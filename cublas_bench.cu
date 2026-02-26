// cuBLAS MXFP8 (microscaling) GEMM baseline benchmark
// Matches the SigLIP2 patch-embed GEMM: C[M,N] = A[M,K] x B[N,K]^T
// MXFP8 E4M3 inputs with UE8M0 per-32-element block scales, FP32 accumulation, BF16 output
// Reports: GEMM only, GEMM+fused_bias, GEMM+fused_bias+unfused_pos, GEMM+unfused_combined
//
// Usage: ./cublas-bench [imgs_per_sm]
//   Default: imgs_per_sm=32 -> M = 32 * 148 * 196 = 928256

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(x) do { \
    cublasStatus_t s = (x); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: status=%d\n", __FILE__, __LINE__, (int)s); \
        exit(1); \
    } \
} while(0)

// Precompute pos_embed as BF16 for the unfused pos_embed kernel
__global__ void precompute_pos_bf16(
    const float* __restrict__ pos_embed,
    __nv_bfloat16* __restrict__ pos_bf16,
    int seq_len, int n_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len * n_dim)
        pos_bf16[idx] = __float2bfloat16(pos_embed[idx]);
}

// Post-processing: C[row, col] += pos_bf16[row % seq_len, col]
// Vectorized: 8 BF16 per thread via 128-bit loads/stores
__global__ void apply_pos_embed(
    __nv_bfloat16* __restrict__ C,
    const __nv_bfloat16* __restrict__ pos_bf16,
    long long total_v8, int N, int seq_len
) {
    long long vid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= total_v8) return;
    long long base = vid * 8;
    int row = (int)(base / N);
    int col = (int)(base % N);
    int pos_row = row % seq_len;

    uint4 cv = *reinterpret_cast<uint4*>(C + base);
    uint4 bv = *reinterpret_cast<const uint4*>(pos_bf16 + (long long)pos_row * N + col);

    uint32_t* cp = reinterpret_cast<uint32_t*>(&cv);
    const uint32_t* bp = reinterpret_cast<const uint32_t*>(&bv);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float c0, c1, b0, b1;
        asm("{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "cvt.rn.f32.bf16 %0, lo;\n\t"
            "cvt.rn.f32.bf16 %1, hi;\n\t"
            "}" : "=f"(c0), "=f"(c1) : "r"(cp[i]));
        asm("{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "cvt.rn.f32.bf16 %0, lo;\n\t"
            "cvt.rn.f32.bf16 %1, hi;\n\t"
            "}" : "=f"(b0), "=f"(b1) : "r"(bp[i]));
        c0 += b0;
        c1 += b1;
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(cp[i]) : "f"(c0), "f"(c1));
    }

    *reinterpret_cast<uint4*>(C + base) = cv;
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);

    const int SM_COUNT = 148;
    const int SEQ_LEN  = 196;
    const int N        = 768;
    const int K        = 768;

    int imgs_per_sm = 32;
    if (argc > 1) imgs_per_sm = atoi(argv[1]);
    const int M = imgs_per_sm * SM_COUNT * SEQ_LEN;

    printf("cuBLAS MXFP8 GEMM baseline (cublasLt)\n");
    printf("  M=%d  N=%d  K=%d  (imgs_per_sm=%d)\n", M, N, K, imgs_per_sm);

    // ── Allocate data tensors ──
    void *d_A, *d_B;
    __nv_bfloat16 *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)M * K));
    CUDA_CHECK(cudaMalloc(&d_B, (size_t)N * K));
    CUDA_CHECK(cudaMalloc(&d_C, (size_t)M * N * sizeof(__nv_bfloat16)));

    CUDA_CHECK(cudaMemset(d_A, 0x3C, (size_t)M * K));
    CUDA_CHECK(cudaMemset(d_B, 0x3C, (size_t)N * K));

    // ── Allocate bias (FP32 for cuBLAS fused epilogue) and pos_embed ──
    float *d_bias, *d_pos;
    __nv_bfloat16 *d_pos_bf16;
    CUDA_CHECK(cudaMalloc(&d_bias, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos, (size_t)SEQ_LEN * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos_bf16, (size_t)SEQ_LEN * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(d_bias, 0, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pos, 0, (size_t)SEQ_LEN * N * sizeof(float)));
    {
        int n_elem = SEQ_LEN * N;
        int tpb = 256;
        precompute_pos_bf16<<<(n_elem + tpb - 1) / tpb, tpb>>>(d_pos, d_pos_bf16, SEQ_LEN, N);
        CUDA_CHECK(cudaGetLastError());
    }

    // ── Allocate MXFP8 scale factor arrays ──
    size_t sf_K = ((size_t)K + 31) / 32;
    size_t sz_scaleA = sf_K * N;
    size_t sz_scaleB = sf_K * M;

    void *d_scaleA, *d_scaleB;
    CUDA_CHECK(cudaMalloc(&d_scaleA, sz_scaleA));
    CUDA_CHECK(cudaMalloc(&d_scaleB, sz_scaleB));
    CUDA_CHECK(cudaMemset(d_scaleA, 0x7F, sz_scaleA));
    CUDA_CHECK(cudaMemset(d_scaleB, 0x7F, sz_scaleB));

    printf("  Scale factors: A=%zu bytes, B=%zu bytes (UE8M0, 1 per 32 along K)\n",
           sz_scaleA, sz_scaleB);

    // ── cuBLASLt setup: plain GEMM (no epilogue bias) ──
    cublasLtHandle_t ltHandle;
    CUBLAS_CHECK(cublasLtCreate(&ltHandle));

    cublasLtMatmulDesc_t matmulDesc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc,
        CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc,
        CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    int32_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc,
        CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc,
        CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc,
        CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scaleA, sizeof(d_scaleA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc,
        CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scaleB, sizeof(d_scaleB)));

    // ── cuBLASLt setup: GEMM + fused bias epilogue ──
    cublasLtMatmulDesc_t matmulDescBias;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDescBias, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDescBias,
        CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDescBias,
        CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDescBias,
        CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDescBias,
        CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDescBias,
        CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scaleA, sizeof(d_scaleA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDescBias,
        CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scaleB, sizeof(d_scaleB)));

    // Fused bias: D[j,i] += bias[j] in col-major = C[row,col] += bias[col] in row-major
    cublasLtEpilogue_t epilogueBias = CUBLASLT_EPILOGUE_BIAS;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDescBias,
        CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogueBias, sizeof(epilogueBias)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDescBias,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(d_bias)));
    cudaDataType_t biasType = CUDA_R_32F;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDescBias,
        CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &biasType, sizeof(biasType)));

    // Matrix layouts
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8F_E4M3, K, N, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8F_E4M3, K, M, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16BF, N, M, N));

    // Heuristics for plain GEMM
    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    size_t workspaceSize = 32 * 1024 * 1024;
    void* d_workspace;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspaceSize));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    cublasLtMatmulHeuristicResult_t heurResult[8];
    int returnedResults = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, layoutA, layoutB, layoutC, layoutC,
        preference, 8, heurResult, &returnedResults));
    if (returnedResults == 0) { fprintf(stderr, "ERROR: No algo (plain)\n"); exit(1); }
    printf("  Found %d plain algorithms\n", returnedResults);

    // Heuristics for GEMM+bias (may not be supported with MXFP8)
    cublasLtMatmulHeuristicResult_t heurBias[8];
    int returnedBias = 0;
    cublasStatus_t biasStatus = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDescBias, layoutA, layoutB, layoutC, layoutC,
        preference, 8, heurBias, &returnedBias);
    bool hasFusedBias = (biasStatus == CUBLAS_STATUS_SUCCESS && returnedBias > 0);
    if (hasFusedBias)
        printf("  Found %d bias-fused algorithms\n", returnedBias);
    else
        printf("  Fused bias NOT SUPPORTED with MXFP8 (status=%d)\n", (int)biasStatus);

    float alpha = 1.0f, beta = 0.0f;

    // apply_pos_embed launch config (vectorized: 8 BF16 per thread)
    long long total_v8 = (long long)M * N / 8;
    int ac_tpb = 256;
    int ac_blocks = (int)((total_v8 + ac_tpb - 1) / ac_tpb);

    // ── Warmup ──
    printf("  Warmup...\n");
    for (int i = 0; i < 3; i++) {
        CUBLAS_CHECK(cublasLtMatmul(ltHandle, matmulDesc,
            &alpha, d_B, layoutA, d_A, layoutB, &beta, d_C, layoutC, d_C, layoutC,
            &heurResult[0].algo, d_workspace, workspaceSize, 0));
        apply_pos_embed<<<ac_blocks, ac_tpb>>>(d_C, d_pos_bf16, total_v8, N, SEQ_LEN);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Verify ──
    {
        __nv_bfloat16 h_C[4];
        CUDA_CHECK(cudaMemcpy(h_C, d_C, 4 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        printf("  C[0,0..3] = %.1f %.1f %.1f %.1f",
            __bfloat162float(h_C[0]), __bfloat162float(h_C[1]),
            __bfloat162float(h_C[2]), __bfloat162float(h_C[3]));
        printf("  (expected %.1f)\n", (float)K * 1.5f * 1.5f);
    }

    const int ITERS = 20;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    double flops = 2.0 * M * N * K;

    // ── 1. GEMM only ──
    printf("  Timing GEMM only (%d iters)...\n", ITERS);
    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; i++) {
        CUBLAS_CHECK(cublasLtMatmul(ltHandle, matmulDesc,
            &alpha, d_B, layoutA, d_A, layoutB, &beta, d_C, layoutC, d_C, layoutC,
            &heurResult[0].algo, d_workspace, workspaceSize, 0));
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_gemm; cudaEventElapsedTime(&ms_gemm, t0, t1); ms_gemm /= ITERS;

    // ── 2. GEMM + fused bias (if supported) ──
    float ms_bias = -1;
    if (hasFusedBias) {
        printf("  Timing GEMM + fused bias (%d iters)...\n", ITERS);
        cudaEventRecord(t0);
        for (int i = 0; i < ITERS; i++) {
            CUBLAS_CHECK(cublasLtMatmul(ltHandle, matmulDescBias,
                &alpha, d_B, layoutA, d_A, layoutB, &beta, d_C, layoutC, d_C, layoutC,
                &heurBias[0].algo, d_workspace, workspaceSize, 0));
        }
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_bias, t0, t1); ms_bias /= ITERS;
    }

    // ── 3. GEMM + unfused pos_embed (best cuBLAS can do for full bias+pos) ──
    printf("  Timing GEMM + unfused bias+pos (%d iters)...\n", ITERS);
    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; i++) {
        CUBLAS_CHECK(cublasLtMatmul(ltHandle, matmulDesc,
            &alpha, d_B, layoutA, d_A, layoutB, &beta, d_C, layoutC, d_C, layoutC,
            &heurResult[0].algo, d_workspace, workspaceSize, 0));
        apply_pos_embed<<<ac_blocks, ac_tpb>>>(d_C, d_pos_bf16, total_v8, N, SEQ_LEN);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_full; cudaEventElapsedTime(&ms_full, t0, t1); ms_full /= ITERS;

    printf("\ncuBLAS MXFP8 results (M=%d N=%d K=%d):\n", M, N, K);
    printf("  GEMM only:                     %.3f ms  %7.2f TFLOPS\n", ms_gemm, flops/ms_gemm/1e9);
    if (hasFusedBias)
        printf("  GEMM + fused bias:             %.3f ms  %7.2f TFLOPS\n", ms_bias, flops/ms_bias/1e9);
    else
        printf("  GEMM + fused bias:             NOT SUPPORTED with MXFP8\n");
    printf("  GEMM + unfused bias+pos:       %.3f ms  %7.2f TFLOPS\n", ms_full, flops/ms_full/1e9);
    printf("  unfused bias+pos overhead: %.3f ms\n", ms_full - ms_gemm);

    // Cleanup
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatmulDescDestroy(matmulDescBias);
    cublasLtDestroy(ltHandle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_workspace); cudaFree(d_scaleA); cudaFree(d_scaleB);
    cudaFree(d_bias); cudaFree(d_pos); cudaFree(d_pos_bf16);

    return 0;
}
