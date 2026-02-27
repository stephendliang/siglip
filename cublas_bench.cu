// cuBLAS FP8 GEMM baseline benchmark (fair comparison)
// Matches the SigLIP2 patch-embed GEMM: C[M,N] = A[M,K] x B[N,K]^T
// Tests BOTH per-tensor scaled FP8 AND MXFP8 (Vec32 UE8M0 block scales)
// Tries ALL heuristic algorithms and reports the best timing for each
// Reports: GEMM only, GEMM+fused_bias, GEMM+unfused_bias+pos
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

// ── Best-of-N algorithm benchmark helper ──
// Tries all heuristic algorithms with warmup, returns best time (ms) and algo index.
struct BenchResult { float ms; int algo_idx; };

static BenchResult bench_best_algo(
    cublasLtHandle_t lt, cublasLtMatmulDesc_t desc,
    const float* alpha, const void* A, cublasLtMatrixLayout_t lA,
    const void* B, cublasLtMatrixLayout_t lB,
    const float* beta, void* C, cublasLtMatrixLayout_t lC,
    cublasLtMatmulHeuristicResult_t* algos, int n_algos,
    void* ws, size_t ws_size,
    int n_iters, cudaEvent_t t0, cudaEvent_t t1)
{
    BenchResult best = {-1.0f, -1};
    for (int a = 0; a < n_algos; a++) {
        // Warmup (3 iters) — skip algo if it fails
        bool ok = true;
        for (int i = 0; i < 3; i++) {
            cublasStatus_t s = cublasLtMatmul(lt, desc, alpha, A, lA, B, lB,
                beta, C, lC, C, lC, &algos[a].algo, ws, ws_size, 0);
            if (s != CUBLAS_STATUS_SUCCESS) { ok = false; break; }
        }
        if (!ok) continue;
        cudaDeviceSynchronize();

        // Timed
        cudaEventRecord(t0);
        for (int i = 0; i < n_iters; i++) {
            cublasLtMatmul(lt, desc, alpha, A, lA, B, lB,
                beta, C, lC, C, lC, &algos[a].algo, ws, ws_size, 0);
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        ms /= n_iters;
        if (best.algo_idx < 0 || ms < best.ms) {
            best.ms = ms;
            best.algo_idx = a;
        }
    }
    return best;
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

    printf("cuBLAS FP8 GEMM baseline (cublasLt, best-of-N algorithms)\n");
    printf("  M=%d  N=%d  K=%d  (imgs_per_sm=%d)\n", M, N, K, imgs_per_sm);

    // ── Allocate data tensors ──
    void *d_A, *d_B;
    __nv_bfloat16 *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)M * K));
    CUDA_CHECK(cudaMalloc(&d_B, (size_t)N * K));
    CUDA_CHECK(cudaMalloc(&d_C, (size_t)M * N * sizeof(__nv_bfloat16)));

    CUDA_CHECK(cudaMemset(d_A, 0x3C, (size_t)M * K));
    CUDA_CHECK(cudaMemset(d_B, 0x3C, (size_t)N * K));

    // ── Allocate bias and pos_embed ──
    float *d_bias, *d_pos;
    __nv_bfloat16 *d_pos_bf16;
    CUDA_CHECK(cudaMalloc(&d_bias, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos, (size_t)SEQ_LEN * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos_bf16, (size_t)SEQ_LEN * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(d_bias, 0, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pos,  0, (size_t)SEQ_LEN * N * sizeof(float)));
    {
        int n_elem = SEQ_LEN * N;
        int tpb = 256;
        precompute_pos_bf16<<<(n_elem + tpb - 1) / tpb, tpb>>>(d_pos, d_pos_bf16, SEQ_LEN, N);
        CUDA_CHECK(cudaGetLastError());
    }

    // ── Allocate MXFP8 scale factor arrays (UE8M0, 1 per 32 elements along K) ──
    size_t sf_K = ((size_t)K + 31) / 32;
    size_t sz_scaleA = sf_K * N;
    size_t sz_scaleB = sf_K * M;

    void *d_scaleA, *d_scaleB;
    CUDA_CHECK(cudaMalloc(&d_scaleA, sz_scaleA));
    CUDA_CHECK(cudaMalloc(&d_scaleB, sz_scaleB));
    CUDA_CHECK(cudaMemset(d_scaleA, 0x7F, sz_scaleA));  // UE8M0 0x7F = 2^(127-127) = 1.0
    CUDA_CHECK(cudaMemset(d_scaleB, 0x7F, sz_scaleB));

    printf("  MXFP8 scale factors: A=%zu bytes, B=%zu bytes (%.1f MB total)\n",
           sz_scaleA, sz_scaleB, (sz_scaleA + sz_scaleB) / (1024.0 * 1024.0));

    // ── Workspace: 256 MB (generous — lets cuBLAS pick its best algorithms) ──
    size_t workspaceSize = 256ULL * 1024 * 1024;
    void* d_workspace;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspaceSize));
    printf("  Workspace: %zu MB\n", workspaceSize / (1024*1024));

    // ── cuBLASLt setup ──
    cublasLtHandle_t ltHandle;
    CUBLAS_CHECK(cublasLtCreate(&ltHandle));

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;

    // Matrix layouts (shared by all descriptors)
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8F_E4M3, K, N, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8F_E4M3, K, M, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16BF, N, M, N));

    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    const int MAX_ALGOS = 32;

    // ════════════════════════════════════════════════════════════
    // Descriptor 1: MXFP8 (Vec32 UE8M0 block scales, no bias)
    // ════════════════════════════════════════════════════════════
    cublasLtMatmulDesc_t descMXFP8;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&descMXFP8, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(descMXFP8,
        CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(descMXFP8,
        CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    int32_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(descMXFP8,
        CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(descMXFP8,
        CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(descMXFP8,
        CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scaleA, sizeof(d_scaleA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(descMXFP8,
        CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scaleB, sizeof(d_scaleB)));

    cublasLtMatmulHeuristicResult_t heurMXFP8[MAX_ALGOS];
    int nMXFP8 = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, descMXFP8, layoutA, layoutB, layoutC, layoutC,
        preference, MAX_ALGOS, heurMXFP8, &nMXFP8));
    printf("  MXFP8: %d algorithms found\n", nMXFP8);

    // ════════════════════════════════════════════════════════════
    // Descriptor 2: Per-tensor scaled FP8 (no MXFP8 block scales)
    // No A_SCALE_MODE/B_SCALE_MODE set — cuBLAS uses per-tensor scaling.
    // No scale pointers — default scale = 1.0.
    // This matches what the custom kernel does (zero scale registers = no scaling).
    // ════════════════════════════════════════════════════════════
    cublasLtMatmulDesc_t descPlain;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&descPlain, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(descPlain,
        CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(descPlain,
        CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    cublasLtMatmulHeuristicResult_t heurPlain[MAX_ALGOS];
    int nPlain = 0;
    cublasStatus_t plainStatus = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, descPlain, layoutA, layoutB, layoutC, layoutC,
        preference, MAX_ALGOS, heurPlain, &nPlain);
    bool hasPlain = (plainStatus == CUBLAS_STATUS_SUCCESS && nPlain > 0);
    if (hasPlain)
        printf("  Per-tensor FP8: %d algorithms found\n", nPlain);
    else
        printf("  Per-tensor FP8: NOT SUPPORTED on this GPU (status=%d, n=%d)\n",
               (int)plainStatus, nPlain);

    // ════════════════════════════════════════════════════════════
    // Descriptor 3: MXFP8 + fused bias epilogue
    // ════════════════════════════════════════════════════════════
    cublasLtMatmulDesc_t descMXFP8Bias;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&descMXFP8Bias, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(descMXFP8Bias,
        CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(descMXFP8Bias,
        CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(descMXFP8Bias,
        CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(descMXFP8Bias,
        CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(descMXFP8Bias,
        CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scaleA, sizeof(d_scaleA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(descMXFP8Bias,
        CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scaleB, sizeof(d_scaleB)));

    cublasLtEpilogue_t epilogueBias = CUBLASLT_EPILOGUE_BIAS;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(descMXFP8Bias,
        CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogueBias, sizeof(epilogueBias)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(descMXFP8Bias,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(d_bias)));
    cudaDataType_t biasType = CUDA_R_32F;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(descMXFP8Bias,
        CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &biasType, sizeof(biasType)));

    cublasLtMatmulHeuristicResult_t heurMXFP8Bias[MAX_ALGOS];
    int nMXFP8Bias = 0;
    cublasStatus_t biasStatus = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, descMXFP8Bias, layoutA, layoutB, layoutC, layoutC,
        preference, MAX_ALGOS, heurMXFP8Bias, &nMXFP8Bias);
    bool hasBias = (biasStatus == CUBLAS_STATUS_SUCCESS && nMXFP8Bias > 0);
    if (hasBias)
        printf("  MXFP8+bias: %d algorithms found\n", nMXFP8Bias);
    else
        printf("  MXFP8+bias: NOT SUPPORTED (status=%d)\n", (int)biasStatus);

    float alpha = 1.0f, beta = 0.0f;

    // apply_pos_embed launch config (vectorized: 8 BF16 per thread)
    long long total_v8 = (long long)M * N / 8;
    int ac_tpb = 256;
    int ac_blocks = (int)((total_v8 + ac_tpb - 1) / ac_tpb);

    // ── Warmup ──
    printf("\n  Warmup...\n");
    for (int i = 0; i < 3; i++) {
        CUBLAS_CHECK(cublasLtMatmul(ltHandle, descMXFP8,
            &alpha, d_B, layoutA, d_A, layoutB, &beta, d_C, layoutC, d_C, layoutC,
            &heurMXFP8[0].algo, d_workspace, workspaceSize, 0));
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

    // ════════════════════════════════════════════════════════════
    // Benchmark: try ALL heuristic algorithms, report best
    // ════════════════════════════════════════════════════════════
    printf("\nBenchmarking (best-of-N algorithms, %d timed iters each)...\n", ITERS);

    // 1. MXFP8 GEMM only
    printf("  [1/5] MXFP8 GEMM only (%d algos)...\n", nMXFP8);
    BenchResult mxfp8_gemm = bench_best_algo(ltHandle, descMXFP8,
        &alpha, d_B, layoutA, d_A, layoutB, &beta, d_C, layoutC,
        heurMXFP8, nMXFP8, d_workspace, workspaceSize, ITERS, t0, t1);

    // 2. Per-tensor FP8 GEMM only
    BenchResult plain_gemm = {-1.0f, -1};
    if (hasPlain) {
        printf("  [2/5] Per-tensor FP8 GEMM only (%d algos)...\n", nPlain);
        plain_gemm = bench_best_algo(ltHandle, descPlain,
            &alpha, d_B, layoutA, d_A, layoutB, &beta, d_C, layoutC,
            heurPlain, nPlain, d_workspace, workspaceSize, ITERS, t0, t1);
    } else {
        printf("  [2/5] Per-tensor FP8: skipped (not supported)\n");
    }

    // 3. MXFP8 + fused bias
    BenchResult mxfp8_bias = {-1.0f, -1};
    if (hasBias) {
        printf("  [3/5] MXFP8 + fused bias (%d algos)...\n", nMXFP8Bias);
        mxfp8_bias = bench_best_algo(ltHandle, descMXFP8Bias,
            &alpha, d_B, layoutA, d_A, layoutB, &beta, d_C, layoutC,
            heurMXFP8Bias, nMXFP8Bias, d_workspace, workspaceSize, ITERS, t0, t1);
    } else {
        printf("  [3/5] MXFP8 + fused bias: skipped (not supported)\n");
    }

    // 4 & 5. Best GEMM + unfused pos_embed (using both MXFP8 and per-tensor best algos)
    // Pick whichever GEMM was faster for the combined timing
    float ms_mxfp8_full = -1.0f;
    if (mxfp8_gemm.algo_idx >= 0) {
        printf("  [4/5] MXFP8 GEMM + unfused pos_embed...\n");
        // Warmup
        for (int i = 0; i < 3; i++) {
            cublasLtMatmul(ltHandle, descMXFP8,
                &alpha, d_B, layoutA, d_A, layoutB, &beta, d_C, layoutC, d_C, layoutC,
                &heurMXFP8[mxfp8_gemm.algo_idx].algo, d_workspace, workspaceSize, 0);
            apply_pos_embed<<<ac_blocks, ac_tpb>>>(d_C, d_pos_bf16, total_v8, N, SEQ_LEN);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(t0);
        for (int i = 0; i < ITERS; i++) {
            cublasLtMatmul(ltHandle, descMXFP8,
                &alpha, d_B, layoutA, d_A, layoutB, &beta, d_C, layoutC, d_C, layoutC,
                &heurMXFP8[mxfp8_gemm.algo_idx].algo, d_workspace, workspaceSize, 0);
            apply_pos_embed<<<ac_blocks, ac_tpb>>>(d_C, d_pos_bf16, total_v8, N, SEQ_LEN);
        }
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_mxfp8_full, t0, t1); ms_mxfp8_full /= ITERS;
    }

    float ms_plain_full = -1.0f;
    if (plain_gemm.algo_idx >= 0) {
        printf("  [5/5] Per-tensor FP8 GEMM + unfused pos_embed...\n");
        for (int i = 0; i < 3; i++) {
            cublasLtMatmul(ltHandle, descPlain,
                &alpha, d_B, layoutA, d_A, layoutB, &beta, d_C, layoutC, d_C, layoutC,
                &heurPlain[plain_gemm.algo_idx].algo, d_workspace, workspaceSize, 0);
            apply_pos_embed<<<ac_blocks, ac_tpb>>>(d_C, d_pos_bf16, total_v8, N, SEQ_LEN);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(t0);
        for (int i = 0; i < ITERS; i++) {
            cublasLtMatmul(ltHandle, descPlain,
                &alpha, d_B, layoutA, d_A, layoutB, &beta, d_C, layoutC, d_C, layoutC,
                &heurPlain[plain_gemm.algo_idx].algo, d_workspace, workspaceSize, 0);
            apply_pos_embed<<<ac_blocks, ac_tpb>>>(d_C, d_pos_bf16, total_v8, N, SEQ_LEN);
        }
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_plain_full, t0, t1); ms_plain_full /= ITERS;
    } else {
        printf("  [5/5] Per-tensor FP8 + pos: skipped\n");
    }

    // ════════════════════════════════════════════════════════════
    // Results
    // ════════════════════════════════════════════════════════════
    printf("\n══════════════════════════════════════════════════════════\n");
    printf("cuBLAS FP8 results (M=%d N=%d K=%d, workspace=%zuMB)\n", M, N, K, workspaceSize/(1024*1024));
    printf("══════════════════════════════════════════════════════════\n");

    printf("\nGEMM only:\n");
    if (mxfp8_gemm.algo_idx >= 0)
        printf("  MXFP8 (block-scaled):       %.3f ms  %7.1f TFLOPS  (algo #%d of %d)\n",
            mxfp8_gemm.ms, flops/mxfp8_gemm.ms/1e9, mxfp8_gemm.algo_idx, nMXFP8);
    else
        printf("  MXFP8 (block-scaled):       FAILED\n");

    if (plain_gemm.algo_idx >= 0)
        printf("  Per-tensor scaled:          %.3f ms  %7.1f TFLOPS  (algo #%d of %d)\n",
            plain_gemm.ms, flops/plain_gemm.ms/1e9, plain_gemm.algo_idx, nPlain);
    else
        printf("  Per-tensor scaled:          NOT SUPPORTED\n");

    // Best pure GEMM
    float best_gemm_ms = 1e9f;
    const char* best_gemm_label = "none";
    if (mxfp8_gemm.algo_idx >= 0 && mxfp8_gemm.ms < best_gemm_ms)
        { best_gemm_ms = mxfp8_gemm.ms; best_gemm_label = "MXFP8"; }
    if (plain_gemm.algo_idx >= 0 && plain_gemm.ms < best_gemm_ms)
        { best_gemm_ms = plain_gemm.ms; best_gemm_label = "per-tensor"; }
    if (best_gemm_ms < 1e9f)
        printf("  >>> Best GEMM:              %.3f ms  %7.1f TFLOPS  (%s)\n",
            best_gemm_ms, flops/best_gemm_ms/1e9, best_gemm_label);

    printf("\nGEMM + fused bias:\n");
    if (mxfp8_bias.algo_idx >= 0)
        printf("  MXFP8+bias:                 %.3f ms  %7.1f TFLOPS  (algo #%d of %d)\n",
            mxfp8_bias.ms, flops/mxfp8_bias.ms/1e9, mxfp8_bias.algo_idx, nMXFP8Bias);
    else
        printf("  MXFP8+bias:                 NOT SUPPORTED\n");

    printf("\nGEMM + unfused bias+pos (end-to-end):\n");
    if (ms_mxfp8_full > 0)
        printf("  MXFP8 + pos:                %.3f ms  %7.1f TFLOPS\n",
            ms_mxfp8_full, flops/ms_mxfp8_full/1e9);
    if (ms_plain_full > 0)
        printf("  Per-tensor + pos:           %.3f ms  %7.1f TFLOPS\n",
            ms_plain_full, flops/ms_plain_full/1e9);

    float best_full_ms = 1e9f;
    if (ms_mxfp8_full > 0 && ms_mxfp8_full < best_full_ms) best_full_ms = ms_mxfp8_full;
    if (ms_plain_full > 0 && ms_plain_full < best_full_ms) best_full_ms = ms_plain_full;
    if (best_full_ms < 1e9f)
        printf("  >>> Best end-to-end:        %.3f ms  %7.1f TFLOPS\n",
            best_full_ms, flops/best_full_ms/1e9);

    if (best_gemm_ms < 1e9f && best_full_ms < 1e9f)
        printf("  Unfused pos overhead:       %.3f ms\n", best_full_ms - best_gemm_ms);

    // Cleanup
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(descMXFP8);
    cublasLtMatmulDescDestroy(descPlain);
    cublasLtMatmulDescDestroy(descMXFP8Bias);
    cublasLtDestroy(ltHandle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_workspace); cudaFree(d_scaleA); cudaFree(d_scaleB);
    cudaFree(d_bias); cudaFree(d_pos); cudaFree(d_pos_bf16);

    return 0;
}
