// cuBLAS MXFP8 (microscaling) GEMM baseline benchmark
// Matches the SigLIP2 patch-embed GEMM: C[M,N] = A[M,K] x B[N,K]^T
// MXFP8 E4M3 inputs with UE8M0 per-32-element block scales, FP32 accumulation, BF16 output
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

    // ── Allocate MXFP8 scale factor arrays ──
    // UE8M0 format: one scale byte per 32 elements along K (innermost dim in col-major)
    // cuBLAS "A" = our B stored as [K,N] col-major: ceil(K/32) * N scale bytes
    // cuBLAS "B" = our A stored as [K,M] col-major: ceil(K/32) * M scale bytes
    size_t sf_K = ((size_t)K + 31) / 32;
    size_t sz_scaleA = sf_K * N;   // cuBLAS A scales (our B)
    size_t sz_scaleB = sf_K * M;   // cuBLAS B scales (our A)

    void *d_scaleA, *d_scaleB;
    CUDA_CHECK(cudaMalloc(&d_scaleA, sz_scaleA));
    CUDA_CHECK(cudaMalloc(&d_scaleB, sz_scaleB));

    // Fill with 0x7F = UE8M0 encoding of 2^0 = 1.0
    CUDA_CHECK(cudaMemset(d_scaleA, 0x7F, sz_scaleA));
    CUDA_CHECK(cudaMemset(d_scaleB, 0x7F, sz_scaleB));

    printf("  Scale factors: A=%zu bytes, B=%zu bytes (UE8M0, 1 per 32 along K)\n",
           sz_scaleA, sz_scaleB);

    // ── cuBLASLt setup ──
    cublasLtHandle_t ltHandle;
    CUBLAS_CHECK(cublasLtCreate(&ltHandle));

    // For row-major A[M,K] x B[N,K]^T = C[M,N]:
    // In col-major (cuBLAS native): compute D[N,M] = B[K,N]^T x A[K,M]
    //   A is [M,K] row-major = [K,M] col-major with ld=K
    //   B is [N,K] row-major = [K,N] col-major with ld=K
    //   transa=T (transpose B to get [N,K]), transb=N (A stays as [K,M])
    cublasLtMatmulDesc_t matmulDesc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc,
        CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc,
        CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Set MXFP8 block-scaled mode for both A and B
    int32_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc,
        CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc,
        CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));

    // Point to scale factor arrays
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc,
        CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scaleA, sizeof(d_scaleA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc,
        CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scaleB, sizeof(d_scaleB)));

    // Matrix layouts: cuBLAS "A" = our weight B, cuBLAS "B" = our input A
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8F_E4M3, K, N, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8F_E4M3, K, M, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16BF, N, M, N));

    // ── Heuristic search ──
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

    if (returnedResults == 0) {
        fprintf(stderr, "ERROR: No cuBLAS algorithm found!\n");
        exit(1);
    }
    printf("  Found %d cuBLAS algorithms\n", returnedResults);

    float alpha = 1.0f, beta = 0.0f;

    // ── Warmup ──
    printf("  Warmup (3 iters)...\n");
    for (int i = 0; i < 3; i++) {
        CUBLAS_CHECK(cublasLtMatmul(ltHandle, matmulDesc,
            &alpha, d_B, layoutA, d_A, layoutB, &beta, d_C, layoutC, d_C, layoutC,
            &heurResult[0].algo, d_workspace, workspaceSize, 0));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Verify ──
    {
        __nv_bfloat16 h_C[4];
        CUDA_CHECK(cudaMemcpy(h_C, d_C, 4 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        printf("  C[0,0..3] = %.1f %.1f %.1f %.1f",
            __bfloat162float(h_C[0]), __bfloat162float(h_C[1]),
            __bfloat162float(h_C[2]), __bfloat162float(h_C[3]));
        float expected = (float)K * 1.5f * 1.5f;
        printf("  (expected %.1f)\n", expected);
    }

    // ── Timed runs ──
    const int ITERS = 20;
    printf("  Timing (%d iters)...\n", ITERS);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; i++) {
        CUBLAS_CHECK(cublasLtMatmul(ltHandle, matmulDesc,
            &alpha, d_B, layoutA, d_A, layoutB, &beta, d_C, layoutC, d_C, layoutC,
            &heurResult[0].algo, d_workspace, workspaceSize, 0));
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= (float)ITERS;

    double flops = 2.0 * M * N * K;
    double tflops = flops / ms / 1e9;

    printf("\ncuBLAS MXFP8:  %.3f ms  %.2f TFLOPS\n", ms, tflops);
    printf("  M=%d  N=%d  K=%d\n", M, N, K);

    // ── Sweep all returned algorithms ──
    if (returnedResults > 1) {
        printf("\n  Algorithm sweep:\n");
        printf("  %-6s %-10s %-10s\n", "Algo#", "ms", "TFLOPS");
        for (int a = 0; a < returnedResults; a++) {
            for (int i = 0; i < 2; i++) {
                cublasLtMatmul(ltHandle, matmulDesc,
                    &alpha, d_B, layoutA, d_A, layoutB, &beta, d_C, layoutC, d_C, layoutC,
                    &heurResult[a].algo, d_workspace, workspaceSize, 0);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            cudaEventRecord(t0);
            for (int i = 0; i < ITERS; i++) {
                cublasLtMatmul(ltHandle, matmulDesc,
                    &alpha, d_B, layoutA, d_A, layoutB, &beta, d_C, layoutC, d_C, layoutC,
                    &heurResult[a].algo, d_workspace, workspaceSize, 0);
            }
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            float ams;
            cudaEventElapsedTime(&ams, t0, t1);
            ams /= (float)ITERS;
            printf("  %-6d %-10.3f %-10.2f\n", a, ams, flops / ams / 1e9);
        }
    }

    // Cleanup
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtDestroy(ltHandle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_workspace);
    cudaFree(d_scaleA);
    cudaFree(d_scaleB);

    return 0;
}
