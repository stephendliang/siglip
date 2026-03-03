// cuBLASLt FP8 benchmark focused on fused-add vs unfused periodic add.
// Matches SigLIP patch-embed shape:
//   D[M,N] = A[M,K] x B[N,K]^T
//
// Benchmarks (best-of-heuristics per mode):
//   1) GEMM only (beta=0)
//   2) GEMM + fused add via beta=1 with full C[M,N]
//   3) GEMM + unfused periodic add kernel (beta=0 + apply_combined)
//   4) PostAdd-only kernel
//
// Usage: ./cublas-bench [imgs_per_sm]

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

constexpr int WARMUP_ITERS = 3;
constexpr int TIMED_ITERS  = 20;
constexpr int POSTADD_TPB  = 256;
constexpr int MAX_ALGOS    = 128;
constexpr size_t WORKSPACE_BYTES = 256ULL * 1024 * 1024;

__global__ void precompute_combined(
    const float* __restrict__ bias,
    const float* __restrict__ pos_embed,
    __nv_bfloat16* __restrict__ combined,
    int seq_len, int n_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len * n_dim) {
        int c = idx % n_dim;
        combined[idx] = __float2bfloat16(bias[c] + pos_embed[idx]);
    }
}

// Vectorized periodic add: D[row, col:col+7] += combined[row % seq_len, col:col+7]
__global__ void apply_combined(
    __nv_bfloat16* __restrict__ D,
    const __nv_bfloat16* __restrict__ combined,
    long long total_v8, int N, int seq_len
) {
    long long vid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= total_v8) return;
    long long base = vid * 8;
    int row = (int)(base / N);
    int col = (int)(base % N);
    int pos_row = row % seq_len;

    uint4 dv = *reinterpret_cast<uint4*>(D + base);
    uint4 bv = *reinterpret_cast<const uint4*>(combined + (long long)pos_row * N + col);

    uint32_t* dp = reinterpret_cast<uint32_t*>(&dv);
    const uint32_t* bp = reinterpret_cast<const uint32_t*>(&bv);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        __nv_bfloat162 d2, b2;
        d2 = *reinterpret_cast<__nv_bfloat162*>(&dp[i]);
        b2 = *reinterpret_cast<const __nv_bfloat162*>(&bp[i]);
        d2 = __hadd2(d2, b2);
        dp[i] = *reinterpret_cast<uint32_t*>(&d2);
    }

    *reinterpret_cast<uint4*>(D + base) = dv;
}

struct BenchResult {
    float ms;
    int algo_idx;
};

static BenchResult bench_best_algo(
    cublasLtHandle_t lt,
    cublasLtMatmulDesc_t desc,
    const float* alpha,
    const void* A, cublasLtMatrixLayout_t lA,
    const void* B, cublasLtMatrixLayout_t lB,
    const float* beta,
    const void* C, cublasLtMatrixLayout_t lC,
    void* D, cublasLtMatrixLayout_t lD,
    const cublasLtMatmulHeuristicResult_t* algos, int n_algos,
    void* ws, size_t ws_size,
    size_t d_bytes,
    cudaEvent_t t0, cudaEvent_t t1
) {
    BenchResult best = {-1.0f, -1};
    for (int a = 0; a < n_algos; a++) {
        bool ok = true;

        if (d_bytes > 0) CUDA_CHECK(cudaMemset(D, 0, d_bytes));

        for (int i = 0; i < WARMUP_ITERS; i++) {
            cublasStatus_t s = cublasLtMatmul(
                lt, desc, alpha,
                A, lA, B, lB,
                beta, C, lC,
                D, lD,
                &algos[a].algo,
                ws, ws_size, 0);
            if (s != CUBLAS_STATUS_SUCCESS) {
                ok = false;
                break;
            }
        }
        if (!ok) continue;

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < TIMED_ITERS; i++) {
            cublasStatus_t s = cublasLtMatmul(
                lt, desc, alpha,
                A, lA, B, lB,
                beta, C, lC,
                D, lD,
                &algos[a].algo,
                ws, ws_size, 0);
            if (s != CUBLAS_STATUS_SUCCESS) {
                ok = false;
                break;
            }
        }
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        if (!ok) continue;

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        ms /= TIMED_ITERS;

        if (best.algo_idx < 0 || ms < best.ms) {
            best.ms = ms;
            best.algo_idx = a;
        }
    }
    return best;
}

static BenchResult bench_best_algo_plus_postadd(
    cublasLtHandle_t lt,
    cublasLtMatmulDesc_t desc,
    const float* alpha,
    const void* A, cublasLtMatrixLayout_t lA,
    const void* B, cublasLtMatrixLayout_t lB,
    const float* beta,
    const void* C, cublasLtMatrixLayout_t lC,
    void* D, cublasLtMatrixLayout_t lD,
    const __nv_bfloat16* d_combined,
    int M, int N, int seq_len,
    const cublasLtMatmulHeuristicResult_t* algos, int n_algos,
    void* ws, size_t ws_size,
    size_t d_bytes,
    cudaEvent_t t0, cudaEvent_t t1
) {
    BenchResult best = {-1.0f, -1};
    long long total_v8 = (long long)M * N / 8;
    int postadd_blocks = (int)((total_v8 + POSTADD_TPB - 1) / POSTADD_TPB);

    for (int a = 0; a < n_algos; a++) {
        bool ok = true;

        if (d_bytes > 0) CUDA_CHECK(cudaMemset(D, 0, d_bytes));

        for (int i = 0; i < WARMUP_ITERS; i++) {
            cublasStatus_t s = cublasLtMatmul(
                lt, desc, alpha,
                A, lA, B, lB,
                beta, C, lC,
                D, lD,
                &algos[a].algo,
                ws, ws_size, 0);
            if (s != CUBLAS_STATUS_SUCCESS) {
                ok = false;
                break;
            }
            apply_combined<<<postadd_blocks, POSTADD_TPB>>>(
                reinterpret_cast<__nv_bfloat16*>(D),
                d_combined,
                total_v8, N, seq_len);
        }
        if (!ok) continue;
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < TIMED_ITERS; i++) {
            cublasStatus_t s = cublasLtMatmul(
                lt, desc, alpha,
                A, lA, B, lB,
                beta, C, lC,
                D, lD,
                &algos[a].algo,
                ws, ws_size, 0);
            if (s != CUBLAS_STATUS_SUCCESS) {
                ok = false;
                break;
            }
            apply_combined<<<postadd_blocks, POSTADD_TPB>>>(
                reinterpret_cast<__nv_bfloat16*>(D),
                d_combined,
                total_v8, N, seq_len);
        }
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        if (!ok) continue;
        CUDA_CHECK(cudaGetLastError());

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        ms /= TIMED_ITERS;

        if (best.algo_idx < 0 || ms < best.ms) {
            best.ms = ms;
            best.algo_idx = a;
        }
    }
    return best;
}

static float bench_postadd_only(
    __nv_bfloat16* d_D,
    const __nv_bfloat16* d_combined,
    int M, int N, int seq_len,
    size_t d_bytes,
    cudaEvent_t t0, cudaEvent_t t1
) {
    long long total_v8 = (long long)M * N / 8;
    int blocks = (int)((total_v8 + POSTADD_TPB - 1) / POSTADD_TPB);

    if (d_bytes > 0) CUDA_CHECK(cudaMemset(d_D, 0, d_bytes));

    for (int i = 0; i < WARMUP_ITERS; i++) {
        apply_combined<<<blocks, POSTADD_TPB>>>(d_D, d_combined, total_v8, N, seq_len);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < TIMED_ITERS; i++) {
        apply_combined<<<blocks, POSTADD_TPB>>>(d_D, d_combined, total_v8, N, seq_len);
    }
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    return ms / TIMED_ITERS;
}

static float to_tflops(double flops, float ms) {
    return ms > 0 ? (float)(flops / ms / 1e9) : 0.0f;
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);

    const int SEQ_LEN = 196;
    const int N = 768;
    const int K = 768;

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    const int SM_COUNT = props.multiProcessorCount;

    int imgs_per_sm = 32;
    if (argc > 1) imgs_per_sm = atoi(argv[1]);

    const int M = imgs_per_sm * SM_COUNT * SEQ_LEN;
    const double flops = 2.0 * M * N * K;

    printf("cuBLASLt FP8 benchmark (fused-add focused)\n");
    printf("  Device: %s (SM %d.%d, SMs=%d)\n", props.name, props.major, props.minor, SM_COUNT);
    printf("  Shape: [%d,%d] x [%d,%d]^T  (imgs_per_sm=%d)\n", M, K, K, N, imgs_per_sm);

    if (M % SEQ_LEN != 0) {
        fprintf(stderr, "Invalid shape: M=%d is not divisible by seq_len=%d\n", M, SEQ_LEN);
        return 1;
    }

    size_t sz_a = (size_t)M * K;  // FP8 bytes
    size_t sz_b = (size_t)N * K;  // FP8 bytes
    size_t sz_cd = (size_t)M * N * sizeof(__nv_bfloat16);

    void* d_A = nullptr;
    void* d_B = nullptr;
    __nv_bfloat16* d_C = nullptr;  // full [M,N] add tensor used by beta=1 fused path
    __nv_bfloat16* d_D = nullptr;  // matmul output

    CUDA_CHECK(cudaMalloc(&d_A, sz_a));
    CUDA_CHECK(cudaMalloc(&d_B, sz_b));
    CUDA_CHECK(cudaMalloc(&d_C, sz_cd));
    CUDA_CHECK(cudaMalloc(&d_D, sz_cd));

    CUDA_CHECK(cudaMemset(d_A, 0x3C, sz_a));
    CUDA_CHECK(cudaMemset(d_B, 0x3C, sz_b));
    CUDA_CHECK(cudaMemset(d_C, 0, sz_cd));
    CUDA_CHECK(cudaMemset(d_D, 0, sz_cd));

    // Build periodic add tensor: combined[seq_len, N] = bias[N] + pos_embed[seq_len, N]
    float* d_bias = nullptr;
    float* d_pos = nullptr;
    __nv_bfloat16* d_combined = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bias, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos, (size_t)SEQ_LEN * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_combined, (size_t)SEQ_LEN * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(d_bias, 0, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pos, 0, (size_t)SEQ_LEN * N * sizeof(float)));

    {
        int elems = SEQ_LEN * N;
        int tpb = 256;
        precompute_combined<<<(elems + tpb - 1) / tpb, tpb>>>(d_bias, d_pos, d_combined, SEQ_LEN, N);
        CUDA_CHECK(cudaGetLastError());
    }

    // Tile combined into full C[M,N] for true fused beta=1 benchmarking.
    {
        int num_images = M / SEQ_LEN;
        size_t tile_bytes = (size_t)SEQ_LEN * N * sizeof(__nv_bfloat16);
        for (int img = 0; img < num_images; img++) {
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<char*>(d_C) + (size_t)img * tile_bytes,
                d_combined,
                tile_bytes,
                cudaMemcpyDeviceToDevice));
        }
    }

    // MXFP8 scale factors (Vec32 UE8M0). Shapes correspond to logical op(A), op(B).
    size_t sf_K = ((size_t)K + 31) / 32;
    size_t sz_scaleA = sf_K * N;
    size_t sz_scaleB = sf_K * M;

    void* d_scaleA = nullptr;
    void* d_scaleB = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scaleA, sz_scaleA));
    CUDA_CHECK(cudaMalloc(&d_scaleB, sz_scaleB));
    CUDA_CHECK(cudaMemset(d_scaleA, 0x7F, sz_scaleA));  // scale = 1.0
    CUDA_CHECK(cudaMemset(d_scaleB, 0x7F, sz_scaleB));

    void* d_workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_workspace, WORKSPACE_BYTES));

    cublasLtHandle_t lt_handle;
    CUBLAS_CHECK(cublasLtCreate(&lt_handle));

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;

    // We call cublasLtMatmul with:
    //   A operand <- d_B layout [K,N], transposed to [N,K]
    //   B operand <- d_A layout [K,M], not transposed [K,M]
    //   C/D layout [N,M] (column-major) == logical row-major [M,N]
    cublasLtMatrixLayout_t layoutA = nullptr;
    cublasLtMatrixLayout_t layoutB = nullptr;
    cublasLtMatrixLayout_t layoutC = nullptr;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8F_E4M3, K, N, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8F_E4M3, K, M, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16BF, N, M, N));

    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &WORKSPACE_BYTES,
        sizeof(WORKSPACE_BYTES)));

    cublasLtMatmulDesc_t desc_mxfp8;
    cublasLtMatmulDesc_t desc_plain;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&desc_mxfp8, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&desc_plain, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        desc_mxfp8, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        desc_mxfp8, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        desc_plain, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        desc_plain, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    int32_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        desc_mxfp8, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        desc_mxfp8, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        desc_mxfp8, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scaleA, sizeof(d_scaleA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        desc_mxfp8, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scaleB, sizeof(d_scaleB)));

    cublasLtMatmulHeuristicResult_t heur_mxfp8[MAX_ALGOS];
    cublasLtMatmulHeuristicResult_t heur_plain[MAX_ALGOS];
    int n_mxfp8 = 0;
    int n_plain = 0;

    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        lt_handle, desc_mxfp8, layoutA, layoutB, layoutC, layoutC,
        preference, MAX_ALGOS, heur_mxfp8, &n_mxfp8));

    cublasStatus_t plain_status = cublasLtMatmulAlgoGetHeuristic(
        lt_handle, desc_plain, layoutA, layoutB, layoutC, layoutC,
        preference, MAX_ALGOS, heur_plain, &n_plain);

    bool has_plain = (plain_status == CUBLAS_STATUS_SUCCESS && n_plain > 0);
    printf("  Heuristics: MXFP8=%d, Per-tensor=%d\n", n_mxfp8, has_plain ? n_plain : 0);

    if (n_mxfp8 <= 0) {
        fprintf(stderr, "No MXFP8 heuristics available.\n");
        return 1;
    }

    float alpha = 1.0f;
    float beta0 = 0.0f;
    float beta1 = 1.0f;

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    printf("  Workspace: %zu MB, timed iters=%d\n", WORKSPACE_BYTES / (1024 * 1024), TIMED_ITERS);
    printf("\nBenchmarking...\n");

    BenchResult mxfp8_gemm = bench_best_algo(
        lt_handle, desc_mxfp8,
        &alpha, d_B, layoutA, d_A, layoutB, &beta0, d_C, layoutC, d_D, layoutC,
        heur_mxfp8, n_mxfp8, d_workspace, WORKSPACE_BYTES, sz_cd, t0, t1);

    BenchResult plain_gemm = {-1.0f, -1};
    if (has_plain) {
        plain_gemm = bench_best_algo(
            lt_handle, desc_plain,
            &alpha, d_B, layoutA, d_A, layoutB, &beta0, d_C, layoutC, d_D, layoutC,
            heur_plain, n_plain, d_workspace, WORKSPACE_BYTES, sz_cd, t0, t1);
    }

    // True fused add: beta=1 with full C[M,N].
    BenchResult mxfp8_fused_add = bench_best_algo(
        lt_handle, desc_mxfp8,
        &alpha, d_B, layoutA, d_A, layoutB, &beta1, d_C, layoutC, d_D, layoutC,
        heur_mxfp8, n_mxfp8, d_workspace, WORKSPACE_BYTES, sz_cd, t0, t1);

    BenchResult plain_fused_add = {-1.0f, -1};
    if (has_plain) {
        plain_fused_add = bench_best_algo(
            lt_handle, desc_plain,
            &alpha, d_B, layoutA, d_A, layoutB, &beta1, d_C, layoutC, d_D, layoutC,
            heur_plain, n_plain, d_workspace, WORKSPACE_BYTES, sz_cd, t0, t1);
    }

    // Unfused periodic add: beta=0 GEMM + postadd kernel. Search best algo end-to-end.
    BenchResult mxfp8_unfused = bench_best_algo_plus_postadd(
        lt_handle, desc_mxfp8,
        &alpha, d_B, layoutA, d_A, layoutB, &beta0, d_C, layoutC, d_D, layoutC,
        d_combined, M, N, SEQ_LEN,
        heur_mxfp8, n_mxfp8, d_workspace, WORKSPACE_BYTES, sz_cd, t0, t1);

    BenchResult plain_unfused = {-1.0f, -1};
    if (has_plain) {
        plain_unfused = bench_best_algo_plus_postadd(
            lt_handle, desc_plain,
            &alpha, d_B, layoutA, d_A, layoutB, &beta0, d_C, layoutC, d_D, layoutC,
            d_combined, M, N, SEQ_LEN,
            heur_plain, n_plain, d_workspace, WORKSPACE_BYTES, sz_cd, t0, t1);
    }

    float ms_postadd_only = bench_postadd_only(d_D, d_combined, M, N, SEQ_LEN, sz_cd, t0, t1);

    printf("\n══════════════════════════════════════════════════════════════════════════\n");
    printf("cuBLASLt FP8 Results (M=%d N=%d K=%d)\n", M, N, K);
    printf("══════════════════════════════════════════════════════════════════════════\n");

    auto print_mode = [&](const char* label, const BenchResult& r) {
        if (r.algo_idx < 0) {
            printf("  %-28s  n/a\n", label);
        } else {
            printf("  %-28s  %7.3f ms  %7.1f TFLOPS  (algo #%d)\n",
                   label, r.ms, to_tflops(flops, r.ms), r.algo_idx);
        }
    };

    printf("\nMXFP8:\n");
    print_mode("GEMM only", mxfp8_gemm);
    print_mode("GEMM + fused add (beta=1)", mxfp8_fused_add);
    print_mode("GEMM + unfused periodic add", mxfp8_unfused);

    if (has_plain) {
        printf("\nPer-tensor FP8:\n");
        print_mode("GEMM only", plain_gemm);
        print_mode("GEMM + fused add (beta=1)", plain_fused_add);
        print_mode("GEMM + unfused periodic add", plain_unfused);
    } else {
        printf("\nPer-tensor FP8: not supported on this device/runtime.\n");
    }

    printf("\nPostAdd-only kernel: %.3f ms\n", ms_postadd_only);

    auto print_overhead = [&](const char* tag, const BenchResult& gemm, const BenchResult& fused, const BenchResult& unfused) {
        if (gemm.algo_idx < 0) return;
        if (fused.algo_idx >= 0) {
            printf("  %-10s fused add overhead:    %+7.3f ms\n", tag, fused.ms - gemm.ms);
        }
        if (unfused.algo_idx >= 0) {
            printf("  %-10s unfused add overhead:  %+7.3f ms\n", tag, unfused.ms - gemm.ms);
        }
        if (fused.algo_idx >= 0 && unfused.algo_idx >= 0) {
            printf("  %-10s fused vs unfused gap:  %+7.3f ms (positive favors fused)\n",
                   tag, unfused.ms - fused.ms);
        }
    };

    printf("\nOverheads relative to GEMM-only:\n");
    print_overhead("MXFP8", mxfp8_gemm, mxfp8_fused_add, mxfp8_unfused);
    if (has_plain) {
        print_overhead("PerTensor", plain_gemm, plain_fused_add, plain_unfused);
    }

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

    CUBLAS_CHECK(cublasLtMatmulDescDestroy(desc_mxfp8));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(desc_plain));
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(layoutA));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(layoutB));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(layoutC));
    CUBLAS_CHECK(cublasLtDestroy(lt_handle));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D));
    CUDA_CHECK(cudaFree(d_scaleA));
    CUDA_CHECK(cudaFree(d_scaleB));
    CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_combined));

    return 0;
}
