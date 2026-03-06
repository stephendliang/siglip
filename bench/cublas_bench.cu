// cuBLASLt FP8 GEMM benchmark with compile-time configurable dimensions and epilogue.
//
// Compile-time config (via -D flags):
//   BENCH_N          Output dimension         (default: 768)
//   BENCH_K          Reduction dimension       (default: 768)
//   BENCH_EPILOGUE   Epilogue type (int):
//     0 = NONE         — GEMM-only
//     1 = PERIODIC_ADD — beta=1 fused add (patch embed)
//     2 = GELU_BIAS    — cuBLAS GELU_BIAS epilogue (FC1)
//     3 = BIAS_ONLY    — cuBLAS BIAS epilogue (FC2)
//
// Usage: ./cublas-bench [imgs_per_sm]   (default: 32)
//   M = imgs_per_sm * SM_COUNT * 196
//
// Examples:
//   Patch embed: make cublas-bench       (N=768, K=768,  PERIODIC_ADD)
//   FC1:         make cublas-bench-fc1   (N=3072, K=768, GELU_BIAS)
//   FC2:         make cublas-bench-fc2   (N=768, K=3072, BIAS_ONLY)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

// ── Compile-time configuration ──

#ifndef BENCH_N
#define BENCH_N 768
#endif
#ifndef BENCH_K
#define BENCH_K 768
#endif
#ifndef BENCH_EPILOGUE
#define BENCH_EPILOGUE 1
#endif

enum class Epilogue { NONE = 0, PERIODIC_ADD = 1, GELU_BIAS = 2, BIAS_ONLY = 3 };

constexpr int N_DIM = BENCH_N;
constexpr int K_DIM = BENCH_K;
constexpr Epilogue EPI = static_cast<Epilogue>(BENCH_EPILOGUE);

static_assert(BENCH_EPILOGUE >= 0 && BENCH_EPILOGUE <= 3, "BENCH_EPILOGUE must be 0-3");
static_assert(N_DIM % 8 == 0, "BENCH_N must be multiple of 8");
static_assert(K_DIM % 32 == 0, "BENCH_K must be multiple of 32 (MXFP8 Vec32)");

constexpr int SEQ_LEN = 196;
constexpr int WARMUP_ITERS = 3;
constexpr int TIMED_ITERS = 20;
constexpr int TPB = 256;
constexpr int MAX_ALGOS = 128;
constexpr size_t WORKSPACE_BYTES = 256ULL * 1024 * 1024;

constexpr const char* EPI_NAME[] = {"none", "periodic_add", "gelu_bias", "bias_only"};
constexpr const char* EPI_SUFFIX[] = {"", " + periodic add", " + bias + GELU", " + bias"};
constexpr const char* FUSED_LABEL[] = {"", "GEMM + fused add (beta=1)", "GEMM + fused bias+GELU", "GEMM + fused bias"};
constexpr const char* UNFUSED_LABEL[] = {"", "GEMM + unfused periodic add", "GEMM + unfused bias+GELU", "GEMM + unfused bias"};
constexpr const char* POST_LABEL[] = {"", "PostAdd-only", "Bias+GELU-only", "Bias-only"};

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

// ── Unfused post-kernels ──

__global__ void precompute_combined(
    const float* __restrict__ bias, const float* __restrict__ pos_embed,
    __nv_bfloat16* __restrict__ combined, int seq_len, int n_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len * n_dim)
        combined[idx] = __float2bfloat16(bias[idx % n_dim] + pos_embed[idx]);
}

__global__ void apply_combined(
    __nv_bfloat16* __restrict__ D, const __nv_bfloat16* __restrict__ combined,
    long long total_v8, int N, int seq_len
) {
    long long vid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= total_v8) return;
    long long base = vid * 8;
    int row = (int)(base / N), col = (int)(base % N);

    uint4 dv = *reinterpret_cast<uint4*>(D + base);
    uint4 bv = *reinterpret_cast<const uint4*>(combined + (long long)(row % seq_len) * N + col);
    uint32_t* dp = reinterpret_cast<uint32_t*>(&dv);
    const uint32_t* bp = reinterpret_cast<const uint32_t*>(&bv);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        __nv_bfloat162 d2 = *reinterpret_cast<__nv_bfloat162*>(&dp[i]);
        __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(&bp[i]);
        d2 = __hadd2(d2, b2);
        dp[i] = *reinterpret_cast<uint32_t*>(&d2);
    }
    *reinterpret_cast<uint4*>(D + base) = dv;
}

__global__ void apply_bias_gelu(
    __nv_bfloat16* __restrict__ D, const float* __restrict__ bias,
    int N, long long total_v8
) {
    long long vid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= total_v8) return;
    long long base = vid * 8;
    int col = (int)(base % N);

    uint4 dv = *reinterpret_cast<uint4*>(D + base);
    __nv_bfloat16* dp = reinterpret_cast<__nv_bfloat16*>(&dv);
    float4 bv0 = __ldg(reinterpret_cast<const float4*>(bias + col));
    float4 bv1 = __ldg(reinterpret_cast<const float4*>(bias + col + 4));
    float b[8] = {bv0.x, bv0.y, bv0.z, bv0.w, bv1.x, bv1.y, bv1.z, bv1.w};

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float x = __bfloat162float(dp[i]) + b[i];
        const float k = 0.7978845608f;
        dp[i] = __float2bfloat16(0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x))));
    }
    *reinterpret_cast<uint4*>(D + base) = dv;
}

__global__ void apply_bias_only(
    __nv_bfloat16* __restrict__ D, const float* __restrict__ bias,
    int N, long long total_v8
) {
    long long vid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= total_v8) return;
    long long base = vid * 8;
    int col = (int)(base % N);

    uint4 dv = *reinterpret_cast<uint4*>(D + base);
    __nv_bfloat16* dp = reinterpret_cast<__nv_bfloat16*>(&dv);
    float4 bv0 = __ldg(reinterpret_cast<const float4*>(bias + col));
    float4 bv1 = __ldg(reinterpret_cast<const float4*>(bias + col + 4));
    float b[8] = {bv0.x, bv0.y, bv0.z, bv0.w, bv1.x, bv1.y, bv1.z, bv1.w};

    #pragma unroll
    for (int i = 0; i < 8; i++)
        dp[i] = __float2bfloat16(__bfloat162float(dp[i]) + b[i]);
    *reinterpret_cast<uint4*>(D + base) = dv;
}

// ── Benchmark helpers ──

struct BenchResult { float ms; int algo_idx; };

static BenchResult bench_best_algo(
    cublasLtHandle_t lt, cublasLtMatmulDesc_t desc, const float* alpha,
    const void* A, cublasLtMatrixLayout_t lA,
    const void* B, cublasLtMatrixLayout_t lB,
    const float* beta,
    const void* C, cublasLtMatrixLayout_t lC,
    void* D, cublasLtMatrixLayout_t lD,
    const cublasLtMatmulHeuristicResult_t* algos, int n_algos,
    void* ws, size_t ws_size, size_t d_bytes,
    cudaEvent_t t0, cudaEvent_t t1
) {
    BenchResult best = {-1.0f, -1};
    for (int a = 0; a < n_algos; a++) {
        bool ok = true;
        if (d_bytes > 0) CUDA_CHECK(cudaMemset(D, 0, d_bytes));
        for (int i = 0; i < WARMUP_ITERS; i++) {
            if (cublasLtMatmul(lt, desc, alpha, A, lA, B, lB, beta, C, lC, D, lD,
                               &algos[a].algo, ws, ws_size, 0) != CUBLAS_STATUS_SUCCESS)
                { ok = false; break; }
        }
        if (!ok) continue;
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < TIMED_ITERS; i++) {
            if (cublasLtMatmul(lt, desc, alpha, A, lA, B, lB, beta, C, lC, D, lD,
                               &algos[a].algo, ws, ws_size, 0) != CUBLAS_STATUS_SUCCESS)
                { ok = false; break; }
        }
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        if (!ok) continue;

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        ms /= TIMED_ITERS;
        if (best.algo_idx < 0 || ms < best.ms) { best.ms = ms; best.algo_idx = a; }
    }
    return best;
}

template <typename PostFn>
static BenchResult bench_best_algo_plus_post(
    cublasLtHandle_t lt, cublasLtMatmulDesc_t desc, const float* alpha,
    const void* A, cublasLtMatrixLayout_t lA,
    const void* B, cublasLtMatrixLayout_t lB,
    const float* beta,
    const void* C, cublasLtMatrixLayout_t lC,
    void* D, cublasLtMatrixLayout_t lD,
    PostFn post_fn,
    const cublasLtMatmulHeuristicResult_t* algos, int n_algos,
    void* ws, size_t ws_size, size_t d_bytes,
    cudaEvent_t t0, cudaEvent_t t1
) {
    BenchResult best = {-1.0f, -1};
    for (int a = 0; a < n_algos; a++) {
        bool ok = true;
        if (d_bytes > 0) CUDA_CHECK(cudaMemset(D, 0, d_bytes));
        for (int i = 0; i < WARMUP_ITERS; i++) {
            if (cublasLtMatmul(lt, desc, alpha, A, lA, B, lB, beta, C, lC, D, lD,
                               &algos[a].algo, ws, ws_size, 0) != CUBLAS_STATUS_SUCCESS)
                { ok = false; break; }
            post_fn();
        }
        if (!ok) continue;
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < TIMED_ITERS; i++) {
            if (cublasLtMatmul(lt, desc, alpha, A, lA, B, lB, beta, C, lC, D, lD,
                               &algos[a].algo, ws, ws_size, 0) != CUBLAS_STATUS_SUCCESS)
                { ok = false; break; }
            post_fn();
        }
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        if (!ok) continue;
        CUDA_CHECK(cudaGetLastError());

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        ms /= TIMED_ITERS;
        if (best.algo_idx < 0 || ms < best.ms) { best.ms = ms; best.algo_idx = a; }
    }
    return best;
}

template <typename PostFn>
static float bench_post_only(PostFn post_fn, size_t d_bytes, void* d_D,
                             cudaEvent_t t0, cudaEvent_t t1) {
    if (d_bytes > 0) CUDA_CHECK(cudaMemset(d_D, 0, d_bytes));
    for (int i = 0; i < WARMUP_ITERS; i++) post_fn();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < TIMED_ITERS; i++) post_fn();
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

// ── Main ──

int main(int argc, char** argv) {
    setbuf(stdout, NULL);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    const int SM_COUNT = props.multiProcessorCount;

    int imgs_per_sm = 32;
    if (argc > 1) imgs_per_sm = atoi(argv[1]);

    const int M = imgs_per_sm * SM_COUNT * SEQ_LEN;
    const double flops = 2.0 * M * (double)N_DIM * K_DIM;
    const size_t sz_a = (size_t)M * K_DIM;
    const size_t sz_b = (size_t)N_DIM * K_DIM;
    const size_t sz_d = (size_t)M * N_DIM * sizeof(__nv_bfloat16);

    constexpr int E = (int)EPI;
    printf("cuBLASLt FP8 benchmark (N=%d K=%d epilogue=%s)\n", N_DIM, K_DIM, EPI_NAME[E]);
    printf("  Device: %s (SM %d.%d, SMs=%d)\n", props.name, props.major, props.minor, SM_COUNT);
    printf("  Shape: [%d,%d] x [%d,%d]^T%s  (imgs_per_sm=%d)\n",
           M, K_DIM, K_DIM, N_DIM, EPI_SUFFIX[E], imgs_per_sm);

    // ── Common allocations ──

    void* d_A = nullptr;
    void* d_B = nullptr;
    __nv_bfloat16* d_D = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, sz_a));
    CUDA_CHECK(cudaMalloc(&d_B, sz_b));
    CUDA_CHECK(cudaMalloc(&d_D, sz_d));
    CUDA_CHECK(cudaMemset(d_A, 0x3C, sz_a));
    CUDA_CHECK(cudaMemset(d_B, 0x3C, sz_b));
    CUDA_CHECK(cudaMemset(d_D, 0, sz_d));

    size_t sf_K = ((size_t)K_DIM + 31) / 32;
    void* d_scaleA = nullptr;
    void* d_scaleB = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scaleA, sf_K * N_DIM));
    CUDA_CHECK(cudaMalloc(&d_scaleB, sf_K * M));
    CUDA_CHECK(cudaMemset(d_scaleA, 0x7F, sf_K * N_DIM));
    CUDA_CHECK(cudaMemset(d_scaleB, 0x7F, sf_K * M));

    void* d_workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_workspace, WORKSPACE_BYTES));

    // ── Epilogue-specific allocations ──

    float* d_bias = nullptr;
    float* d_pos = nullptr;
    __nv_bfloat16* d_combined = nullptr;
    __nv_bfloat16* d_C = nullptr;

    if constexpr (EPI == Epilogue::PERIODIC_ADD) {
        CUDA_CHECK(cudaMalloc(&d_bias, (size_t)N_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pos, (size_t)SEQ_LEN * N_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_combined, (size_t)SEQ_LEN * N_DIM * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemset(d_bias, 0, (size_t)N_DIM * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_pos, 0, (size_t)SEQ_LEN * N_DIM * sizeof(float)));
        {
            int elems = SEQ_LEN * N_DIM;
            precompute_combined<<<(elems + TPB - 1) / TPB, TPB>>>(
                d_bias, d_pos, d_combined, SEQ_LEN, N_DIM);
            CUDA_CHECK(cudaGetLastError());
        }
        // Tile combined into full C[M,N] for beta=1 fused benchmarking
        CUDA_CHECK(cudaMalloc(&d_C, sz_d));
        {
            int num_images = M / SEQ_LEN;
            size_t tile_bytes = (size_t)SEQ_LEN * N_DIM * sizeof(__nv_bfloat16);
            for (int img = 0; img < num_images; img++)
                CUDA_CHECK(cudaMemcpy(reinterpret_cast<char*>(d_C) + (size_t)img * tile_bytes,
                                      d_combined, tile_bytes, cudaMemcpyDeviceToDevice));
        }
    } else if constexpr (EPI == Epilogue::GELU_BIAS || EPI == Epilogue::BIAS_ONLY) {
        CUDA_CHECK(cudaMalloc(&d_bias, (size_t)N_DIM * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_bias, 0, (size_t)N_DIM * sizeof(float)));
    }

    // ── cuBLASLt setup ──

    cublasLtHandle_t lt;
    CUBLAS_CHECK(cublasLtCreate(&lt));

    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WORKSPACE_BYTES, sizeof(WORKSPACE_BYTES)));

    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    int32_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;

    // Layouts: A=[K,N] transa=T, B=[K,M] transb=N, C/D=[N,M]
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8F_E4M3, K_DIM, N_DIM, K_DIM));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8F_E4M3, K_DIM, M, K_DIM));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16BF, N_DIM, M, N_DIM));

    // Descriptor factory
    auto make_desc = [&](bool mxfp8, bool fused) -> cublasLtMatmulDesc_t {
        cublasLtMatmulDesc_t d;
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
        if (mxfp8) {
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scaleA, sizeof(d_scaleA)));
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scaleB, sizeof(d_scaleB)));
        }
        if (fused) {
            if constexpr (EPI == Epilogue::GELU_BIAS) {
                cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_GELU_BIAS;
                CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
            } else if constexpr (EPI == Epilogue::BIAS_ONLY) {
                cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
                CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
            }
            if constexpr (EPI == Epilogue::GELU_BIAS || EPI == Epilogue::BIAS_ONLY) {
                CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(d_bias)));
                cudaDataType_t btype = CUDA_R_32F;
                CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &btype, sizeof(btype)));
            }
        }
        return d;
    };

    // Base descs (GEMM-only)
    auto desc_mxfp8 = make_desc(true, false);
    auto desc_plain = make_desc(false, false);

    cublasLtMatmulHeuristicResult_t heur_mxfp8[MAX_ALGOS], heur_plain[MAX_ALGOS];
    int n_mxfp8 = 0, n_plain = 0;

    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        lt, desc_mxfp8, layoutA, layoutB, layoutC, layoutC,
        pref, MAX_ALGOS, heur_mxfp8, &n_mxfp8));

    cublasStatus_t ps = cublasLtMatmulAlgoGetHeuristic(
        lt, desc_plain, layoutA, layoutB, layoutC, layoutC,
        pref, MAX_ALGOS, heur_plain, &n_plain);
    bool has_plain = (ps == CUBLAS_STATUS_SUCCESS && n_plain > 0);

    printf("  Heuristics: MXFP8=%d, Per-tensor=%d\n", n_mxfp8, has_plain ? n_plain : 0);

    if (n_mxfp8 <= 0) {
        fprintf(stderr, "No MXFP8 heuristics available.\n");
        return 1;
    }

    // Fused descs + heuristics
    cublasLtMatmulDesc_t desc_fused_mxfp8 = nullptr, desc_fused_plain = nullptr;
    cublasLtMatmulHeuristicResult_t heur_fused_mxfp8[MAX_ALGOS], heur_fused_plain[MAX_ALGOS];
    int n_fused_mxfp8 = 0, n_fused_plain = 0;

    if constexpr (EPI == Epilogue::PERIODIC_ADD) {
        // PERIODIC_ADD reuses base descs with beta=1 — same heuristics apply
        desc_fused_mxfp8 = desc_mxfp8;
        n_fused_mxfp8 = n_mxfp8;
        memcpy(heur_fused_mxfp8, heur_mxfp8, n_mxfp8 * sizeof(heur_mxfp8[0]));
        if (has_plain) {
            desc_fused_plain = desc_plain;
            n_fused_plain = n_plain;
            memcpy(heur_fused_plain, heur_plain, n_plain * sizeof(heur_plain[0]));
        }
    } else if constexpr (EPI == Epilogue::GELU_BIAS || EPI == Epilogue::BIAS_ONLY) {
        desc_fused_mxfp8 = make_desc(true, true);
        desc_fused_plain = make_desc(false, true);
        CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
            lt, desc_fused_mxfp8, layoutA, layoutB, layoutC, layoutC,
            pref, MAX_ALGOS, heur_fused_mxfp8, &n_fused_mxfp8));
        cublasStatus_t fs = cublasLtMatmulAlgoGetHeuristic(
            lt, desc_fused_plain, layoutA, layoutB, layoutC, layoutC,
            pref, MAX_ALGOS, heur_fused_plain, &n_fused_plain);
        if (fs != CUBLAS_STATUS_SUCCESS) n_fused_plain = 0;
        printf("  Fused heuristics: MXFP8=%d, Plain=%d\n", n_fused_mxfp8, n_fused_plain);
    }

    float alpha = 1.0f, beta0 = 0.0f, beta1 = 1.0f;
    float* beta_fused = (EPI == Epilogue::PERIODIC_ADD) ? &beta1 : &beta0;
    void* C_fused = (EPI == Epilogue::PERIODIC_ADD) ? (void*)d_C : (void*)d_D;

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    printf("  Workspace: %zu MB, timed iters=%d\n\nBenchmarking...\n", WORKSPACE_BYTES >> 20, TIMED_ITERS);

    // ── GEMM-only ──

    BenchResult mxfp8_gemm = bench_best_algo(lt, desc_mxfp8, &alpha,
        d_B, layoutA, d_A, layoutB, &beta0, d_D, layoutC, d_D, layoutC,
        heur_mxfp8, n_mxfp8, d_workspace, WORKSPACE_BYTES, sz_d, t0, t1);

    BenchResult plain_gemm = {-1.0f, -1};
    if (has_plain)
        plain_gemm = bench_best_algo(lt, desc_plain, &alpha,
            d_B, layoutA, d_A, layoutB, &beta0, d_D, layoutC, d_D, layoutC,
            heur_plain, n_plain, d_workspace, WORKSPACE_BYTES, sz_d, t0, t1);

    // ── Fused + unfused + post-only ──

    BenchResult mxfp8_fused = {-1.0f, -1}, mxfp8_unfused = {-1.0f, -1};
    BenchResult plain_fused = {-1.0f, -1}, plain_unfused = {-1.0f, -1};
    float ms_post_only = -1.0f;

    if constexpr (EPI != Epilogue::NONE) {
        // Fused
        if (n_fused_mxfp8 > 0)
            mxfp8_fused = bench_best_algo(lt, desc_fused_mxfp8, &alpha,
                d_B, layoutA, d_A, layoutB, beta_fused, C_fused, layoutC, d_D, layoutC,
                heur_fused_mxfp8, n_fused_mxfp8, d_workspace, WORKSPACE_BYTES, sz_d, t0, t1);
        if (n_fused_plain > 0)
            plain_fused = bench_best_algo(lt, desc_fused_plain, &alpha,
                d_B, layoutA, d_A, layoutB, beta_fused, C_fused, layoutC, d_D, layoutC,
                heur_fused_plain, n_fused_plain, d_workspace, WORKSPACE_BYTES, sz_d, t0, t1);

        // Unfused (GEMM + post kernel) and post-only
        long long total_v8 = (long long)M * N_DIM / 8;
        int post_blocks = (int)((total_v8 + TPB - 1) / TPB);

        auto run_unfused = [&](auto post_fn) {
            mxfp8_unfused = bench_best_algo_plus_post(lt, desc_mxfp8, &alpha,
                d_B, layoutA, d_A, layoutB, &beta0, d_D, layoutC, d_D, layoutC,
                post_fn, heur_mxfp8, n_mxfp8, d_workspace, WORKSPACE_BYTES, sz_d, t0, t1);
            if (has_plain)
                plain_unfused = bench_best_algo_plus_post(lt, desc_plain, &alpha,
                    d_B, layoutA, d_A, layoutB, &beta0, d_D, layoutC, d_D, layoutC,
                    post_fn, heur_plain, n_plain, d_workspace, WORKSPACE_BYTES, sz_d, t0, t1);
            ms_post_only = bench_post_only(post_fn, sz_d, d_D, t0, t1);
        };

        if constexpr (EPI == Epilogue::PERIODIC_ADD) {
            run_unfused([=]() {
                apply_combined<<<post_blocks, TPB>>>(d_D, d_combined, total_v8, N_DIM, SEQ_LEN);
            });
        } else if constexpr (EPI == Epilogue::GELU_BIAS) {
            run_unfused([=]() {
                apply_bias_gelu<<<post_blocks, TPB>>>(d_D, d_bias, N_DIM, total_v8);
            });
        } else if constexpr (EPI == Epilogue::BIAS_ONLY) {
            run_unfused([=]() {
                apply_bias_only<<<post_blocks, TPB>>>(d_D, d_bias, N_DIM, total_v8);
            });
        }
    }

    // ── Results ──

    printf("\n══════════════════════════════════════════════════════════════════════════\n");
    printf("cuBLASLt FP8 Results (M=%d N=%d K=%d%s)\n", M, N_DIM, K_DIM, EPI_SUFFIX[E]);
    printf("══════════════════════════════════════════════════════════════════════════\n");

    auto pr = [&](const char* label, const BenchResult& r) {
        if (r.algo_idx < 0)
            printf("  %-36s  n/a\n", label);
        else
            printf("  %-36s  %7.3f ms  %7.1f TFLOPS  (algo #%d)\n",
                   label, r.ms, to_tflops(flops, r.ms), r.algo_idx);
    };

    printf("\nMXFP8:\n");
    pr("GEMM only", mxfp8_gemm);
    if constexpr (EPI != Epilogue::NONE) {
        pr(FUSED_LABEL[E], mxfp8_fused);
        pr(UNFUSED_LABEL[E], mxfp8_unfused);
    }

    if (has_plain) {
        printf("\nPer-tensor FP8:\n");
        pr("GEMM only", plain_gemm);
        if constexpr (EPI != Epilogue::NONE) {
            pr(FUSED_LABEL[E], plain_fused);
            pr(UNFUSED_LABEL[E], plain_unfused);
        }
    } else {
        printf("\nPer-tensor FP8: not supported on this device/runtime.\n");
    }

    if constexpr (EPI != Epilogue::NONE) {
        printf("\n%s kernel: %.3f ms\n", POST_LABEL[E], ms_post_only);

        printf("\nOverheads relative to GEMM-only:\n");
        auto oh = [](const char* tag, const BenchResult& gemm,
                     const BenchResult& fused, const BenchResult& unfused) {
            if (gemm.algo_idx < 0) return;
            if (fused.algo_idx >= 0)
                printf("  %-10s fused overhead:      %+7.3f ms\n", tag, fused.ms - gemm.ms);
            if (unfused.algo_idx >= 0)
                printf("  %-10s unfused overhead:    %+7.3f ms\n", tag, unfused.ms - gemm.ms);
            if (fused.algo_idx >= 0 && unfused.algo_idx >= 0)
                printf("  %-10s fused vs unfused:    %+7.3f ms (positive favors fused)\n",
                       tag, unfused.ms - fused.ms);
        };
        oh("MXFP8", mxfp8_gemm, mxfp8_fused, mxfp8_unfused);
        if (has_plain)
            oh("PerTensor", plain_gemm, plain_fused, plain_unfused);
    }

    // ── Cleanup ──

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

    CUBLAS_CHECK(cublasLtMatmulDescDestroy(desc_mxfp8));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(desc_plain));
    if constexpr (EPI == Epilogue::GELU_BIAS || EPI == Epilogue::BIAS_ONLY) {
        CUBLAS_CHECK(cublasLtMatmulDescDestroy(desc_fused_mxfp8));
        CUBLAS_CHECK(cublasLtMatmulDescDestroy(desc_fused_plain));
    }
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(layoutA));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(layoutB));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(layoutC));
    CUBLAS_CHECK(cublasLtDestroy(lt));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_D));
    CUDA_CHECK(cudaFree(d_scaleA));
    CUDA_CHECK(cudaFree(d_scaleB));
    CUDA_CHECK(cudaFree(d_workspace));
    if (d_bias) CUDA_CHECK(cudaFree(d_bias));
    if (d_pos) CUDA_CHECK(cudaFree(d_pos));
    if (d_combined) CUDA_CHECK(cudaFree(d_combined));
    if (d_C) CUDA_CHECK(cudaFree(d_C));

    return 0;
}
