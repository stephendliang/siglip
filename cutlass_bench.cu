// CUTLASS 4.x SM100a per-tensor FP8 GEMM grid search
// ═══════════════════════════════════════════════════
//
// Sweeps tile/cluster configs with FP32 and BF16 epilogue variants.
// Three measurements per config:
//   1. GEMM only (beta=0, FP32 epilogue): pure compute
//   2. Fused FP32 (beta=1, FP32 epilogue): D = float(acc) + float(C) → BF16
//   3. Fused BF16 (beta=1, BF16 epilogue): D = bf16(acc) + C → BF16
//      Matches custom kernel's cvt_add_bf16x2 path.
//
// Build:  make cutlass-bench
// Usage:  ./cutlass-bench [imgs_per_sm]
//         make cutlass-bench-max   (extended tile sweep)

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include <cuda_bf16.h>

using namespace cute;

#ifndef CUTLASS_EXTENDED_SWEEP
#define CUTLASS_EXTENDED_SWEEP 0
#endif

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

// ── Unfused post-processing kernels ──

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

__global__ void apply_combined(
    __nv_bfloat16* __restrict__ C,
    const __nv_bfloat16* __restrict__ combined,
    long long total_v8, int N, int seq_len
) {
    long long vid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= total_v8) return;
    long long base = vid * 8;
    int row = (int)(base / N);
    int col = (int)(base % N);
    int pos_row = row % seq_len;

    uint4 cv = *reinterpret_cast<uint4*>(C + base);
    uint4 bv = *reinterpret_cast<const uint4*>(combined + (long long)pos_row * N + col);

    uint32_t* cp = reinterpret_cast<uint32_t*>(&cv);
    const uint32_t* bp = reinterpret_cast<const uint32_t*>(&bv);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        __nv_bfloat162 c2, b2;
        c2 = *reinterpret_cast<__nv_bfloat162*>(&cp[i]);
        b2 = *reinterpret_cast<const __nv_bfloat162*>(&bp[i]);
        c2 = __hadd2(c2, b2);
        cp[i] = *reinterpret_cast<uint32_t*>(&c2);
    }

    *reinterpret_cast<uint4*>(C + base) = cv;
}

// ═══════════════════════════════════════════════════════════════
// CUTLASS kernel templates — per-tensor FP8 E4M3, BF16 output
// ═══════════════════════════════════════════════════════════════

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// A: [M, K] row-major, FP8 E4M3
using ElementA  = cutlass::float_e4m3_t;
using LayoutA   = cutlass::layout::RowMajor;
constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16

// B: [N, K] col-major (= [K, N] in memory, transposed access)
using ElementB  = cutlass::float_e4m3_t;
using LayoutB   = cutlass::layout::ColumnMajor;
constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 16

// C/D: BF16 row-major
using ElementC  = cutlass::bfloat16_t;
using ElementD  = cutlass::bfloat16_t;
using LayoutC   = cutlass::layout::RowMajor;
using LayoutD   = cutlass::layout::RowMajor;
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 8
constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;  // 8

using ElementAcc = float;

// ── GemmInstance: full CUTLASS type chain parameterized on tile, cluster, epilogue compute type ──
// EpiCompute controls ElementCompute and ElementScalar in LinearCombination.
// float  → FP32 epilogue: D = float(acc) + float(C)
// bf16   → BF16 epilogue: D = bf16(acc) + C  (matches custom kernel)

template <int TM, int TN, int TK, int CM, int CN, typename EpiCompute>
struct GemmInstance {
    using TileShape_    = Shape<Int<TM>, Int<TN>, Int<TK>>;
    using ClusterShape_ = Shape<Int<CM>, Int<CN>, _1>;

    using FusionOp_ = cutlass::epilogue::fusion::LinearCombination<
        ElementD, EpiCompute, ElementC, EpiCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;

    using CollectiveEpilogue_ = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        TileShape_, ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAcc, EpiCompute,
        ElementC, LayoutC, AlignC,
        ElementD, LayoutD, AlignD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOp_
    >::CollectiveOp;

    using CollectiveMainloop_ = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignA,
        ElementB, LayoutB, AlignB,
        ElementAcc,
        TileShape_, ClusterShape_,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue_::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    using GemmKernel_ = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop_,
        CollectiveEpilogue_>;

    using Gemm_ = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_>;

    // Returns average ms, or -1.0f if config is invalid.
    static float run(void* d_A, void* d_B, void* d_C, void* d_D,
                     int M, int N, int K,
                     cutlass::KernelHardwareInfo& hw_info,
                     float alpha_f, float beta_f) {
        using StrideA_ = typename GemmKernel_::StrideA;
        using StrideB_ = typename GemmKernel_::StrideB;
        using StrideC_ = typename GemmKernel_::StrideC;
        using StrideD_ = typename GemmKernel_::StrideD;

        auto stride_a = cutlass::make_cute_packed_stride(StrideA_{}, make_shape(M, K, 1));
        auto stride_b = cutlass::make_cute_packed_stride(StrideB_{}, make_shape(N, K, 1));
        auto stride_c = cutlass::make_cute_packed_stride(StrideC_{}, make_shape(M, N, 1));
        auto stride_d = cutlass::make_cute_packed_stride(StrideD_{}, make_shape(M, N, 1));

        typename Gemm_::Arguments arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {
                reinterpret_cast<ElementA*>(d_A), stride_a,
                reinterpret_cast<ElementB*>(d_B), stride_b
            },
            {
                {},
                reinterpret_cast<ElementC*>(d_C), stride_c,
                reinterpret_cast<ElementD*>(d_D), stride_d
            },
            hw_info
        };

        arguments.epilogue.thread.alpha = EpiCompute(alpha_f);
        arguments.epilogue.thread.beta  = EpiCompute(beta_f);

        Gemm_ gemm;

        if (gemm.can_implement(arguments) != cutlass::Status::kSuccess)
            return -1.0f;

        size_t workspace_size = Gemm_::get_workspace_size(arguments);
        uint8_t* d_workspace = nullptr;
        if (workspace_size > 0)
            CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));

        if (gemm.initialize(arguments, d_workspace) != cutlass::Status::kSuccess) {
            if (d_workspace) cudaFree(d_workspace);
            return -1.0f;
        }

        constexpr int WARMUP = 3;
        constexpr int ITERS  = 20;

        for (int i = 0; i < WARMUP; i++) {
            if (gemm.run() != cutlass::Status::kSuccess) {
                if (d_workspace) cudaFree(d_workspace);
                return -1.0f;
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);

        cudaEventRecord(t0);
        for (int i = 0; i < ITERS; i++) gemm.run();
        cudaEventRecord(t1);
        CUDA_CHECK(cudaEventSynchronize(t1));

        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        ms /= ITERS;

        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        if (d_workspace) cudaFree(d_workspace);

        return ms;
    }
};

// ── Grid search dispatch ──

struct ConfigResult {
    std::string tile_str;
    std::string cluster_str;
    float ms_gemm;
    float ms_fused_fp32;
    float ms_fused_bf16;
};

template <int TM, int TN, int TK, int CM, int CN>
void try_config(const char* tile_str, const char* cluster_str,
                void* d_A, void* d_B, void* d_C, void* d_D,
                int M, int N, int K,
                cutlass::KernelHardwareInfo& hw_info,
                double flops,
                std::vector<ConfigResult>& results) {
    printf("  %-13s %-5s  ", tile_str, cluster_str);
    fflush(stdout);

    float ms_gemm = GemmInstance<TM,TN,TK,CM,CN,float>::run(
        d_A, d_B, d_C, d_D, M, N, K, hw_info, 1.0f, 0.0f);

    float ms_fused_fp32 = ms_gemm >= 0
        ? GemmInstance<TM,TN,TK,CM,CN,float>::run(
              d_A, d_B, d_C, d_D, M, N, K, hw_info, 1.0f, 1.0f)
        : -1.0f;

    float ms_fused_bf16 = GemmInstance<TM,TN,TK,CM,CN,cutlass::bfloat16_t>::run(
        d_A, d_B, d_C, d_D, M, N, K, hw_info, 1.0f, 1.0f);

    if (ms_gemm < 0 && ms_fused_bf16 < 0) {
        printf("SKIP\n");
    } else {
        auto pr = [](float ms) {
            if (ms < 0) printf("  n/a  ");
            else printf("%6.3f ", ms);
        };
        pr(ms_gemm); printf("/ "); pr(ms_fused_fp32); printf("/ "); pr(ms_fused_bf16);
        printf("ms\n");
    }

    results.push_back({tile_str, cluster_str, ms_gemm, ms_fused_fp32, ms_fused_bf16});
}

#define TRY(tm,tn,tk,cm,cn) \
    try_config<tm,tn,tk,cm,cn>(#tm "x" #tn "x" #tk, #cm "x" #cn, \
        d_A, d_B, d_C, d_D, M, N, K, hw_info, flops, results)

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

// ═══════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    setbuf(stdout, NULL);

#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    printf("ERROR: CUTLASS_ARCH_MMA_SM100_SUPPORTED not defined.\n");
    printf("  Requires CUTLASS 4.x, CUDA 12.8+, and -arch=sm_100a.\n");
    return 1;
#else
    const int SM_COUNT = 148;
    const int SEQ_LEN  = 196;
    const int N        = 768;
    const int K        = 768;

    int imgs_per_sm = 32;
    if (argc > 1) imgs_per_sm = atoi(argv[1]);
    const int M = imgs_per_sm * SM_COUNT * SEQ_LEN;

    double flops = 2.0 * M * N * K;

    printf("CUTLASS SM100a FP8 Grid Search\n");
    printf("  [%d, %d] x [%d, %d]^T  (imgs_per_sm=%d, %d images)\n",
           M, K, K, N, imgs_per_sm, M / SEQ_LEN);
    printf("  FP8 E4M3 (per-tensor) -> BF16, acc FP32\n");

    cudaDeviceProp props;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);
    printf("  Device: %s (SM %d.%d)\n\n", props.name, props.major, props.minor);

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    // ── Allocate data ──
    size_t sz_a  = (size_t)M * K;
    size_t sz_b  = (size_t)N * K;
    size_t sz_cd = (size_t)M * N * sizeof(ElementD);

    void *d_A = nullptr, *d_B = nullptr;
    void *d_C = nullptr, *d_D = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, sz_a));
    CUDA_CHECK(cudaMalloc(&d_B, sz_b));
    CUDA_CHECK(cudaMalloc(&d_C, sz_cd));
    CUDA_CHECK(cudaMalloc(&d_D, sz_cd));

    CUDA_CHECK(cudaMemset(d_A, 0x3C, sz_a));
    CUDA_CHECK(cudaMemset(d_B, 0x3C, sz_b));
    CUDA_CHECK(cudaMemset(d_C, 0, sz_cd));
    CUDA_CHECK(cudaMemset(d_D, 0, sz_cd));

    // ── Combined bias+pos_embed → tile into C ──
    float *d_bias, *d_pos;
    __nv_bfloat16 *d_combined;
    CUDA_CHECK(cudaMalloc(&d_bias, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos, (size_t)SEQ_LEN * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_combined, (size_t)SEQ_LEN * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(d_bias, 0, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pos, 0, (size_t)SEQ_LEN * N * sizeof(float)));
    {
        int n_elem = SEQ_LEN * N;
        int tpb = 256;
        precompute_combined<<<(n_elem + tpb - 1) / tpb, tpb>>>(d_bias, d_pos, d_combined, SEQ_LEN, N);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        int num_images = M / SEQ_LEN;
        size_t row_bytes = (size_t)SEQ_LEN * N * sizeof(__nv_bfloat16);
        for (int img = 0; img < num_images; img++) {
            CUDA_CHECK(cudaMemcpy(
                (char*)d_C + img * row_bytes,
                d_combined,
                row_bytes,
                cudaMemcpyDeviceToDevice));
        }
    }
    printf("  C tiled to [%d, %d] (%.1f MB)\n\n", M, N, (double)sz_cd / (1024*1024));

    // ── Grid search ──
    std::vector<ConfigResult> results;

#if CUTLASS_EXTENDED_SWEEP
    constexpr int kConfigCount = 20;
    printf("Sweeping %d tile configs (extended search, GEMM-only / Fused FP32 / Fused BF16)...\n", kConfigCount);
#else
    constexpr int kConfigCount = 13;
    printf("Sweeping %d tile configs (standard search, GEMM-only / Fused FP32 / Fused BF16)...\n", kConfigCount);
#endif

    // 2SM configs (cluster_m=2)
    TRY(256, 128,  64, 2, 1);
    TRY(256, 256,  64, 2, 1);
    TRY(256, 128, 128, 2, 1);
    TRY(256, 256, 128, 2, 1);
    TRY(128, 128,  64, 2, 1);
    TRY(128, 256,  64, 2, 1);
    TRY(128, 128, 128, 2, 1);
    TRY(128, 256, 128, 2, 1);
    TRY(256, 128,  64, 2, 2);
    TRY(256, 256,  64, 2, 2);
    TRY(256, 256, 128, 2, 2);

    // 1SM configs
    TRY(128, 128,  64, 1, 1);
    TRY(128, 256,  64, 1, 1);

#if CUTLASS_EXTENDED_SWEEP
    // Extended 2SM configs (cluster_m=2, cluster_n=2)
    TRY(256, 128, 128, 2, 2);
    TRY(128, 128,  64, 2, 2);
    TRY(128, 256,  64, 2, 2);
    TRY(128, 128, 128, 2, 2);
    TRY(128, 256, 128, 2, 2);

    // Extended 1SM configs
    TRY(128, 128, 128, 1, 1);
    TRY(128, 256, 128, 1, 1);
#endif

    // ── Sort by fused BF16 time (invalid configs last) ──
    std::sort(results.begin(), results.end(), [](const ConfigResult& a, const ConfigResult& b) {
        float av = a.ms_fused_bf16 < 0 ? 1e9f : a.ms_fused_bf16;
        float bv = b.ms_fused_bf16 < 0 ? 1e9f : b.ms_fused_bf16;
        return av < bv;
    });

    // ── Results table ──
    auto tflops = [&](float ms) -> float {
        return ms > 0 ? (float)(flops / ms / 1e9) : 0.0f;
    };

    printf("\n");
    printf("CUTLASS SM100a FP8 Grid Search — [%d, %d] x [%d, %d]^T\n", M, K, K, N);
    printf("════════════════════════════════════════════════════════════════════════════════\n");
    printf("  %-13s %-5s  | %-18s | %-18s | %-18s\n",
           "Tile", "Clust", "GEMM-only", "Fused FP32", "Fused BF16");
    printf("  %-13s %-5s  | %7s %9s | %7s %9s | %7s %9s\n",
           "", "", "ms", "TFLOPS", "ms", "TFLOPS", "ms", "TFLOPS");
    printf("──────────────────────+────────────────────+────────────────────+────────────────────\n");

    auto print_cell = [&](float ms) {
        if (ms < 0) printf("     n/a       n/a ");
        else printf("  %6.3f  %7.0f ", ms, tflops(ms));
    };

    for (auto& r : results) {
        printf("  %-13s %-5s  |", r.tile_str.c_str(), r.cluster_str.c_str());
        print_cell(r.ms_gemm);      printf("|");
        print_cell(r.ms_fused_fp32); printf("|");
        print_cell(r.ms_fused_bf16);
        printf("\n");
    }

    printf("──────────────────────+────────────────────+────────────────────+────────────────────\n");

    // ── Best configs ──
    const ConfigResult* best_gemm = nullptr;
    const ConfigResult* best_fused = nullptr;

    for (auto& r : results) {
        if (r.ms_gemm > 0 && (!best_gemm || r.ms_gemm < best_gemm->ms_gemm))
            best_gemm = &r;
        if (r.ms_fused_bf16 > 0 && (!best_fused || r.ms_fused_bf16 < best_fused->ms_fused_bf16))
            best_fused = &r;
    }

    if (best_gemm)
        printf("  Best GEMM:  %s %s  %.3f ms / %.0f TFLOPS\n",
               best_gemm->tile_str.c_str(), best_gemm->cluster_str.c_str(),
               best_gemm->ms_gemm, tflops(best_gemm->ms_gemm));
    if (best_fused)
        printf("  Best Fused: %s %s  %.3f ms / %.0f TFLOPS (BF16)\n",
               best_fused->tile_str.c_str(), best_fused->cluster_str.c_str(),
               best_fused->ms_fused_bf16, tflops(best_fused->ms_fused_bf16));

    printf("  Custom kernel (fused):       0.530 ms / 2067 TFLOPS\n");

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_bias);
    cudaFree(d_pos);
    cudaFree(d_combined);

    return 0;
#endif
}
