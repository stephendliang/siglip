// CUTLASS 4.x SM100a per-tensor FP8 GEMM policy + tile grid search
// ═══════════════════════════════════════════════════════════════
//
// Compile-time config (via -D flags):
//   BENCH_N          Output dimension         (default: 768)
//   BENCH_K          Reduction dimension       (default: 768)
//   BENCH_EPILOGUE   Epilogue type (int):
//     1 = PERIODIC_ADD — fused periodic table add (patch embed)
//     2 = GELU_BIAS    — fused bias + GELU_taylor (FC1)
//     3 = BIAS_ONLY    — fused bias only (FC2)
//
// Build:  make cutlass-bench        (patch embed: N=768, K=768, PERIODIC_ADD)
//         make cutlass-bench-fc1    (FC1: N=3072, K=768, GELU_BIAS)
//         make cutlass-bench-fc2    (FC2: N=768, K=3072, BIAS_ONLY)
//         make cutlass-bench-max    (extended patch embed sweep)

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
#include "cutlass/arch/arch.h"

#include <cuda_bf16.h>
#include "cutlass/epilogue/thread/activation.h"

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

#if BENCH_EPILOGUE == 1
#include "siglip_periodic_add.hpp"
#endif

using namespace cute;

enum class Epilogue : int { PERIODIC_ADD = 1, GELU_BIAS = 2, BIAS_ONLY = 3 };
constexpr Epilogue EPI = static_cast<Epilogue>(BENCH_EPILOGUE);
constexpr int N_DIM = BENCH_N;
constexpr int K_DIM = BENCH_K;

static_assert(BENCH_EPILOGUE >= 1 && BENCH_EPILOGUE <= 3, "BENCH_EPILOGUE must be 1-3");

#ifndef CUTLASS_EXTENDED_SWEEP
#define CUTLASS_EXTENDED_SWEEP 0
#endif

#ifndef CUTLASS_COMPREHENSIVE_SWEEP
#define CUTLASS_COMPREHENSIVE_SWEEP 1
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

// Unfused bias+GELU kernel for FC1 two-kernel baseline
__global__ void apply_bias_gelu(
    __nv_bfloat16* __restrict__ D,
    const float* __restrict__ bias,
    int N, long long total_v8) {
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

// Unfused bias-only kernel for FC2 two-kernel baseline
__global__ void apply_bias_only(
    __nv_bfloat16* __restrict__ D,
    const float* __restrict__ bias,
    int N, long long total_v8) {
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
        dp[i] = __float2bfloat16(__bfloat162float(dp[i]) + b[i]);
    }
    *reinterpret_cast<uint4*>(D + base) = dv;
}

static float host_gelu_taylor(float x) {
    const float k = 0.7978845608f;
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
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

constexpr int WARMUP_ITERS = 2;
constexpr int TIMED_ITERS  = 10;
constexpr int POSTADD_TPB  = 256;

// Stage policy:
//   MainloopStages=0 -> StageCountAutoCarveout<epilogue_smem>
//   MainloopStages>0 -> explicit StageCount<MainloopStages>
template <int MainloopStages, int CarveoutBytes>
struct StageCountSelector {
    using Type = cutlass::gemm::collective::StageCount<MainloopStages>;
};

template <int CarveoutBytes>
struct StageCountSelector<0, CarveoutBytes> {
    using Type = cutlass::gemm::collective::StageCountAutoCarveout<CarveoutBytes>;
};

// ── Timing helper: warmup + event-timed loop ──

template <typename GemmAdapter>
static float run_gemm_timed(GemmAdapter& gemm,
                            typename GemmAdapter::Arguments& arguments,
                            size_t d_bytes, void* d_D) {
    if (gemm.can_implement(arguments) != cutlass::Status::kSuccess)
        return -1.0f;

    size_t ws = GemmAdapter::get_workspace_size(arguments);
    uint8_t* d_ws = nullptr;
    if (ws > 0) CUDA_CHECK(cudaMalloc(&d_ws, ws));

    if (gemm.initialize(arguments, d_ws) != cutlass::Status::kSuccess) {
        if (d_ws) cudaFree(d_ws);
        return -1.0f;
    }

    if (d_bytes > 0) CUDA_CHECK(cudaMemset(d_D, 0, d_bytes));

    for (int i = 0; i < WARMUP_ITERS; i++) {
        if (gemm.run() != cutlass::Status::kSuccess) {
            if (d_ws) cudaFree(d_ws);
            return -1.0f;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    for (int i = 0; i < TIMED_ITERS; i++) gemm.run();
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= TIMED_ITERS;

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    if (d_ws) cudaFree(d_ws);
    return ms;
}

// Timing helper for GEMM + unfused post-kernel in the same loop
template <typename GemmAdapter, typename PostFn>
static float run_gemm_plus_post_timed(GemmAdapter& gemm,
                                      typename GemmAdapter::Arguments& arguments,
                                      PostFn post_fn,
                                      size_t d_bytes, void* d_D) {
    if (gemm.can_implement(arguments) != cutlass::Status::kSuccess)
        return -1.0f;

    size_t ws = GemmAdapter::get_workspace_size(arguments);
    uint8_t* d_ws = nullptr;
    if (ws > 0) CUDA_CHECK(cudaMalloc(&d_ws, ws));

    if (gemm.initialize(arguments, d_ws) != cutlass::Status::kSuccess) {
        if (d_ws) cudaFree(d_ws);
        return -1.0f;
    }

    if (d_bytes > 0) CUDA_CHECK(cudaMemset(d_D, 0, d_bytes));

    for (int i = 0; i < WARMUP_ITERS; i++) {
        if (gemm.run() != cutlass::Status::kSuccess) {
            if (d_ws) cudaFree(d_ws);
            return -1.0f;
        }
        post_fn();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    for (int i = 0; i < TIMED_ITERS; i++) {
        gemm.run();
        post_fn();
    }
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaGetLastError());

    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= TIMED_ITERS;

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    if (d_ws) cudaFree(d_ws);
    return ms;
}

// ── GemmInstance: full CUTLASS type chain parameterized on tile, cluster, epilogue compute type ──

template <
    int TM, int TN, int TK, int CM, int CN,
    typename EpiCompute,
    typename MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto,
    typename EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto,
    int MainloopStages = 0
>
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
        EpilogueSchedule,
        FusionOp_
    >::CollectiveOp;

    static constexpr int kCarveoutBytes = static_cast<int>(sizeof(typename CollectiveEpilogue_::SharedStorage));
    using StageCountType_ = typename StageCountSelector<MainloopStages, kCarveoutBytes>::Type;

    using CollectiveMainloop_ = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignA,
        ElementB, LayoutB, AlignB,
        ElementAcc,
        TileShape_, ClusterShape_,
        StageCountType_,
        MainloopSchedule
    >::CollectiveOp;

    using GemmKernel_ = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop_,
        CollectiveEpilogue_>;

    using Gemm_ = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_>;

    static constexpr size_t kSmemBytes = sizeof(typename GemmKernel_::SharedStorage);

    static float run(void* d_A, void* d_B, void* d_C, void* d_D,
                     int M, int N, int K,
                     cutlass::KernelHardwareInfo& hw_info,
                     float alpha_f, float beta_f,
                     size_t d_bytes) {
        if constexpr (kSmemBytes > cutlass::arch::sm100_smem_capacity_bytes) {
            return -3.0f;
        } else {
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
            return run_gemm_timed(gemm, arguments, d_bytes, d_D);
        }
    }

    // GEMM + unfused post-kernel, timed together
    template <typename PostFn>
    static float run_gemm_plus_post(
        void* d_A, void* d_B, void* d_C, void* d_D,
        int M, int N, int K,
        cutlass::KernelHardwareInfo& hw_info,
        float alpha_f, float beta_f,
        size_t d_bytes,
        PostFn post_fn
    ) {
        if constexpr (kSmemBytes > cutlass::arch::sm100_smem_capacity_bytes) {
            return -3.0f;
        } else {
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
            return run_gemm_plus_post_timed(gemm, arguments, post_fn, d_bytes, d_D);
        }
    }
};

// ── FusedPeriodicGemmInstance: GEMM + periodic table add fused in epilogue via custom EVT ──
#if BENCH_EPILOGUE == 1

template <
    int TM, int TN, int TK, int CM, int CN,
    typename MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto,
    typename EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto,
    int MainloopStages = 0
>
struct FusedPeriodicGemmInstance {
    using TileShape_    = Shape<Int<TM>, Int<TN>, Int<TK>>;
    using ClusterShape_ = Shape<Int<CM>, Int<CN>, _1>;

    using FusionOp_ = cutlass::epilogue::fusion::SigLipPeriodicAdd<>;

    using CollectiveEpilogue_ = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        TileShape_, ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAcc, float,
        void, LayoutD, AlignD,
        ElementD, LayoutD, AlignD,
        EpilogueSchedule,
        FusionOp_
    >::CollectiveOp;

    static constexpr int kCarveoutBytes = static_cast<int>(sizeof(typename CollectiveEpilogue_::SharedStorage));
    using StageCountType_ = typename StageCountSelector<MainloopStages, kCarveoutBytes>::Type;

    using CollectiveMainloop_ = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignA,
        ElementB, LayoutB, AlignB,
        ElementAcc,
        TileShape_, ClusterShape_,
        StageCountType_,
        MainloopSchedule
    >::CollectiveOp;

    using GemmKernel_ = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop_,
        CollectiveEpilogue_>;

    using Gemm_ = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_>;

    static constexpr size_t kSmemBytes = sizeof(typename GemmKernel_::SharedStorage);

    static float run(void* d_A, void* d_B, void* d_D,
                     const __nv_bfloat16* d_combined,
                     int M, int N, int K, int seq_len,
                     cutlass::KernelHardwareInfo& hw_info,
                     size_t d_bytes) {
        if constexpr (kSmemBytes > cutlass::arch::sm100_smem_capacity_bytes) {
            return -3.0f;
        } else {
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
                    nullptr, stride_c,
                    reinterpret_cast<ElementD*>(d_D), stride_d
                },
                hw_info
            };

            arguments.epilogue.thread.op_1.ptr_combined =
                reinterpret_cast<cutlass::bfloat16_t const*>(d_combined);
            arguments.epilogue.thread.op_1.seq_len = seq_len;

            Gemm_ gemm;
            return run_gemm_timed(gemm, arguments, d_bytes, d_D);
        }
    }
};

#endif // BENCH_EPILOGUE == 1

// ── BiasActGemmInstance: GEMM + per-col bias + activation fused in epilogue ──
// ActivationFn: GELU_taylor (FC1), Identity (FC2)

template <
    template <class> class ActivationFn,
    int TM, int TN, int TK, int CM, int CN,
    typename MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto,
    typename EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto,
    int MainloopStages = 0
>
struct BiasActGemmInstance {
    using TileShape_    = Shape<Int<TM>, Int<TN>, Int<TK>>;
    using ClusterShape_ = Shape<Int<CM>, Int<CN>, _1>;

    using FusionOp_ = cutlass::epilogue::fusion::LinCombPerColBiasEltAct<
        ActivationFn,
        ElementD, float, float, ElementC, float>;

    using CollectiveEpilogue_ = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        TileShape_, ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAcc, float,
        ElementC, LayoutC, AlignC,
        ElementD, LayoutD, AlignD,
        EpilogueSchedule,
        FusionOp_
    >::CollectiveOp;

    static constexpr int kCarveoutBytes = static_cast<int>(sizeof(typename CollectiveEpilogue_::SharedStorage));
    using StageCountType_ = typename StageCountSelector<MainloopStages, kCarveoutBytes>::Type;

    using CollectiveMainloop_ = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignA,
        ElementB, LayoutB, AlignB,
        ElementAcc,
        TileShape_, ClusterShape_,
        StageCountType_,
        MainloopSchedule
    >::CollectiveOp;

    using GemmKernel_ = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop_,
        CollectiveEpilogue_>;

    using Gemm_ = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_>;

    static constexpr size_t kSmemBytes = sizeof(typename GemmKernel_::SharedStorage);

    static float run(void* d_A, void* d_B, void* d_C, void* d_D,
                     const float* d_bias,
                     int M, int N, int K,
                     cutlass::KernelHardwareInfo& hw_info,
                     size_t d_bytes) {
        if constexpr (kSmemBytes > cutlass::arch::sm100_smem_capacity_bytes) {
            return -3.0f;
        } else {
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

            arguments.epilogue.thread.alpha = 1.0f;
            arguments.epilogue.thread.beta = 0.0f;
            arguments.epilogue.thread.bias_ptr = d_bias;

            Gemm_ gemm;
            return run_gemm_timed(gemm, arguments, d_bytes, d_D);
        }
    }
};

// ── Grid search dispatch ──

struct ConfigResult {
    std::string tile_str;
    std::string cluster_str;
    std::string policy_str;
    float ms_gemm          = -1.0f;
    float ms_fused         = -1.0f;  // Primary fused: EVT / bias+GELU / bias
    float ms_fused_fp32    = -1.0f;  // PERIODIC_ADD only
    float ms_fused_bf16    = -1.0f;  // PERIODIC_ADD only
    float ms_post_only     = -1.0f;
    float ms_gemm_post     = -1.0f;
};

static float bench_postadd_only(
    __nv_bfloat16* d_D,
    const __nv_bfloat16* d_combined,
    int M, int N, int seq_len,
    size_t d_bytes
) {
    long long total_v8 = (long long)M * N / 8;
    int blocks = (int)((total_v8 + POSTADD_TPB - 1) / POSTADD_TPB);
    if (d_bytes > 0) CUDA_CHECK(cudaMemset(d_D, 0, d_bytes));

    for (int i = 0; i < WARMUP_ITERS; i++) {
        apply_combined<<<blocks, POSTADD_TPB>>>(d_D, d_combined, total_v8, N, seq_len);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    for (int i = 0; i < TIMED_ITERS; i++) {
        apply_combined<<<blocks, POSTADD_TPB>>>(d_D, d_combined, total_v8, N, seq_len);
    }
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaGetLastError());

    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= TIMED_ITERS;

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms;
}

static float bench_bias_act_only(
    __nv_bfloat16* d_D,
    const float* d_bias,
    int M, int N,
    size_t d_bytes,
    bool do_gelu
) {
    long long total_v8 = (long long)M * N / 8;
    int blocks = (int)((total_v8 + POSTADD_TPB - 1) / POSTADD_TPB);
    if (d_bytes > 0) CUDA_CHECK(cudaMemset(d_D, 0, d_bytes));

    for (int i = 0; i < WARMUP_ITERS; i++) {
        if (do_gelu)
            apply_bias_gelu<<<blocks, POSTADD_TPB>>>(d_D, d_bias, N, total_v8);
        else
            apply_bias_only<<<blocks, POSTADD_TPB>>>(d_D, d_bias, N, total_v8);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    for (int i = 0; i < TIMED_ITERS; i++) {
        if (do_gelu)
            apply_bias_gelu<<<blocks, POSTADD_TPB>>>(d_D, d_bias, N, total_v8);
        else
            apply_bias_only<<<blocks, POSTADD_TPB>>>(d_D, d_bias, N, total_v8);
    }
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaGetLastError());

    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= TIMED_ITERS;

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms;
}

// ── Validation ──

// Validate fused EVT output: spot-check strided samples against CPU reference.
static bool validate_fused_evt(void* d_D, int M, int N, int K, int seq_len) {
    using BF16 = cutlass::bfloat16_t;
    float gemm_even = 2.25f * K;
    float gemm_odd  = 1.50f * K;

    std::vector<BF16> h_row(N);
    CUDA_CHECK(cudaMemcpy(h_row.data(), d_D, N * sizeof(BF16), cudaMemcpyDeviceToHost));
    for (int c = 0; c < N; c++) {
        BF16 gemm_bf16((c & 1) ? gemm_odd : gemm_even);
        BF16 combined_val(float(c + 1) + 3.0f);
        BF16 expected = gemm_bf16 + combined_val;
        if (float(h_row[c]) != float(expected)) {
            printf("\n    ** EVT FAIL [0,%d]: got %.1f expected %.1f **", c,
                   float(h_row[c]), float(expected));
            return false;
        }
    }

    int check_rows[] = {1, 195, 196, 197, M / 2, M - 1};
    for (int ri = 0; ri < 6; ri++) {
        int r = check_rows[ri];
        if (r < 0 || r >= M) continue;
        int c = (r * 7 + 13) % N;
        BF16 h_val;
        CUDA_CHECK(cudaMemcpy(&h_val,
            reinterpret_cast<BF16*>(d_D) + (long long)r * N + c,
            sizeof(BF16), cudaMemcpyDeviceToHost));
        int pr = r % seq_len;
        BF16 gemm_bf16((c & 1) ? gemm_odd : gemm_even);
        BF16 combined_val(float(c + 1) + float(3 * (pr + 1)));
        BF16 expected = gemm_bf16 + combined_val;
        if (float(h_val) != float(expected)) {
            printf("\n    ** EVT FAIL [%d,%d]: got %.1f expected %.1f **", r, c,
                   float(h_val), float(expected));
            return false;
        }
    }
    return true;
}

static bool validate_bias_act_fused(void* d_D, int M, int N, int K, bool do_gelu) {
    using BF16 = cutlass::bfloat16_t;
    struct Spot { int r, c; };
    Spot spots[] = {{0,0}, {0,1}, {0,2}, {0,3}, {0,N-1},
                    {1,0}, {1,1}, {M/2,N/2}, {M-1,N-1}};

    for (auto& s : spots) {
        float b_val = (s.c & 1) ? 1.0f : 1.5f;
        float gemm = (float)K * 1.5f * b_val;
        float bias = (float)(s.c + 1);
        float pre_act = gemm + bias;
        float expected_f32 = do_gelu ? host_gelu_taylor(pre_act) : pre_act;
        BF16 expected_bf16(expected_f32);

        BF16 actual;
        CUDA_CHECK(cudaMemcpy(&actual,
            reinterpret_cast<BF16*>(d_D) + (long long)s.r * N + s.c,
            sizeof(BF16), cudaMemcpyDeviceToHost));

        if (float(actual) != float(expected_bf16)) {
            printf("\n    ** %s FAIL [%d,%d]: got %.1f expected %.1f (gemm=%.1f bias=%.1f) **",
                   do_gelu ? "FC1" : "FC2",
                   s.r, s.c, float(actual), float(expected_bf16), gemm, bias);
            return false;
        }
    }
    return true;
}

// ── Unified try_config_policy ──

template <
    int TM, int TN, int TK, int CM, int CN,
    typename MainloopSchedule,
    typename EpilogueSchedule,
    int MainloopStages
>
void try_config_policy(
    const char* tile_str, const char* cluster_str, const char* policy_str,
    void* d_A, void* d_B, void* d_C, void* d_D,
    const void* epilogue_data,  // d_combined (PERIODIC_ADD) or d_bias (GELU_BIAS/BIAS_ONLY)
    int M, int N, int K, int seq_len,
    cutlass::KernelHardwareInfo& hw_info,
    float ms_post_ref,
    size_t d_bytes,
    std::vector<ConfigResult>& results
) {
    printf("  %-13s %-5s %-20s  ", tile_str, cluster_str, policy_str);
    fflush(stdout);

    ConfigResult r;
    r.tile_str = tile_str;
    r.cluster_str = cluster_str;
    r.policy_str = policy_str;
    r.ms_post_only = ms_post_ref;

    // GEMM-only (beta=0)
#if BENCH_EPILOGUE == 1
    {
        using GI_f32 = GemmInstance<TM,TN,TK,CM,CN,float,MainloopSchedule,EpilogueSchedule,MainloopStages>;
        r.ms_gemm = GI_f32::run(d_A, d_B, d_C, d_D, M, N, K, hw_info, 1.0f, 0.0f, d_bytes);
    }
#else
    {
        using GI = GemmInstance<TM,TN,TK,CM,CN,cutlass::bfloat16_t,MainloopSchedule,EpilogueSchedule,MainloopStages>;
        r.ms_gemm = GI::run(d_A, d_B, d_C, d_D, M, N, K, hw_info, 1.0f, 0.0f, d_bytes);
    }
#endif

#if BENCH_EPILOGUE == 1
    {
        using GI_f32 = GemmInstance<TM,TN,TK,CM,CN,float,MainloopSchedule,EpilogueSchedule,MainloopStages>;
        using GI_bf16 = GemmInstance<TM,TN,TK,CM,CN,cutlass::bfloat16_t,MainloopSchedule,EpilogueSchedule,MainloopStages>;

        r.ms_fused_fp32 = r.ms_gemm >= 0
            ? GI_f32::run(d_A, d_B, d_C, d_D, M, N, K, hw_info, 1.0f, 1.0f, d_bytes)
            : -1.0f;

        r.ms_fused_bf16 = GI_bf16::run(d_A, d_B, d_C, d_D, M, N, K, hw_info, 1.0f, 1.0f, d_bytes);

        using FPI = FusedPeriodicGemmInstance<TM,TN,TK,CM,CN,MainloopSchedule,EpilogueSchedule,MainloopStages>;
        auto d_combined = reinterpret_cast<const __nv_bfloat16*>(epilogue_data);
        r.ms_fused = FPI::run(d_A, d_B, d_D, d_combined, M, N, K, seq_len, hw_info, d_bytes);

        if (r.ms_fused > 0 && !validate_fused_evt(d_D, M, N, K, seq_len))
            r.ms_fused = -2.0f;

        long long total_v8 = (long long)M * N / 8;
        int postadd_blocks = (int)((total_v8 + POSTADD_TPB - 1) / POSTADD_TPB);
        r.ms_gemm_post = GI_bf16::run_gemm_plus_post(
            d_A, d_B, d_C, d_D, M, N, K, hw_info, 1.0f, 0.0f, d_bytes,
            [&]() {
                apply_combined<<<postadd_blocks, POSTADD_TPB>>>(
                    reinterpret_cast<__nv_bfloat16*>(d_D),
                    d_combined, total_v8, N, seq_len);
            });
    }
#elif BENCH_EPILOGUE == 2
    {
        using BGI = BiasActGemmInstance<cutlass::epilogue::thread::GELU_taylor,TM,TN,TK,CM,CN,MainloopSchedule,EpilogueSchedule,MainloopStages>;
        auto d_bias = reinterpret_cast<const float*>(epilogue_data);
        r.ms_fused = BGI::run(d_A, d_B, d_C, d_D, d_bias, M, N, K, hw_info, d_bytes);

        if (r.ms_fused > 0 && !validate_bias_act_fused(d_D, M, N, K, true))
            r.ms_fused = -2.0f;

        long long total_v8 = (long long)M * N / 8;
        int blocks = (int)((total_v8 + POSTADD_TPB - 1) / POSTADD_TPB);
        using GI = GemmInstance<TM,TN,TK,CM,CN,cutlass::bfloat16_t,MainloopSchedule,EpilogueSchedule,MainloopStages>;
        r.ms_gemm_post = GI::run_gemm_plus_post(
            d_A, d_B, d_C, d_D, M, N, K, hw_info, 1.0f, 0.0f, d_bytes,
            [&]() {
                apply_bias_gelu<<<blocks, POSTADD_TPB>>>(
                    reinterpret_cast<__nv_bfloat16*>(d_D),
                    const_cast<float*>(d_bias), N, total_v8);
            });
    }
#elif BENCH_EPILOGUE == 3
    {
        using BOI = BiasActGemmInstance<cutlass::epilogue::thread::Identity,TM,TN,TK,CM,CN,MainloopSchedule,EpilogueSchedule,MainloopStages>;
        auto d_bias = reinterpret_cast<const float*>(epilogue_data);
        r.ms_fused = BOI::run(d_A, d_B, d_C, d_D, d_bias, M, N, K, hw_info, d_bytes);

        if (r.ms_fused > 0 && !validate_bias_act_fused(d_D, M, N, K, false))
            r.ms_fused = -2.0f;

        long long total_v8 = (long long)M * N / 8;
        int blocks = (int)((total_v8 + POSTADD_TPB - 1) / POSTADD_TPB);
        using GI = GemmInstance<TM,TN,TK,CM,CN,cutlass::bfloat16_t,MainloopSchedule,EpilogueSchedule,MainloopStages>;
        r.ms_gemm_post = GI::run_gemm_plus_post(
            d_A, d_B, d_C, d_D, M, N, K, hw_info, 1.0f, 0.0f, d_bytes,
            [&]() {
                apply_bias_only<<<blocks, POSTADD_TPB>>>(
                    reinterpret_cast<__nv_bfloat16*>(d_D),
                    const_cast<float*>(d_bias), N, total_v8);
            });
    }
#endif

    // Print inline results
    auto pr = [](float ms) {
        if (ms == -3.0f) printf(" >SMEM ");
        else if (ms == -2.0f) printf(" FAIL  ");
        else if (ms < 0) printf("  n/a  ");
        else printf("%6.3f ", ms);
    };

    if constexpr (EPI == Epilogue::PERIODIC_ADD) {
        if (r.ms_gemm == -3.0f) {
            printf("SKIP — SMEM exceeded\n");
        } else if (r.ms_gemm < 0 && r.ms_fused_bf16 < 0 && r.ms_fused < 0 && r.ms_gemm_post < 0) {
            printf("SKIP\n");
        } else {
            pr(r.ms_gemm); printf("/ ");
            pr(r.ms_fused_fp32); printf("/ ");
            pr(r.ms_fused_bf16); printf("/ ");
            pr(r.ms_fused); printf("/ ");
            pr(r.ms_post_only); printf("/ ");
            pr(r.ms_gemm_post);
            printf("ms\n");
        }
    } else {
        if (r.ms_gemm == -3.0f) {
            printf("SKIP — SMEM exceeded\n");
        } else if (r.ms_gemm < 0 && r.ms_fused < 0) {
            printf("SKIP\n");
        } else {
            pr(r.ms_gemm); printf("/ ");
            pr(r.ms_fused); printf("/ ");
            pr(r.ms_gemm_post);
            printf("ms\n");
        }
    }

    results.push_back(r);
}

// ── Unified try_config_family ──

template <int TM, int TN, int TK, int CM, int CN>
void try_config_family(
    void* d_A, void* d_B, void* d_C, void* d_D,
    const void* epilogue_data,
    int M, int N, int K, int seq_len,
    cutlass::KernelHardwareInfo& hw_info,
    float ms_post_ref,
    size_t d_bytes,
    std::vector<ConfigResult>& results
) {
    const std::string tile_name = std::to_string(TM) + "x" + std::to_string(TN) + "x" + std::to_string(TK);
    const std::string clust_name = std::to_string(CM) + "x" + std::to_string(CN);

    // Auto policy (baseline)
    try_config_policy<TM,TN,TK,CM,CN,
        cutlass::gemm::collective::KernelScheduleAuto,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        0>(
            tile_name.c_str(),
            clust_name.c_str(),
            "auto/auto/Sauto",
            d_A, d_B, d_C, d_D, epilogue_data,
            M, N, K, seq_len, hw_info, ms_post_ref, d_bytes, results);

    // Explicit policy dispatch — SMEM overflow is handled gracefully inside each
    // Instance::run() via if constexpr on sizeof(SharedStorage), returning -3.0f
    // instead of hitting CUTLASS's static_assert.

#if CUTLASS_COMPREHENSIVE_SWEEP
    if constexpr (CM == 1 && CN == 1 && TM <= 128) {
        try_config_policy<TM,TN,TK,CM,CN,
            cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
            cutlass::epilogue::TmaWarpSpecialized1Sm,
            0>(
                tile_name.c_str(),
                clust_name.c_str(),
                "1sm/1sm/Sauto",
                d_A, d_B, d_C, d_D, epilogue_data,
                M, N, K, seq_len, hw_info, ms_post_ref, d_bytes, results);
        try_config_policy<TM,TN,TK,CM,CN,
            cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
            cutlass::epilogue::TmaWarpSpecialized1Sm,
            3>(
                tile_name.c_str(),
                clust_name.c_str(),
                "1sm/1sm/S3",
                d_A, d_B, d_C, d_D, epilogue_data,
                M, N, K, seq_len, hw_info, ms_post_ref, d_bytes, results);
        try_config_policy<TM,TN,TK,CM,CN,
            cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
            cutlass::epilogue::TmaWarpSpecialized1Sm,
            4>(
                tile_name.c_str(),
                clust_name.c_str(),
                "1sm/1sm/S4",
                d_A, d_B, d_C, d_D, epilogue_data,
                M, N, K, seq_len, hw_info, ms_post_ref, d_bytes, results);
    }

    if constexpr (TM >= 128 && (CM % 2 == 0)) {
        try_config_policy<TM,TN,TK,CM,CN,
            cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            0>(
                tile_name.c_str(),
                clust_name.c_str(),
                "2sm/2sm/Sauto",
                d_A, d_B, d_C, d_D, epilogue_data,
                M, N, K, seq_len, hw_info, ms_post_ref, d_bytes, results);
        try_config_policy<TM,TN,TK,CM,CN,
            cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            3>(
                tile_name.c_str(),
                clust_name.c_str(),
                "2sm/2sm/S3",
                d_A, d_B, d_C, d_D, epilogue_data,
                M, N, K, seq_len, hw_info, ms_post_ref, d_bytes, results);
        try_config_policy<TM,TN,TK,CM,CN,
            cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            4>(
                tile_name.c_str(),
                clust_name.c_str(),
                "2sm/2sm/S4",
                d_A, d_B, d_C, d_D, epilogue_data,
                M, N, K, seq_len, hw_info, ms_post_ref, d_bytes, results);
#if CUTLASS_EXTENDED_SWEEP
        try_config_policy<TM,TN,TK,CM,CN,
            cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            5>(
                tile_name.c_str(),
                clust_name.c_str(),
                "2sm/2sm/S5",
                d_A, d_B, d_C, d_D, epilogue_data,
                M, N, K, seq_len, hw_info, ms_post_ref, d_bytes, results);
#endif
    }
#endif
}

#define TRY(tm,tn,tk,cm,cn) \
    try_config_family<tm,tn,tk,cm,cn>( \
        d_A, d_B, d_C, d_D, epilogue_data, \
        M, N_DIM, K_DIM, SEQ_LEN, hw_info, ms_post_ref, sz_d, results)

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
    const int SEQ_LEN  = 196;

    cudaDeviceProp props;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    const int SM_COUNT = props.multiProcessorCount;

    int imgs_per_sm = 32;
    if (argc > 1) imgs_per_sm = atoi(argv[1]);
    const int M = imgs_per_sm * SM_COUNT * SEQ_LEN;

    double flops = 2.0 * M * N_DIM * K_DIM;

    const char* epi_name =
        EPI == Epilogue::PERIODIC_ADD ? "PERIODIC_ADD (patch embed)" :
        EPI == Epilogue::GELU_BIAS    ? "GELU_BIAS (FC1)" :
                                        "BIAS_ONLY (FC2)";

    printf("CUTLASS SM100a FP8 Grid Search — %s\n", epi_name);
    printf("  [%d, %d] x [%d, %d]^T  (imgs_per_sm=%d, %d images)\n",
           M, K_DIM, K_DIM, N_DIM, imgs_per_sm, M / SEQ_LEN);
    printf("  FP8 E4M3 (per-tensor) -> BF16, acc FP32\n");
    printf("  Device: %s (SM %d.%d, SMs=%d)\n\n", props.name, props.major, props.minor, SM_COUNT);

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    // ── Allocate data ──
    size_t sz_a  = (size_t)M * K_DIM;
    size_t sz_b  = (size_t)N_DIM * K_DIM;
    size_t sz_d  = (size_t)M * N_DIM * sizeof(ElementD);

    void *d_A = nullptr, *d_B = nullptr;
    void *d_C = nullptr, *d_D = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, sz_a));
    CUDA_CHECK(cudaMalloc(&d_B, sz_b));
    CUDA_CHECK(cudaMalloc(&d_C, sz_d));
    CUDA_CHECK(cudaMalloc(&d_D, sz_d));

    CUDA_CHECK(cudaMemset(d_A, 0x3C, sz_a));
    // B: alternating columns — even cols = 0x3C (FP8 1.5), odd cols = 0x38 (FP8 1.0)
    {
        uint8_t* h_B = (uint8_t*)malloc(sz_b);
        for (int n = 0; n < N_DIM; n++)
            memset(h_B + (size_t)n * K_DIM, (n & 1) ? 0x38 : 0x3C, K_DIM);
        CUDA_CHECK(cudaMemcpy(d_B, h_B, sz_b, cudaMemcpyHostToDevice));
        free(h_B);
    }
    CUDA_CHECK(cudaMemset(d_C, 0, sz_d));
    CUDA_CHECK(cudaMemset(d_D, 0, sz_d));

    // epilogue_data: d_combined for PERIODIC_ADD, d_bias for GELU_BIAS/BIAS_ONLY
    const void* epilogue_data = nullptr;
    float ms_post_ref = -1.0f;

    if constexpr (EPI == Epilogue::PERIODIC_ADD) {
        float *d_bias, *d_pos;
        __nv_bfloat16 *d_combined;
        CUDA_CHECK(cudaMalloc(&d_bias, (size_t)N_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pos, (size_t)SEQ_LEN * N_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_combined, (size_t)SEQ_LEN * N_DIM * sizeof(__nv_bfloat16)));
        {
            float* h_bias = (float*)malloc((size_t)N_DIM * sizeof(float));
            float* h_pos  = (float*)malloc((size_t)SEQ_LEN * N_DIM * sizeof(float));
            for (int c = 0; c < N_DIM; c++)
                h_bias[c] = (float)(c + 1);
            for (int r = 0; r < SEQ_LEN; r++)
                for (int c = 0; c < N_DIM; c++)
                    h_pos[r * N_DIM + c] = (float)((r + 1) * 3);
            CUDA_CHECK(cudaMemcpy(d_bias, h_bias, (size_t)N_DIM * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_pos, h_pos, (size_t)SEQ_LEN * N_DIM * sizeof(float), cudaMemcpyHostToDevice));
            free(h_bias);
            free(h_pos);
        }
        {
            int n_elem = SEQ_LEN * N_DIM;
            int tpb = 256;
            precompute_combined<<<(n_elem + tpb - 1) / tpb, tpb>>>(d_bias, d_pos, d_combined, SEQ_LEN, N_DIM);
            CUDA_CHECK(cudaGetLastError());
        }
        // Tile combined into C
        {
            int num_images = M / SEQ_LEN;
            size_t row_bytes = (size_t)SEQ_LEN * N_DIM * sizeof(__nv_bfloat16);
            for (int img = 0; img < num_images; img++) {
                CUDA_CHECK(cudaMemcpy(
                    (char*)d_C + img * row_bytes,
                    d_combined,
                    row_bytes,
                    cudaMemcpyDeviceToDevice));
            }
        }
        printf("  C tiled to [%d, %d] (%.1f MB)\n\n", M, N_DIM, (double)sz_d / (1024*1024));

        ms_post_ref = bench_postadd_only(
            reinterpret_cast<__nv_bfloat16*>(d_D), d_combined,
            M, N_DIM, SEQ_LEN, sz_d);
        printf("  PostAdd-only baseline: %.3f ms\n", ms_post_ref);

        epilogue_data = d_combined;
        // d_bias, d_pos, d_combined kept alive until end
    } else {
        // GELU_BIAS or BIAS_ONLY: allocate bias
        float* d_bias;
        CUDA_CHECK(cudaMalloc(&d_bias, (size_t)N_DIM * sizeof(float)));
        {
            float* h_bias = (float*)malloc((size_t)N_DIM * sizeof(float));
            for (int c = 0; c < N_DIM; c++) h_bias[c] = (float)(c + 1);
            CUDA_CHECK(cudaMemcpy(d_bias, h_bias, (size_t)N_DIM * sizeof(float), cudaMemcpyHostToDevice));
            free(h_bias);
        }

        bool do_gelu = (EPI == Epilogue::GELU_BIAS);
        ms_post_ref = bench_bias_act_only(
            reinterpret_cast<__nv_bfloat16*>(d_D), d_bias,
            M, N_DIM, sz_d, do_gelu);
        printf("  Unfused %s kernel: %.3f ms\n",
               do_gelu ? "bias+GELU" : "bias-only", ms_post_ref);

        epilogue_data = d_bias;
    }

    // ── Grid/policy search ──
    std::vector<ConfigResult> results;

#if CUTLASS_EXTENDED_SWEEP
    printf("Sweeping extended tile list with policy variants...\n");
#else
    printf("Sweeping standard tile list with policy variants...\n");
#endif

    if constexpr (EPI == Epilogue::PERIODIC_ADD) {
        // Patch embed tile configs
        TRY(256,  64,  64, 2, 1);
        TRY(256,  64, 128, 2, 1);
        TRY(256, 128,  64, 2, 1);
        TRY(256, 256,  64, 2, 1);
        TRY(256, 128, 128, 2, 1);
        TRY(256, 256, 128, 2, 1);
        TRY(256, 192,  64, 2, 1);
        TRY(256, 192, 128, 2, 1);
        TRY(128, 128,  64, 2, 1);
        TRY(128, 256,  64, 2, 1);
        TRY(128, 128, 128, 2, 1);
        TRY(128, 256, 128, 2, 1);
        TRY(128, 192,  64, 2, 1);
        TRY(128, 192, 128, 2, 1);
        TRY(256, 128,  64, 2, 2);
        TRY(256, 256,  64, 2, 2);
        TRY(256, 256, 128, 2, 2);
        TRY(256, 192,  64, 2, 2);
        TRY(256, 192, 128, 2, 2);

        TRY( 64, 128,  64, 1, 1);
        TRY( 64, 256,  64, 1, 1);
        TRY( 64, 128, 128, 1, 1);
        TRY( 64, 256, 128, 1, 1);
        TRY(128, 128,  64, 1, 1);
        TRY(128, 256,  64, 1, 1);
        TRY(128, 192,  64, 1, 1);

#if CUTLASS_EXTENDED_SWEEP
        TRY(256, 128, 128, 2, 2);
        TRY(128, 128,  64, 2, 2);
        TRY(128, 256,  64, 2, 2);
        TRY(128, 128, 128, 2, 2);
        TRY(128, 256, 128, 2, 2);
        TRY(128, 192, 128, 2, 2);

        TRY(128, 128, 128, 1, 1);
        TRY(128, 256, 128, 1, 1);
        TRY(128, 192, 128, 1, 1);
#endif
    } else {
        // FC1/FC2 tile configs
        TRY(256, 256, 128, 2, 1);
        TRY(256, 128, 128, 2, 1);
        TRY(256, 192, 128, 2, 1);
        TRY(128, 256, 128, 2, 1);
        TRY(128, 128, 128, 2, 1);
        TRY(128, 192, 128, 2, 1);
        TRY(256, 256,  64, 2, 1);
        TRY(256, 128,  64, 2, 1);

        TRY( 64, 256, 128, 1, 1);
        TRY( 64, 128, 128, 1, 1);
        TRY(128, 256,  64, 1, 1);
        TRY(128, 128,  64, 1, 1);

#if CUTLASS_EXTENDED_SWEEP
        // Extended 2SM configs (cluster_n=2)
        TRY(256, 256, 128, 2, 2);
        TRY(256, 128, 128, 2, 2);
        TRY(256, 192, 128, 2, 2);
        TRY(128, 256, 128, 2, 2);
        TRY(128, 128, 128, 2, 2);
        TRY(128, 192, 128, 2, 2);

        // Extended K=64 variants
        TRY(256, 192,  64, 2, 1);
        TRY(128, 192,  64, 2, 1);
        TRY(128, 128,  64, 2, 1);
        TRY(128, 192,  64, 1, 1);

        // Extended 1SM configs
        TRY(128, 128, 128, 1, 1);
        TRY(128, 256, 128, 1, 1);
        TRY(128, 192, 128, 1, 1);
        TRY( 64, 192, 128, 1, 1);
#endif
    }

    // ── Sort by primary fused metric ──
    std::sort(results.begin(), results.end(), [](const ConfigResult& a, const ConfigResult& b) {
        float av = a.ms_fused > 0 ? a.ms_fused : 1e9f;
        float bv = b.ms_fused > 0 ? b.ms_fused : 1e9f;
        if (av != bv) return av < bv;
        float a2 = a.ms_gemm_post > 0 ? a.ms_gemm_post : 1e9f;
        float b2 = b.ms_gemm_post > 0 ? b.ms_gemm_post : 1e9f;
        if (a2 != b2) return a2 < b2;
        float a3 = a.ms_gemm > 0 ? a.ms_gemm : 1e9f;
        float b3 = b.ms_gemm > 0 ? b.ms_gemm : 1e9f;
        return a3 < b3;
    });

    // ── Results table ──
    auto tflops = [&](float ms) -> float {
        return ms > 0 ? (float)(flops / ms / 1e9) : 0.0f;
    };

    printf("\n");

    if constexpr (EPI == Epilogue::PERIODIC_ADD) {
        printf("CUTLASS SM100a FP8 Policy Search — [%d, %d] x [%d, %d]^T\n", M, K_DIM, K_DIM, N_DIM);
        printf("Total variants tested: %zu\n", results.size());
        printf("═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n");
        printf("  %-13s %-5s %-20s | %-18s | %-18s | %-18s | %-18s | %-10s | %-18s\n",
               "Tile", "Clust", "Policy", "GEMM-only", "Fused FP32", "Fused BF16", "Fused EVT", "PostAdd", "GEMM+PostAdd");
        printf("  %-13s %-5s %-20s | %7s %9s | %7s %9s | %7s %9s | %7s %9s | %7s | %7s %9s\n",
               "", "", "", "ms", "TFLOPS", "ms", "TFLOPS", "ms", "TFLOPS", "ms", "TFLOPS", "ms", "ms", "eTFLOPS");
        printf("──────────────────────────────────────────────+────────────────────+────────────────────+────────────────────+────────────────────+────────────+────────────────────\n");

        auto print_cell = [&](float ms) {
            if (ms == -3.0f) printf("   >SMEM       n/a ");
            else if (ms == -2.0f) printf("    FAIL       n/a ");
            else if (ms < 0) printf("     n/a       n/a ");
            else printf("  %6.3f  %7.0f ", ms, tflops(ms));
        };

        for (auto& r : results) {
            printf("  %-13s %-5s %-20s |", r.tile_str.c_str(), r.cluster_str.c_str(), r.policy_str.c_str());
            print_cell(r.ms_gemm);      printf("|");
            print_cell(r.ms_fused_fp32); printf("|");
            print_cell(r.ms_fused_bf16); printf("|");
            print_cell(r.ms_fused);      printf("|");
            if (r.ms_post_only < 0) printf("   n/a    |");
            else printf(" %7.3f |", r.ms_post_only);
            print_cell(r.ms_gemm_post);
            printf("\n");
        }

        printf("──────────────────────────────────────────────+────────────────────+────────────────────+────────────────────+────────────────────+────────────+────────────────────\n");

        const ConfigResult* best_gemm = nullptr;
        const ConfigResult* best_fused_fp32 = nullptr;
        const ConfigResult* best_fused_bf16 = nullptr;
        const ConfigResult* best_fused = nullptr;
        const ConfigResult* best_gemm_post = nullptr;

        for (auto& r : results) {
            if (r.ms_gemm > 0 && (!best_gemm || r.ms_gemm < best_gemm->ms_gemm))
                best_gemm = &r;
            if (r.ms_fused_fp32 > 0 && (!best_fused_fp32 || r.ms_fused_fp32 < best_fused_fp32->ms_fused_fp32))
                best_fused_fp32 = &r;
            if (r.ms_fused_bf16 > 0 && (!best_fused_bf16 || r.ms_fused_bf16 < best_fused_bf16->ms_fused_bf16))
                best_fused_bf16 = &r;
            if (r.ms_fused > 0 && (!best_fused || r.ms_fused < best_fused->ms_fused))
                best_fused = &r;
            if (r.ms_gemm_post > 0 && (!best_gemm_post || r.ms_gemm_post < best_gemm_post->ms_gemm_post))
                best_gemm_post = &r;
        }

        if (best_gemm)
            printf("  Best GEMM:        %s %s %s  %.3f ms / %.0f TFLOPS\n",
                   best_gemm->tile_str.c_str(), best_gemm->cluster_str.c_str(),
                   best_gemm->policy_str.c_str(),
                   best_gemm->ms_gemm, tflops(best_gemm->ms_gemm));
        if (best_fused_fp32)
            printf("  Best Fused FP32:  %s %s %s  %.3f ms / %.0f TFLOPS\n",
                   best_fused_fp32->tile_str.c_str(), best_fused_fp32->cluster_str.c_str(),
                   best_fused_fp32->policy_str.c_str(),
                   best_fused_fp32->ms_fused_fp32, tflops(best_fused_fp32->ms_fused_fp32));
        if (best_fused_bf16)
            printf("  Best Fused BF16:  %s %s %s  %.3f ms / %.0f TFLOPS\n",
                   best_fused_bf16->tile_str.c_str(), best_fused_bf16->cluster_str.c_str(),
                   best_fused_bf16->policy_str.c_str(),
                   best_fused_bf16->ms_fused_bf16, tflops(best_fused_bf16->ms_fused_bf16));
        if (best_fused)
            printf("  Best Fused EVT:   %s %s %s  %.3f ms / %.0f TFLOPS\n",
                   best_fused->tile_str.c_str(), best_fused->cluster_str.c_str(),
                   best_fused->policy_str.c_str(),
                   best_fused->ms_fused, tflops(best_fused->ms_fused));
        if (best_gemm_post)
            printf("  Best GEMM+PostAdd:%s %s %s  %.3f ms / %.0f TFLOPS\n",
                   best_gemm_post->tile_str.c_str(), best_gemm_post->cluster_str.c_str(),
                   best_gemm_post->policy_str.c_str(),
                   best_gemm_post->ms_gemm_post, tflops(best_gemm_post->ms_gemm_post));
    } else {
        const char* fused_label = (EPI == Epilogue::GELU_BIAS) ? "Fused bias+GELU" : "Fused bias";
        const char* post_label  = (EPI == Epilogue::GELU_BIAS) ? "bias+GELU" : "bias-only";

        printf("%s SM100a FP8 — [%d, %d] x [%d, %d]^T + %s\n",
               (EPI == Epilogue::GELU_BIAS) ? "FC1" : "FC2",
               M, K_DIM, K_DIM, N_DIM, post_label);
        printf("Total variants tested: %zu\n", results.size());
        printf("════════════════════════════════════════════════════════════════════════════════════════════════════════════\n");
        printf("  %-13s %-5s %-20s | %-18s | %-18s | %-18s\n",
               "Tile", "Clust", "Policy", "GEMM-only", fused_label, "GEMM+unfused");
        printf("  %-13s %-5s %-20s | %7s %9s | %7s %9s | %7s %9s\n",
               "", "", "", "ms", "TFLOPS", "ms", "TFLOPS", "ms", "eTFLOPS");
        printf("──────────────────────────────────────────────+────────────────────+────────────────────+────────────────────\n");

        auto print_cell = [&](float ms) {
            if (ms == -3.0f) printf("   >SMEM       n/a ");
            else if (ms == -2.0f) printf("    FAIL       n/a ");
            else if (ms < 0) printf("     n/a       n/a ");
            else printf("  %6.3f  %7.0f ", ms, tflops(ms));
        };

        for (auto& r : results) {
            printf("  %-13s %-5s %-20s |", r.tile_str.c_str(), r.cluster_str.c_str(), r.policy_str.c_str());
            print_cell(r.ms_gemm);  printf("|");
            print_cell(r.ms_fused); printf("|");
            print_cell(r.ms_gemm_post);
            printf("\n");
        }
        printf("──────────────────────────────────────────────+────────────────────+────────────────────+────────────────────\n");

        const ConfigResult* best_gemm = nullptr;
        const ConfigResult* best_fused = nullptr;
        const ConfigResult* best_gemm_post = nullptr;

        for (auto& r : results) {
            if (r.ms_gemm > 0 && (!best_gemm || r.ms_gemm < best_gemm->ms_gemm))
                best_gemm = &r;
            if (r.ms_fused > 0 && (!best_fused || r.ms_fused < best_fused->ms_fused))
                best_fused = &r;
            if (r.ms_gemm_post > 0 && (!best_gemm_post || r.ms_gemm_post < best_gemm_post->ms_gemm_post))
                best_gemm_post = &r;
        }

        if (best_gemm)
            printf("  Best GEMM:         %s %s %s  %.3f ms / %.0f TFLOPS\n",
                   best_gemm->tile_str.c_str(), best_gemm->cluster_str.c_str(),
                   best_gemm->policy_str.c_str(),
                   best_gemm->ms_gemm, tflops(best_gemm->ms_gemm));
        if (best_fused)
            printf("  Best Fused:        %s %s %s  %.3f ms / %.0f TFLOPS\n",
                   best_fused->tile_str.c_str(), best_fused->cluster_str.c_str(),
                   best_fused->policy_str.c_str(),
                   best_fused->ms_fused, tflops(best_fused->ms_fused));
        if (best_gemm_post)
            printf("  Best GEMM+unfused: %s %s %s  %.3f ms / %.0f TFLOPS\n",
                   best_gemm_post->tile_str.c_str(), best_gemm_post->cluster_str.c_str(),
                   best_gemm_post->policy_str.c_str(),
                   best_gemm_post->ms_gemm_post, tflops(best_gemm_post->ms_gemm_post));
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    // epilogue_data (d_combined/d_bias) and related allocs leak at exit — acceptable for a benchmark

    return 0;
#endif
}
