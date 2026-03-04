// CUTLASS 4.x SM100a per-tensor FP8 GEMM policy + tile grid search
// ═══════════════════════════════════════════════════
//
// Sweeps tile/cluster configs plus selected schedule/stage policy variants.
// Six measurements per config:
//   1. GEMM only (beta=0, FP32 epilogue): pure compute
//   2. Fused FP32 (beta=1, FP32 epilogue): D = float(acc) + float(C) → BF16
//   3. Fused BF16 (beta=1, BF16 epilogue): D = bf16(acc) + C → BF16
//   4. Fused EVT: custom EVT visitor fuses periodic table add in epilogue
//   5. PostAdd only: apply_combined kernel on D (unfused)
//   6. GEMM+PostAdd: GEMM(beta=0,BF16 epilogue) + apply_combined
//
// #4 is the truest apples-to-apples comparison: single-kernel GEMM with
// the periodic [seq_len, N] table add fused into the epilogue via EVT.
// #6 is the unfused baseline (two kernels: GEMM + apply_combined).
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
#include "siglip_periodic_add.hpp"

using namespace cute;

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

// ── GemmInstance: full CUTLASS type chain parameterized on tile, cluster, epilogue compute type ──
// EpiCompute controls ElementCompute and ElementScalar in LinearCombination.
// float  → FP32 epilogue: D = float(acc) + float(C)
// bf16   → BF16 epilogue: D = bf16(acc) + C  (matches custom kernel)

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

    // Returns average ms, or -1.0f if config is invalid.
    static float run(void* d_A, void* d_B, void* d_C, void* d_D,
                     int M, int N, int K,
                     cutlass::KernelHardwareInfo& hw_info,
                     float alpha_f, float beta_f,
                     size_t d_bytes) {
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

        if (d_bytes > 0) CUDA_CHECK(cudaMemset(d_D, 0, d_bytes));

        for (int i = 0; i < WARMUP_ITERS; i++) {
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
        for (int i = 0; i < TIMED_ITERS; i++) gemm.run();
        cudaEventRecord(t1);
        CUDA_CHECK(cudaEventSynchronize(t1));

        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        ms /= TIMED_ITERS;

        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        if (d_workspace) cudaFree(d_workspace);

        return ms;
    }

    // GEMM (beta=0 recommended) + unfused periodic add kernel.
    // Returns average total time per iteration.
    static float run_gemm_plus_postadd(
        void* d_A, void* d_B, void* d_C, void* d_D,
        const __nv_bfloat16* d_combined,
        int M, int N, int K, int seq_len,
        cutlass::KernelHardwareInfo& hw_info,
        float alpha_f, float beta_f,
        size_t d_bytes
    ) {
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

        long long total_v8 = (long long)M * N / 8;
        int postadd_blocks = (int)((total_v8 + POSTADD_TPB - 1) / POSTADD_TPB);
        if (d_bytes > 0) CUDA_CHECK(cudaMemset(d_D, 0, d_bytes));

        for (int i = 0; i < WARMUP_ITERS; i++) {
            if (gemm.run() != cutlass::Status::kSuccess) {
                if (d_workspace) cudaFree(d_workspace);
                return -1.0f;
            }
            apply_combined<<<postadd_blocks, POSTADD_TPB>>>(
                reinterpret_cast<__nv_bfloat16*>(d_D),
                d_combined,
                total_v8, N, seq_len);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);

        cudaEventRecord(t0);
        for (int i = 0; i < TIMED_ITERS; i++) {
            gemm.run();
            apply_combined<<<postadd_blocks, POSTADD_TPB>>>(
                reinterpret_cast<__nv_bfloat16*>(d_D),
                d_combined,
                total_v8, N, seq_len);
        }
        cudaEventRecord(t1);
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaGetLastError());

        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        ms /= TIMED_ITERS;

        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        if (d_workspace) cudaFree(d_workspace);
        return ms;
    }
};

// ── FusedPeriodicGemmInstance: GEMM + periodic table add fused in epilogue via custom EVT ──
// Uses SigLipPeriodicAdd (Sm100PeriodicAddNode + Sm90AccFetch) as FusionCallbacks.
// ElementC = void — no source matrix load, no beta*C overhead.
// The periodic [seq_len, N] table is loaded from L2 via __ldg() inside visit().

template <
    int TM, int TN, int TK, int CM, int CN,
    typename MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto,
    typename EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto,
    int MainloopStages = 0
>
struct FusedPeriodicGemmInstance {
    using TileShape_    = Shape<Int<TM>, Int<TN>, Int<TK>>;
    using ClusterShape_ = Shape<Int<CM>, Int<CN>, _1>;

    // Custom EVT: acc + periodic table → BF16 output
    using FusionOp_ = cutlass::epilogue::fusion::SigLipPeriodicAdd<>;

    using CollectiveEpilogue_ = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        TileShape_, ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAcc, float,
        void, LayoutD, AlignD,       // void C — no source load; LayoutD/AlignD as placeholders
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

    static float run(void* d_A, void* d_B, void* d_D,
                     const __nv_bfloat16* d_combined,
                     int M, int N, int K, int seq_len,
                     cutlass::KernelHardwareInfo& hw_info,
                     size_t d_bytes) {
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
                {},                                          // thread (EVT fusion args — set below)
                nullptr, stride_c,                           // C ptr unused (void C)
                reinterpret_cast<ElementD*>(d_D), stride_d
            },
            hw_info
        };

        // Set periodic add table pointer and period
        // EVT tree: Sm90VisitorImplBase<Op0=Sm90AccFetch, Op1=Sm100PeriodicAddNode>
        // Arguments struct: { op_0 (AccFetch, empty), op_1 (PeriodicAddNode) }
        arguments.epilogue.thread.op_1.ptr_combined =
            reinterpret_cast<cutlass::bfloat16_t const*>(d_combined);
        arguments.epilogue.thread.op_1.seq_len = seq_len;

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

        if (d_bytes > 0) CUDA_CHECK(cudaMemset(d_D, 0, d_bytes));

        for (int i = 0; i < WARMUP_ITERS; i++) {
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
        for (int i = 0; i < TIMED_ITERS; i++) gemm.run();
        cudaEventRecord(t1);
        CUDA_CHECK(cudaEventSynchronize(t1));

        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        ms /= TIMED_ITERS;

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
    std::string policy_str;
    float ms_gemm;
    float ms_fused_fp32;
    float ms_fused_bf16;
    float ms_fused_periodic;
    float ms_postadd;
    float ms_gemm_plus_postadd;
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

// Validate fused EVT output: spot-check strided samples against CPU reference.
// Data init: A=0x3C (FP8 1.5), B alternating cols (even=0x3C/1.5, odd=0x38/1.0),
// bias[c]=c+1, pos[r][c]=3*(r+1).
// Expected: D[m,n] = bf16(bf16(gemm_val) + bf16(bias[n] + pos[m%seq_len][n]))
//   where gemm_val = K*1.5*1.5 = 1728.0 (even cols) or K*1.5*1.0 = 1152.0 (odd cols)
static bool validate_fused_evt(void* d_D, int M, int N, int K, int seq_len) {
    using BF16 = cutlass::bfloat16_t;
    float gemm_even = 2.25f * K;  // 1728.0
    float gemm_odd  = 1.50f * K;  // 1152.0

    // Full row 0: all N columns — catches N-indexing and tile-boundary bugs
    std::vector<BF16> h_row(N);
    CUDA_CHECK(cudaMemcpy(h_row.data(), d_D, N * sizeof(BF16), cudaMemcpyDeviceToHost));
    for (int c = 0; c < N; c++) {
        BF16 gemm_bf16((c & 1) ? gemm_odd : gemm_even);
        BF16 combined_val(float(c + 1) + 3.0f);  // row 0: pos_row=0, 3*(0+1)=3
        BF16 expected = gemm_bf16 + combined_val;
        if (float(h_row[c]) != float(expected)) {
            printf("\n    ** EVT FAIL [0,%d]: got %.1f expected %.1f **", c,
                   float(h_row[c]), float(expected));
            return false;
        }
    }

    // Strided rows: catches M-indexing and periodic wrap at seq_len boundary
    int check_rows[] = {1, 195, 196, 197, M / 2, M - 1};
    for (int ri = 0; ri < 6; ri++) {
        int r = check_rows[ri];
        if (r < 0 || r >= M) continue;
        int c = (r * 7 + 13) % N;  // varying column per row
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

template <
    int TM, int TN, int TK, int CM, int CN,
    typename MainloopSchedule,
    typename EpilogueSchedule,
    int MainloopStages
>
void try_config_policy(
    const char* tile_str, const char* cluster_str, const char* policy_str,
    void* d_A, void* d_B, void* d_C, void* d_D,
    const __nv_bfloat16* d_combined,
    int M, int N, int K, int seq_len,
    cutlass::KernelHardwareInfo& hw_info,
    float ms_postadd_ref,
    size_t d_bytes,
    std::vector<ConfigResult>& results
) {
    printf("  %-13s %-5s %-20s  ", tile_str, cluster_str, policy_str);
    fflush(stdout);

    using GI_f32 = GemmInstance<TM,TN,TK,CM,CN,float,MainloopSchedule,EpilogueSchedule,MainloopStages>;
    using GI_bf16 = GemmInstance<TM,TN,TK,CM,CN,cutlass::bfloat16_t,MainloopSchedule,EpilogueSchedule,MainloopStages>;

    float ms_gemm = GI_f32::run(
        d_A, d_B, d_C, d_D, M, N, K, hw_info, 1.0f, 0.0f, d_bytes);

    float ms_fused_fp32 = ms_gemm >= 0
        ? GI_f32::run(
              d_A, d_B, d_C, d_D, M, N, K, hw_info, 1.0f, 1.0f, d_bytes)
        : -1.0f;

    float ms_fused_bf16 = GI_bf16::run(
        d_A, d_B, d_C, d_D, M, N, K, hw_info, 1.0f, 1.0f, d_bytes);

    using FPI = FusedPeriodicGemmInstance<TM,TN,TK,CM,CN,MainloopSchedule,EpilogueSchedule,MainloopStages>;
    float ms_fused_periodic = FPI::run(
        d_A, d_B, d_D, d_combined,
        M, N, K, seq_len, hw_info, d_bytes);

    if (ms_fused_periodic > 0 && !validate_fused_evt(d_D, M, N, K, seq_len))
        ms_fused_periodic = -2.0f;

    float ms_postadd = ms_postadd_ref;

    float ms_gemm_plus_postadd = GI_bf16::run_gemm_plus_postadd(
        d_A, d_B, d_C, d_D, d_combined,
        M, N, K, seq_len,
        hw_info, 1.0f, 0.0f, d_bytes);

    if (ms_gemm < 0 && ms_fused_bf16 < 0 && ms_fused_periodic < 0 && ms_gemm_plus_postadd < 0) {
        printf("SKIP\n");
    } else {
        auto pr = [](float ms) {
            if (ms == -2.0f) printf(" FAIL  ");
            else if (ms < 0) printf("  n/a  ");
            else printf("%6.3f ", ms);
        };
        pr(ms_gemm); printf("/ ");
        pr(ms_fused_fp32); printf("/ ");
        pr(ms_fused_bf16); printf("/ ");
        pr(ms_fused_periodic); printf("/ ");
        pr(ms_postadd); printf("/ ");
        pr(ms_gemm_plus_postadd);
        printf("ms\n");
    }

    results.push_back({
        tile_str, cluster_str, policy_str,
        ms_gemm, ms_fused_fp32, ms_fused_bf16,
        ms_fused_periodic,
        ms_postadd, ms_gemm_plus_postadd
    });
}

template <int TM, int TN, int TK, int CM, int CN>
void try_config_family(
    void* d_A, void* d_B, void* d_C, void* d_D,
    const __nv_bfloat16* d_combined,
    int M, int N, int K, int seq_len,
    cutlass::KernelHardwareInfo& hw_info,
    float ms_postadd_ref,
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
            d_A, d_B, d_C, d_D, d_combined,
            M, N, K, seq_len, hw_info, ms_postadd_ref, d_bytes, results);

#if CUTLASS_COMPREHENSIVE_SWEEP
    // Explicit 1SM policy:
    // Keep this narrow to avoid invalid/oversized instantiations in the sweep.
    // 1SM schedules are only attempted on true 1x1 cluster configs.
    if constexpr (CM == 1 && CN == 1 && TM <= 128) {
        try_config_policy<TM,TN,TK,CM,CN,
            cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
            cutlass::epilogue::TmaWarpSpecialized1Sm,
            0>(
                tile_name.c_str(),
                clust_name.c_str(),
                "1sm/1sm/Sauto",
                d_A, d_B, d_C, d_D, d_combined,
                M, N, K, seq_len, hw_info, ms_postadd_ref, d_bytes, results);
        try_config_policy<TM,TN,TK,CM,CN,
            cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
            cutlass::epilogue::TmaWarpSpecialized1Sm,
            3>(
                tile_name.c_str(),
                clust_name.c_str(),
                "1sm/1sm/S3",
                d_A, d_B, d_C, d_D, d_combined,
                M, N, K, seq_len, hw_info, ms_postadd_ref, d_bytes, results);
        try_config_policy<TM,TN,TK,CM,CN,
            cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
            cutlass::epilogue::TmaWarpSpecialized1Sm,
            4>(
                tile_name.c_str(),
                clust_name.c_str(),
                "1sm/1sm/S4",
                d_A, d_B, d_C, d_D, d_combined,
                M, N, K, seq_len, hw_info, ms_postadd_ref, d_bytes, results);
    }

    if constexpr (TM >= 128 && (CM % 2 == 0)) {
        try_config_policy<TM,TN,TK,CM,CN,
            cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            0>(
                tile_name.c_str(),
                clust_name.c_str(),
                "2sm/2sm/Sauto",
                d_A, d_B, d_C, d_D, d_combined,
                M, N, K, seq_len, hw_info, ms_postadd_ref, d_bytes, results);
        try_config_policy<TM,TN,TK,CM,CN,
            cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            3>(
                tile_name.c_str(),
                clust_name.c_str(),
                "2sm/2sm/S3",
                d_A, d_B, d_C, d_D, d_combined,
                M, N, K, seq_len, hw_info, ms_postadd_ref, d_bytes, results);
        try_config_policy<TM,TN,TK,CM,CN,
            cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            4>(
                tile_name.c_str(),
                clust_name.c_str(),
                "2sm/2sm/S4",
                d_A, d_B, d_C, d_D, d_combined,
                M, N, K, seq_len, hw_info, ms_postadd_ref, d_bytes, results);
#if CUTLASS_EXTENDED_SWEEP
        try_config_policy<TM,TN,TK,CM,CN,
            cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            5>(
                tile_name.c_str(),
                clust_name.c_str(),
                "2sm/2sm/S5",
                d_A, d_B, d_C, d_D, d_combined,
                M, N, K, seq_len, hw_info, ms_postadd_ref, d_bytes, results);
#endif
    }
#endif
}

#define TRY(tm,tn,tk,cm,cn) \
    try_config_family<tm,tn,tk,cm,cn>( \
        d_A, d_B, d_C, d_D, d_combined, \
        M, N, K, SEQ_LEN, hw_info, ms_postadd_ref, sz_cd, results)

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
    const int N        = 768;
    const int K        = 768;

    cudaDeviceProp props;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    const int SM_COUNT = props.multiProcessorCount;

    int imgs_per_sm = 32;
    if (argc > 1) imgs_per_sm = atoi(argv[1]);
    const int M = imgs_per_sm * SM_COUNT * SEQ_LEN;

    double flops = 2.0 * M * N * K;

    printf("CUTLASS SM100a FP8 Grid Search\n");
    printf("  [%d, %d] x [%d, %d]^T  (imgs_per_sm=%d, %d images)\n",
           M, K, K, N, imgs_per_sm, M / SEQ_LEN);
    printf("  FP8 E4M3 (per-tensor) -> BF16, acc FP32\n");

    printf("  Device: %s (SM %d.%d, SMs=%d)\n\n", props.name, props.major, props.minor, SM_COUNT);

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
    // B: alternating columns — even cols = 0x3C (FP8 1.5), odd cols = 0x38 (FP8 1.0)
    // B is [N, K] col-major, so column n occupies contiguous K bytes at offset n*K
    {
        uint8_t* h_B = (uint8_t*)malloc(sz_b);
        for (int n = 0; n < N; n++)
            memset(h_B + (size_t)n * K, (n & 1) ? 0x38 : 0x3C, K);
        CUDA_CHECK(cudaMemcpy(d_B, h_B, sz_b, cudaMemcpyHostToDevice));
        free(h_B);
    }
    CUDA_CHECK(cudaMemset(d_C, 0, sz_cd));
    CUDA_CHECK(cudaMemset(d_D, 0, sz_cd));

    // ── Combined bias+pos_embed → tile into C ──
    float *d_bias, *d_pos;
    __nv_bfloat16 *d_combined;
    CUDA_CHECK(cudaMalloc(&d_bias, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos, (size_t)SEQ_LEN * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_combined, (size_t)SEQ_LEN * N * sizeof(__nv_bfloat16)));
    // Non-uniform bias/pos_embed matching megakernel validation data:
    // bias[c] = c + 1 (column-dependent), pos_embed[r][c] = 3*(r+1) (row-dependent)
    // Exposes N-tile indexing bugs that all-zero data would mask.
    {
        float* h_bias = (float*)malloc((size_t)N * sizeof(float));
        float* h_pos  = (float*)malloc((size_t)SEQ_LEN * N * sizeof(float));
        for (int c = 0; c < N; c++)
            h_bias[c] = (float)(c + 1);
        for (int r = 0; r < SEQ_LEN; r++)
            for (int c = 0; c < N; c++)
                h_pos[r * N + c] = (float)((r + 1) * 3);
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pos, h_pos, (size_t)SEQ_LEN * N * sizeof(float), cudaMemcpyHostToDevice));
        free(h_bias);
        free(h_pos);
    }
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

    // ── Grid/policy search ──
    std::vector<ConfigResult> results;
    float ms_postadd_ref = bench_postadd_only(
        reinterpret_cast<__nv_bfloat16*>(d_D),
        d_combined,
        M, N, SEQ_LEN,
        sz_cd);
    printf("  PostAdd-only baseline: %.3f ms\n", ms_postadd_ref);

#if CUTLASS_EXTENDED_SWEEP
    printf("Sweeping extended tile list with policy variants...\n");
#else
    printf("Sweeping standard tile list with policy variants...\n");
#endif

    // 2SM configs (cluster_m=2)
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

    // 1SM configs
    TRY( 64, 128,  64, 1, 1);
    TRY( 64, 256,  64, 1, 1);
    TRY( 64, 128, 128, 1, 1);
    TRY( 64, 256, 128, 1, 1);
    TRY(128, 128,  64, 1, 1);
    TRY(128, 256,  64, 1, 1);
    TRY(128, 192,  64, 1, 1);

#if CUTLASS_EXTENDED_SWEEP
    // Extended 2SM configs (cluster_m=2, cluster_n=2)
    TRY(256, 128, 128, 2, 2);
    TRY(128, 128,  64, 2, 2);
    TRY(128, 256,  64, 2, 2);
    TRY(128, 128, 128, 2, 2);
    TRY(128, 256, 128, 2, 2);
    TRY(128, 192, 128, 2, 2);

    // Extended 1SM configs
    TRY(128, 128, 128, 1, 1);
    TRY(128, 256, 128, 1, 1);
    TRY(128, 192, 128, 1, 1);
#endif

    // ── Sort by Fused EVT (true apples-to-apples), then GEMM+PostAdd as tiebreak ──
    std::sort(results.begin(), results.end(), [](const ConfigResult& a, const ConfigResult& b) {
        float av = a.ms_fused_periodic > 0 ? a.ms_fused_periodic : 1e9f;
        float bv = b.ms_fused_periodic > 0 ? b.ms_fused_periodic : 1e9f;
        if (av != bv) return av < bv;
        float a2 = a.ms_gemm_plus_postadd > 0 ? a.ms_gemm_plus_postadd : 1e9f;
        float b2 = b.ms_gemm_plus_postadd > 0 ? b.ms_gemm_plus_postadd : 1e9f;
        return a2 < b2;
    });

    // ── Results table ──
    auto tflops = [&](float ms) -> float {
        return ms > 0 ? (float)(flops / ms / 1e9) : 0.0f;
    };

    printf("\n");
    printf("CUTLASS SM100a FP8 Policy Search — [%d, %d] x [%d, %d]^T\n", M, K, K, N);
    printf("Total variants tested: %zu\n", results.size());
    printf("═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  %-13s %-5s %-20s | %-18s | %-18s | %-18s | %-18s | %-10s | %-18s\n",
           "Tile", "Clust", "Policy", "GEMM-only", "Fused FP32", "Fused BF16", "Fused EVT", "PostAdd", "GEMM+PostAdd");
    printf("  %-13s %-5s %-20s | %7s %9s | %7s %9s | %7s %9s | %7s %9s | %7s | %7s %9s\n",
           "", "", "", "ms", "TFLOPS", "ms", "TFLOPS", "ms", "TFLOPS", "ms", "TFLOPS", "ms", "ms", "eTFLOPS");
    printf("──────────────────────────────────────────────+────────────────────+────────────────────+────────────────────+────────────────────+────────────+────────────────────\n");

    auto print_cell = [&](float ms) {
        if (ms == -2.0f) printf("    FAIL       n/a ");
        else if (ms < 0) printf("     n/a       n/a ");
        else printf("  %6.3f  %7.0f ", ms, tflops(ms));
    };

    for (auto& r : results) {
        printf("  %-13s %-5s %-20s |", r.tile_str.c_str(), r.cluster_str.c_str(), r.policy_str.c_str());
        print_cell(r.ms_gemm);      printf("|");
        print_cell(r.ms_fused_fp32); printf("|");
        print_cell(r.ms_fused_bf16); printf("|");
        print_cell(r.ms_fused_periodic); printf("|");
        if (r.ms_postadd < 0) printf("   n/a    |");
        else printf(" %7.3f |", r.ms_postadd);
        print_cell(r.ms_gemm_plus_postadd);
        printf("\n");
    }

    printf("──────────────────────────────────────────────+────────────────────+────────────────────+────────────────────+────────────────────+────────────+────────────────────\n");

    // ── Best configs ──
    const ConfigResult* best_gemm = nullptr;
    const ConfigResult* best_fused_fp32 = nullptr;
    const ConfigResult* best_fused = nullptr;
    const ConfigResult* best_fused_periodic = nullptr;
    const ConfigResult* best_gemm_postadd = nullptr;

    for (auto& r : results) {
        if (r.ms_gemm > 0 && (!best_gemm || r.ms_gemm < best_gemm->ms_gemm))
            best_gemm = &r;
        if (r.ms_fused_fp32 > 0 && (!best_fused_fp32 || r.ms_fused_fp32 < best_fused_fp32->ms_fused_fp32))
            best_fused_fp32 = &r;
        if (r.ms_fused_bf16 > 0 && (!best_fused || r.ms_fused_bf16 < best_fused->ms_fused_bf16))
            best_fused = &r;
        if (r.ms_fused_periodic > 0 && (!best_fused_periodic || r.ms_fused_periodic < best_fused_periodic->ms_fused_periodic))
            best_fused_periodic = &r;
        if (r.ms_gemm_plus_postadd > 0 && (!best_gemm_postadd || r.ms_gemm_plus_postadd < best_gemm_postadd->ms_gemm_plus_postadd))
            best_gemm_postadd = &r;
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
    if (best_fused)
        printf("  Best Fused BF16:  %s %s %s  %.3f ms / %.0f TFLOPS\n",
               best_fused->tile_str.c_str(), best_fused->cluster_str.c_str(),
               best_fused->policy_str.c_str(),
               best_fused->ms_fused_bf16, tflops(best_fused->ms_fused_bf16));
    if (best_fused_periodic)
        printf("  Best Fused EVT:   %s %s %s  %.3f ms / %.0f TFLOPS\n",
               best_fused_periodic->tile_str.c_str(), best_fused_periodic->cluster_str.c_str(),
               best_fused_periodic->policy_str.c_str(),
               best_fused_periodic->ms_fused_periodic, tflops(best_fused_periodic->ms_fused_periodic));
    if (best_gemm_postadd)
        printf("  Best GEMM+PostAdd:%s %s %s  %.3f ms / %.0f TFLOPS\n",
               best_gemm_postadd->tile_str.c_str(), best_gemm_postadd->cluster_str.c_str(),
               best_gemm_postadd->policy_str.c_str(),
               best_gemm_postadd->ms_gemm_plus_postadd, tflops(best_gemm_postadd->ms_gemm_plus_postadd));

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
