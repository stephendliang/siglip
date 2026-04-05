/*
FC2 Hybrid kernel — our optimized mainloop + CUTLASS CollectiveEpilogue.
Shape: [928256,3072]×[3072,768]^T + bias + residual

Goal: best of both worlds.
  Our mainloop:      1.089ms strip (62μs faster than CUTLASS's 1.152ms)
  CUTLASS epilogue:  72μs overhead (vs our 388μs)
  Hybrid ceiling:    1.089 + 0.072 = 1.161ms

Architecture: 8 warps (256 threads), cta_group::2, cluster 2×1
  W0: MMA            ← Phase 2+3a: unrolled K-loop
  W1: Scheduler (CLC)
  W2: MainloopLoad   ← Phase 3a: unrolled TMA load loop
  W3: EpilogueLoad   — CUTLASS (unchanged)
  W4-W7: Epilogue    — CUTLASS (unchanged)

Build:
  make fc2-hybrid              # Phase 1: CUTLASS via custom launch
  make fc2-hybrid-strip        # GEMM-only (STRIP_EPILOGUE)
  make fc2-hybrid-mma          # Phase 2+3a: HybridMainloop (unrolled load + K-loop)

Phase 1: Custom __global__ that delegates to GemmKernel::operator().
  Pure CUTLASS — validates custom launch mechanism.
  Performance: identical to fc2-cutlass (1.224ms fused, 1.152ms strip).

Phase 2 (HYBRID_MMA): HybridMainloop inherits CollectiveMainloop, overrides mma().
  B200 result: 1.220ms fused = identical to Phase 1. K-loop unrolling NOT the gap.

Phase 3a (HYBRID_MMA): Also overrides load() with unrolled TMA load loop.
  Tests whether TMA load scheduling is the source of the 62μs mainloop gap.
  Same cute::copy operations, different loop structure (unroll 5 vs NO_UNROLL).

Phase 3b (future): Custom 7-warp kernel — our PTX mainloop + CUTLASS epilogue.
  Tests warp count hypothesis + captures epilogue quality improvement.
*/

#include <cstdio>
#include <cstdlib>

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
#include "cutlass/device_kernel.h"  /* CUTLASS_GRID_CONSTANT (__grid_constant__) */

#include <cuda_bf16.h>
#include "cutlass/epilogue/thread/activation.h"

using namespace cute;

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

/* ═══════════════════════════════════════════════════════════════════
   Problem dimensions — identical to fc2_w3.cu and fc2_cutlass.cu
   ═══════════════════════════════════════════════════════════════════ */

#define SM_COUNT  148
#define M_TOTAL   928256   /* 4736 images × 196 patches */
#define N_DIM     768
#define K_DIM     3072

/* ═══════════════════════════════════════════════════════════════════
   CUTLASS type setup — same as fc2_cutlass.cu
   ═══════════════════════════════════════════════════════════════════ */

using ElementA   = cutlass::float_e4m3_t;
using LayoutA    = cutlass::layout::RowMajor;
using ElementB   = cutlass::float_e4m3_t;
using LayoutB    = cutlass::layout::ColumnMajor;
using ElementC   = cutlass::bfloat16_t;
using ElementD   = cutlass::bfloat16_t;
using LayoutC    = cutlass::layout::RowMajor;
using LayoutD    = cutlass::layout::RowMajor;
using ElementAcc = float;

constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;

using TileShape    = Shape<_256, _256, _128>;
using ClusterShape = Shape<_2, _1, _1>;

using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;

#ifdef STRIP_EPILOGUE
using FusionOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementAcc>;
#else
using FusionOp = cutlass::epilogue::fusion::LinCombPerColBiasEltAct<
    cutlass::epilogue::thread::Identity,
    ElementD, float, float, ElementC, float>;
#endif

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, float,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    EpilogueSchedule,
    FusionOp
>::CollectiveOp;

constexpr int EpilogueSmemBytes = static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage));

using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<EpilogueSmemBytes>;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAcc,
    TileShape, ClusterShape,
    StageCountType,
    MainloopSchedule
>::CollectiveOp;

/* ═══════════════════════════════════════════════════════════════════
   Phase 2: HybridMainloop — CUTLASS mainloop with our MMA K-loop
   ═══════════════════════════════════════════════════════════════════ */

#ifdef HYBRID_MMA
/*
HybridMainloop inherits CUTLASS's CollectiveMainloop, overrides load() + mma().

Phase 2: mma() override — unrolled K-loop. B200 result: 1.220ms = neutral.
Phase 3a: load() override — unrolled TMA load loop. Tests load codegen.

All types (SharedStorage, Params, TiledMma, etc.) are inherited unchanged,
so GemmUniversal's enable_if on DispatchPolicy::Schedule resolves to the
same SM100 specialization.
*/
struct HybridMainloop : CollectiveMainloop {
    using CollectiveMainloop::CollectiveMainloop;
    using CollectiveMainloop::MainloopPipeline;
    using CollectiveMainloop::MainloopPipelineState;

    /*
    Override load(): identical to base class except _Pragma("unroll 5") on
    the K-tile loop (CUTLASS uses PRAGMA_NO_UNROLL). Tests whether TMA load
    scheduling/codegen is the source of the 62μs mainloop gap.
    */
    template <class LP, class TileCoordMNKL, class KTileIterator>
    CUTLASS_DEVICE auto
    load(MainloopPipeline mainloop_pipeline,
         MainloopPipelineState mainloop_pipe_producer_state,
         LP const& load_inputs,
         TileCoordMNKL const& cta_coord_mnkl,
         KTileIterator k_tile_iter, int k_tile_count)
    {
        using namespace cute;
        using TiledMma_ = typename CollectiveMainloop::TiledMma;

        auto [unused_k_tiles,
              tAgA_mkl, tBgB_nkl, tAsA, tBsB,
              mcast_mask_a, mcast_mask_b] = load_inputs;

        /* Slice out the work coord from partitioned tensors */
        auto tAgA = tAgA_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma_::AtomThrID{}), _, get<3>(cta_coord_mnkl));
        auto tBgB = tBgB_nkl(_, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

        auto barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);

        /* Unrolled K-tile loop (CUTLASS uses PRAGMA_NO_UNROLL) */
        _Pragma("unroll 5")
        for (int ki = 0; ki < k_tile_count; ++ki) {
            mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state, barrier_token);

            using BarrierType = typename MainloopPipeline::ProducerBarrierType;
            BarrierType* tma_barrier = mainloop_pipeline.producer_get_barrier(mainloop_pipe_producer_state);

            int write_stage = mainloop_pipe_producer_state.index();
            ++mainloop_pipe_producer_state;
            barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);

            if (elect_one_sync()) {
                copy(this->observed_tma_load_a_->with(*tma_barrier, mcast_mask_a), tAgA(_,*k_tile_iter), tAsA(_,write_stage));
                copy(this->observed_tma_load_b_->with(*tma_barrier, mcast_mask_b), tBgB(_,*k_tile_iter), tBsB(_,write_stage));
            }

            ++k_tile_iter;
        }

        return make_tuple(mainloop_pipe_producer_state, k_tile_iter);
    }

    /*
    Override mma(): _Pragma("unroll 5") on K-loop (CUTLASS uses NO_UNROLL).
    Phase 2 B200 result: 1.220ms = neutral (K-loop codegen is NOT the gap).
    */
    template <
        class AccumulatorPipeline,
        class FrgEngine, class FrgLayout,
        class MmaParams,
        class CtaTileCoord
    >
    CUTLASS_DEVICE auto
    mma(cute::tuple<MainloopPipeline,
                    AccumulatorPipeline> pipelines,
        cute::tuple<MainloopPipelineState,
                    typename AccumulatorPipeline::PipelineState> pipeline_states,
        cute::tuple<cute::Tensor<FrgEngine, FrgLayout>> const& accumulators_pair,
        MmaParams const& mma_inputs,
        CtaTileCoord cta_tile_coord,
        int k_tile_count)
    {
        static_assert(cute::is_tmem<FrgEngine>::value, "Accumulator must be tmem resident.");
        static_assert(rank(FrgLayout{}) == 3, "Accumulator must be MMA-partitioned: (MMA, MMA_M, MMA_N)");

        auto accumulators = get<0>(accumulators_pair);
        auto [tiled_mma, tCrA, tCrB] = mma_inputs;
        auto [mainloop_pipeline, accumulator_pipeline] = pipelines;
        auto [mainloop_pipe_consumer_state, accumulator_pipe_producer_state] = pipeline_states;

        tiled_mma.accumulate_ = cute::UMMA::ScaleOut::Zero;
        accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);

        /* FC2-specific: K_DIM and TK (128) are compile-time constants.
           Use constexpr trip count so #pragma unroll 5 actually unrolls.
           CUTLASS uses runtime k_tile_count with PRAGMA_NO_UNROLL. */
        constexpr int kKIters = K_DIM / 128;
        (void)k_tile_count;

        auto barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, 0);

        _Pragma("unroll 5")
        for (int ki = 0; ki < kKIters; ++ki) {
            mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state, barrier_token);

            int read_stage = mainloop_pipe_consumer_state.index();
            auto curr_mainloop_pipe_consumer_state = mainloop_pipe_consumer_state;

            ++mainloop_pipe_consumer_state;
            uint32_t skip_wait = (ki + 1 >= kKIters) ? 1U : 0U;
            barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);

            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
                cute::gemm(tiled_mma,
                           tCrA(_,_,k_block,read_stage),
                           tCrB(_,_,k_block,read_stage),
                           accumulators);
                tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
            }
            mainloop_pipeline.consumer_release(curr_mainloop_pipe_consumer_state);
        }

        return mainloop_pipe_consumer_state;
    }
};

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    HybridMainloop,
    CollectiveEpilogue>;
#else
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;
#endif

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

/* ═══════════════════════════════════════════════════════════════════
   Custom kernel — template __global__ matching CUTLASS's device_kernel<>
   ═══════════════════════════════════════════════════════════════════ */

template <typename Operator>
__global__ void
__launch_bounds__(Operator::MaxThreadsPerBlock, 1)
fc2_hybrid_kernel_impl(CUTLASS_GRID_CONSTANT typename Operator::Params const params) {
    extern __shared__ char smem_buf[];
    Operator op;
    op(params, smem_buf);
}

static auto* fc2_hybrid_kernel = fc2_hybrid_kernel_impl<GemmKernel>;

/* ═══════════════════════════════════════════════════════════════════
   Phase 3b: Custom 7-warp kernel — our warp count + CUTLASS epilogue

   Tests the warp count hypothesis: 7 warps (no Sched) vs 8 warps.
   Uses CUTLASS's collective_mainloop.load()/mma() for codegen parity —
   the ONLY difference is fewer warps + static tile scheduling.

   W0: MainloopLoad (CUTLASS TMA loads)
   W1: MMA          (CUTLASS MMA)
   W2: EpilogueLoad (CUTLASS epilogue TMA loads)
   W3-W6: Epilogue  (CUTLASS epilogue store)
   ═══════════════════════════════════════════════════════════════════ */

#ifdef HYBRID_PHASE3

template <typename GK>
__global__ void
__launch_bounds__(224, 1)
fc2_phase3_kernel_impl(CUTLASS_GRID_CONSTANT typename GK::Params const params) {
    using namespace cute;

    using CM = typename GK::CollectiveMainloop;
    using CE = typename GK::CollectiveEpilogue;
    using SharedStorage = typename GK::SharedStorage;

    using MainloopPipeline      = typename GK::MainloopPipeline;
    using MainloopPipelineState  = typename GK::MainloopPipelineState;
    using EpiLoadPipeline       = typename GK::EpiLoadPipeline;
    using EpiLoadPipelineState   = typename GK::EpiLoadPipelineState;
    using EpiStorePipeline      = typename GK::EpiStorePipeline;
    using EpiStorePipelineState  = typename GK::EpiStorePipelineState;
    using AccumulatorPipeline   = typename GK::AccumulatorPipeline;
    using AccumulatorPipelineState = typename GK::AccumulatorPipelineState;
    using LoadOrderBarrier      = typename GK::LoadOrderBarrier;
    using TmemAllocator         = typename GK::TmemAllocator;

    using TileShape       = typename CM::TileShape;
    using TiledMma        = typename CM::TiledMma;
    using AtomThrShapeMNK = typename CM::AtomThrShapeMNK;
    using CtaShape_MNK    = typename CM::CtaShape_MNK;
    using EpilogueTile    = typename GK::EpilogueTile;

    static constexpr bool IsOverlapping = GK::IsOverlappingAccum;
    static constexpr bool HasPeerCta    = size(AtomThrShapeMNK{}) == 2;

    extern __shared__ char smem_buf[];
    SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem_buf);

    int warp_idx = cutlass::canonical_warp_idx_sync();
    uint32_t lane_pred = elect_one_sync();

    auto cluster_shape = cutlass::detail::select_cluster_shape(
        typename CM::DispatchPolicy::ClusterShape{});
    int cluster_size = size(cluster_shape);
    uint32_t cta_rank = block_rank_in_cluster();
    int cta_v = cta_rank % size<0>(typename TiledMma::AtomThrID{});
    bool is_mma_leader = cta_v == 0;
    [[maybe_unused]] uint32_t peer_rank = HasPeerCta ? cta_rank ^ 1 : cta_rank;

    auto problem_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto [M,N,K,L] = problem_MNKL;

    CM collective_mainloop(params.mainloop, cluster_shape, cta_rank);
    CE collective_epilogue(params.epilogue, ss.tensors.epilogue);

    /* TMA descriptor prefetch */
    if (warp_idx == 0 && lane_pred) collective_mainloop.prefetch_tma_descriptors();
    if (warp_idx == 2 && lane_pred) collective_epilogue.prefetch_tma_descriptors(params.epilogue);

    bool epi_load_needed = collective_epilogue.is_producer_load_needed();

    /* ── Warp roles: W0=Load W1=MMA W2=EpiLoad W3-6=Epilogue ── */
    enum { RLoad=0, RMMA=1, REpiL=2, REpi=3 };
    int role = warp_idx < REpi ? warp_idx : REpi;

    /* ── Pipeline init ── */
    typename MainloopPipeline::Params ml_p;
    if (role == RLoad) ml_p.role = MainloopPipeline::ThreadCategory::Producer;
    if (role == RMMA)  ml_p.role = MainloopPipeline::ThreadCategory::Consumer;
    ml_p.is_leader = lane_pred && is_mma_leader && (role == RLoad);
    ml_p.transaction_bytes = CM::TmaTransactionBytes;
    ml_p.initializing_warp = 0;
    MainloopPipeline mainloop_pipe(ss.pipelines.mainloop, ml_p, cluster_shape,
                                   true_type{}, false_type{});

    typename EpiLoadPipeline::Params el_p;
    if (role == REpiL) el_p.role = EpiLoadPipeline::ThreadCategory::Producer;
    if (role == REpi)  el_p.role = EpiLoadPipeline::ThreadCategory::Consumer;
    el_p.dst_blockid = cta_rank;
    el_p.producer_arv_count = 32;
    el_p.consumer_arv_count = GK::NumEpilogueThreads;
    el_p.transaction_bytes = CE::TmaTransactionBytes;
    el_p.initializing_warp = 1;
    EpiLoadPipeline epi_load_pipe(ss.pipelines.epi_load, el_p);

    typename EpiStorePipeline::Params es_p;
    es_p.always_wait = true;
    EpiStorePipeline epi_store_pipe(es_p);

    typename LoadOrderBarrier::Params lo_p;
    lo_p.group_id = (role == RLoad) ? 0 : 1;
    lo_p.group_size = 32;
    lo_p.initializing_warp = 2;
    LoadOrderBarrier load_order(ss.pipelines.load_order, lo_p);

    typename AccumulatorPipeline::Params ap_p;
    if (role == RMMA) ap_p.role = AccumulatorPipeline::ThreadCategory::Producer;
    if (role == REpi) ap_p.role = AccumulatorPipeline::ThreadCategory::Consumer;
    ap_p.producer_arv_count = 1;
    ap_p.consumer_arv_count = size(AtomThrShapeMNK{}) * GK::NumEpilogueThreads;
    ap_p.initializing_warp = 3;
    AccumulatorPipeline acc_pipe(ss.pipelines.accumulator, ap_p, cluster_shape,
                                true_type{}, false_type{});

    /* TMEM allocator + sync barriers */
    TmemAllocator tmem_alloc{};
    cutlass::arch::NamedBarrier tmem_bar(32 + GK::NumEpilogueThreads,
        cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
    auto& tmem_dealloc = ss.pipelines.tmem_dealloc;
    [[maybe_unused]] uint32_t dealloc_phase = 0;
    if (role == RMMA) {
        if constexpr (!IsOverlapping) {
            if (HasPeerCta && lane_pred) tmem_dealloc.init(32);
        } else {
            if (HasPeerCta && lane_pred) tmem_dealloc.init(GK::NumEpilogueThreads * 2);
            else if (lane_pred) tmem_dealloc.init(GK::NumEpilogueThreads);
        }
    }

    /* ── Cluster sync ── */
    cutlass::pipeline_init_arrive_relaxed(cluster_size);

    auto load_inputs = collective_mainloop.load_init(problem_MNKL, ss.tensors.mainloop);

    MainloopPipelineState ml_cons, ml_prod = cutlass::make_producer_start_state<MainloopPipeline>();
    EpiLoadPipelineState  el_cons, el_prod = cutlass::make_producer_start_state<EpiLoadPipeline>();
    EpiStorePipelineState es_prod = cutlass::make_producer_start_state<EpiStorePipeline>();
    AccumulatorPipelineState ac_cons, ac_prod = cutlass::make_producer_start_state<AccumulatorPipeline>();

    auto tmem_st = collective_mainloop.template init_tmem_tensors<EpilogueTile, IsOverlapping>(EpilogueTile{});

    dim3 bid = block_id_in_cluster();
    mainloop_pipe.init_masks(cluster_shape, bid);
    acc_pipe.init_masks(cluster_shape, bid);

    cutlass::pipeline_init_wait(cluster_size);

    /* ── Static tile scheduling (linearize 2D/3D grid → flat cluster ID) ── */
    int bid_linear = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int cid = bid_linear / cluster_size;
    int nc  = (gridDim.x * gridDim.y * gridDim.z) / cluster_size;
    int tm_count = cute::ceil_div(int(M), int(get<0>(CtaShape_MNK{})));
    int tn_count = cute::ceil_div(int(N), int(get<1>(CtaShape_MNK{})));
    int total = tm_count * tn_count;
    int t0 = (int)((long long)cid * total / nc);
    int t1 = (int)((long long)(cid+1) * total / nc);
    int k_tiles = cute::ceil_div(int(K), int(get<2>(TileShape{})));

    /* ═══ WARP DISPATCH ═══ */

    if (role == RLoad) {
        cutlass::arch::wait_on_dependent_grids();
        bool do_arrive = epi_load_needed;
        for (int ti = t0; ti < t1; ti++) {
            int tm = ti / tn_count, tn = ti % tn_count;
            if (tm & 1) tn = tn_count - 1 - tn;
            auto coord = make_coord(tm, tn, _, Int<0>{});
            auto ki = make_coord_iterator(load_inputs.k_tiles);
            int prol = min(int(MainloopPipeline::Stages), k_tiles);

            auto [s1,k1] = collective_mainloop.load(mainloop_pipe, ml_prod, load_inputs, coord, ki, prol);
            ml_prod = s1;
            if (do_arrive) { load_order.arrive(); do_arrive = false; }
            auto [s2,k2] = collective_mainloop.load(mainloop_pipe, ml_prod, load_inputs, coord, k1, k_tiles - prol);
            ml_prod = s2;
        }
        collective_mainloop.load_tail(mainloop_pipe, ml_prod);
    }

    else if (role == RMMA) {
        tmem_alloc.allocate(TmemAllocator::Sm100TmemCapacityColumns, &ss.tmem_base_ptr);
        __syncwarp();
        tmem_bar.arrive();
        uint32_t tptr = ss.tmem_base_ptr;
        collective_mainloop.set_tmem_offsets(tmem_st, tptr);
        auto mma_in = collective_mainloop.mma_init(tmem_st, ss.tensors.mainloop);

        for (int ti = t0; ti < t1; ti++) {
            int tm = ti / tn_count, tn = ti % tn_count;
            if (tm & 1) tn = tn_count - 1 - tn;
            auto coord = make_coord(tm, tn, _, Int<0>{});
            int as = [&]{ if constexpr(IsOverlapping) return ac_prod.phase()^1; else return ac_prod.index(); }();
            if (is_mma_leader) {
                ml_cons = collective_mainloop.mma(
                    make_tuple(mainloop_pipe, acc_pipe),
                    make_tuple(ml_cons, ac_prod),
                    collective_mainloop.slice_accumulator(tmem_st, as),
                    mma_in, coord, k_tiles);
                acc_pipe.producer_commit(ac_prod);
            }
            ++ac_prod;
        }
        cutlass::arch::launch_dependent_grids();
        tmem_alloc.release_allocation_lock();
        if constexpr (!IsOverlapping) {
            if (is_mma_leader) acc_pipe.producer_tail(ac_prod);
            if constexpr (HasPeerCta) {
                tmem_dealloc.arrive(peer_rank, !is_mma_leader);
                tmem_dealloc.wait(dealloc_phase);
                tmem_dealloc.arrive(peer_rank, is_mma_leader);
            }
        } else {
            tmem_dealloc.wait(dealloc_phase);
        }
        tmem_alloc.free(tptr, TmemAllocator::Sm100TmemCapacityColumns);
    }

    else if (role == REpiL) {
        if (epi_load_needed) {
            cutlass::arch::wait_on_dependent_grids();
            bool do_wait = true;
            for (int ti = t0; ti < t1; ti++) {
                int tm = ti / tn_count, tn = ti % tn_count;
                if (tm & 1) tn = tn_count - 1 - tn;
                auto coord = make_coord(tm, tn, _, Int<0>{});
                if (do_wait) { load_order.wait(); do_wait = false; }
                int w = ti - t0;
                bool rev = IsOverlapping && (w % 2 == 0);
                el_prod = collective_epilogue.template load<IsOverlapping>(
                    epi_load_pipe, el_prod, problem_MNKL, CtaShape_MNK{}, coord,
                    TileShape{}, TiledMma{}, ss.tensors.epilogue, rev);
            }
            collective_epilogue.load_tail(epi_load_pipe, el_prod, epi_store_pipe, es_prod);
        }
    }

    else { /* Epilogue W3-W6 */
        tmem_bar.arrive_and_wait();
        uint32_t tptr = ss.tmem_base_ptr;
        collective_mainloop.set_tmem_offsets(tmem_st, tptr);
        bool did_store = false;

        for (int ti = t0; ti < t1; ti++) {
            int tm = ti / tn_count, tn = ti % tn_count;
            if (tm & 1) tn = tn_count - 1 - tn;
            auto coord = make_coord(tm, tn, _, Int<0>{});
            int as = [&]{ if constexpr(IsOverlapping) return ac_cons.phase(); else return ac_cons.index(); }();
            auto acc = get<0>(collective_mainloop.slice_accumulator(tmem_st, as));

            auto [l,s,a] = collective_epilogue.template store<IsOverlapping>(
                epi_load_pipe, el_cons, epi_store_pipe, es_prod,
                acc_pipe, ac_cons,
                problem_MNKL, CtaShape_MNK{}, coord,
                TileShape{}, TiledMma{}, acc, ss.tensors.epilogue);
            el_cons = l; es_prod = s; ac_cons = a;
            did_store = true;
        }

        if constexpr (IsOverlapping) {
            if constexpr (HasPeerCta) tmem_dealloc.arrive(peer_rank);
            tmem_dealloc.arrive();
        }
        if (did_store) {
            collective_epilogue.store_tail(epi_load_pipe, el_cons, epi_store_pipe, es_prod, CtaShape_MNK{});
        }
    }
}

static auto* fc2_phase3_kernel = fc2_phase3_kernel_impl<GemmKernel>;

#endif /* HYBRID_PHASE3 */

/* ═══════════════════════════════════════════════════════════════════
   Host code — identical to fc2_cutlass.cu except launches our kernel
   ═══════════════════════════════════════════════════════════════════ */

__global__ void init_residual(__nv_bfloat16* d, int N, long long total) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        long long row = i / N;
        int col = (int)(i % N);
        d[i] = __float2bfloat16((float)((int)(row % 128)) * 0.25f + (float)col * 0.125f);
    }
}

int main() {
    setbuf(stdout, NULL);

#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    printf("ERROR: CUTLASS_ARCH_MMA_SM100_SUPPORTED not defined.\n");
    return 1;
#else
#if defined(HYBRID_PHASE3)
    printf("FC2 HYBRID Phase 3b — 7-warp custom kernel + CUTLASS epilogue\n");
#elif defined(HYBRID_MMA)
    printf("FC2 HYBRID Phase 3a — HybridMainloop (unrolled load+K-loop) + CUTLASS epilogue\n");
#else
    printf("FC2 HYBRID Phase 1 — pure CUTLASS via custom launch\n");
#endif
#ifdef STRIP_EPILOGUE
    printf("  MODE: GEMM-only (STRIP_EPILOGUE)\n");
#else
    printf("  MODE: Fused (bias + residual)\n");
#endif
    printf("  [%d, %d] x [%d, %d]^T  FP8E4M3 -> BF16\n", M_TOTAL, K_DIM, K_DIM, N_DIM);
    printf("  Tile: 256x256x128, Cluster: 2x1, Schedule: 2SM\n");

    constexpr size_t SmemBytes = sizeof(typename GemmKernel::SharedStorage);
    printf("  SMEM: %zu bytes\n", SmemBytes);

    cudaDeviceProp props;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    printf("  Device: %s (SM %d.%d, SMs=%d)\n\n", props.name, props.major, props.minor,
           props.multiProcessorCount);

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    double flops = 2.0 * M_TOTAL * (double)N_DIM * K_DIM;

    /* Allocate */
    size_t sz_a   = (size_t)M_TOTAL * K_DIM;
    size_t sz_b   = (size_t)N_DIM * K_DIM;
    size_t sz_cd  = (size_t)M_TOTAL * N_DIM * sizeof(ElementD);
    size_t sz_bias = (size_t)N_DIM * sizeof(float);

    void *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_D = nullptr;
    float *d_bias = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, sz_a));
    CUDA_CHECK(cudaMalloc(&d_B, sz_b));
    CUDA_CHECK(cudaMalloc(&d_C, sz_cd));
    CUDA_CHECK(cudaMalloc(&d_D, sz_cd));
    CUDA_CHECK(cudaMalloc(&d_bias, sz_bias));

    CUDA_CHECK(cudaMemset(d_A, 0x3C, sz_a));
    {
        uint8_t* h_B = (uint8_t*)malloc(sz_b);
        for (int n = 0; n < N_DIM; n++)
            memset(h_B + (size_t)n * K_DIM, (n & 1) ? 0x38 : 0x3C, K_DIM);
        CUDA_CHECK(cudaMemcpy(d_B, h_B, sz_b, cudaMemcpyHostToDevice));
        free(h_B);
    }
    {
        float* h_bias = (float*)malloc(sz_bias);
        for (int c = 0; c < N_DIM; c++)
            h_bias[c] = (float)(c + 1);
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias, sz_bias, cudaMemcpyHostToDevice));
        free(h_bias);
    }
#ifndef STRIP_EPILOGUE
    {
        long long total = (long long)M_TOTAL * N_DIM;
        int tpb = 256;
        int bpg = (int)((total + tpb - 1) / tpb);
        init_residual<<<bpg, tpb>>>(reinterpret_cast<__nv_bfloat16*>(d_C), N_DIM, total);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
#else
    CUDA_CHECK(cudaMemset(d_C, 0, sz_cd));
#endif
    CUDA_CHECK(cudaMemset(d_D, 0, sz_cd));
    printf("  Alloc + init done\n");

    /* Use CUTLASS for argument setup — builds TMA descriptors, workspace, etc. */
    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideC;
    using StrideD = typename GemmKernel::StrideD;

    auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M_TOTAL, K_DIM, 1));
    auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N_DIM, K_DIM, 1));
    auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, make_shape(M_TOTAL, N_DIM, 1));
    auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M_TOTAL, N_DIM, 1));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M_TOTAL, N_DIM, K_DIM, 1},
        { reinterpret_cast<ElementA*>(d_A), stride_a,
          reinterpret_cast<ElementB*>(d_B), stride_b },
        { {},
          reinterpret_cast<ElementC*>(d_C), stride_c,
          reinterpret_cast<ElementD*>(d_D), stride_d },
        hw_info
    };

#ifdef STRIP_EPILOGUE
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta = 0.0f;
#else
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta = 1.0f;
    arguments.epilogue.thread.bias_ptr = d_bias;
#endif

    Gemm gemm;
    auto status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        printf("FATAL: can_implement failed: %d\n", (int)status);
        return 1;
    }

    size_t ws = Gemm::get_workspace_size(arguments);
    uint8_t* d_ws = nullptr;
    if (ws > 0) CUDA_CHECK(cudaMalloc(&d_ws, ws));

    status = gemm.initialize(arguments, d_ws);
    if (status != cutlass::Status::kSuccess) {
        printf("FATAL: initialize failed: %d\n", (int)status);
        return 1;
    }

    /* Extract kernel params and launch config from CUTLASS */
    auto kernel_params = gemm.params();
    auto grid = GemmKernel::get_grid_shape(kernel_params);
    auto block = GemmKernel::get_block_shape();

#ifdef HYBRID_PHASE3
    /* Phase 3b: use our 7-warp kernel */
    auto* the_kernel = (void const*)fc2_phase3_kernel;
    dim3 phase3_block(224, 1, 1); /* 7 warps */
#else
    auto* the_kernel = (void const*)fc2_hybrid_kernel;
#endif

    CUDA_CHECK(cudaFuncSetAttribute(the_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SmemBytes));
    CUDA_CHECK(cudaFuncSetAttribute(the_kernel,
        cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

    printf("  CUTLASS initialized (workspace=%zu, grid=%dx%d, block=%d, smem=%zu)\n",
           ws, grid.x, grid.y,
#ifdef HYBRID_PHASE3
           phase3_block.x,
#else
           block.x,
#endif
           SmemBytes);

    /* Cluster launch — match CUTLASS's ClusterLauncher exactly (C API, no __cluster_dims__) */
    auto launch_hybrid = [&](typename GemmKernel::Params& p) {
        cudaLaunchConfig_t config = {};
        config.gridDim = grid;
#ifdef HYBRID_PHASE3
        config.blockDim = phase3_block;
#else
        config.blockDim = block;
#endif
        config.dynamicSmemBytes = SmemBytes;
        cudaLaunchAttribute attrs[2];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim = {2, 1, 1};
        attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[1].val.programmaticStreamSerializationAllowed = 1;
        config.attrs = attrs;
        config.numAttrs = 2;
        void* kernel_args[] = { &p };
        CUDA_CHECK(cudaLaunchKernelExC(&config, the_kernel, kernel_args));
    };

    /* Warmup */
    printf("Launching warmup (2 iters)...\n");
    for (int i = 0; i < 2; i++)
        launch_hybrid(kernel_params);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Warmup done.\n");

    /* Timed: 10 iterations */
    printf("Timing: 10 iterations...\n");
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 10; i++)
        launch_hybrid(kernel_params);
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= 10.0f;
    printf("FC2-HYBRID: %.3f ms  %.2f TFLOPS\n", ms, flops / ms / 1e9);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    /* Checksum run */
    CUDA_CHECK(cudaMemset(d_D, 0, sz_cd));
    launch_hybrid(kernel_params);
    CUDA_CHECK(cudaDeviceSynchronize());

    __nv_bfloat16* h_D = (__nv_bfloat16*)malloc(sz_cd);
    CUDA_CHECK(cudaMemcpy(h_D, d_D, sz_cd, cudaMemcpyDeviceToHost));

    double cksum = 0;
    {
        long long total_elems = (long long)M_TOTAL * N_DIM;
        long long stride = total_elems / 1024;
        for (int i = 0; i < 1024; i++)
            cksum += (double)__bfloat162float(h_D[(long long)i * stride]);
    }

    int errors = 0;
    for (int spot = 0; spot < 32; spot++) {
        long long row = (long long)spot * M_TOTAL / 32;
        int col = (spot * 47) % N_DIM;
        float b_val = (col & 1) ? 1.0f : 1.5f;
        float gemm_val = (float)K_DIM * 1.5f * b_val;
#ifdef STRIP_EPILOGUE
        __nv_bfloat16 expected = __float2bfloat16(gemm_val);
#else
        float bias_f = (float)(col + 1);
        float res_f = __bfloat162float(__float2bfloat16(
            (float)((int)(row % 128)) * 0.25f + (float)col * 0.125f));
        __nv_bfloat16 expected = __float2bfloat16(gemm_val + bias_f + res_f);
#endif
        __nv_bfloat16 actual = h_D[row * N_DIM + col];
        float ef = __bfloat162float(expected);
        float af = __bfloat162float(actual);
        if (ef != af) {
            if (errors < 5)
                printf("  MISMATCH at (%lld,%d): expected %.1f got %.1f\n", row, col, ef, af);
            errors++;
        }
    }
    int valid = (errors == 0) ? 1 : 0;
    printf("Validation: %d/32 spot checks passed%s\n", 32 - errors, valid ? "" : " — FAILED");
    printf("Checksum (1024 strided): %f\n", cksum);
    printf("D[0,0..3] = %.1f %.1f %.1f %.1f\n",
           __bfloat162float(h_D[0]), __bfloat162float(h_D[1]),
           __bfloat162float(h_D[2]), __bfloat162float(h_D[3]));
    printf("@@RESULT ms=%.3f tflops=%.2f checksum=%f valid=%d c0=%.1f\n",
           ms, flops / ms / 1e9, cksum, valid, __bfloat162float(h_D[0]));

    free(h_D);
    if (d_ws) cudaFree(d_ws);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D); cudaFree(d_bias);
    return 0;
#endif
}
