// SM100a Epilogue Visitor: Periodic Table Addition for SigLIP2 Vision Encoder
// ═══════════════════════════════════════════════════════════════════════════
//
// Fuses a periodic [seq_len, N] BF16 table add into the GEMM epilogue:
//
//   D[m, n] = bf16( bf16(acc[m, n]) + combined[m % seq_len, n] )
//
// Numeric semantics match the megakernel: FP32 acc → BF16 first (cvt.rn.bf16x2.f32),
// then BF16 + BF16 add (add.rn.bf16x2). NOT float-domain add then convert.
//
// where combined[i, j] = bias[j] + pos_embed[i, j] is precomputed on host.
// The table (196 × 768 × 2B = 294 KB) is L2-resident on B200.
//
// Architecture: SM100a (Blackwell) only.
// Integration: CUTLASS 4.x CollectiveBuilder, EVT-based epilogue fusion.
//
// This avoids:
//   - Full [M, N] C matrix read (saves ~1.33 GB HBM traffic vs beta=1 fusion)
//   - Separate unfused post-add kernel (saves kernel launch + full D round-trip)
//
// Usage with CUTLASS CollectiveBuilder:
//
//   // Pass as FusionOpOrCallbacks — NOT a tagged FusionOp, uses passthrough
//   using FusionOp = cutlass::epilogue::fusion::SigLipPeriodicAdd<>;
//
//   using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
//       cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
//       TileShape, ClusterShape,
//       cutlass::epilogue::collective::EpilogueTileAuto,
//       float, float,               // ElementAccumulator, ElementCompute
//       void, LayoutC, AlignC,      // void C — no source load needed
//       ElementD, LayoutD, AlignD,
//       cutlass::epilogue::collective::EpilogueScheduleAuto,
//       FusionOp
//   >::CollectiveOp;
//
//   // Set epilogue arguments:
//   arguments.epilogue.thread.op_1.ptr_combined = ptr;
//   arguments.epilogue.thread.op_1.seq_len = 196;
//

#pragma once

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"

namespace cutlass::epilogue::fusion {

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Sm100PeriodicAddNode — Custom EVT compute node for SM100a
//
// Takes accumulator from child (Sm90AccFetch), adds values from a compact
// periodic table indexed by global_row % seq_len, converts to output type.
//
// No SMEM, no TMA: vectorized 128-bit loads from L2-resident global memory via __ldg().
//
// IMPORTANT: The relative coordinate tensor (tCcD) from ConsumerStoreArgs does NOT
// give actual output (m,n) positions on SM100 — it's only for OOB predication.
// We reconstruct the absolute coordinate tensor from the identity tensor, matching
// the SM100 epilogue's own construction (sm100_epilogue_tma_warpspecialized.hpp).
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class ElementOutput_,     // Output element type (typically bfloat16_t)
  class ElementCompute_,    // Compute precision (typically float)
  class ElementTable_,      // Table element type (typically bfloat16_t)
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct Sm100PeriodicAddNode {

  using ElementOutput  = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementTable   = ElementTable_;
  static constexpr auto RoundStyle = RoundStyle_;

  struct SharedStorage { };

  struct Arguments {
    ElementTable const* ptr_combined = nullptr;   // [seq_len, N] compact table in gmem
    int seq_len = 0;                               // Row period (e.g. 196 for SigLIP2)
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return args.ptr_combined != nullptr && args.seq_len > 0;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args,
      void* workspace, cudaStream_t stream, CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  Sm100PeriodicAddNode() { }

  CUTLASS_HOST_DEVICE
  Sm100PeriodicAddNode(Params const& params, SharedStorage const& shared_storage)
      : params_ptr(&params) { }

  Params const* params_ptr = nullptr;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  // ── ConsumerStoreCallbacks ──
  // Carries per-CTA state including the ABSOLUTE coordinate tensor,
  // provides the visit() that does the actual periodic add.

  template <class AbsCoordTensor>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
        Params const* params_ptr_,
        int N_dim_,
        AbsCoordTensor tCcD_abs_)
      : params_ptr(params_ptr_)
      , N_dim(N_dim_)
      , tCcD_abs(tCcD_abs_) { }

    Params const* params_ptr;
    int N_dim;                    // N dimension of the GEMM problem
    AbsCoordTensor tCcD_abs;      // (T2R,T2R_M,T2R_N,EPI_M,EPI_N) — ABSOLUTE global coords

    // visit() is called per (epi_v, epi_m, epi_n) by the epilogue store loop.
    //
    // frg_acc    = raw FP32 accumulator from TMEM (always available)
    // frg_inputs = output of child node Sm90AccFetch (= frg_acc, the accumulator)
    // epi_v      = fragment index within the epilogue subtile for this thread
    // epi_m/n    = epilogue subtile indices within the CTA tile
    //
    // Returns Array<ElementOutput, FragmentSize> written to D via TMA store.

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE Array<ElementOutput, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc,
          int epi_v,
          int epi_m,
          int epi_n,
          Array<ElementInput, FragmentSize> const& frg_inputs) {

      // Match megakernel numeric semantics:
      //   cvt.rn.bf16x2.f32  — FP32 accumulator → BF16 FIRST
      //   add.rn.bf16x2      — BF16 + BF16 table value (BF16-domain add)
      using ConvertToBF16 = NumericArrayConverter<ElementOutput, ElementInput, FragmentSize, RoundStyle>;

      Array<ElementOutput, FragmentSize> frg_acc_bf16 = ConvertToBF16{}(frg_inputs);
      Array<ElementOutput, FragmentSize> frg_result;

      // ── Vectorized table load ──
      // SM100 T2R mapping guarantees: all FragmentSize elements in one visit() call
      // share the same M-row with contiguous N-columns (SM100_TMEM_LOAD_32dp32b1x:
      // each thread gets 32 consecutive N-values from one M-row of the epilogue subtile).
      // Extract coordinates once from element 0, then do 128-bit vectorized loads.
      auto tCcD_mn = tCcD_abs(_,_,_, epi_m, epi_n);
      auto coord_0 = tCcD_mn(epi_v * FragmentSize);
      int global_m = get<0>(coord_0);
      int global_n = get<1>(coord_0);
      int pos_row  = global_m % params_ptr->seq_len;

      // Base address for this thread's contiguous N-column segment in the periodic table
      __nv_bfloat16 const* __restrict__ base =
          reinterpret_cast<__nv_bfloat16 const*>(params_ptr->ptr_combined)
          + static_cast<long long>(pos_row) * N_dim + global_n;

      // 128-bit vectorized loads: 8 BF16 values per int4 (= 4 loads for FragmentSize=32)
      // N-columns are 128-bit aligned (thread starts at multiple-of-32 column offset,
      // table rows are 768*2B = 1536B aligned, base allocation is 256B-aligned via cudaMalloc)
      constexpr int BF16_PER_VEC = sizeof(int4) / sizeof(ElementTable);  // 16/2 = 8
      constexpr int NUM_VECS = FragmentSize / BF16_PER_VEC;
      constexpr int TAIL = FragmentSize % BF16_PER_VEC;

      __nv_bfloat16 tbl_raw[FragmentSize];
      {
        int4 const* __restrict__ src = reinterpret_cast<int4 const*>(base);
        int4* dst = reinterpret_cast<int4*>(tbl_raw);
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < NUM_VECS; ++v) {
          dst[v] = __ldg(src + v);
        }
      }
      // Handle non-multiple-of-8 FragmentSize (unlikely for SM100, but safe)
      if constexpr (TAIL > 0) {
        __nv_bfloat16 const* src_tail = base + NUM_VECS * BF16_PER_VEC;
        CUTLASS_PRAGMA_UNROLL
        for (int t = 0; t < TAIL; ++t) {
          tbl_raw[NUM_VECS * BF16_PER_VEC + t] = __ldg(src_tail + t);
        }
      }

      // BF16 add: bfloat16_t::operator+ uses __hadd on SM80+ (hardware add.rn.bf16)
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        frg_result[i] = frg_acc_bf16[i] + ElementOutput(tbl_raw[i]);
      }

      return frg_result;
    }
  };

  template <
    bool ReferenceSrc,
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [tile_m, tile_n, tile_k, tile_l] = args.tile_coord_mnkl;

    int N_dim = static_cast<int>(cute::size(N));

    // Build ABSOLUTE coordinate tensor — the relative tCcD from ConsumerStoreArgs
    // only works for predication, not for actual (m,n) position extraction on SM100.
    // Replicate the SM100 epilogue's own construction from the identity tensor.
    auto mD_crd = make_identity_tensor(make_shape(M, N));
    auto cD_mn = local_tile(mD_crd, take<0,2>(args.tile_shape_mnk),
                             make_coord(tile_m, tile_n));
    auto thread_t2r = args.tiled_copy.get_thread_slice(args.thread_idx);
    auto tCcD_abs = thread_t2r.partition_D(flat_divide(cD_mn, args.epi_tile));

    return ConsumerStoreCallbacks<decltype(tCcD_abs)>(
        params_ptr,
        N_dim,
        tCcD_abs);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// SigLipPeriodicAdd — Complete EVT tree for fused GEMM + periodic add
//
// Tree structure:
//   Sm90EVT< Sm100PeriodicAddNode,   ← root: adds table values, converts to output
//            Sm90AccFetch             ← leaf: fetches raw accumulator from TMEM
//          >
//
// The tree evaluates bottom-up:
//   1. Sm90AccFetch::visit() returns the FP32 accumulator
//   2. Sm100PeriodicAddNode::visit() receives it as frg_inputs, loads from
//      the compact table via __ldg(), adds, converts to BF16, returns
//
// This is passed directly to CollectiveBuilder as FusionOpOrCallbacks.
// Since it does NOT inherit from FusionOperation, the builder's callbacks
// passthrough (collective_builder.hpp:101) is used: the type itself becomes
// the FusionCallbacks, bypassing the tagged FusionOp dispatch entirely.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class ElementOutput  = cutlass::bfloat16_t,
  class ElementCompute = float,
  class ElementTable   = cutlass::bfloat16_t,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using SigLipPeriodicAdd = Sm90EVT<
  Sm100PeriodicAddNode<ElementOutput, ElementCompute, ElementTable, RoundStyle>,
  Sm90AccFetch
>;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////
