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
//   arguments.epilogue.thread = {
//       {{}, {}},                                    // Sm100PeriodicAddNode args (inner)
//       {}                                           // Sm90AccFetch args (empty)
//   };
//   // Then set the periodic add args:
//   cute::get<0>(cute::get<0>(arguments.epilogue.thread)) = {d_combined, 196};
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
// No SMEM, no TMA: loads directly from L2-resident global memory via __ldg().
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
  // Carries per-CTA state computed in get_consumer_store_callbacks,
  // provides the visit() that does the actual periodic add.

  template <class ThrCoordTensor, class ThrResidue>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
        Params const* params_ptr_,
        int cta_m_base_,
        int cta_n_base_,
        int N_dim_,
        ThrCoordTensor tCcD_,
        ThrResidue residue_tCcD_)
      : params_ptr(params_ptr_)
      , cta_m_base(cta_m_base_)
      , cta_n_base(cta_n_base_)
      , N_dim(N_dim_)
      , tCcD(tCcD_)
      , residue_tCcD(residue_tCcD_) { }

    Params const* params_ptr;
    int cta_m_base;               // tile_coord_m * TileM — global M offset for this CTA
    int cta_n_base;               // tile_coord_n * TileN — global N offset for this CTA
    int N_dim;                    // N dimension of the GEMM problem
    ThrCoordTensor tCcD;          // (CPY,CPY_M,CPY_N,EPI_M,EPI_N) — per-thread coordinates
    ThrResidue residue_tCcD;      // residue for OOB predication

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
      // NOT: float(acc) + float(table) → BF16 (which would preserve FP32 precision through the add)
      using ConvertToBF16 = NumericArrayConverter<ElementOutput, ElementInput, FragmentSize, RoundStyle>;

      Array<ElementOutput, FragmentSize> frg_acc_bf16 = ConvertToBF16{}(frg_inputs);
      Array<ElementOutput, FragmentSize> frg_result;

      auto tCcD_mn = tCcD(_,_,_, epi_m, epi_n);

      int const seq_len = params_ptr->seq_len;
      ElementTable const* __restrict__ ptr = params_ptr->ptr_combined;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        auto coord = tCcD_mn(epi_v * FragmentSize + i);
        int local_m = get<0>(coord);
        int local_n = get<1>(coord);

        // Global coordinates — local_n is CTA-tile-local, needs tile N base
        int global_m = cta_m_base + local_m;
        int global_n = cta_n_base + local_n;
        int pos_row  = global_m % seq_len;

        // L2-resident load from compact [seq_len, N] table
        // Cast to __nv_bfloat16* — __ldg() has no cutlass::bfloat16_t overload
        __nv_bfloat16 raw = __ldg(reinterpret_cast<__nv_bfloat16 const*>(ptr) +
                                  static_cast<long long>(pos_row) * N_dim + global_n);
        ElementOutput tbl_val(raw);

        // BF16 add: bfloat16_t::operator+ promotes to float, adds, rounds back —
        // matches hardware add.rn.bf16x2 semantics: rn_bf16(float(a) + float(b))
        frg_result[i] = frg_acc_bf16[i] + static_cast<ElementOutput>(tbl_val);
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
    // Extract tile coordinates and problem shape
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [tile_m, tile_n, tile_k, tile_l] = args.tile_coord_mnkl;

    // CTA's base M and N offsets in the global output matrix
    int cta_m_base = tile_m * cute::size<0>(args.tile_shape_mnk);
    int cta_n_base = tile_n * cute::size<1>(args.tile_shape_mnk);
    int N_dim = static_cast<int>(cute::size(N));

    return ConsumerStoreCallbacks<decltype(args.tCcD), decltype(args.residue_tCcD)>(
        params_ptr,
        cta_m_base,
        cta_n_base,
        N_dim,
        args.tCcD,
        args.residue_tCcD);
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
