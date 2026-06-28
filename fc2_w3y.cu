/*
  fc2_w3y.cu — FC2 fused-residual port of fc2_w3x (N=768, K=3072).

  EXPERIMENTAL — resadd port Round 3 (W4-does-both). Identical to fc2_w3x
  except -DHAS_RESIDUAL turns on the residual add:

      out = BF16(A·B) + bias + residual

  Design (see CLAUDE.md "FC2 fused status" + memory/project-fc2-resadd-port):
    - NS=6 is preserved by NOT allocating a residual buffer. The residual
      rides the SAME epilogue output-staging double-buffer (OFF_OUT,
      NUM_EPI_STAGES=2), exactly fc2_w3's "each stage holds residual OR
      output sequentially" trick — zero net SMEM over bias-only.
    - W4 (the existing TMA-load warp) ALSO issues the residual TMA loads
      (W0+W2 of fc2_w3 merged into one warp). TMA transfers are async, so
      W4 issues A+B for the mainloop AND residual for the epilogue. The
      open question this kernel exists to answer: does folding residual
      issue into W4 perturb the mainloop TMA cadence (the +41-77% tma_issue
      sensitivity / KERN_3WARP "idle warp isn't free issue" risk)?
      Fallback if it does: -DRESIDUAL_DEDICATED_WARP promotes residual to a
      7th warp (mirrors fc2_w3's W0/W2 split). [TODO: not yet wired.]

  Everything residual-related is gated on HAS_RESIDUAL in gemm_w3x_body.cuh
  and epilogue_ops.cuh, so building WITHOUT it is byte-identical to fc2_w3x
  (proven via the pp-diff SASS-invariance gate; see
  memory/project-w3-workstealing-strip).
*/
#define W3X_FC2
#ifndef NO_RESIDUAL
#define HAS_RESIDUAL
#endif
#include "gemm_w3x_body.cuh"
