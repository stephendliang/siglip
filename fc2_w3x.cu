/*
  fc2_w3x.cu — FC2 bias-only kernel (N=768, K=3072), 6-warp rank-1-shaped
  persistent GEMM. Thin shim: the whole kernel + harness lives in the
  shared gemm_w3x_body.cuh; this file only selects the FC2 identity.
  See that header for the FC1/FC2 gate map.

  Target: beat cuBLASLt rank-1
  nvjet_sm100_qqtst_128x256_128x6_2x1_2cta_v_bz_bias_TNT (1.046 ms,
  K=3072). Best on B200: ~1.001 ms (gflip_blkswap TD=54).
*/
#define W3X_FC2
#include "gemm_w3x_body.cuh"
