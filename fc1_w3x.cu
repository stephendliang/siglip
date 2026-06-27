/*
  fc1_w3x.cu — FC1 GELU+BIAS kernel (N=3072, K=768), 6-warp rank-1-shaped
  persistent GEMM. Thin shim: the whole kernel + harness lives in the
  shared gemm_w3x_body.cuh; this file only selects the FC1 identity.
  See that header for the FC1/FC2 gate map.

  Target: close the ~47 us gap to cuBLASLt MXFP8 rank-1 (1.951 ms) at FC1
  production K=768; beat PerTensor rank-1 (2.414 ms). Best ~2.025 ms
  (zigzag TD=11 + K_STAGGER=1).
*/
#define W3X_FC1
#include "gemm_w3x_body.cuh"
