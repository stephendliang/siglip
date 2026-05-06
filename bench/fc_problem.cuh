/*
  Single source of truth for the FC1/FC2 cuBLASLt problem definition shared
  by cublas_bench.cu and cublaslt_introspect.cu. Layout drift between the
  two harnesses (output [N,M] vs [M,N], FP32 vs BF16 bias) cost months of
  bad reference data; consumers go through this header so any future change
  applies to both.

  Convention (matches the production kernels):

    A = WEIGHT,     col-major [K, N], ldA = K, transA = T  -> A^T = [N, K]
    B = ACTIVATION, col-major [K, M], ldB = K, transB = N  -> B   = [K, M]
    D = OUTPUT,     col-major [N, M], ldD = N
        D = A^T @ B = [N, M]
    bias broadcast along rows -> bias_len = N (output channels)
    bias dtype = BF16 (default)

  Bias dtype is BF16 because cuBLASLt's FP8 fused-bias kernels on sm_100a /
  CUDA 13.0 only enumerate when bias dtype is BF16. FP32 bias dtype gets you
  zero algos and a CUBLAS_STATUS_NOT_SUPPORTED at AlgoGetHeuristic. Verified
  via 4-cell probe ({[N,M], [M,N]} x {BF16, FP32}) — bias dtype is the lever,
  layout orientation is not. Override via DescConfig.bias_dtype if you need
  to test the FP32 path.

  M = batch tokens, N = output channels, K = reduction.
  cublasLtMatmul receives args in (A, B, D) order at the call site.
*/

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublasLt.h>

namespace fc {

enum class Epilogue : int {
    NONE         = 0,
    PERIODIC_ADD = 1,
    GELU_BIAS    = 2,
    BIAS_ONLY    = 3,
};

enum class ScaleMode : int {
    PER_TENSOR = 0,
    MXFP8      = 1,
};

constexpr cudaDataType_t A_DTYPE            = CUDA_R_8F_E4M3;
constexpr cudaDataType_t B_DTYPE            = CUDA_R_8F_E4M3;
constexpr cudaDataType_t D_DTYPE            = CUDA_R_16BF;
constexpr cudaDataType_t DEFAULT_BIAS_DTYPE = CUDA_R_16BF;

inline const char* epi_name(Epilogue e) {
    switch (e) {
        case Epilogue::NONE:         return "none";
        case Epilogue::PERIODIC_ADD: return "periodic_add";
        case Epilogue::GELU_BIAS:    return "gelu_bias";
        case Epilogue::BIAS_ONLY:    return "bias_only";
    }
    return "?";
}

inline bool epi_uses_bias(Epilogue e) {
    return e == Epilogue::GELU_BIAS || e == Epilogue::BIAS_ONLY;
}

#define FC_CUBLAS_CHECK(x) do {                                                 \
    cublasStatus_t _s = (x);                                                    \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                          \
        std::fprintf(stderr, "fc_problem.cuh: cublas %d at %s:%d\n",            \
                     (int)_s, __FILE__, __LINE__);                              \
        std::exit(1);                                                           \
    }                                                                           \
} while (0)

struct Layouts {
    cublasLtMatrixLayout_t A = nullptr;  // [K, N] FP8
    cublasLtMatrixLayout_t B = nullptr;  // [K, M] FP8
    cublasLtMatrixLayout_t D = nullptr;  // [N, M] BF16
};

inline Layouts make_layouts(int M, int N, int K) {
    Layouts l;
    FC_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&l.A, A_DTYPE, K, N, K));
    FC_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&l.B, B_DTYPE, K, M, K));
    FC_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&l.D, D_DTYPE, N, M, N));
    return l;
}

inline void destroy_layouts(Layouts& l) {
    if (l.A) cublasLtMatrixLayoutDestroy(l.A);
    if (l.B) cublasLtMatrixLayoutDestroy(l.B);
    if (l.D) cublasLtMatrixLayoutDestroy(l.D);
    l = {};
}

struct DescConfig {
    Epilogue epi = Epilogue::NONE;
    ScaleMode scale = ScaleMode::PER_TENSOR;
    /*
      Pass d_scaleA/d_scaleB to bind explicit scale pointers. nullptr leaves
      cuBLASLt to default to identity (1.0). For PER_TENSOR both are valid;
      for MXFP8 both pointers are required (block-scale buffers).
    */
    const void* d_scaleA = nullptr;
    const void* d_scaleB = nullptr;
    /*
      Required when epi_uses_bias(epi). Length-N vector in bias_dtype format.
      Default bias_dtype = BF16 (only dtype with fused-bias kernels in
      cuBLASLt sm_100a / CUDA 13.0). Set CUDA_R_32F to test the no-fused-algo
      path explicitly.
    */
    const void* d_bias = nullptr;
    cudaDataType_t bias_dtype = DEFAULT_BIAS_DTYPE;
};

inline cublasLtMatmulDesc_t make_desc(const DescConfig& cfg) {
    cublasLtMatmulDesc_t d = nullptr;
    FC_CUBLAS_CHECK(cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    FC_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    FC_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    if (cfg.scale == ScaleMode::MXFP8) {
        int32_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        FC_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)));
        FC_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)));
    }
    if (cfg.d_scaleA) {
        FC_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &cfg.d_scaleA, sizeof(cfg.d_scaleA)));
    }
    if (cfg.d_scaleB) {
        FC_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &cfg.d_scaleB, sizeof(cfg.d_scaleB)));
    }

    if (epi_uses_bias(cfg.epi)) {
        cublasLtEpilogue_t e = (cfg.epi == Epilogue::GELU_BIAS)
            ? CUBLASLT_EPILOGUE_GELU_BIAS
            : CUBLASLT_EPILOGUE_BIAS;
        FC_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_EPILOGUE, &e, sizeof(e)));
        FC_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &cfg.d_bias, sizeof(cfg.d_bias)));
        FC_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &cfg.bias_dtype, sizeof(cfg.bias_dtype)));
    }
    return d;
}

inline size_t bias_elem_size(cudaDataType_t bias_dtype) {
    switch (bias_dtype) {
        case CUDA_R_16BF: return 2;
        case CUDA_R_16F:  return 2;
        case CUDA_R_32F:  return 4;
        default:          return 4;
    }
}

inline size_t bias_bytes(int N, cudaDataType_t bias_dtype = DEFAULT_BIAS_DTYPE) {
    return (size_t)N * bias_elem_size(bias_dtype);
}

}  // namespace fc
