/*
  fc2_w3x_host.cc — host harness for fc2_w3x.ptx (pure-PTX port).

  Mirrors the timing/validation block of fc2_w3x.cu but:
    - Loads a PTX-compiled cubin via the driver API (cuModuleLoadData).
    - Launches with cluster dims via cuLaunchKernelEx.
    - Uses only 5 kernel params (drops the 4 null profile pointers
      that the PTX port trimmed from the signature).

  Build flow (see Makefile rule fc2-w3x-ptx):
    1. nvcc -arch=compute_100a -code=sm_100a -cubin fc2_w3x.ptx -o fc2_w3x.cubin
    2. ld -r -b binary fc2_w3x.cubin -o fc2_w3x_cubin.o
    3. g++/nvcc links fc2_w3x_host.cc + fc2_w3x_cubin.o + libcuda + libcudart_static
*/

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>

#ifndef M_TOTAL
#define M_TOTAL 928256
#endif
#ifndef N_DIM
#define N_DIM   768
#endif
#ifndef K_DIM
#define K_DIM   3072
#endif

#define TM          128
#define TN          256
#define TK          128
#define SM_COUNT    148
#define CLUSTER_CTAS 2
#define TOTAL_WARPS 6
#define THREADS     (TOTAL_WARPS * 32)

#define TILES_M     ((M_TOTAL + TM * 2 - 1) / (TM * 2))
#define TILES_N     ((N_DIM  + TN - 1) / TN)
#define TOTAL_TILES (TILES_M * TILES_N)
#define K_ITERS     (K_DIM / TK)

#define N_STAGES       6
#define NUM_EPI_STAGES 2
#define SUBPASS_COLS   32
#define ROWS_PER_CTA   TM
#define SUBPASS_BYTES  (ROWS_PER_CTA * SUBPASS_COLS * 2)
#define STAGE_BYTES    32768

#define MAIN_SMEM      (N_STAGES * STAGE_BYTES)
#define OUT_STAGING    (NUM_EPI_STAGES * SUBPASS_BYTES)
#define BIAS_BYTES     (N_DIM * 2)
#define OFF_BIAS_BASE  (MAIN_SMEM + OUT_STAGING)
#define OFF_MBARS      ((OFF_BIAS_BASE + BIAS_BYTES + 127) & ~127)
#define SMEM_BYTES     ((OFF_MBARS + 14*8 + 8 + 127) & ~127)

#define CUDA_CHK(x) do { \
    cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
        exit(1); \
    } \
} while(0)

#define CU_CHK(x) do { \
    CUresult r_ = (x); \
    if (r_ != CUDA_SUCCESS) { \
        const char* s_; cuGetErrorString(r_, &s_); \
        fprintf(stderr, "CU %s:%d: %s\n", __FILE__, __LINE__, s_); \
        exit(1); \
    } \
} while(0)

/*
  Embedded cubin from `ld -r -b binary fc2_w3x.cubin -o fc2_w3x_cubin.o`.
  Symbols _binary_fc2_w3x_cubin_{start,end,size} are emitted by the
  binary-mode linker.
*/
extern "C" const unsigned char _binary_fc2_w3x_cubin_start[];
extern "C" const unsigned char _binary_fc2_w3x_cubin_end[];

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    printf("FC2 W3X PTX kernel — hand-authored port of fc2_w3x.cu\n");
    printf("  [%d,%d] x [%d,%d]^T  NS=%d  THREADS=%d  SMEM=%d B  (cap 228KB)\n",
           M_TOTAL, K_DIM, N_DIM, K_DIM, N_STAGES, THREADS, SMEM_BYTES);

    if (TOTAL_TILES % (SM_COUNT / CLUSTER_CTAS) != 0) {
        fprintf(stderr, "  ERROR: TOTAL_TILES=%d %% num_clusters=%d != 0\n",
                TOTAL_TILES, SM_COUNT / CLUSTER_CTAS);
        return 1;
    }

    CUDA_CHK(cudaSetDevice(0));
    CUDA_CHK(cudaFree(0));
    CU_CHK(cuInit(0));

    /*
      cudaFree(0) already primed the runtime context; cuCtxGetCurrent
      will return a valid handle the module loader can use.  No need to
      create a fresh context (driver API has shifted cuCtxCreate
      signature across CUDA 12/13; runtime-managed context is portable).
    */
    CUcontext ctx;
    CU_CHK(cuCtxGetCurrent(&ctx));
    if (!ctx) {
        fprintf(stderr, "no current CUDA context after cudaFree(0) — unexpected\n");
        return 1;
    }

    /*
      Load cubin from .rodata-embedded blob.  cuModuleLoadData accepts
      either PTX or cubin; we pass cubin since we've already done the
      PTX→cubin step via nvcc in the build pipeline.
    */
    CUmodule mod;
    CU_CHK(cuModuleLoadData(&mod, _binary_fc2_w3x_cubin_start));
    CUfunction fc2_w3x_kernel;
    CU_CHK(cuModuleGetFunction(&fc2_w3x_kernel, mod, "fc2_w3x_kernel"));

    /*
      Opt into the full dynamic SMEM budget (B200: 228 KB/SM).  Without
      this call, the driver caps at 48 KB default.
    */
    CU_CHK(cuFuncSetAttribute(fc2_w3x_kernel,
                              CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                              SMEM_BYTES));

    /* ---- device allocs + packed-layout initialization (matches fc2_w3x.cu) ---- */
    __nv_fp8_e4m3 *d_A=nullptr, *d_B=nullptr;
    __nv_bfloat16 *d_bias=nullptr, *d_C=nullptr;
    size_t sA = (size_t)M_TOTAL * K_DIM;
    size_t sB = (size_t)N_DIM   * K_DIM;
    size_t sC = (size_t)M_TOTAL * N_DIM;
    CUDA_CHK(cudaMalloc(&d_A, sA));
    CUDA_CHK(cudaMalloc(&d_B, sB));
    CUDA_CHK(cudaMalloc(&d_bias, (size_t)N_DIM * sizeof(__nv_bfloat16)));
    CUDA_CHK(cudaMalloc(&d_C, sC * sizeof(__nv_bfloat16)));

    __nv_fp8_e4m3 *hA = (__nv_fp8_e4m3*)malloc(sA);
    __nv_fp8_e4m3 *hB = (__nv_fp8_e4m3*)malloc(sB);
    __nv_bfloat16 *hbias = (__nv_bfloat16*)malloc((size_t)N_DIM * sizeof(__nv_bfloat16));

    /*
      PACKED_TILES layout.  A is packed (a_m_tile, k_block, row, k) so the
      TMA's 2-D tile descriptor can stream it.
    */
    for (size_t a_m_tile = 0; a_m_tile < (size_t)(M_TOTAL / TM); a_m_tile++) {
        for (size_t k_block = 0; k_block < (size_t)K_ITERS; k_block++) {
            for (size_t r = 0; r < (size_t)TM; r++) {
                long long global_row = (long long)a_m_tile * TM + r;
                if (global_row >= M_TOTAL) continue;
                for (size_t k = 0; k < (size_t)TK; k++) {
                    float av = 1.0f + 0.125f * (float)((int)global_row & 7);
                    size_t off = ((a_m_tile * K_ITERS + k_block) * TM + r) * TK + k;
                    hA[off] = static_cast<__nv_fp8_e4m3>(av);
                }
            }
        }
    }
    for (size_t b_n_half = 0; b_n_half < (size_t)(N_DIM / (TN/2)); b_n_half++) {
        for (size_t k_block = 0; k_block < (size_t)K_ITERS; k_block++) {
            for (size_t c = 0; c < (size_t)(TN/2); c++) {
                long long global_col = (long long)b_n_half * (TN/2) + c;
                if (global_col >= N_DIM) continue;
                for (size_t k = 0; k < (size_t)TK; k++) {
                    float bv = (global_col & 1) ? 1.0f : 1.5f;
                    size_t off = ((b_n_half * K_ITERS + k_block) * (TN/2) + c) * TK + k;
                    hB[off] = static_cast<__nv_fp8_e4m3>(bv);
                }
            }
        }
    }
    for (int n = 0; n < N_DIM; n++) hbias[n] = __float2bfloat16(0.25f * (float)(n & 7));

    CUDA_CHK(cudaMemcpy(d_A, hA, sA, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_B, hB, sB, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_bias, hbias, (size_t)N_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemset(d_C, 0, sC * sizeof(__nv_bfloat16)));
    printf("  Alloc + init + pack done\n");

    /* ---- CUtensorMap setup (driver API — same as fc2_w3x.cu) ---- */
    CUtensorMap h_tma_a, h_tma_b, h_tma_c;
    {
        uint64_t a_total_rows = (uint64_t)(M_TOTAL / TM) * (uint64_t)K_ITERS * (uint64_t)TM;
        uint64_t dims[2]    = {(uint64_t)TK, a_total_rows};
        uint64_t strides[1] = {(uint64_t)TK};
        uint32_t box[2]     = {TK, TM};
        uint32_t estrides[2]= {1, 1};
        CU_CHK(cuTensorMapEncodeTiled(&h_tma_a,
            CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)d_A,
            dims, strides, box, estrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }
    {
        uint64_t b_total_rows = (uint64_t)(N_DIM / (TN/2)) * (uint64_t)K_ITERS * (uint64_t)(TN/2);
        uint64_t dims[2]    = {(uint64_t)TK, b_total_rows};
        uint64_t strides[1] = {(uint64_t)TK};
        uint32_t box[2]     = {TK, TN / 2};
        uint32_t estrides[2]= {1, 1};
        CU_CHK(cuTensorMapEncodeTiled(&h_tma_b,
            CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)d_B,
            dims, strides, box, estrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }
    {
        uint64_t dims[2]    = {(uint64_t)N_DIM, (uint64_t)M_TOTAL};
        uint64_t strides[1] = {(uint64_t)N_DIM * sizeof(__nv_bfloat16)};
        uint32_t box[2]     = {SUBPASS_COLS, ROWS_PER_CTA};
        uint32_t estrides[2]= {1, 1};
        CU_CHK(cuTensorMapEncodeTiled(&h_tma_c,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)d_C,
            dims, strides, box, estrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }

    /*
      Kernel param pack: 3 × CUtensorMap (128-byte by-value structs)
      followed by 2 × u64 pointers.  Driver-API launch requires the
      kernel_params array point at addresses of these values.
    */
    void* params[5] = {
        &h_tma_a, &h_tma_b, &h_tma_c,
        &d_bias, &d_C
    };

    /*
      Cluster launch config: grid.x = SM_COUNT, block.x = 192, shared
      memory = SMEM_BYTES, cluster_dim = (2,1,1).
    */
    CUlaunchConfig cfg{};
    cfg.gridDimX = SM_COUNT;
    cfg.gridDimY = 1;
    cfg.gridDimZ = 1;
    cfg.blockDimX = THREADS;
    cfg.blockDimY = 1;
    cfg.blockDimZ = 1;
    cfg.sharedMemBytes = SMEM_BYTES;
    cfg.hStream = 0;

    CUlaunchAttribute attrs[1]{};
    attrs[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    attrs[0].value.clusterDim.x = CLUSTER_CTAS;
    attrs[0].value.clusterDim.y = 1;
    attrs[0].value.clusterDim.z = 1;
    cfg.numAttrs = 1;
    cfg.attrs = attrs;

    auto launch = [&]() {
        CU_CHK(cuLaunchKernelEx(&cfg, fc2_w3x_kernel, params, nullptr));
    };

    /* Warmup */
    const int N_WARMUP = 2;
    printf("Warmup (%d iters)...\n", N_WARMUP);
    for (int i = 0; i < N_WARMUP; i++) launch();
    CUDA_CHK(cudaDeviceSynchronize());

    /* Timed run */
    const int N_TIMED_LAUNCHES = 10;
    printf("Timing %d iters...\n", N_TIMED_LAUNCHES);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < N_TIMED_LAUNCHES; i++) launch();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= (float)N_TIMED_LAUNCHES;
    printf("FC2-W3X-PTX kernel: %.3f ms  %.2f TFLOPS\n",
           ms, 2.0 * M_TOTAL * N_DIM * K_DIM / ms / 1e9);

    /* Validation against CPU reference — spot-check 32 positions */
    __nv_bfloat16* h_C = (__nv_bfloat16*)malloc(sC * sizeof(__nv_bfloat16));
    CUDA_CHK(cudaMemcpy(h_C, d_C, sC * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int spot = 0; spot < 32; spot++) {
        long long row = (long long)spot * M_TOTAL / 32;
        int col = (spot * 47) % N_DIM;
        float av = 1.0f + 0.125f * (float)((int)row & 7);
        float bv = (col & 1) ? 1.0f : 1.5f;
        float expected_ab = av * bv * K_DIM;
        float expected = expected_ab + __bfloat162float(hbias[col]);
        float got = __bfloat162float(h_C[row * N_DIM + col]);
        float rel = fabsf(got - expected) / fabsf(expected);
        if (rel > 0.02f) {
            if (errors < 8) fprintf(stderr, "  MISMATCH [%lld,%d] got=%.1f exp=%.1f\n",
                                    row, col, got, expected);
            errors++;
        }
    }
    printf("%s  errors=%d/32\n", errors == 0 ? "PASS" : "FAIL", errors);
    int valid = (errors == 0) ? 1 : 0;
    float c0 = __bfloat162float(h_C[0]);
    printf("@@RESULT ms=%.4f tflops=%.2f checksum=0.000000 valid=%d c0=%.1f\n",
           ms, 2.0 * M_TOTAL * N_DIM * K_DIM / ms / 1e9, valid, c0);

    free(hA); free(hB); free(hbias); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_bias); cudaFree(d_C);
    cuModuleUnload(mod);
    return errors == 0 ? 0 : 1;
}
