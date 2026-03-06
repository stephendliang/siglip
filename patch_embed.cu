// Hand-tuned from gen.py output — TK=128, SWIZZLE_128B, 4-stage pipeline
// Target: B200  Batch: 4736  GEMM: [928256,768]×[768,768]^T
// Pipeline: 4-stage (parameterized)  K-iters: 6  MMA/iter: 4  idesc: 0x10400010
// Warps: 2+NUM_EPI_WARPS  cta_group::2  __cluster_dims__(2,1,1)
// Warp-specialized: Load(W0) | MMA(W1,cta_group::2,CTA0 only) | Epilogue(W2+,x32 TMEM ld,interleaved TMA stores)  BF16 output
// tcgen05.mma.cta_group::2.kind::f8f6f4  (E4M3 × E4M3 → FP32)
// Each CTA loads own A (128 rows) + half B (128 cols). MMA produces 256×256 output.

#define N_DIM          768
#define K_DIM          768
#include "kernel_common.cuh"

#ifndef CVT_ADD_FUSED
#define CVT_ADD_FUSED 1    // 1=fused asm (asm-local regs), 0=C++ intrinsics (global regs)
#endif

#define COMB_PADDED_ROWS   224          // ceil(196/32)*32 = 7*32
#define COMB_BLOCK_ROWS    32
#define COMB_BLOCK_COLS    32
#define COMB_ROW_BLOCKS    7            // 224/32
#define COMB_COL_BLOCKS    24           // 768/32
#define COMB_BLOCK_ELEMS   1024         // 32*32

__device__ __forceinline__ uint32_t cvt_add_bf16x2(float lo, float hi, uint32_t combined) {
    __nv_bfloat162 acc = __floats2bfloat162_rn(lo, hi);
    __nv_bfloat162 comb;
    memcpy(&comb, &combined, sizeof(uint32_t));
    acc = __hadd2(acc, comb);
    uint32_t result;
    memcpy(&result, &acc, sizeof(uint32_t));
    return result;
}

#define STS_V4(b0,b1,b2,b3, SADDR) \
    asm volatile( \
        "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" \
        :: "r"(SADDR), "r"(b0), "r"(b1), "r"(b2), "r"(b3) : "memory")

// ── Fused CVT+ADD+STS macro — asm-local .reg intermediates avoid global register inflation ──
// Converts 4 pairs of FP32 accumulators to BF16x2, adds combined BF16x2, stores 16B to SMEM.
// With CVT_ADD_FUSED=1, b0-b3 are .reg locals invisible to ptxas's register allocator.
// With CVT_ADD_FUSED=0, falls back to cvt_add_bf16x2() + STS_V4 (global regs).

#if CVT_ADD_FUSED
#define CVT_ADD_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, c0,c1,c2,c3, SADDR) \
    asm volatile( \
        "{\n\t" \
        ".reg .b32 b0, b1, b2, b3;\n\t" \
        "cvt.rn.bf16x2.f32 b0, %1, %0;\n\t" \
        "add.rn.bf16x2 b0, b0, %8;\n\t" \
        "cvt.rn.bf16x2.f32 b1, %3, %2;\n\t" \
        "add.rn.bf16x2 b1, b1, %9;\n\t" \
        "cvt.rn.bf16x2.f32 b2, %5, %4;\n\t" \
        "add.rn.bf16x2 b2, b2, %10;\n\t" \
        "cvt.rn.bf16x2.f32 b3, %7, %6;\n\t" \
        "add.rn.bf16x2 b3, b3, %11;\n\t" \
        "st.shared.v4.b32 [%12], {b0,b1,b2,b3};\n\t" \
        "}" \
        :: "f"(f0),"f"(f1),"f"(f2),"f"(f3), \
           "f"(f4),"f"(f5),"f"(f6),"f"(f7), \
           "r"(c0),"r"(c1),"r"(c2),"r"(c3), \
           "r"(SADDR) \
        : "memory")
#else
#define CVT_ADD_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, c0,c1,c2,c3, SADDR) \
    do { \
        uint32_t _b0 = cvt_add_bf16x2(f0, f1, c0); \
        uint32_t _b1 = cvt_add_bf16x2(f2, f3, c1); \
        uint32_t _b2 = cvt_add_bf16x2(f4, f5, c2); \
        uint32_t _b3 = cvt_add_bf16x2(f6, f7, c3); \
        STS_V4(_b0, _b1, _b2, _b3, SADDR); \
    } while(0)
#endif

#include "kernel_body.cuh"

// ── Host precompute kernel: bias[c] + pos_embed[r,c] → BF16 combined[r,c] ──

__global__ void precompute_combined(
    const float* __restrict__ bias,
    const float* __restrict__ pos_embed,
    __nv_bfloat16* __restrict__ combined,
    int seq_len, int n_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < COMB_PADDED_ROWS * n_dim) {
        int row = idx / n_dim;
        int col = idx % n_dim;
        int real_row = row % seq_len;
        int block_row = row / COMB_BLOCK_COLS;
        int block_col = col / COMB_BLOCK_COLS;
        int row_in_block = row % COMB_BLOCK_ROWS;
        int col_in_block = col % COMB_BLOCK_COLS;
        int blocked_idx = (block_row * COMB_COL_BLOCKS + block_col) * COMB_BLOCK_ELEMS
                        + row_in_block * COMB_BLOCK_COLS + col_in_block;
        combined[blocked_idx] = __float2bfloat16(bias[col] + pos_embed[real_row * n_dim + col]);
    }
}

// ═════════════════════════════════════════════════════════════
// Host
// ═════════════════════════════════════════════════════════════

int main() {
    setbuf(stdout, NULL);
    printf("SigLIP2 patch embed GEMM — tcgen05 cta_group::2 (%d warps [%d epi], cluster of 2)\n",
           2 + NUM_EPI_WARPS, NUM_EPI_WARPS);
    printf("  GEMM: [%d,%d] x [%d,%d]^T  %d-stage pipeline  inline BF16 combined  SMEM-staged coalesced stores  idesc: 0x%08X\n",
           M_TOTAL, K_DIM, N_DIM, K_DIM, N_STAGES, IDESC);

    uint8_t *d_A, *d_B;
    float *d_bias, *d_pos;
    __nv_bfloat16 *d_combined, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A,    (size_t)M_TOTAL * K_DIM));
    CUDA_CHECK(cudaMalloc(&d_B,    (size_t)N_DIM   * K_DIM));
    CUDA_CHECK(cudaMalloc(&d_bias,  (size_t)N_DIM  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos,   (size_t)SEQ_LEN * N_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_combined, (size_t)COMB_PADDED_ROWS * N_DIM * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_C,    (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16)));

    // A: uniform 0x3C (=1.5 in FP8 E4M3)
    // B: alternating rows — even rows 0x3C (1.5), odd rows 0x38 (1.0)
    CUDA_CHECK(cudaMemset(d_A, 0x3C, (size_t)M_TOTAL * K_DIM));
    {
        uint8_t* h_B = (uint8_t*)malloc((size_t)N_DIM * K_DIM);
        for (int n = 0; n < N_DIM; n++)
            memset(h_B + (size_t)n * K_DIM, (n & 1) ? 0x38 : 0x3C, K_DIM);
        CUDA_CHECK(cudaMemcpy(d_B, h_B, (size_t)N_DIM * K_DIM, cudaMemcpyHostToDevice));
        free(h_B);
    }

    // Non-uniform bias/pos_embed
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

    // Precompute combined[r][c] = __float2bfloat16(bias[c] + pos_embed[r*N_DIM+c])
    {
        int n_elem = COMB_PADDED_ROWS * N_DIM;
        int tpb = 256;
        int bpg = (n_elem + tpb - 1) / tpb;
        precompute_combined<<<bpg, tpb>>>(d_bias, d_pos, d_combined, SEQ_LEN, N_DIM);
        CUDA_CHECK(cudaGetLastError());
    }
    printf("  Alloc + precompute done\n");

    CUtensorMap h_tma_a, h_tma_b;
    {
        uint64_t dims[2]    = {(uint64_t)K_DIM, (uint64_t)M_TOTAL};
        uint64_t strides[1] = {(uint64_t)K_DIM};
        uint32_t box[2]     = {TK, TM};
        uint32_t estrides[2]= {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(&h_tma_a,
            CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)d_A,
            dims, strides, box, estrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }
    {
        uint64_t dims[2]    = {(uint64_t)K_DIM, (uint64_t)N_DIM};
        uint64_t strides[1] = {(uint64_t)K_DIM};
        uint32_t box[2]     = {TK, TN/2};
        uint32_t estrides[2]= {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(&h_tma_b,
            CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)d_B,
            dims, strides, box, estrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }

    CUtensorMap h_tma_c;
    {
        uint64_t dims[2]    = {(uint64_t)N_DIM, (uint64_t)M_TOTAL};
        uint64_t strides[1] = {(uint64_t)N_DIM * sizeof(__nv_bfloat16)};
        uint32_t box[2]     = {64, 32};
        uint32_t estrides[2]= {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(&h_tma_c,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)d_C,
            dims, strides, box, estrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }

    CUDA_CHECK(cudaFuncSetAttribute(persistent_gemm<EpilogueOp::BIAS_ADD>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    printf("  TMA descriptors + func attr done\n");

#ifdef TIMING
    long long *d_timing, *d_spread;
    CUDA_CHECK(cudaMalloc(&d_timing, 74 * TIMING_CLUSTER_STRIDE * sizeof(long long)));
    CUDA_CHECK(cudaMemset(d_timing, 0, 74 * TIMING_CLUSTER_STRIDE * sizeof(long long)));
    size_t spread_bytes = (size_t)74 * MAX_SPREAD_TILES * NUM_EPI_WARPS * sizeof(long long);
    CUDA_CHECK(cudaMalloc(&d_spread, spread_bytes));
    CUDA_CHECK(cudaMemset(d_spread, 0, spread_bytes));
#endif

    // ── Warmup: 2 iterations ──
    printf("Launching warmup (2 iters)...\n");
    for (int _i = 0; _i < 2; _i++) {
    persistent_gemm<EpilogueOp::BIAS_ADD><<<SM_COUNT, THREADS, SMEM_BYTES>>>(h_tma_a, h_tma_b, h_tma_c, d_combined, d_C, nullptr
#ifdef TIMING
        , d_timing, d_spread
#endif
    );
    }
    printf("  Waiting for warmup sync...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Warmup done.\n");

    // ── Timed: 10 iterations ──
    printf("Timing: 10 iterations...\n");
    cudaEvent_t _t0, _t1;
    cudaEventCreate(&_t0);
    cudaEventCreate(&_t1);
    cudaEventRecord(_t0);
    for (int _i = 0; _i < 10; _i++) {
    persistent_gemm<EpilogueOp::BIAS_ADD><<<SM_COUNT, THREADS, SMEM_BYTES>>>(h_tma_a, h_tma_b, h_tma_c, d_combined, d_C, nullptr
#ifdef TIMING
        , d_timing, d_spread
#endif
    );
    }
    cudaEventRecord(_t1);
    cudaEventSynchronize(_t1);
    float _ms;
    cudaEventElapsedTime(&_ms, _t0, _t1);
    _ms /= 10.0f;
    printf("Custom kernel: %.3f ms  %.2f TFLOPS\n",
           _ms, 2.0 * M_TOTAL * N_DIM * K_DIM / _ms / 1e9);
    cudaEventDestroy(_t0);
    cudaEventDestroy(_t1);

    // ── Checksum run ──
    persistent_gemm<EpilogueOp::BIAS_ADD><<<SM_COUNT, THREADS, SMEM_BYTES>>>(h_tma_a, h_tma_b, h_tma_c, d_combined, d_C, nullptr
#ifdef TIMING
        , d_timing, d_spread
#endif
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    __nv_bfloat16* h_C = (__nv_bfloat16*)malloc((size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    // Copy combined (blocked layout) back to host for CPU reference validation
    __nv_bfloat16* h_combined = (__nv_bfloat16*)malloc((size_t)COMB_PADDED_ROWS * N_DIM * sizeof(__nv_bfloat16));
    CUDA_CHECK(cudaMemcpy(h_combined, d_combined, (size_t)COMB_PADDED_ROWS * N_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    // Strided checksum: 1024 samples spread across the full output matrix.
    double cksum = 0;
    {
        long long total_elems = (long long)M_TOTAL * N_DIM;
        long long stride = total_elems / 1024;
        for (int i = 0; i < 1024; i++)
            cksum += (double)__bfloat162float(h_C[(long long)i * stride]);
    }

    // CPU reference spot checks: 32 positions spread across the matrix.
    int errors = 0;
    {
        for (int spot = 0; spot < 32; spot++) {
            long long row = (long long)spot * M_TOTAL / 32;
            int col = (spot * 47) % N_DIM;
            int pos_row = (int)(row % SEQ_LEN);

            float b_val = (col & 1) ? 1.0f : 1.5f;
            float gemm_f32 = (float)K_DIM * 1.5f * b_val;
            float gemm_bf16_f = __bfloat162float(__float2bfloat16(gemm_f32));

            int br = pos_row / COMB_BLOCK_ROWS;
            int rir = pos_row % COMB_BLOCK_ROWS;
            int bc = col / COMB_BLOCK_COLS;
            int cic = col % COMB_BLOCK_COLS;
            int cidx = (br * COMB_COL_BLOCKS + bc) * COMB_BLOCK_ELEMS + rir * COMB_BLOCK_COLS + cic;

            float comb_f = __bfloat162float(h_combined[cidx]);
            __nv_bfloat16 expected = __float2bfloat16(gemm_bf16_f + comb_f);
            __nv_bfloat16 actual = h_C[row * N_DIM + col];

            float ef = __bfloat162float(expected);
            float af = __bfloat162float(actual);
            if (ef != af) {
                if (errors < 5)
                    printf("  MISMATCH at (%lld,%d): expected %.1f got %.1f (combined=%.4f gemm=%.1f)\n",
                           row, col, ef, af, comb_f, gemm_bf16_f);
                errors++;
            }
        }
    }
    int valid = (errors == 0) ? 1 : 0;
    printf("Validation: %d/32 spot checks passed%s\n", 32 - errors, valid ? "" : " — FAILED");
    printf("Checksum (1024 strided): %f\n", cksum);
    printf("C[0,0..3] = %.1f %.1f %.1f %.1f\n",
           __bfloat162float(h_C[0]), __bfloat162float(h_C[1]),
           __bfloat162float(h_C[2]), __bfloat162float(h_C[3]));

    // Diagnostic: dump row 0 cols 0-31 actual vs expected combined effect
    printf("DIAG row0 actual:   ");
    for (int c = 0; c < 32; c++) printf("%.0f ", __bfloat162float(h_C[c]));
    printf("\n");
    printf("DIAG row0 expected: ");
    for (int c = 0; c < 32; c++) {
        float b_val = (c & 1) ? 1.0f : 1.5f;
        float g = __bfloat162float(__float2bfloat16((float)K_DIM * 1.5f * b_val));
        int br0 = 0, rir0 = 0, bc0 = c / COMB_BLOCK_COLS, cic0 = c % COMB_BLOCK_COLS;
        int ci = (br0 * COMB_COL_BLOCKS + bc0) * COMB_BLOCK_ELEMS + rir0 * COMB_BLOCK_COLS + cic0;
        float cf = __bfloat162float(h_combined[ci]);
        printf("%.0f ", __bfloat162float(__float2bfloat16(g + cf)));
    }
    printf("\n");
    printf("DIAG combined[0,0..31]: ");
    for (int c = 0; c < 32; c++) {
        int br0 = 0, rir0 = 0, bc0 = c / COMB_BLOCK_COLS, cic0 = c % COMB_BLOCK_COLS;
        int ci = (br0 * COMB_COL_BLOCKS + bc0) * COMB_BLOCK_ELEMS + rir0 * COMB_BLOCK_COLS + cic0;
        printf("%.0f ", __bfloat162float(h_combined[ci]));
    }
    printf("\n");
    free(h_combined);
    printf("@@RESULT ms=%.3f tflops=%.2f checksum=%f valid=%d c0=%.1f\n",
           _ms, 2.0 * M_TOTAL * N_DIM * K_DIM / _ms / 1e9, cksum, valid,
           __bfloat162float(h_C[0]));

#ifdef TIMING
    print_timing(d_timing, d_spread, spread_bytes, _ms);
#endif

    free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_bias); cudaFree(d_pos);
    cudaFree(d_combined); cudaFree(d_C);
    return 0;
}
