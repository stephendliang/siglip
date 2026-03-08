/*
FC1+GELU kernel — derived from patch_embed.cu (patch embed GEMM)
Target: B200  Batch: 4736  GEMM: [928256,768]×[768,3072]^T + bias + GELU
Pipeline: 4-stage (parameterized)  K-iters: 6  MMA/iter: 4  idesc: 0x10400010
Warps: 2+NUM_EPI_WARPS  cta_group::2  __cluster_dims__(2,1,1)
Warp-specialized: Load(W0) | MMA(W1,cta_group::2,CTA0 only) | Epilogue(W2+,x32 TMEM ld,interleaved TMA stores)  BF16 output
tcgen05.mma.cta_group::2.kind::f8f6f4  (E4M3 × E4M3 → FP32)
Each CTA loads own A (128 rows) + half B (128 cols). MMA produces 256×256 output.
Epilogue: FP32 acc + bias → GELU → BF16 CVT → SMEM staging → TMA store
*/

#define N_DIM          3072
#define K_DIM          768
#include "kernel_common.cuh"

/*
GELU approximation (tanh version) — host only, for validation
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
*/
static __host__ __forceinline__ float gelu_fwd(float x) {
    const float k = 0.7978845608f;
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

/*
Fused bias+GELU+CVT+STS — all in one asm block.
Computes GELU(acc+bias) → BF16 CVT → SMEM store using tanh.approx.f32
(single SFU instruction) instead of software tanhf().

For each lane i:
  x = fi + bi                    (acc + bias)
  inner = 0.7978845608 * x * (1 + 0.044715 * x²)
  t = tanh.approx(inner)
  gi = 0.5 * x * (1 + t)         = 0.5 * (x + x*t)

All intermediates in asm-local regs to avoid register inflation.
*/
#define GELU_CVT_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, b0,b1,b2,b3,b4,b5,b6,b7, SADDR) \
    do { \
        asm volatile( \
            "{\n\t" \
            ".reg .f32 x0,x1,x2,x3,x4,x5,x6,x7;\n\t" \
            ".reg .f32 t0,t1,t2,t3,t4,t5,t6,t7;\n\t" \
            ".reg .f32 g0,g1,g2,g3,g4,g5,g6,g7;\n\t" \
            ".reg .b32 o0,o1,o2,o3;\n\t" \
            /* x = acc + bias */ \
            "add.f32 x0, %0, %8;\n\t" \
            "add.f32 x1, %1, %9;\n\t" \
            "add.f32 x2, %2, %10;\n\t" \
            "add.f32 x3, %3, %11;\n\t" \
            "add.f32 x4, %4, %12;\n\t" \
            "add.f32 x5, %5, %13;\n\t" \
            "add.f32 x6, %6, %14;\n\t" \
            "add.f32 x7, %7, %15;\n\t" \
            /* t = tanh.approx(k * x * (1 + k1 * x²))
               rewritten as: inner = k*x + (k*k1)*x³
               where k = 0.7978845608, k1 = 0.044715, k*k1 = 0.035677408136 */ \
            "mul.f32 t0, x0, x0;\n\t" /* t = x² */ \
            "mul.f32 t1, x1, x1;\n\t" \
            "mul.f32 t2, x2, x2;\n\t" \
            "mul.f32 t3, x3, x3;\n\t" \
            "mul.f32 t4, x4, x4;\n\t" \
            "mul.f32 t5, x5, x5;\n\t" \
            "mul.f32 t6, x6, x6;\n\t" \
            "mul.f32 t7, x7, x7;\n\t" \
            "fma.rn.f32 t0, t0, 0f3D122279, 0f3F4C422A;\n\t" /* t = (k*k1)*x² + k */ \
            "fma.rn.f32 t1, t1, 0f3D122279, 0f3F4C422A;\n\t" \
            "fma.rn.f32 t2, t2, 0f3D122279, 0f3F4C422A;\n\t" \
            "fma.rn.f32 t3, t3, 0f3D122279, 0f3F4C422A;\n\t" \
            "fma.rn.f32 t4, t4, 0f3D122279, 0f3F4C422A;\n\t" \
            "fma.rn.f32 t5, t5, 0f3D122279, 0f3F4C422A;\n\t" \
            "fma.rn.f32 t6, t6, 0f3D122279, 0f3F4C422A;\n\t" \
            "fma.rn.f32 t7, t7, 0f3D122279, 0f3F4C422A;\n\t" \
            "mul.f32 t0, t0, x0;\n\t"  /* t = x * ((k*k1)*x² + k) = k*x*(1 + k1*x²) */ \
            "mul.f32 t1, t1, x1;\n\t" \
            "mul.f32 t2, t2, x2;\n\t" \
            "mul.f32 t3, t3, x3;\n\t" \
            "mul.f32 t4, t4, x4;\n\t" \
            "mul.f32 t5, t5, x5;\n\t" \
            "mul.f32 t6, t6, x6;\n\t" \
            "mul.f32 t7, t7, x7;\n\t" \
            "tanh.approx.f32 t0, t0;\n\t" \
            "tanh.approx.f32 t1, t1;\n\t" \
            "tanh.approx.f32 t2, t2;\n\t" \
            "tanh.approx.f32 t3, t3;\n\t" \
            "tanh.approx.f32 t4, t4;\n\t" \
            "tanh.approx.f32 t5, t5;\n\t" \
            "tanh.approx.f32 t6, t6;\n\t" \
            "tanh.approx.f32 t7, t7;\n\t" \
            /* g = 0.5 * (x + x*t) = 0.5 * x * (1 + t) */ \
            "fma.rn.f32 g0, x0, t0, x0;\n\t" /* g = x*t + x */ \
            "fma.rn.f32 g1, x1, t1, x1;\n\t" \
            "fma.rn.f32 g2, x2, t2, x2;\n\t" \
            "fma.rn.f32 g3, x3, t3, x3;\n\t" \
            "fma.rn.f32 g4, x4, t4, x4;\n\t" \
            "fma.rn.f32 g5, x5, t5, x5;\n\t" \
            "fma.rn.f32 g6, x6, t6, x6;\n\t" \
            "fma.rn.f32 g7, x7, t7, x7;\n\t" \
            "mul.f32 g0, g0, 0f3F000000;\n\t" /* g *= 0.5 */ \
            "mul.f32 g1, g1, 0f3F000000;\n\t" \
            "mul.f32 g2, g2, 0f3F000000;\n\t" \
            "mul.f32 g3, g3, 0f3F000000;\n\t" \
            "mul.f32 g4, g4, 0f3F000000;\n\t" \
            "mul.f32 g5, g5, 0f3F000000;\n\t" \
            "mul.f32 g6, g6, 0f3F000000;\n\t" \
            "mul.f32 g7, g7, 0f3F000000;\n\t" \
            /* CVT to BF16x2 + STS */ \
            "cvt.rn.bf16x2.f32 o0, g1, g0;\n\t" \
            "cvt.rn.bf16x2.f32 o1, g3, g2;\n\t" \
            "cvt.rn.bf16x2.f32 o2, g5, g4;\n\t" \
            "cvt.rn.bf16x2.f32 o3, g7, g6;\n\t" \
            "st.shared.v4.b32 [%16], {o0,o1,o2,o3};\n\t" \
            "}" \
            :: "f"(f0),"f"(f1),"f"(f2),"f"(f3), \
               "f"(f4),"f"(f5),"f"(f6),"f"(f7), \
               "f"(b0),"f"(b1),"f"(b2),"f"(b3), \
               "f"(b4),"f"(b5),"f"(b6),"f"(b7), \
               "r"(SADDR) \
            : "memory"); \
    } while(0)

#include "kernel_body.cuh"

// Host

int main() {
    setbuf(stdout, NULL);
    printf("FC1+GELU GEMM — tcgen05 cta_group::2 (%d warps [%d epi], cluster of 2)\n",
           2 + NUM_EPI_WARPS, NUM_EPI_WARPS);
    printf("  GEMM: [%d,%d] x [%d,%d]^T  %d-stage pipeline  bias+GELU  SMEM-staged stores  idesc: 0x%08X\n",
           M_TOTAL, K_DIM, N_DIM, K_DIM, N_STAGES, IDESC);

    uint8_t *d_A, *d_B;
    float *d_bias;
    __nv_bfloat16 *d_C;
    CUDA_CHECK(cudaMalloc(&d_A,    (size_t)M_TOTAL * K_DIM));
    CUDA_CHECK(cudaMalloc(&d_B,    (size_t)N_DIM   * K_DIM));
    CUDA_CHECK(cudaMalloc(&d_bias,  (size_t)N_DIM  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C,    (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16)));

    /*
    A: uniform 0x3C (=1.5 in FP8 E4M3)
    B: alternating rows — even rows 0x3C (1.5), odd rows 0x38 (1.0)
    */
    CUDA_CHECK(cudaMemset(d_A, 0x3C, (size_t)M_TOTAL * K_DIM));
    {
        uint8_t* h_B = (uint8_t*)malloc((size_t)N_DIM * K_DIM);
        for (int n = 0; n < N_DIM; n++)
            memset(h_B + (size_t)n * K_DIM, (n & 1) ? 0x38 : 0x3C, K_DIM);
        CUDA_CHECK(cudaMemcpy(d_B, h_B, (size_t)N_DIM * K_DIM, cudaMemcpyHostToDevice));
        free(h_B);
    }

    // Non-uniform bias: bias[c] = c + 1
    {
        float* h_bias = (float*)malloc((size_t)N_DIM * sizeof(float));
        for (int c = 0; c < N_DIM; c++)
            h_bias[c] = (float)(c + 1);
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias, (size_t)N_DIM * sizeof(float), cudaMemcpyHostToDevice));
        free(h_bias);
    }
    printf("  Alloc done\n");

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

    CUDA_CHECK(cudaFuncSetAttribute(persistent_gemm<EpilogueOp::BIAS_GELU>,
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

    // Warmup: 2 iterations
    printf("Launching warmup (2 iters)...\n");
    for (int _i = 0; _i < 2; _i++) {
    persistent_gemm<EpilogueOp::BIAS_GELU><<<SM_COUNT, THREADS, SMEM_BYTES>>>(h_tma_a, h_tma_b, h_tma_c, d_bias, d_C, nullptr
#ifdef TIMING
        , d_timing, d_spread
#endif
    );
    }
    printf("  Waiting for warmup sync...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Warmup done.\n");

    // Timed: 10 iterations
    printf("Timing: 10 iterations...\n");
    cudaEvent_t _t0, _t1;
    cudaEventCreate(&_t0);
    cudaEventCreate(&_t1);
    cudaEventRecord(_t0);
    for (int _i = 0; _i < 10; _i++) {
    persistent_gemm<EpilogueOp::BIAS_GELU><<<SM_COUNT, THREADS, SMEM_BYTES>>>(h_tma_a, h_tma_b, h_tma_c, d_bias, d_C, nullptr
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
    printf("FC1+GELU kernel: %.3f ms  %.2f TFLOPS\n",
           _ms, 2.0 * M_TOTAL * N_DIM * K_DIM / _ms / 1e9);
    cudaEventDestroy(_t0);
    cudaEventDestroy(_t1);

    // Checksum run
    persistent_gemm<EpilogueOp::BIAS_GELU><<<SM_COUNT, THREADS, SMEM_BYTES>>>(h_tma_a, h_tma_b, h_tma_c, d_bias, d_C, nullptr
#ifdef TIMING
        , d_timing, d_spread
#endif
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    __nv_bfloat16* h_C = (__nv_bfloat16*)malloc((size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    // Strided checksum: 1024 samples spread across the full output matrix
    double cksum = 0;
    {
        long long total_elems = (long long)M_TOTAL * N_DIM;
        long long stride = total_elems / 1024;
        for (int i = 0; i < 1024; i++)
            cksum += (double)__bfloat162float(h_C[(long long)i * stride]);
    }

    // CPU reference spot checks: 32 positions spread across the matrix
    int errors = 0;
    {
        for (int spot = 0; spot < 32; spot++) {
            long long row = (long long)spot * M_TOTAL / 32;
            int col = (spot * 47) % N_DIM;

            float b_val = (col & 1) ? 1.0f : 1.5f;
            float gemm = (float)K_DIM * 1.5f * b_val;
            float bias = (float)(col + 1);
            float expected_f32 = gelu_fwd(gemm + bias);
            __nv_bfloat16 expected = __float2bfloat16(expected_f32);
            __nv_bfloat16 actual = h_C[row * N_DIM + col];

            float ef = __bfloat162float(expected);
            float af = __bfloat162float(actual);
            if (ef != af) {
                if (errors < 5)
                    printf("  MISMATCH at (%lld,%d): expected %.1f got %.1f (gemm=%.1f bias=%.1f gelu=%.4f)\n",
                           row, col, ef, af, gemm, bias, expected_f32);
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

    // Diagnostic: dump row 0 cols 0-7 actual vs expected
    printf("DIAG row0 actual:   ");
    for (int c = 0; c < 8; c++) printf("%.1f ", __bfloat162float(h_C[c]));
    printf("\n");
    printf("DIAG row0 expected: ");
    for (int c = 0; c < 8; c++) {
        float b_val = (c & 1) ? 1.0f : 1.5f;
        float g = (float)K_DIM * 1.5f * b_val;
        float b = (float)(c + 1);
        printf("%.1f ", __bfloat162float(__float2bfloat16(gelu_fwd(g + b))));
    }
    printf("\n");
    printf("@@RESULT ms=%.3f tflops=%.2f checksum=%f valid=%d c0=%.1f\n",
           _ms, 2.0 * M_TOTAL * N_DIM * K_DIM / _ms / 1e9, cksum, valid,
           __bfloat162float(h_C[0]));

#ifdef TIMING
    print_timing(d_timing, d_spread, spread_bytes, _ms);
#endif

    free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_bias); cudaFree(d_C);
    return 0;
}
