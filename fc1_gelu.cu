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
BF16x2 pack + SMEM store for 8 values — shared by all GELU variants.
CVT and STS stay in asm (hardware-specific, no C++ equivalent).
*/
static __device__ __forceinline__ void cvt_sts_v4(
    float g0, float g1, float g2, float g3,
    float g4, float g5, float g6, float g7,
    uint32_t saddr
) {
    uint32_t o0, o1, o2, o3;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o0) : "f"(g0), "f"(g1));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o1) : "f"(g2), "f"(g3));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o2) : "f"(g4), "f"(g5));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o3) : "f"(g6), "f"(g7));
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(saddr), "r"(o0), "r"(o1), "r"(o2), "r"(o3) : "memory");
}

/* ── GELU variants (selected by GELU_VARIANT compile-time flag) ── */

#if GELU_VARIANT == 0
/*
Variant 0 (default): asm tanh.approx, scalar.
8 ops/element: 4 pre-tanh (FADD, FMUL, FFMA, FMUL), 1 MUFU, 3 post-tanh (FMUL, FADD, FMUL).
*/
static __device__ __forceinline__ float gelu_approx(float acc, float bias) {
    float x = acc + bias;
    float inner = x * (0.7978845608f + 0.035677408136f * x * x);
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(inner));
    return 0.5f * (x + x * t);
}

#elif GELU_VARIANT == 1
/*
Variant 1: pure tanhf(), same algebra as variant 0.
No asm constraint — compiler has full scheduling freedom.
nvcc maps tanhf() to MUFU.TANH on SM100a.
*/
static __device__ __forceinline__ float gelu_approx(float acc, float bias) {
    float x = acc + bias;
    float inner = x * (0.7978845608f + 0.035677408136f * x * x);
    float t = tanhf(inner);
    return 0.5f * (x + x * t);
}

#elif GELU_VARIANT == 2
/*
Variant 2: tanhf() + half_x reorder.
Shifts work before tanh: 5 pre-tanh ops, 2 post-tanh ops.
Critical path 32 cycles (saves serial FMUL(0.5*sum) by pre-computing half_x).
*/
static __device__ __forceinline__ float gelu_approx(float acc, float bias) {
    float x = acc + bias;
    float half_x = 0.5f * x;
    float inner = x * (0.7978845608f + 0.035677408136f * x * x);
    float t = tanhf(inner);
    return half_x + half_x * t;
}

#elif GELU_VARIANT == 3
/*
Variant 3: __fmaf_rn intrinsics.
7 ops/element — fuses FMUL+FADD post-tanh into one FFMA.
8 fewer instructions per 8-element group. Critical path: 32 cycles.
*/
static __device__ __forceinline__ float gelu_approx(float acc, float bias) {
    float x = acc + bias;
    float x_sq = x * x;
    float inner = x * __fmaf_rn(0.035677408136f, x_sq, 0.7978845608f);
    float t = tanhf(inner);
    return 0.5f * __fmaf_rn(x, t, x);
}
#endif

#if GELU_VARIANT <= 3
/* Scalar variants: call gelu_approx per element */
#define GELU_CVT_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, b0,b1,b2,b3,b4,b5,b6,b7, SADDR) \
    cvt_sts_v4( \
        gelu_approx(f0, b0), gelu_approx(f1, b1), \
        gelu_approx(f2, b2), gelu_approx(f3, b3), \
        gelu_approx(f4, b4), gelu_approx(f5, b5), \
        gelu_approx(f6, b6), gelu_approx(f7, b7), \
        (SADDR) \
    )

#elif GELU_VARIANT == 4
/*
Variant 4: batched-8, asm tanh.
Explicit 3-phase: all pre-tanh, all 8 MUFU back-to-back, all post-tanh.
Forces maximum MUFU throughput utilization.
~40 live FP32 regs for GELU alone (tight at FC1 baseline ~242).
*/
#define GELU_CVT_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, b0,b1,b2,b3,b4,b5,b6,b7, SADDR) \
    do { \
        float x0 = (f0)+(b0), x1 = (f1)+(b1), x2 = (f2)+(b2), x3 = (f3)+(b3); \
        float x4 = (f4)+(b4), x5 = (f5)+(b5), x6 = (f6)+(b6), x7 = (f7)+(b7); \
        float i0 = x0*(0.7978845608f + 0.035677408136f*x0*x0); \
        float i1 = x1*(0.7978845608f + 0.035677408136f*x1*x1); \
        float i2 = x2*(0.7978845608f + 0.035677408136f*x2*x2); \
        float i3 = x3*(0.7978845608f + 0.035677408136f*x3*x3); \
        float i4 = x4*(0.7978845608f + 0.035677408136f*x4*x4); \
        float i5 = x5*(0.7978845608f + 0.035677408136f*x5*x5); \
        float i6 = x6*(0.7978845608f + 0.035677408136f*x6*x6); \
        float i7 = x7*(0.7978845608f + 0.035677408136f*x7*x7); \
        float t0, t1, t2, t3, t4, t5, t6, t7; \
        asm("tanh.approx.f32 %0, %1;" : "=f"(t0) : "f"(i0)); \
        asm("tanh.approx.f32 %0, %1;" : "=f"(t1) : "f"(i1)); \
        asm("tanh.approx.f32 %0, %1;" : "=f"(t2) : "f"(i2)); \
        asm("tanh.approx.f32 %0, %1;" : "=f"(t3) : "f"(i3)); \
        asm("tanh.approx.f32 %0, %1;" : "=f"(t4) : "f"(i4)); \
        asm("tanh.approx.f32 %0, %1;" : "=f"(t5) : "f"(i5)); \
        asm("tanh.approx.f32 %0, %1;" : "=f"(t6) : "f"(i6)); \
        asm("tanh.approx.f32 %0, %1;" : "=f"(t7) : "f"(i7)); \
        cvt_sts_v4( \
            0.5f*(x0+x0*t0), 0.5f*(x1+x1*t1), \
            0.5f*(x2+x2*t2), 0.5f*(x3+x3*t3), \
            0.5f*(x4+x4*t4), 0.5f*(x5+x5*t5), \
            0.5f*(x6+x6*t6), 0.5f*(x7+x7*t7), \
            (SADDR)); \
    } while(0)

#elif GELU_VARIANT == 5
/*
Variant 5: batched-4+4, asm tanh.
Split into two groups of 4 — lower peak register pressure (~24 live).
Group 2 pre-tanh can overlap with group 1 post-tanh.
*/
#define GELU_CVT_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, b0,b1,b2,b3,b4,b5,b6,b7, SADDR) \
    do { \
        float x0 = (f0)+(b0), x1 = (f1)+(b1), x2 = (f2)+(b2), x3 = (f3)+(b3); \
        float i0 = x0*(0.7978845608f + 0.035677408136f*x0*x0); \
        float i1 = x1*(0.7978845608f + 0.035677408136f*x1*x1); \
        float i2 = x2*(0.7978845608f + 0.035677408136f*x2*x2); \
        float i3 = x3*(0.7978845608f + 0.035677408136f*x3*x3); \
        float t0, t1, t2, t3; \
        asm("tanh.approx.f32 %0, %1;" : "=f"(t0) : "f"(i0)); \
        asm("tanh.approx.f32 %0, %1;" : "=f"(t1) : "f"(i1)); \
        asm("tanh.approx.f32 %0, %1;" : "=f"(t2) : "f"(i2)); \
        asm("tanh.approx.f32 %0, %1;" : "=f"(t3) : "f"(i3)); \
        float g0 = 0.5f*(x0+x0*t0), g1 = 0.5f*(x1+x1*t1); \
        float g2 = 0.5f*(x2+x2*t2), g3 = 0.5f*(x3+x3*t3); \
        float x4 = (f4)+(b4), x5 = (f5)+(b5), x6 = (f6)+(b6), x7 = (f7)+(b7); \
        float i4 = x4*(0.7978845608f + 0.035677408136f*x4*x4); \
        float i5 = x5*(0.7978845608f + 0.035677408136f*x5*x5); \
        float i6 = x6*(0.7978845608f + 0.035677408136f*x6*x6); \
        float i7 = x7*(0.7978845608f + 0.035677408136f*x7*x7); \
        float t4, t5, t6, t7; \
        asm("tanh.approx.f32 %0, %1;" : "=f"(t4) : "f"(i4)); \
        asm("tanh.approx.f32 %0, %1;" : "=f"(t5) : "f"(i5)); \
        asm("tanh.approx.f32 %0, %1;" : "=f"(t6) : "f"(i6)); \
        asm("tanh.approx.f32 %0, %1;" : "=f"(t7) : "f"(i7)); \
        cvt_sts_v4( \
            g0, g1, g2, g3, \
            0.5f*(x4+x4*t4), 0.5f*(x5+x5*t5), \
            0.5f*(x6+x6*t6), 0.5f*(x7+x7*t7), \
            (SADDR)); \
    } while(0)

#elif GELU_VARIANT == 6
/*
Variant 6: batched-8, tanhf().
Same explicit phasing as variant 4 but tanhf() instead of asm.
Maximum compiler freedom + explicit batching.
*/
#define GELU_CVT_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, b0,b1,b2,b3,b4,b5,b6,b7, SADDR) \
    do { \
        float x0 = (f0)+(b0), x1 = (f1)+(b1), x2 = (f2)+(b2), x3 = (f3)+(b3); \
        float x4 = (f4)+(b4), x5 = (f5)+(b5), x6 = (f6)+(b6), x7 = (f7)+(b7); \
        float i0 = x0*(0.7978845608f + 0.035677408136f*x0*x0); \
        float i1 = x1*(0.7978845608f + 0.035677408136f*x1*x1); \
        float i2 = x2*(0.7978845608f + 0.035677408136f*x2*x2); \
        float i3 = x3*(0.7978845608f + 0.035677408136f*x3*x3); \
        float i4 = x4*(0.7978845608f + 0.035677408136f*x4*x4); \
        float i5 = x5*(0.7978845608f + 0.035677408136f*x5*x5); \
        float i6 = x6*(0.7978845608f + 0.035677408136f*x6*x6); \
        float i7 = x7*(0.7978845608f + 0.035677408136f*x7*x7); \
        float t0 = tanhf(i0), t1 = tanhf(i1), t2 = tanhf(i2), t3 = tanhf(i3); \
        float t4 = tanhf(i4), t5 = tanhf(i5), t6 = tanhf(i6), t7 = tanhf(i7); \
        cvt_sts_v4( \
            0.5f*(x0+x0*t0), 0.5f*(x1+x1*t1), \
            0.5f*(x2+x2*t2), 0.5f*(x3+x3*t3), \
            0.5f*(x4+x4*t4), 0.5f*(x5+x5*t5), \
            0.5f*(x6+x6*t6), 0.5f*(x7+x7*t7), \
            (SADDR)); \
    } while(0)

#else
#error "Unknown GELU_VARIANT"
#endif

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
