/*
FC2 kernel — MLP downprojection + residual add
Target: B200  Batch: 4736  GEMM: [928256,3072]×[3072,768]^T + bias + residual
Pipeline: 4-stage (parameterized)  K-iters: 24  MMA/iter: 4  idesc: 0x10400010
Warps: 2+NUM_EPI_WARPS  cta_group::2  __cluster_dims__(2,1,1)
Warp-specialized: Load(W0) | MMA(W1,cta_group::2,CTA0 only) | Epilogue(W2+,x32 TMEM ld,interleaved TMA stores)  BF16 output
tcgen05.mma.cta_group::2.kind::f8f6f4  (E4M3 × E4M3 → FP32)
Each CTA loads own A (128 rows) + half B (128 cols). MMA produces 256×256 output.
Epilogue: FP32 acc + bias + residual(BF16) → BF16 CVT → SMEM staging → TMA store
*/

#define N_DIM          768
#define K_DIM          3072
#define BIAS_BF16      1
#include "kernel_common.cuh"

/*
Fused bias+residual+CVT+STS macro — paths selected by FP32_EPILOGUE and PRE_COMBINE.
f0-f7: FP32 accumulators, b0-b3: BF16x2 bias pairs, r0-r3: BF16x2 residual pairs
PRE_COMBINE=1: bias+residual pre-added before TMEM_WAIT → c0-c3 combined BF16x2.
*/
#if FP32_EPILOGUE == 0
#if PRE_COMBINE
/*
Pre-combined BF16 path: bias+residual already added → 4 CVT + 4 HADD2 + STS = 9 ops.
8 BF16 per STS (vs 12 in original). Section P: 8=+55% overhead, fits better in STS shadow.
*/
#define COMBINED_CVT_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, c0,c1,c2,c3, SADDR) \
    asm volatile( \
        "{\n\t" \
        ".reg .b32 o0, o1, o2, o3;\n\t" \
        "cvt.rn.bf16x2.f32 o0, %1, %0;\n\t" \
        "cvt.rn.bf16x2.f32 o1, %3, %2;\n\t" \
        "cvt.rn.bf16x2.f32 o2, %5, %4;\n\t" \
        "cvt.rn.bf16x2.f32 o3, %7, %6;\n\t" \
        "add.rn.bf16x2 o0, o0, %8;\n\t" \
        "add.rn.bf16x2 o1, o1, %9;\n\t" \
        "add.rn.bf16x2 o2, o2, %10;\n\t" \
        "add.rn.bf16x2 o3, o3, %11;\n\t" \
        "st.shared.v4.b32 [%12], {o0,o1,o2,o3};\n\t" \
        "}" \
        :: "f"(f0),"f"(f1),"f"(f2),"f"(f3), \
           "f"(f4),"f"(f5),"f"(f6),"f"(f7), \
           "r"(c0),"r"(c1),"r"(c2),"r"(c3), \
           "r"(SADDR) \
        : "memory")

/* Keep old macro for non-PRE_COMBINE code in kernel_body.cuh (unused when PRE_COMBINE=1) */
#define BIAS_RES_CVT_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, b0,b1,b2,b3, r0,r1,r2,r3, SADDR) \
    COMBINED_CVT_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, b0,b1,b2,b3, SADDR)
#else
/*
BF16 arithmetic path (original): CVT FP32→BF16x2 early, HADD2 bias, HADD2 residual, STS.
12 BF16 + 1 STS = 13 instructions per macro. Section P: 12 BF16 ≈ +100% STS overhead.
*/
#define BIAS_RES_CVT_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, b0,b1,b2,b3, r0,r1,r2,r3, SADDR) \
    asm volatile( \
        "{\n\t" \
        ".reg .b32 o0, o1, o2, o3;\n\t" \
        "cvt.rn.bf16x2.f32 o0, %1, %0;\n\t" \
        "cvt.rn.bf16x2.f32 o1, %3, %2;\n\t" \
        "cvt.rn.bf16x2.f32 o2, %5, %4;\n\t" \
        "cvt.rn.bf16x2.f32 o3, %7, %6;\n\t" \
        "add.rn.bf16x2 o0, o0, %8;\n\t" \
        "add.rn.bf16x2 o1, o1, %9;\n\t" \
        "add.rn.bf16x2 o2, o2, %10;\n\t" \
        "add.rn.bf16x2 o3, o3, %11;\n\t" \
        "add.rn.bf16x2 o0, o0, %12;\n\t" \
        "add.rn.bf16x2 o1, o1, %13;\n\t" \
        "add.rn.bf16x2 o2, o2, %14;\n\t" \
        "add.rn.bf16x2 o3, o3, %15;\n\t" \
        "st.shared.v4.b32 [%16], {o0,o1,o2,o3};\n\t" \
        "}" \
        :: "f"(f0),"f"(f1),"f"(f2),"f"(f3), \
           "f"(f4),"f"(f5),"f"(f6),"f"(f7), \
           "r"(b0),"r"(b1),"r"(b2),"r"(b3), \
           "r"(r0),"r"(r1),"r"(r2),"r"(r3), \
           "r"(SADDR) \
        : "memory")
#endif

#elif FP32_EPILOGUE == 1
/*
FP32 residual path: unpack residual BF16→FP32, FADD residual in FP32, late CVT, HADD2 bias in BF16.
29 instructions between STS calls — hides 32-cycle STS throughput.
*/
#define BIAS_RES_CVT_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, b0,b1,b2,b3, r0,r1,r2,r3, SADDR) \
    asm volatile( \
        "{\n\t" \
        ".reg .b16 rl0,rh0,rl1,rh1,rl2,rh2,rl3,rh3;\n\t" \
        ".reg .f32 fr0,fr1,fr2,fr3,fr4,fr5,fr6,fr7;\n\t" \
        ".reg .b32 o0,o1,o2,o3;\n\t" \
        "mov.b32 {rl0,rh0}, %12;\n\t" \
        "mov.b32 {rl1,rh1}, %13;\n\t" \
        "mov.b32 {rl2,rh2}, %14;\n\t" \
        "mov.b32 {rl3,rh3}, %15;\n\t" \
        "cvt.rn.f32.bf16 fr0, rl0;\n\t" \
        "cvt.rn.f32.bf16 fr1, rh0;\n\t" \
        "cvt.rn.f32.bf16 fr2, rl1;\n\t" \
        "cvt.rn.f32.bf16 fr3, rh1;\n\t" \
        "cvt.rn.f32.bf16 fr4, rl2;\n\t" \
        "cvt.rn.f32.bf16 fr5, rh2;\n\t" \
        "cvt.rn.f32.bf16 fr6, rl3;\n\t" \
        "cvt.rn.f32.bf16 fr7, rh3;\n\t" \
        "add.f32 %0, %0, fr0;\n\t" \
        "add.f32 %1, %1, fr1;\n\t" \
        "add.f32 %2, %2, fr2;\n\t" \
        "add.f32 %3, %3, fr3;\n\t" \
        "add.f32 %4, %4, fr4;\n\t" \
        "add.f32 %5, %5, fr5;\n\t" \
        "add.f32 %6, %6, fr6;\n\t" \
        "add.f32 %7, %7, fr7;\n\t" \
        "cvt.rn.bf16x2.f32 o0, %1, %0;\n\t" \
        "cvt.rn.bf16x2.f32 o1, %3, %2;\n\t" \
        "cvt.rn.bf16x2.f32 o2, %5, %4;\n\t" \
        "cvt.rn.bf16x2.f32 o3, %7, %6;\n\t" \
        "add.rn.bf16x2 o0, o0, %8;\n\t" \
        "add.rn.bf16x2 o1, o1, %9;\n\t" \
        "add.rn.bf16x2 o2, o2, %10;\n\t" \
        "add.rn.bf16x2 o3, o3, %11;\n\t" \
        "st.shared.v4.b32 [%16], {o0,o1,o2,o3};\n\t" \
        "}" \
        :: "f"(f0),"f"(f1),"f"(f2),"f"(f3), \
           "f"(f4),"f"(f5),"f"(f6),"f"(f7), \
           "r"(b0),"r"(b1),"r"(b2),"r"(b3), \
           "r"(r0),"r"(r1),"r"(r2),"r"(r3), \
           "r"(SADDR) \
        : "memory")

#elif FP32_EPILOGUE == 2
/*
Full FP32 path: unpack both bias+residual BF16→FP32, all arithmetic in FP32, late CVT.
45 instructions between STS calls — maximum compute/STS ratio.
*/
#define BIAS_RES_CVT_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, b0,b1,b2,b3, r0,r1,r2,r3, SADDR) \
    asm volatile( \
        "{\n\t" \
        ".reg .b16 bl0,bh0,bl1,bh1,bl2,bh2,bl3,bh3;\n\t" \
        ".reg .f32 fb0,fb1,fb2,fb3,fb4,fb5,fb6,fb7;\n\t" \
        ".reg .b16 rl0,rh0,rl1,rh1,rl2,rh2,rl3,rh3;\n\t" \
        ".reg .f32 fr0,fr1,fr2,fr3,fr4,fr5,fr6,fr7;\n\t" \
        ".reg .b32 o0,o1,o2,o3;\n\t" \
        "mov.b32 {bl0,bh0}, %8;\n\t" \
        "mov.b32 {bl1,bh1}, %9;\n\t" \
        "mov.b32 {bl2,bh2}, %10;\n\t" \
        "mov.b32 {bl3,bh3}, %11;\n\t" \
        "cvt.rn.f32.bf16 fb0, bl0;\n\t" \
        "cvt.rn.f32.bf16 fb1, bh0;\n\t" \
        "cvt.rn.f32.bf16 fb2, bl1;\n\t" \
        "cvt.rn.f32.bf16 fb3, bh1;\n\t" \
        "cvt.rn.f32.bf16 fb4, bl2;\n\t" \
        "cvt.rn.f32.bf16 fb5, bh2;\n\t" \
        "cvt.rn.f32.bf16 fb6, bl3;\n\t" \
        "cvt.rn.f32.bf16 fb7, bh3;\n\t" \
        "mov.b32 {rl0,rh0}, %12;\n\t" \
        "mov.b32 {rl1,rh1}, %13;\n\t" \
        "mov.b32 {rl2,rh2}, %14;\n\t" \
        "mov.b32 {rl3,rh3}, %15;\n\t" \
        "cvt.rn.f32.bf16 fr0, rl0;\n\t" \
        "cvt.rn.f32.bf16 fr1, rh0;\n\t" \
        "cvt.rn.f32.bf16 fr2, rl1;\n\t" \
        "cvt.rn.f32.bf16 fr3, rh1;\n\t" \
        "cvt.rn.f32.bf16 fr4, rl2;\n\t" \
        "cvt.rn.f32.bf16 fr5, rh2;\n\t" \
        "cvt.rn.f32.bf16 fr6, rl3;\n\t" \
        "cvt.rn.f32.bf16 fr7, rh3;\n\t" \
        "add.f32 %0, %0, fb0;\n\t" \
        "add.f32 %1, %1, fb1;\n\t" \
        "add.f32 %2, %2, fb2;\n\t" \
        "add.f32 %3, %3, fb3;\n\t" \
        "add.f32 %4, %4, fb4;\n\t" \
        "add.f32 %5, %5, fb5;\n\t" \
        "add.f32 %6, %6, fb6;\n\t" \
        "add.f32 %7, %7, fb7;\n\t" \
        "add.f32 %0, %0, fr0;\n\t" \
        "add.f32 %1, %1, fr1;\n\t" \
        "add.f32 %2, %2, fr2;\n\t" \
        "add.f32 %3, %3, fr3;\n\t" \
        "add.f32 %4, %4, fr4;\n\t" \
        "add.f32 %5, %5, fr5;\n\t" \
        "add.f32 %6, %6, fr6;\n\t" \
        "add.f32 %7, %7, fr7;\n\t" \
        "cvt.rn.bf16x2.f32 o0, %1, %0;\n\t" \
        "cvt.rn.bf16x2.f32 o1, %3, %2;\n\t" \
        "cvt.rn.bf16x2.f32 o2, %5, %4;\n\t" \
        "cvt.rn.bf16x2.f32 o3, %7, %6;\n\t" \
        "st.shared.v4.b32 [%16], {o0,o1,o2,o3};\n\t" \
        "}" \
        :: "f"(f0),"f"(f1),"f"(f2),"f"(f3), \
           "f"(f4),"f"(f5),"f"(f6),"f"(f7), \
           "r"(b0),"r"(b1),"r"(b2),"r"(b3), \
           "r"(r0),"r"(r1),"r"(r2),"r"(r3), \
           "r"(SADDR) \
        : "memory")

#else
#error "FP32_EPILOGUE must be 0, 1, or 2"
#endif

#include "kernel_body.cuh"

// Device init kernel for residual matrix
__global__ void init_residual(__nv_bfloat16* __restrict__ res, int n_dim, long long total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int col = (int)(idx % n_dim);
        int row = (int)(idx / n_dim);
        res[idx] = __float2bfloat16((float)(row % 128) * 0.25f + (float)col * 0.125f);
    }
}

// Host

int main() {
    setbuf(stdout, NULL);
    printf("FC2 GEMM — tcgen05 cta_group::2 (%d warps [%d epi], cluster of 2)\n",
           2 + NUM_EPI_WARPS, NUM_EPI_WARPS);
    printf("  GEMM: [%d,%d] x [%d,%d]^T  %d-stage pipeline  bias+residual  SMEM-staged stores  idesc: 0x%08X\n",
           M_TOTAL, K_DIM, N_DIM, K_DIM, N_STAGES, IDESC);

    uint8_t *d_A, *d_B;
    __nv_bfloat16 *d_bias;
    __nv_bfloat16 *d_residual, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A,        (size_t)M_TOTAL * K_DIM));
    CUDA_CHECK(cudaMalloc(&d_B,        (size_t)N_DIM   * K_DIM));
    CUDA_CHECK(cudaMalloc(&d_bias,     (size_t)N_DIM   * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_residual, (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_C,        (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16)));

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

    // Non-uniform bias: bias[c] = bf16(c + 1)
    {
        __nv_bfloat16* h_bias = (__nv_bfloat16*)malloc((size_t)N_DIM * sizeof(__nv_bfloat16));
        for (int c = 0; c < N_DIM; c++)
            h_bias[c] = __float2bfloat16((float)(c + 1));
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias, (size_t)N_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        free(h_bias);
    }

    // Non-uniform residual: residual[row,col] = bf16((row%128)*0.25 + col*0.125)
    {
        long long total = (long long)M_TOTAL * N_DIM;
        int tpb = 256;
        int bpg = (int)((total + tpb - 1) / tpb);
        init_residual<<<bpg, tpb>>>(d_residual, N_DIM, total);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    printf("  Alloc + init done\n");

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

#if TMA_RESIDUAL
    CUtensorMap h_tma_res;
    {
        uint64_t dims[2]    = {(uint64_t)N_DIM, (uint64_t)M_TOTAL};
        uint64_t strides[1] = {(uint64_t)N_DIM * sizeof(__nv_bfloat16)};
        uint32_t box[2]     = {64, 32};
        uint32_t estrides[2]= {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(&h_tma_res,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)d_residual,
            dims, strides, box, estrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }
#endif

    CUDA_CHECK(cudaFuncSetAttribute(persistent_gemm<EpilogueOp::BIAS_RESIDUAL>,
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
    persistent_gemm<EpilogueOp::BIAS_RESIDUAL><<<SM_COUNT, THREADS, SMEM_BYTES>>>(h_tma_a, h_tma_b, h_tma_c, d_bias, d_C, d_residual
#if TMA_RESIDUAL
        , h_tma_res
#endif
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
    persistent_gemm<EpilogueOp::BIAS_RESIDUAL><<<SM_COUNT, THREADS, SMEM_BYTES>>>(h_tma_a, h_tma_b, h_tma_c, d_bias, d_C, d_residual
#if TMA_RESIDUAL
        , h_tma_res
#endif
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
    printf("FC2 kernel: %.3f ms  %.2f TFLOPS\n",
           _ms, 2.0 * M_TOTAL * N_DIM * K_DIM / _ms / 1e9);
    cudaEventDestroy(_t0);
    cudaEventDestroy(_t1);

    // Checksum run
    persistent_gemm<EpilogueOp::BIAS_RESIDUAL><<<SM_COUNT, THREADS, SMEM_BYTES>>>(h_tma_a, h_tma_b, h_tma_c, d_bias, d_C, d_residual
#if TMA_RESIDUAL
        , h_tma_res
#endif
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

    /*
    CPU reference spot checks: 32 positions spread across the matrix
    Kernel computes: bf16(gemm_f32 + bias_f32 + bf16_to_f32(residual_bf16))
    */
    int errors = 0;
    {
        for (int spot = 0; spot < 32; spot++) {
            long long row = (long long)spot * M_TOTAL / 32;
            int col = (spot * 47) % N_DIM;

            float b_val = (col & 1) ? 1.0f : 1.5f;
            float gemm = (float)K_DIM * 1.5f * b_val;
            float res_bf16_f = __bfloat162float(__float2bfloat16(
                (float)((int)row % 128) * 0.25f + (float)col * 0.125f));
            float bias_bf16_f = __bfloat162float(__float2bfloat16((float)(col + 1)));
#if FP32_EPILOGUE == 0
#if PRE_COMBINE
            /* Pre-combined BF16 path: bf16(gemm) + bf16(bias + residual) */
            float combined_f = __bfloat162float(__float2bfloat16(bias_bf16_f + res_bf16_f));
            float acc_rounded = __bfloat162float(__float2bfloat16(gemm));
            __nv_bfloat16 expected = __float2bfloat16(acc_rounded + combined_f);
#else
            /* BF16 path: bf16(bf16(gemm) + bf16(bias)) + residual */
            float acc_rounded = __bfloat162float(__float2bfloat16(gemm));
            float after_bias = __bfloat162float(__float2bfloat16(acc_rounded + bias_bf16_f));
            __nv_bfloat16 expected = __float2bfloat16(after_bias + res_bf16_f);
#endif
#elif FP32_EPILOGUE == 1
            /* FP32 residual path: bf16(f32(acc) + f32(residual)) then bf16 bias add */
            float after_res = __bfloat162float(__float2bfloat16(gemm + res_bf16_f));
            __nv_bfloat16 expected = __float2bfloat16(after_res + bias_bf16_f);
#elif FP32_EPILOGUE == 2
            /* Full FP32 path: bf16(f32(gemm) + f32(bias) + f32(residual)) */
            __nv_bfloat16 expected = __float2bfloat16(gemm + bias_bf16_f + res_bf16_f);
#endif
            __nv_bfloat16 actual = h_C[row * N_DIM + col];

            float ef = __bfloat162float(expected);
            float af = __bfloat162float(actual);
            if (ef != af) {
                if (errors < 5)
                    printf("  MISMATCH at (%lld,%d): expected %.1f got %.1f (gemm=%.1f bias=%.1f res=%.4f)\n",
                           row, col, ef, af, gemm, bias_bf16_f, res_bf16_f);
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
        float gemm_f = (float)K_DIM * 1.5f * b_val;
        float b_r = __bfloat162float(__float2bfloat16((float)(c + 1)));
        float r = __bfloat162float(__float2bfloat16((float)c * 0.125f));
#if FP32_EPILOGUE == 0
#if PRE_COMBINE
        float cmb = __bfloat162float(__float2bfloat16(b_r + r));
        float g_r = __bfloat162float(__float2bfloat16(gemm_f));
        printf("%.1f ", __bfloat162float(__float2bfloat16(g_r + cmb)));
#else
        float g_r = __bfloat162float(__float2bfloat16(gemm_f));
        float ab = __bfloat162float(__float2bfloat16(g_r + b_r));
        printf("%.1f ", __bfloat162float(__float2bfloat16(ab + r)));
#endif
#elif FP32_EPILOGUE == 1
        float ar = __bfloat162float(__float2bfloat16(gemm_f + r));
        printf("%.1f ", __bfloat162float(__float2bfloat16(ar + b_r)));
#elif FP32_EPILOGUE == 2
        printf("%.1f ", __bfloat162float(__float2bfloat16(gemm_f + b_r + r)));
#endif
    }
    printf("\n");
    printf("@@RESULT ms=%.3f tflops=%.2f checksum=%f valid=%d c0=%.1f\n",
           _ms, 2.0 * M_TOTAL * N_DIM * K_DIM / _ms / 1e9, cksum, valid,
           __bfloat162float(h_C[0]));

#ifdef TIMING
    print_timing(d_timing, d_spread, spread_bytes, _ms);
#endif

    free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_bias); cudaFree(d_residual); cudaFree(d_C);
    return 0;
}
