/*
FC2 CUTLASS kernel — CUTLASS GemmUniversal with SM100 TMA warp-specialized epilogue.
Shape: [928256,3072]×[3072,768]^T + bias + residual
Uses CUTLASS's proven epilogue (1.211ms reference) for direct A/B comparison.

Build:  make fc2-cutlass              (fused: bias + residual)
        make fc2-cutlass-strip        (GEMM-only: no epilogue fusion)
*/

#include <cstdio>
#include <cstdlib>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include <cuda_bf16.h>
#include "cutlass/epilogue/thread/activation.h"

using namespace cute;

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

/* Problem dimensions — identical to fc2_w3.cu */
#define SM_COUNT  148
#define M_TOTAL   928256   /* 4736 images × 196 patches */
#define N_DIM     768
#define K_DIM     3072

/* FP8 E4M3 inputs, BF16 C/D, FP32 accumulator */
using ElementA   = cutlass::float_e4m3_t;
using LayoutA    = cutlass::layout::RowMajor;
using ElementB   = cutlass::float_e4m3_t;
using LayoutB    = cutlass::layout::ColumnMajor;
using ElementC   = cutlass::bfloat16_t;
using ElementD   = cutlass::bfloat16_t;
using LayoutC    = cutlass::layout::RowMajor;
using LayoutD    = cutlass::layout::RowMajor;
using ElementAcc = float;

constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;  /* 16 */
constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;  /* 8 */
constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;

/* Tile: 256×256×128, Cluster: 2×1 — matches our hand-tuned kernel */
using TileShape    = Shape<_256, _256, _128>;
using ClusterShape = Shape<_2, _1, _1>;

/* 2SM schedule (cta_group::2) — proven winning config */
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;

#ifdef STRIP_EPILOGUE
/*
GEMM-only: D = alpha * A×B + beta*C, alpha=1, beta=0.
No bias, no residual — just the matmul.
Uses plain BF16 output, no fusion.
*/
using FusionOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementAcc>;
#else
/*
Fused: D = acc + bias + residual
LinCombPerColBiasEltAct with Identity activation, beta=1.
bias_ptr is float*, C is BF16 residual.
*/
using FusionOp = cutlass::epilogue::fusion::LinCombPerColBiasEltAct<
    cutlass::epilogue::thread::Identity,
    ElementD, float, float, ElementC, float>;
#endif

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, float,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    EpilogueSchedule,
    FusionOp
>::CollectiveOp;

constexpr int EpilogueSmemBytes = static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage));

using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<EpilogueSmemBytes>;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAcc,
    TileShape, ClusterShape,
    StageCountType,
    MainloopSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

constexpr size_t SmemBytes = sizeof(typename GemmKernel::SharedStorage);

/* Residual init — matches fc2_w3.cu pattern */
__global__ void init_residual(__nv_bfloat16* d, int N, long long total) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        long long row = i / N;
        int col = (int)(i % N);
        d[i] = __float2bfloat16((float)((int)(row % 128)) * 0.25f + (float)col * 0.125f);
    }
}

int main() {
    setbuf(stdout, NULL);

#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    printf("ERROR: CUTLASS_ARCH_MMA_SM100_SUPPORTED not defined.\n");
    return 1;
#else
    printf("FC2 CUTLASS — GemmUniversal with SM100 TMA warp-specialized epilogue\n");
#ifdef STRIP_EPILOGUE
    printf("  MODE: GEMM-only (STRIP_EPILOGUE — no bias, no residual)\n");
#else
    printf("  MODE: Fused (bias + residual)\n");
#endif
    printf("  [%d, %d] x [%d, %d]^T  FP8E4M3 -> BF16\n", M_TOTAL, K_DIM, K_DIM, N_DIM);
    printf("  Tile: 256x256x128, Cluster: 2x1, Schedule: 2SM\n");
    printf("  SMEM: %zu bytes, Epilogue SMEM: %d bytes\n", SmemBytes, EpilogueSmemBytes);

    cudaDeviceProp props;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    printf("  Device: %s (SM %d.%d, SMs=%d)\n\n", props.name, props.major, props.minor,
           props.multiProcessorCount);

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    double flops = 2.0 * M_TOTAL * (double)N_DIM * K_DIM;

    /* Allocate */
    size_t sz_a   = (size_t)M_TOTAL * K_DIM;
    size_t sz_b   = (size_t)N_DIM * K_DIM;
    size_t sz_cd  = (size_t)M_TOTAL * N_DIM * sizeof(ElementD);
    size_t sz_bias = (size_t)N_DIM * sizeof(float);

    void *d_A = nullptr, *d_B = nullptr;
    void *d_C = nullptr, *d_D = nullptr;
    float *d_bias = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, sz_a));
    CUDA_CHECK(cudaMalloc(&d_B, sz_b));
    CUDA_CHECK(cudaMalloc(&d_C, sz_cd));
    CUDA_CHECK(cudaMalloc(&d_D, sz_cd));
    CUDA_CHECK(cudaMalloc(&d_bias, sz_bias));

    /* Init A: all 0x3C (FP8 1.5) */
    CUDA_CHECK(cudaMemset(d_A, 0x3C, sz_a));

    /* Init B: even cols = 0x3C (FP8 1.5), odd cols = 0x38 (FP8 1.0) — matches fc2_w3 */
    {
        uint8_t* h_B = (uint8_t*)malloc(sz_b);
        for (int n = 0; n < N_DIM; n++)
            memset(h_B + (size_t)n * K_DIM, (n & 1) ? 0x38 : 0x3C, K_DIM);
        CUDA_CHECK(cudaMemcpy(d_B, h_B, sz_b, cudaMemcpyHostToDevice));
        free(h_B);
    }

    /* Init bias: float(c+1) — matches fc2_w3 (bf16(c+1) promoted to float) */
    {
        float* h_bias = (float*)malloc(sz_bias);
        for (int c = 0; c < N_DIM; c++)
            h_bias[c] = (float)(c + 1);
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias, sz_bias, cudaMemcpyHostToDevice));
        free(h_bias);
    }

#ifndef STRIP_EPILOGUE
    /* Init residual (C matrix): bf16(row%128*0.25 + col*0.125) — matches fc2_w3 */
    {
        long long total = (long long)M_TOTAL * N_DIM;
        int tpb = 256;
        int bpg = (int)((total + tpb - 1) / tpb);
        init_residual<<<bpg, tpb>>>(reinterpret_cast<__nv_bfloat16*>(d_C), N_DIM, total);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
#else
    CUDA_CHECK(cudaMemset(d_C, 0, sz_cd));
#endif
    CUDA_CHECK(cudaMemset(d_D, 0, sz_cd));
    printf("  Alloc + init done\n");

    /* Set up CUTLASS arguments */
    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideC;
    using StrideD = typename GemmKernel::StrideD;

    auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M_TOTAL, K_DIM, 1));
    auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N_DIM, K_DIM, 1));
    auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, make_shape(M_TOTAL, N_DIM, 1));
    auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M_TOTAL, N_DIM, 1));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M_TOTAL, N_DIM, K_DIM, 1},
        {
            reinterpret_cast<ElementA*>(d_A), stride_a,
            reinterpret_cast<ElementB*>(d_B), stride_b
        },
        {
            {},
            reinterpret_cast<ElementC*>(d_C), stride_c,
            reinterpret_cast<ElementD*>(d_D), stride_d
        },
        hw_info
    };

#ifdef STRIP_EPILOGUE
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta = 0.0f;
#else
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta = 1.0f;
    arguments.epilogue.thread.bias_ptr = d_bias;
#endif

    Gemm gemm;
    auto status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        printf("FATAL: can_implement failed: %d\n", (int)status);
        return 1;
    }

    size_t ws = Gemm::get_workspace_size(arguments);
    uint8_t* d_ws = nullptr;
    if (ws > 0) CUDA_CHECK(cudaMalloc(&d_ws, ws));

    status = gemm.initialize(arguments, d_ws);
    if (status != cutlass::Status::kSuccess) {
        printf("FATAL: initialize failed: %d\n", (int)status);
        return 1;
    }
    printf("  CUTLASS initialized (workspace=%zu)\n", ws);

    /* Warmup */
    printf("Launching warmup (2 iters)...\n");
    for (int i = 0; i < 2; i++) {
        status = gemm.run();
        if (status != cutlass::Status::kSuccess) {
            printf("FATAL: run failed: %d\n", (int)status);
            return 1;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Warmup done.\n");

    /* Timed: 10 iterations */
    printf("Timing: 10 iterations...\n");
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 10; i++) gemm.run();
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= 10.0f;
    printf("FC2-CUTLASS: %.3f ms  %.2f TFLOPS\n", ms, flops / ms / 1e9);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    /* Checksum run */
    CUDA_CHECK(cudaMemset(d_D, 0, sz_cd));
    gemm.run();
    CUDA_CHECK(cudaDeviceSynchronize());

    __nv_bfloat16* h_D = (__nv_bfloat16*)malloc(sz_cd);
    CUDA_CHECK(cudaMemcpy(h_D, d_D, sz_cd, cudaMemcpyDeviceToHost));

    /* Strided checksum — same as fc2_w3 */
    double cksum = 0;
    {
        long long total_elems = (long long)M_TOTAL * N_DIM;
        long long stride = total_elems / 1024;
        for (int i = 0; i < 1024; i++)
            cksum += (double)__bfloat162float(h_D[(long long)i * stride]);
    }

    /* Spot checks — CUTLASS does FP32 epilogue math: bf16(acc_fp32 + bias_fp32 + res_fp32) */
    int errors = 0;
    for (int spot = 0; spot < 32; spot++) {
        long long row = (long long)spot * M_TOTAL / 32;
        int col = (spot * 47) % N_DIM;
        float b_val = (col & 1) ? 1.0f : 1.5f;
        float gemm_val = (float)K_DIM * 1.5f * b_val;
#ifdef STRIP_EPILOGUE
        __nv_bfloat16 expected = __float2bfloat16(gemm_val);
#else
        float bias_f = (float)(col + 1);
        float res_f = __bfloat162float(__float2bfloat16(
            (float)((int)(row % 128)) * 0.25f + (float)col * 0.125f));
        /* CUTLASS FP32 path: single rounding at the end */
        __nv_bfloat16 expected = __float2bfloat16(gemm_val + bias_f + res_f);
#endif
        __nv_bfloat16 actual = h_D[row * N_DIM + col];
        float ef = __bfloat162float(expected);
        float af = __bfloat162float(actual);
        if (ef != af) {
            if (errors < 5)
                printf("  MISMATCH at (%lld,%d): expected %.1f got %.1f\n", row, col, ef, af);
            errors++;
        }
    }
    int valid = (errors == 0) ? 1 : 0;
    printf("Validation: %d/32 spot checks passed%s\n", 32 - errors, valid ? "" : " — FAILED");
    printf("Checksum (1024 strided): %f\n", cksum);
    printf("D[0,0..3] = %.1f %.1f %.1f %.1f\n",
           __bfloat162float(h_D[0]), __bfloat162float(h_D[1]),
           __bfloat162float(h_D[2]), __bfloat162float(h_D[3]));
    printf("@@RESULT ms=%.3f tflops=%.2f checksum=%f valid=%d c0=%.1f\n",
           ms, flops / ms / 1e9, cksum, valid, __bfloat162float(h_D[0]));

    free(h_D);
    if (d_ws) cudaFree(d_ws);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D); cudaFree(d_bias);
    return 0;
#endif
}
