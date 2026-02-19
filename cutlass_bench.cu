// CUTLASS 4.x Blackwell FP8 GEMM benchmark
// A[M,K] x B[N,K]^T -> C[M,N]   FP8 E4M3 inputs, FP32 accumulate, BF16 output
// Matches cuBLAS bench problem dimensions and output format.
//
// Usage: ./cutlass-bench [imgs_per_sm]

#include <cstdio>
#include <cstdlib>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// ═══════════════════════════════════════════════════════════════
// CUTLASS kernel configuration
// ═══════════════════════════════════════════════════════════════

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// A is [M,K] row-major, B is [N,K] row-major stored as ColumnMajor for B^T
using ElementA       = cutlass::float_e4m3_t;
using ElementB       = cutlass::float_e4m3_t;
using ElementAcc     = float;
using ElementOut     = cutlass::bfloat16_t;      // BF16 output, matching cuBLAS bench
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::ColumnMajor;

constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;    // 16
constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;    // 16
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementOut>::value;  // 8
constexpr int AlignD = AlignC;

// 2SM MMA tile (256 in M requires cluster M divisible by 2)
using MmaTileShape = Shape<_256, _128, _64>;
using ClusterShape = Shape<_2, _1, _1>;

// Simple epilogue: D = alpha * acc + beta * C  (alpha=1, beta=0)
using FusionOp = cutlass::epilogue::fusion::LinearCombination<ElementOut, ElementCompute>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementCompute,
    ElementOut, LayoutC, AlignC,
    ElementOut, LayoutD, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionOp
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAcc,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

// Default 4th parameter = CLC-based tile scheduler for SM100
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

// ═══════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    setbuf(stdout, NULL);

    const int SM_COUNT = 148;
    const int SEQ_LEN  = 196;
    const int N        = 768;
    const int K        = 768;

    int imgs_per_sm = 32;
    if (argc > 1) imgs_per_sm = atoi(argv[1]);
    const int M = imgs_per_sm * SM_COUNT * SEQ_LEN;

    printf("CUTLASS 4.x Blackwell FP8 GEMM Benchmark\n");
    printf("  M=%d  N=%d  K=%d  (imgs_per_sm=%d)\n", M, N, K, imgs_per_sm);
    printf("  Types: FP8 E4M3 x FP8 E4M3 -> BF16 (acc: FP32)\n");
    printf("  Tile: 256x128x64  Cluster: 2x1x1\n\n");

#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    printf("ERROR: CUTLASS_ARCH_MMA_SM100_SUPPORTED not defined.\n");
    printf("  Requires CUDA 12.8+ and sm_100a target.\n");
    return 1;
#else
    // Check device
    cudaDeviceProp props;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);
    printf("  Device: %s (SM %d.%d)\n\n", props.name, props.major, props.minor);

    // KernelHardwareInfo for CLC-based tile scheduler
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    // Strides
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, make_shape(M, N, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1));

    // Allocate device memory
    size_t sz_a  = (size_t)M * K;
    size_t sz_b  = (size_t)N * K;
    size_t sz_cd = (size_t)M * N * sizeof(ElementOut);

    ElementA*   d_A = nullptr;
    ElementB*   d_B = nullptr;
    ElementOut* d_C = nullptr;   // source (beta=0, unused)
    ElementOut* d_D = nullptr;   // output

    cudaMalloc(&d_A, sz_a);
    cudaMalloc(&d_B, sz_b);
    cudaMalloc(&d_C, sz_cd);
    cudaMalloc(&d_D, sz_cd);

    // Fill A,B with 0x3C (1.5 in E4M3), zero C/D
    cudaMemset(d_A, 0x3C, sz_a);
    cudaMemset(d_B, 0x3C, sz_b);
    cudaMemset(d_C, 0, sz_cd);
    cudaMemset(d_D, 0, sz_cd);

    // Construct GEMM arguments: alpha=1, beta=0
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {d_A, stride_a, d_B, stride_b},
        {{}, d_C, stride_c, d_D, stride_d},
        hw_info
    };
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta  = 0.0f;

    // Instantiate GEMM
    Gemm gemm;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    uint8_t* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    cutlass::Status status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        printf("ERROR: CUTLASS cannot implement this problem (status %d)\n", (int)status);
        return 1;
    }

    status = gemm.initialize(arguments, workspace);
    if (status != cutlass::Status::kSuccess) {
        printf("ERROR: CUTLASS initialize failed (status %d)\n", (int)status);
        return 1;
    }

    // Warmup
    printf("  Warmup (3 iters)...\n");
    for (int i = 0; i < 3; i++) {
        status = gemm.run();
        if (status != cutlass::Status::kSuccess) {
            printf("ERROR: CUTLASS run failed on warmup iter %d (status %d)\n", i, (int)status);
            return 1;
        }
    }
    cudaDeviceSynchronize();

    // Verify output
    {
        ElementOut h_D[4];
        cudaMemcpy(h_D, d_D, 4 * sizeof(ElementOut), cudaMemcpyDeviceToHost);
        printf("  C[0,0..3] = %.1f %.1f %.1f %.1f",
            float(h_D[0]), float(h_D[1]), float(h_D[2]), float(h_D[3]));
        float expected = (float)K * 1.5f * 1.5f;
        printf("  (expected %.1f)\n", expected);
    }

    // Timed runs
    const int ITERS = 20;
    printf("  Timing (%d iters)...\n", ITERS);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; i++) {
        gemm.run();
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float total_ms;
    cudaEventElapsedTime(&total_ms, t0, t1);
    float avg_ms = total_ms / ITERS;

    double flops = 2.0 * M * N * K;
    double tflops = flops / (avg_ms * 1e9);

    printf("\nCUTLASS FP8:  %.3f ms  %.2f TFLOPS\n", avg_ms, tflops);
    printf("  M=%d  N=%d  K=%d\n", M, N, K);

    // Cleanup
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    if (workspace) cudaFree(workspace);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    return 0;
#endif
}
