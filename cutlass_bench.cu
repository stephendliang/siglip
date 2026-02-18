// CUTLASS 3.x Blackwell FP8 GEMM reference benchmark
// A[M,K] x B[N,K]^T -> C[M,N]   FP8 E4M3 inputs, FP32 accumulate, FP32 output
// Problem: SigLIP2 patch embed — M=116032, N=768, K=768

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
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// ═══════════════════════════════════════════════════════════════
// Problem dimensions (matching gen.py B200 defaults)
// ═══════════════════════════════════════════════════════════════

constexpr int M = 592 * 196;  // 116,032
constexpr int N = 768;
constexpr int K = 768;

// ═══════════════════════════════════════════════════════════════
// CUTLASS kernel configuration
// ═══════════════════════════════════════════════════════════════

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// A is [M,K] row-major, B is [N,K] row-major → ColumnMajor for B^T
using ElementA   = cutlass::float_e4m3_t;
using ElementB   = cutlass::float_e4m3_t;
using ElementAcc = float;
using ElementOut = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16
constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 16
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementOut>::value; // 4

using MmaTileShape = Shape<_256, _128, _64>;
using ClusterShape = Shape<_2, _2, _1>;

// Simple epilogue: D = alpha * acc + beta * C  (alpha=1, beta=0 → just GEMM result)
using FusionOp = cutlass::epilogue::fusion::LinearCombination<ElementOut, float>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, float,
    ElementOut, LayoutC, AlignC,
    ElementOut, LayoutC, AlignC,
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

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

// ═══════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════

int main() {
    printf("CUTLASS 3.x Blackwell FP8 GEMM Benchmark\n");
    printf("  Problem: [%d, %d] x [%d, %d]^T -> [%d, %d]\n", M, K, N, K, M, N);
    printf("  Types: FP8 E4M3 x FP8 E4M3 -> FP32 (acc: FP32)\n");
    printf("  Tile: 256x128x64  Cluster: 2x2x1\n\n");

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

    // Strides
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, make_shape(M, N, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1));

    // Allocate device memory
    size_t sz_a = (size_t)M * K;
    size_t sz_b = (size_t)N * K;
    size_t sz_c = (size_t)M * N * sizeof(float);

    ElementA* d_A = nullptr;
    ElementB* d_B = nullptr;
    ElementOut* d_C = nullptr;  // source (beta=0, unused)
    ElementOut* d_D = nullptr;  // output

    cudaMalloc(&d_A, sz_a);
    cudaMalloc(&d_B, sz_b);
    cudaMalloc(&d_C, sz_c);
    cudaMalloc(&d_D, sz_c);

    // Fill with random data (as uint8 via curand would work, but let's just zero-init
    // and fill with a simple pattern — for benchmarking, data content doesn't matter)
    cudaMemset(d_A, 0x3C, sz_a);  // ~1.0 in E4M3
    cudaMemset(d_B, 0x3C, sz_b);
    cudaMemset(d_C, 0, sz_c);
    cudaMemset(d_D, 0, sz_c);

    // Construct arguments: alpha=1, beta=0
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {d_A, stride_a, d_B, stride_b},
        {{}, d_C, stride_c, d_D, stride_d}
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
    printf("Warmup: 10 iterations...\n");
    for (int i = 0; i < 10; i++) {
        status = gemm.run();
        if (status != cutlass::Status::kSuccess) {
            printf("ERROR: CUTLASS run failed on warmup iter %d (status %d)\n", i, (int)status);
            return 1;
        }
    }
    cudaDeviceSynchronize();

    // Timed
    constexpr int N_ITER = 100;
    printf("Timing: %d iterations...\n", N_ITER);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    for (int i = 0; i < N_ITER; i++) {
        gemm.run();
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float total_ms;
    cudaEventElapsedTime(&total_ms, t0, t1);
    float avg_ms = total_ms / N_ITER;

    double flops = 2.0 * M * N * K;
    double tflops = flops / (avg_ms * 1e9);

    printf("\nCUTLASS: %.3f ms  %.2f TFLOPS\n", avg_ms, tflops);

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
