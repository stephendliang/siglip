// CUTLASS 4.x Blackwell MXFP8 (microscaling) GEMM benchmark
// A[M,K] x B[N,K]^T -> C[M,N]   MXFP8 E4M3 inputs w/ UE8M0 block scales, FP32 accumulate, BF16 output
// Uses OpClassBlockScaledTensorOp (tcgen05.mma.kind::mxf8f6f4.block_scale)
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
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// ═══════════════════════════════════════════════════════════════
// CUTLASS kernel configuration — MXFP8 (block-scaled)
// ═══════════════════════════════════════════════════════════════

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// MXFP8: mx_float8_t wraps float_e4m3_t data + UE8M0 per-32-element scale factors
using ElementA       = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementB       = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementAcc     = float;
using ElementOut     = cutlass::bfloat16_t;      // BF16 output, matching cuBLAS bench
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::ColumnMajor;

constexpr int AlignA = 16;   // MXFP8 alignment in elements
constexpr int AlignB = 16;
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementOut>::value;  // 8
constexpr int AlignD = AlignC;

// Block-scaled 2SM MMA tile — configurable via TILE_M/N/K and CLUSTER_M/N macros
#ifndef TILE_M
#define TILE_M 256
#endif
#ifndef TILE_N
#define TILE_N 192
#endif
#ifndef TILE_K
#define TILE_K 128
#endif
#ifndef CLUSTER_M
#define CLUSTER_M 2
#endif
#ifndef CLUSTER_N
#define CLUSTER_N 1
#endif
using MmaTileShape = Shape<cute::Int<TILE_M>, cute::Int<TILE_N>, cute::Int<TILE_K>>;
using ClusterShape = Shape<cute::Int<CLUSTER_M>, cute::Int<CLUSTER_N>, _1>;

using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementCompute,
    ElementOut, LayoutC, AlignC,
    ElementOut, LayoutD, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, OperatorClass,
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
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA   = typename Gemm::GemmKernel::StrideA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;

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

    printf("CUTLASS 4.x Blackwell MXFP8 GEMM Benchmark\n");
    printf("  M=%d  N=%d  K=%d  (imgs_per_sm=%d)\n", M, N, K, imgs_per_sm);
    printf("  Types: MXFP8 E4M3 (UE8M0 block scales) -> BF16 (acc: FP32)\n");
    printf("  Tile: %dx%dx%d  Cluster: %dx%dx1\n\n", TILE_M, TILE_N, TILE_K, CLUSTER_M, CLUSTER_N);

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

    // Strides for A, B, C, D
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, make_shape(M, N, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1));

    // Scale factor layouts (interleaved, computed from problem shape)
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    LayoutSFA layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    LayoutSFB layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

    // Allocate device memory for data tensors
    size_t sz_a  = (size_t)M * K;        // FP8 elements = 1 byte each
    size_t sz_b  = (size_t)N * K;
    size_t sz_cd = (size_t)M * N * sizeof(ElementOut);

    ElementA::DataType* d_A = nullptr;
    ElementB::DataType* d_B = nullptr;
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

    // Allocate scale factor arrays (UE8M0, 1 byte per 32 elements along K)
    // filter_zeros gives the actual allocation size
    size_t sz_sfa = size(filter_zeros(layout_sfa));
    size_t sz_sfb = size(filter_zeros(layout_sfb));

    ElementA::ScaleFactorType* d_SFA = nullptr;  // cutlass::float_ue8m0_t
    ElementB::ScaleFactorType* d_SFB = nullptr;

    cudaMalloc(&d_SFA, sz_sfa);
    cudaMalloc(&d_SFB, sz_sfb);

    // Fill scale factors with 0x7F = UE8M0 encoding of 2^0 = 1.0
    cudaMemset(d_SFA, 0x7F, sz_sfa);
    cudaMemset(d_SFB, 0x7F, sz_sfb);

    printf("  Scale factors: SFA=%zu bytes, SFB=%zu bytes\n\n", sz_sfa, sz_sfb);

    // Construct GEMM arguments
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        { // Mainloop: data + scale factors
          d_A, stride_a,
          d_B, stride_b,
          d_SFA, layout_sfa,
          d_SFB, layout_sfb
        },
        { // Epilogue: alpha * acc + beta * C
          {1.0f, 0.0f},
          d_C, stride_c,
          d_D, stride_d
        },
        hw_info
    };

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

    printf("\nCUTLASS MXFP8:  %.3f ms  %.2f TFLOPS\n", avg_ms, tflops);
    printf("  M=%d  N=%d  K=%d\n", M, N, K);

    // Cleanup
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    if (workspace) cudaFree(workspace);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_SFA);
    cudaFree(d_SFB);

    return 0;
#endif
}
