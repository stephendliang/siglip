// CUTLASS 4.x Blackwell MXFP8 (microscaling) GEMM benchmark
// A[M,K] x B[N,K]^T -> C[M,N]   MXFP8 E4M3 inputs w/ UE8M0 block scales, FP32 accumulate, BF16 output
// Uses OpClassBlockScaledTensorOp (tcgen05.mma.kind::mxf8f6f4.block_scale)
// Reports both GEMM-only and GEMM + bias+pos_embed times for apples-to-apples comparison
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

#include <cuda_bf16.h>

using namespace cute;

// ── Post-processing kernels for bias+pos_embed ──

// Precompute combined[r][c] = __float2bfloat16(bias[c] + pos_embed[r*N+c])
__global__ void precompute_combined(
    const float* __restrict__ bias,
    const float* __restrict__ pos_embed,
    __nv_bfloat16* __restrict__ combined,
    int seq_len, int n_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len * n_dim) {
        int c = idx % n_dim;
        combined[idx] = __float2bfloat16(bias[c] + pos_embed[idx]);
    }
}

// Post-processing: C[row, col] += combined[row % seq_len, col]
// Vectorized: 8 BF16 per thread via 128-bit loads/stores
__global__ void apply_combined(
    __nv_bfloat16* __restrict__ C,
    const __nv_bfloat16* __restrict__ combined,
    long long total_v8, int N, int seq_len
) {
    long long vid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= total_v8) return;
    long long base = vid * 8;
    int row = (int)(base / N);
    int col = (int)(base % N);
    int pos_row = row % seq_len;

    uint4 cv = *reinterpret_cast<uint4*>(C + base);
    uint4 bv = *reinterpret_cast<const uint4*>(combined + (long long)pos_row * N + col);

    uint32_t* cp = reinterpret_cast<uint32_t*>(&cv);
    const uint32_t* bp = reinterpret_cast<const uint32_t*>(&bv);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float c0, c1, b0, b1;
        asm("{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "cvt.rn.f32.bf16 %0, lo;\n\t"
            "cvt.rn.f32.bf16 %1, hi;\n\t"
            "}" : "=f"(c0), "=f"(c1) : "r"(cp[i]));
        asm("{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "cvt.rn.f32.bf16 %0, lo;\n\t"
            "cvt.rn.f32.bf16 %1, hi;\n\t"
            "}" : "=f"(b0), "=f"(b1) : "r"(bp[i]));
        c0 += b0;
        c1 += b1;
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(cp[i]) : "f"(c0), "f"(c1));
    }

    *reinterpret_cast<uint4*>(C + base) = cv;
}

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

    printf("  Scale factors: SFA=%zu bytes, SFB=%zu bytes\n", sz_sfa, sz_sfb);

    // ── Allocate bias+pos_embed combined array ──
    float *d_bias, *d_pos;
    __nv_bfloat16 *d_combined;
    cudaMalloc(&d_bias, (size_t)N * sizeof(float));
    cudaMalloc(&d_pos, (size_t)SEQ_LEN * N * sizeof(float));
    cudaMalloc(&d_combined, (size_t)SEQ_LEN * N * sizeof(__nv_bfloat16));
    cudaMemset(d_bias, 0, (size_t)N * sizeof(float));
    cudaMemset(d_pos, 0, (size_t)SEQ_LEN * N * sizeof(float));
    {
        int n_elem = SEQ_LEN * N;
        int tpb = 256;
        precompute_combined<<<(n_elem + tpb - 1) / tpb, tpb>>>(d_bias, d_pos, d_combined, SEQ_LEN, N);
    }
    printf("  Combined bias+pos: %zu bytes\n\n", (size_t)SEQ_LEN * N * sizeof(__nv_bfloat16));

    // apply_combined launch config (vectorized: 8 BF16 per thread)
    long long total_v8 = (long long)M * N / 8;
    int ac_tpb = 256;
    int ac_blocks = (int)((total_v8 + ac_tpb - 1) / ac_tpb);

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
        apply_combined<<<ac_blocks, ac_tpb>>>(
            reinterpret_cast<__nv_bfloat16*>(d_D), d_combined, total_v8, N, SEQ_LEN);
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

    // ── Timed: GEMM only ──
    const int ITERS = 20;
    printf("  Timing GEMM only (%d iters)...\n", ITERS);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; i++) {
        gemm.run();
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms_gemm;
    cudaEventElapsedTime(&ms_gemm, t0, t1);
    ms_gemm /= ITERS;

    double flops = 2.0 * M * N * K;

    // ── Timed: GEMM + bias+pos_embed ──
    printf("  Timing GEMM + bias+pos (%d iters)...\n", ITERS);

    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; i++) {
        gemm.run();
        apply_combined<<<ac_blocks, ac_tpb>>>(
            reinterpret_cast<__nv_bfloat16*>(d_D), d_combined, total_v8, N, SEQ_LEN);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms_full;
    cudaEventElapsedTime(&ms_full, t0, t1);
    ms_full /= ITERS;

    printf("\nCUTLASS MXFP8 (GEMM only):       %.3f ms  %.2f TFLOPS\n", ms_gemm, flops / ms_gemm / 1e9);
    printf("CUTLASS MXFP8 (GEMM+bias+pos):   %.3f ms  %.2f TFLOPS\n", ms_full, flops / ms_full / 1e9);
    printf("  bias+pos overhead: %.3f ms\n", ms_full - ms_gemm);
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
    cudaFree(d_bias);
    cudaFree(d_pos);
    cudaFree(d_combined);

    return 0;
#endif
}
