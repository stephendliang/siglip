// CUTLASS 4.x SM100a per-tensor FP8 GEMM benchmark
// ═══════════════════════════════════════════════════
//
// Per-tensor FP8 E4M3 (OpClassTensorOp) — matches cuBLAS per-tensor path
// and the custom megakernel's MMA mode (tcgen05.mma.kind::f8f6f4).
//
// OLD bench used OpClassBlockScaledTensorOp (MXFP8) which is a different,
// heavier MMA instruction with per-32-element scale factor overhead.
//
// Three measurements:
//   1. GEMM only (beta=0): pure compute, compare with cuBLAS 3001 TFLOPS
//   2. Fused (beta=1): D = acc + C (C = tiled combined[bias+pos_embed])
//      Compare with custom kernel fused path (0.530 ms / 2067 TFLOPS)
//   3. Unfused: GEMM + separate apply_combined kernel
//      Compare with cuBLAS unfused path (0.835 ms)
//
// Compile-time tile/cluster config (override via -D or CUTLASS_TILE=):
//   TILE_M   (default 256)   CLUSTER_M (default 2)
//   TILE_N   (default 128)   CLUSTER_N (default 1)
//   TILE_K   (default 64)
//
// Build:
//   make cutlass-bench
//   make cutlass-bench CUTLASS_TILE="-DTILE_N=256 -DTILE_K=128"
//
// SASS analysis (no GPU needed):
//   cuobjdump --dump-sass cutlass-bench > cutlass_sass.txt
//
//   What to look for in the SASS dump:
//     UTCQMMA.2CTA    — MMA instructions (tensor core issue)
//     UTCL / UTCLD     — TMEM load (tcgen05.ld) — epilogue readback
//     UTCBAR           — MMA commit (tcgen05.commit)
//     LDGSTS / LDSM    — TMA / shared memory loads
//     STS / LDS        — shared memory stores / loads
//     STG              — global store
//     R2UR             — register to uniform register (MMA operand transfer)
//
//   Compare epilogue section register count and TMEM load pattern
//   against: cuobjdump --dump-sass siglip_vision > custom_sass.txt
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
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include <cuda_bf16.h>

using namespace cute;

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

// ── Unfused post-processing kernels (same as cublas_bench.cu) ──

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
        __nv_bfloat162 c2, b2;
        c2 = *reinterpret_cast<__nv_bfloat162*>(&cp[i]);
        b2 = *reinterpret_cast<const __nv_bfloat162*>(&bp[i]);
        c2 = __hadd2(c2, b2);
        cp[i] = *reinterpret_cast<uint32_t*>(&c2);
    }

    *reinterpret_cast<uint4*>(C + base) = cv;
}

// ═══════════════════════════════════════════════════════════════
// CUTLASS kernel — per-tensor FP8 E4M3, BF16 output
// ═══════════════════════════════════════════════════════════════

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// A: [M, K] row-major, FP8 E4M3
using ElementA  = cutlass::float_e4m3_t;
using LayoutA   = cutlass::layout::RowMajor;
constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16

// B: [N, K] col-major (= [K, N] in memory, transposed access)
using ElementB  = cutlass::float_e4m3_t;
using LayoutB   = cutlass::layout::ColumnMajor;
constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 16

// C (source for fused epilogue): BF16
// D (output): BF16
// NOTE: if RowMajor doesn't compile for SM100 TMA epilogue, change to ColumnMajor
using ElementC  = cutlass::bfloat16_t;
using ElementD  = cutlass::bfloat16_t;
#ifdef COL_MAJOR_OUTPUT
using LayoutC   = cutlass::layout::ColumnMajor;
using LayoutD   = cutlass::layout::ColumnMajor;
#else
using LayoutC   = cutlass::layout::RowMajor;
using LayoutD   = cutlass::layout::RowMajor;
#endif
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 8
constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;  // 8

using ElementAcc     = float;
using ElementCompute = float;

// ── Tile and cluster shapes ──
// Override any of these with -D flags at compile time.
// Configs to try for [928256, 768] x [768, 768]^T:
//
//   Default:     256x128x64  cluster 2x1  (CUTLASS example 70 baseline)
//   Wider N:     256x256x64  cluster 2x1  (matches custom kernel TN=256; 3 N-tiles)
//   Deeper K:    256x128x128 cluster 2x1  (matches custom kernel TK=128; 6 K-iters)
//   1SM:         128x128x64  cluster 1x2  (1SM per CTA, cluster along N)
//   Large:       256x256x128 cluster 2x1  (closest to custom kernel tile)

#ifndef TILE_M
#define TILE_M 256
#endif
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef TILE_K
#define TILE_K 64
#endif
#ifndef CLUSTER_M
#define CLUSTER_M 2
#endif
#ifndef CLUSTER_N
#define CLUSTER_N 1
#endif

using TileShape    = Shape<Int<TILE_M>, Int<TILE_N>, Int<TILE_K>>;
using ClusterShape = Shape<Int<CLUSTER_M>, Int<CLUSTER_N>, _1>;

// Fusion: D = alpha * acc + beta * C
// beta=0 → pure GEMM (C not loaded), beta=1 → fused add
using FusionOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, float,
    cutlass::FloatRoundStyle::round_to_nearest>;

// ── Epilogue collective ──
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementCompute,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionOp
>::CollectiveOp;

// ── Mainloop collective ──
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

// ── Kernel + adapter ──
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

// ═══════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    setbuf(stdout, NULL);

#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    printf("ERROR: CUTLASS_ARCH_MMA_SM100_SUPPORTED not defined.\n");
    printf("  Requires CUTLASS 4.x, CUDA 12.8+, and -arch=sm_100a.\n");
    printf("  If compiling for SASS analysis only, ensure CUTLASS is in third_party/cutlass/\n");
    return 1;
#else
    const int SM_COUNT = 148;
    const int SEQ_LEN  = 196;
    const int N        = 768;
    const int K        = 768;

    int imgs_per_sm = 32;
    if (argc > 1) imgs_per_sm = atoi(argv[1]);
    const int M = imgs_per_sm * SM_COUNT * SEQ_LEN;

    printf("CUTLASS SM100a per-tensor FP8 GEMM benchmark\n");
    printf("  M=%d  N=%d  K=%d  (imgs_per_sm=%d, %d images)\n", M, N, K, imgs_per_sm, M / SEQ_LEN);
    printf("  Types: FP8 E4M3 (per-tensor) -> BF16 (acc: FP32)\n");
    printf("  Tile: %dx%dx%d  Cluster: %dx%dx1\n", TILE_M, TILE_N, TILE_K, CLUSTER_M, CLUSTER_N);
#ifdef COL_MAJOR_OUTPUT
    printf("  Output layout: ColumnMajor\n");
#else
    printf("  Output layout: RowMajor\n");
#endif

    // ── Device info ──
    cudaDeviceProp props;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);
    printf("  Device: %s (SM %d.%d)\n\n", props.name, props.major, props.minor);

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    // ── Strides ──
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, make_shape(M, N, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1));

    // ── Allocate data ──
    size_t sz_a  = (size_t)M * K;                     // FP8 = 1 byte each
    size_t sz_b  = (size_t)N * K;
    size_t sz_cd = (size_t)M * N * sizeof(ElementD);   // BF16 = 2 bytes each

    void *d_A = nullptr, *d_B = nullptr;
    void *d_C = nullptr, *d_D = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, sz_a));
    CUDA_CHECK(cudaMalloc(&d_B, sz_b));
    CUDA_CHECK(cudaMalloc(&d_C, sz_cd));
    CUDA_CHECK(cudaMalloc(&d_D, sz_cd));

    // Fill A,B with 0x3C (1.5 in E4M3), zero C/D
    CUDA_CHECK(cudaMemset(d_A, 0x3C, sz_a));
    CUDA_CHECK(cudaMemset(d_B, 0x3C, sz_b));
    CUDA_CHECK(cudaMemset(d_C, 0, sz_cd));
    CUDA_CHECK(cudaMemset(d_D, 0, sz_cd));

    // ── Combined bias+pos_embed (for fused and unfused comparison) ──
    float *d_bias, *d_pos;
    __nv_bfloat16 *d_combined;
    CUDA_CHECK(cudaMalloc(&d_bias, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos, (size_t)SEQ_LEN * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_combined, (size_t)SEQ_LEN * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(d_bias, 0, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pos, 0, (size_t)SEQ_LEN * N * sizeof(float)));
    {
        int n_elem = SEQ_LEN * N;
        int tpb = 256;
        precompute_combined<<<(n_elem + tpb - 1) / tpb, tpb>>>(d_bias, d_pos, d_combined, SEQ_LEN, N);
        CUDA_CHECK(cudaGetLastError());
    }

    // Tile the [196, 768] combined tensor to [M, 768] for the fused epilogue.
    // Each image's 196-row block gets the same combined data.
    // Memory: M * N * 2 bytes = ~1.35 GB for M=928256 — fine for B200 (192 GB).
    {
        int num_images = M / SEQ_LEN;
        size_t row_bytes = (size_t)SEQ_LEN * N * sizeof(__nv_bfloat16);
        for (int img = 0; img < num_images; img++) {
            CUDA_CHECK(cudaMemcpy(
                (char*)d_C + img * row_bytes,
                d_combined,
                row_bytes,
                cudaMemcpyDeviceToDevice));
        }
    }
    printf("  Combined tensor tiled to [%d, %d] (%.1f MB)\n\n", M, N, (double)sz_cd / (1024*1024));

    // apply_combined launch config (8 BF16 per thread)
    long long total_v8 = (long long)M * N / 8;
    int ac_tpb = 256;
    int ac_blocks = (int)((total_v8 + ac_tpb - 1) / ac_tpb);

    // ── GEMM arguments ──
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            reinterpret_cast<ElementA*>(d_A), stride_a,
            reinterpret_cast<ElementB*>(d_B), stride_b
        },
        {
            {},  // fusion thread args (set below)
            reinterpret_cast<ElementC*>(d_C), stride_c,
            reinterpret_cast<ElementD*>(d_D), stride_d
        },
        hw_info
    };

    Gemm gemm;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    uint8_t* d_workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }
    printf("  Workspace: %zu bytes\n", workspace_size);

    cutlass::Status status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        printf("ERROR: CUTLASS cannot implement this problem (status %d)\n", (int)status);
        printf("  Try a different tile/cluster config, or add -DCOL_MAJOR_OUTPUT\n");
        return 1;
    }

    // ── Events ──
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    double flops = 2.0 * M * N * K;

    const int WARMUP = 3;
    const int ITERS  = 20;

    // ════════════════════════════════════════════════════════════
    // 1. GEMM only (beta=0, no C load)
    // ════════════════════════════════════════════════════════════
    printf("\n  [1/3] GEMM only (beta=0)...\n");
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta  = 0.0f;

    status = gemm.initialize(arguments, d_workspace);
    if (status != cutlass::Status::kSuccess) {
        printf("ERROR: initialize failed for GEMM-only (status %d)\n", (int)status);
        return 1;
    }

    for (int i = 0; i < WARMUP; i++) {
        status = gemm.run();
        if (status != cutlass::Status::kSuccess) {
            printf("ERROR: GEMM run failed on warmup iter %d (status %d)\n", i, (int)status);
            return 1;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify
    {
        ElementD h_D[4];
        CUDA_CHECK(cudaMemcpy(h_D, d_D, 4 * sizeof(ElementD), cudaMemcpyDeviceToHost));
        float expected = (float)K * 1.5f * 1.5f;
        printf("  C[0,0..3] = %.1f %.1f %.1f %.1f (expected %.1f)\n",
            float(h_D[0]), float(h_D[1]), float(h_D[2]), float(h_D[3]), expected);
    }

    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; i++) gemm.run();
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms_gemm;
    cudaEventElapsedTime(&ms_gemm, t0, t1);
    ms_gemm /= ITERS;

    // ════════════════════════════════════════════════════════════
    // 2. Fused: D = acc + C (beta=1)
    // ════════════════════════════════════════════════════════════
    printf("  [2/3] Fused GEMM + epilogue (beta=1)...\n");
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta  = 1.0f;

    status = gemm.initialize(arguments, d_workspace);
    if (status != cutlass::Status::kSuccess) {
        printf("ERROR: initialize failed for fused (status %d)\n", (int)status);
        return 1;
    }

    for (int i = 0; i < WARMUP; i++) gemm.run();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; i++) gemm.run();
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms_fused;
    cudaEventElapsedTime(&ms_fused, t0, t1);
    ms_fused /= ITERS;

    // ════════════════════════════════════════════════════════════
    // 3. Unfused: GEMM (beta=0) + separate apply_combined kernel
    // ════════════════════════════════════════════════════════════
    printf("  [3/3] GEMM + unfused apply_combined...\n");
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta  = 0.0f;

    status = gemm.initialize(arguments, d_workspace);
    if (status != cutlass::Status::kSuccess) {
        printf("ERROR: initialize failed for unfused (status %d)\n", (int)status);
        return 1;
    }

    for (int i = 0; i < WARMUP; i++) {
        gemm.run();
        apply_combined<<<ac_blocks, ac_tpb>>>(
            reinterpret_cast<__nv_bfloat16*>(d_D), d_combined, total_v8, N, SEQ_LEN);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; i++) {
        gemm.run();
        apply_combined<<<ac_blocks, ac_tpb>>>(
            reinterpret_cast<__nv_bfloat16*>(d_D), d_combined, total_v8, N, SEQ_LEN);
    }
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms_unfused;
    cudaEventElapsedTime(&ms_unfused, t0, t1);
    ms_unfused /= ITERS;

    // ════════════════════════════════════════════════════════════
    // Results
    // ════════════════════════════════════════════════════════════
    printf("\n══════════════════════════════════════════════════════════\n");
    printf("CUTLASS SM100a per-tensor FP8 (tile %dx%dx%d, cluster %dx%d)\n",
           TILE_M, TILE_N, TILE_K, CLUSTER_M, CLUSTER_N);
    printf("══════════════════════════════════════════════════════════\n");
    printf("  GEMM only:              %.3f ms  %7.1f TFLOPS\n", ms_gemm, flops/ms_gemm/1e9);
    printf("  Fused (acc + C):        %.3f ms  %7.1f TFLOPS\n", ms_fused, flops/ms_fused/1e9);
    printf("  Unfused (GEMM + post):  %.3f ms  %7.1f TFLOPS\n", ms_unfused, flops/ms_unfused/1e9);
    printf("  Fusion overhead:        %.3f ms (fused - GEMM)\n", ms_fused - ms_gemm);
    printf("  Unfused overhead:       %.3f ms (unfused - GEMM)\n", ms_unfused - ms_gemm);
    printf("\n  Reference:\n");
    printf("    cuBLAS per-tensor FP8:  0.365 ms / 3001 TFLOPS (GEMM only)\n");
    printf("    cuBLAS + unfused pos:   0.835 ms\n");
    printf("    Custom kernel (fused):  0.530 ms / 2067 TFLOPS\n");
    printf("  M=%d  N=%d  K=%d\n", M, N, K);

    // ── N-tile analysis ──
    int n_tiles_n = (N + TILE_N - 1) / TILE_N;
    int n_tiles_m = (M + (TILE_M / CLUSTER_M) - 1) / (TILE_M / CLUSTER_M);
    // For 2SM: each cluster processes TILE_M rows across CLUSTER_M CTAs
    // Actual M per cluster = TILE_M (distributed across CLUSTER_M SMs)
    int clusters = SM_COUNT / CLUSTER_M;  // 74 for cluster_m=2
    n_tiles_m = (M + TILE_M - 1) / TILE_M;
    printf("\n  Tile analysis:\n");
    printf("    N-tiles: %d (N=%d / TN=%d)\n", n_tiles_n, N, TILE_N);
    printf("    M-tiles: %d\n", n_tiles_m);
    printf("    Total tiles: %d\n", n_tiles_m * n_tiles_n);
    printf("    Clusters: %d (148 SMs / %d)\n", clusters, CLUSTER_M);
    printf("    K-iters: %d (K=%d / TK=%d)\n", K / TILE_K, K, TILE_K);

    // Cleanup
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    if (d_workspace) cudaFree(d_workspace);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_bias);
    cudaFree(d_pos);
    cudaFree(d_combined);

    return 0;
#endif
}
