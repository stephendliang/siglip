/*
FC2 Hybrid kernel — scaffold for our PTX mainloop + CUTLASS CollectiveEpilogue.
Shape: [928256,3072]×[3072,768]^T + bias + residual

Goal: best of both worlds.
  Our mainloop:      1.095ms strip (62μs faster than CUTLASS's 1.157ms)
  CUTLASS epilogue:  72μs overhead (vs our 388μs)
  Hybrid ceiling:    1.095 + 0.072 = 1.167ms

Architecture: 8 warps (256 threads), cta_group::2, cluster 2×1
  W0: Scheduler (CLC)
  W1: MMA           ← Phase 2: replace with our PTX K-loop via custom CollectiveMainloop
  W2: MainloopLoad  ← Phase 3: replace with our PTX TMA loads
  W3: EpilogueLoad  — CUTLASS (unchanged)
  W4-W7: Epilogue   — CUTLASS (unchanged)

Build:
  make fc2-hybrid              # Phase 1: CUTLASS via custom launch
  make fc2-hybrid-strip        # GEMM-only
  make fc2-hybrid-mma          # Phase 2: (stub, same as Phase 1 until mainloop wrapper ready)

Phase 1 (current): Custom __global__ that delegates to GemmKernel::operator().
  Validates: type extraction, custom launch mechanism, Makefile/bench integration.
  Performance: identical to fc2-cutlass (1.224ms fused, 1.157ms strip).

Phase 2 (next): Override CollectiveMainloop::mma() with our PTX K-loop.
  Approach: Create HybridMainloop class that wraps CUTLASS's CollectiveMainloop,
  overriding only the mma() method. Use with GemmUniversal<..., HybridMainloop, ...>.
  This stays WITHIN GemmKernel::operator() template chain, avoiding
  __host__/__device__ template instantiation errors from CuTe's tapply().

  Key lesson: cannot call CUTLASS epilogue from custom device code — must inject
  our PTX through the CollectiveMainloop interface, not through the kernel dispatch.

Phase 3 (future): Also override CollectiveMainloop::load() with our TMA PTX.
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

/* ═══════════════════════════════════════════════════════════════════
   Problem dimensions — identical to fc2_w3.cu and fc2_cutlass.cu
   ═══════════════════════════════════════════════════════════════════ */

#define SM_COUNT  148
#define M_TOTAL   928256   /* 4736 images × 196 patches */
#define N_DIM     768
#define K_DIM     3072

/* ═══════════════════════════════════════════════════════════════════
   CUTLASS type setup — same as fc2_cutlass.cu
   ═══════════════════════════════════════════════════════════════════ */

using ElementA   = cutlass::float_e4m3_t;
using LayoutA    = cutlass::layout::RowMajor;
using ElementB   = cutlass::float_e4m3_t;
using LayoutB    = cutlass::layout::ColumnMajor;
using ElementC   = cutlass::bfloat16_t;
using ElementD   = cutlass::bfloat16_t;
using LayoutC    = cutlass::layout::RowMajor;
using LayoutD    = cutlass::layout::RowMajor;
using ElementAcc = float;

constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;

using TileShape    = Shape<_256, _256, _128>;
using ClusterShape = Shape<_2, _1, _1>;

using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;

#ifdef STRIP_EPILOGUE
using FusionOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementAcc>;
#else
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

/* ═══════════════════════════════════════════════════════════════════
   PTX helpers — for Phase 2 (our MMA inner loop)
   ═══════════════════════════════════════════════════════════════════ */

static __device__ __forceinline__
uint32_t smem_to_uint(const void* p) {
    return static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(__cvta_generic_to_shared(p)));
}

/* WGMMA descriptor for SWIZZLE_128B mode, SBO=1024 */
static __device__ __forceinline__
uint64_t make_smem_desc(uint32_t addr) {
    uint64_t d = 0;
    constexpr uint32_t SBO = 1024;
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= (uint64_t)((SBO  & 0x3FFFF) >> 4) << 32;
    d |= (1ULL << 46);
    d |= (2ULL << 61);   /* SWIZZLE_128B */
    return d;
}

static __device__ __forceinline__
void tcgen05_commit_mcast(uint32_t mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        :: "r"(mbar_addr), "h"(cta_mask) : "memory");
}

/* ═══════════════════════════════════════════════════════════════════
   Custom kernel — template __global__ matching CUTLASS's device_kernel<>
   ═══════════════════════════════════════════════════════════════════ */

/*
Both Phase 1 and Phase 2 use the same launch pattern as CUTLASS:
  fc2_hybrid_kernel_impl<Operator>(params) → Operator::operator()(params, smem)

Phase 1: Operator = GemmKernel (pure CUTLASS — validates custom launch)
Phase 2: Operator = GemmUniversal<HybridMainloop, ...> (our mainloop + CUTLASS epilogue)
*/

template <typename Operator>
__global__ void
__launch_bounds__(Operator::MaxThreadsPerBlock, 1)
__cluster_dims__(2, 1, 1)
fc2_hybrid_kernel_impl(typename Operator::Params const params) {
    extern __shared__ char smem_buf[];
    Operator op;
    op(params, smem_buf);
}

/* For now, Phase 2 = Phase 1. When HybridMainloop is ready:
   using HybridGemmKernel = GemmUniversal<Shape<int,int,int,int>,
                                           HybridMainloop, CollectiveEpilogue>;
   static auto* fc2_hybrid_kernel = fc2_hybrid_kernel_impl<HybridGemmKernel>; */
static auto* fc2_hybrid_kernel = fc2_hybrid_kernel_impl<GemmKernel>;

/* ═══════════════════════════════════════════════════════════════════
   Host code — identical to fc2_cutlass.cu except launches our kernel
   ═══════════════════════════════════════════════════════════════════ */

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
    printf("FC2 HYBRID — custom kernel launch with CUTLASS types\n");
#ifdef STRIP_EPILOGUE
    printf("  MODE: GEMM-only (STRIP_EPILOGUE)\n");
#else
    printf("  MODE: Fused (bias + residual)\n");
#endif
    printf("  [%d, %d] x [%d, %d]^T  FP8E4M3 -> BF16\n", M_TOTAL, K_DIM, K_DIM, N_DIM);
    printf("  Tile: 256x256x128, Cluster: 2x1, Schedule: 2SM\n");

    constexpr size_t SmemBytes = sizeof(typename GemmKernel::SharedStorage);
    printf("  SMEM: %zu bytes\n", SmemBytes);

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

    void *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_D = nullptr;
    float *d_bias = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, sz_a));
    CUDA_CHECK(cudaMalloc(&d_B, sz_b));
    CUDA_CHECK(cudaMalloc(&d_C, sz_cd));
    CUDA_CHECK(cudaMalloc(&d_D, sz_cd));
    CUDA_CHECK(cudaMalloc(&d_bias, sz_bias));

    CUDA_CHECK(cudaMemset(d_A, 0x3C, sz_a));
    {
        uint8_t* h_B = (uint8_t*)malloc(sz_b);
        for (int n = 0; n < N_DIM; n++)
            memset(h_B + (size_t)n * K_DIM, (n & 1) ? 0x38 : 0x3C, K_DIM);
        CUDA_CHECK(cudaMemcpy(d_B, h_B, sz_b, cudaMemcpyHostToDevice));
        free(h_B);
    }
    {
        float* h_bias = (float*)malloc(sz_bias);
        for (int c = 0; c < N_DIM; c++)
            h_bias[c] = (float)(c + 1);
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias, sz_bias, cudaMemcpyHostToDevice));
        free(h_bias);
    }
#ifndef STRIP_EPILOGUE
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

    /* Use CUTLASS for argument setup — builds TMA descriptors, workspace, etc. */
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
        { reinterpret_cast<ElementA*>(d_A), stride_a,
          reinterpret_cast<ElementB*>(d_B), stride_b },
        { {},
          reinterpret_cast<ElementC*>(d_C), stride_c,
          reinterpret_cast<ElementD*>(d_D), stride_d },
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

    /* Extract kernel params and launch config from CUTLASS */
    auto kernel_params = gemm.params();
    auto grid = GemmKernel::get_grid_shape(kernel_params);
    auto block = GemmKernel::get_block_shape();

    CUDA_CHECK(cudaFuncSetAttribute(fc2_hybrid_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SmemBytes));
    CUDA_CHECK(cudaFuncSetAttribute(fc2_hybrid_kernel,
        cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

    printf("  CUTLASS initialized (workspace=%zu, grid=%dx%d, block=%d, smem=%zu)\n",
           ws, grid.x, grid.y, block.x, SmemBytes);

    /* Cluster launch config — <<<>>> doesn't set up clusters, must use cudaLaunchKernelEx */
    auto launch_hybrid = [&](typename GemmKernel::Params const& p) {
        cudaLaunchConfig_t config = {};
        config.gridDim = grid;
        config.blockDim = block;
        config.dynamicSmemBytes = SmemBytes;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim = {2, 1, 1};
        config.attrs = attrs;
        config.numAttrs = 1;
        CUDA_CHECK(cudaLaunchKernelEx(&config, fc2_hybrid_kernel, p));
    };

    /* Warmup */
    printf("Launching warmup (2 iters)...\n");
    for (int i = 0; i < 2; i++)
        launch_hybrid(kernel_params);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Warmup done.\n");

    /* Timed: 10 iterations */
    printf("Timing: 10 iterations...\n");
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 10; i++)
        launch_hybrid(kernel_params);
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= 10.0f;
    printf("FC2-HYBRID: %.3f ms  %.2f TFLOPS\n", ms, flops / ms / 1e9);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    /* Checksum run */
    CUDA_CHECK(cudaMemset(d_D, 0, sz_cd));
    launch_hybrid(kernel_params);
    CUDA_CHECK(cudaDeviceSynchronize());

    __nv_bfloat16* h_D = (__nv_bfloat16*)malloc(sz_cd);
    CUDA_CHECK(cudaMemcpy(h_D, d_D, sz_cd, cudaMemcpyDeviceToHost));

    double cksum = 0;
    {
        long long total_elems = (long long)M_TOTAL * N_DIM;
        long long stride = total_elems / 1024;
        for (int i = 0; i < 1024; i++)
            cksum += (double)__bfloat162float(h_D[(long long)i * stride]);
    }

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
