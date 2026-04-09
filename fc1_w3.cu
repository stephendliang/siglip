/*
FC1 W3 kernel — warp-specialized persistent GEMM with GELU epilogue
Standalone binary: independent of kernel_common.cuh / kernel_body.cuh
Shape: [928256,768]×[768,3072]^T + bias + GELU
6 warps: W0(TMA A/B) | W1(MMA) | W2-W5(Epilogue: TMEM→GELU→STS→TMA store)
cta_group::2  __cluster_dims__(2,1,1)

Architecture: adapted from fc2_w3.cu for FC1 (no residual, GELU activation).
  - No W2 EpilogueLoad: FC1 has no residual, so no TMA residual loads needed.
  - Epilogue: FP32 GELU(acc + bias) → BF16 CVT → STS → TMA store.
  - 2-stage epilogue double-buffer for STS/TMA store overlap.

Compile-time flags:
  -DSTRIP_EPILOGUE      Skip epilogue (benchmark GEMM core only, valid=0)
  -DGEMM_ONLY           Write D=BF16(A×B), no bias, no GELU (valid=1)
  -DN_STAGES=N          Pipeline depth (default 5, max K_ITERS)
  -DNO_PREFILL          Restore epilogue_mbar wait in W1
  -DTILE_DISPATCH=4     Dedicated scheduler warp, atomicAdd dispatch
  -DNUM_EPI_WARPS=N     Epilogue warp count (1, 2, or 4; default 4)
*/

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>

/* ── Hardware ── */
#define SM_COUNT       148

/* ── Problem dims (overridable via -D flags) ── */
#ifndef M_TOTAL
#define M_TOTAL        928256
#endif
#ifndef N_DIM
#define N_DIM          3072
#endif
#ifndef K_DIM
#define K_DIM          768
#endif

/* ── Tile ── */
#define TM             128
#define TN             256
#define TK             128
#define TILES_M        ((M_TOTAL + TM * 2 - 1) / (TM * 2))
#define TILES_N        (N_DIM / TN)
#define TOTAL_TILES    (TILES_M * TILES_N)
#define K_ITERS        (K_DIM / TK)
#define MMA_K          32
#define MMA_PER_KI     (TK / MMA_K)
#define SNAKE_ORDER    1

/* ── Pipeline ── */
#ifndef N_STAGES
#define N_STAGES       5
#endif
#ifndef K_LOOP_UNROLL
#define K_LOOP_UNROLL  N_STAGES
#endif

/* ── Threads ── */
#ifndef NUM_EPI_WARPS
#define NUM_EPI_WARPS  4
#endif
#define NUM_WARPS      (2 + NUM_EPI_WARPS)   /* W0+W1 + epilogue warps */
#define THREADS        (32 * NUM_WARPS)
#define GROUPS_PER_WARP (4 / NUM_EPI_WARPS)
static_assert(4 % NUM_EPI_WARPS == 0, "NUM_EPI_WARPS must be 1, 2, or 4");

#ifndef TILE_DISPATCH
#define TILE_DISPATCH 0
#endif

#if TILE_DISPATCH == 4
#undef NUM_WARPS
#define NUM_WARPS (2 + NUM_EPI_WARPS + 1)   /* +1 scheduler warp */
#undef THREADS
#define THREADS (32 * NUM_WARPS)
#define SCHED_WARP (2 + NUM_EPI_WARPS)      /* scheduler warp index */
#endif

#if defined(STRIP_EPILOGUE) && defined(GEMM_ONLY)
#error "STRIP_EPILOGUE and GEMM_ONLY are mutually exclusive"
#endif

/* ptxas bar.sync register-operand workaround */
#define _STR(x)  #x
#define _XSTR(x) _STR(x)
#define _EPI_THR_1 32
#define _EPI_THR_2 64
#define _EPI_THR_3 96
#define _EPI_THR_4 128
#define _EPI_THR_X(n) _EPI_THR_##n
#define _EPI_THR(n)   _EPI_THR_X(n)
#define BAR_EPI_SYNC  "bar.sync 1, " _XSTR(_EPI_THR(NUM_EPI_WARPS)) ";"

/* ── SMEM layout ── */
#define STAGE_BYTES    32768                                     /* 16KB A + 16KB B */
#define OFF_TMEM           (N_STAGES * STAGE_BYTES)
#define OFF_TMA_MBAR       (OFF_TMEM + 8)
#define OFF_MMA_MBAR       (OFF_TMA_MBAR + N_STAGES * 8)
#define OFF_MAINLOOP_MBAR  (OFF_MMA_MBAR + N_STAGES * 8)
#define OFF_EPILOGUE_MBAR  (OFF_MAINLOOP_MBAR + 16)
#define _MBAR_END          (OFF_EPILOGUE_MBAR + 16)

#if TILE_DISPATCH == 4
/* W_sched→W0 scheduler pipe: 2-deep FIFO + produce/consume mbarriers */
#define OFF_SCHED_FIFO      _MBAR_END
#define OFF_SCHED_PROD_MBAR (OFF_SCHED_FIFO + 8)
#define OFF_SCHED_CONS_MBAR (OFF_SCHED_PROD_MBAR + 16)
#define OFF_BCAST_TILE      (OFF_SCHED_CONS_MBAR + 16)
#define OFF_TILE_READY_MBAR (OFF_BCAST_TILE + 8)
#define OFF_SCHED_EPOCH     (OFF_TILE_READY_MBAR + 16)
#define _LAYOUT_END         (OFF_SCHED_EPOCH + 8)
#else
#define _LAYOUT_END        _MBAR_END
#endif

#define NUM_EPI_SUBITERS   4
#define NUM_EPI_STAGES     2

/* Bias SMEM: all N_DIM BF16 bias values */
#define OFF_BIAS_SMEM      ((_LAYOUT_END + 15) & ~15)
#define BIAS_SMEM_BYTES    (N_DIM * 2)

/* Epilogue staging: 2-stage double-buffer for STS → TMA store */
#define STAGING_REGION_BYTES  (32 * 128)
#define EPI_STAGE_BYTES       (4 * STAGING_REGION_BYTES)
#define OFF_STAGING           ((OFF_BIAS_SMEM + BIAS_SMEM_BYTES + 1023) & ~1023)
#define SMEM_BYTES            ((OFF_STAGING + NUM_EPI_STAGES * EPI_STAGE_BYTES + 127) & ~127)

/* ── WGMMA / TMEM ── */
#define TMEM_COLS      512
#define IDESC          0x10400010U
#define SBO            1024
#define TMA_BYTES      32768

/* ── Macros ── */
#define PRAGMA_UNROLL(n) _Pragma(_UNROLL_STR(n))
#define _UNROLL_STR2(x) #x
#define _UNROLL_STR(x) _UNROLL_STR2(unroll x)

#define CUDA_CHECK(x) do { \
    cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
        exit(1); \
    } \
} while(0)

#define CU_CHECK(x) do { \
    CUresult r_ = (x); \
    if (r_ != CUDA_SUCCESS) { \
        const char* s_; cuGetErrorString(r_, &s_); \
        fprintf(stderr, "CU %s:%d: %s\n", __FILE__, __LINE__, s_); \
        exit(1); \
    } \
} while(0)

/* ── Device helpers ── */

static __device__ __forceinline__
uint32_t smem_to_uint(const void* p) {
    return static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(__cvta_generic_to_shared(p)));
}

template<typename T>
static __device__ __forceinline__ T warp_uniform(T x) {
    return __shfl_sync(0xFFFFFFFF, x, 0);
}

static __device__ __forceinline__
uint64_t make_smem_desc(uint32_t addr) {
    uint64_t d = 0;
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= (uint64_t)((SBO  & 0x3FFFF) >> 4) << 32;
    d |= (1ULL << 46);
    d |= (2ULL << 61);   /* SWIZZLE_128B */
    return d;
}

static __device__ __forceinline__
void mbar_init(uint32_t addr, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(addr), "r"(count));
}

static __device__ __forceinline__
void mbar_wait(uint32_t addr, uint32_t phase) {
    uint32_t done;
    do {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2, 0x989680;\n\t"
            "selp.b32 %0, 1, 0, p;\n\t"
            "}"
            : "=r"(done) : "r"(addr), "r"(phase));
    } while (!done);
}

static __device__ __forceinline__
void mbar_arrive(uint32_t addr) {
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
        :: "r"(addr) : "memory");
}

static __device__ __forceinline__
void tcgen05_commit_mcast(uint32_t mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        :: "r"(mbar_addr), "h"(cta_mask) : "memory");
}

/* ── TMEM load / wait ── */

#define TMEM_LOAD_X32(r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31, TADDR) \
    asm volatile( \
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 " \
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];" \
        : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3), \
          "=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7), \
          "=f"(r8),"=f"(r9),"=f"(r10),"=f"(r11), \
          "=f"(r12),"=f"(r13),"=f"(r14),"=f"(r15), \
          "=f"(r16),"=f"(r17),"=f"(r18),"=f"(r19), \
          "=f"(r20),"=f"(r21),"=f"(r22),"=f"(r23), \
          "=f"(r24),"=f"(r25),"=f"(r26),"=f"(r27), \
          "=f"(r28),"=f"(r29),"=f"(r30),"=f"(r31) \
        : "r"(TADDR))

#define TMEM_WAIT() \
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory")

/* ── GELU approximation ──
   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
   Rearranged: GELU(x) = 0.5 * (x + x * tanh(x * (0.7979 + 0.03568 * x^2)))
   8 FP32 ops/element: FADD, FMUL, FFMA, FMUL, MUFU.TANH, FMUL, FADD, FMUL */
static __device__ __forceinline__ float gelu_approx(float acc, float bias_f32) {
    float x = acc + bias_f32;
    float inner = x * (0.7978845608f + 0.035677408136f * x * x);
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(inner));
    return 0.5f * (x + x * t);
}

/* Host GELU for validation */
static __host__ __forceinline__ float gelu_fwd(float x) {
    const float k = 0.7978845608f;
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

/* CVT FP32→BF16 + STS.128 */
#define CVT_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, SADDR) \
    asm volatile( \
        "{\n\t" \
        ".reg .b32 o0, o1, o2, o3;\n\t" \
        "cvt.rn.bf16x2.f32 o0, %1, %0;\n\t" \
        "cvt.rn.bf16x2.f32 o1, %3, %2;\n\t" \
        "cvt.rn.bf16x2.f32 o2, %5, %4;\n\t" \
        "cvt.rn.bf16x2.f32 o3, %7, %6;\n\t" \
        "st.shared.v4.b32 [%8], {o0,o1,o2,o3};\n\t" \
        "}" \
        :: "f"(f0),"f"(f1),"f"(f2),"f"(f3), \
           "f"(f4),"f"(f5),"f"(f6),"f"(f7), \
           "r"(SADDR) \
        : "memory")

/* GELU + CVT + STS: 8 FP32 acc + 4 BF16x2 bias → GELU → BF16 → STS.128
   Unpacks BF16 bias to FP32 inline (SHL+AND → reinterpret as FP32). */
#define GELU_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7, b0,b1,b2,b3, SADDR) \
    CVT_STS_V4( \
        gelu_approx(a0, __uint_as_float((b0) << 16)),  \
        gelu_approx(a1, __uint_as_float((b0) & 0xFFFF0000u)), \
        gelu_approx(a2, __uint_as_float((b1) << 16)),  \
        gelu_approx(a3, __uint_as_float((b1) & 0xFFFF0000u)), \
        gelu_approx(a4, __uint_as_float((b2) << 16)),  \
        gelu_approx(a5, __uint_as_float((b2) & 0xFFFF0000u)), \
        gelu_approx(a6, __uint_as_float((b3) << 16)),  \
        gelu_approx(a7, __uint_as_float((b3) & 0xFFFF0000u)), \
        SADDR)

/* GEMM_ONLY: CVT + STS, no bias, no GELU */
#define GEMM_CVT_STS(f0,f1,f2,f3,f4,f5,f6,f7, SADDR) \
    CVT_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, SADDR)

/* TMA store + wait helpers */
#define EPI_STORE(STAGE, NC, PN, PM) do { \
    if (lane == 0) { \
        const uint32_t s_ = smem_to_uint(smem + OFF_STAGING \
            + (STAGE) * EPI_STAGE_BYTES + row_group * STAGING_REGION_BYTES); \
        asm volatile( \
            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group" \
            " [%0, {%1, %2}], [%3];" \
            :: "l"(&tma_c), "r"((PN) + (NC)), "r"((PM) + row_group * 32), \
               "r"(s_) : "memory"); \
        asm volatile("cp.async.bulk.commit_group;" ::: "memory"); \
    } \
} while(0)

#define EPI_WAIT(LAST) do { \
    if (lane == 0) { \
        if (LAST) \
            asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); \
        else \
            asm volatile("cp.async.bulk.wait_group 1;" ::: "memory"); \
    } \
    __syncwarp(); \
    asm volatile(BAR_EPI_SYNC ::: "memory"); \
} while(0)

/* ── K-iteration macro (accumulating, ki >= 1) ── */
#define K_ITER_ACCUM(S) do { \
    mbar_wait(tma_mbar[S], tma_phase[S]); \
    tma_phase[S] ^= 1; \
    asm volatile("tcgen05.fence::after_thread_sync;"); \
    { \
        uint64_t desc_a = desc_a_base[S], desc_b = desc_b_base[S]; \
        asm volatile( \
            "{\n\t" \
            ".reg .pred p;\n\t" \
            "setp.ne.b32 p, 1, 0;\n\t" \
            "tcgen05.mma.cta_group::2.kind::f8f6f4 " \
            "[%0], %1, %2, %3, {%4,%5,%6,%7, %8,%9,%10,%11}, p;\n\t" \
            "}" \
            : \
            : "r"(buf * TN), "l"(desc_a), "l"(desc_b), "r"(IDESC), \
              "r"(0),"r"(0),"r"(0),"r"(0), \
              "r"(0),"r"(0),"r"(0),"r"(0)); \
        for (int sub = 1; sub < MMA_PER_KI; sub++) { \
            desc_a += 2; desc_b += 2; \
            asm volatile( \
                "{\n\t" \
                ".reg .pred p;\n\t" \
                "setp.ne.b32 p, 1, 0;\n\t" \
                "tcgen05.mma.cta_group::2.kind::f8f6f4 " \
                "[%0], %1, %2, %3, {%4,%5,%6,%7, %8,%9,%10,%11}, p;\n\t" \
                "}" \
                : \
                : "r"(buf * TN), "l"(desc_a), "l"(desc_b), "r"(IDESC), \
                  "r"(0),"r"(0),"r"(0),"r"(0), \
                  "r"(0),"r"(0),"r"(0),"r"(0)); \
        } \
    } \
    tcgen05_commit_mcast(mma_mbar[S], 0x3); \
} while(0)


#if TILE_DISPATCH == 4
__device__ int g_tile_ctr;
#endif

/* ════════════════════════════════════════════════════════════════
   KERNEL
   ════════════════════════════════════════════════════════════════ */

__global__ void __launch_bounds__(THREADS, 1)
__cluster_dims__(2, 1, 1)
fc1_w3_kernel(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    const __grid_constant__ CUtensorMap tma_c,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ C
) {
    extern __shared__ __align__(128) char smem[];
    const int sm_id = blockIdx.x;
    const int cta_rank = sm_id & 1;
    const int tid   = threadIdx.x;
    const int warp  = tid / 32;
    const int lane  = tid % 32;

#if TILE_DISPATCH == 0
    const int cluster_id = sm_id / 2;
    const int num_clusters = SM_COUNT / 2;
#endif

    /* ── Mbarrier init ── */
    if (tid == 0) {
        for (int s = 0; s < N_STAGES; s++) {
            mbar_init(smem_to_uint(smem + OFF_TMA_MBAR + s * 8), 2);
            mbar_init(smem_to_uint(smem + OFF_MMA_MBAR + s * 8), 1);
        }
        for (int i = 0; i < 2; i++) {
            mbar_init(smem_to_uint(smem + OFF_MAINLOOP_MBAR + i * 8), 1);
            /* epilogue_mbar: all epilogue warps × 2 CTAs × 32 threads */
            mbar_init(smem_to_uint(smem + OFF_EPILOGUE_MBAR + i * 8),
                      NUM_EPI_WARPS * 2 * 32);
        }

#if TILE_DISPATCH == 4
        for (int i = 0; i < 2; i++) {
            mbar_init(smem_to_uint(smem + OFF_SCHED_PROD_MBAR + i * 8), 32);
            mbar_init(smem_to_uint(smem + OFF_SCHED_CONS_MBAR + i * 8), 32);
            mbar_init(smem_to_uint(smem + OFF_TILE_READY_MBAR + i * 8), 32);
        }
        asm volatile("st.shared.b32 [%0], 0;" :: "r"(smem_to_uint(smem + OFF_SCHED_EPOCH)));
        asm volatile("st.shared.b32 [%0], 0;" :: "r"(smem_to_uint(smem + OFF_SCHED_EPOCH + 4)));
#endif

        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    /* ── TMEM alloc ── */
    if (warp == 1) {
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(smem_to_uint(smem + OFF_TMEM)), "r"(TMEM_COLS));
    }

    /* ── Common state ── */
    uint32_t tma_mbar[N_STAGES], mma_mbar[N_STAGES];
    uint32_t smem_a[N_STAGES], smem_b[N_STAGES];
    for (int s = 0; s < N_STAGES; s++) {
        tma_mbar[s] = smem_to_uint(smem + OFF_TMA_MBAR + s * 8);
        mma_mbar[s] = smem_to_uint(smem + OFF_MMA_MBAR + s * 8);
        smem_a[s]   = smem_to_uint(smem + s * STAGE_BYTES);
        smem_b[s]   = smem_to_uint(smem + s * STAGE_BYTES + 16384);
    }
    const uint32_t mainloop_mbar_addr = smem_to_uint(smem + OFF_MAINLOOP_MBAR);
    const uint32_t epilogue_mbar_addr = smem_to_uint(smem + OFF_EPILOGUE_MBAR);
    /* Clear bit 24: both CTAs arrive on CTA 0's mbar */
    const uint32_t epi_mbar_masked = epilogue_mbar_addr & 0xFEFFFFFF;

#if TILE_DISPATCH == 4
    const uint32_t sched_prod_mbar = smem_to_uint(smem + OFF_SCHED_PROD_MBAR);
    const uint32_t sched_cons_mbar = smem_to_uint(smem + OFF_SCHED_CONS_MBAR);
    const uint32_t tile_ready_mbar = smem_to_uint(smem + OFF_TILE_READY_MBAR);
    const uint32_t bcast_addr      = smem_to_uint(smem + OFF_BCAST_TILE);
    const uint32_t fifo_addr       = smem_to_uint(smem + OFF_SCHED_FIFO);
    const uint32_t cta0_epoch = smem_to_uint(smem + OFF_SCHED_EPOCH) & 0xFEFFFFFFU;
    const uint32_t cta0_fifo  = smem_to_uint(smem + OFF_SCHED_FIFO) & 0xFEFFFFFFU;
    int sched_prod_phase[2] = {0, 0};
    int sched_cons_phase[2] = {0, 0};
    int tile_ready_phase[2] = {0, 0};
#endif

#if TILE_DISPATCH == 4
    int _iter = 0;
    int _prev_tile = -1;
#else
    /* Group-3: each cluster handles a fixed N-tile, strides through M-rows */
    const int tn_fixed = cluster_id % TILES_N;
    const int m_rank = cluster_id / TILES_N;
    const int my_m_stride = (num_clusters - tn_fixed + TILES_N - 1) / TILES_N;
    const int tile_count  = (TILES_M - m_rank + my_m_stride - 1) / my_m_stride;
#endif

    int tma_phase[N_STAGES] = {0};
    int mma_phase[N_STAGES] = {0};

    uint64_t desc_a_base[N_STAGES], desc_b_base[N_STAGES];
    for (int s = 0; s < N_STAGES; s++) {
        desc_a_base[s] = make_smem_desc(smem_a[s]);
        desc_b_base[s] = make_smem_desc(smem_b[s]);
    }

#ifdef NO_PREFILL
    int epi_phase[2] = {1, 1};
#endif
    int ml_phase[2]  = {0, 1};

    /* ── Load bias into SMEM once ── */
    {
        const uint32_t bias_saddr = smem_to_uint(smem + OFF_BIAS_SMEM);
        for (int i = tid; i < N_DIM / 2; i += THREADS) {
            uint32_t val;
            asm volatile("ld.global.b32 %0, [%1];" : "=r"(val) : "l"(bias + i * 2));
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(bias_saddr + i * 4), "r"(val));
        }
    }
    __syncthreads();

#if TILE_DISPATCH == 4
    /* ════════════════════════════════════════════
       SCHEDULER WARP (TD=4): dispatch via atomicAdd, pipe to W0
       ════════════════════════════════════════════ */
    if (warp == SCHED_WARP) {
        const uint32_t epoch_addr = smem_to_uint(smem + OFF_SCHED_EPOCH);
        int _s_iter = 0;
        int _s_buf = 0;
        while (true) {
            if (_s_iter >= 2) {
                mbar_wait(sched_cons_mbar + _s_buf * 8, sched_cons_phase[_s_buf]);
                sched_cons_phase[_s_buf] ^= 1;
            }

            int tile_idx;
            if (cta_rank == 0) {
                if (lane == 0) {
                    asm volatile("atom.global.relaxed.gpu.add.s32 %0, [%1], 1;"
                        : "=r"(tile_idx) : "l"(&g_tile_ctr));
                    asm volatile("st.shared.b32 [%0], %1;"
                        :: "r"(fifo_addr + _s_buf * 4), "r"(tile_idx));
                    asm volatile("fence.acq_rel.cluster;");
                    asm volatile("st.shared.b32 [%0], %1;"
                        :: "r"(epoch_addr + _s_buf * 4), "r"(_s_iter + 1));
                }
            } else {
                if (lane == 0) {
                    int epoch;
                    do {
                        asm volatile("ld.acquire.cluster.shared::cluster.b32 %0, [%1];"
                            : "=r"(epoch) : "r"(cta0_epoch + _s_buf * 4));
                    } while (epoch != _s_iter + 1);
                    asm volatile("ld.shared::cluster.b32 %0, [%1];"
                        : "=r"(tile_idx) : "r"(cta0_fifo + _s_buf * 4));
                    asm volatile("st.shared.b32 [%0], %1;"
                        :: "r"(fifo_addr + _s_buf * 4), "r"(tile_idx));
                }
            }

            mbar_arrive(sched_prod_mbar + _s_buf * 8);

            asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
                : "=r"(tile_idx) : "r"(tile_idx));
            if (tile_idx >= TOTAL_TILES) break;

            _s_buf ^= 1;
            _s_iter++;
        }
        return;
    }
#endif

    /* ════════════════════════════════════════════
       MAIN TILE LOOP
       ════════════════════════════════════════════ */

#if TILE_DISPATCH == 4
    int _pf_tile = TOTAL_TILES;
    int _pf_slot = 1;
    if (warp == 0) {
        mbar_wait(sched_prod_mbar, sched_prod_phase[0]);
        sched_prod_phase[0] ^= 1;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(_pf_tile) : "r"(fifo_addr));
        mbar_arrive(sched_cons_mbar);
    }
    while (true) {
        int tile_idx;
        {
            const int _buf = _iter & 1;
            if (warp == 0) {
                tile_idx = _pf_tile;
                asm volatile("st.shared.b32 [%0], %1;"
                    :: "r"(bcast_addr + _buf * 4), "r"(tile_idx));
                mbar_arrive(tile_ready_mbar + _buf * 8);
            } else {
                mbar_wait(tile_ready_mbar + _buf * 8, tile_ready_phase[_buf]);
                tile_ready_phase[_buf] ^= 1;
                asm volatile("ld.shared.b32 %0, [%1];"
                    : "=r"(tile_idx) : "r"(bcast_addr + _buf * 4));
            }
        }
        if (tile_idx >= TOTAL_TILES) break;
        const int buf = _iter & 1;
#else
    for (int _ti = 0; _ti < tile_count; _ti++) {
        const int _tm = m_rank + _ti * my_m_stride;
        if (_tm >= TILES_M) break;
        const int tile_idx = _tm * TILES_N + tn_fixed;
        const int buf = _ti & 1;
#endif
        int tm = tile_idx / TILES_N;
        int tn = tile_idx % TILES_N;
        if (SNAKE_ORDER && (tm & 1)) tn = TILES_N - 1 - tn;
        const int m_start = tm * TM * 2 + cta_rank * TM;
        const int n_start = tn * TN;
#if TILE_DISPATCH == 4
        const bool has_prev = (_iter > 0);
#else
        const bool has_prev = (_ti > 0);
#endif

        if (warp == 0) {
            /* ── W0: TMA A/B loads ── */
            const uint32_t smem_base = warp_uniform(smem_to_uint(smem));
            for (int ki = 0; ki < K_ITERS; ki++) {
                const int s = ki % N_STAGES;
                const int k_start = ki * TK;
                const uint32_t mma_mbar_s = smem_base + OFF_MMA_MBAR + s * 8;
                const uint32_t tma_mbar_s = (smem_base + OFF_TMA_MBAR + s * 8) & 0xFEFFFFFF;

                if (has_prev || ki >= N_STAGES) {
                    mbar_wait(mma_mbar_s, mma_phase[s]);
                    mma_phase[s] ^= 1;
                }

                if (lane == 0) {
                    const uint32_t a_dst = smem_base + s * STAGE_BYTES;
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
                        ".mbarrier::complete_tx::bytes.cta_group::2"
                        " [%0], [%1, {%2, %3}], [%4];\n\t"
                        "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
                        ".mbarrier::complete_tx::bytes.cta_group::2"
                        " [%5], [%6, {%2, %7}], [%4];\n\t"
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%4], %8;"
                        :: "r"(a_dst), "l"(&tma_a), "r"(k_start), "r"(m_start),
                           "r"(tma_mbar_s), "r"(a_dst + 16384), "l"(&tma_b),
                           "r"(n_start + cta_rank * (TN/2)), "r"(TMA_BYTES)
                        : "memory");
                }
            }

#if TILE_DISPATCH == 4
            /* Prefetch next tile from scheduler FIFO */
            mbar_wait(sched_prod_mbar + _pf_slot * 8, sched_prod_phase[_pf_slot]);
            sched_prod_phase[_pf_slot] ^= 1;
            asm volatile("ld.shared.b32 %0, [%1];"
                : "=r"(_pf_tile) : "r"(fifo_addr + _pf_slot * 4));
            mbar_arrive(sched_cons_mbar + _pf_slot * 8);
            _pf_slot ^= 1;
#endif

        } else if (warp == 1) {
            /* ── W1: MMA ── */
            if (lane == 0 && cta_rank == 0) {
#ifdef NO_PREFILL
                mbar_wait(epilogue_mbar_addr + buf * 8, epi_phase[buf]);
                epi_phase[buf] ^= 1;
#endif

                mbar_wait(tma_mbar[0], tma_phase[0]);
                tma_phase[0] ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");

                /* First K-iteration: initialize accumulator (p=0) */
                {
                    uint64_t desc_a = desc_a_base[0], desc_b = desc_b_base[0];
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "setp.ne.b32 p, 0, 0;\n\t"
                        "tcgen05.mma.cta_group::2.kind::f8f6f4 "
                        "[%0], %1, %2, %3, {%4,%5,%6,%7, %8,%9,%10,%11}, p;\n\t"
                        "}"
                        :
                        : "r"(buf * TN), "l"(desc_a), "l"(desc_b), "r"(IDESC),
                          "r"(0),"r"(0),"r"(0),"r"(0),
                          "r"(0),"r"(0),"r"(0),"r"(0));
                    for (int sub = 1; sub < MMA_PER_KI; sub++) {
                        desc_a += 2; desc_b += 2;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p;\n\t"
                            "setp.ne.b32 p, 1, 0;\n\t"
                            "tcgen05.mma.cta_group::2.kind::f8f6f4 "
                            "[%0], %1, %2, %3, {%4,%5,%6,%7, %8,%9,%10,%11}, p;\n\t"
                            "}"
                            :
                            : "r"(buf * TN), "l"(desc_a), "l"(desc_b), "r"(IDESC),
                              "r"(0),"r"(0),"r"(0),"r"(0),
                              "r"(0),"r"(0),"r"(0),"r"(0));
                    }
                }
                tcgen05_commit_mcast(mma_mbar[0], 0x3);

                PRAGMA_UNROLL(K_LOOP_UNROLL)
                for (int ki = 1; ki < K_ITERS; ki++) {
                    K_ITER_ACCUM(ki % N_STAGES);
                }

                /* Signal epilogue: MMA done for this tile */
                tcgen05_commit_mcast(mainloop_mbar_addr + buf * 8, 0x3);
            }

        } else {
            /* ── W2+: Epilogue ── */
            const int prev_buf = buf ^ 1;
            mbar_wait(mainloop_mbar_addr + prev_buf * 8, ml_phase[prev_buf]);
            ml_phase[prev_buf] ^= 1;

#ifdef STRIP_EPILOGUE
            if (has_prev)
                mbar_arrive(epi_mbar_masked + prev_buf * 8);
#elif defined(GEMM_ONLY)
            /* GEMM-only epilogue: TMEM→CVT→STS→TMA store, no bias, no GELU */
            {
            const int ew = warp - 2;
            const uint32_t xor_val = (lane & 7) << 4;
            const uint32_t sw0 = 0 ^ xor_val, sw1 = 16 ^ xor_val;
            const uint32_t sw2 = 32 ^ xor_val, sw3 = 48 ^ xor_val;
            const uint32_t sw4 = 64 ^ xor_val, sw5 = 80 ^ xor_val;
            const uint32_t sw6 = 96 ^ xor_val, sw7 = 112 ^ xor_val;

            if (has_prev) {
#if TILE_DISPATCH == 4
                const int prev_idx = _prev_tile;
#else
                const int prev_idx = (m_rank + (_ti - 1) * my_m_stride) * TILES_N + tn_fixed;
#endif
                int ptm = prev_idx / TILES_N;
                int ptn = prev_idx % TILES_N;
                if (SNAKE_ORDER && (ptm & 1)) ptn = TILES_N - 1 - ptn;
                const int prev_m = ptm * TM * 2 + cta_rank * TM;
                const int prev_n = ptn * TN;

                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

                for (int si = 0; si < NUM_EPI_SUBITERS; si++) {
                    const int stage = si % NUM_EPI_STAGES;
                    const int nc_base = si * 64;

#if GROUPS_PER_WARP > 1
                    #pragma unroll
                    for (int _rg = 0; _rg < GROUPS_PER_WARP; _rg++) {
                    const int row_group = ew * GROUPS_PER_WARP + _rg;
#else
                    { const int row_group = ew;
#endif
                    const int taddr_base = prev_buf * TN + ((cta_rank * 128 + row_group * 32) << 16);
                    float a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15;
                    float a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31;

                    const uint32_t stage_base = smem_to_uint(smem + OFF_STAGING
                        + stage * EPI_STAGE_BYTES
                        + row_group * STAGING_REGION_BYTES
                        + lane * 128);

                    for (int _ci = 0; _ci < 2; _ci++) {
                        const int chunk = _ci;
                        const int nc = nc_base + chunk * 32;

                        TMEM_LOAD_X32(a0,a1,a2,a3,a4,a5,a6,a7,
                                      a8,a9,a10,a11,a12,a13,a14,a15,
                                      a16,a17,a18,a19,a20,a21,a22,a23,
                                      a24,a25,a26,a27,a28,a29,a30,a31,
                                      taddr_base + nc);

                        const uint32_t rsw0 = chunk ? sw4 : sw0;
                        const uint32_t rsw1 = chunk ? sw5 : sw1;
                        const uint32_t rsw2 = chunk ? sw6 : sw2;
                        const uint32_t rsw3 = chunk ? sw7 : sw3;

                        TMEM_WAIT();

                        GEMM_CVT_STS(a0,a1,a2,a3,a4,a5,a6,a7, stage_base + rsw0);
                        GEMM_CVT_STS(a8,a9,a10,a11,a12,a13,a14,a15, stage_base + rsw1);
                        GEMM_CVT_STS(a16,a17,a18,a19,a20,a21,a22,a23, stage_base + rsw2);
                        GEMM_CVT_STS(a24,a25,a26,a27,a28,a29,a30,a31, stage_base + rsw3);
                    }
                    } /* close row_group */

                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(BAR_EPI_SYNC ::: "memory");

#if GROUPS_PER_WARP > 1
                    if (lane == 0) {
                        #pragma unroll
                        for (int _rg = 0; _rg < GROUPS_PER_WARP; _rg++) {
                            const int rg = ew * GROUPS_PER_WARP + _rg;
                            const uint32_t s_ = smem_to_uint(smem + OFF_STAGING
                                + stage * EPI_STAGE_BYTES + rg * STAGING_REGION_BYTES);
                            asm volatile(
                                "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
                                " [%0, {%1, %2}], [%3];"
                                :: "l"(&tma_c), "r"(prev_n + nc_base), "r"(prev_m + rg * 32),
                                   "r"(s_) : "memory");
                        }
                        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                    }
#else
                    { const int row_group = ew;
                    EPI_STORE(stage, nc_base, prev_n, prev_m); }
#endif
                    EPI_WAIT(si == NUM_EPI_SUBITERS - 1);
                }

                mbar_arrive(epi_mbar_masked + prev_buf * 8);
            }
            }
#else
            /* ── W2-W5: GELU epilogue — TMEM→bias+GELU→CVT→STS→TMA store ── */
            const int ew = warp - 2;
            const uint32_t bias_saddr = smem_to_uint(smem + OFF_BIAS_SMEM);

            const uint32_t xor_val = (lane & 7) << 4;
            const uint32_t sw0 = 0 ^ xor_val, sw1 = 16 ^ xor_val;
            const uint32_t sw2 = 32 ^ xor_val, sw3 = 48 ^ xor_val;
            const uint32_t sw4 = 64 ^ xor_val, sw5 = 80 ^ xor_val;
            const uint32_t sw6 = 96 ^ xor_val, sw7 = 112 ^ xor_val;

            if (has_prev) {
#if TILE_DISPATCH == 4
                const int prev_idx = _prev_tile;
#else
                const int prev_idx = (m_rank + (_ti - 1) * my_m_stride) * TILES_N + tn_fixed;
#endif
                int ptm = prev_idx / TILES_N;
                int ptn = prev_idx % TILES_N;
                if (SNAKE_ORDER && (ptm & 1)) ptn = TILES_N - 1 - ptn;
                const int prev_m = ptm * TM * 2 + cta_rank * TM;
                const int prev_n = ptn * TN;

                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

                for (int si = 0; si < NUM_EPI_SUBITERS; si++) {
                    const int stage = si % NUM_EPI_STAGES;
                    const int nc_base = si * 64;

#if GROUPS_PER_WARP > 1
                    #pragma unroll
                    for (int _rg = 0; _rg < GROUPS_PER_WARP; _rg++) {
                    const int row_group = ew * GROUPS_PER_WARP + _rg;
#else
                    { const int row_group = ew;
#endif
                    const int taddr_base = prev_buf * TN + ((cta_rank * 128 + row_group * 32) << 16);
                    float a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15;
                    float a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31;

                    const uint32_t stage_base = smem_to_uint(smem + OFF_STAGING
                        + stage * EPI_STAGE_BYTES
                        + row_group * STAGING_REGION_BYTES
                        + lane * 128);

                    for (int _ci = 0; _ci < 2; _ci++) {
                        const int chunk = _ci;
                        const int nc = nc_base + chunk * 32;

                        TMEM_LOAD_X32(a0,a1,a2,a3,a4,a5,a6,a7,
                                      a8,a9,a10,a11,a12,a13,a14,a15,
                                      a16,a17,a18,a19,a20,a21,a22,a23,
                                      a24,a25,a26,a27,a28,a29,a30,a31,
                                      taddr_base + nc);

                        /* LDS bias from SMEM (linear, not swizzled) */
                        const uint32_t bs = bias_saddr + (prev_n + nc) * 2;
                        uint4 bv0, bv1, bv2, bv3;
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(bv0.x),"=r"(bv0.y),"=r"(bv0.z),"=r"(bv0.w) : "r"(bs));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(bv1.x),"=r"(bv1.y),"=r"(bv1.z),"=r"(bv1.w) : "r"(bs + 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(bv2.x),"=r"(bv2.y),"=r"(bv2.z),"=r"(bv2.w) : "r"(bs + 32));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(bv3.x),"=r"(bv3.y),"=r"(bv3.z),"=r"(bv3.w) : "r"(bs + 48));

                        const uint32_t rsw0 = chunk ? sw4 : sw0;
                        const uint32_t rsw1 = chunk ? sw5 : sw1;
                        const uint32_t rsw2 = chunk ? sw6 : sw2;
                        const uint32_t rsw3 = chunk ? sw7 : sw3;

                        TMEM_WAIT();

                        GELU_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7,
                            bv0.x,bv0.y,bv0.z,bv0.w, stage_base + rsw0);
                        GELU_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15,
                            bv1.x,bv1.y,bv1.z,bv1.w, stage_base + rsw1);
                        GELU_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23,
                            bv2.x,bv2.y,bv2.z,bv2.w, stage_base + rsw2);
                        GELU_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31,
                            bv3.x,bv3.y,bv3.z,bv3.w, stage_base + rsw3);
                    }
                    } /* close row_group */

                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(BAR_EPI_SYNC ::: "memory");

#if GROUPS_PER_WARP > 1
                    if (lane == 0) {
                        #pragma unroll
                        for (int _rg = 0; _rg < GROUPS_PER_WARP; _rg++) {
                            const int rg = ew * GROUPS_PER_WARP + _rg;
                            const uint32_t s_ = smem_to_uint(smem + OFF_STAGING
                                + stage * EPI_STAGE_BYTES + rg * STAGING_REGION_BYTES);
                            asm volatile(
                                "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
                                " [%0, {%1, %2}], [%3];"
                                :: "l"(&tma_c), "r"(prev_n + nc_base), "r"(prev_m + rg * 32),
                                   "r"(s_) : "memory");
                        }
                        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                    }
#else
                    { const int row_group = ew;
                    EPI_STORE(stage, nc_base, prev_n, prev_m); }
#endif
                    EPI_WAIT(si == NUM_EPI_SUBITERS - 1);
                }

                mbar_arrive(epi_mbar_masked + prev_buf * 8);
            }
#endif /* STRIP_EPILOGUE / GEMM_ONLY */
        }
#if TILE_DISPATCH == 4
        _prev_tile = tile_idx;
        _iter++;
#endif
    }

    /* ══════════════════════════════════════════════
       DRAIN: last tile epilogue
       ══════════════════════════════════════════════ */
    {
#if TILE_DISPATCH == 4
        const int last_idx = _prev_tile;
        const int last_buf = (_iter - 1) & 1;
#else
        const int last_idx = (m_rank + (tile_count - 1) * my_m_stride) * TILES_N + tn_fixed;
        const int last_buf = (tile_count - 1) & 1;
#endif
        int ltm = last_idx / TILES_N;
        int ltn = last_idx % TILES_N;
        if (SNAKE_ORDER && (ltm & 1)) ltn = TILES_N - 1 - ltn;
        const int last_m = ltm * TM * 2 + cta_rank * TM;
        const int last_n = ltn * TN;

        if (warp == 0 || warp == 1) {
            /* W0/W1: nothing to do for drain */
        } else {
#ifdef STRIP_EPILOGUE
            mbar_wait(mainloop_mbar_addr + last_buf * 8, ml_phase[last_buf]);
            ml_phase[last_buf] ^= 1;
            mbar_arrive(epi_mbar_masked + last_buf * 8);
#elif defined(GEMM_ONLY)
            /* GEMM-only drain */
            {
            const int ew = warp - 2;
            const uint32_t xor_val = (lane & 7) << 4;
            const uint32_t sw0 = 0 ^ xor_val, sw1 = 16 ^ xor_val;
            const uint32_t sw2 = 32 ^ xor_val, sw3 = 48 ^ xor_val;
            const uint32_t sw4 = 64 ^ xor_val, sw5 = 80 ^ xor_val;
            const uint32_t sw6 = 96 ^ xor_val, sw7 = 112 ^ xor_val;

            mbar_wait(mainloop_mbar_addr + last_buf * 8, ml_phase[last_buf]);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            ml_phase[last_buf] ^= 1;

            for (int si = 0; si < NUM_EPI_SUBITERS; si++) {
                const int stage = si % NUM_EPI_STAGES;
                const int nc_base = si * 64;

#if GROUPS_PER_WARP > 1
                #pragma unroll
                for (int _rg = 0; _rg < GROUPS_PER_WARP; _rg++) {
                const int row_group = ew * GROUPS_PER_WARP + _rg;
#else
                { const int row_group = ew;
#endif
                const int taddr_base = last_buf * TN + ((cta_rank * 128 + row_group * 32) << 16);
                float a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15;
                float a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31;

                const uint32_t stage_base = smem_to_uint(smem + OFF_STAGING
                    + stage * EPI_STAGE_BYTES
                    + row_group * STAGING_REGION_BYTES
                    + lane * 128);

                for (int _ci = 0; _ci < 2; _ci++) {
                    const int chunk = _ci;
                    const int nc = nc_base + chunk * 32;

                    TMEM_LOAD_X32(a0,a1,a2,a3,a4,a5,a6,a7,
                                  a8,a9,a10,a11,a12,a13,a14,a15,
                                  a16,a17,a18,a19,a20,a21,a22,a23,
                                  a24,a25,a26,a27,a28,a29,a30,a31,
                                  taddr_base + nc);

                    const uint32_t rsw0 = chunk ? sw4 : sw0;
                    const uint32_t rsw1 = chunk ? sw5 : sw1;
                    const uint32_t rsw2 = chunk ? sw6 : sw2;
                    const uint32_t rsw3 = chunk ? sw7 : sw3;

                    TMEM_WAIT();

                    GEMM_CVT_STS(a0,a1,a2,a3,a4,a5,a6,a7, stage_base + rsw0);
                    GEMM_CVT_STS(a8,a9,a10,a11,a12,a13,a14,a15, stage_base + rsw1);
                    GEMM_CVT_STS(a16,a17,a18,a19,a20,a21,a22,a23, stage_base + rsw2);
                    GEMM_CVT_STS(a24,a25,a26,a27,a28,a29,a30,a31, stage_base + rsw3);
                }
                } /* close row_group */

                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(BAR_EPI_SYNC ::: "memory");

#if GROUPS_PER_WARP > 1
                if (lane == 0) {
                    #pragma unroll
                    for (int _rg = 0; _rg < GROUPS_PER_WARP; _rg++) {
                        const int rg = ew * GROUPS_PER_WARP + _rg;
                        const uint32_t s_ = smem_to_uint(smem + OFF_STAGING
                            + stage * EPI_STAGE_BYTES + rg * STAGING_REGION_BYTES);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
                            " [%0, {%1, %2}], [%3];"
                            :: "l"(&tma_c), "r"(last_n + nc_base), "r"(last_m + rg * 32),
                               "r"(s_) : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                }
#else
                { const int row_group = ew;
                EPI_STORE(stage, nc_base, last_n, last_m); }
#endif
                EPI_WAIT(si == NUM_EPI_SUBITERS - 1);
            }

            mbar_arrive(epi_mbar_masked + last_buf * 8);
            }
#else
            /* GELU drain — last tile */
            const int ew = warp - 2;
            const uint32_t bias_saddr = smem_to_uint(smem + OFF_BIAS_SMEM);
            const uint32_t xor_val = (lane & 7) << 4;
            const uint32_t sw0 = 0 ^ xor_val, sw1 = 16 ^ xor_val;
            const uint32_t sw2 = 32 ^ xor_val, sw3 = 48 ^ xor_val;
            const uint32_t sw4 = 64 ^ xor_val, sw5 = 80 ^ xor_val;
            const uint32_t sw6 = 96 ^ xor_val, sw7 = 112 ^ xor_val;

            mbar_wait(mainloop_mbar_addr + last_buf * 8, ml_phase[last_buf]);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            ml_phase[last_buf] ^= 1;

            for (int si = 0; si < NUM_EPI_SUBITERS; si++) {
                const int stage = si % NUM_EPI_STAGES;
                const int nc_base = si * 64;

#if GROUPS_PER_WARP > 1
                #pragma unroll
                for (int _rg = 0; _rg < GROUPS_PER_WARP; _rg++) {
                const int row_group = ew * GROUPS_PER_WARP + _rg;
#else
                { const int row_group = ew;
#endif
                const int taddr_base = last_buf * TN + ((cta_rank * 128 + row_group * 32) << 16);
                float a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15;
                float a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31;

                const uint32_t stage_base = smem_to_uint(smem + OFF_STAGING
                    + stage * EPI_STAGE_BYTES
                    + row_group * STAGING_REGION_BYTES
                    + lane * 128);

                for (int _ci = 0; _ci < 2; _ci++) {
                    const int chunk = _ci;
                    const int nc = nc_base + chunk * 32;

                    TMEM_LOAD_X32(a0,a1,a2,a3,a4,a5,a6,a7,
                                  a8,a9,a10,a11,a12,a13,a14,a15,
                                  a16,a17,a18,a19,a20,a21,a22,a23,
                                  a24,a25,a26,a27,a28,a29,a30,a31,
                                  taddr_base + nc);

                    const uint32_t bs = bias_saddr + (last_n + nc) * 2;
                    uint4 bv0, bv1, bv2, bv3;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(bv0.x),"=r"(bv0.y),"=r"(bv0.z),"=r"(bv0.w) : "r"(bs));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(bv1.x),"=r"(bv1.y),"=r"(bv1.z),"=r"(bv1.w) : "r"(bs + 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(bv2.x),"=r"(bv2.y),"=r"(bv2.z),"=r"(bv2.w) : "r"(bs + 32));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(bv3.x),"=r"(bv3.y),"=r"(bv3.z),"=r"(bv3.w) : "r"(bs + 48));

                    const uint32_t rsw0 = chunk ? sw4 : sw0;
                    const uint32_t rsw1 = chunk ? sw5 : sw1;
                    const uint32_t rsw2 = chunk ? sw6 : sw2;
                    const uint32_t rsw3 = chunk ? sw7 : sw3;

                    TMEM_WAIT();

                    GELU_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7,
                        bv0.x,bv0.y,bv0.z,bv0.w, stage_base + rsw0);
                    GELU_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15,
                        bv1.x,bv1.y,bv1.z,bv1.w, stage_base + rsw1);
                    GELU_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23,
                        bv2.x,bv2.y,bv2.z,bv2.w, stage_base + rsw2);
                    GELU_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31,
                        bv3.x,bv3.y,bv3.z,bv3.w, stage_base + rsw3);
                }
                } /* close row_group */

                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(BAR_EPI_SYNC ::: "memory");

#if GROUPS_PER_WARP > 1
                if (lane == 0) {
                    #pragma unroll
                    for (int _rg = 0; _rg < GROUPS_PER_WARP; _rg++) {
                        const int rg = ew * GROUPS_PER_WARP + _rg;
                        const uint32_t s_ = smem_to_uint(smem + OFF_STAGING
                            + stage * EPI_STAGE_BYTES + rg * STAGING_REGION_BYTES);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
                            " [%0, {%1, %2}], [%3];"
                            :: "l"(&tma_c), "r"(last_n + nc_base), "r"(last_m + rg * 32),
                               "r"(s_) : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                }
#else
                { const int row_group = ew;
                EPI_STORE(stage, nc_base, last_n, last_m); }
#endif
                EPI_WAIT(si == NUM_EPI_SUBITERS - 1);
            }

            mbar_arrive(epi_mbar_masked + last_buf * 8);
#endif /* STRIP_EPILOGUE / GEMM_ONLY drain */
        }
    }

    /* ── TMEM dealloc ── */
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
            :: "r"(0), "r"(TMEM_COLS));
    }
}


/* ════════════════════════════════════════════════════════════════
   HOST
   ════════════════════════════════════════════════════════════════ */

int main() {
    setbuf(stdout, NULL);
#ifdef GEMM_ONLY
    printf("FC1 W3 kernel — %d warps, GEMM_ONLY (D=BF16(A*B), no bias/GELU)\n", NUM_WARPS);
#else
    printf("FC1 W3 kernel — %d warps, GELU epilogue\n", NUM_WARPS);
#endif
    printf("  GEMM: [%d,%d] x [%d,%d]^T  %d-stage pipeline  SMEM: %d bytes\n",
           M_TOTAL, K_DIM, N_DIM, K_DIM, N_STAGES, SMEM_BYTES);
    printf("  Tiles: %dM × %dN = %d total  K_ITERS=%d\n",
           TILES_M, TILES_N, TOTAL_TILES, K_ITERS);

    uint8_t *d_A, *d_B;
    __nv_bfloat16 *d_bias, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A,    (size_t)M_TOTAL * K_DIM));
    CUDA_CHECK(cudaMalloc(&d_B,    (size_t)N_DIM   * K_DIM));
    CUDA_CHECK(cudaMalloc(&d_bias, (size_t)N_DIM   * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_C,    (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16)));

    /* A: uniform 0x3C (1.5 in FP8 E4M3) */
    CUDA_CHECK(cudaMemset(d_A, 0x3C, (size_t)M_TOTAL * K_DIM));
    /* B: alternating rows — even=0x3C(1.5), odd=0x38(1.0) */
    {
        uint8_t* h_B = (uint8_t*)malloc((size_t)N_DIM * K_DIM);
        for (int n = 0; n < N_DIM; n++)
            memset(h_B + (size_t)n * K_DIM, (n & 1) ? 0x38 : 0x3C, K_DIM);
        CUDA_CHECK(cudaMemcpy(d_B, h_B, (size_t)N_DIM * K_DIM, cudaMemcpyHostToDevice));
        free(h_B);
    }
    /* Bias: bias[c] = bf16(c + 1) */
    {
        __nv_bfloat16* h_bias = (__nv_bfloat16*)malloc((size_t)N_DIM * sizeof(__nv_bfloat16));
        for (int c = 0; c < N_DIM; c++)
            h_bias[c] = __float2bfloat16((float)(c + 1));
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias, (size_t)N_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        free(h_bias);
    }
    printf("  Alloc + init done\n");

    /* TMA descriptors */
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

    CUDA_CHECK(cudaFuncSetAttribute(fc1_w3_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    printf("  TMA descriptors + func attr done (SMEM=%d B)\n", SMEM_BYTES);

#if TILE_DISPATCH == 4
    int* d_tile_ctr_ptr;
    CUDA_CHECK(cudaGetSymbolAddress((void**)&d_tile_ctr_ptr, g_tile_ctr));
#define LAUNCH_KERNEL() do { \
    cudaMemsetAsync(d_tile_ctr_ptr, 0, sizeof(int)); \
    fc1_w3_kernel<<<SM_COUNT, THREADS, SMEM_BYTES>>>( \
        h_tma_a, h_tma_b, h_tma_c, d_bias, d_C); \
} while(0)
#else
#define LAUNCH_KERNEL() \
    fc1_w3_kernel<<<SM_COUNT, THREADS, SMEM_BYTES>>>( \
        h_tma_a, h_tma_b, h_tma_c, d_bias, d_C)
#endif

    /* Warmup */
    printf("Launching warmup (2 iters)...\n");
    for (int i = 0; i < 2; i++) {
        LAUNCH_KERNEL();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Warmup done.\n");

    /* Timed: 10 iterations */
    printf("Timing: 10 iterations...\n");
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 10; i++) {
        LAUNCH_KERNEL();
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= 10.0f;
    printf("FC1-W3 kernel: %.3f ms  %.2f TFLOPS\n",
           ms, 2.0 * M_TOTAL * N_DIM * K_DIM / ms / 1e9);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    /* Checksum run */
    LAUNCH_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    __nv_bfloat16* h_C = (__nv_bfloat16*)malloc((size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    /* Strided checksum */
    double cksum = 0;
    {
        long long total_elems = (long long)M_TOTAL * N_DIM;
        long long stride = total_elems / 1024;
        for (int i = 0; i < 1024; i++)
            cksum += (double)__bfloat162float(h_C[(long long)i * stride]);
    }

    /* Spot-check validation */
    int errors = 0;
    for (int spot = 0; spot < 32; spot++) {
        long long row = (long long)spot * M_TOTAL / 32;
        int col = (spot * 47) % N_DIM;
        float b_val = (col & 1) ? 1.0f : 1.5f;
        float gemm = (float)K_DIM * 1.5f * b_val;
        float bias_f = __bfloat162float(__float2bfloat16((float)(col + 1)));
#ifdef GEMM_ONLY
        __nv_bfloat16 expected = __float2bfloat16(gemm);
#else
        float gelu_val = gelu_fwd(gemm + bias_f);
        __nv_bfloat16 expected = __float2bfloat16(gelu_val);
#endif
        __nv_bfloat16 actual = h_C[row * N_DIM + col];
        float ef = __bfloat162float(expected);
        float af = __bfloat162float(actual);
        if (ef != af) {
            if (errors < 5)
                printf("  MISMATCH at (%lld,%d): expected %.1f got %.1f (gemm=%.1f bias=%.1f)\n",
                       row, col, ef, af, gemm, bias_f);
            errors++;
        }
    }
    int valid = (errors == 0) ? 1 : 0;
    printf("Validation: %d/32 spot checks passed%s\n", 32 - errors, valid ? "" : " — FAILED");
    printf("Checksum (1024 strided): %f\n", cksum);
    printf("C[0,0..3] = %.1f %.1f %.1f %.1f\n",
           __bfloat162float(h_C[0]), __bfloat162float(h_C[1]),
           __bfloat162float(h_C[2]), __bfloat162float(h_C[3]));
    printf("@@RESULT ms=%.3f tflops=%.2f checksum=%f valid=%d c0=%.1f\n",
           ms, 2.0 * M_TOTAL * N_DIM * K_DIM / ms / 1e9, cksum, valid,
           __bfloat162float(h_C[0]));

    free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_bias); cudaFree(d_C);
    return 0;
}
