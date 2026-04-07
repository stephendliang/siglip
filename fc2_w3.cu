/*
FC2 W3 kernel — CUTLASS-style shared-SMEM epilogue architecture
Standalone binary: independent of kernel_body.cuh / kernel_common.cuh
Shape: [928256,3072]×[3072,768]^T + bias + residual
7 warps: W0(TMA A/B) | W1(MMA) | W2(EpilogueLoad) | W3-W6(Epilogue)
cta_group::2  __cluster_dims__(2,1,1)

Key architecture: ReuseSmemC + circular producer/consumer epilogue pipeline.
  - 2 shared epilogue stages × 16KB each = 32KB total.
  - Each 16KB stage holds residual OR output sequentially (not both).
  - W2 loads one 64-column residual slice at a time for the PREVIOUS tile,
    pacing against W3-W6 consumer releases.
  - W3-W6 wait/release one stage per sub-iteration instead of consuming
    four fully materialized tile-wide stages.

Barrier protocol (CUTLASS-style):
  - consumed_mbar count=128 (all epilogue threads), not single-thread.
    Structurally guaranteed by BAR.SYNC — eliminates scheduling races.
  - wait_group 1 (allow 1 TMA store in-flight), wait_group 0 on last sub-iter.
  - Deferred consumed: signaled 1 sub-iter after store, when previous store
    is confirmed drained. Both stages signaled on last sub-iter.
  - W2's epi_mbar arrive is after all loads, not before.

Compile-time flags:
  -DFP32_EPILOGUE       FP32 math (FADD, ~0% STS conflict) instead of BF16 (HADD2, 7.5%)
  -DCUTLASS_EPILOGUE    CUTLASS-clone: FP32 res add, BF16 bias, per-group STS, @!PT LDS fences
  -DCUTE_STORE          C++ pointer stores (no asm STS) — tests CuTe store pattern vs asm volatile
  -DCUTLASS_LOOP=N      Loop structure: 1=nounroll si, 2=+nounroll chunk, 3=+C++ FP32 compute
  -DSTRIP_EPILOGUE      Skip epilogue (benchmark GEMM core only, valid=0)
  -DSINGLE_WARP_STORE=1 Only ew==0 issues TMA stores (4 per sub-iter, 1 commit group)
  -DDELAY_TMA_STORE=1   Issue TMA store from sub-iter N at start of sub-iter N+1
  -DNUM_EPI_STAGES=N    Epilogue staging depth (default 2, try 3/4)
  -DNO_PRE_STORE_BAR=1  Remove bar.sync before TMA store (each warp stores own region independently)
  -DNO_POST_STORE_BAR=1 Remove bar.sync after TMA store wait (warps decouple across sub-iters)
  -DCHUNK_REORDER       Odd warps reverse chunk order (1,0 vs 0,1) within each sub-iter
  -DGROUP_REORDER       Rotate STS group order per-warp (4-phase bank conflict elimination)
  -DEPI_REORDER         Shorthand for CHUNK_REORDER + GROUP_REORDER + NO_PRE_STORE_BAR
  -DSELF_LOAD           Per-warp TMA residual load (no W2, no cross-warp sync in epilogue)
  -DSELF_STAGGER=N      With SELF_LOAD: warp ew sleeps ew*N nanoseconds before first sub-iter
                        (0=disabled, ~50=non-overlapping STS, ~200=full isolation)
*/

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

/* ── Hardware ── */
#define SM_COUNT       148

/* ── Problem dims ── */
#define M_TOTAL        928256   /* 4736 images × 196 patches */
#define N_DIM          768
#define K_DIM          3072

/* ── Tile ── */
#define TM             128
#define TN             256
#define TK             128
#define TILES_M        ((M_TOTAL + TM * 2 - 1) / (TM * 2))  /* 3626 */
#define TILES_N        (N_DIM / TN)                           /* 3 */
#define TOTAL_TILES    (TILES_M * TILES_N)                    /* 10878 */
#define K_ITERS        (K_DIM / TK)                           /* 24 */
#define MMA_K          32
#define MMA_PER_KI     (TK / MMA_K)                           /* 4 */
#define SNAKE_ORDER    1

#ifdef M_SNAKE
/* M-directional snake: reverse every other band of ~25 M-rows.
   Band size = num_clusters / TILES_N = wavefront width in M-rows. */
#define M_SNAKE_BAND   ((SM_COUNT / 2 + TILES_N - 1) / TILES_N)
#define M_SNAKE_REMAP(tm) do { \
    int _mb = (tm) / M_SNAKE_BAND; \
    if (_mb & 1) (tm) = _mb * M_SNAKE_BAND + (M_SNAKE_BAND - 1 - ((tm) % M_SNAKE_BAND)); \
} while(0)
#else
#define M_SNAKE_REMAP(tm) ((void)0)
#endif

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
#ifndef NUM_IDLE_WARPS
#define NUM_IDLE_WARPS 0
#endif
#define NUM_WARPS      (3 + NUM_EPI_WARPS + NUM_IDLE_WARPS)  /* W0+W1+W2 + epi + idle */
#define THREADS        (32 * NUM_WARPS)
#define GROUPS_PER_WARP (4 / NUM_EPI_WARPS)  /* row groups each epi warp processes */
static_assert(4 % NUM_EPI_WARPS == 0, "NUM_EPI_WARPS must be 1, 2, or 4");

/*
 * ptxas bug: `bar.sync 1, %0` with register operand gets constant-folded
 * to wrong immediate (128) regardless of actual value.  PTX immediate form
 * (`bar.sync 1, 32;`) works correctly.
 */
#define _STR(x)  #x
#define _XSTR(x) _STR(x)
#define _EPI_THR_1 32
#define _EPI_THR_2 64
#define _EPI_THR_3 96
#define _EPI_THR_4 128
#define _EPI_THR_5 160
#define _EPI_THR_6 192
#define _EPI_THR_7 224
#define _EPI_THR_8 256
#define _EPI_THR_X(n) _EPI_THR_##n
#define _EPI_THR(n)   _EPI_THR_X(n)
#define BAR_EPI_SYNC  "bar.sync 1, " _XSTR(_EPI_THR(NUM_EPI_WARPS)) ";"

/* ── SMEM layout ── */
#define STAGE_BYTES    32768                                    /* 16KB A + 16KB B */
#define OFF_TMEM           (N_STAGES * STAGE_BYTES)
#define OFF_TMA_MBAR       (OFF_TMEM + 8)
#define OFF_MMA_MBAR       (OFF_TMA_MBAR + N_STAGES * 8)
#define OFF_MAINLOOP_MBAR  (OFF_MMA_MBAR + N_STAGES * 8)
#define OFF_EPILOGUE_MBAR  (OFF_MAINLOOP_MBAR + 16)

/* New barriers for W2↔epilogue coordination (2-stage circular load pipe). */
#ifndef NUM_EPI_STAGES
#define NUM_EPI_STAGES     2
#endif
#ifndef SINGLE_WARP_STORE
#define SINGLE_WARP_STORE  0
#endif
#ifndef DELAY_TMA_STORE
#define DELAY_TMA_STORE    0
#endif
#ifndef CPP_EPILOGUE
#define CPP_EPILOGUE       0
#endif
#ifndef CUTLASS_LOOP
#define CUTLASS_LOOP       0
#endif
#ifndef NO_PRE_STORE_BAR
#define NO_PRE_STORE_BAR   0
#endif
#ifndef NO_POST_STORE_BAR
#define NO_POST_STORE_BAR  0
#endif

/* EPI_REORDER: shorthand for the full inter-warp stagger experiment.
   CHUNK_REORDER: odd warps reverse chunk order within sub-iter (2-phase temporal stagger,
     chunks 0/1 hit different SMEM bank halves so concurrent warps avoid bank conflicts).
   GROUP_REORDER: rotate the 4 STS group calls per-warp (groups 0-3 map to non-overlapping
     bank sets under SWIZZLE_128B, rotation eliminates remaining cross-warp bank conflicts).
   Combined with NO_PRE_STORE_BAR to let drift accumulate across sub-iters. */
#ifdef EPI_REORDER
#ifndef CHUNK_REORDER
#define CHUNK_REORDER
#endif
#ifndef GROUP_REORDER
#define GROUP_REORDER
#endif
#undef NO_PRE_STORE_BAR
#define NO_PRE_STORE_BAR 1
#endif

#if defined(GROUP_REORDER) && (defined(CUTLASS_EPILOGUE) || defined(CUTE_STORE) || CUTLASS_LOOP >= 3)
#error "GROUP_REORDER only supported with default BF16 epilogue path"
#endif

#ifdef SELF_LOAD
#undef NO_PRE_STORE_BAR
#define NO_PRE_STORE_BAR 1
#undef NO_POST_STORE_BAR
#define NO_POST_STORE_BAR 1
#if SINGLE_WARP_STORE
#error "SELF_LOAD requires SINGLE_WARP_STORE=0"
#endif
#if DELAY_TMA_STORE
#error "SELF_LOAD incompatible with DELAY_TMA_STORE"
#endif
#endif

#if NO_PRE_STORE_BAR && SINGLE_WARP_STORE
#error "NO_PRE_STORE_BAR requires SINGLE_WARP_STORE=0 (each warp stores its own region)"
#endif
#define NUM_EPI_SUBITERS   4
#ifdef SELF_LOAD
/* Per-warp TMA load barriers replace W2's shared circular pipe */
#define OFF_SELF_LOAD_MBAR (OFF_EPILOGUE_MBAR + 16)
#define _MBAR_END          (OFF_SELF_LOAD_MBAR + NUM_EPI_WARPS * NUM_EPI_STAGES * 8)
#else
#define OFF_LOAD_MBAR      (OFF_EPILOGUE_MBAR + 16)             /* W2→epi: stage ready */
#define OFF_LOAD_CONSUMED  (OFF_LOAD_MBAR + NUM_EPI_STAGES * 8) /* epi→W2: stage released */
#define _MBAR_END          (OFF_LOAD_CONSUMED + NUM_EPI_STAGES * 8)
#endif

/* TILE_DISPATCH: work-stealing tile dispatch mode.
   0 = static contiguous (default)
   1 = atomic + cluster barrier (heavy sync)
   2 = atomic + flag spin (CTA0 writes, CTA1 spins ld.shared::cluster)
   3 = grid-based non-persistent (blockIdx.y = tile, zero dispatch cost) */
#ifndef TILE_DISPATCH
#ifdef ATOMIC_TILES
#define TILE_DISPATCH 1
#else
#define TILE_DISPATCH 0
#endif
#endif

#if TILE_DISPATCH == 1 || TILE_DISPATCH == 2
#define OFF_TILE_SLOT      _MBAR_END
#define _LAYOUT_END        (OFF_TILE_SLOT + 8)
#else
#define _LAYOUT_END        _MBAR_END
#endif

/* Bias SMEM: 256 BF16 = 512 B */
#define OFF_BIAS_SMEM      ((_LAYOUT_END + 15) & ~15)
#define BIAS_SMEM_BYTES    (N_DIM * 2)

/* Epilogue staging: ReuseSmemC — 2-stage circular pipe.
   Each stage holds 128 rows × 64 cols × 2B = 16 KB, used for BOTH residual
   load and output store sequentially (residual overwritten by output after LDS). */
#define STAGING_REGION_BYTES  (32 * 128)                        /* 4096 B: 32 rows × 64 cols × 2B */
#define EPI_STAGE_BYTES    (4 * STAGING_REGION_BYTES)           /* 16384: 128 rows × 64 cols × 2B */

#define OFF_STAGING        ((OFF_BIAS_SMEM + BIAS_SMEM_BYTES + 1023) & ~1023)  /* 1024-align */
/* Stage si: OFF_STAGING + si * EPI_STAGE_BYTES
   Within stage si (ReuseSmemC — same region for load and store):
     data[rg]: OFF_STAGING + si*EPI_STAGE_BYTES + rg*STAGING_REGION_BYTES */

#define SMEM_BYTES         ((OFF_STAGING + NUM_EPI_STAGES * EPI_STAGE_BYTES + 127) & ~127)

/* ── WGMMA / TMEM ── */
#define TMEM_COLS      512
#define IDESC          0x10400010U
#define SBO            1024
#define TMA_BYTES      32768

/* ── Macros ── */
#define PRAGMA_UNROLL(n) _Pragma(_UNROLL_STR(n))
#define _UNROLL_STR2(x) #x
#define _UNROLL_STR(x) _UNROLL_STR2(unroll x)

/* ── Error checks ── */
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
void mbar_arrive_expect_tx(uint32_t addr, uint32_t tx_count) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
        :: "r"(addr), "r"(tx_count) : "memory");
}

static __device__ __forceinline__
void mbar_arrive(uint32_t addr) {
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
        :: "r"(addr) : "memory");
}

static __device__ __forceinline__
void tma_load_2d(uint32_t smem_dst, const void* tma_desc,
                  int32_t c0, int32_t c1, uint32_t mbar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_dst), "l"(tma_desc), "r"(c0), "r"(c1), "r"(mbar)
        : "memory");
}

static __device__ __forceinline__
void tma_load_2d_cta(uint32_t smem_dst, const void* tma_desc,
                      int32_t c0, int32_t c1, uint32_t mbar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_dst), "l"(tma_desc), "r"(c0), "r"(c1), "r"(mbar)
        : "memory");
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

#ifdef FP32_EPILOGUE
/* FP32 epilogue: unpack BF16 to FP32, add in FP32, CVT to BF16, STS.
   FFMA/FADD have ~0% STS conflict (vs 7.5% for HFMA2). Matches CUTLASS. */
static __device__ __forceinline__
void unpack_add_bf16x2(float& a_lo, float& a_hi, uint32_t packed) {
    /* BF16 is top 16 bits of FP32 — shift/mask gives valid float bit pattern */
    a_lo += __uint_as_float(packed << 16);
    a_hi += __uint_as_float(packed & 0xFFFF0000u);
}

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

#define BIAS_RES_CVT_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, b0,b1,b2,b3, r0,r1,r2,r3, SADDR) \
    do { \
        unpack_add_bf16x2(f0,f1,b0); unpack_add_bf16x2(f2,f3,b1); \
        unpack_add_bf16x2(f4,f5,b2); unpack_add_bf16x2(f6,f7,b3); \
        unpack_add_bf16x2(f0,f1,r0); unpack_add_bf16x2(f2,f3,r1); \
        unpack_add_bf16x2(f4,f5,r2); unpack_add_bf16x2(f6,f7,r3); \
        CVT_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7,SADDR); \
    } while(0)

#else /* BF16 epilogue (default) */

/* BF16 compute: 8 FP32 acc + 4 BF16x2 bias + 4 BF16x2 residual → CVT+ADD+ADD → STS */
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

#endif /* FP32_EPILOGUE */

#if CPP_EPILOGUE
/*
 * Break monolithic asm volatile blocks into individual instructions.
 * CVT and ADD: non-volatile asm — compiler/ptxas can freely reorder.
 * STS: volatile (must execute) but NO "memory" clobber — no scheduling barrier.
 *
 * Tests hypothesis: asm volatile + "memory" prevents ptxas from interleaving
 * STS from chunk N with CVT/ADD from chunk N+1. Exact same instructions,
 * just schedulable independently.
 */

#ifdef FP32_EPILOGUE
#undef CVT_STS_V4
#define CVT_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, SADDR) \
    do { \
        uint32_t _s0, _s1, _s2, _s3; \
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(_s0) : "f"(f0), "f"(f1)); \
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(_s1) : "f"(f2), "f"(f3)); \
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(_s2) : "f"(f4), "f"(f5)); \
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(_s3) : "f"(f6), "f"(f7)); \
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" \
            :: "r"(SADDR), "r"(_s0), "r"(_s1), "r"(_s2), "r"(_s3)); \
    } while(0)

#else /* BF16 + CPP_EPILOGUE */
#undef BIAS_RES_CVT_STS_V4
#define BIAS_RES_CVT_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, b0,b1,b2,b3, r0,r1,r2,r3, SADDR) \
    do { \
        uint32_t _o0, _o1, _o2, _o3; \
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(_o0) : "f"(f0), "f"(f1)); \
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(_o1) : "f"(f2), "f"(f3)); \
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(_o2) : "f"(f4), "f"(f5)); \
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(_o3) : "f"(f6), "f"(f7)); \
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_o0) : "r"(_o0), "r"(b0)); \
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_o1) : "r"(_o1), "r"(b1)); \
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_o2) : "r"(_o2), "r"(b2)); \
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_o3) : "r"(_o3), "r"(b3)); \
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_o0) : "r"(_o0), "r"(r0)); \
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_o1) : "r"(_o1), "r"(r1)); \
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_o2) : "r"(_o2), "r"(r2)); \
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_o3) : "r"(_o3), "r"(r3)); \
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" \
            :: "r"(SADDR), "r"(_o0), "r"(_o1), "r"(_o2), "r"(_o3)); \
    } while(0)

#endif /* FP32_EPILOGUE */
#endif /* CPP_EPILOGUE */

#ifdef CUTE_STORE
/*
 * C++ pointer stores instead of asm volatile STS.
 * CuTe's R2S copy uses C++ assignment: dst(i) = src(i), which nvcc
 * compiles to st.shared without asm volatile. Tests whether ptxas
 * schedules C++-generated stores differently from inline asm stores.
 * Compute (CVT, ADD) stays as non-volatile asm (same as CPP_EPILOGUE).
 */
#ifdef CUTLASS_EPILOGUE
#error "CUTE_STORE and CUTLASS_EPILOGUE are mutually exclusive"
#endif

/*
 * CUTE_STORE macro takes a char* pointer (from extern __shared__ smem[])
 * instead of uint32_t address. Callers pass stage_cptr + offset.
 * The __shared__ provenance makes nvcc emit st.shared.v4.b32.
 */
#ifdef FP32_EPILOGUE
#undef CVT_STS_V4
#define CVT_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, SPTR) \
    do { \
        uint32_t _s0, _s1, _s2, _s3; \
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(_s0) : "f"(f0), "f"(f1)); \
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(_s1) : "f"(f2), "f"(f3)); \
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(_s2) : "f"(f4), "f"(f5)); \
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(_s3) : "f"(f6), "f"(f7)); \
        *(uint4*)(SPTR) = make_uint4(_s0, _s1, _s2, _s3); \
    } while(0)

#else /* BF16 + CUTE_STORE */
#undef BIAS_RES_CVT_STS_V4
#define BIAS_RES_CVT_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, b0,b1,b2,b3, r0,r1,r2,r3, SPTR) \
    do { \
        uint32_t _o0, _o1, _o2, _o3; \
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(_o0) : "f"(f0), "f"(f1)); \
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(_o1) : "f"(f2), "f"(f3)); \
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(_o2) : "f"(f4), "f"(f5)); \
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(_o3) : "f"(f6), "f"(f7)); \
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_o0) : "r"(_o0), "r"(b0)); \
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_o1) : "r"(_o1), "r"(b1)); \
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_o2) : "r"(_o2), "r"(b2)); \
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_o3) : "r"(_o3), "r"(b3)); \
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_o0) : "r"(_o0), "r"(r0)); \
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_o1) : "r"(_o1), "r"(r1)); \
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_o2) : "r"(_o2), "r"(r2)); \
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_o3) : "r"(_o3), "r"(r3)); \
        *(uint4*)(SPTR) = make_uint4(_o0, _o1, _o2, _o3); \
    } while(0)
#endif /* FP32_EPILOGUE */
#endif /* CUTE_STORE */

#if CUTLASS_LOOP >= 3
#if defined(CUTLASS_EPILOGUE) || defined(CUTE_STORE)
#error "CUTLASS_LOOP=3 is mutually exclusive with CUTLASS_EPILOGUE and CUTE_STORE"
#endif
/*
 * CUTLASS_LOOP=3: Full C++ epilogue path.
 * No inline asm for loads/compute/stores — only TMEM load+wait.
 * FP32 math: unpack BF16→FP32, FADD residual+bias, CVT back, C++ store.
 * Combined with #pragma unroll 1, tests whether nvcc's C++ code generation
 * path produces structurally different PTX that ptxas schedules better.
 */
#define CPP_FP32_GROUP(a0,a1,a2,a3,a4,a5,a6,a7, bv, rv, SPTR, RSW) \
    do { \
        (a0) += __uint_as_float((rv).x << 16)        + __uint_as_float((bv).x << 16); \
        (a1) += __uint_as_float((rv).x & 0xFFFF0000u) + __uint_as_float((bv).x & 0xFFFF0000u); \
        (a2) += __uint_as_float((rv).y << 16)        + __uint_as_float((bv).y << 16); \
        (a3) += __uint_as_float((rv).y & 0xFFFF0000u) + __uint_as_float((bv).y & 0xFFFF0000u); \
        (a4) += __uint_as_float((rv).z << 16)        + __uint_as_float((bv).z << 16); \
        (a5) += __uint_as_float((rv).z & 0xFFFF0000u) + __uint_as_float((bv).z & 0xFFFF0000u); \
        (a6) += __uint_as_float((rv).w << 16)        + __uint_as_float((bv).w << 16); \
        (a7) += __uint_as_float((rv).w & 0xFFFF0000u) + __uint_as_float((bv).w & 0xFFFF0000u); \
        uint32_t _p0, _p1, _p2, _p3; \
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(_p0) : "f"(a0), "f"(a1)); \
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(_p1) : "f"(a2), "f"(a3)); \
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(_p2) : "f"(a4), "f"(a5)); \
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(_p3) : "f"(a6), "f"(a7)); \
        *(uint4*)((SPTR) + (RSW)) = make_uint4(_p0, _p1, _p2, _p3); \
    } while(0)
#endif /* CUTLASS_LOOP >= 3 */

#ifdef CUTLASS_EPILOGUE
/*
 * CUTLASS-clone epilogue: FP32 residual add → F2FP → BF16 bias add → STS.
 *
 * Splits the monolithic BIAS_RES_CVT_STS_V4 into:
 *   Phase A (C++, non-volatile): unpack residual BF16→FP32 + FADD to acc
 *   Phase B (asm volatile, per group): F2FP → HADD2 bias → STS.128
 *
 * Creates FADD→F2FP→HADD2→STS serial chain per group (4-deep).
 * ptxas should pipeline group N+1's FADD while group N's STS is in-flight.
 * Matches CUTLASS: FFMA→SHF→PRMT→HFMA2→STS pattern.
 */

/* Unpack uint4 of BF16x2 → 8 FP32 values (generates SHF+LOP3 in SASS) */
#define UNPACK_RES_FP32(dst, src) \
    do { \
        (dst)[0] = __uint_as_float((src).x << 16); \
        (dst)[1] = __uint_as_float((src).x & 0xFFFF0000u); \
        (dst)[2] = __uint_as_float((src).y << 16); \
        (dst)[3] = __uint_as_float((src).y & 0xFFFF0000u); \
        (dst)[4] = __uint_as_float((src).z << 16); \
        (dst)[5] = __uint_as_float((src).z & 0xFFFF0000u); \
        (dst)[6] = __uint_as_float((src).w << 16); \
        (dst)[7] = __uint_as_float((src).w & 0xFFFF0000u); \
    } while(0)

/* Per-group: F2FP → HADD2 (bias) → STS.128. One STS per call.
   Input: 8 FP32 accumulators (already have residual added), 4 BF16x2 bias, addr. */
#define CVT_BIAS_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, b0,b1,b2,b3, SADDR) \
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
           "r"(b0),"r"(b1),"r"(b2),"r"(b3), \
           "r"(SADDR) \
        : "memory")

/* LDS pipeline drain + fence in one asm block.
   ptxas DCEs separate drain loads (even with asm volatile).
   Merging with fence.proxy.async prevents removal: ptxas can't remove
   the fence, and the loads inside the same block share its liveness. */
#define LDS_DRAIN_AND_FENCE(SADDR) \
    asm volatile( \
        "{\n\t" \
        ".reg .b32 __dr;\n\t" \
        "ld.shared.b32 __dr, [%0];\n\t" \
        "ld.shared.b32 __dr, [%0];\n\t" \
        "ld.shared.b32 __dr, [%0];\n\t" \
        "ld.shared.b32 __dr, [%0];\n\t" \
        "fence.proxy.async.shared::cta;\n\t" \
        "}" :: "r"(SADDR) : "memory")
#endif /* CUTLASS_EPILOGUE */

/* TMA store + wait helpers — shared between main loop and drain.
   EPI_STORE: issue TMA store(s) + commit_group.
   EPI_WAIT:  wait_group + __syncwarp + bar.sync. */
#if SINGLE_WARP_STORE
#define EPI_STORE(STAGE, NC, PN, PM) do { \
    if (ew == 0 && lane == 0) { \
        for (int rg_ = 0; rg_ < NUM_EPI_WARPS; rg_++) { \
            const uint32_t s_ = smem_to_uint(smem + OFF_STAGING \
                + (STAGE) * EPI_STAGE_BYTES + rg_ * STAGING_REGION_BYTES); \
            asm volatile( \
                "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group" \
                " [%0, {%1, %2}], [%3];" \
                :: "l"(&tma_c), "r"((PN) + (NC)), "r"((PM) + rg_ * 32), \
                   "r"(s_) : "memory"); \
        } \
        asm volatile("cp.async.bulk.commit_group;" ::: "memory"); \
    } \
} while(0)
#define EPI_WAIT_PRED (ew == 0 && lane == 0)
#else
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
#define EPI_WAIT_PRED (lane == 0)
#endif

#if NO_POST_STORE_BAR
#define EPI_WAIT(LAST) do { \
    if (EPI_WAIT_PRED) { \
        if (LAST) \
            asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); \
        else \
            asm volatile("cp.async.bulk.wait_group 1;" ::: "memory"); \
    } \
    __syncwarp(); \
} while(0)
#else
#define EPI_WAIT(LAST) do { \
    if (EPI_WAIT_PRED) { \
        if (LAST) \
            asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); \
        else \
            asm volatile("cp.async.bulk.wait_group 1;" ::: "memory"); \
    } \
    __syncwarp(); \
    asm volatile(BAR_EPI_SYNC ::: "memory"); \
} while(0)
#endif

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


#if TILE_DISPATCH == 1 || TILE_DISPATCH == 2
__device__ int g_tile_ctr;
#endif

/* ════════════════════════════════════════════════════════════════
   KERNEL
   ════════════════════════════════════════════════════════════════ */

__global__ void __launch_bounds__(THREADS, 1)
__cluster_dims__(2, 1, 1)
fc2_w3_kernel(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    const __grid_constant__ CUtensorMap tma_c,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ C,
    const __nv_bfloat16* __restrict__ residual,
    const __grid_constant__ CUtensorMap tma_res
) {
    extern __shared__ __align__(128) char smem[];
#if TILE_DISPATCH == 3
    /* Grid-based: blockIdx.x = CTA rank (0..1), blockIdx.y = tile index */
    const int cta_rank = blockIdx.x;
#else
    const int sm_id = blockIdx.x;
    const int cta_rank = sm_id & 1;
#endif
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
            /* epilogue mbar: W2 + W3-W6 + idle warps arrive.
               (NUM_EPI_WARPS + 1 + NUM_IDLE_WARPS) warps × 2 CTAs × 32 threads */
            mbar_init(smem_to_uint(smem + OFF_EPILOGUE_MBAR + i * 8), (NUM_EPI_WARPS + 1 + NUM_IDLE_WARPS) * 2 * 32);
        }
#ifdef SELF_LOAD
        /* Per-warp TMA load completion barriers */
        for (int ew = 0; ew < NUM_EPI_WARPS; ew++)
            for (int s = 0; s < NUM_EPI_STAGES; s++)
                mbar_init(smem_to_uint(smem + OFF_SELF_LOAD_MBAR + (ew * NUM_EPI_STAGES + s) * 8), 1);
#else
        /* W2→epilogue: stage ready. W2 arrives with expect_tx. */
        for (int s = 0; s < NUM_EPI_STAGES; s++)
            mbar_init(smem_to_uint(smem + OFF_LOAD_MBAR + s * 8), 1);
        /* epilogue→W2: stage released. ALL epilogue threads arrive after
           TMA store completes, structurally guaranteed by BAR.SYNC. */
        for (int s = 0; s < NUM_EPI_STAGES; s++)
            mbar_init(smem_to_uint(smem + OFF_LOAD_CONSUMED + s * 8), NUM_EPI_WARPS * 32);
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
    /* Bit 24 = CTA select in cluster SMEM addressing.  Clear it so both CTAs
       arrive on CTA 0's epilogue mbar (W1 runs on CTA 0 only). */
    const uint32_t epi_mbar_masked = epilogue_mbar_addr & 0xFEFFFFFF;

#if TILE_DISPATCH == 1 || TILE_DISPATCH == 2
    /* Tile slot address: CTA1 reads from CTA0's SMEM (clear bit 24) */
    const uint32_t tile_slot_addr = (smem_to_uint(smem + OFF_TILE_SLOT)) & (cta_rank ? 0xFEFFFFFFU : 0xFFFFFFFFU);
#endif
#if TILE_DISPATCH >= 1
    int _iter = 0;
    int _prev_tile = -1;
#else
    const int tile_stride = num_clusters;  /* strided: cluster 0 gets 0,74,148,... */
#ifdef BIDIR_SNAKE
    /* Even clusters go forward (0,74,148,...), odd clusters go backward (10877,10803,...) */
    const bool reverse = (cluster_id & 1);
    const int fwd_id = reverse ? (num_clusters - 1 - cluster_id) : cluster_id;
    const int tile_count = (TOTAL_TILES - fwd_id + tile_stride - 1) / tile_stride;
#else
    const int tile_count  = ((TILES_M - cluster_id + num_clusters - 1) / num_clusters) * TILES_N;
#endif
#endif

    int tma_phase[N_STAGES] = {0};
    int mma_phase[N_STAGES] = {0};

    uint64_t desc_a_base[N_STAGES], desc_b_base[N_STAGES];
    for (int s = 0; s < N_STAGES; s++) {
        desc_a_base[s] = make_smem_desc(smem_a[s]);
        desc_b_base[s] = make_smem_desc(smem_b[s]);
    }

#if TILE_DISPATCH >= 1
    const int start_buf = 0;
#else
    const int start_buf = 0;
#endif
    int epi_phase[2] = {1, 1};
    int ml_phase[2]  = {start_buf, 1 - start_buf};

#ifndef SELF_LOAD
    /* W2 + epilogue barrier addresses & phases for the circular load pipe. */
    uint32_t load_mbar[NUM_EPI_STAGES];
    uint32_t consumed_mbar[NUM_EPI_STAGES];
    int load_phase[NUM_EPI_STAGES];
    int load_consumed_phase[NUM_EPI_STAGES];
    int load_issue_count = 0;
    for (int s = 0; s < NUM_EPI_STAGES; s++) {
        load_mbar[s] = smem_to_uint(smem + OFF_LOAD_MBAR + s * 8);
        consumed_mbar[s] = smem_to_uint(smem + OFF_LOAD_CONSUMED + s * 8);
        load_phase[s] = 0;
        load_consumed_phase[s] = 0;
    }
#endif

    /* ── Load bias into SMEM once ── */
    {
        const uint32_t bias_saddr = smem_to_uint(smem + OFF_BIAS_SMEM);
        /* 256 BF16 = 128 uint32_t. Spread across all threads (224 threads, each loads ~0-1). */
        for (int i = tid; i < N_DIM / 2; i += THREADS) {
            /* Load 2 BF16 as uint32_t, store to SMEM */
            uint32_t val;
            asm volatile("ld.global.b32 %0, [%1];" : "=r"(val) : "l"(bias + i * 2));
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(bias_saddr + i * 4), "r"(val));
        }
    }
    __syncthreads();

#ifdef WARP_STAGGER
    /*
     * Set P6 = (warp_id & 1) for inter-warp stagger.
     * Epilogue warps W3-W6: P6=1 for {W3,W5}, P6=0 for {W4,W6}.
     * SASS patching replaces NOPs with @P6 YIELD so odd warps yield.
     */
    {
        unsigned ws;
        asm volatile("mov.u32 %0, %%warpid;" : "=r"(ws));
        unsigned bit = ws & 1;
        asm volatile(
            "setp.ne.u32 %%p6, %0, 0;" :: "r"(bit)
        );
    }
#endif

#ifdef LDS_DRAIN
    /*
     * Drain accumulator: XOR'd into drain loads to keep them alive.
     * Initialized to 0 so XOR is identity. ptxas can't prove it's 0
     * because it comes from asm volatile.
     */
    uint32_t drain_acc = 0;
    asm volatile("mov.u32 %0, 0;" : "=r"(drain_acc));
#endif

    /* ════════════════════════════════════════════
       MAIN TILE LOOP
       ════════════════════════════════════════════ */

    /*
     * Tile dispatch modes:
     *   0: Static contiguous — each cluster gets [start, end) range
     *   1: Atomic + cluster barrier — CTA0 atomicAdds, barrier broadcasts
     *   2: Atomic + flag spin — CTA0 atomicAdds + st, CTA1 spins ld.shared::cluster
     *   3: Grid-based — blockIdx.y = tile_idx, non-persistent, zero dispatch cost
     */
#if TILE_DISPATCH == 3
    /* Non-persistent: one tile per cluster, blockIdx.y = tile index */
    {
        const int tile_idx = (int)blockIdx.y;
        const int buf = 0;
        const bool has_prev = false;
#elif TILE_DISPATCH >= 1
    while (true) {
        int tile_idx;
#if TILE_DISPATCH == 1
        /* Cluster barrier: CTA0 atomicAdds, stores to SMEM, full barrier broadcasts */
        if (cta_rank == 0 && tid == 0) {
            int t = atomicAdd(&g_tile_ctr, 1);
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_to_uint(smem + OFF_TILE_SLOT)), "r"(t));
        }
        asm volatile("barrier.cluster.arrive.relaxed.aligned;");
        asm volatile("barrier.cluster.wait.acquire.aligned;");
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(tile_idx) : "r"(tile_slot_addr));
#elif TILE_DISPATCH == 2
        /* Flag spin: CTA0 atomicAdds + writes tile_idx with epoch to SMEM.
           CTA1 thread 0 spins via ld.shared::cluster until epoch matches.
           Epoch = _iter+1 (never 0, so initial SMEM zero means "not ready"). */
        if (cta_rank == 0 && tid == 0) {
            int t = atomicAdd(&g_tile_ctr, 1);
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_to_uint(smem + OFF_TILE_SLOT)), "r"(t));
            /* Write epoch AFTER tile_idx so CTA1 sees consistent data.
               fence ensures st ordering within CTA0's SMEM. */
            asm volatile("fence.acq_rel.cluster;");
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_to_uint(smem + OFF_TILE_SLOT + 4)), "r"(_iter + 1));
        }
        if (cta_rank == 1 && tid == 0) {
            /* Spin until epoch matches — typically resolves in ~20-30 cycles */
            int epoch;
            do {
                asm volatile("ld.acquire.cluster.shared::cluster.b32 %0, [%1];"
                    : "=r"(epoch) : "r"(tile_slot_addr + 4));
            } while (epoch != _iter + 1);
        }
        __syncthreads();
        if (cta_rank == 0) {
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(tile_idx) : "r"(smem_to_uint(smem + OFF_TILE_SLOT)));
        } else {
            asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(tile_idx) : "r"(tile_slot_addr));
        }
#endif
        if (tile_idx >= TOTAL_TILES) break;
        const int buf = _iter & 1;
#else
    for (int _ti = 0; _ti < tile_count; _ti++) {
#ifdef BIDIR_SNAKE
        const int fwd_tile = fwd_id + _ti * tile_stride;
        const int tile_idx = reverse ? (TOTAL_TILES - 1 - fwd_tile) : fwd_tile;
#else
        /* Phase-offset N-batch: each cluster eats all 3 N-tiles per M-row,
           clusters are phase-offset in tn to spread TMA stores */
        const int _tm = cluster_id + (_ti / TILES_N) * num_clusters;
        if (_tm >= TILES_M) break;
        const int tile_idx = _tm * TILES_N + (_ti + cluster_id) % TILES_N;
#endif
        const int buf = _ti & 1;
#endif
        int tm = tile_idx / TILES_N;
        int tn = tile_idx % TILES_N;
        M_SNAKE_REMAP(tm);
        if (SNAKE_ORDER && (tm & 1)) tn = TILES_N - 1 - tn;
        const int m_start = tm * TM * 2 + cta_rank * TM;
        const int n_start = tn * TN;
#if TILE_DISPATCH == 1 || TILE_DISPATCH == 2
        const bool has_prev = (_iter > 0);
#elif TILE_DISPATCH == 0
        const bool has_prev = (_ti > 0);
#endif
        /* TILE_DISPATCH==3: has_prev already set to false above */

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

        } else if (warp == 1) {
            /* ── W1: MMA ── */
            if (lane == 0 && cta_rank == 0) {
                mbar_wait(epilogue_mbar_addr + buf * 8, epi_phase[buf]);
                epi_phase[buf] ^= 1;

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

                /* K-iterations 1..K_ITERS-1 */
                PRAGMA_UNROLL(K_LOOP_UNROLL)
                for (int ki = 1; ki < K_ITERS; ki++) {
                    K_ITER_ACCUM(ki % N_STAGES);
                }

                /* Signal epilogue: MMA done for this tile */
                tcgen05_commit_mcast(mainloop_mbar_addr + buf * 8, 0x3);
            }

        } else if (warp == 2) {
            /* W2 must wait on mainloop_mbar EVERY tile (including tile_start)
               to consume the free-pass phase. Only epilogue work is conditional. */
            const int prev_buf = buf ^ 1;
            mbar_wait(mainloop_mbar_addr + prev_buf * 8, ml_phase[prev_buf]);
            ml_phase[prev_buf] ^= 1;
#if defined(STRIP_EPILOGUE) || defined(SELF_LOAD)
            if (has_prev)
                mbar_arrive(epi_mbar_masked + prev_buf * 8);
#else
            /* ── W2: EpilogueLoad — circular producer for PREVIOUS tile ──
               Stream four 64-col slices through a 2-stage shared pipe. */
            if (has_prev) {
#if TILE_DISPATCH >= 1
                const int prev_idx = _prev_tile;
#elif defined(BIDIR_SNAKE)
                const int prev_fwd = fwd_id + (_ti - 1) * tile_stride;
                const int prev_idx = reverse ? (TOTAL_TILES - 1 - prev_fwd) : prev_fwd;
#else
                const int _pti = _ti - 1;
                const int _ptm = cluster_id + (_pti / TILES_N) * num_clusters;
                const int prev_idx = _ptm * TILES_N + (_pti + cluster_id) % TILES_N;
#endif
                int ptm = prev_idx / TILES_N;
                int ptn = prev_idx % TILES_N;
                M_SNAKE_REMAP(ptm);
                if (SNAKE_ORDER && (ptm & 1)) ptn = TILES_N - 1 - ptn;
                const int prev_m = ptm * TM * 2 + cta_rank * TM;
                const int prev_n = ptn * TN;

                for (int si = 0; si < NUM_EPI_SUBITERS; si++) {
                    const int stage = si % NUM_EPI_STAGES;
                    if (load_issue_count >= NUM_EPI_STAGES) {
                        mbar_wait(consumed_mbar[stage], load_consumed_phase[stage]);
                        load_consumed_phase[stage] ^= 1;
                    }
                    if (lane == 0) {
                        const uint32_t res_dst = smem_to_uint(smem + OFF_STAGING + stage * EPI_STAGE_BYTES);
                        mbar_arrive_expect_tx(load_mbar[stage], EPI_STAGE_BYTES);
                        tma_load_2d_cta(res_dst, &tma_res,
                                        prev_n + si * 64, prev_m, load_mbar[stage]);
                    }
                    load_issue_count++;
                }

                /* Arrive epi_mbar AFTER all loads — prevents W1 from starting
                   next tile's MMA while W2 still issues TMA loads. */
                mbar_arrive(epi_mbar_masked + prev_buf * 8);
            }
#endif /* STRIP_EPILOGUE W2 */

        } else {
            /* W3+ must wait on mainloop_mbar EVERY tile (including tile_start)
               to consume the free-pass phase. Only epilogue work is conditional. */
            const int prev_buf = buf ^ 1;
            mbar_wait(mainloop_mbar_addr + prev_buf * 8, ml_phase[prev_buf]);
            ml_phase[prev_buf] ^= 1;
#if NUM_IDLE_WARPS > 0
            if (warp >= 3 + NUM_EPI_WARPS) {
                /* Idle warps: just arrive at epi_mbar, no epilogue work */
                if (has_prev)
                    mbar_arrive(epi_mbar_masked + prev_buf * 8);
                continue;
            }
#endif
#ifdef STRIP_EPILOGUE
            if (has_prev)
                mbar_arrive(epi_mbar_masked + prev_buf * 8);
#else
            /* ── W3-W6: Epilogue compute — ReuseSmemC, BAR.SYNC coordinated ── */
            const int ew = warp - 3;                           /* 0..NUM_EPI_WARPS-1 */
            const uint32_t bias_saddr = smem_to_uint(smem + OFF_BIAS_SMEM);

            /* Swizzle constants (SWIZZLE_128B: 128-byte rows, XOR with lane-group) */
            const uint32_t xor_val = (lane & 7) << 4;
            const uint32_t sw0 = 0 ^ xor_val, sw1 = 16 ^ xor_val;
            const uint32_t sw2 = 32 ^ xor_val, sw3 = 48 ^ xor_val;
            const uint32_t sw4 = 64 ^ xor_val, sw5 = 80 ^ xor_val;
            const uint32_t sw6 = 96 ^ xor_val, sw7 = 112 ^ xor_val;

            if (has_prev) {
#if TILE_DISPATCH >= 1
                const int prev_idx = _prev_tile;
#elif defined(BIDIR_SNAKE)
                const int prev_fwd = fwd_id + (_ti - 1) * tile_stride;
                const int prev_idx = reverse ? (TOTAL_TILES - 1 - prev_fwd) : prev_fwd;
#else
                const int _pti = _ti - 1;
                const int _ptm = cluster_id + (_pti / TILES_N) * num_clusters;
                const int prev_idx = _ptm * TILES_N + (_pti + cluster_id) % TILES_N;
#endif
                int ptm = prev_idx / TILES_N;
                int ptn = prev_idx % TILES_N;
                M_SNAKE_REMAP(ptm);
                if (SNAKE_ORDER && (ptm & 1)) ptn = TILES_N - 1 - ptn;
                const int prev_m = ptm * TM * 2 + cta_rank * TM;
                const int prev_n = ptn * TN;

                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

#ifdef SELF_LOAD
                uint32_t self_mbar_arr[NUM_EPI_STAGES];
                int self_mbar_phase[NUM_EPI_STAGES] = {0};
                for (int s = 0; s < NUM_EPI_STAGES; s++)
                    self_mbar_arr[s] = smem_to_uint(smem + OFF_SELF_LOAD_MBAR + (ew * NUM_EPI_STAGES + s) * 8);
#endif

#if DELAY_TMA_STORE
                int have_pending = 0;
                int pend_nc, pend_stage;
#endif

#if defined(SELF_LOAD) && defined(SELF_STAGGER) && SELF_STAGGER > 0
                /* Deliberate initial offset: spread warps temporally.
                   Warp 0 starts immediately, warp ew sleeps ew*SELF_STAGGER ns. */
                if (ew > 0) {
                    uint32_t _ns = (uint32_t)ew * SELF_STAGGER;
                    asm volatile("nanosleep.u32 %0;" :: "r"(_ns));
                }
#endif

#if CUTLASS_LOOP >= 1
                PRAGMA_UNROLL(1)
#endif
                for (int si = 0; si < NUM_EPI_SUBITERS; si++) {
                    const int stage = si % NUM_EPI_STAGES;
                    const int nc_base = si * 64;   /* column offset within tile */

#if DELAY_TMA_STORE
                    /* Issue delayed TMA store from previous sub-iter */
                    if (have_pending) {
                        const int row_group = ew;
                        EPI_STORE(pend_stage, pend_nc, prev_n, prev_m);
                    }
                    /* Wait for 2-ago store + consumed signal */
                    if (si >= 2) {
                        EPI_WAIT(0);
                        mbar_arrive(consumed_mbar[(si - 2) % NUM_EPI_STAGES]);
                    }
#endif

#ifdef SELF_LOAD
                    /* Wait for 2-ago TMA store before reusing SMEM stage */
                    if (si >= NUM_EPI_STAGES) {
                        if (lane == 0)
                            asm volatile("cp.async.bulk.wait_group 1;" ::: "memory");
                        __syncwarp();
                    }
                    /* Issue per-warp TMA load: 32 rows × 64 cols */
                    if (lane == 0) {
                        const uint32_t res_dst = smem_to_uint(smem + OFF_STAGING
                            + stage * EPI_STAGE_BYTES + ew * STAGING_REGION_BYTES);
                        mbar_arrive_expect_tx(self_mbar_arr[stage], STAGING_REGION_BYTES);
                        tma_load_2d_cta(res_dst, &tma_res,
                                        prev_n + si * 64, prev_m + ew * 32,
                                        self_mbar_arr[stage]);
                    }
                    mbar_wait(self_mbar_arr[stage], self_mbar_phase[stage]);
                    self_mbar_phase[stage] ^= 1;
#else
                    /* Wait for W2's TMA load to land for this sub-iteration. */
                    mbar_wait(load_mbar[stage], load_phase[stage]);
                    load_phase[stage] ^= 1;
#endif

                    /* Staging SMEM base for drain/fence (row_group=0, valid for any group) */
                    const uint32_t stage_drain = smem_to_uint(smem + OFF_STAGING
                        + stage * EPI_STAGE_BYTES + lane * 128);

                    /* Process 2 chunks of 32 cols each, looping over row groups */
#if GROUPS_PER_WARP > 1
                    #pragma unroll
                    for (int _rg = 0; _rg < GROUPS_PER_WARP; _rg++) {
                    #ifdef EPI_STRIDED
                    const int row_group = ew + _rg * NUM_EPI_WARPS;
#else
                    const int row_group = ew * GROUPS_PER_WARP + _rg;
#endif
#else
                    { const int row_group = ew;
#endif
                    const int taddr_base = prev_buf * TN + ((cta_rank * 128 + row_group * 32) << 16);
                    float a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15;
                    float a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31;

                    /* ReuseSmemC: single SMEM region for both LDS residual and STS output */
                    const uint32_t stage_base = smem_to_uint(smem + OFF_STAGING
                        + stage * EPI_STAGE_BYTES
                        + row_group * STAGING_REGION_BYTES
                        + lane * 128);
#ifdef CUTE_STORE
                    char* stage_cptr = smem + OFF_STAGING
                        + stage * EPI_STAGE_BYTES
                        + row_group * STAGING_REGION_BYTES
                        + lane * 128;
#endif

#if CUTLASS_LOOP >= 2
                    PRAGMA_UNROLL(1)
#endif
                    for (int _ci = 0; _ci < 2; _ci++) {
#ifdef CHUNK_REORDER
                        const int chunk = (ew & 1) ? (1 - _ci) : _ci;
#else
                        const int chunk = _ci;
#endif
                        const int nc = nc_base + chunk * 32;

                        /* TMEM load: 32 FP32 accumulators */
                        TMEM_LOAD_X32(a0,a1,a2,a3,a4,a5,a6,a7,
                                      a8,a9,a10,a11,a12,a13,a14,a15,
                                      a16,a17,a18,a19,a20,a21,a22,a23,
                                      a24,a25,a26,a27,a28,a29,a30,a31,
                                      taddr_base + nc);

#if CUTLASS_LOOP >= 3
                        {
                            const uint32_t rsw0 = chunk ? sw4 : sw0;
                            const uint32_t rsw1 = chunk ? sw5 : sw1;
                            const uint32_t rsw2 = chunk ? sw6 : sw2;
                            const uint32_t rsw3 = chunk ? sw7 : sw3;
                            char* sptr = smem + OFF_STAGING
                                + stage * EPI_STAGE_BYTES
                                + row_group * STAGING_REGION_BYTES
                                + lane * 128;

                            /* C++ reads: bias from linear SMEM */
                            const char* bp = smem + OFF_BIAS_SMEM + (prev_n + nc) * 2;
                            uint4 bv0 = *(const uint4*)(bp);
                            uint4 bv1 = *(const uint4*)(bp + 16);
                            uint4 bv2 = *(const uint4*)(bp + 32);
                            uint4 bv3 = *(const uint4*)(bp + 48);

                            /* C++ reads: residual from swizzled staging SMEM */
                            uint4 rv0 = *(const uint4*)(sptr + rsw0);
                            uint4 rv1 = *(const uint4*)(sptr + rsw1);
                            uint4 rv2 = *(const uint4*)(sptr + rsw2);
                            uint4 rv3 = *(const uint4*)(sptr + rsw3);

                            TMEM_WAIT();

                            /* FP32 compute + C++ store per group */
                            CPP_FP32_GROUP(a0,a1,a2,a3,a4,a5,a6,a7, bv0, rv0, sptr, rsw0);
                            CPP_FP32_GROUP(a8,a9,a10,a11,a12,a13,a14,a15, bv1, rv1, sptr, rsw1);
                            CPP_FP32_GROUP(a16,a17,a18,a19,a20,a21,a22,a23, bv2, rv2, sptr, rsw2);
                            CPP_FP32_GROUP(a24,a25,a26,a27,a28,a29,a30,a31, bv3, rv3, sptr, rsw3);
                        }
#else
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

                        /* LDS residual from shared staging (swizzled, ReuseSmemC) */
                        const uint32_t rsw0 = chunk ? sw4 : sw0;
                        const uint32_t rsw1 = chunk ? sw5 : sw1;
                        const uint32_t rsw2 = chunk ? sw6 : sw2;
                        const uint32_t rsw3 = chunk ? sw7 : sw3;
                        uint4 rv0, rv1, rv2, rv3;
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(rv0.x),"=r"(rv0.y),"=r"(rv0.z),"=r"(rv0.w) : "r"(stage_base + rsw0));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(rv1.x),"=r"(rv1.y),"=r"(rv1.z),"=r"(rv1.w) : "r"(stage_base + rsw1));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(rv2.x),"=r"(rv2.y),"=r"(rv2.z),"=r"(rv2.w) : "r"(stage_base + rsw2));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(rv3.x),"=r"(rv3.y),"=r"(rv3.z),"=r"(rv3.w) : "r"(stage_base + rsw3));

#ifdef CUTLASS_EPILOGUE
                        /* Pre-unpack residual BF16→FP32 (hides in TMEM latency) */
                        float rr0[8], rr1[8], rr2[8], rr3[8];
                        UNPACK_RES_FP32(rr0, rv0);
                        UNPACK_RES_FP32(rr1, rv1);
                        UNPACK_RES_FP32(rr2, rv2);
                        UNPACK_RES_FP32(rr3, rv3);

                        TMEM_WAIT();

                        /* Per-group: FP32 residual add → F2FP → BF16 bias add → STS */
                        a0+=rr0[0]; a1+=rr0[1]; a2+=rr0[2]; a3+=rr0[3];
                        a4+=rr0[4]; a5+=rr0[5]; a6+=rr0[6]; a7+=rr0[7];
                        CVT_BIAS_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7,
                            bv0.x,bv0.y,bv0.z,bv0.w, stage_base + rsw0);

                        a8+=rr1[0]; a9+=rr1[1]; a10+=rr1[2]; a11+=rr1[3];
                        a12+=rr1[4]; a13+=rr1[5]; a14+=rr1[6]; a15+=rr1[7];
                        CVT_BIAS_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15,
                            bv1.x,bv1.y,bv1.z,bv1.w, stage_base + rsw1);

                        a16+=rr2[0]; a17+=rr2[1]; a18+=rr2[2]; a19+=rr2[3];
                        a20+=rr2[4]; a21+=rr2[5]; a22+=rr2[6]; a23+=rr2[7];
                        CVT_BIAS_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23,
                            bv2.x,bv2.y,bv2.z,bv2.w, stage_base + rsw2);

                        a24+=rr3[0]; a25+=rr3[1]; a26+=rr3[2]; a27+=rr3[3];
                        a28+=rr3[4]; a29+=rr3[5]; a30+=rr3[6]; a31+=rr3[7];
                        CVT_BIAS_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31,
                            bv3.x,bv3.y,bv3.z,bv3.w, stage_base + rsw3);
#else
                        TMEM_WAIT();

                        /* STS output to SAME stage region (ReuseSmemC) */
#ifdef CUTE_STORE
                        BIAS_RES_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7,
                            bv0.x,bv0.y,bv0.z,bv0.w,
                            rv0.x,rv0.y,rv0.z,rv0.w, stage_cptr + rsw0);
                        BIAS_RES_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15,
                            bv1.x,bv1.y,bv1.z,bv1.w,
                            rv1.x,rv1.y,rv1.z,rv1.w, stage_cptr + rsw1);
                        BIAS_RES_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23,
                            bv2.x,bv2.y,bv2.z,bv2.w,
                            rv2.x,rv2.y,rv2.z,rv2.w, stage_cptr + rsw2);
                        BIAS_RES_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31,
                            bv3.x,bv3.y,bv3.z,bv3.w,
                            rv3.x,rv3.y,rv3.z,rv3.w, stage_cptr + rsw3);
#else
#ifdef GROUP_REORDER
                        /* Rotate STS group order by ew: at each time step, 4 warps
                           write to 4 disjoint SMEM bank sets (SWIZZLE_128B). */
#define _GR0 BIAS_RES_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7, \
    bv0.x,bv0.y,bv0.z,bv0.w, rv0.x,rv0.y,rv0.z,rv0.w, stage_base + rsw0)
#define _GR1 BIAS_RES_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15, \
    bv1.x,bv1.y,bv1.z,bv1.w, rv1.x,rv1.y,rv1.z,rv1.w, stage_base + rsw1)
#define _GR2 BIAS_RES_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23, \
    bv2.x,bv2.y,bv2.z,bv2.w, rv2.x,rv2.y,rv2.z,rv2.w, stage_base + rsw2)
#define _GR3 BIAS_RES_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31, \
    bv3.x,bv3.y,bv3.z,bv3.w, rv3.x,rv3.y,rv3.z,rv3.w, stage_base + rsw3)
                        switch (ew) {
                        case 0: _GR0; _GR1; _GR2; _GR3; break;
                        case 1: _GR1; _GR2; _GR3; _GR0; break;
                        case 2: _GR2; _GR3; _GR0; _GR1; break;
                        case 3: _GR3; _GR0; _GR1; _GR2; break;
                        }
#undef _GR0
#undef _GR1
#undef _GR2
#undef _GR3
#else
                        BIAS_RES_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7,
                            bv0.x,bv0.y,bv0.z,bv0.w,
                            rv0.x,rv0.y,rv0.z,rv0.w, stage_base + rsw0);
                        BIAS_RES_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15,
                            bv1.x,bv1.y,bv1.z,bv1.w,
                            rv1.x,rv1.y,rv1.z,rv1.w, stage_base + rsw1);
                        BIAS_RES_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23,
                            bv2.x,bv2.y,bv2.z,bv2.w,
                            rv2.x,rv2.y,rv2.z,rv2.w, stage_base + rsw2);
                        BIAS_RES_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31,
                            bv3.x,bv3.y,bv3.z,bv3.w,
                            rv3.x,rv3.y,rv3.z,rv3.w, stage_base + rsw3);
#endif /* GROUP_REORDER */
#endif /* CUTE_STORE */
#endif /* CUTLASS_EPILOGUE */
#endif /* CUTLASS_LOOP >= 3 */
                    }
                    } /* close row_group loop/block */

                    /* FENCE + BAR.SYNC: all epilogue warps' STS must be visible */
#ifdef LDS_DRAIN
                    /*
                     * LSU pipeline drain: 4 LDS from addresses 128B apart
                     * so ptxas can't merge into a single wide load.
                     * Each feeds drain_acc in its own asm block.
                     */
                    { uint32_t _d;
                    asm volatile("ld.shared.b32 %0, [%1];"
                        : "=r"(_d) : "r"(stage_drain) : "memory");
                    drain_acc ^= _d;
                    asm volatile("ld.shared.b32 %0, [%1+128];"
                        : "=r"(_d) : "r"(stage_drain) : "memory");
                    drain_acc ^= _d;
                    asm volatile("ld.shared.b32 %0, [%1+256];"
                        : "=r"(_d) : "r"(stage_drain) : "memory");
                    drain_acc ^= _d;
                    asm volatile("ld.shared.b32 %0, [%1+384];"
                        : "=r"(_d) : "r"(stage_drain) : "memory");
                    drain_acc ^= _d;
                    }
#endif
#ifdef CUTLASS_EPILOGUE
                    LDS_DRAIN_AND_FENCE(stage_drain);
#else
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
#if !NO_PRE_STORE_BAR
                    asm volatile(BAR_EPI_SYNC ::: "memory");
#endif

#ifdef SELF_LOAD
                    { const int row_group = ew;
                    EPI_STORE(stage, nc_base, prev_n, prev_m); }
#elif DELAY_TMA_STORE
                    have_pending = 1;
                    pend_nc = nc_base;
                    pend_stage = stage;
#else
#if GROUPS_PER_WARP > 1
                    /* Batch TMA stores for all row groups into one commit_group */
                    if (lane == 0) {
                        #pragma unroll
                        for (int _rg = 0; _rg < GROUPS_PER_WARP; _rg++) {
                            #ifdef EPI_STRIDED
                            const int rg = ew + _rg * NUM_EPI_WARPS;
#else
                            const int rg = ew * GROUPS_PER_WARP + _rg;
#endif
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
                    if (si > 0)
                        mbar_arrive(consumed_mbar[(si - 1) % NUM_EPI_STAGES]);
                    if (si == NUM_EPI_SUBITERS - 1)
                        mbar_arrive(consumed_mbar[stage]);
#endif
                }

#ifdef SELF_LOAD
                /* Drain all outstanding TMA stores */
                if (lane == 0)
                    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                __syncwarp();
#elif DELAY_TMA_STORE
                /* Drain delayed pipeline: issue last store + drain all */
                { const int row_group = ew;
                EPI_STORE(pend_stage, pend_nc, prev_n, prev_m); }
                EPI_WAIT(1);
                mbar_arrive(consumed_mbar[(NUM_EPI_SUBITERS - 2) % NUM_EPI_STAGES]);
                mbar_arrive(consumed_mbar[pend_stage]);
#endif

                /* Signal W1: TMEM buffer free for the next user of prev_buf. */
#ifdef LDS_DRAIN
                mbar_arrive((epi_mbar_masked ^ drain_acc) + prev_buf * 8);
#else
                mbar_arrive(epi_mbar_masked + prev_buf * 8);
#endif
            }
#endif /* STRIP_EPILOGUE W3-W6 */
        }
#if TILE_DISPATCH >= 1
        _prev_tile = tile_idx;
        _iter++;
#endif
    }

    /* ── Drain: last tile epilogue ── */
    {
#if TILE_DISPATCH == 3
        const int last_idx = (int)blockIdx.y;
        const int last_buf = 0;
#elif TILE_DISPATCH >= 1
        const int last_idx = _prev_tile;
        const int last_buf = (_iter - 1) & 1;
#else
#ifdef BIDIR_SNAKE
        const int last_fwd = fwd_id + (tile_count - 1) * tile_stride;
        const int last_idx = reverse ? (TOTAL_TILES - 1 - last_fwd) : last_fwd;
#else
        const int _lti = tile_count - 1;
        const int _ltm = cluster_id + (_lti / TILES_N) * num_clusters;
        const int last_idx = _ltm * TILES_N + (_lti + cluster_id) % TILES_N;
#endif
        const int last_buf = (tile_count - 1) & 1;
#endif
        int ltm = last_idx / TILES_N;
        int ltn = last_idx % TILES_N;
        M_SNAKE_REMAP(ltm);
        if (SNAKE_ORDER && (ltm & 1)) ltn = TILES_N - 1 - ltn;
        const int last_m = ltm * TM * 2 + cta_rank * TM;
        const int last_n = ltn * TN;

        if (warp == 0) {
            /* W0: nothing to do for drain */
        } else if (warp == 1) {
            /* W1: nothing — already committed mainloop_mbar */
        } else if (warp == 2) {
#if defined(STRIP_EPILOGUE) || defined(SELF_LOAD)
            mbar_wait(mainloop_mbar_addr + last_buf * 8, ml_phase[last_buf]);
            ml_phase[last_buf] ^= 1;
            mbar_arrive(epi_mbar_masked + last_buf * 8);
#else
            /* W2: stream the last tile through the same 2-stage circular pipe. */
            mbar_wait(mainloop_mbar_addr + last_buf * 8, ml_phase[last_buf]);
            ml_phase[last_buf] ^= 1;

            for (int si = 0; si < NUM_EPI_SUBITERS; si++) {
                const int stage = si % NUM_EPI_STAGES;
                if (load_issue_count >= NUM_EPI_STAGES) {
                    mbar_wait(consumed_mbar[stage], load_consumed_phase[stage]);
                    load_consumed_phase[stage] ^= 1;
                }
                if (lane == 0) {
                    const uint32_t res_dst = smem_to_uint(smem + OFF_STAGING + stage * EPI_STAGE_BYTES);
                    mbar_arrive_expect_tx(load_mbar[stage], EPI_STAGE_BYTES);
                    tma_load_2d_cta(res_dst, &tma_res,
                                    last_n + si * 64, last_m, load_mbar[stage]);
                }
                load_issue_count++;
            }

            mbar_arrive(epi_mbar_masked + last_buf * 8);
#endif /* STRIP_EPILOGUE/SELF_LOAD drain W2 */

        } else {
#if NUM_IDLE_WARPS > 0 && !defined(STRIP_EPILOGUE)
            if (warp >= 3 + NUM_EPI_WARPS) {
                /* Idle warps: drain — just arrive at barriers */
                mbar_wait(mainloop_mbar_addr + last_buf * 8, ml_phase[last_buf]);
                ml_phase[last_buf] ^= 1;
                mbar_arrive(epi_mbar_masked + last_buf * 8);
            } else
#endif
            {
#ifdef STRIP_EPILOGUE
            mbar_wait(mainloop_mbar_addr + last_buf * 8, ml_phase[last_buf]);
            ml_phase[last_buf] ^= 1;
            mbar_arrive(epi_mbar_masked + last_buf * 8);
#else
            /* W3-W6: epilogue for last tile (ReuseSmemC) */
            const int ew = warp - 3;
            const uint32_t bias_saddr = smem_to_uint(smem + OFF_BIAS_SMEM);
            const uint32_t xor_val = (lane & 7) << 4;
            const uint32_t sw0 = 0 ^ xor_val, sw1 = 16 ^ xor_val;
            const uint32_t sw2 = 32 ^ xor_val, sw3 = 48 ^ xor_val;
            const uint32_t sw4 = 64 ^ xor_val, sw5 = 80 ^ xor_val;
            const uint32_t sw6 = 96 ^ xor_val, sw7 = 112 ^ xor_val;

            mbar_wait(mainloop_mbar_addr + last_buf * 8, ml_phase[last_buf]);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            ml_phase[last_buf] ^= 1;

#ifdef SELF_LOAD
            uint32_t self_mbar_arr[NUM_EPI_STAGES];
            int self_mbar_phase[NUM_EPI_STAGES] = {0};
            for (int s = 0; s < NUM_EPI_STAGES; s++)
                self_mbar_arr[s] = smem_to_uint(smem + OFF_SELF_LOAD_MBAR + (ew * NUM_EPI_STAGES + s) * 8);
#endif

#ifdef LDS_DRAIN
            uint32_t drain_acc = 0;
            asm volatile("mov.u32 %0, 0;" : "=r"(drain_acc));
#endif

#if DELAY_TMA_STORE
            int have_pending = 0;
            int pend_nc, pend_stage;
#endif

#if defined(SELF_LOAD) && defined(SELF_STAGGER) && SELF_STAGGER > 0
            if (ew > 0) {
                uint32_t _ns = (uint32_t)ew * SELF_STAGGER;
                asm volatile("nanosleep.u32 %0;" :: "r"(_ns));
            }
#endif

#if CUTLASS_LOOP >= 1
            PRAGMA_UNROLL(1)
#endif
            for (int si = 0; si < NUM_EPI_SUBITERS; si++) {
                const int stage = si % NUM_EPI_STAGES;
                const int nc_base = si * 64;

#if DELAY_TMA_STORE
                if (have_pending) {
                    const int row_group = ew;
                    EPI_STORE(pend_stage, pend_nc, last_n, last_m);
                }
                if (si >= 2) {
                    EPI_WAIT(0);
                    mbar_arrive(consumed_mbar[(si - 2) % NUM_EPI_STAGES]);
                }
#endif

#ifdef SELF_LOAD
                /* Wait for 2-ago TMA store before reusing SMEM stage */
                if (si >= NUM_EPI_STAGES) {
                    if (lane == 0)
                        asm volatile("cp.async.bulk.wait_group 1;" ::: "memory");
                    __syncwarp();
                }
                /* Issue per-warp TMA load: 32 rows × 64 cols */
                if (lane == 0) {
                    const uint32_t res_dst = smem_to_uint(smem + OFF_STAGING
                        + stage * EPI_STAGE_BYTES + ew * STAGING_REGION_BYTES);
                    mbar_arrive_expect_tx(self_mbar_arr[stage], STAGING_REGION_BYTES);
                    tma_load_2d_cta(res_dst, &tma_res,
                                    last_n + si * 64, last_m + ew * 32,
                                    self_mbar_arr[stage]);
                }
                mbar_wait(self_mbar_arr[stage], self_mbar_phase[stage]);
                self_mbar_phase[stage] ^= 1;
#else
                mbar_wait(load_mbar[stage], load_phase[stage]);
                load_phase[stage] ^= 1;
#endif

                const uint32_t stage_drain = smem_to_uint(smem + OFF_STAGING
                    + stage * EPI_STAGE_BYTES + lane * 128);

#if GROUPS_PER_WARP > 1
                #pragma unroll
                for (int _rg = 0; _rg < GROUPS_PER_WARP; _rg++) {
                #ifdef EPI_STRIDED
                    const int row_group = ew + _rg * NUM_EPI_WARPS;
#else
                    const int row_group = ew * GROUPS_PER_WARP + _rg;
#endif
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
#ifdef CUTE_STORE
                char* stage_cptr = smem + OFF_STAGING
                    + stage * EPI_STAGE_BYTES
                    + row_group * STAGING_REGION_BYTES
                    + lane * 128;
#endif

#if CUTLASS_LOOP >= 2
                PRAGMA_UNROLL(1)
#endif
                for (int _ci = 0; _ci < 2; _ci++) {
#ifdef CHUNK_REORDER
                    const int chunk = (ew & 1) ? (1 - _ci) : _ci;
#else
                    const int chunk = _ci;
#endif
                    const int nc = nc_base + chunk * 32;

                    TMEM_LOAD_X32(a0,a1,a2,a3,a4,a5,a6,a7,
                                  a8,a9,a10,a11,a12,a13,a14,a15,
                                  a16,a17,a18,a19,a20,a21,a22,a23,
                                  a24,a25,a26,a27,a28,a29,a30,a31,
                                  taddr_base + nc);

#if CUTLASS_LOOP >= 3
                    {
                        const uint32_t rsw0 = chunk ? sw4 : sw0;
                        const uint32_t rsw1 = chunk ? sw5 : sw1;
                        const uint32_t rsw2 = chunk ? sw6 : sw2;
                        const uint32_t rsw3 = chunk ? sw7 : sw3;
                        char* sptr = smem + OFF_STAGING
                            + stage * EPI_STAGE_BYTES
                            + row_group * STAGING_REGION_BYTES
                            + lane * 128;

                        const char* bp = smem + OFF_BIAS_SMEM + (last_n + nc) * 2;
                        uint4 bv0 = *(const uint4*)(bp);
                        uint4 bv1 = *(const uint4*)(bp + 16);
                        uint4 bv2 = *(const uint4*)(bp + 32);
                        uint4 bv3 = *(const uint4*)(bp + 48);

                        uint4 rv0 = *(const uint4*)(sptr + rsw0);
                        uint4 rv1 = *(const uint4*)(sptr + rsw1);
                        uint4 rv2 = *(const uint4*)(sptr + rsw2);
                        uint4 rv3 = *(const uint4*)(sptr + rsw3);

                        TMEM_WAIT();

                        CPP_FP32_GROUP(a0,a1,a2,a3,a4,a5,a6,a7, bv0, rv0, sptr, rsw0);
                        CPP_FP32_GROUP(a8,a9,a10,a11,a12,a13,a14,a15, bv1, rv1, sptr, rsw1);
                        CPP_FP32_GROUP(a16,a17,a18,a19,a20,a21,a22,a23, bv2, rv2, sptr, rsw2);
                        CPP_FP32_GROUP(a24,a25,a26,a27,a28,a29,a30,a31, bv3, rv3, sptr, rsw3);
                    }
#else
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

                    /* LDS residual (swizzled, ReuseSmemC) */
                    const uint32_t rsw0 = chunk ? sw4 : sw0;
                    const uint32_t rsw1 = chunk ? sw5 : sw1;
                    const uint32_t rsw2 = chunk ? sw6 : sw2;
                    const uint32_t rsw3 = chunk ? sw7 : sw3;
                    uint4 rv0, rv1, rv2, rv3;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(rv0.x),"=r"(rv0.y),"=r"(rv0.z),"=r"(rv0.w) : "r"(stage_base + rsw0));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(rv1.x),"=r"(rv1.y),"=r"(rv1.z),"=r"(rv1.w) : "r"(stage_base + rsw1));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(rv2.x),"=r"(rv2.y),"=r"(rv2.z),"=r"(rv2.w) : "r"(stage_base + rsw2));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(rv3.x),"=r"(rv3.y),"=r"(rv3.z),"=r"(rv3.w) : "r"(stage_base + rsw3));

#ifdef CUTLASS_EPILOGUE
                    float rr0[8], rr1[8], rr2[8], rr3[8];
                    UNPACK_RES_FP32(rr0, rv0);
                    UNPACK_RES_FP32(rr1, rv1);
                    UNPACK_RES_FP32(rr2, rv2);
                    UNPACK_RES_FP32(rr3, rv3);

                    TMEM_WAIT();

                    a0+=rr0[0]; a1+=rr0[1]; a2+=rr0[2]; a3+=rr0[3];
                    a4+=rr0[4]; a5+=rr0[5]; a6+=rr0[6]; a7+=rr0[7];
                    CVT_BIAS_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7,
                        bv0.x,bv0.y,bv0.z,bv0.w, stage_base + rsw0);

                    a8+=rr1[0]; a9+=rr1[1]; a10+=rr1[2]; a11+=rr1[3];
                    a12+=rr1[4]; a13+=rr1[5]; a14+=rr1[6]; a15+=rr1[7];
                    CVT_BIAS_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15,
                        bv1.x,bv1.y,bv1.z,bv1.w, stage_base + rsw1);

                    a16+=rr2[0]; a17+=rr2[1]; a18+=rr2[2]; a19+=rr2[3];
                    a20+=rr2[4]; a21+=rr2[5]; a22+=rr2[6]; a23+=rr2[7];
                    CVT_BIAS_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23,
                        bv2.x,bv2.y,bv2.z,bv2.w, stage_base + rsw2);

                    a24+=rr3[0]; a25+=rr3[1]; a26+=rr3[2]; a27+=rr3[3];
                    a28+=rr3[4]; a29+=rr3[5]; a30+=rr3[6]; a31+=rr3[7];
                    CVT_BIAS_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31,
                        bv3.x,bv3.y,bv3.z,bv3.w, stage_base + rsw3);
#else
                    TMEM_WAIT();

#ifdef CUTE_STORE
                    BIAS_RES_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7,
                        bv0.x,bv0.y,bv0.z,bv0.w,
                        rv0.x,rv0.y,rv0.z,rv0.w, stage_cptr + rsw0);
                    BIAS_RES_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15,
                        bv1.x,bv1.y,bv1.z,bv1.w,
                        rv1.x,rv1.y,rv1.z,rv1.w, stage_cptr + rsw1);
                    BIAS_RES_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23,
                        bv2.x,bv2.y,bv2.z,bv2.w,
                        rv2.x,rv2.y,rv2.z,rv2.w, stage_cptr + rsw2);
                    BIAS_RES_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31,
                        bv3.x,bv3.y,bv3.z,bv3.w,
                        rv3.x,rv3.y,rv3.z,rv3.w, stage_cptr + rsw3);
#else
#ifdef GROUP_REORDER
#define _GR0 BIAS_RES_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7, \
    bv0.x,bv0.y,bv0.z,bv0.w, rv0.x,rv0.y,rv0.z,rv0.w, stage_base + rsw0)
#define _GR1 BIAS_RES_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15, \
    bv1.x,bv1.y,bv1.z,bv1.w, rv1.x,rv1.y,rv1.z,rv1.w, stage_base + rsw1)
#define _GR2 BIAS_RES_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23, \
    bv2.x,bv2.y,bv2.z,bv2.w, rv2.x,rv2.y,rv2.z,rv2.w, stage_base + rsw2)
#define _GR3 BIAS_RES_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31, \
    bv3.x,bv3.y,bv3.z,bv3.w, rv3.x,rv3.y,rv3.z,rv3.w, stage_base + rsw3)
                    switch (ew) {
                    case 0: _GR0; _GR1; _GR2; _GR3; break;
                    case 1: _GR1; _GR2; _GR3; _GR0; break;
                    case 2: _GR2; _GR3; _GR0; _GR1; break;
                    case 3: _GR3; _GR0; _GR1; _GR2; break;
                    }
#undef _GR0
#undef _GR1
#undef _GR2
#undef _GR3
#else
                    BIAS_RES_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7,
                        bv0.x,bv0.y,bv0.z,bv0.w,
                        rv0.x,rv0.y,rv0.z,rv0.w, stage_base + rsw0);
                    BIAS_RES_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15,
                        bv1.x,bv1.y,bv1.z,bv1.w,
                        rv1.x,rv1.y,rv1.z,rv1.w, stage_base + rsw1);
                    BIAS_RES_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23,
                        bv2.x,bv2.y,bv2.z,bv2.w,
                        rv2.x,rv2.y,rv2.z,rv2.w, stage_base + rsw2);
                    BIAS_RES_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31,
                        bv3.x,bv3.y,bv3.z,bv3.w,
                        rv3.x,rv3.y,rv3.z,rv3.w, stage_base + rsw3);
#endif /* GROUP_REORDER */
#endif /* CUTE_STORE */
#endif /* CUTLASS_EPILOGUE */
#endif /* CUTLASS_LOOP >= 3 */
                }
                } /* close row_group loop/block */

#ifdef LDS_DRAIN
                asm volatile(
                    "{  .reg .b32 __d;\n\t"
                    "   @%%p5 ld.shared.b32 __d, [%0];\n\t"
                    "   @%%p5 ld.shared.b32 __d, [%0];\n\t"
                    "   @%%p5 ld.shared.b32 __d, [%0];\n\t"
                    "   @%%p5 ld.shared.b32 __d, [%0];\n\t"
                    "}" :: "r"(stage_drain) : "memory");
#endif
#ifdef CUTLASS_EPILOGUE
                LDS_DRAIN_AND_FENCE(stage_drain);
#else
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
#if !NO_PRE_STORE_BAR
                asm volatile(BAR_EPI_SYNC ::: "memory");
#endif

#ifdef SELF_LOAD
                { const int row_group = ew;
                EPI_STORE(stage, nc_base, last_n, last_m); }
#elif DELAY_TMA_STORE
                have_pending = 1;
                pend_nc = nc_base;
                pend_stage = stage;
#else
#if GROUPS_PER_WARP > 1
                if (lane == 0) {
                    #pragma unroll
                    for (int _rg = 0; _rg < GROUPS_PER_WARP; _rg++) {
                        #ifdef EPI_STRIDED
                            const int rg = ew + _rg * NUM_EPI_WARPS;
#else
                            const int rg = ew * GROUPS_PER_WARP + _rg;
#endif
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
                if (si > 0)
                    mbar_arrive(consumed_mbar[(si - 1) % NUM_EPI_STAGES]);
                if (si == NUM_EPI_SUBITERS - 1)
                    mbar_arrive(consumed_mbar[stage]);
#endif
            }

#ifdef SELF_LOAD
            /* Drain all outstanding TMA stores */
            if (lane == 0)
                asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
            __syncwarp();
#elif DELAY_TMA_STORE
            { const int row_group = ew;
            EPI_STORE(pend_stage, pend_nc, last_n, last_m); }
            EPI_WAIT(1);
            mbar_arrive(consumed_mbar[(NUM_EPI_SUBITERS - 2) % NUM_EPI_STAGES]);
            mbar_arrive(consumed_mbar[pend_stage]);
#endif

#ifdef LDS_DRAIN
            mbar_arrive((epi_mbar_masked ^ drain_acc) + last_buf * 8);
#else
            mbar_arrive(epi_mbar_masked + last_buf * 8);
#endif
#endif /* STRIP_EPILOGUE drain W3-W6 */
            } /* close brace for idle-warp else */
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

__global__ void init_residual(__nv_bfloat16* __restrict__ res, int n_dim, long long total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int col = (int)(idx % n_dim);
        int row = (int)(idx / n_dim);
        res[idx] = __float2bfloat16((float)(row % 128) * 0.25f + (float)col * 0.125f);
    }
}

int main() {
    setbuf(stdout, NULL);
#ifdef SELF_LOAD
    printf("FC2 W3 kernel — %d warps (%d idle), SELF_LOAD epilogue (per-warp TMA)\n",
           NUM_WARPS, NUM_IDLE_WARPS);
#else
    printf("FC2 W3 kernel — %d warps (%d idle), shared-SMEM epilogue\n",
           NUM_WARPS, NUM_IDLE_WARPS);
#endif
    printf("  GEMM: [%d,%d] x [%d,%d]^T  %d-stage pipeline  SMEM: %d bytes\n",
           M_TOTAL, K_DIM, N_DIM, K_DIM, N_STAGES, SMEM_BYTES);
    printf("  EPI: stages=%d  SWS=%d  DTS=%d  FP32=%d  CPP=%d\n",
           NUM_EPI_STAGES, SINGLE_WARP_STORE, DELAY_TMA_STORE,
#ifdef FP32_EPILOGUE
           1,
#else
           0,
#endif
           CPP_EPILOGUE);

    uint8_t *d_A, *d_B;
    __nv_bfloat16 *d_bias, *d_residual, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A,        (size_t)M_TOTAL * K_DIM));
    CUDA_CHECK(cudaMalloc(&d_B,        (size_t)N_DIM   * K_DIM));
    CUDA_CHECK(cudaMalloc(&d_bias,     (size_t)N_DIM   * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_residual, (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_C,        (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16)));

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
    /* Residual: residual[row,col] = bf16((row%128)*0.25 + col*0.125) */
    {
        long long total = (long long)M_TOTAL * N_DIM;
        int tpb = 256;
        int bpg = (int)((total + tpb - 1) / tpb);
        init_residual<<<bpg, tpb>>>(d_residual, N_DIM, total);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
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

    CUtensorMap h_tma_res;
    {
        uint64_t dims[2]    = {(uint64_t)N_DIM, (uint64_t)M_TOTAL};
        uint64_t strides[1] = {(uint64_t)N_DIM * sizeof(__nv_bfloat16)};
#ifdef SELF_LOAD
        uint32_t box[2]     = {64, 32};   /* Per-warp: 32 rows × 64 cols */
#else
        uint32_t box[2]     = {64, 128};  /* W2 loads full 128 rows × 64 cols per sub-iter */
#endif
        uint32_t estrides[2]= {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(&h_tma_res,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)d_residual,
            dims, strides, box, estrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }

    CUDA_CHECK(cudaFuncSetAttribute(fc2_w3_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    printf("  TMA descriptors + func attr done (SMEM=%d B)\n", SMEM_BYTES);

#if TILE_DISPATCH == 1 || TILE_DISPATCH == 2
    int* d_tile_ctr_ptr;
    CUDA_CHECK(cudaGetSymbolAddress((void**)&d_tile_ctr_ptr, g_tile_ctr));
#define LAUNCH_KERNEL() do { \
    cudaMemsetAsync(d_tile_ctr_ptr, 0, sizeof(int)); \
    fc2_w3_kernel<<<SM_COUNT, THREADS, SMEM_BYTES>>>( \
        h_tma_a, h_tma_b, h_tma_c, d_bias, d_C, d_residual, h_tma_res); \
} while(0)
#elif TILE_DISPATCH == 3
    /* Grid-based: x=CTA rank (0..1), y=tile index. cluster_dims(2,1,1)
       pairs CTAs 0,1 per tile into a cluster. */
    dim3 grid_dim(2, TOTAL_TILES, 1);
#define LAUNCH_KERNEL() \
    fc2_w3_kernel<<<grid_dim, THREADS, SMEM_BYTES>>>( \
        h_tma_a, h_tma_b, h_tma_c, d_bias, d_C, d_residual, h_tma_res)
#else
#define LAUNCH_KERNEL() \
    fc2_w3_kernel<<<SM_COUNT, THREADS, SMEM_BYTES>>>( \
        h_tma_a, h_tma_b, h_tma_c, d_bias, d_C, d_residual, h_tma_res)
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
    printf("FC2-W3 kernel: %.3f ms  %.2f TFLOPS\n",
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

    int errors = 0;
    for (int spot = 0; spot < 32; spot++) {
        long long row = (long long)spot * M_TOTAL / 32;
        int col = (spot * 47) % N_DIM;
        float b_val = (col & 1) ? 1.0f : 1.5f;
        float gemm = (float)K_DIM * 1.5f * b_val;
        float res_bf16_f = __bfloat162float(__float2bfloat16(
            (float)((int)row % 128) * 0.25f + (float)col * 0.125f));
        float bias_bf16_f = __bfloat162float(__float2bfloat16((float)(col + 1)));
#ifdef FP32_EPILOGUE
        /* FP32 path: bf16(gemm + bias_fp32 + residual_fp32) — single final rounding */
        __nv_bfloat16 expected = __float2bfloat16(gemm + bias_bf16_f + res_bf16_f);
#else
        /* BF16 path: bf16(bf16(gemm) + bf16(bias)) + residual — 3 roundings */
        float acc_rounded = __bfloat162float(__float2bfloat16(gemm));
        float after_bias = __bfloat162float(__float2bfloat16(acc_rounded + bias_bf16_f));
        __nv_bfloat16 expected = __float2bfloat16(after_bias + res_bf16_f);
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
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_bias); cudaFree(d_residual); cudaFree(d_C);
    return 0;
}
