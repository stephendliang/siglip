/*
kernel_common.cuh — shared infrastructure for tcgen05 persistent GEMM kernels
B200 (SM100a), cta_group::2, __cluster_dims__(2,1,1)
Warp-specialized: Load(W0) | MMA(W1) | Epilogue(W2+)

Usage: #define N_DIM and K_DIM before including this header.
*/

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Hardware
#define SM_COUNT       148

// Tuning parameters (overridable via -D at compile time)
#ifndef NUM_EPI_WARPS
#define NUM_EPI_WARPS  4
#endif
#ifndef STAGGER_CYCLES
#define STAGGER_CYCLES 80   // F31: per-warp Phase 1 stagger (sweep: 50, 80, 100, 200)
#endif
#ifndef TMEM_LOAD_WIDTH
#define TMEM_LOAD_WIDTH 32   // 32=1×x32 per 32-col chunk (default), 16=2×x16, 64=1×x64
#endif
#ifndef INTERLEAVE_STRATEGY
#define INTERLEAVE_STRATEGY 2  // 0=all-at-end(CUTLASS-style), 1=per-region, 2=half-batch, 3=three-plus-one
#endif
#ifndef MBAR_EARLY
#define MBAR_EARLY 0           // 0=after Phase 1, 1=after last TMEM_WAIT
#endif
#ifndef PHASE1_UNROLL
#define PHASE1_UNROLL  2
#endif
#ifndef SNAKE_ORDER
#define SNAKE_ORDER 1
#endif
#ifndef N_STAGES
#define N_STAGES       4
#endif
#ifndef K_LOOP_UNROLL
#define K_LOOP_UNROLL  N_STAGES
#endif
#ifndef W0_LOOP_UNROLL
#define W0_LOOP_UNROLL  0    // W0 load K-iter loop: 0=no pragma, 1=no unroll, N=unroll by N
#endif
#ifndef W0_RES_PREFETCH
#define W0_RES_PREFETCH 0    // 0=off, 1=W0 prefetches residual after K-loop (fc2 only)
#endif
#ifndef W0_RES_FULL
#define W0_RES_FULL 0        // 0=off, 1=W0 loads ALL residual (both passes) after K-loop (fc2 only)
#endif
#if W0_RES_PREFETCH && !TMA_RESIDUAL
#error "W0_RES_PREFETCH requires TMA_RESIDUAL >= 1"
#endif
#if W0_RES_FULL && !TMA_RESIDUAL
#error "W0_RES_FULL requires TMA_RESIDUAL >= 1"
#endif
#if W0_RES_FULL && W0_RES_PREFETCH
#error "W0_RES_FULL and W0_RES_PREFETCH are mutually exclusive"
#endif
#ifndef SUB_MMA_UNROLL
#define SUB_MMA_UNROLL  0    // Sub-MMA inner loop: 0=no pragma, 1=no unroll, N=unroll by N
#endif
#ifndef PRELOAD_MODE
#define PRELOAD_MODE 1         // 0=no preload, 1=partial (8 bias), 2=full (all side-data before TMEM_WAIT)
#endif
#ifndef PREFETCH_BEFORE_STORE
#define PREFETCH_BEFORE_STORE 0  // 0=after TMA stores, 1=before TMA stores
#endif
#ifndef GELU_VARIANT
#define GELU_VARIANT 0           // 0=asm tanh, 1=tanhf, 2=tanhf+half_x, 3=fmaf, 4=batch8-asm, 5=batch4+4-asm, 6=batch8-tanhf
#endif
#ifndef TMA_RESIDUAL
#define TMA_RESIDUAL 0           // 0=__ldg residual (default), 1=TMA residual via SMEM staging, 2=TMA preloaded before mainloop wait
#endif
#ifndef DEFERRED_WAIT
#define DEFERRED_WAIT 0          // 0=wait_group before TMEM load (default), 1=wait_group after TMEM load + residual mbar_wait
#endif
#ifndef BATCH_EPILOGUE
#define BATCH_EPILOGUE 0         // 0=fused compute+CVT+STS per 8 elems, 1=compute all 32 then batch CVT+STS
#endif
#ifndef GELU_VECTOR_WIDTH
#define GELU_VECTOR_WIDTH 32     // BATCH_EPILOGUE GELU batch size: 8=per-group, 16=two-batch, 32=full-batch
#endif
#ifndef STORE_TIMING
#define STORE_TIMING 0           // 0=inline TMA stores per INTERLEAVE_STRATEGY, 1=all stores after Phase 1
#endif
#ifndef EPILOGUE_LOOP
#define EPILOGUE_LOOP 0          // 0=unrolled epilogue, 1=#pragma unroll 1 loop body
#endif
#ifndef STS_WIDTH
#define STS_WIDTH 16             // 16=1x st.shared.v4 per call, 32=2x st.shared.v4 per call
#endif
#ifndef EPI_SYNC
#define EPI_SYNC 0               // 0=independent warp poll, 1=bar.sync before epilogue
#endif
#ifndef NUM_PASSES_PARAM
#define NUM_PASSES_PARAM 0       // 0=auto (128 cols/pass), 4=4-pass (64 cols/pass), FC2 TMA_RESIDUAL only
#endif
#ifndef BIAS_SMEM
#define BIAS_SMEM 0              // 0=LDG bias per-chunk, 1=load bias to SMEM once per tile (fc2/fc1 only)
#endif
#ifndef BIAS_BF16
#define BIAS_BF16 0              // 0=FP32 bias, 1=BF16 bias with bf16x2 epilogue arithmetic (fc2 only)
#endif

#if EPILOGUE_LOOP
#undef PHASE1_UNROLL
#define PHASE1_UNROLL 1
#endif

// nvcc doesn't expand macros in #pragma unroll — use _Pragma instead
#define _UNROLL_STR2(x) #x
#define _UNROLL_STR(x) _UNROLL_STR2(unroll x)
#define PRAGMA_UNROLL(n) _Pragma(_UNROLL_STR(n))

// Conditional unroll macros — 0 means no pragma (compiler decides)
#if W0_LOOP_UNROLL > 0
#define MAYBE_UNROLL_W0 _Pragma(_UNROLL_STR(W0_LOOP_UNROLL))
#else
#define MAYBE_UNROLL_W0
#endif
#if SUB_MMA_UNROLL > 0
#define MAYBE_UNROLL_SUB _Pragma(_UNROLL_STR(SUB_MMA_UNROLL))
#else
#define MAYBE_UNROLL_SUB
#endif

// Thread config
#define THREADS        (32 * (2 + NUM_EPI_WARPS))

/*
Problem dimensions — N_DIM must be defined before including this header
*/
#ifndef N_DIM
#error "Define N_DIM before including kernel_common.cuh"
#endif
#define BATCH_SIZE     4736
#define SEQ_LEN        196
#define M_TOTAL        928256
#ifndef K_DIM
#error "Define K_DIM before including kernel_common.cuh"
#endif

// Tile dimensions
#define TM             128
#define TN             256
#define TK             128
#define TILES_M        ((M_TOTAL + TM * 2 - 1) / (TM * 2))    // 3626
#define TILES_N        (N_DIM / TN)
#define TOTAL_TILES    (TILES_M * TILES_N)
#define K_ITERS        (K_DIM / TK)
#define MMA_K          32
#define MMA_PER_KI     (TK / MMA_K)                             // 4

// Pipeline / SMEM layout
#define STAGE_BYTES    32768                                      // 16KB A + 16KB B per stage
#define OFF_TMEM           (N_STAGES * STAGE_BYTES)
#define OFF_TMA_MBAR       (OFF_TMEM + 8)
#define OFF_MMA_MBAR       (OFF_TMA_MBAR + N_STAGES * 8)
#define OFF_MAINLOOP_MBAR  (OFF_MMA_MBAR + N_STAGES * 8)
#define OFF_EPILOGUE_MBAR  (OFF_MAINLOOP_MBAR + 16)
#if BIAS_SMEM
#if BIAS_BF16
#define BIAS_SMEM_BYTES    (TN * 2)          /* 256 bf16 = 512 bytes */
#else
#define BIAS_SMEM_BYTES    (TN * 4)          /* 256 floats = 1024 bytes */
#endif
#else
#define BIAS_SMEM_BYTES    0
#endif
#if TMA_RESIDUAL
#define OFF_RES_MBAR       (OFF_EPILOGUE_MBAR + 16)
#if W0_RES_FULL
#define OFF_RES_CONSUMED_MBAR  (OFF_RES_MBAR + NUM_EPI_WARPS * 8)
#define OFF_RES_PASS_MBAR      (OFF_RES_CONSUMED_MBAR + 8)
#define _MBAR_END              (OFF_RES_PASS_MBAR + 8)
#elif W0_RES_PREFETCH
#define OFF_RES_CONSUMED_MBAR  (OFF_RES_MBAR + NUM_EPI_WARPS * 8)
#define _MBAR_END          (OFF_RES_CONSUMED_MBAR + 8)
#else
#define _MBAR_END          (OFF_RES_MBAR + NUM_EPI_WARPS * 8)
#endif
#define RES_STAGING_OFFSET (2 * STAGING_REGION_BYTES)   // residual regions start after 2 output regions per warp
#else
#define _MBAR_END          (OFF_EPILOGUE_MBAR + 16)
#endif
#if BIAS_SMEM
#define OFF_BIAS_SMEM      ((_MBAR_END + 15) & ~15)                       // 16-align for ld.shared.v4.b32
#define OFF_STAGING        ((OFF_BIAS_SMEM + BIAS_SMEM_BYTES + 1023) & ~1023)  // 1024-align for SWIZZLE_128B
#else
#define OFF_STAGING        ((_MBAR_END + 1023) & ~1023)  // 1024-align for SWIZZLE_128B
#endif
#define STAGING_REGION_ROW_BYTES  128                                               // 64 BF16 cols = 128 bytes (SWIZZLE_128B)
#define STAGING_REGION_BYTES      (32 * STAGING_REGION_ROW_BYTES)                   // 4096 bytes per region (32 rows x 128B)
#define STAGING_WARP_BYTES        (4 * STAGING_REGION_BYTES)                         // 16384 bytes per warp (4 regions x 4096)
#define SMEM_BYTES                ((OFF_STAGING + NUM_EPI_WARPS * STAGING_WARP_BYTES + 127) & ~127)

// WGMMA / TMEM constants
#define TMEM_COLS      512
#define IDESC          0x10400010U
#define SBO            1024
#define TMA_BYTES      32768

// Timing instrumentation
#ifdef TIMING
#define TIMING_CLUSTER_STRIDE 32
#define MAX_SPREAD_TILES 148
#endif

// Error check macros

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

// Device helpers

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
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);            // bits [13:0]  base addr
    d |= (uint64_t)((SBO  & 0x3FFFF) >> 4) << 32;      // bits [45:32] SBO (stride_byte_offset)
    d |= (1ULL << 46);                                  // bit  [46]    LBO
    d |= (2ULL << 61);                                  // bits [63:61] SWIZZLE_128B
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

/*
K-iteration macro (accumulating, for ki >= 1)
Used for ki=1..K_ITERS-1 where accumulator is already initialized.
S is the stage index (0..N_STAGES-1); works with runtime values but best with constants.
*/
#define K_ITER_ACCUM(S) do { \
    mbar_wait(tma_mbar[S], tma_phase[S]); \
    tma_phase[S] ^= 1; \
    asm volatile("tcgen05.fence::after_thread_sync;"); \
    { \
        uint64_t da_ = desc_a_base[S], db_ = desc_b_base[S]; \
        asm volatile( \
            "{\n\t" \
            ".reg .pred p;\n\t" \
            "setp.ne.b32 p, 1, 0;\n\t" \
            "tcgen05.mma.cta_group::2.kind::f8f6f4 " \
            "[%0], %1, %2, %3, {%4,%5,%6,%7, %8,%9,%10,%11}, p;\n\t" \
            "}" \
            : \
            : "r"(buf * TN), "l"(da_), "l"(db_), "r"(IDESC), \
              "r"(0),"r"(0),"r"(0),"r"(0), "r"(0),"r"(0),"r"(0),"r"(0)); \
        MAYBE_UNROLL_SUB \
        for (int sub_ = 1; sub_ < MMA_PER_KI; sub_++) { \
            da_ += 2; db_ += 2; \
            asm volatile( \
                "{\n\t" \
                ".reg .pred p;\n\t" \
                "setp.ne.b32 p, 1, 0;\n\t" \
                "tcgen05.mma.cta_group::2.kind::f8f6f4 " \
                "[%0], %1, %2, %3, {%4,%5,%6,%7, %8,%9,%10,%11}, p;\n\t" \
                "}" \
                : \
                : "r"(buf * TN), "l"(da_), "l"(db_), "r"(IDESC), \
                  "r"(0),"r"(0),"r"(0),"r"(0), "r"(0),"r"(0),"r"(0),"r"(0)); \
        } \
    } \
    tcgen05_commit_mcast(mma_mbar[S], 0x3); \
} while(0)

// TMEM load macros

#define TMEM_LOAD(r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15, TADDR) \
    asm volatile( \
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 " \
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];" \
        : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3), \
          "=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7), \
          "=f"(r8),"=f"(r9),"=f"(r10),"=f"(r11), \
          "=f"(r12),"=f"(r13),"=f"(r14),"=f"(r15) \
        : "r"(TADDR))

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

#define TMEM_LOAD_X64(r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15, \
                      r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31, \
                      r32,r33,r34,r35,r36,r37,r38,r39,r40,r41,r42,r43,r44,r45,r46,r47, \
                      r48,r49,r50,r51,r52,r53,r54,r55,r56,r57,r58,r59,r60,r61,r62,r63, TADDR) \
    asm volatile( \
        "tcgen05.ld.sync.aligned.32x32b.x64.b32 " \
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31," \
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47," \
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];" \
        : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3), \
          "=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7), \
          "=f"(r8),"=f"(r9),"=f"(r10),"=f"(r11), \
          "=f"(r12),"=f"(r13),"=f"(r14),"=f"(r15), \
          "=f"(r16),"=f"(r17),"=f"(r18),"=f"(r19), \
          "=f"(r20),"=f"(r21),"=f"(r22),"=f"(r23), \
          "=f"(r24),"=f"(r25),"=f"(r26),"=f"(r27), \
          "=f"(r28),"=f"(r29),"=f"(r30),"=f"(r31), \
          "=f"(r32),"=f"(r33),"=f"(r34),"=f"(r35), \
          "=f"(r36),"=f"(r37),"=f"(r38),"=f"(r39), \
          "=f"(r40),"=f"(r41),"=f"(r42),"=f"(r43), \
          "=f"(r44),"=f"(r45),"=f"(r46),"=f"(r47), \
          "=f"(r48),"=f"(r49),"=f"(r50),"=f"(r51), \
          "=f"(r52),"=f"(r53),"=f"(r54),"=f"(r55), \
          "=f"(r56),"=f"(r57),"=f"(r58),"=f"(r59), \
          "=f"(r60),"=f"(r61),"=f"(r62),"=f"(r63) \
        : "r"(TADDR))

#if TMEM_LOAD_WIDTH == 32
#define LOAD_32_COLS(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15, \
                     a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31, TADDR) \
    TMEM_LOAD_X32(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15, \
                  a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31, TADDR)
#else
#define LOAD_32_COLS(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15, \
                     a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31, TADDR) \
    TMEM_LOAD(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15, TADDR); \
    TMEM_LOAD(a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31, (TADDR) + 16)
#endif

#define TMEM_WAIT() \
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory")

#define BF16X2_TO_F32(reg, flo, fhi) \
    asm volatile("{\n\t" \
        ".reg .b16 lo, hi;\n\t" \
        "mov.b32 {lo, hi}, %2;\n\t" \
        "cvt.rn.f32.bf16 %0, lo;\n\t" \
        "cvt.rn.f32.bf16 %1, hi;\n\t" \
        "}" : "=f"(flo), "=f"(fhi) : "r"(reg))

/* Non-volatile BF16x2 unpack — compiler can reorder freely */
#define BF16X2_TO_F32_NV(reg, flo, fhi) \
    asm("{\n\t" \
        ".reg .b16 lo, hi;\n\t" \
        "mov.b32 {lo, hi}, %2;\n\t" \
        "cvt.rn.f32.bf16 %0, lo;\n\t" \
        "cvt.rn.f32.bf16 %1, hi;\n\t" \
        "}" : "=f"(flo), "=f"(fhi) : "r"(reg))

/* 8-float CVT+STS: pack 8 FP32 → 4 BF16x2 → 1x st.shared.v4 (16 bytes) */
static __device__ __forceinline__ void cvt_sts_v4(
    float g0, float g1, float g2, float g3,
    float g4, float g5, float g6, float g7,
    uint32_t saddr
) {
    uint32_t o0, o1, o2, o3;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o0) : "f"(g0), "f"(g1));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o1) : "f"(g2), "f"(g3));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o2) : "f"(g4), "f"(g5));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o3) : "f"(g6), "f"(g7));
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(saddr), "r"(o0), "r"(o1), "r"(o2), "r"(o3) : "memory");
}

/* 16-float CVT+STS: pack 16 FP32 → 8 BF16x2 → 2x st.shared.v4 (32 bytes).
   Two addresses needed because SWIZZLE_128B makes addr and addr+16 non-contiguous. */
static __device__ __forceinline__ void cvt_sts_v8(
    float g0, float g1, float g2, float g3,
    float g4, float g5, float g6, float g7,
    float g8, float g9, float g10, float g11,
    float g12, float g13, float g14, float g15,
    uint32_t saddr0, uint32_t saddr1
) {
    uint32_t o0, o1, o2, o3, o4, o5, o6, o7;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o0) : "f"(g0), "f"(g1));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o1) : "f"(g2), "f"(g3));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o2) : "f"(g4), "f"(g5));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o3) : "f"(g6), "f"(g7));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o4) : "f"(g8), "f"(g9));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o5) : "f"(g10), "f"(g11));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o6) : "f"(g12), "f"(g13));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o7) : "f"(g14), "f"(g15));
    asm volatile(
        "st.shared.v4.b32 [%0], {%2,%3,%4,%5};\n\t"
        "st.shared.v4.b32 [%1], {%6,%7,%8,%9};"
        :: "r"(saddr0), "r"(saddr1),
           "r"(o0),"r"(o1),"r"(o2),"r"(o3),
           "r"(o4),"r"(o5),"r"(o6),"r"(o7) : "memory");
}

#define CVT_STS(r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15, SADDR) \
    asm volatile( \
        "{\n\t" \
        ".reg .b32 b0,b1,b2,b3,b4,b5,b6,b7;\n\t" \
        "cvt.rn.bf16x2.f32 b0, %1, %0;\n\t" \
        "cvt.rn.bf16x2.f32 b1, %3, %2;\n\t" \
        "cvt.rn.bf16x2.f32 b2, %5, %4;\n\t" \
        "cvt.rn.bf16x2.f32 b3, %7, %6;\n\t" \
        "cvt.rn.bf16x2.f32 b4, %9, %8;\n\t" \
        "cvt.rn.bf16x2.f32 b5, %11, %10;\n\t" \
        "cvt.rn.bf16x2.f32 b6, %13, %12;\n\t" \
        "cvt.rn.bf16x2.f32 b7, %15, %14;\n\t" \
        "st.shared.v4.b32 [%16], {b0,b1,b2,b3};\n\t" \
        "st.shared.v4.b32 [%16+16], {b4,b5,b6,b7};\n\t" \
        "}" \
        :: "f"(r0),"f"(r1),"f"(r2),"f"(r3), \
           "f"(r4),"f"(r5),"f"(r6),"f"(r7), \
           "f"(r8),"f"(r9),"f"(r10),"f"(r11), \
           "f"(r12),"f"(r13),"f"(r14),"f"(r15), \
           "r"(SADDR) \
        : "memory")
