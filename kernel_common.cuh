// kernel_common.cuh — shared infrastructure for tcgen05 persistent megakernels
// B200 (SM100a), cta_group::2, __cluster_dims__(2,1,1)
// Warp-specialized: Load(W0) | MMA(W1) | Epilogue(W2+)
//
// Usage: #define N_DIM <value> before including this header.

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// ── Hardware ──────────────────────────────────────────────────
#define SM_COUNT       148

// ── Tuning parameters (overridable via -D at compile time) ───
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
#define INTERLEAVE_STRATEGY 2  // 0=all-at-end, 1=per-region, 2=half-batch, 3=three-plus-one
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

// nvcc doesn't expand macros in #pragma unroll — use _Pragma instead
#define _UNROLL_STR2(x) #x
#define _UNROLL_STR(x) _UNROLL_STR2(unroll x)
#define PRAGMA_UNROLL(n) _Pragma(_UNROLL_STR(n))

// ── Thread config ─────────────────────────────────────────────
#define THREADS        (32 * (2 + NUM_EPI_WARPS))

// ── Problem dimensions ────────────────────────────────────────
// N_DIM must be defined before including this header
#ifndef N_DIM
#error "Define N_DIM before including kernel_common.cuh"
#endif
#define BATCH_SIZE     4736
#define SEQ_LEN        196
#define M_TOTAL        928256
#define K_DIM          768

// ── Tile dimensions ───────────────────────────────────────────
#define TM             128
#define TN             256
#define TK             128
#define TILES_M        ((M_TOTAL + TM * 2 - 1) / (TM * 2))    // 3626
#define TILES_N        (N_DIM / TN)
#define TOTAL_TILES    (TILES_M * TILES_N)
#define K_ITERS        (K_DIM / TK)                             // 6
#define MMA_K          32
#define MMA_PER_KI     (TK / MMA_K)                             // 4

// ── Pipeline / SMEM layout ────────────────────────────────────
#define STAGE_BYTES    32768                                      // 16KB A + 16KB B per stage
#define OFF_TMEM           (N_STAGES * STAGE_BYTES)
#define OFF_TMA_MBAR       (OFF_TMEM + 8)
#define OFF_MMA_MBAR       (OFF_TMA_MBAR + N_STAGES * 8)
#define OFF_MAINLOOP_MBAR  (OFF_MMA_MBAR + N_STAGES * 8)
#define OFF_EPILOGUE_MBAR  (OFF_MAINLOOP_MBAR + 16)
#define OFF_STAGING        ((OFF_EPILOGUE_MBAR + 16 + 1023) & ~1023)  // 1024-align for SWIZZLE_128B (addr[6:4] ^= addr[9:7])
#define STAGING_REGION_ROW_BYTES  128                                               // 64 BF16 cols = 128 bytes (SWIZZLE_128B)
#define STAGING_REGION_BYTES      (32 * STAGING_REGION_ROW_BYTES)                   // 4096 bytes per region (32 rows x 128B)
#define STAGING_WARP_BYTES        (4 * STAGING_REGION_BYTES)                         // 16384 bytes per warp (4 regions x 4096)
#define SMEM_BYTES                ((OFF_STAGING + NUM_EPI_WARPS * STAGING_WARP_BYTES + 127) & ~127)

// ── WGMMA / TMEM constants ───────────────────────────────────
#define TMEM_COLS      512
#define IDESC          0x10400010U
#define SBO            1024
#define TMA_BYTES      32768

// ── Timing instrumentation ────────────────────────────────────
#ifdef TIMING
#define TIMING_CLUSTER_STRIDE 32
#define MAX_SPREAD_TILES 148
#endif

// ── Error check macros ────────────────────────────────────────

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

// ── Device helpers ────────────────────────────────────────────

static __device__ __forceinline__
uint32_t smem_to_uint(const void* p) {
    uint32_t r;
    asm volatile("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }"
        : "=r"(r) : "l"(p));
    return r;
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
void tcgen05_commit_mcast(uint32_t mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        :: "r"(mbar_addr), "h"(cta_mask) : "memory");
}

// ── K-iteration macro (accumulating, for ki >= 1) ─────────────
// Used for ki=1..K_ITERS-1 where accumulator is already initialized.
// S is the stage index (0..N_STAGES-1); works with runtime values but best with constants.
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

// ── TMEM load macros ──────────────────────────────────────────

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
