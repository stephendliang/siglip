// Hand-tuned from gen.py output — TK=128, SWIZZLE_128B, 4-stage pipeline
// Target: B200  Batch: 4736  GEMM: [928256,768]×[768,768]^T
// Pipeline: 4-stage (parameterized)  K-iters: 6  MMA/iter: 4  idesc: 0x10400010
// Warps: 2+NUM_EPI_WARPS  cta_group::2  __cluster_dims__(2,1,1)
// Warp-specialized: Load(W0) | MMA(W1,cta_group::2,CTA0 only) | Epilogue(W2+,x32 TMEM ld,interleaved TMA stores)  BF16 output
// tcgen05.mma.cta_group::2.kind::f8f6f4  (E4M3 × E4M3 → FP32)
// Each CTA loads own A (128 rows) + half B (128 cols). MMA produces 256×256 output.

#include <cuda.h>
#include <cuda_bf16.h>
#include <curand.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define SM_COUNT       148
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
#ifndef CVT_ADD_FUSED
#define CVT_ADD_FUSED 1    // 1=fused asm (asm-local regs), 0=C++ intrinsics (global regs)
#endif
// nvcc doesn't expand macros in #pragma unroll — use _Pragma instead
#define _UNROLL_STR2(x) #x
#define _UNROLL_STR(x) _UNROLL_STR2(unroll x)
#define PRAGMA_UNROLL(n) _Pragma(_UNROLL_STR(n))
#define THREADS        (32 * (2 + NUM_EPI_WARPS))
#define BATCH_SIZE     4736
#define SEQ_LEN        196
#define M_TOTAL        928256
#define N_DIM          768
#define K_DIM          768
#define TM             128
#define TN             256
#define TK             128
#define TILES_M        3626
#define TILES_N        3
#define K_ITERS        6
#define TOTAL_TILES    10878
#ifndef N_STAGES
#define N_STAGES       4
#endif
#define STAGE_BYTES    32768                                      // 16KB A + 16KB B per stage
#define OFF_TMEM           (N_STAGES * STAGE_BYTES)
#define OFF_TMA_MBAR       (OFF_TMEM + 8)
#define OFF_MMA_MBAR       (OFF_TMA_MBAR + N_STAGES * 8)
#define OFF_MAINLOOP_MBAR  (OFF_MMA_MBAR + N_STAGES * 8)
#define OFF_EPILOGUE_MBAR  (OFF_MAINLOOP_MBAR + 16)
#define OFF_STAGING        ((OFF_EPILOGUE_MBAR + 16 + 127) & ~127)
#define STAGING_REGION_ROW_BYTES  128                                               // 64 BF16 cols = 128 bytes (SWIZZLE_128B)
#define STAGING_REGION_BYTES      (32 * STAGING_REGION_ROW_BYTES)                   // 4096 bytes per region (32 rows x 128B)
#define STAGING_WARP_BYTES        (4 * STAGING_REGION_BYTES)                         // 16384 bytes per warp (4 regions x 4096)
#define SMEM_BYTES                ((OFF_STAGING + NUM_EPI_WARPS * STAGING_WARP_BYTES + 127) & ~127)
#define TMEM_COLS      512
#define IDESC          0x10400010U
#define SBO            1024
#define TMA_BYTES      32768
#define MMA_K          32
#define MMA_PER_KI     4

#ifdef TIMING
#define TIMING_CLUSTER_STRIDE 32
#define MAX_SPREAD_TILES 148
#endif

#define COMB_PADDED_ROWS   224          // ceil(196/32)*32 = 7*32
#define COMB_BLOCK_ROWS    32
#define COMB_BLOCK_COLS    32
#define COMB_ROW_BLOCKS    7            // 224/32
#define COMB_COL_BLOCKS    24           // 768/32
#define COMB_BLOCK_ELEMS   1024         // 32*32

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

// ── Device helpers ──────────────────────────────────────────

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
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
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

// ── F28: Accumulating K-iteration macro (all sub-MMAs accumulate) ──
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

// ── Pipelined epilogue macros ────────────────────────────────

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

__device__ __forceinline__ uint32_t cvt_add_bf16x2(float lo, float hi, uint32_t combined) {
    __nv_bfloat162 acc = __floats2bfloat162_rn(lo, hi);
    __nv_bfloat162 comb;
    memcpy(&comb, &combined, sizeof(uint32_t));
    acc = __hadd2(acc, comb);
    uint32_t result;
    memcpy(&result, &acc, sizeof(uint32_t));
    return result;
}

#define STS_V4(b0,b1,b2,b3, SADDR) \
    asm volatile( \
        "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" \
        :: "r"(SADDR), "r"(b0), "r"(b1), "r"(b2), "r"(b3) : "memory")

// ── Fused CVT+ADD+STS macro — asm-local .reg intermediates avoid global register inflation ──
// Converts 4 pairs of FP32 accumulators to BF16x2, adds combined BF16x2, stores 16B to SMEM.
// With CVT_ADD_FUSED=1, b0-b3 are .reg locals invisible to ptxas's register allocator.
// With CVT_ADD_FUSED=0, falls back to cvt_add_bf16x2() + STS_V4 (global regs).

#if CVT_ADD_FUSED
#define CVT_ADD_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, c0,c1,c2,c3, SADDR) \
    asm volatile( \
        "{\n\t" \
        ".reg .b32 b0, b1, b2, b3;\n\t" \
        "cvt.rn.bf16x2.f32 b0, %1, %0;\n\t" \
        "add.rn.bf16x2 b0, b0, %8;\n\t" \
        "cvt.rn.bf16x2.f32 b1, %3, %2;\n\t" \
        "add.rn.bf16x2 b1, b1, %9;\n\t" \
        "cvt.rn.bf16x2.f32 b2, %5, %4;\n\t" \
        "add.rn.bf16x2 b2, b2, %10;\n\t" \
        "cvt.rn.bf16x2.f32 b3, %7, %6;\n\t" \
        "add.rn.bf16x2 b3, b3, %11;\n\t" \
        "st.shared.v4.b32 [%12], {b0,b1,b2,b3};\n\t" \
        "}" \
        :: "f"(f0),"f"(f1),"f"(f2),"f"(f3), \
           "f"(f4),"f"(f5),"f"(f6),"f"(f7), \
           "r"(c0),"r"(c1),"r"(c2),"r"(c3), \
           "r"(SADDR) \
        : "memory")
#else
#define CVT_ADD_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, c0,c1,c2,c3, SADDR) \
    do { \
        uint32_t _b0 = cvt_add_bf16x2(f0, f1, c0); \
        uint32_t _b1 = cvt_add_bf16x2(f2, f3, c1); \
        uint32_t _b2 = cvt_add_bf16x2(f4, f5, c2); \
        uint32_t _b3 = cvt_add_bf16x2(f6, f7, c3); \
        STS_V4(_b0, _b1, _b2, _b3, SADDR); \
    } while(0)
#endif

// ── Epilogue: TMEM → inline BF16 add → CVT → 4 swizzle SMEM regions → TMA tensor stores ──
// Phase 1: TMEM readback (all 256 cols) + combined add + CVT → 4 swizzle regions (SWIZZLE_128B)
// Phase 2: 4 TMA tensor stores from swizzle regions → global C

template<int NC_START, int NC_END>
static __device__ __forceinline__
void epilogue_store(
    uint32_t tmem_addr,
    int row_group,
    int lane,
    int gm_base,
    int n_start,
    const __nv_bfloat16* __restrict__ combined,
    int pos_row,
    __nv_bfloat16* __restrict__ C,
    int cta_rank,
    uint32_t staging_saddr,
    uint32_t epi_mbar_addr,
    const CUtensorMap* tma_c_desc
#ifdef TIMING
    , long long& t_phase1_end
#endif
) {
    const int taddr_base = tmem_addr + ((cta_rank * 128 + row_group * 32) << 16);
    const int comb_block_row = pos_row / COMB_BLOCK_ROWS;
    const int comb_row_in_blk = pos_row % COMB_BLOCK_ROWS;
    const __nv_bfloat16* comb_base = combined
        + (long long)comb_block_row * COMB_COL_BLOCKS * COMB_BLOCK_ELEMS
        + comb_row_in_blk * COMB_BLOCK_COLS;

    // 4 swizzle regions per warp, each 32 rows x 64 cols BF16 (SWIZZLE_128B)
    // Only declare staging_rN when used (strategies 0 and 3) to avoid unused-variable warnings.
#if INTERLEAVE_STRATEGY == 0
    const uint32_t staging_r0 = staging_saddr;
    const uint32_t staging_r1 = staging_saddr + STAGING_REGION_BYTES;
    const uint32_t staging_r2 = staging_saddr + 2 * STAGING_REGION_BYTES;
    const uint32_t staging_r3 = staging_saddr + 3 * STAGING_REGION_BYTES;
#elif INTERLEAVE_STRATEGY == 3
    const uint32_t staging_r3 = staging_saddr + 3 * STAGING_REGION_BYTES;
#endif

    // Wait for previous tile's Phase 2 TMA stores before overwriting staging.
    // After ml_wait (~1,342 cycles), TMA stores are long done — this is a true no-op.
    if (lane == 0) {
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
    __syncwarp();

    // Swizzle constants (loop-invariant)
    const uint32_t xor_val = (lane & 7) << 4;
    const uint32_t srow_base = staging_saddr + lane * STAGING_REGION_ROW_BYTES;

#if TMEM_LOAD_WIDTH == 64
    float a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15;
    float a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31;
    float a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44,a45,a46,a47;
    float a48,a49,a50,a51,a52,a53,a54,a55,a56,a57,a58,a59,a60,a61,a62,a63;

    // ═══ Phase 1: all 256 cols → 4 swizzle regions, x64 stride ═══
    TMEM_LOAD_X64(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                  a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                  a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44,a45,a46,a47,
                  a48,a49,a50,a51,a52,a53,a54,a55,a56,a57,a58,a59,a60,a61,a62,a63,
                  taddr_base + NC_START);

    PRAGMA_UNROLL(PHASE1_UNROLL)
    for (int nc = NC_START; nc < NC_END; nc += 64) {
        const uint32_t srow = srow_base + ((nc - NC_START) >> 6) * STAGING_REGION_BYTES;

        // Combined loads (fills TMEM latency window)
        const __nv_bfloat16* comb_ptr = comb_base + (long long)((n_start + nc) / COMB_BLOCK_COLS) * COMB_BLOCK_ELEMS;
        uint4 craw0 = *reinterpret_cast<const uint4*>(comb_ptr);
        uint4 craw1 = *reinterpret_cast<const uint4*>(comb_ptr + 8);

        TMEM_WAIT();

        // F40: MBAR_EARLY — signal TMEM free immediately (data now in registers)
        if (MBAR_EARLY && nc + 64 >= NC_END) {
            if (epi_mbar_addr) mbar_arrive(epi_mbar_addr);
        }

        // First 32 cols: a0..a31 → swizzle region (bytes 0-63)
        {
            CVT_ADD_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7, craw0.x,craw0.y,craw0.z,craw0.w, srow + (0 ^ xor_val));
            CVT_ADD_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15, craw1.x,craw1.y,craw1.z,craw1.w, srow + (16 ^ xor_val));
            craw0 = *reinterpret_cast<const uint4*>(comb_ptr + 16);
            craw1 = *reinterpret_cast<const uint4*>(comb_ptr + 24);
            CVT_ADD_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23, craw0.x,craw0.y,craw0.z,craw0.w, srow + (32 ^ xor_val));
            CVT_ADD_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31, craw1.x,craw1.y,craw1.z,craw1.w, srow + (48 ^ xor_val));
        }

        // Second 32 cols: a32..a63 → swizzle region (bytes 64-127)
        {
            const __nv_bfloat16* comb_ptr2 = comb_base + (long long)((n_start + nc + 32) / COMB_BLOCK_COLS) * COMB_BLOCK_ELEMS;
            craw0 = *reinterpret_cast<const uint4*>(comb_ptr2);
            craw1 = *reinterpret_cast<const uint4*>(comb_ptr2 + 8);
            CVT_ADD_STS_V4(a32,a33,a34,a35,a36,a37,a38,a39, craw0.x,craw0.y,craw0.z,craw0.w, srow + (64 ^ xor_val));
            CVT_ADD_STS_V4(a40,a41,a42,a43,a44,a45,a46,a47, craw1.x,craw1.y,craw1.z,craw1.w, srow + (80 ^ xor_val));
            craw0 = *reinterpret_cast<const uint4*>(comb_ptr2 + 16);
            craw1 = *reinterpret_cast<const uint4*>(comb_ptr2 + 24);
            CVT_ADD_STS_V4(a48,a49,a50,a51,a52,a53,a54,a55, craw0.x,craw0.y,craw0.z,craw0.w, srow + (96 ^ xor_val));
            CVT_ADD_STS_V4(a56,a57,a58,a59,a60,a61,a62,a63, craw1.x,craw1.y,craw1.z,craw1.w, srow + (112 ^ xor_val));
        }

        // F40: Interleaved TMA store(s) — completed region(s) → global
        if (INTERLEAVE_STRATEGY == 1) {
            __syncwarp();
            if (lane == 0) {
                int region_idx = (nc - NC_START) >> 6;
                uint32_t src = staging_saddr + region_idx * STAGING_REGION_BYTES;
                asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                    :: "l"(tma_c_desc), "r"(n_start + nc), "r"(gm_base), "r"(src) : "memory");
            }
        } else if (INTERLEAVE_STRATEGY == 2 && (((nc - NC_START) >> 6) & 1) == 1) {
            // Half-batch: 2 stores after every 2nd region
            __syncwarp();
            if (lane == 0) {
                int region_idx = (nc - NC_START) >> 6;
                uint32_t src0 = staging_saddr + (region_idx - 1) * STAGING_REGION_BYTES;
                uint32_t src1 = staging_saddr + region_idx * STAGING_REGION_BYTES;
                asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                    :: "l"(tma_c_desc), "r"(n_start + nc - 64), "r"(gm_base), "r"(src0) : "memory");
                asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                    :: "l"(tma_c_desc), "r"(n_start + nc), "r"(gm_base), "r"(src1) : "memory");
            }
        } else if (INTERLEAVE_STRATEGY == 3 && ((nc - NC_START) >> 6) == 2) {
            // Three-plus-one: 3 stores after 3rd region
            __syncwarp();
            if (lane == 0) {
                for (int r = 0; r < 3; r++) {
                    uint32_t src = staging_saddr + r * STAGING_REGION_BYTES;
                    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                        :: "l"(tma_c_desc), "r"(n_start + NC_START + r * 64), "r"(gm_base), "r"(src) : "memory");
                }
            }
        }

        if (nc + 64 < NC_END) {
            TMEM_LOAD_X64(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                          a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                          a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44,a45,a46,a47,
                          a48,a49,a50,a51,a52,a53,a54,a55,a56,a57,a58,a59,a60,a61,a62,a63,
                          taddr_base + nc + 64);
        }
    }
#else  // TMEM_LOAD_WIDTH 16 or 32
    float a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15;
    float a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31;

    // ═══ Phase 1: all 256 cols → 4 swizzle regions, x32 stride ═══
    LOAD_32_COLS(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                 a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                 taddr_base + NC_START);

    PRAGMA_UNROLL(PHASE1_UNROLL)
    for (int nc = NC_START; nc < NC_END; nc += 32) {
        const __nv_bfloat16* comb_ptr = comb_base + (long long)((n_start + nc) / COMB_BLOCK_COLS) * COMB_BLOCK_ELEMS;
        uint4 craw0 = *reinterpret_cast<const uint4*>(comb_ptr);
        uint4 craw1 = *reinterpret_cast<const uint4*>(comb_ptr + 8);

        TMEM_WAIT();

        // F40: MBAR_EARLY — signal TMEM free immediately (data now in registers)
        if (MBAR_EARLY && nc + 32 >= NC_END) {
            if (epi_mbar_addr) mbar_arrive(epi_mbar_addr);
        }

        const uint32_t srow = srow_base + ((nc - NC_START) >> 6) * STAGING_REGION_BYTES;
        const int byte_base = ((nc - NC_START) & 63) * 2;
        CVT_ADD_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7, craw0.x,craw0.y,craw0.z,craw0.w, srow + (byte_base ^ xor_val));
        CVT_ADD_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15, craw1.x,craw1.y,craw1.z,craw1.w, srow + ((byte_base + 16) ^ xor_val));

        craw0 = *reinterpret_cast<const uint4*>(comb_ptr + 16);
        craw1 = *reinterpret_cast<const uint4*>(comb_ptr + 24);
        CVT_ADD_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23, craw0.x,craw0.y,craw0.z,craw0.w, srow + ((byte_base + 32) ^ xor_val));
        CVT_ADD_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31, craw1.x,craw1.y,craw1.z,craw1.w, srow + ((byte_base + 48) ^ xor_val));

        // F40: Interleaved TMA store(s) — region completes every 2 x32 iterations
        if (INTERLEAVE_STRATEGY == 1 && ((nc - NC_START) & 63) == 32) {
            int region_idx = (nc - NC_START) >> 6;
            __syncwarp();
            if (lane == 0) {
                uint32_t src = staging_saddr + region_idx * STAGING_REGION_BYTES;
                asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                    :: "l"(tma_c_desc), "r"(n_start + NC_START + region_idx * 64), "r"(gm_base), "r"(src) : "memory");
            }
        } else if (INTERLEAVE_STRATEGY == 2 && ((nc - NC_START) & 63) == 32 && (((nc - NC_START) >> 6) & 1) == 1) {
            // Half-batch: 2 stores after every 2nd region
            int region_idx = (nc - NC_START) >> 6;
            __syncwarp();
            if (lane == 0) {
                uint32_t src0 = staging_saddr + (region_idx - 1) * STAGING_REGION_BYTES;
                uint32_t src1 = staging_saddr + region_idx * STAGING_REGION_BYTES;
                asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                    :: "l"(tma_c_desc), "r"(n_start + NC_START + (region_idx - 1) * 64), "r"(gm_base), "r"(src0) : "memory");
                asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                    :: "l"(tma_c_desc), "r"(n_start + NC_START + region_idx * 64), "r"(gm_base), "r"(src1) : "memory");
            }
        } else if (INTERLEAVE_STRATEGY == 3 && ((nc - NC_START) & 63) == 32 && ((nc - NC_START) >> 6) == 2) {
            // Three-plus-one: 3 stores after 3rd region
            __syncwarp();
            if (lane == 0) {
                for (int r = 0; r < 3; r++) {
                    uint32_t src = staging_saddr + r * STAGING_REGION_BYTES;
                    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                        :: "l"(tma_c_desc), "r"(n_start + NC_START + r * 64), "r"(gm_base), "r"(src) : "memory");
                }
            }
        }

        if (nc + 32 < NC_END) {
            LOAD_32_COLS(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                         a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                         taddr_base + nc + 32);
        }
    }
#endif

#ifdef TIMING
    t_phase1_end = clock64();
#endif

#if INTERLEAVE_STRATEGY == 0
    // Strategy 0 (all-at-end): original behavior — all 4 TMA stores in Phase 2
    __syncwarp();  // Phase 1 SMEM writes visible for Phase 2 TMA reads
    if (!MBAR_EARLY && epi_mbar_addr) mbar_arrive(epi_mbar_addr);

    // ═══ Phase 2: 4 TMA tensor stores → global ═══
    if (lane == 0) {
        int row = gm_base;
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
            :: "l"(tma_c_desc), "r"(n_start + NC_START), "r"(row), "r"(staging_r0) : "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
            :: "l"(tma_c_desc), "r"(n_start + NC_START + 64), "r"(row), "r"(staging_r1) : "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
            :: "l"(tma_c_desc), "r"(n_start + NC_START + 128), "r"(row), "r"(staging_r2) : "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
            :: "l"(tma_c_desc), "r"(n_start + NC_START + 192), "r"(row), "r"(staging_r3) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    }
#elif INTERLEAVE_STRATEGY == 3
    // Strategy 3 (three-plus-one): 3 stores issued inline, 4th store (region 3) here
    __syncwarp();  // ensure region 3 STS visible for TMA read
    if (!MBAR_EARLY && epi_mbar_addr) mbar_arrive(epi_mbar_addr);
    if (lane == 0) {
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
            :: "l"(tma_c_desc), "r"(n_start + NC_START + 192), "r"(gm_base), "r"(staging_r3) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    }
#else
    // Strategies 1, 2: all 4 stores issued inline. Just signal + commit.
    if (!MBAR_EARLY && epi_mbar_addr) mbar_arrive(epi_mbar_addr);
    if (lane == 0) {
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    }
#endif
}

// ── Host precompute kernel: bias[c] + pos_embed[r,c] → BF16 combined[r,c] ──

__global__ void precompute_combined(
    const float* __restrict__ bias,
    const float* __restrict__ pos_embed,
    __nv_bfloat16* __restrict__ combined,
    int seq_len, int n_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < COMB_PADDED_ROWS * n_dim) {
        int row = idx / n_dim;
        int col = idx % n_dim;
        int real_row = row % seq_len;
        int block_row = row / COMB_BLOCK_COLS;
        int block_col = col / COMB_BLOCK_COLS;
        int row_in_block = row % COMB_BLOCK_ROWS;
        int col_in_block = col % COMB_BLOCK_COLS;
        int blocked_idx = (block_row * COMB_COL_BLOCKS + block_col) * COMB_BLOCK_ELEMS
                        + row_in_block * COMB_BLOCK_COLS + col_in_block;
        combined[blocked_idx] = __float2bfloat16(bias[col] + pos_embed[real_row * n_dim + col]);
    }
}

// ═════════════════════════════════════════════════════════════
// Patch embed GEMM — warp-specialized tcgen05 (cta_group::2)
// ═════════════════════════════════════════════════════════════

__global__ void __launch_bounds__(THREADS, 1)
__cluster_dims__(2, 1, 1)
patch_embed_gemm(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    const __grid_constant__ CUtensorMap tma_c,
    const __nv_bfloat16* __restrict__ combined,
    __nv_bfloat16* __restrict__ C
#ifdef TIMING
    , long long* __restrict__ timing_buf   // [74 clusters × TIMING_CLUSTER_STRIDE values]
    , long long* __restrict__ spread_buf   // [74 × MAX_SPREAD_TILES × NUM_EPI_WARPS] per-tile per-warp Phase 1
#endif
) {

    extern __shared__ __align__(128) char smem[];
    const int sm_id = blockIdx.x;
    const int tid   = threadIdx.x;
    const int warp  = tid / 32;
    const int lane  = tid % 32;

    // CTA rank within 2-CTA cluster
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    const int cluster_id = sm_id / 2;       // 0..73
    const int num_clusters = SM_COUNT / 2;   // 74

    // Per-stage SMEM layout: A_tile(16KB) + B_tile(16KB) = 32KB per stage

    // ── Mbarrier init ──
    if (tid == 0) {
        for (int s = 0; s < N_STAGES; s++) {
            mbar_init(smem_to_uint(smem + OFF_TMA_MBAR + s * 8), 2);   // both CTAs arrive at CTA0's TMA mbar
            mbar_init(smem_to_uint(smem + OFF_MMA_MBAR + s * 8), 1);   // CTA0 multicast commits → 1 arrival per CTA
        }
        // Double-buffered mainloop/epilogue mbarriers
        for (int i = 0; i < 2; i++) {
            mbar_init(smem_to_uint(smem + OFF_MAINLOOP_MBAR + i * 8), 1);          // CTA0 multicast commits
            mbar_init(smem_to_uint(smem + OFF_EPILOGUE_MBAR + i * 8), NUM_EPI_WARPS * 2 * 32); // NUM_EPI_WARPS warps × 2 CTAs × 32 threads
        }
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    // ── TMEM alloc: single alloc of TN*2=512 cols, in MMA warp (matmul_v7 pattern) ──
    if (warp == 1) {
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(smem_to_uint(smem + OFF_TMEM)), "r"(TMEM_COLS));
    }

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

    const int tile_start = (int)((long long)cluster_id * TOTAL_TILES / num_clusters);
    const int tile_end   = (int)((long long)(cluster_id + 1) * TOTAL_TILES / num_clusters);

    int tma_phase[N_STAGES] = {0};
    int mma_phase[N_STAGES] = {0};

    // F28: precompute SMEM descriptors — constant across all tiles
    uint64_t desc_a_base[N_STAGES], desc_b_base[N_STAGES];
    for (int s = 0; s < N_STAGES; s++) {
        desc_a_base[s] = make_smem_desc(smem_a[s]);
        desc_b_base[s] = make_smem_desc(smem_b[s]);
    }

    // Double-buffered mbar phase tracking (SUPERIOR pattern).
    // mbar[X] protects tmem[X]. Phase init 1 = "fresh mbar, skip first wait".
    // W1: waits epilogue_mbar[buf] before writing tmem[buf]
    // W2-5: waits mainloop_mbar[buf^1] before reading tmem[buf^1]
    //        arrives epilogue_mbar[buf^1] after reading tmem[buf^1]
    const int start_buf = tile_start & 1;
    int epi_phase[2] = {1, 1};           // W1's epilogue wait phases (both fresh)
    int ml_phase[2]  = {start_buf, 1 - start_buf};  // W2-5: prev_buf on first tile must be fresh(1)

#ifdef TIMING
    long long t_tile_start = 0, t_after_epi = 0, t_after_tma0 = 0, t_kloop_end = 0;
    long long sum_epi_wait = 0, sum_tma0_wait = 0, sum_kloop = 0, sum_total = 0;
    long long min_kloop = 0x7FFFFFFFFFFFFFFFLL, max_kloop = 0;
    long long min_total = 0x7FFFFFFFFFFFFFFFLL, max_total = 0;
    int tile_count = 0;
    // Epilogue Phase 1 timing — ALL epilogue warps (F25)
    long long epi_t0 = 0, epi_t1 = 0;
    long long epi_sum_p1 = 0;
    long long epi_min_p1 = 0x7FFFFFFFFFFFFFFFLL, epi_max_p1 = 0;
    int epi_count = 0;
    // Epilogue ml_wait + Phase 2 timing — W3/ew=1 only (backward compat)
    long long epi_t_before_ml = 0, epi_t2 = 0;
    long long epi_sum_p2 = 0, epi_sum_ml = 0;
    long long epi_min_p2 = 0x7FFFFFFFFFFFFFFFLL, epi_max_p2 = 0;
    long long epi_min_ml = 0x7FFFFFFFFFFFFFFFLL, epi_max_ml = 0;
#endif

    for (int tile_idx = tile_start; tile_idx < tile_end; tile_idx++) {
        const int buf = tile_idx & 1;
        const int tm = tile_idx / TILES_N;
        int tn = tile_idx % TILES_N;
        if (SNAKE_ORDER && (tm & 1)) tn = TILES_N - 1 - tn;  // snake: reverse odd M-rows
        const int m_start = tm * TM * 2 + cta_rank * TM;  // macro tile (256 rows) + CTA offset
        const int n_start = tn * TN;

        // ═══ K-LOOP (W0-1) + OVERLAPPED EPILOGUE (W2-5) ═══
        if (warp == 0) {
            // ── LOAD WARP (W0): TMA async bulk copies (both CTAs) ──
            if (lane == 0) {
                for (int ki = 0; ki < K_ITERS; ki++) {
                    const int s = ki % N_STAGES;
                    const int k_start = ki * TK;

                    if (tile_idx > tile_start || ki >= N_STAGES) {
                        mbar_wait(mma_mbar[s], mma_phase[s]);
                        mma_phase[s] ^= 1;
                    }

                    // Both CTAs arrive at CTA0's TMA mbar (masked)
                    const uint32_t tma_mbar_masked = tma_mbar[s] & 0xFEFFFFFF;
                    tma_load_2d(smem_a[s], &tma_a, k_start, m_start, tma_mbar_masked);
                    tma_load_2d(smem_b[s], &tma_b, k_start, n_start + cta_rank * (TN/2), tma_mbar_masked);
                    mbar_arrive_expect_tx(tma_mbar_masked, TMA_BYTES);
                }
            }
        } else if (warp == 1) {
            // ── MMA WARP (W1): tcgen05.mma → tmem_base + buf*TN (CTA0 only) ──
            if (lane == 0 && cta_rank == 0) {
                // Wait for epilogue to release tmem[buf]
#ifdef TIMING
                t_tile_start = clock64();
#endif
                mbar_wait(epilogue_mbar_addr + buf * 8, epi_phase[buf]);
                epi_phase[buf] ^= 1;
#ifdef TIMING
                t_after_epi = clock64();
#endif

                // ── F28: Manually unrolled K-loop with precomputed descriptors ──
                // ki=0 (s=0): clear accumulator on first sub-MMA
                mbar_wait(tma_mbar[0], tma_phase[0]);
                tma_phase[0] ^= 1;
#ifdef TIMING
                t_after_tma0 = clock64();
#endif
                asm volatile("tcgen05.fence::after_thread_sync;");
                {
                    uint64_t desc_a = desc_a_base[0], desc_b = desc_b_base[0];
                    // First sub-MMA: p=false → clear accumulator
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
                #pragma unroll
                for (int ki = 1; ki < K_ITERS; ki++) {
                    K_ITER_ACCUM(ki % N_STAGES);
                }

                // Signal K-loop done: multicast commit → mainloop_mbar[buf] in both CTAs
                tcgen05_commit_mcast(mainloop_mbar_addr + buf * 8, 0x3);
#ifdef TIMING
                t_kloop_end = clock64();
                long long dt_epi = t_after_epi - t_tile_start;
                long long dt_tma0 = t_after_tma0 - t_after_epi;
                long long dt_kloop = t_kloop_end - t_after_tma0;
                long long dt_total = t_kloop_end - t_tile_start;
                sum_epi_wait += dt_epi;
                sum_tma0_wait += dt_tma0;
                sum_kloop += dt_kloop;
                sum_total += dt_total;
                if (dt_kloop < min_kloop) min_kloop = dt_kloop;
                if (dt_kloop > max_kloop) max_kloop = dt_kloop;
                if (dt_total < min_total) min_total = dt_total;
                if (dt_total > max_total) max_total = dt_total;
                tile_count++;
                // Setup for next tile's t_tile_start
                t_tile_start = clock64();
#endif
            }
        } else {
            // ── OVERLAPPED EPILOGUE (W2+) ──
            const int ew = warp - 2;
            const int row_group = ew % 4;
            const int is_split = (row_group < (NUM_EPI_WARPS - 4)) ? 1 : 0;
            const int col_rank = ew / 4;
            const uint32_t staging_saddr = smem_to_uint(smem + OFF_STAGING + ew * STAGING_WARP_BYTES);

            const int prev_buf = buf ^ 1;

            // Wait for W1's K-loop on tmem[prev_buf] (from previous tile).
            // All epilogue warps poll independently — no bar.sync broadcast needed.
#ifdef TIMING
            if (ew == 1 && lane == 0 && cta_rank == 0)
                epi_t_before_ml = clock64();
#endif
            mbar_wait(mainloop_mbar_addr + prev_buf * 8, ml_phase[prev_buf]);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            ml_phase[prev_buf] ^= 1;
            // F31: stagger Phase 1 start to reduce TMEM scheduling contention
            if (STAGGER_CYCLES > 0 && ew > 0 && lane == 0) {
                long long __stagger_end = clock64() + ew * STAGGER_CYCLES;
                while (clock64() < __stagger_end) {}
            }
            __syncwarp();
#ifdef TIMING
            if (lane == 0 && cta_rank == 0)
                epi_t0 = clock64();  // F25: all epilogue warps timestamp Phase 1 start
#endif

            if (tile_idx > tile_start) {
                // Epilogue: read tmem[prev_buf], add inline BF16 combined, store to global
                const int prev_idx = tile_idx - 1;
                const int ptm = prev_idx / TILES_N;
                int ptn = prev_idx % TILES_N;
                if (SNAKE_ORDER && (ptm & 1)) ptn = TILES_N - 1 - ptn;
                const int prev_m = ptm * TM * 2 + cta_rank * TM;
                const int prev_n = ptn * TN;

                const int gm_base = prev_m + row_group * 32;
                const int pos_row = (gm_base + lane) % SEQ_LEN;
                // Early mbar: signal W1 after Phase 1B (TMEM done), Phase 2B overlaps with K-loop
                const uint32_t epi_mbar_masked = (epilogue_mbar_addr + prev_buf * 8) & 0xFEFFFFFF;
                if (is_split) {
                    if (col_rank == 0)
                        epilogue_store<0, TN/2>(prev_buf * TN, row_group, lane, gm_base, prev_n, combined, pos_row, C, cta_rank, staging_saddr, epi_mbar_masked, &tma_c
#ifdef TIMING
                            , epi_t1
#endif
                        );
                    else
                        epilogue_store<TN/2, TN>(prev_buf * TN, row_group, lane, gm_base, prev_n, combined, pos_row, C, cta_rank, staging_saddr, epi_mbar_masked, &tma_c
#ifdef TIMING
                            , epi_t1
#endif
                        );
                } else {
                    epilogue_store<0, TN>(prev_buf * TN, row_group, lane, gm_base, prev_n, combined, pos_row, C, cta_rank, staging_saddr, epi_mbar_masked, &tma_c
#ifdef TIMING
                        , epi_t1
#endif
                    );
                }
#ifdef TIMING
                // F25: all epilogue warps track Phase 1 + write per-tile spread data
                if (lane == 0 && cta_rank == 0) {
                    long long p1 = epi_t1 - epi_t0;
                    epi_sum_p1 += p1;
                    if (p1 < epi_min_p1) epi_min_p1 = p1;
                    if (p1 > epi_max_p1) epi_max_p1 = p1;
                    epi_count++;
                    // Per-tile per-warp Phase 1 for spread analysis
                    int tile_offset = tile_idx - tile_start - 1;
                    spread_buf[cluster_id * (MAX_SPREAD_TILES * NUM_EPI_WARPS) + tile_offset * NUM_EPI_WARPS + ew] = p1;
                    // ew=1 only: ml_wait + Phase 2 (backward compat)
                    if (ew == 1) {
                        epi_t2 = clock64();
                        long long ml = epi_t0 - epi_t_before_ml;
                        long long p2 = epi_t2 - epi_t1;
                        epi_sum_ml += ml;
                        epi_sum_p2 += p2;
                        if (ml < epi_min_ml) epi_min_ml = ml;
                        if (ml > epi_max_ml) epi_max_ml = ml;
                        if (p2 < epi_min_p2) epi_min_p2 = p2;
                        if (p2 > epi_max_p2) epi_max_p2 = p2;
                    }
                }
#endif
            }
        }
    }  // tile loop

#ifdef TIMING
    // Write timing data — W1 (CTA0, lane 0) writes per-cluster stats
    if (warp == 1 && lane == 0 && cta_rank == 0) {
        long long* out = timing_buf + cluster_id * TIMING_CLUSTER_STRIDE;
        out[0] = sum_epi_wait;
        out[1] = sum_tma0_wait;
        out[2] = sum_kloop;
        out[3] = sum_total;
        out[4] = min_kloop;
        out[5] = max_kloop;
        out[6] = min_total;
        out[7] = max_total;
    }
    // F25: Write per-warp epilogue Phase 1 timing — all epilogue warps
    if (warp >= 2 && lane == 0 && cta_rank == 0) {
        int ew_out = warp - 2;
        long long* out = timing_buf + cluster_id * TIMING_CLUSTER_STRIDE + 8 + ew_out * 4;
        out[0] = epi_sum_p1;
        out[1] = epi_min_p1;
        out[2] = epi_max_p1;
        out[3] = epi_count;
    }
    // Write ew=1 ml_wait + Phase 2 timing (backward compat)
    if (warp == 3 && lane == 0 && cta_rank == 0) {
        long long* out = timing_buf + cluster_id * TIMING_CLUSTER_STRIDE + 24;
        out[0] = epi_sum_p2;
        out[1] = epi_min_p2;
        out[2] = epi_max_p2;
        out[3] = epi_sum_ml;
        out[4] = epi_min_ml;
        out[5] = epi_max_ml;
    }
#endif

    // ── DRAIN (W2+ only): epilogue for the last tile ──
    if (warp >= 2) {
        const int ew = warp - 2;
        const int row_group = ew % 4;
        const int is_split = (row_group < (NUM_EPI_WARPS - 4)) ? 1 : 0;
        const int col_rank = ew / 4;
        const uint32_t staging_saddr = smem_to_uint(smem + OFF_STAGING + ew * STAGING_WARP_BYTES);

        const int last_buf = (tile_end - 1) & 1;

        // Wait for W1's K-loop on tmem[last_buf]
        // All epilogue warps poll independently — no bar.sync broadcast needed.
        mbar_wait(mainloop_mbar_addr + last_buf * 8, ml_phase[last_buf]);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        // F31: stagger Phase 1 start (drain)
        if (STAGGER_CYCLES > 0 && ew > 0 && lane == 0) {
            long long __stagger_end = clock64() + ew * STAGGER_CYCLES;
            while (clock64() < __stagger_end) {}
        }
        __syncwarp();

        // Epilogue store for the last tile
        const int last_idx = tile_end - 1;
        const int ltm = last_idx / TILES_N;
        int ltn = last_idx % TILES_N;
        if (SNAKE_ORDER && (ltm & 1)) ltn = TILES_N - 1 - ltn;
        const int last_m = ltm * TM * 2 + cta_rank * TM;
        const int last_n = ltn * TN;
        const int gm_base = last_m + row_group * 32;
        const int pos_row = (gm_base + lane) % SEQ_LEN;
#ifdef TIMING
        long long drain_t1 = 0;
#endif
        if (is_split) {
            if (col_rank == 0)
                epilogue_store<0, TN/2>(last_buf * TN, row_group, lane, gm_base, last_n, combined, pos_row, C, cta_rank, staging_saddr, 0, &tma_c
#ifdef TIMING
                    , drain_t1
#endif
                );
            else
                epilogue_store<TN/2, TN>(last_buf * TN, row_group, lane, gm_base, last_n, combined, pos_row, C, cta_rank, staging_saddr, 0, &tma_c
#ifdef TIMING
                    , drain_t1
#endif
                );
        } else {
            epilogue_store<0, TN>(last_buf * TN, row_group, lane, gm_base, last_n, combined, pos_row, C, cta_rank, staging_saddr, 0, &tma_c
#ifdef TIMING
                , drain_t1
#endif
            );
        }

        // Wait for drain epilogue's Phase 2B TMA stores to complete
        if (lane == 0) {
            asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        }
        __syncwarp();
    }

    // ── Cluster sync + TMEM dealloc (all warps, both CTAs) ──
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 2) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
            :: "r"(0), "r"(TMEM_COLS));
    }
}

// ═════════════════════════════════════════════════════════════
// Host
// ═════════════════════════════════════════════════════════════

#ifdef TIMING
static int cmp_ll(const void* a, const void* b) {
    long long va = *(const long long*)a;
    long long vb = *(const long long*)b;
    return (va > vb) - (va < vb);
}
#endif

int main() {
    setbuf(stdout, NULL);  // unbuffered stdout for debugging
    printf("SigLIP2 patch embed GEMM — tcgen05 cta_group::2 (%d warps [%d epi], cluster of 2)\n",
           2 + NUM_EPI_WARPS, NUM_EPI_WARPS);
    printf("  GEMM: [%d,%d] x [%d,%d]^T  %d-stage pipeline  inline BF16 combined  SMEM-staged coalesced stores  idesc: 0x%08X\n",
           M_TOTAL, K_DIM, N_DIM, K_DIM, N_STAGES, IDESC);

    uint8_t *d_A, *d_B;
    float *d_bias, *d_pos;
    __nv_bfloat16 *d_combined, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A,    (size_t)M_TOTAL * K_DIM));
    CUDA_CHECK(cudaMalloc(&d_B,    (size_t)N_DIM   * K_DIM));
    CUDA_CHECK(cudaMalloc(&d_bias,  (size_t)N_DIM  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos,   (size_t)SEQ_LEN * N_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_combined, (size_t)COMB_PADDED_ROWS * N_DIM * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_C,    (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16)));

    // A: uniform 0x3C (=1.5 in FP8 E4M3)
    // B: alternating rows — even rows 0x3C (1.5), odd rows 0x38 (1.0)
    // This makes adjacent output columns have different accumulator values:
    //   even cols: K_DIM * 1.5 * 1.5 = 1728.0
    //   odd cols:  K_DIM * 1.5 * 1.0 = 1152.0
    // Detects CVT bf16x2 lane swap bugs that uniform data cannot catch.
    CUDA_CHECK(cudaMemset(d_A, 0x3C, (size_t)M_TOTAL * K_DIM));
    {
        uint8_t* h_B = (uint8_t*)malloc((size_t)N_DIM * K_DIM);
        for (int n = 0; n < N_DIM; n++)
            memset(h_B + (size_t)n * K_DIM, (n & 1) ? 0x38 : 0x3C, K_DIM);
        CUDA_CHECK(cudaMemcpy(d_B, h_B, (size_t)N_DIM * K_DIM, cudaMemcpyHostToDevice));
        free(h_B);
    }

    // Non-uniform bias/pos_embed: makes every output element position-dependent,
    // so layout/permutation bugs (tile ordering, CTA rank, snake direction) are detectable.
    // bias[c] = c + 1 (column-dependent), pos_embed[r][c] = 3*(r+1) (row-dependent)
    {
        float* h_bias = (float*)malloc((size_t)N_DIM * sizeof(float));
        float* h_pos  = (float*)malloc((size_t)SEQ_LEN * N_DIM * sizeof(float));
        for (int c = 0; c < N_DIM; c++)
            h_bias[c] = (float)(c + 1);
        for (int r = 0; r < SEQ_LEN; r++)
            for (int c = 0; c < N_DIM; c++)
                h_pos[r * N_DIM + c] = (float)((r + 1) * 3);
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias, (size_t)N_DIM * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pos, h_pos, (size_t)SEQ_LEN * N_DIM * sizeof(float), cudaMemcpyHostToDevice));
        free(h_bias);
        free(h_pos);
    }

    // Precompute combined[r][c] = __float2bfloat16(bias[c] + pos_embed[r*N_DIM+c])
    {
        int n_elem = COMB_PADDED_ROWS * N_DIM;
        int tpb = 256;
        int bpg = (n_elem + tpb - 1) / tpb;
        precompute_combined<<<bpg, tpb>>>(d_bias, d_pos, d_combined, SEQ_LEN, N_DIM);
        CUDA_CHECK(cudaGetLastError());
    }
    printf("  Alloc + precompute done\n");

    CUtensorMap h_tma_a, h_tma_b;

    // 2D TMA with SWIZZLE_128B: load [TK, height] per stage
    // Global memory: row-major A[M,K], B[N,K] with element=UINT8 (FP8)
    // TK=128 FP8 = 128 bytes = exactly 128B → SWIZZLE_128B
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
        uint32_t box[2]     = {TK, TN/2};   // each CTA loads half of B columns
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
        uint64_t strides[1] = {(uint64_t)N_DIM * sizeof(__nv_bfloat16)};  // 1536 bytes
        uint32_t box[2]     = {64, 32};   // 64 BF16 cols x 32 rows
        uint32_t estrides[2]= {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(&h_tma_c,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)d_C,
            dims, strides, box, estrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }

    CUDA_CHECK(cudaFuncSetAttribute(patch_embed_gemm,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    printf("  TMA descriptors + func attr done\n");

#ifdef TIMING
    long long *d_timing, *d_spread;
    CUDA_CHECK(cudaMalloc(&d_timing, 74 * TIMING_CLUSTER_STRIDE * sizeof(long long)));
    CUDA_CHECK(cudaMemset(d_timing, 0, 74 * TIMING_CLUSTER_STRIDE * sizeof(long long)));
    size_t spread_bytes = (size_t)74 * MAX_SPREAD_TILES * NUM_EPI_WARPS * sizeof(long long);
    CUDA_CHECK(cudaMalloc(&d_spread, spread_bytes));
    CUDA_CHECK(cudaMemset(d_spread, 0, spread_bytes));
#endif

    // ── Warmup: 2 iterations ──
    printf("Launching warmup (2 iters)...\n");
    for (int _i = 0; _i < 2; _i++) {
    patch_embed_gemm<<<SM_COUNT, THREADS, SMEM_BYTES>>>(h_tma_a, h_tma_b, h_tma_c, d_combined, d_C
#ifdef TIMING
        , d_timing, d_spread
#endif
    );
    }
    printf("  Waiting for warmup sync...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Warmup done.\n");

    // ── Timed: 10 iterations ──
    printf("Timing: 10 iterations...\n");
    cudaEvent_t _t0, _t1;
    cudaEventCreate(&_t0);
    cudaEventCreate(&_t1);
    cudaEventRecord(_t0);
    for (int _i = 0; _i < 10; _i++) {
    patch_embed_gemm<<<SM_COUNT, THREADS, SMEM_BYTES>>>(h_tma_a, h_tma_b, h_tma_c, d_combined, d_C
#ifdef TIMING
        , d_timing, d_spread
#endif
    );
    }
    cudaEventRecord(_t1);
    cudaEventSynchronize(_t1);
    float _ms;
    cudaEventElapsedTime(&_ms, _t0, _t1);
    _ms /= 10.0f;
    printf("Custom kernel: %.3f ms  %.2f TFLOPS\n",
           _ms, 2.0 * M_TOTAL * N_DIM * K_DIM / _ms / 1e9);
    cudaEventDestroy(_t0);
    cudaEventDestroy(_t1);

    // ── Checksum run ──
    patch_embed_gemm<<<SM_COUNT, THREADS, SMEM_BYTES>>>(h_tma_a, h_tma_b, h_tma_c, d_combined, d_C
#ifdef TIMING
        , d_timing, d_spread
#endif
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    __nv_bfloat16* h_C = (__nv_bfloat16*)malloc((size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    // Copy combined (blocked layout) back to host for CPU reference validation
    __nv_bfloat16* h_combined = (__nv_bfloat16*)malloc((size_t)COMB_PADDED_ROWS * N_DIM * sizeof(__nv_bfloat16));
    CUDA_CHECK(cudaMemcpy(h_combined, d_combined, (size_t)COMB_PADDED_ROWS * N_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    // Strided checksum: 1024 samples spread across the full output matrix.
    // Catches tile ordering, snake direction, CTA rank, and late-tile bugs
    // that a first-1024-element window would miss entirely.
    double cksum = 0;
    {
        long long total_elems = (long long)M_TOTAL * N_DIM;
        long long stride = total_elems / 1024;
        for (int i = 0; i < 1024; i++)
            cksum += (double)__bfloat162float(h_C[(long long)i * stride]);
    }

    // CPU reference spot checks: 32 positions spread across the matrix.
    // Validates layout correctness (combined blocked-index lookup must match kernel).
    // GEMM result is column-dependent (non-uniform B):
    //   even cols: K_DIM * 1.5 * 1.5 = 1728.0
    //   odd cols:  K_DIM * 1.5 * 1.0 = 1152.0
    int errors = 0;
    {
        for (int spot = 0; spot < 32; spot++) {
            long long row = (long long)spot * M_TOTAL / 32;
            int col = (spot * 47) % N_DIM;  // prime stride for column diversity
            int pos_row = (int)(row % SEQ_LEN);

            // Column-dependent GEMM result: B[even_n]=1.5, B[odd_n]=1.0
            float b_val = (col & 1) ? 1.0f : 1.5f;
            float gemm_f32 = (float)K_DIM * 1.5f * b_val;
            float gemm_bf16_f = __bfloat162float(__float2bfloat16(gemm_f32));

            // Blocked combined layout lookup (must match precompute_combined kernel)
            int br = pos_row / COMB_BLOCK_ROWS;
            int rir = pos_row % COMB_BLOCK_ROWS;
            int bc = col / COMB_BLOCK_COLS;
            int cic = col % COMB_BLOCK_COLS;
            int cidx = (br * COMB_COL_BLOCKS + bc) * COMB_BLOCK_ELEMS + rir * COMB_BLOCK_COLS + cic;

            float comb_f = __bfloat162float(h_combined[cidx]);
            // Reference: bf16(bf16(gemm) + bf16(combined)) — matches kernel's cvt+add
            __nv_bfloat16 expected = __float2bfloat16(gemm_bf16_f + comb_f);
            __nv_bfloat16 actual = h_C[row * N_DIM + col];

            float ef = __bfloat162float(expected);
            float af = __bfloat162float(actual);
            if (ef != af) {
                if (errors < 5)
                    printf("  MISMATCH at (%lld,%d): expected %.1f got %.1f (combined=%.4f gemm=%.1f)\n",
                           row, col, ef, af, comb_f, gemm_bf16_f);
                errors++;
            }
        }
    }
    free(h_combined);

    int valid = (errors == 0) ? 1 : 0;
    printf("Validation: %d/32 spot checks passed%s\n", 32 - errors, valid ? "" : " — FAILED");
    printf("Checksum (1024 strided): %f\n", cksum);
    printf("C[0,0..3] = %.1f %.1f %.1f %.1f\n",
           __bfloat162float(h_C[0]), __bfloat162float(h_C[1]),
           __bfloat162float(h_C[2]), __bfloat162float(h_C[3]));
    printf("@@RESULT ms=%.3f tflops=%.2f checksum=%f valid=%d c0=%.1f\n",
           _ms, 2.0 * M_TOTAL * N_DIM * K_DIM / _ms / 1e9, cksum, valid,
           __bfloat162float(h_C[0]));

#ifdef TIMING
    // Read timing data
    long long h_timing[74 * TIMING_CLUSTER_STRIDE];
    CUDA_CHECK(cudaMemcpy(h_timing, d_timing, sizeof(h_timing), cudaMemcpyDeviceToHost));
    long long* h_spread = (long long*)malloc(spread_bytes);
    CUDA_CHECK(cudaMemcpy(h_spread, d_spread, spread_bytes, cudaMemcpyDeviceToHost));

    // ── Aggregate W1 data across clusters ──
    long long g_epi = 0, g_tma0 = 0, g_kloop = 0, g_total = 0;
    long long g_min_kloop = 0x7FFFFFFFFFFFFFFFLL, g_max_kloop = 0;
    long long g_min_total = 0x7FFFFFFFFFFFFFFFLL, g_max_total = 0;
    int total_tiles = 0;

    for (int c = 0; c < 74; c++) {
        long long* d = h_timing + c * TIMING_CLUSTER_STRIDE;
        int tiles_this = (int)((long long)(c + 1) * TOTAL_TILES / 74) - (int)((long long)c * TOTAL_TILES / 74);
        g_epi += d[0];  g_tma0 += d[1];  g_kloop += d[2];  g_total += d[3];
        if (d[4] < g_min_kloop) g_min_kloop = d[4];
        if (d[5] > g_max_kloop) g_max_kloop = d[5];
        if (d[6] < g_min_total) g_min_total = d[6];
        if (d[7] > g_max_total) g_max_total = d[7];
        total_tiles += tiles_this;
    }

    double clock_ghz = 2.1; // B200 SM clock approx
    printf("\n=== W1 TIMING (clock64, %d tiles across 74 clusters) ===\n", total_tiles);
    printf("  Per-tile averages (cycles / ns at %.1f GHz):\n", clock_ghz);
    printf("    Epilogue mbar wait:  %7lld cycles / %6.1f ns\n", g_epi / total_tiles, (double)g_epi / total_tiles / clock_ghz);
    printf("    TMA stage-0 wait:    %7lld cycles / %6.1f ns\n", g_tma0 / total_tiles, (double)g_tma0 / total_tiles / clock_ghz);
    printf("    K-loop (6 ki × 4 MMA): %7lld cycles / %6.1f ns\n", g_kloop / total_tiles, (double)g_kloop / total_tiles / clock_ghz);
    printf("    Total tile:          %7lld cycles / %6.1f ns\n", g_total / total_tiles, (double)g_total / total_tiles / clock_ghz);
    printf("    Overhead (epi+tma0): %7lld cycles / %6.1f ns  (%.1f%% of tile)\n",
           (g_epi + g_tma0) / total_tiles, (double)(g_epi + g_tma0) / total_tiles / clock_ghz,
           100.0 * (g_epi + g_tma0) / g_total);
    printf("  K-loop range: min=%lld max=%lld (%.1fx spread)\n", g_min_kloop, g_max_kloop,
           g_min_kloop > 0 ? (double)g_max_kloop / g_min_kloop : 0.0);
    printf("  Total tile range: min=%lld max=%lld (%.1fx spread)\n", g_min_total, g_max_total,
           g_min_total > 0 ? (double)g_max_total / g_min_total : 0.0);
    printf("  Expected total cycles (wall clock): %.0f\n", _ms * 1e-3 * clock_ghz * 1e9);

    // ── F25: Aggregate per-warp Phase 1 data ──
    long long gw_sum_p1[NUM_EPI_WARPS] = {0};
    long long gw_min_p1[NUM_EPI_WARPS], gw_max_p1[NUM_EPI_WARPS];
    int gw_count[NUM_EPI_WARPS] = {0};
    for (int w = 0; w < NUM_EPI_WARPS; w++) {
        gw_min_p1[w] = 0x7FFFFFFFFFFFFFFFLL;
        gw_max_p1[w] = 0;
    }
    for (int c = 0; c < 74; c++) {
        long long* d = h_timing + c * TIMING_CLUSTER_STRIDE;
        for (int w = 0; w < NUM_EPI_WARPS; w++) {
            long long* pw = d + 8 + w * 4;
            gw_sum_p1[w] += pw[0];
            if (pw[1] < gw_min_p1[w]) gw_min_p1[w] = pw[1];
            if (pw[2] > gw_max_p1[w]) gw_max_p1[w] = pw[2];
            gw_count[w] += (int)pw[3];
        }
    }
    // Backward-compat ew=1 ml_wait + Phase 2
    long long g_ep2 = 0, g_eml = 0;
    long long g_min_p2 = 0x7FFFFFFFFFFFFFFFLL, g_max_p2 = 0;
    long long g_min_ml = 0x7FFFFFFFFFFFFFFFLL, g_max_ml = 0;
    for (int c = 0; c < 74; c++) {
        long long* d = h_timing + c * TIMING_CLUSTER_STRIDE + 24;
        g_ep2 += d[0];
        if (d[1] < g_min_p2) g_min_p2 = d[1];
        if (d[2] > g_max_p2) g_max_p2 = d[2];
        g_eml += d[3];
        if (d[4] < g_min_ml) g_min_ml = d[4];
        if (d[5] > g_max_ml) g_max_ml = d[5];
    }
    int total_epi_tiles = gw_count[1]; // ew=1 count for backward compat

    // ── F25: Compute per-warp p95 and per-tile inter-warp spread from spread_buf ──
    // Collect per-warp Phase 1 times and per-tile spreads
    int n_spread_tiles = 0;
    for (int c = 0; c < 74; c++) {
        int ts = (int)((long long)c * TOTAL_TILES / 74);
        int te = (int)((long long)(c + 1) * TOTAL_TILES / 74);
        n_spread_tiles += (te - ts - 1); // in-loop epilogue tiles
    }

    long long* warp_p1_all[NUM_EPI_WARPS];
    for (int w = 0; w < NUM_EPI_WARPS; w++)
        warp_p1_all[w] = (long long*)malloc(n_spread_tiles * sizeof(long long));
    long long* tile_spreads = (long long*)malloc(n_spread_tiles * sizeof(long long));

    int idx = 0;
    long long sum_spread = 0;
    long long min_spread_val = 0x7FFFFFFFFFFFFFFFLL, max_spread_val = 0;
    for (int c = 0; c < 74; c++) {
        int ts = (int)((long long)c * TOTAL_TILES / 74);
        int te = (int)((long long)(c + 1) * TOTAL_TILES / 74);
        int epi_tiles_c = te - ts - 1;
        for (int t = 0; t < epi_tiles_c; t++) {
            long long mn = 0x7FFFFFFFFFFFFFFFLL, mx = 0;
            for (int w = 0; w < NUM_EPI_WARPS; w++) {
                long long v = h_spread[c * (MAX_SPREAD_TILES * NUM_EPI_WARPS) + t * NUM_EPI_WARPS + w];
                warp_p1_all[w][idx] = v;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            long long sp = mx - mn;
            tile_spreads[idx] = sp;
            sum_spread += sp;
            if (sp < min_spread_val) min_spread_val = sp;
            if (sp > max_spread_val) max_spread_val = sp;
            idx++;
        }
    }

    // Per-warp p95
    long long gw_p95[NUM_EPI_WARPS];
    for (int w = 0; w < NUM_EPI_WARPS; w++) {
        qsort(warp_p1_all[w], n_spread_tiles, sizeof(long long), cmp_ll);
        gw_p95[w] = warp_p1_all[w][(int)(n_spread_tiles * 0.95)];
    }
    // Spread p95
    qsort(tile_spreads, n_spread_tiles, sizeof(long long), cmp_ll);
    long long p95_spread = tile_spreads[(int)(n_spread_tiles * 0.95)];

    // ── Print F25: Per-warp Phase 1 timing ──
    printf("\n=== EPILOGUE PER-WARP PHASE 1 TIMING (W2-W5, %d tiles across 74 clusters) ===\n", n_spread_tiles);
    printf("  Per-warp Phase 1 (cycles):\n");
    for (int w = 0; w < NUM_EPI_WARPS; w++) {
        long long avg = gw_count[w] > 0 ? gw_sum_p1[w] / gw_count[w] : 0;
        printf("    W%d (ew=%d, rg=%d):  avg=%lld  min=%lld  max=%lld  p95=%lld\n",
               w + 2, w, w, avg, gw_min_p1[w], gw_max_p1[w], gw_p95[w]);
    }
    // Per-warp average spread
    long long warp_avgs[NUM_EPI_WARPS];
    long long avg_min = 0x7FFFFFFFFFFFFFFFLL, avg_max = 0;
    for (int w = 0; w < NUM_EPI_WARPS; w++) {
        warp_avgs[w] = gw_count[w] > 0 ? gw_sum_p1[w] / gw_count[w] : 0;
        if (warp_avgs[w] < avg_min) avg_min = warp_avgs[w];
        if (warp_avgs[w] > avg_max) avg_max = warp_avgs[w];
    }
    printf("  Spread of per-warp averages: %lld cycles (max_avg - min_avg)\n", avg_max - avg_min);
    printf("  Inter-warp spread per tile (max-min Phase 1 across warps):\n");
    printf("    Average: %lld cycles\n", n_spread_tiles > 0 ? sum_spread / n_spread_tiles : 0LL);
    printf("    Min: %lld  Max: %lld  P95: %lld cycles\n", min_spread_val, max_spread_val, p95_spread);
    long long warp_avg_spread = avg_max - avg_min;
    if (warp_avg_spread < 200)
        printf("  => SYMMETRIC (avg spread %lld < 200 cyc): bandwidth-limited, F27 dephasing won't help\n", warp_avg_spread);
    else
        printf("  => ASYMMETRIC (avg spread %lld >= 200 cyc): port-queued or bank-conflict bias, F27 has a target\n", warp_avg_spread);

    // ── Backward-compat: W3/ew=1 full phase timing ──
    printf("\n=== EPILOGUE PHASE TIMING (W3/ew=1, %d tiles across 74 clusters) ===\n", total_epi_tiles);
    if (total_epi_tiles > 0) {
        long long avg_ml = g_eml / total_epi_tiles;
        long long avg_p1 = gw_sum_p1[1] / total_epi_tiles;
        long long avg_p2 = g_ep2 / total_epi_tiles;
        long long avg_total = avg_ml + avg_p1 + avg_p2;
        printf("  Per-tile averages (cycles / ns at %.1f GHz):\n", clock_ghz);
        printf("    Mainloop mbar wait:    %7lld cycles / %6.1f ns  (%.1f%%)\n",
               avg_ml, (double)avg_ml / clock_ghz, 100.0 * avg_ml / avg_total);
        printf("    Phase 1 (TMEM->SMEM):  %7lld cycles / %6.1f ns  (%.1f%%)\n",
               avg_p1, (double)avg_p1 / clock_ghz, 100.0 * avg_p1 / avg_total);
        printf("    Phase 2 (SMEM->global): %7lld cycles / %6.1f ns  (%.1f%%)\n",
               avg_p2, (double)avg_p2 / clock_ghz, 100.0 * avg_p2 / avg_total);
        printf("    Total (wait+work):     %7lld cycles / %6.1f ns\n",
               avg_total, (double)avg_total / clock_ghz);
        printf("    Work only (P1+P2):     %7lld cycles / %6.1f ns\n",
               avg_p1 + avg_p2, (double)(avg_p1 + avg_p2) / clock_ghz);
        printf("  Mainloop wait range: min=%lld max=%lld (%.1fx spread)\n", g_min_ml, g_max_ml,
               g_min_ml > 0 ? (double)g_max_ml / g_min_ml : 0.0);
        printf("  Phase 1 range: min=%lld max=%lld (%.1fx spread)\n", gw_min_p1[1], gw_max_p1[1],
               gw_min_p1[1] > 0 ? (double)gw_max_p1[1] / gw_min_p1[1] : 0.0);
        printf("  Phase 2 range: min=%lld max=%lld (%.1fx spread)\n", g_min_p2, g_max_p2,
               g_min_p2 > 0 ? (double)g_max_p2 / g_min_p2 : 0.0);
    }

    for (int w = 0; w < NUM_EPI_WARPS; w++) free(warp_p1_all[w]);
    free(tile_spreads);
    free(h_spread);
    cudaFree(d_timing);
    cudaFree(d_spread);
#endif

    free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_bias); cudaFree(d_pos);
    cudaFree(d_combined); cudaFree(d_C);
    return 0;
}
