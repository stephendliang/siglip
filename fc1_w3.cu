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
  -DTILE_DISPATCH=N     Static tile swizzle: 0=Group-3 strided, >=8 see
                        tile_dispatch.cuh (default 11 = zigzag)
  -DK_STAGGER=N         Per-cluster K-phase shift (default 1; 0 disables)
  -DNUM_EPI_WARPS=N     Epilogue warp count (1, 2, or 4; default 4)
  -DNUM_EPI_STAGES=N    Epilogue staging ring depth (2-4; default 3; 4 needs
                        BIAS_PER_TILE or LDG_BIAS headroom)
  -DNO_EPI_DECOUPLE     Restore per-subiter cross-warp barriers (lockstep epi)
  -DBIAS_PER_TILE       Per-tile SMEM bias instead of the full 6KB. TD>=8:
                        3-slot cp.async ring (default-on, −133 kcyc); TD<8:
                        two fixed columns at startup (auto at N_STAGES>=6)
  -DNO_BIAS_PER_TILE    Restore the full 6KB SMEM bias (TD>=8, N_STAGES<6)
  -DGELU_F32X2          Packed-pair GELU math (SASS FADD2/FMUL2/FFMA2);
                        tested +31 kcyc — epi is not FP-issue-bound
*/

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>

/* PACKED_TILES is default-on. Opt out with -DNO_PACKED_TILES. */
#ifndef NO_PACKED_TILES
#ifndef PACKED_TILES
#define PACKED_TILES
#endif
#endif

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

/*
 * PREFILL safety: W1 skips epilogue_mbar wait, relying on TMEM double-buffering
 * to hide epilogue latency.  This only works when the K-loop is long enough that
 * W2+ finishes all TMEM reads before W1 overwrites the same buffer 2 tiles later.
 * With K_ITERS < 20, the K-loop is too short — W1 races ahead, the mainloop_mbar
 * parity wraps, and the kernel deadlocks.  Auto-force NO_PREFILL for short K-loops.
 */
#if K_ITERS < 20 && !defined(NO_PREFILL) && !defined(GEMM_ONLY) && !defined(STRIP_EPILOGUE)
#define NO_PREFILL
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

/* Production tune (paired 2026-07-01: −43 kcyc vs TD=0 stride): zigzag
   dispatch + per-cluster K-phase stagger. -DK_STAGGER=0 disables the shift. */
#ifndef TILE_DISPATCH
#define TILE_DISPATCH 11
#endif
#ifndef K_STAGGER
#define K_STAGGER 1
#endif

#include "tile_dispatch.cuh"

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

#define _LAYOUT_END _MBAR_END

#define NUM_EPI_SUBITERS   4
#ifndef NUM_EPI_STAGES
#define NUM_EPI_STAGES     3
#endif

/*
Non-last store pacing tracks the staging ring: ES-1 bulk groups in flight
keeps the oldest slot drained before its STS reuse one subiter later.
*/
#if NUM_EPI_STAGES == 2
#define EPI_WG_NONLAST "cp.async.bulk.wait_group 1;"
#elif NUM_EPI_STAGES == 3
#define EPI_WG_NONLAST "cp.async.bulk.wait_group 2;"
#elif NUM_EPI_STAGES == 4
#define EPI_WG_NONLAST "cp.async.bulk.wait_group 3;"
#else
#error "NUM_EPI_STAGES must be 2, 3, or 4"
#endif

/*
EPI_DECOUPLE: each epi warp's STS and TMA store touch only its own
row_group staging region, so the per-subiter CTA barriers enforce nothing
but lockstep — drop them and let warps self-pace. fence.proxy.async is
per-lane; __syncwarp orders lane 0's TMA store after all lanes' fenced STS.
Production default with ES=3 (paired 2026-07-02: 3619.4 kcyc vs 3955.6
lockstep ES=2 = −336.2 kcyc); -DNO_EPI_DECOUPLE opts out.
*/
#ifndef NO_EPI_DECOUPLE
#ifndef EPI_DECOUPLE
#define EPI_DECOUPLE
#endif
#endif
#ifdef EPI_DECOUPLE
#define EPI_SUBITER_BAR()
#define EPI_PRESTORE_SYNC() __syncwarp()
#else
#define EPI_SUBITER_BAR()   asm volatile(BAR_EPI_SYNC ::: "memory")
#define EPI_PRESTORE_SYNC() asm volatile(BAR_EPI_SYNC ::: "memory")
#endif

/*
 * Bias loading strategy:
 *
 * LDG_BIAS: load bias directly from L1-cached global memory (no SMEM).
 *   Eliminates all bias SMEM, batch tracking, and startup loading.
 *   Bias is 6KB — fits entirely in L1 after first touch.
 *
 * BIAS_PER_TILE: compact per-tile SMEM when N_STAGES >= 6 (full bias exceeds 228KB).
 *   Group-3: 2 N-tiles loaded once at startup (snake order uses 2 N-tiles).
 *   TD>=8: tn changes tile-to-tile, so a 3-slot ring (slot = tile % 3) is
 *   filled by an epi-warp cp.async prefetch each tile. 3 slots, not 2: under
 *   EPI_DECOUPLE warps skew up to a full tile, and slot reuse at distance 2
 *   races a slow reader. Distance 3 is safe by the NO_PREFILL back-pressure
 *   chain: MMA(t+2) waits epilogue_mbar[t&1] = all epi warps done with tile t,
 *   and any warp's tile-(t+3) prefetch is gated on MMA(t+2)'s mainloop arrive.
 */
#ifdef LDG_BIAS
/* No SMEM for bias — direct LDG from L1-cached global */
#define BIAS_SMEM_BYTES    0
#else /* !LDG_BIAS */

/*
Ring default-on at TD>=8 (2026-07-02: −133 kcyc vs the full 6KB SMEM bias at
production NS5/ES3). NO_BIAS_PER_TILE restores full bias but N_STAGES>=6
overrides back on — full bias can never fit at NS=6.
*/
#ifndef BIAS_PER_TILE
#if (!defined(NO_BIAS_PER_TILE) && TILE_DISPATCH >= 8) || N_STAGES >= 6
#define BIAS_PER_TILE 1
#endif
#endif

#ifdef BIAS_PER_TILE
#if TILE_DISPATCH >= 8
#define BIAS_SMEM_BYTES    (TN * 2 * 3)          /* 1536B — 3-slot per-tile ring */
#else
#define BIAS_SMEM_BYTES    (TN * 2 * 2)          /* 1024B — 2 N-tiles for snake */
#endif
#else
#define BIAS_SMEM_BYTES    (N_DIM * 2)            /* 6144B — all bias columns */
#endif

#endif /* LDG_BIAS */
#define OFF_BIAS_SMEM      ((_LAYOUT_END + 15) & ~15)

/* Epilogue staging: 2-stage double-buffer for STS → TMA store */
#define STAGING_REGION_BYTES  (32 * 128)
#define EPI_STAGE_BYTES       (4 * STAGING_REGION_BYTES)
#define OFF_STAGING           ((OFF_BIAS_SMEM + BIAS_SMEM_BYTES + 1023) & ~1023)
#define SMEM_BYTES            ((OFF_STAGING + NUM_EPI_STAGES * EPI_STAGE_BYTES + 127) & ~127)

/*
sm_100a max dynamic SMEM opt-in is 227 KB per block. Infeasible knob combos
(N_STAGES=6 with full SMEM bias or NUM_EPI_STAGES>=3; ES=4 with full SMEM
bias — both fit with BIAS_PER_TILE at 231424B) must die here, not at
cudaFuncSetAttribute.
*/
static_assert(SMEM_BYTES <= 232448, "SMEM layout exceeds 227 KB block ceiling");

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

#ifdef GELU_F32X2
/*
Packed-pair GELU + CVT: gelu_approx's op chain on f32x2 register pairs so
ptxas emits FADD2/FMUL2/FFMA2 (it never auto-packs scalars; probe-verified
2026-07-02). MUFU.TANH has no 2-wide form and stays scalar per element.
Vendor nvjet runs ~5.5 issue-slots/element with this shape vs ~7.5 scalar.
Poly association differs from gelu_approx ((k2*x)*x vs k2*(x*x)) — few-ULP
FP32 drift, absorbed by BF16 rounding and the validation tolerance.
Bias pair arrives as one BF16x2 word; result returns as one BF16x2 word.
*/
static __device__ __forceinline__ uint32_t gelu2_bf16x2(float a_lo, float a_hi,
                                                        uint32_t bias_pair) {
    uint32_t o;
    asm("{\n\t"
        ".reg .b32 blo, bhi;\n\t"
        ".reg .f32 i0, i1, t0, t1;\n\t"
        ".reg .b64 bx, x, s, u, t, k1, k2, hf;\n\t"
        "shl.b32 blo, %3, 16;\n\t"
        "and.b32 bhi, %3, 0xFFFF0000;\n\t"
        "mov.b64 bx, {blo, bhi};\n\t"
        "mov.b64 x, {%1, %2};\n\t"
        "add.rn.f32x2 x, x, bx;\n\t"
        "mul.rn.f32x2 s, x, x;\n\t"
        "mov.b64 k2, {0x3d12220c, 0x3d12220c};\n\t"
        "mov.b64 k1, {0x3f4c422a, 0x3f4c422a};\n\t"
        "fma.rn.f32x2 u, s, k2, k1;\n\t"
        "mul.rn.f32x2 u, u, x;\n\t"
        "mov.b64 {i0, i1}, u;\n\t"
        "tanh.approx.f32 t0, i0;\n\t"
        "tanh.approx.f32 t1, i1;\n\t"
        "mov.b64 t, {t0, t1};\n\t"
        "fma.rn.f32x2 u, x, t, x;\n\t"
        "mov.b64 hf, {0x3f000000, 0x3f000000};\n\t"
        "mul.rn.f32x2 u, u, hf;\n\t"
        "mov.b64 {i0, i1}, u;\n\t"
        "cvt.rn.bf16x2.f32 %0, i1, i0;\n\t"
        "}"
        : "=r"(o)
        : "f"(a_lo), "f"(a_hi), "r"(bias_pair));
    return o;
}
#endif

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
#ifdef GELU_F32X2
#define GELU_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7, b0,b1,b2,b3, SADDR) \
    do { \
        const uint32_t _o0 = gelu2_bf16x2(a0, a1, b0); \
        const uint32_t _o1 = gelu2_bf16x2(a2, a3, b1); \
        const uint32_t _o2 = gelu2_bf16x2(a4, a5, b2); \
        const uint32_t _o3 = gelu2_bf16x2(a6, a7, b3); \
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" \
            :: "r"(SADDR), "r"(_o0), "r"(_o1), "r"(_o2), "r"(_o3) \
            : "memory"); \
    } while(0)
#else
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
#endif /* GELU_F32X2 */

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
            asm volatile(EPI_WG_NONLAST ::: "memory"); \
    } \
    __syncwarp(); \
    EPI_SUBITER_BAR(); \
} while(0)

/* K-iteration macro (accumulating, ki >= 1). Single asm block emits all 4
   MMAs back-to-back so ptxas emits one elect+R2UR.BROADCAST wrapper for the
   group instead of one per MMA — matching rank-1's UTCQMMA emission density. */
static_assert(MMA_PER_KI == 4, "K_ITER_ACCUM hardcodes 4 sub-MMAs per K-iter");
#define K_ITER_ACCUM(S) do { \
    mbar_wait(tma_mbar[S], tma_phase[S]); \
    tma_phase[S] ^= 1; \
    asm volatile("tcgen05.fence::after_thread_sync;"); \
    { \
        uint64_t desc_a = desc_a_base[S], desc_b = desc_b_base[S]; \
        asm volatile( \
            "{\n\t" \
            ".reg .pred p;\n\t" \
            ".reg .b64 da, db;\n\t" \
            ".reg .b32 tc;\n\t" \
            "setp.ne.b32 p, 1, 0;\n\t" \
            "mov.b32 tc, %0;\n\t" \
            "mov.b64 da, %1;\n\t" \
            "mov.b64 db, %2;\n\t" \
            "tcgen05.mma.cta_group::2.kind::f8f6f4 " \
            "[tc], da, db, %3, {%4,%5,%6,%7, %8,%9,%10,%11}, p;\n\t" \
            "add.s64 da, da, 2;\n\t" \
            "add.s64 db, db, 2;\n\t" \
            "tcgen05.mma.cta_group::2.kind::f8f6f4 " \
            "[tc], da, db, %3, {%4,%5,%6,%7, %8,%9,%10,%11}, p;\n\t" \
            "add.s64 da, da, 2;\n\t" \
            "add.s64 db, db, 2;\n\t" \
            "tcgen05.mma.cta_group::2.kind::f8f6f4 " \
            "[tc], da, db, %3, {%4,%5,%6,%7, %8,%9,%10,%11}, p;\n\t" \
            "add.s64 da, da, 2;\n\t" \
            "add.s64 db, db, 2;\n\t" \
            "tcgen05.mma.cta_group::2.kind::f8f6f4 " \
            "[tc], da, db, %3, {%4,%5,%6,%7, %8,%9,%10,%11}, p;\n\t" \
            "}" \
            : \
            : "r"(buf * TN), "l"(desc_a), "l"(desc_b), "r"(IDESC), \
              "r"(0),"r"(0),"r"(0),"r"(0), \
              "r"(0),"r"(0),"r"(0),"r"(0)); \
    } \
    tcgen05_commit_mcast(mma_mbar[S], 0x3); \
} while(0)


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

#if TILE_DISPATCH == 0 || TILE_DISPATCH >= 8
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

#if TILE_DISPATCH >= 8
    /* Static swizzle: each cluster strides a linear index, static_swizzle() remaps. */
    const int tile_count = (TOTAL_TILES + num_clusters - 1) / num_clusters;
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

    /* ── Load bias into SMEM ──
       Skipped in STRIP_EPILOGUE / GEMM_ONLY — epilogue never reads bias in
       those modes, and the global LDG + syncthreads would thrash L2 and
       delay W0's TMA setup. */
#if !defined(STRIP_EPILOGUE) && !defined(GEMM_ONLY)
#ifdef LDG_BIAS
    /* LDG_BIAS: no SMEM bias — direct LDG from L1-cached global in epilogue */
#elif defined(BIAS_PER_TILE)
#if TILE_DISPATCH >= 8
    /* TD>=8: ring slots are filled per-tile by the epi warps' cp.async
       prefetch — nothing to preload */
#else
    /* Group-3: load 2 N-tiles' bias for snake ordering */
    {
        const int tn_b = TILES_N - 1 - tn_fixed;
        const uint32_t bias_saddr = smem_to_uint(smem + OFF_BIAS_SMEM);
        for (int i = tid; i < TN / 2; i += THREADS) {
            uint32_t va, vb;
            asm volatile("ld.global.b32 %0, [%1];" : "=r"(va) : "l"(bias + tn_fixed * TN + i * 2));
            asm volatile("ld.global.b32 %0, [%1];" : "=r"(vb) : "l"(bias + tn_b * TN + i * 2));
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(bias_saddr + i * 4), "r"(va));
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(bias_saddr + TN * 2 + i * 4), "r"(vb));
        }
    }
#endif /* TILE_DISPATCH >= 8 */
#else
    {
        const uint32_t bias_saddr = smem_to_uint(smem + OFF_BIAS_SMEM);
        for (int i = tid; i < N_DIM / 2; i += THREADS) {
            uint32_t val;
            asm volatile("ld.global.b32 %0, [%1];" : "=r"(val) : "l"(bias + i * 2));
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(bias_saddr + i * 4), "r"(val));
        }
    }
#endif
    __syncthreads();
#endif /* !STRIP_EPILOGUE && !GEMM_ONLY */


    /* ════════════════════════════════════════════
       MAIN TILE LOOP
       ════════════════════════════════════════════ */

#if TILE_DISPATCH >= 8
    int prev_tm = 0, prev_tn = 0;
#endif

    for (int _ti = 0; _ti < tile_count; _ti++) {
#if TILE_DISPATCH >= 8
#ifdef N_STAGGER
        /* Per-cluster rotation of the tile-visit order.  tile_count *
           num_clusters == TOTAL_TILES exactly (no invalid slots), so this
           is a bijective permutation of each cluster's tile set. */
        const int _ti_eff = (_ti + cluster_id * N_STAGGER) % tile_count;
#else
        const int _ti_eff = _ti;
#endif
        const int block_idx = _ti_eff * num_clusters + cluster_id;
        if (block_idx >= TOTAL_TILES) break;
        const int tile_idx = static_swizzle(block_idx);
#else
        const int _tm = m_rank + _ti * my_m_stride;
        if (_tm >= TILES_M) break;
        const int tile_idx = _tm * TILES_N + tn_fixed;
#endif
        const int buf = _ti & 1;
        int tm = tile_idx / TILES_N;
        int tn = tile_idx % TILES_N;
        if (SNAKE_ORDER && (tm & 1)) tn = TILES_N - 1 - tn;
#ifdef PACKED_TILES
        const int a_m_tile = tm * 2 + cta_rank;
        const int b_n_half = tn * 2 + cta_rank;
#else
        const int m_start = tm * TM * 2 + cta_rank * TM;
        const int n_start = tn * TN;
#endif
        const bool has_prev = (_ti > 0);

        if (warp == 0) {
            /* ── W0: TMA A/B loads ── */
            const uint32_t smem_base = warp_uniform(smem_to_uint(smem));
#if K_STAGGER
            const int k_shift_b = (cluster_id * K_STAGGER) % K_ITERS;
#endif
            PRAGMA_UNROLL(K_ITERS)
            for (int ki = 0; ki < K_ITERS; ki++) {
                const int s = ki % N_STAGES;
#if K_STAGGER
                const int k_block = (ki + k_shift_b) % K_ITERS;
#else
                const int k_block = ki;
#endif
#ifdef PACKED_TILES
                const int tma_c0   = 0;
                const int tma_a_c1 = (a_m_tile * K_ITERS + k_block) * TM;
                const int tma_b_c1 = (b_n_half * K_ITERS + k_block) * (TN/2);
#else
                const int tma_c0   = k_block * TK;
                const int tma_a_c1 = m_start;
                const int tma_b_c1 = n_start + cta_rank * (TN/2);
#endif
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
                        :: "r"(a_dst), "l"(&tma_a), "r"(tma_c0), "r"(tma_a_c1),
                           "r"(tma_mbar_s), "r"(a_dst + 16384), "l"(&tma_b),
                           "r"(tma_b_c1), "r"(TMA_BYTES)
                        : "memory");
                }
            }


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

                /* First K-iteration: MMA #0 initializes accumulator (p_init=0),
                   MMA #1-3 accumulate (p_acc=1). Combined into one asm block so
                   ptxas emits one elect+broadcast wrapper for all 4 MMAs. */
                {
                    uint64_t desc_a = desc_a_base[0], desc_b = desc_b_base[0];
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p_init, p_acc;\n\t"
                        ".reg .b64 da, db;\n\t"
                        ".reg .b32 tc;\n\t"
                        "setp.ne.b32 p_init, 0, 0;\n\t"
                        "setp.ne.b32 p_acc,  1, 0;\n\t"
                        "mov.b32 tc, %0;\n\t"
                        "mov.b64 da, %1;\n\t"
                        "mov.b64 db, %2;\n\t"
                        "tcgen05.mma.cta_group::2.kind::f8f6f4 "
                        "[tc], da, db, %3, {%4,%5,%6,%7, %8,%9,%10,%11}, p_init;\n\t"
                        "add.s64 da, da, 2;\n\t"
                        "add.s64 db, db, 2;\n\t"
                        "tcgen05.mma.cta_group::2.kind::f8f6f4 "
                        "[tc], da, db, %3, {%4,%5,%6,%7, %8,%9,%10,%11}, p_acc;\n\t"
                        "add.s64 da, da, 2;\n\t"
                        "add.s64 db, db, 2;\n\t"
                        "tcgen05.mma.cta_group::2.kind::f8f6f4 "
                        "[tc], da, db, %3, {%4,%5,%6,%7, %8,%9,%10,%11}, p_acc;\n\t"
                        "add.s64 da, da, 2;\n\t"
                        "add.s64 db, db, 2;\n\t"
                        "tcgen05.mma.cta_group::2.kind::f8f6f4 "
                        "[tc], da, db, %3, {%4,%5,%6,%7, %8,%9,%10,%11}, p_acc;\n\t"
                        "}"
                        :
                        : "r"(buf * TN), "l"(desc_a), "l"(desc_b), "r"(IDESC),
                          "r"(0),"r"(0),"r"(0),"r"(0),
                          "r"(0),"r"(0),"r"(0),"r"(0));
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
#if TILE_DISPATCH >= 8
                const int ptm = prev_tm;
                const int ptn = prev_tn;
#else
                const int prev_idx = (m_rank + (_ti - 1) * my_m_stride) * TILES_N + tn_fixed;
                int ptm = prev_idx / TILES_N;
                int ptn = prev_idx % TILES_N;
                if (SNAKE_ORDER && (ptm & 1)) ptn = TILES_N - 1 - ptn;
#endif
#ifdef PACKED_TILES
                const int prev_m = ((ptm * 2 + cta_rank) * TILES_N + ptn) * TM;
                const int prev_n = 0;
#else
                const int prev_m = ptm * TM * 2 + cta_rank * TM;
                const int prev_n = ptn * TN;
#endif

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
                    EPI_PRESTORE_SYNC();

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
#ifndef LDG_BIAS
            const uint32_t bias_saddr = smem_to_uint(smem + OFF_BIAS_SMEM);
#endif
#if defined(BIAS_PER_TILE) && TILE_DISPATCH >= 8 && !defined(LDG_BIAS)
            /* Ring prefetch: this tile's 512B bias slice, consumed next
               iteration (or in the drain) — a full epi drain of shadow. Each
               epi warp redundantly fills the whole slot (16B/lane); identical
               bytes make the cross-warp overlap benign and keep warps
               decoupled (no election, no barrier). */
            {
                const uint32_t dst = bias_saddr
                    + (uint32_t)((_ti % 3) * (TN * 2)) + lane * 16;
                const __nv_bfloat16* src = bias + tn * TN + lane * 8;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                    :: "r"(dst), "l"(src));
                asm volatile("cp.async.commit_group;" ::: "memory");
            }
#endif

            const uint32_t xor_val = (lane & 7) << 4;
            const uint32_t sw0 = 0 ^ xor_val, sw1 = 16 ^ xor_val;
            const uint32_t sw2 = 32 ^ xor_val, sw3 = 48 ^ xor_val;
            const uint32_t sw4 = 64 ^ xor_val, sw5 = 80 ^ xor_val;
            const uint32_t sw6 = 96 ^ xor_val, sw7 = 112 ^ xor_val;

            if (has_prev) {
#if TILE_DISPATCH >= 8
                const int ptm = prev_tm;
                const int ptn = prev_tn;
#else
                const int prev_idx = (m_rank + (_ti - 1) * my_m_stride) * TILES_N + tn_fixed;
                int ptm = prev_idx / TILES_N;
                int ptn = prev_idx % TILES_N;
                if (SNAKE_ORDER && (ptm & 1)) ptn = TILES_N - 1 - ptn;
#endif
#ifdef PACKED_TILES
                const int prev_m = ((ptm * 2 + cta_rank) * TILES_N + ptn) * TM;
                const int prev_n = 0;
                const int prev_n_bias = ptn * TN;
#else
                const int prev_m = ptm * TM * 2 + cta_rank * TM;
                const int prev_n = ptn * TN;
                const int prev_n_bias = prev_n;
#endif

                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#if defined(BIAS_PER_TILE) && TILE_DISPATCH >= 8 && !defined(LDG_BIAS)
                /* Prev tile's prefetch must have landed; this tile's own
                   (just issued) may still fly → depth 1, not 0. */
                asm volatile("cp.async.wait_group 1;" ::: "memory");
#endif

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

#ifdef LDG_BIAS
                        /* LDG bias directly from L1-cached global memory */
                        const __nv_bfloat16* bp = bias + prev_n_bias + nc;
                        uint4 bv0, bv1, bv2, bv3;
                        asm volatile("ld.global.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(bv0.x),"=r"(bv0.y),"=r"(bv0.z),"=r"(bv0.w) : "l"(bp));
                        asm volatile("ld.global.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(bv1.x),"=r"(bv1.y),"=r"(bv1.z),"=r"(bv1.w) : "l"(bp + 8));
                        asm volatile("ld.global.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(bv2.x),"=r"(bv2.y),"=r"(bv2.z),"=r"(bv2.w) : "l"(bp + 16));
                        asm volatile("ld.global.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(bv3.x),"=r"(bv3.y),"=r"(bv3.z),"=r"(bv3.w) : "l"(bp + 24));
#else
#if defined(BIAS_PER_TILE)
#if TILE_DISPATCH >= 8
                        const uint32_t bs = bias_saddr
                            + (uint32_t)(((_ti - 1) % 3) * (TN * 2))
                            + nc * 2;
#else
                        const uint32_t bs = bias_saddr
                            + ((ptn == tn_fixed) ? 0u : (unsigned)(TN * 2))
                            + nc * 2;
#endif
#else
                        /* LDS bias from SMEM (linear, not swizzled) */
                        const uint32_t bs = bias_saddr + (prev_n_bias + nc) * 2;
#endif
                        uint4 bv0, bv1, bv2, bv3;
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(bv0.x),"=r"(bv0.y),"=r"(bv0.z),"=r"(bv0.w) : "r"(bs));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(bv1.x),"=r"(bv1.y),"=r"(bv1.z),"=r"(bv1.w) : "r"(bs + 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(bv2.x),"=r"(bv2.y),"=r"(bv2.z),"=r"(bv2.w) : "r"(bs + 32));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(bv3.x),"=r"(bv3.y),"=r"(bv3.z),"=r"(bv3.w) : "r"(bs + 48));
#endif

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
                    EPI_PRESTORE_SYNC();

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
#if TILE_DISPATCH >= 8
        prev_tm = tm;
        prev_tn = tn;
#endif
    }

    /* ══════════════════════════════════════════════
       DRAIN: last tile epilogue
       ══════════════════════════════════════════════ */
    {
#if TILE_DISPATCH >= 8
        const int last_idx = static_swizzle((tile_count - 1) * num_clusters + cluster_id);
        const int last_buf = (tile_count - 1) & 1;
#else
        const int last_idx = (m_rank + (tile_count - 1) * my_m_stride) * TILES_N + tn_fixed;
        const int last_buf = (tile_count - 1) & 1;
#endif
        int ltm = last_idx / TILES_N;
        int ltn = last_idx % TILES_N;
        if (SNAKE_ORDER && (ltm & 1)) ltn = TILES_N - 1 - ltn;
#ifdef PACKED_TILES
        const int last_m = ((ltm * 2 + cta_rank) * TILES_N + ltn) * TM;
        const int last_n = 0;
        const int last_n_bias = ltn * TN;
#else
        const int last_m = ltm * TM * 2 + cta_rank * TM;
        const int last_n = ltn * TN;
        const int last_n_bias = last_n;
#endif

        if (warp == 0 || warp == 1) {
            /* W0/W1: nothing to do for drain */
        } else {
            {
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
                EPI_PRESTORE_SYNC();

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
#ifndef LDG_BIAS
            const uint32_t bias_saddr = smem_to_uint(smem + OFF_BIAS_SMEM);
#endif
            const uint32_t xor_val = (lane & 7) << 4;
            const uint32_t sw0 = 0 ^ xor_val, sw1 = 16 ^ xor_val;
            const uint32_t sw2 = 32 ^ xor_val, sw3 = 48 ^ xor_val;
            const uint32_t sw4 = 64 ^ xor_val, sw5 = 80 ^ xor_val;
            const uint32_t sw6 = 96 ^ xor_val, sw7 = 112 ^ xor_val;

            mbar_wait(mainloop_mbar_addr + last_buf * 8, ml_phase[last_buf]);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            ml_phase[last_buf] ^= 1;
#if defined(BIAS_PER_TILE) && TILE_DISPATCH >= 8 && !defined(LDG_BIAS)
            /* Last tile's prefetch (issued in its own loop iteration) */
            asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif

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

#ifdef LDG_BIAS
                    const __nv_bfloat16* bp = bias + last_n_bias + nc;
                    uint4 bv0, bv1, bv2, bv3;
                    asm volatile("ld.global.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(bv0.x),"=r"(bv0.y),"=r"(bv0.z),"=r"(bv0.w) : "l"(bp));
                    asm volatile("ld.global.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(bv1.x),"=r"(bv1.y),"=r"(bv1.z),"=r"(bv1.w) : "l"(bp + 8));
                    asm volatile("ld.global.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(bv2.x),"=r"(bv2.y),"=r"(bv2.z),"=r"(bv2.w) : "l"(bp + 16));
                    asm volatile("ld.global.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(bv3.x),"=r"(bv3.y),"=r"(bv3.z),"=r"(bv3.w) : "l"(bp + 24));
#else
#if defined(BIAS_PER_TILE)
#if TILE_DISPATCH >= 8
                    const uint32_t bs = bias_saddr
                        + (uint32_t)(((tile_count - 1) % 3) * (TN * 2))
                        + nc * 2;
#else
                    const uint32_t bs = bias_saddr
                        + ((ltn == tn_fixed) ? 0u : (unsigned)(TN * 2))
                        + nc * 2;
#endif
#else
                    const uint32_t bs = bias_saddr + (last_n_bias + nc) * 2;
#endif
                    uint4 bv0, bv1, bv2, bv3;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(bv0.x),"=r"(bv0.y),"=r"(bv0.z),"=r"(bv0.w) : "r"(bs));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(bv1.x),"=r"(bv1.y),"=r"(bv1.z),"=r"(bv1.w) : "r"(bs + 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(bv2.x),"=r"(bv2.y),"=r"(bv2.z),"=r"(bv2.w) : "r"(bs + 32));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(bv3.x),"=r"(bv3.y),"=r"(bv3.z),"=r"(bv3.w) : "r"(bs + 48));
#endif

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
                EPI_PRESTORE_SYNC();

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

#ifndef SELF_DIFF
#define SELF_DIFF 0
#endif

/*
  Stream-serialized SM-clock sentinel: single thread reads %%clock64 on the
  default stream before/after the timed loop, so (end-start)/launches is avg
  SM cycles per launch in the same clock domain as every other harness that
  emits @@CYC — cross-binary comparable where wall-ms is DVFS-confounded.
*/
__global__ void read_clock_sentinel(unsigned long long* out) {
    if (threadIdx.x == 0) {
        unsigned long long c;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c));
        out[0] = c;
    }
}

#if SELF_DIFF
/* Double-launch bitwise self-diff: any element differing between two launches
   on identical inputs is a race witness (same detector as fc2_w3's). FC1 runs
   the auto-NO_PREFILL back-pressured path, so expect dirty=0. */
__global__ void count_mismatch(const __nv_bfloat16* __restrict__ a,
                               const __nv_bfloat16* __restrict__ b,
                               long long total,
                               unsigned long long* __restrict__ mm) {
    unsigned long long local = 0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        unsigned short ba = *reinterpret_cast<const unsigned short*>(&a[i]);
        unsigned short bb = *reinterpret_cast<const unsigned short*>(&b[i]);
        if (ba != bb) local++;
    }
    if (local) atomicAdd(mm, local);
}
#endif

#ifdef PACKED_TILES
__global__ void pack_u8(uint8_t* __restrict__ dst, const uint8_t* __restrict__ src,
                        int M, int K, int tile_m, int tile_k) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)M * K;
    if (idx >= total) return;
    int m = (int)(idx / K);
    int k = (int)(idx % K);
    int local_m = m % tile_m;
    int local_k = k % tile_k;
    int tiles_k = K / tile_k;
    long long packed = (long long)(m / tile_m) * tiles_k * tile_m * tile_k
                     + (long long)(k / tile_k) * tile_m * tile_k
                     + (long long)local_m * tile_k + local_k;
    dst[packed] = src[idx];
}
#endif

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
#ifdef PACKED_TILES
    {
        printf("  Packing tiles...\n");
        int tpb = 256;
        {
            uint8_t* d_tmp;
            CUDA_CHECK(cudaMalloc(&d_tmp, (size_t)M_TOTAL * K_DIM));
            long long n = (long long)M_TOTAL * K_DIM;
            pack_u8<<<(int)((n+tpb-1)/tpb), tpb>>>(d_tmp, d_A, M_TOTAL, K_DIM, TM, TK);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(d_A, d_tmp, (size_t)M_TOTAL * K_DIM, cudaMemcpyDeviceToDevice));
            cudaFree(d_tmp);
        }
        {
            uint8_t* d_tmp;
            CUDA_CHECK(cudaMalloc(&d_tmp, (size_t)N_DIM * K_DIM));
            long long n = (long long)N_DIM * K_DIM;
            pack_u8<<<(int)((n+tpb-1)/tpb), tpb>>>(d_tmp, d_B, N_DIM, K_DIM, TN/2, TK);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(d_B, d_tmp, (size_t)N_DIM * K_DIM, cudaMemcpyDeviceToDevice));
            cudaFree(d_tmp);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("  Packing done\n");
    }
#endif
    printf("  Alloc + init done\n");

    /* TMA descriptors */
    CUtensorMap h_tma_a, h_tma_b;
#ifdef PACKED_TILES
    {
        uint64_t a_total_rows = (uint64_t)(M_TOTAL / TM) * K_ITERS * TM;
        uint64_t dims[2]    = {(uint64_t)TK, a_total_rows};
        uint64_t strides[1] = {(uint64_t)TK};
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
        uint64_t b_total_rows = (uint64_t)(N_DIM / (TN/2)) * K_ITERS * (TN/2);
        uint64_t dims[2]    = {(uint64_t)TK, b_total_rows};
        uint64_t strides[1] = {(uint64_t)TK};
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
#else
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
#endif

    CUtensorMap h_tma_c;
#ifdef PACKED_TILES
    {
        uint64_t c_total_rows = (uint64_t)(M_TOTAL / TM) * TILES_N * TM;
        uint64_t dims[2]    = {(uint64_t)TN, c_total_rows};
        uint64_t strides[1] = {(uint64_t)TN * sizeof(__nv_bfloat16)};
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
#else
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
#endif

    CUDA_CHECK(cudaFuncSetAttribute(fc1_w3_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    printf("  TMA descriptors + func attr done (SMEM=%d B)\n", SMEM_BYTES);

#define LAUNCH_KERNEL() \
    fc1_w3_kernel<<<SM_COUNT, THREADS, SMEM_BYTES>>>( \
        h_tma_a, h_tma_b, h_tma_c, d_bias, d_C)

    /* Warmup */
    printf("Launching warmup (2 iters)...\n");
    for (int i = 0; i < 2; i++) {
        LAUNCH_KERNEL();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Warmup done.\n");

    /* Timed: 10 iterations */
    printf("Timing: 10 iterations...\n");
    unsigned long long* d_clk = nullptr;
    CUDA_CHECK(cudaMalloc(&d_clk, 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    read_clock_sentinel<<<1, 1>>>(d_clk + 0);
    cudaEventRecord(t0);
    for (int i = 0; i < 10; i++) {
        LAUNCH_KERNEL();
    }
    cudaEventRecord(t1);
    read_clock_sentinel<<<1, 1>>>(d_clk + 1);
    cudaEventSynchronize(t1);
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= 10.0f;
    printf("FC1-W3 kernel: %.3f ms  %.2f TFLOPS\n",
           ms, 2.0 * M_TOTAL * N_DIM * K_DIM / ms / 1e9);
    {
        unsigned long long clk[2];
        CUDA_CHECK(cudaMemcpy(clk, d_clk, sizeof(clk), cudaMemcpyDeviceToHost));
        printf("@@CYC name=fc1_w3 cyc_avg=%llu launches=10\n",
               (clk[1] > clk[0]) ? (clk[1] - clk[0]) / 10ULL : 0ULL);
    }
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
#ifdef PACKED_TILES
        long long packed_idx = (long long)((int)(row / TM) * TILES_N + col / TN) * TM * TN
                             + (long long)((int)(row % TM)) * TN + (col % TN);
        __nv_bfloat16 actual = h_C[packed_idx];
#else
        __nv_bfloat16 actual = h_C[row * N_DIM + col];
#endif
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

#if SELF_DIFF
    {
        const long long sd_total = (long long)M_TOTAL * N_DIM;
        __nv_bfloat16* d_Cref = nullptr;
        unsigned long long* d_mm = nullptr;
        CUDA_CHECK(cudaMalloc(&d_Cref, (size_t)sd_total * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_mm, sizeof(unsigned long long)));
        const int sd_tpb = 256;
        int sd_bpg = (int)((sd_total + sd_tpb - 1) / sd_tpb);
        if (sd_bpg > 65535) sd_bpg = 65535;
        int dirty = 0; unsigned long long worst = 0;
        printf("@@SELFDIFF_BEGIN launches=%d\n", (int)SELF_DIFF);
        for (int k = 0; k < (int)SELF_DIFF; k++) {
            LAUNCH_KERNEL();
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(d_Cref, d_C, (size_t)sd_total * sizeof(__nv_bfloat16),
                                  cudaMemcpyDeviceToDevice));
            LAUNCH_KERNEL();
            CUDA_CHECK(cudaDeviceSynchronize());
            unsigned long long z = 0;
            CUDA_CHECK(cudaMemcpy(d_mm, &z, sizeof(z), cudaMemcpyHostToDevice));
            count_mismatch<<<sd_bpg, sd_tpb>>>(d_C, d_Cref, sd_total, d_mm);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            unsigned long long n = 0;
            CUDA_CHECK(cudaMemcpy(&n, d_mm, sizeof(n), cudaMemcpyDeviceToHost));
            if (n > 0) { dirty++; if (n > worst) worst = n; }
            printf("@@SELFDIFF iter=%d mismatches=%llu\n", k, n);
        }
        printf("@@SELFDIFF_SUMMARY launches=%d dirty=%d worst=%llu\n",
               (int)SELF_DIFF, dirty, worst);
        cudaFree(d_Cref); cudaFree(d_mm);
    }
#endif

    free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_bias); cudaFree(d_C);
    return 0;
}
