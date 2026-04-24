/*
  fc2_w3x.cu — FC2 bias-only kernel, 6-warp rank-1-shaped persistent

  Target: beat cuBLASLt rank-1 `nvjet_sm100_qqtst_128x256_128x6_2x1_2cta_v_bz_bias_TNT`
          (1.046 ms on B200 at K=3072, FC2 shape).

  Architecture (per CTA, default 6 warps × 32 threads = 192 thr):
    tid   0..127 (W0-W3)  Epilogue warpgroup   setmaxnreg.inc 232
    tid 128..159 (W4)     TMA A+B + bias LDG   setmaxnreg.dec  24
    tid 160..191 (W5)     MMA issuer (CTA 0)   setmaxnreg.dec  24
                          CTA 1: W5 exits early after init barrier

  With -DEPI_2WARP: 4 warps × 32 thr = 128 thr; roles renumbered via
  N_EPI_WARPS/WARP_TMA/WARP_MMA. Each epi warp covers 64 rows (2× rows,
  via an inner ROW_HALVES loop) so output coverage is unchanged.

  Cluster: 2x1, cta_group::2, per-CTA tile 128x256 → cluster output 256x256.
  Total 384 threads / 2-CTA cluster, warps_act=6 on CTA 0, warps_act=5 on CTA 1.

  Key design choices (vs fc2_ws):
    - 6 warps (192 thr) not 8 (256): cleaner launch_bounds, drops dead W6 + W7.
    - setmaxnreg: epi warps get 232 regs, TMA/MMA get 24 — rank-1-shaped.
    - PREFILL (default): W5 MMA for tile N+1 overlaps with W0-W3 epilogue for
      tile N. Safe: TMEM is double-buffered (buf = tile_idx & 1), MMA writes
      buf while epi reads prev_buf. Opt-out via -DNO_PREFILL.
    - PACKED_TILES dispatch only (no #ifdef — this is our cuBLASLt edge).
    - dgswizzle(DG_GROUP_SIZE=8), no work-stealing (persistent, M%SM==0).
    - No residual path — BIAS_ONLY is the contract.

  Constraint: persistent kernel, TOTAL_TILES % num_clusters == 0. FC2 siglip
  shape (M=928256, N=768, K=3072) gives TILES_M=3626, num_clusters=74,
  3626/74=49 exact, TOTAL_TILES=10878, 10878/74=147 exact. Good.
*/

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>

#ifndef M_TOTAL
#define M_TOTAL 928256
#endif
#ifndef N_DIM
#define N_DIM   768
#endif
#ifndef K_DIM
#define K_DIM   3072
#endif

#define TM          128
#define TN          256
#define TK          128
#define SM_COUNT    148
#define CLUSTER_CTAS 2
#define THREADS     (TOTAL_WARPS * 32)

#define TILES_M     ((M_TOTAL + TM * 2 - 1) / (TM * 2))
#define TILES_N     ((N_DIM  + TN - 1) / TN)
#define TOTAL_TILES (TILES_M * TILES_N)
#define K_ITERS     (K_DIM / TK)

#define N_STAGES       6
#define NUM_EPI_STAGES 2
#define NUM_SUBPASSES  (TN / 32)

#ifdef EPI_2WARP
#define N_EPI_WARPS      2
#define ROWS_PER_WARP    64
#define EPI_BARSYNC_ASM  "bar.sync 0, 64;"
#define WARP_TMA         2
#define WARP_MMA         3
#define TOTAL_WARPS      4
#else
#define N_EPI_WARPS      4
#define ROWS_PER_WARP    32
#define EPI_BARSYNC_ASM  "bar.sync 0, 128;"
#define WARP_TMA         4
#define WARP_MMA         5
#define TOTAL_WARPS      6
#endif
#define ROW_HALVES       (ROWS_PER_WARP / 32)

#define SUBPASS_COLS   32
#define ROWS_PER_CTA   TM
#define SUBPASS_BYTES  (ROWS_PER_CTA * SUBPASS_COLS * 2)

#define STAGE_BYTES    32768
#define TMA_BYTES      32768
#define MAIN_SMEM      (N_STAGES * STAGE_BYTES)
#define OUT_STAGING    (NUM_EPI_STAGES * SUBPASS_BYTES)
#define BIAS_BYTES     (N_DIM * 2)

#define OFF_AB         0
#define OFF_OUT        MAIN_SMEM
#define OFF_BIAS       ((OFF_OUT + OUT_STAGING + 127) & ~127)
#define OFF_MBARS      ((OFF_BIAS + BIAS_BYTES + 127) & ~127)

#define MBAR_TMA_FULL       (OFF_MBARS + 0)
#define MBAR_TMA_EMPTY      (MBAR_TMA_FULL + N_STAGES * 8)
#define MBAR_TMEM_READY     (MBAR_TMA_EMPTY + N_STAGES * 8)
#define MBAR_TMEM_CONSUMED  (MBAR_TMEM_READY + 2 * 8)
#define MBARS_END           (MBAR_TMEM_CONSUMED + 2 * 8)

#ifdef XPF_A
/*
  Cross-group A-prefetch (Option A): SMEM-landing TMA for next group's A.
  W4 issues at end of tn=2 tiles into a dedicated 16KB scratch SMEM slot
  with its own mbarrier, cycling parity. Data is never consumed; the issue
  populates L2 as side effect.
*/
#define PF_A_BYTES       16384
#define OFF_PF_A_DATA    ((MBARS_END + 127) & ~127)
#define OFF_PF_A_MBAR    ((OFF_PF_A_DATA + PF_A_BYTES + 127) & ~127)
#define PF_A_END         (OFF_PF_A_MBAR + 8)
#else
#define PF_A_END         MBARS_END
#endif

#define OFF_TMEM       ((PF_A_END + 15) & ~15)
#ifdef PROFILE_KI
#ifdef PROFILE_KI_TN
#define PROF_KI_SLOTS  (K_ITERS * ((N_DIM + TN - 1) / TN))
#else
#define PROF_KI_SLOTS  K_ITERS
#endif
#define PROF_KI_BYTES  (PROF_KI_SLOTS * 8)
#define OFF_PROF_KI    ((OFF_TMEM + 8 + 15) & ~15)
#define SMEM_BYTES     ((OFF_PROF_KI + PROF_KI_BYTES + 127) & ~127)
#else
#define SMEM_BYTES     ((OFF_TMEM + 8 + 127) & ~127)
#endif
/* PROFILE_TILE uses only a register + one global write per tile — no SMEM. */

#define TMEM_COLS    512
#define IDESC        0x10400010U
#define SBO          1024

#ifndef DG_GROUP_SIZE
#define DG_GROUP_SIZE 8
#endif

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

static __device__ __forceinline__
uint32_t smem_to_uint(const void* p) {
    return static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(__cvta_generic_to_shared(p)));
}

static __device__ __forceinline__
uint64_t make_smem_desc(uint32_t addr) {
    uint64_t d = 0;
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= (uint64_t)((SBO & 0x3FFFF) >> 4) << 32;
    d |= (1ULL << 46);
    d |= (2ULL << 61);
    return d;
}

/*
  Dispatch variants for fc2_w3x (all strided layout: lin = c + tt*NC).

  Base: dgswizzle(DG_GROUP_SIZE=G). Within each group of G*TILES_N tiles,
  iterate tm fastest then tn (tn-run length = G).

  Flags (mutually exclusive except where noted):
    DG_ROT (existing P2 probe, 0-delta perf): rotate tn by group_idx —
      spreads group-boundary position across all tn values.
    DG_INNER_T (NEW): transpose within-group decomposition — iterate tn
      fastest then tm (tn-run length = 1). Breaks per-tn sub-run
      coherence; each tile cycles through all TILES_N columns. Bijective:
      same (tm, tn) set within a group, different traversal order.
    DG_STAGGER (NEW): shift lin by c*group_tiles per cluster so cluster c
      starts at group_idx=c instead of group_idx=c/gt. The 74 clusters
      span 74 distinct group_idx values at tt=0 instead of 4.
      Bijectivity proof: lin' = 25c + 74tt (mod 10878) for FC2
      (gt=24, NC=74, TPC=147, TOTAL=10878=NC*TPC). gcd(25, 10878)=1
      and gcd(74, 10878)=74, so the coset {25c + 74t mod 10878 : t} has
      10878/74=147 elements = TPC, and c → (25c mod 74) is a bijection
      on [0, 74), so different c land in different cosets. Bijective.
    DG_GROUP_SIZE=N (build-time): override G ∈ {4, 8, 16, 32}. Trades
      tn-run length vs group count. Affects all variants.

  Sweep suggestion (one binary per config):
    baseline (G=8):                make -B fc2-w3x
    G=4:                           make -B fc2-w3x DFLAGS='-DDG_GROUP_SIZE=4'
    G=16:                          make -B fc2-w3x DFLAGS='-DDG_GROUP_SIZE=16'
    G=32:                          make -B fc2-w3x DFLAGS='-DDG_GROUP_SIZE=32'
    inner-transpose:               make -B fc2-w3x DFLAGS='-DDG_INNER_T'
    cluster-group-stagger:         make -B fc2-w3x DFLAGS='-DDG_STAGGER'
*/
static __device__ __forceinline__
int dgswizzle(int lin) {
    const int group_tiles = TILES_N * DG_GROUP_SIZE;

#ifdef DG_STAGGER
    const int _NC = SM_COUNT / CLUSTER_CTAS;
    const int _TOTAL = TILES_M * TILES_N;
    const int _c = lin - (lin / _NC) * _NC;
    lin = lin + _c * group_tiles;
    if (lin >= _TOTAL) lin -= _TOTAL;
#endif

    const int group_idx = lin / group_tiles;
    const int first_m = group_idx * DG_GROUP_SIZE;
    const int in_group = lin - group_idx * group_tiles;
    const int nig = (first_m + DG_GROUP_SIZE <= TILES_M)
                  ? DG_GROUP_SIZE
                  : TILES_M - first_m;

    int tm_local, tn_raw;
#ifdef DG_INNER_T
    tm_local = in_group / TILES_N;
    tn_raw   = in_group - tm_local * TILES_N;
    if (tm_local >= nig) tm_local = nig - 1;
#else
    tm_local = in_group % nig;
    tn_raw   = in_group / nig;
#endif

#ifdef DG_ROT
    int tn = tn_raw + group_idx;
    while (tn >= TILES_N) tn -= TILES_N;
#else
    const int tn = tn_raw;
#endif
    return (first_m + tm_local) * TILES_N + tn;
}

static __device__ __forceinline__
int dgswizzle_in_group(int lin) {
    const int group_tiles = TILES_N * DG_GROUP_SIZE;
#ifdef DG_STAGGER
    const int _NC = SM_COUNT / CLUSTER_CTAS;
    const int _TOTAL = TILES_M * TILES_N;
    const int _c = lin - (lin / _NC) * _NC;
    int lin_s = lin + _c * group_tiles;
    if (lin_s >= _TOTAL) lin_s -= _TOTAL;
    return lin_s - (lin_s / group_tiles) * group_tiles;
#else
    return lin - (lin / group_tiles) * group_tiles;
#endif
}

/*
  Non-dgswizzle dispatch variants from tile_dispatch.cuh.  Set -DTILE_DISPATCH=N
  where N chooses a static_swizzle (see tile_dispatch.cuh):
     9 zorder, 10 hilbert, 11 zigzag, 13 rowmajor,
    14 ncycle, 15 nflat, 16 nsnake, 17 nlock, 18 checkered,
    19 dg-snake (zigzag within dgswizzle band), 21 ncyrot.
  Unset (default) → dgswizzle as currently shipping.  tile_swizzle() is the
  single entry point; PROFILE_{W4,TILE}'s in_g field is only meaningful under
  dgswizzle (returns 0 otherwise).
*/
#ifdef TILE_DISPATCH
#include "tile_dispatch.cuh"
#endif

static __device__ __forceinline__
int tile_swizzle(int lin) {
#if defined(TILE_DISPATCH) && TILE_DISPATCH >= 8
    return static_swizzle(lin);
#else
    return dgswizzle(lin);
#endif
}

static __device__ __forceinline__
int tile_in_group(int lin) {
#if defined(TILE_DISPATCH) && TILE_DISPATCH >= 8
    (void)lin;
    return 0;
#else
    return dgswizzle_in_group(lin);
#endif
}

static __device__ __forceinline__
void mbar_init(uint32_t addr, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(addr), "r"(count));
}

static __device__ __forceinline__
void mbar_arrive(uint32_t addr) {
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
        :: "r"(addr) : "memory");
}

static __device__ __forceinline__
void mbar_wait(uint32_t addr, uint32_t phase) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "LOOP: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1;\n\t"
                 "@p bra DONE;\n\t"
                 "nanosleep.u32 20;\n\t"
                 "bra LOOP;\n\t"
                 "DONE:\n\t"
                 "}"
        :: "r"(addr), "r"(phase));
}

static __device__ __forceinline__
void tcgen05_commit_mcast(uint32_t mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        :: "r"(mbar_addr), "h"(cta_mask) : "memory");
}

static __device__ __forceinline__
void tma_store(uint32_t smem_src, const CUtensorMap* tma_desc, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tma_desc), "r"(c0), "r"(c1), "r"(smem_src) : "memory");
}

#define TMEM_LOAD_X32(r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31, TADDR) \
    asm volatile( \
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 " \
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];" \
        : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3),"=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7), \
          "=f"(r8),"=f"(r9),"=f"(r10),"=f"(r11),"=f"(r12),"=f"(r13),"=f"(r14),"=f"(r15), \
          "=f"(r16),"=f"(r17),"=f"(r18),"=f"(r19),"=f"(r20),"=f"(r21),"=f"(r22),"=f"(r23), \
          "=f"(r24),"=f"(r25),"=f"(r26),"=f"(r27),"=f"(r28),"=f"(r29),"=f"(r30),"=f"(r31) \
        : "r"(TADDR))

/*
  Lever C — stmatrix-native TMEM load.

  tcgen05.ld.sync.aligned.16x256b.x4.b32 loads 16 TMEM rows × (4 × 256 bits)
  into 16 b32 regs per lane.  Register layout is stmatrix.x4 compatible:
  lane t holds 4 chunks of 4 regs each, one chunk per 8×8 matrix, where the
  chunk for matrix c on lane t=8c+r contains the 8 bf16 values packed as
  4 bf16x2 regs that stmatrix.x4.trans.m8n8 expects (8 rows of col r).

  2 such LDTMs (at TMEM row offsets 0 and +16 via +0x100000 in bits 20..) cover
  our 32-row rh.  This replaces 1 LDTM.32x32b.x32 (which gives lane t all cols
  of row t — wrong layout for stmatrix).

  SASS: LDTM.16dp256bit.x4  (matches rank-1 `..._bz_bias_TNT`).
*/
#define TMEM_LOAD_16X256_X4(r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15, TADDR) \
    asm volatile( \
        "tcgen05.ld.sync.aligned.16x256b.x4.b32 " \
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];" \
        : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3),"=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7), \
          "=f"(r8),"=f"(r9),"=f"(r10),"=f"(r11),"=f"(r12),"=f"(r13),"=f"(r14),"=f"(r15) \
        : "r"(TADDR))

#define TMEM_WAIT() \
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory")

/*
  Lever C — STSM (stmatrix) SMEM store.

  stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 stores 4 8×8 bf16 matrices
  per call, one per lane-octet.  Per lane t=8c+r, the 4 b32 source regs are
  interpreted as 8 bf16 values = column r (rows 0..7) of matrix c.  With the
  `.trans` variant, the matrix is transposed on write so SMEM sees the data
  row-major.

  Lane addresses:  each lane supplies an address; stmatrix uses addresses
  from lane (8c + r), r∈[0,7] as the row-r offset within matrix c's 128B
  tile.  We place matrix c (covering output cols 8c..8c+7) at SMEM offset
  `out_base + row_start*64 + c*16`, with lane t=8c+r providing
  `out_base + (row_start + r)*64 + c*16` → produces linear row-major SMEM
  (row stride 64B = 32 bf16 cols), identical to the layout the current STS
  path writes and the TMA-C store expects (SWIZZLE_NONE).

  NOTE: rank-1's SMEM layout differs — it uses SMEM bit-20 XOR swizzling
  with SWIZZLE_32B-shaped TMA.  Our variant preserves the existing
  SWIZZLE_NONE TMA store descriptor.  This is a valid alternative layout
  for stmatrix; the register routing is what the `.trans` instruction
  demands, the SMEM destination math is ours.

  SASS: STSM.16.MT88.4 (matches rank-1's `bz_bias_TNT`).
*/
#ifndef USE_STMATRIX
#define USE_STMATRIX 0
#endif

#define STSM_X4_TRANS(SADDR, r0, r1, r2, r3) \
    asm volatile( \
        "stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1,%2,%3,%4};" \
        :: "r"(SADDR), "r"(r0), "r"(r1), "r"(r2), "r"(r3))

/*
  CVT_ADD_BF16X2 — native bf16 epilogue math.
  Pack the two fp32 accumulator lanes into one bf16x2 register (F2FP.PACK),
  then add the packed bf16x2 bias in one HADD2. Output is bf16x2 ready
  for the STS.

  Collapses { 2× FADD fp32 + 1× cvt.bf16x2.f32 } per pair into
  { 1× F2FP.PACK + 1× HADD2 }. The acc rounds to bf16 before the add,
  so precision differs from rank-1's fp32-add-then-convert pattern.
  Output-valid to 2% rel for sensible acc scales; SigLIP2 FC2 accs
  stay well within bf16's representable add range.
*/
#define CVT_ADD_BF16X2(p_out, a_lo, a_hi, b_in) \
    asm volatile( \
        "{\n\t" \
        ".reg .b32 ap;\n\t" \
        "cvt.rn.bf16x2.f32 ap, %2, %1;\n\t" \
        "add.rn.bf16x2 %0, ap, %3;\n\t" \
        "}" \
        : "=r"(p_out) \
        : "f"(a_lo), "f"(a_hi), "r"(b_in))

/*
  Async bulk completion fences.  TMA-store only uses the SMEM-read side of
  the bulk path (we never bulk-load here, so there's no global-side state
  to reconcile), but ptxas emits CCTL.IVALL (L1 invalidate) after every
  DEPBAR.LE SB0 sequence to cover the load case conservatively.

  Macro switches:
    NO_BULK_MEMCLBR — drop the "memory" clobber on the wait_group /
      commit_group asm.  Ineffective: SASS is bit-identical to baseline
      because the CCTL.IVALL isn't a clobber artifact but a semantic
      requirement of cp.async.bulk.wait_group.
    WAIT_GROUP_READ — use cp.async.bulk.wait_group.read variant, which
      only waits for the SMEM-read side of bulk ops (sufficient when we
      only bulk-store).  Goal: let ptxas skip CCTL.IVALL.

  Verified that NO_BULK_MEMCLBR produces identical SASS.  WAIT_GROUP_READ
  is the live experiment.
*/
#ifdef NO_BULK_MEMCLBR
#define BULK_ASM(CODE) asm volatile(CODE)
#else
#define BULK_ASM(CODE) asm volatile(CODE ::: "memory")
#endif

#ifdef WAIT_GROUP_READ
#define BULK_WAIT_GROUP(N) "cp.async.bulk.wait_group.read " #N ";"
#else
#define BULK_WAIT_GROUP(N) "cp.async.bulk.wait_group " #N ";"
#endif

/*
  PROFILE_CYCLES — per-warp clock64 phase accumulators.
  Each warp's lane 0 sums cycles per phase; at kernel exit, lane 0 dumps
  prof[] to d_dbg_prof[cluster*2*TOTAL_WARPS + cta_rank*TOTAL_WARPS + warp_id].
  Host aggregates and prints mean cyc/tile per (warp_role, phase).
  Clock64 overhead is ~1-2 cyc per read → ~6 cyc per bracketed phase.
  Use the breakdown for *relative* stall diagnosis, not exact timing.

  Phase semantics per warp role:
    Epi (W0..W_{N_EPI_WARPS-1}):
      0: TMEM ready mbar_wait (per tile)
      1: subpass body — bias LDS + TMEM_LOAD + HFMA2 + CVT + STS + fence
      2: first bar.sync wait (cross-warp STS sync)
      3: tid==0 TMA store (wait_group + cp.async.bulk.tensor + commit)
      4: trailing bar.sync wait (elided under DROP_TRAIL_BARSYNC)
    WARP_TMA:
      0: empty-slot mbar_wait
      1: cp.async.bulk.tensor A+B chain (lane 0)
    WARP_MMA (CTA 0 only):
      0: full-slot mbar_wait
      1: 4× UTCQMMA + commit_mcast(empty)
      2: commit_mcast(tmem_ready) per tile
      3: tmem_consumed wait (NO_PREFILL only)
*/
#ifdef PROFILE_CYCLES
#define PROF_N_PHASES 8
#define PROF_WALL_SLOT 7
#define PROF_READ(v) asm volatile("mov.u64 %0, %%clock64;" : "=l"(v))
#define PROF_BEGIN(tag) uint64_t _pstart_##tag; PROF_READ(_pstart_##tag)
#define PROF_END(tag, ph) do { \
    uint64_t _pend; PROF_READ(_pend); \
    if (lane == 0) prof[ph] += _pend - _pstart_##tag; \
} while(0)
/*
  PROF_WALL_BEGIN/END: bracket the whole per-warp persistent loop with
  clock64 so host can compute a measured wall cyc/tile per warp — no
  hard-coded 1.813 GHz assumption.  Max across warps = critical path.
*/
#define PROF_WALL_BEGIN() uint64_t _wall_t0 = 0; if (lane == 0) PROF_READ(_wall_t0)
#define PROF_WALL_END() do { \
    if (lane == 0) { uint64_t _wall_t1; PROF_READ(_wall_t1); \
                     prof[PROF_WALL_SLOT] += _wall_t1 - _wall_t0; } \
} while(0)
/*
  Accumulate across all timed launches via atomicAdd (one atomic per
  warp per phase per launch — zero hot-path cost; writeout is at kernel
  exit).  Host divides by N_TIMED_LAUNCHES × tiles_per_cluster to get
  per-tile means with 10× more samples than the old last-launch-only
  readback.
*/
#define PROF_WRITEOUT() do { \
    if (lane == 0 && d_dbg_prof != nullptr) { \
        const int _flat = (cluster_id * CLUSTER_CTAS + (int)cta_rank) * TOTAL_WARPS + warp_id; \
        for (int _ph = 0; _ph < PROF_N_PHASES; _ph++) { \
            atomicAdd((unsigned long long*)&d_dbg_prof[_flat * PROF_N_PHASES + _ph], \
                      (unsigned long long)prof[_ph]); \
        } \
    } \
} while(0)
#else
#define PROF_BEGIN(tag)
#define PROF_END(tag, ph)
#define PROF_WALL_BEGIN()
#define PROF_WALL_END()
#define PROF_WRITEOUT()
#endif

/*
  PROFILE_KI — per-K-iter clock64 bracket around W5 MMA's tma_full_mbar wait.
  Accumulated in SMEM (one uint64 per ki, K_ITERS=24 slots) so lane 0 can
  update without register pressure. At kernel exit, W5 lane 0 dumps prof_ki
  to d_dbg_prof_ki[cluster_id * K_ITERS + ki]. Host aggregates across all
  74 clusters, reports mean cyc per (ki, tile). Each SMEM update costs ~30
  cyc so total overhead is ~720 cyc/tile (~6% of wall) — acceptable for
  relative per-ki comparison (cold-start ki vs steady-state).

  Independent of PROFILE_CYCLES: can be enabled alone or together.
*/
#if defined(PROFILE_KI) || defined(PROFILE_TILE) || defined(PROFILE_W5)
#define PROF_KI_READ(v) asm volatile("mov.u64 %0, %%clock64;" : "=l"(v))
#endif

/*
  PROFILE_W5 — per-tile W5 MMA-issuer critical-path diagnostic.  Brackets
  W5's lane-0 serial work.  Captures:
    tile_total_cyc: clock64 delta from tile-start to tile-end in W5.  This
                    IS the wall-contributing W5 per-tile cost (under NS=6
                    pipelined mainloop w/ PREFILL, all clusters feed the
                    same MMA pipeline, so W5 per-tile cyc ≈ wall / clusters).
    mma_asm_sum:    sum of clock64 deltas around the 4×UTCQMMA asm block +
                    following tcgen05_commit_mcast(tma_empty), summed
                    across K_ITERS=24.  Normally small (~5–30 cyc per iter
                    for async issue + commit).  Ballooning here would
                    indicate MMA engine backpressure (TMEM-ready not yet
                    released from prior tile's epi).
    commit_sum:     cyc in the tile-end tcgen05_commit_mcast(mbar_tmem_
                    ready_base + buf*8) — signals epi that current tile's
                    TMEM half is MMA-complete.  Small unless UCGABAR is
                    contended.
  Combine with PROFILE_TILE to get tma_wait_sum.  The breakdown is:
    tile_total_cyc ≈ tma_wait_sum + mma_asm_sum + commit_sum + residual
  where `residual` (idle between phases) surfaces the in_g-structural
  surplus if any.  Per-in_g mean of (tile_total_cyc − tma_wait_sum) is the
  primary diagnostic: if in_g=0 is high here → issue-side or TMEM-handoff
  backpressure; if it's in tma_wait_sum → W4/TMA cold-B effect.

  Pack (2 × u64 per tile):
    word 0: [63:48] tm  [47:44] tn  [43:38] in_g  [37:0] tile_total_cyc
    word 1: [63:32] mma_asm_sum  [31:0] commit_sum
  Output: d_dbg_prof_w5[2*(cluster_id * tiles_per_cluster + tt) + {0,1}].
  Independent of PROFILE_CYCLES / PROFILE_TILE / PROFILE_KI; designed to be
  enabled solo or combined with PROFILE_TILE for the full tma_wait breakdown.
*/

/*
  PROFILE_TILE — per-tile total of W5 MMA full-slot wait, dumped to global so
  host can histogram by (tm, tn) and by tt (tile-sequence index within cluster).
  Uses a single register accumulator reset per tile + one u64 global write at
  end of tile. Packing:
      [63:48] tm  (16 bits, max TILES_M-1)
      [47:40] tn  (8 bits,  max TILES_N-1)
      [39:32] reserved
      [31:0]  cyc (total full-slot wait across K_ITERS for the tile)
  Independent of PROFILE_CYCLES/PROFILE_KI; shares the clock64 bracket when
  combined, so adding PROFILE_TILE on top of PROFILE_KI costs ~6 cyc/tile
  (one add per ki) + ~5 cyc/tile (swizzle) + ~10 cyc/tile (global store).
*/

__global__ void __launch_bounds__(THREADS, 1)
__cluster_dims__(2, 1, 1)
fc2_w3x_kernel(const __grid_constant__ CUtensorMap tma_a,
               const __grid_constant__ CUtensorMap tma_b,
               const __grid_constant__ CUtensorMap tma_c,
               const __nv_bfloat16* __restrict__ d_bias,
               __nv_bfloat16* __restrict__ d_C,
               uint64_t* __restrict__ d_dbg_prof,
               uint64_t* __restrict__ d_dbg_prof_ki,
               uint64_t* __restrict__ d_dbg_prof_tile,
               uint64_t* __restrict__ d_dbg_prof_w5)
{
    (void)d_C;
    (void)d_dbg_prof;
    (void)d_dbg_prof_ki;
    (void)d_dbg_prof_tile;
    (void)d_dbg_prof_w5;

    extern __shared__ __align__(128) uint8_t smem[];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    uint32_t cta_rank;
    asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    const uint16_t pair_mask = 0x3u;

    /*
      setmaxnreg warpgroup asymmetry — rank-1-shaped.
      W0-W3 (tid 0..127): epi compute warpgroup, high reg budget.
      W4-W5 (tid 128..191): TMA + MMA, minimal regs.
      Requires compile-time -maxrregcount so the default is below SETMAXNREG_HI.
    */
#ifdef USE_SETMAXNREG
#ifndef SETMAXNREG_HI
#define SETMAXNREG_HI 192
#endif
#ifndef SETMAXNREG_LO
#define SETMAXNREG_LO 48
#endif
    if (warp_id < N_EPI_WARPS) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(SETMAXNREG_HI));
    } else {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(SETMAXNREG_LO));
    }
#endif

    if (tid == 0) {
        for (int s = 0; s < N_STAGES; s++) {
            mbar_init(smem_to_uint(smem + MBAR_TMA_FULL + s * 8), 2);
            mbar_init(smem_to_uint(smem + MBAR_TMA_EMPTY + s * 8), 1);
        }
        for (int b = 0; b < 2; b++) {
            mbar_init(smem_to_uint(smem + MBAR_TMEM_READY + b * 8), 1);
#ifdef NO_PREFILL
            mbar_init(smem_to_uint(smem + MBAR_TMEM_CONSUMED + b * 8), 2);
#endif
        }
#ifdef XPF_A
        mbar_init(smem_to_uint(smem + OFF_PF_A_MBAR), 1);
#endif
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp_id == 0) {
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(smem_to_uint(smem + OFF_TMEM)), "n"(TMEM_COLS));
    }

    /* WARP_TMA: bias LDG+STS, all 32 lanes cooperate, pre-cluster-barrier. */
    if (warp_id == WARP_TMA) {
        for (int i = lane; i < N_DIM; i += 32) {
            __nv_bfloat16 v = d_bias[i];
            uint16_t bits = *reinterpret_cast<uint16_t*>(&v);
            asm volatile("st.shared.b16 [%0], %1;"
                :: "r"(smem_to_uint(smem + OFF_BIAS + i * 2)), "h"(bits));
        }
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    /* WARP_MMA on CTA 1 is dead — only CTA 0 issues MMA. */
    if (warp_id == WARP_MMA && cta_rank != 0) return;

    const uint32_t taddr_base = *reinterpret_cast<uint32_t*>(smem + OFF_TMEM);

    uint32_t smem_a_arr[N_STAGES], smem_b_arr[N_STAGES];
    uint32_t tma_full_arr[N_STAGES];
    uint32_t tma_full_peer_arr[N_STAGES];
    uint32_t tma_empty_arr[N_STAGES];
    for (int s = 0; s < N_STAGES; s++) {
        smem_a_arr[s]      = smem_to_uint(smem + s * STAGE_BYTES);
        smem_b_arr[s]      = smem_to_uint(smem + s * STAGE_BYTES + 16384);
        uint32_t tf_local  = smem_to_uint(smem + MBAR_TMA_FULL + s * 8);
        tma_full_arr[s]    = tf_local;
        tma_full_peer_arr[s] = tf_local & 0xFEFFFFFFu;
        tma_empty_arr[s]   = smem_to_uint(smem + MBAR_TMA_EMPTY + s * 8);
    }

    uint32_t out_smem_arr[NUM_EPI_STAGES];
    for (int s = 0; s < NUM_EPI_STAGES; s++) {
        out_smem_arr[s]  = smem_to_uint(smem + OFF_OUT + s * SUBPASS_BYTES);
    }

    const uint32_t mbar_tmem_ready_base   = smem_to_uint(smem + MBAR_TMEM_READY);
    const uint32_t smem_bias              = smem_to_uint(smem + OFF_BIAS);

#ifdef NO_PREFILL
    const uint32_t mbar_tmem_consumed_base = smem_to_uint(smem + MBAR_TMEM_CONSUMED);
    const uint32_t mbar_tmem_cons_peer_base = mbar_tmem_consumed_base & 0xFEFFFFFFu;
#endif

    const int num_clusters = SM_COUNT / CLUSTER_CTAS;
    const int cluster_id   = blockIdx.x / CLUSTER_CTAS;
    const int tiles_per_cluster = (TOTAL_TILES + num_clusters - 1) / num_clusters;

    if (warp_id < N_EPI_WARPS) {
        /* =============== Epilogue warpgroup (W0..W_{N_EPI_WARPS-1}) =============== */
        const int row_group = warp_id;

        uint32_t mma_phase[2] = {0, 0};

#ifdef PROFILE_CYCLES
        uint64_t prof[PROF_N_PHASES] = {0};
#endif
        PROF_WALL_BEGIN();

        for (int tt = 0; tt < tiles_per_cluster; tt++) {
            const int lin_tile = cluster_id + tt * num_clusters;
            if (lin_tile >= TOTAL_TILES) break;
            const int swizzled = tile_swizzle(lin_tile);
            const int tm = swizzled / TILES_N;
            const int tn = swizzled % TILES_N;
            const int prev_m = tm * TM * 2;
            const int prev_n = tn * TN;
            const int buf = tt & 1;

            PROF_BEGIN(e0);
            mbar_wait(mbar_tmem_ready_base + buf * 8, mma_phase[buf]);
            PROF_END(e0, 0);
            mma_phase[buf] ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

#ifdef STRIP_EPILOGUE
            (void)prev_m; (void)prev_n;
            (void)taddr_base; (void)buf; (void)cta_rank; (void)row_group;
#ifdef NO_PREFILL
            if (tid == 0) mbar_arrive(mbar_tmem_cons_peer_base + buf * 8);
#endif
#else
            #pragma unroll 1
            for (int sp = 0; sp < NUM_SUBPASSES; sp++) {
                const int es = sp & (NUM_EPI_STAGES - 1);
                const int nc = sp * SUBPASS_COLS;

                PROF_BEGIN(e1);
#if !USE_STMATRIX
                /*
                  Bias is column-only (nc-dependent, row-invariant), so we
                  LDS once per subpass and reuse across ROW_HALVES.
                  Under USE_STMATRIX the bias is re-loaded per-lane inside
                  the rh body (see below) to avoid bp[] stack-spill.
                */
                uint4 bv0, bv1, bv2, bv3;
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(bv0.x), "=r"(bv0.y), "=r"(bv0.z), "=r"(bv0.w)
                    : "r"(smem_bias + (prev_n + nc +  0) * 2));
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(bv1.x), "=r"(bv1.y), "=r"(bv1.z), "=r"(bv1.w)
                    : "r"(smem_bias + (prev_n + nc +  8) * 2));
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(bv2.x), "=r"(bv2.y), "=r"(bv2.z), "=r"(bv2.w)
                    : "r"(smem_bias + (prev_n + nc + 16) * 2));
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(bv3.x), "=r"(bv3.y), "=r"(bv3.z), "=r"(bv3.w)
                    : "r"(smem_bias + (prev_n + nc + 24) * 2));

                uint32_t bp[16];
                bp[ 0]=bv0.x;  bp[ 1]=bv0.y;  bp[ 2]=bv0.z;  bp[ 3]=bv0.w;
                bp[ 4]=bv1.x;  bp[ 5]=bv1.y;  bp[ 6]=bv1.z;  bp[ 7]=bv1.w;
                bp[ 8]=bv2.x;  bp[ 9]=bv2.y;  bp[10]=bv2.z;  bp[11]=bv2.w;
                bp[12]=bv3.x;  bp[13]=bv3.y;  bp[14]=bv3.z;  bp[15]=bv3.w;
#endif

                #pragma unroll
                for (int rh = 0; rh < ROW_HALVES; rh++) {
                    const int row_local_32 = row_group * ROWS_PER_WARP + rh * 32;
                    const uint32_t taddr_tile = taddr_base + buf * TN
                        + ((cta_rank * TM + row_local_32) << 16);

#if USE_STMATRIX
                    /*
                      Lever C path — LDTM.16dp256bit.x4 + STSM.16.MT88.4
                      shape match for cuBLASLt rank-1.

                      2 × LDTM.16dp256bit.x4 load 32 fp32 per lane covering
                      16+16 rows of the 32×32 subpass-rh block.  4 × STSM
                      .x4.trans.m8n8 write the resulting 8 bf16x2 bands per
                      LDTM (4 bf16x2 per STSM × 32 lanes × 2B = 256B... wait
                      = 512B per STSM; 4 STSM = 2048B = rh block).

                      CORRECTNESS — UNVERIFIED on B200.
                        The per-lane register layout produced by
                        LDTM.16x256b.x4 in this configuration is not
                        independently documented here; we trust the memory
                        note (project_lever_c_bugs_confirmed.md) that this
                        variant is stmatrix-native.  Bias lookup and STSM
                        destination addresses below are computed assuming
                        lane t=8c+r holds, in fp32 order, 8 cols of rows r
                        and r+8 (for LDTM #1; +16 rows for LDTM #2).
                        Actual TMEM routing may differ — TRUST B200 FIRST
                        RUN to confirm, and fix routing if row_t_bias or
                        STSM address math is wrong.
                    */
                    float a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15;
                    TMEM_LOAD_16X256_X4(a0,a1,a2,a3,a4,a5,a6,a7,
                                        a8,a9,a10,a11,a12,a13,a14,a15,
                                        taddr_tile + nc);

                    float b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15;
                    TMEM_LOAD_16X256_X4(b0,b1,b2,b3,b4,b5,b6,b7,
                                        b8,b9,b10,b11,b12,b13,b14,b15,
                                        taddr_tile + nc + (16u << 16));
                    TMEM_WAIT();

                    /*
                      F2FP.PACK_AB pairs adjacent fp32 → 1 bf16x2, matching
                      rank-1's packing shape.  We fold the bias add into the
                      pack via a 3-operand inline sequence (cvt + add).  For
                      a rank-1-shape SASS we could switch to FFMA2/FADD on
                      fp32 first, then F2FP.PACK — functionally identical
                      modulo rounding order, but closer to rank-1's insn
                      mix (Grievance 1, for future if fused-residual).

                      Per-lane bias load.  For LDTM.16x256b.x4, lane t=8c+r
                      is assumed to own 8 contiguous cols
                      (nc + 8c .. nc + 8c + 7).  4 bf16x2 bias values cover
                      those 8 cols; same 4 reused across the two 8-row
                      bands of the rh (rows within a band share col
                      assignment).  Load with per-lane divergent
                      LDS.v4.u32 directly from SMEM bias — avoids the
                      stack spill a bp_lane[] array-index would otherwise
                      trigger on ptxas.  Lane->col mapping is unverified;
                      B200 first run will reveal any permutation.
                    */
                    const int lane_c = lane >> 3;
                    const int lane_r = lane & 7;
                    uint32_t bl0, bl1, bl2, bl3;
                    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(bl0), "=r"(bl1), "=r"(bl2), "=r"(bl3)
                        : "r"(smem_bias + (prev_n + nc + lane_c * 8) * 2));

                    uint32_t p[16];
                    CVT_ADD_BF16X2(p[ 0], a0,  a1,  bl0);
                    CVT_ADD_BF16X2(p[ 1], a2,  a3,  bl1);
                    CVT_ADD_BF16X2(p[ 2], a4,  a5,  bl2);
                    CVT_ADD_BF16X2(p[ 3], a6,  a7,  bl3);
                    CVT_ADD_BF16X2(p[ 4], a8,  a9,  bl0);
                    CVT_ADD_BF16X2(p[ 5], a10, a11, bl1);
                    CVT_ADD_BF16X2(p[ 6], a12, a13, bl2);
                    CVT_ADD_BF16X2(p[ 7], a14, a15, bl3);
                    CVT_ADD_BF16X2(p[ 8], b0,  b1,  bl0);
                    CVT_ADD_BF16X2(p[ 9], b2,  b3,  bl1);
                    CVT_ADD_BF16X2(p[10], b4,  b5,  bl2);
                    CVT_ADD_BF16X2(p[11], b6,  b7,  bl3);
                    CVT_ADD_BF16X2(p[12], b8,  b9,  bl0);
                    CVT_ADD_BF16X2(p[13], b10, b11, bl1);
                    CVT_ADD_BF16X2(p[14], b12, b13, bl2);
                    CVT_ADD_BF16X2(p[15], b14, b15, bl3);

                    /*
                      STSM per-lane address: lane t=8c+r supplies the 16B
                      row-of-matrix-c slot in a 512B stmatrix tile.  To
                      produce a linear row-major SMEM layout with row
                      stride 64B (= 32 bf16 cols) and matrix c occupying
                      cols (8c..8c+7), we pass:
                        addr_lane = base + (row_start + r) * 64 + c * 16.
                      Each STSM covers 8 rows; 4 STSMs cover 32 rows.

                      addr uses row_local_32 + {0,8,16,24} for the 8-row
                      starting offset of each STSM.
                    */
                    const uint32_t out_base = out_smem_arr[es];
                    const uint32_t lane_off = lane_r * (SUBPASS_COLS * 2)
                                            + lane_c * 16;
                    STSM_X4_TRANS(out_base + (row_local_32 +  0) * (SUBPASS_COLS * 2) + lane_off,
                                  p[0], p[1], p[2], p[3]);
                    STSM_X4_TRANS(out_base + (row_local_32 +  8) * (SUBPASS_COLS * 2) + lane_off,
                                  p[4], p[5], p[6], p[7]);
                    STSM_X4_TRANS(out_base + (row_local_32 + 16) * (SUBPASS_COLS * 2) + lane_off,
                                  p[8], p[9], p[10], p[11]);
                    STSM_X4_TRANS(out_base + (row_local_32 + 24) * (SUBPASS_COLS * 2) + lane_off,
                                  p[12], p[13], p[14], p[15]);
#else
                    float a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15;
                    float a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31;
                    TMEM_LOAD_X32(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                                  a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                                  taddr_tile + nc);
                    TMEM_WAIT();

                    uint32_t p[16];
                    CVT_ADD_BF16X2(p[ 0], a0,  a1,  bp[ 0]);
                    CVT_ADD_BF16X2(p[ 1], a2,  a3,  bp[ 1]);
                    CVT_ADD_BF16X2(p[ 2], a4,  a5,  bp[ 2]);
                    CVT_ADD_BF16X2(p[ 3], a6,  a7,  bp[ 3]);
                    CVT_ADD_BF16X2(p[ 4], a8,  a9,  bp[ 4]);
                    CVT_ADD_BF16X2(p[ 5], a10, a11, bp[ 5]);
                    CVT_ADD_BF16X2(p[ 6], a12, a13, bp[ 6]);
                    CVT_ADD_BF16X2(p[ 7], a14, a15, bp[ 7]);
                    CVT_ADD_BF16X2(p[ 8], a16, a17, bp[ 8]);
                    CVT_ADD_BF16X2(p[ 9], a18, a19, bp[ 9]);
                    CVT_ADD_BF16X2(p[10], a20, a21, bp[10]);
                    CVT_ADD_BF16X2(p[11], a22, a23, bp[11]);
                    CVT_ADD_BF16X2(p[12], a24, a25, bp[12]);
                    CVT_ADD_BF16X2(p[13], a26, a27, bp[13]);
                    CVT_ADD_BF16X2(p[14], a28, a29, bp[14]);
                    CVT_ADD_BF16X2(p[15], a30, a31, bp[15]);

                    const uint32_t out_base = out_smem_arr[es] + (row_local_32 + lane) * (SUBPASS_COLS * 2);
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                        :: "r"(out_base +  0), "r"(p[0]), "r"(p[1]), "r"(p[2]), "r"(p[3]));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                        :: "r"(out_base + 16), "r"(p[4]), "r"(p[5]), "r"(p[6]), "r"(p[7]));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                        :: "r"(out_base + 32), "r"(p[8]), "r"(p[9]), "r"(p[10]), "r"(p[11]));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                        :: "r"(out_base + 48), "r"(p[12]), "r"(p[13]), "r"(p[14]), "r"(p[15]));
#endif /* USE_STMATRIX */
                }

                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                PROF_END(e1, 1);

                PROF_BEGIN(e2);
#ifndef DROP_LEAD_BARSYNC
                asm volatile(EPI_BARSYNC_ASM ::: "memory");
#endif
                PROF_END(e2, 2);

                if (tid == 0) {
                    PROF_BEGIN(e3);
                    if (tt > 0 || sp >= NUM_EPI_STAGES) {
                        BULK_ASM(BULK_WAIT_GROUP(1));
                    }
                    tma_store(out_smem_arr[es], &tma_c,
                              prev_n + nc, prev_m + cta_rank * ROWS_PER_CTA);
                    BULK_ASM("cp.async.bulk.commit_group;");
                    PROF_END(e3, 3);
                }

#ifndef DROP_TRAIL_BARSYNC
                PROF_BEGIN(e4);
                asm volatile(EPI_BARSYNC_ASM ::: "memory");
                PROF_END(e4, 4);
#endif
            }

#ifdef NO_PREFILL
            if (tid == 0) {
                mbar_arrive(mbar_tmem_cons_peer_base + buf * 8);
            }
#endif
#endif /* STRIP_EPILOGUE */
        }
        if (tid == 0) {
            BULK_ASM(BULK_WAIT_GROUP(0));
        }
        PROF_WALL_END();
        PROF_WRITEOUT();
    }
    else if (warp_id == WARP_TMA) {
        /* =============== WARP_TMA: TMA A+B loader =============== */
        uint32_t tma_empty_phase[N_STAGES] = {0};
        const bool elect = (lane == 0);

#ifdef XPF_A
        const uint32_t pf_a_data_smem = smem_to_uint(smem + OFF_PF_A_DATA);
        const uint32_t pf_a_mbar      = smem_to_uint(smem + OFF_PF_A_MBAR);
        uint32_t       pf_a_phase     = 0;
        bool           pf_a_pending   = false;
#endif

#ifdef PROFILE_CYCLES
        uint64_t prof[PROF_N_PHASES] = {0};
#endif
        PROF_WALL_BEGIN();

        for (int tt = 0; tt < tiles_per_cluster; tt++) {
            const int lin_tile = cluster_id + tt * num_clusters;
            if (lin_tile >= TOTAL_TILES) break;
            const int swizzled = tile_swizzle(lin_tile);
            const int tm = swizzled / TILES_N;
            const int tn = swizzled % TILES_N;
            const int a_m_tile = tm * 2 + cta_rank;
            const int b_n_half = tn * 2 + cta_rank;

            /*
              Rolled K-loop: force ptxas to keep one physical copy of the TMA
              asm body. Default would partially unroll (K_ITERS=24 compile-time
              constant, observed 2×). Rolled costs ~1 BRA per iter but collapses
              UTMALDG count 48→2 in static SASS, plus the ELECT/BSSY/BSYNC/
              R2UR scaffolds that attach to each unrolled copy. See
              docs/W3X_GRIEVANCES_VS_RANK1.md Grievance 6. i-cache footprint
              win only — runtime inst count is unchanged.
            */
            #pragma unroll 1
            for (int ki = 0; ki < K_ITERS; ki++) {
                const int s = ki % N_STAGES;
                if (ki >= N_STAGES || tt > 0) {
                    PROF_BEGIN(t0);
                    mbar_wait(tma_empty_arr[s], tma_empty_phase[s]);
                    PROF_END(t0, 0);
                    tma_empty_phase[s] ^= 1;
                }

                if (elect) {
                    const uint32_t a_dst = smem_a_arr[s];
                    const uint32_t b_dst = smem_b_arr[s];
                    const int tma_c0    = 0;
                    const int tma_a_c1  = (a_m_tile * K_ITERS + ki) * TM;
                    const int tma_b_c1  = (b_n_half * K_ITERS + ki) * (TN / 2);
                    PROF_BEGIN(t1);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
                        ".mbarrier::complete_tx::bytes.cta_group::2"
                        " [%0], [%1, {%2, %3}], [%4];\n\t"
                        "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
                        ".mbarrier::complete_tx::bytes.cta_group::2"
                        " [%5], [%6, {%2, %7}], [%4];\n\t"
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%4], %8;"
                        :: "r"(a_dst), "l"(&tma_a), "r"(tma_c0), "r"(tma_a_c1),
                           "r"(tma_full_peer_arr[s]),
                           "r"(b_dst), "l"(&tma_b), "r"(tma_b_c1),
                           "r"(TMA_BYTES)
                        : "memory");
                    PROF_END(t1, 1);
                }
            }

#if defined(XPF_A) || defined(XPF_B)
            /*
              Cross-group A-prefetch: at end of tn=2 tiles, issue speculative
              load for the corresponding tm in next group's A. Each cluster's
              tm-in-group position covers a different tm of the next group, so
              all 8 next-group A-tiles get prefetched cooperatively.
            */
            if (tn == TILES_N - 1 && elect) {
                const int g_size_pf  = DG_GROUP_SIZE;
                const int tm_local   = tm - (tm / g_size_pf) * g_size_pf;
                const int first_m_nx = (tm / g_size_pf + 1) * g_size_pf;
                const int tm_pf      = first_m_nx + tm_local;
                if (tm_pf < TILES_M) {
                    const int a_m_pf     = tm_pf * 2 + cta_rank;
                    const int tma_a_c1_pf = a_m_pf * K_ITERS * TM;
#ifdef XPF_A
                    if (pf_a_pending) {
                        mbar_wait(pf_a_mbar, pf_a_phase);
                        pf_a_phase ^= 1;
                    }
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
                        ".mbarrier::complete_tx::bytes"
                        " [%0], [%1, {%2, %3}], [%4];\n\t"
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%4], %5;"
                        :: "r"(pf_a_data_smem), "l"(&tma_a),
                           "r"(0), "r"(tma_a_c1_pf),
                           "r"(pf_a_mbar),
                           "r"(PF_A_BYTES)
                        : "memory");
                    pf_a_pending = true;
#endif
#ifdef XPF_B
                    asm volatile(
                        "cp.async.bulk.prefetch.tensor.2d.L2.global.tile"
                        " [%0, {%1, %2}];"
                        :: "l"(&tma_a), "r"(0), "r"(tma_a_c1_pf)
                        : "memory");
#endif
                }
            }
#endif
        }
#ifdef XPF_A
        if (elect && pf_a_pending) {
            mbar_wait(pf_a_mbar, pf_a_phase);
        }
#endif
        PROF_WALL_END();
        PROF_WRITEOUT();
    }
    else /* warp_id == WARP_MMA, cta_rank == 0 */ {
        /* =============== WARP_MMA: MMA issuer (CTA 0 only) =============== */
        uint32_t tma_full_phase[N_STAGES] = {0};
#ifdef NO_PREFILL
        uint32_t tmem_cons_phase[2] = {0, 0};
#endif

#ifdef PROFILE_CYCLES
        uint64_t prof[PROF_N_PHASES] = {0};
#endif

#ifdef PROFILE_KI
        const uint32_t prof_ki_smem = smem_to_uint(smem + OFF_PROF_KI);
        if (lane == 0) {
            #pragma unroll
            for (int k = 0; k < PROF_KI_SLOTS; k++) {
                asm volatile("st.shared.b64 [%0], %1;"
                    :: "r"(prof_ki_smem + k * 8), "l"(0ULL));
            }
        }
#endif

        uint64_t desc_a_base[N_STAGES], desc_b_base[N_STAGES];
        for (int s = 0; s < N_STAGES; s++) {
            desc_a_base[s] = make_smem_desc(smem_a_arr[s]);
            desc_b_base[s] = make_smem_desc(smem_b_arr[s]);
        }

#ifdef NO_PREFILL
        /* Prime both consumed slots (count=2) so tiles 0, 1 don't block. */
        if (lane == 0) {
            mbar_arrive(mbar_tmem_consumed_base + 0);
            mbar_arrive(mbar_tmem_consumed_base + 0);
            mbar_arrive(mbar_tmem_consumed_base + 8);
            mbar_arrive(mbar_tmem_consumed_base + 8);
        }
#endif
        PROF_WALL_BEGIN();

        for (int tt = 0; tt < tiles_per_cluster; tt++) {
            const int lin_tile = cluster_id + tt * num_clusters;
            if (lin_tile >= TOTAL_TILES) break;
            const int buf = tt & 1;
#if defined(PROFILE_KI_TN) || defined(PROFILE_TILE)
            const int _sw_tn = tile_swizzle(lin_tile) % TILES_N;
#endif

#ifdef PROFILE_W5
            uint64_t _w5_tile_t0 = 0;
            uint64_t _w5_mma_sum = 0;
            uint64_t _w5_commit_sum = 0;
            if (lane == 0) PROF_KI_READ(_w5_tile_t0);
#endif

#ifdef NO_PREFILL
            PROF_BEGIN(m3);
            mbar_wait(mbar_tmem_consumed_base + buf * 8, tmem_cons_phase[buf]);
            PROF_END(m3, 3);
            tmem_cons_phase[buf] ^= 1;
#endif

#ifdef PROFILE_TILE
            uint64_t _tile_wait_sum = 0;
#endif

            /*
              W5 K-loop rolled: mirror the W4 rationale. 192 UTCQMMA → 4 in
              static SASS (one per MMA × the 4× fold inside the asm body, not
              per unrolled ki). Each UTCQMMA still emits its own ELECT +
              BRA.U.ANY loop at runtime — that scaffold lives with the
              instruction semantic, not with asm-block structure.
            */
            #pragma unroll 1
            for (int ki = 0; ki < K_ITERS; ki++) {
                const int s = ki % N_STAGES;
#if defined(PROFILE_KI) || defined(PROFILE_TILE)
                uint64_t _ki_start; PROF_KI_READ(_ki_start);
#endif
                PROF_BEGIN(m0);
                mbar_wait(tma_full_arr[s], tma_full_phase[s]);
                PROF_END(m0, 0);
#if defined(PROFILE_KI) || defined(PROFILE_TILE)
                {
                    uint64_t _ki_end; PROF_KI_READ(_ki_end);
                    uint64_t _delta = _ki_end - _ki_start;
#ifdef PROFILE_KI
                    if (lane == 0) {
#ifdef PROFILE_KI_TN
                        const uint32_t _slot = (ki * TILES_N + _sw_tn) * 8;
#else
                        const uint32_t _slot = ki * 8;
#endif
                        uint64_t _old;
                        asm volatile("ld.shared.b64 %0, [%1];"
                            : "=l"(_old) : "r"(prof_ki_smem + _slot));
                        asm volatile("st.shared.b64 [%0], %1;"
                            :: "r"(prof_ki_smem + _slot), "l"(_old + _delta));
                    }
#endif
#ifdef PROFILE_TILE
                    if (lane == 0) _tile_wait_sum += _delta;
#endif
                }
#endif
                tma_full_phase[s] ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

                if (lane == 0) {
                    uint64_t desc_a = desc_a_base[s];
                    uint64_t desc_b = desc_b_base[s];
                    const int accum_flag = (ki == 0) ? 0 : 1;
#ifdef PROFILE_W5
                    uint64_t _w5_m0; PROF_KI_READ(_w5_m0);
#endif
                    PROF_BEGIN(m1);
                    /*
                      Monolithic MMA + empty-slot commit: 4× tcgen05.mma +
                      tcgen05.commit::mbarrier::arrive (ptx-level multicast)
                      under a single asm volatile. Zero SASS opcode delta vs
                      the old split (ptxas already coalesces adjacent
                      predicated asm volatiles, and ELECT/BSSY/R2UR scaffold
                      attaches per-SASS-instruction not per-asm-block), but
                      reads cleaner and keeps the desc_a/desc_b advances
                      inside one block scope.
                    */
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p_init, p_acc;\n\t"
                        ".reg .b64 da, db;\n\t"
                        ".reg .b32 tc;\n\t"
                        "setp.ne.b32 p_init, %12, 0;\n\t"
                        "setp.ne.b32 p_acc,  1, 0;\n\t"
                        "mov.b32 tc, %0;\n\t"
                        "mov.b64 da, %1;\n\t"
                        "mov.b64 db, %2;\n\t"
                        "tcgen05.mma.cta_group::2.kind::f8f6f4 "
                        "[tc], da, db, %3, {%4,%5,%6,%7,%8,%9,%10,%11}, p_init;\n\t"
                        "add.s64 da, da, 2;\n\t"
                        "add.s64 db, db, 2;\n\t"
                        "tcgen05.mma.cta_group::2.kind::f8f6f4 "
                        "[tc], da, db, %3, {%4,%5,%6,%7,%8,%9,%10,%11}, p_acc;\n\t"
                        "add.s64 da, da, 2;\n\t"
                        "add.s64 db, db, 2;\n\t"
                        "tcgen05.mma.cta_group::2.kind::f8f6f4 "
                        "[tc], da, db, %3, {%4,%5,%6,%7,%8,%9,%10,%11}, p_acc;\n\t"
                        "add.s64 da, da, 2;\n\t"
                        "add.s64 db, db, 2;\n\t"
                        "tcgen05.mma.cta_group::2.kind::f8f6f4 "
                        "[tc], da, db, %3, {%4,%5,%6,%7,%8,%9,%10,%11}, p_acc;\n\t"
                        "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
                        ".shared::cluster.multicast::cluster.b64 [%13], %14;\n\t"
                        "}"
                        :
                        : "r"(buf * TN), "l"(desc_a), "l"(desc_b), "r"(IDESC),
                          "r"(0),"r"(0),"r"(0),"r"(0),
                          "r"(0),"r"(0),"r"(0),"r"(0),
                          "r"(accum_flag),
                          "r"(tma_empty_arr[s]), "h"(pair_mask)
                        : "memory");
                    PROF_END(m1, 1);
#ifdef PROFILE_W5
                    { uint64_t _w5_m1; PROF_KI_READ(_w5_m1); _w5_mma_sum += _w5_m1 - _w5_m0; }
#endif
                }
            }

            if (lane == 0) {
#ifdef PROFILE_W5
                uint64_t _w5_c0; PROF_KI_READ(_w5_c0);
#endif
                PROF_BEGIN(m2);
                tcgen05_commit_mcast(mbar_tmem_ready_base + buf * 8, pair_mask);
                PROF_END(m2, 2);
#ifdef PROFILE_W5
                { uint64_t _w5_c1; PROF_KI_READ(_w5_c1); _w5_commit_sum += _w5_c1 - _w5_c0; }
#endif
            }

#ifdef PROFILE_TILE
            if (lane == 0 && d_dbg_prof_tile != nullptr) {
                const int _sw = tile_swizzle(lin_tile);
                const int _tm = _sw / TILES_N;
                const int _tn = _sw % TILES_N;
                const int _ig = tile_in_group(lin_tile);
                const uint64_t _pack =
                      ((uint64_t)(_tm & 0xFFFF) << 48)
                    | ((uint64_t)(_tn & 0xFF)   << 40)
                    | ((uint64_t)(_ig & 0xFF)   << 32)
                    | ((uint64_t)_tile_wait_sum & 0xFFFFFFFFu);
                d_dbg_prof_tile[cluster_id * tiles_per_cluster + tt] = _pack;
            }
#endif

#ifdef PROFILE_W5
            if (lane == 0 && d_dbg_prof_w5 != nullptr) {
                uint64_t _w5_tile_t1; PROF_KI_READ(_w5_tile_t1);
                uint64_t _tile_total = _w5_tile_t1 - _w5_tile_t0;
                const int _sw = tile_swizzle(lin_tile);
                const int _tm = _sw / TILES_N;
                const int _tn = _sw % TILES_N;
                const int _ig = tile_in_group(lin_tile);
                uint64_t _tt_cap = _tile_total;
                if (_tt_cap > 0x3FFFFFFFFFULL) _tt_cap = 0x3FFFFFFFFFULL;
                uint64_t _mma_cap = _w5_mma_sum; if (_mma_cap > 0xFFFFFFFFULL) _mma_cap = 0xFFFFFFFFULL;
                uint64_t _cmt_cap = _w5_commit_sum; if (_cmt_cap > 0xFFFFFFFFULL) _cmt_cap = 0xFFFFFFFFULL;
                uint64_t _w0 =
                      ((uint64_t)(_tm & 0xFFFF) << 48)
                    | ((uint64_t)(_tn & 0xF)    << 44)
                    | ((uint64_t)(_ig & 0x3F)   << 38)
                    | (_tt_cap & 0x3FFFFFFFFFULL);
                uint64_t _w1 = (_mma_cap << 32) | _cmt_cap;
                const size_t _slot = (size_t)(cluster_id * tiles_per_cluster + tt) * 2;
                d_dbg_prof_w5[_slot + 0] = _w0;
                d_dbg_prof_w5[_slot + 1] = _w1;
            }
#endif
        }
        PROF_WALL_END();
        PROF_WRITEOUT();
#ifdef PROFILE_KI
        if (lane == 0 && d_dbg_prof_ki != nullptr) {
            #pragma unroll
            for (int k = 0; k < PROF_KI_SLOTS; k++) {
                uint64_t _v;
                asm volatile("ld.shared.b64 %0, [%1];"
                    : "=l"(_v) : "r"(prof_ki_smem + k * 8));
                d_dbg_prof_ki[cluster_id * PROF_KI_SLOTS + k] = _v;
            }
        }
#endif
    }

    if (warp_id == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
            :: "r"(0), "n"(TMEM_COLS));
    }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    printf("FC2 W3X kernel — 6-warp bias-only rank-1-shaped persistent\n");
    printf("  [%d,%d] x [%d,%d]^T  NS=%d  THREADS=%d  SMEM=%d B  (cap 228KB)\n",
           M_TOTAL, K_DIM, N_DIM, K_DIM, N_STAGES, THREADS, SMEM_BYTES);
    if (SMEM_BYTES > 232448) {
        fprintf(stderr, "  ERROR: SMEM exceeds B200 cap\n"); return 1;
    }
    if (TOTAL_TILES % (SM_COUNT / CLUSTER_CTAS) != 0) {
        fprintf(stderr, "  ERROR: TOTAL_TILES=%d %% num_clusters=%d != 0 — kernel assumes exact division\n",
                TOTAL_TILES, SM_COUNT / CLUSTER_CTAS);
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));

    __nv_fp8_e4m3 *d_A=nullptr, *d_B=nullptr;
    __nv_bfloat16 *d_bias=nullptr, *d_C=nullptr;

    size_t sA = (size_t)M_TOTAL * K_DIM;
    size_t sB = (size_t)N_DIM   * K_DIM;
    size_t sC = (size_t)M_TOTAL * N_DIM;
    CUDA_CHECK(cudaMalloc(&d_A, sA));
    CUDA_CHECK(cudaMalloc(&d_B, sB));
    CUDA_CHECK(cudaMalloc(&d_bias, (size_t)N_DIM * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_C, sC * sizeof(__nv_bfloat16)));

    __nv_fp8_e4m3 *hA = nullptr;
    __nv_fp8_e4m3 *hB = nullptr;
    __nv_bfloat16 *hbias = nullptr;

#ifdef NCU_PROFILE
    /*
      NCU fast path: skip host fills + H2D copies. cudaMemset to a non-NaN
      FP8/BF16 byte pattern gives the kernel valid-looking inputs with
      identical access pattern. Hardware counters are unchanged; only
      numerical output differs (verification block is skipped too).
    */
    CUDA_CHECK(cudaMemset(d_A, 0x3c, sA));
    CUDA_CHECK(cudaMemset(d_B, 0x3c, sB));
    CUDA_CHECK(cudaMemset(d_bias, 0, (size_t)N_DIM * sizeof(__nv_bfloat16)));
#else
    hA = (__nv_fp8_e4m3*)malloc(sA);
    hB = (__nv_fp8_e4m3*)malloc(sB);
    hbias = (__nv_bfloat16*)malloc((size_t)N_DIM * sizeof(__nv_bfloat16));

    /*
      PACKED_TILES layout (fc2_w3 convention, TM per-CTA = 128):
      A packed with tile_m=TM=128 half-slabs. Each (a_m_tile, k_block) pair
      is TM rows × TK cols contiguous. a_m_tile ∈ [0, M_TOTAL/TM).
    */
    for (size_t a_m_tile = 0; a_m_tile < (size_t)(M_TOTAL / TM); a_m_tile++) {
        for (size_t k_block = 0; k_block < (size_t)K_ITERS; k_block++) {
            for (size_t r = 0; r < (size_t)TM; r++) {
                long long global_row = (long long)a_m_tile * TM + r;
                if (global_row >= M_TOTAL) continue;
                for (size_t k = 0; k < (size_t)TK; k++) {
                    float av = 1.0f + 0.125f * (float)((int)global_row & 7);
                    size_t off = ((a_m_tile * K_ITERS + k_block) * TM + r) * TK + k;
                    hA[off] = static_cast<__nv_fp8_e4m3>(av);
                }
            }
        }
    }
    for (size_t b_n_half = 0; b_n_half < (size_t)(N_DIM / (TN/2)); b_n_half++) {
        for (size_t k_block = 0; k_block < (size_t)K_ITERS; k_block++) {
            for (size_t c = 0; c < (size_t)(TN/2); c++) {
                long long global_col = (long long)b_n_half * (TN/2) + c;
                if (global_col >= N_DIM) continue;
                for (size_t k = 0; k < (size_t)TK; k++) {
                    float bv = (global_col & 1) ? 1.0f : 1.5f;
                    size_t off = ((b_n_half * K_ITERS + k_block) * (TN/2) + c) * TK + k;
                    hB[off] = static_cast<__nv_fp8_e4m3>(bv);
                }
            }
        }
    }
    for (int n = 0; n < N_DIM; n++) hbias[n] = __float2bfloat16(0.25f * (float)(n & 7));

    CUDA_CHECK(cudaMemcpy(d_A, hA, sA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, hB, sB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, hbias, (size_t)N_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
#endif
    CUDA_CHECK(cudaMemset(d_C, 0, sC * sizeof(__nv_bfloat16)));
    printf("  Alloc + init + pack done\n");

    CUtensorMap h_tma_a, h_tma_b, h_tma_c;
    {
        uint64_t a_total_rows = (uint64_t)(M_TOTAL / TM) * (uint64_t)K_ITERS * (uint64_t)TM;
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
        uint64_t b_total_rows = (uint64_t)(N_DIM / (TN/2)) * (uint64_t)K_ITERS * (uint64_t)(TN/2);
        uint64_t dims[2]    = {(uint64_t)TK, b_total_rows};
        uint64_t strides[1] = {(uint64_t)TK};
        uint32_t box[2]     = {TK, TN / 2};
        uint32_t estrides[2]= {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(&h_tma_b,
            CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)d_B,
            dims, strides, box, estrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }
    {
        uint64_t dims[2]    = {(uint64_t)N_DIM, (uint64_t)M_TOTAL};
        uint64_t strides[1] = {(uint64_t)N_DIM * sizeof(__nv_bfloat16)};
        uint32_t box[2]     = {SUBPASS_COLS, ROWS_PER_CTA};
        uint32_t estrides[2]= {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(&h_tma_c,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)d_C,
            dims, strides, box, estrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }

    CUDA_CHECK(cudaFuncSetAttribute(fc2_w3x_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    printf("  TMA descriptors + func attr done (SMEM=%d B)\n", SMEM_BYTES);

    uint64_t* d_dbg_prof = nullptr;
    uint64_t* d_dbg_prof_ki = nullptr;
    uint64_t* d_dbg_prof_tile = nullptr;
    uint64_t* d_dbg_prof_w5 = nullptr;
#if defined(PROFILE_CYCLES) || defined(PROFILE_KI) || defined(PROFILE_TILE) || defined(PROFILE_W5)
    const int num_clusters_host = SM_COUNT / CLUSTER_CTAS;
    const int tiles_per_cluster_host = (TOTAL_TILES + num_clusters_host - 1) / num_clusters_host;
#endif
#ifdef PROFILE_CYCLES
    const size_t prof_slots = (size_t)num_clusters_host * CLUSTER_CTAS * TOTAL_WARPS * PROF_N_PHASES;
    const size_t prof_bytes = prof_slots * sizeof(uint64_t);
    CUDA_CHECK(cudaMalloc(&d_dbg_prof, prof_bytes));
    CUDA_CHECK(cudaMemset(d_dbg_prof, 0, prof_bytes));
#endif
#ifdef PROFILE_KI
    const size_t prof_ki_bytes = (size_t)num_clusters_host * PROF_KI_SLOTS * sizeof(uint64_t);
    CUDA_CHECK(cudaMalloc(&d_dbg_prof_ki, prof_ki_bytes));
    CUDA_CHECK(cudaMemset(d_dbg_prof_ki, 0, prof_ki_bytes));
#endif
#ifdef PROFILE_TILE
    const size_t prof_tile_bytes = (size_t)num_clusters_host * tiles_per_cluster_host * sizeof(uint64_t);
    CUDA_CHECK(cudaMalloc(&d_dbg_prof_tile, prof_tile_bytes));
    CUDA_CHECK(cudaMemset(d_dbg_prof_tile, 0, prof_tile_bytes));
#endif
#ifdef PROFILE_W5
    const size_t prof_w5_bytes = (size_t)num_clusters_host * tiles_per_cluster_host * 2 * sizeof(uint64_t);
    CUDA_CHECK(cudaMalloc(&d_dbg_prof_w5, prof_w5_bytes));
    CUDA_CHECK(cudaMemset(d_dbg_prof_w5, 0, prof_w5_bytes));
#endif

    dim3 grid(SM_COUNT, 1, 1);
#define LAUNCH_KERNEL() \
    fc2_w3x_kernel<<<grid, THREADS, SMEM_BYTES>>>( \
        h_tma_a, h_tma_b, h_tma_c, d_bias, d_C, d_dbg_prof, d_dbg_prof_ki, d_dbg_prof_tile, d_dbg_prof_w5)

#ifdef NCU_PROFILE
    const int N_WARMUP = 1;
#else
    const int N_WARMUP = 2;
#endif
    printf("Warmup (%d iters)...\n", N_WARMUP);
    for (int i = 0; i < N_WARMUP; i++) LAUNCH_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

#ifdef PROFILE_CYCLES
    CUDA_CHECK(cudaMemset(d_dbg_prof, 0, prof_bytes));
#endif
#ifdef PROFILE_KI
    CUDA_CHECK(cudaMemset(d_dbg_prof_ki, 0, prof_ki_bytes));
#endif
#ifdef PROFILE_TILE
    CUDA_CHECK(cudaMemset(d_dbg_prof_tile, 0, prof_tile_bytes));
#endif
#ifdef PROFILE_W5
    CUDA_CHECK(cudaMemset(d_dbg_prof_w5, 0, prof_w5_bytes));
#endif
#ifdef NCU_PROFILE
    const int N_TIMED_LAUNCHES = 1;
#else
    const int N_TIMED_LAUNCHES = 10;
#endif
    printf("Timing %d iters...\n", N_TIMED_LAUNCHES);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < N_TIMED_LAUNCHES; i++) LAUNCH_KERNEL();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= (float)N_TIMED_LAUNCHES;
    printf("FC2-W3X kernel: %.3f ms  %.2f TFLOPS\n",
           ms, 2.0 * M_TOTAL * N_DIM * K_DIM / ms / 1e9);

#if defined(PROFILE_CYCLES) || defined(PROFILE_KI) || defined(PROFILE_TILE) || defined(PROFILE_W5)
    const double tile_cyc = (double)ms * 1e-3 * 1.813e9 / tiles_per_cluster_host;
    (void)tile_cyc; /* Kept for PROFILE_KI/_TILE/_W5 host-side scaling. */
#endif

#ifdef PROFILE_CYCLES
    {
        uint64_t* h_prof = (uint64_t*)malloc(prof_bytes);
        CUDA_CHECK(cudaMemcpy(h_prof, d_dbg_prof, prof_bytes, cudaMemcpyDeviceToHost));

        /*
          Aggregate per warp_id across all (cluster, cta) instances where the
          warp actually ran (skip all-zero rows — W_MMA on CTA 1 early-exits).
          PROF_WRITEOUT uses atomicAdd, so readback sums all N_TIMED_LAUNCHES.
          Divide by warp_count × tiles_per_cluster × N_TIMED_LAUNCHES for per-
          tile means with 10× more samples than last-launch-only readback.
        */
        uint64_t warp_sum[TOTAL_WARPS][PROF_N_PHASES] = {{0}};
        int warp_count[TOTAL_WARPS] = {0};
        for (int c = 0; c < num_clusters_host; c++) {
            for (int r = 0; r < CLUSTER_CTAS; r++) {
                for (int w = 0; w < TOTAL_WARPS; w++) {
                    uint64_t row_sum = 0;
                    for (int p = 0; p < PROF_N_PHASES; p++) {
                        row_sum += h_prof[((c * CLUSTER_CTAS + r) * TOTAL_WARPS + w) * PROF_N_PHASES + p];
                    }
                    if (row_sum == 0) continue;
                    warp_count[w]++;
                    for (int p = 0; p < PROF_N_PHASES; p++) {
                        warp_sum[w][p] += h_prof[((c * CLUSTER_CTAS + r) * TOTAL_WARPS + w) * PROF_N_PHASES + p];
                    }
                }
            }
        }

        /*
          Wall cyc/tile: take max across warps of the PROF_WALL_SLOT accumulator
          (loop-enter → loop-exit clock64 bracket).  That's the true critical-
          path length in cycles, independent of clock frequency.  Effective
          clock = wall_cyc / (ms / 1000 / tiles_per_cluster) tells us whether
          the GPU ran at base (1.813 GHz) or boost (1.965 GHz).
        */
        double wall_cyc = 0;
        int wall_src_warp = -1;
        for (int w = 0; w < TOTAL_WARPS; w++) {
            if (warp_count[w] == 0) continue;
            double v = (double)warp_sum[w][PROF_WALL_SLOT]
                     / ((double)warp_count[w] * tiles_per_cluster_host * N_TIMED_LAUNCHES);
            if (v > wall_cyc) { wall_cyc = v; wall_src_warp = w; }
        }
        const double eff_clock_ghz = (ms > 0 && tiles_per_cluster_host > 0)
            ? (wall_cyc * tiles_per_cluster_host * 1e-3 / ms) : 0.0;

        printf("\n[PROFILE] mean cyc/tile per (warp, phase), across %d clusters × %d timed launches\n",
               num_clusters_host, N_TIMED_LAUNCHES);
        printf("  tiles/cluster=%d  wall cyc/tile (measured, W%d) = %.0f  effective clock = %.3f GHz\n",
               tiles_per_cluster_host, wall_src_warp, wall_cyc, eff_clock_ghz);
        printf("  (hard-coded 1.813 GHz would give wall cyc/tile = %.0f — delta %.1f%% indicates actual clock)\n",
               (double)ms * 1e-3 * 1.813e9 / tiles_per_cluster_host,
               wall_cyc > 0 ? 100.0 * (wall_cyc / ((double)ms * 1e-3 * 1.813e9 / tiles_per_cluster_host) - 1.0) : 0.0);

        const char* epi_labels[5] = {
            "tmem_ready wait",
            "subpass body  ",
            "bar.sync 1    ",
            "TMA store (t0)",
            "bar.sync 2    "
        };
        const char* tma_labels[2] = {
            "empty-slot wait",
            "cp.async.tensor"
        };
        const char* mma_labels[4] = {
            "full-slot wait ",
            "4x UTCQMMA     ",
            "tmem_ready cmt ",
            "tmem_cons wait "
        };

        auto dump_warp = [&](int w, const char* tag, const char** labels, int nph) {
            if (warp_count[w] == 0) { printf("  [W%d %s] no data\n", w, tag); return; }
            uint64_t total = 0;
            for (int p = 0; p < nph; p++) total += warp_sum[w][p];
            double denom = (double)warp_count[w] * tiles_per_cluster_host * N_TIMED_LAUNCHES;
            double total_per_tile = (double)total / denom;
            double wall_w = (double)warp_sum[w][PROF_WALL_SLOT] / denom;
            printf("  [W%d %-4s  instances=%3d]   cyc/tile  wall%%\n",
                   w, tag, warp_count[w]);
            for (int p = 0; p < nph; p++) {
                double v = (double)warp_sum[w][p] / denom;
                double pct = wall_cyc > 0 ? 100.0 * v / wall_cyc : 0.0;
                printf("    P%d %s  %8.0f  %5.1f%%\n", p, labels[p], v, pct);
                printf("@@PROF warp=%d tag=%s phase=%d label=%s cyc_per_tile=%.2f wall_pct=%.3f\n",
                       w, tag, p, labels[p], v, pct);
            }
            double tot_pct = wall_cyc > 0 ? 100.0 * total_per_tile / wall_cyc : 0.0;
            printf("    %-19s  %8.0f  %5.1f%%\n", "SUM (instrumented)", total_per_tile, tot_pct);
            printf("@@PROF warp=%d tag=%s phase=SUM label=instrumented cyc_per_tile=%.2f wall_pct=%.3f\n",
                   w, tag, total_per_tile, tot_pct);
            printf("@@PROF warp=%d tag=%s phase=WALL label=wall_bracket cyc_per_tile=%.2f wall_pct=100.000\n",
                   w, tag, wall_w);
        };

        for (int w = 0; w < N_EPI_WARPS; w++) dump_warp(w, "epi",  epi_labels, 5);
        dump_warp(WARP_TMA, "TMA", tma_labels, 2);
        dump_warp(WARP_MMA, "MMA", mma_labels, 4);

        printf("@@PROFMETA wall_cyc_per_tile=%.2f eff_clock_ghz=%.4f tiles_per_cluster=%d launches=%d\n",
               wall_cyc, eff_clock_ghz, tiles_per_cluster_host, N_TIMED_LAUNCHES);

        free(h_prof);
    }
#endif

#ifdef PROFILE_KI
    {
        uint64_t* h_prof_ki = (uint64_t*)malloc(prof_ki_bytes);
        CUDA_CHECK(cudaMemcpy(h_prof_ki, d_dbg_prof_ki, prof_ki_bytes, cudaMemcpyDeviceToHost));

        /*
          W5 MMA per-ki full-slot wait. With PROFILE_KI_TN, slots are
          per-(ki, tn) with TILES_N tn buckets (= PROF_KI_SLOTS = K_ITERS*TILES_N);
          without it, slots are per-ki only. Aggregate across 74 clusters; each
          cluster's slot = sum over tiles_per_cluster tiles of clock64 delta.
          Divide by per-tile-count (total or per-tn) for mean cyc per ki per tile.
          Last-launch-only semantics (same as PROFILE_CYCLES).

          NOTE on per-tn denominators: each cluster covers
          tiles_per_cluster tiles, and under dgswizzle each tn is visited
          tiles_per_cluster/TILES_N times per cluster (exact for FC2 shape:
          147 tiles, 49 per tn). So the per-(ki,tn) mean uses
          (num_clusters * tiles_per_cluster / TILES_N).
        */
        const int tn_host = (N_DIM + TN - 1) / TN;
        printf("\n[PROFILE_KI] W5 MMA per-ki full-slot wait  (clusters=%d, tiles/cluster=%d)\n",
               num_clusters_host, tiles_per_cluster_host);

#ifdef PROFILE_KI_TN
        printf("  ki        tn=0          tn=1          tn=2         all_tn  wall%%\n");
        printf("        cyc/vis      cyc/vis      cyc/vis      cyc/tile\n");
        const double denom_tn  = (double)num_clusters_host * tiles_per_cluster_host / tn_host;
        const double denom_all = (double)num_clusters_host * tiles_per_cluster_host;
        uint64_t grand_total = 0;
        double cum_all = 0.0;
        for (int k = 0; k < K_ITERS; k++) {
            uint64_t s_tn[4] = {0,0,0,0};
            for (int c = 0; c < num_clusters_host; c++) {
                for (int n = 0; n < tn_host && n < 4; n++) {
                    s_tn[n] += h_prof_ki[c * PROF_KI_SLOTS + k * tn_host + n];
                }
            }
            uint64_t s_all = 0;
            for (int n = 0; n < tn_host && n < 4; n++) s_all += s_tn[n];
            grand_total += s_all;
            double per_ki_all = (double)s_all / denom_all;
            cum_all += per_ki_all;
            double pct = tile_cyc > 0 ? 100.0 * per_ki_all / tile_cyc : 0.0;
            printf("  %2d", k);
            for (int n = 0; n < 3; n++) {
                if (n < tn_host) {
                    printf("   %7.0f  ", (double)s_tn[n] / denom_tn);
                } else {
                    printf("   %7s  ", "---");
                }
            }
            printf("    %7.0f  %4.1f%%\n", per_ki_all, pct);
        }
        double tot_per_tile = (double)grand_total / denom_all;
        double tot_pct = tile_cyc > 0 ? 100.0 * tot_per_tile / tile_cyc : 0.0;
        printf("  --- sum across K_ITERS=%d (all_tn) ------\n", K_ITERS);
        printf("                                            %7.0f  %4.1f%%  (should ≈ W%d MMA P0)\n",
               tot_per_tile, tot_pct, WARP_MMA);

        printf("\n  per-tn totals (across all ki):\n");
        uint64_t s_tn_tot[4] = {0,0,0,0};
        for (int k = 0; k < K_ITERS; k++) {
            for (int c = 0; c < num_clusters_host; c++) {
                for (int n = 0; n < tn_host && n < 4; n++) {
                    s_tn_tot[n] += h_prof_ki[c * PROF_KI_SLOTS + k * tn_host + n];
                }
            }
        }
        for (int n = 0; n < tn_host && n < 3; n++) {
            double per_vis = (double)s_tn_tot[n] / denom_tn;
            printf("    tn=%d: %7.0f cyc/visit  (sum across %d ki = cyc spent waiting on tn=%d B per tile)\n",
                   n, per_vis, K_ITERS, n);
        }
#else
        printf("  ki  cyc/ki   wall%%   cum-cyc/tile\n");
        double denom = (double)num_clusters_host * tiles_per_cluster_host;
        double cum = 0.0;
        uint64_t grand_total = 0;
        for (int k = 0; k < K_ITERS; k++) {
            uint64_t s = 0;
            for (int c = 0; c < num_clusters_host; c++) s += h_prof_ki[c * K_ITERS + k];
            grand_total += s;
            double per_ki = (double)s / denom;
            cum += per_ki;
            double pct = tile_cyc > 0 ? 100.0 * per_ki / tile_cyc : 0.0;
            printf("  %2d   %6.0f   %4.1f%%   %7.0f\n", k, per_ki, pct, cum);
        }
        double tot_per_tile = (double)grand_total / denom;
        double tot_pct = tile_cyc > 0 ? 100.0 * tot_per_tile / tile_cyc : 0.0;
        printf("  --- sum across K_ITERS=%d ------\n", K_ITERS);
        printf("       %6.0f   %4.1f%%  (should ≈ PROFILE W%d MMA P0 cyc/tile)\n",
               tot_per_tile, tot_pct, WARP_MMA);
        (void)tn_host;
#endif
        free(h_prof_ki);
    }
#endif

#ifdef PROFILE_TILE
    {
        uint64_t* h_prof_tile = (uint64_t*)malloc(prof_tile_bytes);
        CUDA_CHECK(cudaMemcpy(h_prof_tile, d_dbg_prof_tile, prof_tile_bytes, cudaMemcpyDeviceToHost));

        /*
          Per-tile W5 MMA full-slot wait, last-launch-only (same as PROFILE_KI).
          Packed u64 = [tm:16][tn:8][in_g:8][cyc:32]. Aggregate across all
          (cluster, tt) entries, bucketed four ways:
            (1) by tm_bin × tn                 — spatial histogram
            (2) by tt   (sequence index)       — L2-warmup trajectory
            (3) by tn   (aggregated over tm)   — N-column effect
            (4) by in_g (0..23, dgswizzle position-in-group)
                                               — group-boundary effect
        */
        const int TOTAL_ENTRIES = num_clusters_host * tiles_per_cluster_host;
        const int TM_BINS = 16;
        const int tiles_m_host = M_TOTAL / TM / 2;
        const int tiles_n_host = N_DIM / TN;
        const int bin_width = (tiles_m_host + TM_BINS - 1) / TM_BINS;
        const int IG_SLOTS = tiles_n_host * DG_GROUP_SIZE;

        uint64_t bin_cyc[TM_BINS][3]   = {{0}};
        uint32_t bin_cnt[TM_BINS][3]   = {{0}};
        uint64_t bin_max[TM_BINS][3]   = {{0}};
        uint64_t tt_cyc[256]           = {0};
        uint32_t tt_cnt[256]           = {0};
        uint64_t tn_cyc[8]             = {0};
        uint32_t tn_cnt[8]             = {0};
        uint64_t ig_cyc[32]            = {0};
        uint32_t ig_cnt[32]            = {0};
        uint64_t ig_max[32]            = {0};
        uint64_t grand_cyc_sum = 0;
        uint32_t grand_cnt     = 0;
        uint32_t cyc_min = 0xFFFFFFFFu, cyc_max = 0;

        for (int c = 0; c < num_clusters_host; c++) {
            for (int t = 0; t < tiles_per_cluster_host; t++) {
                uint64_t p = h_prof_tile[c * tiles_per_cluster_host + t];
                if (p == 0) continue;
                int      tm  = (int)((p >> 48) & 0xFFFFu);
                int      tn  = (int)((p >> 40) & 0xFFu);
                int      ig  = (int)((p >> 32) & 0xFFu);
                uint32_t cyc = (uint32_t)(p & 0xFFFFFFFFu);

                int mb = tm / bin_width; if (mb >= TM_BINS) mb = TM_BINS - 1;
                if (tn < 3) {
                    bin_cyc[mb][tn] += cyc;
                    bin_cnt[mb][tn] += 1;
                    if (cyc > bin_max[mb][tn]) bin_max[mb][tn] = cyc;
                }
                if (tn < 8) {
                    tn_cyc[tn] += cyc;
                    tn_cnt[tn] += 1;
                }
                if (t < 256) {
                    tt_cyc[t] += cyc;
                    tt_cnt[t] += 1;
                }
                if (ig < 32) {
                    ig_cyc[ig] += cyc;
                    ig_cnt[ig] += 1;
                    if (cyc > ig_max[ig]) ig_max[ig] = cyc;
                }
                grand_cyc_sum += cyc;
                grand_cnt     += 1;
                if (cyc < cyc_min) cyc_min = cyc;
                if (cyc > cyc_max) cyc_max = cyc;
            }
        }

        (void)TOTAL_ENTRIES;
        const double tile_cyc_local = tile_cyc;
        double mean_per_tile = grand_cnt ? (double)grand_cyc_sum / grand_cnt : 0.0;
        printf("\n[PROFILE_TILE] W5 MMA per-tile full-slot wait  (entries=%u, bin_width=%d tm)\n",
               grand_cnt, bin_width);
        printf("  overall: mean=%.0f cyc (%.1f%% wall)  min=%u  max=%u\n",
               mean_per_tile, tile_cyc_local > 0 ? 100.0 * mean_per_tile / tile_cyc_local : 0.0,
               cyc_min == 0xFFFFFFFFu ? 0 : cyc_min, cyc_max);

        printf("\n  by M-row bin × N-col tile  (mean cyc/tile; max in parens)\n");
        printf("  tm_bin   tm_range      tn=0           tn=1           tn=2           row_mean\n");
        for (int b = 0; b < TM_BINS; b++) {
            int tm_lo = b * bin_width;
            int tm_hi = (b + 1) * bin_width - 1; if (tm_hi >= tiles_m_host) tm_hi = tiles_m_host - 1;
            uint64_t row_sum = 0; uint32_t row_cnt = 0;
            for (int n = 0; n < 3; n++) { row_sum += bin_cyc[b][n]; row_cnt += bin_cnt[b][n]; }
            if (row_cnt == 0) continue;
            printf("  %4d   [%4d..%4d]", b, tm_lo, tm_hi);
            for (int n = 0; n < 3; n++) {
                if (bin_cnt[b][n] == 0) { printf("   %4s           ", "---"); continue; }
                double m = (double)bin_cyc[b][n] / bin_cnt[b][n];
                printf("   %5.0f (%5lu)", m, (unsigned long)bin_max[b][n]);
            }
            double rm = (double)row_sum / row_cnt;
            printf("   %5.0f\n", rm);
        }

        printf("\n  by N-col (aggregated over all tm)\n");
        printf("  tn    cyc/tile   wall%%   count\n");
        for (int n = 0; n < tiles_n_host && n < 8; n++) {
            if (tn_cnt[n] == 0) continue;
            double m = (double)tn_cyc[n] / tn_cnt[n];
            double pct = tile_cyc_local > 0 ? 100.0 * m / tile_cyc_local : 0.0;
            printf("  %2d    %7.0f   %4.1f%%   %u\n", n, m, pct, tn_cnt[n]);
        }

        printf("\n  by tt (tile-sequence index within cluster, averaged over clusters)\n");
        printf("   tt   cyc/tile  (first 32 entries)\n");
        int n_show = tiles_per_cluster_host < 32 ? tiles_per_cluster_host : 32;
        for (int t = 0; t < n_show; t++) {
            if (tt_cnt[t] == 0) continue;
            double m = (double)tt_cyc[t] / tt_cnt[t];
            printf("   %3d   %7.0f\n", t, m);
        }

        printf("\n  by in_g (position within dgswizzle group; 0=group boundary)\n");
        printf("  in_g  raw_tn  cyc/tile    max    count\n");
        for (int ig = 0; ig < IG_SLOTS && ig < 32; ig++) {
            if (ig_cnt[ig] == 0) continue;
            double m = (double)ig_cyc[ig] / ig_cnt[ig];
            int raw_tn = ig / DG_GROUP_SIZE;
            printf("   %3d    %3d   %7.0f  %5lu    %u\n",
                   ig, raw_tn, m, (unsigned long)ig_max[ig], ig_cnt[ig]);
        }
        free(h_prof_tile);
    }
#endif

#ifdef PROFILE_W5
    {
        uint64_t* h_prof_w5 = (uint64_t*)malloc(prof_w5_bytes);
        CUDA_CHECK(cudaMemcpy(h_prof_w5, d_dbg_prof_w5, prof_w5_bytes, cudaMemcpyDeviceToHost));

        const int tiles_n_host = N_DIM / TN;
        const int IG_SLOTS = tiles_n_host * DG_GROUP_SIZE;

        uint64_t tn_t[8] = {0}, tn_m[8] = {0}, tn_c[8] = {0};
        uint32_t tn_cnt[8] = {0};
        uint64_t ig_t[32] = {0}, ig_m[32] = {0}, ig_c[32] = {0};
        uint32_t ig_cnt[32] = {0};
        uint64_t grand_t = 0, grand_m = 0, grand_c = 0;
        uint32_t grand_cnt = 0;
        uint64_t t_min = ~0ULL, t_max = 0;

        for (int c = 0; c < num_clusters_host; c++) {
            for (int t = 0; t < tiles_per_cluster_host; t++) {
                const size_t slot = (size_t)(c * tiles_per_cluster_host + t) * 2;
                uint64_t w0 = h_prof_w5[slot + 0];
                uint64_t w1 = h_prof_w5[slot + 1];
                if (w0 == 0 && w1 == 0) continue;
                int      tn     = (int)((w0 >> 44) & 0xFu);
                int      ig     = (int)((w0 >> 38) & 0x3Fu);
                uint64_t tot    = w0 & 0x3FFFFFFFFFULL;
                uint32_t mma_s  = (uint32_t)(w1 >> 32);
                uint32_t cmt_s  = (uint32_t)(w1 & 0xFFFFFFFFu);
                if (tn < 8) {
                    tn_t[tn]   += tot;
                    tn_m[tn]   += mma_s;
                    tn_c[tn]   += cmt_s;
                    tn_cnt[tn]++;
                }
                if (ig < 32) {
                    ig_t[ig]   += tot;
                    ig_m[ig]   += mma_s;
                    ig_c[ig]   += cmt_s;
                    ig_cnt[ig]++;
                }
                grand_t   += tot;
                grand_m   += mma_s;
                grand_c   += cmt_s;
                grand_cnt++;
                if (tot < t_min) t_min = tot;
                if (tot > t_max) t_max = tot;
            }
        }

        double mean_t = grand_cnt ? (double)grand_t / grand_cnt : 0.0;
        double mean_m = grand_cnt ? (double)grand_m / grand_cnt : 0.0;
        double mean_c = grand_cnt ? (double)grand_c / grand_cnt : 0.0;

        printf("\n[PROFILE_W5] W5 MMA-issuer per-tile diagnostic  (entries=%u)\n", grand_cnt);
        printf("  wall cyc/tile @ 1.813 GHz = %.0f  (W5 total cyc/tile ≈ wall when MMA-bound)\n", tile_cyc);
        printf("  overall: tile_total mean=%.0f  min=%llu  max=%llu  (%.1f%% wall)\n",
               mean_t, (unsigned long long)(t_min == ~0ULL ? 0 : t_min), (unsigned long long)t_max,
               tile_cyc > 0 ? 100.0 * mean_t / tile_cyc : 0.0);
        printf("           mma_asm    mean=%.0f  (per-iter≈%.0f, %.1f%% wall)\n",
               mean_m, mean_m / K_ITERS, tile_cyc > 0 ? 100.0 * mean_m / tile_cyc : 0.0);
        printf("           commit     mean=%.0f  (%.1f%% wall)\n",
               mean_c, tile_cyc > 0 ? 100.0 * mean_c / tile_cyc : 0.0);
        printf("  residual (= total − mma − commit): %.0f  (%.1f%% wall) ← combine with PROFILE_TILE tma_wait for finer split\n",
               mean_t - mean_m - mean_c,
               tile_cyc > 0 ? 100.0 * (mean_t - mean_m - mean_c) / tile_cyc : 0.0);

        printf("\n  by N-col (aggregated over all tm)\n");
        printf("  tn   tile_total   mma_asm   commit   residual   count\n");
        for (int n = 0; n < tiles_n_host && n < 8; n++) {
            if (tn_cnt[n] == 0) continue;
            double mt = (double)tn_t[n] / tn_cnt[n];
            double mm = (double)tn_m[n] / tn_cnt[n];
            double mc = (double)tn_c[n] / tn_cnt[n];
            printf("  %2d   %9.0f   %7.0f   %6.0f   %8.0f   %u\n",
                   n, mt, mm, mc, mt - mm - mc, tn_cnt[n]);
        }

        printf("\n  by in_g (position within dgswizzle group)\n");
        printf("  in_g  raw_tn  tile_total   mma_asm   commit   residual   count\n");
        for (int ig = 0; ig < IG_SLOTS && ig < 32; ig++) {
            if (ig_cnt[ig] == 0) continue;
            double mt = (double)ig_t[ig] / ig_cnt[ig];
            double mm = (double)ig_m[ig] / ig_cnt[ig];
            double mc = (double)ig_c[ig] / ig_cnt[ig];
            int raw_tn = ig / DG_GROUP_SIZE;
            printf("   %3d    %3d   %9.0f   %7.0f   %6.0f   %8.0f   %u\n",
                   ig, raw_tn, mt, mm, mc, mt - mm - mc, ig_cnt[ig]);
        }
        printf("\n  diagnosis:\n");
        printf("    A (epi-TMEM backpressure): residual high at in_g=0..3 only   → cross-group B-prefetch\n");
        printf("    B (issue-side gaps):       mma_asm high at all in_g          → manual UR desc handling\n");
        printf("    C (cold B at group start): residual AND tma_wait high in_g=0 → B-prefetch at group end\n");
        printf("    D (uniform):               all buckets within ±3%% of mean    → declare done at wall\n");
        free(h_prof_w5);
    }
#endif

#if defined(STRIP_EPILOGUE) || defined(NCU_PROFILE)
    int errors = 0;
    int valid = 0;
    float c0 = 0.0f;
#else
    LAUNCH_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    __nv_bfloat16* h_C = (__nv_bfloat16*)malloc(sC * sizeof(__nv_bfloat16));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sC * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int spot = 0; spot < 32; spot++) {
        long long row = (long long)spot * M_TOTAL / 32;
        int col = (spot * 47) % N_DIM;
        float av = 1.0f + 0.125f * (float)((int)row & 7);
        float bv = (col & 1) ? 1.0f : 1.5f;
        float expected_ab = av * bv * K_DIM;
#if defined(GEMM_ONLY)
        float expected = expected_ab;
#else
        float expected = expected_ab + __bfloat162float(hbias[col]);
#endif
        float got = __bfloat162float(h_C[row * N_DIM + col]);
        float rel = fabsf(got - expected) / fabsf(expected);
        if (rel > 0.02f) {
            if (errors < 8) fprintf(stderr, "  MISMATCH [%lld,%d] got=%.1f exp=%.1f\n", row, col, got, expected);
            errors++;
        }
    }
    printf("%s  errors=%d/32\n", errors == 0 ? "PASS" : "FAIL", errors);
    int valid = (errors == 0) ? 1 : 0;
    float c0 = __bfloat162float(h_C[0]);
#endif

    printf("@@RESULT ms=%.4f tflops=%.2f checksum=0.000000 valid=%d c0=%.1f\n",
           ms, 2.0 * M_TOTAL * N_DIM * K_DIM / ms / 1e9, valid, c0);

    free(hA); free(hB); free(hbias);
#if !defined(STRIP_EPILOGUE) && !defined(NCU_PROFILE)
    free(h_C);
#endif
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_bias); cudaFree(d_C);
#ifdef PROFILE_CYCLES
    if (d_dbg_prof) cudaFree(d_dbg_prof);
#endif
#ifdef PROFILE_KI
    if (d_dbg_prof_ki) cudaFree(d_dbg_prof_ki);
#endif
#ifdef PROFILE_TILE
    if (d_dbg_prof_tile) cudaFree(d_dbg_prof_tile);
#endif
#ifdef PROFILE_W5
    if (d_dbg_prof_w5) cudaFree(d_dbg_prof_w5);
#endif
    return errors == 0 ? 0 : 1;
}
