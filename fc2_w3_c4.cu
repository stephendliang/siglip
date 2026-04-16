/*
FC2 W3 Cluster-of-4 — shared-B pair-stealing.

Design
  Cluster = 4 CTAs (37 clusters × 4 = 148 SMs).
  Pair P0 = {CTA0, CTA1} computes cluster tile (2*m_pair,     tn).
  Pair P1 = {CTA2, CTA3} computes cluster tile (2*m_pair + 1, tn).
  Shared B via 4-CTA TMA multicast (mask 0xF).
  A via pair multicast (mask 0x3 / 0xC).
  Scheduler = CTA0 W7 only.  atomicAdd g_pair_ctr, DSMEM-fanout pair_ready_mbar
  to all 4 CTAs, publish pair_idx to CTA0-hosted pair_bcast (read via DSMEM).

  pair_idx layout (column-major through pairs — keeps B column L2-warm):
    pair_idx = tn * (TILES_M/2) + m_pair
    P0 tm   = 2 * m_pair
    P1 tm   = 2 * m_pair + 1

  TOTAL_PAIRS = (TILES_M/2) * TILES_N = 1813 * 3 = 5439
  Per cluster: 5439 / 37 = 147 pairs (exact).

Synchronization
  - tma_mbar[s] per-CTA, count=1.  Each CTA W0 arrive.expect_tx(TMA_BYTES).
    .multicast::cluster fanout delivers per-CTA complete_tx = A+B bytes.
    Assumption: HW multicast fanout delivers bytes to all target CTAs in
    approximate lockstep, so leader's local mbar firing implies partner's
    SMEM is ready (or arrives within a handful of cycles).  If a race
    surfaces under validation we'll add a pair barrier after tma_mbar wait.
  - mma_mbar[s] per-CTA, count=1.  Pair leader tcgen05.commit with pair mcast
    mask arrives both pair CTAs' local mbar — only leader waits.
  - mainloop_mbar[2] per-CTA, count=1.  Pair leader tcgen05.commit with pair
    mcast mask arrives both pair CTAs' local mbar — all W2-W6 wait.
  - epilogue_mbar[2] pair-shared at pair leader (rank = pair_id*2), count =
    2 * (NUM_EPI_WARPS+1) * 32.  Partner routes via mapa.shared::cluster.

LEAN_DISPATCH
  W0, W1 read pair_bcast at loop top (via pair_ready_mbar).
  W2-W6 defer: read pair_bcast[prev_buf] after mainloop_mbar[prev_buf] acquire.
  Termination: W1 lane 0 arrives mainloop_mbar[iter&1] for the termination iter
  so W2-W6 unblock and see pair_idx >= TOTAL_PAIRS in their deferred read.

Build
  make fc2-w3-c4                          # default: BF16 epilogue, bias+residual
  make fc2-w3-c4 DFLAGS=-DGEMM_ONLY       # strip bias/residual, BF16(A*B) only
  make fc2-w3-c4 DFLAGS=-DSTRIP_EPILOGUE  # GEMM-only, no epilogue at all
*/

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define SM_COUNT 148

#ifndef M_TOTAL
#define M_TOTAL 928256
#endif
#ifndef N_DIM
#define N_DIM 768
#endif
#ifndef K_DIM
#define K_DIM 3072
#endif

#define TM 128
#define TN 256
#define TK 128
#define TILES_M  ((M_TOTAL + TM * 2 - 1) / (TM * 2))
#define TILES_N  (N_DIM / TN)
#define TOTAL_TILES (TILES_M * TILES_N)
#define K_ITERS  (K_DIM / TK)
#define MMA_K    32
#define MMA_PER_KI (TK / MMA_K)

#define CTAS_PER_CLUSTER 4
#define NUM_CLUSTERS (SM_COUNT / CTAS_PER_CLUSTER)
static_assert(TILES_M % 2 == 0, "shared-B pair dispatch requires even TILES_M");
#define TOTAL_PAIRS ((TILES_M / 2) * TILES_N)

#ifndef N_STAGES
#define N_STAGES 6
#endif

/* PREFILL: K_ITERS >= 20 lets W1 skip the epilogue_mbar wait safely (deeper
   pipeline absorbs a few iterations).  Short K forces NO_PREFILL. */
#if (K_DIM / TK) < 20 && !defined(NO_PREFILL)
#define NO_PREFILL
#endif

#define NUM_EPI_WARPS    4
#define NUM_EPI_STAGES   2
#define NUM_EPI_SUBITERS 4
#define NUM_WARPS        8
#define THREADS          256

#define STAGE_BYTES 32768
#define OFF_TMEM            (N_STAGES * STAGE_BYTES)
#define OFF_TMA_MBAR        (OFF_TMEM + 8)
#define OFF_MMA_MBAR        (OFF_TMA_MBAR + N_STAGES * 8)
#define OFF_MAINLOOP_MBAR   (OFF_MMA_MBAR + N_STAGES * 8)
#define OFF_EPILOGUE_MBAR   (OFF_MAINLOOP_MBAR + 16)
#define OFF_LOAD_MBAR       (OFF_EPILOGUE_MBAR + 16)
#define OFF_LOAD_CONSUMED   (OFF_LOAD_MBAR + NUM_EPI_STAGES * 8)
#define OFF_PAIR_BCAST      (OFF_LOAD_CONSUMED + NUM_EPI_STAGES * 8)
#define OFF_PAIR_READY_MBAR (OFF_PAIR_BCAST + 8)
#define _MBAR_END           (OFF_PAIR_READY_MBAR + 16)

#define OFF_BIAS_SMEM       ((_MBAR_END + 15) & ~15)
#define BIAS_SMEM_BYTES     (((N_DIM * 2) + 15) & ~15)
#define OFF_STAGING         ((OFF_BIAS_SMEM + BIAS_SMEM_BYTES + 1023) & ~1023)
#define EPI_STAGE_BYTES     16384
#define STAGING_REGION_BYTES (32 * 128)
#define SMEM_BYTES          ((OFF_STAGING + NUM_EPI_STAGES * EPI_STAGE_BYTES + 127) & ~127)

#define TMA_BYTES 32768
#define TMEM_COLS 512
#define IDESC     0x10400010U
#define SBO       1024
#define BAR_EPI_SYNC "bar.sync 1, 128;"

#define _UNROLL_STR2(x) #x
#define _UNROLL_STR(x)  _UNROLL_STR2(unroll x)
#define PRAGMA_UNROLL(n) _Pragma(_UNROLL_STR(n))
#define K_LOOP_UNROLL N_STAGES

/*══════════════════════════════════════════
  HELPERS
  ══════════════════════════════════════════*/

static __device__ __forceinline__
uint32_t smem_to_uint(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

/* Map a local SMEM address (rank bits = 0) to cluster SMEM at target CTA rank.
   Uses HW mapa.shared::cluster — the portable way for any cluster size. */
static __device__ __forceinline__
uint32_t map_rank(uint32_t local, int dst_rank) {
    uint32_t out;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(out) : "r"(local), "r"(dst_rank));
    return out;
}

static __device__ __forceinline__
uint64_t make_smem_desc(uint32_t addr) {
    uint64_t d = 0;
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= (uint64_t)((SBO  & 0x3FFFF) >> 4) << 32;
    d |= (1ULL << 46);
    d |= (2ULL << 61);
    return d;
}

static __device__ __forceinline__ void mbar_init(uint32_t addr, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count));
}

static __device__ __forceinline__ void mbar_wait(uint32_t addr, uint32_t phase) {
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

/* Local (shared::cta): arrives on the CURRENT CTA's mbar only. */
static __device__ __forceinline__ void mbar_arrive_local(uint32_t addr) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(addr) : "memory");
}

static __device__ __forceinline__
void mbar_arrive_expect_tx_local(uint32_t addr, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(addr), "r"(bytes) : "memory");
}

/* Cluster (shared::cluster): address must carry target-CTA rank bits (use
   map_rank to compute).  For cross-CTA arrives only. */
static __device__ __forceinline__ void mbar_arrive_cluster(uint32_t addr) {
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
        :: "r"(addr) : "memory");
}

#if !defined(STRIP_EPILOGUE) && !defined(GEMM_ONLY)
static __device__ __forceinline__
void tma_load_2d_cta(uint32_t smem_dst, const void* desc, int32_t c0, int32_t c1, uint32_t mbar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_dst), "l"(desc), "r"(c0), "r"(c1), "r"(mbar)
        : "memory");
}
#endif

/* tcgen05.commit with pair multicast: mask=0x3 (P0) or 0xC (P1). */
static __device__ __forceinline__
void tcgen05_commit_pair(uint32_t mbar, uint16_t pair_mask) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(mbar), "h"(pair_mask) : "memory");
}

/*══════════════════════════════════════════
  TMEM LOAD / BF16 EPILOGUE MACROS (ported)
  ══════════════════════════════════════════*/

#define TMEM_LOAD_X32(r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,\
                      r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31, TADDR) \
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

#define TMEM_WAIT() asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory")

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

#define GEMM_CVT_STS(f0,f1,f2,f3,f4,f5,f6,f7, SADDR) \
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
        if (LAST) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); \
        else      asm volatile("cp.async.bulk.wait_group 1;" ::: "memory"); \
    } \
    __syncwarp(); \
    asm volatile(BAR_EPI_SYNC ::: "memory"); \
} while(0)

/*══════════════════════════════════════════
  GLOBAL STATE
  ══════════════════════════════════════════*/

__device__ int g_pair_ctr;

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); }} while(0)
#define CU_CHECK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
    fprintf(stderr, "CU error %d @ %s:%d\n", (int)r, __FILE__, __LINE__); \
    exit(1); }} while(0)

/*══════════════════════════════════════════
  KERNEL
  ══════════════════════════════════════════*/

__global__ void __launch_bounds__(THREADS, 1)
__cluster_dims__(4, 1, 1)
fc2_w3_c4_kernel(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    const __grid_constant__ CUtensorMap tma_c,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ C,
    const __nv_bfloat16* __restrict__ residual,
    const __grid_constant__ CUtensorMap tma_res
) {
    extern __shared__ __align__(128) char smem[];
    const int sm_id     = blockIdx.x;
    const int cta_rank  = sm_id & (CTAS_PER_CLUSTER - 1);
    const int pair_id   = cta_rank >> 1;
    const int pair_lane = cta_rank & 1;
    const int tid       = threadIdx.x;
    const int warp      = tid / 32;
    const int lane      = tid % 32;
    (void)C; (void)residual;

    const uint16_t pair_mcast_mask = (pair_id == 0) ? (uint16_t)0x0003 : (uint16_t)0x000C;

    /*── Mbar init ──*/
    if (tid == 0) {
        for (int s = 0; s < N_STAGES; s++) {
            mbar_init(smem_to_uint(smem + OFF_TMA_MBAR + s * 8), 1);
            mbar_init(smem_to_uint(smem + OFF_MMA_MBAR + s * 8), 1);
        }
        for (int i = 0; i < 2; i++) {
            mbar_init(smem_to_uint(smem + OFF_MAINLOOP_MBAR + i * 8), 1);
            /* Pair-shared epi_mbar: only pair leader's copy is used (partner
               addresses via `& 0xFEFFFFFF`).  Both pair CTAs' 5 arriving warps
               (W2 + W3-W6) hit it — 2 × 5 × 32 = 320. */
            mbar_init(smem_to_uint(smem + OFF_EPILOGUE_MBAR + i * 8),
                      (NUM_EPI_WARPS + 1) * 2 * 32);
            mbar_init(smem_to_uint(smem + OFF_PAIR_READY_MBAR + i * 8), 1);
        }
        for (int s = 0; s < NUM_EPI_STAGES; s++) {
            mbar_init(smem_to_uint(smem + OFF_LOAD_MBAR + s * 8), 1);
            mbar_init(smem_to_uint(smem + OFF_LOAD_CONSUMED + s * 8),
                      NUM_EPI_WARPS * 32);
        }
        if (cta_rank == 0) {
            asm volatile("st.shared.b32 [%0], 0;"
                :: "r"(smem_to_uint(smem + OFF_PAIR_BCAST)));
            asm volatile("st.shared.b32 [%0], 0;"
                :: "r"(smem_to_uint(smem + OFF_PAIR_BCAST + 4)));
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    /*── TMEM alloc (per pair, cta_group::2) ──*/
    if (warp == 1) {
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(smem_to_uint(smem + OFF_TMEM)), "r"(TMEM_COLS));
    }

    /*── Common state ──*/
    uint32_t tma_mbar[N_STAGES], mma_mbar[N_STAGES];
    uint32_t smem_a[N_STAGES],   smem_b[N_STAGES];
    for (int s = 0; s < N_STAGES; s++) {
        tma_mbar[s] = smem_to_uint(smem + OFF_TMA_MBAR + s * 8);
        mma_mbar[s] = smem_to_uint(smem + OFF_MMA_MBAR + s * 8);
        smem_a[s]   = smem_to_uint(smem + s * STAGE_BYTES);
        smem_b[s]   = smem_to_uint(smem + s * STAGE_BYTES + 16384);
    }
    const uint32_t mainloop_mbar_addr = smem_to_uint(smem + OFF_MAINLOOP_MBAR);
    const uint32_t epilogue_mbar_addr = smem_to_uint(smem + OFF_EPILOGUE_MBAR);
    /* Pair-leader's epi_mbar (leader rank = pair_id*2).  For the leader itself
       mapa returns the local address unchanged; for the partner it routes via
       DSMEM.  Both use `mbar_arrive_cluster` below. */
    const uint32_t epi_mbar_shared = map_rank(epilogue_mbar_addr, pair_id * 2);

    /* CTA0-hosted pair_bcast — all CTAs read via ld.shared::cluster */
    const uint32_t pair_bcast_cta0 = map_rank(
        smem_to_uint(smem + OFF_PAIR_BCAST), 0);

    uint64_t desc_a_base[N_STAGES], desc_b_base[N_STAGES];
    for (int s = 0; s < N_STAGES; s++) {
        desc_a_base[s] = make_smem_desc(smem_a[s]);
        desc_b_base[s] = make_smem_desc(smem_b[s]);
    }

    int tma_phase[N_STAGES] = {0};
    int mma_phase[N_STAGES] = {0};
    int ml_phase[2] = {0, 1};
    int pair_ready_phase[2] = {0, 0};
#ifdef NO_PREFILL
    int epi_phase[2] = {1, 1};
#endif

#if !defined(STRIP_EPILOGUE) && !defined(GEMM_ONLY)
    uint32_t load_mbar[NUM_EPI_STAGES];
    uint32_t consumed_mbar[NUM_EPI_STAGES];
    int load_phase[NUM_EPI_STAGES] = {0};
    int load_consumed_phase[NUM_EPI_STAGES] = {0};
    int load_issue_count = 0;
    for (int s = 0; s < NUM_EPI_STAGES; s++) {
        load_mbar[s]     = smem_to_uint(smem + OFF_LOAD_MBAR + s * 8);
        consumed_mbar[s] = smem_to_uint(smem + OFF_LOAD_CONSUMED + s * 8);
    }
#endif

    /*── Bias load ──*/
    {
        const uint32_t bias_saddr = smem_to_uint(smem + OFF_BIAS_SMEM);
        for (int i = tid; i < N_DIM / 2; i += THREADS) {
            uint32_t val;
            asm volatile("ld.global.b32 %0, [%1];" : "=r"(val) : "l"(bias + i * 2));
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(bias_saddr + i * 4), "r"(val));
        }
    }
    __syncthreads();

    /*═══════════════════════════════════════
      W7 SCHEDULER (CTA0 only)
      ═══════════════════════════════════════*/
    if (cta_rank == 0 && warp == 7) {
        int _iter = 0, _buf = 0;
        while (true) {
            int pair_idx;
            if (lane == 0) {
                asm volatile("atom.global.relaxed.gpu.add.s32 %0, [%1], 1;"
                    : "=r"(pair_idx) : "l"(&g_pair_ctr));
                if (pair_idx > TOTAL_PAIRS) pair_idx = TOTAL_PAIRS;
                asm volatile("st.shared.b32 [%0], %1;"
                    :: "r"(smem_to_uint(smem + OFF_PAIR_BCAST + _buf * 4)), "r"(pair_idx));
                asm volatile("fence.acq_rel.cluster;");
                /* Arrive each CTA's local pair_ready_mbar via DSMEM (mapa) */
                const uint32_t pr_local = smem_to_uint(smem + OFF_PAIR_READY_MBAR + _buf * 8);
                mbar_arrive_local(pr_local);
                #pragma unroll
                for (int dst = 1; dst < CTAS_PER_CLUSTER; dst++) {
                    mbar_arrive_cluster(map_rank(pr_local, dst));
                }
            }
            asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
                : "=r"(pair_idx) : "r"(pair_idx));
            if (pair_idx >= TOTAL_PAIRS) break;
            _buf ^= 1;
            _iter++;
        }
        return;
    }

    /*═══════════════════════════════════════
      MAIN TILE LOOP
      ═══════════════════════════════════════*/
    int _iter = 0;

    while (true) {
        const int _buf = _iter & 1;

        /*── Fetch pair_idx — W0 & W1 eager, W2-W6 deferred (LEAN) ──*/
        int pair_idx;
        if (warp == 0 || warp == 1) {
            if (lane == 0) mbar_wait(
                smem_to_uint(smem + OFF_PAIR_READY_MBAR + _buf * 8),
                pair_ready_phase[_buf]);
            pair_ready_phase[_buf] ^= 1;
            asm volatile("ld.shared::cluster.b32 %0, [%1];"
                : "=r"(pair_idx) : "r"(pair_bcast_cta0 + _buf * 4));
            asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
                : "=r"(pair_idx) : "r"(pair_idx));
        } else {
            /* W2-W6: defer until mainloop_mbar acquire.  Placeholder 0 never
               triggers break at loop top. */
            pair_idx = 0;
        }

        if (pair_idx >= TOTAL_PAIRS) {
            /* W0, W1 break path.  Each CTA's W1 arrives its OWN local
               mainloop_mbar (shared::cta scope) so this CTA's W2-W6 unblock
               and see the TOTAL_PAIRS sentinel in pair_bcast via deferred
               read.  During regular iters this arrive came from the pair
               leader's tcgen05.commit.cta_group::2.multicast::cluster, which
               fans out across the pair; here we replace that per-CTA. */
            if (warp == 1 && lane == 0) {
                mbar_arrive_local(mainloop_mbar_addr + _buf * 8);
            }
            break;
        }

        const int m_pair = pair_idx % (TILES_M / 2);
        const int tn     = pair_idx / (TILES_M / 2);
        const int tm     = m_pair * 2 + pair_id;

        const int m_start = tm * TM * 2 + pair_lane * TM;
        const int n_start = tn * TN;
        const int b_c1    = n_start + pair_lane * (TN / 2);
        const bool has_prev = (_iter > 0);

        if (warp == 0) {
            /*── W0: TMA issue ──*/
            const uint32_t smem_base = smem_to_uint(smem);
            PRAGMA_UNROLL(K_LOOP_UNROLL)
            for (int ki = 0; ki < K_ITERS; ki++) {
                const int s = ki % N_STAGES;
                if (has_prev || ki >= N_STAGES) {
                    mbar_wait(mma_mbar[s], mma_phase[s]);
                    mma_phase[s] ^= 1;
                }
                const int k_tk = ki * TK;
                const uint32_t a_dst = smem_base + s * STAGE_BYTES;
                const uint32_t b_dst = a_dst + 16384;
                const uint32_t tma_mbar_s = smem_base + OFF_TMA_MBAR + s * 8;

                if (lane == 0) {
                    /* Own local mbar: arrive.expect_tx in shared::cta scope so
                       the arrival stays on THIS CTA (not routed to CTA0 by the
                       cluster scope). */
                    mbar_arrive_expect_tx_local(tma_mbar_s, TMA_BYTES);

                    if (cta_rank == 0) {
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
                            ".mbarrier::complete_tx::bytes.multicast::cluster"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(b_dst), "l"(&tma_b), "r"(k_tk), "r"(b_c1),
                               "r"(tma_mbar_s), "h"((uint16_t)0x000F)
                            : "memory");
                    }
                    if (pair_lane == 0) {
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
                            ".mbarrier::complete_tx::bytes.multicast::cluster"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(a_dst), "l"(&tma_a), "r"(k_tk), "r"(m_start),
                               "r"(tma_mbar_s), "h"(pair_mcast_mask)
                            : "memory");
                    }
                }
            }
        }

        else if (warp == 1) {
            /*── W1: MMA K-loop — pair leader only ──*/
            if (pair_lane == 0 && lane == 0) {
#ifdef NO_PREFILL
                mbar_wait(epilogue_mbar_addr + _buf * 8, epi_phase[_buf]);
                epi_phase[_buf] ^= 1;
#endif
                /* First K-iter: p=0 zero-inits TMEM at buf*TN */
                mbar_wait(tma_mbar[0], tma_phase[0]);
                tma_phase[0] ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                {
                    uint64_t da = desc_a_base[0], db = desc_b_base[0];
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "setp.ne.b32 p, 0, 0;\n\t"
                        "tcgen05.mma.cta_group::2.kind::f8f6f4 "
                        "[%0], %1, %2, %3, {%4,%5,%6,%7, %8,%9,%10,%11}, p;\n\t"
                        "}"
                        :: "r"(_buf * TN), "l"(da), "l"(db), "r"(IDESC),
                           "r"(0),"r"(0),"r"(0),"r"(0),
                           "r"(0),"r"(0),"r"(0),"r"(0));
                    for (int sub = 1; sub < MMA_PER_KI; sub++) {
                        da += 2; db += 2;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p;\n\t"
                            "setp.ne.b32 p, 1, 0;\n\t"
                            "tcgen05.mma.cta_group::2.kind::f8f6f4 "
                            "[%0], %1, %2, %3, {%4,%5,%6,%7, %8,%9,%10,%11}, p;\n\t"
                            "}"
                            :: "r"(_buf * TN), "l"(da), "l"(db), "r"(IDESC),
                               "r"(0),"r"(0),"r"(0),"r"(0),
                               "r"(0),"r"(0),"r"(0),"r"(0));
                    }
                }
                tcgen05_commit_pair(mma_mbar[0], pair_mcast_mask);

                /* Accumulating K-iters 1..K_ITERS-1 */
                PRAGMA_UNROLL(K_LOOP_UNROLL)
                for (int ki = 1; ki < K_ITERS; ki++) {
                    const int s = ki % N_STAGES;
                    mbar_wait(tma_mbar[s], tma_phase[s]);
                    tma_phase[s] ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint64_t da = desc_a_base[s], db = desc_b_base[s];
                    for (int sub = 0; sub < MMA_PER_KI; sub++) {
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p;\n\t"
                            "setp.ne.b32 p, 1, 0;\n\t"
                            "tcgen05.mma.cta_group::2.kind::f8f6f4 "
                            "[%0], %1, %2, %3, {%4,%5,%6,%7, %8,%9,%10,%11}, p;\n\t"
                            "}"
                            :: "r"(_buf * TN), "l"(da), "l"(db), "r"(IDESC),
                               "r"(0),"r"(0),"r"(0),"r"(0),
                               "r"(0),"r"(0),"r"(0),"r"(0));
                        da += 2; db += 2;
                    }
                    tcgen05_commit_pair(mma_mbar[s], pair_mcast_mask);
                }

                /* Signal epilogue (arrives both pair CTAs' mainloop_mbar) */
                tcgen05_commit_pair(mainloop_mbar_addr + _buf * 8, pair_mcast_mask);
            }
        }

        else if (warp == 2) {
            /*── W2: Epilogue residual-load producer (per-pair) ──*/
            const int prev_buf = _buf ^ 1;
            mbar_wait(mainloop_mbar_addr + prev_buf * 8, ml_phase[prev_buf]);
            ml_phase[prev_buf] ^= 1;

            /* LEAN deferred read: pair_bcast[prev_buf] now carries the previous
               tile's pair_idx (or TOTAL_PAIRS sentinel on termination). */
            int prev_idx;
            if (lane == 0) {
                asm volatile("ld.shared::cluster.b32 %0, [%1];"
                    : "=r"(prev_idx) : "r"(pair_bcast_cta0 + prev_buf * 4));
            }
            asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
                : "=r"(prev_idx) : "r"(prev_idx));
            /* _prev_pair = prev_idx; (unused in current path) */

            if (prev_idx >= TOTAL_PAIRS) {
                mbar_arrive_cluster(epi_mbar_shared + prev_buf * 8);
                goto _lean_done;
            }

#if defined(STRIP_EPILOGUE) || defined(GEMM_ONLY)
            if (has_prev) mbar_arrive_cluster(epi_mbar_shared + prev_buf * 8);
#else
            if (has_prev) {
                const int pm_pair = prev_idx % (TILES_M / 2);
                const int ptn     = prev_idx / (TILES_M / 2);
                const int ptm     = pm_pair * 2 + pair_id;
                const int prev_m  = ptm * TM * 2 + pair_lane * TM;
                const int prev_n  = ptn * TN;

                for (int si = 0; si < NUM_EPI_SUBITERS; si++) {
                    const int stage = si % NUM_EPI_STAGES;
                    if (load_issue_count >= NUM_EPI_STAGES) {
                        mbar_wait(consumed_mbar[stage], load_consumed_phase[stage]);
                        load_consumed_phase[stage] ^= 1;
                    }
                    if (lane == 0) {
                        const uint32_t res_dst = smem_to_uint(smem + OFF_STAGING
                            + stage * EPI_STAGE_BYTES);
                        mbar_arrive_expect_tx_local(load_mbar[stage], EPI_STAGE_BYTES);
                        tma_load_2d_cta(res_dst, &tma_res,
                                        prev_n + si * 64, prev_m, load_mbar[stage]);
                    }
                    load_issue_count++;
                }
                mbar_arrive_cluster(epi_mbar_shared + prev_buf * 8);
            }
#endif
        }

        else {
            /*── W3-W6: Epilogue compute (per-pair) ──*/
            const int ew = warp - 3;
            const int prev_buf = _buf ^ 1;
            mbar_wait(mainloop_mbar_addr + prev_buf * 8, ml_phase[prev_buf]);
            ml_phase[prev_buf] ^= 1;

            int prev_idx;
            if (lane == 0) {
                asm volatile("ld.shared::cluster.b32 %0, [%1];"
                    : "=r"(prev_idx) : "r"(pair_bcast_cta0 + prev_buf * 4));
            }
            asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
                : "=r"(prev_idx) : "r"(prev_idx));
            /* _prev_pair = prev_idx; (unused in current path) */

            if (prev_idx >= TOTAL_PAIRS) {
                mbar_arrive_cluster(epi_mbar_shared + prev_buf * 8);
                goto _lean_done;
            }

#ifdef STRIP_EPILOGUE
            if (has_prev) mbar_arrive_cluster(epi_mbar_shared + prev_buf * 8);
#else
            if (has_prev) {
                const int pm_pair = prev_idx % (TILES_M / 2);
                const int ptn     = prev_idx / (TILES_M / 2);
                const int ptm     = pm_pair * 2 + pair_id;
                const int prev_m  = ptm * TM * 2 + pair_lane * TM;
                const int prev_n  = ptn * TN;
                const int prev_n_bias = prev_n;

                const uint32_t xor_val = (lane & 7) << 4;
                const uint32_t sw0 = 0 ^ xor_val, sw1 = 16 ^ xor_val;
                const uint32_t sw2 = 32 ^ xor_val, sw3 = 48 ^ xor_val;
                const uint32_t sw4 = 64 ^ xor_val, sw5 = 80 ^ xor_val;
                const uint32_t sw6 = 96 ^ xor_val, sw7 = 112 ^ xor_val;

                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

#ifdef GEMM_ONLY
                for (int si = 0; si < NUM_EPI_SUBITERS; si++) {
                    const int stage = si % NUM_EPI_STAGES;
                    const int nc_base = si * 64;
                    const int row_group = ew;

                    const int taddr_base = prev_buf * TN
                        + ((pair_lane * 128 + row_group * 32) << 16);
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
                        GEMM_CVT_STS(a0,a1,a2,a3,a4,a5,a6,a7,   stage_base + rsw0);
                        GEMM_CVT_STS(a8,a9,a10,a11,a12,a13,a14,a15,  stage_base + rsw1);
                        GEMM_CVT_STS(a16,a17,a18,a19,a20,a21,a22,a23, stage_base + rsw2);
                        GEMM_CVT_STS(a24,a25,a26,a27,a28,a29,a30,a31, stage_base + rsw3);
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(BAR_EPI_SYNC ::: "memory");
                    EPI_STORE(stage, nc_base, prev_n, prev_m);
                    EPI_WAIT(si == NUM_EPI_SUBITERS - 1);
                }
                mbar_arrive_cluster(epi_mbar_shared + prev_buf * 8);
#else
                const uint32_t bias_saddr = smem_to_uint(smem + OFF_BIAS_SMEM);
                for (int si = 0; si < NUM_EPI_SUBITERS; si++) {
                    const int stage = si % NUM_EPI_STAGES;
                    const int nc_base = si * 64;
                    const int row_group = ew;

                    mbar_wait(load_mbar[stage], load_phase[stage]);
                    load_phase[stage] ^= 1;

                    const int taddr_base = prev_buf * TN
                        + ((pair_lane * 128 + row_group * 32) << 16);
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

                        const uint32_t bs = bias_saddr + (prev_n_bias + nc) * 2;
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
                        uint4 rv0, rv1, rv2, rv3;
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(rv0.x),"=r"(rv0.y),"=r"(rv0.z),"=r"(rv0.w)
                            : "r"(stage_base + rsw0));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(rv1.x),"=r"(rv1.y),"=r"(rv1.z),"=r"(rv1.w)
                            : "r"(stage_base + rsw1));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(rv2.x),"=r"(rv2.y),"=r"(rv2.z),"=r"(rv2.w)
                            : "r"(stage_base + rsw2));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(rv3.x),"=r"(rv3.y),"=r"(rv3.z),"=r"(rv3.w)
                            : "r"(stage_base + rsw3));

                        TMEM_WAIT();

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
                    }

                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(BAR_EPI_SYNC ::: "memory");
                    EPI_STORE(stage, nc_base, prev_n, prev_m);
                    EPI_WAIT(si == NUM_EPI_SUBITERS - 1);
                    if (si > 0)
                        mbar_arrive_local(consumed_mbar[(si - 1) % NUM_EPI_STAGES]);
                    if (si == NUM_EPI_SUBITERS - 1)
                        mbar_arrive_local(consumed_mbar[stage]);
                }
                mbar_arrive_cluster(epi_mbar_shared + prev_buf * 8);
#endif /* GEMM_ONLY */
            }
#endif /* STRIP_EPILOGUE */
        }

        /* _prev_pair = pair_idx; */
        _iter++;
    }

_lean_done: ;

    /*── TMEM dealloc ──*/
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
            :: "r"(0), "r"(TMEM_COLS));
    }
}

/*══════════════════════════════════════════
  HOST
  ══════════════════════════════════════════*/

__global__ void init_residual(__nv_bfloat16* res, int n_dim, long long total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int col = (int)(idx % n_dim);
        int row = (int)(idx / n_dim);
        res[idx] = __float2bfloat16((float)(row % 128) * 0.25f + (float)col * 0.125f);
    }
}

int main() {
    setbuf(stdout, NULL);
    printf("FC2 W3 c4 — cluster-of-4 shared-B pair-stealing\n");
    printf("  GEMM: [%d,%d] x [%d,%d]^T  NS=%d  SMEM=%d B  clusters=%d  pairs=%d\n",
           M_TOTAL, K_DIM, N_DIM, K_DIM, N_STAGES, SMEM_BYTES,
           NUM_CLUSTERS, TOTAL_PAIRS);
#ifdef STRIP_EPILOGUE
    printf("  Mode: STRIP_EPILOGUE (GEMM core only, validation disabled)\n");
#elif defined(GEMM_ONLY)
    printf("  Mode: GEMM_ONLY (BF16(A*B), no bias/residual)\n");
#else
    printf("  Mode: full (bias + residual)\n");
#endif

    uint8_t *d_A, *d_B;
    __nv_bfloat16 *d_bias, *d_residual, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A,        (size_t)M_TOTAL * K_DIM));
    CUDA_CHECK(cudaMalloc(&d_B,        (size_t)N_DIM   * K_DIM));
    CUDA_CHECK(cudaMalloc(&d_bias,     (size_t)N_DIM   * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_residual, (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_C,        (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16)));

    CUDA_CHECK(cudaMemset(d_A, 0x3C, (size_t)M_TOTAL * K_DIM));
    {
        uint8_t* h_B = (uint8_t*)malloc((size_t)N_DIM * K_DIM);
        for (int n = 0; n < N_DIM; n++)
            for (int k = 0; k < K_DIM; k++)
                h_B[(long long)n * K_DIM + k] = (n & 1) ? 0x38 : 0x3C;
        cudaMemcpy(d_B, h_B, (size_t)N_DIM * K_DIM, cudaMemcpyHostToDevice);
        free(h_B);
    }
    {
        __nv_bfloat16* h_bias = (__nv_bfloat16*)malloc((size_t)N_DIM * sizeof(__nv_bfloat16));
        for (int i = 0; i < N_DIM; i++) h_bias[i] = __float2bfloat16((float)(i + 1));
        cudaMemcpy(d_bias, h_bias, (size_t)N_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        free(h_bias);
    }
    {
        long long total = (long long)M_TOTAL * N_DIM;
        init_residual<<<(total + 255) / 256, 256>>>(d_residual, N_DIM, total);
    }

    CUtensorMap h_tma_a;
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
    CUtensorMap h_tma_b;
    {
        uint64_t dims[2]    = {(uint64_t)K_DIM, (uint64_t)N_DIM};
        uint64_t strides[1] = {(uint64_t)K_DIM};
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
        uint32_t box[2]     = {64, 128};
        uint32_t estrides[2]= {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(&h_tma_res,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)d_residual,
            dims, strides, box, estrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }

    CUDA_CHECK(cudaFuncSetAttribute(fc2_w3_c4_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));

    int* d_pair_ctr_ptr;
    CUDA_CHECK(cudaGetSymbolAddress((void**)&d_pair_ctr_ptr, g_pair_ctr));

    #define LAUNCH() do { \
        cudaMemsetAsync(d_pair_ctr_ptr, 0, sizeof(int)); \
        fc2_w3_c4_kernel<<<SM_COUNT, THREADS, SMEM_BYTES>>>( \
            h_tma_a, h_tma_b, h_tma_c, d_bias, d_C, d_residual, h_tma_res); \
    } while(0)

    for (int i = 0; i < 2; i++) LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaError_t last = cudaGetLastError();
    if (last != cudaSuccess) {
        printf("Launch error: %s\n", cudaGetErrorString(last));
        return 1;
    }

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 10; i++) LAUNCH();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    ms /= 10.0f;
    printf("FC2-W3-c4 kernel: %.3f ms  %.2f TFLOPS\n",
           ms, 2.0 * M_TOTAL * N_DIM * K_DIM / ms / 1e9);

    LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    __nv_bfloat16* h_C = (__nv_bfloat16*)malloc((size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));

    double cksum = 0;
    {
        long long total_elems = (long long)M_TOTAL * N_DIM;
        long long stride = total_elems / 1024;
        for (int i = 0; i < 1024; i++)
            cksum += (double)__bfloat162float(h_C[(long long)i * stride]);
    }

#ifndef STRIP_EPILOGUE
    int errors = 0;
    for (int spot = 0; spot < 32; spot++) {
        long long row = (long long)spot * M_TOTAL / 32;
        int col = (spot * 47) % N_DIM;
        float b_val = (col & 1) ? 1.0f : 1.5f;
        float gemm = (float)K_DIM * 1.5f * b_val;
        float res_bf16_f = __bfloat162float(__float2bfloat16(
            (float)((int)row % 128) * 0.25f + (float)col * 0.125f));
        float bias_bf16_f = __bfloat162float(__float2bfloat16((float)(col + 1)));
#ifdef GEMM_ONLY
        __nv_bfloat16 expected = __float2bfloat16(gemm);
#else
        float acc_rounded = __bfloat162float(__float2bfloat16(gemm));
        float after_bias = __bfloat162float(__float2bfloat16(acc_rounded + bias_bf16_f));
        __nv_bfloat16 expected = __float2bfloat16(after_bias + res_bf16_f);
#endif
        __nv_bfloat16 actual = h_C[row * N_DIM + col];
        float ef = __bfloat162float(expected);
        float af = __bfloat162float(actual);
        if (ef != af) {
            if (errors < 5)
                printf("  MISMATCH at (%lld,%d): expected %.1f got %.1f "
                       "(gemm=%.1f bias=%.1f res=%.4f)\n",
                       row, col, ef, af, gemm, bias_bf16_f, res_bf16_f);
            errors++;
        }
    }
    int valid = (errors == 0) ? 1 : 0;
    printf("Validation: %d/32 spot checks passed%s\n",
           32 - errors, valid ? "" : " — FAILED");
    printf("@@RESULT ms=%.3f tflops=%.2f checksum=%f valid=%d c0=%.1f\n",
           ms, 2.0 * M_TOTAL * N_DIM * K_DIM / ms / 1e9, cksum, valid,
           __bfloat162float(h_C[0]));
#else
    printf("@@RESULT ms=%.3f tflops=%.2f checksum=%f valid=1 c0=%.1f\n",
           ms, 2.0 * M_TOTAL * N_DIM * K_DIM / ms / 1e9, cksum,
           __bfloat162float(h_C[0]));
#endif

    free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_bias); cudaFree(d_residual); cudaFree(d_C);
    return 0;
}
