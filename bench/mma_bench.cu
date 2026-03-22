/*
MMA microbenchmarks for SM100a (Blackwell) — tcgen05 WGMMA characterization

Measures UTCQMMA throughput, latency, scheduling overhead, and overlap
with other operations. Determines whether ptxas generates optimal K-loop
schedules or leaves performance on the table (like the epilogue).

Sections:
  A. MMA issue throughput — ILP sweep (1/4/8/16 back-to-back UTCQMMA)
  B. K-iteration decomposition — fence + 4×MMA + commit (sub-component costs)
  C. MMA latency — full pipeline: issue → commit → mbar_wait → tcgen05.ld
  D. TMEM load throughput — tcgen05.ld.sync x16/x32/x64, ILP sweep
  E. MMA shadow budget — STS/BF16 between MMA issue and commit
  F. MMA + TMA overlap — concurrent TMA load during MMA compute
  G. Full K-loop pipeline — 1/4/8/16/24 iterations amortized throughput
  H. TMEM double-buffer — read buf0 while MMA writes buf1
  I. STS + TMEM_LD overlap — can epilogue STS and tcgen05.ld run concurrently?
  J. MMA compute time — commit to mbar_wait return (tensor core latency)
  K. Cross-warp epilogue handoff — W1 commit → W2 mbar_wait + TMEM_LD
  L. Multi-warp contention — 4 epilogue warps STS/LDS/BF16 during MMA
  N. Full epilogue pipeline — TMEM_LD+F2FP+HADD2+STS+fence during MMA
  O. Combined MMA shadow — full epilogue work in single-warp shadow window
  P. W0 TMA dispatch during MMA — TMA load contention test
  Q. fence.proxy.async + mbar_arrive during MMA
  R. Pipelined K-loop with TMA — W0 TMA + W1 MMA (realistic FC2 pattern)
  S. TMA store during MMA — cp.async.bulk.tensor store contention
  M. Multi-SM scaling — all 74 clusters, per-cluster throughput variance

Requires: SM100a (B200), cta_group::2, __cluster_dims__(2,1,1)
Build: make mma-bench && ./mma-bench
*/

#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>

#define WARMUP 32
#define REPS   256

/* Tile dimensions (matching FC2) */
#define BENCH_TK 128                /* K tile = 128 bytes for FP8 */
#define BENCH_TM 128                /* M tile per CTA */
#define BENCH_TN 256                /* N tile (both CTAs combined) */

/* SMEM layout per CTA */
#define MBAR_OFF   0
#define N_MBAR     32
#define TMEM_OFF   (N_MBAR * 8)    /* 256: TMEM address from alloc */
#define A_OFF      512              /* 512B-aligned for TMA SWIZZLE_128B */
#define A_SZ       (BENCH_TK * BENCH_TM)        /* 16384 = 16KB */
#define B_OFF      (A_OFF + A_SZ)               /* 16896 */
#define B_SZ       (BENCH_TK * BENCH_TN / 2)    /* 16384 = 16KB */
#define STAGE_SZ   (A_SZ + B_SZ)                /* 32768 = 32KB */
#define N_STAGES   4
#define WORK_OFF   (A_OFF + N_STAGES * STAGE_SZ)
#define WORK_SZ    16384
#define SMEM_TOT   (WORK_OFF + WORK_SZ)          /* ~147KB */

/* Mbarrier slots */
#define MB_TMA(s)  (MBAR_OFF + (s) * 8)
#define MB_MMA(s)  (MBAR_OFF + (N_STAGES + (s)) * 8)
#define MB_DONE    (MBAR_OFF + (2 * N_STAGES + 1) * 8)

/* SMEM A/B stage addresses */
#define STAGE_A(s) (A_OFF + (s) * STAGE_SZ)
#define STAGE_B(s) (B_OFF + (s) * STAGE_SZ)

/* Global data (FP8 = UINT8) */
#define GA_K  (BENCH_TK * N_STAGES)     /* 512 */
#define GA_M  (BENCH_TM * 2)            /* 256, room for 2 CTAs */
#define GB_K  (BENCH_TK * N_STAGES)     /* 512 */
#define GB_N  BENCH_TN                   /* 256 */

/* WGMMA constants */
#define IDESC       0x10400010U
#define SBO         1024
#define TMEM_COLS   512

#define CU_CHECK(x) do { CUresult _r = (x); if (_r) { const char *_s; cuGetErrorString(_r, &_s); fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, _s); exit(1); } } while(0)
#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1); } } while(0)

/* ═══════ PTX helpers ═══════ */

__device__ __forceinline__ long long clk() {
    long long r;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(r) :: "memory");
    return r;
}

__device__ __forceinline__ uint32_t to_smem(const void *p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint64_t make_desc(uint32_t addr) {
    uint64_t d = 0;
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= (uint64_t)((SBO & 0x3FFFF) >> 4) << 32;
    d |= (1ULL << 46);
    d |= (2ULL << 61);  /* SWIZZLE_128B */
    return d;
}

__device__ __forceinline__ void mb_init(uint32_t a, int n) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(n));
}

__device__ __forceinline__ void mb_expect_tx(uint32_t a, int b) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(a), "r"(b) : "memory");
}

__device__ __forceinline__ void mb_arrive(uint32_t a) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                 :: "r"(a) : "memory");
}

__device__ __forceinline__ void mb_wait(uint32_t a, int ph) {
    uint32_t done;
    do {
        asm volatile(
            "{\n\t.reg .pred p;\n\t"
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2, 0x989680;\n\t"
            "selp.b32 %0, 1, 0, p;\n\t}"
            : "=r"(done) : "r"(a), "r"(ph));
    } while (!done);
}

__device__ __forceinline__ void mma_fence() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

__device__ __forceinline__
void mma_issue(uint32_t taddr, uint64_t da, uint64_t db, int accum) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f8f6f4 "
        "[%0], %1, %2, %3, {%5,%6,%7,%8, %9,%10,%11,%12}, p;\n\t"
        "}"
        :
        : "r"(taddr), "l"(da), "l"(db), "r"(IDESC), "r"(accum),
          "r"(0),"r"(0),"r"(0),"r"(0), "r"(0),"r"(0),"r"(0),"r"(0));
}

__device__ __forceinline__
void mma_commit(uint32_t mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        :: "r"(mbar_addr), "h"((uint16_t)0x3) : "memory");
}

__device__ __forceinline__
void tma_ld(uint32_t dst, const void *tm, int x, int y, uint32_t mb) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tm), "r"(x), "r"(y), "r"(mb) : "memory");
}

/* 4 sub-MMAs = one K-iteration (matches FC2: MMA_PER_KI=4, MMA_K=32, TK=128) */
#define MMA_4SUB(taddr, da, db, accum) do { \
    mma_issue((taddr), (da),     (db),     (accum)); \
    mma_issue((taddr), (da) + 2, (db) + 2, 1); \
    mma_issue((taddr), (da) + 4, (db) + 4, 1); \
    mma_issue((taddr), (da) + 6, (db) + 6, 1); \
} while(0)

/* ═══════ Common setup ═══════ */

/*
All MMA bench kernels share:
 - 2 CTAs in a cluster (cta_group::2)
 - TMEM alloc (512 cols)
 - TMA load of N stages of A+B data
 - CTA1 idles after setup (stays alive for cluster SMEM)
 - CTA0 does all measurements (lane 0 for MMA, all lanes for TMEM load)

Returns: false for CTA1 (caller should go to teardown).
*/
struct Setup {
    uint32_t sb;
    int lane, cta_rank;
    uint64_t da[N_STAGES], db[N_STAGES];
    uint32_t taddr;
};

__device__ bool bench_init(Setup &s,
                           const CUtensorMap &tma_a, const CUtensorMap &tma_b,
                           int n_load = 1) {
    extern __shared__ __align__(128) char smem[];
    s.sb = to_smem(smem);
    s.lane = threadIdx.x;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(s.cta_rank));

    if (s.lane == 0)
        for (int i = 0; i < N_MBAR; i++)
            mb_init(s.sb + MBAR_OFF + i * 8, 1);
    __syncwarp();
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    /* TMEM alloc — collective, all threads in both CTAs */
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(s.sb + TMEM_OFF), "r"(TMEM_COLS));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(s.taddr) : "r"(s.sb + TMEM_OFF));

    /* Compute SMEM descriptors per stage */
    for (int i = 0; i < N_STAGES; i++) {
        s.da[i] = make_desc(s.sb + STAGE_A(i));
        s.db[i] = make_desc(s.sb + STAGE_B(i));
    }

    /* TMA load: each CTA loads its own A + B */
    for (int st = 0; st < n_load && st < N_STAGES; st++) {
        if (s.lane == 0) {
            uint32_t mb = s.sb + MB_TMA(st);
            mb_expect_tx(mb, A_SZ + B_SZ);
            tma_ld(s.sb + STAGE_A(st), &tma_a,
                   st * BENCH_TK, s.cta_rank * BENCH_TM, mb);
            tma_ld(s.sb + STAGE_B(st), &tma_b,
                   st * BENCH_TK, s.cta_rank * (BENCH_TN / 2), mb);
        }
        mb_wait(s.sb + MB_TMA(st), 0);
    }

    /* Sync: both CTAs ready */
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    return s.cta_rank == 0;
}

__device__ void bench_fini(Setup &s) {
    /* Drain MMA pipeline: commit any uncommitted ops + wait for completion.
       Without this, dealloc hangs because TMEM is still being written. */
    if (s.lane == 0 && s.cta_rank == 0) {
        mma_commit(s.sb + MB_DONE);
        mb_wait(s.sb + MB_DONE, 0);
    }
    __syncwarp();

    /* Cluster barrier so both CTAs are done */
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    /* TMEM dealloc */
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
        :: "r"(0), "r"(TMEM_COLS));
}

/* ═══════ A. MMA issue throughput — ILP sweep ═══════ */

template <int N_MMA>
__global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void k_mma_tput(__grid_constant__ const CUtensorMap tma_a,
                __grid_constant__ const CUtensorMap tma_b,
                long long *out) {
    Setup s;
    if (!bench_init(s, tma_a, tma_b)) { bench_fini(s); return; }

    uint64_t da = s.da[0], db = s.db[0];

    if (s.lane == 0) {
        /* Warmup: init accumulator then repeat */
        mma_fence();
        mma_issue(0, da, db, 0);
        for (int w = 0; w < WARMUP * N_MMA; w++)
            mma_issue(0, da, db, 1);

        /* Measure */
        long long tot = 0;
        for (int rep = 0; rep < REPS; rep++) {
            long long t0 = clk();
            #pragma unroll
            for (int i = 0; i < N_MMA; i++)
                mma_issue(0, da, db, 1);
            long long t1 = clk();
            tot += t1 - t0;
        }
        out[0] = tot;
    }
    bench_fini(s);
}

/* ═══════ B. K-iteration decomposition ═══════ */

/* Fence only */
__global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void k_fence_only(__grid_constant__ const CUtensorMap tma_a,
                  __grid_constant__ const CUtensorMap tma_b,
                  long long *out) {
    Setup s;
    if (!bench_init(s, tma_a, tma_b)) { bench_fini(s); return; }

    if (s.lane == 0) {
        for (int w = 0; w < WARMUP; w++)
            mma_fence();
        long long tot = 0;
        for (int rep = 0; rep < REPS; rep++) {
            long long t0 = clk();
            mma_fence();
            long long t1 = clk();
            tot += t1 - t0;
        }
        out[0] = tot;
    }
    bench_fini(s);
}

/* Commit only (need MMA issued first) */
__global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void k_commit_only(__grid_constant__ const CUtensorMap tma_a,
                   __grid_constant__ const CUtensorMap tma_b,
                   long long *out) {
    Setup s;
    if (!bench_init(s, tma_a, tma_b)) { bench_fini(s); return; }

    uint64_t da = s.da[0], db = s.db[0];
    uint32_t mbar0 = s.sb + MB_MMA(0);
    uint32_t mbar1 = s.sb + MB_MMA(1);

    if (s.lane == 0) {
        mma_fence();
        mma_issue(0, da, db, 0);
    }

    /* Warmup: all lanes wait to prevent mbar phase overrun */
    int ph0 = 0, ph1 = 0;
    for (int w = 0; w < WARMUP; w++) {
        if (s.lane == 0) {
            MMA_4SUB(0, da, db, 1);
            mma_commit(mbar0);
        }
        mb_wait(mbar0, ph0); ph0 ^= 1;
    }

    /* Measure: time commit dispatch only */
    long long tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        uint32_t mbar = (rep & 1) ? mbar1 : mbar0;
        int &ph = (rep & 1) ? ph1 : ph0;
        if (s.lane == 0) {
            MMA_4SUB(0, da, db, 1);
            long long t0 = clk();
            mma_commit(mbar);
            long long t1 = clk();
            tot += t1 - t0;
        }
        mb_wait(mbar, ph); ph ^= 1;
    }
    if (s.lane == 0) out[0] = tot;
    bench_fini(s);
}

/* Full K-iteration: fence + 4×MMA + commit + wait */
__global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void k_kiter_full(__grid_constant__ const CUtensorMap tma_a,
                  __grid_constant__ const CUtensorMap tma_b,
                  long long *out) {
    Setup s;
    if (!bench_init(s, tma_a, tma_b)) { bench_fini(s); return; }

    uint64_t da = s.da[0], db = s.db[0];
    uint32_t mbar0 = s.sb + MB_MMA(0), mbar1 = s.sb + MB_MMA(1);

    /* Warmup */
    int ph0 = 0, ph1 = 0;
    for (int w = 0; w < WARMUP; w++) {
        uint32_t mbar = (w & 1) ? mbar1 : mbar0;
        int &ph = (w & 1) ? ph1 : ph0;
        if (s.lane == 0) {
            mma_fence();
            MMA_4SUB(0, da, db, w > 0 ? 1 : 0);
            mma_commit(mbar);
        }
        mb_wait(mbar, ph); ph ^= 1;
    }

    /* Measure */
    long long tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        uint32_t mbar = (rep & 1) ? mbar1 : mbar0;
        int &ph = (rep & 1) ? ph1 : ph0;
        long long t0 = clk();
        if (s.lane == 0) {
            mma_fence();
            MMA_4SUB(0, da, db, 1);
            mma_commit(mbar);
        }
        mb_wait(mbar, ph); ph ^= 1;
        long long t1 = clk();
        if (s.lane == 0) tot += t1 - t0;
    }

    if (s.lane == 0) out[0] = tot;
    bench_fini(s);
}

/* ═══════ C. MMA latency — issue to TMEM readable ═══════ */

__global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void k_mma_lat(__grid_constant__ const CUtensorMap tma_a,
               __grid_constant__ const CUtensorMap tma_b,
               long long *out) {
    Setup s;
    if (!bench_init(s, tma_a, tma_b)) { bench_fini(s); return; }

    uint64_t da = s.da[0], db = s.db[0];
    uint32_t mbar0 = s.sb + MB_MMA(0), mbar1 = s.sb + MB_MMA(1);

    float r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;

    /* Warmup */
    int ph0 = 0, ph1 = 0;
    for (int w = 0; w < WARMUP; w++) {
        uint32_t mbar = (w & 1) ? mbar1 : mbar0;
        int &ph = (w & 1) ? ph1 : ph0;
        if (s.lane == 0) {
            mma_fence();
            MMA_4SUB(0, da, db, w > 0 ? 1 : 0);
            mma_commit(mbar);
        }
        mb_wait(mbar, ph); ph ^= 1;
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
            : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3),
              "=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7),
              "=f"(r8),"=f"(r9),"=f"(r10),"=f"(r11),
              "=f"(r12),"=f"(r13),"=f"(r14),"=f"(r15)
            : "r"(0));
    }

    /* Measure: fence → MMA×4 → commit → wait → TMEM load */
    long long tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        uint32_t mbar = (rep & 1) ? mbar1 : mbar0;
        int &ph = (rep & 1) ? ph1 : ph0;
        long long t0 = clk();
        if (s.lane == 0) {
            mma_fence();
            MMA_4SUB(0, da, db, 1);
            mma_commit(mbar);
        }
        mb_wait(mbar, ph); ph ^= 1;
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
            : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3),
              "=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7),
              "=f"(r8),"=f"(r9),"=f"(r10),"=f"(r11),
              "=f"(r12),"=f"(r13),"=f"(r14),"=f"(r15)
            : "r"(0));
        long long t1 = clk();
        if (s.lane == 0) tot += t1 - t0;
    }

    if (s.lane == 0) {
        out[0] = tot;
        /* Verification: r0 should be ~288.0 for A=B=1.5 FP8, K=128 */
        out[1] = __float_as_int(r0);
    }
    bench_fini(s);
}

/* ═══════ D. TMEM load throughput ═══════ */

template <int N_LOADS>
__global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void k_tmem_load(__grid_constant__ const CUtensorMap tma_a,
                 __grid_constant__ const CUtensorMap tma_b,
                 long long *out) {
    Setup s;
    if (!bench_init(s, tma_a, tma_b)) { bench_fini(s); return; }

    uint64_t da = s.da[0], db = s.db[0];
    uint32_t mbar = s.sb + MB_MMA(0);

    float r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;

    /* Pre-fill TMEM via MMA */
    if (s.lane == 0) {
        mma_fence();
        MMA_4SUB(0, da, db, 0);
        mma_commit(mbar);
    }
    mb_wait(mbar, 0);

    /* Warmup */
    for (int w = 0; w < WARMUP; w++) {
        #pragma unroll
        for (int i = 0; i < N_LOADS; i++)
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3),
                  "=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7),
                  "=f"(r8),"=f"(r9),"=f"(r10),"=f"(r11),
                  "=f"(r12),"=f"(r13),"=f"(r14),"=f"(r15)
                : "r"(i * 32));
    }

    /* Measure */
    long long tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        long long t0 = clk();
        #pragma unroll
        for (int i = 0; i < N_LOADS; i++)
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3),
                  "=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7),
                  "=f"(r8),"=f"(r9),"=f"(r10),"=f"(r11),
                  "=f"(r12),"=f"(r13),"=f"(r14),"=f"(r15)
                : "r"(i * 32));
        long long t1 = clk();
        tot += t1 - t0;
    }

    if (s.lane == 0) out[0] = tot;
    bench_fini(s);
}

/* x32 variant */
template <int N_LOADS>
__global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void k_tmem_load_x32(__grid_constant__ const CUtensorMap tma_a,
                     __grid_constant__ const CUtensorMap tma_b,
                     long long *out) {
    Setup s;
    if (!bench_init(s, tma_a, tma_b)) { bench_fini(s); return; }

    uint64_t da = s.da[0], db = s.db[0];
    uint32_t mbar = s.sb + MB_MMA(0);

    float r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;
    float s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15;

    /* Pre-fill TMEM */
    if (s.lane == 0) {
        mma_fence();
        MMA_4SUB(0, da, db, 0);
        mma_commit(mbar);
    }
    mb_wait(mbar, 0);

    for (int w = 0; w < WARMUP; w++) {
        #pragma unroll
        for (int i = 0; i < N_LOADS; i++)
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3),
                  "=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7),
                  "=f"(r8),"=f"(r9),"=f"(r10),"=f"(r11),
                  "=f"(r12),"=f"(r13),"=f"(r14),"=f"(r15),
                  "=f"(s0),"=f"(s1),"=f"(s2),"=f"(s3),
                  "=f"(s4),"=f"(s5),"=f"(s6),"=f"(s7),
                  "=f"(s8),"=f"(s9),"=f"(s10),"=f"(s11),
                  "=f"(s12),"=f"(s13),"=f"(s14),"=f"(s15)
                : "r"(i * 64));
    }

    long long tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        long long t0 = clk();
        #pragma unroll
        for (int i = 0; i < N_LOADS; i++)
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3),
                  "=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7),
                  "=f"(r8),"=f"(r9),"=f"(r10),"=f"(r11),
                  "=f"(r12),"=f"(r13),"=f"(r14),"=f"(r15),
                  "=f"(s0),"=f"(s1),"=f"(s2),"=f"(s3),
                  "=f"(s4),"=f"(s5),"=f"(s6),"=f"(s7),
                  "=f"(s8),"=f"(s9),"=f"(s10),"=f"(s11),
                  "=f"(s12),"=f"(s13),"=f"(s14),"=f"(s15)
                : "r"(i * 64));
        long long t1 = clk();
        tot += t1 - t0;
    }

    if (s.lane == 0) out[0] = tot;
    bench_fini(s);
}

/* ═══════ E. MMA shadow budget — STS/BF16 between MMA and commit ═══════ */
/* Measures how much LSU/compute work fits "free" in the MMA shadow window.
   Lane 0 issues MMA, then ALL lanes do N_OPS STS (or BF16) before commit. */

template <int N_STS>
__global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void k_mma_shadow_sts(__grid_constant__ const CUtensorMap tma_a,
                      __grid_constant__ const CUtensorMap tma_b,
                      long long *out) {
    Setup s;
    if (!bench_init(s, tma_a, tma_b)) { bench_fini(s); return; }

    uint64_t da = s.da[0], db = s.db[0];
    uint32_t mbar0 = s.sb + MB_MMA(0), mbar1 = s.sb + MB_MMA(1);
    uint32_t sts_addr = s.sb + WORK_OFF + s.lane * 16;
    uint32_t v = 0x3c003c00u | s.lane;

    /* Warmup */
    int ph0 = 0, ph1 = 0;
    for (int w = 0; w < WARMUP; w++) {
        uint32_t mbar = (w & 1) ? mbar1 : mbar0;
        int &ph = (w & 1) ? ph1 : ph0;
        if (s.lane == 0) {
            mma_fence();
            MMA_4SUB(0, da, db, w > 0 ? 1 : 0);
        }
        #pragma unroll
        for (int i = 0; i < N_STS; i++)
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                         :: "r"(sts_addr), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
        if (s.lane == 0) mma_commit(mbar);
        mb_wait(mbar, ph); ph ^= 1;
    }

    /* Measure */
    long long tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        uint32_t mbar = (rep & 1) ? mbar1 : mbar0;
        int &ph = (rep & 1) ? ph1 : ph0;
        long long t0 = clk();
        if (s.lane == 0) {
            mma_fence();
            MMA_4SUB(0, da, db, 1);
        }
        #pragma unroll
        for (int i = 0; i < N_STS; i++)
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                         :: "r"(sts_addr), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
        if (s.lane == 0) mma_commit(mbar);
        mb_wait(mbar, ph); ph ^= 1;
        long long t1 = clk();
        if (s.lane == 0) tot += t1 - t0;
    }

    if (s.lane == 0) out[0] = tot;
    bench_fini(s);
}

template <int N_BF16>
__global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void k_mma_shadow_bf16(__grid_constant__ const CUtensorMap tma_a,
                       __grid_constant__ const CUtensorMap tma_b,
                       long long *out) {
    Setup s;
    if (!bench_init(s, tma_a, tma_b)) { bench_fini(s); return; }

    uint64_t da = s.da[0], db = s.db[0];
    uint32_t mbar0 = s.sb + MB_MMA(0), mbar1 = s.sb + MB_MMA(1);
    uint32_t bf = 0x3c003c00u, v = 0x3c003c00u;

    int ph0 = 0, ph1 = 0;
    for (int w = 0; w < WARMUP; w++) {
        uint32_t mbar = (w & 1) ? mbar1 : mbar0;
        int &ph = (w & 1) ? ph1 : ph0;
        if (s.lane == 0) {
            mma_fence();
            MMA_4SUB(0, da, db, w > 0 ? 1 : 0);
        }
        #pragma unroll
        for (int i = 0; i < N_BF16; i++)
            asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(bf) : "r"(v));
        if (s.lane == 0) mma_commit(mbar);
        mb_wait(mbar, ph); ph ^= 1;
    }

    long long tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        uint32_t mbar = (rep & 1) ? mbar1 : mbar0;
        int &ph = (rep & 1) ? ph1 : ph0;
        long long t0 = clk();
        if (s.lane == 0) {
            mma_fence();
            MMA_4SUB(0, da, db, 1);
        }
        #pragma unroll
        for (int i = 0; i < N_BF16; i++)
            asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(bf) : "r"(v));
        if (s.lane == 0) mma_commit(mbar);
        mb_wait(mbar, ph); ph ^= 1;
        long long t1 = clk();
        if (s.lane == 0) tot += t1 - t0;
    }

    if (s.lane == 0) out[0] = tot;
    if (bf == 0xDEAD) out[1] = bf;   /* prevent DCE */
    bench_fini(s);
}

/* ═══════ F. MMA + TMA overlap ═══════ */

__global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void k_mma_tma_overlap(__grid_constant__ const CUtensorMap tma_a,
                       __grid_constant__ const CUtensorMap tma_b,
                       long long *out) {
    Setup s;
    if (!bench_init(s, tma_a, tma_b, 2)) { bench_fini(s); return; }

    uint64_t da = s.da[0], db = s.db[0];
    uint32_t mbar_mma0 = s.sb + MB_MMA(0), mbar_mma1 = s.sb + MB_MMA(1);
    uint32_t mbar_tma = s.sb + MB_TMA(1);  /* reuse stage 1 TMA mbar */

    /* Measure: MMA on stage 0 data while TMA reloads stage 1 */
    int mph0 = 0, mph1 = 0, tph = 1;
    for (int w = 0; w < WARMUP; w++) {
        uint32_t mbar = (w & 1) ? mbar_mma1 : mbar_mma0;
        int &ph = (w & 1) ? mph1 : mph0;
        if (s.lane == 0) {
            /* Start TMA reload into stage 1 */
            mb_expect_tx(mbar_tma, A_SZ + B_SZ);
            tma_ld(s.sb + STAGE_A(1), &tma_a,
                   0, s.cta_rank * BENCH_TM, mbar_tma);
            tma_ld(s.sb + STAGE_B(1), &tma_b,
                   0, s.cta_rank * (BENCH_TN / 2), mbar_tma);
            /* MMA on stage 0 */
            mma_fence();
            MMA_4SUB(0, da, db, w > 0 ? 1 : 0);
            mma_commit(mbar);
        }
        mb_wait(mbar_tma, tph); tph ^= 1;
        mb_wait(mbar, ph); ph ^= 1;
    }

    /* Baseline: MMA only (no TMA) */
    long long t_mma_only = 0;
    for (int rep = 0; rep < REPS; rep++) {
        uint32_t mbar = (rep & 1) ? mbar_mma1 : mbar_mma0;
        int &ph = (rep & 1) ? mph1 : mph0;
        long long t0 = clk();
        if (s.lane == 0) {
            mma_fence();
            MMA_4SUB(0, da, db, 1);
            mma_commit(mbar);
        }
        mb_wait(mbar, ph); ph ^= 1;
        long long t1 = clk();
        if (s.lane == 0) t_mma_only += t1 - t0;
    }

    /* MMA + TMA concurrent */
    long long t_mma_tma = 0;
    for (int rep = 0; rep < REPS; rep++) {
        uint32_t mbar = (rep & 1) ? mbar_mma1 : mbar_mma0;
        int &ph = (rep & 1) ? mph1 : mph0;
        long long t0 = clk();
        if (s.lane == 0) {
            mb_expect_tx(mbar_tma, A_SZ + B_SZ);
            tma_ld(s.sb + STAGE_A(1), &tma_a,
                   0, s.cta_rank * BENCH_TM, mbar_tma);
            tma_ld(s.sb + STAGE_B(1), &tma_b,
                   0, s.cta_rank * (BENCH_TN / 2), mbar_tma);
            mma_fence();
            MMA_4SUB(0, da, db, 1);
            mma_commit(mbar);
        }
        mb_wait(mbar_tma, tph); tph ^= 1;
        mb_wait(mbar, ph); ph ^= 1;
        long long t1 = clk();
        if (s.lane == 0) t_mma_tma += t1 - t0;
    }

    if (s.lane == 0) {
        out[0] = t_mma_only;
        out[1] = t_mma_tma;
    }
    bench_fini(s);
}

/* ═══════ G. Full K-loop pipeline ═══════ */

template <int K_ITERS>
__global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void k_kloop(__grid_constant__ const CUtensorMap tma_a,
             __grid_constant__ const CUtensorMap tma_b,
             long long *out) {
    Setup s;
    int n_load = K_ITERS < N_STAGES ? K_ITERS : N_STAGES;
    if (!bench_init(s, tma_a, tma_b, n_load)) { bench_fini(s); return; }

    uint32_t mbar[N_STAGES];
    for (int i = 0; i < N_STAGES; i++)
        mbar[i] = s.sb + MB_MMA(i);
    int ph[N_STAGES] = {0};

    /* Warmup: serialized K-loop */
    for (int w = 0; w < 4; w++) {
        for (int ki = 0; ki < K_ITERS; ki++) {
            int st = ki % N_STAGES;
            if (ki >= N_STAGES) { mb_wait(mbar[st], ph[st]); ph[st] ^= 1; }
            if (s.lane == 0) {
                mma_fence();
                MMA_4SUB(0, s.da[st], s.db[st], ki > 0 || w > 0 ? 1 : 0);
                mma_commit(mbar[st]);
            }
        }
        /* Drain */
        for (int i = 0; i < N_STAGES && i < K_ITERS; i++) {
            int st = (K_ITERS - 1 - i) % N_STAGES;
            if (i < K_ITERS) { mb_wait(mbar[st], ph[st]); ph[st] ^= 1; }
        }
    }

    /* Measure: pipelined K-loop */
    long long tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        long long t0 = clk();
        for (int ki = 0; ki < K_ITERS; ki++) {
            int st = ki % N_STAGES;
            if (ki >= N_STAGES) { mb_wait(mbar[st], ph[st]); ph[st] ^= 1; }
            if (s.lane == 0) {
                mma_fence();
                MMA_4SUB(0, s.da[st], s.db[st], 1);
                mma_commit(mbar[st]);
            }
        }
        for (int i = 0; i < N_STAGES && i < K_ITERS; i++) {
            int st = (K_ITERS - 1 - i) % N_STAGES;
            mb_wait(mbar[st], ph[st]); ph[st] ^= 1;
        }
        long long t1 = clk();
        if (s.lane == 0) tot += t1 - t0;
    }

    if (s.lane == 0) out[0] = tot;
    bench_fini(s);
}

/* ═══════ H. TMEM double-buffer — read buf0 while MMA writes buf1 ═══════ */

__global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void k_tmem_dbuf(__grid_constant__ const CUtensorMap tma_a,
                 __grid_constant__ const CUtensorMap tma_b,
                 long long *out) {
    Setup s;
    if (!bench_init(s, tma_a, tma_b)) { bench_fini(s); return; }

    uint64_t da = s.da[0], db = s.db[0];
    uint32_t mbar0 = s.sb + MB_MMA(0), mbar1 = s.sb + MB_MMA(1);
    float r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;

    /* Fill both TMEM buffers */
    if (s.lane == 0) {
        mma_fence();
        MMA_4SUB(0, da, db, 0);               /* buf0 */
        mma_commit(mbar0);
    }
    mb_wait(mbar0, 0);
    if (s.lane == 0) {
        mma_fence();
        MMA_4SUB(BENCH_TN, da, db, 0);        /* buf1 */
        mma_commit(mbar1);
    }
    mb_wait(mbar1, 0);

    /* Warmup */
    int ph0 = 1, ph1 = 1;
    for (int w = 0; w < WARMUP; w++) {
        int wr_buf = w & 1;
        int rd_buf = 1 - wr_buf;
        uint32_t mbar = wr_buf ? mbar1 : mbar0;
        int &ph = wr_buf ? ph1 : ph0;
        if (s.lane == 0) {
            mma_fence();
            MMA_4SUB(wr_buf * BENCH_TN, da, db, 1);
        }
        /* Read from other buffer while MMA writes */
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
            : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3),
              "=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7),
              "=f"(r8),"=f"(r9),"=f"(r10),"=f"(r11),
              "=f"(r12),"=f"(r13),"=f"(r14),"=f"(r15)
            : "r"(rd_buf * BENCH_TN));
        if (s.lane == 0) mma_commit(mbar);
        mb_wait(mbar, ph); ph ^= 1;
    }

    /* Baseline: MMA only, no TMEM read */
    long long t_mma_only = 0;
    for (int rep = 0; rep < REPS; rep++) {
        int wr_buf = rep & 1;
        uint32_t mbar = wr_buf ? mbar1 : mbar0;
        int &ph = wr_buf ? ph1 : ph0;
        long long t0 = clk();
        if (s.lane == 0) {
            mma_fence();
            MMA_4SUB(wr_buf * BENCH_TN, da, db, 1);
            mma_commit(mbar);
        }
        mb_wait(mbar, ph); ph ^= 1;
        long long t1 = clk();
        if (s.lane == 0) t_mma_only += t1 - t0;
    }

    /* MMA write buf X + TMEM read buf Y concurrent */
    long long t_mma_rd = 0;
    for (int rep = 0; rep < REPS; rep++) {
        int wr_buf = rep & 1;
        int rd_buf = 1 - wr_buf;
        uint32_t mbar = wr_buf ? mbar1 : mbar0;
        int &ph = wr_buf ? ph1 : ph0;
        long long t0 = clk();
        if (s.lane == 0) {
            mma_fence();
            MMA_4SUB(wr_buf * BENCH_TN, da, db, 1);
        }
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
            : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3),
              "=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7),
              "=f"(r8),"=f"(r9),"=f"(r10),"=f"(r11),
              "=f"(r12),"=f"(r13),"=f"(r14),"=f"(r15)
            : "r"(rd_buf * BENCH_TN));
        if (s.lane == 0) mma_commit(mbar);
        mb_wait(mbar, ph); ph ^= 1;
        long long t1 = clk();
        if (s.lane == 0) t_mma_rd += t1 - t0;
    }

    if (s.lane == 0) {
        out[0] = t_mma_only;
        out[1] = t_mma_rd;
    }
    bench_fini(s);
}

/* ═══════ I. STS + TMEM_LD overlap ═══════ */
/* After MMA is complete, can STS and tcgen05.ld run concurrently?
   TMEM_LD uses the tensor memory port, STS uses shared memory.
   Tests: TMEM_LD only, STS only, both together. */

template <int N_STS>
__global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void k_sts_tmem_overlap(__grid_constant__ const CUtensorMap tma_a,
                        __grid_constant__ const CUtensorMap tma_b,
                        long long *out) {
    Setup s;
    if (!bench_init(s, tma_a, tma_b)) { bench_fini(s); return; }

    uint64_t da = s.da[0], db = s.db[0];
    uint32_t mbar = s.sb + MB_MMA(0);
    uint32_t sts_addr = s.sb + WORK_OFF + s.lane * 16;
    uint32_t v = 0x3c003c00u | s.lane;
    float r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;

    /* Pre-fill TMEM */
    if (s.lane == 0) { mma_fence(); MMA_4SUB(0, da, db, 0); mma_commit(mbar); }
    mb_wait(mbar, 0);

    /* Warmup all 3 patterns */
    for (int w = 0; w < WARMUP; w++) {
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
            : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3),"=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7),
              "=f"(r8),"=f"(r9),"=f"(r10),"=f"(r11),"=f"(r12),"=f"(r13),"=f"(r14),"=f"(r15)
            : "r"(0));
        #pragma unroll
        for (int i = 0; i < N_STS; i++)
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                         :: "r"(sts_addr), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
    }

    /* 1: TMEM_LD only */
    long long t_ld = 0;
    for (int rep = 0; rep < REPS; rep++) {
        long long t0 = clk();
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
            : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3),"=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7),
              "=f"(r8),"=f"(r9),"=f"(r10),"=f"(r11),"=f"(r12),"=f"(r13),"=f"(r14),"=f"(r15)
            : "r"(0));
        t_ld += clk() - t0;
    }

    /* 2: STS only */
    long long t_sts = 0;
    for (int rep = 0; rep < REPS; rep++) {
        long long t0 = clk();
        #pragma unroll
        for (int i = 0; i < N_STS; i++)
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                         :: "r"(sts_addr), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
        t_sts += clk() - t0;
    }

    /* 3: Both together */
    long long t_both = 0;
    for (int rep = 0; rep < REPS; rep++) {
        long long t0 = clk();
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
            : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3),"=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7),
              "=f"(r8),"=f"(r9),"=f"(r10),"=f"(r11),"=f"(r12),"=f"(r13),"=f"(r14),"=f"(r15)
            : "r"(0));
        #pragma unroll
        for (int i = 0; i < N_STS; i++)
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                         :: "r"(sts_addr), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
        t_both += clk() - t0;
    }

    if (s.lane == 0) { out[0] = t_ld; out[1] = t_sts; out[2] = t_both; }
    bench_fini(s);
}

/* ═══════ J. MMA compute time — commit to mbar_wait ═══════ */
/* Isolates tensor core computation latency: time from commit dispatch
   to mbar_wait returning. MMA dispatch happens before timing starts. */

__global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void k_mma_compute_lat(__grid_constant__ const CUtensorMap tma_a,
                       __grid_constant__ const CUtensorMap tma_b,
                       long long *out) {
    Setup s;
    if (!bench_init(s, tma_a, tma_b)) { bench_fini(s); return; }

    uint64_t da = s.da[0], db = s.db[0];
    uint32_t mbar0 = s.sb + MB_MMA(0), mbar1 = s.sb + MB_MMA(1);
    int ph0 = 0, ph1 = 0;

    for (int w = 0; w < WARMUP; w++) {
        uint32_t mbar = (w & 1) ? mbar1 : mbar0;
        int &ph = (w & 1) ? ph1 : ph0;
        if (s.lane == 0) { mma_fence(); MMA_4SUB(0, da, db, w > 0 ? 1 : 0); mma_commit(mbar); }
        mb_wait(mbar, ph); ph ^= 1;
    }

    long long tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        uint32_t mbar = (rep & 1) ? mbar1 : mbar0;
        int &ph = (rep & 1) ? ph1 : ph0;
        if (s.lane == 0) { mma_fence(); MMA_4SUB(0, da, db, 1); }
        long long t0 = clk();
        if (s.lane == 0) mma_commit(mbar);
        mb_wait(mbar, ph); ph ^= 1;
        long long t1 = clk();
        if (s.lane == 0) tot += t1 - t0;
    }

    if (s.lane == 0) out[0] = tot;
    bench_fini(s);
}

/* ═══════ K. Cross-warp epilogue handoff ═══════ */
/* W1 (MMA warp) issues MMA + commit.
   W2 (epilogue warp) waits on mbar then does TMEM_LD.
   Measures inter-warp handoff latency: commit → mbar_wait → TMEM_LD.
   Separately reports wait time and TMEM_LD time. */

#define MW_THREADS 192  /* 6 warps × 32, matching FC2 */

struct MWSetup {
    uint32_t sb;
    int lane, warp, cta_rank;
    uint64_t da[N_STAGES], db[N_STAGES];
};

__device__ bool mw_init(MWSetup &s,
                        const CUtensorMap &tma_a, const CUtensorMap &tma_b,
                        int n_load = 1) {
    extern __shared__ __align__(128) char smem[];
    s.sb = to_smem(smem);
    s.lane = threadIdx.x % 32;
    s.warp = threadIdx.x / 32;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(s.cta_rank));

    if (s.warp == 0 && s.lane == 0)
        for (int i = 0; i < N_MBAR; i++)
            mb_init(s.sb + MBAR_OFF + i * 8, 1);
    __syncthreads();
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    /* Warp 1 allocates TMEM (collective within warp, both CTAs) */
    if (s.warp == 1)
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(s.sb + TMEM_OFF), "r"(TMEM_COLS));
    __syncthreads();

    for (int i = 0; i < N_STAGES; i++) {
        s.da[i] = make_desc(s.sb + STAGE_A(i));
        s.db[i] = make_desc(s.sb + STAGE_B(i));
    }

    /* Warp 0 fills WORK area for LDS tests */
    if (s.warp == 0) {
        uint32_t a = s.sb + WORK_OFF + s.lane * 16;
        uint32_t v = 0x3c003c00u | s.lane;
        for (int i = 0; i < WORK_SZ / 512; i++)
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                :: "r"(a + i * 512), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
    }

    for (int st = 0; st < n_load && st < N_STAGES; st++) {
        if (s.warp == 0 && s.lane == 0) {
            uint32_t mb = s.sb + MB_TMA(st);
            mb_expect_tx(mb, A_SZ + B_SZ);
            tma_ld(s.sb + STAGE_A(st), &tma_a, st * BENCH_TK, s.cta_rank * BENCH_TM, mb);
            tma_ld(s.sb + STAGE_B(st), &tma_b, st * BENCH_TK, s.cta_rank * (BENCH_TN / 2), mb);
        }
        __syncthreads();
        if (s.warp == 0) mb_wait(s.sb + MB_TMA(st), 0);
        __syncthreads();
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    return s.cta_rank == 0;
}

__device__ void mw_fini(MWSetup &s) {
    /* Drain MMA pipeline (warp 1 lane 0 is the only MMA issuer) */
    if (s.warp == 1 && s.lane == 0 && s.cta_rank == 0) {
        mma_commit(s.sb + MB_DONE);
        mb_wait(s.sb + MB_DONE, 0);
    }
    __syncthreads();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    if (s.warp == 1)
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
            :: "r"(0), "r"(TMEM_COLS));
}

__global__ __launch_bounds__(96, 1) __cluster_dims__(2, 1, 1)
void k_xwarp_handoff(__grid_constant__ const CUtensorMap tma_a,
                     __grid_constant__ const CUtensorMap tma_b,
                     long long *out) {
    MWSetup s;
    if (!mw_init(s, tma_a, tma_b)) { mw_fini(s); return; }

    uint64_t da = s.da[0], db = s.db[0];
    uint32_t mbar0 = s.sb + MB_MMA(0), mbar1 = s.sb + MB_MMA(1);

    if (s.warp == 1) {
        /* MMA producer: fence → MMA×4 → commit → wait (serialized to
           prevent mbar phase overrun — W1 dispatches ~3x faster than
           W2 consumes, so without backpressure W1 cycles the mbar past
           W2's expected phase → deadlock) */
        int ph0 = 0, ph1 = 0;
        for (int rep = 0; rep < WARMUP + REPS; rep++) {
            uint32_t mbar = (rep & 1) ? mbar1 : mbar0;
            int &ph = (rep & 1) ? ph1 : ph0;
            if (s.lane == 0) {
                mma_fence();
                MMA_4SUB(0, da, db, rep > 0 ? 1 : 0);
                mma_commit(mbar);
            }
            mb_wait(mbar, ph); ph ^= 1;
        }
    } else if (s.warp == 2) {
        /* Epilogue consumer: mbar_wait → TMEM_LD, timed */
        float r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;
        int ph0 = 0, ph1 = 0;
        for (int w = 0; w < WARMUP; w++) {
            uint32_t mbar = (w & 1) ? mbar1 : mbar0;
            int &ph = (w & 1) ? ph1 : ph0;
            mb_wait(mbar, ph); ph ^= 1;
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3),"=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7),
                  "=f"(r8),"=f"(r9),"=f"(r10),"=f"(r11),"=f"(r12),"=f"(r13),"=f"(r14),"=f"(r15)
                : "r"(0));
        }
        long long tot_wait = 0, tot_ld = 0;
        for (int rep = 0; rep < REPS; rep++) {
            uint32_t mbar = (rep & 1) ? mbar1 : mbar0;
            int &ph = (rep & 1) ? ph1 : ph0;
            long long t0 = clk();
            mb_wait(mbar, ph); ph ^= 1;
            long long t1 = clk();
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3),"=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7),
                  "=f"(r8),"=f"(r9),"=f"(r10),"=f"(r11),"=f"(r12),"=f"(r13),"=f"(r14),"=f"(r15)
                : "r"(0));
            long long t2 = clk();
            if (s.lane == 0) { tot_wait += t1 - t0; tot_ld += t2 - t1; }
        }
        if (s.lane == 0) { out[0] = tot_wait; out[1] = tot_ld; }
    }
    /* W0 idles */
    mw_fini(s);
}

/* ═══════ L. Multi-warp contention ═══════ */
/* W1 does MMA. W2-W5 do continuous STS/LDS/BF16 work.
   Measures W1's K-iteration time under varying epilogue pressure.
   EPI_STS=0: no epilogue work (baseline). EPI_STS>0: N STS per iteration per warp. */

template <int EPI_STS>
__global__ __launch_bounds__(MW_THREADS, 1) __cluster_dims__(2, 1, 1)
void k_mw_sts(__grid_constant__ const CUtensorMap tma_a,
              __grid_constant__ const CUtensorMap tma_b,
              long long *out) {
    MWSetup s;
    if (!mw_init(s, tma_a, tma_b)) { mw_fini(s); return; }

    uint64_t da = s.da[0], db = s.db[0];
    uint32_t mbar0 = s.sb + MB_MMA(0), mbar1 = s.sb + MB_MMA(1);

    if (s.warp == 1) {
        /* MMA warp: timed K-iterations */
        int ph0 = 0, ph1 = 0;
        for (int w = 0; w < WARMUP; w++) {
            uint32_t mbar = (w & 1) ? mbar1 : mbar0;
            int &ph = (w & 1) ? ph1 : ph0;
            if (s.lane == 0) { mma_fence(); MMA_4SUB(0, da, db, w > 0 ? 1 : 0); mma_commit(mbar); }
            mb_wait(mbar, ph); ph ^= 1;
        }
        long long tot = 0;
        for (int rep = 0; rep < REPS; rep++) {
            uint32_t mbar = (rep & 1) ? mbar1 : mbar0;
            int &ph = (rep & 1) ? ph1 : ph0;
            long long t0 = clk();
            if (s.lane == 0) { mma_fence(); MMA_4SUB(0, da, db, 1); mma_commit(mbar); }
            mb_wait(mbar, ph); ph ^= 1;
            long long t1 = clk();
            if (s.lane == 0) tot += t1 - t0;
        }
        if (s.lane == 0) out[0] = tot;
    } else if (s.warp >= 2 && EPI_STS > 0) {
        /* Epilogue warps: continuous STS to create dispatch contention */
        uint32_t addr = s.sb + WORK_OFF + (s.warp - 2) * 1024 + s.lane * 16;
        uint32_t v = 0x3c003c00u | s.lane;
        #pragma unroll 1
        for (int i = 0; i < (WARMUP + REPS) * EPI_STS; i++)
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                         :: "r"(addr), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
    }
    /* W0 idles */
    mw_fini(s);
}

/* Mixed epilogue: LDS + BF16 + STS per iteration (realistic FC2 pattern) */
__global__ __launch_bounds__(MW_THREADS, 1) __cluster_dims__(2, 1, 1)
void k_mw_mixed(__grid_constant__ const CUtensorMap tma_a,
                __grid_constant__ const CUtensorMap tma_b,
                long long *out) {
    MWSetup s;
    if (!mw_init(s, tma_a, tma_b)) { mw_fini(s); return; }

    uint64_t da = s.da[0], db = s.db[0];
    uint32_t mbar0 = s.sb + MB_MMA(0), mbar1 = s.sb + MB_MMA(1);

    if (s.warp == 1) {
        int ph0 = 0, ph1 = 0;
        for (int w = 0; w < WARMUP; w++) {
            uint32_t mbar = (w & 1) ? mbar1 : mbar0;
            int &ph = (w & 1) ? ph1 : ph0;
            if (s.lane == 0) { mma_fence(); MMA_4SUB(0, da, db, w > 0 ? 1 : 0); mma_commit(mbar); }
            mb_wait(mbar, ph); ph ^= 1;
        }
        long long tot = 0;
        for (int rep = 0; rep < REPS; rep++) {
            uint32_t mbar = (rep & 1) ? mbar1 : mbar0;
            int &ph = (rep & 1) ? ph1 : ph0;
            long long t0 = clk();
            if (s.lane == 0) { mma_fence(); MMA_4SUB(0, da, db, 1); mma_commit(mbar); }
            mb_wait(mbar, ph); ph ^= 1;
            long long t1 = clk();
            if (s.lane == 0) tot += t1 - t0;
        }
        if (s.lane == 0) out[0] = tot;
    } else if (s.warp >= 2) {
        /* Realistic epilogue: 4 LDS + 4 BF16 HADD2 + 4 STS per iteration */
        uint32_t lds_addr = s.sb + WORK_OFF + (s.warp - 2) * 1024 + s.lane * 16;
        uint32_t sts_addr = s.sb + WORK_OFF + 8192 + (s.warp - 2) * 1024 + s.lane * 16;
        uint32_t r0, r1, r2, r3;
        uint32_t bf = 0x3c003c00u, v = 0x3c003c00u;
        #pragma unroll 1
        for (int i = 0; i < (WARMUP + REPS) * 4; i++) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(lds_addr));
            asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(bf) : "r"(v));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                         :: "r"(sts_addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
        }
        if (bf == 0xDEAD) out[1] = bf;
    }
    mw_fini(s);
}

/* ═══════ N. Full epilogue pipeline during MMA ═══════ */
/* W1 does MMA. W2-W5 run realistic FC2 epilogue:
   TMEM_LD → F2FP (f32→bf16) → HADD2 (bias add) → STS.128 → fence.proxy.async.
   Tests whether the FULL epilogue sequence contends (not just STS like section L). */

__global__ __launch_bounds__(MW_THREADS, 1) __cluster_dims__(2, 1, 1)
void k_mw_full_epi(__grid_constant__ const CUtensorMap tma_a,
                   __grid_constant__ const CUtensorMap tma_b,
                   long long *out) {
    MWSetup s;
    if (!mw_init(s, tma_a, tma_b)) { mw_fini(s); return; }

    uint64_t da = s.da[0], db = s.db[0];
    uint32_t mbar0 = s.sb + MB_MMA(0), mbar1 = s.sb + MB_MMA(1);

    if (s.warp == 1) {
        int ph0 = 0, ph1 = 0;
        for (int w = 0; w < WARMUP; w++) {
            uint32_t mbar = (w & 1) ? mbar1 : mbar0;
            int &ph = (w & 1) ? ph1 : ph0;
            if (s.lane == 0) { mma_fence(); MMA_4SUB(0, da, db, w > 0 ? 1 : 0); mma_commit(mbar); }
            mb_wait(mbar, ph); ph ^= 1;
        }
        long long tot = 0;
        for (int rep = 0; rep < REPS; rep++) {
            uint32_t mbar = (rep & 1) ? mbar1 : mbar0;
            int &ph = (rep & 1) ? ph1 : ph0;
            long long t0 = clk();
            if (s.lane == 0) { mma_fence(); MMA_4SUB(0, da, db, 1); mma_commit(mbar); }
            mb_wait(mbar, ph); ph ^= 1;
            long long t1 = clk();
            if (s.lane == 0) tot += t1 - t0;
        }
        if (s.lane == 0) out[0] = tot;
    } else if (s.warp >= 2) {
        float r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;
        uint32_t sts_addr = s.sb + WORK_OFF + (s.warp - 2) * 1024 + s.lane * 16;
        uint32_t bf = 0x3c003c00u;
        #pragma unroll 1
        for (int i = 0; i < (WARMUP + REPS) * 4; i++) {
            /* TMEM_LD: read 32 cols of FP32 accumulators */
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3),"=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7),
                  "=f"(r8),"=f"(r9),"=f"(r10),"=f"(r11),"=f"(r12),"=f"(r13),"=f"(r14),"=f"(r15)
                : "r"(0));
            /* F2FP: convert 8 f32 pairs → 4 bf16x2 */
            uint32_t b0, b1, b2, b3;
            asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b0) : "f"(r0), "f"(r1));
            asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b1) : "f"(r2), "f"(r3));
            asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b2) : "f"(r4), "f"(r5));
            asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b3) : "f"(r6), "f"(r7));
            /* HADD2: add bias */
            asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(b0) : "r"(bf));
            asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(b1) : "r"(bf));
            asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(b2) : "r"(bf));
            asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(b3) : "r"(bf));
            /* STS.128 */
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                         :: "r"(sts_addr), "r"(b0), "r"(b1), "r"(b2), "r"(b3) : "memory");
            /* fence.proxy.async (required before TMA store in FC2) */
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        if (bf == 0xDEAD) out[1] = bf;
    }
    mw_fini(s);
}

/* ═══════ O. Combined MMA shadow budget ═══════ */
/* Full epilogue work inside a single warp's MMA shadow window:
   TMEM_LD → F2FP → HADD2 → STS → fence.proxy.async, all between dispatch and commit.
   Section E tested STS and BF16 separately. This tests the realistic combined workload. */

template <int N_CHUNKS>
__global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void k_mma_shadow_combined(__grid_constant__ const CUtensorMap tma_a,
                           __grid_constant__ const CUtensorMap tma_b,
                           long long *out) {
    Setup s;
    if (!bench_init(s, tma_a, tma_b)) { bench_fini(s); return; }

    uint64_t da = s.da[0], db = s.db[0];
    uint32_t mbar0 = s.sb + MB_MMA(0), mbar1 = s.sb + MB_MMA(1);
    uint32_t sts_addr = s.sb + WORK_OFF + s.lane * 16;
    uint32_t bf = 0x3c003c00u;
    float r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;

    /* Pre-fill TMEM */
    if (s.lane == 0) { mma_fence(); MMA_4SUB(0, da, db, 0); mma_commit(mbar0); }
    mb_wait(mbar0, 0);

    /* Warmup */
    int ph0 = 1, ph1 = 0;
    for (int w = 0; w < WARMUP; w++) {
        uint32_t mbar = (w & 1) ? mbar1 : mbar0;
        int &ph = (w & 1) ? ph1 : ph0;
        if (s.lane == 0) {
            mma_fence();
            MMA_4SUB(0, da, db, 1);
        }
        /* Epilogue work in MMA shadow */
        for (int c = 0; c < N_CHUNKS; c++) {
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3),"=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7),
                  "=f"(r8),"=f"(r9),"=f"(r10),"=f"(r11),"=f"(r12),"=f"(r13),"=f"(r14),"=f"(r15)
                : "r"(0));
            uint32_t b0, b1, b2, b3;
            asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b0) : "f"(r0), "f"(r1));
            asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b1) : "f"(r2), "f"(r3));
            asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b2) : "f"(r4), "f"(r5));
            asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b3) : "f"(r6), "f"(r7));
            asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(b0) : "r"(bf));
            asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(b1) : "r"(bf));
            asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(b2) : "r"(bf));
            asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(b3) : "r"(bf));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                         :: "r"(sts_addr), "r"(b0), "r"(b1), "r"(b2), "r"(b3) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        if (s.lane == 0) mma_commit(mbar);
        mb_wait(mbar, ph); ph ^= 1;
    }

    /* Measure */
    long long tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        uint32_t mbar = (rep & 1) ? mbar1 : mbar0;
        int &ph = (rep & 1) ? ph1 : ph0;
        long long t0 = clk();
        if (s.lane == 0) {
            mma_fence();
            MMA_4SUB(0, da, db, 1);
        }
        for (int c = 0; c < N_CHUNKS; c++) {
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3),"=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7),
                  "=f"(r8),"=f"(r9),"=f"(r10),"=f"(r11),"=f"(r12),"=f"(r13),"=f"(r14),"=f"(r15)
                : "r"(0));
            uint32_t b0, b1, b2, b3;
            asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b0) : "f"(r0), "f"(r1));
            asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b1) : "f"(r2), "f"(r3));
            asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b2) : "f"(r4), "f"(r5));
            asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b3) : "f"(r6), "f"(r7));
            asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(b0) : "r"(bf));
            asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(b1) : "r"(bf));
            asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(b2) : "r"(bf));
            asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(b3) : "r"(bf));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                         :: "r"(sts_addr), "r"(b0), "r"(b1), "r"(b2), "r"(b3) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        if (s.lane == 0) mma_commit(mbar);
        mb_wait(mbar, ph); ph ^= 1;
        long long t1 = clk();
        if (s.lane == 0) tot += t1 - t0;
    }
    if (s.lane == 0) out[0] = tot;
    if (bf == 0xDEAD) out[2] = bf;
    bench_fini(s);
}

/* ═══════ P. W0 TMA dispatch during MMA ═══════ */
/* W0 continuously issues TMA loads. W1 does MMA (timed).
   Tests whether TMA dispatch from a separate warp contends with MMA. */

__global__ __launch_bounds__(MW_THREADS, 1) __cluster_dims__(2, 1, 1)
void k_mw_tma_mma(__grid_constant__ const CUtensorMap tma_a,
                   __grid_constant__ const CUtensorMap tma_b,
                   long long *out) {
    MWSetup s;
    if (!mw_init(s, tma_a, tma_b)) { mw_fini(s); return; }

    uint64_t da = s.da[0], db = s.db[0];
    uint32_t mma_mbar0 = s.sb + MB_MMA(0), mma_mbar1 = s.sb + MB_MMA(1);
    /* W0 uses TMA mbar slots 2,3 (0 was used by mw_init) */
    uint32_t tma_m0 = s.sb + MB_TMA(2), tma_m1 = s.sb + MB_TMA(3);

    if (s.warp == 1) {
        /* MMA warp: timed K-iterations */
        int ph0 = 0, ph1 = 0;
        for (int w = 0; w < WARMUP; w++) {
            uint32_t mbar = (w & 1) ? mma_mbar1 : mma_mbar0;
            int &ph = (w & 1) ? ph1 : ph0;
            if (s.lane == 0) { mma_fence(); MMA_4SUB(0, da, db, w > 0 ? 1 : 0); mma_commit(mbar); }
            mb_wait(mbar, ph); ph ^= 1;
        }
        long long tot = 0;
        for (int rep = 0; rep < REPS; rep++) {
            uint32_t mbar = (rep & 1) ? mma_mbar1 : mma_mbar0;
            int &ph = (rep & 1) ? ph1 : ph0;
            long long t0 = clk();
            if (s.lane == 0) { mma_fence(); MMA_4SUB(0, da, db, 1); mma_commit(mbar); }
            mb_wait(mbar, ph); ph ^= 1;
            long long t1 = clk();
            if (s.lane == 0) tot += t1 - t0;
        }
        if (s.lane == 0) out[0] = tot;
    } else if (s.warp == 0 && s.lane == 0) {
        /* TMA warp: continuous TMA loads with backpressure */
        int tph0 = 0, tph1 = 0;
        for (int i = 0; i < (WARMUP + REPS) * 2; i++) {
            uint32_t tmb = (i & 1) ? tma_m1 : tma_m0;
            int &tph = (i & 1) ? tph1 : tph0;
            mb_expect_tx(tmb, A_SZ + B_SZ);
            tma_ld(s.sb + STAGE_A(2), &tma_a,
                   0, s.cta_rank * BENCH_TM, tmb);
            tma_ld(s.sb + STAGE_B(2), &tma_b,
                   0, s.cta_rank * (BENCH_TN / 2), tmb);
            mb_wait(tmb, tph); tph ^= 1;
        }
    }
    mw_fini(s);
}

/* ═══════ Q. fence.proxy.async + mbar_arrive during MMA ═══════ */
/* W2-W5 do continuous fence.proxy.async + mbar_arrive.
   Isolates whether these specific operations contend with MMA
   (section L tested STS/LDS but not fence/mbar). */

template <int N_FENCE>
__global__ __launch_bounds__(MW_THREADS, 1) __cluster_dims__(2, 1, 1)
void k_mw_fence(__grid_constant__ const CUtensorMap tma_a,
                __grid_constant__ const CUtensorMap tma_b,
                long long *out) {
    MWSetup s;
    if (!mw_init(s, tma_a, tma_b)) { mw_fini(s); return; }

    uint64_t da = s.da[0], db = s.db[0];
    uint32_t mbar0 = s.sb + MB_MMA(0), mbar1 = s.sb + MB_MMA(1);
    /* Epilogue warps use their own mbar for arrive testing */
    uint32_t epi_mbar = s.sb + MBAR_OFF + (12 + s.warp) * 8;

    if (s.warp == 1) {
        int ph0 = 0, ph1 = 0;
        for (int w = 0; w < WARMUP; w++) {
            uint32_t mbar = (w & 1) ? mbar1 : mbar0;
            int &ph = (w & 1) ? ph1 : ph0;
            if (s.lane == 0) { mma_fence(); MMA_4SUB(0, da, db, w > 0 ? 1 : 0); mma_commit(mbar); }
            mb_wait(mbar, ph); ph ^= 1;
        }
        long long tot = 0;
        for (int rep = 0; rep < REPS; rep++) {
            uint32_t mbar = (rep & 1) ? mbar1 : mbar0;
            int &ph = (rep & 1) ? ph1 : ph0;
            long long t0 = clk();
            if (s.lane == 0) { mma_fence(); MMA_4SUB(0, da, db, 1); mma_commit(mbar); }
            mb_wait(mbar, ph); ph ^= 1;
            long long t1 = clk();
            if (s.lane == 0) tot += t1 - t0;
        }
        if (s.lane == 0) out[0] = tot;
    } else if (s.warp >= 2 && N_FENCE > 0) {
        #pragma unroll 1
        for (int i = 0; i < (WARMUP + REPS) * N_FENCE; i++) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (s.lane == 0)
                mb_arrive(epi_mbar);
        }
    }
    mw_fini(s);
}

/* ═══════ R. Pipelined K-loop with interleaved TMA ═══════ */
/* W0 does TMA loads, W1 does MMA — the real FC2 K-loop pattern.
   Pipeline: W0 fills stage s+N_STAGES via TMA while W1 processes stage s via MMA.
   TMA mbars signal data ready (W0→W1), MMA mbars signal stage free (W1→W0).
   Measures the true K-loop floor with realistic TMA interleaving. */

#define KLOOP_ITERS 24  /* FC2: K=3072, TK=128 → 24 iterations */

__global__ __launch_bounds__(MW_THREADS, 1) __cluster_dims__(2, 1, 1)
void k_mw_real_kloop(__grid_constant__ const CUtensorMap tma_a,
                     __grid_constant__ const CUtensorMap tma_b,
                     long long *out) {
    MWSetup s;
    if (!mw_init(s, tma_a, tma_b, N_STAGES)) { mw_fini(s); return; }

    uint32_t mma_mbar[N_STAGES], tma_mbar[N_STAGES];
    int mma_ph[N_STAGES], tma_ph[N_STAGES];
    for (int i = 0; i < N_STAGES; i++) {
        mma_mbar[i] = s.sb + MB_MMA(i);
        tma_mbar[i] = s.sb + MB_TMA(i);
        mma_ph[i] = 0;
        tma_ph[i] = 1; /* mw_init loaded all N_STAGES, so phase 0 done */
    }

    if (s.warp == 1) {
        /* W1: MMA warp — process stages in pipeline order */

        /* Warmup (4 full K-loops) */
        for (int w = 0; w < 4; w++) {
            for (int ki = 0; ki < KLOOP_ITERS; ki++) {
                int st = ki % N_STAGES;
                /* Wait for TMA data (first pass of first warmup: already loaded) */
                if (ki >= N_STAGES || w > 0) {
                    mb_wait(tma_mbar[st], tma_ph[st]); tma_ph[st] ^= 1;
                }
                /* Wait for previous MMA on this slot.
                   Always skip for ki < N_STAGES — the drain (or initial fill)
                   already confirmed these stages are free. Using w > 0 here
                   would add an extra wait per stage per K-loop (drain + ki=0),
                   consuming a phase flip that hasn't been produced yet. */
                if (ki >= N_STAGES) {
                    mb_wait(mma_mbar[st], mma_ph[st]); mma_ph[st] ^= 1;
                }
                if (s.lane == 0) {
                    mma_fence();
                    MMA_4SUB(0, s.da[st], s.db[st], ki > 0 || w > 0 ? 1 : 0);
                    mma_commit(mma_mbar[st]);
                }
            }
            /* Drain: wait for last N_STAGES commits */
            for (int i = 0; i < N_STAGES && i < KLOOP_ITERS; i++) {
                int st = (KLOOP_ITERS - N_STAGES + i) % N_STAGES;
                mb_wait(mma_mbar[st], mma_ph[st]); mma_ph[st] ^= 1;
            }
        }

        /* Measure */
        long long tot = 0;
        for (int rep = 0; rep < REPS; rep++) {
            long long t0 = clk();
            for (int ki = 0; ki < KLOOP_ITERS; ki++) {
                int st = ki % N_STAGES;
                mb_wait(tma_mbar[st], tma_ph[st]); tma_ph[st] ^= 1;
                if (ki >= N_STAGES) {
                    mb_wait(mma_mbar[st], mma_ph[st]); mma_ph[st] ^= 1;
                }
                if (s.lane == 0) {
                    mma_fence();
                    MMA_4SUB(0, s.da[st], s.db[st], 1);
                    mma_commit(mma_mbar[st]);
                }
            }
            for (int i = 0; i < N_STAGES && i < KLOOP_ITERS; i++) {
                int st = (KLOOP_ITERS - N_STAGES + i) % N_STAGES;
                mb_wait(mma_mbar[st], mma_ph[st]); mma_ph[st] ^= 1;
            }
            long long t1 = clk();
            if (s.lane == 0) tot += t1 - t0;
        }
        if (s.lane == 0) out[0] = tot;

    } else if (s.warp == 0) {
        /* W0: TMA warp — reload stages for W1's pipeline.
           For each K-iter (beyond initial fill), wait for MMA to free the stage,
           then issue TMA loads. Coordinates wrap around the small tensor. */
        for (int rep = 0; rep < 4 + REPS; rep++) {
            for (int ki = 0; ki < KLOOP_ITERS; ki++) {
                int st = ki % N_STAGES;
                /* Wait for MMA to finish with this stage before overwriting */
                if (ki >= N_STAGES || rep > 0) {
                    mb_wait(mma_mbar[st], mma_ph[st]); mma_ph[st] ^= 1;
                }
                /* Issue TMA loads (only lane 0) */
                if (s.lane == 0) {
                    int coord_k = (ki % (GA_K / BENCH_TK)) * BENCH_TK;
                    mb_expect_tx(tma_mbar[st], A_SZ + B_SZ);
                    tma_ld(s.sb + STAGE_A(st), &tma_a,
                           coord_k, s.cta_rank * BENCH_TM, tma_mbar[st]);
                    tma_ld(s.sb + STAGE_B(st), &tma_b,
                           coord_k, s.cta_rank * (BENCH_TN / 2), tma_mbar[st]);
                }
            }
        }
    }
    /* W2-W5 idle */
    mw_fini(s);
}

/* ═══════ S. TMA store during MMA ═══════ */
/* W2-W5 do TMA stores (cp.async.bulk.tensor.2d) while W1 does MMA.
   Tests whether TMA store contends with MMA compute.
   Uses a BF16 output tensor map passed as third kernel argument. */

__device__ __forceinline__
void tma_st(const void *tm, int x, int y, uint32_t smem_src) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
        :: "l"(tm), "r"(x), "r"(y), "r"(smem_src) : "memory");
}

template <int N_STORE>
__global__ __launch_bounds__(MW_THREADS, 1) __cluster_dims__(2, 1, 1)
void k_mw_tma_store(__grid_constant__ const CUtensorMap tma_a,
                    __grid_constant__ const CUtensorMap tma_b,
                    __grid_constant__ const CUtensorMap tma_c,
                    long long *out) {
    MWSetup s;
    if (!mw_init(s, tma_a, tma_b)) { mw_fini(s); return; }

    uint64_t da = s.da[0], db = s.db[0];
    uint32_t mbar0 = s.sb + MB_MMA(0), mbar1 = s.sb + MB_MMA(1);

    /* Pre-fill WORK area with valid BF16 data for TMA stores */
    if (s.warp == 0) {
        uint32_t v = 0x3c003c00u;
        for (int i = s.lane; i < WORK_SZ / 4; i += 32)
            asm volatile("st.shared.b32 [%0], %1;"
                :: "r"(s.sb + WORK_OFF + i * 4), "r"(v) : "memory");
    }
    __syncthreads();

    if (s.warp == 1) {
        int ph0 = 0, ph1 = 0;
        for (int w = 0; w < WARMUP; w++) {
            uint32_t mbar = (w & 1) ? mbar1 : mbar0;
            int &ph = (w & 1) ? ph1 : ph0;
            if (s.lane == 0) { mma_fence(); MMA_4SUB(0, da, db, w > 0 ? 1 : 0); mma_commit(mbar); }
            mb_wait(mbar, ph); ph ^= 1;
        }
        long long tot = 0;
        for (int rep = 0; rep < REPS; rep++) {
            uint32_t mbar = (rep & 1) ? mbar1 : mbar0;
            int &ph = (rep & 1) ? ph1 : ph0;
            long long t0 = clk();
            if (s.lane == 0) { mma_fence(); MMA_4SUB(0, da, db, 1); mma_commit(mbar); }
            mb_wait(mbar, ph); ph ^= 1;
            long long t1 = clk();
            if (s.lane == 0) tot += t1 - t0;
        }
        if (s.lane == 0) out[0] = tot;
    } else if (s.warp >= 2 && N_STORE > 0) {
        /* TMA store from WORK area to global via tma_c.
           Each store: fence.proxy.async → TMA store → commit_group → wait_group.
           box=[64,32] BF16 = 4096 bytes per store. */
        int col = (s.warp - 2) * 64;    /* each warp stores different 64-col region */
        int row = s.cta_rank * BENCH_TM; /* row offset for this CTA */
        uint32_t src = s.sb + WORK_OFF + (s.warp - 2) * 4096;
        #pragma unroll 1
        for (int i = 0; i < (WARMUP + REPS) * N_STORE; i++) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (s.lane == 0) {
                tma_st(&tma_c, col, row, src);
                asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
            }
        }
    }
    mw_fini(s);
}

/* ═══════ M. Multi-SM scaling ═══════ */
/* Launch on all 74 clusters (148 SMs). Each measures K-iteration throughput.
   Stresses memory subsystem and reveals system-level contention. */

#define MAX_CLUSTERS 74

__global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void k_multi_sm(__grid_constant__ const CUtensorMap tma_a,
                __grid_constant__ const CUtensorMap tma_b,
                long long *out) {
    Setup s;
    if (!bench_init(s, tma_a, tma_b)) { bench_fini(s); return; }

    uint64_t da = s.da[0], db = s.db[0];
    uint32_t mbar0 = s.sb + MB_MMA(0), mbar1 = s.sb + MB_MMA(1);
    int cluster_id = blockIdx.x / 2;

    int ph0 = 0, ph1 = 0;
    for (int w = 0; w < WARMUP; w++) {
        uint32_t mbar = (w & 1) ? mbar1 : mbar0;
        int &ph = (w & 1) ? ph1 : ph0;
        if (s.lane == 0) { mma_fence(); MMA_4SUB(0, da, db, w > 0 ? 1 : 0); mma_commit(mbar); }
        mb_wait(mbar, ph); ph ^= 1;
    }

    long long tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        uint32_t mbar = (rep & 1) ? mbar1 : mbar0;
        int &ph = (rep & 1) ? ph1 : ph0;
        long long t0 = clk();
        if (s.lane == 0) { mma_fence(); MMA_4SUB(0, da, db, 1); mma_commit(mbar); }
        mb_wait(mbar, ph); ph ^= 1;
        long long t1 = clk();
        if (s.lane == 0) tot += t1 - t0;
    }

    if (s.lane == 0) out[cluster_id] = tot;
    bench_fini(s);
}

/* ═══════ Host ═══════ */

int main() {
    CUDA_CHECK(cudaSetDevice(0));

    /* Global data */
    uint8_t *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)GA_K * GA_M));
    CUDA_CHECK(cudaMalloc(&d_B, (size_t)GB_K * GB_N));
    CUDA_CHECK(cudaMemset(d_A, 0x3C, (size_t)GA_K * GA_M));   /* FP8 E4M3 = 1.5 */
    CUDA_CHECK(cudaMemset(d_B, 0x3C, (size_t)GB_K * GB_N));

    /* BF16 output for TMA store tests (section S) */
    /* [N_DIM, M_DIM] = [256, 256] BF16 — enough for one tile */
    #define GC_N 256
    #define GC_M 256
    __nv_bfloat16 *d_C;
    CUDA_CHECK(cudaMalloc(&d_C, (size_t)GC_N * GC_M * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(d_C, 0, (size_t)GC_N * GC_M * sizeof(__nv_bfloat16)));

    /* Tensor maps */
    CUtensorMap tma_a, tma_b, tma_c;
    {
        uint64_t dims[2]    = {(uint64_t)GA_K, (uint64_t)GA_M};
        uint64_t strides[1] = {(uint64_t)GA_K};
        uint32_t box[2]     = {BENCH_TK, BENCH_TM};
        uint32_t estrides[2]= {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(&tma_a,
            CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)d_A,
            dims, strides, box, estrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }
    {
        uint64_t dims[2]    = {(uint64_t)GB_K, (uint64_t)GB_N};
        uint64_t strides[1] = {(uint64_t)GB_K};
        uint32_t box[2]     = {BENCH_TK, BENCH_TN / 2};
        uint32_t estrides[2]= {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(&tma_b,
            CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)d_B,
            dims, strides, box, estrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }
    {
        /* TMA store descriptor: BF16 [N=256, M=256], box=[64,32] matching FC2 */
        uint64_t dims[2]    = {(uint64_t)GC_N, (uint64_t)GC_M};
        uint64_t strides[1] = {(uint64_t)GC_N * sizeof(__nv_bfloat16)};
        uint32_t box[2]     = {64, 32};
        uint32_t estrides[2]= {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(&tma_c,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)d_C,
            dims, strides, box, estrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }

    /* Set max SMEM for all kernels */
    auto set_smem = [](const void *fn) {
        CUDA_CHECK(cudaFuncSetAttribute(fn,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOT));
    };

    set_smem((const void*)k_mma_tput<1>);
    set_smem((const void*)k_mma_tput<4>);
    set_smem((const void*)k_mma_tput<8>);
    set_smem((const void*)k_mma_tput<16>);
    set_smem((const void*)k_fence_only);
    set_smem((const void*)k_commit_only);
    set_smem((const void*)k_kiter_full);
    set_smem((const void*)k_mma_lat);
    set_smem((const void*)k_tmem_load<1>);
    set_smem((const void*)k_tmem_load<2>);
    set_smem((const void*)k_tmem_load<4>);
    set_smem((const void*)k_tmem_load<8>);
    set_smem((const void*)k_tmem_load_x32<1>);
    set_smem((const void*)k_tmem_load_x32<2>);
    set_smem((const void*)k_tmem_load_x32<4>);
    set_smem((const void*)k_mma_shadow_sts<0>);
    set_smem((const void*)k_mma_shadow_sts<1>);
    set_smem((const void*)k_mma_shadow_sts<2>);
    set_smem((const void*)k_mma_shadow_sts<4>);
    set_smem((const void*)k_mma_shadow_sts<8>);
    set_smem((const void*)k_mma_shadow_sts<16>);
    set_smem((const void*)k_mma_shadow_bf16<0>);
    set_smem((const void*)k_mma_shadow_bf16<4>);
    set_smem((const void*)k_mma_shadow_bf16<8>);
    set_smem((const void*)k_mma_shadow_bf16<16>);
    set_smem((const void*)k_mma_shadow_bf16<32>);
    set_smem((const void*)k_mma_shadow_bf16<64>);
    set_smem((const void*)k_mma_tma_overlap);
    set_smem((const void*)k_kloop<1>);
    set_smem((const void*)k_kloop<4>);
    set_smem((const void*)k_kloop<8>);
    set_smem((const void*)k_kloop<16>);
    set_smem((const void*)k_kloop<24>);
    set_smem((const void*)k_tmem_dbuf);
    set_smem((const void*)k_sts_tmem_overlap<1>);
    set_smem((const void*)k_sts_tmem_overlap<4>);
    set_smem((const void*)k_sts_tmem_overlap<8>);
    set_smem((const void*)k_mma_compute_lat);
    set_smem((const void*)k_xwarp_handoff);
    set_smem((const void*)k_mw_sts<0>);
    set_smem((const void*)k_mw_sts<4>);
    set_smem((const void*)k_mw_sts<8>);
    set_smem((const void*)k_mw_sts<16>);
    set_smem((const void*)k_mw_mixed);
    set_smem((const void*)k_mw_full_epi);
    set_smem((const void*)k_mma_shadow_combined<0>);
    set_smem((const void*)k_mma_shadow_combined<1>);
    set_smem((const void*)k_mma_shadow_combined<2>);
    set_smem((const void*)k_mma_shadow_combined<4>);
    set_smem((const void*)k_mma_shadow_combined<8>);
    set_smem((const void*)k_mw_tma_mma);
    set_smem((const void*)k_mw_fence<0>);
    set_smem((const void*)k_mw_fence<4>);
    set_smem((const void*)k_mw_fence<8>);
    set_smem((const void*)k_mw_fence<16>);
    set_smem((const void*)k_mw_real_kloop);
    set_smem((const void*)k_mw_tma_store<0>);
    set_smem((const void*)k_mw_tma_store<1>);
    set_smem((const void*)k_mw_tma_store<2>);
    set_smem((const void*)k_mw_tma_store<4>);
    set_smem((const void*)k_multi_sm);

    long long *d_out, h[256];
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(h)));
    auto reset = [&]() { CUDA_CHECK(cudaMemset(d_out, 0, sizeof(h))); };
    const char *cur = "";
    auto sync = [&]() {
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { fprintf(stderr, "launch [%s]: %s\n", cur, cudaGetErrorString(e)); exit(1); }
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { fprintf(stderr, "kernel [%s]: %s\n", cur, cudaGetErrorString(e)); exit(1); }
        CUDA_CHECK(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
    };

    /* Launch config: 2 blocks (1 cluster), 32 threads */
    dim3 grid(2), block(32);

    printf("MMA Microbenchmark — SM100a (tcgen05.mma.cta_group::2.kind::f8f6f4)\n");
    printf("REPS=%d, WARMUP=%d, IDESC=0x%08X, SMEM=%d bytes\n\n", REPS, WARMUP, IDESC, SMEM_TOT);

    /* ═══ A. MMA ISSUE THROUGHPUT ═══ */
    printf("═══ A. MMA Issue Throughput (back-to-back, same descriptor, accumulating) ═══\n\n");
    for (int n : {1, 4, 8, 16}) {
        cur = "A.mma_tput";
        reset();
        switch (n) {
            case 1:  k_mma_tput<1> <<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 4:  k_mma_tput<4> <<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 8:  k_mma_tput<8> <<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 16: k_mma_tput<16><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
        }
        sync();
        double cpm = (double)h[0] / (REPS * n);
        printf("  ILP=%2d:  %6.1f cyc/MMA  (total %.0f cyc for %d ops)\n",
               n, cpm, (double)h[0] / REPS, n);
    }

    /* ═══ B. K-ITERATION DECOMPOSITION ═══ */
    printf("\n═══ B. K-Iteration Decomposition ═══\n\n");

    cur = "B.fence";
    reset(); k_fence_only<<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); sync();
    double t_fence = (double)h[0] / REPS;
    printf("  fence only:              %6.1f cyc\n", t_fence);

    cur = "B.commit";
    reset(); k_commit_only<<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); sync();
    double t_commit = (double)h[0] / REPS;
    printf("  commit only (dispatch):  %6.1f cyc\n", t_commit);

    cur = "B.kiter_full";
    reset(); k_kiter_full<<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); sync();
    double t_kiter = (double)h[0] / REPS;
    printf("  fence+4×MMA+commit+wait: %6.1f cyc  (4 sub-MMAs = 1 K-iteration)\n", t_kiter);

    /* 4×MMA only from section A */
    reset(); k_mma_tput<4><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); sync();
    double t_4mma = (double)h[0] / REPS;
    printf("  4×MMA dispatch only:     %6.1f cyc\n", t_4mma);
    printf("  overhead (kiter - 4×MMA):%6.1f cyc  (fence + commit + wait)\n", t_kiter - t_4mma);

    /* ═══ C. MMA LATENCY ═══ */
    printf("\n═══ C. MMA Latency (fence → 4×MMA → commit → mbar_wait → TMEM_LD) ═══\n\n");
    cur = "C.mma_lat";
    reset(); k_mma_lat<<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); sync();
    double t_lat = (double)h[0] / REPS;
    float verify_val;
    memcpy(&verify_val, &h[1], sizeof(float));
    printf("  full pipeline latency:   %6.1f cyc\n", t_lat);
    printf("  verify TMEM[0,0]:        %.1f  (expect ~288.0 for A=B=1.5 FP8 K=128)\n", verify_val);

    /* ═══ D. TMEM LOAD THROUGHPUT ═══ */
    printf("\n═══ D. TMEM Load Throughput (tcgen05.ld.sync.aligned) ═══\n\n");

    printf("  x16 (16 regs, 32 cols):\n");
    for (int n : {1, 2, 4, 8}) {
        cur = "D.tmem_x16";
        reset();
        switch (n) {
            case 1: k_tmem_load<1><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 2: k_tmem_load<2><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 4: k_tmem_load<4><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 8: k_tmem_load<8><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
        }
        sync();
        double cpl = (double)h[0] / (REPS * n);
        printf("    ILP=%d:  %6.1f cyc/load  (%.1f cyc total for %d loads)\n",
               n, cpl, (double)h[0] / REPS, n);
    }

    printf("  x32 (32 regs, 64 cols):\n");
    for (int n : {1, 2, 4}) {
        cur = "D.tmem_x32";
        reset();
        switch (n) {
            case 1: k_tmem_load_x32<1><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 2: k_tmem_load_x32<2><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 4: k_tmem_load_x32<4><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
        }
        sync();
        double cpl = (double)h[0] / (REPS * n);
        printf("    ILP=%d:  %6.1f cyc/load  (%.1f cyc total for %d loads)\n",
               n, cpl, (double)h[0] / REPS, n);
    }

    printf("  x64 (2×x32 sequential, 64 cols):\n");
    for (int n : {1, 2}) {
        cur = "D.tmem_x64";
        reset();
        /* x64 = 2 back-to-back x32 loads covering 64 columns */
        switch (n) {
            case 1: k_tmem_load_x32<2><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 2: k_tmem_load_x32<4><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
        }
        sync();
        double cpl = (double)h[0] / (REPS * n);
        printf("    ILP=%d:  %6.1f cyc/x64  (%.1f cyc total, 2×x32 each)\n",
               n, cpl, (double)h[0] / REPS);
    }

    /* ═══ E. MMA SHADOW BUDGET ═══ */
    printf("\n═══ E. MMA Shadow Budget (work between MMA dispatch and commit+wait) ═══\n\n");

    /* Baseline: MMA + 0 STS */
    cur = "E.sts";
    reset(); k_mma_shadow_sts<0><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); sync();
    double t_base = (double)h[0] / REPS;
    printf("  STS.128 between MMA and commit (32 cyc/STS throughput):\n");
    printf("    STS= 0:  %6.1f cyc  (baseline)\n", t_base);

    for (int n : {1, 2, 4, 8, 16}) {
        reset();
        switch (n) {
            case 1:  k_mma_shadow_sts<1> <<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 2:  k_mma_shadow_sts<2> <<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 4:  k_mma_shadow_sts<4> <<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 8:  k_mma_shadow_sts<8> <<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 16: k_mma_shadow_sts<16><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
        }
        sync();
        double t = (double)h[0] / REPS;
        double delta = t - t_base;
        printf("    STS=%2d:  %6.1f cyc  (%+.1f vs base, %.1f%% hidden)\n",
               n, t, delta, delta > 0 ? 100.0 * (1.0 - delta / (n * 32.0)) : 100.0);
    }

    /* BF16 HADD2 shadow */
    printf("\n  BF16 HADD2 between MMA and commit (2 cyc/op throughput):\n");
    reset(); k_mma_shadow_bf16<0><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); sync();
    double t_base_bf = (double)h[0] / REPS;
    printf("    BF16= 0:  %6.1f cyc  (baseline)\n", t_base_bf);

    for (int n : {4, 8, 16, 32, 64}) {
        cur = "E.bf16";
        reset();
        switch (n) {
            case 4:  k_mma_shadow_bf16<4> <<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 8:  k_mma_shadow_bf16<8> <<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 16: k_mma_shadow_bf16<16><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 32: k_mma_shadow_bf16<32><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 64: k_mma_shadow_bf16<64><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
        }
        sync();
        double t = (double)h[0] / REPS;
        double delta = t - t_base_bf;
        printf("    BF16=%2d:  %6.1f cyc  (%+.1f vs base, %.1f%% hidden)\n",
               n, t, delta, delta > 0 ? 100.0 * (1.0 - delta / (n * 2.0)) : 100.0);
    }

    /* ═══ F. MMA + TMA OVERLAP ═══ */
    printf("\n═══ F. MMA + TMA Overlap (concurrent TMA load during MMA) ═══\n\n");
    cur = "F.mma_tma";
    reset(); k_mma_tma_overlap<<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); sync();
    {
        double t_mma = (double)h[0] / REPS;
        double t_both = (double)h[1] / REPS;
        printf("  MMA only (K_ITER):       %6.1f cyc\n", t_mma);
        printf("  MMA + TMA concurrent:    %6.1f cyc\n", t_both);
        printf("  overhead:                %+.1f cyc (%.1f%%)\n",
               t_both - t_mma, 100.0 * (t_both / t_mma - 1.0));
    }

    /* ═══ G. FULL K-LOOP PIPELINE ═══ */
    printf("\n═══ G. Full K-Loop Pipeline (N_STAGES=%d, amortized) ═══\n\n", N_STAGES);
    for (int k : {1, 4, 8, 16, 24}) {
        cur = "G.kloop";
        reset();
        switch (k) {
            case 1:  k_kloop<1> <<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 4:  k_kloop<4> <<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 8:  k_kloop<8> <<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 16: k_kloop<16><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 24: k_kloop<24><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
        }
        sync();
        double total = (double)h[0] / REPS;
        double per_iter = total / k;
        printf("  K=%2d:  %7.1f cyc total  %6.1f cyc/iter  (4 sub-MMAs/iter)\n",
               k, total, per_iter);
    }

    /* ═══ H. TMEM DOUBLE-BUFFER ═══ */
    printf("\n═══ H. TMEM Double-Buffer (read buf0 while MMA writes buf1) ═══\n\n");
    cur = "H.dbuf";
    reset(); k_tmem_dbuf<<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); sync();
    {
        double t_mma = (double)h[0] / REPS;
        double t_both = (double)h[1] / REPS;
        printf("  MMA only (per K_ITER):   %6.1f cyc\n", t_mma);
        printf("  MMA + TMEM read:         %6.1f cyc\n", t_both);
        printf("  overhead:                %+.1f cyc (%.1f%%)\n",
               t_both - t_mma, 100.0 * (t_both / t_mma - 1.0));
    }

    /* ═══ I. STS + TMEM_LD OVERLAP ═══ */
    printf("\n═══ I. STS + TMEM_LD Overlap (after MMA complete) ═══\n\n");
    for (int n : {1, 4, 8}) {
        cur = "I.overlap";
        reset();
        switch (n) {
            case 1: k_sts_tmem_overlap<1><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 4: k_sts_tmem_overlap<4><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            case 8: k_sts_tmem_overlap<8><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
        }
        sync();
        double t_ld = (double)h[0] / REPS;
        double t_sts = (double)h[1] / REPS;
        double t_both = (double)h[2] / REPS;
        double expected = t_ld + t_sts;
        printf("  STS=%d:  TMEM_LD=%.1f  STS=%.1f  both=%.1f  (%.1f%% vs sequential)\n",
               n, t_ld, t_sts, t_both, 100.0 * (t_both / expected - 1.0));
    }

    /* ═══ J. MMA COMPUTE TIME ═══ */
    printf("\n═══ J. MMA Compute Time (commit → mbar_wait, tensor core latency) ═══\n\n");
    cur = "J.compute";
    reset(); k_mma_compute_lat<<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); sync();
    {
        double t = (double)h[0] / REPS;
        printf("  commit + mbar_wait:      %6.1f cyc  (MMA dispatched before timing)\n", t);
        printf("  cf. full K_ITER (B):     %6.1f cyc  (includes fence + dispatch)\n", t_kiter);
    }

    /* ═══ K. CROSS-WARP EPILOGUE HANDOFF ═══ */
    printf("\n═══ K. Cross-Warp Handoff (W1 MMA+commit → W2 mbar_wait+TMEM_LD) ═══\n\n");
    cur = "K.xwarp";
    {
        dim3 grid3(2), block3(96);
        reset(); k_xwarp_handoff<<<grid3, block3, SMEM_TOT>>>(tma_a, tma_b, d_out); sync();
        double t_wait = (double)h[0] / REPS;
        double t_ld = (double)h[1] / REPS;
        printf("  W2 mbar_wait (for W1):   %6.1f cyc\n", t_wait);
        printf("  W2 TMEM_LD (after wait): %6.1f cyc\n", t_ld);
        printf("  total handoff:           %6.1f cyc\n", t_wait + t_ld);
    }

    /* ═══ L. MULTI-WARP CONTENTION ═══ */
    printf("\n═══ L. Multi-Warp Contention (W1=MMA, W2-W5=epilogue STS) ═══\n\n");
    {
        dim3 grid_mw(2), block_mw(MW_THREADS);
        printf("  W1 K-iter time with N epilogue warps doing continuous STS:\n");
        for (int sts : {0, 4, 8, 16}) {
            cur = "L.mw_sts";
            reset();
            switch (sts) {
                case 0:  k_mw_sts<0> <<<grid_mw, block_mw, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
                case 4:  k_mw_sts<4> <<<grid_mw, block_mw, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
                case 8:  k_mw_sts<8> <<<grid_mw, block_mw, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
                case 16: k_mw_sts<16><<<grid_mw, block_mw, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            }
            sync();
            double t = (double)h[0] / REPS;
            printf("    EPI_STS=%2d:  %6.1f cyc/iter", sts, t);
            if (sts == 0) printf("  (baseline — no epilogue work)");
            printf("\n");
        }

        printf("\n  Mixed epilogue (4× LDS+BF16+STS per warp, realistic FC2):\n");
        cur = "L.mw_mixed";
        reset(); k_mw_mixed<<<grid_mw, block_mw, SMEM_TOT>>>(tma_a, tma_b, d_out); sync();
        printf("    mixed:       %6.1f cyc/iter\n", (double)h[0] / REPS);
    }

    /* ═══ N. FULL EPILOGUE PIPELINE DURING MMA ═══ */
    printf("\n═══ N. Full Epilogue Pipeline (TMEM_LD+F2FP+HADD2+STS+fence during MMA) ═══\n\n");
    cur = "N.full_epi";
    {
        dim3 grid_mw(2), block_mw(MW_THREADS);
        reset(); k_mw_full_epi<<<grid_mw, block_mw, SMEM_TOT>>>(tma_a, tma_b, d_out); sync();
        double t = (double)h[0] / REPS;
        printf("  W1 K-iter with full epilogue pipeline on W2-W5:\n");
        printf("    full_epi:    %6.1f cyc/iter\n", t);
        printf("  cf. baseline (L, no epi):  %6.1f cyc/iter\n", (double)h[0] / REPS); /* re-use for clarity */

        /* Run L baseline for direct comparison */
        reset(); k_mw_sts<0><<<grid_mw, block_mw, SMEM_TOT>>>(tma_a, tma_b, d_out); sync();
        double t_base = (double)h[0] / REPS;
        printf("    no_epi:      %6.1f cyc/iter  (L baseline re-run)\n", t_base);
        printf("    overhead:   %+6.1f cyc  (%.1f%%)\n", t - t_base, 100.0 * (t / t_base - 1.0));
    }

    /* ═══ O. COMBINED MMA SHADOW BUDGET ═══ */
    printf("\n═══ O. Combined MMA Shadow (TMEM_LD+F2FP+HADD2+STS+fence in shadow window) ═══\n\n");
    {
        double base_o = 0;
        for (int n : {0, 1, 2, 4, 8}) {
            cur = "O.combined";
            reset();
            switch (n) {
                case 0: k_mma_shadow_combined<0><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
                case 1: k_mma_shadow_combined<1><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
                case 2: k_mma_shadow_combined<2><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
                case 4: k_mma_shadow_combined<4><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
                case 8: k_mma_shadow_combined<8><<<grid, block, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            }
            sync();
            double t = (double)h[0] / REPS;
            if (n == 0) base_o = t;
            printf("  chunks=%d:  %6.1f cyc", n, t);
            if (n == 0) printf("  (baseline)");
            else printf("  (%+.1f vs base, %.1f%% hidden)", t - base_o, base_o > 0 ? 100.0 * (1.0 - (t - base_o) / (t - base_o + base_o)) : 0.0);
            /* Each chunk = 1 TMEM_LD + 4 F2FP + 4 HADD2 + 1 STS + 1 fence */
            printf("\n");
        }
        printf("  (each chunk = TMEM_LD.x16 + 4×F2FP + 4×HADD2 + STS.128 + fence.proxy.async)\n");
    }

    /* ═══ P. W0 TMA DISPATCH DURING MMA ═══ */
    printf("\n═══ P. W0 TMA Dispatch During MMA (TMA contention test) ═══\n\n");
    cur = "P.tma_mma";
    {
        dim3 grid_mw(2), block_mw(MW_THREADS);
        reset(); k_mw_tma_mma<<<grid_mw, block_mw, SMEM_TOT>>>(tma_a, tma_b, d_out); sync();
        double t_tma = (double)h[0] / REPS;

        /* Baseline: W0 idle */
        reset(); k_mw_sts<0><<<grid_mw, block_mw, SMEM_TOT>>>(tma_a, tma_b, d_out); sync();
        double t_base = (double)h[0] / REPS;

        printf("  W1 K-iter with W0 doing continuous TMA loads:\n");
        printf("    W0 TMA active: %6.1f cyc/iter\n", t_tma);
        printf("    W0 idle:       %6.1f cyc/iter  (baseline)\n", t_base);
        printf("    overhead:     %+6.1f cyc  (%.1f%%)\n",
               t_tma - t_base, 100.0 * (t_tma / t_base - 1.0));
    }

    /* ═══ Q. FENCE.PROXY.ASYNC + MBAR DURING MMA ═══ */
    printf("\n═══ Q. fence.proxy.async + mbar_arrive During MMA ═══\n\n");
    {
        dim3 grid_mw(2), block_mw(MW_THREADS);
        printf("  W1 K-iter with W2-W5 doing fence+mbar_arrive:\n");
        for (int n : {0, 4, 8, 16}) {
            cur = "Q.fence";
            reset();
            switch (n) {
                case 0:  k_mw_fence<0> <<<grid_mw, block_mw, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
                case 4:  k_mw_fence<4> <<<grid_mw, block_mw, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
                case 8:  k_mw_fence<8> <<<grid_mw, block_mw, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
                case 16: k_mw_fence<16><<<grid_mw, block_mw, SMEM_TOT>>>(tma_a, tma_b, d_out); break;
            }
            sync();
            double t = (double)h[0] / REPS;
            printf("    N_FENCE=%2d:  %6.1f cyc/iter", n, t);
            if (n == 0) printf("  (baseline)");
            printf("\n");
        }
    }

    /* ═══ R. PIPELINED K-LOOP WITH TMA ═══ */
    printf("\n═══ R. Pipelined K-Loop (W0=TMA, W1=MMA, %d iters, %d stages) ═══\n\n",
           KLOOP_ITERS, N_STAGES);
    cur = "R.kloop_tma";
    {
        dim3 grid_mw(2), block_mw(MW_THREADS);
        reset(); k_mw_real_kloop<<<grid_mw, block_mw, SMEM_TOT>>>(tma_a, tma_b, d_out); sync();
        double t_tma_kloop = (double)h[0] / REPS;
        double per_iter = t_tma_kloop / KLOOP_ITERS;
        printf("  W0 TMA + W1 MMA pipelined K-loop:\n");
        printf("    total:      %8.1f cyc  (%d iters)\n", t_tma_kloop, KLOOP_ITERS);
        printf("    per-iter:   %8.1f cyc/iter\n", per_iter);
        printf("  cf. MMA-only K-loop (G, K=%d):  per-iter=%.1f cyc\n", KLOOP_ITERS,
               12614.0 / KLOOP_ITERS); /* from section G output, hardcoded for reference */
        printf("  cf. single K-iter (B):           %.1f cyc\n", t_kiter);
        printf("  TMA overhead:  %.1f cyc/iter (%.1f%%)\n",
               per_iter - 525.6, 100.0 * (per_iter / 525.6 - 1.0));
    }

    /* ═══ S. TMA STORE DURING MMA ═══ */
    printf("\n═══ S. TMA Store During MMA (cp.async.bulk.tensor.2d contention) ═══\n\n");
    {
        dim3 grid_mw(2), block_mw(MW_THREADS);
        printf("  W1 K-iter with W2-W5 doing TMA stores:\n");
        for (int n : {0, 1, 2, 4}) {
            cur = "S.tma_store";
            reset();
            switch (n) {
                case 0: k_mw_tma_store<0><<<grid_mw, block_mw, SMEM_TOT>>>(tma_a, tma_b, tma_c, d_out); break;
                case 1: k_mw_tma_store<1><<<grid_mw, block_mw, SMEM_TOT>>>(tma_a, tma_b, tma_c, d_out); break;
                case 2: k_mw_tma_store<2><<<grid_mw, block_mw, SMEM_TOT>>>(tma_a, tma_b, tma_c, d_out); break;
                case 4: k_mw_tma_store<4><<<grid_mw, block_mw, SMEM_TOT>>>(tma_a, tma_b, tma_c, d_out); break;
            }
            sync();
            double t = (double)h[0] / REPS;
            printf("    N_STORE=%d:  %6.1f cyc/iter", n, t);
            if (n == 0) printf("  (baseline)");
            printf("\n");
        }
        printf("  (each store = fence.proxy.async + TMA store 64×32 BF16 + commit + wait)\n");
    }

    /* ═══ M. MULTI-SM SCALING ═══ */
    printf("\n═══ M. Multi-SM Scaling (%d clusters, K-iter throughput) ═══\n\n", MAX_CLUSTERS);
    cur = "M.multi_sm";
    {
        dim3 grid_all(MAX_CLUSTERS * 2), block1(32);
        reset(); k_multi_sm<<<grid_all, block1, SMEM_TOT>>>(tma_a, tma_b, d_out); sync();
        double mn = 1e18, mx = 0, sum = 0;
        int valid = 0;
        for (int i = 0; i < MAX_CLUSTERS; i++) {
            if (h[i] == 0) continue;
            double t = (double)h[i] / REPS;
            sum += t; valid++;
            if (t < mn) mn = t;
            if (t > mx) mx = t;
        }
        if (valid > 0) {
            double avg = sum / valid;
            printf("  %d clusters reporting:\n", valid);
            printf("    avg:  %6.1f cyc/iter\n", avg);
            printf("    min:  %6.1f cyc/iter\n", mn);
            printf("    max:  %6.1f cyc/iter\n", mx);
            printf("    spread: %.1f%%\n", 100.0 * (mx - mn) / avg);
        }
        /* Single cluster for comparison (from section B) */
        printf("  single cluster (B):  %6.1f cyc/iter\n", t_kiter);
    }

    printf("\nDone.\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
