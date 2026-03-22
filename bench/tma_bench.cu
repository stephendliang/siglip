/*
Comprehensive TMA microbenchmarks for SM100a (Blackwell)

Sections:
  A. TMA load latency — serial (single outstanding)
  B. TMA load throughput — pipeline depth sweep (1/2/4/8/16)
  C. Transfer size sweep — 512B, 1KB, 2KB, 4KB, 8KB
  D. Swizzle mode sweep — NONE, 32B, 64B, 128B (latency + contention)
  E. LSU baselines — LDS.128, STS.128, LDS+STS interleaved
  F. Single-warp contention — TMA+LDS, TMA+STS, TMA+LDS+STS
  G. Pipelined TMA + LDS — TMA pipe-2/4/8 while LDS runs (bandwidth stress)
  H. TMA store — latency, TMA_store+LDS contention
  I. Multi-warp — 2/4/6 warps: W0=TMA, rest=LDS
  J. Cold data — L2-miss TMA latency (64MB tensor)

Contention ratio: ~1.0 = independent, >1.0 = SMEM port sharing.
Build: make tma-bench && ./tma-bench
*/

#include <cstdio>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda.h>

#define WARMUP 64
#define REPS 512
#define LDS_OPS 512
#define STS_OPS 64

/* SMEM layout
   Max TMA buffer = pipe-16 at 4KB = 65536 bytes
   WORK region sits after that for kernels needing LDS/STS buffers */
#define MBAR_OFF 0
#define N_MBAR 32
#define TMA_BUF 256             /* 256B-aligned for TMA */
#define TMA_STAGE_MAX 8192      /* max per-stage size (8KB, for size sweep) */
#define MAX_PIPE 16
#define WORK_OFF (TMA_BUF + 4096 * MAX_PIPE)  /* 65792: after pipe-16 @ 4KB */
#define WORK_SZ 8192
#define SMEM_MAX (WORK_OFF + WORK_SZ)  /* 73984 = ~72KB, well within 228KB */

/* Global tensor */
#define GW 1024
#define GH 1024
/* Cold data tensor (64MB > B200 L2) */
#define COLD_GH 32768  /* 32768 × 1024 × 2B = 64MB */

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

__device__ __forceinline__ void mb_init(uint32_t a, int n) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(n));
}

__device__ __forceinline__ void mb_expect_tx(uint32_t a, int b) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(a), "r"(b) : "memory");
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

__device__ __forceinline__ void tma_ld(uint32_t dst, const void *tm, int x, int y, uint32_t mb) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tm), "r"(x), "r"(y), "r"(mb) : "memory");
}

__device__ __forceinline__ void fence_async() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void tma_st(const void *tm, int x, int y, uint32_t src) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
        :: "l"(tm), "r"(x), "r"(y), "r"(src) : "memory");
}

__device__ __forceinline__ void bulk_commit() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ __forceinline__ void bulk_wait_all() {
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

/* ═══════ A. TMA load latency (serial) — parameterized ═══════ */

__global__ __launch_bounds__(32, 1)
void k_tma_lat(__grid_constant__ const CUtensorMap tm, int bytes, long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem), mb = sb + MBAR_OFF, buf = sb + TMA_BUF;
    int lane = threadIdx.x, ph = 0;
    if (lane == 0) mb_init(mb, 1);
    __syncwarp();

    for (int i = 0; i < WARMUP; i++) {
        if (lane == 0) { mb_expect_tx(mb, bytes); tma_ld(buf, &tm, 0, 0, mb); }
        mb_wait(mb, ph); ph ^= 1;
    }
    long long tot = 0;
    for (int i = 0; i < REPS; i++) {
        if (lane == 0) mb_expect_tx(mb, bytes);
        long long t0 = clk();
        if (lane == 0) tma_ld(buf, &tm, 0, 0, mb);
        mb_wait(mb, ph);
        long long t1 = clk();
        tot += t1 - t0; ph ^= 1;
    }
    if (lane == 0) out[0] = tot;
}

/* Cold data: vary y coordinate each rep to miss L2 */
__global__ __launch_bounds__(32, 1)
void k_tma_lat_cold(__grid_constant__ const CUtensorMap tm, int bytes, int box_y, long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem), mb = sb + MBAR_OFF, buf = sb + TMA_BUF;
    int lane = threadIdx.x, ph = 0;
    if (lane == 0) mb_init(mb, 1);
    __syncwarp();

    for (int i = 0; i < WARMUP; i++) {
        int y = (i * box_y) % COLD_GH;
        if (lane == 0) { mb_expect_tx(mb, bytes); tma_ld(buf, &tm, 0, y, mb); }
        mb_wait(mb, ph); ph ^= 1;
    }
    long long tot = 0;
    for (int i = 0; i < REPS; i++) {
        int y = ((WARMUP + i) * box_y) % COLD_GH;
        if (lane == 0) mb_expect_tx(mb, bytes);
        long long t0 = clk();
        if (lane == 0) tma_ld(buf, &tm, 0, y, mb);
        mb_wait(mb, ph);
        long long t1 = clk();
        tot += t1 - t0; ph ^= 1;
    }
    if (lane == 0) out[0] = tot;
}

/* ═══════ B. TMA load throughput (pipelined) ═══════ */

template <int D>
__global__ __launch_bounds__(32, 1)
void k_tma_pipe(__grid_constant__ const CUtensorMap tm, int bytes, int stage, int box_y, long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    int lane = threadIdx.x, ph = 0;
    if (lane == 0)
        for (int i = 0; i < D; i++) mb_init(sb + MBAR_OFF + i*8, 1);
    __syncwarp();

    for (int r = 0; r < WARMUP; r++) {
        for (int i = 0; i < D; i++) {
            uint32_t m = sb + MBAR_OFF + i*8;
            if (lane == 0) { mb_expect_tx(m, bytes); tma_ld(sb + TMA_BUF + i*stage, &tm, 0, (i*box_y)%GH, m); }
        }
        for (int i = 0; i < D; i++) mb_wait(sb + MBAR_OFF + i*8, ph);
        ph ^= 1;
    }
    long long t0 = clk();
    for (int r = 0; r < REPS; r++) {
        for (int i = 0; i < D; i++) {
            uint32_t m = sb + MBAR_OFF + i*8;
            if (lane == 0) { mb_expect_tx(m, bytes); tma_ld(sb + TMA_BUF + i*stage, &tm, 0, (i*box_y)%GH, m); }
        }
        for (int i = 0; i < D; i++) mb_wait(sb + MBAR_OFF + i*8, ph);
        ph ^= 1;
    }
    long long t1 = clk();
    if (lane == 0) { out[0] = t1 - t0; out[1] = (long long)(REPS * D); }
}

/* ═══════ E. LSU baselines ═══════ */

__global__ __launch_bounds__(32, 1)
void k_lds(long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    int lane = threadIdx.x;
    uint32_t addr = sb + WORK_OFF + lane * 16;
    uint32_t v = 0x3c003c00u | lane;
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                 :: "r"(addr), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
    __syncwarp();

    uint32_t r0, r1, r2, r3, acc = 0;
    for (int i = 0; i < WARMUP; i++) {
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr)); acc ^= r0;
    }
    long long tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        long long t0 = clk();
        #pragma unroll 1
        for (int i = 0; i < LDS_OPS; i++) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr)); acc ^= r0;
        }
        long long t1 = clk();
        tot += t1 - t0;
    }
    if (lane == 0) out[0] = tot;
    if (acc == 0xDEAD) out[1] = acc;
}

__global__ __launch_bounds__(32, 1)
void k_sts(long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    int lane = threadIdx.x;
    uint32_t addr = sb + WORK_OFF + lane * 16;
    uint32_t v = 0x3c003c00u | lane;
    for (int i = 0; i < WARMUP; i++)
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                     :: "r"(addr), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
    long long tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        long long t0 = clk();
        #pragma unroll 1
        for (int i = 0; i < STS_OPS; i++)
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                         :: "r"(addr), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
        long long t1 = clk();
        tot += t1 - t0;
    }
    if (lane == 0) out[0] = tot;
}

__global__ __launch_bounds__(32, 1)
void k_lds_sts(long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    int lane = threadIdx.x;
    uint32_t lds_a = sb + WORK_OFF + lane * 16;
    uint32_t sts_a = sb + WORK_OFF + 4096 + lane * 16;
    uint32_t v = 0x3c003c00u | lane;
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                 :: "r"(lds_a), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
    __syncwarp();

    uint32_t r0, r1, r2, r3, acc = 0;
    for (int i = 0; i < WARMUP; i++) {
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(lds_a)); acc ^= r0;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                     :: "r"(sts_a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
    }

    int groups = 48;
    long long tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        long long t0 = clk();
        #pragma unroll 1
        for (int g = 0; g < groups; g++) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(lds_a)); acc ^= r0;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(lds_a)); acc ^= r0;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(lds_a)); acc ^= r0;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(lds_a)); acc ^= r0;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(lds_a)); acc ^= r0;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(lds_a)); acc ^= r0;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(lds_a)); acc ^= r0;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sts_a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sts_a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sts_a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sts_a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
        }
        long long t1 = clk();
        tot += t1 - t0;
    }
    if (lane == 0) { out[0] = tot; out[1] = (long long)(groups * 7); out[2] = (long long)(groups * 4); }
    if (acc == 0xDEAD) out[3] = acc;
}

/* ═══════ F. Single-warp contention ═══════ */

__global__ __launch_bounds__(32, 1)
void k_tma_lds(__grid_constant__ const CUtensorMap tm, int bytes, long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    int lane = threadIdx.x;
    uint32_t mb = sb + MBAR_OFF, tma_buf = sb + TMA_BUF;
    uint32_t lds_a = sb + WORK_OFF + lane * 16;
    if (lane == 0) mb_init(mb, 1);
    uint32_t v = 0x3c003c00u | lane;
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                 :: "r"(lds_a), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
    __syncwarp();

    uint32_t r0, r1, r2, r3, acc = 0; int ph = 0;
    for (int i = 0; i < WARMUP; i++) {
        if (lane == 0) { mb_expect_tx(mb, bytes); tma_ld(tma_buf, &tm, 0, 0, mb); }
        for (int j = 0; j < 8; j++) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(lds_a)); acc ^= r0;
        }
        mb_wait(mb, ph); ph ^= 1;
    }
    long long lds_tot = 0, tma_tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        if (lane == 0) mb_expect_tx(mb, bytes);
        long long ta = clk();
        if (lane == 0) tma_ld(tma_buf, &tm, 0, 0, mb);
        long long t0 = clk();
        #pragma unroll 1
        for (int i = 0; i < LDS_OPS; i++) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(lds_a)); acc ^= r0;
        }
        long long t1 = clk();
        mb_wait(mb, ph); long long tb = clk();
        lds_tot += t1 - t0; tma_tot += tb - ta; ph ^= 1;
    }
    if (lane == 0) { out[0] = lds_tot; out[1] = tma_tot; }
    if (acc == 0xDEAD) out[2] = acc;
}

__global__ __launch_bounds__(32, 1)
void k_tma_sts(__grid_constant__ const CUtensorMap tm, int bytes, long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    int lane = threadIdx.x;
    uint32_t mb = sb + MBAR_OFF, tma_buf = sb + TMA_BUF;
    uint32_t sts_a = sb + WORK_OFF + lane * 16;
    if (lane == 0) mb_init(mb, 1);
    __syncwarp();

    uint32_t v = 0x3c003c00u | lane; int ph = 0;
    for (int i = 0; i < WARMUP; i++) {
        if (lane == 0) { mb_expect_tx(mb, bytes); tma_ld(tma_buf, &tm, 0, 0, mb); }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                     :: "r"(sts_a), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
        mb_wait(mb, ph); ph ^= 1;
    }
    long long sts_tot = 0, tma_tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        if (lane == 0) mb_expect_tx(mb, bytes);
        long long ta = clk();
        if (lane == 0) tma_ld(tma_buf, &tm, 0, 0, mb);
        long long t0 = clk();
        #pragma unroll 1
        for (int i = 0; i < STS_OPS; i++)
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                         :: "r"(sts_a), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
        long long t1 = clk();
        mb_wait(mb, ph); long long tb = clk();
        sts_tot += t1 - t0; tma_tot += tb - ta; ph ^= 1;
    }
    if (lane == 0) { out[0] = sts_tot; out[1] = tma_tot; }
}

__global__ __launch_bounds__(32, 1)
void k_tma_lds_sts(__grid_constant__ const CUtensorMap tm, int bytes, long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    int lane = threadIdx.x;
    uint32_t mb = sb + MBAR_OFF, tma_buf = sb + TMA_BUF;
    uint32_t lds_a = sb + WORK_OFF + lane * 16;
    uint32_t sts_a = sb + WORK_OFF + 4096 + lane * 16;
    if (lane == 0) mb_init(mb, 1);
    uint32_t v = 0x3c003c00u | lane;
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                 :: "r"(lds_a), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
    __syncwarp();

    uint32_t r0, r1, r2, r3, acc = 0; int ph = 0;
    for (int i = 0; i < WARMUP; i++) {
        if (lane == 0) { mb_expect_tx(mb, bytes); tma_ld(tma_buf, &tm, 0, 0, mb); }
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(lds_a)); acc ^= r0;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                     :: "r"(sts_a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
        mb_wait(mb, ph); ph ^= 1;
    }
    long long work_tot = 0, tma_tot = 0;
    int groups = 48;
    for (int rep = 0; rep < REPS; rep++) {
        if (lane == 0) mb_expect_tx(mb, bytes);
        long long ta = clk();
        if (lane == 0) tma_ld(tma_buf, &tm, 0, 0, mb);
        long long t0 = clk();
        #pragma unroll 1
        for (int g = 0; g < groups; g++) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(lds_a)); acc ^= r0;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(lds_a)); acc ^= r0;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(lds_a)); acc ^= r0;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(lds_a)); acc ^= r0;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(lds_a)); acc ^= r0;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(lds_a)); acc ^= r0;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(lds_a)); acc ^= r0;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sts_a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sts_a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sts_a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sts_a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
        }
        long long t1 = clk();
        mb_wait(mb, ph); long long tb = clk();
        work_tot += t1 - t0; tma_tot += tb - ta; ph ^= 1;
    }
    if (lane == 0) { out[0] = work_tot; out[1] = tma_tot; }
    if (acc == 0xDEAD) out[2] = acc;
}

/* ═══════ G. Pipelined TMA + LDS (bandwidth stress) ═══════ */

template <int D>
__global__ __launch_bounds__(32, 1)
void k_pipe_tma_lds(__grid_constant__ const CUtensorMap tm, int bytes, int stage, int box_y, long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    int lane = threadIdx.x;
    uint32_t lds_a = sb + WORK_OFF + lane * 16;

    if (lane == 0)
        for (int i = 0; i < D; i++) mb_init(sb + MBAR_OFF + i*8, 1);
    uint32_t v = 0x3c003c00u | lane;
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                 :: "r"(lds_a), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
    __syncwarp();

    uint32_t r0, r1, r2, r3, acc = 0; int ph = 0;
    for (int r = 0; r < WARMUP; r++) {
        for (int i = 0; i < D; i++) {
            uint32_t m = sb + MBAR_OFF + i*8;
            if (lane == 0) { mb_expect_tx(m, bytes); tma_ld(sb + TMA_BUF + i*stage, &tm, 0, (i*box_y)%GH, m); }
        }
        for (int j = 0; j < 8; j++) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(lds_a)); acc ^= r0;
        }
        for (int i = 0; i < D; i++) mb_wait(sb + MBAR_OFF + i*8, ph);
        ph ^= 1;
    }

    long long lds_tot = 0, tma_tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        for (int i = 0; i < D; i++) {
            uint32_t m = sb + MBAR_OFF + i*8;
            if (lane == 0) { mb_expect_tx(m, bytes); }
        }
        long long ta = clk();
        for (int i = 0; i < D; i++) {
            uint32_t m = sb + MBAR_OFF + i*8;
            if (lane == 0) tma_ld(sb + TMA_BUF + i*stage, &tm, 0, (i*box_y)%GH, m);
        }
        long long t0 = clk();
        #pragma unroll 1
        for (int i = 0; i < LDS_OPS; i++) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(lds_a)); acc ^= r0;
        }
        long long t1 = clk();
        for (int i = 0; i < D; i++) mb_wait(sb + MBAR_OFF + i*8, ph);
        long long tb = clk();
        lds_tot += t1 - t0; tma_tot += tb - ta; ph ^= 1;
    }
    if (lane == 0) { out[0] = lds_tot; out[1] = tma_tot; }
    if (acc == 0xDEAD) out[2] = acc;
}

/* ═══════ H. TMA store ═══════ */

__global__ __launch_bounds__(32, 1)
void k_tma_store_lat(__grid_constant__ const CUtensorMap tm, int bytes, long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    int lane = threadIdx.x;
    uint32_t src = sb + TMA_BUF;

    /* Fill SMEM source buffer */
    uint32_t v = 0x3c003c00u | lane;
    for (int i = 0; i < bytes / (32 * 16); i++)
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                     :: "r"(src + lane * 16 + i * 512), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
    __syncwarp();

    for (int i = 0; i < WARMUP; i++) {
        fence_async();
        if (lane == 0) tma_st(&tm, 0, 0, src);
        bulk_commit(); bulk_wait_all();
    }

    long long tot = 0;
    for (int i = 0; i < REPS; i++) {
        fence_async();
        long long t0 = clk();
        if (lane == 0) tma_st(&tm, 0, 0, src);
        bulk_commit(); bulk_wait_all();
        long long t1 = clk();
        tot += t1 - t0;
    }
    if (lane == 0) out[0] = tot;
}

/* TMA store in flight while LDS loop runs.
   Key: TMA store READS from SMEM (same direction as LDS!) */
__global__ __launch_bounds__(32, 1)
void k_tma_store_lds(__grid_constant__ const CUtensorMap tm_store, int bytes, long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    int lane = threadIdx.x;
    uint32_t store_src = sb + TMA_BUF;
    uint32_t lds_a = sb + WORK_OFF + lane * 16;

    uint32_t v = 0x3c003c00u | lane;
    for (int i = 0; i < bytes / (32 * 16); i++)
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                     :: "r"(store_src + lane * 16 + i * 512), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                 :: "r"(lds_a), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
    __syncwarp();

    uint32_t r0, r1, r2, r3, acc = 0;
    for (int i = 0; i < WARMUP; i++) {
        fence_async();
        if (lane == 0) tma_st(&tm_store, 0, 0, store_src);
        bulk_commit();
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(lds_a)); acc ^= r0;
        bulk_wait_all();
    }

    long long lds_tot = 0, tma_tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        fence_async();
        long long ta = clk();
        if (lane == 0) tma_st(&tm_store, 0, 0, store_src);
        bulk_commit();
        long long t0 = clk();
        #pragma unroll 1
        for (int i = 0; i < LDS_OPS; i++) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(lds_a)); acc ^= r0;
        }
        long long t1 = clk();
        bulk_wait_all();
        long long tb = clk();
        lds_tot += t1 - t0; tma_tot += tb - ta;
    }
    if (lane == 0) { out[0] = lds_tot; out[1] = tma_tot; }
    if (acc == 0xDEAD) out[2] = acc;
}

/* ═══════ I. Multi-warp TMA + LDS ═══════ */

template <int NW>
__global__ __launch_bounds__(NW * 32, 1)
void k_multi_tma_lds(__grid_constant__ const CUtensorMap tm, int bytes, long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    uint32_t mb = sb + MBAR_OFF, tma_buf = sb + TMA_BUF;
    uint32_t lds_a = sb + WORK_OFF + lane * 16;

    if (threadIdx.x == 0) mb_init(mb, 1);
    uint32_t v = 0x3c003c00u | lane;
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                 :: "r"(lds_a), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
    __syncthreads();

    uint32_t r0, r1, r2, r3, acc = 0; int ph = 0;
    for (int i = 0; i < WARMUP; i++) {
        if (wid == 0 && lane == 0) { mb_expect_tx(mb, bytes); tma_ld(tma_buf, &tm, 0, 0, mb); }
        if (wid > 0) {
            for (int j = 0; j < 8; j++) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(lds_a)); acc ^= r0;
            }
        }
        if (wid == 0) mb_wait(mb, ph);
        __syncthreads(); ph ^= 1;
    }

    long long tma_tot = 0, lds_tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        if (wid == 0 && lane == 0) mb_expect_tx(mb, bytes);
        long long ta = clk();
        if (wid == 0 && lane == 0) tma_ld(tma_buf, &tm, 0, 0, mb);
        long long t0 = clk();
        if (wid > 0) {
            #pragma unroll 1
            for (int i = 0; i < LDS_OPS; i++) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(lds_a)); acc ^= r0;
            }
        }
        long long t1 = clk();
        if (wid == 0) mb_wait(mb, ph);
        __syncthreads(); long long tb = clk();
        if (wid == 0) tma_tot += tb - ta;
        if (wid == 1) lds_tot += t1 - t0;
        ph ^= 1;
    }
    if (wid == 0 && lane == 0) out[0] = tma_tot;
    if (wid == 1 && lane == 0) out[1] = lds_tot;
    if (acc == 0xDEAD) out[2] = acc;
}

/* ═══════ K. LDS from TMA buffer (read-back, same swizzled region) ═══════ */
/* FC2 loads residual via TMA into SMEM, then LDS reads from that SAME buffer.
   Tests whether reading from a TMA-written swizzled region differs from
   reading from a thread-initialized region (as the baseline tests do). */

__global__ __launch_bounds__(32, 1)
void k_tma_read_back(__grid_constant__ const CUtensorMap tm, int bytes, long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    int lane = threadIdx.x;
    uint32_t mb = sb + MBAR_OFF, buf = sb + TMA_BUF;
    uint32_t lds_a = buf + lane * 16;  /* read FROM TMA buffer */
    if (lane == 0) mb_init(mb, 1);
    __syncwarp();

    uint32_t r0, r1, r2, r3, acc = 0; int ph = 0;
    /* Prime: TMA load so buffer has valid data */
    if (lane == 0) { mb_expect_tx(mb, bytes); tma_ld(buf, &tm, 0, 0, mb); }
    mb_wait(mb, ph); ph ^= 1;

    for (int i = 0; i < WARMUP; i++) {
        if (lane == 0) { mb_expect_tx(mb, bytes); tma_ld(buf, &tm, 0, 0, mb); }
        mb_wait(mb, ph); ph ^= 1;
        #pragma unroll 1
        for (int j = 0; j < LDS_OPS; j++) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(lds_a)); acc ^= r0;
        }
    }

    long long tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        /* Reload via TMA (refreshes swizzled data in buffer) */
        if (lane == 0) { mb_expect_tx(mb, bytes); tma_ld(buf, &tm, 0, 0, mb); }
        mb_wait(mb, ph); ph ^= 1;
        /* Now LDS from the TMA buffer */
        long long t0 = clk();
        #pragma unroll 1
        for (int i = 0; i < LDS_OPS; i++) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(lds_a)); acc ^= r0;
        }
        long long t1 = clk();
        tot += t1 - t0;
    }
    if (lane == 0) out[0] = tot;
    if (acc == 0xDEAD) out[1] = acc;
}

/* ═══════ L. Double-buffered TMA + LDS (exact FC2 pattern) ═══════ */
/* TMA writes to buffer A while LDS reads from buffer B (previously loaded).
   This is the FC2 epilogue double-buffer: TMA loads residual for next tile
   into one buffer while epilogue warps read residual from the other buffer. */

__global__ __launch_bounds__(32, 1)
void k_dbuf_tma_lds(__grid_constant__ const CUtensorMap tm, int bytes, int stage, long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    int lane = threadIdx.x;
    uint32_t mbuf[2] = {sb + MBAR_OFF, sb + MBAR_OFF + 8};
    uint32_t buf[2] = {sb + TMA_BUF, sb + TMA_BUF + (uint32_t)stage};

    if (lane == 0) { mb_init(mbuf[0], 1); mb_init(mbuf[1], 1); }
    __syncwarp();

    uint32_t r0, r1, r2, r3, acc = 0;
    int ph[2] = {0, 0};

    /* Prime both buffers */
    if (lane == 0) { mb_expect_tx(mbuf[0], bytes); tma_ld(buf[0], &tm, 0, 0, mbuf[0]); }
    mb_wait(mbuf[0], ph[0]); ph[0] ^= 1;
    if (lane == 0) { mb_expect_tx(mbuf[1], bytes); tma_ld(buf[1], &tm, 0, 0, mbuf[1]); }
    mb_wait(mbuf[1], ph[1]); ph[1] ^= 1;

    for (int i = 0; i < WARMUP; i++) {
        int cur = i & 1, prev = 1 - cur;
        if (lane == 0) { mb_expect_tx(mbuf[cur], bytes); tma_ld(buf[cur], &tm, 0, 0, mbuf[cur]); }
        uint32_t lds_a = buf[prev] + lane * 16;
        #pragma unroll 1
        for (int j = 0; j < 16; j++) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(lds_a)); acc ^= r0;
        }
        mb_wait(mbuf[cur], ph[cur]); ph[cur] ^= 1;
    }

    long long lds_tot = 0, tma_tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        int cur = rep & 1, prev = 1 - cur;
        if (lane == 0) mb_expect_tx(mbuf[cur], bytes);
        long long ta = clk();
        if (lane == 0) tma_ld(buf[cur], &tm, 0, 0, mbuf[cur]);
        /* LDS from OTHER buffer (previously loaded) while TMA writes current */
        uint32_t lds_a = buf[prev] + lane * 16;
        long long t0 = clk();
        #pragma unroll 1
        for (int i = 0; i < LDS_OPS; i++) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(lds_a)); acc ^= r0;
        }
        long long t1 = clk();
        mb_wait(mbuf[cur], ph[cur]); ph[cur] ^= 1;
        long long tb = clk();
        lds_tot += t1 - t0; tma_tot += tb - ta;
    }
    if (lane == 0) { out[0] = lds_tot; out[1] = tma_tot; }
    if (acc == 0xDEAD) out[2] = acc;
}

/* ═══════ M. Dual TMA: load + store simultaneously ═══════ */
/* FC2 pipelines K-loop TMA loads (GMEM→SMEM) with epilogue TMA stores
   (SMEM→GMEM). Both use the TMA engine. Tests if they compete. */

__global__ __launch_bounds__(32, 1)
void k_dual_tma(__grid_constant__ const CUtensorMap tm_load, __grid_constant__ const CUtensorMap tm_store, int bytes, long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    int lane = threadIdx.x;
    uint32_t mb = sb + MBAR_OFF;
    uint32_t load_buf = sb + TMA_BUF;
    uint32_t store_buf = sb + TMA_BUF + TMA_STAGE_MAX;

    if (lane == 0) mb_init(mb, 1);
    /* Fill store source buffer */
    uint32_t v = 0x3c003c00u | lane;
    for (int i = 0; i < bytes / (32 * 16) + 1; i++)
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                     :: "r"(store_buf + lane * 16 + i * 512), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
    __syncwarp();

    int ph = 0;
    for (int i = 0; i < WARMUP; i++) {
        if (lane == 0) mb_expect_tx(mb, bytes);
        fence_async();
        if (lane == 0) {
            tma_ld(load_buf, &tm_load, 0, 0, mb);
            tma_st(&tm_store, 0, 0, store_buf);
        }
        bulk_commit();
        mb_wait(mb, ph); ph ^= 1;
        bulk_wait_all();
    }

    long long tot = 0;
    for (int i = 0; i < REPS; i++) {
        if (lane == 0) mb_expect_tx(mb, bytes);
        fence_async();
        long long t0 = clk();
        if (lane == 0) {
            tma_ld(load_buf, &tm_load, 0, 0, mb);
            tma_st(&tm_store, 0, 0, store_buf);
        }
        bulk_commit();
        mb_wait(mb, ph); ph ^= 1;
        bulk_wait_all();
        long long t1 = clk();
        tot += t1 - t0;
    }
    if (lane == 0) out[0] = tot;
}

/* ═══════ N. Mbarrier overhead (isolation) ═══════ */
/* Measures mbarrier init/arrive/wait overhead without any TMA or LSU work.
   FC2 timing shows epi_wait = 107 cycles — how much is barrier cost? */

__global__ __launch_bounds__(32, 1)
void k_mbar_overhead(long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    int lane = threadIdx.x;
    uint32_t mb = sb + MBAR_OFF;

    if (lane == 0) mb_init(mb, 1);
    __syncwarp();

    int ph = 0;
    /* arrive+wait with 0 tx bytes — barrier completes immediately after arrive */
    for (int i = 0; i < WARMUP; i++) {
        if (lane == 0) {
            asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(mb) : "memory");
        }
        mb_wait(mb, ph); ph ^= 1;
    }

    long long tot = 0;
    for (int i = 0; i < REPS; i++) {
        long long t0 = clk();
        if (lane == 0) {
            asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(mb) : "memory");
        }
        mb_wait(mb, ph);
        long long t1 = clk();
        tot += t1 - t0; ph ^= 1;
    }
    if (lane == 0) out[0] = tot;
}

/* ═══════ O. LDS ILP sweep — latency vs throughput ═══════ */
/* FC2 epilogue issues 7 independent LDS.128 for residual.
   At ILP=1 (serial chain), LDS takes ~25 cyc/op (latency-bound).
   At ILP=7+, LDS should approach 4 cyc/op (throughput).
   This test measures the actual scaling. */

template <int ILP>
__global__ __launch_bounds__(32, 1)
void k_lds_ilp(long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    int lane = threadIdx.x;

    /* Init ILP separate SMEM locations */
    for (int j = 0; j < ILP; j++) {
        uint32_t a = sb + WORK_OFF + j * 512 + lane * 16;
        uint32_t v = 0x3c003c00u | (lane + j);
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                     :: "r"(a), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
    }
    __syncwarp();

    /* ILP independent accumulators — no cross-chain dependencies */
    uint32_t acc[ILP];
    uint32_t addr[ILP];
    for (int j = 0; j < ILP; j++) {
        acc[j] = 0;
        addr[j] = sb + WORK_OFF + j * 512 + lane * 16;
    }

    uint32_t r0, r1, r2, r3;
    int ops_per_iter = ILP;
    int iters = 512 / ILP;  /* total ops = iters * ILP = 512 */

    for (int i = 0; i < WARMUP; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr[j]));
            acc[j] ^= r0;
        }
    }

    long long tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        long long t0 = clk();
        #pragma unroll 1
        for (int i = 0; i < iters; i++) {
            #pragma unroll
            for (int j = 0; j < ILP; j++) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr[j]));
                acc[j] ^= r0;
            }
        }
        long long t1 = clk();
        tot += t1 - t0;
    }

    uint32_t sink = 0;
    for (int j = 0; j < ILP; j++) sink ^= acc[j];
    if (lane == 0) out[0] = tot;
    if (sink == 0xDEAD) out[1] = (long long)sink;
}

/* ═══════ P. STS interleaving with BF16 compute ═══════ */
/* CP-SAT scheduler assumes BF16 ops fill the 32-cycle STS gaps.
   Test: STS followed by N BF16 ops, measure if BF16 hides in STS shadow. */

template <int BF16_BETWEEN>
__global__ __launch_bounds__(32, 1)
void k_sts_bf16(long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    int lane = threadIdx.x;

    uint32_t sts_a = sb + WORK_OFF + lane * 16;
    uint32_t v = 0x3c003c00u | lane;  /* BF16 1.0 pair */
    uint32_t bf = v;

    for (int i = 0; i < WARMUP; i++) {
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                     :: "r"(sts_a), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
        #pragma unroll
        for (int j = 0; j < BF16_BETWEEN; j++)
            asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(bf) : "r"(v));
    }

    int n_sts = 64;
    long long tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        long long t0 = clk();
        #pragma unroll 1
        for (int i = 0; i < n_sts; i++) {
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                         :: "r"(sts_a), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
            #pragma unroll
            for (int j = 0; j < BF16_BETWEEN; j++)
                asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(bf) : "r"(v));
        }
        long long t1 = clk();
        tot += t1 - t0;
    }
    if (lane == 0) out[0] = tot;
    if (bf == 0xDEAD) out[1] = (long long)bf;
}

/* ═══════ Q. Realistic FC2 epilogue fragment ═══════ */
/* One epilogue group: 7× LDS.128 (independent) → HADD2 → F2FP → 4× STS.128
   No LDTM (would need TMEM), but captures the LDS→compute→STS critical path. */

__global__ __launch_bounds__(32, 1)
void k_epilogue_fragment(long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    int lane = threadIdx.x;

    /* Init 7 LDS source regions (simulate residual in SMEM) */
    for (int j = 0; j < 7; j++) {
        uint32_t a = sb + WORK_OFF + j * 512 + lane * 16;
        uint32_t v = 0x3c003c00u | (lane + j);
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                     :: "r"(a), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
    }
    __syncwarp();

    uint32_t lds_addr[7];
    for (int j = 0; j < 7; j++)
        lds_addr[j] = sb + WORK_OFF + j * 512 + lane * 16;
    uint32_t sts_a = sb + WORK_OFF + 4096 + lane * 16;

    uint32_t r0, r1, r2, r3;
    uint32_t bias = 0x3c003c00u;  /* BF16 1.0 pair */
    uint32_t acc_val = 0x3c003c00u;

    for (int i = 0; i < WARMUP; i++) {
        /* 7 independent LDS (residual) */
        #pragma unroll
        for (int j = 0; j < 7; j++)
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(lds_addr[j]));
        /* BF16 compute: HADD2 (residual + bias) + F2FP (convert) */
        asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(r0) : "r"(bias));
        asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(r0) : "r"(acc_val));
        /* 4 STS (output staging) */
        #pragma unroll
        for (int j = 0; j < 4; j++)
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                         :: "r"(sts_a + j * 512), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
    }

    int groups = 48;
    long long tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        long long t0 = clk();
        #pragma unroll 1
        for (int g = 0; g < groups; g++) {
            /* 7 independent LDS */
            #pragma unroll
            for (int j = 0; j < 7; j++)
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(lds_addr[j]));
            /* compute */
            asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(r0) : "r"(bias));
            asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(r0) : "r"(acc_val));
            /* 4 STS */
            #pragma unroll
            for (int j = 0; j < 4; j++)
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                             :: "r"(sts_a + j * 512), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
        }
        long long t1 = clk();
        tot += t1 - t0;
    }
    if (lane == 0) { out[0] = tot; out[1] = (long long)groups; }
}

/* ═══════ R. TMA issue rate ═══════ */
/* How fast can lane 0 ISSUE (not complete) N TMA loads back-to-back?
   Measures dispatch overhead of cp.async.bulk.tensor instruction. */

__global__ __launch_bounds__(32, 1)
void k_tma_issue_rate(__grid_constant__ const CUtensorMap tm, int bytes, long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    int lane = threadIdx.x;

    /* Init 8 mbarriers */
    if (lane == 0)
        for (int i = 0; i < 8; i++) mb_init(sb + MBAR_OFF + i*8, 1);
    __syncwarp();

    /* Warmup: issue 8 TMAs, wait all */
    int ph = 0;
    for (int w = 0; w < WARMUP; w++) {
        for (int i = 0; i < 8; i++) {
            uint32_t m = sb + MBAR_OFF + i*8;
            if (lane == 0) { mb_expect_tx(m, bytes); tma_ld(sb + TMA_BUF + i*4096, &tm, 0, 0, m); }
        }
        for (int i = 0; i < 8; i++) mb_wait(sb + MBAR_OFF + i*8, ph);
        ph ^= 1;
    }

    /* Measure: time to ISSUE 8 TMAs (not wait for completion) */
    long long issue_tot = 0, complete_tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        for (int i = 0; i < 8; i++) {
            uint32_t m = sb + MBAR_OFF + i*8;
            if (lane == 0) mb_expect_tx(m, bytes);
        }
        long long t0 = clk();
        if (lane == 0) {
            #pragma unroll
            for (int i = 0; i < 8; i++)
                tma_ld(sb + TMA_BUF + i*4096, &tm, 0, 0, sb + MBAR_OFF + i*8);
        }
        long long t1 = clk();
        for (int i = 0; i < 8; i++) mb_wait(sb + MBAR_OFF + i*8, ph);
        long long t2 = clk();
        issue_tot += t1 - t0;
        complete_tot += t2 - t0;
        ph ^= 1;
    }
    if (lane == 0) { out[0] = issue_tot; out[1] = complete_tot; }
}

/* ═══════ S. fence.proxy.async overhead ═══════ */

__global__ __launch_bounds__(32, 1)
void k_fence_overhead(long long *out) {
    long long tot = 0;
    for (int i = 0; i < WARMUP; i++)
        fence_async();
    for (int i = 0; i < REPS; i++) {
        long long t0 = clk();
        fence_async();
        long long t1 = clk();
        tot += t1 - t0;
    }
    if (threadIdx.x == 0) out[0] = tot;
}

/* ═══════ T. Mbarrier breakdown: arrive vs polling ═══════ */

__global__ __launch_bounds__(32, 1)
void k_mbar_arrive_only(long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    int lane = threadIdx.x;
    /* Use 2 mbarriers alternating, just measure arrive cost (no wait) */
    uint32_t mb0 = sb + MBAR_OFF, mb1 = sb + MBAR_OFF + 8;
    if (lane == 0) { mb_init(mb0, 1); mb_init(mb1, 1); }
    __syncwarp();

    /* arrive + immediate wait on a trivial barrier (no TX) */
    int ph = 0;
    for (int i = 0; i < WARMUP; i++) {
        if (lane == 0)
            asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(mb0) : "memory");
        mb_wait(mb0, ph); ph ^= 1;
    }

    /* Measure just the arrive instruction (separate from wait) */
    long long arrive_tot = 0, wait_tot = 0;
    for (int i = 0; i < REPS; i++) {
        long long t0 = clk();
        if (lane == 0)
            asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(mb0) : "memory");
        long long t1 = clk();
        mb_wait(mb0, ph);
        long long t2 = clk();
        arrive_tot += t1 - t0;
        wait_tot += t2 - t1;
        ph ^= 1;
    }
    if (lane == 0) { out[0] = arrive_tot; out[1] = wait_tot; }
}

/* ═══════ U. LDS bank conflict sweep ═══════ */
/* Vary LDS stride to test bank conflict sensitivity.
   Stride 16B = 1 bank gap (conflict-free for 32 lanes).
   Stride 32B = 2 bank gap. Stride 4B = same bank (max conflict). */

template <int STRIDE>
__global__ __launch_bounds__(32, 1)
void k_lds_stride(long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    int lane = threadIdx.x;

    /* Fill enough SMEM for all strides */
    for (int i = 0; i < 64; i++) {
        uint32_t a = sb + WORK_OFF + i * 64 + (lane % 4) * 16;
        uint32_t v = 0x3c003c00u | lane;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                     :: "r"(a), "r"(v), "r"(v), "r"(v), "r"(v) : "memory");
    }
    __syncwarp();

    uint32_t addr = sb + WORK_OFF + lane * STRIDE;
    uint32_t r0, r1, r2, r3, acc = 0;

    for (int i = 0; i < WARMUP; i++) {
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr)); acc ^= r0;
    }

    long long tot = 0;
    for (int rep = 0; rep < REPS; rep++) {
        long long t0 = clk();
        #pragma unroll 1
        for (int i = 0; i < LDS_OPS; i++) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr)); acc ^= r0;
        }
        long long t1 = clk();
        tot += t1 - t0;
    }
    if (lane == 0) out[0] = tot;
    if (acc == 0xDEAD) out[1] = (long long)acc;
}

/* ═══════ V. Multi-SM TMA bandwidth ═══════ */
/* Launch on all SMs simultaneously to test system-level TMA bandwidth.
   Each block does serial TMA loads; total bandwidth = per-SM × N_SM. */

__global__ __launch_bounds__(32, 1)
void k_tma_multi_sm(__grid_constant__ const CUtensorMap tm, int bytes, int box_y, long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    int lane = threadIdx.x;
    uint32_t mb = sb + MBAR_OFF, buf = sb + TMA_BUF;
    int bid = blockIdx.x;

    if (lane == 0) mb_init(mb, 1);
    __syncwarp();

    int ph = 0;
    /* Use different y coords per block to avoid L2 sharing */
    int y_base = (bid * box_y * 4) % (GH - box_y);

    for (int i = 0; i < WARMUP; i++) {
        int y = (y_base + i * box_y) % (GH - box_y);
        if (lane == 0) { mb_expect_tx(mb, bytes); tma_ld(buf, &tm, 0, y, mb); }
        mb_wait(mb, ph); ph ^= 1;
    }

    long long tot = 0;
    for (int i = 0; i < REPS; i++) {
        int y = (y_base + (WARMUP + i) * box_y) % (GH - box_y);
        if (lane == 0) mb_expect_tx(mb, bytes);
        long long t0 = clk();
        if (lane == 0) tma_ld(buf, &tm, 0, y, mb);
        mb_wait(mb, ph);
        long long t1 = clk();
        tot += t1 - t0; ph ^= 1;
    }
    /* Block 0 reports; all blocks write to separate slots */
    if (lane == 0) out[bid] = tot;
}

/* ═══════ Host ═══════ */

void make_tmap(CUtensorMap *tm, nv_bfloat16 *ptr, int gw, int gh,
               int box_x, int box_y, CUtensorMapSwizzle swizzle) {
    uint64_t dims[2] = {(uint64_t)gw, (uint64_t)gh};
    uint64_t strides[1] = {(uint64_t)gw * sizeof(nv_bfloat16)};
    uint32_t box[2] = {(uint32_t)box_x, (uint32_t)box_y};
    uint32_t estrides[2] = {1, 1};
    CU_CHECK(cuTensorMapEncodeTiled(tm,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr,
        dims, strides, box, estrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));

    /* Global data: warm (2MB) + cold (64MB) */
    nv_bfloat16 *d_warm, *d_cold, *d_store;
    CUDA_CHECK(cudaMalloc(&d_warm, (size_t)GW * GH * 2));
    CUDA_CHECK(cudaMemset(d_warm, 0x3c, (size_t)GW * GH * 2));
    CUDA_CHECK(cudaMalloc(&d_cold, (size_t)GW * COLD_GH * 2));
    CUDA_CHECK(cudaMemset(d_cold, 0x3c, (size_t)GW * COLD_GH * 2));
    CUDA_CHECK(cudaMalloc(&d_store, (size_t)GW * GH * 2));

    /* Tensor maps — size sweep (SWIZZLE_128B, box_x=64, vary box_y) */
    struct SizeConfig { int box_x, box_y, bytes; const char *name; CUtensorMap tm; };
    SizeConfig sizes[] = {
        {64,  4,   512, " 512B"}, {64,  8,  1024, "  1KB"},
        {64, 16,  2048, "  2KB"}, {64, 32,  4096, "  4KB"},
        {64, 64,  8192, "  8KB"},
    };
    for (auto &s : sizes) {
        s.bytes = s.box_x * 2 * s.box_y;
        make_tmap(&s.tm, d_warm, GW, GH, s.box_x, s.box_y, CU_TENSOR_MAP_SWIZZLE_128B);
    }

    /* Tensor maps — swizzle sweep (4KB, vary swizzle + box_x) */
    struct SwizzleConfig {
        int box_x, box_y, bytes; CUtensorMapSwizzle sw; const char *name; CUtensorMap tm;
    };
    SwizzleConfig swizzles[] = {
        {64, 32, 4096, CU_TENSOR_MAP_SWIZZLE_NONE,  " NONE"},
        {16, 128,4096, CU_TENSOR_MAP_SWIZZLE_32B,   "  32B"},
        {32, 64, 4096, CU_TENSOR_MAP_SWIZZLE_64B,   "  64B"},
        {64, 32, 4096, CU_TENSOR_MAP_SWIZZLE_128B,  " 128B"},
    };
    for (auto &s : swizzles)
        make_tmap(&s.tm, d_warm, GW, GH, s.box_x, s.box_y, s.sw);

    /* Cold data tensor map */
    CUtensorMap tm_cold;
    make_tmap(&tm_cold, d_cold, GW, COLD_GH, 64, 32, CU_TENSOR_MAP_SWIZZLE_128B);

    /* Store tensor map */
    CUtensorMap tm_store;
    make_tmap(&tm_store, d_store, GW, GH, 64, 32, CU_TENSOR_MAP_SWIZZLE_128B);

    /* Default 4KB/128B map for baselines */
    CUtensorMap &tmap = sizes[3].tm;
    int default_bytes = 4096;

    /* Set max dynamic SMEM for all kernels */
    auto set_smem = [](const void *fn) {
        CUDA_CHECK(cudaFuncSetAttribute(fn,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
    };
    set_smem((const void*)k_tma_lat);
    set_smem((const void*)k_tma_lat_cold);
    set_smem((const void*)k_tma_pipe<1>);
    set_smem((const void*)k_tma_pipe<2>);
    set_smem((const void*)k_tma_pipe<4>);
    set_smem((const void*)k_tma_pipe<8>);
    set_smem((const void*)k_tma_pipe<16>);
    set_smem((const void*)k_lds);
    set_smem((const void*)k_sts);
    set_smem((const void*)k_lds_sts);
    set_smem((const void*)k_tma_lds);
    set_smem((const void*)k_tma_sts);
    set_smem((const void*)k_tma_lds_sts);
    set_smem((const void*)k_pipe_tma_lds<2>);
    set_smem((const void*)k_pipe_tma_lds<4>);
    set_smem((const void*)k_pipe_tma_lds<8>);
    set_smem((const void*)k_tma_store_lat);
    set_smem((const void*)k_tma_store_lds);
    set_smem((const void*)k_multi_tma_lds<2>);
    set_smem((const void*)k_multi_tma_lds<4>);
    set_smem((const void*)k_multi_tma_lds<6>);
    set_smem((const void*)k_tma_read_back);
    set_smem((const void*)k_dbuf_tma_lds);
    set_smem((const void*)k_dual_tma);
    set_smem((const void*)k_mbar_overhead);
    set_smem((const void*)k_lds_ilp<1>);
    set_smem((const void*)k_lds_ilp<2>);
    set_smem((const void*)k_lds_ilp<4>);
    set_smem((const void*)k_lds_ilp<7>);
    set_smem((const void*)k_lds_ilp<8>);
    set_smem((const void*)k_lds_ilp<16>);
    set_smem((const void*)k_sts_bf16<0>);
    set_smem((const void*)k_sts_bf16<1>);
    set_smem((const void*)k_sts_bf16<2>);
    set_smem((const void*)k_sts_bf16<4>);
    set_smem((const void*)k_sts_bf16<8>);
    set_smem((const void*)k_sts_bf16<15>);
    set_smem((const void*)k_epilogue_fragment);
    set_smem((const void*)k_tma_issue_rate);
    set_smem((const void*)k_fence_overhead);
    set_smem((const void*)k_mbar_arrive_only);
    set_smem((const void*)k_lds_stride<4>);
    set_smem((const void*)k_lds_stride<16>);
    set_smem((const void*)k_lds_stride<32>);
    set_smem((const void*)k_lds_stride<64>);
    set_smem((const void*)k_lds_stride<128>);
    set_smem((const void*)k_tma_multi_sm);

    long long *d_out, h[256];  /* enough for multi-SM (up to 148 blocks) */
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(h)));
    auto reset = [&]() { CUDA_CHECK(cudaMemset(d_out, 0, sizeof(h))); };
    const char *cur_test = "";
    auto sync = [&]() {
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { fprintf(stderr, "launch error [%s]: %s\n", cur_test, cudaGetErrorString(e)); exit(1); }
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { fprintf(stderr, "kernel error [%s]: %s\n", cur_test, cudaGetErrorString(e)); exit(1); }
        CUDA_CHECK(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
    };

    printf("TMA Comprehensive Microbenchmark — SM100a\n");
    printf("REPS=%d, WARMUP=%d\n\n", REPS, WARMUP);

    /* ═══ A. TMA LOAD LATENCY ═══ */
    printf("═══ A. TMA Load Latency (serial, L2-warm) ═══\n\n");
    cur_test = "A.tma_lat";
    reset(); k_tma_lat<<<1, 32, SMEM_MAX>>>(tmap, default_bytes, d_out); sync();
    double tma_lat = (double)h[0] / REPS;
    printf("  4KB (64×32 BF16, SW128):     %8.1f cyc  (%.1f B/cyc)\n\n", tma_lat, 4096.0 / tma_lat);

    /* ═══ B. PIPELINE DEPTH SWEEP ═══ */
    printf("═══ B. Pipeline Depth Sweep (4KB, SWIZZLE_128B) ═══\n\n");
    for (int d : {1, 2, 4, 8, 16}) {
        cur_test = "B.tma_pipe";
        reset();
        int stage = 4096;
        switch(d) {
            case 1:  k_tma_pipe<1> <<<1, 32, SMEM_MAX>>>(tmap, default_bytes, stage, 32, d_out); break;
            case 2:  k_tma_pipe<2> <<<1, 32, SMEM_MAX>>>(tmap, default_bytes, stage, 32, d_out); break;
            case 4:  k_tma_pipe<4> <<<1, 32, SMEM_MAX>>>(tmap, default_bytes, stage, 32, d_out); break;
            case 8:  k_tma_pipe<8> <<<1, 32, SMEM_MAX>>>(tmap, default_bytes, stage, 32, d_out); break;
            case 16: k_tma_pipe<16><<<1, 32, SMEM_MAX>>>(tmap, default_bytes, stage, 32, d_out); break;
        }
        sync();
        double cpl = (double)h[0] / h[1];
        printf("  pipe=%2d:  %8.1f cyc/load  (%.1f B/cyc, %.1f%% of serial)\n",
               d, cpl, 4096.0 / cpl, 100.0 * cpl / tma_lat);
    }

    /* ═══ C. TRANSFER SIZE SWEEP ═══ */
    printf("\n═══ C. Transfer Size Sweep (SWIZZLE_128B, serial) ═══\n\n");
    for (auto &s : sizes) {
        cur_test = s.name;
        reset(); k_tma_lat<<<1, 32, SMEM_MAX>>>(s.tm, s.bytes, d_out); sync();
        double lat = (double)h[0] / REPS;
        printf("  %s (%2d×%3d):  %8.1f cyc  (%.1f B/cyc)\n",
               s.name, s.box_x, s.box_y, lat, s.bytes / lat);
    }

    /* ═══ D. SWIZZLE MODE SWEEP ═══ */
    printf("\n═══ D. Swizzle Mode Sweep (4KB, serial latency + TMA+LDS contention) ═══\n\n");
    for (auto &s : swizzles) {
        cur_test = s.name;
        reset(); k_tma_lat<<<1, 32, SMEM_MAX>>>(s.tm, s.bytes, d_out); sync();
        double lat = (double)h[0] / REPS;
        reset(); k_tma_lds<<<1, 32, SMEM_MAX>>>(s.tm, s.bytes, d_out); sync();
        double lds_w = (double)h[0] / REPS;
        printf("  SW%s:  lat=%6.1f cyc  TMA+LDS: LDS=%.1f cyc (ratio vs SW128 baseline)\n",
               s.name, lat, lds_w);
    }

    /* ═══ E. LSU BASELINES ═══ */
    printf("\n═══ E. LSU Baselines (single warp, no TMA) ═══\n\n");
    cur_test = "E.lds";
    reset(); k_lds<<<1, 32, SMEM_MAX>>>(d_out); sync();
    double lds_base = (double)h[0] / REPS;
    printf("  LDS.128:    %8.1f cyc  (%d ops, %.2f cyc/op)\n", lds_base, LDS_OPS, lds_base / LDS_OPS);

    cur_test = "E.sts";
    reset(); k_sts<<<1, 32, SMEM_MAX>>>(d_out); sync();
    double sts_base = (double)h[0] / REPS;
    printf("  STS.128:    %8.1f cyc  (%d ops, %.2f cyc/op)\n", sts_base, STS_OPS, sts_base / STS_OPS);

    cur_test = "E.lds_sts";
    reset(); k_lds_sts<<<1, 32, SMEM_MAX>>>(d_out); sync();
    double lds_sts_time = (double)h[0] / REPS;
    int ls_lds = (int)h[1], ls_sts = (int)h[2];
    double expected = ls_lds * (lds_base / LDS_OPS) + ls_sts * (sts_base / STS_OPS);
    printf("  LDS+STS:    %8.1f cyc  (%d LDS + %d STS, %.1f%% vs independent)\n",
           lds_sts_time, ls_lds, ls_sts, 100.0 * (lds_sts_time / expected - 1.0));

    /* ═══ F. SINGLE-WARP CONTENTION ═══ */
    printf("\n═══ F. Single-Warp Contention (TMA load 4KB + LSU) ═══\n\n");

    cur_test = "F.tma_lds";
    reset(); k_tma_lds<<<1, 32, SMEM_MAX>>>(tmap, default_bytes, d_out); sync();
    double lds_w_tma = (double)h[0] / REPS, tma_w_lds = (double)h[1] / REPS;
    printf("  TMA+LDS:  LDS=%7.1f (base=%7.1f, r=%.3f)  TMA=%7.1f (base=%7.1f, r=%.3f)\n",
           lds_w_tma, lds_base, lds_w_tma/lds_base, tma_w_lds, tma_lat, tma_w_lds/tma_lat);

    cur_test = "F.tma_sts";
    reset(); k_tma_sts<<<1, 32, SMEM_MAX>>>(tmap, default_bytes, d_out); sync();
    double sts_w_tma = (double)h[0] / REPS, tma_w_sts = (double)h[1] / REPS;
    printf("  TMA+STS:  STS=%7.1f (base=%7.1f, r=%.3f)  TMA=%7.1f (base=%7.1f, r=%.3f)\n",
           sts_w_tma, sts_base, sts_w_tma/sts_base, tma_w_sts, tma_lat, tma_w_sts/tma_lat);

    cur_test = "F.tma_lds_sts";
    reset(); k_tma_lds_sts<<<1, 32, SMEM_MAX>>>(tmap, default_bytes, d_out); sync();
    double work_w = (double)h[0] / REPS, tma_w_all = (double)h[1] / REPS;
    printf("  TMA+ALL:  work=%7.1f (base=%7.1f, r=%.3f)  TMA=%7.1f (base=%7.1f, r=%.3f)\n",
           work_w, lds_sts_time, work_w/lds_sts_time, tma_w_all, tma_lat, tma_w_all/tma_lat);

    /* ═══ G. PIPELINED TMA + LDS ═══ */
    printf("\n═══ G. Pipelined TMA + LDS (multiple TMA in flight during LDS) ═══\n\n");
    for (int d : {2, 4, 8}) {
        cur_test = "G.pipe_tma_lds";
        reset();
        switch(d) {
            case 2: k_pipe_tma_lds<2><<<1, 32, SMEM_MAX>>>(tmap, default_bytes, 4096, 32, d_out); break;
            case 4: k_pipe_tma_lds<4><<<1, 32, SMEM_MAX>>>(tmap, default_bytes, 4096, 32, d_out); break;
            case 8: k_pipe_tma_lds<8><<<1, 32, SMEM_MAX>>>(tmap, default_bytes, 4096, 32, d_out); break;
        }
        sync();
        double lds_p = (double)h[0] / REPS, tma_p = (double)h[1] / REPS;
        printf("  pipe=%d:  LDS=%7.1f (base=%7.1f, r=%.3f)  TMA=%7.1f\n",
               d, lds_p, lds_base, lds_p/lds_base, tma_p);
    }

    /* ═══ H. TMA STORE ═══ */
    printf("\n═══ H. TMA Store (SMEM→GMEM, 4KB) ═══\n\n");
    cur_test = "H.tma_store_lat";
    reset(); k_tma_store_lat<<<1, 32, SMEM_MAX>>>(tm_store, default_bytes, d_out); sync();
    double store_lat = (double)h[0] / REPS;
    printf("  Store latency:   %8.1f cyc\n", store_lat);

    cur_test = "H.tma_store_lds";
    reset(); k_tma_store_lds<<<1, 32, SMEM_MAX>>>(tm_store, default_bytes, d_out); sync();
    double lds_w_store = (double)h[0] / REPS, store_w_lds = (double)h[1] / REPS;
    printf("  Store+LDS:  LDS=%7.1f (base=%7.1f, r=%.3f)  Store=%7.1f (base=%7.1f, r=%.3f)\n",
           lds_w_store, lds_base, lds_w_store/lds_base, store_w_lds, store_lat, store_w_lds/store_lat);

    /* ═══ I. MULTI-WARP ═══ */
    printf("\n═══ I. Multi-Warp (W0=TMA, rest=LDS, 4KB) ═══\n\n");
    for (int nw : {2, 4, 6}) {
        cur_test = "I.multi_tma_lds";
        reset();
        switch(nw) {
            case 2: k_multi_tma_lds<2><<<1, nw*32, SMEM_MAX>>>(tmap, default_bytes, d_out); break;
            case 4: k_multi_tma_lds<4><<<1, nw*32, SMEM_MAX>>>(tmap, default_bytes, d_out); break;
            case 6: k_multi_tma_lds<6><<<1, nw*32, SMEM_MAX>>>(tmap, default_bytes, d_out); break;
        }
        sync();
        double tma_m = (double)h[0] / REPS, lds_m = (double)h[1] / REPS;
        printf("  %dW (%d LDS warps):  LDS=%7.1f (1W base=%7.1f, r=%.3f)  TMA=%7.1f (r=%.3f)\n",
               nw, nw-1, lds_m, lds_base, lds_m/lds_base, tma_m, tma_m/tma_lat);
    }

    /* ═══ J. COLD DATA ═══ */
    printf("\n═══ J. Cold Data (64MB tensor, L2 miss, 4KB) ═══\n\n");
    cur_test = "J.cold";
    reset(); k_tma_lat_cold<<<1, 32, SMEM_MAX>>>(tm_cold, default_bytes, 32, d_out); sync();
    double cold_lat = (double)h[0] / REPS;
    printf("  Cold TMA latency:  %8.1f cyc  (warm: %.1f, ratio: %.1fx)\n",
           cold_lat, tma_lat, cold_lat / tma_lat);

    /* ═══ K. LDS FROM TMA BUFFER ═══ */
    printf("\n═══ K. LDS from TMA Buffer (read-back, same swizzled region) ═══\n\n");
    cur_test = "K.read_back";
    reset(); k_tma_read_back<<<1, 32, SMEM_MAX>>>(tmap, default_bytes, d_out); sync();
    double lds_from_tma = (double)h[0] / REPS;
    printf("  LDS from TMA buf:  %8.1f cyc  (%d ops, %.2f cyc/op)\n", lds_from_tma, LDS_OPS, lds_from_tma / LDS_OPS);
    printf("  LDS from init buf: %8.1f cyc  (ratio: %.3f)\n", lds_base, lds_from_tma / lds_base);

    /* ═══ L. DOUBLE-BUFFERED TMA + LDS ═══ */
    printf("\n═══ L. Double-Buffered TMA + LDS (exact FC2 pattern) ═══\n\n");
    cur_test = "L.dbuf_tma_lds";
    reset(); k_dbuf_tma_lds<<<1, 32, SMEM_MAX>>>(tmap, default_bytes, 4096, d_out); sync();
    double dbuf_lds = (double)h[0] / REPS, dbuf_tma = (double)h[1] / REPS;
    printf("  LDS (from prev TMA buf): %7.1f (base=%7.1f, r=%.3f)\n",
           dbuf_lds, lds_base, dbuf_lds / lds_base);
    printf("  TMA (concurrent):        %7.1f (base=%7.1f, r=%.3f)\n",
           dbuf_tma, tma_lat, dbuf_tma / tma_lat);

    /* ═══ M. DUAL TMA ═══ */
    printf("\n═══ M. Dual TMA: Load + Store Simultaneously ═══\n\n");
    cur_test = "M.dual_tma";
    reset(); k_dual_tma<<<1, 32, SMEM_MAX>>>(tmap, tm_store, default_bytes, d_out); sync();
    double dual_tma = (double)h[0] / REPS;
    printf("  Dual TMA (load+store):  %8.1f cyc  (load-only: %.1f, ratio: %.3f)\n",
           dual_tma, tma_lat, dual_tma / tma_lat);

    /* ═══ N. MBARRIER OVERHEAD ═══ */
    printf("\n═══ N. Mbarrier Overhead (arrive+wait, no TMA/LSU) ═══\n\n");
    cur_test = "N.mbar";
    reset(); k_mbar_overhead<<<1, 32, SMEM_MAX>>>(d_out); sync();
    double mbar_cost = (double)h[0] / REPS;
    printf("  Mbarrier arrive+wait:  %8.1f cyc\n", mbar_cost);

    /* ═══ O. LDS ILP SWEEP ═══ */
    printf("\n═══ O. LDS.128 ILP Sweep (latency → throughput) ═══\n\n");
    for (int ilp : {1, 2, 4, 7, 8, 16}) {
        cur_test = "O.lds_ilp";
        reset();
        switch(ilp) {
            case 1:  k_lds_ilp<1> <<<1, 32, SMEM_MAX>>>(d_out); break;
            case 2:  k_lds_ilp<2> <<<1, 32, SMEM_MAX>>>(d_out); break;
            case 4:  k_lds_ilp<4> <<<1, 32, SMEM_MAX>>>(d_out); break;
            case 7:  k_lds_ilp<7> <<<1, 32, SMEM_MAX>>>(d_out); break;
            case 8:  k_lds_ilp<8> <<<1, 32, SMEM_MAX>>>(d_out); break;
            case 16: k_lds_ilp<16><<<1, 32, SMEM_MAX>>>(d_out); break;
        }
        sync();
        double t = (double)h[0] / REPS;
        printf("  ILP=%2d:  %8.1f cyc  (512 ops, %.2f cyc/op)\n", ilp, t, t / 512.0);
    }

    /* ═══ P. STS + BF16 INTERLEAVING ═══ */
    printf("\n═══ P. STS.128 + BF16 Interleaving (does BF16 hide in STS shadow?) ═══\n\n");
    for (int bf : {0, 1, 2, 4, 8, 15}) {
        cur_test = "P.sts_bf16";
        reset();
        switch(bf) {
            case 0:  k_sts_bf16<0> <<<1, 32, SMEM_MAX>>>(d_out); break;
            case 1:  k_sts_bf16<1> <<<1, 32, SMEM_MAX>>>(d_out); break;
            case 2:  k_sts_bf16<2> <<<1, 32, SMEM_MAX>>>(d_out); break;
            case 4:  k_sts_bf16<4> <<<1, 32, SMEM_MAX>>>(d_out); break;
            case 8:  k_sts_bf16<8> <<<1, 32, SMEM_MAX>>>(d_out); break;
            case 15: k_sts_bf16<15><<<1, 32, SMEM_MAX>>>(d_out); break;
        }
        sync();
        double t = (double)h[0] / REPS;
        double per_sts = t / 64.0;
        printf("  STS + %2d×HADD2:  %8.1f cyc  (64 STS, %.2f cyc/STS, %.1f%% vs STS-only)\n",
               bf, t, per_sts, 100.0 * (per_sts / (sts_base / STS_OPS) - 1.0));
    }

    /* ═══ Q. EPILOGUE FRAGMENT ═══ */
    printf("\n═══ Q. FC2 Epilogue Fragment (7×LDS → HADD2 → F2FP → 4×STS, 48 groups) ═══\n\n");
    cur_test = "Q.epilogue";
    reset(); k_epilogue_fragment<<<1, 32, SMEM_MAX>>>(d_out); sync();
    double epi_time = (double)h[0] / REPS;
    int epi_groups = (int)h[1];
    printf("  Total:     %8.1f cyc  (%d groups, %.1f cyc/group)\n", epi_time, epi_groups, epi_time / epi_groups);
    printf("  Per group: 7 LDS + 2 BF16 + 4 STS = 13 insns\n");

    /* ═══ R. TMA ISSUE RATE ═══ */
    printf("\n═══ R. TMA Issue Rate (8 back-to-back TMA loads, dispatch overhead) ═══\n\n");
    cur_test = "R.tma_issue";
    reset(); k_tma_issue_rate<<<1, 32, SMEM_MAX>>>(tmap, default_bytes, d_out); sync();
    double issue_time = (double)h[0] / REPS;
    double complete_time = (double)h[1] / REPS;
    printf("  Issue 8 TMAs:    %8.1f cyc  (%.1f cyc/issue)\n", issue_time, issue_time / 8.0);
    printf("  Complete all:    %8.1f cyc  (%.1f cyc/load incl wait)\n", complete_time, complete_time / 8.0);

    /* ═══ S. FENCE OVERHEAD ═══ */
    printf("\n═══ S. fence.proxy.async Overhead ═══\n\n");
    cur_test = "S.fence";
    reset(); k_fence_overhead<<<1, 32, SMEM_MAX>>>(d_out); sync();
    double fence_cost = (double)h[0] / REPS;
    printf("  fence.proxy.async:  %8.1f cyc\n", fence_cost);

    /* ═══ T. MBARRIER BREAKDOWN ═══ */
    printf("\n═══ T. Mbarrier Breakdown (arrive vs try_wait polling) ═══\n\n");
    cur_test = "T.mbar_breakdown";
    reset(); k_mbar_arrive_only<<<1, 32, SMEM_MAX>>>(d_out); sync();
    double arrive_cost = (double)h[0] / REPS;
    double wait_cost = (double)h[1] / REPS;
    printf("  Arrive:    %8.1f cyc\n", arrive_cost);
    printf("  Wait:      %8.1f cyc\n", wait_cost);
    printf("  Total:     %8.1f cyc  (section N measured: %.1f)\n", arrive_cost + wait_cost, mbar_cost);

    /* ═══ U. LDS BANK CONFLICT SWEEP ═══ */
    printf("\n═══ U. LDS.128 Bank Conflict Sweep (stride per lane) ═══\n\n");
    for (int stride : {4, 16, 32, 64, 128}) {
        cur_test = "U.lds_stride";
        reset();
        switch(stride) {
            case 4:   k_lds_stride<4>  <<<1, 32, SMEM_MAX>>>(d_out); break;
            case 16:  k_lds_stride<16> <<<1, 32, SMEM_MAX>>>(d_out); break;
            case 32:  k_lds_stride<32> <<<1, 32, SMEM_MAX>>>(d_out); break;
            case 64:  k_lds_stride<64> <<<1, 32, SMEM_MAX>>>(d_out); break;
            case 128: k_lds_stride<128><<<1, 32, SMEM_MAX>>>(d_out); break;
        }
        sync();
        double t = (double)h[0] / REPS;
        printf("  stride=%3dB:  %8.1f cyc  (512 ops, %.2f cyc/op)\n", stride, t, t / 512.0);
    }

    /* ═══ V. MULTI-SM TMA BANDWIDTH ═══ */
    printf("\n═══ V. Multi-SM TMA Bandwidth (all SMs, serial loads) ═══\n\n");
    {
        int n_sm = 0;
        cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, 0);
        if (n_sm > 256) n_sm = 256;  /* cap to output buffer */
        if (n_sm > 0) {
            cur_test = "V.multi_sm";
            CUDA_CHECK(cudaMemset(d_out, 0, n_sm * sizeof(long long)));
            k_tma_multi_sm<<<n_sm, 32, SMEM_MAX>>>(tmap, default_bytes, 32, d_out);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h, d_out, n_sm * sizeof(long long), cudaMemcpyDeviceToHost));
            double sum = 0, mn = 1e18, mx = 0;
            for (int i = 0; i < n_sm; i++) {
                double t = (double)h[i] / REPS;
                sum += t; if (t < mn) mn = t; if (t > mx) mx = t;
            }
            double avg = sum / n_sm;
            double bw_per_sm = 4096.0 / avg;  /* B/cyc per SM */
            double bw_total = bw_per_sm * n_sm;
            printf("  %d SMs:  avg=%.1f cyc/load  min=%.1f  max=%.1f\n", n_sm, avg, mn, mx);
            printf("  Per-SM: %.1f B/cyc  Total: %.1f B/cyc (%.1f GB/s @ 2.1GHz)\n",
                   bw_per_sm, bw_total, bw_total * 2.1);
        }
    }

    /* ═══ SUMMARY ═══ */
    printf("\n═══ SUMMARY ═══\n");
    printf("TMA load (4KB, L2-warm):  %.1f cyc  (mbarrier overhead: %.1f cyc)\n", tma_lat, mbar_cost);
    printf("TMA store (4KB):          %.1f cyc\n", store_lat);
    printf("TMA load (4KB, L2-cold):  %.1f cyc  (%.1fx warm)\n", cold_lat, cold_lat / tma_lat);
    printf("Dual TMA (load+store):    %.1f cyc  (%.1fx load-only)\n", dual_tma, dual_tma / tma_lat);
    printf("\nContention ratios (1.0 = independent, >1.0 = port sharing):\n");
    printf("  TMA_load + LDS:         LDS %.3f  TMA %.3f\n", lds_w_tma/lds_base, tma_w_lds/tma_lat);
    printf("  TMA_load + STS:         STS %.3f  TMA %.3f\n", sts_w_tma/sts_base, tma_w_sts/tma_lat);
    printf("  TMA_load + ALL:        work %.3f  TMA %.3f\n", work_w/lds_sts_time, tma_w_all/tma_lat);
    printf("  TMA_store + LDS:        LDS %.3f  store %.3f\n", lds_w_store/lds_base, store_w_lds/store_lat);
    printf("  Dbuf TMA + LDS (FC2):   LDS %.3f  TMA %.3f\n", dbuf_lds/lds_base, dbuf_tma/tma_lat);
    printf("  LDS from TMA buf:       %.3f (vs init buf)\n", lds_from_tma/lds_base);

    cudaFree(d_warm); cudaFree(d_cold); cudaFree(d_store); cudaFree(d_out);
    return 0;
}
