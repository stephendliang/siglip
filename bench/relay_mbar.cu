/*
Approach (2) proof-of-concept: 1D multicast + manual mbarrier relay

Proves that CTA0 can:
  1. Issue cp.async.bulk.shared::cluster.global...multicast::cluster
     (data deposited into BOTH CTAs' SMEM)
  2. Wait on LOCAL mbar (HW completion for issuing CTA)
  3. Issue mbarrier.complete_tx.relaxed.cluster.shared::cluster to PEER mbar
  4. PEER CTA waits on its own local mbar and wakes from the relay

Three test stages:
  0. Smoke test: just complete_tx to peer mbar (no multicast), verify it works
  A. Correctness: CTA0 multicast loads + relay → CTA1 verifies data
  B. Latency: pipelined relay vs cta_group::2 tensor (reference)

Build: make relay-mbar && ./relay-mbar

2-CTA cluster required (sm_100a).
*/

#include <cstdio>
#include <cstdint>
#include <cuda.h>

#define WARMUP 128
#define REPS   1024
#ifndef XFER
#define XFER   16384   /* 16KB = FC2 A-tile size */
#endif

#define CU_CHECK(x) do { CUresult _r = (x); if (_r) { const char *_s; cuGetErrorString(_r, &_s); fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, _s); exit(1); } } while(0)
#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1); } } while(0)

/* SMEM layout per CTA:
   [0..N*8)           N mbars × 8B
   [512..512+N*XFER)  N buffers
   Auto-size MAX_PIPE to fit 228KB SMEM.
*/
#define MAX_PIPE_RAW ((228*1024 - 512) / XFER)
#define MAX_PIPE (MAX_PIPE_RAW > 4 ? 4 : MAX_PIPE_RAW)
#define MBAR_OFF 0
#define BUF_OFF  512
#define SMEM_BYTES (BUF_OFF + MAX_PIPE * XFER)

#define GCOLS 128
#define GROWS (XFER / GCOLS * MAX_PIPE * 4)

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

__device__ __forceinline__ void mb_complete_tx_cluster(uint32_t addr, int bytes) {
    asm volatile(
        "mbarrier.complete_tx.relaxed.cluster.shared::cluster.b64 [%0], %1;"
        :: "r"(addr), "r"(bytes) : "memory");
}

__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive;\n\t"
                 "barrier.cluster.wait;" ::: "memory");
}

/* ═══════ Kernel 0: Smoke test — just complete_tx to peer ═══════
   No multicast, no TMA. CTA0 manually signals CTA1's mbar.
   Proves mbarrier.complete_tx.shared::cluster works on SM100a. */

__global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void k_smoke(int *result) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    uint32_t mb_addr = sb + MBAR_OFF;
    int lane = threadIdx.x;

    uint32_t cta_rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    if (lane == 0) mb_init(mb_addr, 1);
    __syncwarp();
    cluster_sync();

    if (cta_rank == 0) {
        /* CTA0: manually signal CTA1's mbar */
        if (lane == 0) {
            uint32_t peer_mb = mb_addr ^ (1u << 24);
            mb_complete_tx_cluster(peer_mb, XFER);
        }
    } else {
        /* CTA1: expect bytes, wait for CTA0's signal */
        if (lane == 0) {
            mb_expect_tx(mb_addr, XFER);
            mb_wait(mb_addr, 0);
        }
        __syncwarp();
        if (lane == 0) *result = 1;  /* reached = success */
    }
    cluster_sync();
}

/* ═══════ Kernel A: Correctness test ═══════
   CTA0 loads known pattern with multicast, relays to CTA1.
   CTA1 waits on local mbar, verifies data.
   Single iteration, 1 warp per CTA. */

__global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void k_correctness(const uint8_t *src, int *result) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    uint32_t mb_addr = sb + MBAR_OFF;
    uint32_t buf = sb + BUF_OFF;
    int lane = threadIdx.x;

    uint32_t cta_rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    if (lane == 0) mb_init(mb_addr, 1);
    __syncwarp();
    cluster_sync();

    if (cta_rank == 0) {
        if (lane == 0) {
            mb_expect_tx(mb_addr, XFER);
            asm volatile(
                "cp.async.bulk.shared::cluster.global"
                ".mbarrier::complete_tx::bytes.multicast::cluster"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(buf), "l"((uint64_t)src), "r"(XFER),
                   "r"(mb_addr), "h"((uint16_t)0x3)
                : "memory");
            mb_wait(mb_addr, 0);
            /* Fence: ensure async proxy data is committed before relay */
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            uint32_t peer_mb = mb_addr ^ (1u << 24);
            mb_complete_tx_cluster(peer_mb, XFER);
        }
    } else {
        if (lane == 0) {
            mb_expect_tx(mb_addr, XFER);
            mb_wait(mb_addr, 0);
        }
        __syncwarp();
        /* Fence: data arrived via async proxy (multicast TMA) */
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

        int ok = 1;
        const uint8_t *my_buf = (const uint8_t *)(smem + BUF_OFF);
        for (int i = lane * (XFER / 32); i < (lane + 1) * (XFER / 32); i++) {
            if (my_buf[i] != src[i]) { ok = 0; break; }
        }
        unsigned ballot = __ballot_sync(0xFFFFFFFF, ok);
        if (lane == 0) *result = (ballot == 0xFFFFFFFF) ? 1 : 0;
    }
    cluster_sync();
}

/* ═══════ Kernel B: 1D multicast + relay (pipelined latency) ═══════ */

template <int PIPE>
__global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void k_relay(const uint8_t *src, long long *out_cta0, long long *out_cta1) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    uint32_t mb[MAX_PIPE], buf[MAX_PIPE];
    int ph[MAX_PIPE];
    int lane = threadIdx.x;
    uint32_t cta_rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    for (int i = 0; i < PIPE; i++) {
        mb[i] = sb + MBAR_OFF + i * 8;
        buf[i] = sb + BUF_OFF + i * XFER;
        ph[i] = 0;
        if (lane == 0) mb_init(mb[i], 1);
    }
    __syncwarp();
    cluster_sync();

    uint32_t peer_mb[MAX_PIPE];
    for (int i = 0; i < PIPE; i++)
        peer_mb[i] = mb[i] ^ (1u << 24);

    if (cta_rank == 0) {
        for (int i = 0; i < WARMUP; i++) {
            int s = i % PIPE;
            if (lane == 0) {
                mb_expect_tx(mb[s], XFER);
                uint64_t gaddr = (uint64_t)src + (uint64_t)(i % 64) * XFER;
                asm volatile(
                    "cp.async.bulk.shared::cluster.global"
                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(buf[s]), "l"(gaddr), "r"(XFER),
                       "r"(mb[s]), "h"((uint16_t)0x3)
                    : "memory");
            }
            mb_wait(mb[s], ph[s]); ph[s] ^= 1;
            if (lane == 0) mb_complete_tx_cluster(peer_mb[s], XFER);
        }

        long long t0 = clk();
        for (int i = 0; i < REPS; i++) {
            int s = i % PIPE;
            if (lane == 0) {
                mb_expect_tx(mb[s], XFER);
                uint64_t gaddr = (uint64_t)src + (uint64_t)(i % 64) * XFER;
                asm volatile(
                    "cp.async.bulk.shared::cluster.global"
                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(buf[s]), "l"(gaddr), "r"(XFER),
                       "r"(mb[s]), "h"((uint16_t)0x3)
                    : "memory");
            }
            mb_wait(mb[s], ph[s]); ph[s] ^= 1;
            if (lane == 0) mb_complete_tx_cluster(peer_mb[s], XFER);
        }
        long long t1 = clk();
        if (lane == 0) *out_cta0 = t1 - t0;
    } else {
        for (int i = 0; i < WARMUP; i++) {
            int s = i % PIPE;
            if (lane == 0) mb_expect_tx(mb[s], XFER);
            mb_wait(mb[s], ph[s]); ph[s] ^= 1;
        }

        long long t0 = clk();
        for (int i = 0; i < REPS; i++) {
            int s = i % PIPE;
            if (lane == 0) mb_expect_tx(mb[s], XFER);
            mb_wait(mb[s], ph[s]); ph[s] ^= 1;
        }
        long long t1 = clk();
        if (lane == 0) *out_cta1 = t1 - t0;
    }
    cluster_sync();
}

/* ═══════ Kernel C: tensor.2d + cta_group::2 reference ═══════
   Both CTAs issue loads (cta_group::2 requires both to participate).
   Each CTA uses its OWN mbar (count=1): expect_tx arrives at own mbar,
   cta_group::2 HW complete_tx goes to each CTA's referenced mbar.
   mb_wait is .shared::cta (CTA-local), so each CTA must satisfy its own. */

template <int PIPE>
__global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void k_tensor_cg2(const __grid_constant__ CUtensorMap tm, long long *out_cta0, long long *out_cta1) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    uint32_t mb[MAX_PIPE], buf[MAX_PIPE];
    int ph[MAX_PIPE];
    int lane = threadIdx.x;
    uint32_t cta_rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    int rows = XFER / GCOLS;

    for (int i = 0; i < PIPE; i++) {
        mb[i] = sb + MBAR_OFF + i * 8;
        buf[i] = sb + BUF_OFF + i * XFER;
        ph[i] = 0;
        if (lane == 0) mb_init(mb[i], 1);  /* count=1: own CTA's expect_tx only */
    }
    __syncwarp();
    cluster_sync();

    /* Each CTA's mbar in shared::cluster space (bit 24 = cta_rank) */
    uint32_t my_mb[MAX_PIPE];
    for (int i = 0; i < PIPE; i++)
        my_mb[i] = mb[i] | (cta_rank << 24);

    /* Both CTAs issue loads + expect_tx to their OWN mbar.
       cta_group::2 completes each CTA's referenced mbar independently. */
    for (int i = 0; i < WARMUP; i++) {
        int s = i % PIPE;
        if (lane == 0) {
            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                :: "r"(my_mb[s]), "r"(XFER) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
                ".mbarrier::complete_tx::bytes.cta_group::2"
                " [%0], [%1, {%2, %3}], [%4];"
                :: "r"(buf[s]), "l"(&tm), "r"(0), "r"((i * rows) % GROWS),
                   "r"(my_mb[s])
                : "memory");
        }
        mb_wait(mb[s], ph[s]); ph[s] ^= 1;
    }

    long long t0 = clk();
    for (int i = 0; i < REPS; i++) {
        int s = i % PIPE;
        if (lane == 0) {
            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                :: "r"(my_mb[s]), "r"(XFER) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
                ".mbarrier::complete_tx::bytes.cta_group::2"
                " [%0], [%1, {%2, %3}], [%4];"
                :: "r"(buf[s]), "l"(&tm), "r"(0), "r"((i * rows) % GROWS),
                   "r"(my_mb[s])
                : "memory");
        }
        mb_wait(mb[s], ph[s]); ph[s] ^= 1;
    }
    long long t1 = clk();

    if (cta_rank == 0) {
        if (lane == 0) *out_cta0 = t1 - t0;
    } else {
        if (lane == 0) *out_cta1 = t1 - t0;
    }
    cluster_sync();
}

/* ═══════ Kernel D: 1D bulk copy, per-CTA mbar (no cta_group, no multicast) ═══════
   Apples-to-apples vs k_tensor_cg2: same per-CTA mbar protocol (count=1),
   but plain cp.async.bulk.shared::cluster.global instead of tensor.2d cta_group::2.
   Each CTA independently loads XFER bytes to its own SMEM. No cross-CTA coordination
   in the load instruction — only the mbar tracks completion. */

template <int PIPE>
__global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void k_1d_percta(const uint8_t *src, long long *out_cta0, long long *out_cta1) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    uint32_t mb[MAX_PIPE], buf[MAX_PIPE];
    int ph[MAX_PIPE];
    int lane = threadIdx.x;
    uint32_t cta_rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    for (int i = 0; i < PIPE; i++) {
        mb[i] = sb + MBAR_OFF + i * 8;
        buf[i] = sb + BUF_OFF + i * XFER;
        ph[i] = 0;
        if (lane == 0) mb_init(mb[i], 1);
    }
    __syncwarp();
    cluster_sync();

    /* Per-CTA addresses in .shared::cluster space (bit 24 = cta_rank).
       Without cta_group::2, HW writes to the literal cluster address. */
    uint32_t my_mb[MAX_PIPE], my_buf[MAX_PIPE];
    for (int i = 0; i < PIPE; i++) {
        my_mb[i] = mb[i] | (cta_rank << 24);
        my_buf[i] = buf[i] | (cta_rank << 24);
    }

    for (int i = 0; i < WARMUP; i++) {
        int s = i % PIPE;
        if (lane == 0) {
            uint64_t gaddr = (uint64_t)src + (uint64_t)(i % 64) * XFER;
            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                :: "r"(my_mb[s]), "r"(XFER) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global"
                ".mbarrier::complete_tx::bytes"
                " [%0], [%1], %2, [%3];"
                :: "r"(my_buf[s]), "l"(gaddr), "r"(XFER),
                   "r"(my_mb[s])
                : "memory");
        }
        mb_wait(mb[s], ph[s]); ph[s] ^= 1;
    }

    long long t0 = clk();
    for (int i = 0; i < REPS; i++) {
        int s = i % PIPE;
        if (lane == 0) {
            uint64_t gaddr = (uint64_t)src + (uint64_t)(i % 64) * XFER;
            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                :: "r"(my_mb[s]), "r"(XFER) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global"
                ".mbarrier::complete_tx::bytes"
                " [%0], [%1], %2, [%3];"
                :: "r"(my_buf[s]), "l"(gaddr), "r"(XFER),
                   "r"(my_mb[s])
                : "memory");
        }
        mb_wait(mb[s], ph[s]); ph[s] ^= 1;
    }
    long long t1 = clk();

    if (cta_rank == 0) {
        if (lane == 0) *out_cta0 = t1 - t0;
    } else {
        if (lane == 0) *out_cta1 = t1 - t0;
    }
    cluster_sync();
}

/* ═══════ Host ═══════ */

/* Returns true if tensor map was created (SWIZZLE_128B needs power-of-2 rows) */
bool make_tmap(CUtensorMap *tm, uint8_t *ptr) {
    int rows = XFER / GCOLS;
    uint64_t dims[2]    = {GCOLS, GROWS};
    uint64_t strides[1] = {GCOLS};
    uint32_t box[2]     = {(uint32_t)GCOLS, (uint32_t)rows};
    uint32_t estrides[2]= {1, 1};
    CUresult r = cuTensorMapEncodeTiled(tm,
        CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)ptr,
        dims, strides, box, estrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) return false;
    return true;
}

int main() {
    CU_CHECK(cuInit(0));

    size_t src_bytes = (size_t)XFER * 64;
    uint8_t *d_src;
    CUDA_CHECK(cudaMalloc(&d_src, src_bytes));
    {
        uint8_t *h = (uint8_t*)malloc(src_bytes);
        for (size_t i = 0; i < src_bytes; i++) h[i] = (uint8_t)(i & 0xFF);
        CUDA_CHECK(cudaMemcpy(d_src, h, src_bytes, cudaMemcpyHostToDevice));
        free(h);
    }

    int *d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));

    long long *d_t0, *d_t1;
    CUDA_CHECK(cudaMalloc(&d_t0, sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_t1, sizeof(long long)));

    CUtensorMap tm;
    bool have_tmap = make_tmap(&tm, d_src);
    if (!have_tmap)
        printf("  (tensor map failed for XFER=%dKB — cg2 tests skipped)\n\n", XFER/1024);

    int smem = SMEM_BYTES;
    auto set_smem = [&](const void *fn) {
        CUDA_CHECK(cudaFuncSetAttribute(fn,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    };

    /* ── Test 0: Smoke — just complete_tx to peer ── */
    printf("═══ Smoke: mbarrier.complete_tx.shared::cluster ═══\n");
    set_smem((const void*)k_smoke);
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(int)));
    k_smoke<<<2, 32, smem>>>(d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    int h_smoke;
    CUDA_CHECK(cudaMemcpy(&h_smoke, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  complete_tx relay: %s\n\n", h_smoke ? "PASS" : "FAIL");
    if (!h_smoke) { printf("FATAL: complete_tx to peer mbar broken on this GPU\n"); return 1; }

    /* ── Test A: Correctness ── */
    printf("═══ Correctness: 1D multicast + relay ═══\n");
    set_smem((const void*)k_correctness);
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(int)));
    k_correctness<<<2, 32, smem>>>(d_src, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    int h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  CTA1 data verify: %s\n\n", h_result ? "PASS" : "FAIL");
    if (!h_result) { printf("FAIL — aborting latency tests\n"); return 1; }

    /* ── Test B: Latency comparison ── */
    printf("═══ Latency: 1D per-CTA vs 1D relay vs tensor cg2 (XFER=%dKB, REPS=%d) ═══\n",
           XFER / 1024, REPS);
    printf("  %-24s %8s %8s %8s\n", "variant", "CTA0 cy", "CTA1 cy", "CTA1/iter");
    printf("  %-24s %8s %8s %8s\n", "-------", "-------", "-------", "---------");

    auto run_1d = [&](const char *label, int pipe, auto fn) {
        set_smem((const void*)fn);
        fn<<<2, 32, smem>>>(d_src, d_t0, d_t1);
        CUDA_CHECK(cudaDeviceSynchronize());
        long long h0, h1;
        CUDA_CHECK(cudaMemcpy(&h0, d_t0, sizeof(long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h1, d_t1, sizeof(long long), cudaMemcpyDeviceToHost));
        printf("  %-24s %8lld %8lld %8.1f\n", label, h0, h1, (double)h1 / REPS);
    };

    auto run_ref = [&](const char *label, int pipe, auto fn) {
        set_smem((const void*)fn);
        fn<<<2, 32, smem>>>(tm, d_t0, d_t1);
        CUDA_CHECK(cudaDeviceSynchronize());
        long long h0, h1;
        CUDA_CHECK(cudaMemcpy(&h0, d_t0, sizeof(long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h1, d_t1, sizeof(long long), cudaMemcpyDeviceToHost));
        printf("  %-24s %8lld %8lld %8.1f\n", label, h0, h1, (double)h1 / REPS);
    };

    run_1d("1D per-CTA pipe=1", 1, k_1d_percta<1>);
    run_1d("1D relay pipe=1", 1, k_relay<1>);
    if (have_tmap) run_ref("tensor cg2 pipe=1", 1, k_tensor_cg2<1>);
#if MAX_PIPE >= 2
    printf("\n");
    run_1d("1D per-CTA pipe=2", 2, k_1d_percta<2>);
    run_1d("1D relay pipe=2", 2, k_relay<2>);
    if (have_tmap) run_ref("tensor cg2 pipe=2", 2, k_tensor_cg2<2>);
#endif
#if MAX_PIPE >= 4
    printf("\n");
    run_1d("1D per-CTA pipe=4", 4, k_1d_percta<4>);
    run_1d("1D relay pipe=4", 4, k_relay<4>);
    if (have_tmap) run_ref("tensor cg2 pipe=4", 4, k_tensor_cg2<4>);
#endif

    printf("\nRelay overhead = CTA1(relay) - CTA1(cg2) per iteration\n");

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_t0));
    CUDA_CHECK(cudaFree(d_t1));
    return 0;
}
