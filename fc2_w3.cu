/*
FC2 W3 kernel — CUTLASS-style shared-SMEM epilogue architecture
Standalone binary: independent of kernel_body.cuh / kernel_common.cuh
Shape: [928256,3072]×[3072,768]^T + bias + residual
7 warps: W0(TMA A/B) | W1(MMA) | W2(EpilogueLoad) | W3-W6(Epilogue)
cta_group::2  __cluster_dims__(2,1,1)

Compile-time flags:
  -DFP32_EPILOGUE    FP32 math (FADD, ~0% STS conflict) instead of BF16 (HADD2, 7.5%)
  -DSTRIP_EPILOGUE   Skip epilogue (benchmark GEMM core only, valid=0)
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

/* ── Pipeline ── */
#ifndef N_STAGES
#define N_STAGES       5
#endif
#ifndef K_LOOP_UNROLL
#define K_LOOP_UNROLL  N_STAGES
#endif

/* ── Threads: 7 warps = 224 ── */
#define NUM_EPI_WARPS  4
#define THREADS        (32 * 7)   /* W0 + W1 + W2(loader) + W3-W6(epilogue) */

/* ── SMEM layout ── */
#define STAGE_BYTES    32768                                    /* 16KB A + 16KB B */
#define OFF_TMEM           (N_STAGES * STAGE_BYTES)
#define OFF_TMA_MBAR       (OFF_TMEM + 8)
#define OFF_MMA_MBAR       (OFF_TMA_MBAR + N_STAGES * 8)
#define OFF_MAINLOOP_MBAR  (OFF_MMA_MBAR + N_STAGES * 8)
#define OFF_EPILOGUE_MBAR  (OFF_MAINLOOP_MBAR + 16)

/* New barriers for W2↔epilogue coordination */
#define OFF_LOAD_MBAR      (OFF_EPILOGUE_MBAR + 16)            /* W2→epi: load done (2 stages) */
#define OFF_LOAD_CONSUMED  (OFF_LOAD_MBAR + 16)                /* epi→W2: stage consumed (2 stages) */
#define _MBAR_END          (OFF_LOAD_CONSUMED + 16)

/* Bias SMEM: 256 BF16 = 512 B */
#define OFF_BIAS_SMEM      ((_MBAR_END + 15) & ~15)
#define BIAS_SMEM_BYTES    (TN * 2)

/* Epilogue staging: 2 double-buffered stages, each 128 rows × 64 cols × 2B = 16 KB
   Option A: separate load/store regions (simpler, +32 KB) */
#define STAGING_REGION_BYTES  (32 * 128)                        /* 4096 B: 32 rows × 64 cols × 2B */

/* ReuseSmemC=false for now (Option A): residual and output in separate regions.
   Each stage = 16 KB residual + 16 KB output = 32 KB. 2 stages = 64 KB.
   Actually: residual 128r×64c = 16KB, output 128r×64c = 16KB. */
#define STAGE_RES_BYTES    (4 * STAGING_REGION_BYTES)           /* 16384: 4 row_groups × 4096 */
#define STAGE_OUT_BYTES    (4 * STAGING_REGION_BYTES)           /* 16384: 4 row_groups × 4096 */
#define STAGE_TOTAL_BYTES  (STAGE_RES_BYTES + STAGE_OUT_BYTES)  /* 32768 per stage */

#define OFF_STAGING        ((OFF_BIAS_SMEM + BIAS_SMEM_BYTES + 1023) & ~1023)  /* 1024-align */
/* Stage s: OFF_STAGING + s * STAGE_TOTAL_BYTES
   Within stage s:
     residual[rg]: OFF_STAGING + s*STAGE_TOTAL_BYTES + rg*STAGING_REGION_BYTES
     output[rg]:   OFF_STAGING + s*STAGE_TOTAL_BYTES + STAGE_RES_BYTES + rg*STAGING_REGION_BYTES */

#define SMEM_BYTES         ((OFF_STAGING + 2 * STAGE_TOTAL_BYTES + 127) & ~127)

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
    const int sm_id = blockIdx.x;
    const int tid   = threadIdx.x;
    const int warp  = tid / 32;
    const int lane  = tid % 32;

    const int cta_rank = sm_id & 1;
    const int cluster_id = sm_id / 2;
    const int num_clusters = SM_COUNT / 2;

    /* ── Mbarrier init ── */
    if (tid == 0) {
        for (int s = 0; s < N_STAGES; s++) {
            mbar_init(smem_to_uint(smem + OFF_TMA_MBAR + s * 8), 2);
            mbar_init(smem_to_uint(smem + OFF_MMA_MBAR + s * 8), 1);
        }
        for (int i = 0; i < 2; i++) {
            mbar_init(smem_to_uint(smem + OFF_MAINLOOP_MBAR + i * 8), 1);
            /* epilogue mbar: all epilogue warps (W3-W6) + W2 arrive.
               (NUM_EPI_WARPS + 1) warps × 2 CTAs × 32 threads */
            mbar_init(smem_to_uint(smem + OFF_EPILOGUE_MBAR + i * 8), (NUM_EPI_WARPS + 1) * 2 * 32);
        }
        /* W2→epilogue: load done (double-buffered). W2 arrives with expect_tx. */
        for (int s = 0; s < 2; s++)
            mbar_init(smem_to_uint(smem + OFF_LOAD_MBAR + s * 8), 1);
        /* epilogue→W2: load consumed (double-buffered). All 4 epi warps arrive. */
        for (int s = 0; s < 2; s++)
            mbar_init(smem_to_uint(smem + OFF_LOAD_CONSUMED + s * 8), NUM_EPI_WARPS);

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

    const int tile_start = (int)((long long)cluster_id * TOTAL_TILES / num_clusters);
    const int tile_end   = (int)((long long)(cluster_id + 1) * TOTAL_TILES / num_clusters);

    int tma_phase[N_STAGES] = {0};
    int mma_phase[N_STAGES] = {0};

    uint64_t desc_a_base[N_STAGES], desc_b_base[N_STAGES];
    for (int s = 0; s < N_STAGES; s++) {
        desc_a_base[s] = make_smem_desc(smem_a[s]);
        desc_b_base[s] = make_smem_desc(smem_b[s]);
    }

    const int start_buf = tile_start & 1;
    int epi_phase[2] = {1, 1};
    int ml_phase[2]  = {start_buf, 1 - start_buf};

    /* W2 + epilogue barrier addresses & phases (persist across tiles) */
    const uint32_t load_mbar[2] = {
        smem_to_uint(smem + OFF_LOAD_MBAR),
        smem_to_uint(smem + OFF_LOAD_MBAR + 8)
    };
    const uint32_t consumed_mbar[2] = {
        smem_to_uint(smem + OFF_LOAD_CONSUMED),
        smem_to_uint(smem + OFF_LOAD_CONSUMED + 8)
    };
    int load_phase[2] = {0, 0};          /* epilogue side: wait phase for load_mbar */
    int load_consumed_phase[2] = {0, 0}; /* W2 side: wait phase for consumed_mbar */

    /* ── Load bias into SMEM once ── */
    {
        const uint32_t bias_saddr = smem_to_uint(smem + OFF_BIAS_SMEM);
        /* 256 BF16 = 128 uint32_t. Spread across all threads (224 threads, each loads ~0-1). */
        for (int i = tid; i < TN / 2; i += THREADS) {
            /* Load 2 BF16 as uint32_t, store to SMEM */
            uint32_t val;
            asm volatile("ld.global.b32 %0, [%1];" : "=r"(val) : "l"(bias + i * 2));
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(bias_saddr + i * 4), "r"(val));
        }
    }
    __syncthreads();

    /* ════════════════════════════════════════════
       MAIN TILE LOOP
       ════════════════════════════════════════════ */

    for (int tile_idx = tile_start; tile_idx < tile_end; tile_idx++) {
        const int buf = tile_idx & 1;
        const int tm = tile_idx / TILES_N;
        int tn = tile_idx % TILES_N;
        if (SNAKE_ORDER && (tm & 1)) tn = TILES_N - 1 - tn;
        const int m_start = tm * TM * 2 + cta_rank * TM;
        const int n_start = tn * TN;

        if (warp == 0) {
            /* ── W0: TMA A/B loads ── */
            const uint32_t smem_base = warp_uniform(smem_to_uint(smem));
            for (int ki = 0; ki < K_ITERS; ki++) {
                const int s = ki % N_STAGES;
                const int k_start = ki * TK;
                const uint32_t mma_mbar_s = smem_base + OFF_MMA_MBAR + s * 8;
                const uint32_t tma_mbar_s = (smem_base + OFF_TMA_MBAR + s * 8) & 0xFEFFFFFF;

                if (tile_idx > tile_start || ki >= N_STAGES) {
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
#ifdef STRIP_EPILOGUE
            if (tile_idx > tile_start) {
                const int prev_buf = buf ^ 1;
                mbar_wait(mainloop_mbar_addr + prev_buf * 8, ml_phase[prev_buf]);
                ml_phase[prev_buf] ^= 1;
                mbar_arrive(epilogue_mbar_addr + prev_buf * 8);
            }
#else
            /* ── W2: EpilogueLoad — TMA residual into shared staging ── */
            if (tile_idx > tile_start) {
                const int prev_buf = buf ^ 1;
                const int prev_idx = tile_idx - 1;
                const int ptm = prev_idx / TILES_N;
                int ptn = prev_idx % TILES_N;
                if (SNAKE_ORDER && (ptm & 1)) ptn = TILES_N - 1 - ptn;
                const int prev_m = ptm * TM * 2 + cta_rank * TM;
                const int prev_n = ptn * TN;

                mbar_wait(mainloop_mbar_addr + prev_buf * 8, ml_phase[prev_buf]);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                ml_phase[prev_buf] ^= 1;

                for (int si = 0; si < 4; si++) {
                    const int stg = si & 1;
                    /* Wait for previous consumer to free this stage (skip first 2) */
                    if (si >= 2) {
                        mbar_wait(consumed_mbar[stg], load_consumed_phase[stg]);
                        load_consumed_phase[stg] ^= 1;
                    }
                    if (lane == 0) {
                        const uint32_t res_dst = smem_to_uint(smem + OFF_STAGING + stg * STAGE_TOTAL_BYTES);
                        mbar_arrive_expect_tx(load_mbar[stg], STAGE_RES_BYTES);
                        tma_load_2d_cta(res_dst, &tma_res,
                                        prev_n + si * 64, prev_m, load_mbar[stg]);
                    }
                }
                mbar_arrive(epilogue_mbar_addr + prev_buf * 8);
            }
#endif /* STRIP_EPILOGUE W2 */

        } else {
#ifdef STRIP_EPILOGUE
            if (tile_idx > tile_start) {
                const int prev_buf = buf ^ 1;
                mbar_wait(mainloop_mbar_addr + prev_buf * 8, ml_phase[prev_buf]);
                ml_phase[prev_buf] ^= 1;
                mbar_arrive(epilogue_mbar_addr + prev_buf * 8);
            }
#else
            /* ── W3-W6: Epilogue compute — shared SMEM, BAR.SYNC coordinated ── */
            const int ew = warp - 3;                           /* 0..3 */
            const int row_group = ew;                          /* rows ew*32..(ew+1)*32-1 */
            const uint32_t bias_saddr = smem_to_uint(smem + OFF_BIAS_SMEM);

            /* Swizzle constants (SWIZZLE_128B: 128-byte rows, XOR with lane-group) */
            const uint32_t xor_val = (lane & 7) << 4;
            const uint32_t sw0 = 0 ^ xor_val, sw1 = 16 ^ xor_val;
            const uint32_t sw2 = 32 ^ xor_val, sw3 = 48 ^ xor_val;
            const uint32_t sw4 = 64 ^ xor_val, sw5 = 80 ^ xor_val;
            const uint32_t sw6 = 96 ^ xor_val, sw7 = 112 ^ xor_val;

            if (tile_idx > tile_start) {
                const int prev_buf = buf ^ 1;
                const int prev_idx = tile_idx - 1;
                const int ptm = prev_idx / TILES_N;
                int ptn = prev_idx % TILES_N;
                if (SNAKE_ORDER && (ptm & 1)) ptn = TILES_N - 1 - ptn;
                const int prev_m = ptm * TM * 2 + cta_rank * TM;
                const int prev_n = ptn * TN;
                const int gm_base = prev_m + row_group * 32;
                const int taddr_base = prev_buf * TN + ((cta_rank * 128 + row_group * 32) << 16);

                mbar_wait(mainloop_mbar_addr + prev_buf * 8, ml_phase[prev_buf]);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                ml_phase[prev_buf] ^= 1;

                for (int si = 0; si < 4; si++) {
                    const int stg = si & 1;
                    const int nc_base = si * 64;   /* column offset within tile */

                    /* Wait for W2's TMA load to land */
                    mbar_wait(load_mbar[stg], load_phase[stg]);
                    load_phase[stg] ^= 1;

                    /* Wait for previous TMA store to finish reading output SMEM
                       (needed when reusing same stage: si >= 2) */
                    if (si >= 2) {
                        if (lane == 0) {
                            asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                        }
                        __syncwarp();
                    }

                    /* Process 2 chunks of 32 cols each */
                    float a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15;
                    float a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31;

                    for (int chunk = 0; chunk < 2; chunk++) {
                        const int nc = nc_base + chunk * 32;

                        /* TMEM load: 32 FP32 accumulators */
                        TMEM_LOAD_X32(a0,a1,a2,a3,a4,a5,a6,a7,
                                      a8,a9,a10,a11,a12,a13,a14,a15,
                                      a16,a17,a18,a19,a20,a21,a22,a23,
                                      a24,a25,a26,a27,a28,a29,a30,a31,
                                      taddr_base + nc);

                        /* LDS bias from SMEM (linear, not swizzled) */
                        const uint32_t bs = bias_saddr + nc * 2;
                        uint4 bv0, bv1, bv2, bv3;
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(bv0.x),"=r"(bv0.y),"=r"(bv0.z),"=r"(bv0.w) : "r"(bs));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(bv1.x),"=r"(bv1.y),"=r"(bv1.z),"=r"(bv1.w) : "r"(bs + 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(bv2.x),"=r"(bv2.y),"=r"(bv2.z),"=r"(bv2.w) : "r"(bs + 32));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(bv3.x),"=r"(bv3.y),"=r"(bv3.z),"=r"(bv3.w) : "r"(bs + 48));

                        /* LDS residual from shared staging (swizzled) */
                        const uint32_t rs = smem_to_uint(smem + OFF_STAGING
                            + stg * STAGE_TOTAL_BYTES
                            + row_group * STAGING_REGION_BYTES
                            + lane * 128);
                        const uint32_t rsw0 = chunk ? sw4 : sw0;
                        const uint32_t rsw1 = chunk ? sw5 : sw1;
                        const uint32_t rsw2 = chunk ? sw6 : sw2;
                        const uint32_t rsw3 = chunk ? sw7 : sw3;
                        uint4 rv0, rv1, rv2, rv3;
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(rv0.x),"=r"(rv0.y),"=r"(rv0.z),"=r"(rv0.w) : "r"(rs + rsw0));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(rv1.x),"=r"(rv1.y),"=r"(rv1.z),"=r"(rv1.w) : "r"(rs + rsw1));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(rv2.x),"=r"(rv2.y),"=r"(rv2.z),"=r"(rv2.w) : "r"(rs + rsw2));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(rv3.x),"=r"(rv3.y),"=r"(rv3.z),"=r"(rv3.w) : "r"(rs + rsw3));

                        TMEM_WAIT();

                        /* Output staging — STS to per-warp region within output area */
                        const uint32_t os = smem_to_uint(smem + OFF_STAGING
                            + stg * STAGE_TOTAL_BYTES + STAGE_RES_BYTES
                            + row_group * STAGING_REGION_BYTES
                            + lane * 128);

                        BIAS_RES_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7,
                            bv0.x,bv0.y,bv0.z,bv0.w,
                            rv0.x,rv0.y,rv0.z,rv0.w, os + rsw0);
                        BIAS_RES_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15,
                            bv1.x,bv1.y,bv1.z,bv1.w,
                            rv1.x,rv1.y,rv1.z,rv1.w, os + rsw1);
                        BIAS_RES_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23,
                            bv2.x,bv2.y,bv2.z,bv2.w,
                            rv2.x,rv2.y,rv2.z,rv2.w, os + rsw2);
                        BIAS_RES_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31,
                            bv3.x,bv3.y,bv3.z,bv3.w,
                            rv3.x,rv3.y,rv3.z,rv3.w, os + rsw3);
                    }

                    /* FENCE + BAR.SYNC: all 4 epilogue warps' STS must be visible */
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("bar.sync 1, %0;" :: "r"(NUM_EPI_WARPS * 32) : "memory");

                    /* Signal W2: this stage's residual data has been consumed */
                    if (lane == 0) {
                        mbar_arrive(consumed_mbar[stg]);
                    }

                    /* TMA store: each warp stores its 32-row × 64-col output */
                    if (lane == 0) {
                        const uint32_t out_src = smem_to_uint(smem + OFF_STAGING
                            + stg * STAGE_TOTAL_BYTES + STAGE_RES_BYTES
                            + row_group * STAGING_REGION_BYTES);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
                            " [%0, {%1, %2}], [%3];"
                            :: "l"(&tma_c), "r"(prev_n + nc_base), "r"(gm_base),
                               "r"(out_src) : "memory");
                        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                    }

                    /* No second bar.sync needed: each warp's TMA store reads from its own
                       row_group region. wait_group 0 at si+2 handles per-warp timing. */
                }

                /* Signal W1: TMEM buffer free for next tile */
                mbar_arrive(epilogue_mbar_addr + prev_buf * 8);
            }
#endif /* STRIP_EPILOGUE W3-W6 */
        }
    }

    /* ── Drain: last tile epilogue ── */
    {
        const int last_idx = tile_end - 1;
        const int last_buf = last_idx & 1;
        const int ltm = last_idx / TILES_N;
        int ltn = last_idx % TILES_N;
        if (SNAKE_ORDER && (ltm & 1)) ltn = TILES_N - 1 - ltn;
        const int last_m = ltm * TM * 2 + cta_rank * TM;
        const int last_n = ltn * TN;

        if (warp == 0) {
            /* W0: nothing to do for drain */
        } else if (warp == 1) {
            /* W1: nothing — already committed mainloop_mbar */
        } else if (warp == 2) {
#ifdef STRIP_EPILOGUE
            mbar_wait(mainloop_mbar_addr + last_buf * 8, ml_phase[last_buf]);
            ml_phase[last_buf] ^= 1;
            mbar_arrive(epilogue_mbar_addr + last_buf * 8);
#else
            /* W2: load residual for last tile */
            mbar_wait(mainloop_mbar_addr + last_buf * 8, ml_phase[last_buf]);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            ml_phase[last_buf] ^= 1;

            for (int si = 0; si < 4; si++) {
                const int stg = si & 1;
                if (si >= 2) {
                    mbar_wait(consumed_mbar[stg], load_consumed_phase[stg]);
                    load_consumed_phase[stg] ^= 1;
                }
                if (lane == 0) {
                    const uint32_t res_dst = smem_to_uint(smem + OFF_STAGING + stg * STAGE_TOTAL_BYTES);
                    mbar_arrive_expect_tx(load_mbar[stg], STAGE_RES_BYTES);
                    tma_load_2d_cta(res_dst, &tma_res,
                                    last_n + si * 64, last_m, load_mbar[stg]);
                }
            }
            mbar_arrive(epilogue_mbar_addr + last_buf * 8);
#endif /* STRIP_EPILOGUE drain W2 */

        } else {
#ifdef STRIP_EPILOGUE
            mbar_wait(mainloop_mbar_addr + last_buf * 8, ml_phase[last_buf]);
            ml_phase[last_buf] ^= 1;
            mbar_arrive(epilogue_mbar_addr + last_buf * 8);
#else
            /* W3-W6: epilogue for last tile */
            const int ew = warp - 3;
            const int row_group = ew;
            const uint32_t bias_saddr = smem_to_uint(smem + OFF_BIAS_SMEM);
            const uint32_t xor_val = (lane & 7) << 4;
            const uint32_t sw0 = 0 ^ xor_val, sw1 = 16 ^ xor_val;
            const uint32_t sw2 = 32 ^ xor_val, sw3 = 48 ^ xor_val;
            const uint32_t sw4 = 64 ^ xor_val, sw5 = 80 ^ xor_val;
            const uint32_t sw6 = 96 ^ xor_val, sw7 = 112 ^ xor_val;
            const int gm_base = last_m + row_group * 32;
            const int taddr_base = last_buf * TN + ((cta_rank * 128 + row_group * 32) << 16);

            mbar_wait(mainloop_mbar_addr + last_buf * 8, ml_phase[last_buf]);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            ml_phase[last_buf] ^= 1;

            for (int si = 0; si < 4; si++) {
                const int stg = si & 1;
                const int nc_base = si * 64;

                mbar_wait(load_mbar[stg], load_phase[stg]);
                load_phase[stg] ^= 1;

                if (si >= 2) {
                    if (lane == 0) {
                        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                    }
                    __syncwarp();
                }

                float a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15;
                float a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31;

                for (int chunk = 0; chunk < 2; chunk++) {
                    const int nc = nc_base + chunk * 32;

                    TMEM_LOAD_X32(a0,a1,a2,a3,a4,a5,a6,a7,
                                  a8,a9,a10,a11,a12,a13,a14,a15,
                                  a16,a17,a18,a19,a20,a21,a22,a23,
                                  a24,a25,a26,a27,a28,a29,a30,a31,
                                  taddr_base + nc);

                    const uint32_t bs = bias_saddr + nc * 2;
                    uint4 bv0, bv1, bv2, bv3;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(bv0.x),"=r"(bv0.y),"=r"(bv0.z),"=r"(bv0.w) : "r"(bs));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(bv1.x),"=r"(bv1.y),"=r"(bv1.z),"=r"(bv1.w) : "r"(bs + 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(bv2.x),"=r"(bv2.y),"=r"(bv2.z),"=r"(bv2.w) : "r"(bs + 32));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(bv3.x),"=r"(bv3.y),"=r"(bv3.z),"=r"(bv3.w) : "r"(bs + 48));

                    const uint32_t rs = smem_to_uint(smem + OFF_STAGING
                        + stg * STAGE_TOTAL_BYTES
                        + row_group * STAGING_REGION_BYTES
                        + lane * 128);
                    const uint32_t rsw0 = chunk ? sw4 : sw0;
                    const uint32_t rsw1 = chunk ? sw5 : sw1;
                    const uint32_t rsw2 = chunk ? sw6 : sw2;
                    const uint32_t rsw3 = chunk ? sw7 : sw3;
                    uint4 rv0, rv1, rv2, rv3;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(rv0.x),"=r"(rv0.y),"=r"(rv0.z),"=r"(rv0.w) : "r"(rs + rsw0));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(rv1.x),"=r"(rv1.y),"=r"(rv1.z),"=r"(rv1.w) : "r"(rs + rsw1));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(rv2.x),"=r"(rv2.y),"=r"(rv2.z),"=r"(rv2.w) : "r"(rs + rsw2));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(rv3.x),"=r"(rv3.y),"=r"(rv3.z),"=r"(rv3.w) : "r"(rs + rsw3));

                    TMEM_WAIT();

                    const uint32_t os = smem_to_uint(smem + OFF_STAGING
                        + stg * STAGE_TOTAL_BYTES + STAGE_RES_BYTES
                        + row_group * STAGING_REGION_BYTES
                        + lane * 128);

                    BIAS_RES_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7,
                        bv0.x,bv0.y,bv0.z,bv0.w,
                        rv0.x,rv0.y,rv0.z,rv0.w, os + rsw0);
                    BIAS_RES_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15,
                        bv1.x,bv1.y,bv1.z,bv1.w,
                        rv1.x,rv1.y,rv1.z,rv1.w, os + rsw1);
                    BIAS_RES_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23,
                        bv2.x,bv2.y,bv2.z,bv2.w,
                        rv2.x,rv2.y,rv2.z,rv2.w, os + rsw2);
                    BIAS_RES_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31,
                        bv3.x,bv3.y,bv3.z,bv3.w,
                        rv3.x,rv3.y,rv3.z,rv3.w, os + rsw3);
                }

                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("bar.sync 1, %0;" :: "r"(NUM_EPI_WARPS * 32) : "memory");

                if (lane == 0) {
                    mbar_arrive(consumed_mbar[stg]);
                }

                if (lane == 0) {
                    const uint32_t out_src = smem_to_uint(smem + OFF_STAGING
                        + stg * STAGE_TOTAL_BYTES + STAGE_RES_BYTES
                        + row_group * STAGING_REGION_BYTES);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
                        " [%0, {%1, %2}], [%3];"
                        :: "l"(&tma_c), "r"(last_n + nc_base), "r"(gm_base),
                           "r"(out_src) : "memory");
                    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                }
            }

            /* Wait for final TMA stores to complete before kernel exit */
            if (lane == 0) {
                asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
            }
            __syncwarp();

            mbar_arrive(epilogue_mbar_addr + last_buf * 8);
#endif /* STRIP_EPILOGUE drain W3-W6 */
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
    printf("FC2 W3 kernel — 7 warps, shared-SMEM epilogue\n");
    printf("  GEMM: [%d,%d] x [%d,%d]^T  %d-stage pipeline  SMEM: %d bytes\n",
           M_TOTAL, K_DIM, N_DIM, K_DIM, N_STAGES, SMEM_BYTES);

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
        uint32_t box[2]     = {64, 128};  /* W2 loads full 128 rows × 64 cols per sub-iter */
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

    /* Warmup */
    printf("Launching warmup (2 iters)...\n");
    for (int i = 0; i < 2; i++) {
        fc2_w3_kernel<<<SM_COUNT, THREADS, SMEM_BYTES>>>(
            h_tma_a, h_tma_b, h_tma_c, d_bias, d_C, d_residual, h_tma_res);
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
        fc2_w3_kernel<<<SM_COUNT, THREADS, SMEM_BYTES>>>(
            h_tma_a, h_tma_b, h_tma_c, d_bias, d_C, d_residual, h_tma_res);
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
    fc2_w3_kernel<<<SM_COUNT, THREADS, SMEM_BYTES>>>(
        h_tma_a, h_tma_b, h_tma_c, d_bias, d_C, d_residual, h_tma_res);
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
