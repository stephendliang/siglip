/*
  fc2_ws.cu — FC2 GEMM mimicking cuBLASLt rank-1 structure (warp-specialized)
    target: nvjet_sm100_qqtst_128x256_128x6_2x1_2cta_v_bz_bias_TNT

  Per-CTA tile 128x256 (cluster tile 256x256 via cta_group::2).
  8 warps × 32 threads = 256 threads per CTA, 2x1 cluster.
  Flat register allocation — NO setmaxnreg.

  Thread-role map (tid-threshold dispatch):
    tid   0..127 (warps 0-3)  Epilogue warpgroup (bar.sync 0,128)
    tid 128..159 (warp   4)   TMA A+B multicast loader (also loads bias at start)
    tid 160..191 (warp   5)   TMA residual loader (per-sub-pass handshake)
    tid 192..223 (warp   6)   idle (mirror rank-1's dead path)
    tid 224..255 (warp   7)   MMA issuer (CTA 0 only, 4x UTCQMMA per K-stage)

  Mbar topology:
    MBAR_TMA_FULL[6]    — on CTA 0, count=2, both CTAs' W4 arrive via shared::cluster
    MBAR_TMA_EMPTY[6]   — per-CTA, count=1, MMA multicasts arrive via tcgen05.commit
    MBAR_RES_FULL[2]    — per-CTA, count=1, W5 arrives locally
    MBAR_RES_EMPTY[2]   — per-CTA, count=1, Epi arrives locally
    MBAR_TMEM_READY     — per-CTA, count=1, MMA multicasts arrive
    MBAR_TMEM_CONSUMED  — on CTA 0, count=2, both CTAs' Epi arrive via shared::cluster

  Dispatch: dgswizzle, DG_GROUP_SIZE=8, PACKED_TILES hardcoded.
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
#define THREADS     256

#define TILES_M     ((M_TOTAL + TM * 2 - 1) / (TM * 2))
#define TILES_N     ((N_DIM  + TN - 1) / TN)
#define TOTAL_TILES (TILES_M * TILES_N)
#define K_ITERS     (K_DIM / TK)

#define N_STAGES       6
#define NUM_EPI_STAGES 2
#define NUM_SUBPASSES  (TN / 32)

#define SUBPASS_COLS   32
#define ROWS_PER_CTA   TM
#define SUBPASS_BYTES  (ROWS_PER_CTA * SUBPASS_COLS * 2)

#define STAGE_BYTES    32768
#define TMA_BYTES      32768
#define MAIN_SMEM      (N_STAGES * STAGE_BYTES)
#define OUT_STAGING    (NUM_EPI_STAGES * SUBPASS_BYTES)
#define RES_STAGING    (NUM_EPI_STAGES * SUBPASS_BYTES)
#define BIAS_BYTES     (N_DIM * 2)

#define OFF_AB         0
#define OFF_OUT        MAIN_SMEM
#define OFF_RES        (OFF_OUT + OUT_STAGING)
#define OFF_BIAS       ((OFF_RES + RES_STAGING + 127) & ~127)
#define OFF_MBARS      ((OFF_BIAS + BIAS_BYTES + 127) & ~127)

#define MBAR_TMA_FULL       (OFF_MBARS + 0)
#define MBAR_TMA_EMPTY      (MBAR_TMA_FULL + N_STAGES * 8)
#define MBAR_RES_FULL       (MBAR_TMA_EMPTY + N_STAGES * 8)
#define MBAR_RES_EMPTY      (MBAR_RES_FULL + NUM_EPI_STAGES * 8)
#define MBAR_TMEM_READY     (MBAR_RES_EMPTY + NUM_EPI_STAGES * 8)
#define MBAR_TMEM_CONSUMED  (MBAR_TMEM_READY + 8)
#define MBARS_END           (MBAR_TMEM_CONSUMED + 8)

#define OFF_TMEM       ((MBARS_END + 15) & ~15)
#define SMEM_BYTES     ((OFF_TMEM + 8 + 127) & ~127)

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

static __device__ __forceinline__
int dgswizzle(int lin) {
    const int group_tiles = TILES_N * DG_GROUP_SIZE;
    const int group_idx = lin / group_tiles;
    const int first_m = group_idx * DG_GROUP_SIZE;
    const int in_group = lin % group_tiles;
    if (first_m + DG_GROUP_SIZE <= TILES_M) {
        return (first_m + in_group % DG_GROUP_SIZE) * TILES_N
             + in_group / DG_GROUP_SIZE;
    }
    const int tail = TILES_M - first_m;
    return (first_m + in_group % tail) * TILES_N + in_group / tail;
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

static __device__ __forceinline__
void tma_load_cta(uint32_t smem_dst, const CUtensorMap* tma_desc,
                  int32_t c0, int32_t c1, uint32_t mbar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_dst), "l"(tma_desc), "r"(c0), "r"(c1), "r"(mbar)
        : "memory");
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

#define TMEM_WAIT() \
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory")

__global__ void __launch_bounds__(THREADS, 1)
__cluster_dims__(2, 1, 1)
fc2_ws_kernel(const __grid_constant__ CUtensorMap tma_a,
              const __grid_constant__ CUtensorMap tma_b,
              const __grid_constant__ CUtensorMap tma_c,
              const __grid_constant__ CUtensorMap tma_res,
              const __nv_bfloat16* __restrict__ d_bias,
              __nv_bfloat16* __restrict__ d_C,
              const __nv_bfloat16* __restrict__ d_res)
{
    (void)d_C; (void)d_res;

    extern __shared__ __align__(128) uint8_t smem[];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    uint32_t cta_rank;
    asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    const uint16_t pair_mask = 0x3u;

    /* === Init (per-CTA, but MBAR_TMA_FULL / MBAR_TMEM_CONSUMED only on CTA 0) === */
    if (tid == 0) {
        for (int s = 0; s < N_STAGES; s++) {
            if (cta_rank == 0) {
                mbar_init(smem_to_uint(smem + MBAR_TMA_FULL + s * 8), 2);
            }
            mbar_init(smem_to_uint(smem + MBAR_TMA_EMPTY + s * 8), 1);
        }
        for (int s = 0; s < NUM_EPI_STAGES; s++) {
            mbar_init(smem_to_uint(smem + MBAR_RES_FULL  + s * 8), 1);
            mbar_init(smem_to_uint(smem + MBAR_RES_EMPTY + s * 8), 1);
        }
        mbar_init(smem_to_uint(smem + MBAR_TMEM_READY), 1);
        if (cta_rank == 0) {
            mbar_init(smem_to_uint(smem + MBAR_TMEM_CONSUMED), 2);
        }
    }
    if (warp_id == 0) {
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(smem_to_uint(smem + OFF_TMEM)), "n"(TMEM_COLS));
    }

    /* Bias LDG+STS in warp 4, all 32 lanes cooperate. Happens pre-barrier. */
    if (warp_id == 4) {
        for (int i = lane; i < N_DIM; i += 32) {
            __nv_bfloat16 v = d_bias[i];
            uint16_t bits = *reinterpret_cast<uint16_t*>(&v);
            asm volatile("st.shared.b16 [%0], %1;"
                :: "r"(smem_to_uint(smem + OFF_BIAS + i * 2)), "h"(bits));
        }
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");

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

    uint32_t res_full_arr[NUM_EPI_STAGES], res_empty_arr[NUM_EPI_STAGES];
    uint32_t out_smem_arr[NUM_EPI_STAGES], res_smem_arr[NUM_EPI_STAGES];
    for (int s = 0; s < NUM_EPI_STAGES; s++) {
        res_full_arr[s]  = smem_to_uint(smem + MBAR_RES_FULL  + s * 8);
        res_empty_arr[s] = smem_to_uint(smem + MBAR_RES_EMPTY + s * 8);
        out_smem_arr[s]  = smem_to_uint(smem + OFF_OUT + s * SUBPASS_BYTES);
        res_smem_arr[s]  = smem_to_uint(smem + OFF_RES + s * SUBPASS_BYTES);
    }

    const uint32_t mbar_tmem_ready        = smem_to_uint(smem + MBAR_TMEM_READY);
    const uint32_t mbar_tmem_consumed     = smem_to_uint(smem + MBAR_TMEM_CONSUMED);
    const uint32_t mbar_tmem_cons_peer    = mbar_tmem_consumed & 0xFEFFFFFFu;
    const uint32_t smem_bias              = smem_to_uint(smem + OFF_BIAS);

    const int num_clusters = SM_COUNT / CLUSTER_CTAS;
    const int cluster_id   = blockIdx.x / CLUSTER_CTAS;
    const int tiles_per_cluster = (TOTAL_TILES + num_clusters - 1) / num_clusters;

    if (warp_id < 4) {
        /* =============== Epilogue warpgroup (tid 0..127) =============== */
        const int wg_warp   = warp_id;
        const int row_group = wg_warp;

        uint32_t mma_phase = 0;
        uint32_t res_full_phase[NUM_EPI_STAGES] = {0, 0};

        for (int tt = 0; tt < tiles_per_cluster; tt++) {
            const int lin_tile = cluster_id + tt * num_clusters;
            if (lin_tile >= TOTAL_TILES) break;
            const int swizzled = dgswizzle(lin_tile);
            const int tm = swizzled / TILES_N;
            const int tn = swizzled % TILES_N;
            const int prev_m = tm * TM * 2;
            const int prev_n = tn * TN;
            const int buf = lin_tile & 1;

            mbar_wait(mbar_tmem_ready, mma_phase);
            mma_phase ^= 1;

            const uint32_t taddr_tile = taddr_base + buf * TN
                + ((cta_rank * TM + row_group * 32) << 16);

#ifdef STRIP_EPILOGUE
            (void)prev_m; (void)prev_n; (void)taddr_tile;
            if (tid == 0) mbar_arrive(mbar_tmem_cons_peer);
#else
            #pragma unroll 1
            for (int sp = 0; sp < NUM_SUBPASSES; sp++) {
                const int es = sp & (NUM_EPI_STAGES - 1);
                const int nc = sp * SUBPASS_COLS;

#if !defined(GEMM_ONLY) && !defined(BIAS_ONLY)
                mbar_wait(res_full_arr[es], res_full_phase[es]);
                res_full_phase[es] ^= 1;
#endif

                float a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15;
                float a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31;
                TMEM_LOAD_X32(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                              a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                              taddr_tile + nc);
                TMEM_WAIT();

                float o[32];
#ifdef GEMM_ONLY
                (void)prev_n;
                o[0]=a0;   o[1]=a1;   o[2]=a2;   o[3]=a3;
                o[4]=a4;   o[5]=a5;   o[6]=a6;   o[7]=a7;
                o[8]=a8;   o[9]=a9;   o[10]=a10; o[11]=a11;
                o[12]=a12; o[13]=a13; o[14]=a14; o[15]=a15;
                o[16]=a16; o[17]=a17; o[18]=a18; o[19]=a19;
                o[20]=a20; o[21]=a21; o[22]=a22; o[23]=a23;
                o[24]=a24; o[25]=a25; o[26]=a26; o[27]=a27;
                o[28]=a28; o[29]=a29; o[30]=a30; o[31]=a31;
#else
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

                __nv_bfloat162 b[16];
                uint32_t* bptr = reinterpret_cast<uint32_t*>(b);
                bptr[0]=bv0.x;  bptr[1]=bv0.y;  bptr[2]=bv0.z;  bptr[3]=bv0.w;
                bptr[4]=bv1.x;  bptr[5]=bv1.y;  bptr[6]=bv1.z;  bptr[7]=bv1.w;
                bptr[8]=bv2.x;  bptr[9]=bv2.y;  bptr[10]=bv2.z; bptr[11]=bv2.w;
                bptr[12]=bv3.x; bptr[13]=bv3.y; bptr[14]=bv3.z; bptr[15]=bv3.w;

#ifdef BIAS_ONLY
                o[0]  = a0  + __bfloat162float(b[0].x);
                o[1]  = a1  + __bfloat162float(b[0].y);
                o[2]  = a2  + __bfloat162float(b[1].x);
                o[3]  = a3  + __bfloat162float(b[1].y);
                o[4]  = a4  + __bfloat162float(b[2].x);
                o[5]  = a5  + __bfloat162float(b[2].y);
                o[6]  = a6  + __bfloat162float(b[3].x);
                o[7]  = a7  + __bfloat162float(b[3].y);
                o[8]  = a8  + __bfloat162float(b[4].x);
                o[9]  = a9  + __bfloat162float(b[4].y);
                o[10] = a10 + __bfloat162float(b[5].x);
                o[11] = a11 + __bfloat162float(b[5].y);
                o[12] = a12 + __bfloat162float(b[6].x);
                o[13] = a13 + __bfloat162float(b[6].y);
                o[14] = a14 + __bfloat162float(b[7].x);
                o[15] = a15 + __bfloat162float(b[7].y);
                o[16] = a16 + __bfloat162float(b[8].x);
                o[17] = a17 + __bfloat162float(b[8].y);
                o[18] = a18 + __bfloat162float(b[9].x);
                o[19] = a19 + __bfloat162float(b[9].y);
                o[20] = a20 + __bfloat162float(b[10].x);
                o[21] = a21 + __bfloat162float(b[10].y);
                o[22] = a22 + __bfloat162float(b[11].x);
                o[23] = a23 + __bfloat162float(b[11].y);
                o[24] = a24 + __bfloat162float(b[12].x);
                o[25] = a25 + __bfloat162float(b[12].y);
                o[26] = a26 + __bfloat162float(b[13].x);
                o[27] = a27 + __bfloat162float(b[13].y);
                o[28] = a28 + __bfloat162float(b[14].x);
                o[29] = a29 + __bfloat162float(b[14].y);
                o[30] = a30 + __bfloat162float(b[15].x);
                o[31] = a31 + __bfloat162float(b[15].y);
#else
                /* Residual: lane handles row (row_group*32 + lane), cols 0..31 for this sub-pass.
                   Staging layout: row r, col c → offset (r * SUBPASS_COLS + c) * 2.
                   64 bytes per row = 4× v4. */
                const int res_row_off = (row_group * 32 + lane) * (SUBPASS_COLS * 2);
                uint4 rv0, rv1, rv2, rv3;
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(rv0.x), "=r"(rv0.y), "=r"(rv0.z), "=r"(rv0.w)
                    : "r"(res_smem_arr[es] + res_row_off + 0));
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(rv1.x), "=r"(rv1.y), "=r"(rv1.z), "=r"(rv1.w)
                    : "r"(res_smem_arr[es] + res_row_off + 16));
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(rv2.x), "=r"(rv2.y), "=r"(rv2.z), "=r"(rv2.w)
                    : "r"(res_smem_arr[es] + res_row_off + 32));
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(rv3.x), "=r"(rv3.y), "=r"(rv3.z), "=r"(rv3.w)
                    : "r"(res_smem_arr[es] + res_row_off + 48));

                if (tid == 0) mbar_arrive(res_empty_arr[es]);

                __nv_bfloat162 r[16];
                uint32_t* rptr = reinterpret_cast<uint32_t*>(r);
                rptr[0]=rv0.x;  rptr[1]=rv0.y;  rptr[2]=rv0.z;  rptr[3]=rv0.w;
                rptr[4]=rv1.x;  rptr[5]=rv1.y;  rptr[6]=rv1.z;  rptr[7]=rv1.w;
                rptr[8]=rv2.x;  rptr[9]=rv2.y;  rptr[10]=rv2.z; rptr[11]=rv2.w;
                rptr[12]=rv3.x; rptr[13]=rv3.y; rptr[14]=rv3.z; rptr[15]=rv3.w;

                o[0]  = a0  + __bfloat162float(b[0].x)  + __bfloat162float(r[0].x);
                o[1]  = a1  + __bfloat162float(b[0].y)  + __bfloat162float(r[0].y);
                o[2]  = a2  + __bfloat162float(b[1].x)  + __bfloat162float(r[1].x);
                o[3]  = a3  + __bfloat162float(b[1].y)  + __bfloat162float(r[1].y);
                o[4]  = a4  + __bfloat162float(b[2].x)  + __bfloat162float(r[2].x);
                o[5]  = a5  + __bfloat162float(b[2].y)  + __bfloat162float(r[2].y);
                o[6]  = a6  + __bfloat162float(b[3].x)  + __bfloat162float(r[3].x);
                o[7]  = a7  + __bfloat162float(b[3].y)  + __bfloat162float(r[3].y);
                o[8]  = a8  + __bfloat162float(b[4].x)  + __bfloat162float(r[4].x);
                o[9]  = a9  + __bfloat162float(b[4].y)  + __bfloat162float(r[4].y);
                o[10] = a10 + __bfloat162float(b[5].x)  + __bfloat162float(r[5].x);
                o[11] = a11 + __bfloat162float(b[5].y)  + __bfloat162float(r[5].y);
                o[12] = a12 + __bfloat162float(b[6].x)  + __bfloat162float(r[6].x);
                o[13] = a13 + __bfloat162float(b[6].y)  + __bfloat162float(r[6].y);
                o[14] = a14 + __bfloat162float(b[7].x)  + __bfloat162float(r[7].x);
                o[15] = a15 + __bfloat162float(b[7].y)  + __bfloat162float(r[7].y);
                o[16] = a16 + __bfloat162float(b[8].x)  + __bfloat162float(r[8].x);
                o[17] = a17 + __bfloat162float(b[8].y)  + __bfloat162float(r[8].y);
                o[18] = a18 + __bfloat162float(b[9].x)  + __bfloat162float(r[9].x);
                o[19] = a19 + __bfloat162float(b[9].y)  + __bfloat162float(r[9].y);
                o[20] = a20 + __bfloat162float(b[10].x) + __bfloat162float(r[10].x);
                o[21] = a21 + __bfloat162float(b[10].y) + __bfloat162float(r[10].y);
                o[22] = a22 + __bfloat162float(b[11].x) + __bfloat162float(r[11].x);
                o[23] = a23 + __bfloat162float(b[11].y) + __bfloat162float(r[11].y);
                o[24] = a24 + __bfloat162float(b[12].x) + __bfloat162float(r[12].x);
                o[25] = a25 + __bfloat162float(b[12].y) + __bfloat162float(r[12].y);
                o[26] = a26 + __bfloat162float(b[13].x) + __bfloat162float(r[13].x);
                o[27] = a27 + __bfloat162float(b[13].y) + __bfloat162float(r[13].y);
                o[28] = a28 + __bfloat162float(b[14].x) + __bfloat162float(r[14].x);
                o[29] = a29 + __bfloat162float(b[14].y) + __bfloat162float(r[14].y);
                o[30] = a30 + __bfloat162float(b[15].x) + __bfloat162float(r[15].x);
                o[31] = a31 + __bfloat162float(b[15].y) + __bfloat162float(r[15].y);
#endif
#endif

                uint32_t p[16];
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;"
                        : "=r"(p[i]) : "f"(o[2*i]), "f"(o[2*i + 1]));
                }

                const uint32_t out_base = out_smem_arr[es] + (row_group * 32 + lane) * (SUBPASS_COLS * 2);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                    :: "r"(out_base +  0), "r"(p[0]), "r"(p[1]), "r"(p[2]), "r"(p[3]));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                    :: "r"(out_base + 16), "r"(p[4]), "r"(p[5]), "r"(p[6]), "r"(p[7]));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                    :: "r"(out_base + 32), "r"(p[8]), "r"(p[9]), "r"(p[10]), "r"(p[11]));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                    :: "r"(out_base + 48), "r"(p[12]), "r"(p[13]), "r"(p[14]), "r"(p[15]));

                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("bar.sync 0, 128;" ::: "memory");

                if (tid == 0) {
                    if (sp >= NUM_EPI_STAGES) {
                        asm volatile("cp.async.bulk.wait_group 1;" ::: "memory");
                    }
                    tma_store(out_smem_arr[es], &tma_c,
                              prev_n + nc, prev_m + cta_rank * ROWS_PER_CTA);
                    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                }

                asm volatile("bar.sync 0, 128;" ::: "memory");
            }

            if (tid == 0) {
                asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                mbar_arrive(mbar_tmem_cons_peer);
            }
#endif /* STRIP_EPILOGUE */
        }
    }
    else if (warp_id == 4) {
        /* =============== TMA A+B loader (warp 4) =============== */
        uint32_t tma_empty_phase[N_STAGES] = {0};
        const bool elect = (lane == 0);

        for (int tt = 0; tt < tiles_per_cluster; tt++) {
            const int lin_tile = cluster_id + tt * num_clusters;
            if (lin_tile >= TOTAL_TILES) break;
            const int swizzled = dgswizzle(lin_tile);
            const int tm = swizzled / TILES_N;
            const int tn = swizzled % TILES_N;
            const int a_m_tile = tm * 2 + cta_rank;
            const int b_n_half = tn * 2 + cta_rank;

            for (int ki = 0; ki < K_ITERS; ki++) {
                const int s = ki % N_STAGES;
                if (ki >= N_STAGES || tt > 0) {
                    mbar_wait(tma_empty_arr[s], tma_empty_phase[s]);
                    tma_empty_phase[s] ^= 1;
                }

                if (elect) {
                    const uint32_t a_dst = smem_a_arr[s];
                    const uint32_t b_dst = smem_b_arr[s];
                    const int tma_c0    = 0;
                    const int tma_a_c1  = (a_m_tile * K_ITERS + ki) * TM;
                    const int tma_b_c1  = (b_n_half * K_ITERS + ki) * (TN / 2);
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
                }
            }
        }
    }
    else if (warp_id == 5) {
        /* =============== TMA residual loader (warp 5) =============== */
#if !defined(STRIP_EPILOGUE) && !defined(GEMM_ONLY) && !defined(BIAS_ONLY)
        uint32_t res_empty_phase[NUM_EPI_STAGES] = {0, 0};
        const bool elect = (lane == 0);

        for (int tt = 0; tt < tiles_per_cluster; tt++) {
            const int lin_tile = cluster_id + tt * num_clusters;
            if (lin_tile >= TOTAL_TILES) break;
            const int swizzled = dgswizzle(lin_tile);
            const int tm = swizzled / TILES_N;
            const int tn = swizzled % TILES_N;
            const int prev_m = tm * TM * 2;
            const int prev_n = tn * TN;

            for (int sp = 0; sp < NUM_SUBPASSES; sp++) {
                const int es = sp & (NUM_EPI_STAGES - 1);
                if (sp >= NUM_EPI_STAGES || tt > 0) {
                    mbar_wait(res_empty_arr[es], res_empty_phase[es]);
                    res_empty_phase[es] ^= 1;
                }

                if (elect) {
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                        :: "r"(res_full_arr[es]), "r"(SUBPASS_BYTES) : "memory");
                    tma_load_cta(res_smem_arr[es], &tma_res,
                                 prev_n + sp * SUBPASS_COLS,
                                 prev_m + cta_rank * ROWS_PER_CTA,
                                 res_full_arr[es]);
                }
            }
        }
#endif
    }
    else if (warp_id == 6) {
        /* idle (mirror rank-1's dead path) */
    }
    else if (cta_rank == 0) {
        /* =============== MMA issuer (warp 7, CTA 0 only) =============== */
        uint32_t tma_full_phase[N_STAGES] = {0};
        uint32_t tmem_cons_phase = 0;

        uint64_t desc_a_base[N_STAGES], desc_b_base[N_STAGES];
        for (int s = 0; s < N_STAGES; s++) {
            desc_a_base[s] = make_smem_desc(smem_a_arr[s]);
            desc_b_base[s] = make_smem_desc(smem_b_arr[s]);
        }

        /* Initial consumed-arrives from both CTAs prime the first tile */
        if (lane == 0) {
            mbar_arrive(mbar_tmem_consumed);
            mbar_arrive(mbar_tmem_consumed);
        }

        for (int tt = 0; tt < tiles_per_cluster; tt++) {
            const int lin_tile = cluster_id + tt * num_clusters;
            if (lin_tile >= TOTAL_TILES) break;
            const int buf = lin_tile & 1;

            mbar_wait(mbar_tmem_consumed, tmem_cons_phase);
            tmem_cons_phase ^= 1;

            for (int ki = 0; ki < K_ITERS; ki++) {
                const int s = ki % N_STAGES;
                mbar_wait(tma_full_arr[s], tma_full_phase[s]);
                tma_full_phase[s] ^= 1;

                if (lane == 0) {
                    uint64_t desc_a = desc_a_base[s];
                    uint64_t desc_b = desc_b_base[s];
                    const int accum_flag = (ki == 0) ? 0 : 1;
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
                        "}"
                        :
                        : "r"(buf * TN), "l"(desc_a), "l"(desc_b), "r"(IDESC),
                          "r"(0),"r"(0),"r"(0),"r"(0),
                          "r"(0),"r"(0),"r"(0),"r"(0),
                          "r"(accum_flag));
                    tcgen05_commit_mcast(tma_empty_arr[s], pair_mask);
                }
            }

            if (lane == 0) tcgen05_commit_mcast(mbar_tmem_ready, pair_mask);
        }
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    if (warp_id == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
            :: "r"(0), "n"(TMEM_COLS));
    }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    printf("FC2 WS kernel — mimics rank-1 (256 thr, 8 warps, flat regs)\n");
    printf("  [%d,%d] x [%d,%d]^T  NS=%d  SMEM=%d B  (cap 228KB)\n",
           M_TOTAL, K_DIM, N_DIM, K_DIM, N_STAGES, SMEM_BYTES);
    if (SMEM_BYTES > 232448) {
        fprintf(stderr, "  ERROR: SMEM exceeds B200 cap\n"); return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));

    __nv_fp8_e4m3 *d_A=nullptr, *d_B=nullptr;
    __nv_bfloat16 *d_bias=nullptr, *d_residual=nullptr, *d_C=nullptr;

    size_t sA = (size_t)M_TOTAL * K_DIM;
    size_t sB = (size_t)N_DIM   * K_DIM;
    size_t sC = (size_t)M_TOTAL * N_DIM;
    CUDA_CHECK(cudaMalloc(&d_A, sA));
    CUDA_CHECK(cudaMalloc(&d_B, sB));
    CUDA_CHECK(cudaMalloc(&d_bias, (size_t)N_DIM * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_residual, sC * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_C, sC * sizeof(__nv_bfloat16)));

    __nv_fp8_e4m3 *hA = (__nv_fp8_e4m3*)malloc(sA);
    __nv_fp8_e4m3 *hB = (__nv_fp8_e4m3*)malloc(sB);
    __nv_bfloat16 *hbias = (__nv_bfloat16*)malloc((size_t)N_DIM * sizeof(__nv_bfloat16));
    __nv_bfloat16 *hres  = (__nv_bfloat16*)malloc(sC * sizeof(__nv_bfloat16));

    /* PACKED_TILES layout (fc2_w3 convention, TM per-CTA = 128):
       A packed with tile_m=TM=128 half-slabs.  Each (a_m_tile, k_block) pair
       is TM rows × TK cols contiguous.  a_m_tile ∈ [0, M_TOTAL/TM). */
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
    for (size_t i = 0; i < sC; i++) hres[i] = __float2bfloat16(0.125f);

    CUDA_CHECK(cudaMemcpy(d_A, hA, sA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, hB, sB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, hbias, (size_t)N_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_residual, hres, sC * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, sC * sizeof(__nv_bfloat16)));
    printf("  Alloc + init + pack done\n");

    CUtensorMap h_tma_a, h_tma_b, h_tma_c, h_tma_res;
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
    {
        uint64_t dims[2]    = {(uint64_t)N_DIM, (uint64_t)M_TOTAL};
        uint64_t strides[1] = {(uint64_t)N_DIM * sizeof(__nv_bfloat16)};
        uint32_t box[2]     = {SUBPASS_COLS, ROWS_PER_CTA};
        uint32_t estrides[2]= {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(&h_tma_res,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)d_residual,
            dims, strides, box, estrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }

    CUDA_CHECK(cudaFuncSetAttribute(fc2_ws_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    printf("  TMA descriptors + func attr done (SMEM=%d B)\n", SMEM_BYTES);

    dim3 grid(SM_COUNT, 1, 1);
#define LAUNCH_KERNEL() \
    fc2_ws_kernel<<<grid, THREADS, SMEM_BYTES>>>( \
        h_tma_a, h_tma_b, h_tma_c, h_tma_res, d_bias, d_C, d_residual)

    printf("Warmup (2 iters)...\n");
    for (int i = 0; i < 2; i++) LAUNCH_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Timing 10 iters...\n");
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 10; i++) LAUNCH_KERNEL();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= 10.0f;
    printf("FC2-WS kernel: %.3f ms  %.2f TFLOPS\n",
           ms, 2.0 * M_TOTAL * N_DIM * K_DIM / ms / 1e9);

#if defined(STRIP_EPILOGUE)
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
#elif defined(BIAS_ONLY)
        float expected = expected_ab + __bfloat162float(hbias[col]);
#else
        float expected = expected_ab + __bfloat162float(hbias[col]) + __bfloat162float(hres[row * N_DIM + col]);
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

    free(hA); free(hB); free(hbias); free(hres);
#ifndef STRIP_EPILOGUE
    free(h_C);
#endif
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_bias); cudaFree(d_residual); cudaFree(d_C);
    return errors == 0 ? 0 : 1;
}
