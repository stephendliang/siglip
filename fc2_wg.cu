/*
fc2_wg.cu — FC2 warpgroup-specialized kernel, standalone.

Architecture shape matches rank-1 (nvjet_sm100_qqtst_128x256_128x6_2x1_2cta_v):
  256 threads = 2 warpgroups (WG0 consumer, WG1 producer).

  Consumer WG (tid 0-127, setmaxnreg.inc 232):
    Unified control flow — all 128 threads walk the same epilogue loop.
    Per-warp offsets (row group 0..3) are derived from wg_warp, stored in
    registers ptxas can lift to uniform class.

  Producer WG (tid 128-255, setmaxnreg.dec 40):
    Sub-specialized by wg_warp (the way rank-1 does it — BRA per warp, but
    within each sub-warp the control flow is uniform across all 32 lanes):
      wg_warp 0 (tid 128-159): TMA A+B loader for every K-stage of every tile.
      wg_warp 1 (tid 160-191): TMA residual loader (per tile, 4 slices).
      wg_warp 2 (tid 192-223, CTA 0 only): MMA issuer via tcgen05.mma.cta_group::2.
      wg_warp 3 + (wg_warp 2 CTA 1): exit.

  Inside every sub-warp the TMA/MMA asm blocks are NOT lane-gated. All 32
  lanes issue the same instruction with uniform operands; the hardware
  collapses to a single uniform-datapath op (UTMALDG / UTCQMMA). That is
  what keeps ELECT + BSYNC out of the final SASS.

Layout:
  cta_group::2 cluster (2 CTAs), NS=6 pipeline stages, dgswizzle dispatch,
  PACKED_TILES DRAM layout. No experimental variants.

Supported compile flags:
  -DGEMM_ONLY        D = BF16(A·B), no bias/residual (baseline vs cublas gemm)
  -DSTRIP_EPILOGUE   Consumer arrives at barriers only (MMA-core baseline)
*/

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define SM_COUNT     148

#ifndef M_TOTAL
#define M_TOTAL      928256
#endif
#ifndef N_DIM
#define N_DIM        768
#endif
#ifndef K_DIM
#define K_DIM        3072
#endif

#define TM           128
#define TN           256
#define TK           128
#define TILES_M      ((M_TOTAL + TM * 2 - 1) / (TM * 2))
#define TILES_N      (N_DIM / TN)
#define TOTAL_TILES  (TILES_M * TILES_N)
#define K_ITERS      (K_DIM / TK)
#define MMA_K        32
#define MMA_PER_KI   (TK / MMA_K)
static_assert(MMA_PER_KI == 4, "MMA_PER_KI must be 4 for this kernel");

#define N_STAGES     6
#define NUM_EPI_STAGES 2
#define NUM_EPI_SUBITERS 4

#define STAGE_BYTES  32768

#define TMA_BOX_COLS 64
#define TMA_BOX_ROWS 32
#define EPI_COL_STRIDE 64
#define EPI_PASS_CHUNKS 2

#define STAGING_REGION_ROW_BYTES 128
#define STAGING_REGION_BYTES     (32 * STAGING_REGION_ROW_BYTES)
#define STAGE_BLOCK_BYTES        (4 * STAGING_REGION_BYTES)
#define EPI_STAGE_BYTES          (NUM_EPI_SUBITERS * STAGE_BLOCK_BYTES)

#define OFF_TMEM           (N_STAGES * STAGE_BYTES)
#define OFF_TMA_MBAR       (OFF_TMEM + 8)
#define OFF_MMA_MBAR       (OFF_TMA_MBAR + N_STAGES * 8)
#define OFF_MAINLOOP_MBAR  (OFF_MMA_MBAR + N_STAGES * 8)
#define OFF_EPILOGUE_MBAR  (OFF_MAINLOOP_MBAR + 16)
#define OFF_LOAD_MBAR      (OFF_EPILOGUE_MBAR + 16)
#define OFF_LOAD_CONSUMED  (OFF_LOAD_MBAR + NUM_EPI_STAGES * 8)
#define OFF_BIAS_SMEM      ((OFF_LOAD_CONSUMED + NUM_EPI_STAGES * 8 + 15) & ~15)
#define BIAS_SMEM_BYTES    (N_DIM * 2)
#define OFF_STAGING        ((OFF_BIAS_SMEM + BIAS_SMEM_BYTES + 1023) & ~1023)
#define SMEM_BYTES         ((OFF_STAGING + NUM_EPI_STAGES * EPI_STAGE_BYTES + 127) & ~127)

#define TMEM_COLS   512
#define IDESC       0x10400010U
#define SBO         1024

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
int dgswizzle(int block_idx) {
    const int group_tiles = TILES_N * DG_GROUP_SIZE;
    const int group_idx = block_idx / group_tiles;
    const int first_m = group_idx * DG_GROUP_SIZE;
    const int in_group = block_idx % group_tiles;
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

__global__ void __launch_bounds__(256, 1)
__cluster_dims__(2, 1, 1)
fc2_wg_kernel(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    const __grid_constant__ CUtensorMap tma_c,
    const __grid_constant__ CUtensorMap tma_res,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ C,
    const __nv_bfloat16* __restrict__ residual
) {
    extern __shared__ __align__(128) char smem[];
    const int sm_id = blockIdx.x;
    const int cta_rank = sm_id & 1;
    const int cluster_id = sm_id >> 1;
    const int num_clusters = SM_COUNT / 2;
    const int tid = threadIdx.x;
    const int wg_id = tid >> 7;
    const int wg_tid = tid & 127;
    const int wg_warp = wg_tid >> 5;
    const int lane = tid & 31;

    if (tid == 0) {
        for (int s = 0; s < N_STAGES; s++) {
            mbar_init(smem_to_uint(smem + OFF_TMA_MBAR + s * 8), 2);
            mbar_init(smem_to_uint(smem + OFF_MMA_MBAR + s * 8), 1);
        }
        for (int i = 0; i < 2; i++) {
            mbar_init(smem_to_uint(smem + OFF_MAINLOOP_MBAR + i * 8), 1);
            mbar_init(smem_to_uint(smem + OFF_EPILOGUE_MBAR + i * 8), 256);
        }
        for (int s = 0; s < NUM_EPI_STAGES; s++) {
            mbar_init(smem_to_uint(smem + OFF_LOAD_MBAR + s * 8), 1);
            mbar_init(smem_to_uint(smem + OFF_LOAD_CONSUMED + s * 8), 128);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    if (wg_id == 1 && wg_warp == 0) {
        asm volatile(
            "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(smem_to_uint(smem + OFF_TMEM)), "n"(TMEM_COLS));
        asm volatile(
            "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
    asm volatile("bar.sync 0, 256;");
    const uint32_t taddr_base = *reinterpret_cast<uint32_t*>(smem + OFF_TMEM);

    uint32_t smem_a[N_STAGES], smem_b[N_STAGES];
    uint32_t tma_mbar_arr[N_STAGES], mma_mbar_arr[N_STAGES];
    for (int s = 0; s < N_STAGES; s++) {
        smem_a[s] = smem_to_uint(smem + s * STAGE_BYTES);
        smem_b[s] = smem_to_uint(smem + s * STAGE_BYTES + 16384);
        tma_mbar_arr[s] = smem_to_uint(smem + OFF_TMA_MBAR + s * 8);
        mma_mbar_arr[s] = smem_to_uint(smem + OFF_MMA_MBAR + s * 8);
    }
    const uint32_t mainloop_mbar_a = smem_to_uint(smem + OFF_MAINLOOP_MBAR);
    const uint32_t epi_mbar_a = smem_to_uint(smem + OFF_EPILOGUE_MBAR);
    const uint32_t load_mbar_a = smem_to_uint(smem + OFF_LOAD_MBAR);
    const uint32_t load_consumed_a = smem_to_uint(smem + OFF_LOAD_CONSUMED);
    const uint32_t epi_mbar_masked = epi_mbar_a & 0xFEFFFFFFU;

    if (wg_id == 0) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(232));
    } else {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(40));
    }

#ifndef GEMM_ONLY
    if (wg_id == 0) {
        const int bi = wg_tid * 4;
        if (bi < N_DIM) {
            uint4 v = *reinterpret_cast<const uint4*>(bias + bi);
            uint32_t sm = smem_to_uint(smem + OFF_BIAS_SMEM + bi * 2);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                :: "r"(sm), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
        }
        asm volatile("bar.sync 1, 128;");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
#endif

    if (wg_id == 1) {
        if (wg_warp == 0) {
            uint32_t mma_phase[N_STAGES] = {0};
            int stage = 0;
            int round = 0;

            const int num_tiles_my = (TOTAL_TILES - cluster_id + num_clusters - 1) / num_clusters;
            for (int tc = 0; tc < num_tiles_my; tc++) {
                const int block_idx = tc * num_clusters + cluster_id;
                const int flat = dgswizzle(block_idx);
                const int tm = flat / TILES_N;
                const int tn = flat - tm * TILES_N;
                const int packed_m = (tm * 2 + cta_rank) * TILES_N;

                #pragma unroll 1
                for (int ki = 0; ki < K_ITERS; ki++) {
                    if (round >= 1) {
                        mbar_wait(mma_mbar_arr[stage], mma_phase[stage]);
                        mma_phase[stage] ^= 1;
                    }

                    const int a_row = (packed_m * TM) + ki * TM;
                    const int b_row_base = tn * (TN / 2) * K_ITERS + ki * (TN / 2);

                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2"
                        " [%0], [%1, {%2, %3}], [%4];\n\t"
                        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2"
                        " [%5], [%6, {%2, %7}], [%4];\n\t"
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%4], %8;"
                        :: "r"(smem_a[stage]), "l"(&tma_a), "r"(0), "r"(a_row),
                           "r"(tma_mbar_arr[stage]),
                           "r"(smem_b[stage]), "l"(&tma_b), "r"(b_row_base),
                           "r"(32768)
                        : "memory");

                    stage++;
                    if (stage == N_STAGES) { stage = 0; round++; }
                }
            }
        } else if (wg_warp == 1) {
#if !defined(GEMM_ONLY) && !defined(STRIP_EPILOGUE)
            uint32_t consumed_phase = 1;
            int issue_count = 0;
            const int num_tiles_my = (TOTAL_TILES - cluster_id + num_clusters - 1) / num_clusters;

            for (int tc = 0; tc < num_tiles_my; tc++) {
                const int block_idx = tc * num_clusters + cluster_id;
                const int flat = dgswizzle(block_idx);
                const int tm = flat / TILES_N;
                const int tn = flat - tm * TILES_N;
                const int epi_stage = tc & (NUM_EPI_STAGES - 1);

                if (issue_count >= NUM_EPI_STAGES) {
                    mbar_wait(load_consumed_a + epi_stage * 8, consumed_phase);
                    if (epi_stage == NUM_EPI_STAGES - 1) consumed_phase ^= 1;
                }

                const int packed_m = ((tm * 2 + cta_rank) * TILES_N + tn) * TM;
                const uint32_t stage_base = smem_to_uint(smem + OFF_STAGING + epi_stage * EPI_STAGE_BYTES);
                const uint32_t lmbar = load_mbar_a + epi_stage * 8;
                mbar_arrive_expect_tx(lmbar, EPI_STAGE_BYTES);

                #pragma unroll
                for (int si = 0; si < NUM_EPI_SUBITERS; si++) {
                    const uint32_t dst = stage_base + si * STAGE_BLOCK_BYTES;
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                        " [%0], [%1, {%2, %3}], [%4];"
                        :: "r"(dst), "l"(&tma_res),
                           "r"(si * EPI_COL_STRIDE), "r"(packed_m),
                           "r"(lmbar)
                        : "memory");
                }
                issue_count++;
            }
#endif
        } else if (wg_warp == 2 && cta_rank == 0) {
            uint32_t tma_phase_arr[N_STAGES] = {0};
            uint32_t epi_phase_c[2] = {0, 0};
            int stage = 0;

            uint64_t desc_a_base[N_STAGES], desc_b_base[N_STAGES];
            #pragma unroll
            for (int s = 0; s < N_STAGES; s++) {
                desc_a_base[s] = make_smem_desc(smem_a[s]);
                desc_b_base[s] = make_smem_desc(smem_b[s]);
            }

            const int num_tiles_my = (TOTAL_TILES - cluster_id + num_clusters - 1) / num_clusters;

            for (int tc = 0; tc < num_tiles_my; tc++) {
                const int buf = tc & 1;
                const uint32_t tc_tmem = (uint32_t)(buf * TN);

                if (tc >= 2) {
                    mbar_wait(epi_mbar_a + buf * 8, epi_phase_c[buf]);
                    epi_phase_c[buf] ^= 1;
                }

                asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

                {
                    mbar_wait(tma_mbar_arr[stage], tma_phase_arr[stage]);
                    tma_phase_arr[stage] ^= 1;
                    uint64_t desc_a = desc_a_base[stage];
                    uint64_t desc_b = desc_b_base[stage];
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
                        : "r"(tc_tmem), "l"(desc_a), "l"(desc_b), "r"(IDESC),
                          "r"(0),"r"(0),"r"(0),"r"(0),
                          "r"(0),"r"(0),"r"(0),"r"(0));
                    asm volatile(
                        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                        :: "r"(mma_mbar_arr[stage]), "h"((uint16_t)0x3)
                        : "memory");
                    stage++;
                    if (stage == N_STAGES) stage = 0;
                }

                #pragma unroll 1
                for (int ki = 1; ki < K_ITERS; ki++) {
                    mbar_wait(tma_mbar_arr[stage], tma_phase_arr[stage]);
                    tma_phase_arr[stage] ^= 1;
                    uint64_t desc_a = desc_a_base[stage];
                    uint64_t desc_b = desc_b_base[stage];
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p_acc;\n\t"
                        ".reg .b64 da, db;\n\t"
                        ".reg .b32 tc;\n\t"
                        "setp.ne.b32 p_acc, 1, 0;\n\t"
                        "mov.b32 tc, %0;\n\t"
                        "mov.b64 da, %1;\n\t"
                        "mov.b64 db, %2;\n\t"
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
                        "add.s64 da, da, 2;\n\t"
                        "add.s64 db, db, 2;\n\t"
                        "tcgen05.mma.cta_group::2.kind::f8f6f4 "
                        "[tc], da, db, %3, {%4,%5,%6,%7, %8,%9,%10,%11}, p_acc;\n\t"
                        "}"
                        :
                        : "r"(tc_tmem), "l"(desc_a), "l"(desc_b), "r"(IDESC),
                          "r"(0),"r"(0),"r"(0),"r"(0),
                          "r"(0),"r"(0),"r"(0),"r"(0));
                    asm volatile(
                        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                        :: "r"(mma_mbar_arr[stage]), "h"((uint16_t)0x3)
                        : "memory");
                    stage++;
                    if (stage == N_STAGES) stage = 0;
                }

                asm volatile(
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                    :: "r"(mainloop_mbar_a + buf * 8), "h"((uint16_t)0x3)
                    : "memory");
            }
        }
    } else {
        uint32_t main_phase_arr[2] = {0, 0};
#ifndef GEMM_ONLY
        uint32_t res_phase_arr[NUM_EPI_STAGES] = {0, 0};
#endif
        const int num_tiles_my = (TOTAL_TILES - cluster_id + num_clusters - 1) / num_clusters;

#ifdef STRIP_EPILOGUE
        for (int tc = 0; tc < num_tiles_my; tc++) {
            const int buf = tc & 1;
            mbar_wait(mainloop_mbar_a + buf * 8, main_phase_arr[buf]);
            main_phase_arr[buf] ^= 1;
            mbar_arrive(epi_mbar_masked + buf * 8);
        }
#else
        const uint32_t bias_saddr = smem_to_uint(smem + OFF_BIAS_SMEM);
        const uint32_t xor_val = (uint32_t)(lane & 7) << 4;
        const uint32_t sw0 = 0 ^ xor_val, sw1 = 16 ^ xor_val;
        const uint32_t sw2 = 32 ^ xor_val, sw3 = 48 ^ xor_val;
        const uint32_t sw4 = 64 ^ xor_val, sw5 = 80 ^ xor_val;
        const uint32_t sw6 = 96 ^ xor_val, sw7 = 112 ^ xor_val;

        for (int tc = 0; tc < num_tiles_my; tc++) {
            const int block_idx = tc * num_clusters + cluster_id;
            const int flat = dgswizzle(block_idx);
            const int tm = flat / TILES_N;
            const int tn = flat - tm * TILES_N;
            const int buf = tc & 1;
            const int epi_stage = tc & (NUM_EPI_STAGES - 1);

            mbar_wait(mainloop_mbar_a + buf * 8, main_phase_arr[buf]);
            main_phase_arr[buf] ^= 1;
#ifndef GEMM_ONLY
            mbar_wait(load_mbar_a + epi_stage * 8, res_phase_arr[epi_stage]);
            res_phase_arr[epi_stage] ^= 1;
#endif

            const int row_group = wg_warp;
            const int prev_n_bias = tn * TN;
            const uint32_t taddr_rg = taddr_base + (uint32_t)(buf * TN)
                + ((uint32_t)(cta_rank * 128 + row_group * 32) << 16);

            #pragma unroll 1
            for (int si = 0; si < NUM_EPI_SUBITERS; si++) {
                const int nc_base = si * EPI_COL_STRIDE;
                const int stage_ = epi_stage;
                const uint32_t stage_base = smem_to_uint(smem + OFF_STAGING
                    + stage_ * EPI_STAGE_BYTES
                    + si * STAGE_BLOCK_BYTES
                    + row_group * STAGING_REGION_BYTES
                    + lane * STAGING_REGION_ROW_BYTES);

                float a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15;
                float a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31;

                #pragma unroll
                for (int _ci = 0; _ci < EPI_PASS_CHUNKS; _ci++) {
                    const int chunk = _ci;
                    const int nc = nc_base + chunk * 32;
                    TMEM_LOAD_X32(a0,a1,a2,a3,a4,a5,a6,a7,
                                  a8,a9,a10,a11,a12,a13,a14,a15,
                                  a16,a17,a18,a19,a20,a21,a22,a23,
                                  a24,a25,a26,a27,a28,a29,a30,a31,
                                  taddr_rg + nc);

                    const uint32_t rsw0 = (chunk & 1) ? sw4 : sw0;
                    const uint32_t rsw1 = (chunk & 1) ? sw5 : sw1;
                    const uint32_t rsw2 = (chunk & 1) ? sw6 : sw2;
                    const uint32_t rsw3 = (chunk & 1) ? sw7 : sw3;
                    TMEM_WAIT();

#ifdef GEMM_ONLY
                    GEMM_CVT_STS(a0,a1,a2,a3,a4,a5,a6,a7, stage_base + rsw0);
                    GEMM_CVT_STS(a8,a9,a10,a11,a12,a13,a14,a15, stage_base + rsw1);
                    GEMM_CVT_STS(a16,a17,a18,a19,a20,a21,a22,a23, stage_base + rsw2);
                    GEMM_CVT_STS(a24,a25,a26,a27,a28,a29,a30,a31, stage_base + rsw3);
#else
                    const uint32_t bs = bias_saddr + (uint32_t)((prev_n_bias + nc) * 2);
                    uint4 bv0, bv1, bv2, bv3;
                    asm("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(bv0.x),"=r"(bv0.y),"=r"(bv0.z),"=r"(bv0.w) : "r"(bs));
                    asm("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(bv1.x),"=r"(bv1.y),"=r"(bv1.z),"=r"(bv1.w) : "r"(bs + 16));
                    asm("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(bv2.x),"=r"(bv2.y),"=r"(bv2.z),"=r"(bv2.w) : "r"(bs + 32));
                    asm("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(bv3.x),"=r"(bv3.y),"=r"(bv3.z),"=r"(bv3.w) : "r"(bs + 48));

                    uint4 rv0, rv1, rv2, rv3;
                    asm("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(rv0.x),"=r"(rv0.y),"=r"(rv0.z),"=r"(rv0.w) : "r"(stage_base + rsw0));
                    asm("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(rv1.x),"=r"(rv1.y),"=r"(rv1.z),"=r"(rv1.w) : "r"(stage_base + rsw1));
                    asm("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(rv2.x),"=r"(rv2.y),"=r"(rv2.z),"=r"(rv2.w) : "r"(stage_base + rsw2));
                    asm("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(rv3.x),"=r"(rv3.y),"=r"(rv3.z),"=r"(rv3.w) : "r"(stage_base + rsw3));

                    BIAS_RES_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7,
                                        bv0.x, bv0.y, bv0.z, bv0.w,
                                        rv0.x, rv0.y, rv0.z, rv0.w,
                                        stage_base + rsw0);
                    BIAS_RES_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15,
                                        bv1.x, bv1.y, bv1.z, bv1.w,
                                        rv1.x, rv1.y, rv1.z, rv1.w,
                                        stage_base + rsw1);
                    BIAS_RES_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23,
                                        bv2.x, bv2.y, bv2.z, bv2.w,
                                        rv2.x, rv2.y, rv2.z, rv2.w,
                                        stage_base + rsw2);
                    BIAS_RES_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31,
                                        bv3.x, bv3.y, bv3.z, bv3.w,
                                        rv3.x, rv3.y, rv3.z, rv3.w,
                                        stage_base + rsw3);
#endif
                }

                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("bar.sync 1, 128;");

                const int prev_m_global = ((tm * 2 + cta_rank) * TILES_N + tn) * TM;
                const int cc = 0;
                const int cr = prev_m_global + row_group * 32;
                const uint32_t src_smem = smem_to_uint(smem + OFF_STAGING
                    + stage_ * EPI_STAGE_BYTES
                    + si * STAGE_BLOCK_BYTES
                    + row_group * STAGING_REGION_BYTES);

                asm volatile(
                    "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
                    " [%0, {%1, %2}], [%3];"
                    :: "l"(&tma_c), "r"(cc + nc_base), "r"(cr), "r"(src_smem)
                    : "memory");
                asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                if (si == NUM_EPI_SUBITERS - 1) {
                    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
                } else {
                    asm volatile("cp.async.bulk.wait_group.read 3;" ::: "memory");
                }
            }

#ifndef GEMM_ONLY
            mbar_arrive(load_consumed_a + epi_stage * 8);
#endif
            mbar_arrive(epi_mbar_masked + buf * 8);
        }
#endif
    }
}

__global__ void init_A(uint8_t* A, long long total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int row = (int)(idx / K_DIM);
    A[idx] = (uint8_t)(0x38 + (row & 7));
}

__global__ void init_residual(__nv_bfloat16* res, int n_dim, long long total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int col = (int)(idx % n_dim);
    int row = (int)(idx / n_dim);
    res[idx] = __float2bfloat16((float)(row % 128) * 0.25f + (float)col * 0.125f);
}

__global__ void pack_u8(uint8_t* dst, const uint8_t* src, int M, int K, int tile_m, int tile_k) {
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

__global__ void pack_bf16(__nv_bfloat16* dst, const __nv_bfloat16* src, int M, int N, int tile_m, int tile_n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)M * N;
    if (idx >= total) return;
    int m = (int)(idx / N);
    int n = (int)(idx % N);
    int tiles_n = N / tile_n;
    long long packed = (long long)(m / tile_m) * tiles_n * tile_m * tile_n
                     + (long long)(n / tile_n) * tile_m * tile_n
                     + (long long)(m % tile_m) * tile_n + (n % tile_n);
    dst[packed] = src[idx];
}

int main() {
    setbuf(stdout, NULL);
    printf("FC2 WG kernel — warpgroup-specialized (consumer + producer)\n");
    printf("  GEMM: [%d,%d] x [%d,%d]^T  NS=%d  SMEM=%d B\n",
           M_TOTAL, K_DIM, N_DIM, K_DIM, N_STAGES, SMEM_BYTES);

    uint8_t *d_A, *d_B;
    __nv_bfloat16 *d_bias, *d_residual, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A,        (size_t)M_TOTAL * K_DIM));
    CUDA_CHECK(cudaMalloc(&d_B,        (size_t)N_DIM   * K_DIM));
    CUDA_CHECK(cudaMalloc(&d_bias,     (size_t)N_DIM   * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_residual, (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_C,        (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16)));

    {
        long long total = (long long)M_TOTAL * K_DIM;
        int tpb = 256;
        init_A<<<(int)((total+tpb-1)/tpb), tpb>>>(d_A, total);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        uint8_t* h_B = (uint8_t*)malloc((size_t)N_DIM * K_DIM);
        for (int n = 0; n < N_DIM; n++)
            memset(h_B + (size_t)n * K_DIM, (n & 1) ? 0x38 : 0x3C, K_DIM);
        CUDA_CHECK(cudaMemcpy(d_B, h_B, (size_t)N_DIM * K_DIM, cudaMemcpyHostToDevice));
        free(h_B);
    }
    {
        __nv_bfloat16* h_bias = (__nv_bfloat16*)malloc((size_t)N_DIM * sizeof(__nv_bfloat16));
        for (int c = 0; c < N_DIM; c++)
            h_bias[c] = __float2bfloat16((float)(c + 1));
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias, (size_t)N_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        free(h_bias);
    }
    {
        long long total = (long long)M_TOTAL * N_DIM;
        int tpb = 256;
        init_residual<<<(int)((total+tpb-1)/tpb), tpb>>>(d_residual, N_DIM, total);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        int tpb = 256;
        uint8_t* d_tmp;
        CUDA_CHECK(cudaMalloc(&d_tmp, (size_t)M_TOTAL * K_DIM));
        long long n = (long long)M_TOTAL * K_DIM;
        pack_u8<<<(int)((n+tpb-1)/tpb), tpb>>>(d_tmp, d_A, M_TOTAL, K_DIM, TM, TK);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(d_A, d_tmp, (size_t)M_TOTAL * K_DIM, cudaMemcpyDeviceToDevice));
        cudaFree(d_tmp);

        CUDA_CHECK(cudaMalloc(&d_tmp, (size_t)N_DIM * K_DIM));
        n = (long long)N_DIM * K_DIM;
        pack_u8<<<(int)((n+tpb-1)/tpb), tpb>>>(d_tmp, d_B, N_DIM, K_DIM, TN/2, TK);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(d_B, d_tmp, (size_t)N_DIM * K_DIM, cudaMemcpyDeviceToDevice));
        cudaFree(d_tmp);

        __nv_bfloat16* d_tmp16;
        CUDA_CHECK(cudaMalloc(&d_tmp16, (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16)));
        n = (long long)M_TOTAL * N_DIM;
        pack_bf16<<<(int)((n+tpb-1)/tpb), tpb>>>(d_tmp16, d_residual, M_TOTAL, N_DIM, TM, TN);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(d_residual, d_tmp16, (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToDevice));
        cudaFree(d_tmp16);

        CUDA_CHECK(cudaDeviceSynchronize());
    }
    printf("  Alloc + init + pack done\n");

    CUtensorMap h_tma_a, h_tma_b, h_tma_c, h_tma_res;
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
    {
        uint64_t c_total_rows = (uint64_t)(M_TOTAL / TM) * TILES_N * TM;
        uint64_t dims[2]    = {(uint64_t)TN, c_total_rows};
        uint64_t strides[1] = {(uint64_t)TN * sizeof(__nv_bfloat16)};
        uint32_t box[2]     = {TMA_BOX_COLS, TMA_BOX_ROWS};
        uint32_t estrides[2]= {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(&h_tma_c,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)d_C,
            dims, strides, box, estrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }
    {
        uint64_t r_total_rows = (uint64_t)(M_TOTAL / TM) * TILES_N * TM;
        uint64_t dims[2]    = {(uint64_t)TN, r_total_rows};
        uint64_t strides[1] = {(uint64_t)TN * sizeof(__nv_bfloat16)};
        uint32_t box[2]     = {TMA_BOX_COLS, TM};
        uint32_t estrides[2]= {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(&h_tma_res,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)d_residual,
            dims, strides, box, estrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }

    CUDA_CHECK(cudaFuncSetAttribute(fc2_wg_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    printf("  TMA descriptors + func attr done (SMEM=%d B)\n", SMEM_BYTES);

    dim3 grid(SM_COUNT, 1, 1);
#define LAUNCH_KERNEL() \
    fc2_wg_kernel<<<grid, 256, SMEM_BYTES>>>( \
        h_tma_a, h_tma_b, h_tma_c, h_tma_res, d_bias, d_C, d_residual)

    printf("Warmup (2 iters)...\n");
    for (int i = 0; i < 2; i++) LAUNCH_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Timing 10 iters...\n");
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 10; i++) LAUNCH_KERNEL();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= 10.0f;
    printf("FC2-WG kernel: %.3f ms  %.2f TFLOPS\n",
           ms, 2.0 * M_TOTAL * N_DIM * K_DIM / ms / 1e9);

    LAUNCH_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    __nv_bfloat16* h_C = (__nv_bfloat16*)malloc((size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    auto a_val_of = [](long long r) {
        return 1.0f + 0.125f * (float)((int)r & 7);
    };
    int errors = 0;
    for (int spot = 0; spot < 32; spot++) {
        long long row = (long long)spot * M_TOTAL / 32;
        int col = (spot * 47) % N_DIM;
        float b_val = (col & 1) ? 1.0f : 1.5f;
        float a_val = a_val_of(row);
        float gemm = (float)K_DIM * a_val * b_val;
        float res_bf16_f = __bfloat162float(__float2bfloat16(
            (float)((int)row % 128) * 0.25f + (float)col * 0.125f));
        float bias_bf16_f = __bfloat162float(__float2bfloat16((float)(col + 1)));
#ifdef GEMM_ONLY
        __nv_bfloat16 expected = __float2bfloat16(gemm);
#else
        __nv_bfloat16 expected = __float2bfloat16(
            __bfloat162float(__float2bfloat16(gemm + bias_bf16_f)) + res_bf16_f);
#endif
        __nv_bfloat16 got = h_C[(long long)row * N_DIM + col];
        float fexp = __bfloat162float(expected);
        float fgot = __bfloat162float(got);
        if (fabsf(fgot - fexp) > fabsf(fexp) * 0.01f + 1e-3f) {
            if (errors < 4)
                printf("  MISMATCH row=%lld col=%d  got=%f  expected=%f\n",
                       row, col, fgot, fexp);
            errors++;
        }
    }
    if (errors == 0) printf("VALID (32 spot checks passed)\n");
    else printf("INVALID (%d/32 spot checks failed)\n", errors);

    free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_bias); cudaFree(d_residual); cudaFree(d_C);
    return errors == 0 ? 0 : 1;
}
