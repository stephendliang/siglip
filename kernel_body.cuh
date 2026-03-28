/*
kernel_body.cuh — shared kernel body for tcgen05 persistent GEMM kernels
Contains epilogue_store template and persistent_gemm kernel template.
Each .cu file: #define N_DIM → #include "kernel_common.cuh" → define transform macros → #include "kernel_body.cuh"
*/

#pragma once

#if REG_PAD
static __device__ volatile int d_regpad_skip = 1;
#endif

// Epilogue operation selector
enum class EpilogueOp : int { BIAS_ADD = 0, BIAS_GELU = 1, BIAS_RESIDUAL = 2 };

template<EpilogueOp Op> struct EpilogueSideData;
template<> struct EpilogueSideData<EpilogueOp::BIAS_ADD>      { using type = const __nv_bfloat16*; };
template<> struct EpilogueSideData<EpilogueOp::BIAS_GELU>     { using type = const float*; };
template<> struct EpilogueSideData<EpilogueOp::BIAS_RESIDUAL>  {
#if BIAS_BF16
    using type = const __nv_bfloat16*;
#else
    using type = const float*;
#endif
};
template<EpilogueOp Op> using SideDataPtr = typename EpilogueSideData<Op>::type;

/*
Stub macros for dead if-constexpr branches —
preprocessor expands before template instantiation; stubs provide syntactically valid no-ops.
*/
#ifndef CVT_ADD_STS_V4
#define CVT_ADD_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, c0,c1,c2,c3, SADDR) ((void)0)
constexpr bool HAS_CVT_ADD = false;
#else
constexpr bool HAS_CVT_ADD = true;
#endif
#ifndef GELU_CVT_STS_V4
#define GELU_CVT_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, b0,b1,b2,b3,b4,b5,b6,b7, SADDR) ((void)0)
constexpr bool HAS_GELU_CVT = false;
#else
constexpr bool HAS_GELU_CVT = true;
#endif
#ifndef BIAS_RES_CVT_STS_V4
#define BIAS_RES_CVT_STS_V4(f0,f1,f2,f3,f4,f5,f6,f7, b0,b1,b2,b3,b4,b5,b6,b7, r0,r1,r2,r3, SADDR) ((void)0)
constexpr bool HAS_BIAS_RES_CVT = false;
#else
constexpr bool HAS_BIAS_RES_CVT = true;
#endif
// Stub COMB_* constants for dead BIAS_ADD branch in fc1_gelu compilation
#ifndef COMB_BLOCK_ROWS
#define COMB_BLOCK_ROWS 1
#define COMB_BLOCK_COLS 1
#define COMB_COL_BLOCKS 1
#define COMB_BLOCK_ELEMS 1
#endif

/*
Epilogue: TMEM → transform → CVT → swizzle SMEM regions → TMA tensor stores
Transform selected by Op: BIAS_ADD (combined table), BIAS_GELU (bias+GELU), BIAS_RESIDUAL (bias+residual)
*/

template<int NC_START, int NC_END, EpilogueOp Op, bool FIRST_PASS_PRELOADED = false>
static __device__
#if EPI_NOINLINE
__noinline__
#else
__forceinline__
#endif
void epilogue_store(
    uint32_t tmem_addr,
    int row_group,
    int lane,
    int gm_base,
    int n_start,
    SideDataPtr<Op> __restrict__ side_data,
    __nv_bfloat16* __restrict__ C,
    const __nv_bfloat16* __restrict__ residual,
    int cta_rank,
    uint32_t staging_saddr,
    uint32_t epi_mbar_addr,
    const CUtensorMap* tma_c_desc
#if TMA_RESIDUAL
    , const CUtensorMap* tma_res_desc
    , uint32_t res_mbar_addr
    , uint32_t res_staging_saddr
#endif
#if STAGES_C >= 2
    , uint32_t sc_release_mbar_base
    , int sc_phase
#endif
#if W0_RES_PREFETCH || W0_RES_FULL || EPI_LOAD_WARP
    , uint32_t res_consumed_mbar_addr
#endif
#if !EPI_PIPELINE && (W0_RES_FULL || EPI_LOAD_WARP)
    , uint32_t res_pass_mbar_addr
#endif
#if EPI_PIPELINE
    , int pipe_load_phase
#endif
#ifdef TIMING
    , long long& t_phase1_end
#endif
) {
#if BIAS_SMEM
    extern __shared__ __align__(128) char smem[];
#endif
    const int taddr_base = tmem_addr + ((cta_rank * 128 + row_group * 32) << 16);

    // BIAS_ADD: precompute combined table base pointer (pos_row computed here)
    const __nv_bfloat16* comb_base = nullptr;
    if constexpr (Op == EpilogueOp::BIAS_ADD) {
        const int pos_row = (gm_base + lane) % SEQ_LEN;
        const int comb_block_row = pos_row / COMB_BLOCK_ROWS;
        const int comb_row_in_blk = pos_row % COMB_BLOCK_ROWS;
        comb_base = side_data
            + (long long)comb_block_row * COMB_COL_BLOCKS * COMB_BLOCK_ELEMS
            + comb_row_in_blk * COMB_BLOCK_COLS;
    }

    // BIAS_RESIDUAL: precompute row pointer into full residual matrix
    const __nv_bfloat16* res_row = nullptr;
    if constexpr (Op == EpilogueOp::BIAS_RESIDUAL) {
        res_row = residual + (long long)(gm_base + lane) * N_DIM + n_start;
    }

    constexpr int N_REGIONS = (NC_END - NC_START) / 64;

    // Wait for previous tile's Phase 2 TMA stores before overwriting staging.
    if (lane == 0) {
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
    __syncwarp();

    // Swizzle constants (loop-invariant)
    const uint32_t xor_val = (lane & 7) << 4;
    const uint32_t srow_base = staging_saddr + lane * STAGING_REGION_ROW_BYTES;

#if TMA_RESIDUAL
    /*
    Two-pass TMA residual epilogue: loads residual via TMA into SMEM (coalesced)
    instead of scattered per-lane __ldg. Each pass processes 128 cols using
    output regions 0-1 and residual regions 2-3. Phase toggles twice per call
    (2 passes) so mbarrier phase is consistent across calls.
    */
    if constexpr (Op == EpilogueOp::BIAS_RESIDUAL && (NC_END - NC_START) >= 256) {
#if EPI_PIPELINE
        /* Pipelined epilogue: 4 sub-iterations of 64 cols, one per pipeline stage */
        constexpr int LOCAL_PASSES = EPI_PIPELINE_STAGES;
        constexpr int PASS_COLS = 64;
#elif NUM_PASSES_PARAM == 0
        constexpr int PASS_COLS = 128;
        constexpr int LOCAL_PASSES = (NC_END - NC_START) / PASS_COLS;
#else
        constexpr int LOCAL_PASSES = NUM_PASSES_PARAM;
        constexpr int PASS_COLS = (NC_END - NC_START) / LOCAL_PASSES;
#endif
        constexpr int PASS_REGIONS = PASS_COLS / 64;
        (void)PASS_REGIONS;  /* suppress unused warning for EPI_PIPELINE (uses pipe_stg directly) */
        const int taddr_base = tmem_addr + ((cta_rank * 128 + row_group * 32) << 16);
#if !EPI_PIPELINE
        int res_phase = 0;
#endif

#if BIAS_SMEM
#if BIAS_BF16
        /* Load tile's 256 bias bf16 into SMEM once. 1 uint4 LDG per lane = 32 LDG. */
        {
            reinterpret_cast<uint4*>(smem + OFF_BIAS_SMEM)[lane] =
                __ldg(reinterpret_cast<const uint4*>(side_data + n_start) + lane);
        }
#else
        /* Load tile's 256 bias floats into SMEM once. All warps write same values
           (idempotent — no cross-warp sync needed). 2 float4 LDG per lane = 64 LDG. */
        {
            float* bias_smem = reinterpret_cast<float*>(smem + OFF_BIAS_SMEM);
            const float* bias_src = side_data + n_start;
            reinterpret_cast<float4*>(bias_smem)[lane * 2]     = __ldg(reinterpret_cast<const float4*>(bias_src) + lane * 2);
            reinterpret_cast<float4*>(bias_smem)[lane * 2 + 1] = __ldg(reinterpret_cast<const float4*>(bias_src) + lane * 2 + 1);
        }
#endif
        __syncwarp();
        const uint32_t bias_smem_base = smem_to_uint(smem + OFF_BIAS_SMEM);
#endif

        /* Precomputed swizzle offsets — 8 groups of 16 bytes within a 128-byte region row */
        const uint32_t sw0 = 0 ^ xor_val, sw1 = 16 ^ xor_val, sw2 = 32 ^ xor_val, sw3 = 48 ^ xor_val;
        const uint32_t sw4 = 64 ^ xor_val, sw5 = 80 ^ xor_val, sw6 = 96 ^ xor_val, sw7 = 112 ^ xor_val;

#if TMEM_LOAD_WIDTH == 64
        float a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15;
        float a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31;
        float a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44,a45,a46,a47;
        float a48,a49,a50,a51,a52,a53,a54,a55,a56,a57,a58,a59,a60,a61,a62,a63;

        for (int pass = 0; pass < LOCAL_PASSES; pass++) {
            const int pnc_s = NC_START + pass * PASS_COLS;
            const int pnc_e = pnc_s + PASS_COLS;

#if !W0_RES_FULL && !EPI_LOAD_WARP
            if (!(FIRST_PASS_PRELOADED && pass == 0)) {
                if (lane == 0) {
                    mbar_arrive_expect_tx(res_mbar_addr, PASS_REGIONS * STAGING_REGION_BYTES);
                    tma_load_2d_cta(res_staging_saddr, tma_res_desc,
                                    n_start + pnc_s, gm_base, res_mbar_addr);
                    if constexpr (PASS_REGIONS >= 2) {
                        tma_load_2d_cta(res_staging_saddr + STAGING_REGION_BYTES, tma_res_desc,
                                        n_start + pnc_s + 64, gm_base, res_mbar_addr);
                    }
                }
            }
#endif

#if !DEFERRED_WAIT
            if (pass > 0) {
                if (lane == 0) {
                    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                }
                __syncwarp();
            }
#endif

            /* x64: load 64 cols at a time, 1 region per iteration */
            TMEM_LOAD_X64(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                          a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                          a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44,a45,a46,a47,
                          a48,a49,a50,a51,a52,a53,a54,a55,a56,a57,a58,a59,a60,a61,a62,a63,
                          taddr_base + pnc_s);

            mbar_wait(res_mbar_addr, res_phase);
            res_phase ^= 1;

#if DEFERRED_WAIT
            if (pass > 0) {
                if (lane == 0) {
                    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                }
                __syncwarp();
            }
#endif

#if EPILOGUE_LOOP
#pragma unroll 1
#else
            PRAGMA_UNROLL(PHASE1_UNROLL)
#endif
            for (int nc = pnc_s; nc < pnc_e; nc += 64) {
                const int ri = (nc - pnc_s) >> 6;
                const uint32_t srow = srow_base + ri * STAGING_REGION_BYTES;
                const uint32_t rs = res_staging_saddr + ri * STAGING_REGION_BYTES
                    + lane * STAGING_REGION_ROW_BYTES;

#if BIAS_BF16
#if BIAS_SMEM
                const uint32_t bs = bias_smem_base + nc * 2;
                uint4 bv0, bv1, bv2, bv3, bv4, bv5, bv6, bv7;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(bv0.x),"=r"(bv0.y),"=r"(bv0.z),"=r"(bv0.w) : "r"(bs));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(bv1.x),"=r"(bv1.y),"=r"(bv1.z),"=r"(bv1.w) : "r"(bs + 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(bv2.x),"=r"(bv2.y),"=r"(bv2.z),"=r"(bv2.w) : "r"(bs + 32));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(bv3.x),"=r"(bv3.y),"=r"(bv3.z),"=r"(bv3.w) : "r"(bs + 48));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(bv4.x),"=r"(bv4.y),"=r"(bv4.z),"=r"(bv4.w) : "r"(bs + 64));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(bv5.x),"=r"(bv5.y),"=r"(bv5.z),"=r"(bv5.w) : "r"(bs + 80));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(bv6.x),"=r"(bv6.y),"=r"(bv6.z),"=r"(bv6.w) : "r"(bs + 96));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(bv7.x),"=r"(bv7.y),"=r"(bv7.z),"=r"(bv7.w) : "r"(bs + 112));
#else
                const uint4* bp = reinterpret_cast<const uint4*>(side_data + n_start + nc);
                uint4 bv0 = __ldg(bp);     uint4 bv1 = __ldg(bp + 1);
                uint4 bv2 = __ldg(bp + 2); uint4 bv3 = __ldg(bp + 3);
                uint4 bv4 = __ldg(bp + 4); uint4 bv5 = __ldg(bp + 5);
                uint4 bv6 = __ldg(bp + 6); uint4 bv7 = __ldg(bp + 7);
#endif
#else /* !BIAS_BF16 */
#if BIAS_SMEM
                /* Load 64 bias floats from SMEM (linear, no swizzle) */
                const uint32_t bs = bias_smem_base + nc * 4;
                float4 bv0, bv1, bv2, bv3, bv4, bv5, bv6, bv7;
                float4 bv8, bv9, bv10, bv11, bv12, bv13, bv14, bv15;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv0.x),"=f"(bv0.y),"=f"(bv0.z),"=f"(bv0.w) : "r"(bs));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv1.x),"=f"(bv1.y),"=f"(bv1.z),"=f"(bv1.w) : "r"(bs + 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv2.x),"=f"(bv2.y),"=f"(bv2.z),"=f"(bv2.w) : "r"(bs + 32));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv3.x),"=f"(bv3.y),"=f"(bv3.z),"=f"(bv3.w) : "r"(bs + 48));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv4.x),"=f"(bv4.y),"=f"(bv4.z),"=f"(bv4.w) : "r"(bs + 64));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv5.x),"=f"(bv5.y),"=f"(bv5.z),"=f"(bv5.w) : "r"(bs + 80));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv6.x),"=f"(bv6.y),"=f"(bv6.z),"=f"(bv6.w) : "r"(bs + 96));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv7.x),"=f"(bv7.y),"=f"(bv7.z),"=f"(bv7.w) : "r"(bs + 112));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv8.x),"=f"(bv8.y),"=f"(bv8.z),"=f"(bv8.w) : "r"(bs + 128));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv9.x),"=f"(bv9.y),"=f"(bv9.z),"=f"(bv9.w) : "r"(bs + 144));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv10.x),"=f"(bv10.y),"=f"(bv10.z),"=f"(bv10.w) : "r"(bs + 160));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv11.x),"=f"(bv11.y),"=f"(bv11.z),"=f"(bv11.w) : "r"(bs + 176));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv12.x),"=f"(bv12.y),"=f"(bv12.z),"=f"(bv12.w) : "r"(bs + 192));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv13.x),"=f"(bv13.y),"=f"(bv13.z),"=f"(bv13.w) : "r"(bs + 208));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv14.x),"=f"(bv14.y),"=f"(bv14.z),"=f"(bv14.w) : "r"(bs + 224));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv15.x),"=f"(bv15.y),"=f"(bv15.z),"=f"(bv15.w) : "r"(bs + 240));
#else
                const float* bp = side_data + n_start + nc;
                float4 bv0 = __ldg(reinterpret_cast<const float4*>(bp));
                float4 bv1 = __ldg(reinterpret_cast<const float4*>(bp + 4));
                float4 bv2 = __ldg(reinterpret_cast<const float4*>(bp + 8));
                float4 bv3 = __ldg(reinterpret_cast<const float4*>(bp + 12));
                float4 bv4 = __ldg(reinterpret_cast<const float4*>(bp + 16));
                float4 bv5 = __ldg(reinterpret_cast<const float4*>(bp + 20));
                float4 bv6 = __ldg(reinterpret_cast<const float4*>(bp + 24));
                float4 bv7 = __ldg(reinterpret_cast<const float4*>(bp + 28));
                float4 bv8 = __ldg(reinterpret_cast<const float4*>(bp + 32));
                float4 bv9 = __ldg(reinterpret_cast<const float4*>(bp + 36));
                float4 bv10 = __ldg(reinterpret_cast<const float4*>(bp + 40));
                float4 bv11 = __ldg(reinterpret_cast<const float4*>(bp + 44));
                float4 bv12 = __ldg(reinterpret_cast<const float4*>(bp + 48));
                float4 bv13 = __ldg(reinterpret_cast<const float4*>(bp + 52));
                float4 bv14 = __ldg(reinterpret_cast<const float4*>(bp + 56));
                float4 bv15 = __ldg(reinterpret_cast<const float4*>(bp + 60));
#endif
#endif /* BIAS_BF16 */

                /* Load 64 cols of residual from SMEM (swizzled) */
                uint4 rv0, rv1, rv2, rv3, rv4, rv5, rv6, rv7;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(rv0.x),"=r"(rv0.y),"=r"(rv0.z),"=r"(rv0.w) : "r"(rs + sw0));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(rv1.x),"=r"(rv1.y),"=r"(rv1.z),"=r"(rv1.w) : "r"(rs + sw1));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(rv2.x),"=r"(rv2.y),"=r"(rv2.z),"=r"(rv2.w) : "r"(rs + sw2));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(rv3.x),"=r"(rv3.y),"=r"(rv3.z),"=r"(rv3.w) : "r"(rs + sw3));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(rv4.x),"=r"(rv4.y),"=r"(rv4.z),"=r"(rv4.w) : "r"(rs + sw4));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(rv5.x),"=r"(rv5.y),"=r"(rv5.z),"=r"(rv5.w) : "r"(rs + sw5));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(rv6.x),"=r"(rv6.y),"=r"(rv6.z),"=r"(rv6.w) : "r"(rs + sw6));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(rv7.x),"=r"(rv7.y),"=r"(rv7.z),"=r"(rv7.w) : "r"(rs + sw7));

#if PRE_COMBINE && BIAS_BF16
                /*
                Pre-add bias+residual before TMEM_WAIT: 32 HADD2 ops (all independent,
                ~16 cyc throughput at full ILP). Executes during TMEM stall — essentially free.
                Reduces BF16/STS from 12→8, moving closer to the 4-BF16 free zone (section P).
                */
                uint32_t cv0x, cv0y, cv0z, cv0w, cv1x, cv1y, cv1z, cv1w;
                uint32_t cv2x, cv2y, cv2z, cv2w, cv3x, cv3y, cv3z, cv3w;
                uint32_t cv4x, cv4y, cv4z, cv4w, cv5x, cv5y, cv5z, cv5w;
                uint32_t cv6x, cv6y, cv6z, cv6w, cv7x, cv7y, cv7z, cv7w;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv0x) : "r"(bv0.x), "r"(rv0.x));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv0y) : "r"(bv0.y), "r"(rv0.y));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv0z) : "r"(bv0.z), "r"(rv0.z));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv0w) : "r"(bv0.w), "r"(rv0.w));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv1x) : "r"(bv1.x), "r"(rv1.x));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv1y) : "r"(bv1.y), "r"(rv1.y));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv1z) : "r"(bv1.z), "r"(rv1.z));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv1w) : "r"(bv1.w), "r"(rv1.w));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv2x) : "r"(bv2.x), "r"(rv2.x));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv2y) : "r"(bv2.y), "r"(rv2.y));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv2z) : "r"(bv2.z), "r"(rv2.z));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv2w) : "r"(bv2.w), "r"(rv2.w));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv3x) : "r"(bv3.x), "r"(rv3.x));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv3y) : "r"(bv3.y), "r"(rv3.y));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv3z) : "r"(bv3.z), "r"(rv3.z));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv3w) : "r"(bv3.w), "r"(rv3.w));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv4x) : "r"(bv4.x), "r"(rv4.x));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv4y) : "r"(bv4.y), "r"(rv4.y));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv4z) : "r"(bv4.z), "r"(rv4.z));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv4w) : "r"(bv4.w), "r"(rv4.w));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv5x) : "r"(bv5.x), "r"(rv5.x));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv5y) : "r"(bv5.y), "r"(rv5.y));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv5z) : "r"(bv5.z), "r"(rv5.z));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv5w) : "r"(bv5.w), "r"(rv5.w));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv6x) : "r"(bv6.x), "r"(rv6.x));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv6y) : "r"(bv6.y), "r"(rv6.y));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv6z) : "r"(bv6.z), "r"(rv6.z));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv6w) : "r"(bv6.w), "r"(rv6.w));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv7x) : "r"(bv7.x), "r"(rv7.x));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv7y) : "r"(bv7.y), "r"(rv7.y));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv7z) : "r"(bv7.z), "r"(rv7.z));
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(cv7w) : "r"(bv7.w), "r"(rv7.w));
#endif

                TMEM_WAIT();

                if (MBAR_EARLY && pass == LOCAL_PASSES - 1 && nc + 64 >= pnc_e) {
                    if (epi_mbar_addr) mbar_arrive(epi_mbar_addr);
                }

#if BIAS_BF16
#if PRE_COMBINE
                COMBINED_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7,
                    cv0x,cv0y,cv0z,cv0w, srow + sw0);
                COMBINED_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15,
                    cv1x,cv1y,cv1z,cv1w, srow + sw1);
                COMBINED_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23,
                    cv2x,cv2y,cv2z,cv2w, srow + sw2);
                COMBINED_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31,
                    cv3x,cv3y,cv3z,cv3w, srow + sw3);
                COMBINED_CVT_STS_V4(a32,a33,a34,a35,a36,a37,a38,a39,
                    cv4x,cv4y,cv4z,cv4w, srow + sw4);
                COMBINED_CVT_STS_V4(a40,a41,a42,a43,a44,a45,a46,a47,
                    cv5x,cv5y,cv5z,cv5w, srow + sw5);
                COMBINED_CVT_STS_V4(a48,a49,a50,a51,a52,a53,a54,a55,
                    cv6x,cv6y,cv6z,cv6w, srow + sw6);
                COMBINED_CVT_STS_V4(a56,a57,a58,a59,a60,a61,a62,a63,
                    cv7x,cv7y,cv7z,cv7w, srow + sw7);
#else
                BIAS_RES_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7,
                    bv0.x,bv0.y,bv0.z,bv0.w,
                    rv0.x,rv0.y,rv0.z,rv0.w, srow + sw0);
                BIAS_RES_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15,
                    bv1.x,bv1.y,bv1.z,bv1.w,
                    rv1.x,rv1.y,rv1.z,rv1.w, srow + sw1);
                BIAS_RES_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23,
                    bv2.x,bv2.y,bv2.z,bv2.w,
                    rv2.x,rv2.y,rv2.z,rv2.w, srow + sw2);
                BIAS_RES_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31,
                    bv3.x,bv3.y,bv3.z,bv3.w,
                    rv3.x,rv3.y,rv3.z,rv3.w, srow + sw3);
                BIAS_RES_CVT_STS_V4(a32,a33,a34,a35,a36,a37,a38,a39,
                    bv4.x,bv4.y,bv4.z,bv4.w,
                    rv4.x,rv4.y,rv4.z,rv4.w, srow + sw4);
                BIAS_RES_CVT_STS_V4(a40,a41,a42,a43,a44,a45,a46,a47,
                    bv5.x,bv5.y,bv5.z,bv5.w,
                    rv5.x,rv5.y,rv5.z,rv5.w, srow + sw5);
                BIAS_RES_CVT_STS_V4(a48,a49,a50,a51,a52,a53,a54,a55,
                    bv6.x,bv6.y,bv6.z,bv6.w,
                    rv6.x,rv6.y,rv6.z,rv6.w, srow + sw6);
                BIAS_RES_CVT_STS_V4(a56,a57,a58,a59,a60,a61,a62,a63,
                    bv7.x,bv7.y,bv7.z,bv7.w,
                    rv7.x,rv7.y,rv7.z,rv7.w, srow + sw7);
#endif
#else
                /* 8× BIAS_RES_CVT_STS_V4 — one full 64-col region */
                BIAS_RES_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7,
                    bv0.x,bv0.y,bv0.z,bv0.w,bv1.x,bv1.y,bv1.z,bv1.w,
                    rv0.x,rv0.y,rv0.z,rv0.w, srow + sw0);
                BIAS_RES_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15,
                    bv2.x,bv2.y,bv2.z,bv2.w,bv3.x,bv3.y,bv3.z,bv3.w,
                    rv1.x,rv1.y,rv1.z,rv1.w, srow + sw1);
                BIAS_RES_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23,
                    bv4.x,bv4.y,bv4.z,bv4.w,bv5.x,bv5.y,bv5.z,bv5.w,
                    rv2.x,rv2.y,rv2.z,rv2.w, srow + sw2);
                BIAS_RES_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31,
                    bv6.x,bv6.y,bv6.z,bv6.w,bv7.x,bv7.y,bv7.z,bv7.w,
                    rv3.x,rv3.y,rv3.z,rv3.w, srow + sw3);
                BIAS_RES_CVT_STS_V4(a32,a33,a34,a35,a36,a37,a38,a39,
                    bv8.x,bv8.y,bv8.z,bv8.w,bv9.x,bv9.y,bv9.z,bv9.w,
                    rv4.x,rv4.y,rv4.z,rv4.w, srow + sw4);
                BIAS_RES_CVT_STS_V4(a40,a41,a42,a43,a44,a45,a46,a47,
                    bv10.x,bv10.y,bv10.z,bv10.w,bv11.x,bv11.y,bv11.z,bv11.w,
                    rv5.x,rv5.y,rv5.z,rv5.w, srow + sw5);
                BIAS_RES_CVT_STS_V4(a48,a49,a50,a51,a52,a53,a54,a55,
                    bv12.x,bv12.y,bv12.z,bv12.w,bv13.x,bv13.y,bv13.z,bv13.w,
                    rv6.x,rv6.y,rv6.z,rv6.w, srow + sw6);
                BIAS_RES_CVT_STS_V4(a56,a57,a58,a59,a60,a61,a62,a63,
                    bv14.x,bv14.y,bv14.z,bv14.w,bv15.x,bv15.y,bv15.z,bv15.w,
                    rv7.x,rv7.y,rv7.z,rv7.w, srow + sw7);
#endif

#if PREFETCH_BEFORE_STORE
                if (nc + 64 < pnc_e) {
                    TMEM_LOAD_X64(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                                  a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                                  a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44,a45,a46,a47,
                                  a48,a49,a50,a51,a52,a53,a54,a55,a56,a57,a58,a59,a60,a61,a62,a63,
                                  taddr_base + nc + 64);
                }
#endif

                /* Interleaved TMA stores — x64 completes one region per iteration */
#if !STORE_TIMING
                if (INTERLEAVE_STRATEGY == 1) {
                    __syncwarp();
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (lane == 0) {
                        uint32_t src = staging_saddr + ri * STAGING_REGION_BYTES;
                        asm volatile(
                            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                            :: "l"(tma_c_desc), "r"(n_start + nc),
                               "r"(gm_base), "r"(src) : "memory");
                    }
                } else if (INTERLEAVE_STRATEGY >= 2 && (ri & 1) == 1) {
                    __syncwarp();
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (lane == 0) {
                        uint32_t src0 = staging_saddr + (ri - 1) * STAGING_REGION_BYTES;
                        uint32_t src1 = staging_saddr + ri * STAGING_REGION_BYTES;
                        asm volatile(
                            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                            :: "l"(tma_c_desc), "r"(n_start + nc - 64),
                               "r"(gm_base), "r"(src0) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                            :: "l"(tma_c_desc), "r"(n_start + nc),
                               "r"(gm_base), "r"(src1) : "memory");
                    }
                }
#endif /* !STORE_TIMING */

#if !PREFETCH_BEFORE_STORE
                if (nc + 64 < pnc_e) {
                    TMEM_LOAD_X64(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                                  a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                                  a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44,a45,a46,a47,
                                  a48,a49,a50,a51,a52,a53,a54,a55,a56,a57,a58,a59,a60,a61,a62,a63,
                                  taddr_base + nc + 64);
                }
#endif
            }

            /* Commit this pass's TMA output stores */
            if (STORE_TIMING || INTERLEAVE_STRATEGY == 0
                || (INTERLEAVE_STRATEGY >= 2 && PASS_REGIONS < 2)) {
                __syncwarp();
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (lane == 0) {
                    for (int r = 0; r < PASS_REGIONS; r++) {
                        uint32_t src = staging_saddr + r * STAGING_REGION_BYTES;
                        asm volatile(
                            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                            :: "l"(tma_c_desc), "r"(n_start + pnc_s + r * 64),
                               "r"(gm_base), "r"(src) : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                }
            } else {
                if (lane == 0) {
                    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                }
            }
#if W0_RES_FULL || EPI_LOAD_WARP
            /* Signal pass consumed so loader warp can load next pass */
            if (pass < LOCAL_PASSES - 1) {
                if (res_pass_mbar_addr && lane == 0) mbar_arrive(res_pass_mbar_addr);
            }
#endif
        }
#else  /* TMEM_LOAD_WIDTH != 64 — x32 path */
        float a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15;
        float a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31;

        for (int pass = 0; pass < LOCAL_PASSES; pass++) {
            const int pnc_s = NC_START + pass * PASS_COLS;
            const int pnc_e = pnc_s + PASS_COLS;

#if EPI_PIPELINE
            /* Per-stage addressing: stage buffer = pipeline base + stage * STAGE_C_BYTES + warp * region.
               ReuseSmemC: residual LDS and output STS use the same region (sequential). */
            const uint32_t pipe_stg = staging_saddr + pass * STAGE_C_BYTES + row_group * STAGING_REGION_BYTES;
            const uint32_t pass_res_staging = pipe_stg;
            const uint32_t srow_base = pipe_stg + lane * STAGING_REGION_ROW_BYTES;
#elif STAGES_C >= 2
            /* W0 already loaded residual into stage[pass] — just compute staging addr */
            const uint32_t pass_res_staging = res_staging_saddr + pass * 2 * STAGING_REGION_BYTES;
#else
            const uint32_t pass_res_staging = res_staging_saddr;
#if !W0_RES_FULL && !EPI_LOAD_WARP
            if (!(FIRST_PASS_PRELOADED && pass == 0)) {
                if (lane == 0) {
#if STRIP_EPI_TMA_LOAD
                    /* Arrive with 0 tx bytes — mbar_wait passes immediately, no TMA traffic */
                    mbar_arrive_expect_tx(res_mbar_addr, 0);
#else
                    mbar_arrive_expect_tx(res_mbar_addr, PASS_REGIONS * STAGING_REGION_BYTES);
                    tma_load_2d_cta(res_staging_saddr, tma_res_desc,
                                    n_start + pnc_s, gm_base, res_mbar_addr);
                    if constexpr (PASS_REGIONS >= 2) {
                        tma_load_2d_cta(res_staging_saddr + STAGING_REGION_BYTES, tma_res_desc,
                                        n_start + pnc_s + 64, gm_base, res_mbar_addr);
                    }
#endif
                }
            }
#endif
#endif

#if EPI_PIPELINE
            /* No inter-pass wait_group — each sub-iter has its own stage buffer */
#elif !DEFERRED_WAIT
            if (pass > 0) {
                if (lane == 0) {
                    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                }
                __syncwarp();
            }
#endif

#if !STRIP_EPI_COMPUTE
            LOAD_32_COLS(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                         a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                         taddr_base + pnc_s);
#endif

#if EPI_PIPELINE
            /* Wait for W2's TMA loads for this pipeline stage.
               Phase alternates per tile — all 4 stages use the same phase. */
            mbar_wait(res_mbar_addr + pass * 8, pipe_load_phase);
#elif STAGES_C >= 2
            /* Wait for W0's TMA load to complete — per-(warp,stage) mbar.
               Per-pass mbars all share the same phase (W0 arrives once per tile per mbar),
               so use sc_phase (tracked per-tile in persistent_gemm), not res_phase. */
            mbar_wait(res_mbar_addr + pass * 8, sc_phase);
#else
            mbar_wait(res_mbar_addr, res_phase);
            res_phase ^= 1;
#endif

#if !EPI_PIPELINE && DEFERRED_WAIT
            if (pass > 0) {
                if (lane == 0) {
                    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                }
                __syncwarp();
            }
#endif

#if BRANCHLESS_EPI && STAGES_C >= 2 && BIAS_BF16
            /*
            Branchless epilogue: predicated TMA + deferred stores + full unroll.
            All TMA/mbar ops use predicated execution (no divergent branches,
            no BSSY/BSYNC, minimal R2UR). Inner loop fully unrolled (no back-edge BRA).
            TMA stores deferred to after all compute (no interleaved IS1 branches).
            Only remaining branches: TMEM_WAIT (unavoidable, one per 32-col chunk).
            */
            const uint32_t is_lane0 = (lane == 0) ? 1u : 0u;

#pragma unroll
            for (int nc = pnc_s; nc < pnc_e; nc += 32) {
                const int chunk_in_pass = nc - pnc_s;
                const int res_ri = chunk_in_pass >> 6;
                const int half = (chunk_in_pass >> 5) & 1;

                const uint32_t bs = bias_smem_base + nc * 2;
                uint4 bv0, bv1, bv2, bv3;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(bv0.x),"=r"(bv0.y),"=r"(bv0.z),"=r"(bv0.w) : "r"(bs));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(bv1.x),"=r"(bv1.y),"=r"(bv1.z),"=r"(bv1.w) : "r"(bs + 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(bv2.x),"=r"(bv2.y),"=r"(bv2.z),"=r"(bv2.w) : "r"(bs + 32));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(bv3.x),"=r"(bv3.y),"=r"(bv3.z),"=r"(bv3.w) : "r"(bs + 48));

                const uint32_t rs = pass_res_staging
                    + res_ri * STAGING_REGION_BYTES + lane * STAGING_REGION_ROW_BYTES;
                const uint32_t rsw0 = half ? sw4 : sw0;
                const uint32_t rsw1 = half ? sw5 : sw1;
                const uint32_t rsw2 = half ? sw6 : sw2;
                const uint32_t rsw3 = half ? sw7 : sw3;
                uint4 rv0, rv1, rv2, rv3;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(rv0.x),"=r"(rv0.y),"=r"(rv0.z),"=r"(rv0.w) : "r"(rs + rsw0));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(rv1.x),"=r"(rv1.y),"=r"(rv1.z),"=r"(rv1.w) : "r"(rs + rsw1));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(rv2.x),"=r"(rv2.y),"=r"(rv2.z),"=r"(rv2.w) : "r"(rs + rsw2));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(rv3.x),"=r"(rv3.y),"=r"(rv3.z),"=r"(rv3.w) : "r"(rs + rsw3));

                TMEM_WAIT();

                const uint32_t srow = srow_base + res_ri * STAGING_REGION_BYTES;

                BIAS_RES_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7,
                    bv0.x,bv0.y,bv0.z,bv0.w,
                    rv0.x,rv0.y,rv0.z,rv0.w, srow + rsw0);
                BIAS_RES_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15,
                    bv1.x,bv1.y,bv1.z,bv1.w,
                    rv1.x,rv1.y,rv1.z,rv1.w, srow + rsw1);
                BIAS_RES_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23,
                    bv2.x,bv2.y,bv2.z,bv2.w,
                    rv2.x,rv2.y,rv2.z,rv2.w, srow + rsw2);
                BIAS_RES_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31,
                    bv3.x,bv3.y,bv3.z,bv3.w,
                    rv3.x,rv3.y,rv3.z,rv3.w, srow + rsw3);

                if (nc + 32 < pnc_e) {
                    LOAD_32_COLS(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                                 a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                                 taddr_base + nc + 32);
                }
            }

            /* Deferred TMA stores — all regions at once, predicated (no branches) */
            __syncwarp();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            for (int r = 0; r < PASS_REGIONS; r++) {
                uint32_t src = staging_saddr + r * STAGING_REGION_BYTES;
                pred_tma_store_2d(tma_c_desc, n_start + pnc_s + r * 64,
                                  gm_base, src, is_lane0);
            }
            pred_commit_group(is_lane0);

            /* Signal W0 that this stage's buffer is free — predicated */
            pred_mbar_arrive(sc_release_mbar_base + pass * 8, is_lane0);
#else /* original branched path */
#if STRIP_EPI_COMPUTE
            /* Skip all compute — just issue TMA stores with stale staging data.
               Isolates TMA load+store bandwidth contention on K-loop. */
            {
#if !STRIP_EPI_TMA_STORE
                __syncwarp();
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (lane == 0) {
                    for (int r = 0; r < PASS_REGIONS; r++) {
                        uint32_t src = staging_saddr + r * STAGING_REGION_BYTES;
                        asm volatile(
                            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                            :: "l"(tma_c_desc), "r"(n_start + pnc_s + r * 64),
                               "r"(gm_base), "r"(src) : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                }
#endif /* !STRIP_EPI_TMA_STORE */
            }
#else /* !STRIP_EPI_COMPUTE */
#if EPILOGUE_LOOP
#pragma unroll 1
#else
            PRAGMA_UNROLL(PHASE1_UNROLL)
#endif
            for (int nc = pnc_s; nc < pnc_e; nc += 32) {
                const int chunk_in_pass = nc - pnc_s;
                const int res_ri = chunk_in_pass >> 6;
                const int half = (chunk_in_pass >> 5) & 1;  /* 0=first 32 cols, 1=second 32 cols */

#if BIAS_BF16
#if BIAS_SMEM
                const uint32_t bs = bias_smem_base + nc * 2;
                uint4 bv0, bv1, bv2, bv3;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(bv0.x),"=r"(bv0.y),"=r"(bv0.z),"=r"(bv0.w) : "r"(bs));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(bv1.x),"=r"(bv1.y),"=r"(bv1.z),"=r"(bv1.w) : "r"(bs + 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(bv2.x),"=r"(bv2.y),"=r"(bv2.z),"=r"(bv2.w) : "r"(bs + 32));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(bv3.x),"=r"(bv3.y),"=r"(bv3.z),"=r"(bv3.w) : "r"(bs + 48));
#else
                const uint4* bp = reinterpret_cast<const uint4*>(side_data + n_start + nc);
                uint4 bv0 = __ldg(bp);     uint4 bv1 = __ldg(bp + 1);
                uint4 bv2 = __ldg(bp + 2); uint4 bv3 = __ldg(bp + 3);
#endif
#else /* !BIAS_BF16 */
#if BIAS_SMEM
                const uint32_t bs = bias_smem_base + nc * 4;
                float4 bv0, bv1, bv2, bv3, bv4, bv5, bv6, bv7;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv0.x),"=f"(bv0.y),"=f"(bv0.z),"=f"(bv0.w) : "r"(bs));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv1.x),"=f"(bv1.y),"=f"(bv1.z),"=f"(bv1.w) : "r"(bs + 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv2.x),"=f"(bv2.y),"=f"(bv2.z),"=f"(bv2.w) : "r"(bs + 32));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv3.x),"=f"(bv3.y),"=f"(bv3.z),"=f"(bv3.w) : "r"(bs + 48));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv4.x),"=f"(bv4.y),"=f"(bv4.z),"=f"(bv4.w) : "r"(bs + 64));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv5.x),"=f"(bv5.y),"=f"(bv5.z),"=f"(bv5.w) : "r"(bs + 80));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv6.x),"=f"(bv6.y),"=f"(bv6.z),"=f"(bv6.w) : "r"(bs + 96));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(bv7.x),"=f"(bv7.y),"=f"(bv7.z),"=f"(bv7.w) : "r"(bs + 112));
#else
                const float* bp = side_data + n_start + nc;
                float4 bv0 = __ldg(reinterpret_cast<const float4*>(bp));
                float4 bv1 = __ldg(reinterpret_cast<const float4*>(bp + 4));
                float4 bv2 = __ldg(reinterpret_cast<const float4*>(bp + 8));
                float4 bv3 = __ldg(reinterpret_cast<const float4*>(bp + 12));
                float4 bv4 = __ldg(reinterpret_cast<const float4*>(bp + 16));
                float4 bv5 = __ldg(reinterpret_cast<const float4*>(bp + 20));
                float4 bv6 = __ldg(reinterpret_cast<const float4*>(bp + 24));
                float4 bv7 = __ldg(reinterpret_cast<const float4*>(bp + 28));
#endif
#endif /* BIAS_BF16 */

                /* Residual from SMEM — use precomputed swizzle offsets */
                const uint32_t rs = pass_res_staging
                    + res_ri * STAGING_REGION_BYTES + lane * STAGING_REGION_ROW_BYTES;
                const uint32_t rsw0 = half ? sw4 : sw0;
                const uint32_t rsw1 = half ? sw5 : sw1;
                const uint32_t rsw2 = half ? sw6 : sw2;
                const uint32_t rsw3 = half ? sw7 : sw3;
                uint4 rv0, rv1, rv2, rv3;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(rv0.x),"=r"(rv0.y),"=r"(rv0.z),"=r"(rv0.w) : "r"(rs + rsw0));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(rv1.x),"=r"(rv1.y),"=r"(rv1.z),"=r"(rv1.w) : "r"(rs + rsw1));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(rv2.x),"=r"(rv2.y),"=r"(rv2.z),"=r"(rv2.w) : "r"(rs + rsw2));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(rv3.x),"=r"(rv3.y),"=r"(rv3.z),"=r"(rv3.w) : "r"(rs + rsw3));

                TMEM_WAIT();

                if (MBAR_EARLY && pass == LOCAL_PASSES - 1 && nc + 32 >= pnc_e) {
                    if (epi_mbar_addr) mbar_arrive(epi_mbar_addr);
                }

                /* Output staging — use precomputed swizzle offsets */
                const uint32_t srow = srow_base + res_ri * STAGING_REGION_BYTES;

#if BIAS_BF16
                BIAS_RES_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7,
                    bv0.x,bv0.y,bv0.z,bv0.w,
                    rv0.x,rv0.y,rv0.z,rv0.w, srow + rsw0);
                BIAS_RES_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15,
                    bv1.x,bv1.y,bv1.z,bv1.w,
                    rv1.x,rv1.y,rv1.z,rv1.w, srow + rsw1);
                BIAS_RES_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23,
                    bv2.x,bv2.y,bv2.z,bv2.w,
                    rv2.x,rv2.y,rv2.z,rv2.w, srow + rsw2);
                BIAS_RES_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31,
                    bv3.x,bv3.y,bv3.z,bv3.w,
                    rv3.x,rv3.y,rv3.z,rv3.w, srow + rsw3);
#else
                BIAS_RES_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7,
                    bv0.x,bv0.y,bv0.z,bv0.w,bv1.x,bv1.y,bv1.z,bv1.w,
                    rv0.x,rv0.y,rv0.z,rv0.w, srow + rsw0);
                BIAS_RES_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15,
                    bv2.x,bv2.y,bv2.z,bv2.w,bv3.x,bv3.y,bv3.z,bv3.w,
                    rv1.x,rv1.y,rv1.z,rv1.w, srow + rsw1);
                BIAS_RES_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23,
                    bv4.x,bv4.y,bv4.z,bv4.w,bv5.x,bv5.y,bv5.z,bv5.w,
                    rv2.x,rv2.y,rv2.z,rv2.w, srow + rsw2);
                BIAS_RES_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31,
                    bv6.x,bv6.y,bv6.z,bv6.w,bv7.x,bv7.y,bv7.z,bv7.w,
                    rv3.x,rv3.y,rv3.z,rv3.w, srow + rsw3);
#endif

#if PREFETCH_BEFORE_STORE
                if (nc + 32 < pnc_e) {
                    LOAD_32_COLS(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                                 a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                                 taddr_base + nc + 32);
                }
#endif

                /* Interleaved TMA stores within 128-col pass (2 output regions) */
#if !STORE_TIMING && !EPI_PIPELINE && !STRIP_EPI_TMA_STORE
                if (INTERLEAVE_STRATEGY == 1 && half == 1) {
                    __syncwarp();
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (lane == 0) {
                        uint32_t src = staging_saddr + res_ri * STAGING_REGION_BYTES;
                        asm volatile(
                            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                            :: "l"(tma_c_desc), "r"(n_start + pnc_s + res_ri * 64),
                               "r"(gm_base), "r"(src) : "memory");
                    }
                } else if (INTERLEAVE_STRATEGY >= 2 && half == 1 && (res_ri & 1) == 1) {
                    __syncwarp();
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (lane == 0) {
                        uint32_t src0 = staging_saddr;
                        uint32_t src1 = staging_saddr + STAGING_REGION_BYTES;
                        asm volatile(
                            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                            :: "l"(tma_c_desc), "r"(n_start + pnc_s),
                               "r"(gm_base), "r"(src0) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                            :: "l"(tma_c_desc), "r"(n_start + pnc_s + 64),
                               "r"(gm_base), "r"(src1) : "memory");
                    }
                }
#endif /* !STORE_TIMING && !STRIP_EPI_TMA_STORE */

#if !PREFETCH_BEFORE_STORE
                if (nc + 32 < pnc_e) {
                    LOAD_32_COLS(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                                 a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                                 taddr_base + nc + 32);
                }
#endif
#if EPI_BAR_CHUNK
                /* Sync all epilogue warps after each 32-col chunk — serializes concurrent STS/compute */
                asm volatile("bar.sync 1, %0;" :: "r"(NUM_EPI_WARPS * 32) : "memory");
#endif
            }

#if EPI_PIPELINE
            /* EPI_PIPELINE: one TMA store per sub-iter from the stage buffer */
            __syncwarp();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (lane == 0) {
                asm volatile(
                    "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                    :: "l"(tma_c_desc), "r"(n_start + pnc_s),
                       "r"(gm_base), "r"(pipe_stg) : "memory");
                asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            }
#else
#if !STRIP_EPI_TMA_STORE
            /* Commit this pass's TMA output stores */
            if (STORE_TIMING || INTERLEAVE_STRATEGY == 0
                || (INTERLEAVE_STRATEGY >= 2 && PASS_REGIONS < 2)) {
                __syncwarp();
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (lane == 0) {
                    for (int r = 0; r < PASS_REGIONS; r++) {
                        uint32_t src = staging_saddr + r * STAGING_REGION_BYTES;
                        asm volatile(
                            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                            :: "l"(tma_c_desc), "r"(n_start + pnc_s + r * 64),
                               "r"(gm_base), "r"(src) : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                }
            } else {
                if (lane == 0) {
                    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                }
            }
#endif /* !STRIP_EPI_TMA_STORE */
#if STAGES_C >= 2
            /* Signal W0 that this stage's buffer is free */
            if (lane == 0) mbar_arrive(sc_release_mbar_base + pass * 8);
#elif W0_RES_FULL || EPI_LOAD_WARP
            /* Signal pass consumed so loader warp can load next pass */
            if (pass < LOCAL_PASSES - 1) {
                if (res_pass_mbar_addr && lane == 0) mbar_arrive(res_pass_mbar_addr);
            }
#endif
#endif /* EPI_PIPELINE */
#endif /* !STRIP_EPI_COMPUTE */
#endif /* BRANCHLESS_EPI */
#if EPI_BAR_PASS
            /* Sync all epilogue warps between passes — reduces concurrent activity during K-loop */
            asm volatile("bar.sync 1, %0;" :: "r"(NUM_EPI_WARPS * 32) : "memory");
#endif
        }
#endif  /* TMEM_LOAD_WIDTH == 64 */

#if EPI_PIPELINE
        /* Wait for all TMA stores to complete, then signal stages free for W2 */
        if (lane == 0) {
            asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        }
        __syncwarp();
        if (res_consumed_mbar_addr && lane == 0)
            mbar_arrive(res_consumed_mbar_addr);
#elif W0_RES_PREFETCH || W0_RES_FULL || EPI_LOAD_WARP
        if (res_consumed_mbar_addr && lane == 0)
            mbar_arrive(res_consumed_mbar_addr);
#endif

#ifdef TIMING
        t_phase1_end = clock64();
#endif
        if (!MBAR_EARLY && epi_mbar_addr) mbar_arrive(epi_mbar_addr);
        return;
    }
#endif

#if TMEM_LOAD_WIDTH == 64
    float a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15;
    float a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31;
    float a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44,a45,a46,a47;
    float a48,a49,a50,a51,a52,a53,a54,a55,a56,a57,a58,a59,a60,a61,a62,a63;

    // Phase 1: all cols → swizzle regions, x64 stride
    TMEM_LOAD_X64(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                  a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                  a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44,a45,a46,a47,
                  a48,a49,a50,a51,a52,a53,a54,a55,a56,a57,a58,a59,a60,a61,a62,a63,
                  taddr_base + NC_START);

#if EPILOGUE_LOOP
#pragma unroll 1
#else
    PRAGMA_UNROLL(PHASE1_UNROLL)
#endif
    for (int nc = NC_START; nc < NC_END; nc += 64) {
        const uint32_t srow = srow_base + ((nc - NC_START) >> 6) * STAGING_REGION_BYTES;

        // Side-data variables
        uint4 craw0 = {}, craw1 = {};
        const __nv_bfloat16* comb_ptr = nullptr;
        float4 bv0 = {}, bv1 = {};
        uint4 rv0 = {};
#if PRELOAD_MODE == 2
        uint4 craw2 = {}, craw3 = {}, craw4 = {}, craw5 = {}, craw6 = {}, craw7 = {};
#endif

#if PRELOAD_MODE >= 1
        // Preload side data (fills TMEM latency window)
        if constexpr (Op == EpilogueOp::BIAS_ADD) {
            comb_ptr = comb_base + (long long)((n_start + nc) / COMB_BLOCK_COLS) * COMB_BLOCK_ELEMS;
            craw0 = *reinterpret_cast<const uint4*>(comb_ptr);
            craw1 = *reinterpret_cast<const uint4*>(comb_ptr + 8);
#if PRELOAD_MODE == 2
            craw2 = *reinterpret_cast<const uint4*>(comb_ptr + 16);
            craw3 = *reinterpret_cast<const uint4*>(comb_ptr + 24);
            const __nv_bfloat16* comb_ptr2_pre = comb_base + (long long)((n_start + nc + 32) / COMB_BLOCK_COLS) * COMB_BLOCK_ELEMS;
            craw4 = *reinterpret_cast<const uint4*>(comb_ptr2_pre);
            craw5 = *reinterpret_cast<const uint4*>(comb_ptr2_pre + 8);
            craw6 = *reinterpret_cast<const uint4*>(comb_ptr2_pre + 16);
            craw7 = *reinterpret_cast<const uint4*>(comb_ptr2_pre + 24);
#endif
        } else if constexpr (Op == EpilogueOp::BIAS_GELU) {
            const float* bp = reinterpret_cast<const float*>(side_data) + n_start + nc;
            bv0 = __ldg(reinterpret_cast<const float4*>(bp));
            bv1 = __ldg(reinterpret_cast<const float4*>(bp + 4));
#if !BIAS_BF16
        } else if constexpr (Op == EpilogueOp::BIAS_RESIDUAL) {
            const float* bp = side_data + n_start + nc;
            bv0 = __ldg(reinterpret_cast<const float4*>(bp));
            bv1 = __ldg(reinterpret_cast<const float4*>(bp + 4));
            rv0 = __ldg(reinterpret_cast<const uint4*>(res_row + nc));
#endif
        }
#endif

        TMEM_WAIT();

#if PRELOAD_MODE == 0
        if constexpr (Op == EpilogueOp::BIAS_ADD) {
            comb_ptr = comb_base + (long long)((n_start + nc) / COMB_BLOCK_COLS) * COMB_BLOCK_ELEMS;
            craw0 = *reinterpret_cast<const uint4*>(comb_ptr);
            craw1 = *reinterpret_cast<const uint4*>(comb_ptr + 8);
        } else if constexpr (Op == EpilogueOp::BIAS_GELU) {
            const float* bp = reinterpret_cast<const float*>(side_data) + n_start + nc;
            bv0 = __ldg(reinterpret_cast<const float4*>(bp));
            bv1 = __ldg(reinterpret_cast<const float4*>(bp + 4));
#if !BIAS_BF16
        } else if constexpr (Op == EpilogueOp::BIAS_RESIDUAL) {
            const float* bp = side_data + n_start + nc;
            bv0 = __ldg(reinterpret_cast<const float4*>(bp));
            bv1 = __ldg(reinterpret_cast<const float4*>(bp + 4));
            rv0 = __ldg(reinterpret_cast<const uint4*>(res_row + nc));
#endif
        }
#endif

        if (MBAR_EARLY && nc + 64 >= NC_END) {
            if (epi_mbar_addr) mbar_arrive(epi_mbar_addr);
        }

        // Transform: accumulator → (op-specific) → BF16 → SMEM
        if constexpr (Op == EpilogueOp::BIAS_ADD) {
#if PRELOAD_MODE == 2
            {
                CVT_ADD_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7, craw0.x,craw0.y,craw0.z,craw0.w, srow + (0 ^ xor_val));
                CVT_ADD_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15, craw1.x,craw1.y,craw1.z,craw1.w, srow + (16 ^ xor_val));
                CVT_ADD_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23, craw2.x,craw2.y,craw2.z,craw2.w, srow + (32 ^ xor_val));
                CVT_ADD_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31, craw3.x,craw3.y,craw3.z,craw3.w, srow + (48 ^ xor_val));
            }
            {
                CVT_ADD_STS_V4(a32,a33,a34,a35,a36,a37,a38,a39, craw4.x,craw4.y,craw4.z,craw4.w, srow + (64 ^ xor_val));
                CVT_ADD_STS_V4(a40,a41,a42,a43,a44,a45,a46,a47, craw5.x,craw5.y,craw5.z,craw5.w, srow + (80 ^ xor_val));
                CVT_ADD_STS_V4(a48,a49,a50,a51,a52,a53,a54,a55, craw6.x,craw6.y,craw6.z,craw6.w, srow + (96 ^ xor_val));
                CVT_ADD_STS_V4(a56,a57,a58,a59,a60,a61,a62,a63, craw7.x,craw7.y,craw7.z,craw7.w, srow + (112 ^ xor_val));
            }
#else
            {
                CVT_ADD_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7, craw0.x,craw0.y,craw0.z,craw0.w, srow + (0 ^ xor_val));
                CVT_ADD_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15, craw1.x,craw1.y,craw1.z,craw1.w, srow + (16 ^ xor_val));
                craw0 = *reinterpret_cast<const uint4*>(comb_ptr + 16);
                craw1 = *reinterpret_cast<const uint4*>(comb_ptr + 24);
                CVT_ADD_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23, craw0.x,craw0.y,craw0.z,craw0.w, srow + (32 ^ xor_val));
                CVT_ADD_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31, craw1.x,craw1.y,craw1.z,craw1.w, srow + (48 ^ xor_val));
            }
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
#endif
        } else if constexpr (Op == EpilogueOp::BIAS_GELU) {
            const float* bp = side_data + n_start + nc;
            {
                GELU_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7, bv0.x,bv0.y,bv0.z,bv0.w,bv1.x,bv1.y,bv1.z,bv1.w, srow + (0 ^ xor_val));
                float4 bv2 = __ldg(reinterpret_cast<const float4*>(bp + 8));
                float4 bv3 = __ldg(reinterpret_cast<const float4*>(bp + 12));
                GELU_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15, bv2.x,bv2.y,bv2.z,bv2.w,bv3.x,bv3.y,bv3.z,bv3.w, srow + (16 ^ xor_val));
                float4 bv4 = __ldg(reinterpret_cast<const float4*>(bp + 16));
                float4 bv5 = __ldg(reinterpret_cast<const float4*>(bp + 20));
                GELU_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23, bv4.x,bv4.y,bv4.z,bv4.w,bv5.x,bv5.y,bv5.z,bv5.w, srow + (32 ^ xor_val));
                float4 bv6 = __ldg(reinterpret_cast<const float4*>(bp + 24));
                float4 bv7 = __ldg(reinterpret_cast<const float4*>(bp + 28));
                GELU_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31, bv6.x,bv6.y,bv6.z,bv6.w,bv7.x,bv7.y,bv7.z,bv7.w, srow + (48 ^ xor_val));
            }
            {
                float4 bv8 = __ldg(reinterpret_cast<const float4*>(bp + 32));
                float4 bv9 = __ldg(reinterpret_cast<const float4*>(bp + 36));
                GELU_CVT_STS_V4(a32,a33,a34,a35,a36,a37,a38,a39, bv8.x,bv8.y,bv8.z,bv8.w,bv9.x,bv9.y,bv9.z,bv9.w, srow + (64 ^ xor_val));
                float4 bv10 = __ldg(reinterpret_cast<const float4*>(bp + 40));
                float4 bv11 = __ldg(reinterpret_cast<const float4*>(bp + 44));
                GELU_CVT_STS_V4(a40,a41,a42,a43,a44,a45,a46,a47, bv10.x,bv10.y,bv10.z,bv10.w,bv11.x,bv11.y,bv11.z,bv11.w, srow + (80 ^ xor_val));
                float4 bv12 = __ldg(reinterpret_cast<const float4*>(bp + 48));
                float4 bv13 = __ldg(reinterpret_cast<const float4*>(bp + 52));
                GELU_CVT_STS_V4(a48,a49,a50,a51,a52,a53,a54,a55, bv12.x,bv12.y,bv12.z,bv12.w,bv13.x,bv13.y,bv13.z,bv13.w, srow + (96 ^ xor_val));
                float4 bv14 = __ldg(reinterpret_cast<const float4*>(bp + 56));
                float4 bv15 = __ldg(reinterpret_cast<const float4*>(bp + 60));
                GELU_CVT_STS_V4(a56,a57,a58,a59,a60,a61,a62,a63, bv14.x,bv14.y,bv14.z,bv14.w,bv15.x,bv15.y,bv15.z,bv15.w, srow + (112 ^ xor_val));
            }
        } else if constexpr (Op == EpilogueOp::BIAS_RESIDUAL) {
#if !TMA_RESIDUAL
            /* Non-TMA_RESIDUAL x64 path: load bias+residual via __ldg.
               When TMA_RESIDUAL=1 this entire branch is dead (early return above). */
            static_assert(HAS_BIAS_RES_CVT, "BIAS_RESIDUAL requires BIAS_RES_CVT_STS_V4 macro — define before #include \"kernel_body.cuh\"");
#if BIAS_BF16
            {
                uint4 bv1 = __ldg(reinterpret_cast<const uint4*>(side_data + n_start + nc + 8));
                uint4 rv1 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 8));
                BIAS_RES_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7, bv0.x,bv0.y,bv0.z,bv0.w, rv0.x,rv0.y,rv0.z,rv0.w, srow + (0 ^ xor_val));
                uint4 bv2 = __ldg(reinterpret_cast<const uint4*>(side_data + n_start + nc + 16));
                uint4 rv2 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 16));
                BIAS_RES_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15, bv1.x,bv1.y,bv1.z,bv1.w, rv1.x,rv1.y,rv1.z,rv1.w, srow + (16 ^ xor_val));
                uint4 bv3 = __ldg(reinterpret_cast<const uint4*>(side_data + n_start + nc + 24));
                uint4 rv3 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 24));
                BIAS_RES_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23, bv2.x,bv2.y,bv2.z,bv2.w, rv2.x,rv2.y,rv2.z,rv2.w, srow + (32 ^ xor_val));
                BIAS_RES_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31, bv3.x,bv3.y,bv3.z,bv3.w, rv3.x,rv3.y,rv3.z,rv3.w, srow + (48 ^ xor_val));
            }
            {
                uint4 bv4 = __ldg(reinterpret_cast<const uint4*>(side_data + n_start + nc + 32));
                uint4 rv4 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 32));
                uint4 bv5 = __ldg(reinterpret_cast<const uint4*>(side_data + n_start + nc + 40));
                uint4 rv5 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 40));
                BIAS_RES_CVT_STS_V4(a32,a33,a34,a35,a36,a37,a38,a39, bv4.x,bv4.y,bv4.z,bv4.w, rv4.x,rv4.y,rv4.z,rv4.w, srow + (64 ^ xor_val));
                uint4 bv6 = __ldg(reinterpret_cast<const uint4*>(side_data + n_start + nc + 48));
                uint4 rv6 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 48));
                BIAS_RES_CVT_STS_V4(a40,a41,a42,a43,a44,a45,a46,a47, bv5.x,bv5.y,bv5.z,bv5.w, rv5.x,rv5.y,rv5.z,rv5.w, srow + (80 ^ xor_val));
                uint4 bv7 = __ldg(reinterpret_cast<const uint4*>(side_data + n_start + nc + 56));
                uint4 rv7 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 56));
                BIAS_RES_CVT_STS_V4(a48,a49,a50,a51,a52,a53,a54,a55, bv6.x,bv6.y,bv6.z,bv6.w, rv6.x,rv6.y,rv6.z,rv6.w, srow + (96 ^ xor_val));
                BIAS_RES_CVT_STS_V4(a56,a57,a58,a59,a60,a61,a62,a63, bv7.x,bv7.y,bv7.z,bv7.w, rv7.x,rv7.y,rv7.z,rv7.w, srow + (112 ^ xor_val));
            }
#else
            const float* bp = side_data + n_start + nc;
            {
                BIAS_RES_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7, bv0.x,bv0.y,bv0.z,bv0.w,bv1.x,bv1.y,bv1.z,bv1.w, rv0.x,rv0.y,rv0.z,rv0.w, srow + (0 ^ xor_val));
                float4 bv2 = __ldg(reinterpret_cast<const float4*>(bp + 8));
                float4 bv3 = __ldg(reinterpret_cast<const float4*>(bp + 12));
                uint4 rv1 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 8));
                BIAS_RES_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15, bv2.x,bv2.y,bv2.z,bv2.w,bv3.x,bv3.y,bv3.z,bv3.w, rv1.x,rv1.y,rv1.z,rv1.w, srow + (16 ^ xor_val));
                float4 bv4 = __ldg(reinterpret_cast<const float4*>(bp + 16));
                float4 bv5 = __ldg(reinterpret_cast<const float4*>(bp + 20));
                uint4 rv2 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 16));
                BIAS_RES_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23, bv4.x,bv4.y,bv4.z,bv4.w,bv5.x,bv5.y,bv5.z,bv5.w, rv2.x,rv2.y,rv2.z,rv2.w, srow + (32 ^ xor_val));
                float4 bv6 = __ldg(reinterpret_cast<const float4*>(bp + 24));
                float4 bv7 = __ldg(reinterpret_cast<const float4*>(bp + 28));
                uint4 rv3 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 24));
                BIAS_RES_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31, bv6.x,bv6.y,bv6.z,bv6.w,bv7.x,bv7.y,bv7.z,bv7.w, rv3.x,rv3.y,rv3.z,rv3.w, srow + (48 ^ xor_val));
            }
            {
                float4 bv8 = __ldg(reinterpret_cast<const float4*>(bp + 32));
                float4 bv9 = __ldg(reinterpret_cast<const float4*>(bp + 36));
                uint4 rv4 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 32));
                BIAS_RES_CVT_STS_V4(a32,a33,a34,a35,a36,a37,a38,a39, bv8.x,bv8.y,bv8.z,bv8.w,bv9.x,bv9.y,bv9.z,bv9.w, rv4.x,rv4.y,rv4.z,rv4.w, srow + (64 ^ xor_val));
                float4 bv10 = __ldg(reinterpret_cast<const float4*>(bp + 40));
                float4 bv11 = __ldg(reinterpret_cast<const float4*>(bp + 44));
                uint4 rv5 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 40));
                BIAS_RES_CVT_STS_V4(a40,a41,a42,a43,a44,a45,a46,a47, bv10.x,bv10.y,bv10.z,bv10.w,bv11.x,bv11.y,bv11.z,bv11.w, rv5.x,rv5.y,rv5.z,rv5.w, srow + (80 ^ xor_val));
                float4 bv12 = __ldg(reinterpret_cast<const float4*>(bp + 48));
                float4 bv13 = __ldg(reinterpret_cast<const float4*>(bp + 52));
                uint4 rv6 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 48));
                BIAS_RES_CVT_STS_V4(a48,a49,a50,a51,a52,a53,a54,a55, bv12.x,bv12.y,bv12.z,bv12.w,bv13.x,bv13.y,bv13.z,bv13.w, rv6.x,rv6.y,rv6.z,rv6.w, srow + (96 ^ xor_val));
                float4 bv14 = __ldg(reinterpret_cast<const float4*>(bp + 56));
                float4 bv15 = __ldg(reinterpret_cast<const float4*>(bp + 60));
                uint4 rv7 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 56));
                BIAS_RES_CVT_STS_V4(a56,a57,a58,a59,a60,a61,a62,a63, bv14.x,bv14.y,bv14.z,bv14.w,bv15.x,bv15.y,bv15.z,bv15.w, rv7.x,rv7.y,rv7.z,rv7.w, srow + (112 ^ xor_val));
            }
#endif
#endif /* !TMA_RESIDUAL */
        }

#if PREFETCH_BEFORE_STORE
        if (nc + 64 < NC_END) {
            TMEM_LOAD_X64(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                          a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                          a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44,a45,a46,a47,
                          a48,a49,a50,a51,a52,a53,a54,a55,a56,a57,a58,a59,a60,a61,a62,a63,
                          taddr_base + nc + 64);
        }
#endif

        // Interleaved TMA store(s) — fence.proxy.async bridges sync→async proxy
#if !STORE_TIMING
        if (INTERLEAVE_STRATEGY == 1) {
            __syncwarp();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (lane == 0) {
                int region_idx = (nc - NC_START) >> 6;
                uint32_t src = staging_saddr + region_idx * STAGING_REGION_BYTES;
                asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                    :: "l"(tma_c_desc), "r"(n_start + nc), "r"(gm_base), "r"(src) : "memory");
            }
        } else if (INTERLEAVE_STRATEGY == 2 && (((nc - NC_START) >> 6) & 1) == 1) {
            __syncwarp();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (lane == 0) {
                int region_idx = (nc - NC_START) >> 6;
                uint32_t src0 = staging_saddr + (region_idx - 1) * STAGING_REGION_BYTES;
                uint32_t src1 = staging_saddr + region_idx * STAGING_REGION_BYTES;
                asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                    :: "l"(tma_c_desc), "r"(n_start + nc - 64), "r"(gm_base), "r"(src0) : "memory");
                asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                    :: "l"(tma_c_desc), "r"(n_start + nc), "r"(gm_base), "r"(src1) : "memory");
            }
        } else if (INTERLEAVE_STRATEGY == 3 && ((nc - NC_START) >> 6) == (N_REGIONS < 3 ? N_REGIONS - 1 : 2)) {
            constexpr int INLINE_REGIONS = N_REGIONS < 3 ? N_REGIONS : 3;
            __syncwarp();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (lane == 0) {
                for (int r = 0; r < INLINE_REGIONS; r++) {
                    uint32_t src = staging_saddr + r * STAGING_REGION_BYTES;
                    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                        :: "l"(tma_c_desc), "r"(n_start + NC_START + r * 64), "r"(gm_base), "r"(src) : "memory");
                }
            }
        }
#endif /* !STORE_TIMING */

#if !PREFETCH_BEFORE_STORE
        if (nc + 64 < NC_END) {
            TMEM_LOAD_X64(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                          a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                          a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44,a45,a46,a47,
                          a48,a49,a50,a51,a52,a53,a54,a55,a56,a57,a58,a59,a60,a61,a62,a63,
                          taddr_base + nc + 64);
        }
#endif
    }
#else  // TMEM_LOAD_WIDTH 16 or 32
    float a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15;
    float a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31;

    // Phase 1: all cols → swizzle regions, x32 stride
    LOAD_32_COLS(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                 a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                 taddr_base + NC_START);

#if EPILOGUE_LOOP
#pragma unroll 1
#else
    PRAGMA_UNROLL(PHASE1_UNROLL)
#endif
    for (int nc = NC_START; nc < NC_END; nc += 32) {
        // Side-data variables
        uint4 craw0 = {}, craw1 = {};
        const __nv_bfloat16* comb_ptr = nullptr;
        const float* bp = nullptr;
        float4 bv0 = {}, bv1 = {};
        uint4 rv0 = {};
#if PRELOAD_MODE == 2
        uint4 craw2 = {}, craw3 = {};
        float4 bv2 = {}, bv3 = {}, bv4 = {}, bv5 = {}, bv6 = {}, bv7 = {};
        uint4 rv1 = {}, rv2 = {}, rv3 = {};
#endif

#if PRELOAD_MODE >= 1
        // Preload side data (fills TMEM latency window)
        if constexpr (Op == EpilogueOp::BIAS_ADD) {
            comb_ptr = comb_base + (long long)((n_start + nc) / COMB_BLOCK_COLS) * COMB_BLOCK_ELEMS;
            craw0 = *reinterpret_cast<const uint4*>(comb_ptr);
            craw1 = *reinterpret_cast<const uint4*>(comb_ptr + 8);
#if PRELOAD_MODE == 2
            craw2 = *reinterpret_cast<const uint4*>(comb_ptr + 16);
            craw3 = *reinterpret_cast<const uint4*>(comb_ptr + 24);
#endif
        } else if constexpr (Op == EpilogueOp::BIAS_GELU) {
            bp = reinterpret_cast<const float*>(side_data) + n_start + nc;
            bv0 = __ldg(reinterpret_cast<const float4*>(bp));
            bv1 = __ldg(reinterpret_cast<const float4*>(bp + 4));
#if PRELOAD_MODE == 2
            /* Full preload: all 32 bias values before TMEM_WAIT.
               Fills the TMEM latency window with useful LDG traffic and
               removes interleaved LDG from the GELU critical path. */
            bv2 = __ldg(reinterpret_cast<const float4*>(bp + 8));
            bv3 = __ldg(reinterpret_cast<const float4*>(bp + 12));
            bv4 = __ldg(reinterpret_cast<const float4*>(bp + 16));
            bv5 = __ldg(reinterpret_cast<const float4*>(bp + 20));
            bv6 = __ldg(reinterpret_cast<const float4*>(bp + 24));
            bv7 = __ldg(reinterpret_cast<const float4*>(bp + 28));
#endif
#if !BIAS_BF16
        } else if constexpr (Op == EpilogueOp::BIAS_RESIDUAL) {
            bp = side_data + n_start + nc;
            bv0 = __ldg(reinterpret_cast<const float4*>(bp));
            bv1 = __ldg(reinterpret_cast<const float4*>(bp + 4));
#if PRELOAD_MODE == 2
            bv2 = __ldg(reinterpret_cast<const float4*>(bp + 8));
            bv3 = __ldg(reinterpret_cast<const float4*>(bp + 12));
            bv4 = __ldg(reinterpret_cast<const float4*>(bp + 16));
            bv5 = __ldg(reinterpret_cast<const float4*>(bp + 20));
            bv6 = __ldg(reinterpret_cast<const float4*>(bp + 24));
            bv7 = __ldg(reinterpret_cast<const float4*>(bp + 28));
            rv1 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 8));
            rv2 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 16));
            rv3 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 24));
#endif
            rv0 = __ldg(reinterpret_cast<const uint4*>(res_row + nc));
#endif
        }
#endif

        TMEM_WAIT();

#if PRELOAD_MODE == 0
        if constexpr (Op == EpilogueOp::BIAS_ADD) {
            comb_ptr = comb_base + (long long)((n_start + nc) / COMB_BLOCK_COLS) * COMB_BLOCK_ELEMS;
            craw0 = *reinterpret_cast<const uint4*>(comb_ptr);
            craw1 = *reinterpret_cast<const uint4*>(comb_ptr + 8);
        } else if constexpr (Op == EpilogueOp::BIAS_GELU) {
            bp = reinterpret_cast<const float*>(side_data) + n_start + nc;
            bv0 = __ldg(reinterpret_cast<const float4*>(bp));
            bv1 = __ldg(reinterpret_cast<const float4*>(bp + 4));
#if !BIAS_BF16
        } else if constexpr (Op == EpilogueOp::BIAS_RESIDUAL) {
            bp = side_data + n_start + nc;
            bv0 = __ldg(reinterpret_cast<const float4*>(bp));
            bv1 = __ldg(reinterpret_cast<const float4*>(bp + 4));
            rv0 = __ldg(reinterpret_cast<const uint4*>(res_row + nc));
#endif
        }
#endif

        if (MBAR_EARLY && nc + 32 >= NC_END) {
            if (epi_mbar_addr) mbar_arrive(epi_mbar_addr);
        }

        const uint32_t srow = srow_base + ((nc - NC_START) >> 6) * STAGING_REGION_BYTES;
        const int byte_base = ((nc - NC_START) & 63) * 2;

        // Transform: accumulator → (op-specific) → BF16 → SMEM
        if constexpr (Op == EpilogueOp::BIAS_ADD) {
#if PRELOAD_MODE == 2
            CVT_ADD_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7, craw0.x,craw0.y,craw0.z,craw0.w, srow + (byte_base ^ xor_val));
            CVT_ADD_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15, craw1.x,craw1.y,craw1.z,craw1.w, srow + ((byte_base + 16) ^ xor_val));
            CVT_ADD_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23, craw2.x,craw2.y,craw2.z,craw2.w, srow + ((byte_base + 32) ^ xor_val));
            CVT_ADD_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31, craw3.x,craw3.y,craw3.z,craw3.w, srow + ((byte_base + 48) ^ xor_val));
#else
            CVT_ADD_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7, craw0.x,craw0.y,craw0.z,craw0.w, srow + (byte_base ^ xor_val));
            CVT_ADD_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15, craw1.x,craw1.y,craw1.z,craw1.w, srow + ((byte_base + 16) ^ xor_val));

            craw0 = *reinterpret_cast<const uint4*>(comb_ptr + 16);
            craw1 = *reinterpret_cast<const uint4*>(comb_ptr + 24);
            CVT_ADD_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23, craw0.x,craw0.y,craw0.z,craw0.w, srow + ((byte_base + 32) ^ xor_val));
            CVT_ADD_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31, craw1.x,craw1.y,craw1.z,craw1.w, srow + ((byte_base + 48) ^ xor_val));
#endif
        } else if constexpr (Op == EpilogueOp::BIAS_GELU) {
#if BATCH_EPILOGUE && defined(HAS_GELU_APPROX)
            /*
            Batched epilogue: compute GELU_VECTOR_WIDTH elements, then store.
            Width 32: all 32 GELU → 4x store. Width 16: 2 × (16 GELU → 2x store).
            Width 8: 4 × (8 GELU → 1x store). Compiler pipelines MUFU within batch.
            Requires GELU_VARIANT 0-3 (standalone gelu_approx function).
            */
#if GELU_VECTOR_WIDTH == 32
#if PRELOAD_MODE < 2
            float4 bv2 = __ldg(reinterpret_cast<const float4*>(bp + 8));
            float4 bv3 = __ldg(reinterpret_cast<const float4*>(bp + 12));
            float4 bv4 = __ldg(reinterpret_cast<const float4*>(bp + 16));
            float4 bv5 = __ldg(reinterpret_cast<const float4*>(bp + 20));
            float4 bv6 = __ldg(reinterpret_cast<const float4*>(bp + 24));
            float4 bv7 = __ldg(reinterpret_cast<const float4*>(bp + 28));
#endif
            a0  = gelu_approx(a0,  bv0.x); a1  = gelu_approx(a1,  bv0.y);
            a2  = gelu_approx(a2,  bv0.z); a3  = gelu_approx(a3,  bv0.w);
            a4  = gelu_approx(a4,  bv1.x); a5  = gelu_approx(a5,  bv1.y);
            a6  = gelu_approx(a6,  bv1.z); a7  = gelu_approx(a7,  bv1.w);
            a8  = gelu_approx(a8,  bv2.x); a9  = gelu_approx(a9,  bv2.y);
            a10 = gelu_approx(a10, bv2.z); a11 = gelu_approx(a11, bv2.w);
            a12 = gelu_approx(a12, bv3.x); a13 = gelu_approx(a13, bv3.y);
            a14 = gelu_approx(a14, bv3.z); a15 = gelu_approx(a15, bv3.w);
            a16 = gelu_approx(a16, bv4.x); a17 = gelu_approx(a17, bv4.y);
            a18 = gelu_approx(a18, bv4.z); a19 = gelu_approx(a19, bv4.w);
            a20 = gelu_approx(a20, bv5.x); a21 = gelu_approx(a21, bv5.y);
            a22 = gelu_approx(a22, bv5.z); a23 = gelu_approx(a23, bv5.w);
            a24 = gelu_approx(a24, bv6.x); a25 = gelu_approx(a25, bv6.y);
            a26 = gelu_approx(a26, bv6.z); a27 = gelu_approx(a27, bv6.w);
            a28 = gelu_approx(a28, bv7.x); a29 = gelu_approx(a29, bv7.y);
            a30 = gelu_approx(a30, bv7.z); a31 = gelu_approx(a31, bv7.w);
#if STS_WIDTH == 32
            cvt_sts_v8(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                       srow + (byte_base ^ xor_val), srow + ((byte_base + 16) ^ xor_val));
            cvt_sts_v8(a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                       srow + ((byte_base + 32) ^ xor_val), srow + ((byte_base + 48) ^ xor_val));
#else
            cvt_sts_v4(a0,a1,a2,a3,a4,a5,a6,a7, srow + (byte_base ^ xor_val));
            cvt_sts_v4(a8,a9,a10,a11,a12,a13,a14,a15, srow + ((byte_base + 16) ^ xor_val));
            cvt_sts_v4(a16,a17,a18,a19,a20,a21,a22,a23, srow + ((byte_base + 32) ^ xor_val));
            cvt_sts_v4(a24,a25,a26,a27,a28,a29,a30,a31, srow + ((byte_base + 48) ^ xor_val));
#endif
#elif GELU_VECTOR_WIDTH == 16
#if PRELOAD_MODE < 2
            float4 bv2 = __ldg(reinterpret_cast<const float4*>(bp + 8));
            float4 bv3 = __ldg(reinterpret_cast<const float4*>(bp + 12));
#endif
            a0  = gelu_approx(a0,  bv0.x); a1  = gelu_approx(a1,  bv0.y);
            a2  = gelu_approx(a2,  bv0.z); a3  = gelu_approx(a3,  bv0.w);
            a4  = gelu_approx(a4,  bv1.x); a5  = gelu_approx(a5,  bv1.y);
            a6  = gelu_approx(a6,  bv1.z); a7  = gelu_approx(a7,  bv1.w);
            a8  = gelu_approx(a8,  bv2.x); a9  = gelu_approx(a9,  bv2.y);
            a10 = gelu_approx(a10, bv2.z); a11 = gelu_approx(a11, bv2.w);
            a12 = gelu_approx(a12, bv3.x); a13 = gelu_approx(a13, bv3.y);
            a14 = gelu_approx(a14, bv3.z); a15 = gelu_approx(a15, bv3.w);
#if STS_WIDTH == 32
            cvt_sts_v8(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                       srow + (byte_base ^ xor_val), srow + ((byte_base + 16) ^ xor_val));
#else
            cvt_sts_v4(a0,a1,a2,a3,a4,a5,a6,a7, srow + (byte_base ^ xor_val));
            cvt_sts_v4(a8,a9,a10,a11,a12,a13,a14,a15, srow + ((byte_base + 16) ^ xor_val));
#endif
#if PRELOAD_MODE < 2
            float4 bv4 = __ldg(reinterpret_cast<const float4*>(bp + 16));
            float4 bv5 = __ldg(reinterpret_cast<const float4*>(bp + 20));
            float4 bv6 = __ldg(reinterpret_cast<const float4*>(bp + 24));
            float4 bv7 = __ldg(reinterpret_cast<const float4*>(bp + 28));
#endif
            a16 = gelu_approx(a16, bv4.x); a17 = gelu_approx(a17, bv4.y);
            a18 = gelu_approx(a18, bv4.z); a19 = gelu_approx(a19, bv4.w);
            a20 = gelu_approx(a20, bv5.x); a21 = gelu_approx(a21, bv5.y);
            a22 = gelu_approx(a22, bv5.z); a23 = gelu_approx(a23, bv5.w);
            a24 = gelu_approx(a24, bv6.x); a25 = gelu_approx(a25, bv6.y);
            a26 = gelu_approx(a26, bv6.z); a27 = gelu_approx(a27, bv6.w);
            a28 = gelu_approx(a28, bv7.x); a29 = gelu_approx(a29, bv7.y);
            a30 = gelu_approx(a30, bv7.z); a31 = gelu_approx(a31, bv7.w);
#if STS_WIDTH == 32
            cvt_sts_v8(a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                       srow + ((byte_base + 32) ^ xor_val), srow + ((byte_base + 48) ^ xor_val));
#else
            cvt_sts_v4(a16,a17,a18,a19,a20,a21,a22,a23, srow + ((byte_base + 32) ^ xor_val));
            cvt_sts_v4(a24,a25,a26,a27,a28,a29,a30,a31, srow + ((byte_base + 48) ^ xor_val));
#endif
#else /* GELU_VECTOR_WIDTH == 8 */
            a0  = gelu_approx(a0,  bv0.x); a1  = gelu_approx(a1,  bv0.y);
            a2  = gelu_approx(a2,  bv0.z); a3  = gelu_approx(a3,  bv0.w);
            a4  = gelu_approx(a4,  bv1.x); a5  = gelu_approx(a5,  bv1.y);
            a6  = gelu_approx(a6,  bv1.z); a7  = gelu_approx(a7,  bv1.w);
            cvt_sts_v4(a0,a1,a2,a3,a4,a5,a6,a7, srow + (byte_base ^ xor_val));
#if PRELOAD_MODE < 2
            float4 bv2 = __ldg(reinterpret_cast<const float4*>(bp + 8));
            float4 bv3 = __ldg(reinterpret_cast<const float4*>(bp + 12));
#endif
            a8  = gelu_approx(a8,  bv2.x); a9  = gelu_approx(a9,  bv2.y);
            a10 = gelu_approx(a10, bv2.z); a11 = gelu_approx(a11, bv2.w);
            a12 = gelu_approx(a12, bv3.x); a13 = gelu_approx(a13, bv3.y);
            a14 = gelu_approx(a14, bv3.z); a15 = gelu_approx(a15, bv3.w);
            cvt_sts_v4(a8,a9,a10,a11,a12,a13,a14,a15, srow + ((byte_base + 16) ^ xor_val));
#if PRELOAD_MODE < 2
            float4 bv4 = __ldg(reinterpret_cast<const float4*>(bp + 16));
            float4 bv5 = __ldg(reinterpret_cast<const float4*>(bp + 20));
#endif
            a16 = gelu_approx(a16, bv4.x); a17 = gelu_approx(a17, bv4.y);
            a18 = gelu_approx(a18, bv4.z); a19 = gelu_approx(a19, bv4.w);
            a20 = gelu_approx(a20, bv5.x); a21 = gelu_approx(a21, bv5.y);
            a22 = gelu_approx(a22, bv5.z); a23 = gelu_approx(a23, bv5.w);
            cvt_sts_v4(a16,a17,a18,a19,a20,a21,a22,a23, srow + ((byte_base + 32) ^ xor_val));
#if PRELOAD_MODE < 2
            float4 bv6 = __ldg(reinterpret_cast<const float4*>(bp + 24));
            float4 bv7 = __ldg(reinterpret_cast<const float4*>(bp + 28));
#endif
            a24 = gelu_approx(a24, bv6.x); a25 = gelu_approx(a25, bv6.y);
            a26 = gelu_approx(a26, bv6.z); a27 = gelu_approx(a27, bv6.w);
            a28 = gelu_approx(a28, bv7.x); a29 = gelu_approx(a29, bv7.y);
            a30 = gelu_approx(a30, bv7.z); a31 = gelu_approx(a31, bv7.w);
            cvt_sts_v4(a24,a25,a26,a27,a28,a29,a30,a31, srow + ((byte_base + 48) ^ xor_val));
#endif /* GELU_VECTOR_WIDTH */
#elif PRELOAD_MODE == 2
            GELU_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7, bv0.x,bv0.y,bv0.z,bv0.w,bv1.x,bv1.y,bv1.z,bv1.w, srow + (byte_base ^ xor_val));
            GELU_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15, bv2.x,bv2.y,bv2.z,bv2.w,bv3.x,bv3.y,bv3.z,bv3.w, srow + ((byte_base + 16) ^ xor_val));
            GELU_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23, bv4.x,bv4.y,bv4.z,bv4.w,bv5.x,bv5.y,bv5.z,bv5.w, srow + ((byte_base + 32) ^ xor_val));
            GELU_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31, bv6.x,bv6.y,bv6.z,bv6.w,bv7.x,bv7.y,bv7.z,bv7.w, srow + ((byte_base + 48) ^ xor_val));
#else
            GELU_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7, bv0.x,bv0.y,bv0.z,bv0.w,bv1.x,bv1.y,bv1.z,bv1.w, srow + (byte_base ^ xor_val));
            float4 bv2 = __ldg(reinterpret_cast<const float4*>(bp + 8));
            float4 bv3 = __ldg(reinterpret_cast<const float4*>(bp + 12));
            GELU_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15, bv2.x,bv2.y,bv2.z,bv2.w,bv3.x,bv3.y,bv3.z,bv3.w, srow + ((byte_base + 16) ^ xor_val));

            float4 bv4 = __ldg(reinterpret_cast<const float4*>(bp + 16));
            float4 bv5 = __ldg(reinterpret_cast<const float4*>(bp + 20));
            GELU_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23, bv4.x,bv4.y,bv4.z,bv4.w,bv5.x,bv5.y,bv5.z,bv5.w, srow + ((byte_base + 32) ^ xor_val));
            float4 bv6 = __ldg(reinterpret_cast<const float4*>(bp + 24));
            float4 bv7 = __ldg(reinterpret_cast<const float4*>(bp + 28));
            GELU_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31, bv6.x,bv6.y,bv6.z,bv6.w,bv7.x,bv7.y,bv7.z,bv7.w, srow + ((byte_base + 48) ^ xor_val));
#endif
        } else if constexpr (Op == EpilogueOp::BIAS_RESIDUAL) {
#if BIAS_BF16
            {
                uint4 bbv0 = __ldg(reinterpret_cast<const uint4*>(side_data + n_start + nc));
                uint4 rrv0 = __ldg(reinterpret_cast<const uint4*>(res_row + nc));
                uint4 bbv1 = __ldg(reinterpret_cast<const uint4*>(side_data + n_start + nc + 8));
                uint4 rrv1 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 8));
                BIAS_RES_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7, bbv0.x,bbv0.y,bbv0.z,bbv0.w, rrv0.x,rrv0.y,rrv0.z,rrv0.w, srow + (byte_base ^ xor_val));
                uint4 bbv2 = __ldg(reinterpret_cast<const uint4*>(side_data + n_start + nc + 16));
                uint4 rrv2 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 16));
                BIAS_RES_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15, bbv1.x,bbv1.y,bbv1.z,bbv1.w, rrv1.x,rrv1.y,rrv1.z,rrv1.w, srow + ((byte_base + 16) ^ xor_val));
                uint4 bbv3 = __ldg(reinterpret_cast<const uint4*>(side_data + n_start + nc + 24));
                uint4 rrv3 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 24));
                BIAS_RES_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23, bbv2.x,bbv2.y,bbv2.z,bbv2.w, rrv2.x,rrv2.y,rrv2.z,rrv2.w, srow + ((byte_base + 32) ^ xor_val));
                BIAS_RES_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31, bbv3.x,bbv3.y,bbv3.z,bbv3.w, rrv3.x,rrv3.y,rrv3.z,rrv3.w, srow + ((byte_base + 48) ^ xor_val));
            }
#else /* !BIAS_BF16 */
#if BATCH_EPILOGUE
            /*
            Batched epilogue: load all side-data, compute all 32 bias+residual adds
            as C++ (non-volatile BF16 unpack — compiler can reorder freely),
            then batch 4x cvt_sts_v4. Eliminates 4 compute→STS serialization points.
            */
            {
#if PRELOAD_MODE < 2
                float4 bv2 = __ldg(reinterpret_cast<const float4*>(bp + 8));
                float4 bv3 = __ldg(reinterpret_cast<const float4*>(bp + 12));
                float4 bv4 = __ldg(reinterpret_cast<const float4*>(bp + 16));
                float4 bv5 = __ldg(reinterpret_cast<const float4*>(bp + 20));
                float4 bv6 = __ldg(reinterpret_cast<const float4*>(bp + 24));
                float4 bv7 = __ldg(reinterpret_cast<const float4*>(bp + 28));
                uint4 rv1 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 8));
                uint4 rv2 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 16));
                uint4 rv3 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 24));
#endif
                /* Compute phase: unpack BF16 residual (non-volatile) + bias add for all 32 */
                float rf0, rf1;
                BF16X2_TO_F32_NV(rv0.x, rf0, rf1); a0  = a0  + bv0.x + rf0; a1  = a1  + bv0.y + rf1;
                BF16X2_TO_F32_NV(rv0.y, rf0, rf1); a2  = a2  + bv0.z + rf0; a3  = a3  + bv0.w + rf1;
                BF16X2_TO_F32_NV(rv0.z, rf0, rf1); a4  = a4  + bv1.x + rf0; a5  = a5  + bv1.y + rf1;
                BF16X2_TO_F32_NV(rv0.w, rf0, rf1); a6  = a6  + bv1.z + rf0; a7  = a7  + bv1.w + rf1;
                BF16X2_TO_F32_NV(rv1.x, rf0, rf1); a8  = a8  + bv2.x + rf0; a9  = a9  + bv2.y + rf1;
                BF16X2_TO_F32_NV(rv1.y, rf0, rf1); a10 = a10 + bv2.z + rf0; a11 = a11 + bv2.w + rf1;
                BF16X2_TO_F32_NV(rv1.z, rf0, rf1); a12 = a12 + bv3.x + rf0; a13 = a13 + bv3.y + rf1;
                BF16X2_TO_F32_NV(rv1.w, rf0, rf1); a14 = a14 + bv3.z + rf0; a15 = a15 + bv3.w + rf1;
                BF16X2_TO_F32_NV(rv2.x, rf0, rf1); a16 = a16 + bv4.x + rf0; a17 = a17 + bv4.y + rf1;
                BF16X2_TO_F32_NV(rv2.y, rf0, rf1); a18 = a18 + bv4.z + rf0; a19 = a19 + bv4.w + rf1;
                BF16X2_TO_F32_NV(rv2.z, rf0, rf1); a20 = a20 + bv5.x + rf0; a21 = a21 + bv5.y + rf1;
                BF16X2_TO_F32_NV(rv2.w, rf0, rf1); a22 = a22 + bv5.z + rf0; a23 = a23 + bv5.w + rf1;
                BF16X2_TO_F32_NV(rv3.x, rf0, rf1); a24 = a24 + bv6.x + rf0; a25 = a25 + bv6.y + rf1;
                BF16X2_TO_F32_NV(rv3.y, rf0, rf1); a26 = a26 + bv6.z + rf0; a27 = a27 + bv6.w + rf1;
                BF16X2_TO_F32_NV(rv3.z, rf0, rf1); a28 = a28 + bv7.x + rf0; a29 = a29 + bv7.y + rf1;
                BF16X2_TO_F32_NV(rv3.w, rf0, rf1); a30 = a30 + bv7.z + rf0; a31 = a31 + bv7.w + rf1;
            }
            /* Store phase */
#if STS_WIDTH == 32
            cvt_sts_v8(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                       srow + (byte_base ^ xor_val), srow + ((byte_base + 16) ^ xor_val));
            cvt_sts_v8(a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                       srow + ((byte_base + 32) ^ xor_val), srow + ((byte_base + 48) ^ xor_val));
#else
            cvt_sts_v4(a0,a1,a2,a3,a4,a5,a6,a7, srow + (byte_base ^ xor_val));
            cvt_sts_v4(a8,a9,a10,a11,a12,a13,a14,a15, srow + ((byte_base + 16) ^ xor_val));
            cvt_sts_v4(a16,a17,a18,a19,a20,a21,a22,a23, srow + ((byte_base + 32) ^ xor_val));
            cvt_sts_v4(a24,a25,a26,a27,a28,a29,a30,a31, srow + ((byte_base + 48) ^ xor_val));
#endif
#elif PRELOAD_MODE == 2
            BIAS_RES_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7, bv0.x,bv0.y,bv0.z,bv0.w,bv1.x,bv1.y,bv1.z,bv1.w, rv0.x,rv0.y,rv0.z,rv0.w, srow + (byte_base ^ xor_val));
            BIAS_RES_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15, bv2.x,bv2.y,bv2.z,bv2.w,bv3.x,bv3.y,bv3.z,bv3.w, rv1.x,rv1.y,rv1.z,rv1.w, srow + ((byte_base + 16) ^ xor_val));
            BIAS_RES_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23, bv4.x,bv4.y,bv4.z,bv4.w,bv5.x,bv5.y,bv5.z,bv5.w, rv2.x,rv2.y,rv2.z,rv2.w, srow + ((byte_base + 32) ^ xor_val));
            BIAS_RES_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31, bv6.x,bv6.y,bv6.z,bv6.w,bv7.x,bv7.y,bv7.z,bv7.w, rv3.x,rv3.y,rv3.z,rv3.w, srow + ((byte_base + 48) ^ xor_val));
#else
            BIAS_RES_CVT_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7, bv0.x,bv0.y,bv0.z,bv0.w,bv1.x,bv1.y,bv1.z,bv1.w, rv0.x,rv0.y,rv0.z,rv0.w, srow + (byte_base ^ xor_val));
            float4 bv2 = __ldg(reinterpret_cast<const float4*>(bp + 8));
            float4 bv3 = __ldg(reinterpret_cast<const float4*>(bp + 12));
            uint4 rv1 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 8));
            BIAS_RES_CVT_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15, bv2.x,bv2.y,bv2.z,bv2.w,bv3.x,bv3.y,bv3.z,bv3.w, rv1.x,rv1.y,rv1.z,rv1.w, srow + ((byte_base + 16) ^ xor_val));

            float4 bv4 = __ldg(reinterpret_cast<const float4*>(bp + 16));
            float4 bv5 = __ldg(reinterpret_cast<const float4*>(bp + 20));
            uint4 rv2 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 16));
            BIAS_RES_CVT_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23, bv4.x,bv4.y,bv4.z,bv4.w,bv5.x,bv5.y,bv5.z,bv5.w, rv2.x,rv2.y,rv2.z,rv2.w, srow + ((byte_base + 32) ^ xor_val));
            float4 bv6 = __ldg(reinterpret_cast<const float4*>(bp + 24));
            float4 bv7 = __ldg(reinterpret_cast<const float4*>(bp + 28));
            uint4 rv3 = __ldg(reinterpret_cast<const uint4*>(res_row + nc + 24));
            BIAS_RES_CVT_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31, bv6.x,bv6.y,bv6.z,bv6.w,bv7.x,bv7.y,bv7.z,bv7.w, rv3.x,rv3.y,rv3.z,rv3.w, srow + ((byte_base + 48) ^ xor_val));
#endif
#endif /* BIAS_BF16 */
        }

#if PREFETCH_BEFORE_STORE
        if (nc + 32 < NC_END) {
            LOAD_32_COLS(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                         a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                         taddr_base + nc + 32);
        }
#endif

        // Interleaved TMA store(s) — region completes every 2 x32 iterations
#if !STORE_TIMING
        if (INTERLEAVE_STRATEGY == 1 && ((nc - NC_START) & 63) == 32) {
            int region_idx = (nc - NC_START) >> 6;
            __syncwarp();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (lane == 0) {
                uint32_t src = staging_saddr + region_idx * STAGING_REGION_BYTES;
                asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                    :: "l"(tma_c_desc), "r"(n_start + NC_START + region_idx * 64), "r"(gm_base), "r"(src) : "memory");
            }
        } else if (INTERLEAVE_STRATEGY == 2 && ((nc - NC_START) & 63) == 32 && (((nc - NC_START) >> 6) & 1) == 1) {
            int region_idx = (nc - NC_START) >> 6;
            __syncwarp();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (lane == 0) {
                uint32_t src0 = staging_saddr + (region_idx - 1) * STAGING_REGION_BYTES;
                uint32_t src1 = staging_saddr + region_idx * STAGING_REGION_BYTES;
                asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                    :: "l"(tma_c_desc), "r"(n_start + NC_START + (region_idx - 1) * 64), "r"(gm_base), "r"(src0) : "memory");
                asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                    :: "l"(tma_c_desc), "r"(n_start + NC_START + region_idx * 64), "r"(gm_base), "r"(src1) : "memory");
            }
        } else if (INTERLEAVE_STRATEGY == 3 && ((nc - NC_START) & 63) == 32 && ((nc - NC_START) >> 6) == (N_REGIONS < 3 ? N_REGIONS - 1 : 2)) {
            constexpr int INLINE_REGIONS = N_REGIONS < 3 ? N_REGIONS : 3;
            __syncwarp();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (lane == 0) {
                for (int r = 0; r < INLINE_REGIONS; r++) {
                    uint32_t src = staging_saddr + r * STAGING_REGION_BYTES;
                    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                        :: "l"(tma_c_desc), "r"(n_start + NC_START + r * 64), "r"(gm_base), "r"(src) : "memory");
                }
            }
        }
#endif /* !STORE_TIMING */

#if !PREFETCH_BEFORE_STORE
        if (nc + 32 < NC_END) {
            LOAD_32_COLS(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                         a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                         taddr_base + nc + 32);
        }
#endif
    }
#endif

#ifdef TIMING
    t_phase1_end = clock64();
#endif

#if STORE_TIMING || INTERLEAVE_STRATEGY == 0
    // All-at-end: single fence + all TMA stores (STORE_TIMING=1 or strategy 0)
    __syncwarp();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    if (!MBAR_EARLY && epi_mbar_addr) mbar_arrive(epi_mbar_addr);

    if (lane == 0) {
        int row = gm_base;
        PRAGMA_UNROLL(N_REGIONS)
        for (int r = 0; r < N_REGIONS; r++) {
            uint32_t src = staging_saddr + r * STAGING_REGION_BYTES;
            asm volatile(
                "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                :: "l"(tma_c_desc), "r"(n_start + NC_START + r * 64), "r"(row), "r"(src) : "memory");
        }
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    }
#elif INTERLEAVE_STRATEGY == 3
    // Strategy 3 (three-plus-one): inline stores cover first 3 regions, Phase 2 handles last
    __syncwarp();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if (!MBAR_EARLY && epi_mbar_addr) mbar_arrive(epi_mbar_addr);
    if (lane == 0) {
        if constexpr (N_REGIONS > 3) {
            uint32_t src = staging_saddr + 3 * STAGING_REGION_BYTES;
            asm volatile(
                "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                :: "l"(tma_c_desc), "r"(n_start + NC_START + 192), "r"(gm_base), "r"(src) : "memory");
        }
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    }
#else
    // Strategies 1, 2: all stores issued inline. Just signal + commit.
    if (!MBAR_EARLY && epi_mbar_addr) mbar_arrive(epi_mbar_addr);
    if (lane == 0) {
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    }
#endif
}

#if (TMA_RESIDUAL >= 2 || W0_RES_PREFETCH || W0_RES_FULL) && !EPI_PIPELINE
#define EPI_PRELOADED true
#else
#define EPI_PRELOADED false
#endif

/*
Persistent GEMM — warp-specialized tcgen05 (cta_group::2)
*/

template<EpilogueOp Op>
__global__ void __launch_bounds__(THREADS, 1)
__cluster_dims__(2, 1, 1)
persistent_gemm(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    const __grid_constant__ CUtensorMap tma_c,
    SideDataPtr<Op> __restrict__ side_data,
    __nv_bfloat16* __restrict__ C,
    const __nv_bfloat16* __restrict__ residual
#if TMA_RESIDUAL
    , const __grid_constant__ CUtensorMap tma_res
#endif
#ifdef TIMING
    , long long* __restrict__ timing_buf
    , long long* __restrict__ spread_buf
#endif
) {

    extern __shared__ __align__(128) char smem[];
    const int sm_id = blockIdx.x;
    const int tid   = threadIdx.x;
    const int warp  = tid / 32;
    const int lane  = tid % 32;

    /* cluster_dims(2,1,1): CTA rank = position within 2-CTA cluster.
       Derived from blockIdx.x to avoid asm → keeps uniform chain for UR allocation. */
    const int cta_rank = sm_id & 1;
    const int cluster_id = sm_id / 2;
    const int num_clusters = SM_COUNT / 2;

    // Mbarrier init
    if (tid == 0) {
        for (int s = 0; s < N_STAGES; s++) {
            mbar_init(smem_to_uint(smem + OFF_TMA_MBAR + s * 8), 2);
            mbar_init(smem_to_uint(smem + OFF_MMA_MBAR + s * 8), 1);
        }
        for (int i = 0; i < 2; i++) {
            mbar_init(smem_to_uint(smem + OFF_MAINLOOP_MBAR + i * 8), 1);
            mbar_init(smem_to_uint(smem + OFF_EPILOGUE_MBAR + i * 8), NUM_EPI_WARPS * 2 * 32);
        }
#if TMA_RESIDUAL
#if EPI_PIPELINE
        /* Pipeline epilogue mbars: per-stage load + tile-free */
        for (int s = 0; s < EPI_PIPELINE_STAGES; s++)
            mbar_init(smem_to_uint(smem + OFF_PIPE_LOAD_MBAR + s * 8), 1);
        mbar_init(smem_to_uint(smem + OFF_PIPE_FREE_MBAR), NUM_EPI_WARPS);
#else
        for (int w = 0; w < NUM_EPI_WARPS; w++)
            mbar_init(smem_to_uint(smem + OFF_RES_MBAR + w * 8), 1);
#if STAGES_C >= 2
        /* StagesC load mbar: one per (warp, stage). W0 arrives with TMA expect_tx. */
        for (int i = 0; i < NUM_EPI_WARPS * STAGES_C; i++)
            mbar_init(smem_to_uint(smem + OFF_SC_LOAD_MBAR + i * 8), 1);
        /* StagesC release mbar: one per stage. All consumer warps arrive when done. */
        for (int s = 0; s < STAGES_C; s++)
            mbar_init(smem_to_uint(smem + OFF_SC_REL_MBAR + s * 8), NUM_EPI_WARPS);
#elif W0_RES_FULL || EPI_LOAD_WARP
        mbar_init(smem_to_uint(smem + OFF_RES_CONSUMED_MBAR), NUM_EPI_WARPS);
        mbar_init(smem_to_uint(smem + OFF_RES_PASS_MBAR), NUM_EPI_WARPS);
#elif W0_RES_PREFETCH
        mbar_init(smem_to_uint(smem + OFF_RES_CONSUMED_MBAR), NUM_EPI_WARPS);
#endif
#endif /* EPI_PIPELINE */
#endif
#if EPI_REUSE_SMEM
        mbar_init(smem_to_uint(smem + OFF_EPI_DONE_MBAR), NUM_EPI_WARPS);
#endif
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    // TMEM alloc
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
#if EPI_REUSE_SMEM
    int epi_done_phase = 0;
#endif

    uint64_t desc_a_base[N_STAGES], desc_b_base[N_STAGES];
    for (int s = 0; s < N_STAGES; s++) {
        desc_a_base[s] = make_smem_desc(smem_a[s]);
        desc_b_base[s] = make_smem_desc(smem_b[s]);
    }

    const int start_buf = tile_start & 1;
    int epi_phase[2] = {1, 1};
    int ml_phase[2]  = {start_buf, 1 - start_buf};
#if EPI_PIPELINE
    int pipe_free_phase = 0;
#elif STAGES_C >= 2
    int sc_rel_phase[STAGES_C] = {0};
    int sc_consumer_phase = 0;
#elif W0_RES_FULL || EPI_LOAD_WARP
    int res_consumed_phase = 0;
    int res_pass_phase = 0;
#elif W0_RES_PREFETCH
    int res_consumed_phase = 0;
#endif

#ifdef TIMING
    long long t_tile_start = 0, t_after_epi = 0, t_after_tma0 = 0, t_kloop_end = 0;
    long long sum_epi_wait = 0, sum_tma0_wait = 0, sum_kloop = 0, sum_total = 0;
    long long min_kloop = 0x7FFFFFFFFFFFFFFFLL, max_kloop = 0;
    long long min_total = 0x7FFFFFFFFFFFFFFFLL, max_total = 0;
    int tile_count = 0;
    long long gt_tile_start = 0, gt_kloop_end = 0, sum_gt_total = 0;
    long long epi_t0 = 0, epi_t1 = 0;
    long long epi_sum_p1 = 0;
    long long epi_min_p1 = 0x7FFFFFFFFFFFFFFFLL, epi_max_p1 = 0;
    int epi_count = 0;
    long long epi_t_before_ml = 0, epi_t2 = 0;
    long long epi_sum_p2 = 0, epi_sum_ml = 0;
    long long epi_min_p2 = 0x7FFFFFFFFFFFFFFFLL, epi_max_p2 = 0;
    long long epi_min_ml = 0x7FFFFFFFFFFFFFFFLL, epi_max_ml = 0;
#endif

#if OVERLAP_EPI_WAIT
    uint32_t epi_pf_ready_ = 0;
#endif

    for (int tile_idx = tile_start; tile_idx < tile_end; tile_idx++) {
        const int buf = tile_idx & 1;
        const int tm = tile_idx / TILES_N;
        int tn = tile_idx % TILES_N;
        if (SNAKE_ORDER && (tm & 1)) tn = TILES_N - 1 - tn;
        const int m_start = tm * TM * 2 + cta_rank * TM;
        const int n_start = tn * TN;

        if (warp == 0) {
            /*
            LOAD WARP (W0) — R2UR fix: all threads compute addresses as
            integer offsets from smem_base (uniform), eliminating array lookups
            that break the compiler's uniformity tracking. Only lane 0 issues TMA.
            */
            const uint32_t smem_base = warp_uniform(smem_to_uint(smem));
            MAYBE_UNROLL_W0
            for (int ki = 0; ki < K_ITERS; ki++) {
                const int s = ki % N_STAGES;
                const int k_start = ki * TK;
                const uint32_t mma_mbar_s = smem_base + OFF_MMA_MBAR + s * 8;
                const uint32_t tma_mbar_s = (smem_base + OFF_TMA_MBAR + s * 8) & 0xFEFFFFFF;

                if (tile_idx > tile_start || ki >= N_STAGES) {
                    mbar_wait(mma_mbar_s, mma_phase[s]);
                    mma_phase[s] ^= 1;
                }

#if EPI_REUSE_SMEM
                /* Wait for previous tile's epilogue to release borrowed pipeline stages.
                   Only needed once per tile, at the first K-iter using a borrowed stage. */
                if (ki == EPI_FIRST_BORROW_KI && tile_idx > tile_start) {
                    mbar_wait(smem_base + OFF_EPI_DONE_MBAR, epi_done_phase);
                    epi_done_phase ^= 1;
                }
#endif

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
#if STAGES_C >= 2
            /*
            StagesC: W0 pre-loads ALL residual passes for the CURRENT tile.
            The epilogue for this tile runs in the NEXT iteration, so the data
            will be ready by then. No waiting for consumer here — just issue
            TMA loads. Backpressure via release_mbar prevents overwriting a
            buffer the consumer is still reading (only needed from tile_start+2).
            */
            if (lane == 0) {
                if constexpr (Op == EpilogueOp::BIAS_RESIDUAL) {
                    for (int pass = 0; pass < 2; pass++) {
                        /* Wait for consumer to release this stage (skip first tile — no consumer yet) */
                        if (tile_idx > tile_start) {
                            mbar_wait(smem_base + OFF_SC_REL_MBAR + pass * 8,
                                      sc_rel_phase[pass]);
                            sc_rel_phase[pass] ^= 1;
                        }
                        /* Issue TMA loads for all epilogue warps into stage[pass] */
                        for (int ew = 0; ew < NUM_EPI_WARPS; ew++) {
                            const int gm = m_start + ew * 32;
                            const uint32_t load_mbar = smem_base + OFF_SC_LOAD_MBAR
                                + (ew * STAGES_C + pass) * 8;
                            const uint32_t rstg = smem_base + OFF_STAGING
                                + ew * STAGING_WARP_BYTES
                                + RES_STAGING_OFFSET + pass * 2 * STAGING_REGION_BYTES;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;\n\t"
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                " [%2], [%3, {%4, %5}], [%0];\n\t"
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                " [%6], [%3, {%7, %5}], [%0];"
                                :: "r"(load_mbar), "r"(2 * STAGING_REGION_BYTES),
                                   "r"(rstg), "l"(&tma_res),
                                   "r"(n_start + pass * 128), "r"(gm),
                                   "r"(rstg + STAGING_REGION_BYTES),
                                   "r"(n_start + pass * 128 + 64)
                                : "memory");
                        }
                    }
                }
            }
#elif W0_RES_FULL && !STRIP_EPILOGUE
            if (lane == 0) {
                if constexpr (Op == EpilogueOp::BIAS_RESIDUAL) {
                    /*
                    Load residual for the PREVIOUS tile — the one the epilogue is
                    processing concurrently in this iteration. Handshake:
                      W0 loads pass 0 → epilogue consumes → signals res_pass_mbar →
                      W0 loads pass 1 → epilogue consumes → signals res_consumed_mbar.
                    Previous code loaded for the CURRENT tile and deadlocked because
                    the current tile's epilogue runs in the NEXT iteration.
                    */
                    if (tile_idx > tile_start) {
                        /* Wait for tile before previous to finish reading residual */
                        if (tile_idx > tile_start + 1) {
                            mbar_wait(smem_base + OFF_RES_CONSUMED_MBAR, res_consumed_phase);
                            res_consumed_phase ^= 1;
                        }
                        /* Compute previous tile's coordinates */
                        const int prev_idx = tile_idx - 1;
                        const int ptm = prev_idx / TILES_N;
                        int ptn = prev_idx % TILES_N;
                        if (SNAKE_ORDER && (ptm & 1)) ptn = TILES_N - 1 - ptn;
                        const int prev_m = ptm * TM * 2 + cta_rank * TM;
                        const int prev_n = ptn * TN;

                        for (int pass = 0; pass < 2; pass++) {
                            if (pass > 0) {
                                mbar_wait(smem_base + OFF_RES_PASS_MBAR, res_pass_phase);
                                res_pass_phase ^= 1;
                            }
                            for (int ew = 0; ew < NUM_EPI_WARPS; ew++) {
                                const int gm = prev_m + ew * 32;
                                const uint32_t rmbar = smem_base + OFF_RES_MBAR + ew * 8;
                                const uint32_t rstg = smem_base + OFF_STAGING
                                    + ew * STAGING_WARP_BYTES + RES_STAGING_OFFSET;
#if NUM_PASSES_PARAM == 4
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;\n\t"
                                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                    " [%2], [%3, {%4, %5}], [%0];"
                                    :: "r"(rmbar), "r"(STAGING_REGION_BYTES),
                                       "r"(rstg), "l"(&tma_res), "r"(prev_n + pass * 64), "r"(gm)
                                    : "memory");
#else
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;\n\t"
                                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                    " [%2], [%3, {%4, %5}], [%0];\n\t"
                                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                    " [%6], [%3, {%7, %5}], [%0];"
                                    :: "r"(rmbar), "r"(2 * STAGING_REGION_BYTES),
                                       "r"(rstg), "l"(&tma_res), "r"(prev_n + pass * 128), "r"(gm),
                                       "r"(rstg + STAGING_REGION_BYTES), "r"(prev_n + pass * 128 + 64)
                                    : "memory");
#endif
                            }
                        }
                    }
                }
            }
#elif W0_RES_PREFETCH && !STRIP_EPILOGUE
            if (lane == 0) {
                if constexpr (Op == EpilogueOp::BIAS_RESIDUAL) {
                    /* Wait for prev tile's epilogue to finish reading residual */
                    if (tile_idx > tile_start) {
                        mbar_wait(smem_base + OFF_RES_CONSUMED_MBAR, res_consumed_phase);
                        res_consumed_phase ^= 1;
                    }
                    /* Prefetch pass-0 residual for ALL epilogue warps */
                    for (int ew = 0; ew < NUM_EPI_WARPS; ew++) {
                        const int gm = m_start + ew * 32;
                        const uint32_t rmbar = smem_base + OFF_RES_MBAR + ew * 8;
                        const uint32_t rstg = smem_base + OFF_STAGING
                            + ew * STAGING_WARP_BYTES + RES_STAGING_OFFSET;
#if NUM_PASSES_PARAM == 4
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;\n\t"
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%2], [%3, {%4, %5}], [%0];"
                            :: "r"(rmbar), "r"(STAGING_REGION_BYTES),
                               "r"(rstg), "l"(&tma_res), "r"(n_start), "r"(gm)
                            : "memory");
#else
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;\n\t"
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%2], [%3, {%4, %5}], [%0];\n\t"
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                            " [%6], [%3, {%7, %5}], [%0];"
                            :: "r"(rmbar), "r"(2 * STAGING_REGION_BYTES),
                               "r"(rstg), "l"(&tma_res), "r"(n_start), "r"(gm),
                               "r"(rstg + STAGING_REGION_BYTES), "r"(n_start + 64)
                            : "memory");
#endif
                    }
                }
            }
#endif
        } else if (warp == 1) {
            // MMA WARP (W1)
            if (lane == 0 && cta_rank == 0) {
#ifdef TIMING
                t_tile_start = clock64();
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_tile_start));
#endif
#if OVERLAP_EPI_WAIT
                if (!epi_pf_ready_)
#endif
                    mbar_wait(epilogue_mbar_addr + buf * 8, epi_phase[buf]);
#if OVERLAP_EPI_WAIT
                epi_pf_ready_ = 0;
#endif
                epi_phase[buf] ^= 1;
#ifdef TIMING
                t_after_epi = clock64();
#endif

                mbar_wait(tma_mbar[0], tma_phase[0]);
                tma_phase[0] ^= 1;
#ifdef TIMING
                t_after_tma0 = clock64();
#endif
                asm volatile("tcgen05.fence::after_thread_sync;");
#if BATCH_MMA
                {
                    uint64_t da0 = desc_a_base[0], db0 = desc_b_base[0];
                    uint64_t da1 = da0+2, da2 = da0+4, da3 = da0+6;
                    uint64_t db1 = db0+2, db2 = db0+4, db3 = db0+6;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p0, p1;\n\t"
                        "setp.ne.b32 p0, 0, 0;\n\t"
                        "setp.ne.b32 p1, 1, 0;\n\t"
                        "tcgen05.mma.cta_group::2.kind::f8f6f4 "
                        "[%0], %1, %2, %3, {%10,%11,%12,%13, %14,%15,%16,%17}, p0;\n\t"
                        "tcgen05.mma.cta_group::2.kind::f8f6f4 "
                        "[%0], %4, %5, %3, {%10,%11,%12,%13, %14,%15,%16,%17}, p1;\n\t"
                        "tcgen05.mma.cta_group::2.kind::f8f6f4 "
                        "[%0], %6, %7, %3, {%10,%11,%12,%13, %14,%15,%16,%17}, p1;\n\t"
                        "tcgen05.mma.cta_group::2.kind::f8f6f4 "
                        "[%0], %8, %9, %3, {%10,%11,%12,%13, %14,%15,%16,%17}, p1;\n\t"
                        "}"
                        :
                        : "r"(buf * TN), "l"(da0), "l"(db0), "r"(IDESC),
                          "l"(da1), "l"(db1), "l"(da2), "l"(db2), "l"(da3), "l"(db3),
                          "r"(0),"r"(0),"r"(0),"r"(0),
                          "r"(0),"r"(0),"r"(0),"r"(0));
                }
#else
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
                    MAYBE_UNROLL_SUB
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
#endif
                tcgen05_commit_mcast(mma_mbar[0], 0x3);
#if PREFETCH_MBAR
                uint32_t pf_ready_ = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred pf;\n\t"
                    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 pf, [%1], %2, 0x989680;\n\t"
                    "selp.b32 %0, 1, 0, pf;\n\t"
                    "}"
                    : "=r"(pf_ready_) : "r"(tma_mbar[1 % N_STAGES]), "r"(tma_phase[1 % N_STAGES]));
#endif
                PRAGMA_UNROLL(K_LOOP_UNROLL)
                for (int ki = 1; ki < K_ITERS; ki++) {
                    K_ITER_ACCUM(ki % N_STAGES);
#if PREFETCH_MBAR
                    if (ki + 1 < K_ITERS) {
                        const int ns_ = (ki + 1) % N_STAGES;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred pf;\n\t"
                            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 pf, [%1], %2, 0x989680;\n\t"
                            "selp.b32 %0, 1, 0, pf;\n\t"
                            "}"
                            : "=r"(pf_ready_) : "r"(tma_mbar[ns_]), "r"(tma_phase[ns_]));
                    }
#endif
                }

#if OVERLAP_EPI_WAIT
                {
                    const int next_buf = buf ^ 1;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred pf;\n\t"
                        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 pf, [%1], %2, 0x989680;\n\t"
                        "selp.b32 %0, 1, 0, pf;\n\t"
                        "}"
                        : "=r"(epi_pf_ready_)
                        : "r"(epilogue_mbar_addr + next_buf * 8), "r"(epi_phase[next_buf]));
                }
#endif
                tcgen05_commit_mcast(mainloop_mbar_addr + buf * 8, 0x3);
#ifdef TIMING
                t_kloop_end = clock64();
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_kloop_end));
                long long dt_epi = t_after_epi - t_tile_start;
                long long dt_tma0 = t_after_tma0 - t_after_epi;
                long long dt_kloop = t_kloop_end - t_after_tma0;
                long long dt_total = t_kloop_end - t_tile_start;
                sum_epi_wait += dt_epi;
                sum_tma0_wait += dt_tma0;
                sum_kloop += dt_kloop;
                sum_total += dt_total;
                sum_gt_total += gt_kloop_end - gt_tile_start;
                if (dt_kloop < min_kloop) min_kloop = dt_kloop;
                if (dt_kloop > max_kloop) max_kloop = dt_kloop;
                if (dt_total < min_total) min_total = dt_total;
                if (dt_total > max_total) max_total = dt_total;
                timing_buf[cluster_id * TIMING_CLUSTER_STRIDE + 32 + tile_count] = dt_total;
                tile_count++;
                t_tile_start = clock64();
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_tile_start));
#endif
            }
        }
#if EPI_PIPELINE && !STRIP_EPILOGUE
        else if (warp == 2) {
            /*
            EPI_PIPELINE (W2) — fire-and-forget residual TMA loader.
            Loads residual for the PREVIOUS tile into 4 pipeline stages (one per 64-col
            sub-iteration). All 16 TMA loads (4 stages × 4 warps) issued at once.
            W2 is idle for the rest of the tile — no spinning, no dispatch contention.
            */
            if constexpr (Op == EpilogueOp::BIAS_RESIDUAL) {
                if (lane == 0) {
                    const uint32_t smem_base = smem_to_uint(smem);
                    if (tile_idx > tile_start) {
                        /* Wait for previous tile's stages to be free (skip first tile) */
                        if (tile_idx > tile_start + 1) {
                            mbar_wait(smem_base + OFF_PIPE_FREE_MBAR, pipe_free_phase);
                            pipe_free_phase ^= 1;
                        }
                        const int prev_idx = tile_idx - 1;
                        const int ptm = prev_idx / TILES_N;
                        int ptn = prev_idx % TILES_N;
                        if (SNAKE_ORDER && (ptm & 1)) ptn = TILES_N - 1 - ptn;
                        const int prev_m = ptm * TM * 2 + cta_rank * TM;
                        const int prev_n = ptn * TN;

                        /* Fire all stages × warps TMA loads (fire-and-forget) */
                        for (int si = 0; si < EPI_PIPELINE_STAGES; si++) {
                            const uint32_t stage_mbar = smem_base + OFF_PIPE_LOAD_MBAR + si * 8;
                            mbar_arrive_expect_tx(stage_mbar, NUM_EPI_WARPS * STAGING_REGION_BYTES);
                            for (int ew = 0; ew < NUM_EPI_WARPS; ew++) {
                                const int gm = prev_m + (ew % 4) * 32;
                                const uint32_t rstg = smem_base + OFF_STAGING
                                    + si * STAGE_C_BYTES + ew * STAGING_REGION_BYTES;
                                tma_load_2d_cta(rstg, &tma_res, prev_n + si * 64, gm, stage_mbar);
                            }
                        }
                    }
                }
            }
        }
#elif EPI_LOAD_WARP && !STRIP_EPILOGUE
        else if (warp == 2) {
            /* Legacy serial EPI_LOAD_WARP — kept for reference, dead path with EPI_PIPELINE */
            if constexpr (Op == EpilogueOp::BIAS_RESIDUAL) {
                if (lane == 0) {
                    const uint32_t smem_base = smem_to_uint(smem);
                    if (tile_idx > tile_start) {
                        if (tile_idx > tile_start + 1) {
                            mbar_wait(smem_base + OFF_RES_CONSUMED_MBAR, res_consumed_phase);
                            res_consumed_phase ^= 1;
                        }
                        const int prev_idx = tile_idx - 1;
                        const int ptm = prev_idx / TILES_N;
                        int ptn = prev_idx % TILES_N;
                        if (SNAKE_ORDER && (ptm & 1)) ptn = TILES_N - 1 - ptn;
                        const int prev_m = ptm * TM * 2 + cta_rank * TM;
                        const int prev_n = ptn * TN;

                        for (int pass = 0; pass < 2; pass++) {
                            if (pass > 0) {
                                mbar_wait(smem_base + OFF_RES_PASS_MBAR, res_pass_phase);
                                res_pass_phase ^= 1;
                            }
                            for (int ew = 0; ew < NUM_EPI_WARPS; ew++) {
                                const int gm = prev_m + (ew % 4) * 32;
                                const uint32_t rmbar = smem_base + OFF_RES_MBAR + ew * 8;
                                const uint32_t rstg = smem_base + OFF_STAGING
                                    + ew * STAGING_WARP_BYTES + RES_STAGING_OFFSET;
#if NUM_PASSES_PARAM == 4
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;\n\t"
                                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                    " [%2], [%3, {%4, %5}], [%0];"
                                    :: "r"(rmbar), "r"(STAGING_REGION_BYTES),
                                       "r"(rstg), "l"(&tma_res), "r"(prev_n + pass * 64), "r"(gm)
                                    : "memory");
#else
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;\n\t"
                                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                    " [%2], [%3, {%4, %5}], [%0];\n\t"
                                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                    " [%6], [%3, {%7, %5}], [%0];"
                                    :: "r"(rmbar), "r"(2 * STAGING_REGION_BYTES),
                                       "r"(rstg), "l"(&tma_res), "r"(prev_n + pass * 128), "r"(gm),
                                       "r"(rstg + STAGING_REGION_BYTES), "r"(prev_n + pass * 128 + 64)
                                    : "memory");
#endif
                            }
                        }
                    }
                }
            }
        }
#endif
        else {
            // OVERLAPPED EPILOGUE (W2+EPI_LOAD_WARP .. )
            const int ew = warp - 2 - EPI_LOAD_WARP;
            const int row_group = ew % 4;
#if NUM_EPI_WARPS > 4
            const int is_split = (row_group < (NUM_EPI_WARPS - 4)) ? 1 : 0;
            const int col_rank = ew / 4;
#endif
#if EPI_PIPELINE
            const uint32_t staging_saddr = smem_to_uint(smem + OFF_STAGING);
#else
            const uint32_t staging_saddr = smem_to_uint(smem + OFF_STAGING + ew * STAGING_WARP_BYTES);
#endif

            const int prev_buf = buf ^ 1;

            /* Hoist prev-tile coords above mainloop wait (pure arithmetic) */
            int prev_n = 0;
            int gm_base = 0;
            if (tile_idx > tile_start) {
                const int prev_idx = tile_idx - 1;
                const int ptm = prev_idx / TILES_N;
                int ptn = prev_idx % TILES_N;
                if (SNAKE_ORDER && (ptm & 1)) ptn = TILES_N - 1 - ptn;
                const int prev_m = ptm * TM * 2 + cta_rank * TM;
                prev_n = ptn * TN;
                gm_base = prev_m + row_group * 32;
            }

#if TMA_RESIDUAL >= 2 && !W0_RES_PREFETCH && !W0_RES_FULL && !EPI_PIPELINE && !STRIP_EPILOGUE
            /* Preload first-pass residual before mainloop wait — TMA flies during idle */
            if constexpr (Op == EpilogueOp::BIAS_RESIDUAL) {
                if (tile_idx > tile_start && lane == 0) {
                    const uint32_t res_mbar = smem_to_uint(smem + OFF_RES_MBAR + ew * 8);
                    const uint32_t res_stg = staging_saddr + RES_STAGING_OFFSET;
#if NUM_EPI_WARPS > 4
                    const int nc0 = is_split ? (col_rank * (TN/2)) : 0;
#else
                    const int nc0 = 0;
#endif
#if NUM_PASSES_PARAM == 4
                    mbar_arrive_expect_tx(res_mbar, STAGING_REGION_BYTES);
                    tma_load_2d_cta(res_stg, &tma_res, prev_n + nc0, gm_base, res_mbar);
#else
                    mbar_arrive_expect_tx(res_mbar, 2 * STAGING_REGION_BYTES);
                    tma_load_2d_cta(res_stg, &tma_res, prev_n + nc0, gm_base, res_mbar);
                    tma_load_2d_cta(res_stg + STAGING_REGION_BYTES, &tma_res, prev_n + nc0 + 64, gm_base, res_mbar);
#endif
                }
            }
#endif

#ifdef TIMING
            if (ew == 1 && lane == 0 && cta_rank == 0)
                epi_t_before_ml = clock64();
#endif
            mbar_wait(mainloop_mbar_addr + prev_buf * 8, ml_phase[prev_buf]);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#if EPI_SYNC
            asm volatile("bar.sync 1, %0;" :: "r"(NUM_EPI_WARPS * 32) : "memory");
#endif
            ml_phase[prev_buf] ^= 1;
            if (STAGGER_CYCLES > 0 && ew > 0 && lane == 0) {
                long long __stagger_end = clock64() + ew * STAGGER_CYCLES;
                while (clock64() < __stagger_end) {}
            }
            __syncwarp();
#ifdef TIMING
            if (lane == 0 && cta_rank == 0)
                epi_t0 = clock64();
#endif

#if STRIP_EPILOGUE
            if (tile_idx > tile_start) {
                const uint32_t epi_mbar_masked = (epilogue_mbar_addr + prev_buf * 8) & 0xFEFFFFFF;
#if EPI_DELAY
                /* Busy-wait BEFORE arrive — tests dispatch + TMEM release timing */
                {
                    long long __delay_end = clock64() + EPI_DELAY;
                    while (clock64() < __delay_end) {}
                }
#endif
                mbar_arrive(epi_mbar_masked);
#if NOP_EPILOGUE
                /* Busy-wait AFTER arrive — tests dispatch pressure only (TMEM already free) */
                {
                    long long __nop_end = clock64() + NOP_EPILOGUE;
                    while (clock64() < __nop_end) {}
                }
#endif
            }
#else
            if (tile_idx > tile_start) {
                const uint32_t epi_mbar_masked = (epilogue_mbar_addr + prev_buf * 8) & 0xFEFFFFFF;
#if REG_PAD
                /* Runtime skip: full epilogue compiled (186 regs) but not executed — tests RF occupancy */
                int __rp;
                asm volatile("ld.global.s32 %0, [%1];" : "=r"(__rp) : "l"(&d_regpad_skip));
                if (__rp) {
                    mbar_arrive(epi_mbar_masked);
                } else {
#endif
#if NUM_EPI_WARPS > 4
                if (is_split) {
                    if (col_rank == 0)
                        epilogue_store<0, TN/2, Op, EPI_PRELOADED>(prev_buf * TN, row_group, lane, gm_base, prev_n, side_data, C, residual, cta_rank, staging_saddr, epi_mbar_masked, &tma_c
#if TMA_RESIDUAL
                            #if STAGES_C >= 2
                            , &tma_res, smem_to_uint(smem + OFF_SC_LOAD_MBAR + ew * STAGES_C * 8), staging_saddr + RES_STAGING_OFFSET
#else
                            , &tma_res, smem_to_uint(smem + OFF_RES_MBAR + ew * 8), staging_saddr + RES_STAGING_OFFSET
#endif
#endif
#if STAGES_C >= 2
                            , smem_to_uint(smem + OFF_SC_REL_MBAR)
                            , sc_consumer_phase
#endif
#if W0_RES_FULL || W0_RES_PREFETCH || EPI_LOAD_WARP
                            , smem_to_uint(smem + OFF_RES_CONSUMED_MBAR)
#endif
#if W0_RES_FULL || EPI_LOAD_WARP
                            , smem_to_uint(smem + OFF_RES_PASS_MBAR)
#endif
#ifdef TIMING
                            , epi_t1
#endif
                        );
                    else
                        epilogue_store<TN/2, TN, Op, EPI_PRELOADED>(prev_buf * TN, row_group, lane, gm_base, prev_n, side_data, C, residual, cta_rank, staging_saddr, epi_mbar_masked, &tma_c
#if TMA_RESIDUAL
                            #if STAGES_C >= 2
                            , &tma_res, smem_to_uint(smem + OFF_SC_LOAD_MBAR + ew * STAGES_C * 8), staging_saddr + RES_STAGING_OFFSET
#else
                            , &tma_res, smem_to_uint(smem + OFF_RES_MBAR + ew * 8), staging_saddr + RES_STAGING_OFFSET
#endif
#endif
#if STAGES_C >= 2
                            , smem_to_uint(smem + OFF_SC_REL_MBAR)
                            , sc_consumer_phase
#endif
#if W0_RES_FULL || W0_RES_PREFETCH || EPI_LOAD_WARP
                            , smem_to_uint(smem + OFF_RES_CONSUMED_MBAR)
#endif
#if W0_RES_FULL || EPI_LOAD_WARP
                            , smem_to_uint(smem + OFF_RES_PASS_MBAR)
#endif
#ifdef TIMING
                            , epi_t1
#endif
                        );
                } else
#endif
                {
                    epilogue_store<0, TN, Op, EPI_PRELOADED>(prev_buf * TN, row_group, lane, gm_base, prev_n, side_data, C, residual, cta_rank, staging_saddr, epi_mbar_masked, &tma_c
#if TMA_RESIDUAL
#if EPI_PIPELINE
                        , &tma_res, smem_to_uint(smem + OFF_PIPE_LOAD_MBAR), 0
#elif STAGES_C >= 2
                        , &tma_res, smem_to_uint(smem + OFF_SC_LOAD_MBAR + ew * STAGES_C * 8), staging_saddr + RES_STAGING_OFFSET
#else
                        , &tma_res, smem_to_uint(smem + OFF_RES_MBAR + ew * 8), staging_saddr + RES_STAGING_OFFSET
#endif
#endif
#if STAGES_C >= 2
                        , smem_to_uint(smem + OFF_SC_REL_MBAR)
                        , sc_consumer_phase
#endif
#if EPI_PIPELINE
                        , smem_to_uint(smem + OFF_PIPE_FREE_MBAR)
                        , (tile_idx - tile_start - 1) & 1
#elif W0_RES_FULL || W0_RES_PREFETCH || EPI_LOAD_WARP
                        , smem_to_uint(smem + OFF_RES_CONSUMED_MBAR)
#endif
#if !EPI_PIPELINE && (W0_RES_FULL || EPI_LOAD_WARP)
                        , smem_to_uint(smem + OFF_RES_PASS_MBAR)
#endif
#ifdef TIMING
                        , epi_t1
#endif
                    );
                }
#if REG_PAD
                }
#endif
#if STAGES_C >= 2
                sc_consumer_phase ^= 1;
#endif
#ifdef TIMING
                if (lane == 0 && cta_rank == 0) {
                    long long p1 = epi_t1 - epi_t0;
                    epi_sum_p1 += p1;
                    if (p1 < epi_min_p1) epi_min_p1 = p1;
                    if (p1 > epi_max_p1) epi_max_p1 = p1;
                    epi_count++;
                    int tile_offset = tile_idx - tile_start - 1;
                    spread_buf[cluster_id * (MAX_SPREAD_TILES * NUM_EPI_WARPS) + tile_offset * NUM_EPI_WARPS + ew] = p1;
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
#if EPI_REUSE_SMEM
            /* Signal W0 that staging SMEM (borrowed pipeline stages) is free.
               On tile_start (no epilogue): forward-signal so W0 doesn't block.
               On subsequent tiles: signals after Phase 1 completes. */
            if (lane == 0) mbar_arrive(smem_to_uint(smem + OFF_EPI_DONE_MBAR));
#endif
#endif /* !STRIP_EPILOGUE */
        }
    }  // tile loop

#ifdef TIMING
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
        out[30] = sum_gt_total;
    }
    if (warp >= 2 + EPI_LOAD_WARP && lane == 0 && cta_rank == 0) {
        int ew_out = warp - 2 - EPI_LOAD_WARP;
        long long* out = timing_buf + cluster_id * TIMING_CLUSTER_STRIDE + 8 + ew_out * 4;
        out[0] = epi_sum_p1;
        out[1] = epi_min_p1;
        out[2] = epi_max_p1;
        out[3] = epi_count;
    }
    if (warp == 3 + EPI_LOAD_WARP && lane == 0 && cta_rank == 0) {
        long long* out = timing_buf + cluster_id * TIMING_CLUSTER_STRIDE + 24;
        out[0] = epi_sum_p2;
        out[1] = epi_min_p2;
        out[2] = epi_max_p2;
        out[3] = epi_sum_ml;
        out[4] = epi_min_ml;
        out[5] = epi_max_ml;
    }
#endif

#if W0_RES_FULL && !STRIP_EPILOGUE
    /*
    W0 drain: load residual for the last tile. Runs concurrently with
    the drain epilogue (W2+). Same pass handshake as the tile loop.
    */
    if (warp == 0 && tile_end > tile_start && lane == 0) {
        if constexpr (Op == EpilogueOp::BIAS_RESIDUAL) {
            const uint32_t smem_base = smem_to_uint(smem);
            /* Wait for previous tile's epilogue to finish with residual */
            if (tile_end - tile_start > 1) {
                mbar_wait(smem_base + OFF_RES_CONSUMED_MBAR, res_consumed_phase);
            }
            const int last_idx = tile_end - 1;
            const int ltm = last_idx / TILES_N;
            int ltn = last_idx % TILES_N;
            if (SNAKE_ORDER && (ltm & 1)) ltn = TILES_N - 1 - ltn;
            const int drain_m = ltm * TM * 2 + cta_rank * TM;
            const int drain_n = ltn * TN;

            for (int pass = 0; pass < 2; pass++) {
                if (pass > 0) {
                    mbar_wait(smem_base + OFF_RES_PASS_MBAR, res_pass_phase);
                }
                for (int ew = 0; ew < NUM_EPI_WARPS; ew++) {
                    const int gm = drain_m + ew * 32;
                    const uint32_t rmbar = smem_base + OFF_RES_MBAR + ew * 8;
                    const uint32_t rstg = smem_base + OFF_STAGING
                        + ew * STAGING_WARP_BYTES + RES_STAGING_OFFSET;
#if NUM_PASSES_PARAM == 4
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;\n\t"
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                        " [%2], [%3, {%4, %5}], [%0];"
                        :: "r"(rmbar), "r"(STAGING_REGION_BYTES),
                           "r"(rstg), "l"(&tma_res), "r"(drain_n + pass * 64), "r"(gm)
                        : "memory");
#else
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;\n\t"
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                        " [%2], [%3, {%4, %5}], [%0];\n\t"
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                        " [%6], [%3, {%7, %5}], [%0];"
                        :: "r"(rmbar), "r"(2 * STAGING_REGION_BYTES),
                           "r"(rstg), "l"(&tma_res), "r"(drain_n + pass * 128), "r"(gm),
                           "r"(rstg + STAGING_REGION_BYTES), "r"(drain_n + pass * 128 + 64)
                        : "memory");
#endif
                }
            }
        }
    }
#endif

#if EPI_PIPELINE && !STRIP_EPILOGUE
    /*
    W2 drain: fire-and-forget load for the last tile into pipeline stages.
    */
    if (warp == 2 && tile_end > tile_start && lane == 0) {
        if constexpr (Op == EpilogueOp::BIAS_RESIDUAL) {
            const uint32_t smem_base = smem_to_uint(smem);
            if (tile_end - tile_start > 1) {
                mbar_wait(smem_base + OFF_PIPE_FREE_MBAR, pipe_free_phase);
            }
            const int last_idx = tile_end - 1;
            const int ltm = last_idx / TILES_N;
            int ltn = last_idx % TILES_N;
            if (SNAKE_ORDER && (ltm & 1)) ltn = TILES_N - 1 - ltn;
            const int drain_m = ltm * TM * 2 + cta_rank * TM;
            const int drain_n = ltn * TN;

            for (int si = 0; si < EPI_PIPELINE_STAGES; si++) {
                const uint32_t stage_mbar = smem_base + OFF_PIPE_LOAD_MBAR + si * 8;
                mbar_arrive_expect_tx(stage_mbar, NUM_EPI_WARPS * STAGING_REGION_BYTES);
                for (int ew = 0; ew < NUM_EPI_WARPS; ew++) {
                    const int gm = drain_m + (ew % 4) * 32;
                    const uint32_t rstg = smem_base + OFF_STAGING
                        + si * STAGE_C_BYTES + ew * STAGING_REGION_BYTES;
                    tma_load_2d_cta(rstg, &tma_res, drain_n + si * 64, gm, stage_mbar);
                }
            }
        }
    }
#elif EPI_LOAD_WARP && !STRIP_EPILOGUE
    /*
    W2 drain: load residual for the last tile. Runs concurrently with
    the drain epilogue. Same pass handshake as the tile loop.
    */
    if (warp == 2 && tile_end > tile_start && lane == 0) {
        if constexpr (Op == EpilogueOp::BIAS_RESIDUAL) {
            const uint32_t smem_base = smem_to_uint(smem);
            if (tile_end - tile_start > 1) {
                mbar_wait(smem_base + OFF_RES_CONSUMED_MBAR, res_consumed_phase);
            }
            const int last_idx = tile_end - 1;
            const int ltm = last_idx / TILES_N;
            int ltn = last_idx % TILES_N;
            if (SNAKE_ORDER && (ltm & 1)) ltn = TILES_N - 1 - ltn;
            const int drain_m = ltm * TM * 2 + cta_rank * TM;
            const int drain_n = ltn * TN;

            for (int pass = 0; pass < 2; pass++) {
                if (pass > 0) {
                    mbar_wait(smem_base + OFF_RES_PASS_MBAR, res_pass_phase);
                }
                for (int ew = 0; ew < NUM_EPI_WARPS; ew++) {
                    const int gm = drain_m + (ew % 4) * 32;
                    const uint32_t rmbar = smem_base + OFF_RES_MBAR + ew * 8;
                    const uint32_t rstg = smem_base + OFF_STAGING
                        + ew * STAGING_WARP_BYTES + RES_STAGING_OFFSET;
#if NUM_PASSES_PARAM == 4
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;\n\t"
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                        " [%2], [%3, {%4, %5}], [%0];"
                        :: "r"(rmbar), "r"(STAGING_REGION_BYTES),
                           "r"(rstg), "l"(&tma_res), "r"(drain_n + pass * 64), "r"(gm)
                        : "memory");
#else
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;\n\t"
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                        " [%2], [%3, {%4, %5}], [%0];\n\t"
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                        " [%6], [%3, {%7, %5}], [%0];"
                        :: "r"(rmbar), "r"(2 * STAGING_REGION_BYTES),
                           "r"(rstg), "l"(&tma_res), "r"(drain_n + pass * 128), "r"(gm),
                           "r"(rstg + STAGING_REGION_BYTES), "r"(drain_n + pass * 128 + 64)
                        : "memory");
#endif
                }
            }
        }
    }
#endif

#if !STRIP_EPILOGUE
    // DRAIN (epilogue warps only): epilogue for the last tile
    if (warp >= 2 + EPI_LOAD_WARP) {
        const int ew = warp - 2 - EPI_LOAD_WARP;
        const int row_group = ew % 4;
#if NUM_EPI_WARPS > 4
        const int is_split = (row_group < (NUM_EPI_WARPS - 4)) ? 1 : 0;
        const int col_rank = ew / 4;
#endif
#if EPI_PIPELINE
        const uint32_t staging_saddr = smem_to_uint(smem + OFF_STAGING);
#else
        const uint32_t staging_saddr = smem_to_uint(smem + OFF_STAGING + ew * STAGING_WARP_BYTES);
#endif

        const int last_buf = (tile_end - 1) & 1;

        /* Hoist drain coords above mainloop wait (pure arithmetic) */
        const int last_idx = tile_end - 1;
        const int ltm = last_idx / TILES_N;
        int ltn = last_idx % TILES_N;
        if (SNAKE_ORDER && (ltm & 1)) ltn = TILES_N - 1 - ltn;
        const int last_m = ltm * TM * 2 + cta_rank * TM;
        const int last_n = ltn * TN;
        const int gm_base = last_m + row_group * 32;

#if TMA_RESIDUAL >= 2 && !W0_RES_PREFETCH && !W0_RES_FULL && !EPI_LOAD_WARP
        /* Preload first-pass residual before mainloop wait — TMA flies during idle */
        if constexpr (Op == EpilogueOp::BIAS_RESIDUAL) {
            if (lane == 0) {
                const uint32_t res_mbar = smem_to_uint(smem + OFF_RES_MBAR + ew * 8);
                const uint32_t res_stg = staging_saddr + RES_STAGING_OFFSET;
#if NUM_EPI_WARPS > 4
                const int nc0 = is_split ? (col_rank * (TN/2)) : 0;
#else
                const int nc0 = 0;
#endif
#if NUM_PASSES_PARAM == 4
                mbar_arrive_expect_tx(res_mbar, STAGING_REGION_BYTES);
                tma_load_2d_cta(res_stg, &tma_res, last_n + nc0, gm_base, res_mbar);
#else
                mbar_arrive_expect_tx(res_mbar, 2 * STAGING_REGION_BYTES);
                tma_load_2d_cta(res_stg, &tma_res, last_n + nc0, gm_base, res_mbar);
                tma_load_2d_cta(res_stg + STAGING_REGION_BYTES, &tma_res, last_n + nc0 + 64, gm_base, res_mbar);
#endif
            }
        }
#endif

        mbar_wait(mainloop_mbar_addr + last_buf * 8, ml_phase[last_buf]);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#if EPI_SYNC
        asm volatile("bar.sync 1, %0;" :: "r"(NUM_EPI_WARPS * 32) : "memory");
#endif
        if (STAGGER_CYCLES > 0 && ew > 0 && lane == 0) {
            long long __stagger_end = clock64() + ew * STAGGER_CYCLES;
            while (clock64() < __stagger_end) {}
        }
        __syncwarp();

#ifdef TIMING
        long long drain_t1 = 0;
#endif
#if REG_PAD
        {
            int __rp;
            asm volatile("ld.global.s32 %0, [%1];" : "=r"(__rp) : "l"(&d_regpad_skip));
            if (!__rp) {
#endif
#if NUM_EPI_WARPS > 4
        if (is_split) {
            if (col_rank == 0)
                epilogue_store<0, TN/2, Op, EPI_PRELOADED>(last_buf * TN, row_group, lane, gm_base, last_n, side_data, C, residual, cta_rank, staging_saddr, 0, &tma_c
#if TMA_RESIDUAL
#if STAGES_C >= 2
                    , &tma_res, smem_to_uint(smem + OFF_SC_LOAD_MBAR + ew * STAGES_C * 8), staging_saddr + RES_STAGING_OFFSET
#else
                    , &tma_res, smem_to_uint(smem + OFF_RES_MBAR + ew * 8), staging_saddr + RES_STAGING_OFFSET
#endif
#endif
#if STAGES_C >= 2
                    , smem_to_uint(smem + OFF_SC_REL_MBAR)
                    , sc_consumer_phase
#endif
#if W0_RES_FULL || EPI_LOAD_WARP
                    , 0
                    , smem_to_uint(smem + OFF_RES_PASS_MBAR)
#elif W0_RES_PREFETCH
                    , 0
#endif
#ifdef TIMING
                    , drain_t1
#endif
                );
            else
                epilogue_store<TN/2, TN, Op, EPI_PRELOADED>(last_buf * TN, row_group, lane, gm_base, last_n, side_data, C, residual, cta_rank, staging_saddr, 0, &tma_c
#if TMA_RESIDUAL
#if STAGES_C >= 2
                    , &tma_res, smem_to_uint(smem + OFF_SC_LOAD_MBAR + ew * STAGES_C * 8), staging_saddr + RES_STAGING_OFFSET
#else
                    , &tma_res, smem_to_uint(smem + OFF_RES_MBAR + ew * 8), staging_saddr + RES_STAGING_OFFSET
#endif
#endif
#if STAGES_C >= 2
                    , smem_to_uint(smem + OFF_SC_REL_MBAR)
                    , sc_consumer_phase
#endif
#if W0_RES_FULL || EPI_LOAD_WARP
                    , 0
                    , smem_to_uint(smem + OFF_RES_PASS_MBAR)
#elif W0_RES_PREFETCH
                    , 0
#endif
#ifdef TIMING
                    , drain_t1
#endif
                );
        } else
#endif
        {
            epilogue_store<0, TN, Op, EPI_PRELOADED>(last_buf * TN, row_group, lane, gm_base, last_n, side_data, C, residual, cta_rank, staging_saddr, 0, &tma_c
#if TMA_RESIDUAL
#if EPI_PIPELINE
                , &tma_res, smem_to_uint(smem + OFF_PIPE_LOAD_MBAR), 0
#elif STAGES_C >= 2
                , &tma_res, smem_to_uint(smem + OFF_SC_LOAD_MBAR + ew * STAGES_C * 8), staging_saddr + RES_STAGING_OFFSET
#else
                , &tma_res, smem_to_uint(smem + OFF_RES_MBAR + ew * 8), staging_saddr + RES_STAGING_OFFSET
#endif
#endif
#if STAGES_C >= 2
                , smem_to_uint(smem + OFF_SC_REL_MBAR)
                , sc_consumer_phase
#endif
#if EPI_PIPELINE
                , (uint32_t)0
                , (tile_end - tile_start - 1) & 1
#elif W0_RES_FULL || EPI_LOAD_WARP
                , 0
#if !EPI_PIPELINE
                , smem_to_uint(smem + OFF_RES_PASS_MBAR)
#endif
#elif W0_RES_PREFETCH
                , 0
#endif
#ifdef TIMING
                , drain_t1
#endif
            );
        }
#if REG_PAD
            }
        }
#endif

        if (lane == 0) {
            asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        }
        __syncwarp();
    }
#endif /* !STRIP_EPILOGUE */

    // Cluster sync + TMEM dealloc
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 2) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
            :: "r"(0), "r"(TMEM_COLS));
    }
}

/*
Host timing readback + analysis (shared between all kernels)
*/

#ifdef TIMING
static int cmp_ll(const void* a, const void* b) {
    long long va = *(const long long*)a;
    long long vb = *(const long long*)b;
    return (va > vb) - (va < vb);
}

static void print_timing(long long* d_timing, long long* d_spread, size_t spread_bytes, float _ms) {
    long long h_timing[74 * TIMING_CLUSTER_STRIDE];
    CUDA_CHECK(cudaMemcpy(h_timing, d_timing, sizeof(h_timing), cudaMemcpyDeviceToHost));
    long long* h_spread = (long long*)malloc(spread_bytes);
    CUDA_CHECK(cudaMemcpy(h_spread, d_spread, spread_bytes, cudaMemcpyDeviceToHost));

    // Aggregate W1 data across clusters
    long long g_epi = 0, g_tma0 = 0, g_kloop = 0, g_total = 0;
    long long g_min_kloop = 0x7FFFFFFFFFFFFFFFLL, g_max_kloop = 0;
    long long g_min_total = 0x7FFFFFFFFFFFFFFFLL, g_max_total = 0;
    long long g_gt_total = 0;
    int total_tiles = 0;

    for (int c = 0; c < 74; c++) {
        long long* d = h_timing + c * TIMING_CLUSTER_STRIDE;
        int tiles_this = (int)((long long)(c + 1) * TOTAL_TILES / 74) - (int)((long long)c * TOTAL_TILES / 74);
        g_epi += d[0];  g_tma0 += d[1];  g_kloop += d[2];  g_total += d[3];
        if (d[4] < g_min_kloop) g_min_kloop = d[4];
        if (d[5] > g_max_kloop) g_max_kloop = d[5];
        if (d[6] < g_min_total) g_min_total = d[6];
        if (d[7] > g_max_total) g_max_total = d[7];
        g_gt_total += d[30];
        total_tiles += tiles_this;
    }

    /* Derive actual SM clock from globaltimer (fixed 1 GHz = nanoseconds)
       vs clock64 (SM cycles). Ratio = actual SM frequency. */
    double clock_ghz = (g_gt_total > 0) ? (double)g_total / (double)g_gt_total : 2.1;
    printf("\n=== W1 TIMING (clock64, %d tiles across 74 clusters) ===\n", total_tiles);
    printf("  SM clock (measured): %.3f GHz  (clock64/globaltimer)\n", clock_ghz);
    printf("  Per-tile averages (cycles / ns at %.3f GHz):\n", clock_ghz);
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
    printf("  Predicted wall from tile cycles: %.3f ms  (actual: %.3f ms)\n",
           (double)g_total / (74.0 * clock_ghz * 1e6), _ms);

    // Per-warp Phase 1 data
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
    int total_epi_tiles = gw_count[1];

    // Per-warp p95 and inter-warp spread
    int n_spread_tiles = 0;
    for (int c = 0; c < 74; c++) {
        int ts = (int)((long long)c * TOTAL_TILES / 74);
        int te = (int)((long long)(c + 1) * TOTAL_TILES / 74);
        n_spread_tiles += (te - ts - 1);
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

    long long gw_p95[NUM_EPI_WARPS];
    for (int w = 0; w < NUM_EPI_WARPS; w++) {
        qsort(warp_p1_all[w], n_spread_tiles, sizeof(long long), cmp_ll);
        gw_p95[w] = warp_p1_all[w][(int)(n_spread_tiles * 0.95)];
    }
    qsort(tile_spreads, n_spread_tiles, sizeof(long long), cmp_ll);
    long long p95_spread = tile_spreads[(int)(n_spread_tiles * 0.95)];

    printf("\n=== EPILOGUE PER-WARP PHASE 1 TIMING (W2-W5, %d tiles across 74 clusters) ===\n", n_spread_tiles);
    printf("  Per-warp Phase 1 (cycles):\n");
    for (int w = 0; w < NUM_EPI_WARPS; w++) {
        long long avg = gw_count[w] > 0 ? gw_sum_p1[w] / gw_count[w] : 0;
        printf("    W%d (ew=%d, rg=%d):  avg=%lld  min=%lld  max=%lld  p95=%lld\n",
               w + 2, w, w, avg, gw_min_p1[w], gw_max_p1[w], gw_p95[w]);
    }
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

    // Backward-compat: W3/ew=1 full phase timing
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

    /* Per-tile W1 total cycle histogram */
    printf("\n=== W1 PER-TILE TOTAL CYCLE DISTRIBUTION (%d tiles across 74 clusters) ===\n", total_tiles);
    {
        long long* all_totals = (long long*)malloc(total_tiles * sizeof(long long));
        int aidx = 0;
        for (int c = 0; c < 74; c++) {
            long long* d = h_timing + c * TIMING_CLUSTER_STRIDE;
            int tiles_c = (int)((long long)(c + 1) * TOTAL_TILES / 74) - (int)((long long)c * TOTAL_TILES / 74);
            for (int t = 0; t < tiles_c && aidx < total_tiles; t++)
                all_totals[aidx++] = d[32 + t];
        }
        qsort(all_totals, aidx, sizeof(long long), cmp_ll);
        if (aidx > 0) {
            long long p1v = all_totals[(int)(aidx * 0.01)];
            long long p5v = all_totals[(int)(aidx * 0.05)];
            long long p25v = all_totals[(int)(aidx * 0.25)];
            long long p50v = all_totals[aidx / 2];
            long long p75v = all_totals[(int)(aidx * 0.75)];
            long long p95v = all_totals[(int)(aidx * 0.95)];
            long long p99v = all_totals[(int)(aidx * 0.99)];
            printf("  Percentiles (cycles):  p1=%lld  p5=%lld  p25=%lld  p50=%lld  p75=%lld  p95=%lld  p99=%lld\n",
                   p1v, p5v, p25v, p50v, p75v, p95v, p99v);
            printf("  Range: min=%lld  max=%lld  (%.2fx spread)\n",
                   all_totals[0], all_totals[aidx - 1],
                   all_totals[0] > 0 ? (double)all_totals[aidx - 1] / all_totals[0] : 0.0);
            /* 10-bin histogram */
            long long bin_lo = all_totals[0];
            long long bin_hi = all_totals[aidx - 1];
            if (bin_hi > bin_lo) {
                const int NBINS = 10;
                int bins[10] = {0};
                double bin_width = (double)(bin_hi - bin_lo) / NBINS;
                for (int i = 0; i < aidx; i++) {
                    int b = (int)((all_totals[i] - bin_lo) / bin_width);
                    if (b >= NBINS) b = NBINS - 1;
                    bins[b]++;
                }
                printf("  Histogram (%lld - %lld, bin=%.0f cyc):\n", bin_lo, bin_hi, bin_width);
                for (int b = 0; b < NBINS; b++) {
                    long long lo = bin_lo + (long long)(b * bin_width);
                    long long hi = bin_lo + (long long)((b + 1) * bin_width);
                    int bar_len = bins[b] * 40 / (aidx > 0 ? aidx : 1);
                    if (bins[b] > 0 && bar_len == 0) bar_len = 1;
                    printf("    [%6lld-%6lld] %5d ", lo, hi, bins[b]);
                    for (int x = 0; x < bar_len; x++) printf("#");
                    printf("\n");
                }
            }
        }
        free(all_totals);
    }

    for (int w = 0; w < NUM_EPI_WARPS; w++) free(warp_p1_all[w]);
    free(tile_spreads);
    free(h_spread);
    cudaFree(d_timing);
    cudaFree(d_spread);
}
#endif
