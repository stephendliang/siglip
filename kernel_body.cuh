// kernel_body.cuh — shared kernel body for tcgen05 persistent megakernels
// Contains epilogue_store template and persistent_gemm kernel template.
// Each .cu file: #define N_DIM → #include "kernel_common.cuh" → define transform macros → #include "kernel_body.cuh"

#pragma once

// ── Epilogue operation selector ──────────────────────────────
enum class EpilogueOp : int { BIAS_ADD = 0, BIAS_GELU = 1 };

template<EpilogueOp Op> struct EpilogueSideData;
template<> struct EpilogueSideData<EpilogueOp::BIAS_ADD>  { using type = const __nv_bfloat16*; };
template<> struct EpilogueSideData<EpilogueOp::BIAS_GELU> { using type = const float*; };
template<EpilogueOp Op> using SideDataPtr = typename EpilogueSideData<Op>::type;

// ── Stub macros for dead if-constexpr branches ───────────────
// Preprocessor expands before template instantiation; stubs provide syntactically valid no-ops.
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
// Stub COMB_* constants for dead BIAS_ADD branch in fc1_gelu compilation
#ifndef COMB_BLOCK_ROWS
#define COMB_BLOCK_ROWS 1
#define COMB_BLOCK_COLS 1
#define COMB_COL_BLOCKS 1
#define COMB_BLOCK_ELEMS 1
#endif

// ── Epilogue: TMEM → transform → CVT → swizzle SMEM regions → TMA tensor stores ──
// Transform selected by Op: BIAS_ADD (combined table add) or BIAS_GELU (bias + GELU)

template<int NC_START, int NC_END, EpilogueOp Op>
static __device__ __forceinline__
void epilogue_store(
    uint32_t tmem_addr,
    int row_group,
    int lane,
    int gm_base,
    int n_start,
    SideDataPtr<Op> __restrict__ side_data,
    __nv_bfloat16* __restrict__ C,
    int cta_rank,
    uint32_t staging_saddr,
    uint32_t epi_mbar_addr,
    const CUtensorMap* tma_c_desc
#ifdef TIMING
    , long long& t_phase1_end
#endif
) {
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

    constexpr int N_REGIONS = (NC_END - NC_START) / 64;

    // Wait for previous tile's Phase 2 TMA stores before overwriting staging.
    if (lane == 0) {
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
    __syncwarp();

    // Swizzle constants (loop-invariant)
    const uint32_t xor_val = (lane & 7) << 4;
    const uint32_t srow_base = staging_saddr + lane * STAGING_REGION_ROW_BYTES;

#if TMEM_LOAD_WIDTH == 64
    float a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15;
    float a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31;
    float a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44,a45,a46,a47;
    float a48,a49,a50,a51,a52,a53,a54,a55,a56,a57,a58,a59,a60,a61,a62,a63;

    // ═══ Phase 1: all cols → swizzle regions, x64 stride ═══
    TMEM_LOAD_X64(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                  a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                  a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44,a45,a46,a47,
                  a48,a49,a50,a51,a52,a53,a54,a55,a56,a57,a58,a59,a60,a61,a62,a63,
                  taddr_base + NC_START);

    PRAGMA_UNROLL(PHASE1_UNROLL)
    for (int nc = NC_START; nc < NC_END; nc += 64) {
        const uint32_t srow = srow_base + ((nc - NC_START) >> 6) * STAGING_REGION_BYTES;

        // Preload side data (fills TMEM latency window)
        uint4 craw0 = {}, craw1 = {};
        const __nv_bfloat16* comb_ptr = nullptr;
        float4 bv0 = {}, bv1 = {};
        if constexpr (Op == EpilogueOp::BIAS_ADD) {
            static_assert(HAS_CVT_ADD, "BIAS_ADD requires CVT_ADD_STS_V4 macro — define before #include \"kernel_body.cuh\"");
            comb_ptr = comb_base + (long long)((n_start + nc) / COMB_BLOCK_COLS) * COMB_BLOCK_ELEMS;
            craw0 = *reinterpret_cast<const uint4*>(comb_ptr);
            craw1 = *reinterpret_cast<const uint4*>(comb_ptr + 8);
        } else if constexpr (Op == EpilogueOp::BIAS_GELU) {
            static_assert(HAS_GELU_CVT, "BIAS_GELU requires GELU_CVT_STS_V4 macro — define before #include \"kernel_body.cuh\"");
            const float* bp = side_data + n_start + nc;
            bv0 = __ldg(reinterpret_cast<const float4*>(bp));
            bv1 = __ldg(reinterpret_cast<const float4*>(bp + 4));
        }

        TMEM_WAIT();

        if (MBAR_EARLY && nc + 64 >= NC_END) {
            if (epi_mbar_addr) mbar_arrive(epi_mbar_addr);
        }

        // Transform: accumulator → (op-specific) → BF16 → SMEM
        if constexpr (Op == EpilogueOp::BIAS_ADD) {
            // First 32 cols: a0..a31
            {
                CVT_ADD_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7, craw0.x,craw0.y,craw0.z,craw0.w, srow + (0 ^ xor_val));
                CVT_ADD_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15, craw1.x,craw1.y,craw1.z,craw1.w, srow + (16 ^ xor_val));
                craw0 = *reinterpret_cast<const uint4*>(comb_ptr + 16);
                craw1 = *reinterpret_cast<const uint4*>(comb_ptr + 24);
                CVT_ADD_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23, craw0.x,craw0.y,craw0.z,craw0.w, srow + (32 ^ xor_val));
                CVT_ADD_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31, craw1.x,craw1.y,craw1.z,craw1.w, srow + (48 ^ xor_val));
            }
            // Second 32 cols: a32..a63
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
        } else if constexpr (Op == EpilogueOp::BIAS_GELU) {
            const float* bp = side_data + n_start + nc;
            // First 32 cols: a0..a31
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
            // Second 32 cols: a32..a63
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
        }

        // Interleaved TMA store(s) — fence.proxy.async bridges sync→async proxy
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

        if (nc + 64 < NC_END) {
            TMEM_LOAD_X64(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                          a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                          a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44,a45,a46,a47,
                          a48,a49,a50,a51,a52,a53,a54,a55,a56,a57,a58,a59,a60,a61,a62,a63,
                          taddr_base + nc + 64);
        }
    }
#else  // TMEM_LOAD_WIDTH 16 or 32
    float a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15;
    float a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31;

    // ═══ Phase 1: all cols → swizzle regions, x32 stride ═══
    LOAD_32_COLS(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                 a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                 taddr_base + NC_START);

    PRAGMA_UNROLL(PHASE1_UNROLL)
    for (int nc = NC_START; nc < NC_END; nc += 32) {
        // Preload side data (fills TMEM latency window)
        uint4 craw0 = {}, craw1 = {};
        const __nv_bfloat16* comb_ptr = nullptr;
        const float* bp = nullptr;
        float4 bv0 = {}, bv1 = {};
        if constexpr (Op == EpilogueOp::BIAS_ADD) {
            static_assert(HAS_CVT_ADD, "BIAS_ADD requires CVT_ADD_STS_V4 macro — define before #include \"kernel_body.cuh\"");
            comb_ptr = comb_base + (long long)((n_start + nc) / COMB_BLOCK_COLS) * COMB_BLOCK_ELEMS;
            craw0 = *reinterpret_cast<const uint4*>(comb_ptr);
            craw1 = *reinterpret_cast<const uint4*>(comb_ptr + 8);
        } else if constexpr (Op == EpilogueOp::BIAS_GELU) {
            static_assert(HAS_GELU_CVT, "BIAS_GELU requires GELU_CVT_STS_V4 macro — define before #include \"kernel_body.cuh\"");
            bp = side_data + n_start + nc;
            bv0 = __ldg(reinterpret_cast<const float4*>(bp));
            bv1 = __ldg(reinterpret_cast<const float4*>(bp + 4));
        }

        TMEM_WAIT();

        if (MBAR_EARLY && nc + 32 >= NC_END) {
            if (epi_mbar_addr) mbar_arrive(epi_mbar_addr);
        }

        const uint32_t srow = srow_base + ((nc - NC_START) >> 6) * STAGING_REGION_BYTES;
        const int byte_base = ((nc - NC_START) & 63) * 2;

        // Transform: accumulator → (op-specific) → BF16 → SMEM
        if constexpr (Op == EpilogueOp::BIAS_ADD) {
            CVT_ADD_STS_V4(a0,a1,a2,a3,a4,a5,a6,a7, craw0.x,craw0.y,craw0.z,craw0.w, srow + (byte_base ^ xor_val));
            CVT_ADD_STS_V4(a8,a9,a10,a11,a12,a13,a14,a15, craw1.x,craw1.y,craw1.z,craw1.w, srow + ((byte_base + 16) ^ xor_val));

            craw0 = *reinterpret_cast<const uint4*>(comb_ptr + 16);
            craw1 = *reinterpret_cast<const uint4*>(comb_ptr + 24);
            CVT_ADD_STS_V4(a16,a17,a18,a19,a20,a21,a22,a23, craw0.x,craw0.y,craw0.z,craw0.w, srow + ((byte_base + 32) ^ xor_val));
            CVT_ADD_STS_V4(a24,a25,a26,a27,a28,a29,a30,a31, craw1.x,craw1.y,craw1.z,craw1.w, srow + ((byte_base + 48) ^ xor_val));
        } else if constexpr (Op == EpilogueOp::BIAS_GELU) {
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
        }

        // Interleaved TMA store(s) — region completes every 2 x32 iterations
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

        if (nc + 32 < NC_END) {
            LOAD_32_COLS(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                         a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                         taddr_base + nc + 32);
        }
    }
#endif

#ifdef TIMING
    t_phase1_end = clock64();
#endif

#if INTERLEAVE_STRATEGY == 0
    // Strategy 0 (all-at-end): all TMA stores in Phase 2
    __syncwarp();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    if (!MBAR_EARLY && epi_mbar_addr) mbar_arrive(epi_mbar_addr);

    if (lane == 0) {
        int row = gm_base;
        #pragma unroll
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

// ═════════════════════════════════════════════════════════════
// Persistent GEMM — warp-specialized tcgen05 (cta_group::2)
// ═════════════════════════════════════════════════════════════

template<EpilogueOp Op>
__global__ void __launch_bounds__(THREADS, 1)
__cluster_dims__(2, 1, 1)
persistent_gemm(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    const __grid_constant__ CUtensorMap tma_c,
    SideDataPtr<Op> __restrict__ side_data,
    __nv_bfloat16* __restrict__ C
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

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    const int cluster_id = sm_id / 2;
    const int num_clusters = SM_COUNT / 2;

    // ── Mbarrier init ──
    if (tid == 0) {
        for (int s = 0; s < N_STAGES; s++) {
            mbar_init(smem_to_uint(smem + OFF_TMA_MBAR + s * 8), 2);
            mbar_init(smem_to_uint(smem + OFF_MMA_MBAR + s * 8), 1);
        }
        for (int i = 0; i < 2; i++) {
            mbar_init(smem_to_uint(smem + OFF_MAINLOOP_MBAR + i * 8), 1);
            mbar_init(smem_to_uint(smem + OFF_EPILOGUE_MBAR + i * 8), NUM_EPI_WARPS * 2 * 32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    // ── TMEM alloc ──
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

    uint64_t desc_a_base[N_STAGES], desc_b_base[N_STAGES];
    for (int s = 0; s < N_STAGES; s++) {
        desc_a_base[s] = make_smem_desc(smem_a[s]);
        desc_b_base[s] = make_smem_desc(smem_b[s]);
    }

    const int start_buf = tile_start & 1;
    int epi_phase[2] = {1, 1};
    int ml_phase[2]  = {start_buf, 1 - start_buf};

#ifdef TIMING
    long long t_tile_start = 0, t_after_epi = 0, t_after_tma0 = 0, t_kloop_end = 0;
    long long sum_epi_wait = 0, sum_tma0_wait = 0, sum_kloop = 0, sum_total = 0;
    long long min_kloop = 0x7FFFFFFFFFFFFFFFLL, max_kloop = 0;
    long long min_total = 0x7FFFFFFFFFFFFFFFLL, max_total = 0;
    int tile_count = 0;
    long long epi_t0 = 0, epi_t1 = 0;
    long long epi_sum_p1 = 0;
    long long epi_min_p1 = 0x7FFFFFFFFFFFFFFFLL, epi_max_p1 = 0;
    int epi_count = 0;
    long long epi_t_before_ml = 0, epi_t2 = 0;
    long long epi_sum_p2 = 0, epi_sum_ml = 0;
    long long epi_min_p2 = 0x7FFFFFFFFFFFFFFFLL, epi_max_p2 = 0;
    long long epi_min_ml = 0x7FFFFFFFFFFFFFFFLL, epi_max_ml = 0;
#endif

    for (int tile_idx = tile_start; tile_idx < tile_end; tile_idx++) {
        const int buf = tile_idx & 1;
        const int tm = tile_idx / TILES_N;
        int tn = tile_idx % TILES_N;
        if (SNAKE_ORDER && (tm & 1)) tn = TILES_N - 1 - tn;
        const int m_start = tm * TM * 2 + cta_rank * TM;
        const int n_start = tn * TN;

        if (warp == 0) {
            // ── LOAD WARP (W0) ──
            if (lane == 0) {
                for (int ki = 0; ki < K_ITERS; ki++) {
                    const int s = ki % N_STAGES;
                    const int k_start = ki * TK;

                    if (tile_idx > tile_start || ki >= N_STAGES) {
                        mbar_wait(mma_mbar[s], mma_phase[s]);
                        mma_phase[s] ^= 1;
                    }

                    const uint32_t tma_mbar_masked = tma_mbar[s] & 0xFEFFFFFF;
                    tma_load_2d(smem_a[s], &tma_a, k_start, m_start, tma_mbar_masked);
                    tma_load_2d(smem_b[s], &tma_b, k_start, n_start + cta_rank * (TN/2), tma_mbar_masked);
                    mbar_arrive_expect_tx(tma_mbar_masked, TMA_BYTES);
                }
            }
        } else if (warp == 1) {
            // ── MMA WARP (W1) ──
            if (lane == 0 && cta_rank == 0) {
#ifdef TIMING
                t_tile_start = clock64();
#endif
                mbar_wait(epilogue_mbar_addr + buf * 8, epi_phase[buf]);
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
                #pragma unroll
                for (int ki = 1; ki < K_ITERS; ki++) {
                    K_ITER_ACCUM(ki % N_STAGES);
                }

                tcgen05_commit_mcast(mainloop_mbar_addr + buf * 8, 0x3);
#ifdef TIMING
                t_kloop_end = clock64();
                long long dt_epi = t_after_epi - t_tile_start;
                long long dt_tma0 = t_after_tma0 - t_after_epi;
                long long dt_kloop = t_kloop_end - t_after_tma0;
                long long dt_total = t_kloop_end - t_tile_start;
                sum_epi_wait += dt_epi;
                sum_tma0_wait += dt_tma0;
                sum_kloop += dt_kloop;
                sum_total += dt_total;
                if (dt_kloop < min_kloop) min_kloop = dt_kloop;
                if (dt_kloop > max_kloop) max_kloop = dt_kloop;
                if (dt_total < min_total) min_total = dt_total;
                if (dt_total > max_total) max_total = dt_total;
                tile_count++;
                t_tile_start = clock64();
#endif
            }
        } else {
            // ── OVERLAPPED EPILOGUE (W2+) ──
            const int ew = warp - 2;
            const int row_group = ew % 4;
            const int is_split = (row_group < (NUM_EPI_WARPS - 4)) ? 1 : 0;
            const int col_rank = ew / 4;
            const uint32_t staging_saddr = smem_to_uint(smem + OFF_STAGING + ew * STAGING_WARP_BYTES);

            const int prev_buf = buf ^ 1;

#ifdef TIMING
            if (ew == 1 && lane == 0 && cta_rank == 0)
                epi_t_before_ml = clock64();
#endif
            mbar_wait(mainloop_mbar_addr + prev_buf * 8, ml_phase[prev_buf]);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
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

            if (tile_idx > tile_start) {
                const int prev_idx = tile_idx - 1;
                const int ptm = prev_idx / TILES_N;
                int ptn = prev_idx % TILES_N;
                if (SNAKE_ORDER && (ptm & 1)) ptn = TILES_N - 1 - ptn;
                const int prev_m = ptm * TM * 2 + cta_rank * TM;
                const int prev_n = ptn * TN;

                const int gm_base = prev_m + row_group * 32;
                const uint32_t epi_mbar_masked = (epilogue_mbar_addr + prev_buf * 8) & 0xFEFFFFFF;
                if (is_split) {
                    if (col_rank == 0)
                        epilogue_store<0, TN/2, Op>(prev_buf * TN, row_group, lane, gm_base, prev_n, side_data, C, cta_rank, staging_saddr, epi_mbar_masked, &tma_c
#ifdef TIMING
                            , epi_t1
#endif
                        );
                    else
                        epilogue_store<TN/2, TN, Op>(prev_buf * TN, row_group, lane, gm_base, prev_n, side_data, C, cta_rank, staging_saddr, epi_mbar_masked, &tma_c
#ifdef TIMING
                            , epi_t1
#endif
                        );
                } else {
                    epilogue_store<0, TN, Op>(prev_buf * TN, row_group, lane, gm_base, prev_n, side_data, C, cta_rank, staging_saddr, epi_mbar_masked, &tma_c
#ifdef TIMING
                        , epi_t1
#endif
                    );
                }
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
    }
    if (warp >= 2 && lane == 0 && cta_rank == 0) {
        int ew_out = warp - 2;
        long long* out = timing_buf + cluster_id * TIMING_CLUSTER_STRIDE + 8 + ew_out * 4;
        out[0] = epi_sum_p1;
        out[1] = epi_min_p1;
        out[2] = epi_max_p1;
        out[3] = epi_count;
    }
    if (warp == 3 && lane == 0 && cta_rank == 0) {
        long long* out = timing_buf + cluster_id * TIMING_CLUSTER_STRIDE + 24;
        out[0] = epi_sum_p2;
        out[1] = epi_min_p2;
        out[2] = epi_max_p2;
        out[3] = epi_sum_ml;
        out[4] = epi_min_ml;
        out[5] = epi_max_ml;
    }
#endif

    // ── DRAIN (W2+ only): epilogue for the last tile ──
    if (warp >= 2) {
        const int ew = warp - 2;
        const int row_group = ew % 4;
        const int is_split = (row_group < (NUM_EPI_WARPS - 4)) ? 1 : 0;
        const int col_rank = ew / 4;
        const uint32_t staging_saddr = smem_to_uint(smem + OFF_STAGING + ew * STAGING_WARP_BYTES);

        const int last_buf = (tile_end - 1) & 1;

        mbar_wait(mainloop_mbar_addr + last_buf * 8, ml_phase[last_buf]);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        if (STAGGER_CYCLES > 0 && ew > 0 && lane == 0) {
            long long __stagger_end = clock64() + ew * STAGGER_CYCLES;
            while (clock64() < __stagger_end) {}
        }
        __syncwarp();

        const int last_idx = tile_end - 1;
        const int ltm = last_idx / TILES_N;
        int ltn = last_idx % TILES_N;
        if (SNAKE_ORDER && (ltm & 1)) ltn = TILES_N - 1 - ltn;
        const int last_m = ltm * TM * 2 + cta_rank * TM;
        const int last_n = ltn * TN;
        const int gm_base = last_m + row_group * 32;
#ifdef TIMING
        long long drain_t1 = 0;
#endif
        if (is_split) {
            if (col_rank == 0)
                epilogue_store<0, TN/2, Op>(last_buf * TN, row_group, lane, gm_base, last_n, side_data, C, cta_rank, staging_saddr, 0, &tma_c
#ifdef TIMING
                    , drain_t1
#endif
                );
            else
                epilogue_store<TN/2, TN, Op>(last_buf * TN, row_group, lane, gm_base, last_n, side_data, C, cta_rank, staging_saddr, 0, &tma_c
#ifdef TIMING
                    , drain_t1
#endif
                );
        } else {
            epilogue_store<0, TN, Op>(last_buf * TN, row_group, lane, gm_base, last_n, side_data, C, cta_rank, staging_saddr, 0, &tma_c
#ifdef TIMING
                , drain_t1
#endif
            );
        }

        if (lane == 0) {
            asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        }
        __syncwarp();
    }

    // ── Cluster sync + TMEM dealloc ──
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 2) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
            :: "r"(0), "r"(TMEM_COLS));
    }
}

// ═════════════════════════════════════════════════════════════
// Host timing readback + analysis (shared between all kernels)
// ═════════════════════════════════════════════════════════════

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

    // ── Aggregate W1 data across clusters ──
    long long g_epi = 0, g_tma0 = 0, g_kloop = 0, g_total = 0;
    long long g_min_kloop = 0x7FFFFFFFFFFFFFFFLL, g_max_kloop = 0;
    long long g_min_total = 0x7FFFFFFFFFFFFFFFLL, g_max_total = 0;
    int total_tiles = 0;

    for (int c = 0; c < 74; c++) {
        long long* d = h_timing + c * TIMING_CLUSTER_STRIDE;
        int tiles_this = (int)((long long)(c + 1) * TOTAL_TILES / 74) - (int)((long long)c * TOTAL_TILES / 74);
        g_epi += d[0];  g_tma0 += d[1];  g_kloop += d[2];  g_total += d[3];
        if (d[4] < g_min_kloop) g_min_kloop = d[4];
        if (d[5] > g_max_kloop) g_max_kloop = d[5];
        if (d[6] < g_min_total) g_min_total = d[6];
        if (d[7] > g_max_total) g_max_total = d[7];
        total_tiles += tiles_this;
    }

    double clock_ghz = 2.1;
    printf("\n=== W1 TIMING (clock64, %d tiles across 74 clusters) ===\n", total_tiles);
    printf("  Per-tile averages (cycles / ns at %.1f GHz):\n", clock_ghz);
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
    printf("  Expected total cycles (wall clock): %.0f\n", _ms * 1e-3 * clock_ghz * 1e9);

    // ── Per-warp Phase 1 data ──
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

    // ── Per-warp p95 and inter-warp spread ──
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

    // ── Backward-compat: W3/ew=1 full phase timing ──
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

    for (int w = 0; w < NUM_EPI_WARPS; w++) free(warp_p1_all[w]);
    free(tile_spreads);
    free(h_spread);
    cudaFree(d_timing);
    cudaFree(d_spread);
}
#endif
