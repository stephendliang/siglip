// Hand-tuned from gen.py output — TK=128, SWIZZLE_128B, 6-stage pipeline
// Target: B200  Batch: 4736  GEMM: [928256,768]×[768,768]^T
// Pipeline: 6-stage  K-iters: 6  MMA/iter: 4  idesc: 0x10400010
// Warps: 2+NUM_EPI_WARPS  cta_group::2  __cluster_dims__(2,1,1)
// Warp-specialized: Load(W0) | MMA(W1,cta_group::2,CTA0 only) | Epilogue(W2+,x32 TMEM ld,col-split)  BF16 output
// tcgen05.mma.cta_group::2.kind::f8f6f4  (E4M3 × E4M3 → FP32)
// Each CTA loads own A (128 rows) + half B (128 cols). MMA produces 256×256 output.

#include <cuda.h>
#include <cuda_bf16.h>
#include <curand.h>
#include <cstdint>
#include <cstdio>

#define SM_COUNT       148
#define NUM_EPI_WARPS  5
#define THREADS        (32 * (2 + NUM_EPI_WARPS))
#define BATCH_SIZE     4736
#define SEQ_LEN        196
#define M_TOTAL        928256
#define N_DIM          768
#define K_DIM          768
#define TM             128
#define TN             256
#define TK             128
#define TILES_M        3626
#define TILES_N        3
#define K_ITERS        6
#define TOTAL_TILES    10878
#define N_STAGES       6
#define OFF_TMEM           196608
#define OFF_TMA_MBAR       196616
#define OFF_MMA_MBAR       196664
#define OFF_MAINLOOP_MBAR  196712
#define OFF_EPILOGUE_MBAR  196728
#define SMEM_BYTES         196864
#define TMEM_COLS      512
#define IDESC          0x10400010U
#define SBO            1024
#define TMA_BYTES      32768
#define MMA_K          32
#define MMA_PER_KI     4

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

// ── Device helpers ──────────────────────────────────────────

static __device__ __forceinline__
uint32_t smem_to_uint(const void* p) {
    uint32_t r;
    asm volatile("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }"
        : "=r"(r) : "l"(p));
    return r;
}

static __device__ __forceinline__
uint64_t make_smem_desc(uint32_t addr) {
    uint64_t d = 0;
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);            // bits [13:0]  base addr
    d |= (uint64_t)((SBO  & 0x3FFFF) >> 4) << 32;      // bits [45:32] SBO (stride_byte_offset)
    d |= (1ULL << 46);                                  // bit  [46]    LBO
    d |= (2ULL << 61);                                  // bits [63:61] SWIZZLE_128B
    return d;
}

static __device__ __forceinline__
void mbar_init(uint32_t addr, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(addr), "r"(count));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
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
void tcgen05_commit_mcast(uint32_t mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        :: "r"(mbar_addr), "h"(cta_mask) : "memory");
}

// ── Pipelined epilogue macros ────────────────────────────────

#define TMEM_LOAD(r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15, TADDR) \
    asm volatile( \
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 " \
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];" \
        : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3), \
          "=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7), \
          "=f"(r8),"=f"(r9),"=f"(r10),"=f"(r11), \
          "=f"(r12),"=f"(r13),"=f"(r14),"=f"(r15) \
        : "r"(TADDR))

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

#define CVT_STG(r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15, GPTR) \
    asm volatile( \
        "{\n\t" \
        ".reg .b32 b0,b1,b2,b3,b4,b5,b6,b7;\n\t" \
        "cvt.rn.bf16x2.f32 b0, %1, %0;\n\t" \
        "cvt.rn.bf16x2.f32 b1, %3, %2;\n\t" \
        "cvt.rn.bf16x2.f32 b2, %5, %4;\n\t" \
        "cvt.rn.bf16x2.f32 b3, %7, %6;\n\t" \
        "cvt.rn.bf16x2.f32 b4, %9, %8;\n\t" \
        "cvt.rn.bf16x2.f32 b5, %11, %10;\n\t" \
        "cvt.rn.bf16x2.f32 b6, %13, %12;\n\t" \
        "cvt.rn.bf16x2.f32 b7, %15, %14;\n\t" \
        "st.global.v8.b32 [%16], {b0,b1,b2,b3,b4,b5,b6,b7};\n\t" \
        "}" \
        :: "f"(r0),"f"(r1),"f"(r2),"f"(r3), \
           "f"(r4),"f"(r5),"f"(r6),"f"(r7), \
           "f"(r8),"f"(r9),"f"(r10),"f"(r11), \
           "f"(r12),"f"(r13),"f"(r14),"f"(r15), \
           "l"(GPTR) \
        : "memory")

#define BF16X2_TO_F32(reg, flo, fhi) \
    asm volatile("{\n\t" \
        ".reg .b16 lo, hi;\n\t" \
        "mov.b32 {lo, hi}, %2;\n\t" \
        "cvt.rn.f32.bf16 %0, lo;\n\t" \
        "cvt.rn.f32.bf16 %1, hi;\n\t" \
        "}" : "=f"(flo), "=f"(fhi) : "r"(reg))

// ── Unified epilogue: x32 TMEM readback → inline BF16 combined add → FP32→BF16 → st.global.v8 ──
// Single-buffer x32 TMEM loads (32 cols/load). All 4 BF16 combined uint4 loads
// issued before TMEM_WAIT to overlap with TMEM latency. No A/B double-buffering needed.

template<int NC_START, int NC_END>
static __device__ __forceinline__
void epilogue_store(
    uint32_t tmem_addr,
    int row_group,
    int lane,
    int gm_base,
    int n_start,
    const __nv_bfloat16* __restrict__ combined,
    int pos_row,
    __nv_bfloat16* __restrict__ C,
    int cta_rank
) {
    __nv_bfloat16* row_out = C + (long long)(gm_base + lane) * N_DIM + n_start;
    const int taddr_base = tmem_addr + ((cta_rank * 128 + row_group * 32) << 16);
    const __nv_bfloat16* comb_row = combined + (long long)pos_row * N_DIM + n_start;

    // Single buffer: 32 FP32 registers for x32 TMEM load
    float a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15;
    float a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31;

    // Prefetch first 32-column chunk
    TMEM_LOAD_X32(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                  a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                  taddr_base + NC_START);

    for (int nc = NC_START; nc < NC_END; nc += 32) {
        // ── Load ALL BF16 combined before TMEM_WAIT (overlap with TMEM latency) ──
        // First 16 BF16 values (cols nc..nc+15)
        uint4 craw0 = *reinterpret_cast<const uint4*>(comb_row + nc);
        uint4 craw1 = *reinterpret_cast<const uint4*>(comb_row + nc + 8);
        float s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15;
        BF16X2_TO_F32(craw0.x, s0, s1);
        BF16X2_TO_F32(craw0.y, s2, s3);
        BF16X2_TO_F32(craw0.z, s4, s5);
        BF16X2_TO_F32(craw0.w, s6, s7);
        BF16X2_TO_F32(craw1.x, s8, s9);
        BF16X2_TO_F32(craw1.y, s10, s11);
        BF16X2_TO_F32(craw1.z, s12, s13);
        BF16X2_TO_F32(craw1.w, s14, s15);

        // Second 16 BF16 values (cols nc+16..nc+31) — loaded but NOT converted yet
        craw0 = *reinterpret_cast<const uint4*>(comb_row + nc + 16);
        craw1 = *reinterpret_cast<const uint4*>(comb_row + nc + 24);

        // ── Wait for TMEM data ──
        TMEM_WAIT();

        // ── Process first 16 columns ──
        a0+=s0; a1+=s1; a2+=s2; a3+=s3;
        a4+=s4; a5+=s5; a6+=s6; a7+=s7;
        a8+=s8; a9+=s9; a10+=s10; a11+=s11;
        a12+=s12; a13+=s13; a14+=s14; a15+=s15;

        CVT_STG(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15, row_out + nc);

        // ── Convert second 16 BF16 (data already in craw regs from loads above) ──
        BF16X2_TO_F32(craw0.x, s0, s1);
        BF16X2_TO_F32(craw0.y, s2, s3);
        BF16X2_TO_F32(craw0.z, s4, s5);
        BF16X2_TO_F32(craw0.w, s6, s7);
        BF16X2_TO_F32(craw1.x, s8, s9);
        BF16X2_TO_F32(craw1.y, s10, s11);
        BF16X2_TO_F32(craw1.z, s12, s13);
        BF16X2_TO_F32(craw1.w, s14, s15);

        // ── Process second 16 columns ──
        a16+=s0; a17+=s1; a18+=s2; a19+=s3;
        a20+=s4; a21+=s5; a22+=s6; a23+=s7;
        a24+=s8; a25+=s9; a26+=s10; a27+=s11;
        a28+=s12; a29+=s13; a30+=s14; a31+=s15;

        CVT_STG(a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31, row_out + nc + 16);

        // ── Prefetch next 32-column chunk ──
        if (nc + 32 < NC_END)
            TMEM_LOAD_X32(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
                          a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,
                          taddr_base + nc + 32);
    }
}

// ── Host precompute kernel: bias[c] + pos_embed[r,c] → BF16 combined[r,c] ──

__global__ void precompute_combined(
    const float* __restrict__ bias,
    const float* __restrict__ pos_embed,
    __nv_bfloat16* __restrict__ combined,
    int seq_len, int n_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len * n_dim) {
        int c = idx % n_dim;
        combined[idx] = __float2bfloat16(bias[c] + pos_embed[idx]);
    }
}

// ═════════════════════════════════════════════════════════════
// Patch embed GEMM — warp-specialized tcgen05 (cta_group::2)
// ═════════════════════════════════════════════════════════════

__global__ void __launch_bounds__(THREADS, 1)
__cluster_dims__(2, 1, 1)
patch_embed_gemm(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    const __nv_bfloat16* __restrict__ combined,
    __nv_bfloat16* __restrict__ C
) {

    extern __shared__ __align__(128) char smem[];
    const int sm_id = blockIdx.x;
    const int tid   = threadIdx.x;
    const int warp  = tid / 32;
    const int lane  = tid % 32;

    // CTA rank within 2-CTA cluster
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    const int cluster_id = sm_id / 2;       // 0..73
    const int num_clusters = SM_COUNT / 2;   // 74

    // Per-stage SMEM offsets (6 stages, A=16384 + B/2=16384 = 32768 per stage)
    static constexpr int off_a[N_STAGES] = {0, 32768, 65536, 98304, 131072, 163840};
    static constexpr int off_b[N_STAGES] = {16384, 49152, 81920, 114688, 147456, 180224};

    // ── Mbarrier init ──
    if (tid == 0) {
        for (int s = 0; s < N_STAGES; s++) {
            mbar_init(smem_to_uint(smem + OFF_TMA_MBAR + s * 8), 2);   // both CTAs arrive at CTA0's TMA mbar
            mbar_init(smem_to_uint(smem + OFF_MMA_MBAR + s * 8), 1);   // CTA0 multicast commits → 1 arrival per CTA
        }
        // Double-buffered mainloop/epilogue mbarriers
        for (int i = 0; i < 2; i++) {
            mbar_init(smem_to_uint(smem + OFF_MAINLOOP_MBAR + i * 8), 1);          // CTA0 multicast commits
            mbar_init(smem_to_uint(smem + OFF_EPILOGUE_MBAR + i * 8), NUM_EPI_WARPS * 2 * 32); // NUM_EPI_WARPS warps × 2 CTAs × 32 threads
        }
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    // ── TMEM alloc: single alloc of TN*2=512 cols, in MMA warp (matmul_v7 pattern) ──
    if (warp == 1) {
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(smem_to_uint(smem + OFF_TMEM)), "r"(TMEM_COLS));
    }

    uint32_t tma_mbar[N_STAGES], mma_mbar[N_STAGES];
    uint32_t smem_a[N_STAGES], smem_b[N_STAGES];
    for (int s = 0; s < N_STAGES; s++) {
        tma_mbar[s] = smem_to_uint(smem + OFF_TMA_MBAR + s * 8);
        mma_mbar[s] = smem_to_uint(smem + OFF_MMA_MBAR + s * 8);
        smem_a[s]   = smem_to_uint(smem + off_a[s]);
        smem_b[s]   = smem_to_uint(smem + off_b[s]);
    }
    const uint32_t mainloop_mbar_addr = smem_to_uint(smem + OFF_MAINLOOP_MBAR);
    const uint32_t epilogue_mbar_addr = smem_to_uint(smem + OFF_EPILOGUE_MBAR);

    const int tile_start = (int)((long long)cluster_id * TOTAL_TILES / num_clusters);
    const int tile_end   = (int)((long long)(cluster_id + 1) * TOTAL_TILES / num_clusters);

    int tma_phase[N_STAGES] = {0};
    int mma_phase[N_STAGES] = {0};

    // Double-buffered mbar phase tracking (SUPERIOR pattern).
    // mbar[X] protects tmem[X]. Phase init 1 = "fresh mbar, skip first wait".
    // W1: waits epilogue_mbar[buf] before writing tmem[buf]
    // W2-5: waits mainloop_mbar[buf^1] before reading tmem[buf^1]
    //        arrives epilogue_mbar[buf^1] after reading tmem[buf^1]
    const int start_buf = tile_start & 1;
    int epi_phase[2] = {1, 1};           // W1's epilogue wait phases (both fresh)
    int ml_phase[2]  = {start_buf, 1 - start_buf};  // W2-5: prev_buf on first tile must be fresh(1)

    for (int tile_idx = tile_start; tile_idx < tile_end; tile_idx++) {
        const int buf = tile_idx & 1;
        const int tm = tile_idx / TILES_N;
        int tn = tile_idx % TILES_N;
        if (tm & 1) tn = TILES_N - 1 - tn;  // snake: reverse odd M-rows
        const int m_start = tm * TM * 2 + cta_rank * TM;  // macro tile (256 rows) + CTA offset
        const int n_start = tn * TN;

        // ═══ K-LOOP (W0-1) + OVERLAPPED EPILOGUE (W2-5) ═══
        if (warp == 0) {
            // ── LOAD WARP (W0): TMA async bulk copies (both CTAs) ──
            if (lane == 0) {
                for (int ki = 0; ki < K_ITERS; ki++) {
                    const int s = ki % N_STAGES;
                    const int k_start = ki * TK;

                    if (tile_idx > tile_start || ki >= N_STAGES) {
                        mbar_wait(mma_mbar[s], mma_phase[s]);
                        mma_phase[s] ^= 1;
                    }

                    // Both CTAs arrive at CTA0's TMA mbar (masked)
                    const uint32_t tma_mbar_masked = tma_mbar[s] & 0xFEFFFFFF;
                    tma_load_2d(smem_a[s], &tma_a, k_start, m_start, tma_mbar_masked);
                    tma_load_2d(smem_b[s], &tma_b, k_start, n_start + cta_rank * (TN/2), tma_mbar_masked);
                    mbar_arrive_expect_tx(tma_mbar_masked, TMA_BYTES);
                }
            }
        } else if (warp == 1) {
            // ── MMA WARP (W1): tcgen05.mma → tmem_base + buf*TN (CTA0 only) ──
            if (lane == 0 && cta_rank == 0) {
                // Wait for epilogue to release tmem[buf]
                mbar_wait(epilogue_mbar_addr + buf * 8, epi_phase[buf]);
                epi_phase[buf] ^= 1;

                for (int ki = 0; ki < K_ITERS; ki++) {
                    const int s = ki % N_STAGES;

                    mbar_wait(tma_mbar[s], tma_phase[s]);
                    tma_phase[s] ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");

                    uint64_t desc_a = make_smem_desc(smem_a[s]);
                    uint64_t desc_b = make_smem_desc(smem_b[s]);

                    {
                        uint32_t accumulate = (ki == 0) ? 0 : 1;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p;\n\t"
                            "setp.ne.b32 p, %4, 0;\n\t"
                            "tcgen05.mma.cta_group::2.kind::f8f6f4 "
                            "[%0], %1, %2, %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p;\n\t"
                            "}"
                            :
                            : "r"(buf * TN), "l"(desc_a), "l"(desc_b), "r"(IDESC),
                              "r"(accumulate),
                              "r"(0), "r"(0), "r"(0), "r"(0),
                              "r"(0), "r"(0), "r"(0), "r"(0));
                    }

                    for (int sub = 1; sub < MMA_PER_KI; sub++) {
                        desc_a += 2;
                        desc_b += 2;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p;\n\t"
                            "setp.ne.b32 p, %4, 0;\n\t"
                            "tcgen05.mma.cta_group::2.kind::f8f6f4 "
                            "[%0], %1, %2, %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p;\n\t"
                            "}"
                            :
                            : "r"(buf * TN), "l"(desc_a), "l"(desc_b), "r"(IDESC),
                              "r"(1),
                              "r"(0), "r"(0), "r"(0), "r"(0),
                              "r"(0), "r"(0), "r"(0), "r"(0));
                    }

                    // Multicast commit → MMA mbar in both CTAs
                    tcgen05_commit_mcast(mma_mbar[s], 0x3);
                }

                // Signal K-loop done: multicast commit → mainloop_mbar[buf] in both CTAs
                tcgen05_commit_mcast(mainloop_mbar_addr + buf * 8, 0x3);
            }
        } else {
            // ── OVERLAPPED EPILOGUE (W2+) ──
            const int ew = warp - 2;
            const int row_group = ew % 4;
            const int is_split = (row_group < (NUM_EPI_WARPS - 4)) ? 1 : 0;
            const int col_rank = ew / 4;

            const int prev_buf = buf ^ 1;

            // Wait for W1's K-loop on tmem[prev_buf] (from previous tile).
            // All epilogue warps poll independently — no bar.sync broadcast needed.
            mbar_wait(mainloop_mbar_addr + prev_buf * 8, ml_phase[prev_buf]);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            ml_phase[prev_buf] ^= 1;

            if (tile_idx > tile_start) {
                // Epilogue: read tmem[prev_buf], add inline BF16 combined, store to global
                const int prev_idx = tile_idx - 1;
                const int ptm = prev_idx / TILES_N;
                int ptn = prev_idx % TILES_N;
                if (ptm & 1) ptn = TILES_N - 1 - ptn;
                const int prev_m = ptm * TM * 2 + cta_rank * TM;
                const int prev_n = ptn * TN;

                const int gm_base = prev_m + row_group * 32;
                const int pos_row = (gm_base + lane) % SEQ_LEN;
                if (is_split) {
                    if (col_rank == 0)
                        epilogue_store<0, TN/2>(prev_buf * TN, row_group, lane, gm_base, prev_n, combined, pos_row, C, cta_rank);
                    else
                        epilogue_store<TN/2, TN>(prev_buf * TN, row_group, lane, gm_base, prev_n, combined, pos_row, C, cta_rank);
                } else {
                    epilogue_store<0, TN>(prev_buf * TN, row_group, lane, gm_base, prev_n, combined, pos_row, C, cta_rank);
                }

                // Signal: done reading tmem[prev_buf] — arrive at CTA0's epilogue mbar
                const uint32_t epi_mbar_masked = (epilogue_mbar_addr + prev_buf * 8) & 0xFEFFFFFF;
                mbar_arrive(epi_mbar_masked);
            }
        }
    }  // tile loop

    // ── DRAIN (W2+ only): epilogue for the last tile ──
    if (warp >= 2) {
        const int ew = warp - 2;
        const int row_group = ew % 4;
        const int is_split = (row_group < (NUM_EPI_WARPS - 4)) ? 1 : 0;
        const int col_rank = ew / 4;

        const int last_buf = (tile_end - 1) & 1;

        // Wait for W1's K-loop on tmem[last_buf]
        // All epilogue warps poll independently — no bar.sync broadcast needed.
        mbar_wait(mainloop_mbar_addr + last_buf * 8, ml_phase[last_buf]);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

        // Epilogue store for the last tile
        const int last_idx = tile_end - 1;
        const int ltm = last_idx / TILES_N;
        int ltn = last_idx % TILES_N;
        if (ltm & 1) ltn = TILES_N - 1 - ltn;
        const int last_m = ltm * TM * 2 + cta_rank * TM;
        const int last_n = ltn * TN;
        const int gm_base = last_m + row_group * 32;
        const int pos_row = (gm_base + lane) % SEQ_LEN;
        if (is_split) {
            if (col_rank == 0)
                epilogue_store<0, TN/2>(last_buf * TN, row_group, lane, gm_base, last_n, combined, pos_row, C, cta_rank);
            else
                epilogue_store<TN/2, TN>(last_buf * TN, row_group, lane, gm_base, last_n, combined, pos_row, C, cta_rank);
        } else {
            epilogue_store<0, TN>(last_buf * TN, row_group, lane, gm_base, last_n, combined, pos_row, C, cta_rank);
        }
    }

    // ── Cluster sync + TMEM dealloc (all warps, both CTAs) ──
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 2) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
            :: "r"(0), "r"(TMEM_COLS));
    }
}

// ═════════════════════════════════════════════════════════════
// Host
// ═════════════════════════════════════════════════════════════

int main() {
    setbuf(stdout, NULL);  // unbuffered stdout for debugging
    printf("SigLIP2 patch embed GEMM — tcgen05 cta_group::2 (%d warps [%d epi], cluster of 2)\n",
           2 + NUM_EPI_WARPS, NUM_EPI_WARPS);
    printf("  GEMM: [%d,%d] x [%d,%d]^T  6-stage pipeline  inline BF16 combined  st.global stores  idesc: 0x%08X\n",
           M_TOTAL, K_DIM, N_DIM, K_DIM, IDESC);

    uint8_t *d_A, *d_B;
    float *d_bias, *d_pos;
    __nv_bfloat16 *d_combined, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A,    (size_t)M_TOTAL * K_DIM));
    CUDA_CHECK(cudaMalloc(&d_B,    (size_t)N_DIM   * K_DIM));
    CUDA_CHECK(cudaMalloc(&d_bias,  (size_t)N_DIM  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos,   (size_t)SEQ_LEN * N_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_combined, (size_t)SEQ_LEN * N_DIM * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_C,    (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16)));

    // Fill A, B with 0x3C (=1.5 in FP8 E4M3, a valid finite value)
    CUDA_CHECK(cudaMemset(d_A, 0x3C, (size_t)M_TOTAL * K_DIM));
    CUDA_CHECK(cudaMemset(d_B, 0x3C, (size_t)N_DIM * K_DIM));
    CUDA_CHECK(cudaMemset(d_bias, 0, (size_t)N_DIM * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pos,  0, (size_t)SEQ_LEN * N_DIM * sizeof(float)));

    // Precompute combined[r][c] = __float2bfloat16(bias[c] + pos_embed[r*N_DIM+c])
    {
        int n_elem = SEQ_LEN * N_DIM;
        int tpb = 256;
        int bpg = (n_elem + tpb - 1) / tpb;
        precompute_combined<<<bpg, tpb>>>(d_bias, d_pos, d_combined, SEQ_LEN, N_DIM);
        CUDA_CHECK(cudaGetLastError());
    }
    printf("  Alloc + precompute done\n");

    CUtensorMap h_tma_a, h_tma_b;

    // 2D TMA with SWIZZLE_128B: load [TK, height] per stage
    // Global memory: row-major A[M,K], B[N,K] with element=UINT8 (FP8)
    // TK=128 FP8 = 128 bytes = exactly 128B → SWIZZLE_128B
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
        uint32_t box[2]     = {TK, TN/2};   // each CTA loads half of B columns
        uint32_t estrides[2]= {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(&h_tma_b,
            CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)d_B,
            dims, strides, box, estrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }

    CUDA_CHECK(cudaFuncSetAttribute(patch_embed_gemm,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    printf("  TMA descriptors + func attr done\n");

    // ── Warmup: 2 iterations ──
    printf("Launching warmup (2 iters)...\n");
    for (int _i = 0; _i < 2; _i++) {
    patch_embed_gemm<<<SM_COUNT, THREADS, SMEM_BYTES>>>(h_tma_a, h_tma_b, d_combined, d_C);
    }
    printf("  Waiting for warmup sync...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Warmup done.\n");

    // ── Timed: 10 iterations ──
    printf("Timing: 10 iterations...\n");
    cudaEvent_t _t0, _t1;
    cudaEventCreate(&_t0);
    cudaEventCreate(&_t1);
    cudaEventRecord(_t0);
    for (int _i = 0; _i < 10; _i++) {
    patch_embed_gemm<<<SM_COUNT, THREADS, SMEM_BYTES>>>(h_tma_a, h_tma_b, d_combined, d_C);
    }
    cudaEventRecord(_t1);
    cudaEventSynchronize(_t1);
    float _ms;
    cudaEventElapsedTime(&_ms, _t0, _t1);
    _ms /= 10.0f;
    printf("Custom kernel: %.3f ms  %.2f TFLOPS\n",
           _ms, 2.0 * M_TOTAL * N_DIM * K_DIM / _ms / 1e9);
    cudaEventDestroy(_t0);
    cudaEventDestroy(_t1);

    // ── Checksum run ──
    patch_embed_gemm<<<SM_COUNT, THREADS, SMEM_BYTES>>>(h_tma_a, h_tma_b, d_combined, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    __nv_bfloat16* h_C = (__nv_bfloat16*)malloc((size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    double cksum = 0;
    for (int i = 0; i < 1024 && i < M_TOTAL * N_DIM; i++) cksum += __bfloat162float(h_C[i]);
    printf("Checksum (first 1024): %f\n", cksum);
    printf("C[0,0..3] = %.1f %.1f %.1f %.1f\n",
           __bfloat162float(h_C[0]), __bfloat162float(h_C[1]),
           __bfloat162float(h_C[2]), __bfloat162float(h_C[3]));

    free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_bias); cudaFree(d_pos);
    cudaFree(d_combined); cudaFree(d_C);
    return 0;
}
