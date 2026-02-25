// Hand-tuned from gen.py output — TK=128, SWIZZLE_128B, 5-stage pipeline
// Target: B200  Batch: 4736  GEMM: [928256,768]×[768,768]^T
// Pipeline: 5-stage  K-iters: 6  MMA/iter: 4  idesc: 0x08200010
// Warps: 6 (192 threads)  cta_group::1
// Warp-specialized: Load(W0) | MMA(W1,cta_group::1) | Epilogue(W2-5)  BF16 output
// tcgen05.mma.cta_group::1.kind::f8f6f4  (E4M3 × E4M3 → FP32)

#include <cuda.h>
#include <cuda_bf16.h>
#include <curand.h>
#include <cstdint>
#include <cstdio>

#define SM_COUNT       148
#define THREADS        192
#define BATCH_SIZE     4736
#define SEQ_LEN        196
#define M_TOTAL        928256
#define N_DIM          768
#define K_DIM          768
#define TM             128
#define TN             128
#define TK             128
#define TILES_M        7252
#define TILES_N        6
#define K_ITERS        6
#define TOTAL_TILES    43512
#define N_STAGES       5
#define OFF_SUMS_BUF       163840
#define OFF_TMEM_0         229376
#define OFF_TMEM_1         229380
#define OFF_TMA_MBAR       229384
#define OFF_MMA_MBAR       229424
#define OFF_MAINLOOP_MBAR  229464
#define OFF_EPILOGUE_MBAR  229472
#define SMEM_BYTES         229504
#define TMEM_COLS      128
#define IDESC          0x08200010U
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
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(addr), "r"(tx_count) : "memory");
}

static __device__ __forceinline__
void mbar_arrive(uint32_t addr) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(addr) : "memory");
}

static __device__ __forceinline__
void tma_load_2d(uint32_t smem_dst, const void* tma_desc,
                  int32_t c0, int32_t c1, uint32_t mbar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_dst), "l"(tma_desc), "r"(c0), "r"(c1), "r"(mbar)
        : "memory");
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

// ── Unified epilogue: TMEM readback → sums add → FP32→BF16 → st.global.v8 ──
// Software-pipelined: double-buffered TMEM loads (A/B) across column pairs.
// Sums (bias+pos_embed) are pre-computed in SMEM with [ew][col][lane] layout,
// stride 32 between columns. Both overlapped and drain paths use this function.

static __device__ __forceinline__
void epilogue_store(
    uint32_t tmem_addr,
    int ew,
    int lane,
    int gm_base,
    int n_start,
    const float* sums,  // smem + OFF_SUMS_BUF + ew*16384 + lane*4, stride 32
    __nv_bfloat16* __restrict__ C
) {
    __nv_bfloat16* row_out = C + (long long)(gm_base + lane) * N_DIM + n_start;
    const int taddr_base = tmem_addr + (ew * 32 << 16);

    // Double-buffered TMEM registers
    float a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15;
    float b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15;

    // Prefetch first chunk into buffer A
    TMEM_LOAD(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15, taddr_base);

    for (int nc = 0; nc < TN; nc += 32) {
        // ── Chunk nc (buffer A) ──
        float s0=sums[(nc+0)*32], s1=sums[(nc+1)*32];
        float s2=sums[(nc+2)*32], s3=sums[(nc+3)*32];
        float s4=sums[(nc+4)*32], s5=sums[(nc+5)*32];
        float s6=sums[(nc+6)*32], s7=sums[(nc+7)*32];
        float s8=sums[(nc+8)*32], s9=sums[(nc+9)*32];
        float s10=sums[(nc+10)*32], s11=sums[(nc+11)*32];
        float s12=sums[(nc+12)*32], s13=sums[(nc+13)*32];
        float s14=sums[(nc+14)*32], s15=sums[(nc+15)*32];

        TMEM_WAIT();
        TMEM_LOAD(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15, taddr_base + nc + 16);

        a0+=s0; a1+=s1; a2+=s2; a3+=s3;
        a4+=s4; a5+=s5; a6+=s6; a7+=s7;
        a8+=s8; a9+=s9; a10+=s10; a11+=s11;
        a12+=s12; a13+=s13; a14+=s14; a15+=s15;

        CVT_STG(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15, row_out + nc);

        // ── Chunk nc+16 (buffer B) ──
        s0=sums[(nc+16)*32]; s1=sums[(nc+17)*32];
        s2=sums[(nc+18)*32]; s3=sums[(nc+19)*32];
        s4=sums[(nc+20)*32]; s5=sums[(nc+21)*32];
        s6=sums[(nc+22)*32]; s7=sums[(nc+23)*32];
        s8=sums[(nc+24)*32]; s9=sums[(nc+25)*32];
        s10=sums[(nc+26)*32]; s11=sums[(nc+27)*32];
        s12=sums[(nc+28)*32]; s13=sums[(nc+29)*32];
        s14=sums[(nc+30)*32]; s15=sums[(nc+31)*32];

        TMEM_WAIT();
        if (nc + 32 < TN)
            TMEM_LOAD(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15, taddr_base + nc + 32);

        b0+=s0; b1+=s1; b2+=s2; b3+=s3;
        b4+=s4; b5+=s5; b6+=s6; b7+=s7;
        b8+=s8; b9+=s9; b10+=s10; b11+=s11;
        b12+=s12; b13+=s13; b14+=s14; b15+=s15;

        CVT_STG(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15, row_out + nc + 16);
    }
}

// ═════════════════════════════════════════════════════════════
// Patch embed GEMM — warp-specialized tcgen05 (cta_group::1)
// ═════════════════════════════════════════════════════════════

__global__ void __launch_bounds__(192, 1)
patch_embed_gemm(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    const float*   __restrict__ bias,
    const float*   __restrict__ pos_embed,
    __nv_bfloat16* __restrict__ C
) {

    extern __shared__ __align__(128) char smem[];
    const int sm_id = blockIdx.x;
    const int tid   = threadIdx.x;
    const int warp  = tid / 32;
    const int lane  = tid % 32;

    // Per-stage SMEM offsets (codegen constants)
    static constexpr int off_a[N_STAGES] = {0, 32768, 65536, 98304, 131072};
    static constexpr int off_b[N_STAGES] = {16384, 49152, 81920, 114688, 147456};

    // ── TMEM allocation: 2 buffers (cta_group::1) ──
    // One warp calls alloc (all lanes converged); other warps skip.
    // No relinquish needed — only required if other warps also want to alloc.
    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(smem_to_uint(smem + OFF_TMEM_0)), "r"(TMEM_COLS));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(smem_to_uint(smem + OFF_TMEM_1)), "r"(TMEM_COLS));
    }
    __syncthreads();
    const uint32_t tmem_base0 = *(volatile uint32_t*)(smem + OFF_TMEM_0);
    const uint32_t tmem_base1 = *(volatile uint32_t*)(smem + OFF_TMEM_1);
    const uint32_t tmem_base[2] = {tmem_base0, tmem_base1};

    // ── Mbarrier init ──
    if (tid == 0) {
        for (int s = 0; s < N_STAGES; s++) {
            mbar_init(smem_to_uint(smem + OFF_TMA_MBAR + s * 8), 1);
            mbar_init(smem_to_uint(smem + OFF_MMA_MBAR + s * 8), 1);
        }
        mbar_init(smem_to_uint(smem + OFF_MAINLOOP_MBAR), 1);   // W1 arrives once
        mbar_init(smem_to_uint(smem + OFF_EPILOGUE_MBAR), 4);   // W2-5 each arrive once
    }
    __syncthreads();

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

    const int tile_start = (int)((long long)sm_id * TOTAL_TILES / SM_COUNT);
    const int tile_end   = (int)((long long)(sm_id + 1) * TOTAL_TILES / SM_COUNT);

    int tma_phase[N_STAGES] = {0};
    int mma_phase[N_STAGES] = {0};
    int mainloop_phase = 0;
    int epilogue_phase = 0;
    int first_tile = 1;

    for (int tile_idx = tile_start; tile_idx < tile_end; tile_idx++) {
        const int buf = tile_idx & 1;
        const int tm = tile_idx / TILES_N;
        int tn = tile_idx % TILES_N;
        if (tm & 1) tn = TILES_N - 1 - tn;  // snake: reverse odd M-rows
        const int m_start = tm * TM;
        const int n_start = tn * TN;

        // ═══ K-LOOP (W0-1) + OVERLAPPED EPILOGUE (W2-3) ═══
        if (warp == 0) {
            // ── LOAD WARP (W0): TMA async bulk copies ──
            if (lane == 0) {
                for (int ki = 0; ki < K_ITERS; ki++) {
                    const int s = ki % N_STAGES;
                    const int k_start = ki * TK;

                    if (!(first_tile && ki < N_STAGES)) {
                        mbar_wait(mma_mbar[s], mma_phase[s]);
                        mma_phase[s] ^= 1;
                    }

                    mbar_arrive_expect_tx(tma_mbar[s], TMA_BYTES);
                    tma_load_2d(smem_a[s], &tma_a, k_start, m_start, tma_mbar[s]);
                    tma_load_2d(smem_b[s], &tma_b, k_start, n_start, tma_mbar[s]);
                }
            }
        } else if (warp == 1) {
            // ── MMA WARP (W1): tcgen05.mma.cta_group::1 → tmem_base[buf] ──
            if (lane == 0) {
                // Wait for previous epilogue to release TMEM buffer
                if (!first_tile) {
                    mbar_wait(epilogue_mbar_addr, epilogue_phase);
                    epilogue_phase ^= 1;
                }

                for (int ki = 0; ki < K_ITERS; ki++) {
                    const int s = ki % N_STAGES;

                    mbar_wait(tma_mbar[s], tma_phase[s]);
                    tma_phase[s] ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");

                    uint64_t desc_a = make_smem_desc(smem_a[s]);
                    uint64_t desc_b = make_smem_desc(smem_b[s]);

                    // First sub-MMA: clear accum on ki==0, accumulate otherwise
                    {
                        uint32_t accumulate = (ki == 0) ? 0 : 1;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p;\n\t"
                            "setp.ne.b32 p, %4, 0;\n\t"
                            "tcgen05.mma.cta_group::1.kind::f8f6f4 "
                            "[%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t"
                            "}"
                            :
                            : "r"(tmem_base[buf]), "l"(desc_a), "l"(desc_b), "r"(IDESC),
                              "r"(accumulate),
                              "r"(0), "r"(0), "r"(0), "r"(0));
                    }

                    // Remaining sub-MMAs: advance desc by 32B, always accumulate
                    for (int sub = 1; sub < MMA_PER_KI; sub++) {
                        desc_a += 2;  // 32 bytes / 16 = 2 descriptor units
                        desc_b += 2;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p;\n\t"
                            "setp.ne.b32 p, %4, 0;\n\t"
                            "tcgen05.mma.cta_group::1.kind::f8f6f4 "
                            "[%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t"
                            "}"
                            :
                            : "r"(tmem_base[buf]), "l"(desc_a), "l"(desc_b), "r"(IDESC),
                              "r"(1),
                              "r"(0), "r"(0), "r"(0), "r"(0));
                    }

                    asm volatile(
                        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                        :: "r"(mma_mbar[s]) : "memory");
                }

                // Signal K-loop done — TMEM data ready for epilogue
                asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
                mbar_arrive(mainloop_mbar_addr);
            }
        } else {
            // ── OVERLAPPED EPILOGUE (W2-5): TMEM readback + SMEM sums + store ──
            const int ew = warp - 2;  // 0,1,2,3 for warps 2-5

            if (!first_tile) {
                // Wait for MMA to finish — TMEM data ready
                mbar_wait(mainloop_mbar_addr, mainloop_phase);
                mainloop_phase ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

                const int prev_idx = tile_idx - 1;
                const int ptm = prev_idx / TILES_N;
                int ptn = prev_idx % TILES_N;
                if (ptm & 1) ptn = TILES_N - 1 - ptn;  // snake
                const int prev_m = ptm * TM;
                const int prev_n = ptn * TN;
                const uint32_t prev_tmem = tmem_base[buf ^ 1];

                const int gm_base = prev_m + ew * 32;
                const float* sums = (const float*)(smem + OFF_SUMS_BUF + ew * 16384 + lane * 4);

                epilogue_store(prev_tmem, ew, lane, gm_base, prev_n, sums, C);
            }

            // Signal epilogue done (dummy on first tile to prime the mbarrier)
            if (lane == 0) mbar_arrive(epilogue_mbar_addr);

            // Prefetch bias+pos sums into SMEM for current tile
            // (consumed by next iteration's overlapped epilogue — same warp/lane)
            {
                const int gm_base_cur = m_start + ew * 32;
                const int pos_row = (gm_base_cur + lane) % SEQ_LEN;
                float* sums_out = (float*)(smem + OFF_SUMS_BUF + ew * 16384 + lane * 4);
                const float* bp_cur = bias + n_start;
                const float* pp_cur = pos_embed + (long long)pos_row * N_DIM + n_start;
                for (int j = 0; j < TN; j++)
                    sums_out[j * 32] = bp_cur[j] + pp_cur[j];
            }
        }

        first_tile = 0;
    }  // tile loop

    // ── Drain epilogue: warps 0-3 write the last tile ──
    {
        const int num_tiles = tile_end - tile_start;
        const int drain_phase = (num_tiles - 1) & 1;
        // Wait for last tile's MMA to finish
        mbar_wait(mainloop_mbar_addr, drain_phase);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    }
    if (warp < 4) {
        const int last_idx = tile_end - 1;
        const int last_buf = last_idx & 1;
        const int ltm = last_idx / TILES_N;
        int ltn = last_idx % TILES_N;
        if (ltm & 1) ltn = TILES_N - 1 - ltn;
        const int last_m = ltm * TM;
        const int last_n = ltn * TN;
        const uint32_t last_tmem = tmem_base[last_buf];

        const int ew = warp;
        const int gm_base = last_m + ew * 32;
        const int pos_row = (gm_base + lane) % SEQ_LEN;

        // Compute sums into SMEM (same warp reads them — no cross-warp sync needed)
        float* sums_out = (float*)(smem + OFF_SUMS_BUF + ew * 16384 + lane * 4);
        const float* bp = bias + last_n;
        const float* pp = pos_embed + (long long)pos_row * N_DIM + last_n;
        for (int j = 0; j < TN; j++)
            sums_out[j * 32] = bp[j] + pp[j];

        epilogue_store(last_tmem, ew, lane, gm_base, last_n,
                       (const float*)sums_out, C);
    }

    __syncthreads();  // all warps done before dealloc


    // ── TMEM dealloc: 2 buffers (warp 0 only, cta_group::1) ──
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :: "r"(tmem_base[0]), "r"(TMEM_COLS));
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :: "r"(tmem_base[1]), "r"(TMEM_COLS));
    }
}

// ═════════════════════════════════════════════════════════════
// Host
// ═════════════════════════════════════════════════════════════

int main() {
    setbuf(stdout, NULL);  // unbuffered stdout for debugging
    printf("SigLIP2 patch embed GEMM — tcgen05 cta_group::1 (6 warps)\n");
    printf("  GEMM: [%d,%d] x [%d,%d]^T  5-stage pipeline  SMEM sums  st.global stores  idesc: 0x%08X\n",
           M_TOTAL, K_DIM, N_DIM, K_DIM, IDESC);

    uint8_t *d_A, *d_B;
    float *d_bias, *d_pos;
    __nv_bfloat16 *d_C;
    CUDA_CHECK(cudaMalloc(&d_A,    (size_t)M_TOTAL * K_DIM));
    CUDA_CHECK(cudaMalloc(&d_B,    (size_t)N_DIM   * K_DIM));
    CUDA_CHECK(cudaMalloc(&d_bias,  (size_t)N_DIM  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos,   (size_t)SEQ_LEN * N_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C,    (size_t)M_TOTAL * N_DIM * sizeof(__nv_bfloat16)));

    // Fill A, B with 0x3C (=1.5 in FP8 E4M3, a valid finite value)
    CUDA_CHECK(cudaMemset(d_A, 0x3C, (size_t)M_TOTAL * K_DIM));
    CUDA_CHECK(cudaMemset(d_B, 0x3C, (size_t)N_DIM * K_DIM));
    CUDA_CHECK(cudaMemset(d_bias, 0, (size_t)N_DIM * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pos,  0, (size_t)SEQ_LEN * N_DIM * sizeof(float)));
    printf("  Alloc + RNG done\n");

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
        uint32_t box[2]     = {TK, TN};
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
    patch_embed_gemm<<<SM_COUNT, THREADS, SMEM_BYTES>>>(h_tma_a, h_tma_b, d_bias, d_pos, d_C);
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
    patch_embed_gemm<<<SM_COUNT, THREADS, SMEM_BYTES>>>(h_tma_a, h_tma_b, d_bias, d_pos, d_C);
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
    patch_embed_gemm<<<SM_COUNT, THREADS, SMEM_BYTES>>>(h_tma_a, h_tma_b, d_bias, d_pos, d_C);
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
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_bias); cudaFree(d_pos); cudaFree(d_C);
    return 0;
}