// Hand-tuned from gen.py output — TK=128, SWIZZLE_128B, 7-stage pipeline
// Target: B200  Batch: 4736  GEMM: [928256,768]×[768,768]^T
// Pipeline: 7-stage  K-iters: 6  MMA/iter: 4  idesc: 0x08200010
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
#define N_STAGES       6
#define OFF_TMEM_0         196608
#define OFF_TMEM_1         196612
#define OFF_TMA_MBAR       196616
#define OFF_MMA_MBAR       196664
#define OFF_MAINLOOP_MBAR  196712
#define OFF_EPILOGUE_MBAR  196720
#define SMEM_BYTES         196736
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
        "st.global.v4.b32 [%16], {b0,b1,b2,b3};\n\t" \
        "st.global.v4.b32 [%17], {b4,b5,b6,b7};\n\t" \
        "}" \
        :: "f"(r0),"f"(r1),"f"(r2),"f"(r3), \
           "f"(r4),"f"(r5),"f"(r6),"f"(r7), \
           "f"(r8),"f"(r9),"f"(r10),"f"(r11), \
           "f"(r12),"f"(r13),"f"(r14),"f"(r15), \
           "l"(GPTR), "l"((GPTR) + 8) \
        : "memory")

// ── Shared epilogue: TMEM readback → bias+pos add → FP32→BF16 → st.global ──

static __device__ __forceinline__
void epilogue_store(
    uint32_t tmem_addr,
    int ew,
    int lane,
    int gm_base,
    int n_start,
    const float* __restrict__ bp,
    const float* __restrict__ pp,
    __nv_bfloat16* __restrict__ C
) {
    __nv_bfloat16* row_out = C + (long long)(gm_base + lane) * N_DIM + n_start;

    for (int nc = 0; nc < TN; nc += 16) {
        float v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15;
        int taddr = tmem_addr + (ew * 32 << 16) + nc;

        TMEM_LOAD(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15, taddr);

        float s0=bp[nc]+pp[nc], s1=bp[nc+1]+pp[nc+1];
        float s2=bp[nc+2]+pp[nc+2], s3=bp[nc+3]+pp[nc+3];
        float s4=bp[nc+4]+pp[nc+4], s5=bp[nc+5]+pp[nc+5];
        float s6=bp[nc+6]+pp[nc+6], s7=bp[nc+7]+pp[nc+7];
        float s8=bp[nc+8]+pp[nc+8], s9=bp[nc+9]+pp[nc+9];
        float s10=bp[nc+10]+pp[nc+10], s11=bp[nc+11]+pp[nc+11];
        float s12=bp[nc+12]+pp[nc+12], s13=bp[nc+13]+pp[nc+13];
        float s14=bp[nc+14]+pp[nc+14], s15=bp[nc+15]+pp[nc+15];

        TMEM_WAIT();

        v0+=s0; v1+=s1; v2+=s2; v3+=s3;
        v4+=s4; v5+=s5; v6+=s6; v7+=s7;
        v8+=s8; v9+=s9; v10+=s10; v11+=s11;
        v12+=s12; v13+=s13; v14+=s14; v15+=s15;

        CVT_STG(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15, row_out + nc);
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
    static constexpr int off_a[N_STAGES] = {0, 32768, 65536, 98304, 131072, 163840};
    static constexpr int off_b[N_STAGES] = {16384, 49152, 81920, 114688, 147456, 180224};

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
            // ── OVERLAPPED EPILOGUE (W2-5): TMEM readback + bias/pos + store ──
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

                const int ew = warp - 2;  // 0,1,2,3 for warps 2-5
                const int gm_base = prev_m + ew * 32;
                const int pos_row = (gm_base + lane) % SEQ_LEN;
                const float* bp = bias + prev_n;
                const float* pp = pos_embed + (long long)pos_row * N_DIM + prev_n;

                epilogue_store(prev_tmem, ew, lane, gm_base, prev_n, bp, pp, C);
            }

            // Signal epilogue done (dummy on first tile to prime the mbarrier)
            if (lane == 0) mbar_arrive(epilogue_mbar_addr);
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
        const float* bp = bias + last_n;
        const float* pp = pos_embed + (long long)pos_row * N_DIM + last_n;

        epilogue_store(last_tmem, ew, lane, gm_base, last_n, bp, pp, C);
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
    printf("SigLIP2 patch embed GEMM — tcgen05 cta_group::1 (4 warps)\n");
    printf("  GEMM: [%d,%d] x [%d,%d]^T  6-stage pipeline  st.global stores  idesc: 0x%08X\n",
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