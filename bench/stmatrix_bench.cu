/*
stmatrix SMEM layout + throughput characterization for SM100a

GO/NO-GO gate for replacing STS.128 with stmatrix in fc2_w3 epilogue.

Tests:
  A. Layout characterization — stmatrix m8n8.x4 non-transposed: where do bytes land?
  B. Layout characterization — stmatrix m8n8.x4 transposed: how does .trans change it?
  C. Reference — STS.128 with our SWIZZLE_128B addressing
  D. Throughput comparison — STS.128 vs STSM at 1/2/4/8 warps
  E. Cross-warp contention — 4 epilogue warps concurrent STS vs STSM

Build: make stmatrix-bench && ./stmatrix-bench
*/

#include <cstdio>
#include <cstdint>
#include <cuda.h>

#define WARMUP 32
#define REPS   512

#define CU_CHECK(x) do { CUresult _r = (x); if (_r) { \
    const char *_s; cuGetErrorString(_r, &_s); \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, _s); exit(1); \
} } while(0)

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(1); \
} } while(0)

/*
 * Helper: encode a unique 16-bit value for (lane, element_index).
 * Decode: lane = val >> 5, elem = val & 0x1F.
 * Max: lane=31, elem=31 → 31*32+31 = 1023 < 65535.
 */
static __device__ __forceinline__
uint32_t encode_pair(int lane, int e0, int e1) {
    uint16_t lo = (uint16_t)(lane * 32 + e0);
    uint16_t hi = (uint16_t)(lane * 32 + e1);
    return (uint32_t)lo | ((uint32_t)hi << 16);
}

/* ════════════════════════════════════════════════════════════════════
 * Test A: stmatrix x4 NON-transposed layout
 *
 * stmatrix.sync.aligned.x4.m8n8.shared.b16 [addr], {r0,r1,r2,r3}
 * 4 tiles of m8n8: threads 0-7=tile0, 8-15=tile1, 16-23=tile2, 24-31=tile3
 * Each tile: 8 rows × 8 cols × 2B = 128B, contiguous at 16B row stride
 * Each thread provides its row address within the 128B tile.
 * ════════════════════════════════════════════════════════════════════ */
__global__ void test_stsm_nontrans(uint16_t* out) {
    __shared__ alignas(128) uint8_t smem[2048];

    int lane = threadIdx.x % 32;
    int tile = lane / 8;
    int row  = lane % 8;

    /* Zero SMEM */
    for (int i = lane * 4; i < 2048; i += 128)
        *(uint32_t*)(smem + i) = 0xDEADDEAD;
    __syncthreads();

    /* Prepare 4 regs: 8 BF16 values = 4 × uint32 (BF16x2 packed)
     * Element index within this thread: 0..7
     * Stored as reg0={e0,e1}, reg1={e2,e3}, reg2={e4,e5}, reg3={e6,e7} */
    uint32_t r0 = encode_pair(lane, 0, 1);
    uint32_t r1 = encode_pair(lane, 2, 3);
    uint32_t r2 = encode_pair(lane, 4, 5);
    uint32_t r3 = encode_pair(lane, 6, 7);

    /* Address: each tile gets a 128B-aligned block.
     * Within tile: thread (lane%8) addresses row (lane%8) at stride 16B.
     * Tiles at offsets: 0, 128, 256, 384. */
    uint32_t addr = (uint32_t)(uintptr_t)__cvta_generic_to_shared(smem)
                  + tile * 128 + row * 16;

    asm volatile(
        "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};"
        :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
    );
    __syncthreads();

    /* Thread 0 dumps all 512 bytes (256 uint16 values) */
    if (lane == 0) {
        for (int i = 0; i < 256; i++)
            out[i] = *(uint16_t*)(smem + i * 2);
    }
}

/* ════════════════════════════════════════════════════════════════════
 * Test B: stmatrix x4 TRANSPOSED layout
 *
 * stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [addr], {r0,r1,r2,r3}
 * Same addresses as test A, but .trans reorders data across threads.
 * ═══════════��════════════════════════════════════════════════════════ */
__global__ void test_stsm_trans(uint16_t* out) {
    __shared__ alignas(128) uint8_t smem[2048];

    int lane = threadIdx.x % 32;
    int tile = lane / 8;
    int row  = lane % 8;

    for (int i = lane * 4; i < 2048; i += 128)
        *(uint32_t*)(smem + i) = 0xDEADDEAD;
    __syncthreads();

    uint32_t r0 = encode_pair(lane, 0, 1);
    uint32_t r1 = encode_pair(lane, 2, 3);
    uint32_t r2 = encode_pair(lane, 4, 5);
    uint32_t r3 = encode_pair(lane, 6, 7);

    uint32_t addr = (uint32_t)(uintptr_t)__cvta_generic_to_shared(smem)
                  + tile * 128 + row * 16;

    asm volatile(
        "stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};"
        :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
    );
    __syncthreads();

    if (lane == 0) {
        for (int i = 0; i < 256; i++)
            out[i] = *(uint16_t*)(smem + i * 2);
    }
}

/* ════════════════════════════════════════════════════════════════════
 * Test C: STS.128 with SWIZZLE_128B reference
 *
 * Each lane stores 16B (4 regs) at lane*128 + swizzle_offset within
 * a 4096B region. This matches fc2_w3's epilogue exactly.
 * ═════════════════���══════════════════════════════════════════════════ */
__global__ void test_sts128_swizzle(uint16_t* out) {
    __shared__ alignas(128) uint8_t smem[4096];

    int lane = threadIdx.x % 32;

    for (int i = lane * 4; i < 4096; i += 128)
        *(uint32_t*)(smem + i) = 0xDEADDEAD;
    __syncthreads();

    uint32_t r0 = encode_pair(lane, 0, 1);
    uint32_t r1 = encode_pair(lane, 2, 3);
    uint32_t r2 = encode_pair(lane, 4, 5);
    uint32_t r3 = encode_pair(lane, 6, 7);

    /* SWIZZLE_128B: 128B rows, XOR with (lane & 7) << 4 */
    uint32_t base = (uint32_t)(uintptr_t)__cvta_generic_to_shared(smem) + lane * 128;
    uint32_t xor_val = (lane & 7) << 4;
    uint32_t sw0 = 0 ^ xor_val;

    asm volatile(
        "st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
        :: "r"(base + sw0), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
    );
    __syncthreads();

    /* Dump all 4096 bytes (2048 uint16 values) */
    if (lane == 0) {
        for (int i = 0; i < 2048; i++)
            out[i] = *(uint16_t*)(smem + i * 2);
    }
}

/* ════════════════════════════════════════════════════════════════════
 * Test D/E: Throughput benchmarks
 *
 * Measure cycles for repeated STS.128 vs stmatrix at various warp counts.
 * Each warp does 4 stores per iteration (matching our epilogue chunk).
 * ════════════════════════════════════════════════���═══════════════════ */
__global__ void bench_sts128(uint32_t* cycles_out, int nwarps) {
    extern __shared__ __align__(128) uint8_t smem[];

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (warp >= nwarps) return;

    /* Each warp gets its own 4096B region (32 rows × 128B) */
    uint32_t base = (uint32_t)(uintptr_t)__cvta_generic_to_shared(smem)
                  + warp * 4096 + lane * 128;
    uint32_t xor_val = (lane & 7) << 4;
    uint32_t sw0 = 0 ^ xor_val;
    uint32_t sw1 = 16 ^ xor_val;
    uint32_t sw2 = 32 ^ xor_val;
    uint32_t sw3 = 48 ^ xor_val;

    uint32_t r0 = lane, r1 = lane + 32, r2 = lane + 64, r3 = lane + 96;

    /* Warmup */
    for (int i = 0; i < WARMUP; i++) {
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(base+sw0), "r"(r0),"r"(r1),"r"(r2),"r"(r3) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(base+sw1), "r"(r0),"r"(r1),"r"(r2),"r"(r3) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(base+sw2), "r"(r0),"r"(r1),"r"(r2),"r"(r3) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(base+sw3), "r"(r0),"r"(r1),"r"(r2),"r"(r3) : "memory");
    }
    __syncthreads();

    uint32_t t0, t1;
    asm volatile("bar.sync 0; mov.u32 %0, %%clock;" : "=r"(t0));

    for (int i = 0; i < REPS; i++) {
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(base+sw0), "r"(r0),"r"(r1),"r"(r2),"r"(r3) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(base+sw1), "r"(r0),"r"(r1),"r"(r2),"r"(r3) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(base+sw2), "r"(r0),"r"(r1),"r"(r2),"r"(r3) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(base+sw3), "r"(r0),"r"(r1),"r"(r2),"r"(r3) : "memory");
    }

    asm volatile("bar.sync 0; mov.u32 %0, %%clock;" : "=r"(t1));

    if (lane == 0)
        cycles_out[warp] = t1 - t0;
}

__global__ void bench_stsm_nontrans(uint32_t* cycles_out, int nwarps) {
    extern __shared__ __align__(128) uint8_t smem[];

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (warp >= nwarps) return;

    int tile = lane / 8;
    int row  = lane % 8;

    /* Each warp gets 4 × 128B tiles = 512B per stmatrix call, 4 calls → 2048B */
    uint32_t warp_base = (uint32_t)(uintptr_t)__cvta_generic_to_shared(smem) + warp * 2048;

    /* 4 stmatrix calls at offsets 0, 512, 1024, 1536 within warp region */
    uint32_t addr0 = warp_base + 0    + tile * 128 + row * 16;
    uint32_t addr1 = warp_base + 512  + tile * 128 + row * 16;
    uint32_t addr2 = warp_base + 1024 + tile * 128 + row * 16;
    uint32_t addr3 = warp_base + 1536 + tile * 128 + row * 16;

    uint32_t r0 = lane, r1 = lane + 32, r2 = lane + 64, r3 = lane + 96;

    /* Warmup */
    for (int i = 0; i < WARMUP; i++) {
        asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1,%2,%3,%4};" :: "r"(addr0), "r"(r0),"r"(r1),"r"(r2),"r"(r3));
        asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1,%2,%3,%4};" :: "r"(addr1), "r"(r0),"r"(r1),"r"(r2),"r"(r3));
        asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1,%2,%3,%4};" :: "r"(addr2), "r"(r0),"r"(r1),"r"(r2),"r"(r3));
        asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1,%2,%3,%4};" :: "r"(addr3), "r"(r0),"r"(r1),"r"(r2),"r"(r3));
    }
    __syncthreads();

    uint32_t t0, t1;
    asm volatile("bar.sync 0; mov.u32 %0, %%clock;" : "=r"(t0));

    for (int i = 0; i < REPS; i++) {
        asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1,%2,%3,%4};" :: "r"(addr0), "r"(r0),"r"(r1),"r"(r2),"r"(r3));
        asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1,%2,%3,%4};" :: "r"(addr1), "r"(r0),"r"(r1),"r"(r2),"r"(r3));
        asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1,%2,%3,%4};" :: "r"(addr2), "r"(r0),"r"(r1),"r"(r2),"r"(r3));
        asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1,%2,%3,%4};" :: "r"(addr3), "r"(r0),"r"(r1),"r"(r2),"r"(r3));
    }

    asm volatile("bar.sync 0; mov.u32 %0, %%clock;" : "=r"(t1));

    if (lane == 0)
        cycles_out[warp] = t1 - t0;
}

__global__ void bench_stsm_trans(uint32_t* cycles_out, int nwarps) {
    extern __shared__ __align__(128) uint8_t smem[];

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (warp >= nwarps) return;

    int tile = lane / 8;
    int row  = lane % 8;

    uint32_t warp_base = (uint32_t)(uintptr_t)__cvta_generic_to_shared(smem) + warp * 2048;
    uint32_t addr0 = warp_base + 0    + tile * 128 + row * 16;
    uint32_t addr1 = warp_base + 512  + tile * 128 + row * 16;
    uint32_t addr2 = warp_base + 1024 + tile * 128 + row * 16;
    uint32_t addr3 = warp_base + 1536 + tile * 128 + row * 16;

    uint32_t r0 = lane, r1 = lane + 32, r2 = lane + 64, r3 = lane + 96;

    for (int i = 0; i < WARMUP; i++) {
        asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1,%2,%3,%4};" :: "r"(addr0), "r"(r0),"r"(r1),"r"(r2),"r"(r3));
        asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1,%2,%3,%4};" :: "r"(addr1), "r"(r0),"r"(r1),"r"(r2),"r"(r3));
        asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1,%2,%3,%4};" :: "r"(addr2), "r"(r0),"r"(r1),"r"(r2),"r"(r3));
        asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1,%2,%3,%4};" :: "r"(addr3), "r"(r0),"r"(r1),"r"(r2),"r"(r3));
    }
    __syncthreads();

    uint32_t t0, t1;
    asm volatile("bar.sync 0; mov.u32 %0, %%clock;" : "=r"(t0));

    for (int i = 0; i < REPS; i++) {
        asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1,%2,%3,%4};" :: "r"(addr0), "r"(r0),"r"(r1),"r"(r2),"r"(r3));
        asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1,%2,%3,%4};" :: "r"(addr1), "r"(r0),"r"(r1),"r"(r2),"r"(r3));
        asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1,%2,%3,%4};" :: "r"(addr2), "r"(r0),"r"(r1),"r"(r2),"r"(r3));
        asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1,%2,%3,%4};" :: "r"(addr3), "r"(r0),"r"(r1),"r"(r2),"r"(r3));
    }

    asm volatile("bar.sync 0; mov.u32 %0, %%clock;" : "=r"(t1));

    if (lane == 0)
        cycles_out[warp] = t1 - t0;
}

/* ════════════════════════════════════════════════════════════════════
 * Test F/G: Epilogue-realistic contention
 *
 * Our actual epilogue per group: LDS×1 → F2FP×4 → HADD2×8 → STS×1
 * 4 groups per iteration = 4 LDS + 16 F2FP + 32 HADD2 + 4 STS
 *
 * Test F: BF16 compute chain (short — our kernel)
 * Test G: FP32 compute chain (long — CUTLASS approach, more interleave room)
 *
 * Each variant tested with STS.128 and stmatrix, at 4 warps (our epilogue).
 * ════════════════════════════════════════════════════════════════════ */

/* Helper macro: BF16 epilogue compute chain (LDS → F2FP → HADD2×2 → store) */
#define EPI_BF16_COMPUTE(lds_addr) \
    uint32_t d0, d1, d2, d3; \
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" \
        : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3) : "r"(lds_addr)); \
    uint32_t o0, o1, o2, o3; \
    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o0) : "f"(__int_as_float(d0)), "f"(__int_as_float(d1))); \
    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o1) : "f"(__int_as_float(d0)), "f"(__int_as_float(d1))); \
    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o2) : "f"(__int_as_float(d2)), "f"(__int_as_float(d3))); \
    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o3) : "f"(__int_as_float(d2)), "f"(__int_as_float(d3))); \
    asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(o0) : "r"(d0)); \
    asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(o1) : "r"(d1)); \
    asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(o2) : "r"(d2)); \
    asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(o3) : "r"(d3)); \
    asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(o0) : "r"(d0)); \
    asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(o1) : "r"(d1)); \
    asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(o2) : "r"(d2)); \
    asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(o3) : "r"(d3));

/* Helper macro: FP32 epilogue compute chain (LDS → FADD×8 → F2FP → HADD2 → store) */
#define EPI_FP32_COMPUTE(lds_addr) \
    uint32_t d0, d1, d2, d3; \
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" \
        : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3) : "r"(lds_addr)); \
    float f0 = __int_as_float(d0), f1 = __int_as_float(d1); \
    float f2 = __int_as_float(d2), f3 = __int_as_float(d3); \
    float f4 = f0, f5 = f1, f6 = f2, f7 = f3; \
    asm volatile("add.rn.f32 %0, %0, %1;" : "+f"(f0) : "f"(f4)); \
    asm volatile("add.rn.f32 %0, %0, %1;" : "+f"(f1) : "f"(f5)); \
    asm volatile("add.rn.f32 %0, %0, %1;" : "+f"(f2) : "f"(f6)); \
    asm volatile("add.rn.f32 %0, %0, %1;" : "+f"(f3) : "f"(f7)); \
    asm volatile("add.rn.f32 %0, %0, %1;" : "+f"(f4) : "f"(f0)); \
    asm volatile("add.rn.f32 %0, %0, %1;" : "+f"(f5) : "f"(f1)); \
    asm volatile("add.rn.f32 %0, %0, %1;" : "+f"(f6) : "f"(f2)); \
    asm volatile("add.rn.f32 %0, %0, %1;" : "+f"(f7) : "f"(f3)); \
    uint32_t o0, o1, o2, o3; \
    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o0) : "f"(f0), "f"(f1)); \
    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o1) : "f"(f2), "f"(f3)); \
    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o2) : "f"(f4), "f"(f5)); \
    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o3) : "f"(f6), "f"(f7)); \
    asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(o0) : "r"(d0)); \
    asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(o1) : "r"(d1)); \
    asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(o2) : "r"(d2)); \
    asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(o3) : "r"(d3));

/* Shared init helper for epilogue benches */
#define EPI_INIT_LDS_SOURCE(smem, warp_off, lane) do { \
    if (lane < 16) { \
        uint32_t ia_ = (uint32_t)(uintptr_t)__cvta_generic_to_shared(smem) \
                     + warp_off + lane * 16; \
        uint32_t v_ = lane + 1; \
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" \
            :: "r"(ia_), "r"(v_),"r"(v_),"r"(v_),"r"(v_)); \
    } \
} while(0)

/* ── BF16 + STS.128 (our baseline) ── */
__global__ void bench_epi_bf16_sts(uint32_t* cycles_out, int nwarps) {
    extern __shared__ __align__(128) uint8_t smem[];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (warp >= nwarps) return;

    /* 4KB store + 256B LDS source per warp */
    uint32_t warp_off = warp * 4352;
    uint32_t store_base = (uint32_t)(uintptr_t)__cvta_generic_to_shared(smem)
                        + warp_off + lane * 128;
    uint32_t lds_base   = (uint32_t)(uintptr_t)__cvta_generic_to_shared(smem)
                        + warp_off + 4096;
    uint32_t xor_val = (lane & 7) << 4;
    uint32_t sws[4] = {0^xor_val, 16^xor_val, 32^xor_val, 48^xor_val};

    EPI_INIT_LDS_SOURCE(smem, warp_off + 4096, lane);
    __syncthreads();

    for (int i = 0; i < WARMUP; i++) {
        #pragma unroll 1
        for (int g = 0; g < 4; g++) {
            EPI_BF16_COMPUTE(lds_base);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                :: "r"(store_base + sws[g]), "r"(o0),"r"(o1),"r"(o2),"r"(o3) : "memory");
        }
    }
    __syncthreads();

    uint32_t t0, t1;
    asm volatile("bar.sync 0; mov.u32 %0, %%clock;" : "=r"(t0));
    for (int i = 0; i < REPS; i++) {
        #pragma unroll 1
        for (int g = 0; g < 4; g++) {
            EPI_BF16_COMPUTE(lds_base);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                :: "r"(store_base + sws[g]), "r"(o0),"r"(o1),"r"(o2),"r"(o3) : "memory");
        }
    }
    asm volatile("bar.sync 0; mov.u32 %0, %%clock;" : "=r"(t1));
    if (lane == 0) cycles_out[warp] = t1 - t0;
}

/* ── BF16 + stmatrix (proposed replacement) ── */
__global__ void bench_epi_bf16_stsm(uint32_t* cycles_out, int nwarps) {
    extern __shared__ __align__(128) uint8_t smem[];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (warp >= nwarps) return;

    int tile = lane / 8, row = lane % 8;
    uint32_t warp_off = warp * 2304; /* 2KB store + 256B LDS source */
    uint32_t stsm_base = (uint32_t)(uintptr_t)__cvta_generic_to_shared(smem) + warp_off;
    uint32_t lds_base  = (uint32_t)(uintptr_t)__cvta_generic_to_shared(smem) + warp_off + 2048;
    uint32_t addrs[4] = {
        stsm_base + 0    + tile*128 + row*16,
        stsm_base + 512  + tile*128 + row*16,
        stsm_base + 1024 + tile*128 + row*16,
        stsm_base + 1536 + tile*128 + row*16,
    };

    EPI_INIT_LDS_SOURCE(smem, warp_off + 2048, lane);
    __syncthreads();

    for (int i = 0; i < WARMUP; i++) {
        #pragma unroll 1
        for (int g = 0; g < 4; g++) {
            EPI_BF16_COMPUTE(lds_base);
            asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1,%2,%3,%4};"
                :: "r"(addrs[g]), "r"(o0),"r"(o1),"r"(o2),"r"(o3));
        }
    }
    __syncthreads();

    uint32_t t0, t1;
    asm volatile("bar.sync 0; mov.u32 %0, %%clock;" : "=r"(t0));
    for (int i = 0; i < REPS; i++) {
        #pragma unroll 1
        for (int g = 0; g < 4; g++) {
            EPI_BF16_COMPUTE(lds_base);
            asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1,%2,%3,%4};"
                :: "r"(addrs[g]), "r"(o0),"r"(o1),"r"(o2),"r"(o3));
        }
    }
    asm volatile("bar.sync 0; mov.u32 %0, %%clock;" : "=r"(t1));
    if (lane == 0) cycles_out[warp] = t1 - t0;
}

/* ── FP32 + STS.128 (longer compute chain, CUTLASS-like) ── */
__global__ void bench_epi_fp32_sts(uint32_t* cycles_out, int nwarps) {
    extern __shared__ __align__(128) uint8_t smem[];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (warp >= nwarps) return;

    uint32_t warp_off = warp * 4352;
    uint32_t store_base = (uint32_t)(uintptr_t)__cvta_generic_to_shared(smem)
                        + warp_off + lane * 128;
    uint32_t lds_base   = (uint32_t)(uintptr_t)__cvta_generic_to_shared(smem)
                        + warp_off + 4096;
    uint32_t xor_val = (lane & 7) << 4;
    uint32_t sws[4] = {0^xor_val, 16^xor_val, 32^xor_val, 48^xor_val};

    EPI_INIT_LDS_SOURCE(smem, warp_off + 4096, lane);
    __syncthreads();

    for (int i = 0; i < WARMUP; i++) {
        #pragma unroll 1
        for (int g = 0; g < 4; g++) {
            EPI_FP32_COMPUTE(lds_base);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                :: "r"(store_base + sws[g]), "r"(o0),"r"(o1),"r"(o2),"r"(o3) : "memory");
        }
    }
    __syncthreads();

    uint32_t t0, t1;
    asm volatile("bar.sync 0; mov.u32 %0, %%clock;" : "=r"(t0));
    for (int i = 0; i < REPS; i++) {
        #pragma unroll 1
        for (int g = 0; g < 4; g++) {
            EPI_FP32_COMPUTE(lds_base);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                :: "r"(store_base + sws[g]), "r"(o0),"r"(o1),"r"(o2),"r"(o3) : "memory");
        }
    }
    asm volatile("bar.sync 0; mov.u32 %0, %%clock;" : "=r"(t1));
    if (lane == 0) cycles_out[warp] = t1 - t0;
}

/* ── FP32 + stmatrix (CUTLASS-like: long compute + STSM) ── */
__global__ void bench_epi_fp32_stsm(uint32_t* cycles_out, int nwarps) {
    extern __shared__ __align__(128) uint8_t smem[];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (warp >= nwarps) return;

    int tile = lane / 8, row = lane % 8;
    uint32_t warp_off = warp * 2304;
    uint32_t stsm_base = (uint32_t)(uintptr_t)__cvta_generic_to_shared(smem) + warp_off;
    uint32_t lds_base  = (uint32_t)(uintptr_t)__cvta_generic_to_shared(smem) + warp_off + 2048;
    uint32_t addrs[4] = {
        stsm_base + 0    + tile*128 + row*16,
        stsm_base + 512  + tile*128 + row*16,
        stsm_base + 1024 + tile*128 + row*16,
        stsm_base + 1536 + tile*128 + row*16,
    };

    EPI_INIT_LDS_SOURCE(smem, warp_off + 2048, lane);
    __syncthreads();

    for (int i = 0; i < WARMUP; i++) {
        #pragma unroll 1
        for (int g = 0; g < 4; g++) {
            EPI_FP32_COMPUTE(lds_base);
            asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1,%2,%3,%4};"
                :: "r"(addrs[g]), "r"(o0),"r"(o1),"r"(o2),"r"(o3));
        }
    }
    __syncthreads();

    uint32_t t0, t1;
    asm volatile("bar.sync 0; mov.u32 %0, %%clock;" : "=r"(t0));
    for (int i = 0; i < REPS; i++) {
        #pragma unroll 1
        for (int g = 0; g < 4; g++) {
            EPI_FP32_COMPUTE(lds_base);
            asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1,%2,%3,%4};"
                :: "r"(addrs[g]), "r"(o0),"r"(o1),"r"(o2),"r"(o3));
        }
    }
    asm volatile("bar.sync 0; mov.u32 %0, %%clock;" : "=r"(t1));
    if (lane == 0) cycles_out[warp] = t1 - t0;
}

/* ════════════════════════════════════════════════════════════════════ */

void print_layout(const char* name, const uint16_t* data, int total_vals,
                  int rows, int cols)
{
    printf("\n=== %s ===\n", name);
    printf("Format: L<lane>E<elem>  (value = lane*32 + elem)\n");
    printf("DEAD = untouched\n\n");

    for (int r = 0; r < rows; r++) {
        printf("  row %2d (byte %4d): ", r, r * cols * 2);
        for (int c = 0; c < cols; c++) {
            int idx = r * cols + c;
            if (idx >= total_vals) { printf("  OOB  "); continue; }
            uint16_t v = data[idx];
            if (v == 0xDEAD)
                printf("  __   ");
            else
                printf("L%02dE%d ", v / 32, v % 32);
        }
        printf("\n");
    }
}

void run_throughput(const char* name, void(*kern)(uint32_t*,int), int nwarps,
                    int smem_bytes)
{
    uint32_t h_cycles[8] = {};
    uint32_t* d_cycles;
    CUDA_CHECK(cudaMalloc(&d_cycles, 8 * sizeof(uint32_t)));

    kern<<<1, nwarps * 32, smem_bytes>>>(d_cycles, nwarps);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_cycles, d_cycles, nwarps * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    /* Find max across warps (= wall-clock for all warps) */
    uint32_t maxc = 0;
    for (int w = 0; w < nwarps; w++)
        if (h_cycles[w] > maxc) maxc = h_cycles[w];

    float per_iter = (float)maxc / REPS;
    float per_store = per_iter / 4.0f;  /* 4 stores per iter */
    printf("  %s  %dW: %8.1f cyc/iter  %6.1f cyc/store  (max warp: %u total)\n",
           name, nwarps, per_iter, per_store, maxc);

    cudaFree(d_cycles);
}

int main() {
    CU_CHECK(cuInit(0));
    printf("stmatrix layout + throughput benchmark (SM100a)\n");
    printf("================================================\n");

    /* ── Layout tests ── */
    uint16_t* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, 4096 * sizeof(uint16_t)));

    /* Test A: stmatrix non-transposed */
    {
        uint16_t h_out[256];
        CUDA_CHECK(cudaMemset(d_out, 0, 256 * sizeof(uint16_t)));
        test_stsm_nontrans<<<1, 32>>>(d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out, d_out, 256 * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        /* 4 tiles × 8 rows × 8 cols = 32 rows × 8 cols, stored as 4 × 128B blocks */
        print_layout("stmatrix x4 m8n8 NON-transposed (512B = 4 tiles × 128B)",
                     h_out, 256, 32, 8);
    }

    /* Test B: stmatrix transposed */
    {
        uint16_t h_out[256];
        CUDA_CHECK(cudaMemset(d_out, 0, 256 * sizeof(uint16_t)));
        test_stsm_trans<<<1, 32>>>(d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out, d_out, 256 * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        print_layout("stmatrix x4 m8n8 TRANSPOSED (512B = 4 tiles × 128B)",
                     h_out, 256, 32, 8);
    }

    /* Test C: STS.128 SWIZZLE_128B reference */
    {
        uint16_t h_out[2048];
        CUDA_CHECK(cudaMemset(d_out, 0, 2048 * sizeof(uint16_t)));
        test_sts128_swizzle<<<1, 32>>>(d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out, d_out, 2048 * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        /* 32 lanes × 128B/lane = 4096B total. Each lane stores 16B at swizzled offset. */
        print_layout("STS.128 SWIZZLE_128B (4096B = 32 lanes × 128B rows)",
                     h_out, 2048, 32, 64);
    }

    /* ── Throughput tests ── */
    printf("\n\n=== Throughput: 4 stores/iter × %d iters ===\n", REPS);
    printf("  (STS.128 stores 16B/thread, STSM stores 512B/warp = same total)\n\n");

    int warp_counts[] = {1, 2, 4, 8};
    for (int wi = 0; wi < 4; wi++) {
        int nw = warp_counts[wi];
        int smem_sts  = nw * 4096;   /* 4KB per warp for STS.128 */
        int smem_stsm = nw * 2048;   /* 2KB per warp for stmatrix */

        run_throughput("STS.128     ", bench_sts128,         nw, smem_sts);
        run_throughput("STSM (norm) ", bench_stsm_nontrans,  nw, smem_stsm);
        run_throughput("STSM (trans)", bench_stsm_trans,      nw, smem_stsm);
        printf("\n");
    }

    /* ── Epilogue-realistic contention (THE key test) ── */
    printf("\n=== Epilogue-realistic: LDS + compute + store, 4 groups/iter × %d iters ===\n", REPS);
    printf("  BF16 chain: LDS.v4 → F2FP×4 → HADD2×8 → store (our kernel)\n");
    printf("  FP32 chain: LDS.v4 → FADD×8 → F2FP×4 → HADD2×4 → store (CUTLASS-like)\n\n");

    {
        int epi_warp_counts[] = {1, 2, 4};
        for (int wi = 0; wi < 3; wi++) {
            int nw = epi_warp_counts[wi];

            run_throughput("BF16+STS.128", bench_epi_bf16_sts,  nw, nw * 4352);
            run_throughput("BF16+STSM   ", bench_epi_bf16_stsm, nw, nw * 2304);
            run_throughput("FP32+STS.128", bench_epi_fp32_sts,  nw, nw * 4352);
            run_throughput("FP32+STSM   ", bench_epi_fp32_stsm, nw, nw * 2304);
            printf("\n");
        }
    }

    cudaFree(d_out);
    printf("Done.\n");
    return 0;
}
