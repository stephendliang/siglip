/*
  tma_swizzle_probe.cu — standalone validator for CU_TENSOR_MAP_SWIZZLE_64B

  Mirrors fc2_w3x.cu's epilogue STS chain (4 × st.shared.v4.b32 per lane at
  offsets {0,16,32,48}) writing identity-indexed BF16 markers into an SMEM
  staging, then issues a single TMA store with the target swizzle mode and
  reads back from GMEM to check each element matches its expected marker.

  Build variants (see Makefile):
    tma-swizzle-probe       SWIZZLE_NONE + XOR=0   (baseline — must pass)
    tma-swizzle-probe-sw    SWIZZLE_64B  + XOR = (row & 3) << 4
    tma-swizzle-probe-sw-xN SWIZZLE_64B  + alternate XOR formula N (1..4)
                            for iterating if the default XOR is wrong

  What the probe tells you:
    - PASS  => the STS XOR matches the TMA swizzle => port to fc2_w3x safely.
    - FAIL  => the first mismatches show (row, col) with expected vs got; the
              "chunk-permutation map" section reports, per row mod 4, which
              logical chunk TMA actually read from each physical chunk slot.
              Use that to redesign the XOR (or drop to SWIZZLE_NONE).

  Kernel shape: 8 warps × 32 lanes = 256 threads, 1 block, cta_group::1.
                Staging = 32 BF16 cols × 256 rows × 2 B = 16 KB.
  TMA box:      {32, 256}. GMEM dims: {32, 256}. No multicast, no cluster.
*/

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define ROWS    256
#define COLS    32
#define ROW_BYTES (COLS * 2)
#define STAGE_BYTES (ROWS * ROW_BYTES)

#ifndef XOR_MODE
/*
  XOR_MODE selects the SMEM-side chunk permutation that claims to match the
  TMA descriptor's swizzle. 0 = no XOR (pair with SWIZZLE_NONE). Others are
  candidate formulas for SWIZZLE_64B (documented NVIDIA convention:
  new_chunk = chunk ^ (row & 3), i.e. XOR the chunk-byte offset (0/16/32/48)
  with ((row & 3) << 4).  Alternates cover the common "off by a shift"
  mistakes so a single rebuild picks the winner without editing source.
*/
#define XOR_MODE 0
#endif

#define CUDA_CHECK(x) do { \
    cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } \
} while(0)
#define CU_CHECK(x) do { \
    CUresult r_ = (x); \
    if (r_ != CUDA_SUCCESS) { const char* s_; cuGetErrorString(r_, &s_); \
        fprintf(stderr, "CU %s:%d: %s\n", __FILE__, __LINE__, s_); exit(1); } \
} while(0)

static __device__ __forceinline__
uint32_t smem_to_uint(const void* p) {
    return static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(__cvta_generic_to_shared(p)));
}

static __device__ __forceinline__ uint32_t xor_mask_for(uint32_t row) {
#if XOR_MODE == 0
    (void)row; return 0;
#elif XOR_MODE == 1
    return (row & 3) << 4;
#elif XOR_MODE == 2
    return ((row >> 1) & 3) << 4;
#elif XOR_MODE == 3
    return (row & 7) << 4;
#elif XOR_MODE == 4
    return ((row >> 2) & 3) << 4;
#else
    #error "unknown XOR_MODE"
#endif
}

__global__ __launch_bounds__(256, 1)
void probe_kernel(const __grid_constant__ CUtensorMap tma_c,
                  __nv_bfloat16* /*d_C — TMA handles it*/) {
    extern __shared__ __align__(128) uint8_t smem[];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int row     = warp_id * 32 + lane;

    __nv_bfloat16 vals[32];
    #pragma unroll
    for (int c = 0; c < 32; c++) {
        vals[c] = __float2bfloat16((float)(row * 32 + c));
    }
    const uint32_t* p = reinterpret_cast<const uint32_t*>(vals);

    const uint32_t row_base = smem_to_uint(smem) + row * ROW_BYTES;
    const uint32_t xor_m    = xor_mask_for(row);

    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(row_base + ( 0 ^ xor_m)), "r"(p[0]), "r"(p[1]), "r"(p[2]), "r"(p[3]));
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(row_base + (16 ^ xor_m)), "r"(p[4]), "r"(p[5]), "r"(p[6]), "r"(p[7]));
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(row_base + (32 ^ xor_m)), "r"(p[8]), "r"(p[9]), "r"(p[10]), "r"(p[11]));
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(row_base + (48 ^ xor_m)), "r"(p[12]), "r"(p[13]), "r"(p[14]), "r"(p[15]));

    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    if (tid == 0) {
        const uint32_t src = smem_to_uint(smem);
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
            " [%0, {%1, %2}], [%3];"
            :: "l"(&tma_c), "r"(0), "r"(0), "r"(src) : "memory");
        asm volatile("cp.async.bulk.commit_group;");
        asm volatile("cp.async.bulk.wait_group 0;");
    }
}

int main(int, char**) {
    const char* mode_label;
    int swizzle_is_64b;
#if XOR_MODE == 0
    mode_label = "SWIZZLE_NONE baseline (XOR=0)";
    swizzle_is_64b = 0;
#else
    mode_label = "SWIZZLE_64B candidate";
    swizzle_is_64b = 1;
#endif
    printf("tma_swizzle_probe: %s  XOR_MODE=%d\n", mode_label, XOR_MODE);
    printf("  box = {%d cols, %d rows}  stage = %d B  row stride = %d B\n",
           COLS, ROWS, STAGE_BYTES, ROW_BYTES);

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));

    __nv_bfloat16* d_C = nullptr;
    size_t sC = (size_t)ROWS * COLS;
    CUDA_CHECK(cudaMalloc(&d_C, sC * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(d_C, 0, sC * sizeof(__nv_bfloat16)));

    CUtensorMap h_tma_c;
    {
        uint64_t dims[2]    = {(uint64_t)COLS, (uint64_t)ROWS};
        uint64_t strides[1] = {(uint64_t)COLS * sizeof(__nv_bfloat16)};
        uint32_t box[2]     = {COLS, ROWS};
        uint32_t estrides[2]= {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(&h_tma_c,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)d_C,
            dims, strides, box, estrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle_is_64b ? CU_TENSOR_MAP_SWIZZLE_64B
                           : CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }

    int smem_bytes = STAGE_BYTES + 128;
    CUDA_CHECK(cudaFuncSetAttribute(probe_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    probe_kernel<<<1, 256, smem_bytes>>>(h_tma_c, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    __nv_bfloat16* h_C = (__nv_bfloat16*)malloc(sC * sizeof(__nv_bfloat16));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sC * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    int errors = 0;
    int shown = 0;
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            float expected = (float)(r * 32 + c);
            float got      = __bfloat162float(h_C[r * COLS + c]);
            if (expected != got) {
                errors++;
                if (shown < 8) {
                    printf("  MISMATCH [r=%3d, c=%2d]  expected=%.1f  got=%.1f\n",
                           r, c, expected, got);
                    shown++;
                }
            }
        }
    }

    if (errors > 0) {
        printf("\n  chunk-permutation map — for each r mod 4, decode the physical-chunk\n");
        printf("  assignment actually observed in GMEM. Each row has 4 chunks of 8 BF16.\n");
        printf("  A correctly-matched XOR gives identity [0,1,2,3] for all rows.\n\n");
        for (int rmod = 0; rmod < 4; rmod++) {
            int sample_row = rmod;
            int out_chunk_src[4] = {-1, -1, -1, -1};
            for (int c = 0; c < COLS; c++) {
                float got_f = __bfloat162float(h_C[sample_row * COLS + c]);
                int val = (int)got_f;
                int src_row   = val / 32;
                int src_col   = val - src_row * 32;
                int phys_chunk = c >> 3;
                int logi_chunk = src_col >> 3;
                if (src_row == sample_row && out_chunk_src[phys_chunk] == -1) {
                    out_chunk_src[phys_chunk] = logi_chunk;
                }
            }
            printf("  r=%d (r mod 4 = %d): phys_chunk -> logi_chunk  [%d, %d, %d, %d]\n",
                   sample_row, rmod,
                   out_chunk_src[0], out_chunk_src[1], out_chunk_src[2], out_chunk_src[3]);
        }
        printf("\n  Interpretation: if row r shows [a,b,c,d], TMA reads logical chunk a from\n");
        printf("  physical slot 0, chunk b from slot 1, etc. To make identity match, STS\n");
        printf("  must place logical chunk k into physical slot k' where [k'] = k in the map.\n");
    }

    printf("\n%s: errors=%d/%d\n",
           errors == 0 ? "PASS" : "FAIL", errors, (int)sC);

    free(h_C);
    cudaFree(d_C);
    return errors == 0 ? 0 : 1;
}
