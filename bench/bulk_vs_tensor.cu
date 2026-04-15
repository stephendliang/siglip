/*
cp.async.bulk (1D raw) vs cp.async.bulk.tensor.2d — latency & throughput

Isolates TMA descriptor/swizzle overhead by comparing:
  1. tensor.2d  SWIZZLE_128B  (what our kernel uses)
  2. tensor.2d  SWIZZLE_NONE  (no swizzle, same descriptor path)
  3. raw 1D bulk              (no descriptor at all)

Single CTA, single warp — no cluster complications.
Transfer sizes: 4KB, 8KB, 16KB, 32KB (FC2 A-tile = 16KB).
Pipeline depths: 1 (latency), 2, 4, 6 (FC2 default).

Build: make bulk-vs-tensor && ./bulk-vs-tensor
*/

#include <cstdio>
#include <cstdint>
#include <cuda.h>

#define WARMUP 128
#define REPS   1024

#define CU_CHECK(x) do { CUresult _r = (x); if (_r) { const char *_s; cuGetErrorString(_r, &_s); fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, _s); exit(1); } } while(0)
#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1); } } while(0)

/* Max 6 stages × 32KB = 192KB SMEM. Mbars at the front. */
#define MAX_PIPE    6
#define MAX_XFER    32768
#define MBAR_OFF    0
#define BUF_OFF     512      /* 512B-aligned, after 6×8=48 bytes of mbars */
#define SMEM_BYTES  (BUF_OFF + MAX_PIPE * MAX_XFER)

/* Global tensor: 128 cols × enough rows for any transfer size.
   Using UINT8 to match FC2's FP8 layout. */
#define GCOLS  128
#define GROWS  (MAX_XFER / GCOLS * MAX_PIPE * 4)  /* plenty of rows */

__device__ __forceinline__ long long clk() {
    long long r;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(r) :: "memory");
    return r;
}

__device__ __forceinline__ uint32_t to_smem(const void *p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mb_init(uint32_t a, int n) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(n));
}

__device__ __forceinline__ void mb_expect_tx(uint32_t a, int b) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(a), "r"(b) : "memory");
}

__device__ __forceinline__ void mb_wait(uint32_t a, int ph) {
    uint32_t done;
    do {
        asm volatile(
            "{\n\t.reg .pred p;\n\t"
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2, 0x989680;\n\t"
            "selp.b32 %0, 1, 0, p;\n\t}"
            : "=r"(done) : "r"(a), "r"(ph));
    } while (!done);
}

/* ═══════ Kernel: tensor.2d load (serial + pipelined) ═══════ */

template <int PIPE, int XFER>
__global__ __launch_bounds__(32, 1)
void k_tensor(const __grid_constant__ CUtensorMap tm, long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    uint32_t mb[MAX_PIPE], buf[MAX_PIPE];
    int ph[MAX_PIPE];
    int lane = threadIdx.x;
    int rows = XFER / GCOLS;

    for (int i = 0; i < PIPE; i++) {
        mb[i] = sb + MBAR_OFF + i * 8;
        buf[i] = sb + BUF_OFF + i * XFER;
        ph[i] = 0;
        if (lane == 0) mb_init(mb[i], 1);
    }
    __syncwarp();

    /* warmup */
    for (int i = 0; i < WARMUP; i++) {
        int s = i % PIPE;
        if (lane == 0) {
            mb_expect_tx(mb[s], XFER);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.tile"
                ".mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                :: "r"(buf[s]), "l"(&tm), "r"(0), "r"((i * rows) % GROWS), "r"(mb[s])
                : "memory");
        }
        mb_wait(mb[s], ph[s]); ph[s] ^= 1;
    }

    /* timed */
    long long t0 = clk();
    for (int i = 0; i < REPS; i++) {
        int s = i % PIPE;
        if (lane == 0) {
            mb_expect_tx(mb[s], XFER);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.tile"
                ".mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                :: "r"(buf[s]), "l"(&tm), "r"(0), "r"((i * rows) % GROWS), "r"(mb[s])
                : "memory");
        }
        mb_wait(mb[s], ph[s]); ph[s] ^= 1;
    }
    long long t1 = clk();
    if (lane == 0) *out = t1 - t0;
}

/* ═══════ Kernel: raw 1D bulk copy (serial + pipelined) ═══════ */

template <int PIPE, int XFER>
__global__ __launch_bounds__(32, 1)
void k_bulk(const uint8_t *src, long long *out) {
    extern __shared__ __align__(128) char smem[];
    uint32_t sb = to_smem(smem);
    uint32_t mb[MAX_PIPE], buf_addr[MAX_PIPE];
    int ph[MAX_PIPE];
    int lane = threadIdx.x;

    for (int i = 0; i < PIPE; i++) {
        mb[i] = sb + MBAR_OFF + i * 8;
        buf_addr[i] = sb + BUF_OFF + i * XFER;
        ph[i] = 0;
        if (lane == 0) mb_init(mb[i], 1);
    }
    __syncwarp();

    /* warmup */
    for (int i = 0; i < WARMUP; i++) {
        int s = i % PIPE;
        if (lane == 0) {
            uint64_t gaddr = (uint64_t)src + (uint64_t)(i % 64) * XFER;
            mb_expect_tx(mb[s], XFER);
            asm volatile(
                "cp.async.bulk.shared::cta.global"
                ".mbarrier::complete_tx::bytes"
                " [%0], [%1], %2, [%3];"
                :: "r"(buf_addr[s]), "l"(gaddr), "r"(XFER), "r"(mb[s])
                : "memory");
        }
        mb_wait(mb[s], ph[s]); ph[s] ^= 1;
    }

    /* timed */
    long long t0 = clk();
    for (int i = 0; i < REPS; i++) {
        int s = i % PIPE;
        if (lane == 0) {
            uint64_t gaddr = (uint64_t)src + (uint64_t)(i % 64) * XFER;
            mb_expect_tx(mb[s], XFER);
            asm volatile(
                "cp.async.bulk.shared::cta.global"
                ".mbarrier::complete_tx::bytes"
                " [%0], [%1], %2, [%3];"
                :: "r"(buf_addr[s]), "l"(gaddr), "r"(XFER), "r"(mb[s])
                : "memory");
        }
        mb_wait(mb[s], ph[s]); ph[s] ^= 1;
    }
    long long t1 = clk();
    if (lane == 0) *out = t1 - t0;
}

/* ═══════ Host ═══════ */

void make_tmap(CUtensorMap *tm, uint8_t *ptr, CUtensorMapSwizzle swizzle) {
    uint64_t dims[2]    = {GCOLS, GROWS};
    uint64_t strides[1] = {GCOLS};
    uint32_t box[2]     = {GCOLS, MAX_XFER / GCOLS};  /* will re-create per xfer size */
    uint32_t estrides[2]= {1, 1};
    CU_CHECK(cuTensorMapEncodeTiled(tm,
        CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)ptr,
        dims, strides, box, estrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

void make_tmap_sized(CUtensorMap *tm, uint8_t *ptr, CUtensorMapSwizzle swizzle, int xfer) {
    int rows = xfer / GCOLS;
    uint64_t dims[2]    = {GCOLS, GROWS};
    uint64_t strides[1] = {GCOLS};
    uint32_t box[2]     = {(uint32_t)GCOLS, (uint32_t)rows};
    uint32_t estrides[2]= {1, 1};
    CU_CHECK(cuTensorMapEncodeTiled(tm,
        CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)ptr,
        dims, strides, box, estrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

template <int PIPE, int XFER>
float run_tensor(CUtensorMap &tm, long long *d_out) {
    k_tensor<PIPE, XFER><<<1, 32, SMEM_BYTES>>>(tm, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    long long h;
    CUDA_CHECK(cudaMemcpy(&h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
    return (float)h / REPS;
}

template <int PIPE, int XFER>
float run_bulk(uint8_t *d_src, long long *d_out) {
    k_bulk<PIPE, XFER><<<1, 32, SMEM_BYTES>>>(d_src, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    long long h;
    CUDA_CHECK(cudaMemcpy(&h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
    return (float)h / REPS;
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFuncSetAttribute(k_tensor<1, 4096>,  cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_tensor<1, 8192>,  cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_tensor<1, 16384>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_tensor<1, 32768>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_tensor<2, 4096>,  cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_tensor<2, 8192>,  cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_tensor<2, 16384>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_tensor<2, 32768>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_tensor<4, 4096>,  cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_tensor<4, 8192>,  cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_tensor<4, 16384>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_tensor<4, 32768>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_tensor<6, 4096>,  cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_tensor<6, 8192>,  cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_tensor<6, 16384>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_tensor<6, 32768>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_bulk<1, 4096>,  cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_bulk<1, 8192>,  cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_bulk<1, 16384>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_bulk<1, 32768>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_bulk<2, 4096>,  cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_bulk<2, 8192>,  cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_bulk<2, 16384>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_bulk<2, 32768>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_bulk<4, 4096>,  cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_bulk<4, 8192>,  cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_bulk<4, 16384>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_bulk<4, 32768>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_bulk<6, 4096>,  cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_bulk<6, 8192>,  cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_bulk<6, 16384>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(k_bulk<6, 32768>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));

    /* Allocate global data */
    size_t data_bytes = (size_t)GCOLS * GROWS;
    uint8_t *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMemset(d_data, 0x42, data_bytes));

    long long *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(long long)));

    printf("cp.async.bulk (1D raw) vs cp.async.bulk.tensor.2d\n");
    printf("  Global tensor: %d × %d UINT8 (%.1f MB)\n", GCOLS, GROWS, data_bytes / 1e6);
    printf("  SMEM: %d bytes, REPS=%d, WARMUP=%d\n\n", SMEM_BYTES, REPS, WARMUP);

    /* Create tensor maps for each transfer size */
    struct { int xfer; const char *label; } sizes[] = {
        {4096,  " 4KB"}, {8192,  " 8KB"}, {16384, "16KB"}, {32768, "32KB"}
    };
    int pipes[] = {1, 2, 4, 6};

    for (auto &sz : sizes) {
        CUtensorMap tm_swz128, tm_none;
        make_tmap_sized(&tm_swz128, d_data, CU_TENSOR_MAP_SWIZZLE_128B, sz.xfer);
        make_tmap_sized(&tm_none,   d_data, CU_TENSOR_MAP_SWIZZLE_NONE, sz.xfer);

        printf("═══ %s per load ═══\n", sz.label);
        printf("  %-7s  %10s  %10s  %10s\n", "pipe", "tensor+S128", "tensor+NONE", "raw_1D");
        printf("  %-7s  %10s  %10s  %10s\n", "----", "-----------", "-----------", "------");

        for (int p : pipes) {
            float cyc_s128 = 0, cyc_none = 0, cyc_bulk = 0;

            /* Dispatch by template params — ugly but necessary */
            #define CASE(P, X) \
                if (p == P && sz.xfer == X) { \
                    cyc_s128 = run_tensor<P, X>(tm_swz128, d_out); \
                    cyc_none = run_tensor<P, X>(tm_none, d_out); \
                    cyc_bulk = run_bulk<P, X>(d_data, d_out); \
                }
            CASE(1, 4096)  CASE(1, 8192)  CASE(1, 16384)  CASE(1, 32768)
            CASE(2, 4096)  CASE(2, 8192)  CASE(2, 16384)  CASE(2, 32768)
            CASE(4, 4096)  CASE(4, 8192)  CASE(4, 16384)  CASE(4, 32768)
            CASE(6, 4096)  CASE(6, 8192)  CASE(6, 16384)  CASE(6, 32768)
            #undef CASE

            printf("  pipe=%-2d  %8.1f cy  %8.1f cy  %8.1f cy\n",
                   p, cyc_s128, cyc_none, cyc_bulk);
        }
        printf("\n");
    }

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
