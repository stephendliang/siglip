// Minimal tcgen05 TMEM alloc/dealloc test
#include <cstdio>
#include <cstdint>

__device__ __forceinline__
uint32_t smem_to_uint(const void* p) {
    uint32_t r;
    asm volatile("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }"
        : "=r"(r) : "l"(p));
    return r;
}

__global__ void __launch_bounds__(128, 1)
test_tmem() {
    __shared__ __align__(128) char smem[256];
    const int tid = threadIdx.x;
    const int warp = tid / 32;

    // Store TMEM addresses in smem
    uint32_t* tmem_addr0 = (uint32_t*)(smem + 0);
    uint32_t* tmem_addr1 = (uint32_t*)(smem + 4);

    // Test 1: Single alloc (warp 0), no relinquish
    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(smem_to_uint(tmem_addr0)), "r"(128));
    }
    __syncthreads();

    if (tid == 0)
        printf("Block %d: alloc 1 OK, tmem_addr = %u\n", blockIdx.x, *tmem_addr0);

    // Test 2: Second alloc (warp 0)
    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(smem_to_uint(tmem_addr1)), "r"(128));
    }
    __syncthreads();

    if (tid == 0)
        printf("Block %d: alloc 2 OK, tmem_addr = %u, %u\n", blockIdx.x, *tmem_addr0, *tmem_addr1);

    // Dealloc both (warp 0 only)
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :: "r"(*tmem_addr0), "r"(128));
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :: "r"(*tmem_addr1), "r"(128));
    }

    if (tid == 0)
        printf("Block %d: dealloc OK\n", blockIdx.x);
}

int main() {
    printf("Testing tcgen05 TMEM alloc/dealloc...\n");
    test_tmem<<<2, 128>>>();
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("FAILED: %s\n", cudaGetErrorString(e));
        return 1;
    }
    printf("SUCCESS\n");
    return 0;
}
