## Action List

### 1. Replace `__syncthreads()` with mbarrier decoupling (line 394-397)
**Delete:**
```
asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
__syncthreads();
asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
```
**Replace with:** SUPERIOR.cu's pattern — epilogue warps wait on `mainloop_mbar`, signal `epilogue_mbar` when done. MMA warp waits on `epilogue_mbar` before next tile. TMA warp waits on `mma_mbar` only. Three independent stages.

### 2. Switch TMA stores back to st.global (lines 375-384)
**Delete:** CVT_STS + tma_store_fence + __syncwarp + tma_store_wait_1 + tma_store_2d + tma_store_commit + double-buffered SMEM staging.
**Replace with:**
```
CVT to bf16x2 in registers → st.global.v8.b32 [C_ptr + offset], {b0..b7}
```
Fire-and-forget. Zero sync points per chunk. Kills all SMEM staging, fences, syncwarps, TMA waits.

### 3. Prefetch bias+pos into SMEM during K-loop (lines 357-364)
**Currently:** 32 LDG per chunk inside epilogue inner loop.
**Fix:** Epilogue warps (W2-5) are idle during first_tile's K-loop. Use them to TMA-load or cooperatively LDG bias (512B, trivial) and pos_embed (2KB/warp/chunk) into SMEM. Then epilogue reads from SMEM via LDS instead of LDG. Removes all global memory reads from the critical inner loop.

### 4. Remove TMA store SMEM buffers (OFF_TMA_BUF, lines 342-346)
Once you do #2, the 8KB of TMA store staging buffers (4 warps × 2 buffers × 1024B) are freed. Use that SMEM for bias/pos prefetch from #3.
