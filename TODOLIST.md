Here's the complete action list, covering everything in the epilogue that needs to change:

## Complete Action List

### 1. Replace `__syncthreads` with mbarriers (line 394-397)

The `tcgen05.fence::before + __syncthreads + tcgen05.fence::after` forces all 6 warps to rendezvous at every tile boundary. Replace with the mbarrier protocol from SUPERIOR.cu:
- **mainloop_mbar**: MMA warp (W1) signals via `tcgen05.commit` when K-loop finishes → epilogue warps wait on it
- **epilogue_mbar**: Epilogue warps `mbarrier_arrive` when done → MMA warp waits on it before next tile
- **mma_mbar**: Already exists — W1 signals after each K-iter → W0 waits before reusing SMEM stage

This fully decouples the 3 warp groups. No warp ever waits for a group it doesn't depend on.

### 2. Replace TMA stores with `st.global.v8.b32` (lines 375-384)

The current inner loop per-iteration:
```
CVT_STS → tma_store_fence → __syncwarp → lane==0 { tma_store_wait_1, tma_store_2d, tma_store_commit }
```
This serializes the loop: `__syncwarp` blocks all 32 lanes, `tma_store_wait_1` blocks lane 0, and lanes 1-31 diverge off. Replace with:
```
CVT + st.global.v8.b32   (all 32 lanes store independently)
```
Eliminates 5 instructions/barriers from the inner loop. No fence, no syncwarp, no wait, no lane-0 divergence. Every lane contributes every cycle.

### 3. Make W0/W1 join epilogue after K-loop

Currently W0 (TMA) and W1 (MMA) are stalled on mbarriers while W2-5 run the epilogue. That's **4-way** TMEM interleaving. After W0 finishes its last TMA load and W1 finishes its last `tcgen05.commit`, they should fall through into the epilogue path — giving **6-way** TMEM interleaving. 50% more warps hiding the ~200-cycle LDTM latency.

The drain path (line 402) already proves this works: `if (warp < 4)` has W0-W3 all doing epilogue. Generalize this to the overlapped path. Each warp handles `128 / 6 ≈ 21-22` rows instead of 32.

### 4. Prefetch bias + pos_embed into SMEM during K-loop (lines 357-364)

Currently every epilogue iteration issues **32 LDG instructions** (16 floats bias + 16 floats pos_embed) per warp. All 32 lanes load the **same** 16 addresses (broadcast). That's 128B of global memory reads per warp per iteration, competing with TMEM reads for the memory pipeline.

Fix: W2-5 are idle during the K-loop. Use them to `cp.async` or plain-load bias (128 floats = 512B) and pos_embed (128 floats × 32 rows = 16KB per warp, 64KB total) into SMEM. Then the epilogue inner loop reads from SMEM — zero LDG, zero L2 pressure.

### 5. Delete TMA store SMEM buffers (OFF_TMA_BUF = 196608, 8KB)

`OFF_TMA_BUF` allocates 8 double-buffered SMEM regions (4 warps × 2 × 1024B) used as staging for TMA stores. After #2 (st.global), these are dead. Reclaim the 8KB for bias/pos prefetch buffers from #4.

### 6. Delete `tma_c` descriptor and host-side TMA store setup

After #2, TMA stores are gone. Remove:
- `tma_c` kernel parameter (line 193)
- `tma_store_2d`, `tma_store_fence`, `tma_store_commit`, `tma_store_wait_0/1` helper functions (lines 118-146)
- Host-side `h_tma_c` TensorMap creation (lines 540-553)
- All `smem_base[]`/`smem_lane[]` address computation (lines 342-346, 418-422)

### 7. Unify overlapped + drain epilogue into one function

The overlapped epilogue (lines 327-391) and drain epilogue (lines 402-467) are near-identical copies. Extract into a `__device__ __forceinline__` function that takes `(tmem_base, row_start, col_start, bias_ptr, pos_ptr, C_ptr, N)`. Reduces code size, simplifies maintenance, makes #2/#3/#4 changes single-site.

---

**Dependency order**: #7 first (refactor), then #2 + #6 (kill TMA stores), then #5 (reclaim SMEM), then #1 (mbarriers), then #4 (prefetch), then #3 (6-way interleaving). Items #2+#6 and #5 can be done together. #1 and #3 are tightly coupled.
