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

### 3. Software-pipeline TMEM loads across column iterations (lines 349-387)

The overlapped epilogue inner loop is serial per column chunk:
```
for nc = 0..TN step 16:
    tcgen05.ld(col=nc)         // issue async TMEM load
    bias+pos math              // ~16 FADDs hiding latency
    TMEM_WAIT                  // STALL — 16 FADDs aren't enough to cover ~200 cycles
    FADD (apply bias+pos)      // 16 FADDs
    CVT + store                // convert FP32→BF16, write to global
```

The `TMEM_WAIT` stalls because each `tcgen05.ld.sync.aligned.32x32b.x16` has ~200-cycle latency but there's only ~16 instructions of independent work between issue and wait. The 4-way warp interleaving (W2-W5) helps — the scheduler can run another warp while one is stalled — but it's not enough if all 4 warps hit their `TMEM_WAIT` before any load completes.

**Fix**: Overlap adjacent column iterations by issuing the next `tcgen05.ld` before waiting on the current one:
```
tcgen05.ld(col=0)                // prefetch first chunk
for nc = 0..TN-16:
    bias+pos math(nc)            // independent work
    TMEM_WAIT                    // wait for current chunk
    tcgen05.ld(col=nc+16)        // issue NEXT chunk immediately after wait returns
    FADD + CVT + store(nc)       // process current chunk (overlaps with next load)
// handle last chunk
TMEM_WAIT
FADD + CVT + store(last)
```

This doubles the effective overlap window: the load for chunk N+1 is in flight during the entire compute phase of chunk N (~32 instructions of FADD+CVT+store). Combined with 4-way warp interleaving, this should fully hide TMEM latency.

**Note on "6-way" (W0/W1 joining)**: The original plan to have W0/W1 join the overlapped epilogue was wrong — they're **busy** running the K-loop for the next tile during the overlapped epilogue (that's what "overlapped" means). W0/W1 only become free for the drain epilogue (last tile), where they already participate (line 402: `warp < 4` uses W0-W3). Making the drain use all 6 warps (`warp < 6`) is a trivial extension but low-impact since the drain is only 1 tile.

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

**Dependency order**: #7 first (refactor), then #1 (mbarriers), then #2 + #5 + #6 (st.global + dead code removal), then #3 (pipeline TMEM loads — straightforward once #2 simplifies the loop body), then #4 (prefetch — needs SMEM reclaimed by #5).
