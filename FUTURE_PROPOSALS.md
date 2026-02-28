# Future Proposals

Baseline: **0.630 ms / 1739 TFLOPS**, 222 regs, 0 spills. L1 tex at 85% (primary ceiling).
Reference: cuBLAS GEMM-only = 0.365 ms / 3001 TFLOPS. cuBLAS + unfused pos_embed = 0.835 ms.

**Critical finding (F10/F11):** The kernel is **K-loop-bound, not epilogue-bound**. The epilogue completes within the K-loop's shadow — combined load stalls occur in slack time and don't affect wall clock. Epilogue-only optimizations are unlikely to help unless they also affect K-loop throughput or are large enough to flip the overlap balance (dangerous — see F11's 44% regression when `.cg` made the epilogue the new critical path).

---

## SHOULD TRY

### 2. Offline combined relayout (epilogue-friendly blocked format)

**Problem:** The combined tensor is stored as `[196, 768]` row-major (1536B row stride). Each epilogue warp's 32 lanes read from 32 consecutive position rows at the same column offset — generating 32 scattered cache line accesses per `uint4` load. F2 tried to fix this at runtime with a "Phase 0" coalescing loop and regressed 36% due to the loop's code size and register bloat.

**Insight:** F2 failed because of *runtime* overhead, not because coalesced combined access is a bad idea. The coalescing work can be moved entirely offline into `precompute_combined`, which runs once before the timed kernel.

**Fix:** Rearrange the combined tensor into a blocked layout where each 32-row × 32-col block is stored contiguously (32 × 32 × 2B = 2 KB per block). Within a block, data is ordered so that a warp's 32 lanes reading their respective rows at a given column chunk hit a contiguous 2 KB region instead of 32 scattered 128B cache lines.

**Implementation:** Change `precompute_combined` kernel (trivial, outside timed region) + change epilogue indexing from `pos_row * N_DIM + col` to block-offset calculation (~5 lines).

**Design questions:**
- Block dimensions: 32×32 (matches nc iteration and row_group size) is natural, but needs the epilogue `comb_row` indexing to change from `pos_row * N_DIM + col` to a block-offset calculation
- The `pos_row` for each lane is `(gm_base + lane) % 196` — adjacent lanes DO get adjacent rows. The blocking exploits this by making adjacent rows contiguous within each column chunk
- **Wraparound hazard:** 196 / 32 = 6.125 — the last block has only 4 rows (196 - 192 = 4), requiring padding or special handling. Near the 196 boundary, 32 lanes can span two non-contiguous pos_row ranges (e.g., gm_base=180 → lanes 0-15 get rows 180-195, lanes 16-31 get rows 0-15). The block indexing must handle this correctly.
- ~~With `.cg` (proposal 1b), the relayout may be redundant~~ — F11 killed `.cg` (44% regression). L1 bypass is not viable for scattered combined loads. Relayout remains the only path to fixing the access pattern.

**Interaction with other proposals:** Pairs naturally with #3 (TMA-load combined). A blocked layout can be made TMA-compatible (swizzled), enabling a single TMA bulk copy of each 32×256 block into SMEM instead of 4 scattered `uint4` loads per nc iteration.

**Priority:** Lower than before. F10 proved the epilogue is K-loop-bound — combined load stalls are in slack time. This optimization only helps if combined loads contribute enough L1 traffic to indirectly slow the K-loop (possible via shared L1 bandwidth with TMA/MMA), but that's speculative.

**Go/no-go:** Zero runtime cost (offline relayout). Keep unless the indexing change introduces bugs or the blocked layout doesn't improve L1/L2 sector efficiency in ncu.

---

### 3. TMA-load combined tile into SMEM

**Problem:** Phase 1 issues 4 uncoalesced `uint4` global loads per nc iteration (32 cols), totaling 32 loads per tile per warp. These compete for L1 bandwidth with Phase 2 stores and generate excess L2 sectors. F5 (SMEM prefetch during K-loop) was rejected because manual SMEM staging was slower than L1-cached inline loads — but F5 used explicit load loops, not TMA.

**Fix:** Use TMA bulk copy (`cp.async.bulk.tensor.2d`) to load the combined chunk needed for the current tile into SMEM. TMA is a single instruction issued by lane 0 — zero register pressure on epilogue warps, no unrolled loop, no code bloat.

**Scheduling opportunity:** W0 (TMA load warp) is idle during the epilogue phase. It could issue the combined TMA load for the *next* tile's epilogue while W2-W6 are processing the current tile. This prefetch overlaps combined data movement with epilogue compute — similar to how W0 prefetches A/B tiles during the K-loop.

**SMEM budget:** 17 KB free. A useful chunk is 32 rows × 256 cols × 2B = 16 KB (one row_group's worth). Fits, but only one chunk at a time — warps would need to share or take turns. Alternatively, if combined with proposal #2 (offline relayout), the TMA descriptor can fetch a pre-blocked tile directly into SMEM with perfect layout.

**New coordination required:** W0 currently only synchronizes with W1 (via TMA mbarriers). Loading combined for the epilogue means W0 needs a new mbarrier path to signal W2-W6 that the combined data is ready in SMEM. Adds complexity to the warp synchronization protocol.

**Interaction with F10/F11:** F10 (software pipelining) was neutral — epilogue stalls aren't on the critical path. F11 (`.cg`) regressed 44% by saturating L2. TMA-loading combined into SMEM remains a viable alternative to reduce L1 traffic from combined loads, but only matters if that L1 traffic indirectly contends with K-loop TMA loads (speculative).

**Go/no-go:** Lower priority given F10's finding that the epilogue is K-loop-bound. Only pursue if ncu profiling shows combined loads contributing significant L1 contention with TMA pipe. Kill if mbarrier coordination cost exceeds the L1 relief.

---

### 4. Asymmetric work redistribution across epilogue warps

**Problem:** Epilogue work is imbalanced. W2 and W6 each process 128 cols (4 nc iterations) via column-split of row_group 0. W3/W4/W5 each process 256 cols (8 nc iterations). The critical path is determined by the 8-iteration warps — W2 and W6 idle for ~50% of the epilogue.

| Warp | Row group | Columns | nc iters | Status |
|------|-----------|---------|----------|--------|
| W2 | 0 | 0-127 | 4 | **idle 50%** |
| W6 | 0 | 128-255 | 4 | **idle 50%** |
| W3 | 1 | 0-255 | 8 | critical path |
| W4 | 2 | 0-255 | 8 | critical path |
| W5 | 3 | 0-255 | 8 | critical path |

**Fix:** After W2 finishes row_group 0 (cols 0-127), have it process row_group 3 (cols 0-127). W6 finishes row_group 0 (cols 128-255), then processes row_group 3 (cols 128-255). W5 is freed from row_group 3.

New distribution:

| Warp | Work | nc iters |
|------|------|----------|
| W2 | rg0 cols 0-127, then rg3 cols 0-127 | 4+4=8 |
| W6 | rg0 cols 128-255, then rg3 cols 128-255 | 4+4=8 |
| W3 | rg1 cols 0-255 | 8 |
| W4 | rg2 cols 0-255 | 8 |
| W5 | removed, or takes partial work | 0 |

All active warps do 8 iterations. No idle time. W5 either becomes a 5th balanced worker (if we find work to give it) or gets reassigned.

**Key constraint:** TMEM loads are per-row_group (`taddr_base` depends on `row_group`). A warp processing two row_groups needs two sequential `epilogue_store` calls with different `taddr_base` and `pos_row`. This is two function calls, two `__syncwarp` barriers, two Phase 2 passes — but fills the 50% idle window that was previously wasted.

**Does NOT increase TMEM concurrency:** W2/W6 process their second row_group sequentially after the first, not concurrently. This avoids the TMEM bandwidth saturation seen in F3 (6 concurrent warps regressed).

**Risk (HIGH — likely dead end):**
- **F10 confirmed the idle time is NOT on the critical path.** The epilogue completes within the K-loop's shadow. W2/W6 finishing early is irrelevant — they wait for `mainloop_mbar` anyway. Redistributing work only matters if the epilogue becomes the longer leg, which it currently isn't.
- Two `epilogue_store` calls per warp means two TMEM prefetch setups and two `__syncwarp` barriers. W2/W6 arrive at `epilogue_mbar` later, delaying W1's next MMA start.
- Reducing active warps from 5 to 4 (W5 freed) reduces TMEM contention (good, per F3) but also reduces parallelism for hiding TMEM latency (bad).
- Similar structural risk profile to F3/F7 — past experiments show TMEM/synchronization changes have unpredictable effects.
- **Danger:** Making the epilogue take *longer* per warp (two sequential `epilogue_store` calls) risks flipping the overlap balance, as F11 demonstrated catastrophically.

**Go/no-go:** Likely not worth pursuing given F10's K-loop-bound finding. Only reconsider if a future change (e.g., TN=128) makes the epilogue the longer leg.

---

## LAST RESORT

### 6. TMA stores for Phase 2

**Problem:** Phase 2 executes ~3.5M `ld.shared` + ~3.5M `st.global` instructions per kernel invocation. These go through L1 (stores) and the SMEM pipe (loads), contributing to L1 pressure (85%) and short_scoreboard stalls (1.1%).

**Fix:** Replace Phase 2's manual ld.shared → st.global loop with a single TMA `cp.async.bulk.tensor.2d` store from SMEM to global memory per chunk.

**Why it helps:**
- Eliminates ~7M instructions (ld.shared + st.global) entirely
- TMA stores bypass L1, going SMEM → L2 → DRAM
- Reduces both L1 pressure and instruction count

**Staging reuse problem:** TMA stores are asynchronous at the instruction level, but the **SMEM staging buffer cannot be overwritten** until the TMA store completes reading from it. Phase 1 of the NEXT nc iteration would overwrite staging while TMA is still reading. Workarounds:
1. **Double-buffered staging:** 84 KB × 2 = 168 KB additional. Impossible (17 KB headroom).
2. **TMA completion fence before next Phase 1:** Brings back the stall, negating the async benefit.
3. **Single-shot TMA after all Phase 1 iterations:** Requires entire tile output in SMEM simultaneously (128 rows × 256 cols × 2B = 64 KB). Doesn't fit per-warp staging (16,896 bytes).

None of these are clean. The "issue and proceed immediately" assumption does not hold without an explicit completion-safe staging reuse protocol.

**Hard part:** TMA stores require SMEM data in the output tensor's memory layout. Currently Phase 1 writes row-per-thread (lane i writes its row contiguously), and Phase 2 transposes this for coalesced output. For TMA, Phase 1 would need to write data in the final row-major output layout within SMEM, which requires lane coordination (each lane owns different rows but TMA needs columns-contiguous). This is a full layout rework.

Additionally: TMA store descriptors need to be set up for the output tensor, the SMEM swizzle pattern must match, and TMA engine occupancy is shared with the W0 load warp.

**Effort:** High (full day+). Layout rework + staging reuse protocol + TMA descriptor setup + correctness validation.

**Go/no-go:** Only pursue if all earlier proposals plateau and L1 pressure remains the dominant bottleneck. The staging reuse constraint makes this significantly harder than it appears at first glance.
