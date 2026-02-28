# Future Proposals

Baseline: **0.579 ms / 1892 TFLOPS**, 223 regs, 0 spills. L1 tex at 68%, L2 at 49%, DRAM at 32%.
Reference: cuBLAS GEMM-only = 0.365 ms / 3001 TFLOPS. cuBLAS + unfused pos_embed = 0.835 ms.

**Critical findings:**
- **K-loop-bound** (F10/F11): The epilogue completes within the K-loop's shadow.
- **Combined loads fully optimized** (F13/F14): Blocked relayout gave near-perfect L1 locality. Three L1-bypass variants (cp.async SMEM staging +40%, ld.global.cg +6%, ld.global.cs +28%) all regressed — L1 hits at ~30 cycles are optimal and overlap with TMEM latency.
- **L1 at 68% is the new equilibrium** — remaining L1 traffic is K-loop inherent (TMA fills, MMA), not from combined loads.

**Tested and ruled out:** Proposals 2 (offline relayout → committed as F13) and 3 (TMA/L1-bypass combined loads → F14, all variants failed). See EXPERIMENTS.md.

---

## SHOULD TRY

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
