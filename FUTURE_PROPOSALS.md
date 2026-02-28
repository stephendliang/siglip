# Future Proposals

Baseline: **0.630 ms / 1739 TFLOPS**, 222 regs, 0 spills. L1 tex at 85% (primary ceiling).
Reference: cuBLAS GEMM-only = 0.365 ms / 3001 TFLOPS. cuBLAS + unfused pos_embed = 0.835 ms.

---

## MUST TRY

### 1a. Software-pipeline combined tensor loads

**Problem:** In the Phase 1 epilogue loop, the 4 global `uint4` loads for the combined (bias+pos_embed) tensor are issued and immediately consumed by `BF16X2_TO_F32` conversions. These loads are uncoalesced (32 lanes × different rows, 1536B stride), so each takes ~50-200 cycles. The warp stalls waiting for the data before it even reaches `TMEM_WAIT()`, burning latency-hiding time that should be spent under the TMEM fence.

```
// Current: load → immediate consume → STALL → TMEM_WAIT (arrives late)
craw0 = *(comb_row + nc);           // issue uncoalesced global load
BF16X2_TO_F32(craw0.x, s0, s1);    // STALL here waiting for craw0
...
TMEM_WAIT();                         // warp already burned 100+ stall cycles getting here
```

**Fix:** Issue loads for iteration N+1 at the end of iteration N. By the time they're consumed in iteration N+1, they've had the entire TMEM_WAIT + FADD + CVT_STS + prefetch duration to complete.

```
// PROLOGUE: issue first TMEM prefetch + first 4 combined loads
TMEM_LOAD_X32(..., taddr + NC_START);
next_craw0 = *(comb_row + NC_START);
next_craw1 = *(comb_row + NC_START + 8);
next_craw2 = *(comb_row + NC_START + 16);
next_craw3 = *(comb_row + NC_START + 24);

for (nc = NC_START; nc < NC_END; nc += 32) {
    curr_craw0..3 = next_craw0..3;   // register rename (free)

    if (nc + 32 < NC_END) {           // issue NEXT iteration's loads NOW
        next_craw0 = *(comb_row + nc + 32);
        next_craw1 = *(comb_row + nc + 40);
        next_craw2 = *(comb_row + nc + 48);
        next_craw3 = *(comb_row + nc + 56);
    }

    BF16X2_TO_F32(curr_craw0..3);     // no stall — data loaded last iteration
    TMEM_WAIT();                       // next_craw loading in background
    FADD + CVT_STS ...
    if (nc + 32 < NC_END)
        TMEM_LOAD_X32(...);            // issued sooner (no prior stall)
}
```

**Register cost:** +16 to +24 regs. Current code uses 2 × `uint4` (8 regs) with reuse across first/second halves. Pipelining requires 4 × `uint4` for next-iteration data (16 regs) live across the loop boundary, plus 4 × `uint4` for current-iteration data (16 regs) during consumption — peak 32 regs for combined data vs current 8. Compiler register renaming in the fully-unrolled loop may reduce the effective cost. Hard per-thread ceiling is **255 regs** (ISA encoding limit). At 222 + 24 = 246, that's 9 regs of headroom — tight if the compiler extends other live ranges.

**Icache note:** The `<0, 256>` template unrolls to 8 iterations. Software pipelining adds ~4 LDGs per unrolled iteration = 32 extra instructions in the unrolled body. Icache requests already went +314% from SMEM staging. Monitor `l1tex__t_requests_pipe_lsu_mem_global_op_ld` and instruction cache hit rate.

**Expected effect:** TMEM_LOAD_X32 issued earlier each iteration → better TMEM overlap. Combined load stalls moved out of the critical path. Measurable even without `.cg`.

**Go/no-go:** Keep if spills = 0 and runtime improves. Kill immediately if spills appear or runtime regresses > 1%. Check `ptxas` register count before running.

---

### 1b. Combined loads bypass L1 via `.cg` (sweep alongside 1a)

**Problem:** L1 tex throughput is at **85%** — the primary bottleneck. Phase 1's uncoalesced combined loads and Phase 2's coalesced `st.global` stores compete for the same L1 bandwidth. The combined tensor (196 × 768 × 2B = 294 KB) barely exceeds L1 capacity (~256 KB), causing thrashing under multi-warp access.

**Fix:** Replace standard loads with `ld.global.cg` (cache-global, bypass L1):

```c
#define LDG_CG_V4(reg, ptr) \
    asm volatile("ld.global.cg.v4.b32 {%0,%1,%2,%3}, [%4];" \
        : "=r"(reg.x), "=r"(reg.y), "=r"(reg.z), "=r"(reg.w) \
        : "l"(ptr))
```

**Relationship to 1a:** `.cg` without pipelining is actively harmful — bypassing L1 increases per-access latency from ~50 to ~200 cycles (L2 path), making the existing pre-`TMEM_WAIT()` stall worse. With pipelining, the latency is hidden behind the previous iteration's TMEM + compute work. However, 1b is NOT a hard dependency on 1a — it should be tested as a **2-point sweep**: `1a alone` vs `1a + .cg`.

**Why it might work:** The combined tensor fits trivially in L2 (48+ MB on B200). L2 is at 54% utilization. Routing combined loads through L2 instead of L1 drops L1 pressure, giving Phase 2's coalesced stores uncontested L1 access.

**Why it might not:** `.cg` does NOT fix the uncoalesced access pattern — it just moves scatter from L1 to L2. Each lane's `uint4` load still generates a separate 128B cache line fetch. A single warp's 4 loads generate 128 scattered L2 accesses (16 KB of L2 traffic for 2 KB of useful data, 8× amplification). All 148 SMs × 5 warps × 8 iterations doing this simultaneously puts real pressure on L2 — the 54% headroom may not be as generous as it looks.

**Zero register cost** beyond 1a.

**Go/no-go:** Test `1a alone` vs `1a + .cg`. Monitor `l1tex__throughput`, `lts__throughput`, `long_scoreboard`, and wall time. Keep `.cg` only if time improves AND L2 throughput stays below ~75%. Kill if L2 becomes the new limiter.

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
- With `.cg` (proposal 1b), the relayout may be redundant — if combined loads bypass L1 anyway, their access pattern through L2 matters less. Test 1b first; if L1 drops well below 85%, this becomes lower priority

**Interaction with other proposals:** Pairs naturally with #3 (TMA-load combined). A blocked layout can be made TMA-compatible (swizzled), enabling a single TMA bulk copy of each 32×256 block into SMEM instead of 4 scattered `uint4` loads per nc iteration.

**Priority:** First SHOULD TRY. Becomes MUST TRY if 1b fails to relieve L1 pressure (L1 stays above ~80% after 1a+1b sweep).

**Go/no-go:** Zero runtime cost (offline relayout). Keep unless the indexing change introduces bugs or the blocked layout doesn't improve L1/L2 sector efficiency in ncu.

---

### 3. TMA-load combined tile into SMEM

**Problem:** Phase 1 issues 4 uncoalesced `uint4` global loads per nc iteration (32 cols), totaling 32 loads per tile per warp. These compete for L1 bandwidth with Phase 2 stores and generate excess L2 sectors. F5 (SMEM prefetch during K-loop) was rejected because manual SMEM staging was slower than L1-cached inline loads — but F5 used explicit load loops, not TMA.

**Fix:** Use TMA bulk copy (`cp.async.bulk.tensor.2d`) to load the combined chunk needed for the current tile into SMEM. TMA is a single instruction issued by lane 0 — zero register pressure on epilogue warps, no unrolled loop, no code bloat.

**Scheduling opportunity:** W0 (TMA load warp) is idle during the epilogue phase. It could issue the combined TMA load for the *next* tile's epilogue while W2-W6 are processing the current tile. This prefetch overlaps combined data movement with epilogue compute — similar to how W0 prefetches A/B tiles during the K-loop.

**SMEM budget:** 17 KB free. A useful chunk is 32 rows × 256 cols × 2B = 16 KB (one row_group's worth). Fits, but only one chunk at a time — warps would need to share or take turns. Alternatively, if combined with proposal #2 (offline relayout), the TMA descriptor can fetch a pre-blocked tile directly into SMEM with perfect layout.

**New coordination required:** W0 currently only synchronizes with W1 (via TMA mbarriers). Loading combined for the epilogue means W0 needs a new mbarrier path to signal W2-W6 that the combined data is ready in SMEM. Adds complexity to the warp synchronization protocol.

**Interaction with 1a/1b:** If software-pipelined `.cg` loads (1a+1b) drop L1 below 85% and eliminate the combined load stalls, TMA-loading combined becomes less valuable. Test 1a+1b first.

**Go/no-go:** Only pursue if 1a/1b leave L1 above ~80% and combined loads remain a significant contributor to L1 traffic. Kill if mbarrier coordination cost exceeds the L1 relief.

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

**Risk (medium-high):**
- The idle time may not be on the critical path. The epilogue is overlapped with the K-loop. If K-loop time > max(warp epilogue time), W2/W6 finishing early is irrelevant — they wait for mainloop_mbar anyway. The imbalance only matters if the epilogue is the longer leg.
- Two `epilogue_store` calls per warp means two TMEM prefetch setups and two `__syncwarp` barriers. W2/W6 arrive at `epilogue_mbar` later, delaying W1's next MMA start.
- Reducing active warps from 5 to 4 (W5 freed) reduces TMEM contention (good, per F3) but also reduces parallelism for hiding TMEM latency (bad).
- Similar structural risk profile to F3/F7 — past experiments show TMEM/synchronization changes have unpredictable effects.

**Go/no-go:** Exploratory only, pursue after 1a/1b/2. Kill immediately if TMEM-related stalls (`long_scoreboard`) increase or if no clear wall-time improvement.

---

### 5. TN=128 revisit with current epilogue stack

**Problem:** TN=128 was last tested at 1190 TFLOPS under a completely different kernel: x16 TMEM loads, no SMEM staging, no 5th warp, 6 pipeline stages. The epilogue has fundamentally changed since then. TN=128 may behave differently under the current architecture.

**What changes with TN=128:**
- TILES_N: 3 → 6 (768/128), TOTAL_TILES: 10,878 → 21,756
- TMEM_COLS: 512 → 256 (128×2 for double buffering)
- Epilogue per tile: 4 nc iterations (128 cols / 32) instead of 8 — every warp finishes in half the time
- Work balance improves: 4 row_groups × 4 iters = 16 units / 5 warps is more even than 32 units / 5 warps
- SMEM staging per warp shrinks: (128×2 + 16 pad) × 32 rows = 8,448 bytes vs 16,896. Total staging: 42 KB vs 84 KB — frees ~42 KB
- Freed SMEM + TMEM could enable deeper pipeline or other uses

**What gets worse:**
- 2× more tiles = 2× more tile transitions (mbarrier waits, TMEM prefetch setup, tile index computation)
- K-loop is unchanged (6 iters) but MMA work per tile halves (128×128 instead of 256×128 per CTA) — MMA pipe utilization drops
- Each tile produces less output per K-loop overhead — worse compute-to-overhead ratio

**Sweep parameters:** Must co-sweep with N_STAGES (2/3/4) and NUM_EPI_WARPS (3/4/5) since the optimal balance point shifts. The smaller tile means less epilogue work, so fewer epilogue warps might suffice (freeing registers). Also check if register count drops enough for `__launch_bounds__(THREADS, 2)` — though at 222 regs this seems unlikely without significant code shrinkage.

**Go/no-go:** Pure parameter sweep — low implementation risk. Keep if any sweep point beats 0.630 ms. The main value is diagnostic: if TN=128 with fewer epilogue warps is competitive, it reveals that the current epilogue is K-loop-bottlenecked (not epilogue-bottlenecked), which informs whether proposals 2/4 are worth pursuing.

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
