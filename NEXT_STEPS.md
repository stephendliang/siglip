# Next Steps for Optimization

Current state: 0.630 ms / 1739 TFLOPS, L1 throughput-bound (85%), 222 regs, ~211 KB SMEM, 1 cluster/SM.

Ordered by effort and expected payoff.

## 1. Split x32→2×x16 TMEM Prefetch (Earlier Overlap)

**Effort: medium (restructure epilogue loop, ~30 min)**

Currently `TMEM_LOAD_X32` prefetch is issued at the END of each Phase 1 loop iteration (after both CVT_STS calls). Splitting into 2× `TMEM_LOAD_X16` allows issuing the first half right after the first CVT_STS consumes `a0..a15`, giving ~half an iteration more overlap with TMEM latency.

**Restructured loop:**
```
TMEM_LOAD_X16(a0..a15, NC_START)
TMEM_LOAD_X16(a16..a31, NC_START+16)
for nc = NC_START..NC_END step 32:
    load combined, TMEM_WAIT, add first 16, CVT_STS(a0..a15)
    TMEM_LOAD_X16(a0..a15, nc+32)       ← issued HERE (earlier)
    convert+add second 16, CVT_STS(a16..a31)
    TMEM_LOAD_X16(a16..a31, nc+32+16)
```

**Why it could help:** `long_scoreboard` (TMEM) is 4.4%, the single largest stall. More overlap time = fewer TMEM wait cycles. x16 vs x32 is bandwidth-neutral (CLAUDE.md confirms perf-neutral). No extra registers needed since the loads target the same a0..a31.

**Risk:** Two `tcgen05.ld` instructions vs one = 1 extra instruction per chunk. `tcgen05.wait::ld` is a global fence (waits for ALL outstanding TMEM loads), so the second x16 of iteration N+1 must complete before the first TMEM_WAIT of iteration N+1. If the first x16 prefetch overlaps enough, the second one might still be in-flight at TMEM_WAIT.

## 2. Coalesced `combined` Loads (Phase 0 Staging)

**Effort: medium-high (~1 hr)**

Current `combined` tensor loads are uncoalesced: each thread reads a different row (1536B stride), causing excess L2 sector overfetch. Add a Phase 0: coalesced warp-cooperative load into SMEM, then read from SMEM during Phase 1.

**Caveats:**
- This is sector **overfetch**, not read-modify-write amplification (RMW is a store phenomenon).
- "Should eliminate remaining 33% excess L2 sectors" is a hypothesis — excess may not reach zero even if combined-load overfetch is the dominant source.
- Phase 0 and Phase 1 both need the SMEM staging buffer. Must sequence Phase 0 to completion before Phase 1 begins writing, or use a separate SMEM region (~17 KB headroom is tight).
- With L1 at 85%, reducing L2 overfetch may have limited impact unless it also frees L1 bandwidth for other traffic.

## 3. TMA Store from SMEM (Re-test)

**Effort: high (~2 hr)**

Replace Phase 2 `st.global` with `cp.async.bulk.tensor.2d` from SMEM to global, bypassing L1 entirely.

**Blockers to address before attempting:**
- Prior TMA store approach failed due to lane-0 serialization (`cp.async.bulk.wait` stalls, single-lane issue). Must explain how the new approach avoids the same `stalled_wait` path.
- SMEM staging layout is row-per-thread (not contiguous 2D). TMA tensor stores require a contiguous rectangular tile in SMEM. Would need a layout change or an additional transpose step.
- **Kill criteria:** if `stalled_wait` exceeds 3% in initial testing, abandon.

## 4. 2 Clusters/SM Occupancy (TN=128 Sweep)

**Effort: very high (major rework)**

Reduce SMEM below ~114 KB (TN=128, fewer stages/warps) to fit 2 clusters (4 CTAs) per SM.

**Why this is harder than it sounds:**
- The kernel is **throughput-bound** (85% L1), not latency-bound. `long_scoreboard` is only 4.4%. Doubling occupancy helps hide latency, but if L1 bandwidth is the ceiling, more warps just increase L1 contention.
- TMEM budget: 512 cols/SM total. TN=128 double-buffered = 256 cols/cluster. Two clusters = 512 cols. Fits exactly, no margin.
- Launch config and tile distribution are hardcoded for 1 cluster/SM (`SM_COUNT` CTAs, fixed snake ordering). Needs non-trivial changes to launch geometry, tile mapping, and cluster partitioning.
- Previous TN=128 test hit 1190 TFLOPS (with x16 loads, pre-staging). Worth re-testing with current epilogue, but as a data point, not an assumed win.

## Ruled out

- **Phase 2 unroll sweep** (tested): unroll 2→0.636 ms, 4→0.634 ms, **8→0.630 ms (winner)**, 16→0.630 ms, 32→0.636 ms. Unroll 8 is the sweet spot: better ILP without I-cache penalty. Applied.
- **Store cache-policy `.cs`/`.cg`** (tested): `.cs` (streaming) regressed +1.1% (0.640 ms). `.cg` (L2-only) was neutral (0.634 ms). Coalesced stores from SMEM staging already minimize L1 write pressure — bypassing L1 doesn't help because the stores aren't polluting useful cache lines.
- **SMEM bank-conflict swizzle** (infeasible): Additive per-row offset (`(row>>3)*4` bytes) eliminates 4-way bank conflicts in theory, but `st.shared.v4.b32` / `ld.shared.v4.b32` require 16B alignment. Breaking stride-4 bank patterns requires 1/2/3-word offsets, which are inherently misaligned for v4 ops. XOR-based swizzle would need matching deswizzle in Phase 2 and adds complexity for a stall that's only 1.1% (`short_scoreboard`).
- **I-cache pressure reduction**: 314% increase in I-cache requests was noted in profiling, but `stalled_imc_miss` is not a measurable stall source. Not a bottleneck — don't optimize it.
- **Warp-specialized epilogue (producer/consumer split)**: Requires SMEM double-buffering of the staging area (~84 KB extra). Budget is ~17 KB free. Not feasible without first shrinking the pipeline or tile.
