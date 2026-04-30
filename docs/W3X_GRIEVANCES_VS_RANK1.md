# fc2_w3x — what's left to try

Status (2026-04-25): fc2_w3x bias-only at 1.007 ms, beats cuBLASLt rank-1
(1.046 ms) by 39 µs on FC2 K=3072. W5 MMA-ceiling-bound at ~12482 cyc/tile
≈ 24 × 520 cyc/iter. Tensor pipe 95.84% active → 43 µs absolute non-tensor
slack. Hardware MMA-retirement floor 0.896 ms (base) / 0.827 ms (boost).
Gap to floor: ~100 µs (base), 180 µs (boost). Per ncu, that's "staging +
pipeline bubbles, not compute-bound" — the regime where structural topology
changes pay off and SASS cleanups don't.

This doc was originally a list of 9 SASS-level deltas vs rank-1's
`128x256 NS=6 2x1 2cta` listing (architectural twin of our kernel). All 9
have been investigated since; section 1 below summarizes status. The
forward-looking content is sections 2-4: Tier 1-3 untested strategies
ranked by upside/effort.

Reference SASS: `rank1.sass` is a `cuda-gdb` dump of cuBLASLt listing 2
(`128x256 NS=6 2x1 2cta`, the architectural twin of fc2_w3x) captured
2026-04-19. Used as a *data point* on alternative design choices, not
as a target — fc2_w3x at 1.007 ms beats both rank-1 listings (listing 1
at 1.0454, listing 2 at 1.0457). Header decodes:
```
nvjet_sm100_qqtst_128x256_128x6_2x1_2cta_v_bz_bias_TNT
  <<<(21756,1,1),(256,1,1)>>>   cluster dim (2,1,1)
```
- Grid `(21756, 1, 1)` = X-axis only. The `2x1` and `1x2` labels in
  cuBLASLt kernel names are *logical* cluster shape; the actual grid
  axis is always X. Confirmed empirically — see DEAD list below.
- 21756 CTAs = 10878 clusters = `TOTAL_TILES` exactly: rank-1 launches
  **one cluster per tile, non-persistent** (~74 waves to drain). We
  use 73 persistent clusters that loop over tiles. Different scheduling
  model — see Tier 3 #4. Note: this scheduling alone doesn't make
  rank-1 faster; it loses 39 µs to us anyway. The relevant question is
  whether non-persistent + our hand-tuned per-tile work would be faster
  than our current persistent kernel.

Listing 1 (`176x128 NS=8 1x2 2cta`) is a different tile shape — see
Tier 1 #1. Reference target is the **hardware MMA-retirement floor**
(0.896 ms base, 0.827 ms boost), not rank-1.

## 1. SASS-level grievances — exhausted

| # | Grievance | Status | Verdict |
|---|---|---|---|
| G1 | No packed BF16 epilogue (FFMA2/F2FP.PACK) | Implemented (`native BF16 epilogue`) | DEAD as wall lever (±0 µs, cleaner) |
| G2 | No stmatrix (STSM) | STSM `bcce329` lands SASS shape match | At-baseline n=10 (Δ −0.4 µs ≈ noise floor) |
| G3 | Per-thread IMAD addressing | Bundled with G4 | DEAD |
| G4 | 495 R2UR round-trips | `dead_r2ur_elect_smem_fix.md` | DEAD — ptxas won't ULDS-promote inside `if(lane==0)`; UIADD3=706 already uses UR via constant prop |
| G5 | Per-asm ELECT/BSSY/BSYNC scaffold | K-loop single-asm DEAD; **multi-block consolidation untested** | See Tier 2 #2 |
| G6 | Fully-unrolled K-loop vs tight loop | K_UNROLL sweep B200 n=3 (Apr 25) | DEAD as wall lever; multiples-of-N_STAGES tie default, non-multiples regress 87–197 µs. Explicit `K_UNROLL=24` = 39% SASS shrink at parity wall |
| G7 | NANOSLEEP count | Never tested as perf | Not a lever (fires on miss only) |
| G8 | No ACQBULK fence | Never tested as perf | Code-cleanliness only |
| G9 | TMA batch granularity | Never tested as perf | Revisit during fused-residual port |
| — | Cluster-axis swap (Y/Z) | 2 attempts: bit-mask v1, mapa+cudaLaunchKernelEx v2 | DEAD — B200 runtime rejects (1,2,1)/(1,1,2) with "cluster misconfiguration". Hardware/driver constraint. rank-1 also uses X-axis (verified in `rank1.sass`) |

**What this means:** the static-SASS-cleanup ceiling is now ~5-10 µs of
remaining upside (G5 multi-block, possibly G7/G8/G9 if a future regime
shift makes them load-bearing). The original 15-25 µs estimate baked in
G2 (since proven noise-floor) and G4 (since proven structural).

To go materially below 1.007 ms, structural changes are needed: tile
shape, cluster topology, register budget, dispatch model, or MMA shape.

## 2. Tier 1 — structural changes, 5-15 µs upside, moderate effort

### Strategy 1.1 — Tile shape sweep with deeper NS

We've only ever benched fc2_w3x at one shape: `128x256 NS=6 2x1 2cta`
(per-CTA 128x256, cluster output 256x256). cuBLASLt listing 1
(`176x128 NS=8 1x2 2cta`) is a different point in the design space —
smaller per-CTA tile, deeper pipeline, N-axis cluster split. cuBLASLt
itself runs slower than fc2_w3x at both shapes, so this isn't about
mimicking — it's about whether deeper NS at a smaller tile would let
*our* hand-tuned per-tile work hide more TMA latency.

**Why it's a lever:** at 256x256 NS=6, each pipeline stage holds 256x128 A
+ 256x128 B in FP8 = ~64 KB per stage × 6 stages ≈ 227 KB of 228 KB SMEM
(NS=7 doesn't fit). At 128x256 (half the per-stage A bytes), NS=7-8
should fit. Deeper NS = more in-flight TMA loads = better latency hide
when MMA is faster than fill.

**Why it might not be:** smaller tiles → more total tiles → more tile-
launch overhead (BSSY/BSYNC, mbarrier rotates, scheduler atomicAdd).
Persistent kernel amortizes some of this, but not all. Also, our 256x256
choice was made because larger tiles minimize cluster-sync frequency on
B operand multicast.

**Sweep matrix:** (TM, TN, NS) ∈ {128, 256} × {128, 256} × {6, 7, 8},
filter for SMEM ≤ 228 KB. ~9 valid configs. Each requires `make -B
fc2-w3x DFLAGS='-DTM=... -DTN=... -DN_STAGES=...'` and a wall test.

**Constraint to verify per config:** kernel asserts `TOTAL_TILES %
num_clusters == 0` (CLAUDE.md). Compute TOTAL_TILES = ceil(M_TOTAL /
(TM × 2)) × ceil(N_DIM / TN) and confirm it's a multiple of 74 before
queueing each build. Some shape combinations will fall out of the sweep
on this constraint alone.

**Realistic upside:** 5-15 µs if NS=7-8 fits and meaningfully deepens
TMA-latency hide. Could be 0 if the smaller-tile launch overhead
cancels the pipeline gain.

### Strategy 1.2 — Register-budget tuning via `__launch_bounds__` / `.maxnreg`

The K_UNROLL sweep showed regs=64 vs 66 toggles wall by 100+ µs (the UR
datapath gate). That's a side-effect observation, not a swept lever.
Forcing ptxas to a different reg target via `__launch_bounds__(threads,
min_blocks)` or PTX-level `.maxnreg N` could push it onto a different
scheduling regime.

**Why it's a lever:** regs=64 at threads=192 already leaves slack — B200
SM has 64K registers, 192 threads × 64 regs = 12K regs per CTA, well
under the 64K cap. Forcing regs=48 or regs=80 changes ptxas's spill/fill
calculus and can unlock different instruction schedules.

**Sweep:** `__launch_bounds__(192, N)` for N ∈ {1, 2, 4, 8, 16}, then
also raw `.maxnreg` values via PTX. ~10 builds, single-file change.

**Risk:** spill (regs too low) is a hard wall regression; relaxed regs
(regs too high) eats SM occupancy. Only the sweet spot pays.

**Realistic upside:** 3-10 µs if there's a reg-budget the K_UNROLL
sweep didn't accidentally hit. Highly uncertain — could be 0.

## 3. Tier 2 — validated paths, small wins

### Strategy 2.1 — Explicit `K_UNROLL=24` as default

The K_UNROLL sweep (Apr 25) flagged anomalous variance on the default
build: max−min = 7.9 µs on n=3 vs ≤0.5 µs for u6/u12/u24. If the n=10
re-test confirms the spread is real, switching the default build to
explicit `K_UNROLL=24` saves 5-8 µs on outlier runs and 2-3 µs on
means, plus delivers a 39% SASS shrink (10029→6077 lines) for free.

**Test:** interleaved n=10 of `./fc2-w3x` (default) vs `./fc2-w3x` with
`-DK_UNROLL=24` baked in. Split @@RESULTs by line parity to kill
sequential drift.

**Decision:**
- If variance gap is real → make `K_UNROLL=24` the Makefile default.
- If gap collapses → keep default, treat the SASS shrink as cosmetic.

**Realistic upside:** 2-8 µs depending on what n=10 says.

### Strategy 2.2 — ASM block consolidation (Grievance 5 expanded)

Each `asm volatile` block wrapped in a `@P0` predicate gets ptxas
scaffolding: BSSY (start branch-sync region), ELECT (pick lane),
PTX body, BSYNC (rejoin). w3x has 292 ELECT, 79 BSSY, 79 BSYNC →
~450 insts of pure scaffold per kernel.

Memory says single-block merge in the K-loop has been done (DEAD, 0 µs).
But fc2_w3x.cu has ~10 separate asm blocks per tile in W4+W5 outside the
K-loop (descriptor advance, mbarrier ops, scheduler increment, epilogue
TMEM loads, STS issues, fence emits). Consolidating to 3-4 larger PTX
bodies would cut scaffold ~3×.

**Why it's a lever:** scaffold runs on the uniform / branch pipe, not
tensor. Critical-path overlap is limited but nonzero — at 95.84% tensor
active, 4.16% slack ≈ 43 µs. Some scaffold cycles fall into that slack;
some fall on critical path during transitions (tile boundaries, store
barriers).

**Why it might not be:** the doc's original estimate was 3-5 µs, "pipe
already has slack." That estimate stands.

**Cost:** invasive. Each merge needs careful PTX rewriting to preserve
correctness across now-grouped operations (memory ordering, barrier
semantics, register allocation across the larger asm body).

**Realistic upside:** 3-5 µs if all consolidation lands cleanly.

## 4. Tier 3 — high-effort structural rewrites

### Strategy 3.1 — Stream-K dispatch with our hand-tuned per-tile body

Persistent dispatch hits scheduling ceilings: cluster wavefront timing,
tile-bijection L2 staggering, store-barrier overlap. Stream-K decomposes
the K-axis across SMs and reduces partial accumulators, fundamentally
different load-balancing model.

**Why it might be a lever:** at 1.007 ms / 1.046 ms (us / rank-1), both
kernels are persistent. Stream-K is a different point in the design
space — could expose pipeline parallelism that persistent doesn't.

**Why it might not be:** we have a CUTLASS-static reference that runs at
1.244 ms (185 µs slower than us at parity tile shape). That suggests
CUTLASS's epilogue is the bottleneck in their kernel, not their
dispatch. Grafting our epilogue onto CUTLASS's dispatch would test
whether stream-K's load balancing helps — but most CUTLASS infrastructure
is incompatible with our hand-rolled tcgen05 pipeline.

**Cost:** weeks of refactoring. Probably untestable without a partial
CUTLASS dependency.

**Realistic upside:** highly uncertain. Could be 20+ µs if stream-K
fundamentally outruns persistent at this M; could be a regression.

### Strategy 3.2 — Different MMA K-shape (K=64 or K=256 per iter)

Currently `tcgen05.mma.sync` with K=128 per iter, K_ITERS=24. The
hardware MMA-retirement floor (460 cyc/iter at K=128) scales with K.

**K=64** doubles iter count to 48, halves per-iter retirement (~230
cyc), shortens the pipeline-bubble length. Could let NS effectively
deepen for free (more iters, same staging). Risk: more iter-boundary
overhead (descriptor advance, mbarrier toggle).

**K=256** halves iter count to 12, doubles per-iter retirement (~920
cyc). Risk: iter is now longer than TMA load, MMA blocks waiting for
fill regardless of NS depth.

**Why it might be a lever:** the 460→520 cyc/iter staging gap is
bubbles between MMA dispatches. Shorter iters → shorter bubbles
proportionally.

**Cost:** every PTX MMA emission, descriptor sizing, and TMA box dim
needs rewriting. Plus correctness re-validation.

**Realistic upside:** unknown. Plausibly 5-20 µs; plausibly negative if
the iter-overhead shift goes the wrong way.

### Strategy 3.3 — W4 slack-time recovery via geometry change

PROFILE_W4 shows W4 53% idle (`empty_wait` 6558 cyc/tile, 53% of wall).
Memory frames this as structural ("W5 backpressure, not W4 underused")
in current geometry. But that finding is at 256x256 NS=6 2x1. A new
tile shape (Tier 1 #1) could shift the backpressure point — if W5
bound moves, W4 might gain real work.

**Why it depends on Tier 1 #1:** can't unlock W4 slack while keeping
the geometry that creates the W5 backpressure. Tile-shape sweep is a
prerequisite.

**Realistic upside:** 5-15 µs if Tier 1 lands a geometry where W5 isn't
the bottleneck, and W4 cycles can be re-pipelined for prefetch /
metadata staging.

### Strategy 3.4 — Non-persistent dispatch (one cluster per tile)

We chose persistent dispatch (73 clusters × 149 tiles each) early in
fc2_w3x's life and never sweeped the alternative. Non-persistent =
launch one cluster per tile, ~74 waves drain via HW scheduler.
Different scheduling model in the design space, untested for fc2_w3x.

cuBLASLt and CUTLASS both happen to use non-persistent — and both
land slower than us (rank-1 +39 µs, CUTLASS-static +185 µs) — so the
*model itself* clearly isn't a free win; epilogue/MMA throughput
dominate. The question is whether non-persistent + our hand-tuned
per-tile work would beat our current persistent.

**Why it might be a lever:**
- Hardware scheduler picks cluster→SM placement per launch. Non-
  persistent gets fresh L2-aware placement each wave; persistent
  locks in placement at kernel launch and never rebalances. At
  10878 clusters / 74 waves = 147 cluster placements per wave,
  driver may stagger L2 sets better than our static tile-dispatch
  scheme can.
- Tile-boundary state (mbarrier rotates, register reset, TMEM dealloc)
  is amortized differently. Persistent stages 6 K-iters of next tile
  inside current epilogue (PREFILL); non-persistent has cleaner tile
  starts but pays full launch overhead per cluster.
- 95.84% tensor active says staging is the gap — and staging is
  exactly where these two models differ.

**Why it might not be:**
- Persistent kernel's main win is PREFILL (10 µs at K=3072 per
  CLAUDE.md). Going non-persistent loses that, has to be made up
  by better placement or cleaner pipeline drain.
- 10878 cluster launches × per-launch overhead (~few hundred ns?)
  is a real tax. cuBLASLt and CUTLASS absorb this because their
  per-tile work is heavier; ours is leaner, so the launch tax bites
  proportionally harder.
- Our static dispatch (dgswizzle TD=8) already hits 67.65% L2 hit
  rate with 1.043× DRAM amp. Driver-managed placement would have to
  beat that to recover the launch overhead.

**Cost:**
- Strip the persistent loop: `for (int tt = 0; tt < tiles_per_cluster;
  tt++)` collapses to `tt = 0` execution.
- Compute `cluster_id` from blockIdx.x directly (no
  `cluster_id + tt * num_clusters` formula).
- Disable PREFILL (`#undef PREFILL` or auto-guard tightens since
  `tiles_per_cluster=1` triggers the K_ITERS<20 short-circuit anyway).
- Grid becomes `(2 * TOTAL_TILES, 1, 1) = (21756, 1, 1)`.

**Realistic upside:** plausibly 10-30 µs if HW scheduler placement is
materially better non-persistent; plausibly negative if launch
overhead dominates. A clean-room rewrite path; ~half a day of work
to get a working non-persistent variant.

## Hard ceiling

Tensor pipe 95.84% active = absolute non-tensor idle ceiling 43 µs.
Adding all Tier 1+2 fixes (~15-30 µs realistic total) lands somewhere in
the **0.978-0.992 ms range**, not sub-950 µs. Closing further to the
hardware floor (0.896 ms base / 0.827 ms boost) requires Tier 3 — a
structural rewrite (stream-K, K-shape change, non-persistent dispatch,
or full geometry shift), not an additive cleanup.

The honest cap on incremental work: ~30 µs from current 1.007 ms.

## Structural differences from rank-1

Listed for awareness, not as targets to chase. fc2_w3x beats rank-1
by 39 µs at our shape — these differences haven't held us back. They
matter only as data points on alternative design choices.

- 557 more BRA when fully unrolled — only relevant if i-cache pressure
  becomes load-bearing.
- 32 STSM ops — STSM lands the same shape and is at-baseline.
- Tight looped K-body — explored via K_UNROLL sweep; tied default at
  multiples-of-NS, regressed at non-multiples.
- Non-persistent dispatch — see Tier 3 #4. Both rank-1 and CUTLASS
  use it; both lose to us. So it's not a free win. Whether it pairs
  with our hand-tuned per-tile work for a net gain is open.
