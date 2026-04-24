# fc2_w3x grievances vs cuBLASLt rank-1

Status: fc2_w3x beats rank-1 by 39 µs (1.007 ms vs 1.046 ms) on FC2 K=3072
BIAS_ONLY. Tensor pipe 95.84% active, so absolute non-tensor idle ceiling is
~43 µs. Most grievances below are code-quality issues with small-to-zero
perf upside in this regime, because the pipes they run on already have slack.
Listed anyway so future-us knows what to try first if the critical path
shifts (e.g., after a fused-residual port that adds epilogue work).

Opcode counts are from `cuobjdump --dump-sass fc2-w3x` vs local
`rank1.sass` (128x256 NS=6 2x1 2cta variant, the architectural twin).

## Grievance 1 — no packed BF16/FP32 epilogue ops

| opcode | rank1 | w3x |
|---|---|---|
| FFMA2  | 256   | 0   |
| FADD   | 256   | 32  |
| F2FP   | 128   | 16  |

Rank-1's epilogue computes `acc[0:1] + bias[0:1]` with one `FFMA2` and
converts 2 fp32 lanes to bf16x2 with one `F2FP.PACK`. We emit scalar
`FADD` + scalar `F2FP`, one column at a time. ~8× more epilogue math
insts than rank-1.

**Why it's probably not a perf lever**: FMA/FP32 and XU pipes have 43 µs
of slack under the tensor-pipe ceiling. Cutting 320 scalar insts →
~128 packed insts frees cycles on pipes that are already idle.

**When it would become one**: fused-residual port. Fused adds another
bias+add per column. If we keep scalar ops, epilogue doubles. Packed
keeps it at parity with current cycle count. Move here when porting
to fused.

**Fix**: rewrite epilogue arith as `__hfma2` / `__hadd2` on bf16x2, or
emit inline PTX `fma.rn.bf16x2` / `cvt.pack.bf16x2.sat.f32x2`.

## Grievance 2 — no stmatrix (STSM), plain STS instead

| opcode | rank1 | w3x |
|---|---|---|
| STSM | 32 | 0  |
| STS  | 1  | 24 |

Rank-1 uses `stmatrix` to write TMEM→SMEM in a warp-cooperative pattern
that matches TMA store box layout. We use plain STS after manual pack.

Known blocker (see `project_lever_c_bugs_confirmed.md`): Lever C
(`USE_STMATRIX=1`) fails because our `LDTM.32x32b.x32` reg layout
doesn't line up with stmatrix's expected layout + address-swizzle
collisions. Fix = swap to `LDTM.16dp256bit.x4` variant. Not attempted.

**Perf upside**: uncertain. Our STS count is 24 per iter, not 192, so
this is already reasonably consolidated. stmatrix would collapse
24→~8–12 shared stores and may reduce barrier stalls on the SMEM port.
Realistic: 5–10 µs if it works.

## Grievance 3 — per-thread addressing arithmetic

| opcode | rank1 | w3x |
|---|---|---|
| IMAD  | 5   | 421 |
| UIMAD | 58  | 0   |
| UPRMT | 31  | 0   |
| UISETP| 51  | 1   |

Rank-1 does descriptor advance, tile offsets, and loop predicates on
the uniform datapath (UR regs, UIMAD/UIADD3/UPRMT/UISETP). We compute
the same math per-thread in R regs, duplicating work 32× across lanes
of a warp.

**Why we got here**: most of our scalar math lives in C++ between
`asm volatile` blocks. ptxas can lift it to UR only when it sees it's
warp-uniform — which it does, mostly (`UIADD3 706`), but the R-side
remains populated because several round-trips through R exist
(see Grievance 4).

**Perf upside**: ~0 cycles in this kernel. FMA pipe has slack; per-thread
IMAD retires in the shadow. But it DOES eat R registers — 32 lanes ×
N scalar regs. Fewer R regs would let ptxas schedule the real work
better. Probably 1–2 µs if it helps at all.

## Grievance 4 — 495 R2UR round-trips

| opcode | rank1 | w3x |
|---|---|---|
| R2UR | 3 | 495 |

We compute values in R and then broadcast to UR. Rank-1 keeps scalars
uniform from the start. Each R2UR is a real cycle on the regular
datapath.

**Why**: our inline PTX blocks take arguments as `"r"` (register)
constraints. ptxas can't pass a `u` (uniform) value into an `r` slot,
so it emits R2UR before every asm block that needs a uniform value.

**Fix**: change PTX operand constraints from `"r"` to `"u"` where the
value is warp-uniform (tile index, descriptor base, SMEM offset
constants). Touches every asm block in `fc2_w3x.cu`.

**Perf upside**: small but real — 495 cycles × 147 tiles = ~73K
cycles ≈ 37 µs *if* any of it is on the critical path. Probably only
5–10% is on the critical path, so 2–4 µs realistic.

## Grievance 5 — per-`asm volatile` ELECT/BSSY/BSYNC scaffolds

| opcode | rank1 | w3x |
|---|---|---|
| ELECT | 0 | 292 |
| BSSY  | 0 | 79  |
| BSYNC | 0 | 79  |

Every `asm volatile` block in our source that's wrapped in a `@P0`
predicate gets ptxas scaffolding: `BSSY` to set up a branch-sync
region, `ELECT` to pick one lane, the PTX body, `BSYNC` to rejoin.
~450 insts of pure scaffold.

Rank-1 gets single-lane dispatch via `UGETNEXTWORKID` into a UR
predicate (`UP0`) — one uniform check, no BSSY/BSYNC.

**Fix**: merge adjacent `asm volatile` blocks into fewer, larger PTX
bodies so the scaffold amortizes. Currently we have ~10 separate asm
blocks per tile in W4 and W5; merging to 3–4 would cut scaffold
~3×.

**Perf upside**: scaffold is on the uniform/branch pipe. Limited
critical-path overlap. Realistic: 3–5 µs if ALL of it is removed,
proportionally less for partial merges.

## Grievance 6 — fully-unrolled K-loop vs tight-loop

| opcode | rank1 | w3x |
|---|---|---|
| UTCQMMA | 8   | 192 |
| BRA     | 87  | 557 |

Rank-1 runs a tight 24-iter K-loop through a backward BRA with 8
UTCQMMAs per pass. We fully unroll, emitting 192 UTCQMMAs static.
That's 4.96K static inst count vs 1.63K for rank-1 — 3× bigger
binary.

**Not a runtime grievance** — the dynamic instruction stream is
similar. But static bytes matter for i-cache footprint. 9.9K lines
of SASS vs 1.65K for rank-1 means w3x might be hitting more i-cache
misses on the first tile of the persistent loop.

**Fix**: add `#pragma unroll 4` (or 1 for no unroll) on the K-loop in
W5. Would need to benchmark carefully — we rolled it up intentionally
at some point to avoid an unrelated issue, check commit history
before reverting.

**Perf upside**: ~0 after steady state (i-cache warm). Possibly 0.5–1 µs
on the first few tiles of each launch, invisible at steady-state.

## Grievance 7 — NANOSLEEP count

| opcode | rank1 | w3x |
|---|---|---|
| NANOSLEEP | 17 | 76 |

Our mbarrier `try_wait` loops spin with `nanosleep 0xc350` (50 µs)
on failure. Rank-1 has fewer nanosleeps (17 vs 76), likely because
its mbarrier phase predictions succeed more often on the fast path.

**Fix**: review each `try_wait` site. Some could be replaced with
`mbarrier.wait` (hardware sleep, no polling) or have the nanosleep
duration reduced.

**Perf upside**: nanosleep only fires on miss. Under warm-cluster
steady-state, miss rate should be <1% per site. Not a lever.

## Grievance 8 — no ACQBULK fence after cluster barrier wait

| opcode | rank1 | w3x |
|---|---|---|
| ACQBULK | 2 | 0 |

Rank-1 emits `ACQBULK` (bulk-async acquire fence) after cluster
barrier waits. We do not. ACQBULK serializes the bulk async proxy
with subsequent ops. Our tile boundaries do a generic `fence.proxy.async`
instead, which is broader and may be stricter than needed.

Task #24 claims this was added, but the current kernel emits 0 ACQBULK.
Either the task was reverted or the macro guarding it is off by default.
Check `fc2_w3x.cu` for `#ifdef ACQBULK` or similar.

**Perf upside**: unclear. Fence strength is usually secondary to fence
placement. If we're already correct with the broader fence, switching
to ACQBULK is a code-cleanliness change.

## Grievance 9 — batched TMA ops with worse granularity

| opcode | rank1 | w3x |
|---|---|---|
| UTMALDG       | 5 | 48 |
| UTMACMDFLUSH  | 8 | 1  |
| UTMASTG       | 8 | 1  |
| LDTM          | 16 | 1 |

These look like grievances but some may be advantages. We batch TMA
store into 1 large UTMASTG + 1 UTMACMDFLUSH; rank-1 issues 8 smaller
ones. Same for LDTM (1 big vs 16 small). Both patterns are valid —
batch reduces issue overhead, smaller chunks give finer arrival
ordering to the epilogue.

Only UTMALDG is clearly higher for us (48 vs 5). That's because we
fully unroll TMA loads (K=24 iters × 2 ops ≈ 48) vs rank-1's looped
body.

**Perf relevance**: if the fused-residual port adds a residual TMA
load per tile, revisit batching to keep it at 1 op.

## Priority ranking

If a future change lands us in a regime where these matter (fused
port, dispatch shift, K-scaling), attack in this order:

1. **Grievance 1 (packed epilogue math)** — free win during fused
   port, required if epilogue work doubles.
2. **Grievance 4 (R2UR round-trips)** — most mechanical fix, just
   change `"r"` to `"u"` operand constraints. 2–4 µs realistic.
3. **Grievance 2 (stmatrix / Lever C)** — known work, known fix path,
   5–10 µs if it works.
4. **Grievance 5 (ELECT/BSSY scaffold)** — invasive (merge asm blocks)
   but 3–5 µs on offer.
5. **Grievance 3 (UR addressing)** — bundles with Grievance 4.
6. Rest — cosmetic.

## What rank-1 does that we shouldn't copy

Not every delta is a grievance. Rank-1 has:

- 557 more BRA instructions when fully unrolled — not wanted.
- 32 STSM ops — only wanted if LDTM reg-layout fix lands.
- Tight looped K-body — possibly bad for persistent-kernel tail overlap.

## Hard ceiling

Don't forget: ncu reports tensor pipe 95.84% active. ~43 µs of
absolute non-tensor idle. Adding all grievance fixes together
(~15–25 µs realistic total) lands somewhere in the 0.985–0.995 ms
range, not sub-950 µs. Closing to the hardware floor (~0.83 ms at
boost, 0.90 ms at base) requires structural changes (wider tile,
split-K, different epilogue topology), not SASS cleanups.
