# GPU Benchmarking — Accurate & Reliable

Terse playbook for sub-µs effect detection on shared/throttling GPUs (vast.ai,
shared B200). Built from the fc2_w3x sweep machinery in `tools/anova_1way.py`
and `tools/sweep_fc2_w3x_*.sh`.

## TL;DR recipe

1. Time in **cycles** via `clock64()`, not ms — clock-frequency invariant.
2. **Pass-major** loop: outer pass p, inner variant v. Emit
   `@@SAMPLE pass=p variant=v cyc=Y` per launch.
3. **Trim** the first 33–50% of passes (cold-start L2 + thermal ramp).
4. Analyze **paired by pass** (residuals = sample − per-pass mean).
5. Report **AUC**, **Cohen's d**, **η²**, **mean rank**, **win%**.
   Do not report p-values.

```bash
REPS=128 SWEEP=dgsw,dgsnake,gflip tools/sweep_fc2_w3x_swizzle.sh
# auto-runs: anova_1way.py --metric cyc --paired rep --trim 0.33
```

## Why wall time (ms) lies

- **Clock frequency drifts.** B200 boost 1.965 GHz → base 1.813 GHz is ~8%.
  A "−1 µs win" can be 8% slower silicon running 8% faster algorithm.
- **Thermal throttling** on vast.ai / unlocked clocks adds a 1–10% slow ramp
  over the first ~30 s of a sweep.
- **L2 cold-start** inflates the first 1–10 launches; ratio depends on dataset.
- **Other tenants** on shared SMs / PCIe / NVLink jitter wall, not cycles.

Locked clocks (`nvidia-smi -lgc`) fix the first one but require root and
aren't available on vast.ai. Cycles dodge it.

## Cycles via `clock64()`

PTX intrinsic, 1-cycle SM clock counter, monotonic per CTA. Embed
entry/exit reads in the kernel:

```cuda
__global__ void kern(..., uint64_t* d_wall_cyc) {
    uint64_t t0 = 0;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    /* ... kernel body ... */
    if (threadIdx.x == 0 && warp_id == 0) {
        uint64_t t1; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        d_wall_cyc[blockIdx.x] += (t1 - t0);
    }
}
```

Host: `cudaMemset` before the timed pass; `cudaMemcpy` after sync; report
`max_over_CTAs / N_TIMED_LAUNCHES`. Cost: ~2–3 regs, 0 spills.

Cycles eliminate ~95% of thermal/clock-induced σ — the dominant noise on
unlocked hardware.

## Pass-major interleaving (randomized block design)

**Don't** run "100 reps of A, then 100 reps of B." Every drift between the
two batches contaminates the comparison.

**Do** run "pass 1 of {A,B,C}, pass 2 of {A,B,C}, ..." Each pass is a
*block*; the within-block A-vs-B contrast cancels per-pass drift.

```
for p in 1..REPS:
    for v in active_variants:
        launch v; record (p, v, cyc)
```

Emit one line per launch:
```
@@SAMPLE pass=4 variant=gflip cyc=12487
```

This is the standard randomized block design from agricultural stats. The
"block" is a pass; the "treatment" is the variant. Drift between pass 4
and pass 5 doesn't matter — only the within-pass differences.

## Trim the first 33–50%

The first N passes are L2-cold and thermally cool. Drop them. Sort each
variant's samples by `pass`, take the back half:

```python
def trim_rows(rows, frac, paired):
    by_v = group_by(rows, factor)
    out = []
    for v, items in by_v.items():
        items.sort(key=lambda r: int(r[paired]))
        drop = int(len(items) * frac)
        out.extend(items[drop:])
    return out
```

`--trim 0.33` is the default. `0.5` for sketchier hardware. Verify by
plotting cyc vs pass — should be flat after the trim point.

## Mean subtraction (residuals) for paired data

Once paired, replace each sample with its residual against the per-pass mean:

```python
def pair_residuals(rows, factor, metric, paired):
    by_block = group_by(rows, paired)
    for block, items in by_block.items():
        mean = sum(float(r[metric]) for r in items) / len(items)
        for r in items: r[metric] = float(r[metric]) - mean
    return rows
```

Variance plummets when there's real drift between passes. Report the
residual σ alongside raw σ — the ratio tells you how much drift the
pass-major interleave actually cancelled.

## Why p-values fail at large n

The t-statistic scales with √n. At n=2000:

- 0.1 µs effect with σ=2 µs → t = 0.05·√2000 ≈ 2.2 → "p<0.05 SIGNIFICANT"
- But the effect is **2.5% of one σ** — practically invisible.

p-values answer "is there *any* effect" — useless when n is big enough
that even noise has p<0.001. The right question is "is the effect *large
enough to care*."

**Strip p-values from your reports.** Use effect sizes.

## AUC (Mann-Whitney, empirical)

P(random sample from Y < random sample from X). Range [0,1], 0.5 = no
distinction. Computed via bisect:

```python
def empirical_auc(xs, ys):
    xs_sorted = sorted(xs)
    s = 0.0
    for y in ys:
        lo = bisect_left(xs_sorted, y)
        hi = bisect_right(xs_sorted, y)
        s += lo + 0.5 * (hi - lo)
    return s / (len(xs) * len(ys))
```

Fold to [0.5, 1.0] for verdict bands:

| AUC (folded) | Verdict |
|---|---|
| < 0.55 | TIE |
| < 0.65 | WEAK |
| < 0.75 | MODERATE |
| < 0.85 | STRONG |
| ≥ 0.85 | DECISIVE |

n-independent, distribution-free, ties handled correctly.

## Cohen's d (standardized effect size)

```
d = (μ_y − μ_x) / pooled_σ
```

Interpretation: |d| < 0.2 trivial, < 0.5 small, < 0.8 medium, ≥ 0.8 large.
Complementary to AUC — d gives signed magnitude in σ-units, AUC gives
overlap probability. Report both.

## η² (eta-squared, variance explained)

```
η² = SS_between / SS_total
```

Across-variant share of total variance. n-invariant, unlike F-stat.

| η² | Verdict |
|---|---|
| < 0.01 | negligible |
| < 0.06 | small |
| < 0.14 | medium |
| ≥ 0.14 | large |

Use as the omnibus "is anything different" replacement for the ANOVA
F-test p-value.

## Mean rank + win count (Friedman-style)

Within each pass, sort variants by metric ascending and assign ranks 1..k
(average ties). Track per-variant mean rank and win share (k-way tie at
min splits 1/k each).

| metric | random-pick baseline | what it tells you |
|---|---|---|
| mean_rank | (k+1)/2 | central tendency across paired blocks |
| σ_rank | depends on k | rank stability |
| win% | 100/k | fraction of passes where this was *the* fastest |

Mean rank is more robust than mean cycles because it ignores absolute
scale and outliers. Win% catches "always 2nd by µs" vs "1st half the time
but 4th the other half" — same mean cycles, very different stories.

This is essentially Friedman's test without the χ² conversion. The χ²
p-value is just as useless at large n as the t-version.

## Kendall-τ (rank correlation across two metrics)

Use case: "does the cycle ranking match the wall-ms ranking?" or "does
the η²-ordering match the AUC-ordering?" Or: rank stability across two
independent runs of the same sweep.

τ ∈ [−1, 1]. Compute pair-wise concordance:

```python
def kendall_tau(xs, ys):
    n = len(xs); pairs = c = d = 0
    for i in range(n):
        for j in range(i+1, n):
            sx = (xs[i] > xs[j]) - (xs[i] < xs[j])
            sy = (ys[i] > ys[j]) - (ys[i] < ys[j])
            if sx*sy > 0: c += 1
            elif sx*sy < 0: d += 1
            pairs += 1
    return (c - d) / pairs
```

|τ| ≥ 0.7 = strong agreement, ≥ 0.5 moderate, < 0.3 weak. If wall-ms
and cycles disagree (low τ) on the *same* hardware run, your wall-ms is
clock-noise, not signal.

## Sample size — what to budget

Cycles + paired + trimmed reduces n required by 5–20× vs raw ms. Rough
guide for B200 fc2_w3x-class kernels (12k cyc/tile, σ_cyc ~ 50–100):

| effect (cyc/tile) | n needed for AUC ≥ 0.65 |
|---|---|
| 100 (≈5 µs) | ~16 |
| 50 (≈2.5 µs) | ~64 |
| 20 (≈1 µs) | ~256 |
| 10 (≈0.5 µs) | ~1024 |

Below 10 cyc effect size you're chasing measurement floor — invest in
microbenchmarks, not full-kernel timing.

## Anti-patterns to retire

- **Reporting "mean ± σ" without n.** Useless. Always report n.
- **Per-variant batched runs** ("100 of A then 100 of B"). Drift contamination.
- **ms when locked clocks aren't available.** Use cyc.
- **p-values at n > 100.** They're always significant; they tell you nothing.
- **Skipping trim.** Cold L2 + thermal ramp is real and large.
- **One-shot wall-clock comparisons** ("ours: 1.005, theirs: 1.046"). Need
  paired interleave to claim a µs-scale win.
- **Believing "this kernel is faster" from a 3-rep median.** σ on 3 reps
  is undefined; the median is one of three points.

## Reference: the analyzer

`tools/anova_1way.py` implements all of the above:

```bash
python3 tools/anova_1way.py wall_data.csv \
    --factor swizzle \
    --metric cyc \
    --paired rep \
    --trim 0.33 \
    --out compare.txt
```

Output sections:
1. Per-cell stats (n, mean, σ, range)
2. Residual stats (paired only)
3. ANOVA summary with η²
4. Per-pass mean-rank + win% (paired only)
5. Pairwise effect size vs fastest cell (Cohen's d + AUC verdict)

The sweep harnesses (`tools/sweep_fc2_w3x_swizzle.sh`,
`tools/sweep_fc2_w3x_tile.sh`) emit pass-major `@@SAMPLE` lines and call
the analyzer with the right flags by default.
