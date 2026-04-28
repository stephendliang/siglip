#!/usr/bin/env python3
"""
1-way effect-size analysis for sweep CSVs with one factor.

Reads a CSV with columns including {factor, <metric>} and runs:

  - per-cell stats (mean / σ / SE)
  - 1-way ANOVA: between-cell vs within-cell variance,
    summarized by η² (proportion of variance explained), NOT p-values
  - pairwise effect size of every cell against the fastest cell:
    empirical AUC (Mann-Whitney) + Cohen's d, NOT t / p

Why no p-values: with n in the thousands, the t-statistic scales with √n
and any tiny effect becomes "STRONG" by p<<1e-9. Effect-size measures are
sample-size-independent and answer the practical question "how much do
these distributions actually overlap?"

  AUC = P(x < y) for x drawn from cell A, y drawn from cell B.
        0.5 = indistinguishable (full overlap).
        1.0 = perfect separation, A always faster.
  d   = Cohen's d, (μ_y − μ_x) / pooled_σ.  Standardized mean diff.

Verdict bands by AUC (folded to [0.5, 1.0]):
  TIE       AUC <0.55     practically indistinguishable
  WEAK      0.55–0.65     clear shift, heavy overlap
  MODERATE  0.65–0.75     shift visible in histograms
  STRONG    0.75–0.85     mostly disjoint
  DECISIVE  ≥0.85         distributions barely overlap

Thermal-throttle defenses (vast.ai-grade noise):

  --metric cyc        Use integer wall cycles instead of ms; clock-frequency-
                      invariant, eliminates ~95% of thermal-induced σ.
  --paired COL        Block-pair samples by COL (e.g. rep / pass) and
                      analyze residuals = sample − per-block mean.  Cancels
                      whatever drift correlates with the block index.
  --trim FRAC         Drop the first FRAC fraction of samples per cell
                      (sorted by --paired column if given, else input order).
                      Removes cold-start ramp.

Combine all three for vast.ai locked-clock equivalence:

    python3 anova_1way.py wall_data.csv --factor swizzle \
        --metric cyc --paired rep --trim 0.33

Usage:
  python3 tools/anova_1way.py wall_data.csv \
        --factor tile_shape \
        [--metric ms|cyc] [--paired COL] [--trim FRAC] [--out compare.txt]
"""
import argparse
import bisect
import csv
import math
import sys


def empirical_auc(xs, ys):
    """Mann-Whitney U / (n_x * n_y) — empirical P(x < y).

    No distributional assumption.  AUC=0.5 is full overlap;
    AUC=1.0 means every x_i < every y_j.  Ties contribute 0.5
    (standard convention).
    """
    n_x, n_y = len(xs), len(ys)
    if n_x == 0 or n_y == 0:
        return 0.5
    ys_sorted = sorted(ys)
    total = 0.0
    for x in xs:
        lo = bisect.bisect_left(ys_sorted, x)
        hi = bisect.bisect_right(ys_sorted, x)
        total += (n_y - hi) + 0.5 * (hi - lo)
    return total / (n_x * n_y)


def cohens_d(xs, ys):
    """Standardized mean difference: (μ_y − μ_x) / pooled_σ.
    Positive d means ys is slower (assuming larger = slower).
    """
    n_x, n_y = len(xs), len(ys)
    if n_x < 2 or n_y < 2:
        return 0.0
    mx = sum(xs) / n_x
    my = sum(ys) / n_y
    vx = sum((x - mx) ** 2 for x in xs) / (n_x - 1)
    vy = sum((y - my) ** 2 for y in ys) / (n_y - 1)
    pooled_var = ((n_x - 1) * vx + (n_y - 1) * vy) / (n_x + n_y - 2)
    if pooled_var <= 0:
        return 0.0
    return (my - mx) / math.sqrt(pooled_var)


def auc_verdict(auc):
    a = abs(auc - 0.5) + 0.5
    if a < 0.55:
        return "TIE"
    if a < 0.65:
        return "WEAK"
    if a < 0.75:
        return "MODERATE"
    if a < 0.85:
        return "STRONG"
    return "DECISIVE"


def eta_squared_verdict(eta):
    if eta < 0.01:
        return "negligible"
    if eta < 0.06:
        return "small"
    if eta < 0.14:
        return "medium"
    return "large"


def anova_1way(rows, factor, metric="ms"):
    levels = sorted({r[factor] for r in rows})
    cells = {lv: [] for lv in levels}
    for r in rows:
        cells[r[factor]].append(float(r[metric]))

    k = len(levels)
    ns = [len(cells[lv]) for lv in levels]
    if not ns or min(ns) < 2:
        raise ValueError(f"insufficient data per cell (min n={min(ns) if ns else 0})")

    all_obs = [v for lv in levels for v in cells[lv]]
    grand = sum(all_obs) / len(all_obs)
    cell_means = {lv: sum(vs) / len(vs) for lv, vs in cells.items()}

    ss_between = sum(len(cells[lv]) * (cell_means[lv] - grand) ** 2 for lv in levels)
    ss_within  = sum((v - cell_means[lv]) ** 2 for lv in levels for v in cells[lv])
    ss_total   = ss_between + ss_within

    df_between = k - 1
    df_within  = sum(ns) - k

    ms_between = ss_between / df_between if df_between else 0.0
    ms_within  = ss_within  / df_within  if df_within  else 0.0
    F = ms_between / ms_within if ms_within > 0 else float("inf")
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    return {
        "levels": levels,
        "cells": cells,
        "cell_means": cell_means,
        "ss_between": ss_between,
        "ss_within":  ss_within,
        "df_between": df_between,
        "df_within":  df_within,
        "ms_between": ms_between,
        "ms_within":  ms_within,
        "F": F,
        "eta_sq": eta_sq,
    }


def trim_rows(rows, factor, trim, paired):
    """Drop the first `trim` fraction of samples per cell (per factor level).

    If `paired` is given, sort each cell by paired column (numeric) before
    dropping; otherwise drop in input order.
    """
    if trim <= 0.0:
        return rows
    by_cell = {}
    for r in rows:
        by_cell.setdefault(r[factor], []).append(r)
    kept = []
    for lv, group in by_cell.items():
        if paired is not None:
            group = sorted(group, key=lambda x: float(x[paired]))
        drop = int(len(group) * trim)
        kept.extend(group[drop:])
    return kept


def pair_residuals(rows, paired, metric):
    """In-place: replace rows[i][metric] with the residual of the per-`paired`
    block mean.  E.g. paired='rep' subtracts mean(ms[*, rep=p]) from each
    sample at rep=p.  Removes whatever drift correlates with the block index.
    """
    by_block = {}
    for r in rows:
        by_block.setdefault(r[paired], []).append(float(r[metric]))
    block_mean = {b: sum(vs) / len(vs) for b, vs in by_block.items()}
    out = []
    for r in rows:
        nr = dict(r)
        nr[metric] = float(r[metric]) - block_mean[r[paired]]
        out.append(nr)
    return out, block_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--factor", required=True,
                    help="column name for the factor (≥2 levels, e.g. tile_shape)")
    ap.add_argument("--metric", default="ms", choices=("ms", "cyc"),
                    help="response variable column (default: ms)")
    ap.add_argument("--paired", default=None,
                    help="block-pair column (e.g. rep, pass).  Pairwise AUC/d "
                         "runs on residuals = sample − per-block mean.")
    ap.add_argument("--trim", type=float, default=0.0,
                    help="drop first FRAC of samples per cell, sorted by "
                         "--paired if given (e.g. 0.33 drops cold-start third)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"ERROR: empty CSV {args.csv}", file=sys.stderr)
        sys.exit(1)

    needed_cols = [args.factor, args.metric, "variant"]
    if args.paired is not None:
        needed_cols.append(args.paired)
    for needed in needed_cols:
        if needed not in rows[0]:
            print(f"ERROR: column '{needed}' missing from CSV", file=sys.stderr)
            sys.exit(1)

    n_raw = len(rows)
    if args.trim > 0.0:
        rows = trim_rows(rows, args.factor, args.trim, args.paired)

    raw_by_variant = {}
    for r in rows:
        raw_by_variant.setdefault(r["variant"], []).append(float(r[args.metric]))

    if args.paired is not None:
        rows_ana, _block_mean = pair_residuals(rows, args.paired, args.metric)
    else:
        rows_ana = rows

    res = anova_1way(rows_ana, args.factor, args.metric)
    cells = res["cells"]
    cell_means = res["cell_means"]

    is_cyc = (args.metric == "cyc")
    unit_lbl = "cyc" if is_cyc else "µs"
    scale = 1.0 if is_cyc else 1000.0
    delta_lbl = "cyc" if is_cyc else "µs"

    lines = []
    lines.append(f"effect-size analysis: factor = {args.factor}  metric = {args.metric}")
    if args.paired is not None:
        lines.append(f"paired by {args.paired} (pairwise AUC/d run on residuals; "
                     f"per-cell means below are absolute pre-pairing)")
    if args.trim > 0.0:
        lines.append(f"trim = {args.trim:.2f} ({n_raw} → {len(rows)} rows after dropping "
                     f"first {int(args.trim*100)}% per cell)")
    lines.append(f"reading {args.csv}")
    lines.append("")

    lines.append(f"per-cell stats ({unit_lbl}):")
    for v in sorted(raw_by_variant):
        xs = raw_by_variant[v]
        n = len(xs)
        m = sum(xs) / n
        var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
        s = math.sqrt(var)
        se = s / math.sqrt(n) if n > 0 else 0.0
        lines.append(f"  {v:24s}  n={n:5d}  mean = {m*scale:10.2f}  "
                     f"σ = {s*scale:7.3f}  SE = {se*scale:6.3f}")
    lines.append("")

    if args.paired is not None:
        lines.append(f"per-cell residual stats (after subtracting per-{args.paired} mean):")
        resid_cells = res["cells"]
        for v in sorted(resid_cells):
            xs = resid_cells[v]
            n = len(xs)
            m = sum(xs) / n
            var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
            s = math.sqrt(var)
            se = s / math.sqrt(n) if n > 0 else 0.0
            lines.append(f"  {v:24s}  n={n:5d}  Δ̄ = {m*scale:+9.3f}  "
                         f"σ = {s*scale:7.3f}  SE = {se*scale:6.3f}")
        lines.append("")

    lines.append("ANOVA summary (η² = SS_between / SS_total — proportion of "
                 "variance explained by factor):")
    lines.append(f"  {'source':14s}  {'SS':>16s}  {'df':>5s}  {'MS':>16s}")
    lines.append(f"  {args.factor:14s}  {res['ss_between']:16.3f}  "
                 f"{res['df_between']:5d}  {res['ms_between']:16.3f}")
    lines.append(f"  {'residual':14s}  {res['ss_within']:16.3f}  "
                 f"{res['df_within']:5d}  {res['ms_within']:16.3f}")
    lines.append(f"  η² = {res['eta_sq']:.4f}  ({eta_squared_verdict(res['eta_sq'])} effect)"
                 f"   F = {res['F']:.2f}")
    lines.append("  η² bands (Cohen): <0.01 negligible, <0.06 small, <0.14 medium, ≥0.14 large.")
    lines.append("")

    sorted_cells = sorted(cell_means.items(), key=lambda kv: kv[1])
    ref_lv, ref_mean = sorted_cells[0]
    lines.append(f"pairwise effect size vs fastest = {ref_lv}:")
    lines.append(f"  AUC bands: <0.55 TIE, <0.65 WEAK, <0.75 MODERATE, "
                 f"<0.85 STRONG, ≥0.85 DECISIVE")
    lines.append(f"  {'variant':22s}  {'Δ':>14s}        {'d':>7s}    "
                 f"{'AUC':>5s}    verdict")
    ref_xs = cells[ref_lv]
    for lv, _ in sorted_cells[1:]:
        xs = cells[lv]
        if len(xs) < 2:
            continue
        auc = empirical_auc(ref_xs, xs)
        d = cohens_d(ref_xs, xs)
        v = auc_verdict(auc)
        if abs(auc - 0.5) < 0.05:
            sign = "practically indistinguishable"
        elif auc > 0.5:
            sign = "slower"
        else:
            sign = "faster"
        delta_raw = sum(xs) / len(xs) - sum(ref_xs) / len(ref_xs)
        delta_disp = delta_raw if is_cyc else delta_raw * 1000.0
        lines.append(f"  {lv:22s}  Δ = {delta_disp:+10.3f} {delta_lbl}  "
                     f"d = {d:+6.3f}   AUC = {auc:.3f}   {v:9s} ({sign})")

    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
