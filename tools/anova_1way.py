#!/usr/bin/env python3
"""
1-way ANOVA + per-cell stats for sweep CSVs with one factor.

Reads a CSV with columns including {factor, <metric>} and runs:

  - per-cell stats (mean / σ / SE)
  - 1-way ANOVA: between-cell vs within-cell variance, F-test
  - pairwise Welch t-test of every cell against the fastest cell

F-distribution p-values via the regularized incomplete beta function
(Numerical Recipes Lentz continued fraction). stdlib only.

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
import csv
import math
import sys


def regularized_beta(x, a, b):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(x, a, b) / a
    else:
        return 1.0 - bt * _betacf(1.0 - x, b, a) / b


def _betacf(x, a, b):
    EPS = 3e-7
    FPMIN = 1e-30
    MAXIT = 200
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            return h
    return h


def f_pvalue(F, df1, df2):
    if F <= 0.0:
        return 1.0
    x = df2 / (df2 + df1 * F)
    return regularized_beta(x, df2 / 2.0, df1 / 2.0)


def t_pvalue_two_sided(t, df):
    if df <= 0:
        return 1.0
    x = df / (df + t * t)
    return regularized_beta(x, df / 2.0, 0.5)


def welch_t(xs, ys):
    nx, ny = len(xs), len(ys)
    if nx < 2 or ny < 2:
        return None
    mx = sum(xs) / nx
    my = sum(ys) / ny
    vx = sum((x - mx) ** 2 for x in xs) / (nx - 1)
    vy = sum((y - my) ** 2 for y in ys) / (ny - 1)
    se = math.sqrt(vx / nx + vy / ny)
    if se == 0.0:
        return None
    t = (mx - my) / se
    num = (vx / nx + vy / ny) ** 2
    den = (vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1)
    df = num / den if den > 0 else float("inf")
    return {"t": t, "df": df, "delta_us": (mx - my) * 1000.0,
            "se_us": se * 1000.0, "mx": mx, "my": my}


def verdict(absz):
    if absz < 1.96:
        return "TIE"
    if absz < 2.58:
        return "WEAK"
    if absz < 3.29:
        return "MODERATE"
    return "STRONG"


def fmt_p(p):
    if p < 1e-9:
        return "<1e-9"
    if p < 1e-6:
        return f"{p:.1e}"
    return f"{p:.4f}"


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

    df_between = k - 1
    df_within  = sum(ns) - k

    ms_between = ss_between / df_between if df_between else 0.0
    ms_within  = ss_within  / df_within  if df_within  else 0.0
    F = ms_between / ms_within if ms_within > 0 else float("inf")
    p = f_pvalue(F, df_between, df_within)

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
        "p": p,
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
                    help="block-pair column (e.g. rep, pass).  ANOVA runs on "
                         "residuals = sample − per-block mean. Restores cell "
                         "means in the output for human-readable absolute values.")
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
    lines.append(f"1-way ANOVA: factor = {args.factor}  metric = {args.metric}")
    if args.paired is not None:
        lines.append(f"paired by {args.paired} (analysis runs on residuals; "
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

    ss_unit = f"{args.metric}²"
    lines.append(f"ANOVA table (units: {ss_unit} for SS, {ss_unit} for MS):")
    lines.append(f"  {'source':14s}  {'SS':>14s}  {'df':>5s}  {'MS':>14s}  "
                 f"{'F':>9s}  {'p':>10s}  verdict")
    F_proxy = math.sqrt(res["F"]) if res["F"] != float("inf") else 999.0
    v_overall = verdict(F_proxy)
    lines.append(f"  {args.factor:14s}  {res['ss_between']:14.6f}  "
                 f"{res['df_between']:5d}  {res['ms_between']:14.6f}  "
                 f"{res['F']:9.3f}  {fmt_p(res['p']):>10s}  {v_overall}")
    lines.append(f"  {'residual':14s}  {res['ss_within']:14.6f}  "
                 f"{res['df_within']:5d}  {res['ms_within']:14.6f}        —          —    —")
    lines.append("")
    lines.append("Verdict bands by F (proxy via √F vs |z|): "
                 "TIE <1.96, WEAK <2.58, MODERATE <3.29, STRONG ≥3.29.")
    lines.append("")

    sorted_cells = sorted(cell_means.items(), key=lambda kv: kv[1])
    ref_lv, ref_mean = sorted_cells[0]
    lines.append(f"pairwise Welch t (each cell vs fastest = {ref_lv}):")
    ref_xs = cells[ref_lv]
    for lv, _ in sorted_cells[1:]:
        result = welch_t(cells[lv], ref_xs)
        if result is None:
            continue
        v = verdict(abs(result["t"]))
        sign = "slower" if result["t"] > 0 else "faster"
        if abs(result["t"]) < 1.96:
            sign = "indistinguishable"
        p_two = t_pvalue_two_sided(result["t"], result["df"])
        delta = result["delta_us"] if not is_cyc else (result["mx"] - result["my"])
        se = result["se_us"] if not is_cyc else (result["se_us"] / 1000.0)
        lines.append(f"  {lv:22s}  Δ = {delta:+10.3f} {delta_lbl}   "
                     f"SE = {se:7.3f}   "
                     f"t = {result['t']:+7.3f}   df ≈ {result['df']:5.1f}   "
                     f"p = {fmt_p(p_two):>10s}   {v:9s} ({sign})")

    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
