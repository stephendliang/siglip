#!/usr/bin/env python3
"""
1-way ANOVA + per-cell stats for sweep CSVs with one factor.

Reads a CSV with columns including {factor, ms} and runs:

  - per-cell stats (mean / σ / SE)
  - 1-way ANOVA: between-cell vs within-cell variance, F-test
  - pairwise Welch t-test of every cell against the slowest cell

F-distribution p-values via the regularized incomplete beta function
(Numerical Recipes Lentz continued fraction). stdlib only.

Usage:
  python3 tools/anova_1way.py wall_data.csv \
        --factor tile_shape \
        [--out compare.txt]
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


def anova_1way(rows, factor):
    levels = sorted({r[factor] for r in rows})
    cells = {lv: [] for lv in levels}
    for r in rows:
        cells[r[factor]].append(float(r["ms"]))

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--factor", required=True,
                    help="column name for the factor (≥2 levels, e.g. tile_shape)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"ERROR: empty CSV {args.csv}", file=sys.stderr)
        sys.exit(1)

    for needed in (args.factor, "ms", "variant"):
        if needed not in rows[0]:
            print(f"ERROR: column '{needed}' missing from CSV", file=sys.stderr)
            sys.exit(1)

    res = anova_1way(rows, args.factor)
    cells = res["cells"]
    cell_means = res["cell_means"]

    lines = []
    lines.append(f"1-way ANOVA: factor = {args.factor}")
    lines.append(f"reading {args.csv}")
    lines.append("")

    by_variant = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(float(r["ms"]))

    lines.append("per-cell stats (µs):")
    for v in sorted(by_variant):
        xs = by_variant[v]
        n = len(xs)
        m = sum(xs) / n
        var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
        s = math.sqrt(var)
        se = s / math.sqrt(n) if n > 0 else 0.0
        lines.append(f"  {v:24s}  n={n:3d}  mean = {m*1000:8.2f}  "
                     f"σ = {s*1000:5.3f}  SE = {se*1000:5.3f}")
    lines.append("")

    lines.append("ANOVA table (units: ms² for SS, ms² for MS):")
    lines.append(f"  {'source':14s}  {'SS':>12s}  {'df':>4s}  {'MS':>12s}  "
                 f"{'F':>9s}  {'p':>10s}  verdict")
    F_proxy = math.sqrt(res["F"]) if res["F"] != float("inf") else 999.0
    v_overall = verdict(F_proxy)
    lines.append(f"  {args.factor:14s}  {res['ss_between']:12.6f}  "
                 f"{res['df_between']:4d}  {res['ms_between']:12.6f}  "
                 f"{res['F']:9.3f}  {fmt_p(res['p']):>10s}  {v_overall}")
    lines.append(f"  {'residual':14s}  {res['ss_within']:12.6f}  "
                 f"{res['df_within']:4d}  {res['ms_within']:12.6f}        —          —    —")
    lines.append("")
    lines.append("Verdict bands by F (proxy via √F vs |z|): "
                 "TIE <1.96, WEAK <2.58, MODERATE <3.29, STRONG ≥3.29.")
    lines.append("")

    sorted_cells = sorted(cell_means.items(), key=lambda kv: -kv[1])
    ref_lv, ref_mean = sorted_cells[0]
    lines.append(f"pairwise Welch t (each cell vs slowest = {ref_lv}):")
    ref_xs = cells[ref_lv]
    for lv in sorted(cells):
        if lv == ref_lv:
            continue
        result = welch_t(cells[lv], ref_xs)
        if result is None:
            continue
        v = verdict(abs(result["t"]))
        sign = "faster" if result["t"] < 0 else "slower"
        if abs(result["t"]) < 1.96:
            sign = "indistinguishable"
        p_two = t_pvalue_two_sided(result["t"], result["df"])
        lines.append(f"  {lv:22s}  Δ = {result['delta_us']:+8.3f} µs   "
                     f"SE = {result['se_us']:5.3f}   "
                     f"t = {result['t']:+7.3f}   df ≈ {result['df']:5.1f}   "
                     f"p = {fmt_p(p_two):>10s}   {v:9s} ({sign})")

    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
