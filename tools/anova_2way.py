#!/usr/bin/env python3
"""
2-way ANOVA + per-cell stats for sweep CSVs with two factors.

Reads a CSV with columns including {factor-a, factor-b, ms} and runs:

  - per-cell stats (mean / σ / SE)
  - 2-way ANOVA: factor A main effect, factor B main effect, A×B interaction
  - pairwise Welch t-test of every cell against the slowest cell

F-distribution p-values via the regularized incomplete beta function
(Numerical Recipes Lentz continued fraction). stdlib only.

Usage:
  python3 tools/anova_2way.py wall_data.csv \
        --factor-a tile_shape \
        --factor-b multicast \
        [--out compare.txt]
"""
import argparse
import csv
import math
import sys


def regularized_beta(x, a, b):
    """I_x(a, b) — regularized incomplete beta function."""
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
    """Continued fraction for incomplete beta (Lentz's method)."""
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
    """P(F_{df1,df2} >= F)."""
    if F <= 0.0:
        return 1.0
    x = df2 / (df2 + df1 * F)
    return regularized_beta(x, df2 / 2.0, df1 / 2.0)


def t_pvalue_two_sided(t, df):
    """Two-sided p-value for Student's t."""
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
        return f"<1e-9"
    if p < 1e-6:
        return f"{p:.1e}"
    if p < 1e-3:
        return f"{p:.4f}"
    return f"{p:.4f}"


def anova_2way(rows, factor_a, factor_b):
    """
    Returns (a_levels, b_levels, cell_data, ss, df, ms, F, p)
    where cell_data[(ai, bj)] = list of measurements.
    """
    a_levels = sorted({r[factor_a] for r in rows})
    b_levels = sorted({r[factor_b] for r in rows})
    cells = {(ai, bj): [] for ai in a_levels for bj in b_levels}
    for r in rows:
        cells[(r[factor_a], r[factor_b])].append(float(r["ms"]))

    a, b = len(a_levels), len(b_levels)
    ns = [len(cells[(ai, bj)]) for ai in a_levels for bj in b_levels]
    if not ns or min(ns) < 2:
        raise ValueError(f"insufficient data per cell (min n={min(ns) if ns else 0})")
    n = min(ns)
    if max(ns) != n:
        raise ValueError(f"unbalanced cells (n in [{min(ns)}, {max(ns)}]); 2-way ANOVA needs balanced")

    all_obs = [v for k in cells for v in cells[k]]
    grand = sum(all_obs) / len(all_obs)

    cell_means = {k: sum(vs) / len(vs) for k, vs in cells.items()}
    a_means = {ai: sum(cell_means[(ai, bj)] for bj in b_levels) / b for ai in a_levels}
    b_means = {bj: sum(cell_means[(ai, bj)] for ai in a_levels) / a for bj in b_levels}

    ss_a = n * b * sum((a_means[ai] - grand) ** 2 for ai in a_levels)
    ss_b = n * a * sum((b_means[bj] - grand) ** 2 for bj in b_levels)
    ss_ab = n * sum(
        (cell_means[(ai, bj)] - a_means[ai] - b_means[bj] + grand) ** 2
        for ai in a_levels for bj in b_levels
    )
    ss_err = sum(
        (v - cell_means[k]) ** 2 for k, vs in cells.items() for v in vs
    )

    df_a = a - 1
    df_b = b - 1
    df_ab = (a - 1) * (b - 1)
    df_err = a * b * (n - 1)

    ms = {
        "A": ss_a / df_a if df_a else 0.0,
        "B": ss_b / df_b if df_b else 0.0,
        "AB": ss_ab / df_ab if df_ab else 0.0,
        "err": ss_err / df_err if df_err else 0.0,
    }
    F = {
        "A": ms["A"] / ms["err"] if ms["err"] > 0 else float("inf"),
        "B": ms["B"] / ms["err"] if ms["err"] > 0 else float("inf"),
        "AB": ms["AB"] / ms["err"] if ms["err"] > 0 else float("inf"),
    }
    p = {
        "A": f_pvalue(F["A"], df_a, df_err),
        "B": f_pvalue(F["B"], df_b, df_err),
        "AB": f_pvalue(F["AB"], df_ab, df_err),
    }
    ss = {"A": ss_a, "B": ss_b, "AB": ss_ab, "err": ss_err}
    df = {"A": df_a, "B": df_b, "AB": df_ab, "err": df_err}

    return a_levels, b_levels, cells, cell_means, a_means, b_means, ss, df, ms, F, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--factor-a", required=True,
                    help="column name for factor A (≥2 levels, e.g. tile_shape)")
    ap.add_argument("--factor-b", required=True,
                    help="column name for factor B (≥2 levels, e.g. multicast)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"ERROR: empty CSV {args.csv}", file=sys.stderr)
        sys.exit(1)

    for needed in (args.factor_a, args.factor_b, "ms", "variant"):
        if needed not in rows[0]:
            print(f"ERROR: column '{needed}' missing from CSV", file=sys.stderr)
            sys.exit(1)

    a_levels, b_levels, cells, cell_means, a_means, b_means, ss, df, ms, F, p = \
        anova_2way(rows, args.factor_a, args.factor_b)

    lines = []
    lines.append(f"2-way ANOVA: factor A = {args.factor_a}, factor B = {args.factor_b}")
    lines.append(f"reading {args.csv}")
    lines.append("")

    # Per-variant cell stats (variant column maps each row → its cell label).
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

    # Marginal means.
    lines.append(f"marginal means by {args.factor_a} (averaged over {args.factor_b}):")
    for ai in a_levels:
        lines.append(f"  {ai:18s}  mean = {a_means[ai]*1000:8.2f} µs")
    lines.append(f"marginal means by {args.factor_b} (averaged over {args.factor_a}):")
    for bj in b_levels:
        lines.append(f"  {bj:18s}  mean = {b_means[bj]*1000:8.2f} µs")
    lines.append("")

    # ANOVA table.
    lines.append("ANOVA table (units: ms² for SS, ms² for MS):")
    lines.append(f"  {'source':14s}  {'SS':>12s}  {'df':>4s}  {'MS':>12s}  "
                 f"{'F':>9s}  {'p':>10s}  verdict")
    for src, label in [("A", args.factor_a), ("B", args.factor_b),
                       ("AB", f"{args.factor_a}×{args.factor_b}"),
                       ("err", "residual")]:
        if src == "err":
            lines.append(f"  {label:14s}  {ss[src]:12.6f}  {df[src]:4d}  "
                         f"{ms[src]:12.6f}        —          —    —")
        else:
            absz_proxy = math.sqrt(F[src])
            v = verdict(absz_proxy)
            lines.append(f"  {label:14s}  {ss[src]:12.6f}  {df[src]:4d}  "
                         f"{ms[src]:12.6f}  {F[src]:9.3f}  {fmt_p(p[src]):>10s}  {v}")
    lines.append("")
    lines.append("Verdict bands by F (proxy via √F vs |z|): "
                 "TIE <1.96, WEAK <2.58, MODERATE <3.29, STRONG ≥3.29.")
    lines.append("")

    # Pairwise Welch vs slowest cell.
    cell_mean_list = sorted(cell_means.items(), key=lambda kv: -kv[1])
    ref_key, ref_mean = cell_mean_list[0]
    ref_label = f"{ref_key[0]}/{ref_key[1]}"
    lines.append(f"pairwise Welch t (each cell vs slowest = {ref_label}):")
    ref_xs = cells[ref_key]
    for k in sorted(cells):
        if k == ref_key:
            continue
        res = welch_t(cells[k], ref_xs)
        if res is None:
            continue
        v = verdict(abs(res["t"]))
        sign = "faster" if res["t"] < 0 else "slower"
        if abs(res["t"]) < 1.96:
            sign = "indistinguishable"
        label = f"{k[0]}/{k[1]}"
        p_two = t_pvalue_two_sided(res["t"], res["df"])
        lines.append(f"  {label:22s}  Δ = {res['delta_us']:+8.3f} µs   "
                     f"SE = {res['se_us']:5.3f}   "
                     f"t = {res['t']:+7.3f}   df ≈ {res['df']:5.1f}   "
                     f"p = {fmt_p(p_two):>10s}   {v:9s} ({sign})")

    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
