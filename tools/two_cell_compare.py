#!/usr/bin/env python3
"""
Head-to-head comparison of exactly two variants from a wall_data.csv.

Reads the CSV produced by sweep_fc2_w3x_combo.sh (when configured for
the 2-cell mode), computes per-variant mean / SE / σ, runs a Welch's
t-test between the two, and prints a single recommendation line.

No regression model — with only 2 cells the OLS design is rank-deficient
on most predictors, so we drop it and answer the only question that's
identifiable: A vs B, mean Δ at what z.

stdlib only.
"""
import argparse
import csv
import math
import sys


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
    return {"mx": mx, "my": my, "vx": vx, "vy": vy,
            "nx": nx, "ny": ny, "se": se, "t": t, "df": df}


def fmt_ms_to_us(x):
    return x * 1000.0


def load(path):
    cells = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            v = row["variant"]
            ms = float(row["ms"])
            cells.setdefault(v, []).append(ms)
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cells = load(args.csv)
    if len(cells) != 2:
        print(f"ERROR: expected exactly 2 variants, got {len(cells)}: "
              f"{sorted(cells)}", file=sys.stderr)
        sys.exit(1)

    names = sorted(cells, key=lambda k: -sum(cells[k]) / len(cells[k]))
    a, b = names[0], names[1]
    xs, ys = cells[a], cells[b]

    lines = []
    lines.append(f"two-cell comparison: {a} vs {b}")
    lines.append(f"reading {args.csv}")
    lines.append("")

    for name, xs_ in [(a, xs), (b, ys)]:
        n = len(xs_)
        m = sum(xs_) / n
        v = sum((x - m) ** 2 for x in xs_) / (n - 1) if n > 1 else 0.0
        s = math.sqrt(v)
        se = s / math.sqrt(n) if n > 0 else 0.0
        lines.append(f"  {name}  n={n:3d}  "
                     f"mean = {fmt_ms_to_us(m):8.2f} µs   "
                     f"σ = {fmt_ms_to_us(s):5.3f} µs   "
                     f"SE = {fmt_ms_to_us(se):5.3f} µs")

    res = welch_t(xs, ys)
    lines.append("")
    if res is None:
        lines.append("  ERROR: insufficient data for Welch t.")
    else:
        delta_us = fmt_ms_to_us(res["mx"] - res["my"])
        se_us    = fmt_ms_to_us(res["se"])
        lines.append(f"  Δ ({a} − {b}) = {delta_us:+.3f} µs"
                     f"   SE = {se_us:.3f} µs"
                     f"   t = {res['t']:+.3f}   df ≈ {res['df']:.1f}")

        absz = abs(res["t"])
        if absz < 1.96:
            verdict = "TIE — within ±1.96σ; can't distinguish."
        elif absz < 2.58:
            verdict = "WEAK — significant at α=0.05 but not α=0.01."
        elif absz < 3.29:
            verdict = "MODERATE — significant at α=0.01."
        else:
            verdict = "STRONG — |t| ≥ 3.29 (α≈0.001)."

        if res["t"] > 0:
            faster = b
        else:
            faster = a
        lines.append(f"  verdict: {verdict}")
        if absz >= 1.96:
            lines.append(f"  fastest: {faster}  (mean lower by "
                         f"{abs(delta_us):.2f} µs)")

    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
