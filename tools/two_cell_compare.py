#!/usr/bin/env python3
"""
Pairwise comparison of N variants from a wall_data.csv.

Reads the CSV produced by sweep_fc2_w3x_combo.sh, computes per-variant
mean / SE / σ, and runs Welch's t-tests:

  - if a variant named "vbase" is present, it's used as the reference
    and every other cell is compared against it
  - otherwise the slowest-mean cell is used as reference

Verdict bands per |t|: TIE (<1.96), WEAK (<2.58), MODERATE (<3.29),
STRONG (>=3.29).

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


def verdict(absz):
    if absz < 1.96:
        return "TIE"
    if absz < 2.58:
        return "WEAK"
    if absz < 3.29:
        return "MODERATE"
    return "STRONG"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cells = load(args.csv)
    if len(cells) < 2:
        print(f"ERROR: need >=2 variants, got {len(cells)}: {sorted(cells)}",
              file=sys.stderr)
        sys.exit(1)

    if "vbase" in cells:
        ref = "vbase"
    else:
        ref = max(cells, key=lambda k: sum(cells[k]) / len(cells[k]))

    others = sorted(k for k in cells if k != ref)

    lines = []
    lines.append(f"variant comparison: ref = {ref}")
    lines.append(f"reading {args.csv}")
    lines.append("")

    for name in [ref] + others:
        xs = cells[name]
        n = len(xs)
        m = sum(xs) / n
        v = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
        s = math.sqrt(v)
        se = s / math.sqrt(n) if n > 0 else 0.0
        lines.append(f"  {name:8s}  n={n:3d}  "
                     f"mean = {fmt_ms_to_us(m):8.2f} µs   "
                     f"σ = {fmt_ms_to_us(s):5.3f} µs   "
                     f"SE = {fmt_ms_to_us(se):5.3f} µs")

    lines.append("")
    lines.append(f"pairwise Welch t (each vs {ref}):")
    for name in others:
        res = welch_t(cells[name], cells[ref])
        if res is None:
            lines.append(f"  {name:8s} vs {ref}: insufficient data")
            continue
        delta_us = fmt_ms_to_us(res["mx"] - res["my"])
        se_us    = fmt_ms_to_us(res["se"])
        v = verdict(abs(res["t"]))
        sign = "faster" if res["t"] < 0 else "slower"
        if abs(res["t"]) < 1.96:
            sign = "indistinguishable"
        lines.append(f"  {name:8s}  Δ = {delta_us:+7.3f} µs   "
                     f"SE = {se_us:5.3f}   "
                     f"t = {res['t']:+6.3f}   "
                     f"df ≈ {res['df']:5.1f}   "
                     f"{v:9s} ({sign})")

    if len(others) >= 2:
        lines.append("")
        lines.append("pairwise among non-ref cells:")
        for i, a in enumerate(others):
            for b in others[i + 1:]:
                res = welch_t(cells[a], cells[b])
                if res is None:
                    continue
                delta_us = fmt_ms_to_us(res["mx"] - res["my"])
                se_us    = fmt_ms_to_us(res["se"])
                v = verdict(abs(res["t"]))
                lines.append(f"  {a} vs {b}:  Δ = {delta_us:+7.3f} µs   "
                             f"SE = {se_us:5.3f}   "
                             f"t = {res['t']:+6.3f}   "
                             f"df ≈ {res['df']:5.1f}   {v}")

    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
