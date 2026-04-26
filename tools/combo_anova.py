#!/usr/bin/env python3
"""
OLS regression on the 2^7 fc2_w3x combo sweep.

Reads wall_data.csv produced by sweep_fc2_w3x_combo.sh, fits a linear model
on:
  - 7 main effects (one per lever)
  - 21 two-way interactions
  - 35 three-way interactions (reported but with a stricter alpha)
  - intercept

Reports each effect with its β, σ(β), |β/σ| (z-stat with σ²=residual), and a
Bonferroni-corrected significance flag at α=0.01 across the 7 mains + 21
twos = 28 effects (3-way included for visibility but not Bonferroni-counted).

Decision rule baked into the report header:
  - if no main or 2-way effect passes |β/σ| > Bonferroni-Z: KILL ALL 7 LEVERS
  - if any does: report the winning combo, recommend cross with dispatches
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path


def parse_csv(path: Path):
    rows = []
    with path.open() as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts))
            rows.append(row)
    return header, rows


LEVERS = ["EPI_2WARP", "DROP_LEAD", "DROP_TRAIL",
          "WAIT_GROUP", "NO_BULK_MEMCLBR"]


def build_design(rows, include_3way=True):
    n = len(rows)
    feats = list(LEVERS)
    twos = [(a, b) for a, b in itertools.combinations(LEVERS, 2)]
    feats += [f"{a}:{b}" for a, b in twos]
    threes = []
    if include_3way:
        threes = list(itertools.combinations(LEVERS, 3))
        feats += [f"{a}:{b}:{c}" for a, b, c in threes]

    p = len(feats) + 1
    X = [[1.0] + [0.0] * (p - 1) for _ in range(n)]
    y = [0.0] * n

    for i, row in enumerate(rows):
        y[i] = float(row["ms"])
        bits = {lev: int(row[lev]) for lev in LEVERS}
        col = 1
        for lev in LEVERS:
            X[i][col] = float(bits[lev])
            col += 1
        for a, b in twos:
            X[i][col] = float(bits[a] * bits[b])
            col += 1
        for a, b, c in threes:
            X[i][col] = float(bits[a] * bits[b] * bits[c])
            col += 1
    return X, y, ["intercept"] + feats


def matmul(A, B):
    n, m = len(A), len(A[0])
    p = len(B[0])
    out = [[0.0] * p for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        for k in range(m):
            a = Ai[k]
            if a == 0.0:
                continue
            Bk = B[k]
            outi = out[i]
            for j in range(p):
                outi[j] += a * Bk[j]
    return out


def transpose(A):
    n, m = len(A), len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]


def matvec(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        pivot = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > abs(M[pivot][i]):
                pivot = k
        M[i], M[pivot] = M[pivot], M[i]
        if abs(M[i][i]) < 1e-12:
            raise RuntimeError(f"singular pivot at row {i}")
        inv = 1.0 / M[i][i]
        for j in range(i, n + 1):
            M[i][j] *= inv
        for k in range(n):
            if k == i:
                continue
            f = M[k][i]
            if f == 0.0:
                continue
            for j in range(i, n + 1):
                M[k][j] -= f * M[i][j]
    return [M[i][n] for i in range(n)]


def invert(A):
    n = len(A)
    M = [row[:] + [1.0 if j == i else 0.0 for j in range(n)]
         for i, row in enumerate(A)]
    for i in range(n):
        pivot = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > abs(M[pivot][i]):
                pivot = k
        M[i], M[pivot] = M[pivot], M[i]
        if abs(M[i][i]) < 1e-12:
            raise RuntimeError(f"singular at row {i}")
        inv = 1.0 / M[i][i]
        for j in range(2 * n):
            M[i][j] *= inv
        for k in range(n):
            if k == i:
                continue
            f = M[k][i]
            if f == 0.0:
                continue
            for j in range(2 * n):
                M[k][j] -= f * M[i][j]
    return [[M[i][j + n] for j in range(n)] for i in range(n)]


def fit_ols(X, y):
    Xt = transpose(X)
    XtX = matmul(Xt, X)
    Xty = matvec(Xt, y)
    XtX_inv = invert(XtX)
    beta = matvec(XtX_inv, Xty)
    yhat = matvec(X, beta)
    resid = [y[i] - yhat[i] for i in range(len(y))]
    n = len(y)
    p = len(beta)
    rss = sum(r * r for r in resid)
    sigma2 = rss / max(n - p, 1)
    se = [math.sqrt(sigma2 * XtX_inv[i][i]) for i in range(p)]
    return beta, se, sigma2, resid


def bonferroni_z(alpha, n_tests):
    a = alpha / n_tests
    z = 0.0
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        p_two = math.erfc(mid / math.sqrt(2))
        if p_two > a:
            lo = mid
        else:
            hi = mid
        z = mid
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--no-3way", action="store_true",
                    help="Skip 3-way interaction terms (set this for "
                         "fractional designs where 3-ways alias with "
                         "lower-order effects).")
    args = ap.parse_args()

    header, rows = parse_csv(args.csv)
    if not rows:
        print("no rows in csv", file=sys.stderr)
        sys.exit(1)

    X, y, names = build_design(rows, include_3way=not args.no_3way)
    beta, se, sigma2, resid = fit_ols(X, y)

    n_obs = len(y)
    p = len(beta)
    rmse = math.sqrt(sigma2)

    n_main = len(LEVERS)
    n_two = n_main * (n_main - 1) // 2
    n_three = n_main * (n_main - 1) * (n_main - 2) // 6
    n_correct = n_main + n_two
    z_thresh = bonferroni_z(args.alpha, n_correct)
    z_thresh_uncorr = 2.576

    rows_out = []
    for i in range(p):
        n_colons = names[i].count(":")
        if names[i] == "intercept":
            kind = "INT"
        elif n_colons == 0:
            kind = "MAIN"
        elif n_colons == 1:
            kind = "2WAY"
        else:
            kind = "3WAY"
        z = beta[i] / se[i] if se[i] > 0 else 0.0
        sig = ""
        if kind == "INT":
            pass
        elif kind in ("MAIN", "2WAY"):
            if abs(z) > z_thresh:
                sig = "***"
            elif abs(z) > z_thresh_uncorr:
                sig = "*"
        else:
            if abs(z) > z_thresh_uncorr:
                sig = "*"
        rows_out.append((kind, names[i], beta[i], se[i], z, sig))

    main_2way_max = max(
        (abs(r[4]) for r in rows_out if r[0] in ("MAIN", "2WAY")),
        default=0.0,
    )

    sigs = [r for r in rows_out
            if r[0] in ("MAIN", "2WAY") and abs(r[4]) > z_thresh]

    lines = []
    lines.append("=" * 78)
    lines.append("fc2_w3x combinatorial wash-lever sweep — OLS regression")
    lines.append("=" * 78)
    lines.append(f"n_obs={n_obs}  p={p}  rmse={rmse*1000:.2f} µs")
    lines.append(f"  variants seen:    {len(set(r['variant'] for r in rows))} / {1 << len(LEVERS)}")
    lines.append(f"  reps × variants:  {n_obs}")
    lines.append(f"  σ_resid:          {rmse*1000:.2f} µs (within-cell wall noise)")
    lines.append("")
    lines.append(f"Bonferroni-corrected α={args.alpha} over {n_correct} mains+2way → |z| > {z_thresh:.3f}")
    lines.append(f"Uncorrected α=0.01 → |z| > {z_thresh_uncorr:.3f}")
    lines.append("")

    lines.append("DECISION:")
    if not sigs:
        lines.append(f"  KILL ALL {len(LEVERS)} LEVERS — no main or 2-way effect passes Bonferroni.")
        lines.append(f"  max |z| across mains+2way = {main_2way_max:.2f} (threshold {z_thresh:.2f})")
    else:
        helps = [r for r in sigs if r[2] < 0]
        hurts = [r for r in sigs if r[2] > 0]
        if helps and not hurts:
            lines.append(f"  FLIP DEFAULTS — {len(helps)} effect(s) significantly help (β<0):")
            for kind, nm, b, s, z, sig in helps:
                lines.append(f"    {kind:5s}  {nm:35s}  β={b*1000:+7.2f} µs  z={z:+6.2f}")
        elif hurts and not helps:
            lines.append(f"  KILL THE LEVER(S) — {len(hurts)} effect(s) significantly HURT (β>0).")
            lines.append("  Default is already off; the macro just bloats the source. Delete:")
            for kind, nm, b, s, z, sig in hurts:
                lines.append(f"    {kind:5s}  {nm:35s}  β={b*1000:+7.2f} µs  z={z:+6.2f}")
        else:
            lines.append(f"  MIXED — {len(helps)} help, {len(hurts)} hurt:")
            for kind, nm, b, s, z, sig in sigs:
                direction = "HELPS" if b < 0 else "HURTS"
                lines.append(f"    {kind:5s}  {nm:35s}  β={b*1000:+7.2f} µs  z={z:+6.2f}  [{direction}]")
    lines.append("")

    lines.append("-" * 78)
    lines.append(f"{'kind':5s}  {'name':35s}  {'β (µs)':>10s}  {'σ (µs)':>9s}  {'z':>7s}  sig")
    lines.append("-" * 78)
    by_kind_then_z = sorted(rows_out,
                            key=lambda r: (r[0] != "INT",
                                           r[0] != "MAIN",
                                           r[0] != "2WAY",
                                           -abs(r[4])))
    for kind, nm, b, s, z, sig in by_kind_then_z:
        lines.append(f"{kind:5s}  {nm:35s}  {b*1000:>+10.2f}  {s*1000:>9.3f}  {z:>+7.2f}  {sig}")

    lines.append("")
    lines.append("-" * 78)
    lines.append("Per-variant cell means (top-10 fastest):")
    lines.append("-" * 78)
    cells = defaultdict(list)
    for r in rows:
        cells[r["variant"]].append(float(r["ms"]))
    cell_means = [(v, sum(xs) / len(xs), len(xs)) for v, xs in cells.items()]
    cell_means.sort(key=lambda t: t[1])

    baseline_name = "v" + "0" * len(LEVERS)
    lines.append(f"{'variant':10s}  {'bits':10s}  {'n':>3s}  {'mean ms':>9s}  {'µs vs '+baseline_name:>14s}")
    base_mean = next((m for v, m, n in cell_means if v == baseline_name), cell_means[0][1])
    for v, m, n in cell_means[:10]:
        lines.append(f"{v:10s}  {v[1:]:10s}  {n:>3d}  {m:>9.4f}  {(m-base_mean)*1000:>+14.2f}")

    lines.append("")
    lines.append(f"{'baseline':10s}  v0000000   mean = {base_mean:.4f} ms")
    fastest_v, fastest_m, _ = cell_means[0]
    lines.append(f"{'fastest':10s}  {fastest_v}  mean = {fastest_m:.4f} ms  Δ = {(fastest_m-base_mean)*1000:+.2f} µs")
    slowest_v, slowest_m, _ = cell_means[-1]
    lines.append(f"{'slowest':10s}  {slowest_v}  mean = {slowest_m:.4f} ms  Δ = {(slowest_m-base_mean)*1000:+.2f} µs")
    lines.append("")
    lines.append(f"Spread (max − min cell mean) = {(slowest_m - fastest_m)*1000:.2f} µs")

    out = "\n".join(lines) + "\n"
    args.out.write_text(out)
    print(out)


if __name__ == "__main__":
    main()
