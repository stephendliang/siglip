#!/usr/bin/env python3
"""
Unsupervised clustering + regression on the swizzle metrics from
analyze_swizzle.py, joined to the n=200 wall measurements (B200,
2026-04-27, fc2_w3x BIAS_ONLY at K=3072).

Goal: discover whether the metric vector explains the wall ranking.
Specifically — does dgsnake/gflip/tn2br cluster TOGETHER and APART from
dgsw on the metric vector?

  - If YES → the analyzer's metric set captures the lever.  The feature(s)
    that load on the separating PC tell us what the lever is.
  - If NO  → the analyzer is blind to the lever.  The hardware-level
    cluster→tile-within-group permutation has no expression in our
    wavefront-shape metrics; we'd need physical-cache-affinity probing
    (ncu) instead.

Usage:
    python3 tools/analyze_swizzle.py --csv /tmp/swizzle_metrics.csv
    python3 tools/cluster_swizzle.py /tmp/swizzle_metrics.csv

stdlib + numpy + sklearn.  No CLI flags — it's a one-shot diagnostic.
"""
import argparse
import csv
import sys

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler


# n=200, 2026-04-27 sweep result (means in µs).  Variants we have wall data
# for; chet/pmix/ncycle/etc. are computed but unlabeled.
WALL_NS200 = {
    "dgsnake":      1002.22,
    "gflip":        1002.24,
    "tn2br":        1002.33,
    "checkered":    1002.39,
    "dgsw_G8":      1002.56,
    "dg4_pingpong": 1003.01,
    "dg4_diag":     1003.03,
    "dgsw_G4":      1003.08,
    "dgsw_G2":      1003.88,
    "rowmajor":     1004.21,
    "zigzag":       1004.32,
    "dgsw_G16":     1006.85,
    "dgsw_G32":     1008.40,
}


# Pretty group hints for printing — not used in any computation.
GROUP_HINT = {
    "dgsnake":      "dgsw_G8 + asym",
    "gflip":        "dgsw_G8 + asym",
    "tn2br":        "dgsw_G8 + asym",
    "checkered":    "dgsw_G8-shape",
    "dgsw_G8":      "dgsw_G8 baseline",
    "dgsw_G4":      "small G",
    "dgsw_G2":      "small G",
    "dg4_diag":     "small G",
    "dg4_pingpong": "small G",
    "dgsw_G16":     "large G",
    "dgsw_G32":     "large G",
    "zigzag":       "linear",
    "rowmajor":     "linear",
    "dgsw_G3":      "small G",
    "dgsw_G6":      "small G",
    "ingh":         "asym broken",
    "chet":         "hybrid",
    "pmix":         "hybrid",
    "ncycle":       "ncycle family",
    "ncyrot":       "ncycle family",
    "nflat":        "ncycle family",
    "nsnake":       "ncycle family",
}


# Numeric features — drop bookkeeping columns.
SKIP_COLS = {"name", "bijective"}


def load_metrics(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def split_features(rows):
    feature_names = [k for k in rows[0].keys() if k not in SKIP_COLS]
    names = [r["name"] for r in rows]
    X = np.array(
        [[float(r[k]) for k in feature_names] for r in rows], dtype=np.float64
    )
    return names, feature_names, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", default="/tmp/swizzle_metrics.csv", nargs="?")
    args = ap.parse_args()

    rows = load_metrics(args.csv)
    names, feature_names, X = split_features(rows)

    print(f"loaded {len(names)} variants, {len(feature_names)} features")
    print()

    keep = [i for i, n in enumerate(names) if n in WALL_NS200]
    drop = [i for i, n in enumerate(names) if n not in WALL_NS200]
    print(f"wall-labeled: {len(keep)}    unlabeled (kept for clustering only): "
          f"{len(drop)}")
    print()

    Xs = StandardScaler().fit_transform(X)
    Xs_lab = Xs[keep]
    names_lab = [names[i] for i in keep]
    y_lab = np.array([WALL_NS200[n] for n in names_lab])
    y_lab_delta = y_lab - y_lab.min()

    print("=" * 76)
    print("PCA on standardized features (all variants)")
    print("=" * 76)
    pca = PCA(n_components=min(6, len(feature_names), Xs.shape[0]))
    Z = pca.fit_transform(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    for i, (ev, c) in enumerate(zip(pca.explained_variance_ratio_, cum)):
        print(f"  PC{i+1}  var={ev*100:5.1f}%  cumulative={c*100:5.1f}%")
    print()
    print("Top |loading| on PC1 / PC2 / PC3:")
    for pc in range(min(3, pca.n_components_)):
        order = np.argsort(-np.abs(pca.components_[pc]))
        top = [(feature_names[i], pca.components_[pc][i]) for i in order[:6]]
        print(f"  PC{pc+1}: " + ", ".join(f"{n}={v:+.2f}" for n, v in top))
    print()

    print("=" * 76)
    print("PC1/PC2 coordinates (sorted by wall delta where labeled)")
    print("=" * 76)
    print(f"{'name':14s}  {'PC1':>7s}  {'PC2':>7s}  {'PC3':>7s}  "
          f"{'wall_us':>9s}  {'delta':>6s}  hint")
    order_lab = sorted(keep, key=lambda i: WALL_NS200[names[i]])
    for i in order_lab:
        n = names[i]
        wall = WALL_NS200[n]
        d = wall - y_lab.min()
        hint = GROUP_HINT.get(n, "")
        print(f"  {n:12s}  {Z[i, 0]:+7.2f}  {Z[i, 1]:+7.2f}  {Z[i, 2]:+7.2f}  "
              f"{wall:9.2f}  {d:+6.2f}  {hint}")
    print("  -- unlabeled --")
    for i in drop:
        n = names[i]
        hint = GROUP_HINT.get(n, "")
        print(f"  {n:12s}  {Z[i, 0]:+7.2f}  {Z[i, 1]:+7.2f}  {Z[i, 2]:+7.2f}  "
              f"{'—':>9s}  {'—':>6s}  {hint}")
    print()

    print("=" * 76)
    print("k-means clustering on full feature space (k=4)")
    print("=" * 76)
    km = KMeans(n_clusters=4, random_state=0, n_init=10)
    cluster_labels = km.fit_predict(Xs)
    by_cluster = {}
    for n, lbl, i in zip(names, cluster_labels, range(len(names))):
        by_cluster.setdefault(int(lbl), []).append((n, WALL_NS200.get(n)))
    for cid in sorted(by_cluster):
        members = by_cluster[cid]
        labeled = [(n, w) for n, w in members if w is not None]
        avg_wall = np.mean([w for _, w in labeled]) if labeled else None
        print(f"  cluster {cid}  size={len(members):2d}"
              + (f"  mean_wall={avg_wall:8.3f} µs"
                 if avg_wall is not None else "  (no wall data)"))
        for n, w in members:
            wstr = f"{w:8.2f}" if w is not None else "       —"
            hint = GROUP_HINT.get(n, "")
            print(f"      {n:14s}  wall={wstr}   {hint}")
    print()

    print("=" * 76)
    print("hierarchical (Ward) cluster tree on labeled set")
    print("=" * 76)
    for k in (2, 3, 4, 5):
        ag = AgglomerativeClustering(n_clusters=k, linkage="ward")
        lbls = ag.fit_predict(Xs_lab)
        groups = {}
        for n, l in zip(names_lab, lbls):
            groups.setdefault(int(l), []).append(n)
        print(f"  k={k}: " + " | ".join(
            "{" + ",".join(sorted(g, key=lambda n: WALL_NS200[n])) + "}"
            for g in groups.values()
        ))
    print()

    print("=" * 76)
    print("Linear regression: wall_delta ~ standardized features (Lasso α=0.05)")
    print("=" * 76)
    lasso = Lasso(alpha=0.05, max_iter=20000)
    lasso.fit(Xs_lab, y_lab_delta)
    pred = lasso.predict(Xs_lab)
    r2 = 1 - np.sum((y_lab_delta - pred) ** 2) / np.sum(
        (y_lab_delta - y_lab_delta.mean()) ** 2
    )
    print(f"  R² = {r2:.4f}    intercept = {lasso.intercept_:+.3f}")
    coefs = list(zip(feature_names, lasso.coef_))
    coefs.sort(key=lambda kv: -abs(kv[1]))
    print(f"  {'feature':18s}  {'β (std)':>9s}  {'note'}")
    for n, c in coefs:
        if abs(c) < 1e-4:
            continue
        print(f"  {n:18s}  {c:+9.3f}")
    print()

    print("=" * 76)
    print("Plain OLS for comparison (no shrinkage)")
    print("=" * 76)
    ols = LinearRegression()
    ols.fit(Xs_lab, y_lab_delta)
    pred = ols.predict(Xs_lab)
    r2 = 1 - np.sum((y_lab_delta - pred) ** 2) / np.sum(
        (y_lab_delta - y_lab_delta.mean()) ** 2
    )
    print(f"  R² = {r2:.4f}    intercept = {ols.intercept_:+.3f}")
    coefs = list(zip(feature_names, ols.coef_))
    coefs.sort(key=lambda kv: -abs(kv[1]))
    print(f"  {'feature':18s}  {'β (std)':>9s}")
    for n, c in coefs[:10]:
        print(f"  {n:18s}  {c:+9.3f}")
    print()

    print("=" * 76)
    print("DIAGNOSIS: front-tier separation")
    print("=" * 76)
    front_tier = ["dgsnake", "gflip", "tn2br"]
    baseline = "dgsw_G8"
    front_idx = [names.index(n) for n in front_tier]
    base_idx = names.index(baseline)
    front_centroid = Xs[front_idx].mean(axis=0)
    base_vec = Xs[base_idx]
    diff = front_centroid - base_vec
    abs_diff = np.abs(diff)
    order = np.argsort(-abs_diff)
    print(f"feature-space distance between {{dgsnake, gflip, tn2br}} centroid")
    print(f"and dgsw_G8: ‖Δ‖ = {np.linalg.norm(diff):.3f} std-units")
    print()
    print("Top features by |Δ| (front centroid − dgsw_G8), in std-units:")
    print(f"  {'feature':18s}  {'Δ (std)':>9s}  {'front':>8s}  {'dgsw_G8':>8s}")
    raw_front = X[front_idx].mean(axis=0)
    raw_base = X[base_idx]
    for i in order[:10]:
        print(f"  {feature_names[i]:18s}  {diff[i]:+9.3f}  "
              f"{raw_front[i]:8.3f}  {raw_base[i]:8.3f}")
    print()

    if abs_diff.max() < 0.3:
        verdict = (
            "ANALYZER BLIND. Front tier is metric-indistinguishable from\n"
            "  dgsw_G8 (max |Δ| < 0.3 std-units). The +0.33 µs lever lives\n"
            "  outside the wavefront-shape metric set — likely SM→L2-partition\n"
            "  cache affinity from the cluster→tile-within-group permutation,\n"
            "  which only ncu can probe."
        )
    elif abs_diff.max() < 1.0:
        verdict = (
            f"ANALYZER WEAK SIGNAL. Top feature {feature_names[order[0]]} "
            f"separates by\n  {abs_diff[order[0]]:.2f} std-units — small but "
            f"present. The lever may be partly captured."
        )
    else:
        verdict = (
            f"ANALYZER CAPTURES LEVER. Top feature "
            f"{feature_names[order[0]]} separates\n"
            f"  front-tier from dgsw_G8 by {abs_diff[order[0]]:.2f} "
            f"std-units."
        )
    print("VERDICT:")
    print(f"  {verdict}")


if __name__ == "__main__":
    main()
