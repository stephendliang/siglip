#!/usr/bin/env python3
"""
Predict the wall tier of an unseen swizzle by k-nearest-neighbor in the
structural feature space.

For each unlabeled variant (no entry in the wall summary), find its 3
nearest labeled neighbors by standardized Euclidean distance on the
analyze_swizzle.py feature vector, and report:

  - the 3 NN names + their tiers + distance
  - vote-based tier prediction (majority of NN tiers)
  - "blind-spot warning" if the variant lies outside the convex hull of
    the labeled set (any feature exceeds the labeled min/max range)
  - a per-feature delta vs. the FRONT-tier centroid for the strongest
    levers (top-3 |τ_AUC|), to flag direction-correctness

Usage:
  python3 tools/study_summary.py summary.txt /tmp/swizzle_metrics.csv  # prerequisite (computes τ)
  python3 tools/predict_swizzle.py summary.txt /tmp/swizzle_metrics.csv

Stdlib only.
"""
import argparse
import csv
import math
import re
import sys

# Reuse parsers from study_summary.py
sys.path.insert(0, "/home/sdl/net/siglip/tools")
from study_summary import parse_summary, parse_metrics, NAME_MAP, tier_label, kendall_tau


def standardize(rows, feature_names):
    """Compute mean/sd per feature on labeled rows, return (means, sds)."""
    means, sds = {}, {}
    for f in feature_names:
        vals = [r[f] for r in rows]
        m = sum(vals) / len(vals)
        v = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
        means[f] = m
        sds[f] = math.sqrt(v) if v > 1e-12 else 1.0
    return means, sds


def euclid(r1, r2, feature_names, means, sds):
    return math.sqrt(sum(
        ((r1[f] - means[f]) / sds[f] - (r2[f] - means[f]) / sds[f]) ** 2
        for f in feature_names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", help="anova_1way.py summary.txt")
    ap.add_argument("metrics", help="analyze_swizzle.py metrics CSV")
    ap.add_argument("--k", type=int, default=3, help="# nearest neighbors")
    args = ap.parse_args()

    wall, fastest = parse_summary(args.summary)
    metrics = parse_metrics(args.metrics)
    feature_names = list(next(iter(metrics.values())).keys())

    # Build labeled set: structural metrics ∩ wall data via NAME_MAP.
    # Inverse map: structural-name → wall-name (for tier lookup).
    inv_map = {sn: wn for wn, sn in NAME_MAP.items()}

    labeled = []  # rows with both structural + wall
    unlabeled = []  # rows with only structural
    for sname, feats in metrics.items():
        wname = inv_map.get(sname)
        if wname and wname in wall:
            tier = tier_label(wall[wname].get("verdict", ""))
            labeled.append({"sname": sname, "wname": wname, "tier": tier,
                            "auc": wall[wname].get("auc", 0.5),
                            "delta": wall[wname].get("delta", 0.0),
                            **feats})
        else:
            unlabeled.append({"sname": sname, **feats})

    if not labeled or not unlabeled:
        print(f"labeled={len(labeled)}, unlabeled={len(unlabeled)} — nothing to predict")
        return

    # Standardize on labeled-set statistics so unseen variants project into
    # the same coordinate system.
    means, sds = standardize(labeled, feature_names)

    # Top-τ features (use labeled set)
    aucs = [r["auc"] for r in labeled]
    tau_by_feat = sorted(
        ((f, kendall_tau([r[f] for r in labeled], aucs)) for f in feature_names),
        key=lambda x: -abs(x[1]))
    top_levers = [f for f, _ in tau_by_feat[:5]]

    # Front-tier centroid for direction check
    front_rows = [r for r in labeled if r["tier"] == "FRONT"]
    front_centroid = {f: sum(r[f] for r in front_rows) / len(front_rows)
                      for f in feature_names}
    front_centroid_std = {f: (front_centroid[f] - means[f]) / sds[f]
                          for f in feature_names}

    # Labeled feature range (for hull check)
    feat_min = {f: min(r[f] for r in labeled) for f in feature_names}
    feat_max = {f: max(r[f] for r in labeled) for f in feature_names}

    print(f"=== Prediction for {len(unlabeled)} unlabeled variants "
          f"(k={args.k} NN over {len(labeled)} labeled) ===\n")
    print(f"top τ_AUC features (labeled set): "
          + ", ".join(f"{f}({t:+.2f})" for f, t in tau_by_feat[:5]))
    print(f"front centroid: " + ", ".join(
        f"{f}={front_centroid[f]:.2f}" for f in top_levers))
    print()

    for u in unlabeled:
        # Distances to all labeled
        dists = []
        for L in labeled:
            d = euclid(u, L, feature_names, means, sds)
            dists.append((d, L["wname"], L["tier"], L["auc"], L["delta"]))
        dists.sort()
        nn = dists[:args.k]

        # Vote
        from collections import Counter
        vote = Counter(t for _, _, t, _, _ in nn)
        tier_vote, tier_n = vote.most_common(1)[0]
        # Mean AUC of NN as numeric prediction
        mean_auc = sum(a for _, _, _, a, _ in nn) / args.k
        mean_delta = sum(d for _, _, _, _, d in nn) / args.k

        # Hull check: any feature outside labeled range?
        out_of_hull = []
        for f in feature_names:
            if u[f] < feat_min[f] - 1e-9 or u[f] > feat_max[f] + 1e-9:
                std_ext = (u[f] - front_centroid[f]) / sds[f] if sds[f] else 0
                out_of_hull.append((f, u[f], feat_min[f], feat_max[f], std_ext))

        # Direction check on top levers
        direction_pulls = []
        for f in top_levers:
            u_std = (u[f] - means[f]) / sds[f]
            front_std = front_centroid_std[f]
            # τ sign: positive τ = feature rises with worse wall
            tau = dict(tau_by_feat)[f]
            # Direction: if τ>0, lower is better (closer to front).
            # If unseen is BELOW front on a τ>0 feature, that's GOOD.
            delta_from_front = u_std - front_std
            tau_aligned = -delta_from_front * (1 if tau > 0 else -1)
            direction_pulls.append((f, u[f], front_centroid[f], delta_from_front, tau_aligned, tau))

        print(f"--- {u['sname']} ---")
        print(f"  predicted tier: {tier_vote} ({tier_n}/{args.k} vote, mean_AUC={mean_auc:.3f}, mean_Δ={mean_delta:+.0f} cyc)")
        print(f"  {args.k}-NN:")
        for d, wn, t, a, dl in nn:
            print(f"    d={d:.2f}   {wn:10s}  [{t:5s}]  AUC={a:.3f}  Δ={dl:+.0f}")
        if out_of_hull:
            print(f"  ⚠ OUTSIDE labeled hull on {len(out_of_hull)} features (analyzer extrapolating):")
            for f, v, lo, hi, std_ext in out_of_hull[:5]:
                print(f"    {f}: u={v:.2f}, labeled∈[{lo:.2f},{hi:.2f}], "
                      f"{abs(std_ext):.1f} σ {'above' if v > hi else 'below'} front")
        print(f"  direction on top-τ levers (positive τ_aligned = pull toward FRONT):")
        for f, uv, fv, dfs, tau_al, tau in direction_pulls:
            print(f"    {f:18s}: u={uv:8.2f}  front={fv:8.2f}  Δstd={dfs:+5.2f}  τ_aligned={tau_al:+5.2f}  (τ={tau:+.2f})")
        print()


if __name__ == "__main__":
    main()
