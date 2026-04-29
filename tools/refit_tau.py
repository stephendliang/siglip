#!/usr/bin/env python3
"""Refit the bloom-filter τ weights against the consolidated wall labels.

Reads:
  data/wall_labels_canonical.csv   (one row per variant; from build_wall_labels.py)
  data/swizzle_metrics_refit.csv   (from analyze_swizzle.py --csv)

For every structural metric, computes Kendall-τ against delta_cyc_med
across all wall-labeled variants.  Bootstrap-resamples by variant
(n_resamples=1000 by default) to test sign stability — sign-stable axes
are those whose τ direction is preserved in ≥ AGREEMENT_THRESHOLD of
bootstrap samples (default 0.95).

Outputs:
  - stdout: per-metric τ + 95% CI + sign-agreement fraction, sorted by |τ|
  - recommended SIGN_STABLE_TAU dict (drop-in replacement for
    bloom_filter.py:44-54)
  - recommended GFLIP_REF dict (centroid of gflip variant for normaliser)
  - recommended WORTHY threshold = max(0, score(slowest_known_winner))
    minus a safety margin.  Never reject a known winner.

Usage:
  python3 tools/refit_tau.py
  python3 tools/refit_tau.py --boot 2000 --min-tau 0.30 --agreement 0.95
"""
import argparse
import csv
import os
import random
import sys
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_boot import kendall_tau, percentile


WALL_TO_METRIC = {
    "dgsw":         "dgsw_G8",
    "dg2":          "dgsw_G2",
    "dg4":          "dgsw_G4",
    "dg16":         "dgsw_G16",
    "dg32":         "dgsw_G32",
    "dgsnake":      "dgsnake",
    "dgsnake_G4":   "dgsnake_G4",
    "dgsnake_G16":  "dgsnake_G16",
    "checkered":    "checkered",
    "zigzag":       "zigzag",
    "rowmajor":     "rowmajor",
    "gflip":        "gflip",
    "gflip_snrot":  "gflip_snrot",
    "gflip_lmrev":  "gflip_lmrev",
    "gflip_blkswap":   "gflip_blkswap",
    "gflip_cidperm":   "gflip_cidperm",
    "gflip_blklmrev":  "gflip_blklmrev",
    "gflip_blkmul3":   "gflip_blkmul3",
    "gflip_quartswap": "gflip_quartswap",
    "tn2br":        "tn2br",
    "dg4diag":      "dg4_diag",
    "dg4pp":        "dg4_pingpong",
    "g4swap":       "dg_g4swap",
    "lmrev":        "dg_lmrev",
    "comboAB":      "dg_combo_ab",
    "comboAC":      "dg_combo_ac",
    "snrot1":       "dg_sn_rot1",
    "snrot2":       "dg_sn_rot2",
    "lmsn":         "dg_lmsn",
    "dg_snlmrev":   "dg_sn_lmrev",
    "dg_psh3":      "dg_phasesh3",
    "dg_sn_psh3":   "dg_sn_phasesh3",
    "dg_tt_phase":  "dg_tt_phase",
    "dg_antisnake": "dg_antisnake",
    "wfd_latin":    "wfd_latin",
}


METRIC_AXES = [
    "cluster_tm_corr",
    "cluster_tn_corr",
    "adj_tn_diff",
    "adj_tm_diff",
    "tm_extent_mean",
    "tm_density_mean",
    "intra_tn_run",
    "intra_tm_run",
    "tm_tick_spread",
    "tick_irreg_mean",
    "tick_irreg_p95",
    "wf_uniq_tm_max",
    "wf_uniq_tm_mean",
    "wf_uniq_tn_mean",
    "wf_tm_entropy",
    "wf_tn_entropy",
    "l2_warm_w8",
    "l2_warm_w24",
    "fresh_tm_p50",
    "fresh_tm_total",
    "same_tn_mass_mean",
    "pair_same_tn",
    "cluster_tm_var",
    "pc_tn_traj_dist",
]


def to_num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_labels(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["variant"]] = {
                "delta_cyc_med": float(r["delta_cyc_med"]),
                "n_passes": int(r["n_passes"]),
                "source": r["source"],
                "sweep_id": r["sweep_id"],
            }
    return out


def read_metrics(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("bijective", "True") != "True":
                continue
            out[r["name"]] = {k: to_num(r.get(k)) for k in METRIC_AXES}
            out[r["name"]]["bijective"] = True
    return out


def join(labels, metrics):
    """Yield (wall_name, metric_name, label, metric_dict) for each
    label-metric pair we can match."""
    out = []
    for wall_name, lab in labels.items():
        mname = WALL_TO_METRIC.get(wall_name, wall_name)
        m = metrics.get(mname)
        if m is None:
            continue
        out.append((wall_name, mname, lab, m))
    return out


def tau_with_bootstrap(joined, axis, n_boot, seed=0xC0FFEE):
    pairs = [(j[2]["delta_cyc_med"], j[3].get(axis))
             for j in joined if j[3].get(axis) is not None]
    if len(pairs) < 5:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    point = kendall_tau(xs, ys)
    rng = random.Random(seed)
    boots = []
    n = len(pairs)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bxs = [xs[i] for i in idx]
        bys = [ys[i] for i in idx]
        if len(set(bxs)) < 2 or len(set(bys)) < 2:
            continue
        boots.append(kendall_tau(bxs, bys))
    boots.sort()
    if not boots:
        return {"tau": point, "ci_lo": point, "ci_hi": point,
                "agree": 0.0, "n_pairs": n}
    lo = percentile(boots, 2.5)
    hi = percentile(boots, 97.5)
    same_sign = sum(1 for t in boots
                    if (t > 0) == (point > 0)) / len(boots)
    return {"tau": point, "ci_lo": lo, "ci_hi": hi,
            "agree": same_sign, "n_pairs": n}


def score_with_tau(metrics_dict, tau_vec, ref_vec):
    """Same signature as bloom_filter.score_vs_gflip."""
    s = 0.0
    for feat, tau in tau_vec.items():
        v = metrics_dict.get(feat)
        ref = ref_vec.get(feat)
        if v is None or ref is None:
            continue
        delta = v - ref
        s += -tau * delta
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels",
                    default="data/wall_labels_canonical.csv")
    ap.add_argument("--metrics",
                    default="data/swizzle_metrics_refit.csv")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--min-tau", type=float, default=0.30,
                    help="drop axes whose |τ| < min")
    ap.add_argument("--agreement", type=float, default=0.95,
                    help="bootstrap sign-stability fraction required")
    ap.add_argument("--ref-variant", default="gflip_blkswap",
                    help="centroid for the score (replaces GFLIP_REF; "
                         "default gflip_blkswap = current best)")
    args = ap.parse_args()

    labels = read_labels(args.labels)
    metrics = read_metrics(args.metrics)
    joined = join(labels, metrics)
    print(f"joined {len(joined)} (variant, metric) pairs from "
          f"{len(labels)} wall-labeled / {len(metrics)} metric'd")
    unmapped = [v for v in labels if v not in WALL_TO_METRIC and v not in metrics]
    if unmapped:
        print(f"  unmapped (no metric): {unmapped}")

    print()
    print(f"Kendall-τ vs delta_cyc_med "
          f"(n={len(joined)} variants, boot={args.boot}):")
    print(f"  agreement = fraction of bootstrap τ with same sign as point")
    print()
    print(f"  {'metric':22s} {'τ':>7s} {'95% CI':>20s} {'agree':>7s} verdict")
    rows = []
    for axis in METRIC_AXES:
        r = tau_with_bootstrap(joined, axis, args.boot)
        if r is None:
            continue
        r["axis"] = axis
        rows.append(r)
    rows.sort(key=lambda r: -abs(r["tau"]))

    sign_stable = {}
    for r in rows:
        verdict = ""
        if r["agree"] >= args.agreement and abs(r["tau"]) >= args.min_tau:
            verdict = "** SIGN-STABLE"
            sign_stable[r["axis"]] = round(r["tau"], 2)
        elif r["agree"] >= args.agreement:
            verdict = "(stable but |τ| < min)"
        elif abs(r["tau"]) >= args.min_tau:
            verdict = "(strong but unstable)"
        ci_str = f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"
        print(f"  {r['axis']:22s} {r['tau']:+7.3f} {ci_str:>20s} "
              f"{r['agree']:7.3f} {verdict}")

    print()
    print(f"=== {len(sign_stable)} sign-stable axes (|τ|≥{args.min_tau}, "
          f"agree≥{args.agreement}) ===")
    print()
    print("SIGN_STABLE_TAU = {")
    for k, v in sorted(sign_stable.items(), key=lambda kv: -abs(kv[1])):
        print(f"    {k!r:24s}: {v:+.2f},")
    print("}")
    print()

    ref_metric_name = WALL_TO_METRIC.get(args.ref_variant, args.ref_variant)
    ref_metrics = metrics.get(ref_metric_name)
    if ref_metrics:
        print(f"REF = {args.ref_variant!r}  (metric name: {ref_metric_name!r})")
        print(f"GFLIP_REF = {{   # rename to BLKSWAP_REF when bloom_filter.py is updated")
        for k in sign_stable:
            v = ref_metrics.get(k)
            if v is not None:
                print(f"    {k!r:24s}: {v:.4f},")
        for k in ("tm_density_mean", "wf_uniq_tm_mean", "intra_tn_run"):
            if k not in sign_stable:
                v = ref_metrics.get(k)
                if v is not None:
                    print(f"    {k!r:24s}: {v:.4f},")
        print("}")
    print()

    print("=== threshold calibration (zero false-negatives mandate) ===")
    known_winners = ["gflip_snrot", "gflip_lmrev", "gflip_blkswap"]
    if ref_metrics:
        scores = []
        for w in known_winners + ["gflip_cidperm"]:
            mname = WALL_TO_METRIC.get(w, w)
            mdict = metrics.get(mname)
            if mdict is None:
                continue
            s = score_with_tau(mdict, sign_stable, ref_metrics)
            scores.append((w, s, labels.get(w, {}).get("delta_cyc_med")))
            tag = "WINNER" if w in known_winners else "(loser)"
            print(f"  {w:25s} score={s:+.4f}  Δcyc={labels.get(w, {}).get('delta_cyc_med', '?'):>+8.0f}  {tag}")
        worst_winner = min(s[1] for s in scores if s[0] in known_winners)
        print()
        print(f"  worst winner score        = {worst_winner:+.4f}")
        print(f"  recommended WORTHY thresh = {worst_winner - 0.10:+.4f}  "
              f"(0.10 safety margin)")
        print(f"    → any score ≥ this = WORTHY (no false negative on known winners)")


if __name__ == "__main__":
    main()
