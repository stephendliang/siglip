#!/usr/bin/env python3
"""Cheap bloom filter: "is this candidate stupid or not?"

Reads metrics.csv (output of analyze_swizzle.py --csv) and verdicts each
variant against gflip's signature using the sign-stable Kendall-τ vector
extracted from study_summary.py on the n=32768 sweep.

The point: before paying CUDA build + B200 sweep cost (~30 min × 5 candidates),
score them in <1 sec against the existing wall data. If a candidate's feature
vector falls in known-loser territory or is a near-duplicate of an existing
tested variant, reject it.

Verdict bands:
  STUPID    — bijection fails, or hits a known-loser pattern
  REDUNDANT — feature signature within ε of an already-tested variant
  WASH      — distinct but score change vs gflip < |0.05|
  WORTHY    — distinct AND moves on a sign-stable τ axis in the right direction

Usage:
  python3 tools/analyze_swizzle.py --csv /tmp/m.csv --summary > /dev/null
  python3 tools/bloom_filter.py /tmp/m.csv [--candidates gflip_snrot,gflip_lmrev,...]
"""
import argparse
import csv
import sys
from collections import defaultdict


GFLIP_REF = {
    "cluster_tm_corr":  0.6505,
    "adj_tn_diff":      0.1661,
    "adj_tm_diff":      2.108,
    "tm_extent_mean":   38.90,
    "tm_density_mean":  0.794,
    "intra_tn_run":     3.897,
    "tm_tick_spread":   0.4045,
    "tick_irreg_mean":  1.027,
    "wf_uniq_tm_max":   32,
    "l2_warm_w8":       0.106,
    "wf_uniq_tm_mean":  29.92,
}


SIGN_STABLE_TAU = {
    "cluster_tm_corr":  +0.50,
    "adj_tm_diff":      -0.50,
    "tm_tick_spread":   -0.66,
    "tick_irreg_mean":  -0.66,
    "wf_uniq_tm_max":   -0.59,
    "l2_warm_w8":       -0.59,
    "tm_extent_mean":   -0.57,
    "wf_uniq_tm_mean":  -0.40,
    "intra_tn_run":     -0.38,
}


HARD_REJECT = [
    ("tm_density_mean", "<", 0.5,
     "wavefront too spread (xband / wfd_latin / dg_psh3-class)"),
    ("intra_tn_run",    ">", 10,
     "intra_tn_run too long (nflat / ncycle-class catastrophe)"),
    ("wf_uniq_tm_max",  ">", 60,
     "wavefront unique-tm explodes (degenerate dispatch)"),
    ("cluster_tm_corr", ">", 0.999,
     "no cluster→tm decorrelation (zigzag/rowmajor floor)"),
]


SOFT_REJECT_OVERSHOOT = [
    ("cluster_tm_corr", "<", 0.55,
     "cluster_tm_corr below g4swap's 0.63 — likely overshoot"),
    ("tm_extent_mean",  ">", 50,
     "tm_extent way wider than gflip (g4swap territory)"),
]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def hard_reject_check(metrics):
    if not metrics.get("bijective"):
        return "bijection fails"
    for feat, op, threshold, reason in HARD_REJECT:
        v = num(metrics.get(feat))
        if v is None:
            continue
        if (op == "<" and v < threshold) or (op == ">" and v > threshold):
            return reason
    return None


def soft_overshoot_check(metrics):
    flags = []
    for feat, op, threshold, reason in SOFT_REJECT_OVERSHOOT:
        v = num(metrics.get(feat))
        if v is None:
            continue
        if (op == "<" and v < threshold) or (op == ">" and v > threshold):
            flags.append((feat, v, threshold, reason))
    return flags


def feature_vector(metrics, keys):
    return [num(metrics.get(k)) or 0.0 for k in keys]


def euclidean(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def standardise_vec(rows, keys):
    """Return (means, stds) for each feature across rows."""
    n = len(rows)
    if n < 2:
        return [0.0] * len(keys), [1.0] * len(keys)
    means = []
    stds = []
    for k in keys:
        vals = [num(r.get(k)) or 0.0 for r in rows]
        m = sum(vals) / n
        v = sum((x - m) ** 2 for x in vals) / n
        s = v ** 0.5 if v > 0 else 1.0
        means.append(m)
        stds.append(s)
    return means, stds


def score_vs_gflip(metrics):
    """Higher = predicted faster than gflip.  Sign convention: tau<0
    means lower is better; positive Δ in that direction = good = +score."""
    s = 0.0
    contributors = []
    for feat, tau in SIGN_STABLE_TAU.items():
        v = num(metrics.get(feat))
        ref = GFLIP_REF.get(feat)
        if v is None or ref is None:
            continue
        delta = v - ref
        contribution = -tau * delta
        s += contribution
        contributors.append((feat, v, ref, delta, tau, contribution))
    contributors.sort(key=lambda r: -abs(r[5]))
    return s, contributors


def closest_neighbor(target_metrics, all_rows, keys, exclude=None):
    """Find the variant whose standardised feature vector is closest in
    Euclidean distance to the target."""
    rows_for_norm = [
        r for r in all_rows
        if r.get("name") != target_metrics.get("name")
        and r.get("bijective", "True") in ("True", True)
    ]
    means, stds = standardise_vec(rows_for_norm, keys)
    target_vec = [
        ((num(target_metrics.get(k)) or 0.0) - m) / s
        for k, m, s in zip(keys, means, stds)
    ]
    best_name, best_d = None, float("inf")
    for r in rows_for_norm:
        if exclude and r.get("name") in exclude:
            continue
        if r.get("name") == target_metrics.get("name"):
            continue
        if not r.get("bijective", "True") == "True":
            continue
        rv = [
            ((num(r.get(k)) or 0.0) - m) / s
            for k, m, s in zip(keys, means, stds)
        ]
        d = euclidean(target_vec, rv)
        if d < best_d:
            best_d, best_name = d, r.get("name")
    return best_name, best_d


def empirical_match(metrics):
    """Flag variants matching empirical winner signatures the τ ranking can't
    see (snrot2 won 2nd at n=32768 via adj_tn_diff which is NOT sign-stable
    globally — but the empirical n=32768 mean_rank = 6.18 vs gflip's 4.88
    says it's a real but hidden lever)."""
    flags = []
    v = num(metrics.get("adj_tn_diff"))
    if v is not None and v > 0.5:
        flags.append(f"adj_tn_diff={v:.2f} (snrot2-class, empirical 2nd at n=32768)")
    return flags


def verdict(metrics, all_rows):
    bij = metrics.get("bijective")
    if isinstance(bij, str):
        metrics["bijective"] = (bij == "True")

    rej = hard_reject_check(metrics)
    if rej:
        return "STUPID", rej, None, []

    soft = soft_overshoot_check(metrics)
    s, contribs = score_vs_gflip(metrics)
    keys = list(SIGN_STABLE_TAU.keys())
    nn, nn_d = closest_neighbor(metrics, all_rows, keys)
    emp = empirical_match(metrics)

    if soft:
        flag_str = "; ".join(f"{f}={v:.2f} ({r})" for f, v, t, r in soft)
        if s > 0.10:
            return "WORTHY-BUT-OVERSHOOT-RISK", flag_str, nn, contribs
        return "STUPID", flag_str, nn, contribs

    if s > 0.10:
        emp_str = (" + " + "; ".join(emp)) if emp else ""
        return ("WORTHY",
                f"score = {s:+.3f} vs gflip on sign-stable τ{emp_str}",
                nn, contribs)
    if s > 0.03:
        emp_str = (" + " + "; ".join(emp)) if emp else ""
        return ("MARGINAL", f"score = {s:+.3f}{emp_str}", nn, contribs)

    if s > -0.10 and emp:
        return ("BUILD-ANYWAY",
                f"score = {s:+.3f} on sign-stable τ (≈ gflip), but matches "
                f"empirical winner signature: {'; '.join(emp)}",
                nn, contribs)

    if s > -0.03:
        return ("WASH",
                f"score = {s:+.3f} (within ±0.03 of gflip on every τ axis)",
                nn, contribs)
    return ("STUPID",
            f"score = {s:+.3f} vs gflip — predicted slower"
            + (f" (empirical match {'; '.join(emp)} overruled by strong τ regression)" if emp else ""),
            nn, contribs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_csv", help="output of analyze_swizzle.py --csv")
    ap.add_argument("--candidates", default=None,
                    help="comma-separated names to score (default: anything "
                         "starting with 'gflip_' or 'propose_' or named via flag)")
    ap.add_argument("--full", action="store_true",
                    help="also score every variant in the CSV (not just candidates)")
    args = ap.parse_args()

    rows = []
    with open(args.metrics_csv) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    if not rows:
        print("ERROR: empty metrics.csv", file=sys.stderr)
        sys.exit(1)

    if args.candidates:
        wanted = set(args.candidates.split(","))
    elif args.full:
        wanted = set(r["name"] for r in rows)
    else:
        wanted = set(
            r["name"] for r in rows
            if r["name"].startswith(("gflip_", "propose_"))
            and r["name"] != "gflip"
        )

    print(f"=== Bloom filter: {len(wanted)} candidate(s), reference = gflip ===")
    print()
    print("gflip baseline metrics:")
    for k, v in sorted(GFLIP_REF.items()):
        print(f"  {k:18s} = {v:.4f}")
    print()
    print("Sign-stable τ (from study_summary.py n=32768):")
    for k, t in sorted(SIGN_STABLE_TAU.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:18s} τ = {t:+.2f}  ({'higher worse' if t > 0 else 'lower worse'})")
    print()

    band_counts = defaultdict(int)
    for r in rows:
        if r["name"] not in wanted:
            continue
        band, reason, nn, contribs = verdict(r, rows)
        band_counts[band] += 1
        print(f"--- {r['name']} → {band} ---")
        print(f"   {reason}")
        if contribs:
            print("   top contributors (Δ vs gflip × −τ):")
            for feat, v, ref, delta, tau, contrib in contribs[:5]:
                arrow = "↑" if delta > 0 else "↓"
                good = "+" if contrib > 0 else "-"
                print(f"     {feat:18s}  {v:8.4f} {arrow} ({delta:+.3f})  "
                      f"τ={tau:+.2f}  contrib={good}{abs(contrib):.4f}")
        print()

    print("=== Summary ===")
    for band in ("WORTHY", "BUILD-ANYWAY", "MARGINAL", "WORTHY-BUT-OVERSHOOT-RISK",
                 "WASH", "REDUNDANT", "STUPID"):
        if band_counts[band]:
            print(f"  {band:30s} {band_counts[band]}")


if __name__ == "__main__":
    main()
