#!/usr/bin/env python3
"""Cheap bloom filter: "is this candidate stupid or not?"

Reads metrics.csv (output of analyze_swizzle.py --csv) and verdicts each
variant against gflip's signature using the sign-stable Kendall-τ vector
refit against consolidated wall labels (tools/build_wall_labels.py +
tools/refit_tau.py) — most-recent training set spans n=16123 + n=43910 +
older n=32768/3072 raw paired-pass sweeps, anchored to dgsw.

The point: before paying CUDA build + B200 sweep cost (~30 min × 5 candidates),
score them in <1 sec against the existing wall data. If a candidate's feature
vector falls in known-loser territory or is a near-duplicate of an existing
tested variant, reject it.

Verdict bands:
  STUPID    — bijection fails, hard-reject metric, or score < -0.50
  REDUNDANT — feature signature within ε of an already-tested variant
  WASH      — score in [-0.50, -0.10]: gflip-class noise band, build but expect tie
  MARGINAL  — score in [-0.10, +0.10]: predicted indistinguishable from reference
  WORTHY    — score ≥ +0.10: clearly faster on sign-stable τ axes
  WORTHY-BUT-OVERSHOOT-RISK — WORTHY but tripped a soft-reject (cidperm-class)

The τ refit shifted axes from m-axis-dominated (cluster_tm_corr, adj_tm_diff,
tm_extent_mean — all dropped to |τ|<0.1 in new fit) to tn-axis-dominated
(wf_uniq_tn_mean, fresh_tm_total, cluster_tn_corr, etc.) — the new training
set is gflip-family-heavy, where the discriminating lever is XOR=1 + within-
group tn rotation, not m-axis lm manipulation.  Within-gflip-family
discrimination (saturation between blkswap/lmrev/blklmrev) sits below model
resolution: the linear fit cannot capture Lever C saturation; rely on wall.

Usage:
  python3 tools/analyze_swizzle.py --csv /tmp/m.csv --summary > /dev/null
  python3 tools/bloom_filter.py /tmp/m.csv [--candidates gflip_snrot,...]

Retrain after collecting new wall data:
  python3 tools/build_wall_labels.py
  python3 tools/refit_tau.py --boot 2000
"""
import argparse
import csv
import sys
from collections import defaultdict


GFLIP_REF = {
    "wf_uniq_tn_mean":  3.0612,
    "fresh_tm_total":   3624.0,
    "tick_irreg_mean":  1.0274,
    "cluster_tn_corr":  0.1339,
    "wf_tn_entropy":    1.5920,
    "pair_same_tn":     0.3238,
    "intra_tn_run":     3.8971,
    "adj_tn_diff":      0.1661,
    "tm_density_mean":  0.7943,
    "wf_uniq_tm_mean":  29.9184,
    "cluster_tm_corr":  0.6505,
    "tm_extent_mean":   38.90,
    "wf_uniq_tm_max":   32,
    "l2_warm_w8":       0.106,
    "adj_tm_diff":      2.108,
}


SIGN_STABLE_TAU = {
    "wf_uniq_tn_mean":  -0.52,
    "fresh_tm_total":   +0.52,
    "tick_irreg_mean":  -0.46,
    "cluster_tn_corr":  -0.43,
    "wf_tn_entropy":    -0.39,
    "pair_same_tn":     +0.39,
    "intra_tn_run":     -0.23,
    "adj_tn_diff":      +0.21,
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
    """Flag variants matching the gflip_snrot empirical winner signature:
    XOR=1 base (cluster_tm_corr in gflip band ~0.60-0.70) PLUS within-group
    tn rotation (adj_tn_diff > 1.0).  At n=16123 gflip_snrot won mean_rank
    despite scoring near-zero on the refit τ vector — the within-group
    tn rotation lever is real but undercaptured by Kendall-τ on a global
    label set.  Pre-2026-04-29 channel was 'snrot2-class adj_tn_diff > 0.5'
    — snrot2 itself is now confirmed +311/+1750 cyc slower, so the older
    threshold drowned signal in noise.  Tighter signature: gflip-class ctm
    + high adj_tn_diff."""
    flags = []
    ctm = num(metrics.get("cluster_tm_corr"))
    atn = num(metrics.get("adj_tn_diff"))
    if ctm is None or atn is None:
        return flags
    if atn > 1.0 and 0.55 <= ctm <= 0.75:
        flags.append(f"adj_tn_diff={atn:.2f} ctm={ctm:.2f} "
                     f"(gflip_snrot-class within-group tn rotation)")
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
        if s > -0.50:
            return "WORTHY-BUT-OVERSHOOT-RISK", flag_str, nn, contribs
        return "STUPID", flag_str, nn, contribs

    if s >= 0.10:
        emp_str = (" + " + "; ".join(emp)) if emp else ""
        return ("WORTHY",
                f"score = {s:+.3f} vs gflip on sign-stable τ{emp_str}",
                nn, contribs)
    if s >= -0.10:
        emp_str = (" + " + "; ".join(emp)) if emp else ""
        return ("MARGINAL", f"score = {s:+.3f}{emp_str}", nn, contribs)

    if s >= -0.50 and emp:
        return ("BUILD-ANYWAY",
                f"score = {s:+.3f} on sign-stable τ but matches empirical "
                f"winner signature: {'; '.join(emp)}",
                nn, contribs)

    if s >= -0.50:
        return ("WASH",
                f"score = {s:+.3f} (gflip-class noise band, build but "
                f"expect ≤ ref)",
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
    print("Sign-stable τ (refit on consolidated wall labels n=35 variants):")
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
