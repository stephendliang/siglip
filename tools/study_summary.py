#!/usr/bin/env python3
"""
Study an existing fc2_w3x sweep summary.txt: parse the per-cell wall stats
(mean_cyc, σ_resid, mean_rank, win%, AUC, Cohen's d) and join them with
the structural metrics from analyze_swizzle.py to ask "why does the
front tier win?".

Output:
  - per-variant table: wall metrics × top-τ structural features
  - Kendall-τ ranking of every structural feature against AUC and mean_rank
  - tier-mean deltas: front-tier centroid vs mid/back/floor on each feature
  - top |feature delta| candidates → the levers that the analyzer can see
    (vs analyzer-blind levers, which show up as small deltas)

Stdlib only — no scipy/sklearn dependency, since this is meant to run on
the same CPU VPS that builds the kernel.

Usage:
  python3 tools/analyze_swizzle.py --csv /tmp/swizzle_metrics.csv
  python3 tools/study_summary.py summary.txt /tmp/swizzle_metrics.csv
"""
import argparse
import csv
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_boot import bootstrap_ci, fmt_ci


# Wall name → structural CSV name mapping (analyze_swizzle.py uses _G8 /
# dg4_diag / dg_combo_ab style; the binary's VARIANT_TABLE uses the
# shorter dgsw / dg4diag / comboAB).
NAME_MAP = {
    "dgsw":      "dgsw_G8",
    "dg2":       "dgsw_G2",
    "dg4":       "dgsw_G4",
    "dg16":      "dgsw_G16",
    "dg32":      "dgsw_G32",
    "dgsnake":   "dgsnake",
    "checkered": "checkered",
    "zigzag":    "zigzag",
    "rowmajor":  "rowmajor",
    "gflip":     "gflip",
    "tn2br":     "tn2br",
    "dg4diag":   "dg4_diag",
    "dg4pp":     "dg4_pingpong",
    "g4swap":    "dg_g4swap",
    "lmrev":     "dg_lmrev",
    "comboAB":   "dg_combo_ab",
    "comboAC":   "dg_combo_ac",
    "snrot1":    "dg_sn_rot1",
    "snrot2":    "dg_sn_rot2",
    "lmsn":      "dg_lmsn",
}


def parse_summary(path):
    """Parse anova_1way.py paired summary.txt → dict[name] = wall stats."""
    out = {}
    text = open(path).read()

    # Section: per-cell residual stats
    m = re.search(r"per-cell residual stats.*?(?=\n[A-Z]|\nANOVA|\Z)", text, re.S)
    if m:
        for line in m.group(0).splitlines():
            mm = re.match(r"\s+(\S+)\s+n=\s*\d+\s+Δ̄\s*=\s*([+-]?[\d.]+)\s+σ\s*=\s*([\d.]+)", line)
            if mm:
                out.setdefault(mm.group(1), {})["resid_mean"] = float(mm.group(2))
                out[mm.group(1)]["resid_sigma"] = float(mm.group(3))

    # Section: per-cell stats (cyc) — mean and raw σ
    m = re.search(r"per-cell stats \(cyc\):.*?(?=\nper-cell residual)", text, re.S)
    if m:
        for line in m.group(0).splitlines():
            mm = re.match(r"\s+(\S+)\s+n=\s*\d+\s+mean\s*=\s*([\d.]+)\s+σ\s*=\s*([\d.]+)", line)
            if mm:
                out.setdefault(mm.group(1), {})["mean_cyc"] = float(mm.group(2))
                out[mm.group(1)]["raw_sigma"] = float(mm.group(3))

    # Section: per-pass rank analysis
    m = re.search(r"per-pass rank analysis.*?(?=\npairwise|\Z)", text, re.S)
    if m:
        for line in m.group(0).splitlines():
            mm = re.match(r"\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*/\s*\d+\s+([\d.]+)%", line)
            if mm:
                out.setdefault(mm.group(1), {})["mean_rank"] = float(mm.group(2))
                out[mm.group(1)]["sigma_rank"] = float(mm.group(3))
                out[mm.group(1)]["wins"] = float(mm.group(4))
                out[mm.group(1)]["win_pct"] = float(mm.group(5))

    # Section: pairwise Welch — Δ, d, AUC, verdict
    m = re.search(r"pairwise effect size vs fastest = (\S+):(.*)\Z", text, re.S)
    fastest = None
    if m:
        fastest = m.group(1)
        for line in m.group(2).splitlines():
            mm = re.match(r"\s+(\S+)\s+Δ\s*=\s*([+-]?[\d.]+)\s*cyc\s+d\s*=\s*([+-]?[\d.]+)\s+AUC\s*=\s*([\d.]+)\s+(\S+)", line)
            if mm:
                out.setdefault(mm.group(1), {})["delta"] = float(mm.group(2))
                out[mm.group(1)]["d"] = float(mm.group(3))
                out[mm.group(1)]["auc"] = float(mm.group(4))
                out[mm.group(1)]["verdict"] = mm.group(5)

    # Reference cell vs itself
    if fastest and fastest in out:
        out[fastest].setdefault("delta", 0.0)
        out[fastest].setdefault("d", 0.0)
        out[fastest].setdefault("auc", 0.5)
        out[fastest].setdefault("verdict", "REF")

    return out, fastest


def parse_metrics(path):
    """structural metrics CSV → dict[name] = {feature: value}."""
    rows = list(csv.DictReader(open(path)))
    out = {}
    for r in rows:
        name = r.pop("name")
        r.pop("bijective", None)
        out[name] = {k: float(v) for k, v in r.items()}
    return out


def kendall_tau(xs, ys):
    """Kendall τ-b correlation, stdlib-only."""
    assert len(xs) == len(ys)
    n = len(xs)
    if n < 2:
        return 0.0
    concordant = discordant = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1; continue
            if dy == 0:
                ty += 1; continue
            if (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    n0 = n * (n - 1) / 2
    den = math.sqrt((n0 - tx) * (n0 - ty))
    return (concordant - discordant) / den if den > 0 else 0.0


def tier_label(verdict):
    if verdict in ("TIE", "REF"): return "FRONT"
    if verdict == "WEAK":         return "FRONT"
    if verdict == "MODERATE":     return "MID"
    if verdict == "STRONG":       return "BACK"
    if verdict == "DECISIVE":     return "FLOOR"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", help="summary.txt from anova_1way.py")
    ap.add_argument("metrics", help="structural metrics CSV from analyze_swizzle.py")
    ap.add_argument("--boot", type=int, default=0,
                    help="bootstrap CI resamples on Kendall τ (default 0 = off; "
                         "1000 typical, iid resample over cells)")
    args = ap.parse_args()

    wall, fastest = parse_summary(args.summary)
    metrics = parse_metrics(args.metrics)
    feature_names = list(next(iter(metrics.values())).keys())

    # Join wall × structural via NAME_MAP.  Print missing.
    joined = []
    for wname, w in wall.items():
        sname = NAME_MAP.get(wname)
        if sname is None or sname not in metrics:
            print(f"# skip {wname}: no structural row (sname={sname})", file=sys.stderr)
            continue
        feats = metrics[sname]
        joined.append({"name": wname, **w, **{f"feat_{k}": v for k, v in feats.items()}})

    # Sort by AUC (front first).
    joined.sort(key=lambda r: r.get("auc", 1.0))

    # === Section 1: per-variant table ============================
    print(f"=== {len(joined)} variants joined (fastest reference: {fastest}) ===\n")
    print(f"{'variant':10s} {'verd':8s} {'Δcyc':>7s} {'d':>5s} {'AUC':>5s} "
          f"{'mean_rk':>7s} {'σ_rk':>5s} {'win%':>5s} {'σ_resid':>7s}")
    print("-" * 70)
    for r in joined:
        print(f"{r['name']:10s} {r.get('verdict',''):8s} "
              f"{r.get('delta',0):>+7.0f} {r.get('d',0):>+5.2f} "
              f"{r.get('auc',0):>5.3f} {r.get('mean_rank',0):>7.2f} "
              f"{r.get('sigma_rank',0):>5.2f} {r.get('win_pct',0):>5.2f} "
              f"{r.get('resid_sigma',0):>7.0f}")
    print()

    # === Section 2: Kendall-τ vs each wall metric ================
    print("=== Kendall-τ between structural features and wall ranking ===\n")
    print("(positive τ = feature value rises with worse wall; |τ|≈0 means analyzer-blind)\n")
    aucs = [r.get("auc", 0.5) for r in joined]
    ranks = [r.get("mean_rank", 0.0) for r in joined]
    deltas = [r.get("delta", 0.0) for r in joined]
    rsig = [r.get("resid_sigma", 0.0) for r in joined]
    rows = []
    boot_rows = {}
    for f in feature_names:
        xs = [r[f"feat_{f}"] for r in joined]
        rows.append((f,
                     kendall_tau(xs, aucs),
                     kendall_tau(xs, ranks),
                     kendall_tau(xs, deltas),
                     kendall_tau(xs, rsig)))
        if args.boot > 0:
            paired = list(zip(xs, aucs))

            def tau_of(samp):
                if len(samp) < 2:
                    return 0.0
                return kendall_tau([p[0] for p in samp], [p[1] for p in samp])

            t_p, lo, hi = bootstrap_ci(tau_of, paired, n_resamples=args.boot)
            sign_stable = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
            boot_rows[f] = (lo, hi, sign_stable)
    rows.sort(key=lambda x: -abs(x[1]))
    if args.boot > 0:
        print(f"{'feature':22s} {'τ_AUC':>7s} {'CI95':>17s} {'sign':>6s}  "
              f"{'τ_rank':>7s} {'τ_delta':>8s} {'τ_σresid':>9s}")
    else:
        print(f"{'feature':22s} {'τ_AUC':>7s} {'τ_rank':>7s} {'τ_delta':>8s} {'τ_σresid':>9s}")
    print("-" * (90 if args.boot > 0 else 60))
    for f, t_auc, t_rank, t_delta, t_rsig in rows:
        if args.boot > 0:
            lo, hi, stable = boot_rows[f]
            ci = f"[{lo:+.2f},{hi:+.2f}]"
            sgn = "stable" if stable else "FLIPS"
            print(f"{f:22s} {t_auc:>+7.3f} {ci:>17s} {sgn:>6s}  "
                  f"{t_rank:>+7.3f} {t_delta:>+8.3f} {t_rsig:>+9.3f}")
        else:
            print(f"{f:22s} {t_auc:>+7.3f} {t_rank:>+7.3f} {t_delta:>+8.3f} {t_rsig:>+9.3f}")
    print()

    # === Section 2a: pairwise feature τ (collinearity) ==========
    print("=== Pairwise feature collinearity (Kendall τ between structural features) ===\n")
    print("(features at |τ|≥0.7 are redundant axes — top-τ ranking double-counts them)\n")
    feat_vals = {f: [r[f"feat_{f}"] for r in joined] for f in feature_names}
    pair_taus = []
    for i, fi in enumerate(feature_names):
        for fj in feature_names[i + 1:]:
            t = kendall_tau(feat_vals[fi], feat_vals[fj])
            pair_taus.append((fi, fj, t))
    pair_taus.sort(key=lambda x: -abs(x[2]))

    high_collinear = [p for p in pair_taus if abs(p[2]) >= 0.7]
    print(f"  pairs at |τ|≥0.7: {len(high_collinear)} of "
          f"{len(pair_taus)} pairs ({100*len(high_collinear)/max(1,len(pair_taus)):.1f}%)")
    print(f"  top-15 collinear pairs:")
    print(f"  {'feature_a':22s}  {'feature_b':22s}  {'τ':>6s}")
    for fa, fb, t in pair_taus[:15]:
        print(f"  {fa:22s}  {fb:22s}  {t:>+6.3f}")
    print()

    # Greedy cluster: walk τ table descending; merge features into clusters
    # whenever they connect at |τ|≥0.7 to an existing member.
    parent = {f: f for f in feature_names}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for fa, fb, t in pair_taus:
        if abs(t) >= 0.7:
            union(fa, fb)
    clusters = {}
    for f in feature_names:
        clusters.setdefault(find(f), []).append(f)
    n_clusters = len(clusters)
    print(f"  Effective dimensionality: {n_clusters} feature clusters at |τ|≥0.7 "
          f"(from {len(feature_names)} raw features)")
    # Pick a representative per cluster: highest |τ_AUC| among its members
    aucs_local = aucs
    tau_lookup = {}
    for f in feature_names:
        tau_lookup[f] = abs(kendall_tau(feat_vals[f], aucs_local))
    for rep, members in sorted(clusters.items(),
                               key=lambda kv: -max(tau_lookup[m] for m in kv[1])):
        members_s = sorted(members, key=lambda m: -tau_lookup[m])
        best = members_s[0]
        rest = ", ".join(members_s[1:]) if len(members_s) > 1 else "—"
        print(f"    {best:22s} (|τ_AUC|={tau_lookup[best]:.2f})"
              f"  proxies for: {rest}")
    print()

    # === Section 2b: tier-conditional Kendall τ ==================
    print("=== Tier-conditional Kendall-τ (within-tier levers) ===\n")
    print("(τ across all cells can flip sign at tier boundaries, hiding the "
          "within-tier lever.)\n(τ within a tier asks: among cells already "
          "at this tier, what predicts position?)\n")
    tier_groups = {
        "FRONT": [r for r in joined if tier_label(r.get("verdict", "")) == "FRONT"],
        "MID":   [r for r in joined if tier_label(r.get("verdict", "")) == "MID"],
        "BACK":  [r for r in joined if tier_label(r.get("verdict", "")) == "BACK"],
        "FLOOR": [r for r in joined if tier_label(r.get("verdict", "")) == "FLOOR"],
    }
    for tier, members in tier_groups.items():
        if len(members) < 3:
            print(f"--- {tier} (n={len(members)}): too few cells for τ "
                  f"(need ≥3) — skipping ---\n")
            continue
        t_aucs = [r.get("auc", 0.5) for r in members]
        t_ranks = [r.get("mean_rank", 0.0) for r in members]
        rows_t = []
        for f in feature_names:
            xs = [r[f"feat_{f}"] for r in members]
            t_auc = kendall_tau(xs, t_aucs)
            t_rank = kendall_tau(xs, t_ranks)
            rows_t.append((f, t_auc, t_rank, xs))
        rows_t.sort(key=lambda x: -abs(x[1]))
        print(f"--- {tier} (n={len(members)}: "
              f"{[r['name'] for r in members]}) ---")
        print(f"  {'feature':22s} {'τ_AUC':>7s} {'τ_rank':>7s}", end="")
        if args.boot > 0:
            print(f" {'CI95(τ_AUC)':>17s} {'sign':>6s}")
        else:
            print()
        for f, t_auc, t_rank, xs in rows_t[:8]:  # top 8 only
            line = f"  {f:22s} {t_auc:>+7.3f} {t_rank:>+7.3f}"
            if args.boot > 0:
                paired = list(zip(xs, t_aucs))
                def tau_of(samp, _f=f):
                    if len(samp) < 2:
                        return 0.0
                    return kendall_tau([p[0] for p in samp],
                                       [p[1] for p in samp])
                _, lo, hi = bootstrap_ci(tau_of, paired, n_resamples=args.boot)
                stable = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
                ci = f"[{lo:+.2f},{hi:+.2f}]"
                sgn = "stable" if stable else "FLIPS"
                line += f" {ci:>17s} {sgn:>6s}"
            print(line)
        print()

    # === Section 3: tier-mean feature deltas =====================
    print("=== Tier-mean structural features (FRONT − OTHER tiers) ===\n")
    print("(large |Δ| = feature distinguishes tier; small = analyzer-blind to lever)\n")
    front = [r for r in joined if tier_label(r.get("verdict", "")) == "FRONT"]
    mid   = [r for r in joined if tier_label(r.get("verdict", "")) == "MID"]
    back  = [r for r in joined if tier_label(r.get("verdict", "")) == "BACK"]
    floor = [r for r in joined if tier_label(r.get("verdict", "")) == "FLOOR"]
    print(f"FRONT (n={len(front)}): {[r['name'] for r in front]}")
    print(f"MID   (n={len(mid)}): {[r['name'] for r in mid]}")
    print(f"BACK  (n={len(back)}): {[r['name'] for r in back]}")
    print(f"FLOOR (n={len(floor)}): {[r['name'] for r in floor]}")
    print()

    def feat_mean(rs, f):
        return sum(r[f"feat_{f}"] for r in rs) / len(rs) if rs else float("nan")

    # Compute a normalization scale so tiny vs huge features both readable.
    feat_scales = {}
    for f in feature_names:
        all_vals = [r[f"feat_{f}"] for r in joined]
        if not all_vals:
            feat_scales[f] = 1.0; continue
        s = max(all_vals) - min(all_vals)
        feat_scales[f] = s if s > 1e-9 else 1.0

    print(f"{'feature':22s} {'FRONT':>10s} {'MID':>10s} {'BACK':>10s} {'FLOOR':>10s} {'|F-M|/range':>12s}")
    print("-" * 80)
    rows = []
    for f in feature_names:
        fmean = feat_mean(front, f); mmean = feat_mean(mid, f)
        bmean = feat_mean(back, f);  flmean = feat_mean(floor, f)
        norm = abs(fmean - mmean) / feat_scales[f] if not math.isnan(mmean) else 0.0
        rows.append((f, fmean, mmean, bmean, flmean, norm))
    rows.sort(key=lambda x: -x[5])
    for f, fm, mm, bm, flm, nrm in rows:
        def s(v):
            return "    nan   " if math.isnan(v) else f"{v:>10.3f}"
        print(f"{f:22s} {s(fm)} {s(mm)} {s(bm)} {s(flm)} {nrm:>12.3f}")
    print()

    # === Section 4: verdict =====================================
    print("=== Verdict ===\n")
    top_tau = max(abs(r[1]) for r in [
        (f, kendall_tau([j[f"feat_{f}"] for j in joined], aucs), 0, 0, 0)
        for f in feature_names])
    if top_tau >= 0.7:
        v = "STRONG: top feature τ ≥ 0.7"
    elif top_tau >= 0.5:
        v = "MEDIUM: top feature τ ≥ 0.5 — analyzer captures the lever partially"
    elif top_tau >= 0.3:
        v = "WEAK: top feature τ ∈ [0.3, 0.5) — feature set carries some signal"
    else:
        v = "BLIND: top feature |τ| < 0.3 — analyzer can't see the lever"
    print(f"top |τ| (vs AUC) = {top_tau:.3f} → {v}\n")
    print("Use τ_AUC > 0.5 features as the candidate levers.  Use the tier-mean")
    print("table to read direction (FRONT lower or higher than rest).")
    print("σ_resid column tells you which dispatches are also the most CONSISTENT")
    print("(low σ_resid = stable rank-to-rank, separate axis from speed).")


if __name__ == "__main__":
    main()
