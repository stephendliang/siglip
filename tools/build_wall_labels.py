#!/usr/bin/env python3
"""Consolidate every available paired-pass wall sweep into a single canonical
per-variant wall label: trimmed median of (cyc[v,p] - cyc[dgsw,p]) across
passes within each sweep.  Paired Δ is cohort-invariant within a sweep
(both samples drawn from the same launch p), so labels can be pooled across
sweeps even when sweeps contain different variant sets.

Sources:
  - Raw paired-pass:  data/fc2_w3x_swizzle_*/wall_data.csv
  - Manual override:  data/wall_labels_manual.csv  (for variants only seen
                       in chat-pasted aggregate tables, e.g. round-5/6/7
                       cells.  Columns: variant,delta_cyc_med,n_passes,
                       sweep_id,note)

Outputs:
  - data/wall_labels.csv             (long: one row per (variant, sweep))
  - data/wall_labels_canonical.csv   (one row per variant; aggregated)

Anchor = dgsw (always present in every sweep; in metrics CSV maps to
dgsw_G8).  delta_cyc_med < 0 means faster than dgsw.

Trim is applied the same way as tools/anova_1way.py: drop the first 33%
of reps per variant by rep order (cold L2 + thermal ramp), then pair
remaining reps to anchor.

Usage:
  python3 tools/build_wall_labels.py
  python3 tools/build_wall_labels.py --anchor dgsw --trim 0.33 --boot 500
"""
import argparse
import csv
import glob
import os
import sys
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_boot import bootstrap_ci


def read_wall_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "variant": r["variant"],
                "rep": int(r["rep"]),
                "cyc": int(r["cyc"]),
            })
    return rows


def trim_first_frac(rows_for_variant, frac):
    if frac <= 0:
        return rows_for_variant
    rows_sorted = sorted(rows_for_variant, key=lambda r: r["rep"])
    drop = int(len(rows_sorted) * frac)
    return rows_sorted[drop:]


def paired_deltas(rows, variant, anchor, trim):
    by_v = [r for r in rows if r["variant"] == variant]
    by_a = [r for r in rows if r["variant"] == anchor]
    if not by_v or not by_a:
        return []
    by_v = trim_first_frac(by_v, trim)
    by_a = trim_first_frac(by_a, trim)
    a_map = {r["rep"]: r["cyc"] for r in by_a}
    deltas = []
    for r in by_v:
        if r["rep"] in a_map:
            deltas.append(r["cyc"] - a_map[r["rep"]])
    return deltas


def summarise_sweep(sweep_path, anchor, trim, boot):
    rows = read_wall_csv(sweep_path)
    variants = sorted({r["variant"] for r in rows})
    sweep_id = os.path.basename(os.path.dirname(sweep_path))
    if anchor not in variants:
        return sweep_id, []
    out = []
    for v in variants:
        deltas = paired_deltas(rows, v, anchor, trim)
        if len(deltas) < 50:
            continue
        med = median(deltas)
        if boot > 0 and v != anchor:
            try:
                point, lo, hi = bootstrap_ci(
                    lambda xs: median(xs),
                    deltas,
                    n_resamples=boot,
                    ci=95.0,
                )
            except Exception:
                point, lo, hi = med, med, med
        else:
            point, lo, hi = med, med, med
        out.append({
            "variant": v,
            "sweep_id": sweep_id,
            "delta_cyc_med": int(round(med)),
            "n_passes": len(deltas),
            "ci_lo": int(round(lo)),
            "ci_hi": int(round(hi)),
            "source": "raw",
        })
    return sweep_id, out


def read_manual(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            out.append({
                "variant": r["variant"].strip(),
                "sweep_id": r.get("sweep_id", "manual").strip(),
                "delta_cyc_med": int(round(float(r["delta_cyc_med"]))),
                "n_passes": int(r["n_passes"]),
                "ci_lo": int(round(float(r.get("ci_lo", r["delta_cyc_med"])))),
                "ci_hi": int(round(float(r.get("ci_hi", r["delta_cyc_med"])))),
                "source": "manual",
                "note": r.get("note", ""),
            })
    return out


def canonicalise(long_rows):
    """Pick one label per variant.

    Prefer most-recent sweep_id (lex sort, dates are YYYYMMDD_HHMMSS).
    Within ties, prefer largest n_passes.  Manual entries get a synthetic
    sweep_id of 'manual_*' that sorts lower than dated sweeps unless they're
    the only source for the variant.
    """
    by_v = {}
    for r in long_rows:
        by_v.setdefault(r["variant"], []).append(r)
    canon = []
    for v, rs in by_v.items():
        rs_sorted = sorted(
            rs,
            key=lambda r: (r["source"] == "manual", r["sweep_id"], r["n_passes"]),
            reverse=True,
        )
        pick = rs_sorted[0]
        all_meds = sorted(r["delta_cyc_med"] for r in rs)
        if len(rs) > 1:
            disagree_cyc = all_meds[-1] - all_meds[0]
        else:
            disagree_cyc = 0
        canon.append({
            **pick,
            "n_sources": len(rs),
            "disagree_cyc": disagree_cyc,
        })
    return sorted(canon, key=lambda r: r["delta_cyc_med"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="dgsw")
    ap.add_argument("--trim", type=float, default=0.33)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--manual",
                    default="data/wall_labels_manual.csv")
    ap.add_argument("--out-long",
                    default="data/wall_labels.csv")
    ap.add_argument("--out-canonical",
                    default="data/wall_labels_canonical.csv")
    args = ap.parse_args()

    sweep_paths = sorted(glob.glob(
        os.path.join(args.data_dir, "fc2_w3x_swizzle_*", "wall_data.csv")
    ))
    print(f"found {len(sweep_paths)} raw sweep(s)")
    long_rows = []
    for p in sweep_paths:
        sid, sweep_rows = summarise_sweep(p, args.anchor, args.trim, args.boot)
        if not sweep_rows:
            print(f"  SKIP {sid} — anchor {args.anchor!r} not in sweep")
            continue
        print(f"  {sid}: {len(sweep_rows)} variants, "
              f"{sweep_rows[0]['n_passes']} paired passes")
        long_rows.extend(sweep_rows)

    manual_rows = read_manual(args.manual)
    print(f"manual override: {len(manual_rows)} entries from {args.manual}")
    long_rows.extend(manual_rows)

    cols_long = ["variant", "sweep_id", "delta_cyc_med", "n_passes",
                 "ci_lo", "ci_hi", "source"]
    with open(args.out_long, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols_long, extrasaction="ignore")
        w.writeheader()
        for r in sorted(long_rows,
                        key=lambda r: (r["variant"], r["sweep_id"])):
            w.writerow(r)
    print(f"wrote long-format → {args.out_long}  ({len(long_rows)} rows)")

    canon = canonicalise(long_rows)
    cols_canon = ["variant", "delta_cyc_med", "n_passes", "ci_lo", "ci_hi",
                  "n_sources", "disagree_cyc", "sweep_id", "source"]
    with open(args.out_canonical, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols_canon, extrasaction="ignore")
        w.writeheader()
        for r in canon:
            w.writerow(r)
    print(f"wrote canonical    → {args.out_canonical}  ({len(canon)} variants)")

    print()
    print(f"{'variant':25s} {'Δcyc':>8s} {'n':>6s} {'src':>4s} {'sweeps':>6s} {'disagree':>8s}")
    for r in canon:
        print(f"  {r['variant']:23s} {r['delta_cyc_med']:+8d} "
              f"{r['n_passes']:6d} {r['source']:>4s} "
              f"{r['n_sources']:6d} {r['disagree_cyc']:8d}")


if __name__ == "__main__":
    main()
