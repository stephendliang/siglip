#!/usr/bin/env python3
"""
TD=22 hyperparameter sweep.

Enumerates (d1, d2, d3a, d3b, threshold, flip_base) combinations organized by
named families.  For each candidate:
  1. Validates bijection over [0, TM*TN) -> (tm, tn).
  2. Computes tile-visit features via tile_regress.features().
  3. (If --predict) fits a Ridge on existing bench data, predicts fused ms.

Non-bijective configs are silently dropped.  Emits a CSV of all valid candidates
(with features and optional prediction) and prints a ranked table.

Usage:
    tools/td22_sweep.py [--layer fc2] [--bench data/bench_20260418_175934]
                       [--top 30] [--out td22_sweep.csv] [--no-predict]
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tile_regress as tr


FEATURE_KEYS = [
    "a_reuse", "b_reuse", "tm_jump", "tn_jump",
    "a_reuse_w2", "b_reuse_w2", "tm_jump2", "tn_jump2", "path_curve",
    "store_conc", "store_max", "tn_entropy", "tm_entropy",
    "unique_tn_w2", "unique_tn_w4",
    "tn_carry_w1", "tn_carry_w2", "tn_carry_w4",
]


def is_bijection(cfg):
    TM, TN = cfg["TILES_M"], cfg["TILES_N"]
    total = TM * TN
    seen = np.zeros(total, dtype=bool)
    for b in range(total):
        idx = tr.static_swizzle(22, b, cfg)
        if not (0 <= idx < total) or seen[idx]:
            return False
        seen[idx] = True
    return bool(seen.all())


def build_candidates(TM, TN):
    """Systematic enumeration over (d2, d3a, d3b, threshold, flip_base) with
    d1 pinned to total (no-op outer wrap).  Non-bijective combos are
    filtered out downstream; here we just produce the grid."""
    total = TM * TN

    d2_opts  = sorted(set([TM, 2 * TM, 3 * TM, total, total // 2, total // 3,
                           TM // 2, TM // 3, TM * TN, TM + 1, TM - 1]))
    d2_opts  = [d for d in d2_opts if d > 0]
    d3a_opts = [1, 2, 3]
    d3b_opts = [0, 1, 2, 3]
    th_opts  = [0, 1, 2, 3, 10, 100, 1000, total]
    fb_opts  = sorted(set([0, 1, TM // 4 - 1, TM // 2 - 1, TM - 2, TM - 1,
                           TM, TM + 1]))

    cands = []
    for d2 in d2_opts:
        for d3a in d3a_opts:
            for d3b in d3b_opts:
                for th in th_opts:
                    for fb in fb_opts:
                        name = f"d2={d2}_d3a={d3a}_d3b={d3b}_th={th}_fb={fb}"
                        cands.append({
                            "name":         name,
                            "RK_D1":        total,
                            "RK_D2":        d2,
                            "RK_D3A":       d3a,
                            "RK_D3B":       d3b,
                            "RK_THRESHOLD": th,
                            "RK_FLIP_BASE": fb,
                        })
    return cands


def feature_signature(feats):
    """Hash of rounded feature values — two configs with identical features
    predict identical performance.  Used to dedupe the sweep."""
    keys = sorted(FEATURE_KEYS)
    return tuple(round(feats.get(k, 0.0), 6) for k in keys)


def fit_predictor(layer, bench_dir, feature_keys):
    """Fit a Ridge on existing bench variants' (features, ms) for fused mode."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    if not os.path.isdir(bench_dir):
        raise RuntimeError(f"bench dir not found: {bench_dir}")
    df = tr.load_bench(bench_dir)
    feat_df = tr.build_feature_frame(df)
    sub = feat_df[(feat_df["layer"] == layer) & (feat_df["mode"] == "fused")].copy()
    if len(sub) < 3:
        raise RuntimeError(f"not enough {layer}/fused rows in {bench_dir}")
    missing = [k for k in feature_keys if k not in sub.columns]
    if missing:
        raise RuntimeError(f"features missing from bench frame: {missing}")
    X = sub[feature_keys].values.astype(float)
    y = sub["ms"].values.astype(float)
    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=1.0).fit(scaler.transform(X), y)
    return scaler, model, sub


def reference_features(layer):
    """Compute features for the known baselines so sweep output can be read
    in context.  Returns list of (name, feats) for dgsw, zigzag, ncycle."""
    out = []
    for name, td in [("dgswizzle", 8), ("zigzag", 11), ("ncycle", 14),
                     ("dgsnake", 19), ("default", 0), ("rowmajor", 13)]:
        try:
            feats = tr.features(layer, td, ks=0, ns=0)
            out.append((name, feats))
        except Exception as e:
            print(f"  ref {name} failed: {e}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="fc2", choices=["fc1", "fc2"])
    ap.add_argument("--bench", default="data/bench_20260418_175934")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", default="td22_sweep.csv")
    ap.add_argument("--no-predict", action="store_true")
    args = ap.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    cfg_base = dict(tr.LAYERS[args.layer])
    TM, TN = cfg_base["TILES_M"], cfg_base["TILES_N"]
    print(f"layer={args.layer}  TM={TM} TN={TN} total={TM*TN}", file=sys.stderr)

    cands = build_candidates(TM, TN)
    print(f"generated {len(cands)} candidates", file=sys.stderr)

    valid = []
    for c in cands:
        params = {k: v for k, v in c.items() if k.startswith("RK_")}
        cfg = dict(cfg_base)
        cfg.update(params)
        if is_bijection(cfg):
            valid.append(c)
    print(f"{len(valid)}/{len(cands)} bijective", file=sys.stderr)

    rows = []
    seen_sigs = {}
    for c in valid:
        params = {k: v for k, v in c.items() if k.startswith("RK_")}
        feats = tr.features(args.layer, 22, ks=0, ns=0, params=params)
        sig = feature_signature(feats)
        if sig in seen_sigs:
            seen_sigs[sig] += 1
            continue
        seen_sigs[sig] = 1
        row = dict(c)
        for k in FEATURE_KEYS:
            row[k] = feats.get(k, float("nan"))
        rows.append(row)
    print(f"{len(rows)} unique feature signatures "
          f"(deduped from {len(valid)} bijective configs)", file=sys.stderr)

    if not args.no_predict:
        try:
            scaler, model, ref_df = fit_predictor(args.layer, args.bench, FEATURE_KEYS)
            Xc = np.array([[r[k] for k in FEATURE_KEYS] for r in rows], dtype=float)
            preds = model.predict(scaler.transform(Xc))
            for r, p in zip(rows, preds):
                r["pred_ms"] = float(p)
            rows.sort(key=lambda r: r["pred_ms"])
            print(f"fit Ridge on {len(ref_df)} rows, training-min ms = {ref_df['ms'].min():.4f}",
                  file=sys.stderr)
        except Exception as e:
            print(f"prediction skipped: {e}", file=sys.stderr)

    with open(args.out, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"wrote {args.out}", file=sys.stderr)

    print("\n== Known baselines (for reference) ==")
    baselines = reference_features(args.layer)
    cols = ["name", "a_reuse", "b_reuse_w2", "tn_jump", "path_curve", "store_conc"]
    print("  ".join(f"{c:<16}" for c in cols))
    for name, f in baselines:
        vals = [name] + [f"{f.get(c, 0):.4f}" for c in cols[1:]]
        print("  ".join(f"{v:<16}" for v in vals))

    print(f"\n== Top {min(args.top, len(rows))} TD=22 candidates ==")
    show_cols = ["name"]
    if "pred_ms" in rows[0]:
        show_cols.append("pred_ms")
    show_cols += ["a_reuse", "b_reuse_w2", "tn_jump", "path_curve", "store_conc"]
    print("  ".join(f"{c:<18}" for c in show_cols))
    for r in rows[:args.top]:
        vals = []
        for c in show_cols:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        print("  ".join(f"{v:<18}" for v in vals))


if __name__ == "__main__":
    main()
