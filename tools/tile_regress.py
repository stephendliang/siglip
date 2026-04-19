#!/usr/bin/env python3
"""
tile_regress.py — explain fused/gemm/strip wall-time from tile-sequence features.

Replicates static_swizzle (TD=8..21, plus stride/default as TD=0) from
tile_dispatch.cuh in pure Python, emits per-cluster tile visit sequences,
extracts ordering features, and regresses against measured ms from
data/bench_20260418_*/  wall files (and optionally stagger/kstagger CSVs).

Usage:
    ./tools/tile_regress.py                             # all modes, both layers
    ./tools/tile_regress.py --bench data/bench_.../     # specific bench dir
    ./tools/tile_regress.py --extra data/stagger_*/results.csv  # include stagger data
    ./tools/tile_regress.py --dump-features            # emit per-variant feature CSV
"""

import argparse
import csv
import glob
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score


LAYERS = {
    "fc1": dict(TILES_M=3626, TILES_N=12, K_ITERS=6, NC=74, TOTAL_TILES=43512,
                tile_count=588, DG_GROUP_SIZE=8, CK_GROUP_M=8, CK_GROUP_N=5),
    "fc2": dict(TILES_M=3626, TILES_N=3, K_ITERS=24, NC=74, TOTAL_TILES=10878,
                tile_count=147, DG_GROUP_SIZE=8, CK_GROUP_M=8, CK_GROUP_N=2),
}

DISPATCH_TD = {
    "default":   0,
    "rowmajor":  13,
    "zigzag":    11,
    "zorder":    9,
    "hilbert":   10,
    "dgswizzle": 8,
    "ncycle":    14,
    "nflat":     15,
    "nsnake":    16,
    "checkered": 18,
    "dgsnake":   19,
    "ncyrot":    21,
}


def static_swizzle(td, block_idx, cfg):
    """Replicates static_swizzle() from tile_dispatch.cuh. Returns flat tile_idx
    tm*TILES_N + tn, from which (tm, tn) is derived."""
    TM, TN, NC = cfg["TILES_M"], cfg["TILES_N"], cfg["NC"]

    if td == 0:
        cid = block_idx % NC
        ti = block_idx // NC
        tn = cid % TN
        m_rank = cid // TN
        stride = (NC - tn + TN - 1) // TN
        tm = m_rank + ti * stride
        if tm >= TM: tm = TM - 1
        return tm * TN + tn

    if td == 8:
        G = cfg["DG_GROUP_SIZE"]
        gt = TN * G
        gi = block_idx // gt
        first_m = gi * G
        in_g = block_idx % gt
        nig = min(G, TM - first_m)
        tm = first_m + in_g % nig
        tn = in_g // nig
        return tm * TN + tn

    if td == 9:
        ZB = 4
        num_bn = (TN + ZB - 1) // ZB
        num_m_full = TM // ZB
        tail_m = TM - num_m_full * ZB
        last_bn_n = TN - (num_bn - 1) * ZB
        fmt = ZB * TN
        fmt_tot = num_m_full * fmt
        if block_idx < fmt_tot:
            bm = block_idx // fmt
            within = block_idx % fmt
        else:
            bm = num_m_full
            within = block_idx - fmt_tot
        num_m = ZB if bm < num_m_full else tail_m
        bn = 0
        for b in range(num_bn):
            n_in_b = last_bn_n if b == num_bn - 1 else ZB
            tib = num_m * n_in_b
            if within < tib:
                bn = b
                break
            within -= tib
        num_n = last_bn_n if bn == num_bn - 1 else ZB
        zr = zc = 0
        count = 0
        for z in range(16):
            r = ((z >> 1) & 1) | ((z >> 2) & 2)
            col = (z & 1) | ((z >> 1) & 2)
            if r >= num_m or col >= num_n: continue
            if count == within:
                zr, zc = r, col
                break
            count += 1
        tm = bm * ZB + zr
        tn = bn * ZB + zc
        if tm >= TM: tm = TM - 1
        if tn >= TN: tn = TN - 1
        return tm * TN + tn

    if td == 10:
        HB = 4
        num_bn = (TN + HB - 1) // HB
        num_m_full = TM // HB
        tail_m = TM - num_m_full * HB
        last_bn_n = TN - (num_bn - 1) * HB
        fmt = HB * TN
        fmt_tot = num_m_full * fmt
        h_x = [0,1,1,0,0,0,1,1,2,2,3,3,3,2,2,3]
        h_y = [0,0,1,1,2,3,3,2,2,3,3,2,1,1,0,0]
        if block_idx < fmt_tot:
            bm = block_idx // fmt
            within = block_idx % fmt
        else:
            bm = num_m_full
            within = block_idx - fmt_tot
        num_m = HB if bm < num_m_full else tail_m
        bn = 0
        for b in range(num_bn):
            n_in_b = last_bn_n if b == num_bn - 1 else HB
            tib = num_m * n_in_b
            if within < tib:
                bn = b
                break
            within -= tib
        num_n = last_bn_n if bn == num_bn - 1 else HB
        hr = hc = 0
        count = 0
        for d in range(16):
            r, col = h_x[d], h_y[d]
            if r >= num_m or col >= num_n: continue
            if count == within:
                hr, hc = r, col
                break
            count += 1
        tm = bm * HB + hr
        tn = bn * HB + hc
        if tm >= TM: tm = TM - 1
        if tn >= TN: tn = TN - 1
        return tm * TN + tn

    if td == 11:
        tm = block_idx // TN
        tn = block_idx % TN
        if tm >= TM: tm = TM - 1
        if tm & 1: tn = TN - 1 - tn
        return tm * TN + tn

    if td == 12:
        tn = block_idx // TM
        tm = block_idx % TM
        if tn >= TN:
            tn = TN - 1
            tm = TM - 1
        return tm * TN + tn

    if td == 13:
        tm = block_idx // TN
        tn = block_idx % TN
        if tm >= TM: tm = TM - 1
        return tm * TN + tn

    if td == 14:
        c = block_idx % NC
        ti_local = block_idx // NC
        super_ = ti_local // TN
        tn = ti_local % TN
        tm = super_ * NC + c
        if tm >= TM: tm = TM - 1
        return tm * TN + tn

    if td == 15:
        BAND_SIZE = (NC + TN - 1) // TN
        c = block_idx % NC
        ti_local = block_idx // NC
        band = c // BAND_SIZE
        band_lane = c % BAND_SIZE
        super_ = ti_local // TN
        sub = ti_local % TN
        tn = (sub + band) % TN
        tm = super_ * NC + band * BAND_SIZE + band_lane
        if tm >= TM: tm = TM - 1
        return tm * TN + tn

    if td == 16:
        c = block_idx % NC
        ti_local = block_idx // NC
        super_ = ti_local // TN
        sub = ti_local % TN
        tn = TN - 1 - sub if (super_ & 1) else sub
        tm = super_ * NC + c
        if tm >= TM: tm = TM - 1
        return tm * TN + tn

    if td == 18:
        GM = cfg["CK_GROUP_M"]
        GN = cfg["CK_GROUP_N"]
        stripes = (TN + GN - 1) // GN
        row_tiles = GM * TN
        row_group = block_idx // row_tiles
        in_row = block_idx % row_tiles
        first_m = row_group * GM
        if first_m + GM <= TM:
            full_ss = GM * GN
            interior = (stripes - 1) * full_ss
            if stripes == 1 or in_row < interior:
                stripe = 0 if stripes == 1 else in_row // full_ss
                in_s = in_row - stripe * full_ss
            else:
                stripe = stripes - 1
                in_s = in_row - interior
            tm = first_m + in_s % GM
            tn = stripe * GN + in_s // GM
        else:
            num_m = TM - first_m
            dm = num_m if num_m > 0 else 1
            full_ss = dm * GN
            interior = (stripes - 1) * full_ss
            if stripes == 1 or in_row < interior:
                stripe = 0 if stripes == 1 else in_row // full_ss
                in_s = in_row - stripe * full_ss
            else:
                stripe = stripes - 1
                in_s = in_row - interior
            tm = first_m + in_s % dm
            tn = stripe * GN + in_s // dm
        if tm >= TM: tm = TM - 1
        if tn >= TN: tn = TN - 1
        return tm * TN + tn

    if td == 19:
        G = cfg["DG_GROUP_SIZE"]
        gt = TN * G
        gi = block_idx // gt
        first_m = gi * G
        in_g = block_idx % gt
        nig = min(G, TM - first_m)
        local_m = in_g % nig
        local_n = in_g // nig
        if local_m & 1:
            local_n = TN - 1 - local_n
        tm = first_m + local_m
        if tm >= TM: tm = TM - 1
        return tm * TN + local_n

    if td == 21:
        if NC != 148 // 2:
            raise ValueError("TD=21 expects SM_COUNT=148")
        c = block_idx % NC
        ti_local = block_idx // NC
        super_ = ti_local // TN
        sub = ti_local % TN
        if TN == 3:
            tn_shift = (c >= 25) + (c >= 50)
        elif TN == 12:
            tn_shift = sum(c >= t for t in (7, 13, 19, 25, 31, 37, 44, 50, 56, 62, 68))
        else:
            raise ValueError(f"TD=21 no threshold table for TILES_N={TN}")
        tn = sub + tn_shift
        if tn >= TN: tn -= TN
        tm = super_ * NC + c
        if tm >= TM: tm = TM - 1
        return tm * TN + tn

    raise ValueError(f"Unknown TD={td}")


def emit_sequence(layer, td, ks=0, ns=0):
    """Returns per-cluster tile sequences as np.array of shape [NC, tile_count, 2]
    with (tm, tn) pairs.  ks shifts per-cluster K-iter start offset (does not
    change tile order).  ns rotates the per-cluster tile-visit order."""
    cfg = LAYERS[layer]
    NC = cfg["NC"]
    TN = cfg["TILES_N"]
    tc = cfg["tile_count"]
    seq = np.empty((NC, tc, 2), dtype=np.int32)
    for c in range(NC):
        for ti in range(tc):
            ti_eff = (ti + c * ns) % tc if ns else ti
            block_idx = ti_eff * NC + c
            if block_idx >= cfg["TOTAL_TILES"]:
                seq[c, ti] = seq[c, ti - 1] if ti > 0 else (0, 0)
                continue
            flat = static_swizzle(td, block_idx, cfg)
            tm, tn = divmod(flat, TN)
            seq[c, ti] = (tm, tn)
    return seq


def features(layer, td, ks=0, ns=0):
    """Compute tile-sequence features for a (layer, td, ks, ns) variant.

    Features:
      Within-cluster locality:
        a_reuse:  fraction of (c, s) where tm[c,s+1] == tm[c,s]  (A stays → A hit)
        b_reuse:  fraction where tn[c,s+1] == tn[c,s]            (B stays → B hit)
        tm_jump:  mean |tm[c,s+1] - tm[c,s]|                     (DRAM amp proxy for A)
      Cross-cluster synchrony (at each step s):
        store_conc:  mean_s (sum_tn (count_clusters_on_tn choose 2)) / (NC choose 2)
        tn_entropy:  mean_s entropy of tn distribution across NC clusters
        tm_entropy:  mean_s entropy of tm distribution across NC clusters
        n_active_tn: mean_s distinct tn values / TILES_N
        n_active_tm: mean_s distinct tm values / NC
      K-phase:
        ks:          raw K_STAGGER
        ks_odd:      1 if K_STAGGER is odd else 0
        k_phase_div: distinct (c*ks) % K_ITERS values / K_ITERS
      N-stagger:
        ns:          raw N_STAGGER
        ns_nontriv:  1 if ns > 0 else 0
    """
    cfg = LAYERS[layer]
    NC, TN, TM, KI = cfg["NC"], cfg["TILES_N"], cfg["TILES_M"], cfg["K_ITERS"]
    seq = emit_sequence(layer, td, ks, ns)
    tm_seq = seq[..., 0]
    tn_seq = seq[..., 1]

    a_reuse = float((tm_seq[:, 1:] == tm_seq[:, :-1]).mean())
    b_reuse = float((tn_seq[:, 1:] == tn_seq[:, :-1]).mean())
    tm_jump = float(np.abs(tm_seq[:, 1:].astype(np.int64) - tm_seq[:, :-1]).mean())
    tn_jump = float(np.abs(tn_seq[:, 1:].astype(np.int64) - tn_seq[:, :-1]).mean())

    # Cross-cluster stats per step.  Vectorize with bincount along axis 0.
    tc = seq.shape[1]
    store_pair = 0.0
    tn_ent = 0.0
    tm_ent = 0.0
    n_tn = 0.0
    n_tm = 0.0
    store_max_share = 0.0
    pair_norm = NC * (NC - 1) / 2
    for s in range(tc):
        tns = tn_seq[:, s]
        tms = tm_seq[:, s]
        tn_counts = np.bincount(tns, minlength=TN).astype(np.float64)
        # store pair contention: C(k,2) summed over tn buckets
        store_pair += (tn_counts * (tn_counts - 1) / 2).sum()
        store_max_share = max(store_max_share, tn_counts.max() / NC)
        p = tn_counts[tn_counts > 0] / NC
        tn_ent += -np.sum(p * np.log2(p))
        n_tn += (tn_counts > 0).sum()

        # tm counts - big bincount
        tm_counts = np.bincount(tms, minlength=TM).astype(np.float64)
        p = tm_counts[tm_counts > 0] / NC
        tm_ent += -np.sum(p * np.log2(p))
        n_tm += (tm_counts > 0).sum()

    store_conc = (store_pair / tc) / pair_norm
    tn_ent /= tc
    tm_ent /= tc
    n_active_tn = (n_tn / tc) / TN
    n_active_tm = (n_tm / tc) / NC

    # K-phase
    k_phases = [(c * ks) % KI for c in range(NC)]
    k_phase_div = len(set(k_phases)) / KI

    return {
        "a_reuse":      a_reuse,
        "b_reuse":      b_reuse,
        "tm_jump":      tm_jump,
        "tn_jump":      tn_jump,
        "store_conc":   store_conc,
        "store_max":    store_max_share,
        "tn_entropy":   tn_ent,
        "tm_entropy":   tm_ent,
        "n_active_tn":  n_active_tn,
        "n_active_tm":  n_active_tm,
        "ks":           float(ks),
        "ks_odd":       float(ks & 1),
        "k_phase_div":  k_phase_div,
        "ns":           float(ns),
        "ns_nontriv":   float(ns > 0),
    }


RESULT_RE = re.compile(r'@@RESULT\s+ms=([0-9.]+)')


def parse_wall_file(path):
    try:
        with open(path) as f:
            for line in f:
                m = RESULT_RE.search(line)
                if m: return float(m.group(1))
    except FileNotFoundError:
        return None
    return None


def load_bench(bench_dir):
    """Parse */_wall_r1.txt files in a bench dir. Returns DataFrame with cols
    layer, dispatch, mode, ms."""
    rows = []
    for path in sorted(glob.glob(os.path.join(bench_dir, "*_wall_r1.txt"))):
        name = os.path.basename(path).replace("_wall_r1.txt", "")
        # name like fc1_zigzag_p_fused, fc2_tail_lean_p_fused, cutlass_fc2_fused
        m = re.match(r'^(fc[12])_(.+)_p_(fused|gemm|strip)$', name)
        if not m:
            continue
        layer, disp, mode = m.group(1), m.group(2), m.group(3)
        ms = parse_wall_file(path)
        if ms is None: continue
        rows.append(dict(layer=layer, dispatch=disp, mode=mode, ms=ms,
                         kstagger=0, nstagger=0))
    return pd.DataFrame(rows)


def load_stagger_csv(path):
    """Parse a stagger_sweep CSV (cols: layer,dispatch,kstagger[,nstagger],mode,ms,...)"""
    df = pd.read_csv(path)
    if "nstagger" not in df.columns:
        df["nstagger"] = 0
    # coerce ms to float; drop BUILD_FAIL/RUN_FAIL rows
    df["ms"] = pd.to_numeric(df["ms"], errors="coerce")
    df = df.dropna(subset=["ms"])
    df["kstagger"] = df["kstagger"].astype(int)
    df["nstagger"] = df["nstagger"].astype(int)
    return df[["layer", "dispatch", "kstagger", "nstagger", "mode", "ms"]]


def build_feature_frame(df):
    """Attach feature columns to each (layer, dispatch, kstagger, nstagger) row
    where the dispatch is simulatable."""
    cache = {}
    rows = []
    for _, r in df.iterrows():
        disp = r["dispatch"]
        if disp not in DISPATCH_TD:
            continue
        td = DISPATCH_TD[disp]
        key = (r["layer"], td, int(r["kstagger"]), int(r["nstagger"]))
        if key not in cache:
            try:
                cache[key] = features(r["layer"], td,
                                      int(r["kstagger"]), int(r["nstagger"]))
            except Exception as e:
                print(f"  feature fail for {key}: {e}", file=sys.stderr)
                cache[key] = None
        feat = cache[key]
        if feat is None: continue
        row = dict(r); row.update(feat); row["td"] = td
        rows.append(row)
    return pd.DataFrame(rows)


FEATURE_COLS = [
    "a_reuse", "b_reuse", "tm_jump", "tn_jump",
    "store_conc", "store_max", "tn_entropy", "tm_entropy",
    "n_active_tn", "n_active_tm",
    "ks", "ks_odd", "k_phase_div", "ns", "ns_nontriv",
]


def fit_report(df, layer, mode, feature_cols=FEATURE_COLS):
    sub = df[(df.layer == layer) & (df["mode"] == mode)].copy()
    if len(sub) < 4:
        print(f"[{layer} {mode}]  n={len(sub)} — too few samples, skip")
        return
    X = sub[feature_cols].values.astype(np.float64)
    y = sub["ms"].values
    keep = [i for i in range(X.shape[1]) if X[:, i].std() > 1e-9]
    cols = [feature_cols[i] for i in keep]
    X = X[:, keep]
    if X.shape[1] == 0:
        print(f"[{layer} {mode}]  all features zero-variance")
        return

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # Feature table sorted by y
    sub_sorted = sub.sort_values("ms").reset_index(drop=True)
    print(f"\n=== {layer} {mode}   n={len(sub)} ===")
    print("  Per-variant features (sorted by ms):")
    tag_w = 20
    hdr = f"  {'variant':<{tag_w}} {'ms':>6}  " + "  ".join(f"{c:>7}" for c in cols)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in sub_sorted.iterrows():
        tag = r["dispatch"]
        if r["kstagger"] or r["nstagger"]:
            tag += f" k{r['kstagger']}n{r['nstagger']}"
        row = f"  {tag:<{tag_w}} {r['ms']:>6.3f}  " + "  ".join(
            f"{r[c]:>7.3f}" for c in cols)
        print(row)

    # Feature correlation with y (univariate) — safer than multi-feature coefs at n=9
    print("\n  Univariate correlations with ms:")
    corrs = []
    for i, c in enumerate(cols):
        if X[:, i].std() < 1e-9:
            continue
        r = np.corrcoef(X[:, i], y)[0, 1]
        corrs.append((c, r))
    corrs.sort(key=lambda t: -abs(t[1]))
    for c, r in corrs:
        bar = "+" if r > 0 else "-"
        print(f"    {c:<14}  r={r:+.3f}  {bar * int(abs(r) * 20)}")

    # Ridge coefficients (reference — unstable at n=9)
    ridge = Ridge(alpha=1.0).fit(Xs, y)
    y_pred_train = ridge.predict(Xs)
    r2_train = 1.0 - np.var(y - y_pred_train) / np.var(y)

    # LOOCV R² (honest)
    if len(sub) >= 5:
        loo = LeaveOneOut()
        y_preds_loo = np.zeros_like(y)
        for tr, te in loo.split(Xs):
            m = Ridge(alpha=1.0).fit(Xs[tr], y[tr])
            y_preds_loo[te] = m.predict(Xs[te])
        r2_loo = 1.0 - np.var(y - y_preds_loo) / np.var(y)
    else:
        r2_loo = float('nan')

    print(f"\n  Ridge fit  R²_train={r2_train:.3f}  R²_loo={r2_loo:.3f}")
    order = np.argsort(-np.abs(ridge.coef_))
    print(f"  {'feature':<14}  {'β (std)':>9}")
    for i in order[:6]:
        print(f"    {cols[i]:<14}  {ridge.coef_[i]:>+9.4f}")

    # Lasso for sparse selection
    try:
        lasso = LassoCV(cv=min(5, len(sub) - 1), max_iter=10000).fit(Xs, y)
        picked = [(cols[i], lasso.coef_[i]) for i in range(len(cols))
                  if abs(lasso.coef_[i]) > 1e-6]
        picked.sort(key=lambda t: -abs(t[1]))
        print(f"\n  Lasso (α={lasso.alpha_:.4f}) kept {len(picked)}/{len(cols)} features:")
        for c, b in picked:
            print(f"    {c:<14}  {b:>+9.4f}")
    except Exception as e:
        print(f"  Lasso failed: {e}")

    # Residuals
    sub2 = sub.assign(pred=y_pred_train, resid=y - y_pred_train).sort_values("resid")
    print(f"\n  Residuals (negative = faster than model predicts):")
    for _, r in sub2.iterrows():
        tag = f"{r['dispatch']}"
        if r["kstagger"] or r["nstagger"]:
            tag += f" k{r['kstagger']}n{r['nstagger']}"
        print(f"    {tag:<22} ms={r['ms']:.3f}  pred={r['pred']:.3f}  resid={r['resid']:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/bench_20260418_034637",
                    help="bench dir with _wall_r1.txt files")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra stagger/kstagger CSV files to merge")
    ap.add_argument("--dump-features", action="store_true")
    ap.add_argument("--layer", choices=["fc1", "fc2", "both"], default="both")
    args = ap.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    frames = []
    if args.bench and os.path.isdir(args.bench):
        df = load_bench(args.bench)
        print(f"loaded {len(df)} rows from {args.bench}")
        frames.append(df)
    for p in args.extra:
        for mp in glob.glob(p):
            df = load_stagger_csv(mp)
            print(f"loaded {len(df)} rows from {mp}")
            frames.append(df)

    if not frames:
        print("no data loaded", file=sys.stderr)
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    # collapse duplicates (same config seen twice across sources) by taking mean
    df = df.groupby(["layer", "dispatch", "kstagger", "nstagger", "mode"],
                    as_index=False)["ms"].mean()

    feat_df = build_feature_frame(df)
    print(f"\n{len(feat_df)} rows with features (out of {len(df)}; dropped "
          f"{len(df) - len(feat_df)} non-simulatable dispatches)")
    print(f"  dispatches kept: {sorted(feat_df['dispatch'].unique())}")

    if args.dump_features:
        feat_df.to_csv("data/tile_features.csv", index=False)
        print("wrote data/tile_features.csv")

    layers = ["fc1", "fc2"] if args.layer == "both" else [args.layer]
    for layer in layers:
        for mode in ["fused", "gemm", "strip"]:
            fit_report(feat_df, layer, mode)


if __name__ == "__main__":
    main()
