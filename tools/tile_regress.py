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


def _drift_cross_features(tn_seq, tm_seq, cfg, windows=(2, 4)):
    """Drift-windowed cross-cluster features.  SMs don't march in lockstep — at
    any wall-clock moment, cluster c is at step s_c and cluster c' is at
    s_c' ≠ s_c.  For window w, we approximate "clusters that are currently
    co-active" as those within |Δs| ≤ w.  Pairs of clusters hitting the same
    tn (B-column) within this window contend on the store path; the cluster
    × (2w+1) set of (tm,tn) currently in flight is the rough L2 working set.
    """
    NC, tc = tn_seq.shape
    TN, TM = cfg["TILES_N"], cfg["TILES_M"]
    out = {}
    for w in windows:
        tot_store = 0.0
        tot_uniq_ab = 0.0
        tot_uniq_tn = 0.0
        tot_uniq_tm = 0.0
        tot_pairs_norm = 0.0
        count = 0
        K = 2 * w + 1
        for s in range(tc):
            lo, hi = max(0, s - w), min(tc, s + w + 1)
            tns_w = tn_seq[:, lo:hi].ravel()
            tms_w = tm_seq[:, lo:hi].ravel()
            slots = tns_w.size
            # Drift-windowed store contention
            tn_counts = np.bincount(tns_w, minlength=TN).astype(np.float64)
            tot_store += (tn_counts * (tn_counts - 1) / 2).sum()
            tot_pairs_norm += slots * (slots - 1) / 2
            # Working-set size in window
            flat = tms_w.astype(np.int64) * TN + tns_w.astype(np.int64)
            tot_uniq_ab += np.unique(flat).size / slots
            tot_uniq_tn += np.unique(tns_w).size / TN
            tot_uniq_tm += np.unique(tms_w).size / min(slots, TM)
            count += 1
        out[f"store_drift_w{w}"] = tot_store / max(1e-9, tot_pairs_norm)
        out[f"unique_ab_w{w}"] = tot_uniq_ab / count
        out[f"unique_tn_w{w}"] = tot_uniq_tn / count
        out[f"unique_tm_w{w}"] = tot_uniq_tm / count
    return out


def _carry_features(tn_seq, tm_seq, windows=(1, 2, 4)):
    """L2 carry-over: at step s, fraction of cluster-tile events whose tile
    appeared at step s-w..s-1 across any cluster.  High → same (tm,tn) still
    hot in L2 when the next cluster needs it.  tn_carry_w captures B-tile
    temporal locality; ab_carry captures full-tile reuse (rare but free)."""
    NC, tc = tn_seq.shape
    out = {}
    # pre-flatten per-step sets once
    tn_sets = [set(tn_seq[:, s].tolist()) for s in range(tc)]
    tm_sets = [set(tm_seq[:, s].tolist()) for s in range(tc)]
    ab_sets = [set(zip(tm_seq[:, s].tolist(), tn_seq[:, s].tolist()))
               for s in range(tc)]
    for w in windows:
        tot_tn = tot_tm = tot_ab = 0.0
        count = 0
        for s in range(w, tc):
            prev_tn, prev_tm, prev_ab = set(), set(), set()
            for d in range(1, w + 1):
                if s - d >= 0:
                    prev_tn |= tn_sets[s - d]
                    prev_tm |= tm_sets[s - d]
                    prev_ab |= ab_sets[s - d]
            cur_tn_arr = tn_seq[:, s]
            cur_tm_arr = tm_seq[:, s]
            hit_tn = sum(1 for v in cur_tn_arr if v in prev_tn)
            hit_tm = sum(1 for v in cur_tm_arr if v in prev_tm)
            hit_ab = sum(1 for p in zip(cur_tm_arr.tolist(), cur_tn_arr.tolist())
                         if p in prev_ab)
            tot_tn += hit_tn / NC
            tot_tm += hit_tm / NC
            tot_ab += hit_ab / NC
            count += 1
        out[f"tn_carry_w{w}"] = tot_tn / max(1, count)
        out[f"tm_carry_w{w}"] = tot_tm / max(1, count)
        out[f"ab_carry_w{w}"] = tot_ab / max(1, count)
    return out


def _load_store_collide(tn_seq, K):
    """P(tn[c, s] == tn[c', s - K])  — models mainloop/epilogue asymmetry:
    tile_s's mainloop loads column tn[c,s] while tile_(s-K)'s epilogue still
    drains column tn[c,s-K] across any cluster.  K is the number of K-loop
    iterations (tile lifetime in mainloop units)."""
    NC, tc = tn_seq.shape
    if tc <= K:
        return 0.0
    hits = 0
    total = 0
    for s in range(K, tc):
        prev_tn = set(tn_seq[:, s - K].tolist())
        cur = tn_seq[:, s]
        hits += sum(1 for v in cur if v in prev_tn)
        total += NC
    return hits / max(1, total)


def _autocorr(seq, lag):
    """Mean across clusters of Pearson autocorrelation at given lag."""
    NC, tc = seq.shape
    if tc <= lag + 1:
        return 0.0
    accs = []
    for c in range(NC):
        x = seq[c].astype(np.float64)
        a = x[:-lag]
        b = x[lag:]
        va = a - a.mean()
        vb = b - b.mean()
        denom = math.sqrt((va * va).sum() * (vb * vb).sum())
        if denom < 1e-12:
            continue
        accs.append(float((va * vb).sum() / denom))
    return float(np.mean(accs)) if accs else 0.0


def _two_hop_features(tn_seq, tm_seq):
    """2-step sequence features.  hilbert vs zorder differ in path smoothness:
    hilbert's 2-hop jumps are bounded, zorder's alternate.  Captures something
    per-step jump alone misses."""
    if tm_seq.shape[1] < 3:
        return dict(tm_jump2=0.0, tn_jump2=0.0, a_reuse_w2=0.0, b_reuse_w2=0.0,
                    path_curve=0.0)
    tm_d2 = np.abs(tm_seq[:, 2:].astype(np.int64) - tm_seq[:, :-2])
    tn_d2 = np.abs(tn_seq[:, 2:].astype(np.int64) - tn_seq[:, :-2])
    a_r_w2 = float(((tm_seq[:, 2:] == tm_seq[:, 1:-1]) |
                    (tm_seq[:, 2:] == tm_seq[:, :-2])).mean())
    b_r_w2 = float(((tn_seq[:, 2:] == tn_seq[:, 1:-1]) |
                    (tn_seq[:, 2:] == tn_seq[:, :-2])).mean())
    # path_curve: |Δ2hop| / (|Δ1| + |Δ2|).  1.0 = collinear, <1.0 = curved.
    d1_tm = tm_seq[:, 1:].astype(np.int64) - tm_seq[:, :-1]
    d1_tn = tn_seq[:, 1:].astype(np.int64) - tn_seq[:, :-1]
    step1 = np.abs(d1_tm[:, :-1]) + np.abs(d1_tn[:, :-1])
    step2 = np.abs(d1_tm[:, 1:]) + np.abs(d1_tn[:, 1:])
    cum = np.abs(d1_tm[:, :-1] + d1_tm[:, 1:]) + np.abs(d1_tn[:, :-1] + d1_tn[:, 1:])
    denom = step1 + step2
    mask = denom > 0
    curve = float((cum[mask] / denom[mask]).mean()) if mask.any() else 1.0
    return dict(tm_jump2=float(tm_d2.mean()), tn_jump2=float(tn_d2.mean()),
                a_reuse_w2=a_r_w2, b_reuse_w2=b_r_w2, path_curve=curve)


def features(layer, td, ks=0, ns=0):
    """Compute tile-sequence features for a (layer, td, ks, ns) variant.

    Categories:
      Within-cluster locality (1-hop and 2-hop):
        a_reuse, b_reuse, tm_jump, tn_jump
        a_reuse_w2, b_reuse_w2, tm_jump2, tn_jump2, path_curve
      Cross-cluster synchronous (step-aligned):
        store_conc, store_max, tn_entropy, tm_entropy, n_active_tn, n_active_tm
      Cross-cluster drift-windowed (w=2,4):
        store_drift_w2, store_drift_w4
        unique_ab_w{2,4}, unique_tn_w{2,4}, unique_tm_w{2,4}
      L2 carry-over (tile reappearance across time):
        tn_carry_w{1,2,4}, tm_carry_w{1,2,4}, ab_carry_w{1,2,4}
      K-phase: ks, ks_odd, k_phase_div, k_phase_max
      N-stagger: ns, ns_nontriv
      Layer-scaling interactions (K_ITERS as stall multiplier):
        a_reuse_KI, b_reuse_KI, store_conc_KI
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
    phase_counts = Counter(k_phases)
    k_phase_max = max(phase_counts.values()) / NC

    feats = {
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
        "k_phase_max":  k_phase_max,
        "ns":           float(ns),
        "ns_nontriv":   float(ns > 0),
    }
    feats.update(_two_hop_features(tn_seq, tm_seq))
    feats.update(_drift_cross_features(tn_seq, tm_seq, cfg, windows=(2, 4)))
    feats.update(_carry_features(tn_seq, tm_seq, windows=(1, 2, 4, 8, 16, 32)))
    # Mainloop/epilogue asymmetry: load_tn now vs store_tn K steps ago
    feats["collide_K"]  = _load_store_collide(tn_seq, KI)
    feats["collide_K2"] = _load_store_collide(tn_seq, KI // 2 or 1)
    # Autocorrelation of (tm, tn) sequences
    for lag in (4, 8, 16):
        feats[f"tm_autocorr_{lag}"] = _autocorr(tm_seq, lag)
        feats[f"tn_autocorr_{lag}"] = _autocorr(tn_seq, lag)
    # Layer-scaling interactions: same feature, different weight per layer via K_ITERS.
    feats["a_reuse_KI"]    = a_reuse * KI
    feats["b_reuse_KI"]    = b_reuse * KI
    feats["store_conc_KI"] = store_conc * KI
    return feats


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
    # within-cluster 1-hop
    "a_reuse", "b_reuse", "tm_jump", "tn_jump",
    # within-cluster 2-hop
    "a_reuse_w2", "b_reuse_w2", "tm_jump2", "tn_jump2", "path_curve",
    # cross-cluster synchronous
    "store_conc", "store_max", "tn_entropy", "tm_entropy",
    "n_active_tn", "n_active_tm",
    # cross-cluster drift-windowed
    "store_drift_w2", "store_drift_w4",
    "unique_ab_w2", "unique_ab_w4",
    "unique_tn_w2", "unique_tn_w4",
    "unique_tm_w2", "unique_tm_w4",
    # L2 carry-over short window
    "tn_carry_w1", "tm_carry_w1", "ab_carry_w1",
    "tn_carry_w2", "tm_carry_w2", "ab_carry_w2",
    "tn_carry_w4", "tm_carry_w4", "ab_carry_w4",
    # L2 carry-over long window (block-lifetime scale)
    "tn_carry_w8", "tm_carry_w8", "ab_carry_w8",
    "tn_carry_w16", "tm_carry_w16", "ab_carry_w16",
    "tn_carry_w32", "tm_carry_w32", "ab_carry_w32",
    # mainloop/epilogue asymmetry (K-lag collision)
    "collide_K", "collide_K2",
    # sequence autocorrelation at multi-scale lags
    "tm_autocorr_4", "tn_autocorr_4",
    "tm_autocorr_8", "tn_autocorr_8",
    "tm_autocorr_16", "tn_autocorr_16",
    # K-phase / stagger
    "ks", "ks_odd", "k_phase_div", "k_phase_max", "ns", "ns_nontriv",
    # layer-scaling interactions
    "a_reuse_KI", "b_reuse_KI", "store_conc_KI",
]


def _ridge_loo_rmse(Xk, y, alpha=1.0):
    """Fast LOOCV rmse for Ridge at a fixed feature subset.  Centers y so that
    the no-intercept Ridge on standardized X behaves like a standard ridge fit
    with intercept (the intercept is the LOO-train mean of y)."""
    n, k = Xk.shape
    y_pred = np.empty(n)
    I = np.eye(k) * alpha
    for i in range(n):
        tr = np.delete(np.arange(n), i)
        Xtr = Xk[tr]
        ytr = y[tr]
        ymean = ytr.mean()
        try:
            beta = np.linalg.solve(Xtr.T @ Xtr + I, Xtr.T @ (ytr - ymean))
        except np.linalg.LinAlgError:
            return float('inf')
        y_pred[i] = Xk[i] @ beta + ymean
    return float(np.sqrt(((y - y_pred) ** 2).mean()))


def _best_subset(Xs, y, cols, k, top_n=5, alpha=1.0):
    """Exhaustive best-subset: pick k features minimizing LOOCV rmse."""
    from itertools import combinations
    p = Xs.shape[1]
    if k > p:
        return []
    results = []
    for idx in combinations(range(p), k):
        rmse = _ridge_loo_rmse(Xs[:, list(idx)], y, alpha=alpha)
        results.append((rmse, idx))
    results.sort(key=lambda t: t[0])
    return [(r, [cols[i] for i in idx]) for r, idx in results[:top_n]]


def _loo_rmse(model_ctor, Xs, y):
    loo = LeaveOneOut()
    y_preds = np.zeros_like(y)
    for tr, te in loo.split(Xs):
        m = model_ctor().fit(Xs[tr], y[tr])
        y_preds[te] = m.predict(Xs[te])
    rmse = float(np.sqrt(((y - y_preds) ** 2).mean()))
    r2 = 1.0 - float(((y - y_preds) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return rmse, r2, y_preds


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
    y_std = float(y.std())

    # Univariate correlations (used both for reporting and top-k filter)
    corrs = []
    for i, c in enumerate(cols):
        r = np.corrcoef(X[:, i], y)[0, 1]
        if not np.isnan(r):
            corrs.append((i, c, r))
    corrs.sort(key=lambda t: -abs(t[2]))

    sub_sorted = sub.sort_values("ms").reset_index(drop=True)
    print(f"\n=== {layer} {mode}   n={len(sub)}   σ(ms)={y_std:.3f} ===")
    print("  Per-variant features (sorted by ms, top-6 correlators):")
    top6 = [c for _, c, _ in corrs[:6]]
    tag_w = 20
    hdr = f"  {'variant':<{tag_w}} {'ms':>6}  " + "  ".join(f"{c:>9}" for c in top6)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in sub_sorted.iterrows():
        tag = r["dispatch"]
        if r["kstagger"] or r["nstagger"]:
            tag += f" k{r['kstagger']}n{r['nstagger']}"
        row = f"  {tag:<{tag_w}} {r['ms']:>6.3f}  " + "  ".join(
            f"{r[c]:>9.3f}" for c in top6)
        print(row)

    print(f"\n  Univariate correlations with ms  (|r|>0.3 shown):")
    for _, c, r in corrs:
        if abs(r) < 0.30:
            continue
        bar = ("+" if r > 0 else "-") * int(abs(r) * 20)
        print(f"    {c:<16}  r={r:+.3f}  {bar}")

    # Ridge (all features) — baseline
    ridge = Ridge(alpha=1.0).fit(Xs, y)
    rmse_ridge_train = float(np.sqrt(((y - ridge.predict(Xs)) ** 2).mean()))
    if len(sub) >= 5:
        rmse_ridge_loo, r2_ridge_loo, _ = _loo_rmse(lambda: Ridge(alpha=1.0), Xs, y)
    else:
        rmse_ridge_loo, r2_ridge_loo = float('nan'), float('nan')

    # Ridge on top-k features (k=3, 5) — reduce overfit at small n
    reports = [(f"ridge_all ({len(cols)}f)", rmse_ridge_train, rmse_ridge_loo, r2_ridge_loo)]
    for k in (3, 5):
        if k >= len(cols):
            continue
        idx = [i for i, _, _ in corrs[:k]]
        Xk = Xs[:, idx]
        m = Ridge(alpha=1.0).fit(Xk, y)
        tr = float(np.sqrt(((y - m.predict(Xk)) ** 2).mean()))
        if len(sub) >= 5:
            lo_rmse, lo_r2, _ = _loo_rmse(lambda: Ridge(alpha=1.0), Xk, y)
        else:
            lo_rmse, lo_r2 = float('nan'), float('nan')
        reports.append((f"ridge_top{k}", tr, lo_rmse, lo_r2))

    # Exhaustive best-subset k=3 — finds combos that beat univariate filtering
    # when signal lives in outlier-rare features (e.g. b_reuse_w2 only fires on hilbert).
    best3 = _best_subset(Xs, y, cols, k=3, top_n=3)
    if best3:
        rmse_bs3 = best3[0][0]
        r2_bs3 = 1.0 - (rmse_bs3 ** 2 * len(y)) / ((y - y.mean()) ** 2).sum()
        reports.append(("best_subset k=3", float('nan'), rmse_bs3, r2_bs3))

    # LassoCV — sparse selection
    try:
        lasso = LassoCV(cv=min(5, len(sub) - 1), max_iter=50000).fit(Xs, y)
        rmse_lasso_train = float(np.sqrt(((y - lasso.predict(Xs)) ** 2).mean()))
        if len(sub) >= 5:
            rmse_lasso_loo, r2_lasso_loo, _ = _loo_rmse(
                lambda: LassoCV(cv=min(5, len(sub) - 1), max_iter=50000), Xs, y)
        else:
            rmse_lasso_loo, r2_lasso_loo = float('nan'), float('nan')
        picked = [(cols[i], lasso.coef_[i]) for i in range(len(cols))
                  if abs(lasso.coef_[i]) > 1e-6]
        picked.sort(key=lambda t: -abs(t[1]))
        reports.append((f"lasso ({len(picked)}f, α={lasso.alpha_:.4f})",
                        rmse_lasso_train, rmse_lasso_loo, r2_lasso_loo))
    except Exception as e:
        picked = []
        print(f"  Lasso failed: {e}")

    # Gradient boosting — catches nonlinear interactions, reg'd hard for small n
    try:
        gbr_kwargs = dict(n_estimators=30, max_depth=2, learning_rate=0.1,
                          min_samples_leaf=2, random_state=0)
        gbr = GradientBoostingRegressor(**gbr_kwargs).fit(Xs, y)
        rmse_gbr_train = float(np.sqrt(((y - gbr.predict(Xs)) ** 2).mean()))
        if len(sub) >= 5:
            rmse_gbr_loo, r2_gbr_loo, _ = _loo_rmse(
                lambda: GradientBoostingRegressor(**gbr_kwargs), Xs, y)
        else:
            rmse_gbr_loo, r2_gbr_loo = float('nan'), float('nan')
        reports.append(("gbr(d=2,n=30)", rmse_gbr_train, rmse_gbr_loo, r2_gbr_loo))
    except Exception as e:
        print(f"  GBR failed: {e}")

    print(f"\n  Model                       rmse_train  rmse_loo  R²_loo")
    print(f"  --------------------------- ----------- --------- -------")
    for name, tr, lo, r2 in reports:
        print(f"  {name:<28} {tr:>10.4f}  {lo:>8.4f}  {r2:>+6.3f}")

    # Lasso selected features
    if picked:
        print(f"\n  Lasso features (kept {len(picked)}/{len(cols)}):")
        for c, b in picked:
            print(f"    {c:<16}  {b:>+9.4f}")

    # Top-3 best-subset triples
    if best3:
        print(f"\n  Top best-subset k=3 triples (exhaustive search over {len(cols)} features):")
        for rmse, feats in best3:
            print(f"    rmse_loo={rmse:.4f}  {{{', '.join(feats)}}}")

    # Residuals under best-LOO model
    best = min(reports, key=lambda r: r[2] if not np.isnan(r[2]) else float('inf'))
    print(f"\n  Best LOO model: {best[0]}  rmse={best[2]:.4f} ms")
    y_pred_train = ridge.predict(Xs)
    sub2 = sub.assign(pred=y_pred_train, resid=y - y_pred_train).sort_values("resid")
    print(f"  Ridge-all residuals (negative = faster than model predicts):")
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
