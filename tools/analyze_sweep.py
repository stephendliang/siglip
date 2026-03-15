#!/usr/bin/env python3
"""Analyze grid search sweep data: parameter importance.

Three methods, each addressing different weaknesses:
  1. eta-squared (one-way ANOVA) — simple, but confounded by tiered search imbalance
  2. Balanced-subset eta-squared — filters to subsets where only one param varies
  3. Random forest permutation importance — diagnostic only (confounded on tiered data)

Usage:
    python3 tools/analyze_sweep.py data/sweep_results_run4.csv
    python3 tools/analyze_sweep.py data/session_*/sweep_patch_embed.csv
    python3 tools/analyze_sweep.py data/session_*/sweep_*.csv   # all layers
"""

import csv
import sys
import os
from collections import defaultdict

try:
    import numpy as np
    from scipy import stats
except ImportError:
    print("Error: numpy and scipy required.", file=sys.stderr)
    sys.exit(1)

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def load_sweep(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r.get('status') == 'OK']
    return rows


def detect_params(rows):
    """Find columns that are tuning parameters (not output metrics)."""
    skip = {'regs', 'spills', 'smem_kb', 'ms', 'tflops', 'status', 'dflags'}
    params = []
    for col in rows[0].keys():
        if col in skip:
            continue
        vals = set(r[col] for r in rows)
        if len(vals) >= 2:
            params.append(col)
    return params


def compute_eta_sq(rows, param, ms_all=None):
    """Compute eta-squared for one parameter over given rows."""
    if ms_all is None:
        ms_all = np.array([float(r['ms']) for r in rows])
    ss_total = np.sum((ms_all - ms_all.mean()) ** 2)
    if ss_total == 0:
        return 0.0, [], 0.0, 1.0

    levels = sorted(set(r[param] for r in rows), key=lambda x: float(x))
    if len(levels) < 2:
        return 0.0, [], 0.0, 1.0

    groups = []
    level_stats = []
    for lv in levels:
        vals = np.array([float(r['ms']) for r in rows if r[param] == lv])
        groups.append(vals)
        level_stats.append((lv, len(vals), vals.mean(),
                            vals.std(ddof=1) if len(vals) > 1 else 0.0))

    grand_mean = ms_all.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    eta_sq = ss_between / ss_total

    if all(len(g) >= 2 for g in groups):
        f_stat, p_val = stats.f_oneway(*groups)
    else:
        f_stat, p_val = 0.0, 1.0

    return eta_sq, level_stats, f_stat, p_val


def balanced_eta_sq(rows, param, all_params):
    """Compute eta-squared on subsets where only `param` varies.

    Groups rows by all OTHER parameters, keeping only groups where
    `param` takes multiple values. This controls for confounds from
    the tiered search pinning.
    """
    other_params = [p for p in all_params if p != param]

    # Group rows by their other-param signature
    buckets = defaultdict(list)
    for r in rows:
        key = tuple(r[p] for p in other_params)
        buckets[key].append(r)

    # Keep only buckets where param varies
    controlled_rows = []
    for key, bucket in buckets.items():
        vals = set(r[param] for r in bucket)
        if len(vals) >= 2:
            controlled_rows.extend(bucket)

    if len(controlled_rows) < 4:
        return None, [], 0, 0.0, 1.0

    ms_ctrl = np.array([float(r['ms']) for r in controlled_rows])
    eta_sq, level_stats, f_stat, p_val = compute_eta_sq(
        controlled_rows, param, ms_ctrl)
    return eta_sq, level_stats, len(controlled_rows), f_stat, p_val


def rf_importance(rows, params):
    """Random forest permutation importance for all parameters."""
    # Encode params as numeric features
    X = np.array([[float(r[p]) for p in params] for r in rows])
    y = np.array([float(r['ms']) for r in rows])

    rf = RandomForestRegressor(n_estimators=200, max_depth=8,
                               random_state=42, n_jobs=-1)
    rf.fit(X, y)
    r2 = rf.score(X, y)

    result = permutation_importance(rf, X, y, n_repeats=30,
                                    random_state=42, n_jobs=-1)
    return {p: (result.importances_mean[i], result.importances_std[i])
            for i, p in enumerate(params)}, r2


def sig_str(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'ns'


def analyze_file(path):
    rows = load_sweep(path)
    if not rows:
        print(f"  No valid rows in {path}")
        return

    params = detect_params(rows)
    ms_all = np.array([float(r['ms']) for r in rows])

    label = os.path.basename(path).replace('.csv', '')
    print(f"\n{'=' * 78}")
    print(f"  {label}  ({len(rows)} configs, ms range {ms_all.min():.3f}-{ms_all.max():.3f})")
    print(f"{'=' * 78}")

    ss_total = np.sum((ms_all - ms_all.mean()) ** 2)
    if ss_total == 0:
        print("  All configs have identical ms — nothing to analyze.")
        return

    # --- Method 1: raw eta-squared ---
    raw_results = {}
    for param in params:
        eta_sq, level_stats, f_stat, p_val = compute_eta_sq(rows, param, ms_all)
        if level_stats:
            raw_results[param] = {
                'eta_sq': eta_sq, 'levels': level_stats,
                'f_stat': f_stat, 'p_val': p_val,
            }

    # --- Method 2: balanced-subset eta-squared ---
    bal_results = {}
    for param in params:
        eta_sq, level_stats, n_ctrl, f_stat, p_val = balanced_eta_sq(
            rows, param, params)
        if eta_sq is not None:
            bal_results[param] = {
                'eta_sq': eta_sq, 'levels': level_stats,
                'n': n_ctrl, 'f_stat': f_stat, 'p_val': p_val,
            }

    # --- Method 3: random forest ---
    rf_results = None
    rf_r2 = None
    if HAS_SKLEARN:
        rf_results, rf_r2 = rf_importance(rows, params)

    # --- Combined table ---
    print(f"\n{'Parameter Importance — Three Methods':^78}")
    print("-" * 78)
    hdr_rf = '  RF perm' if rf_results else ''
    print(f"  {'Parameter':22s}  {'raw eta^2':>9s}  {'bal eta^2':>9s} {'(n)':>5s}"
          f"{hdr_rf}  {'best level':>10s}")
    print("  " + "-" * 74)

    # Sort by balanced eta-sq where available, fall back to raw
    def sort_key(p):
        bal = bal_results.get(p, {}).get('eta_sq', 0)
        raw = raw_results.get(p, {}).get('eta_sq', 0)
        n_bal = bal_results.get(p, {}).get('n', 0)
        # Use balanced η² when we have enough balanced pairs, raw as fallback
        if n_bal >= 10:
            return bal
        return raw

    for param in sorted(params, key=sort_key, reverse=True):
        raw = raw_results.get(param, {})
        bal = bal_results.get(param, {})
        raw_eta = f"{raw['eta_sq']:.4f} {sig_str(raw['p_val'])}" if raw else '     n/a'
        bal_eta = f"{bal['eta_sq']:.4f} {sig_str(bal['p_val'])}" if bal else '     n/a'
        bal_n = f"({bal['n']:3d})" if bal else '     '

        rf_col = ''
        if rf_results and param in rf_results:
            imp, std = rf_results[param]
            rf_col = f"  {imp:.4f}±{std:.4f}"

        # Best level from balanced if available, else raw
        best_lvl = ''
        lvls = bal.get('levels') or raw.get('levels', [])
        if lvls:
            best = min(lvls, key=lambda x: x[2])
            best_lvl = f"  {best[0]:>6s}"

        print(f"  {param:22s}  {raw_eta:>9s}  {bal_eta:>9s} {bal_n}"
              f"{rf_col}{best_lvl}")

    if rf_r2 is not None:
        print(f"\n  Random forest R² = {rf_r2:.4f} (fit quality)")

    # --- Check for imbalance ---
    imbalanced = []
    for param in params:
        levels = sorted(set(r[param] for r in rows), key=lambda x: float(x))
        counts = [sum(1 for r in rows if r[param] == lv) for lv in levels]
        ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
        if ratio > 5:
            imbalanced.append((param, levels, counts, ratio))

    if imbalanced:
        print(f"\n  Imbalanced parameters (max/min ratio > 5x):")
        for param, levels, counts, ratio in imbalanced:
            dist = ', '.join(f'{lv}:{c}' for lv, c in zip(levels, counts))
            print(f"    {param:22s}  {ratio:5.0f}x  [{dist}]")

    # --- Per-level details for top params ---
    print(f"\n{'Per-level means (top params by balanced eta^2)':^78}")
    print("-" * 78)

    ranked = sorted(params, key=sort_key, reverse=True)
    for param in ranked:
        bal = bal_results.get(param, {})
        raw = raw_results.get(param, {})
        eta = bal.get('eta_sq', raw.get('eta_sq', 0))
        if eta < 0.005:
            continue
        source = 'balanced' if bal.get('levels') else 'raw'
        lvls = bal.get('levels') or raw.get('levels', [])
        n_note = f", n={bal['n']}" if 'n' in bal else ''
        print(f"\n  {param} (eta^2={eta:.4f}, {source}{n_note}):")
        best_mean = min(lv[2] for lv in lvls)
        for lv, n, mean, std in lvls:
            delta_us = (mean - best_mean) * 1000
            marker = ' <-- best' if mean == best_mean else ''
            print(f"    {lv:>6s}  n={n:3d}  mean={mean:.4f} ms  std={std:.4f}  "
                  f"+{delta_us:6.1f} us{marker}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <sweep_csv> [sweep_csv ...]")
        sys.exit(1)

    if not HAS_SKLEARN:
        print("Note: sklearn not found, random forest analysis disabled.",
              file=sys.stderr)

    for path in sys.argv[1:]:
        if not os.path.exists(path):
            print(f"Not found: {path}")
            continue
        analyze_file(path)

    print()


if __name__ == '__main__':
    main()
