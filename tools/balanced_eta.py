"""
Compute balanced-subset eta-squared from tiered sweep CSV data.
Filters to rows where only the target param varies (all others constant).

Usage:
    python3 balanced_eta.py data/session_20260315_020611/sweep_fc1_gelu.csv
"""
import csv
import sys
from collections import defaultdict

PARAMS = [
    'N_STAGES', 'NUM_EPI_WARPS', 'TMEM_LOAD_WIDTH', 'INTERLEAVE_STRATEGY',
    'MBAR_EARLY', 'STAGGER_CYCLES', 'PHASE1_UNROLL', 'SNAKE_ORDER',
    'CVT_ADD_FUSED', 'K_LOOP_UNROLL', 'W0_LOOP_UNROLL', 'SUB_MMA_UNROLL',
    'PRELOAD_MODE', 'PREFETCH_BEFORE_STORE', 'GELU_VARIANT', 'TMA_RESIDUAL',
    'BATCH_EPILOGUE', 'GELU_VECTOR_WIDTH', 'STORE_TIMING',
    'EPILOGUE_LOOP', 'STS_WIDTH', 'EPI_SYNC', 'NUM_PASSES_PARAM',
]


def balanced_eta_sq(rows, param):
    """
    Group rows by all-other-params signature. Keep groups where the target
    param has >=2 distinct values. Compute one-way ANOVA eta-squared on
    this balanced subset only.
    """
    all_params = [p for p in PARAMS if p in rows[0] and
                  len(set(r.get(p, '') for r in rows)) >= 2]
    other_params = [p for p in all_params if p != param]

    # Group by other-param signature
    groups = defaultdict(list)
    for r in rows:
        sig = tuple(r.get(p, '') for p in other_params)
        groups[sig].append(r)

    # Keep groups with >=2 distinct values of target param
    balanced = []
    for sig, members in groups.items():
        vals = set(m.get(param, '') for m in members)
        if len(vals) >= 2:
            balanced.extend(members)

    if len(balanced) < 3:
        return None, 0, {}

    ms_vals = [float(r['ms']) for r in balanced]
    grand_mean = sum(ms_vals) / len(ms_vals)
    ss_total = sum((m - grand_mean) ** 2 for m in ms_vals)
    if ss_total == 0:
        return 0.0, len(balanced), {}

    levels = defaultdict(list)
    for r in balanced:
        levels[r.get(param, '')].append(float(r['ms']))
    if len(levels) < 2:
        return None, 0, {}

    ss_between = sum(
        len(v) * (sum(v) / len(v) - grand_mean) ** 2
        for v in levels.values()
    )
    eta = ss_between / ss_total
    level_means = {k: sum(v) / len(v) for k, v in levels.items()}
    return eta, len(balanced), level_means


def main():
    path = sys.argv[1]
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r.get('status') == 'OK']

    print(f'Loaded {len(rows)} OK configs from {path}')
    print(f'Best: {min(float(r["ms"]) for r in rows):.3f} ms\n')

    # Find active params (>=2 distinct values in data)
    active = []
    for p in PARAMS:
        vals = set(r.get(p, '') for r in rows)
        vals.discard('')
        if len(vals) >= 2:
            active.append(p)

    print(f'{"Param":30s} {"Bal-η²":>8} {"N_bal":>6}  Levels (value:mean_ms(count))')
    print('-' * 90)

    results = []
    for p in active:
        eta, n_bal, level_means = balanced_eta_sq(rows, p)
        if eta is None:
            continue

        # Rebuild level counts for display
        other_params = [pp for pp in active if pp != p]
        groups = defaultdict(list)
        for r in rows:
            sig = tuple(r.get(pp, '') for pp in other_params)
            groups[sig].append(r)
        balanced = []
        for sig, members in groups.items():
            vals = set(m.get(p, '') for m in members)
            if len(vals) >= 2:
                balanced.extend(members)
        level_counts = defaultdict(int)
        for r in balanced:
            level_counts[r.get(p, '')] += 1

        level_str = '  '.join(
            f'{v}:{level_means[v]:.3f}({level_counts[v]})'
            for v in sorted(level_means, key=lambda x: level_means[x])
        )
        results.append((eta, n_bal, p, level_str))

    results.sort(key=lambda x: x[0], reverse=True)
    for eta, n_bal, p, level_str in results:
        print(f'{p:30s} {eta:8.4f} {n_bal:6d}  {level_str}')


if __name__ == '__main__':
    main()
