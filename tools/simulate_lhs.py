#!/usr/bin/env python3
"""Bootstrap convergence simulation: random vs tiered search strategies.

Loads sweep CSV data (ground truth from exhaustive tiered search) and simulates
how many configs each strategy needs to find the optimum. No GPU needed.

Usage:
    python3 tools/simulate_lhs.py data/session_*/sweep_fc1_gelu.csv
    python3 tools/simulate_lhs.py data/sweep_results_local.csv --trials 50000
"""

import argparse
import csv
import random
import sys


def load_csv(path):
    """Load ms values from sweep CSV (OK configs only)."""
    ms_list = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('status') == 'OK':
                try:
                    ms_list.append(float(row['ms']))
                except (ValueError, KeyError):
                    continue
    return ms_list


def bootstrap_random(ms_list, n, trials=10000, seed=42):
    """Bootstrap: draw n random configs, track min(ms) across trials.

    Returns (mean_best, p5, p95).
    """
    rng = random.Random(seed)
    bests = []
    for _ in range(trials):
        sample = rng.choices(ms_list, k=n)
        bests.append(min(sample))
    bests.sort()
    mean_best = sum(bests) / len(bests)
    p5 = bests[int(0.05 * len(bests))]
    p95 = bests[int(0.95 * len(bests))]
    return mean_best, p5, p95


def replay_sequential(ms_list, seed=42):
    """Simulate sequential search in shuffled order (CSV is sorted by ms).

    Shuffles to remove sort bias, then tracks best-so-far at each step.
    Returns list of (n_configs, best_ms_so_far) at each step.
    """
    rng = random.Random(seed)
    shuffled = list(ms_list)
    rng.shuffle(shuffled)
    progression = []
    best_so_far = float('inf')
    for i, ms in enumerate(shuffled, 1):
        if ms < best_so_far:
            best_so_far = ms
        progression.append((i, best_so_far))
    return progression


def configs_to_threshold(ms_list, threshold_ms, trials=10000, seed=42):
    """How many random configs needed to reach within threshold_ms of optimum?

    Returns (mean_n, p5_n, p95_n).
    """
    rng = random.Random(seed)
    target = min(ms_list) + threshold_ms
    counts = []
    for _ in range(trials):
        order = list(range(len(ms_list)))
        rng.shuffle(order)
        best = float('inf')
        for i, idx in enumerate(order, 1):
            best = min(best, ms_list[idx])
            if best <= target:
                counts.append(i)
                break
        else:
            counts.append(len(ms_list))
    counts.sort()
    mean_n = sum(counts) / len(counts)
    p5 = counts[int(0.05 * len(counts))]
    p95 = counts[int(0.95 * len(counts))]
    return mean_n, p5, p95


def main():
    parser = argparse.ArgumentParser(
        description='Bootstrap convergence simulation for search strategies')
    parser.add_argument('csv_path', help='Sweep CSV with ground truth results')
    parser.add_argument('--trials', type=int, default=10000,
                        help='Bootstrap trials (default: 10000)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    ms_list = load_csv(args.csv_path)
    if not ms_list:
        print(f'No OK configs found in {args.csv_path}', file=sys.stderr)
        sys.exit(1)

    optimum = min(ms_list)
    total = len(ms_list)
    print(f'Loaded {total} OK configs from {args.csv_path}')
    print(f'Optimum: {optimum:.3f} ms')
    print()

    # Convergence table
    sample_sizes = [10, 20, 30, 40, 60, 80, 100, 200, 500]
    sample_sizes = [n for n in sample_sizes if n <= total]
    if total not in sample_sizes:
        sample_sizes.append(total)

    print(f'{"N":>6}  {"Mean best":>10}  {"P5":>10}  {"P95":>10}  {"Hit opt":>8}')
    print('-' * 52)
    for n in sample_sizes:
        mean_best, p5, p95 = bootstrap_random(ms_list, n, trials=args.trials, seed=args.seed)
        hit_pct = sum(1 for _ in range(args.trials)
                      if min(random.Random(args.seed + _).choices(ms_list, k=n)) <= optimum + 0.001
                      ) / args.trials * 100
        print(f'{n:>6}  {mean_best:>10.3f}  {p5:>10.3f}  {p95:>10.3f}  {hit_pct:>7.1f}%')
    print()

    # Sequential replay (shuffled — CSV is sorted by ms, not execution order)
    progression = replay_sequential(ms_list, seed=args.seed)
    print('Sequential replay (shuffled order, best-so-far):')
    shown = set()
    for n, best in progression:
        if n in [1, 5, 10, 50, 100, 200, 500, total] or best <= optimum:
            if n not in shown:
                print(f'  After {n:>5} configs: {best:.3f} ms')
                shown.add(n)
                if best <= optimum:
                    break
    print()

    # Configs to reach within 0.5% of optimum
    threshold_abs = optimum * 0.005
    mean_n, p5_n, p95_n = configs_to_threshold(
        ms_list, threshold_abs, trials=args.trials, seed=args.seed)
    print(f'Configs to reach within 0.5% of optimum ({optimum + threshold_abs:.3f} ms):')
    print(f'  Random: mean={mean_n:.0f}, P5={p5_n}, P95={p95_n}')

    # Find where sequential first hits the threshold
    target = optimum + threshold_abs
    for n, best in progression:
        if best <= target:
            print(f'  Sequential (shuffled): {n} configs')
            break
    print()

    # Summary comparison
    mean60, p5_60, _ = bootstrap_random(ms_list, 60, trials=args.trials, seed=args.seed)
    for n_seq, best_seq in progression:
        if best_seq <= mean60:
            print(f'Random N=60 (mean {mean60:.3f} ms) matches sequential at N={n_seq}')
            break
    print(f'Random N=60 P5={p5_60:.3f} ms vs exhaustive={optimum:.3f} ms')


if __name__ == '__main__':
    main()
