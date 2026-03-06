#!/usr/bin/env python3
"""Unified benchmark comparison: cuBLAS vs CUTLASS vs our kernel.

Builds all targets, runs each benchmark N times per layer, collects fused
timing, sorts results, and performs one-way ANOVA + pairwise Welch's t-tests.

Requires B200 hardware for execution.

Usage:
    python3 tools/compare_all.py                          # all layers, 10 runs
    python3 tools/compare_all.py --runs 20                # more runs for tighter CI
    python3 tools/compare_all.py --layer patch_embed      # single layer
    python3 tools/compare_all.py --layer fc1 fc2          # subset of layers
    python3 tools/compare_all.py --skip-build             # reuse existing binaries
    python3 tools/compare_all.py --grid-search            # run grid search first
    python3 tools/compare_all.py --csv data/compare.csv   # save raw samples
    python3 tools/compare_all.py --cutlass-mode max       # extended CUTLASS sweep
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

try:
    import numpy as np
    from scipy import stats
except ImportError:
    print("Error: numpy and scipy required. Install with: pip install numpy scipy",
          file=sys.stderr)
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# ── Layer definitions ──

LAYERS = {
    'patch_embed': {
        'our_target': 'siglip_vision',
        'our_binary': './siglip_vision',
        'our_source': 'patch_embed.cu',
        'cublas_target': 'cublas-bench',
        'cublas_binary': './cublas-bench',
        'cutlass_target': 'cutlass-bench',
        'cutlass_target_max': 'cutlass-bench-max',
        'cutlass_binary': './cutlass-bench',
        'cutlass_binary_max': './cutlass-bench-max',
        'N': 768, 'K': 768,
        'epilogue': 'PERIODIC_ADD',
        'label': 'Patch Embed (768x768)',
    },
    'fc1': {
        'our_target': 'fc1-gelu',
        'our_binary': './fc1-gelu',
        'our_source': 'fc1_gelu.cu',
        'cublas_target': 'cublas-bench-fc1',
        'cublas_binary': './cublas-bench-fc1',
        'cutlass_target': 'cutlass-bench-fc1',
        'cutlass_target_max': 'cutlass-bench-fc1-max',
        'cutlass_binary': './cutlass-bench-fc1',
        'cutlass_binary_max': './cutlass-bench-fc1-max',
        'N': 3072, 'K': 768,
        'epilogue': 'GELU_BIAS',
        'label': 'FC1+GELU (768x3072)',
    },
    'fc2': {
        'our_target': 'fc2',
        'our_binary': './fc2',
        'our_source': 'fc2.cu',
        'cublas_target': 'cublas-bench-fc2',
        'cublas_binary': './cublas-bench-fc2',
        'cutlass_target': 'cutlass-bench-fc2',
        'cutlass_target_max': 'cutlass-bench-fc2-max',
        'cutlass_binary': './cutlass-bench-fc2',
        'cutlass_binary_max': './cutlass-bench-fc2-max',
        'N': 768, 'K': 3072,
        'epilogue': 'BIAS_ONLY',
        'label': 'FC2+Bias (3072x768)',
    },
}


# ── Output parsers ──

def parse_our_kernel(output):
    """Parse @@RESULT line from our kernel. Returns dict with ms, tflops, valid."""
    for line in output.splitlines():
        m = re.match(r'@@RESULT ms=([\d.]+) tflops=([\d.]+) checksum=[\d.]+ valid=([01])', line)
        if m:
            return {
                'ms': float(m.group(1)),
                'tflops': float(m.group(2)),
                'valid': int(m.group(3)),
            }
    return None


def parse_cublas(output):
    """Parse cuBLAS bench output. Returns dict of {metric: (ms, tflops, algo)}."""
    results = {}
    # Parse lines like:  Label                     0.365 ms  3001.0 TFLOPS  (algo #5)
    for line in output.splitlines():
        m = re.match(r'\s+(.+?)\s{2,}([\d.]+) ms\s+([\d.]+) TFLOPS\s+\(algo #(\d+)\)', line)
        if m:
            label = m.group(1).strip()
            results[label] = {
                'ms': float(m.group(2)),
                'tflops': float(m.group(3)),
                'algo': int(m.group(4)),
            }
    # Parse scaling mode context
    current_mode = None
    tagged = {}
    for line in output.splitlines():
        if 'MXFP8:' in line:
            current_mode = 'mxfp8'
        elif 'Per-tensor FP8:' in line:
            current_mode = 'per_tensor'
        m = re.match(r'\s+(.+?)\s{2,}([\d.]+) ms\s+([\d.]+) TFLOPS\s+\(algo #(\d+)\)', line)
        if m and current_mode:
            label = m.group(1).strip()
            key = f"{current_mode}_{label}"
            tagged[key] = {
                'ms': float(m.group(2)),
                'tflops': float(m.group(3)),
                'algo': int(m.group(4)),
            }
    return tagged


def parse_cublas_best_fused(output, epilogue):
    """Extract the best fused time from cuBLAS output for fair comparison."""
    tagged = parse_cublas(output)

    # Priority: per-tensor fused > mxfp8 fused > per-tensor gemm+unfused
    fused_labels = {
        'PERIODIC_ADD': 'GEMM + fused add (beta=1)',
        'GELU_BIAS': 'GEMM + fused bias+GELU',
        'BIAS_ONLY': 'GEMM + fused bias',
    }
    unfused_labels = {
        'PERIODIC_ADD': 'GEMM + unfused periodic add',
        'GELU_BIAS': 'GEMM + unfused bias+GELU',
        'BIAS_ONLY': 'GEMM + unfused bias',
    }

    fused_label = fused_labels[epilogue]
    unfused_label = unfused_labels[epilogue]

    # Try per-tensor fused first, then mxfp8 fused, then unfused
    for prefix in ['per_tensor', 'mxfp8']:
        key = f"{prefix}_{fused_label}"
        if key in tagged:
            return tagged[key]['ms'], f"cuBLAS {prefix} fused"

    for prefix in ['per_tensor', 'mxfp8']:
        key = f"{prefix}_{unfused_label}"
        if key in tagged:
            return tagged[key]['ms'], f"cuBLAS {prefix} unfused"

    return None, None


def parse_cublas_gemm_only(output):
    """Extract best GEMM-only time from cuBLAS (reference)."""
    tagged = parse_cublas(output)
    for prefix in ['per_tensor', 'mxfp8']:
        key = f"{prefix}_GEMM only"
        if key in tagged:
            return tagged[key]['ms']
    return None


def parse_cutlass_best_fused(output, epilogue):
    """Extract best fused time from CUTLASS bench output."""
    # For PERIODIC_ADD, look for "Best Fused EVT" or "Best GEMM+PostAdd"
    # For GELU_BIAS / BIAS_ONLY, look for "Best Fused"
    best_ms = None

    if epilogue == 'PERIODIC_ADD':
        # Prefer EVT fused, fallback to GEMM+PostAdd
        for pattern in [r'Best Fused EVT:\s+\S+\s+\S+\s+\S+\s+([\d.]+) ms',
                        r'Best Fused BF16:\s+\S+\s+\S+\s+\S+\s+([\d.]+) ms',
                        r'Best GEMM\+PostAdd:\s*\S+\s+\S+\s+\S+\s+([\d.]+) ms']:
            m = re.search(pattern, output)
            if m:
                ms = float(m.group(1))
                if best_ms is None or ms < best_ms:
                    best_ms = ms
    else:
        m = re.search(r'Best Fused:\s+\S+\s+\S+\s+\S+\s+([\d.]+) ms', output)
        if m:
            best_ms = float(m.group(1))
        # Also check GEMM+unfused as fallback
        if best_ms is None:
            m = re.search(r'Best GEMM\+unfused:\s+\S+\s+\S+\s+\S+\s+([\d.]+) ms', output)
            if m:
                best_ms = float(m.group(1))

    return best_ms


def parse_cutlass_gemm_only(output):
    """Extract best GEMM-only time from CUTLASS."""
    m = re.search(r'Best GEMM:\s+\S+\s+\S+\s+\S+\s+([\d.]+) ms', output)
    return float(m.group(1)) if m else None


# ── Build ──

def build_targets(targets, verbose=False):
    """Build make targets. Returns True on success."""
    for target in targets:
        print(f"  Building {target} ...", end=' ', flush=True)
        result = subprocess.run(
            ['make', target], capture_output=True, text=True, cwd=ROOT_DIR)
        if result.returncode != 0:
            print("FAILED")
            if verbose:
                print(result.stderr[-500:])
            return False
        print("ok")
    return True


# ── Run helpers ──

def run_binary(binary, timeout=60):
    """Run a binary, return (stdout, success)."""
    try:
        result = subprocess.run(
            [binary], capture_output=True, text=True, timeout=timeout, cwd=ROOT_DIR)
        return result.stdout + result.stderr, result.returncode == 0
    except subprocess.TimeoutExpired:
        return "", False


def collect_our_samples(binary, n_runs, label="our kernel"):
    """Run our kernel n_runs times, return list of ms values."""
    samples = []
    for i in range(n_runs):
        output, ok = run_binary(binary, timeout=30)
        if not ok:
            print(f"    {label} run {i+1}/{n_runs}: FAILED")
            continue
        parsed = parse_our_kernel(output)
        if parsed is None:
            print(f"    {label} run {i+1}/{n_runs}: no @@RESULT")
            continue
        if not parsed['valid']:
            print(f"    {label} run {i+1}/{n_runs}: INVALID")
            continue
        samples.append(parsed['ms'])
        print(f"    {label} run {i+1}/{n_runs}: {parsed['ms']:.3f} ms / "
              f"{parsed['tflops']:.0f} TFLOPS")
    return samples


def collect_cublas_samples(binary, epilogue, n_runs):
    """Run cuBLAS bench n_runs times, return list of best fused ms values."""
    samples = []
    gemm_samples = []
    for i in range(n_runs):
        output, ok = run_binary(binary, timeout=120)
        if not ok:
            print(f"    cuBLAS run {i+1}/{n_runs}: FAILED")
            continue
        ms, desc = parse_cublas_best_fused(output, epilogue)
        gemm_ms = parse_cublas_gemm_only(output)
        if ms is None:
            print(f"    cuBLAS run {i+1}/{n_runs}: could not parse fused time")
            continue
        samples.append(ms)
        if gemm_ms is not None:
            gemm_samples.append(gemm_ms)
        suffix = f" (GEMM-only: {gemm_ms:.3f} ms)" if gemm_ms else ""
        print(f"    cuBLAS run {i+1}/{n_runs}: {ms:.3f} ms [{desc}]{suffix}")
    return samples, gemm_samples


def collect_cutlass_samples(binary, epilogue, n_runs):
    """Run CUTLASS bench n_runs times, return list of best fused ms values."""
    samples = []
    gemm_samples = []
    for i in range(n_runs):
        output, ok = run_binary(binary, timeout=300)
        if not ok:
            print(f"    CUTLASS run {i+1}/{n_runs}: FAILED")
            continue
        ms = parse_cutlass_best_fused(output, epilogue)
        gemm_ms = parse_cutlass_gemm_only(output)
        if ms is None:
            print(f"    CUTLASS run {i+1}/{n_runs}: could not parse fused time")
            continue
        samples.append(ms)
        if gemm_ms is not None:
            gemm_samples.append(gemm_ms)
        suffix = f" (GEMM-only: {gemm_ms:.3f} ms)" if gemm_ms else ""
        print(f"    CUTLASS run {i+1}/{n_runs}: {ms:.3f} ms{suffix}")
    return samples, gemm_samples


# ── Statistics ──

def cohens_d(a, b):
    """Cohen's d effect size."""
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 +
                          (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def sig_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def print_summary(name, samples):
    """Print summary statistics for a sample array."""
    a = np.array(samples)
    iqr = np.percentile(a, 75) - np.percentile(a, 25)
    print(f"  {name:20s}  n={len(a):2d}  mean={a.mean():.4f}  std={a.std(ddof=1):.4f}  "
          f"min={a.min():.4f}  max={a.max():.4f}  median={np.median(a):.4f}  IQR={iqr:.4f}")


def run_anova(groups, layer_label):
    """Run ANOVA + pairwise tests on groups dict {name: samples}."""
    names = list(groups.keys())
    arrays = [np.array(groups[n]) for n in names]

    # Need at least 2 groups with 2+ samples each
    valid = [(n, a) for n, a in zip(names, arrays) if len(a) >= 2]
    if len(valid) < 2:
        print(f"  Not enough data for ANOVA (need >= 2 groups with >= 2 samples)")
        return

    valid_names = [v[0] for v in valid]
    valid_arrays = [v[1] for v in valid]

    print(f"\n{'Summary Statistics':^72}")
    print("=" * 72)
    for name, arr in zip(valid_names, valid_arrays):
        print_summary(name, arr)

    # Sort by mean (ascending = fastest first)
    ranked = sorted(zip(valid_names, valid_arrays), key=lambda x: x[1].mean())
    print(f"\n  Ranking (fastest first):")
    for i, (name, arr) in enumerate(ranked, 1):
        flops = 2.0 * 928256 * 768  # approximate — layer-dependent
        # Just report ms ranking
        print(f"    #{i}: {name:20s}  {arr.mean():.4f} ms "
              f"({'+' if i > 1 else ''}{(arr.mean() - ranked[0][1].mean()) * 1000:.1f} us vs best)"
              if i > 1 else
              f"    #{i}: {name:20s}  {arr.mean():.4f} ms (best)")

    # One-way ANOVA
    print(f"\n{'One-way ANOVA':^72}")
    print("=" * 72)
    if len(valid_arrays) >= 2:
        f_stat, p_anova = stats.f_oneway(*valid_arrays)
        print(f"  F({len(valid_arrays)-1}, {sum(len(a) for a in valid_arrays) - len(valid_arrays)}) "
              f"= {f_stat:.3f},  p = {p_anova:.2e}  "
              f"{'(significant)' if p_anova < 0.05 else '(not significant)'}")
    else:
        print("  Cannot compute (need >= 2 groups)")
        p_anova = 1.0

    # Pairwise Welch's t-tests
    print(f"\n{'Pairwise Welchs t-tests':^72}")
    print("=" * 72)
    print(f"  {'Comparison':36s}  {'t-stat':>8s}  {'p-value':>10s}  {'d':>7s}  {'Sig':>4s}  {'Delta':>10s}")
    print("  " + "-" * 80)
    for i in range(len(valid_names)):
        for j in range(i + 1, len(valid_names)):
            a, b = valid_arrays[i], valid_arrays[j]
            t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
            d = cohens_d(a, b)
            diff_us = (a.mean() - b.mean()) * 1000
            sig = sig_stars(p_val)
            print(f"  {valid_names[i]:>17s} vs {valid_names[j]:<16s}  "
                  f"{t_stat:8.3f}  {p_val:10.2e}  {d:7.3f}  {sig:>4s}  {diff_us:+8.1f} us")

    n = min(len(a) for a in valid_arrays)
    print(f"\n  n = {n}-{max(len(a) for a in valid_arrays)} runs, alpha = 0.05")
    print(f"  ns = not significant, * p<0.05, ** p<0.01, *** p<0.001")

    return p_anova


# ── Grid search (optional) ──

def run_grid_search(kernel, tier='all'):
    """Run grid search and return the best config's dflags."""
    print(f"\n  Running grid search for {kernel} (tier={tier})...")
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, 'grid_search.py'),
           '--kernel', kernel, '--tier', tier]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=ROOT_DIR)
    if result.returncode != 0:
        print(f"  Grid search failed: {result.stderr[-300:]}")
        return None

    # Find the best config from output
    best_ms = None
    best_dflags = None
    for line in result.stdout.splitlines():
        # Grid search prints: [N/M] dflags ... ms / TFLOPS
        m = re.search(r'([\d.]+) ms / [\d.]+ TFLOPS', line)
        if m:
            ms = float(m.group(1))
            if best_ms is None or ms < best_ms:
                best_ms = ms
                # Extract dflags from the line
                dm = re.search(r'\] (.+?) \.\.\.', line)
                best_dflags = dm.group(1).strip() if dm else ''

    if best_ms is not None:
        print(f"  Best: {best_ms:.3f} ms  dflags: {best_dflags or '(defaults)'}")
    return best_dflags


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description='Unified benchmark: cuBLAS vs CUTLASS vs our kernel')
    parser.add_argument('--layer', nargs='+', choices=list(LAYERS.keys()),
                        default=list(LAYERS.keys()),
                        help='Layers to benchmark (default: all)')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of repeated runs per approach (default: 10)')
    parser.add_argument('--skip-build', action='store_true',
                        help='Skip build phase (reuse existing binaries)')
    parser.add_argument('--grid-search', action='store_true',
                        help='Run grid search before repeated measurements')
    parser.add_argument('--grid-tier', default='all',
                        help='Grid search tier (default: all)')
    parser.add_argument('--cutlass-mode', choices=['standard', 'max'], default='max',
                        help='CUTLASS sweep mode (default: max)')
    parser.add_argument('--csv', default=None,
                        help='Save raw samples to CSV')
    parser.add_argument('--imgs-per-sm', type=int, default=32,
                        help='Images per SM for M dimension (default: 32)')
    args = parser.parse_args()

    os.chdir(ROOT_DIR)

    print("=" * 72)
    print("  Unified Benchmark: cuBLAS vs CUTLASS vs Our Kernel")
    print(f"  Layers: {', '.join(args.layer)}")
    print(f"  Runs per approach: {args.runs}")
    print(f"  CUTLASS mode: {args.cutlass_mode}")
    print("=" * 72)

    # ── Build phase ──
    if not args.skip_build:
        print("\nBuild phase:")
        targets = set()
        for layer_name in args.layer:
            layer = LAYERS[layer_name]
            targets.add(layer['our_target'])
            targets.add(layer['cublas_target'])
            if args.cutlass_mode == 'max':
                targets.add(layer['cutlass_target_max'])
            else:
                targets.add(layer['cutlass_target'])

        if not build_targets(sorted(targets), verbose=True):
            print("\nBuild failed. Aborting.")
            sys.exit(1)
        print()

    # ── Optional grid search ──
    if args.grid_search:
        print("\nGrid search phase:")
        for layer_name in args.layer:
            kernel_name = layer_name if layer_name != 'fc1' else 'fc1_gelu'
            run_grid_search(kernel_name, args.grid_tier)

    # ── Collection phase ──
    all_raw = []  # for CSV: (layer, approach, run_idx, ms)
    layer_results = {}  # layer -> {approach: [ms, ...]}

    for layer_name in args.layer:
        layer = LAYERS[layer_name]
        use_max = args.cutlass_mode == 'max'
        cutlass_binary = layer['cutlass_binary_max'] if use_max else layer['cutlass_binary']

        print(f"\n{'=' * 72}")
        print(f"  {layer['label']}")
        print(f"{'=' * 72}")

        groups = {}

        # Our kernel
        print(f"\n  Our kernel ({layer['our_binary']}):")
        our_samples = collect_our_samples(layer['our_binary'], args.runs)
        if our_samples:
            groups['Our kernel'] = our_samples
            for i, ms in enumerate(our_samples):
                all_raw.append((layer_name, 'our_kernel', i, ms))

        # cuBLAS
        print(f"\n  cuBLAS ({layer['cublas_binary']}):")
        cublas_fused, cublas_gemm = collect_cublas_samples(
            layer['cublas_binary'], layer['epilogue'], args.runs)
        if cublas_fused:
            groups['cuBLAS fused'] = cublas_fused
            for i, ms in enumerate(cublas_fused):
                all_raw.append((layer_name, 'cublas_fused', i, ms))
        if cublas_gemm:
            groups['cuBLAS GEMM-only'] = cublas_gemm
            for i, ms in enumerate(cublas_gemm):
                all_raw.append((layer_name, 'cublas_gemm', i, ms))

        # CUTLASS
        print(f"\n  CUTLASS ({cutlass_binary}):")
        cutlass_fused, cutlass_gemm = collect_cutlass_samples(
            cutlass_binary, layer['epilogue'], args.runs)
        if cutlass_fused:
            groups['CUTLASS fused'] = cutlass_fused
            for i, ms in enumerate(cutlass_fused):
                all_raw.append((layer_name, 'cutlass_fused', i, ms))
        if cutlass_gemm:
            groups['CUTLASS GEMM-only'] = cutlass_gemm
            for i, ms in enumerate(cutlass_gemm):
                all_raw.append((layer_name, 'cutlass_gemm', i, ms))

        layer_results[layer_name] = groups

    # ── Analysis phase ──

    print(f"\n\n{'#' * 72}")
    print(f"{'#':>2}  RESULTS & STATISTICAL ANALYSIS")
    print(f"{'#' * 72}")

    for layer_name in args.layer:
        layer = LAYERS[layer_name]
        groups = layer_results[layer_name]

        print(f"\n{'=' * 72}")
        print(f"  {layer['label']}")
        print(f"{'=' * 72}")

        if not groups:
            print("  No data collected.")
            continue

        # Filter to fused-only groups for ANOVA (exclude GEMM-only reference)
        fused_groups = {k: v for k, v in groups.items() if 'GEMM-only' not in k}
        run_anova(fused_groups, layer['label'])

        # Also show GEMM-only comparison if available
        gemm_groups = {k: v for k, v in groups.items() if 'GEMM-only' in k}
        if gemm_groups:
            print(f"\n  GEMM-only reference (not fused):")
            for name, samples in sorted(gemm_groups.items(), key=lambda x: np.mean(x[1])):
                a = np.array(samples)
                print(f"    {name:20s}  mean={a.mean():.4f} ms  std={a.std(ddof=1):.4f}")

    # ── Cross-layer summary ──

    if len(args.layer) > 1:
        print(f"\n\n{'=' * 72}")
        print(f"  CROSS-LAYER SUMMARY (mean ms, fused)")
        print(f"{'=' * 72}")
        print(f"  {'Layer':20s}  {'Our kernel':>12s}  {'cuBLAS':>12s}  {'CUTLASS':>12s}  {'Best':>12s}")
        print("  " + "-" * 72)
        for layer_name in args.layer:
            groups = layer_results[layer_name]
            row = {'Our kernel': None, 'cuBLAS fused': None, 'CUTLASS fused': None}
            for k in row:
                if k in groups and groups[k]:
                    row[k] = np.mean(groups[k])
            vals = {k: v for k, v in row.items() if v is not None}
            best = min(vals, key=vals.get) if vals else "?"

            def fmt(v):
                return f"{v:12.4f}" if v is not None else f"{'n/a':>12s}"

            print(f"  {LAYERS[layer_name]['label']:20s}  "
                  f"{fmt(row['Our kernel'])}  {fmt(row['cuBLAS fused'])}  "
                  f"{fmt(row['CUTLASS fused'])}  {best}")

    # ── Save CSV ──

    if args.csv:
        csv_path = os.path.join(ROOT_DIR, args.csv) if not os.path.isabs(args.csv) else args.csv
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['layer', 'approach', 'run', 'ms'])
            for row in all_raw:
                writer.writerow(row)
        print(f"\nRaw samples saved to: {csv_path}")

    print(f"\nDone. {args.runs} runs x {len(args.layer)} layers x 3 approaches.")


if __name__ == '__main__':
    main()
