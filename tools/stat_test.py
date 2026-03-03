#!/usr/bin/env python3
"""Statistical comparison of TMEM load width variants (x16, x32, x64).

Runs each binary N times, extracts kernel timing, performs pairwise
Welch's t-tests and reports effect sizes (Cohen's d).
"""

import subprocess
import re
import sys
import numpy as np
from scipy import stats

VARIANTS = {
    "x16 (baseline)": "./siglip_vision",
    "x32":            "./siglip_x32",
    "x64":            "./siglip_x64",
}

N_RUNS = 30  # per variant


def run_once(binary: str) -> float:
    """Run binary, return kernel time in ms."""
    result = subprocess.run([binary], capture_output=True, text=True, timeout=30)
    output = result.stdout + result.stderr
    m = re.search(r"Custom kernel:\s+([\d.]+)\s+ms", output)
    if not m:
        raise RuntimeError(f"Could not parse timing from {binary}:\n{output}")
    return float(m.group(1))


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 +
                          (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_RUNS
    print(f"Running {n} iterations per variant...\n")

    timings = {}
    for name, binary in VARIANTS.items():
        samples = []
        for i in range(n):
            t = run_once(binary)
            samples.append(t)
            print(f"  {name:16s}  run {i+1:2d}/{n}  {t:.3f} ms")
        timings[name] = np.array(samples)
        print()

    # Summary statistics
    print("=" * 72)
    print(f"{'Variant':16s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}  {'Median':>8s}  {'IQR':>8s}")
    print("-" * 72)
    for name, t in timings.items():
        iqr = np.percentile(t, 75) - np.percentile(t, 25)
        print(f"{name:16s}  {t.mean():8.4f}  {t.std(ddof=1):8.4f}  {t.min():8.4f}  "
              f"{t.max():8.4f}  {np.median(t):8.4f}  {iqr:8.4f}")

    # Pairwise Welch's t-tests
    names = list(timings.keys())
    print(f"\n{'Pairwise Welchs t-tests':^72}")
    print("=" * 72)
    print(f"{'Comparison':32s}  {'t-stat':>8s}  {'p-value':>10s}  {'Cohen d':>8s}  {'Sig?':>6s}")
    print("-" * 72)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = timings[names[i]], timings[names[j]]
            t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
            d = cohens_d(a, b)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            diff_us = (a.mean() - b.mean()) * 1000  # microseconds
            print(f"{names[i]:>15s} vs {names[j]:<14s}  {t_stat:8.3f}  {p_val:10.6f}  {d:8.3f}  {sig:>6s}")
            print(f"{'':32s}  mean diff = {diff_us:+.1f} µs")

    # One-way ANOVA
    print(f"\n{'One-way ANOVA':^72}")
    print("=" * 72)
    f_stat, p_anova = stats.f_oneway(*timings.values())
    print(f"F = {f_stat:.3f},  p = {p_anova:.6f}  {'(significant)' if p_anova < 0.05 else '(not significant)'}")

    print(f"\nn = {n} runs per variant, α = 0.05")
    print("ns = not significant, * p<0.05, ** p<0.01, *** p<0.001")


if __name__ == "__main__":
    main()
