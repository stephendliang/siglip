#!/usr/bin/env python3
"""Analyze a B200 session output directory.

Reads machine info, compare.txt (ANOVA), ncu profiles, and cuBLAS SASS
captures. Prints a structured summary for pasting into Claude Code.

Usage:
    python3 tools/analyze_session.py data/session_YYYYMMDD_HHMMSS/
"""

import os
import re
import sys


def read_file(path):
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return None


def summarize_machine(outdir):
    txt = read_file(os.path.join(outdir, 'machine_info.txt'))
    if not txt:
        return
    print("=== Machine ===")
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith('---'):
            continue
        for key in ['Driver Version', 'CUDA Version', 'Product Name',
                     'GPU Max Clock', 'SM Clock', 'Memory Clock',
                     'nvcc: NVIDIA', 'release', 'GPU Current Clocks']:
            if key.lower() in line.lower():
                print(f"  {line}")
                break
    for line in txt.splitlines():
        if re.match(r'^[0-9a-f]{40}$', line.strip()):
            print(f"  git: {line.strip()[:12]}")
    print()


def summarize_comparison(outdir):
    txt = read_file(os.path.join(outdir, 'compare.txt'))
    if not txt:
        return

    print("=== Comparison (cuBLAS vs CUTLASS vs Ours) ===")

    # compare_all.py output has ={72} lines around layer labels AND around
    # sub-headers (Summary Statistics, One-way ANOVA, Pairwise Welch's).
    # Instead of splitting on = lines (which fragments the body), search
    # for each layer's block and extract ranking + ANOVA from the full text.
    layer_pattern = re.compile(
        r'={50,}\n\s+(Patch Embed|FC1\+GELU|FC2\+Bias).*?\n={50,}\n(.*?)(?=\n={50,}\n\s+(?:Patch Embed|FC1\+GELU|FC2\+Bias|CROSS-LAYER)|\Z)',
        re.DOTALL)
    for m in layer_pattern.finditer(txt):
        layer = m.group(1)
        body = m.group(2)

        ranking = re.findall(r'#\d+:\s+(\S.*?\S)\s+([\d.]+) ms', body)
        if ranking:
            print(f"\n  {layer}:")
            for name, ms in ranking:
                print(f"    {name:20s}  {ms} ms")

        anova = re.findall(r'F\(\d+.*?p = [\d.e+-]+.*', body)
        for line in anova:
            print(f"    ANOVA: {line.strip()}")

    # Cross-layer summary
    m = re.search(r'CROSS-LAYER SUMMARY.*?\n(.*?)(?=\n\n\n|\Z)', txt, re.DOTALL)
    if m:
        print(f"\n  Cross-layer summary:")
        for line in m.group(1).strip().splitlines():
            print(f"    {line.strip()}")

    print()


def summarize_ncu(outdir):
    found = False
    for name in ['patch_embed', 'fc1-gelu', 'fc2']:
        path = os.path.join(outdir, f'source_counters_{name}.csv')
        if not os.path.exists(path):
            continue
        if not found:
            print("=== ncu Source Counters ===")
            found = True
        size = os.path.getsize(path)
        print(f"  {name}: {size:,} bytes")
        print(f"    Analyze: python3 tools/analyze_source_counters.py {path}")

    full_rep = os.path.join(outdir, 'siglip_full.ncu-rep')
    if os.path.exists(full_rep):
        size = os.path.getsize(full_rep)
        print(f"  Full profile: {size:,} bytes ({full_rep})")

    if found:
        print()


def summarize_cublas_sass(outdir):
    try:
        sass_files = sorted(f for f in os.listdir(outdir)
                            if f.startswith('cublas_sass_'))
    except OSError:
        return
    if not sass_files:
        return
    print("=== cuBLAS SASS Captures ===")
    for f in sass_files:
        path = os.path.join(outdir, f)
        size = os.path.getsize(path)
        txt = read_file(path)
        mma_count = len(re.findall(r'UTCQMMA', txt)) if txt else 0
        # Extract layer from filename: cublas_sass_cublas-bench-fc1_<hash>.txt
        layer = "unknown"
        m = re.match(r'cublas_sass_(cublas-bench(?:-fc[12])?)_', f)
        if m:
            tag = m.group(1)
            layer = {'cublas-bench': 'patch_embed',
                     'cublas-bench-fc1': 'FC1',
                     'cublas-bench-fc2': 'FC2'}.get(tag, tag)
        if size < 100:
            print(f"  {f} [{layer}]: {size} bytes (likely empty — cuobjdump may have failed)")
        else:
            print(f"  {f} [{layer}]: {size:,} bytes, {mma_count} UTCQMMA instructions")
            if mma_count > 0:
                print(f"    Analyze: python3 tools/sass_analysis.py {path}")
    print()


def summarize_failures(outdir):
    log = read_file(os.path.join(outdir, 'session.log'))
    if not log:
        return
    fails = [l for l in log.splitlines() if 'FAIL' in l or 'WARN' in l]
    if fails:
        print("=== Warnings/Failures ===")
        for f in fails:
            print(f"  {f}")
        print()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <session_dir>")
        sys.exit(1)

    # Support glob expansion: use the last (most recent) directory
    outdir = sys.argv[-1].rstrip('/')
    if not os.path.isdir(outdir):
        print(f"Not a directory: {outdir}")
        sys.exit(1)

    print(f"Session: {os.path.basename(outdir)}")
    print()

    summarize_machine(outdir)
    summarize_comparison(outdir)
    summarize_ncu(outdir)
    summarize_cublas_sass(outdir)
    summarize_failures(outdir)

    print("Paste this output into Claude Code for interpretation.")


if __name__ == '__main__':
    main()
