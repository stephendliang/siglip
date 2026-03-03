#!/usr/bin/env python3
"""Compare SASS dumps between CUTLASS and custom kernel.

Extracts key instruction counts from cuobjdump --dump-sass output:
- MMA instructions (UTCQMMA)
- TMEM loads (LDTM) with width breakdown (x16/x32/x64)
- TMEM waits/fences
- Register-to-uniform moves (R2UR)
- Shared memory ops (STS, LDS)
- Global stores (STG)
- FP conversions (F2FP BF16 pack)
- BF16 epilogue math (HFMA2, HADD2)
- Total instruction count per kernel function

Usage:
    python3 compare_sass.py sass/cutlass_default.txt sass/custom_kernel.txt
    python3 compare_sass.py sass/cutlass_256x256x64.txt sass/custom_kernel.txt
"""
import sys
import re
from collections import Counter

# Opcodes to count (case-insensitive regex match against the full instruction line)
CATEGORIES = {
    'MMA (UTCQMMA)':       r'UTCQMMA',
    'TMEM load (LDTM)':    r'\bLDTM\b',
    '  LDTM.x16':          r'\bLDTM\.x16\b',
    '  LDTM.x32':          r'\bLDTM\.x32\b',
    '  LDTM.x64':          r'\bLDTM\.x64\b',
    'MMA commit (UTCBAR)': r'UTCBAR',
    'TMEM fence':          r'MEMBAR.*TMEM|UTCFENCE|FENCE\.VIEW',
    'R2UR':                r'\bR2UR\b',
    'ELECT':               r'\bELECT\b',
    'PLOP3':               r'\bPLOP3\b',
    'F2FP (BF16 pack)':    r'\bF2FP\b',
    'HFMA2 (BF16 fma)':    r'\bHFMA2\b',
    'HADD2 (BF16 add)':    r'\bHADD2\b',
    'FMUL':                r'\bFMUL\b',
    'FFMA':                r'\bFFMA\b',
    'STS (st.shared)':     r'\bSTS\b',
    'LDS (ld.shared)':     r'\bLDS\b',
    'STG (st.global)':     r'\bSTG\b',
    'LDG (ld.global)':     r'\bLDG\b',
    'SYNCS (barrier)':     r'\bSYNCS\b',
    'NANOSLEEP':           r'NANOSLEEP',
    'SETP':                r'\bSETP\b',
    'IMAD':                r'\bIMAD\b',
    'Total instructions':  None,  # special: count all instruction lines
}


def parse_sass(filename):
    """Parse a SASS dump, return per-function instruction counts."""
    functions = {}
    current_fn = None
    counts = Counter()
    total = 0

    with open(filename) as f:
        for line in f:
            # Function header: "Function : NAME" or ".text.NAME:"
            m = re.match(r'\s*(?:\.text\.([\w.:]+)|Function\s*:\s*(\S+))', line)
            if m:
                if current_fn and total > 0:
                    counts['Total instructions'] = total
                    functions[current_fn] = dict(counts)
                current_fn = m.group(1) or m.group(2)
                counts = Counter()
                total = 0
                continue

            # Instruction line: has /*ADDR*/ followed by opcode
            m = re.match(r'\s*/\*[0-9a-f]+\*/\s+(\S+)', line)
            if not m:
                continue

            total += 1

            for cat, pattern in CATEGORIES.items():
                if pattern is None:
                    continue
                if re.search(pattern, line, re.IGNORECASE):
                    counts[cat] += 1

    # Last function
    if current_fn and total > 0:
        counts['Total instructions'] = total
        functions[current_fn] = dict(counts)

    return functions


def find_gemm_kernel(functions):
    """Find the main GEMM kernel (largest function with MMA instructions)."""
    best = None
    best_mma = 0
    for name, counts in functions.items():
        mma = counts.get('MMA (UTCQMMA)', 0)
        if mma > best_mma:
            best_mma = mma
            best = name
    return best


def print_comparison(name_a, fns_a, name_b, fns_b):
    """Print side-by-side comparison of the main GEMM kernels."""
    kernel_a = find_gemm_kernel(fns_a)
    kernel_b = find_gemm_kernel(fns_b)

    if not kernel_a:
        print(f"WARNING: No MMA instructions found in {name_a}")
        return
    if not kernel_b:
        print(f"WARNING: No MMA instructions found in {name_b}")
        return

    counts_a = fns_a[kernel_a]
    counts_b = fns_b[kernel_b]

    # Truncate long kernel names
    short_a = kernel_a[:50] + '...' if len(kernel_a) > 50 else kernel_a
    short_b = kernel_b[:50] + '...' if len(kernel_b) > 50 else kernel_b

    print(f"\n{'=' * 72}")
    print(f"  SASS Comparison")
    print(f"{'=' * 72}")
    print(f"  A: {name_a}")
    print(f"     {short_a}")
    print(f"  B: {name_b}")
    print(f"     {short_b}")
    print()

    hdr = f"  {'Category':<25s}  {'A':>8s}  {'B':>8s}  {'Delta':>8s}"
    print(hdr)
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*8}")

    for cat in CATEGORIES:
        a = counts_a.get(cat, 0)
        b = counts_b.get(cat, 0)
        delta = a - b
        sign = '+' if delta > 0 else ''
        if a == 0 and b == 0:
            continue
        print(f"  {cat:<25s}  {a:>8d}  {b:>8d}  {sign}{delta:>7d}")

    print()

    # Summary stats
    total_a = counts_a.get('Total instructions', 0)
    total_b = counts_b.get('Total instructions', 0)
    mma_a = counts_a.get('MMA (UTCQMMA)', 0)
    mma_b = counts_b.get('MMA (UTCQMMA)', 0)

    if mma_a > 0:
        overhead_a = (total_a - mma_a) / mma_a
        print(f"  A: {overhead_a:.1f} overhead instructions per MMA ({total_a} total / {mma_a} MMA)")
    if mma_b > 0:
        overhead_b = (total_b - mma_b) / mma_b
        print(f"  B: {overhead_b:.1f} overhead instructions per MMA ({total_b} total / {mma_b} MMA)")

    # TMEM load width summary
    ldtm_a = counts_a.get('TMEM load (LDTM)', 0)
    ldtm_b = counts_b.get('TMEM load (LDTM)', 0)
    if ldtm_a > 0 or ldtm_b > 0:
        print(f"\n  TMEM load pattern:")
        for label in ['  LDTM.x16', '  LDTM.x32', '  LDTM.x64']:
            a = counts_a.get(label, 0)
            b = counts_b.get(label, 0)
            if a > 0 or b > 0:
                print(f"    A: {a:>3d} {label.strip():<10s}  B: {b:>3d} {label.strip()}")

    # Function count summary
    print(f"\n  Functions in A: {len(fns_a)}")
    print(f"  Functions in B: {len(fns_b)}")

    # List all functions with instruction counts
    print(f"\n  All functions in A (by size):")
    for name, counts in sorted(fns_a.items(), key=lambda x: x[1].get('Total instructions', 0), reverse=True)[:10]:
        t = counts.get('Total instructions', 0)
        m = counts.get('MMA (UTCQMMA)', 0)
        short = name[:60] + '...' if len(name) > 60 else name
        print(f"    {t:>6d} instr  {m:>3d} MMA  {short}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <sass_a.txt> <sass_b.txt>")
        print(f"  Compare SASS dumps from cuobjdump --dump-sass")
        sys.exit(1)

    fns_a = parse_sass(sys.argv[1])
    fns_b = parse_sass(sys.argv[2])

    print_comparison(sys.argv[1], fns_a, sys.argv[2], fns_b)
