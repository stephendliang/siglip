#!/usr/bin/env python3
"""
ANOVA-style analysis of ncu CSV data across kernel variants.
Reads all CSVs from a session directory, pivots into a metric×variant table,
and highlights the biggest differences between strip/fused and w3/cutlass.
"""
import csv, sys, os, glob
from collections import defaultdict

def load_csv(path):
    """Returns {metric_name: (value_str, unit)}"""
    metrics = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            name = row["Metric Name"]
            val = row["Metric Value"]
            unit = row["Metric Unit"]
            metrics[name] = (val, unit)
    return metrics

def parse_val(s):
    """Parse ncu metric value string to float, or None."""
    if s in ("n/a", "", "N/A"):
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None

def main():
    if len(sys.argv) < 2:
        datadir = max(glob.glob("data/ncu_*"), key=os.path.getmtime)
        print(f"Auto-selected: {datadir}")
    else:
        datadir = sys.argv[1]

    csvfiles = sorted(glob.glob(os.path.join(datadir, "*.csv")))
    if not csvfiles:
        print(f"No CSVs in {datadir}")
        sys.exit(1)

    # Load all variants
    variants = {}
    for path in csvfiles:
        name = os.path.basename(path).replace(".csv", "")
        variants[name] = load_csv(path)

    vnames = sorted(variants.keys())
    all_metrics = sorted(set(m for v in variants.values() for m in v))

    # === Table 1: Full metric table ===
    print("=" * 120)
    print("FULL METRIC TABLE")
    print("=" * 120)

    # Header
    hdr = f"{'Metric':<62} {'Unit':<8}"
    for vn in vnames:
        hdr += f" {vn:>12}"
    print(hdr)
    print("-" * len(hdr))

    rows_for_analysis = []  # (metric, unit, {variant: float_val})

    for metric in all_metrics:
        unit = ""
        vals = {}
        for vn in vnames:
            if metric in variants[vn]:
                val_str, u = variants[vn][metric]
                unit = u
                vals[vn] = parse_val(val_str)
            else:
                vals[vn] = None

        row = f"{metric:<62} {unit:<8}"
        for vn in vnames:
            v = vals[vn]
            if v is None:
                row += f" {'n/a':>12}"
            elif v >= 1e6:
                row += f" {v:>12,.0f}"
            elif v >= 100:
                row += f" {v:>12,.1f}"
            else:
                row += f" {v:>12.2f}"
        print(row)
        rows_for_analysis.append((metric, unit, vals))

    # === Comparison analyses ===

    comparisons = [
        ("Q1: w3 fusion cost (fused - gemm)", "w3_fused", "w3_gemm"),
        ("Q1b: CUTLASS fusion cost (fused - strip)", "cutlass_fused", "cutlass_strip"),
        ("Q2: w3 vs CUTLASS fused (18us gap)", "w3_fused", "cutlass_fused"),
        ("Q3: w3 vs CUTLASS GEMM (46us gap)", "w3_gemm", "cutlass_strip"),
        ("Q5: epi1 vs epi4 fused (warp count)", "epi1_fused", "w3_fused"),
        ("Q6: w3 output cost (gemm - strip)", "w3_gemm", "w3_strip"),
        ("Q8: inline vs sched (TD=6 vs TD=4)", "w3_inline", "w3_sched"),
        ("Q9: inline vs cutlass fused", "w3_inline", "cutlass_fused"),
        ("Q10: prefill vs baseline fused", "w3_prefill", "w3_fused"),
        ("Q11: inline+prefill vs cutlass fused", "w3_inline_prefill", "cutlass_fused"),
        ("Q12: NS7 vs NS6 fused", "w3_ns7", "w3_fused"),
        ("Q13: NS7 vs cutlass fused", "w3_ns7", "cutlass_fused"),
        ("Q14: LDG fused vs w3 fused", "ldg_fused", "w3_fused"),
        ("Q15: LDG fused vs cutlass fused", "ldg_fused", "cutlass_fused"),
        ("Q16: LDG gemm vs w3 gemm", "ldg_gemm", "w3_gemm"),
    ]

    for title, a, b in comparisons:
        if a not in variants or b not in variants:
            continue
        print()
        print("=" * 120)
        print(f"{title}")
        print(f"  A = {a},  B = {b}")
        print(f"  Positive delta% = A is HIGHER than B")
        print("=" * 120)

        diffs = []
        for metric, unit, vals in rows_for_analysis:
            va, vb = vals.get(a), vals.get(b)
            if va is None or vb is None:
                continue
            if vb == 0 and va == 0:
                continue

            abs_diff = va - vb
            if vb != 0:
                pct = (va - vb) / abs(vb) * 100
            else:
                pct = float('inf') if va > 0 else float('-inf')

            diffs.append((abs(pct), pct, abs_diff, va, vb, metric, unit))

        diffs.sort(reverse=True)

        hdr2 = f"{'Metric':<58} {'Unit':<8} {'A':>12} {'B':>12} {'Diff':>12} {'Diff%':>8}"
        print(hdr2)
        print("-" * len(hdr2))

        for _, pct, abs_diff, va, vb, metric, unit in diffs[:30]:
            def fmt(v):
                if v >= 1e6:
                    return f"{v:>12,.0f}"
                elif v >= 100:
                    return f"{v:>12,.1f}"
                else:
                    return f"{v:>12.2f}"

            pct_str = f"{pct:>+7.1f}%" if abs(pct) < 1e6 else f"{'>>':>8}"
            diff_str = f"{abs_diff:>+12,.1f}" if abs(abs_diff) >= 1 else f"{abs_diff:>+12.2f}"
            print(f"{metric:<58} {unit:<8} {fmt(va)} {fmt(vb)} {diff_str} {pct_str}")

    # === Key stall breakdown ===
    print()
    print("=" * 120)
    print("STALL BREAKDOWN (warps_issue_stalled_*)")
    print("=" * 120)

    stall_metrics = [m for m in all_metrics if "warps_issue_stalled" in m]

    for vn in vnames:
        print(f"\n  {vn}:")
        stalls = []
        for m in stall_metrics:
            if m in variants[vn]:
                v = parse_val(variants[vn][m][0])
                if v is not None:
                    stalls.append((v, m))
        stalls.sort(reverse=True)
        total = sum(v for v, _ in stalls)
        for v, m in stalls:
            short = m.replace("smsp__warps_issue_stalled_", "")
            pct = v / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"    {short:<30} {v:>12,.1f}  ({pct:>5.1f}%) {bar}")
        print(f"    {'TOTAL':<30} {total:>12,.1f}")

    # === Fused-Strip delta for stalls ===
    print()
    print("=" * 120)
    print("STALL DELTA: fused - strip (what epilogue adds)")
    print("=" * 120)

    pairs = [("w3", "w3_fused", "w3_strip"), ("cutlass", "cutlass_fused", "cutlass_strip"),
             ("epi1", "epi1_fused", "w3_strip")]

    hdr3 = f"{'Stall type':<30}"
    for label, _, _ in pairs:
        hdr3 += f" {label+' Δ':>12} {label+' Δ%':>8}"
    print(hdr3)
    print("-" * len(hdr3))

    for m in stall_metrics:
        short = m.replace("smsp__warps_issue_stalled_", "")
        row = f"{short:<30}"
        for label, fused, strip in pairs:
            vf = parse_val(variants.get(fused, {}).get(m, ("n/a",))[0]) if fused in variants else None
            vs = parse_val(variants.get(strip, {}).get(m, ("n/a",))[0]) if strip in variants else None
            if vf is not None and vs is not None:
                delta = vf - vs
                dpct = (vf - vs) / abs(vs) * 100 if vs != 0 else float('inf')
                row += f" {delta:>+12,.1f} {dpct:>+7.1f}%"
            else:
                row += f" {'n/a':>12} {'n/a':>8}"
        print(row)

if __name__ == "__main__":
    main()
