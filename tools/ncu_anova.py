#!/usr/bin/env python3
"""
ANOVA-style analysis of ncu CSV data across kernel variants.

Consumes the label schema produced by tools/bench.sh:
    {layer}_{dispatch}_{pk}_{mode}.csv    (pk in {p, np}, mode in {fused, gemm, strip})
    cutlass_fc2_fused.csv, cutlass_fc2_strip.csv, cublas_fc{1,2}_gemm.csv

Produces:
  1. Full metric × variant table.
  2. Per (layer, pk, mode) cross-dispatch comparison vs default.
  3. Per (layer, dispatch, pk) fused-vs-gemm and gemm-vs-strip delta (what
     the epilogue / residual add).
  4. Stall breakdown per variant (sorted, percent of total).
  5. Fused - strip stall delta, per (layer, dispatch, pk).
"""
import csv, sys, os, glob, re
from collections import defaultdict

UNIT_SCALE = {
    "Gbyte": 1e9, "Mbyte": 1e6, "Kbyte": 1e3, "byte": 1,
    "Gsector": 1e9, "Msector": 1e6, "Ksector": 1e3, "sector": 1,
}

def load_csv(path):
    metrics = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            name = row.get("Metric Name", "")
            val  = row.get("Metric Value", "")
            unit = row.get("Metric Unit", "")
            if not name: continue
            metrics[name] = (val, unit)
    return metrics

def parse_val(s, unit=""):
    if s in ("n/a", "", "N/A"): return None
    try: v = float(s.replace(",", ""))
    except ValueError: return None
    return v * UNIT_SCALE.get(unit, 1.0)

def parse_label(lbl):
    """Return (layer, dispatch, pk, mode) or None for baselines/bad."""
    if lbl.startswith(("cutlass_", "cublas_")):
        return None
    parts = lbl.split("_")
    if len(parts) < 4: return None
    layer = parts[0]
    if layer not in ("fc1", "fc2"): return None
    mode = parts[-1]
    pk   = parts[-2]
    disp = "_".join(parts[1:-2])
    return layer, disp, pk, mode

def fmt_num(v):
    if v is None: return "n/a"
    av = abs(v)
    if av >= 1e9: return f"{v/1e9:>11.2f}G"
    if av >= 1e6: return f"{v/1e6:>11.2f}M"
    if av >= 1e3: return f"{v/1e3:>11.1f}K"
    return f"{v:>12.2f}"

def fmt_pct(v):
    if v is None: return "n/a"
    if abs(v) >= 1e6: return ">>>"
    return f"{v:>+7.1f}%"

def main():
    if len(sys.argv) < 2:
        cands = glob.glob("data/bench_*") + glob.glob("data/ncu_*")
        if not cands:
            print("usage: ncu_anova.py <bench_or_ncu_dir>"); sys.exit(1)
        datadir = max(cands, key=os.path.getmtime)
        print(f"Auto-selected: {datadir}\n")
    else:
        datadir = sys.argv[1]

    csvfiles = sorted(glob.glob(os.path.join(datadir, "*.csv")))
    csvfiles = [p for p in csvfiles if not os.path.basename(p).startswith("full_")]
    if not csvfiles:
        print(f"No CSVs in {datadir}"); sys.exit(1)

    variants = {}
    for path in csvfiles:
        name = os.path.basename(path).replace(".csv", "")
        variants[name] = load_csv(path)

    vnames = sorted(variants.keys())
    all_metrics = sorted(set(m for v in variants.values() for m in v))

    def get(vn, metric):
        v, u = variants[vn].get(metric, ("", ""))
        return parse_val(v, u)

    # ── Table 1: full metric × variant ──
    print("=" * 120)
    print("FULL METRIC TABLE  (directory: %s)" % datadir)
    print("=" * 120)
    hdr = f"{'Metric':<62} {'Unit':<8}"
    col_w = max(10, min(16, 180 // max(1, len(vnames))))
    for vn in vnames: hdr += f" {vn[:col_w]:>{col_w}}"
    print(hdr); print("-" * min(len(hdr), 180))

    rows_for_analysis = []
    for metric in all_metrics:
        unit = ""
        vals = {}
        for vn in vnames:
            if metric in variants[vn]:
                val_str, u = variants[vn][metric]
                unit = u
                vals[vn] = parse_val(val_str, u)
            else:
                vals[vn] = None
        row = f"{metric:<62} {unit:<8}"
        for vn in vnames:
            v = vals[vn]
            s = "n/a" if v is None else (f"{v:,.0f}" if abs(v) >= 1e6 else (f"{v:,.1f}" if abs(v) >= 100 else f"{v:.2f}"))
            row += f" {s:>{col_w}}"
        print(row)
        rows_for_analysis.append((metric, unit, vals))

    # ── Group variants by parsed label ──
    parsed = {}   # vn -> (layer, disp, pk, mode)
    baselines = []
    for vn in vnames:
        p = parse_label(vn)
        if p is None: baselines.append(vn)
        else: parsed[vn] = p

    by_group = defaultdict(dict)  # (layer, pk, mode) -> {disp: vn}
    by_triple = defaultdict(dict) # (layer, disp, pk)  -> {mode: vn}
    for vn, (layer, disp, pk, mode) in parsed.items():
        by_group[(layer, pk, mode)][disp] = vn
        by_triple[(layer, disp, pk)][mode] = vn

    # ── Cross-dispatch comparison (per layer, pk, mode) ──
    def pairwise_block(title, a, b, metrics_filter=None):
        print(); print("=" * 120); print(title)
        print(f"  A = {a},  B = {b}   (positive delta% = A higher than B)")
        print("=" * 120)
        diffs = []
        for metric, unit, vals in rows_for_analysis:
            if metrics_filter and metric not in metrics_filter: continue
            va, vb = vals.get(a), vals.get(b)
            if va is None or vb is None: continue
            if vb == 0 and va == 0: continue
            abs_d = va - vb
            pct = (va - vb) / abs(vb) * 100 if vb != 0 else (float('inf') if va > 0 else float('-inf'))
            diffs.append((abs(pct), pct, abs_d, va, vb, metric, unit))
        diffs.sort(reverse=True)
        hdr2 = f"{'Metric':<58} {'Unit':<8} {'A':>13} {'B':>13} {'Diff':>13} {'Diff%':>9}"
        print(hdr2); print("-" * len(hdr2))
        for _, pct, abs_d, va, vb, metric, unit in diffs[:25]:
            diff_str = f"{abs_d:>+13,.1f}" if abs(abs_d) >= 1 else f"{abs_d:>+13.3f}"
            va_str = f"{va:,.0f}" if abs(va) >= 1e6 else (f"{va:,.1f}" if abs(va) >= 100 else f"{va:.2f}")
            vb_str = f"{vb:,.0f}" if abs(vb) >= 1e6 else (f"{vb:,.1f}" if abs(vb) >= 100 else f"{vb:.2f}")
            print(f"{metric:<58} {unit:<8} {va_str:>13} {vb_str:>13} {diff_str} {fmt_pct(pct):>9}")

    print("\n" + "#" * 120)
    print("# CROSS-DISPATCH COMPARISONS (within same layer + pk + mode; baseline = default, else alphabetically first)")
    print("#" * 120)
    for (layer, pk, mode), disps in sorted(by_group.items()):
        if len(disps) < 2: continue
        base = disps.get("default") or sorted(disps.values())[0]
        base_disp = next(d for d,v in disps.items() if v == base)
        for disp, vn in sorted(disps.items()):
            if vn == base: continue
            pairwise_block(
                f"{layer} pk={pk} mode={mode}: {disp} vs {base_disp}",
                vn, base,
            )

    # ── Decomposition per (layer, disp, pk): fused-vs-gemm, gemm-vs-strip ──
    print("\n" + "#" * 120)
    print("# MODE DECOMPOSITION (per layer+dispatch+pk: fused−gemm, gemm−strip)")
    print("#" * 120)
    for (layer, disp, pk), modes in sorted(by_triple.items()):
        f, g, s = modes.get("fused"), modes.get("gemm"), modes.get("strip")
        if f and g:
            pairwise_block(f"{layer} {disp} pk={pk}: fused - gemm  (residual load + bias mul cost)", f, g)
        if g and s:
            pairwise_block(f"{layer} {disp} pk={pk}: gemm - strip  (output store cost)", g, s)

    # ── CUTLASS / cuBLAS cross-compare against best w3 variant, if present ──
    print("\n" + "#" * 120)
    print("# BASELINE COMPARISONS (cutlass / cublas vs best w3 per layer+mode)")
    print("#" * 120)
    for bl in baselines:
        m = re.match(r"(cutlass|cublas)_(fc\d)_(\w+)", bl)
        if not m: continue
        _src, layer, mode = m.groups()
        cands = by_group.get((layer, "p", mode), {})
        if not cands: cands = by_group.get((layer, "np", mode), {})
        if not cands: continue
        peer = cands.get("default") or cands.get("lean") or sorted(cands.values())[0]
        pairwise_block(f"{bl} vs {peer}", peer, bl)

    # ── Stall breakdown per variant ──
    print("\n" + "=" * 120); print("STALL BREAKDOWN (per variant, sorted)"); print("=" * 120)
    stall_metrics = [m for m in all_metrics if "warps_issue_stalled" in m]
    for vn in vnames:
        stalls = []
        for m in stall_metrics:
            v = get(vn, m)
            if v is not None: stalls.append((v, m))
        if not stalls: continue
        stalls.sort(reverse=True)
        total = sum(v for v, _ in stalls) or 1.0
        print(f"\n  {vn}:")
        for v, m in stalls:
            short = m.replace("smsp__warps_issue_stalled_", "").replace(".avg", "")
            pct = v / total * 100
            bar = "█" * int(pct / 2)
            print(f"    {short:<28} {v:>12,.1f}  ({pct:>5.1f}%) {bar}")
        print(f"    {'TOTAL':<28} {total:>12,.1f}")

    # ── Stall delta fused-strip per (layer, disp, pk) ──
    print("\n" + "=" * 120); print("FUSED − STRIP STALL DELTA  (what the epilogue adds, per triple)"); print("=" * 120)
    triples = [(l, d, p) for (l, d, p), modes in by_triple.items() if "fused" in modes and "strip" in modes]
    if not triples:
        print("  (no triples have both fused and strip)")
    else:
        hdr3 = f"{'stall':<28}"
        for l, d, p in sorted(triples): hdr3 += f" {(d+'/'+l+'/'+p)[:14]:>14}"
        print(hdr3); print("-" * min(len(hdr3), 180))
        for m in stall_metrics:
            short = m.replace("smsp__warps_issue_stalled_", "").replace(".avg", "")
            row = f"{short:<28}"
            for l, d, p in sorted(triples):
                vf = get(by_triple[(l,d,p)]["fused"], m)
                vs = get(by_triple[(l,d,p)]["strip"], m)
                if vf is None or vs is None: row += f" {'n/a':>14}"; continue
                delta = vf - vs
                row += f" {delta:>+14,.0f}"
            print(row)

if __name__ == "__main__":
    main()
