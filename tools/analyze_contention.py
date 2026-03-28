#!/usr/bin/env python3
"""Analyze FC2 contention bench results — wall time decomposition + ncu metric scaling.

Usage:
    python3 tools/analyze_contention.py data/contention_*/

Reads:
    wall_summary.txt     — already formatted by the bench script
    ncu/*.csv            — ncu --csv output per variant
    wall_results.txt     — raw @@RESULT lines

Outputs analysis to stdout.
"""

import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_ncu_csv(filepath):
    """Parse ncu --csv --page raw output into {metric_name: value}."""
    metrics = {}
    try:
        with open(filepath) as f:
            # Skip ncu header lines (start with ==)
            lines = []
            for line in f:
                if line.startswith('"') or line.startswith('metric') or ',' in line:
                    lines.append(line)

            if not lines:
                return metrics

            # Find the actual CSV data — ncu CSV has a header row then data rows
            # Format: "ID","Metric Name","Metric Unit","Metric Value"
            # or multi-kernel: extra columns for kernel name, invocation, etc.
            reader = csv.reader(lines)
            header = None
            for row in reader:
                if not row:
                    continue
                # Find header row
                if any('Metric Name' in col for col in row):
                    header = row
                    name_idx = next(i for i, c in enumerate(row) if 'Metric Name' in c)
                    val_idx = next(i for i, c in enumerate(row) if 'Metric Value' in c)
                    continue
                if header is None:
                    continue
                try:
                    name = row[name_idx].strip()
                    val_str = row[val_idx].strip().replace(',', '')
                    if val_str and val_str != 'n/a':
                        try:
                            val = float(val_str)
                            # If multiple kernel invocations, take the last (main kernel)
                            metrics[name] = val
                        except ValueError:
                            pass
                except (IndexError, StopIteration):
                    continue
    except Exception as e:
        print(f"  WARN: failed to parse {filepath}: {e}", file=sys.stderr)
    return metrics


def parse_wall_results(filepath):
    """Parse wall_results.txt into {label: {ms, regs, valid}}."""
    results = {}
    try:
        with open(filepath) as f:
            for line in f:
                ms_m = re.search(r'ms=([0-9.]+)', line)
                label_m = re.search(r'label=(\S+)', line)
                regs_m = re.search(r'regs=(\d+)', line)
                valid_m = re.search(r'valid=([01])', line)
                if ms_m and label_m:
                    results[label_m.group(1)] = {
                        'ms': float(ms_m.group(1)),
                        'regs': int(regs_m.group(1)) if regs_m else 0,
                        'valid': int(valid_m.group(1)) if valid_m else -1,
                    }
    except FileNotFoundError:
        pass
    return results


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <contention_dir>")
        sys.exit(1)

    outdir = Path(sys.argv[1].rstrip('/'))
    if not outdir.is_dir():
        print(f"Not a directory: {outdir}")
        sys.exit(1)

    print(f"{'='*72}")
    print(f"  FC2 CONTENTION ANALYSIS — {outdir.name}")
    print(f"{'='*72}")

    # ── Wall time results ──
    wall = parse_wall_results(outdir / 'wall_results.txt')
    if wall:
        strip_ms = wall.get('strip', {}).get('ms', 0)

        print(f"\n{'─'*72}")
        print("  WALL TIME — WARP SCALING")
        print(f"{'─'*72}")
        warp_labels = ['strip', 'w1', 'w2', 'w3', 'baseline']
        warp_counts = [0, 1, 2, 3, 4]
        print(f"  {'Warps':>5} {'Label':>12} {'ms':>8} {'Delta':>8} {'Per-warp':>10}")
        print(f"  {'─'*5:>5} {'─'*12:>12} {'─'*8:>8} {'─'*8:>8} {'─'*10:>10}")
        prev_ms = strip_ms
        for wc, label in zip(warp_counts, warp_labels):
            if label in wall:
                ms = wall[label]['ms']
                delta = ms - strip_ms
                per_warp = delta / wc if wc > 0 else 0
                print(f"  {wc:>5} {label:>12} {ms:>8.3f} {delta:>+8.3f} {per_warp:>10.1f}μs" if wc > 0
                      else f"  {wc:>5} {label:>12} {ms:>8.3f} {'—':>8} {'—':>10}")
                prev_ms = ms

        # Per-operation at 1 warp
        w1_full = wall.get('w1_full', {}).get('ms', 0)
        if w1_full and strip_ms:
            w1_overhead = w1_full - strip_ms
            print(f"\n{'─'*72}")
            print(f"  1-WARP DECOMPOSITION (total overhead: {w1_overhead*1000:.0f}μs)")
            print(f"{'─'*72}")
            w1_ops = [
                ('w1_full', 'Full epilogue'),
                ('w1_only_tma_load', 'TMA loads only'),
                ('w1_only_tma_store', 'TMA stores only'),
                ('w1_only_compute', 'STS+BF16 only'),
                ('w1_no_tma_load', 'No TMA loads'),
                ('w1_no_tma_store', 'No TMA stores'),
                ('w1_no_compute', 'No STS+BF16'),
            ]
            print(f"  {'Operation':>20} {'ms':>8} {'Delta':>8} {'% of 1W':>8}")
            print(f"  {'─'*20:>20} {'─'*8:>8} {'─'*8:>8} {'─'*8:>8}")
            for label, desc in w1_ops:
                if label in wall:
                    ms = wall[label]['ms']
                    delta = ms - strip_ms
                    pct = delta / w1_overhead * 100 if w1_overhead > 0 else 0
                    print(f"  {desc:>20} {ms:>8.3f} {delta:>+8.3f} {pct:>7.1f}%")

            # Additivity check
            only_load = wall.get('w1_only_tma_load', {}).get('ms', 0)
            only_store = wall.get('w1_only_tma_store', {}).get('ms', 0)
            only_compute = wall.get('w1_only_compute', {}).get('ms', 0)
            if only_load and only_store and only_compute:
                sum_parts = (only_load - strip_ms) + (only_store - strip_ms) + (only_compute - strip_ms)
                print(f"\n  Additivity check:")
                print(f"    Sum of parts:  {sum_parts*1000:.0f}μs")
                print(f"    Full 1W:       {w1_overhead*1000:.0f}μs")
                ratio = sum_parts / w1_overhead if w1_overhead > 0 else 0
                print(f"    Ratio:         {ratio:.2f}x {'(additive)' if 0.8 < ratio < 1.2 else '(non-additive — contention between operations)'}")

        # Per-operation at 2 warps
        w2_full = wall.get('w2_full', {}).get('ms', 0)
        if w2_full and strip_ms:
            w2_overhead = w2_full - strip_ms
            print(f"\n{'─'*72}")
            print(f"  2-WARP DECOMPOSITION (total overhead: {w2_overhead*1000:.0f}μs)")
            print(f"{'─'*72}")
            w2_ops = [
                ('w2_full', 'Full epilogue'),
                ('w2_only_tma_load', 'TMA loads only'),
                ('w2_only_tma_store', 'TMA stores only'),
                ('w2_only_compute', 'STS+BF16 only'),
            ]
            print(f"  {'Operation':>20} {'ms':>8} {'Delta':>8} {'% of 2W':>8}")
            print(f"  {'─'*20:>20} {'─'*8:>8} {'─'*8:>8} {'─'*8:>8}")
            for label, desc in w2_ops:
                if label in wall:
                    ms = wall[label]['ms']
                    delta = ms - strip_ms
                    pct = delta / w2_overhead * 100 if w2_overhead > 0 else 0
                    print(f"  {desc:>20} {ms:>8.3f} {delta:>+8.3f} {pct:>7.1f}%")

        # Thermal drift
        strip_end = wall.get('strip_end', {}).get('ms', 0)
        if strip_end and strip_ms:
            drift = strip_end - strip_ms
            print(f"\n  Thermal drift: strip start={strip_ms:.3f}ms, end={strip_end:.3f}ms, delta={drift*1000:+.0f}μs")

    # ── NCU metrics ──
    ncu_dir = outdir / 'ncu'
    if ncu_dir.is_dir():
        ncu_files = sorted(ncu_dir.glob('ncu_*.csv'))
        if ncu_files:
            # Parse all
            all_ncu = {}
            for f in ncu_files:
                label = f.stem  # ncu_strip, ncu_w1, etc.
                metrics = parse_ncu_csv(f)
                if metrics:
                    all_ncu[label] = metrics

            if all_ncu:
                # ── Stall breakdown scaling ──
                print(f"\n{'─'*72}")
                print("  NCU STALL BREAKDOWN — WARP SCALING")
                print(f"{'─'*72}")

                stall_names = [
                    ('long_scoreboard', 'L2/Global'),
                    ('short_scoreboard', 'SMEM/Local'),
                    ('mio_throttle', 'MIO throttle'),
                    ('math_pipe_throttle', 'Math throttle'),
                    ('not_selected', 'Not selected'),
                    ('barrier', 'Barrier'),
                    ('membar', 'Membar'),
                    ('wait', 'Wait'),
                    ('drain', 'Drain'),
                    ('sleeping', 'Sleeping'),
                    ('misc', 'Misc'),
                ]

                ordered_labels = ['ncu_strip', 'ncu_w1', 'ncu_w2', 'ncu_w3', 'ncu_baseline']
                present = [l for l in ordered_labels if l in all_ncu]

                if present:
                    # Header
                    hdr = f"  {'Stall Reason':>18}"
                    for label in present:
                        short = label.replace('ncu_', '')
                        hdr += f" {short:>10}"
                    # Add delta columns
                    if 'ncu_strip' in all_ncu and 'ncu_baseline' in all_ncu:
                        hdr += f" {'Δ(b-s)':>10}"
                    print(hdr)
                    print(f"  {'─'*18}" + ''.join(f" {'─'*10}" for _ in present) +
                          (" " + "─"*10 if 'ncu_strip' in all_ncu and 'ncu_baseline' in all_ncu else ""))

                    for key_part, desc in stall_names:
                        metric_name = f"smsp__warps_issue_stalled_{key_part}_per_warp_active.pct"
                        row = f"  {desc:>18}"
                        vals = []
                        for label in present:
                            v = all_ncu[label].get(metric_name, None)
                            vals.append(v)
                            if v is not None:
                                row += f" {v:>9.1f}%"
                            else:
                                row += f" {'—':>10}"
                        # Delta
                        if len(vals) >= 2 and vals[0] is not None and vals[-1] is not None:
                            delta = vals[-1] - vals[0]
                            row += f" {delta:>+9.1f}%"
                        print(row)

                # ── Memory throughput scaling ──
                print(f"\n{'─'*72}")
                print("  NCU MEMORY THROUGHPUT — WARP SCALING")
                print(f"{'─'*72}")

                mem_names = [
                    ('lts__throughput.avg.pct_of_peak_sustained_elapsed', 'L2 throughput %'),
                    ('dram__throughput.avg.pct_of_peak_sustained_elapsed', 'DRAM throughput %'),
                    ('l1tex__throughput.avg.pct_of_peak_sustained_elapsed', 'L1 throughput %'),
                    ('lts__t_sectors_op_read.sum', 'L2 read sectors'),
                    ('lts__t_sectors_op_write.sum', 'L2 write sectors'),
                    ('lts__t_sectors.sum', 'L2 total sectors'),
                    ('dram__bytes_read.sum', 'DRAM read bytes'),
                    ('dram__bytes_write.sum', 'DRAM write bytes'),
                ]

                if present:
                    hdr = f"  {'Metric':>20}"
                    for label in present:
                        short = label.replace('ncu_', '')
                        hdr += f" {short:>14}"
                    print(hdr)
                    print(f"  {'─'*20}" + ''.join(f" {'─'*14}" for _ in present))

                    for metric_name, desc in mem_names:
                        row = f"  {desc:>20}"
                        for label in present:
                            v = all_ncu[label].get(metric_name, None)
                            if v is not None:
                                if v > 1e9:
                                    row += f" {v/1e9:>13.1f}G"
                                elif v > 1e6:
                                    row += f" {v/1e6:>13.1f}M"
                                elif v > 1e3:
                                    row += f" {v/1e3:>13.1f}K"
                                else:
                                    row += f" {v:>14.1f}"
                            else:
                                row += f" {'—':>14}"
                        print(row)

                # ── Pipe utilization scaling ──
                print(f"\n{'─'*72}")
                print("  NCU PIPE UTILIZATION — WARP SCALING")
                print(f"{'─'*72}")

                pipe_names = [
                    ('sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed', 'LSU pipe %'),
                    ('sm__inst_executed_pipe_tex.avg.pct_of_peak_sustained_elapsed', 'TEX pipe %'),
                    ('sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed', 'FMA pipe %'),
                    ('sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_elapsed', 'ALU pipe %'),
                    ('sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed', 'Shared pipe %'),
                    ('sm__throughput.avg.pct_of_peak_sustained_elapsed', 'SM throughput %'),
                    ('smsp__inst_executed.sum', 'Total insns'),
                    ('sm__mio_pq_read_cycles_active.avg.pct_of_peak_sustained_elapsed', 'MIO read %'),
                    ('sm__mio_pq_write_cycles_active.avg.pct_of_peak_sustained_elapsed', 'MIO write %'),
                ]

                if present:
                    hdr = f"  {'Metric':>18}"
                    for label in present:
                        short = label.replace('ncu_', '')
                        hdr += f" {short:>14}"
                    print(hdr)
                    print(f"  {'─'*18}" + ''.join(f" {'─'*14}" for _ in present))

                    for metric_name, desc in pipe_names:
                        row = f"  {desc:>18}"
                        for label in present:
                            v = all_ncu[label].get(metric_name, None)
                            if v is not None:
                                if v > 1e9:
                                    row += f" {v/1e9:>13.1f}G"
                                elif v > 1e6:
                                    row += f" {v/1e6:>13.1f}M"
                                else:
                                    row += f" {v:>14.1f}"
                            else:
                                row += f" {'—':>14}"
                        print(row)

                # ── Per-operation stall breakdown at 1 warp ──
                w1_ncu_labels = ['ncu_w1_only_tma_load', 'ncu_w1_only_tma_store', 'ncu_w1_only_compute']
                w1_present = [l for l in w1_ncu_labels if l in all_ncu]
                if w1_present:
                    # Add strip as reference
                    if 'ncu_strip' in all_ncu:
                        w1_present = ['ncu_strip'] + w1_present

                    print(f"\n{'─'*72}")
                    print("  NCU STALL BREAKDOWN — PER-OPERATION @ 1 WARP")
                    print(f"{'─'*72}")

                    hdr = f"  {'Stall Reason':>18}"
                    for label in w1_present:
                        short = label.replace('ncu_', '')
                        hdr += f" {short:>16}"
                    print(hdr)
                    print(f"  {'─'*18}" + ''.join(f" {'─'*16}" for _ in w1_present))

                    for key_part, desc in stall_names:
                        metric_name = f"smsp__warps_issue_stalled_{key_part}_per_warp_active.pct"
                        row = f"  {desc:>18}"
                        for label in w1_present:
                            v = all_ncu[label].get(metric_name, None)
                            if v is not None:
                                row += f" {v:>15.1f}%"
                            else:
                                row += f" {'—':>16}"
                        print(row)

                    # Memory + pipe for per-operation
                    print(f"\n  {'Metric':>18}", end="")
                    for label in w1_present:
                        short = label.replace('ncu_', '')
                        print(f" {short:>16}", end="")
                    print()
                    print(f"  {'─'*18}" + ''.join(f" {'─'*16}" for _ in w1_present))

                    for metric_name, desc in mem_names + pipe_names:
                        row = f"  {desc:>18}"
                        has_data = False
                        for label in w1_present:
                            v = all_ncu[label].get(metric_name, None)
                            if v is not None:
                                has_data = True
                                if v > 1e9:
                                    row += f" {v/1e9:>15.1f}G"
                                elif v > 1e6:
                                    row += f" {v/1e6:>15.1f}M"
                                else:
                                    row += f" {v:>16.1f}"
                            else:
                                row += f" {'—':>16}"
                        if has_data:
                            print(row)

    print(f"\n{'='*72}")
    print("  INTERPRETATION GUIDE")
    print(f"{'='*72}")
    print("""
  Wall time decomposition:
    - If sum of (only_tma_load + only_tma_store + only_compute) ≈ full 1W overhead:
      Operations contend INDEPENDENTLY for different resources.
    - If sum >> full: Operations share the same resource (double-counting).
    - If sum << full: Overhead comes from warp EXISTENCE, not specific operations.

  NCU stall scaling:
    - long_scoreboard increase = L2/DRAM bandwidth contention
    - short_scoreboard increase = SMEM port contention
    - mio_throttle increase = TMA unit saturation
    - not_selected increase = warp scheduler pressure (too many warps)
    - barrier increase = mbarrier/BAR.SYNC waits

  Look for: which metric increases LINEARLY from 0→4 warps at ~same rate as
  wall time? That's the dominant contention source.
""")


if __name__ == '__main__':
    main()
