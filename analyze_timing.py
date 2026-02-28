#!/usr/bin/env python3
"""Analyze clock64 timing output from the SigLIP megakernel timing build.

Parses the kernel's printf output and computes derived metrics:
equilibrium analysis, ceiling projections, bottleneck identification.

Usage:
    ./siglip_timing 2>&1 | python3 analyze_timing.py
    python3 analyze_timing.py clock64_timing.txt
    python3 analyze_timing.py clock64_timing.txt --cublas 3001  # override cuBLAS TFLOPS
"""
import sys
import re


def parse_timing(lines):
    """Parse kernel printf output into a dict of metrics."""
    d = {}

    for line in lines:
        line = line.strip()

        # Kernel perf line: "Custom kernel: 0.533 ms  2052.90 TFLOPS"
        m = re.search(r'Custom kernel:\s+([\d.]+)\s+ms\s+([\d.]+)\s+TFLOPS', line)
        if m:
            d['kernel_ms'] = float(m.group(1))
            d['kernel_tflops'] = float(m.group(2))

        # Tile count: "=== W1 TIMING (clock64, 10878 tiles across 74 clusters) ==="
        m = re.search(r'clock64,\s+(\d+)\s+tiles\s+across\s+(\d+)\s+clusters', line)
        if m:
            d['tiles'] = int(m.group(1))
            d['clusters'] = int(m.group(2))

        # W1 metrics: "    Epilogue mbar wait:     1056 cycles /  502.9 ns"
        m = re.search(r'Epilogue mbar wait:\s+(\d+)\s+cycles', line)
        if m:
            d['w1_epi_wait'] = int(m.group(1))

        m = re.search(r'TMA stage-0 wait:\s+(\d+)\s+cycles', line)
        if m:
            d['w1_tma0_wait'] = int(m.group(1))

        m = re.search(r'K-loop.*?:\s+(\d+)\s+cycles', line)
        if m and 'w1_kloop' not in d:  # first match only
            d['w1_kloop'] = int(m.group(1))

        m = re.search(r'Total tile:\s+(\d+)\s+cycles', line)
        if m:
            d['w1_total'] = int(m.group(1))

        # W1 ranges
        m = re.search(r'K-loop range:\s+min=(\d+)\s+max=(\d+)', line)
        if m:
            d['w1_kloop_min'] = int(m.group(1))
            d['w1_kloop_max'] = int(m.group(2))

        m = re.search(r'Total tile range:\s+min=(\d+)\s+max=(\d+)', line)
        if m:
            d['w1_total_min'] = int(m.group(1))
            d['w1_total_max'] = int(m.group(2))

        # Wall clock cycles
        m = re.search(r'Expected total cycles.*?:\s+([\d.]+)', line)
        if m:
            d['wall_cycles'] = float(m.group(1))

        # Epilogue tile count
        m = re.search(r'W3/ew=1,\s+(\d+)\s+tiles', line)
        if m:
            d['epi_tiles'] = int(m.group(1))

        # Epilogue metrics
        m = re.search(r'Mainloop mbar wait:\s+(\d+)\s+cycles', line)
        if m:
            d['epi_ml_wait'] = int(m.group(1))

        m = re.search(r'Phase 1.*?:\s+(\d+)\s+cycles', line)
        if m and 'epi_phase1' not in d:
            d['epi_phase1'] = int(m.group(1))

        m = re.search(r'Phase 2.*?:\s+(\d+)\s+cycles', line)
        if m and 'epi_phase2' not in d:
            d['epi_phase2'] = int(m.group(1))

        m = re.search(r'Work only.*?:\s+(\d+)\s+cycles', line)
        if m:
            d['epi_work'] = int(m.group(1))

        # Epilogue ranges
        m = re.search(r'Mainloop wait range:\s+min=(\d+)\s+max=(\d+)', line)
        if m:
            d['epi_ml_min'] = int(m.group(1))
            d['epi_ml_max'] = int(m.group(2))

        m = re.search(r'Phase 1 range:\s+min=(\d+)\s+max=(\d+)', line)
        if m:
            d['epi_p1_min'] = int(m.group(1))
            d['epi_p1_max'] = int(m.group(2))

        m = re.search(r'Phase 2 range:\s+min=(\d+)\s+max=(\d+)', line)
        if m:
            d['epi_p2_min'] = int(m.group(1))
            d['epi_p2_max'] = int(m.group(2))

        # Warp/epi config from banner
        m = re.search(r'(\d+)\s+warps\s+\[(\d+)\s+epi\]', line)
        if m:
            d['total_warps'] = int(m.group(1))
            d['epi_warps'] = int(m.group(2))

        # GEMM dimensions
        m = re.search(r'GEMM:\s+\[(\d+),(\d+)\]\s*x\s*\[(\d+),(\d+)\]', line)
        if m:
            d['M'] = int(m.group(1))
            d['K'] = int(m.group(2))
            d['N'] = int(m.group(4))

    return d


def analyze(d, cublas_tflops=3001):
    """Print derived analysis from parsed timing data."""

    # Check we have the minimum data
    required = ['w1_epi_wait', 'w1_tma0_wait', 'w1_kloop', 'w1_total',
                'epi_ml_wait', 'epi_phase1', 'epi_phase2']
    missing = [k for k in required if k not in d]
    if missing:
        print(f"ERROR: Missing fields: {missing}", file=sys.stderr)
        print("Is this a valid timing build output?", file=sys.stderr)
        sys.exit(1)

    ms = d.get('kernel_ms', 0)
    tflops = d.get('kernel_tflops', 0)
    tiles = d.get('tiles', 10878)

    w1_epi = d['w1_epi_wait']
    w1_tma = d['w1_tma0_wait']
    w1_kloop = d['w1_kloop']
    w1_total = d['w1_total']
    w1_productive = w1_tma + w1_kloop

    epi_ml = d['epi_ml_wait']
    epi_p1 = d['epi_phase1']
    epi_p2 = d['epi_phase2']
    epi_work = d.get('epi_work', epi_p1 + epi_p2)
    epi_total = epi_ml + epi_p1 + epi_p2

    # "Effective epilogue" = what W1 cares about: ml_wait + Phase 1
    # Phase 2B is overlapped (runs after early mbar arrive)
    # NOTE: syncwarp overhead before early mbar arrive is NOT included — unmeasured.
    # F25 should instrument this. If significant (50-100 cyc), deficit and all
    # projections shift. Current model assumes syncwarp ≈ 0 (conservative).
    epi_effective = epi_ml + epi_p1
    deficit = epi_effective - w1_productive

    print("=" * 68)
    print("  clock64 Timing Analysis")
    print("=" * 68)
    if ms > 0:
        print(f"  {ms:.3f} ms / {tflops:.0f} TFLOPS (timing build)")
    print()

    # --- Cycle budget ---
    print("--- CYCLE BUDGET (per tile) ---")
    print(f"  W1:       epi_wait({w1_epi:,}) + TMA0({w1_tma:,}) + K-loop({w1_kloop:,}) = {w1_total:,}")
    print(f"  Epilogue: ml_wait({epi_ml:,}) + Phase1({epi_p1:,}) + Phase2B({epi_p2:,}) = {epi_total:,}")
    print(f"  W1 productive (TMA0+K-loop): {w1_productive:,}")
    print(f"  Epilogue work (P1+P2B):      {epi_work:,}")
    print()

    # --- Equilibrium ---
    print("--- EQUILIBRIUM ---")
    print(f"  Effective epilogue (ml_wait + Phase1):        {epi_effective:,}")
    print(f"  W1 productive (TMA0 + K-loop):               {w1_productive:,}")
    if deficit > 0:
        print(f"  Deficit: {deficit:,} cycles (epilogue {deficit} slower → amplifies to {w1_epi:,} epi_wait)")
        print(f"  Amplification factor: {w1_epi/deficit:.1f}x (double-buffer lag)")
        status = "EPILOGUE-BOUND"
    elif deficit < 0:
        print(f"  Surplus: {-deficit:,} cycles (W1 slower → epilogue waits more)")
        status = "COMPUTE-BOUND"
    else:
        print(f"  Perfectly balanced")
        status = "BALANCED"
    print(f"  Status: {status}")
    print()

    # --- Overhead decomposition ---
    print("--- OVERHEAD DECOMPOSITION ---")
    w1_overhead = w1_epi + w1_tma
    pct_epi = 100.0 * w1_epi / w1_total
    pct_tma = 100.0 * w1_tma / w1_total
    pct_kloop = 100.0 * w1_kloop / w1_total
    print(f"  W1 tile = {w1_total:,} cycles:")
    print(f"    K-loop:   {w1_kloop:>6,} ({pct_kloop:4.1f}%) — productive MMA + overhead")
    print(f"    epi_wait: {w1_epi:>6,} ({pct_epi:4.1f}%) — blocked on TMEM buffer")
    print(f"    TMA0:     {w1_tma:>6,} ({pct_tma:4.1f}%) — DRAM latency for A matrix")
    print(f"  Epilogue tile = {epi_total:,} cycles:")
    pct_ml = 100.0 * epi_ml / epi_total if epi_total else 0
    pct_p1 = 100.0 * epi_p1 / epi_total if epi_total else 0
    pct_p2 = 100.0 * epi_p2 / epi_total if epi_total else 0
    print(f"    Phase 1:  {epi_p1:>6,} ({pct_p1:4.1f}%) — TMEM readback + BF16 add + CVT → SMEM")
    print(f"    ml_wait:  {epi_ml:>6,} ({pct_ml:4.1f}%) — waiting for W1 to commit")
    print(f"    Phase 2B: {epi_p2:>6,} ({pct_p2:4.1f}%) — SMEM → global (overlapped with K-loop)")
    print()

    # --- Ceiling projections ---
    if tflops > 0:
        # Use production TFLOPS if available, fall back to timing build
        base_tflops = tflops

        print("--- CEILING PROJECTIONS ---")
        # 1. Eliminate epi_wait
        tile_no_epi = w1_productive
        speedup_no_epi = w1_total / tile_no_epi
        tflops_no_epi = base_tflops * speedup_no_epi
        print(f"  Eliminate epi_wait:        {tile_no_epi:,} cyc/tile  {speedup_no_epi:.2f}x  → {tflops_no_epi:.0f} TFLOPS")

        # 2. Also eliminate TMA0
        tile_no_both = w1_kloop
        speedup_no_both = w1_total / tile_no_both
        tflops_no_both = base_tflops * speedup_no_both
        print(f"  Also eliminate TMA0_wait:  {tile_no_both:,} cyc/tile  {speedup_no_both:.2f}x  → {tflops_no_both:.0f} TFLOPS")

        # 3. Gap to cuBLAS
        gap = cublas_tflops / base_tflops
        print(f"  cuBLAS reference:          {cublas_tflops} TFLOPS  (gap: {gap:.2f}x)")
        remaining = cublas_tflops / tflops_no_both if tflops_no_both > 0 else 999
        print(f"  After eliminating all overhead: {remaining:.2f}x remaining gap (K-loop efficiency)")
        print()

        # --- What-if scenarios ---
        print("--- WHAT-IF SCENARIOS ---")

        # F22: BF16 epilogue arithmetic (save 100-300 cycles from Phase 1)
        for save in [100, 200, 300]:
            new_p1 = epi_p1 - save
            new_epi_effective = epi_ml + new_p1
            new_deficit = new_epi_effective - w1_productive
            if new_deficit > 0:
                # Still epilogue-bound, estimate new epi_wait
                # Rough: epi_wait scales with deficit (amplification ~constant)
                amp = w1_epi / deficit if deficit > 0 else 3.0
                new_epi_wait = max(0, int(new_deficit * amp))
                new_tile = new_epi_wait + w1_productive
            else:
                new_epi_wait = 0
                new_tile = w1_productive
            speedup = w1_total / new_tile if new_tile > 0 else 1
            proj = base_tflops * speedup
            label = f"Phase1 -{save}cyc"
            print(f"  {label}: epi_wait→{new_epi_wait:,}, tile→{new_tile:,}, {speedup:.2f}x → {proj:.0f} TFLOPS")

        # F28: K-loop savings (alone, then paired with F22)
        for ksave in [150, 300]:
            # Alone: deficit grows
            new_kloop = w1_kloop - ksave
            new_productive = w1_tma + new_kloop
            new_deficit_alone = epi_effective - new_productive
            if new_deficit_alone > 0:
                amp = w1_epi / deficit if deficit > 0 else 3.0
                new_epi_wait = int(new_deficit_alone * amp)
                new_tile = new_epi_wait + new_productive
            else:
                new_epi_wait = 0
                new_tile = new_productive
            speedup = w1_total / new_tile if new_tile > 0 else 1
            proj = base_tflops * speedup
            print(f"  K-loop -{ksave}cyc alone: epi_wait→{new_epi_wait:,}, tile→{new_tile:,}, {speedup:.2f}x → {proj:.0f} TFLOPS")

        # Paired: F22 (-200) + F28 (-300)
        for p1_save, k_save in [(200, 150), (200, 300), (300, 300)]:
            new_p1 = epi_p1 - p1_save
            new_kloop = w1_kloop - k_save
            new_productive = w1_tma + new_kloop
            new_epi_eff = epi_ml + new_p1
            new_def = new_epi_eff - new_productive
            if new_def > 0:
                amp = w1_epi / deficit if deficit > 0 else 3.0
                new_epi_wait = int(new_def * amp)
                new_tile = new_epi_wait + new_productive
            else:
                new_epi_wait = 0
                new_tile = new_productive
            speedup = w1_total / new_tile if new_tile > 0 else 1
            proj = base_tflops * speedup
            print(f"  F22(-{p1_save})+F28(-{k_save}): epi_wait→{new_epi_wait:,}, tile→{new_tile:,}, {speedup:.2f}x → {proj:.0f} TFLOPS")

        print(f"  (Amplification model: epi_wait ≈ deficit × {w1_epi/deficit:.1f}. Approximate for small perturbations.)")
        print()

    # --- Jitter ---
    print("--- JITTER ---")
    if 'w1_kloop_min' in d:
        spread = d['w1_kloop_max'] / d['w1_kloop_min'] if d['w1_kloop_min'] > 0 else 0
        print(f"  K-loop:  min={d['w1_kloop_min']:,}  max={d['w1_kloop_max']:,}  ({spread:.1f}x)")
    if 'w1_total_min' in d:
        spread = d['w1_total_max'] / d['w1_total_min'] if d['w1_total_min'] > 0 else 0
        print(f"  Total:   min={d['w1_total_min']:,}  max={d['w1_total_max']:,}  ({spread:.1f}x)")
    if 'epi_ml_min' in d:
        spread = d['epi_ml_max'] / d['epi_ml_min'] if d['epi_ml_min'] > 0 else 0
        print(f"  ml_wait: min={d['epi_ml_min']:,}  max={d['epi_ml_max']:,}  ({spread:.1f}x)")
    if 'epi_p1_min' in d:
        spread = d['epi_p1_max'] / d['epi_p1_min'] if d['epi_p1_min'] > 0 else 0
        print(f"  Phase1:  min={d['epi_p1_min']:,}  max={d['epi_p1_max']:,}  ({spread:.1f}x)")
    print()

    # --- Comparison with production build ---
    if 'wall_cycles' in d and tiles > 0:
        expected_cycles = tiles * w1_total
        wall = d['wall_cycles']
        overhead = 100.0 * (wall - expected_cycles) / wall if wall > 0 else 0
        print("--- TIMING BUILD vs WALL CLOCK ---")
        print(f"  Sum of tile cycles (tiles × avg): {expected_cycles:,.0f}")
        print(f"  Wall clock cycles:                {wall:,.0f}")
        if expected_cycles > wall:
            print(f"  Tiles overlap (persistent kernel): {100*(expected_cycles-wall)/expected_cycles:.1f}% overlap")
        else:
            print(f"  Non-tile overhead: {overhead:.1f}% (setup, drain, cluster sync)")
        print()


if __name__ == '__main__':
    cublas = 3001

    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        with open(sys.argv[1]) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    for i, arg in enumerate(sys.argv):
        if arg == '--cublas' and i + 1 < len(sys.argv):
            cublas = float(sys.argv[i + 1])

    d = parse_timing(lines)
    analyze(d, cublas_tflops=cublas)
