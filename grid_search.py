#!/usr/bin/env python3
"""Compile-time grid search for megakernel.cu parameters.

Enumerates parameter combos, prunes invalid configs (SMEM budget, thread count),
compiles with -D flags, runs with timeout + hang detection, collects results
into a sorted table + CSV.

Usage:
    python3 grid_search.py --tier 1          # structure: N_STAGES x NUM_EPI_WARPS (~5 configs)
    python3 grid_search.py --tier 2          # epilogue: INTER x MBAR x STAG x TMEM (~128 configs)
    python3 grid_search.py --tier 3          # tuning: PHASE1_UNROLL x SNAKE_ORDER (~6 configs)
    python3 grid_search.py --tier all        # sequential 1->2->3, pinning winners
    python3 grid_search.py --full-cross      # all parameters crossed (~3000 configs)
    python3 grid_search.py --only N_STAGES STAGGER_CYCLES --fix MBAR_EARLY=1
"""

import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
import tempfile
import time

# ── Defaults (match megakernel.cu #ifndef values) ──
DEFAULTS = {
    'N_STAGES': 4,
    'NUM_EPI_WARPS': 4,
    'TMEM_LOAD_WIDTH': 32,
    'INTERLEAVE_STRATEGY': 2,
    'MBAR_EARLY': 0,
    'STAGGER_CYCLES': 80,
    'PHASE1_UNROLL': 2,
    'SNAKE_ORDER': 1,
    'CVT_ADD_FUSED': 1,
}

# ── Parameter ranges ──
RANGES = {
    'N_STAGES': [3, 4, 5],
    'NUM_EPI_WARPS': [4, 5],
    'TMEM_LOAD_WIDTH': [32, 64],
    'INTERLEAVE_STRATEGY': [0, 1, 2, 3],
    'MBAR_EARLY': [0, 1],
    'STAGGER_CYCLES': [0, 40, 60, 80, 100, 120, 160, 200],
    'PHASE1_UNROLL': [1, 2, 4],
    'SNAKE_ORDER': [0, 1],
    'CVT_ADD_FUSED': [0, 1],
}

# ── Tier definitions ──
TIER_PARAMS = {
    1: ['N_STAGES', 'NUM_EPI_WARPS'],
    2: ['INTERLEAVE_STRATEGY', 'MBAR_EARLY', 'STAGGER_CYCLES', 'TMEM_LOAD_WIDTH'],
    3: ['PHASE1_UNROLL', 'SNAKE_ORDER', 'CVT_ADD_FUSED'],
}

# ── Build config ──
NVCC = 'nvcc'
ARCH = 'sm_100a'
CFLAGS = f'-gencode arch=compute_100a,code={ARCH} -O3 --ptxas-options=-v'
LDFLAGS = '-lcurand -lcuda'
SRC = 'megakernel.cu'
COMPILE_TIMEOUT = 120
RUN_TIMEOUT = 30
SMEM_LIMIT = 233472  # 228 KB


def is_valid(cfg):
    """Pre-compile constraint check. Returns (valid, reason)."""
    n_stages = cfg['N_STAGES']
    num_epi = cfg['NUM_EPI_WARPS']

    # Thread count
    threads = 32 * (2 + num_epi)
    if threads > 1024:
        return False, f'threads {threads} > 1024'

    # Row group coverage
    if num_epi < 4:
        return False, 'NUM_EPI_WARPS < 4'

    # SMEM budget
    stage_bytes = 32768  # TK=128: 128*128 + 128*128 = 32KB
    off_tmem = n_stages * stage_bytes
    off_tma_mbar = off_tmem + 8
    off_mma_mbar = off_tma_mbar + n_stages * 8
    off_mainloop_mbar = off_mma_mbar + n_stages * 8
    off_epilogue_mbar = off_mainloop_mbar + 16
    off_staging = (off_epilogue_mbar + 16 + 127) & ~127
    staging_warp_bytes = 4 * 32 * 128  # 16384
    smem_total = (off_staging + num_epi * staging_warp_bytes + 127) & ~127
    if smem_total > SMEM_LIMIT:
        return False, f'SMEM {smem_total} > {SMEM_LIMIT}'

    return True, 'ok'


def smem_kb(cfg):
    """Compute SMEM usage in KB."""
    n_stages = cfg['N_STAGES']
    num_epi = cfg['NUM_EPI_WARPS']
    stage_bytes = 32768
    off_tmem = n_stages * stage_bytes
    off_tma_mbar = off_tmem + 8
    off_mma_mbar = off_tma_mbar + n_stages * 8
    off_mainloop_mbar = off_mma_mbar + n_stages * 8
    off_epilogue_mbar = off_mainloop_mbar + 16
    off_staging = (off_epilogue_mbar + 16 + 127) & ~127
    staging_warp_bytes = 4 * 32 * 128
    smem_total = (off_staging + num_epi * staging_warp_bytes + 127) & ~127
    return smem_total / 1024


def parse_ptxas(stderr):
    """Parse ptxas output for registers and spills."""
    regs = 0
    spills_store = 0
    spills_load = 0
    for line in stderr.splitlines():
        m = re.search(r'Used (\d+) registers', line)
        if m:
            regs = max(regs, int(m.group(1)))
        m = re.search(r'(\d+) bytes spill stores', line)
        if m:
            spills_store = max(spills_store, int(m.group(1)))
        m = re.search(r'(\d+) bytes spill loads', line)
        if m:
            spills_load = max(spills_load, int(m.group(1)))
    return regs, spills_store + spills_load


def parse_result(line):
    """Parse @@RESULT line. Returns (ms, tflops, checksum, valid, c0) or None."""
    m = re.match(r'@@RESULT ms=([\d.]+) tflops=([\d.]+) checksum=([\d.]+) valid=([01]) c0=([\d.]+)', line)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4)), float(m.group(5))


def make_dflags(cfg):
    """Build -D flags string, only for values that differ from defaults."""
    parts = []
    for k, v in sorted(cfg.items()):
        if v != DEFAULTS[k]:
            parts.append(f'-D{k}={v}')
    return ' '.join(parts)


def run_config(cfg, binary_path, repeat=1):
    """Compile and run a single config. Returns result dict."""
    dflags = make_dflags(cfg)
    cmd = f'{NVCC} {CFLAGS} {dflags} {SRC} -o {binary_path} {LDFLAGS}'

    result = {**cfg, 'status': 'UNKNOWN', 'ms': float('inf'), 'tflops': 0.0,
              'regs': 0, 'spills': 0, 'smem_kb': smem_kb(cfg), 'dflags': dflags}

    # Compile
    try:
        comp = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=COMPILE_TIMEOUT)
    except subprocess.TimeoutExpired:
        result['status'] = 'COMPILE_TIMEOUT'
        return result

    if comp.returncode != 0:
        result['status'] = 'COMPILE_FAIL'
        result['error'] = comp.stderr[-500:] if comp.stderr else ''
        return result

    regs, spills = parse_ptxas(comp.stderr)
    result['regs'] = regs
    result['spills'] = spills

    if regs > 255:
        result['status'] = 'SKIP_REGS'
        return result
    if spills > 0:
        result['status'] = 'SKIP_SPILLS'
        return result

    # Run (potentially multiple times)
    best_ms = float('inf')
    best_tflops = 0.0
    checksum = 0.0
    c0 = 0.0

    for rep in range(repeat):
        try:
            run = subprocess.run([binary_path], capture_output=True, text=True,
                                 timeout=RUN_TIMEOUT)
        except subprocess.TimeoutExpired:
            result['status'] = 'HANG'
            return result

        if run.returncode != 0:
            result['status'] = 'RUNTIME_ERROR'
            result['error'] = run.stderr[-500:] if run.stderr else ''
            return result

        # Parse @@RESULT
        parsed = None
        for line in run.stdout.splitlines():
            if line.startswith('@@RESULT'):
                parsed = parse_result(line)
                break

        if parsed is None:
            result['status'] = 'NO_RESULT'
            result['error'] = run.stdout[-500:]
            return result

        ms, tflops, cksum, valid_flag, c0_val = parsed

        # Binary self-validates against CPU reference (32 spot checks)
        if not valid_flag:
            result['status'] = 'VALIDATION_FAILED'
            result['ms'] = ms
            result['tflops'] = tflops
            return result

        if ms < best_ms:
            best_ms = ms
            best_tflops = tflops
        checksum = cksum
        c0 = c0_val

    result['status'] = 'OK'
    result['ms'] = best_ms
    result['tflops'] = best_tflops
    return result


def enumerate_configs(sweep_params, fixed):
    """Generate all configs from sweep_params × fixed values."""
    param_names = sorted(sweep_params.keys())
    param_values = [sweep_params[p] for p in param_names]

    for combo in itertools.product(*param_values):
        cfg = dict(fixed)
        cfg.update(zip(param_names, combo))
        yield cfg


def print_table(results, file=sys.stdout):
    """Print sorted results table."""
    # Sort by ms (ascending), with non-OK at the end
    ok_results = [r for r in results if r['status'] == 'OK']
    other_results = [r for r in results if r['status'] != 'OK']
    ok_results.sort(key=lambda r: r['ms'])

    if ok_results:
        best_ms = ok_results[0]['ms']
    else:
        best_ms = None

    header = (f'{"#":>3}  {"STG":>3}  {"EPI":>3}  {"INTER":>5}  {"MBAR":>4}  {"TMEM":>4}  '
              f'{"STAG":>4}  {"PH1U":>4}  {"SNAKE":>5}  {"FUSE":>4}  {"REGS":>4}  {"SMEM":>7}  '
              f'{"MS":>7}  {"TFLOPS":>7}  {"STATUS"}')
    print(header, file=file)
    print('-' * len(header), file=file)

    for i, r in enumerate(ok_results + other_results, 1):
        status = r['status']
        if status == 'OK' and best_ms is not None and r['ms'] == best_ms:
            status = 'BEST'

        ms_str = f'{r["ms"]:.3f}' if r['ms'] < float('inf') else '  -'
        tflops_str = f'{r["tflops"]:.0f}' if r['tflops'] > 0 else '  -'

        print(f'{i:>3}  {r["N_STAGES"]:>3}  {r["NUM_EPI_WARPS"]:>3}  '
              f'{r["INTERLEAVE_STRATEGY"]:>5}  {r["MBAR_EARLY"]:>4}  '
              f'{r["TMEM_LOAD_WIDTH"]:>4}  {r["STAGGER_CYCLES"]:>4}  '
              f'{r["PHASE1_UNROLL"]:>4}  {r["SNAKE_ORDER"]:>5}  '
              f'{r["CVT_ADD_FUSED"]:>4}  '
              f'{r["regs"]:>4}  {r["smem_kb"]:>6.1f}K  '
              f'{ms_str:>7}  {tflops_str:>7}  {status}',
              file=file)


def write_csv(results, path):
    """Write results to CSV."""
    ok_results = [r for r in results if r['status'] == 'OK']
    other_results = [r for r in results if r['status'] != 'OK']
    ok_results.sort(key=lambda r: r['ms'])
    all_sorted = ok_results + other_results

    fields = ['N_STAGES', 'NUM_EPI_WARPS', 'INTERLEAVE_STRATEGY', 'MBAR_EARLY',
              'TMEM_LOAD_WIDTH', 'STAGGER_CYCLES', 'PHASE1_UNROLL', 'SNAKE_ORDER',
              'CVT_ADD_FUSED', 'regs', 'spills', 'smem_kb', 'ms', 'tflops', 'status', 'dflags']

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in all_sorted:
            writer.writerow(r)


def run_sweep(sweep_params, fixed, repeat=1):
    """Run a sweep over sweep_params with fixed values pinned."""
    configs = list(enumerate_configs(sweep_params, fixed))

    # Pre-compile pruning
    valid = []
    pruned = 0
    for cfg in configs:
        ok, reason = is_valid(cfg)
        if ok:
            valid.append(cfg)
        else:
            pruned += 1

    total = len(configs)
    print(f'Sweep: {total} total, {len(valid)} valid, {pruned} pruned')
    if not valid:
        print('No valid configs!')
        return []

    # Estimate time
    est_min = len(valid) * 5 / 60  # ~5s compile+run per config
    print(f'Estimated time: ~{est_min:.0f} min ({len(valid)} configs × ~5s each)')
    print()

    results = []
    binary_path = os.path.join(tempfile.gettempdir(), 'sweep_bin')

    for i, cfg in enumerate(valid):
        dflags = make_dflags(cfg)
        label = dflags if dflags else '(defaults)'
        print(f'[{i+1}/{len(valid)}] {label} ...', end=' ', flush=True)

        t0 = time.time()
        result = run_config(cfg, binary_path, repeat=repeat)
        dt = time.time() - t0

        if result['status'] == 'OK':
            print(f'{result["ms"]:.3f} ms / {result["tflops"]:.0f} TFLOPS  '
                  f'({result["regs"]} regs, {dt:.1f}s)')
        else:
            print(f'{result["status"]}  ({dt:.1f}s)')

        results.append(result)

    # Cleanup
    if os.path.exists(binary_path):
        os.unlink(binary_path)

    return results


def get_best(results):
    """Return the best OK config, or None."""
    ok = [r for r in results if r['status'] == 'OK']
    if not ok:
        return None
    return min(ok, key=lambda r: r['ms'])


def main():
    parser = argparse.ArgumentParser(description='Grid search for megakernel.cu parameters')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--tier', choices=['1', '2', '3', 'all'],
                      help='Tiered search (1=structure, 2=epilogue, 3=tuning, all=sequential)')
    mode.add_argument('--full-cross', action='store_true',
                      help='Full cross-product of all parameters')
    parser.add_argument('--fix', nargs='*', default=[], metavar='PARAM=VAL',
                        help='Pin specific parameters (e.g. --fix MBAR_EARLY=1 N_STAGES=4)')
    parser.add_argument('--only', nargs='*', default=[], metavar='PARAM',
                        help='Sweep only these parameters, pin rest at defaults')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Run each config N times, report best (default: 1)')
    parser.add_argument('--csv', default='sweep_results.csv',
                        help='Output CSV path (default: sweep_results.csv)')
    args = parser.parse_args()

    # Parse --fix
    fixed_overrides = {}
    for f in args.fix:
        if '=' not in f:
            print(f'Error: --fix arg must be PARAM=VAL, got: {f}', file=sys.stderr)
            sys.exit(1)
        k, v = f.split('=', 1)
        if k not in DEFAULTS:
            print(f'Error: unknown parameter: {k}', file=sys.stderr)
            sys.exit(1)
        fixed_overrides[k] = int(v)

    # Validate --only
    for p in args.only:
        if p not in RANGES:
            print(f'Error: unknown parameter: {p}', file=sys.stderr)
            sys.exit(1)

    all_results = []

    if args.only:
        # Custom sweep: only specified params
        sweep_params = {p: RANGES[p] for p in args.only}
        fixed = dict(DEFAULTS)
        fixed.update(fixed_overrides)
        for p in args.only:
            fixed.pop(p, None)

        print(f'=== Custom sweep: {", ".join(args.only)} ===')
        results = run_sweep(sweep_params, fixed, repeat=args.repeat)
        all_results.extend(results)

    elif args.full_cross:
        # Full cross-product
        sweep_params = dict(RANGES)
        fixed = dict(fixed_overrides)
        # Remove swept params from fixed
        for p in sweep_params:
            fixed.pop(p, None)

        print('=== Full cross-product sweep ===')
        results = run_sweep(sweep_params, fixed, repeat=args.repeat)
        all_results.extend(results)

    elif args.tier == 'all':
        # Sequential tiers, pinning winners
        winners = dict(DEFAULTS)
        winners.update(fixed_overrides)

        for tier_num in [1, 2, 3]:
            tier_params = TIER_PARAMS[tier_num]
            sweep_params = {p: RANGES[p] for p in tier_params if p not in fixed_overrides}
            if not sweep_params:
                print(f'\n=== Tier {tier_num}: all params pinned by --fix, skipping ===')
                continue

            fixed = {k: v for k, v in winners.items() if k not in sweep_params}

            print(f'\n=== Tier {tier_num}: {", ".join(sweep_params.keys())} ===')
            results = run_sweep(sweep_params, fixed, repeat=args.repeat)
            all_results.extend(results)

            best = get_best(results)
            if best:
                for p in tier_params:
                    winners[p] = best[p]
                print(f'\nTier {tier_num} winner: {make_dflags(best) or "(defaults)"} '
                      f'→ {best["ms"]:.3f} ms / {best["tflops"]:.0f} TFLOPS')
            else:
                print(f'\nTier {tier_num}: no valid results!')

    else:
        # Single tier
        tier_num = int(args.tier)
        tier_params = TIER_PARAMS[tier_num]
        sweep_params = {p: RANGES[p] for p in tier_params if p not in fixed_overrides}
        fixed = dict(DEFAULTS)
        fixed.update(fixed_overrides)
        for p in sweep_params:
            fixed.pop(p, None)

        print(f'=== Tier {tier_num}: {", ".join(sweep_params.keys())} ===')
        results = run_sweep(sweep_params, fixed, repeat=args.repeat)
        all_results.extend(results)

    # Summary
    if all_results:
        print('\n' + '=' * 80)
        print('RESULTS (sorted by ms)')
        print('=' * 80)
        print_table(all_results)

        write_csv(all_results, args.csv)
        print(f'\nCSV written to {args.csv}')

        best = get_best(all_results)
        if best:
            dflags = make_dflags(best)
            print(f'\nBest: {dflags or "(defaults)"} → {best["ms"]:.3f} ms / '
                  f'{best["tflops"]:.0f} TFLOPS ({best["regs"]} regs)')


if __name__ == '__main__':
    main()
