#!/usr/bin/env python3
"""
Analyze warp-scaling benchmark output.

Reads the stdout from ./warp-scaling, parses per-warp cycle counts,
and produces:
  - S-tests: scaling curves (throughput vs warp count, normalized to 1-warp)
  - X-tests: cross-pipe independence (overhead vs single-pipe baselines)
  - F-tests: foreground/background interference
  - B-tests: BAR.SYNC effect on compute warp throughput

Usage:
  ./warp-scaling > data/warp_scaling.txt
  python3 tools/analyze_warp_scaling.py data/warp_scaling.txt
  python3 tools/analyze_warp_scaling.py data/warp_scaling.txt --csv data/warp_scaling.csv
"""

import argparse
import re
import sys
from collections import defaultdict


def parse_output(lines):
    """Parse warp-scaling benchmark output into structured data.

    Returns list of dicts: {name, category, n_warps, warp_cpis: [float], max_cpi, min_cpi}
    """
    results = []
    for line in lines:
        line = line.rstrip()
        # Match data lines: "name  N  W0_cpi  W1_cpi ... max  min"
        # Skip headers and section markers
        if not line or line.startswith('=') or line.startswith('Warp') or \
           line.startswith('REPS') or line.startswith('@@') or \
           'Test' in line and 'Warps' in line:
            continue
        if line.startswith('---'):
            continue

        # Try to parse as a data line
        parts = line.split()
        if len(parts) < 4:
            continue

        name = parts[0]
        try:
            n_warps = int(parts[1])
        except ValueError:
            continue

        # Parse per-warp CPIs (skip '-' entries)
        cpis = []
        for p in parts[2:2+n_warps]:
            try:
                cpis.append(float(p))
            except ValueError:
                cpis.append(None)

        # Parse max/min (last 2 numeric entries)
        try:
            max_cpi = float(parts[-2])
            min_cpi = float(parts[-1])
        except (ValueError, IndexError):
            max_cpi = max(c for c in cpis if c is not None) if cpis else 0
            min_cpi = min(c for c in cpis if c is not None) if cpis else 0

        # Determine category from name prefix
        if name.startswith('S_'):
            cat = 'S'
        elif name.startswith('X_'):
            cat = 'X'
        elif name.startswith('F_'):
            cat = 'F'
        elif name.startswith('P_'):
            cat = 'P'
        elif name.startswith('B_'):
            cat = 'B'
        elif name.startswith('N_'):
            cat = 'N'
        elif name.startswith('A_'):
            cat = 'A'
        else:
            cat = '?'

        results.append({
            'name': name,
            'category': cat,
            'n_warps': n_warps,
            'warp_cpis': cpis,
            'max_cpi': max_cpi,
            'min_cpi': min_cpi,
        })

    return results


def analyze_scaling(results):
    """S-tests: show throughput scaling per pipe."""
    s_tests = [r for r in results if r['category'] == 'S']
    if not s_tests:
        return

    # Group by pipe
    pipes = defaultdict(list)
    for r in s_tests:
        # Name format: S_{PIPE}_w{N}
        m = re.match(r'S_(\w+)_w(\d+)', r['name'])
        if m:
            pipe = m.group(1)
            pipes[pipe].append(r)

    print('\n' + '='*80)
    print('S-TESTS: Same-pipe throughput scaling')
    print('='*80)

    for pipe in sorted(pipes.keys()):
        tests = sorted(pipes[pipe], key=lambda r: r['n_warps'])
        baseline = tests[0]['max_cpi'] if tests else 1

        print(f'\n  {pipe}:')
        print(f'    {"Warps":>5s}  {"Max CPI":>8s}  {"Slowdown":>8s}  {"Per-warp TP":>11s}  Bar')

        for t in tests:
            nw = t['n_warps']
            cpi = t['max_cpi']
            slowdown = cpi / baseline if baseline > 0 else 0
            tp = 1.0 / cpi if cpi > 0 else 0
            bar_len = int(slowdown * 20)
            bar = '#' * min(bar_len, 60)
            print(f'    {nw:5d}  {cpi:8.2f}  {slowdown:7.2f}x  {tp:11.4f}  {bar}')

        # Key metric: is throughput per-warp or per-SM?
        if len(tests) >= 2:
            w1 = tests[0]['max_cpi']
            w8 = tests[-1]['max_cpi']
            ratio = w8 / w1 if w1 > 0 else 0
            if ratio > 6:
                verdict = 'SHARED (per-SM pipe)'
            elif ratio > 3:
                verdict = 'PARTIALLY SHARED'
            elif ratio > 1.5:
                verdict = 'SUB-PARTITIONED'
            else:
                verdict = 'PER-WARP (independent)'
            print(f'    → 8w/1w ratio = {ratio:.1f}x → {verdict}')


def analyze_cross_pipe(results):
    """X-tests: cross-pipe independence."""
    x_tests = [r for r in results if r['category'] == 'X']
    if not x_tests:
        return

    # Get single-warp baselines from S-tests
    s_tests = [r for r in results if r['category'] == 'S']
    baselines = {}
    for r in s_tests:
        m = re.match(r'S_(\w+)_w1', r['name'])
        if m:
            baselines[m.group(1)] = r['max_cpi']

    print('\n' + '='*80)
    print('X-TESTS: Cross-pipe independence')
    print('='*80)

    for t in x_tests:
        print(f'\n  {t["name"]} ({t["n_warps"]} warps):')
        for w in range(t['n_warps']):
            cpi = t['warp_cpis'][w] if w < len(t['warp_cpis']) else None
            if cpi is None:
                continue
            print(f'    W{w}: {cpi:.2f} cyc/insn', end='')
            # TODO: we'd need to know which pipe each warp was on to compare
            # to baseline. For now just print raw.
            print()


def analyze_fg_bg(results):
    """F-tests: foreground/background interference."""
    f_tests = [r for r in results if r['category'] == 'F']
    if not f_tests:
        return

    print('\n' + '='*80)
    print('F-TESTS: Foreground/Background interference')
    print('='*80)

    for t in f_tests:
        print(f'\n  {t["name"]} ({t["n_warps"]} warps):')
        for w in range(t['n_warps']):
            cpi = t['warp_cpis'][w] if w < len(t['warp_cpis']) else None
            if cpi is None:
                continue
            print(f'    W{w}: {cpi:.2f} cyc/insn')
        spread = t['max_cpi'] - t['min_cpi']
        print(f'    Spread: {spread:.2f} cyc/insn ({spread/t["min_cpi"]*100:.1f}%)')


def analyze_barsync(results):
    """B-tests: BAR.SYNC effect on compute warps."""
    b_tests = [r for r in results if r['category'] == 'B']
    if not b_tests:
        return

    print('\n' + '='*80)
    print('B-TESTS: BAR.SYNC effect on compute warp throughput')
    print('='*80)
    print('  Key question: do synchronized idle gaps in "epi" warps')
    print('  improve throughput of "kloop" compute warps?')

    # Find baseline (no bar) and bar variants
    baseline = None
    bar_tests = []
    for t in b_tests:
        if 'nobar' in t['name']:
            baseline = t
        else:
            bar_tests.append(t)

    if baseline:
        # Compute warp CPIs are the LAST n_compute warps
        # In our gen: bar warps are first, compute warps are last
        m = re.match(r'B_(\d+)(?:no)?bar_(\d+)cmp', baseline['name'])
        if m:
            n_bar = int(m.group(1))
            n_cmp = int(m.group(2))
            base_cmp_cpis = baseline['warp_cpis'][n_bar:n_bar+n_cmp]
            base_cmp_avg = sum(c for c in base_cmp_cpis if c) / max(len([c for c in base_cmp_cpis if c]), 1)

            print(f'\n  Baseline (no BAR.SYNC): compute warp avg = {base_cmp_avg:.2f} cyc/insn')

            for t in sorted(bar_tests, key=lambda x: x['name']):
                m2 = re.match(r'B_(\d+)bar_(\d+)cmp_i(\d+)', t['name'])
                if m2:
                    nb = int(m2.group(1))
                    nc = int(m2.group(2))
                    interval = int(m2.group(3))
                    cmp_cpis = t['warp_cpis'][nb:nb+nc]
                    cmp_avg = sum(c for c in cmp_cpis if c) / max(len([c for c in cmp_cpis if c]), 1)
                    delta = (cmp_avg - base_cmp_avg) / base_cmp_avg * 100
                    arrow = '↑ WORSE' if delta > 2 else '↓ BETTER' if delta < -2 else '≈ NOISE'
                    print(f'  {t["name"]}: compute avg = {cmp_avg:.2f} cyc/insn '
                          f'({delta:+.1f}% vs baseline) {arrow}')

    print()
    for t in b_tests + ([baseline] if baseline else []):
        print(f'\n  {t["name"]}:')
        for w in range(t['n_warps']):
            cpi = t['warp_cpis'][w] if w < len(t['warp_cpis']) else None
            if cpi is None:
                continue
            print(f'    W{w}: {cpi:.2f} cyc/insn')


def analyze_prodcons(results):
    """P-tests: producer-consumer (CUTLASS W3 pattern) comparison."""
    p_tests = [r for r in results if r['category'] == 'P']
    if not p_tests:
        return

    print('\n' + '='*80)
    print('P-TESTS: Producer-consumer (CUTLASS W3 architecture)')
    print('='*80)
    print('  Compares: LDG self-load (our arch) vs LDS pre-loaded (CUTLASS)')
    print('  Load warp (lw): dedicated warp that pre-loads then idles (W3 sim)')
    print()

    # Group by scenario (n_kloop)
    groups = defaultdict(list)
    for t in p_tests:
        name = t['name']
        # Extract kloop count from name
        if '_2kl' in name:
            kl = 2
        elif '_1kl' in name:
            kl = 1
        else:
            kl = 0
        groups[kl].append(t)

    for kl in sorted(groups.keys()):
        tests = groups[kl]
        print(f'  --- {kl} K-loop warps ---')
        for t in tests:
            n_epi = sum(1 for c in t['warp_cpis'][:4] if c and c > 0)
            epi_cpis = [c for c in t['warp_cpis'][:4] if c and c > 0]
            epi_avg = sum(epi_cpis) / len(epi_cpis) if epi_cpis else 0

            has_lw = '_lw' in t['name']
            is_ldg = '_ldg' in t['name']

            # K-loop warp CPIs are at the end
            kl_cpis = []
            if kl > 0:
                kl_start = n_epi + (1 if has_lw else 0)
                kl_cpis = [c for c in t['warp_cpis'][kl_start:kl_start+kl] if c and c > 0]
            kl_avg = sum(kl_cpis) / len(kl_cpis) if kl_cpis else 0

            mode = 'LDG(self)' if is_ldg else 'LDS(smem)'
            lw_str = '+LW' if has_lw else '    '
            kl_str = f'  kloop={kl_avg:.1f}' if kl_cpis else ''
            print(f'    {t["name"]:<28s} {mode} {lw_str}  epi_avg={epi_avg:7.2f}{kl_str}')

        # Key comparisons within this group
        ldg = [t for t in tests if '_ldg' in t['name'] and '_lw' not in t['name']]
        lds = [t for t in tests if '_lds' in t['name'] and '_lw' not in t['name']]
        lds_lw = [t for t in tests if '_lds' in t['name'] and '_lw' in t['name']]

        if ldg and lds:
            ldg_max = ldg[0]['max_cpi']
            lds_max = lds[0]['max_cpi']
            delta = (ldg_max - lds_max) / lds_max * 100 if lds_max > 0 else 0
            print(f'    → LDG vs LDS: {delta:+.1f}% ({"LDG slower" if delta > 0 else "LDS slower"})')
        if lds and lds_lw:
            lds_max = lds[0]['max_cpi']
            lw_max = lds_lw[0]['max_cpi']
            delta = (lw_max - lds_max) / lds_max * 100 if lds_max > 0 else 0
            print(f'    → Load warp effect: {delta:+.1f}% ({"WORSE" if delta > 0 else "BETTER — idle warp helps"})')
        print()


def analyze_nanosleep(results):
    """N-tests: nanosleep actual duration calibration."""
    n_tests = [r for r in results if r['category'] == 'N']
    if not n_tests:
        return

    print('\n' + '='*80)
    print('N-TESTS: Nanosleep calibration')
    print('='*80)
    print('  Measures actual sleep cycles for different nanosleep.u32 values')
    print('  (validates P-test load warp idle simulation)')
    print()

    print(f'  {"Test":<20s}  {"ns_value":>8s}  {"cyc/call":>10s}  {"~ns @1.8GHz":>12s}')
    for t in n_tests:
        m = re.match(r'N_sleep_(\d+)', t['name'])
        if not m:
            continue
        ns_val = int(m.group(1))
        # CPI is cycles per nanosleep call (insns_per_iter=1)
        cpi = t['warp_cpis'][0] if t['warp_cpis'] else 0
        # Estimate ns at typical clock
        est_ns = cpi / 1.8 if cpi else 0  # ~1.8 GHz typical
        print(f'  {t["name"]:<20s}  {ns_val:>8d}  {cpi:>10.1f}  {est_ns:>10.1f}ns')


def analyze_asymmetric(results):
    """A-tests: asymmetric duration — dispatch recovery after epi warps exit."""
    a_tests = [r for r in results if r['category'] == 'A']
    if not a_tests:
        return

    # Get steady-state baseline from F-tests (F_4ldgmix_2ffma)
    f_baseline = [r for r in results if r['name'] == 'F_4ldgmix_2ffma']

    print('\n' + '='*80)
    print('A-TESTS: Asymmetric duration (dispatch recovery)')
    print('='*80)
    print('  Epi warps exit early. Compare compute warp CPI to steady-state F-test.')
    print('  Lower compute CPI = dispatch contention drops after epi warps finish.')
    print()

    if f_baseline:
        fb = f_baseline[0]
        # Compute warp CPIs are the last n_compute warps
        fb_compute_cpis = [c for c in fb['warp_cpis'][4:] if c and c > 0]
        fb_compute_avg = sum(fb_compute_cpis) / len(fb_compute_cpis) if fb_compute_cpis else 0
        print(f'  Baseline (F_4ldgmix_2ffma): compute warp avg = {fb_compute_avg:.2f} cyc/insn')
    else:
        fb_compute_avg = 0

    for t in a_tests:
        m = re.match(r'A_(\d+)ldgmix_(\d+)ffma_(half|quarter)', t['name'])
        if not m:
            print(f'\n  {t["name"]}: (unparsed)')
            continue
        n_epi = int(m.group(1))
        n_cmp = int(m.group(2))
        fraction = m.group(3)

        epi_cpis = [c for c in t['warp_cpis'][:n_epi] if c and c > 0]
        cmp_cpis = [c for c in t['warp_cpis'][n_epi:n_epi+n_cmp] if c and c > 0]
        epi_avg = sum(epi_cpis) / len(epi_cpis) if epi_cpis else 0
        cmp_avg = sum(cmp_cpis) / len(cmp_cpis) if cmp_cpis else 0

        print(f'\n  {t["name"]} ({t["n_warps"]} warps, epi={fraction}):')
        print(f'    Epi avg:     {epi_avg:.2f} cyc/insn (ran {"REPS/2" if fraction == "half" else "REPS/4"})')
        print(f'    Compute avg: {cmp_avg:.2f} cyc/insn (ran full REPS)')
        if fb_compute_avg > 0:
            delta = (cmp_avg - fb_compute_avg) / fb_compute_avg * 100
            arrow = '↓ BETTER' if delta < -2 else '↑ WORSE' if delta > 2 else '≈ SAME'
            print(f'    vs steady-state: {delta:+.1f}% {arrow}')
            print(f'    → {"YES — dispatch recovery helps" if delta < -5 else "NO — contention persists even with short epi"}')


def write_csv(results, path):
    """Write flat CSV for external analysis."""
    with open(path, 'w') as f:
        f.write('name,category,n_warps,warp_id,cyc_per_insn,max_cpi,min_cpi\n')
        for r in results:
            for w, cpi in enumerate(r['warp_cpis']):
                if cpi is not None:
                    f.write(f'{r["name"]},{r["category"]},{r["n_warps"]},'
                            f'{w},{cpi:.4f},{r["max_cpi"]:.4f},{r["min_cpi"]:.4f}\n')
    print(f'\nCSV written to {path}')


def main():
    parser = argparse.ArgumentParser(description='Analyze warp-scaling benchmark output')
    parser.add_argument('input', help='Output file from ./warp-scaling')
    parser.add_argument('--csv', help='Write CSV output')
    args = parser.parse_args()

    with open(args.input) as f:
        lines = f.readlines()

    results = parse_output(lines)
    if not results:
        print('No results parsed. Is the input file from ./warp-scaling?')
        sys.exit(1)

    print(f'Parsed {len(results)} test results')

    analyze_scaling(results)
    analyze_cross_pipe(results)
    analyze_fg_bg(results)
    analyze_prodcons(results)
    analyze_barsync(results)
    analyze_nanosleep(results)
    analyze_asymmetric(results)

    if args.csv:
        write_csv(results, args.csv)


if __name__ == '__main__':
    main()
