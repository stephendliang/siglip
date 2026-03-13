#!/usr/bin/env python3
"""Static analysis of GELU variant instruction scheduling.

Models instruction-level scheduling for each GELU_VARIANT (0-6) to predict
relative performance before B200 testing. Uses calibrated latencies from
sass_analysis.py.

Usage:
    python3 tools/analyze_gelu_variants.py              # comparison table
    python3 tools/analyze_gelu_variants.py --variant 3  # single variant detail
    python3 tools/analyze_gelu_variants.py --verbose     # instruction-by-instruction schedule
"""

import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional

# Import calibrated latencies from sass_analysis.py
sys.path.insert(0, __file__ and __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from sass_analysis import LATENCY


@dataclass
class SynthInstr:
    """Synthetic instruction for scheduling simulation."""
    opcode: str          # FADD, FMUL, FFMA, MUFU, F2FP, STS
    dst: list            # destination register names
    src: list            # source register names
    latency: int         # issue-to-result latency (cycles)
    elem: int            # element index (0-7) for grouping
    phase: str           # 'pre_tanh', 'tanh', 'post_tanh', 'cvt', 'sts'
    tag: str = ''        # human-readable description

    @property
    def pipe(self):
        if self.opcode == 'MUFU':
            return 'SFU'
        if self.opcode == 'STS':
            return 'SMEM'
        if self.opcode == 'F2FP':
            return 'CVT'
        return 'FP'


def _lat(opcode):
    return LATENCY.get(opcode, 4)


def _r(prefix, elem, suffix=''):
    return f'{prefix}{elem}{suffix}'


def emit_variant_0():
    """Variant 0: asm tanh, scalar — 8 ops/element."""
    instrs = []
    for i in range(8):
        x = _r('x', i)
        inner = _r('inner', i)
        t = _r('t', i)
        xsq = _r('xsq', i)
        xt = _r('xt', i)
        sum_ = _r('sum', i)
        g = _r('g', i)

        instrs.append(SynthInstr('FADD', [x], [_r('acc', i), _r('bias', i)],
                                 _lat('FADD'), i, 'pre_tanh', f'x{i} = acc{i} + bias{i}'))
        instrs.append(SynthInstr('FMUL', [xsq], [x, x],
                                 _lat('FMUL'), i, 'pre_tanh', f'xsq{i} = x{i} * x{i}'))
        instrs.append(SynthInstr('FFMA', [inner+'_pre'], [xsq, '__const_k1', '__const_k'],
                                 _lat('FFMA'), i, 'pre_tanh', f'inner{i}_pre = k1*xsq{i} + k'))
        instrs.append(SynthInstr('FMUL', [inner], [x, inner+'_pre'],
                                 _lat('FMUL'), i, 'pre_tanh', f'inner{i} = x{i} * inner{i}_pre'))
        instrs.append(SynthInstr('MUFU', [t], [inner],
                                 _lat('MUFU'), i, 'tanh', f't{i} = tanh(inner{i})'))
        instrs.append(SynthInstr('FMUL', [xt], [x, t],
                                 _lat('FMUL'), i, 'post_tanh', f'xt{i} = x{i} * t{i}'))
        instrs.append(SynthInstr('FADD', [sum_], [x, xt],
                                 _lat('FADD'), i, 'post_tanh', f'sum{i} = x{i} + xt{i}'))
        instrs.append(SynthInstr('FMUL', [g], [sum_, '__const_half'],
                                 _lat('FMUL'), i, 'post_tanh', f'g{i} = 0.5 * sum{i}'))
    _append_cvt_sts(instrs)
    return instrs


def emit_variant_1():
    """Variant 1: pure tanhf(), same algebra — compiler maps to MUFU.TANH."""
    return emit_variant_0()  # same instruction sequence; difference is asm constraint


def emit_variant_2():
    """Variant 2: tanhf() + half_x reorder — 5 pre, 2 post."""
    instrs = []
    for i in range(8):
        x = _r('x', i)
        hx = _r('halfx', i)
        inner = _r('inner', i)
        t = _r('t', i)
        hxt = _r('hxt', i)
        g = _r('g', i)

        instrs.append(SynthInstr('FADD', [x], [_r('acc', i), _r('bias', i)],
                                 _lat('FADD'), i, 'pre_tanh', f'x{i} = acc{i} + bias{i}'))
        instrs.append(SynthInstr('FMUL', [hx], [x, '__const_half'],
                                 _lat('FMUL'), i, 'pre_tanh', f'halfx{i} = 0.5 * x{i}'))
        xsq = _r('xsq', i)
        instrs.append(SynthInstr('FMUL', [xsq], [x, x],
                                 _lat('FMUL'), i, 'pre_tanh', f'xsq{i} = x{i} * x{i}'))
        instrs.append(SynthInstr('FFMA', [inner+'_pre'], [xsq, '__const_k1', '__const_k'],
                                 _lat('FFMA'), i, 'pre_tanh', f'inner{i}_pre = k1*xsq{i} + k'))
        instrs.append(SynthInstr('FMUL', [inner], [x, inner+'_pre'],
                                 _lat('FMUL'), i, 'pre_tanh', f'inner{i} = x{i} * inner{i}_pre'))
        instrs.append(SynthInstr('MUFU', [t], [inner],
                                 _lat('MUFU'), i, 'tanh', f't{i} = tanh(inner{i})'))
        instrs.append(SynthInstr('FMUL', [hxt], [hx, t],
                                 _lat('FMUL'), i, 'post_tanh', f'hxt{i} = halfx{i} * t{i}'))
        instrs.append(SynthInstr('FADD', [g], [hx, hxt],
                                 _lat('FADD'), i, 'post_tanh', f'g{i} = halfx{i} + hxt{i}'))
    _append_cvt_sts(instrs)
    return instrs


def emit_variant_3():
    """Variant 3: __fmaf_rn — 7 ops/element."""
    instrs = []
    for i in range(8):
        x = _r('x', i)
        xsq = _r('xsq', i)
        inner_pre = _r('inner_pre', i)
        inner = _r('inner', i)
        t = _r('t', i)
        g = _r('g', i)

        instrs.append(SynthInstr('FADD', [x], [_r('acc', i), _r('bias', i)],
                                 _lat('FADD'), i, 'pre_tanh', f'x{i} = acc{i} + bias{i}'))
        instrs.append(SynthInstr('FMUL', [xsq], [x, x],
                                 _lat('FMUL'), i, 'pre_tanh', f'xsq{i} = x{i} * x{i}'))
        instrs.append(SynthInstr('FFMA', [inner_pre], [xsq, '__const_k1', '__const_k'],
                                 _lat('FFMA'), i, 'pre_tanh', f'inner_pre{i} = fma(k1, xsq{i}, k)'))
        instrs.append(SynthInstr('FMUL', [inner], [x, inner_pre],
                                 _lat('FMUL'), i, 'pre_tanh', f'inner{i} = x{i} * inner_pre{i}'))
        instrs.append(SynthInstr('MUFU', [t], [inner],
                                 _lat('MUFU'), i, 'tanh', f't{i} = tanh(inner{i})'))
        instrs.append(SynthInstr('FFMA', [g], [x, t, x],
                                 _lat('FFMA'), i, 'post_tanh', f'g{i} = fma(x{i}, t{i}, x{i})'))
        instrs.append(SynthInstr('FMUL', [_r('gf', i)], [g, '__const_half'],
                                 _lat('FMUL'), i, 'post_tanh', f'gf{i} = 0.5 * g{i}'))
    # variant 3 outputs gf0..gf7
    _append_cvt_sts(instrs, prefix='gf')
    return instrs


def _emit_batched_8_asm():
    """Variant 4: batched-8, asm tanh — explicit 3-phase pipeline."""
    instrs = []
    # Phase 1: all pre-tanh
    for i in range(8):
        x = _r('x', i)
        xsq = _r('xsq', i)
        instrs.append(SynthInstr('FADD', [x], [_r('acc', i), _r('bias', i)],
                                 _lat('FADD'), i, 'pre_tanh', f'x{i} = acc{i} + bias{i}'))
        instrs.append(SynthInstr('FMUL', [xsq], [x, x],
                                 _lat('FMUL'), i, 'pre_tanh', f'xsq{i} = x{i} * x{i}'))
        instrs.append(SynthInstr('FFMA', [_r('ip', i)], [xsq, '__const_k1', '__const_k'],
                                 _lat('FFMA'), i, 'pre_tanh', f'ip{i} = k1*xsq{i} + k'))
        instrs.append(SynthInstr('FMUL', [_r('inner', i)], [x, _r('ip', i)],
                                 _lat('FMUL'), i, 'pre_tanh', f'inner{i} = x{i} * ip{i}'))
    # Phase 2: all tanh back-to-back
    for i in range(8):
        instrs.append(SynthInstr('MUFU', [_r('t', i)], [_r('inner', i)],
                                 _lat('MUFU'), i, 'tanh', f't{i} = tanh(inner{i})'))
    # Phase 3: all post-tanh
    for i in range(8):
        x = _r('x', i)
        t = _r('t', i)
        xt = _r('xt', i)
        sum_ = _r('sum', i)
        g = _r('g', i)
        instrs.append(SynthInstr('FMUL', [xt], [x, t],
                                 _lat('FMUL'), i, 'post_tanh', f'xt{i} = x{i} * t{i}'))
        instrs.append(SynthInstr('FADD', [sum_], [x, xt],
                                 _lat('FADD'), i, 'post_tanh', f'sum{i} = x{i} + xt{i}'))
        instrs.append(SynthInstr('FMUL', [g], [sum_, '__const_half'],
                                 _lat('FMUL'), i, 'post_tanh', f'g{i} = 0.5 * sum{i}'))
    _append_cvt_sts(instrs)
    return instrs


def emit_variant_4():
    return _emit_batched_8_asm()


def _emit_batched_4_4_asm():
    """Variant 5: batched-4+4, asm tanh — two groups, overlapped."""
    instrs = []
    # Group 1: elements 0-3
    for i in range(4):
        x = _r('x', i)
        xsq = _r('xsq', i)
        instrs.append(SynthInstr('FADD', [x], [_r('acc', i), _r('bias', i)],
                                 _lat('FADD'), i, 'pre_tanh', f'x{i} = acc{i} + bias{i}'))
        instrs.append(SynthInstr('FMUL', [xsq], [x, x],
                                 _lat('FMUL'), i, 'pre_tanh', f'xsq{i} = x{i} * x{i}'))
        instrs.append(SynthInstr('FFMA', [_r('ip', i)], [xsq, '__const_k1', '__const_k'],
                                 _lat('FFMA'), i, 'pre_tanh', f'ip{i} = k1*xsq{i} + k'))
        instrs.append(SynthInstr('FMUL', [_r('inner', i)], [x, _r('ip', i)],
                                 _lat('FMUL'), i, 'pre_tanh', f'inner{i} = x{i} * ip{i}'))
    for i in range(4):
        instrs.append(SynthInstr('MUFU', [_r('t', i)], [_r('inner', i)],
                                 _lat('MUFU'), i, 'tanh', f't{i} = tanh(inner{i})'))
    for i in range(4):
        instrs.append(SynthInstr('FMUL', [_r('xt', i)], [_r('x', i), _r('t', i)],
                                 _lat('FMUL'), i, 'post_tanh', f'xt{i} = x{i} * t{i}'))
        instrs.append(SynthInstr('FADD', [_r('sum', i)], [_r('x', i), _r('xt', i)],
                                 _lat('FADD'), i, 'post_tanh', f'sum{i} = x{i} + xt{i}'))
        instrs.append(SynthInstr('FMUL', [_r('g', i)], [_r('sum', i), '__const_half'],
                                 _lat('FMUL'), i, 'post_tanh', f'g{i} = 0.5 * sum{i}'))
    # Group 2: elements 4-7
    for i in range(4, 8):
        x = _r('x', i)
        xsq = _r('xsq', i)
        instrs.append(SynthInstr('FADD', [x], [_r('acc', i), _r('bias', i)],
                                 _lat('FADD'), i, 'pre_tanh', f'x{i} = acc{i} + bias{i}'))
        instrs.append(SynthInstr('FMUL', [xsq], [x, x],
                                 _lat('FMUL'), i, 'pre_tanh', f'xsq{i} = x{i} * x{i}'))
        instrs.append(SynthInstr('FFMA', [_r('ip', i)], [xsq, '__const_k1', '__const_k'],
                                 _lat('FFMA'), i, 'pre_tanh', f'ip{i} = k1*xsq{i} + k'))
        instrs.append(SynthInstr('FMUL', [_r('inner', i)], [x, _r('ip', i)],
                                 _lat('FMUL'), i, 'pre_tanh', f'inner{i} = x{i} * ip{i}'))
    for i in range(4, 8):
        instrs.append(SynthInstr('MUFU', [_r('t', i)], [_r('inner', i)],
                                 _lat('MUFU'), i, 'tanh', f't{i} = tanh(inner{i})'))
    for i in range(4, 8):
        instrs.append(SynthInstr('FMUL', [_r('xt', i)], [_r('x', i), _r('t', i)],
                                 _lat('FMUL'), i, 'post_tanh', f'xt{i} = x{i} * t{i}'))
        instrs.append(SynthInstr('FADD', [_r('sum', i)], [_r('x', i), _r('xt', i)],
                                 _lat('FADD'), i, 'post_tanh', f'sum{i} = x{i} + xt{i}'))
        instrs.append(SynthInstr('FMUL', [_r('g', i)], [_r('sum', i), '__const_half'],
                                 _lat('FMUL'), i, 'post_tanh', f'g{i} = 0.5 * sum{i}'))
    _append_cvt_sts(instrs)
    return instrs


def emit_variant_5():
    return _emit_batched_4_4_asm()


def emit_variant_6():
    """Variant 6: batched-8, tanhf() — same structure as 4, compiler freedom."""
    return _emit_batched_8_asm()  # same instruction model; difference is asm constraint


def _append_cvt_sts(instrs, prefix='g'):
    """Append 4x F2FP converts + 1 STS.V4 for 8 values."""
    for pair in range(4):
        i0 = pair * 2
        i1 = pair * 2 + 1
        instrs.append(SynthInstr('F2FP', [_r('o', pair)],
                                 [_r(prefix, i0), _r(prefix, i1)],
                                 _lat('F2FP'), i0, 'cvt',
                                 f'o{pair} = cvt_bf16x2({prefix}{i0}, {prefix}{i1})'))
    instrs.append(SynthInstr('STS', [],
                             ['o0', 'o1', 'o2', 'o3', '__saddr'],
                             _lat('STS'), 0, 'sts',
                             'st.shared.v4 [saddr], {o0,o1,o2,o3}'))


EMITTERS = {
    0: ('asm tanh, scalar (default)', emit_variant_0),
    1: ('tanhf(), same algebra', emit_variant_1),
    2: ('tanhf() + half_x reorder', emit_variant_2),
    3: ('__fmaf_rn intrinsics', emit_variant_3),
    4: ('batched-8, asm tanh', emit_variant_4),
    5: ('batched-4+4, asm tanh', emit_variant_5),
    6: ('batched-8, tanhf()', emit_variant_6),
}


def build_deps(instrs):
    """Build RAW dependency edges. Returns list of (producer_idx, consumer_idx)."""
    last_writer = {}  # reg_name -> instr_idx
    edges = []
    for idx, instr in enumerate(instrs):
        for src in instr.src:
            if src.startswith('__'):
                continue
            if src in last_writer:
                edges.append((last_writer[src], idx))
        for dst in instr.dst:
            last_writer[dst] = idx
    return edges


def schedule(instrs):
    """Greedy list scheduler. Returns (cycle_count, schedule_trace, per_instr_start_cycle).

    Tracks per-register ready times and per-pipe occupancy.
    Pipe throughputs: FP=1/cyc, SFU=1/cyc, CVT=1/cyc, SMEM=1/32cyc (serialized).
    """
    n = len(instrs)
    edges = build_deps(instrs)

    # Build predecessors per instruction
    preds = [set() for _ in range(n)]
    for p, c in edges:
        preds[c].add(p)

    reg_ready = {}   # reg -> cycle when result available
    pipe_ready = {'FP': 0, 'SFU': 0, 'CVT': 0, 'SMEM': 0}
    start_cycle = [None] * n
    done = [False] * n
    trace = []
    scheduled = 0
    cycle = 0

    while scheduled < n:
        # Find all ready instructions
        ready = []
        for idx in range(n):
            if done[idx]:
                continue
            # Check all predecessors are done
            if any(not done[p] for p in preds[idx]):
                continue
            # Check source registers are ready
            earliest = 0
            for src in instrs[idx].src:
                if src.startswith('__'):
                    continue
                earliest = max(earliest, reg_ready.get(src, 0))
            # Check pipe availability
            pipe = instrs[idx].pipe
            earliest = max(earliest, pipe_ready[pipe])
            ready.append((earliest, idx))

        if not ready:
            cycle += 1
            continue

        # Pick the one with earliest start; break ties by highest latency (longest chain first)
        ready.sort(key=lambda x: (x[0], -instrs[x[1]].latency))
        earliest_cycle, best_idx = ready[0]

        if earliest_cycle > cycle:
            cycle = earliest_cycle

        instr = instrs[best_idx]
        start_cycle[best_idx] = cycle
        done[best_idx] = True
        scheduled += 1

        # Update pipe occupancy
        pipe = instr.pipe
        if pipe == 'SMEM':
            pipe_ready[pipe] = cycle + instr.latency
        else:
            pipe_ready[pipe] = cycle + 1

        # Update register ready times
        for dst in instr.dst:
            reg_ready[dst] = cycle + instr.latency

        trace.append((cycle, best_idx, instr))
        cycle += 1  # advance at least 1 cycle per issue

    total_cycles = max(start_cycle[i] + instrs[i].latency for i in range(n))
    return total_cycles, trace, start_cycle


def critical_path(instrs):
    """Longest weighted dependency path through the DAG (cycles)."""
    n = len(instrs)
    edges = build_deps(instrs)

    # Forward adjacency
    succs = [[] for _ in range(n)]
    for p, c in edges:
        succs[p].append(c)

    # DP: longest path from each node
    memo = [None] * n

    def dp(idx):
        if memo[idx] is not None:
            return memo[idx]
        lat = instrs[idx].latency
        best = lat
        for s in succs[idx]:
            best = max(best, lat + dp(s))
        memo[idx] = best
        return best

    return max(dp(i) for i in range(n))


def peak_live_regs(instrs, start_cycle):
    """Approximate peak live FP32 register count."""
    n = len(instrs)
    edges = build_deps(instrs)

    # For each register, find define cycle and last-use cycle
    reg_def = {}
    reg_last_use = {}
    for idx, instr in enumerate(instrs):
        for dst in instr.dst:
            reg_def[dst] = start_cycle[idx]
        for src in instr.src:
            if src.startswith('__'):
                continue
            if src not in reg_last_use or start_cycle[idx] > reg_last_use[src]:
                reg_last_use[src] = start_cycle[idx]

    # Count live at each cycle
    all_regs = set(reg_def.keys())
    if not all_regs:
        return 0
    max_cycle = max(start_cycle[i] + instrs[i].latency for i in range(n))
    peak = 0
    for cyc in range(max_cycle + 1):
        live = 0
        for reg in all_regs:
            if reg_def.get(reg, max_cycle + 1) <= cyc <= reg_last_use.get(reg, -1):
                live += 1
        peak = max(peak, live)
    return peak


def instruction_mix(instrs):
    """Count instructions by opcode."""
    counts = {}
    for instr in instrs:
        counts[instr.opcode] = counts.get(instr.opcode, 0) + 1
    return counts


def mufu_gaps(trace):
    """Return list of cycle gaps between consecutive MUFU issues."""
    mufu_cycles = [cyc for cyc, idx, instr in trace if instr.opcode == 'MUFU']
    if len(mufu_cycles) < 2:
        return []
    return [mufu_cycles[i+1] - mufu_cycles[i] for i in range(len(mufu_cycles) - 1)]


def analyze_variant(variant_id, verbose=False):
    """Analyze a single variant. Returns metrics dict."""
    name, emitter = EMITTERS[variant_id]
    instrs = emitter()
    total_instrs = len(instrs)
    mix = instruction_mix(instrs)
    cp = critical_path(instrs)
    sched_cycles, trace, start_cycles = schedule(instrs)
    gaps = mufu_gaps(trace)
    peak = peak_live_regs(instrs, start_cycles)

    result = {
        'id': variant_id,
        'name': name,
        'total_instrs': total_instrs,
        'critical_path': cp,
        'scheduled_cycles': sched_cycles,
        'peak_live_regs': peak,
        'mix': mix,
        'mufu_gaps': gaps,
        'mufu_gap_avg': sum(gaps) / len(gaps) if gaps else 0,
        'mufu_gap_min': min(gaps) if gaps else 0,
    }

    if verbose:
        result['trace'] = trace
        result['instrs'] = instrs

    return result


def print_comparison_table(results):
    """Print side-by-side comparison of all variants."""
    print('GELU Variant Comparison (8 elements + CVT + STS per call)')
    print('=' * 100)

    hdr = f'{"V":>2}  {"Name":<28}  {"Instrs":>6}  {"CritPath":>8}  {"Sched":>6}  {"Regs":>4}  {"MUFU gap":>8}  {"Mix"}'
    print(hdr)
    print('-' * 100)

    for r in results:
        mix = r['mix']
        mix_str = '  '.join(f'{k}:{v}' for k, v in sorted(mix.items()))
        gap_str = f'{r["mufu_gap_avg"]:.1f}' if r['mufu_gaps'] else '-'
        print(f'{r["id"]:>2}  {r["name"]:<28}  {r["total_instrs"]:>6}  '
              f'{r["critical_path"]:>8}  {r["scheduled_cycles"]:>6}  '
              f'{r["peak_live_regs"]:>4}  {gap_str:>8}  {mix_str}')

    print()
    # Find best
    best_cp = min(results, key=lambda r: r['critical_path'])
    best_sc = min(results, key=lambda r: r['scheduled_cycles'])
    best_regs = min(results, key=lambda r: r['peak_live_regs'])
    print(f'Best critical path:    variant {best_cp["id"]} ({best_cp["critical_path"]} cyc)')
    print(f'Best scheduled cycles: variant {best_sc["id"]} ({best_sc["scheduled_cycles"]} cyc)')
    print(f'Lowest register pressure: variant {best_regs["id"]} ({best_regs["peak_live_regs"]} regs)')

    # Relative improvement table
    baseline = results[0]['scheduled_cycles']
    print(f'\nRelative to variant 0 ({baseline} scheduled cycles):')
    for r in results:
        delta = r['scheduled_cycles'] - baseline
        pct = delta / baseline * 100 if baseline else 0
        sign = '+' if delta >= 0 else ''
        print(f'  V{r["id"]}: {sign}{delta} cyc ({sign}{pct:.1f}%)')


def print_verbose_trace(result):
    """Print instruction-by-instruction schedule trace."""
    print(f'\nVariant {result["id"]}: {result["name"]}')
    print(f'  Total instructions: {result["total_instrs"]}')
    print(f'  Critical path: {result["critical_path"]} cycles')
    print(f'  Scheduled: {result["scheduled_cycles"]} cycles')
    print(f'  Peak live regs: {result["peak_live_regs"]}')
    print()

    print(f'  {"Cyc":>4}  {"Pipe":<5}  {"Op":<5}  {"Lat":>3}  {"Elem":>4}  {"Phase":<10}  {"Description"}')
    print(f'  {"-"*4}  {"-"*5}  {"-"*5}  {"-"*3}  {"-"*4}  {"-"*10}  {"-"*30}')

    for cyc, idx, instr in result['trace']:
        print(f'  {cyc:>4}  {instr.pipe:<5}  {instr.opcode:<5}  {instr.latency:>3}  '
              f'{instr.elem:>4}  {instr.phase:<10}  {instr.tag}')

    if result['mufu_gaps']:
        print(f'\n  MUFU gaps: {result["mufu_gaps"]}')
        print(f'  MUFU gap avg: {result["mufu_gap_avg"]:.1f}, min: {result["mufu_gap_min"]}')


def main():
    parser = argparse.ArgumentParser(description='GELU variant static scheduling analysis')
    parser.add_argument('--variant', type=int, choices=list(EMITTERS.keys()),
                        help='Analyze single variant in detail')
    parser.add_argument('--verbose', action='store_true',
                        help='Show instruction-by-instruction schedule trace')
    args = parser.parse_args()

    if args.variant is not None:
        result = analyze_variant(args.variant, verbose=True)
        print_verbose_trace(result)
        return

    results = [analyze_variant(v, verbose=args.verbose) for v in sorted(EMITTERS.keys())]
    print_comparison_table(results)

    if args.verbose:
        for r in results:
            print_verbose_trace(r)


if __name__ == '__main__':
    main()
