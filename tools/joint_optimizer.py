#!/usr/bin/env python3
"""
Joint instruction selection + scheduling optimizer for FC2 epilogue.

Stage A: Per-chunk (32 cols = 16 BF16 pairs).
Stage B: Multi-chunk sub-iteration (n × 32 cols + boundary ops).

Uses CP-SAT to jointly decide which pairs use BF16 vs ALU path
AND optimal instruction ordering, minimizing makespan.

Usage:
  python3 tools/joint_optimizer.py                          # sweep k=0..16, 1 chunk
  python3 tools/joint_optimizer.py --k 5                    # fixed k=5, 1 chunk
  python3 tools/joint_optimizer.py --joint                  # joint, 1 chunk (Stage A)
  python3 tools/joint_optimizer.py --chunks 2               # sweep k=0..16, 2 chunks
  python3 tools/joint_optimizer.py --chunks 2 --joint       # joint, 2 chunks (Stage B)
  python3 tools/joint_optimizer.py --chunks 2 --joint --time 600
"""

import argparse
import sys
import time as time_module
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional

from ortools.sat.python import cp_model

from lowering_rules import (
    Pipe, PIPE_THROUGHPUT, INSN_LATENCY, BF16_SLOW,
)


# ---------------------------------------------------------------------------
# Concrete instruction for the CP-SAT model
# ---------------------------------------------------------------------------

@dataclass
class Insn:
    """One concrete SASS instruction with scheduling metadata."""
    idx: int                # unique global index
    mnemonic: str           # e.g., 'FFMA', 'HFMA2', 'LDS.128', 'STS.128'
    pipe: Pipe
    throughput: int         # min cycles to next same-pipe issue
    latency: int            # cycles until result available
    reads: List[str]        # symbolic register names read
    writes: List[str]       # symbolic register names written
    pair_idx: int = -1      # which of 16 pairs this belongs to (-1 = fixed/shared)
    is_alu_path: bool = False  # True if this insn only exists in the ALU path
    is_bf16_path: bool = False # True if this insn only exists in the BF16 path
    comment: str = ''
    chunk_idx: int = 0      # which 32-col chunk (0, 1, ...)


# ---------------------------------------------------------------------------
# Chunk instruction generator (internal)
# ---------------------------------------------------------------------------

def _generate_chunk(k_alu: int = None, chunk_idx: int = 0,
                    start_idx: int = 0) -> Tuple[List[Insn], int]:
    """Generate all instructions for one 32-col chunk.

    Args:
        k_alu: None for joint (both paths), int for fixed (first k pairs on ALU).
        chunk_idx: chunk identifier (0, 1, ...). Prefixes register names.
        start_idx: starting instruction index for globally unique IDs.

    Returns:
        (insns, next_free_idx)

    Fixed per-chunk:
      1× LDTM.x32 → 32 TMEM regs
      1× DEPBAR (wait for TMEM)
      4× LDS.128 bias → 16 bias regs
      4× LDS.128 residual → 16 res regs
      4× STS.128 (store 16 BF16 packed results)

    Per pair (BF16 path): F2FP + HFMA2 + HADD2 = 3 insns, 8 BF16 cyc
    Per pair (ALU path): LOP3+SHF+2×FFMA+LOP3+SHF+2×FFMA+F2FP = 9 insns, 16 ALU + 2 BF16 cyc
    """
    c = chunk_idx
    p = f'c{c}_'  # register name prefix
    joint = (k_alu is None)
    insns = []
    idx = start_idx

    def add(mnemonic, pipe, throughput, latency, reads, writes,
            pair=-1, alu_only=False, bf16_only=False, comment=''):
        nonlocal idx
        insns.append(Insn(
            idx=idx, mnemonic=mnemonic, pipe=pipe,
            throughput=throughput, latency=latency,
            reads=reads, writes=writes,
            pair_idx=pair, is_alu_path=alu_only, is_bf16_path=bf16_only,
            comment=comment, chunk_idx=c,
        ))
        idx += 1

    # --- Fixed: LDTM ---
    tmem_writes = [f'{p}tmem_{i}' for i in range(32)]
    add('LDTM', Pipe.TMEM, 20, 20, [f'{p}tmem_addr'], tmem_writes,
        comment=f'LDTM.x32 chunk {c}')

    # --- Fixed: DEPBAR (wait for TMEM) ---
    add('DEPBAR', Pipe.BARRIER, 0, 0, tmem_writes, tmem_writes,
        comment=f'DEPBAR.LE SB0 chunk {c}')

    # --- Fixed: 4× LDS.128 bias ---
    for g in range(4):
        w = [f'{p}bias_{g*4+j}' for j in range(4)]
        add('LDS.128', Pipe.LOAD, 4, 20, [f'{p}bias_saddr'], w,
            comment=f'LDS.128 bias group {g} chunk {c}')

    # --- Fixed: 4× LDS.128 residual ---
    for g in range(4):
        w = [f'{p}res_{g*4+j}' for j in range(4)]
        add('LDS.128', Pipe.LOAD, 4, 20, [f'{p}res_saddr'], w,
            comment=f'LDS.128 res group {g} chunk {c}')

    # --- Per-pair compute ---
    for pair in range(16):
        tmem_lo = f'{p}tmem_{pair*2}'
        tmem_hi = f'{p}tmem_{pair*2+1}'
        bias_reg = f'{p}bias_{pair}'
        res_reg = f'{p}res_{pair}'
        acc_lo = f'{p}acc_fp32_lo_{pair}'
        acc_hi = f'{p}acc_fp32_hi_{pair}'

        # In joint mode, use path-specific output registers so both
        # paths create correct RAW deps to STS independently.
        # Without this, build_deps last_writer sees ALU path's F2FP_late
        # as the only writer of cvt → loses BF16 path's HADD2 → STS dep.
        if joint:
            cvt_bf16 = f'{p}cvt_bf16_{pair}'
            cvt_alu = f'{p}cvt_alu_{pair}'
        else:
            cvt_bf16 = f'{p}cvt_{pair}'
            cvt_alu = f'{p}cvt_{pair}'

        use_alu = (k_alu is not None and pair < k_alu)
        use_bf16 = (k_alu is not None and pair >= k_alu)

        # --- BF16 path ---
        if use_bf16 or joint:
            bf16_only = joint
            add('F2FP', Pipe.BF16, 2, 4,
                [tmem_hi, tmem_lo], [cvt_bf16],
                pair=pair, bf16_only=bf16_only,
                comment=f'F2FP pair {pair} chunk {c} (BF16)')
            add('HFMA2', Pipe.BF16, 3, 4,
                [cvt_bf16, bias_reg], [cvt_bf16],
                pair=pair, bf16_only=bf16_only,
                comment=f'HFMA2 bias pair {pair} chunk {c} (BF16)')
            add('HADD2', Pipe.BF16, 3, 4,
                [cvt_bf16, res_reg], [cvt_bf16],
                pair=pair, bf16_only=bf16_only,
                comment=f'HADD2 res pair {pair} chunk {c} (BF16)')

        # --- ALU path ---
        if use_alu or joint:
            alu_only = joint
            add('LOP3', Pipe.ALU, 2, 4,
                [bias_reg], [f'{p}bias_fp32_hi_{pair}'],
                pair=pair, alu_only=alu_only,
                comment=f'LOP3 unpack bias hi pair {pair} chunk {c}')
            add('SHF', Pipe.ALU, 2, 4,
                [bias_reg], [f'{p}bias_fp32_lo_{pair}'],
                pair=pair, alu_only=alu_only,
                comment=f'SHF unpack bias lo pair {pair} chunk {c}')
            add('FFMA', Pipe.ALU, 2, 4,
                [tmem_lo, f'{p}bias_fp32_lo_{pair}'], [acc_lo],
                pair=pair, alu_only=alu_only,
                comment=f'FFMA bias lo pair {pair} chunk {c}')
            add('FFMA', Pipe.ALU, 2, 4,
                [tmem_hi, f'{p}bias_fp32_hi_{pair}'], [acc_hi],
                pair=pair, alu_only=alu_only,
                comment=f'FFMA bias hi pair {pair} chunk {c}')
            add('LOP3', Pipe.ALU, 2, 4,
                [res_reg], [f'{p}res_fp32_hi_{pair}'],
                pair=pair, alu_only=alu_only,
                comment=f'LOP3 unpack res hi pair {pair} chunk {c}')
            add('SHF', Pipe.ALU, 2, 4,
                [res_reg], [f'{p}res_fp32_lo_{pair}'],
                pair=pair, alu_only=alu_only,
                comment=f'SHF unpack res lo pair {pair} chunk {c}')
            add('FFMA', Pipe.ALU, 2, 4,
                [acc_lo, f'{p}res_fp32_lo_{pair}'], [acc_lo],
                pair=pair, alu_only=alu_only,
                comment=f'FFMA res lo pair {pair} chunk {c}')
            add('FFMA', Pipe.ALU, 2, 4,
                [acc_hi, f'{p}res_fp32_hi_{pair}'], [acc_hi],
                pair=pair, alu_only=alu_only,
                comment=f'FFMA res hi pair {pair} chunk {c}')
            add('F2FP', Pipe.BF16, 2, 4,
                [acc_hi, acc_lo], [cvt_alu],
                pair=pair, alu_only=alu_only,
                comment=f'F2FP pair {pair} chunk {c} (ALU late)')

    # --- Fixed: 4× STS.128 ---
    for g in range(4):
        reads = []
        for j in range(4):
            pair = g * 4 + j
            if joint:
                # Read from both path-specific outputs — solver enforces
                # the active path's dep, skips the inactive one
                reads.extend([f'{p}cvt_bf16_{pair}', f'{p}cvt_alu_{pair}'])
            else:
                reads.append(f'{p}cvt_{pair}')
        reads.append(f'{p}sts_addr_{g}')
        add('STS.128', Pipe.STORE, 32, 0, reads, [f'{p}sts_done_{g}'],
            comment=f'STS.128 group {g} chunk {c}')

    return insns, idx


def generate_chunk_instructions(k_alu: int = None) -> List[Insn]:
    """Generate all instructions for one 32-col chunk (Stage A interface).

    If k_alu is None: generates BOTH paths for all 16 pairs (for joint model).
    If k_alu is int: generates ALU path for first k_alu pairs, BF16 for rest.
    """
    insns, _ = _generate_chunk(k_alu=k_alu, chunk_idx=0, start_idx=0)
    return insns


# ---------------------------------------------------------------------------
# Sub-iteration instruction generator (Stage B)
# ---------------------------------------------------------------------------

def generate_subiter_instructions(n_chunks: int = 2,
                                  k_alu=None) -> List[Insn]:
    """Generate instructions for one sub-iteration (n_chunks × 32 cols).

    Includes per-chunk compute + IS1 sub-iteration boundary ops
    (WARPSYNC, FENCE, TMA store, commit) after all chunks' STS.

    Args:
        n_chunks: number of 32-col chunks per sub-iteration.
        k_alu: None (joint), int (same k for all chunks), or list (per-chunk).
    """
    all_insns = []
    idx = 0

    for c in range(n_chunks):
        if k_alu is None:
            ck = None
        elif isinstance(k_alu, (list, tuple)):
            ck = k_alu[c]
        else:
            ck = k_alu
        chunk_insns, idx = _generate_chunk(k_alu=ck, chunk_idx=c, start_idx=idx)
        all_insns.extend(chunk_insns)

    # --- Sub-iteration boundary ops (IS1 pattern) ---
    # All STS must complete before TMA can read staging SMEM.

    sts_done_regs = [f'c{c}_sts_done_{g}'
                     for c in range(n_chunks) for g in range(4)]

    # WARPSYNC: all lanes confirm STS issued
    all_insns.append(Insn(
        idx=idx, mnemonic='WARPSYNC', pipe=Pipe.BARRIER,
        throughput=0, latency=2,
        reads=sts_done_regs, writes=['sync_done'],
        comment='__syncwarp() before fence'))
    idx += 1

    # FENCE: make STS writes visible to async TMA engine (~10 cyc)
    all_insns.append(Insn(
        idx=idx, mnemonic='FENCE', pipe=Pipe.BARRIER,
        throughput=0, latency=10,
        reads=['sync_done'], writes=['fence_done'],
        comment='fence.proxy.async.shared::cta'))
    idx += 1

    # UTMASTG: issue TMA store from staging SMEM (async, fire-and-forget)
    all_insns.append(Insn(
        idx=idx, mnemonic='UTMASTG', pipe=Pipe.TMA,
        throughput=0, latency=0,
        reads=['fence_done'], writes=['tma_issued'],
        comment='TMA store from staging SMEM'))
    idx += 1

    # COMMIT: batch TMA operations
    all_insns.append(Insn(
        idx=idx, mnemonic='COMMIT', pipe=Pipe.CONTROL,
        throughput=0, latency=0,
        reads=['tma_issued'], writes=[],
        comment='cp.async.bulk.commit_group'))
    idx += 1

    return all_insns


# ---------------------------------------------------------------------------
# Dependency graph builder
# ---------------------------------------------------------------------------

def build_deps(insns: List[Insn]) -> List[Tuple[int, int, str, int]]:
    """Build dependency edges from symbolic read/write sets.

    Returns list of (src_idx, dst_idx, dep_type, latency).
    Works correctly for multi-chunk: register names are chunk-prefixed,
    so no spurious cross-chunk dependencies are created.
    """
    edges = []
    last_writer: Dict[str, int] = {}
    last_readers: Dict[str, List[int]] = {}

    for insn in insns:
        i = insn.idx

        for r in insn.reads:
            # RAW: reader depends on last writer
            if r in last_writer:
                w = last_writer[r]
                lat = insns[w].latency
                edges.append((w, i, 'RAW', lat))

        for w in insn.writes:
            # WAW: writer depends on last writer (ordering)
            if w in last_writer:
                prev = last_writer[w]
                edges.append((prev, i, 'WAW', 0))

            # WAR: writer depends on last readers (can't overwrite before read)
            if w in last_readers:
                for reader in last_readers[w]:
                    if reader != i:
                        edges.append((reader, i, 'WAR', 0))

        # Update tracking
        for w in insn.writes:
            last_writer[w] = i
            last_readers[w] = []

        for r in insn.reads:
            last_readers.setdefault(r, []).append(i)

    return edges


# ---------------------------------------------------------------------------
# Core CP-SAT solver
# ---------------------------------------------------------------------------

def _solve_model(insns: List[Insn], joint: bool = False,
                 time_limit: float = 60.0,
                 verbose: bool = True) -> Optional[Dict]:
    """Build and solve CP-SAT model for any instruction list.

    Handles both fixed-path and joint (path selection as CP-SAT variable) modes.
    Uses IntervalVar + no_overlap per pipe for efficient throughput modeling.
    """
    N = len(insns)
    edges = build_deps(insns)

    model = cp_model.CpModel()
    max_horizon = N * 20 if joint else N * 40

    # --- Discover pairs for joint model ---
    pair_keys = sorted(set(
        (insn.chunk_idx, insn.pair_idx)
        for insn in insns if insn.pair_idx >= 0
    )) if joint else []

    # --- Path selection variables (joint only) ---
    use_alu: Dict[Tuple[int, int], cp_model.IntVar] = {}
    active: List = [None] * N

    if joint:
        for key in pair_keys:
            use_alu[key] = model.new_bool_var(f'alu_c{key[0]}_p{key[1]}')

        for i, insn in enumerate(insns):
            active[i] = model.new_bool_var(f'act_{i}')
            key = (insn.chunk_idx, insn.pair_idx)

            if insn.pair_idx < 0:
                model.add(active[i] == 1)
            elif insn.is_alu_path:
                model.add(active[i] == use_alu[key])
            elif insn.is_bf16_path:
                model.add(active[i] == 1).only_enforce_if(use_alu[key].negated())
                model.add(active[i] == 0).only_enforce_if(use_alu[key])
            else:
                model.add(active[i] == 1)

    # --- Issue time per instruction ---
    time_var = [model.new_int_var(0, max_horizon, f't_{i}') for i in range(N)]

    if joint:
        for i in range(N):
            model.add(time_var[i] == 0).only_enforce_if(active[i].negated())

    # --- Dependency constraints ---
    for src, dst, dep_type, latency in edges:
        src_insn = insns[src]
        dst_insn = insns[dst]

        if joint:
            # Skip deps between different paths of same pair in same chunk
            if (src_insn.pair_idx == dst_insn.pair_idx >= 0 and
                src_insn.chunk_idx == dst_insn.chunk_idx and
                src_insn.is_alu_path != dst_insn.is_alu_path and
                (src_insn.is_alu_path or src_insn.is_bf16_path) and
                (dst_insn.is_alu_path or dst_insn.is_bf16_path)):
                continue

            both_act = model.new_bool_var(f'dep_{src}_{dst}')
            model.add_min_equality(both_act, [active[src], active[dst]])

            if dep_type == 'RAW' and latency > 0:
                model.add(time_var[dst] >= time_var[src] + latency).only_enforce_if(both_act)
            else:
                model.add(time_var[dst] >= time_var[src] + 1).only_enforce_if(both_act)
        else:
            if dep_type == 'RAW' and latency > 0:
                model.add(time_var[dst] >= time_var[src] + latency)
            else:
                model.add(time_var[dst] >= time_var[src] + 1)

    # --- Per-pipe throughput via no_overlap ---
    pipes: Dict[Pipe, List[int]] = {}
    for insn in insns:
        if insn.throughput > 0:
            pipes.setdefault(insn.pipe, []).append(insn.idx)

    for pipe, indices in pipes.items():
        intervals = []
        for i in indices:
            tp = insns[i].throughput
            if joint:
                iv = model.new_optional_interval_var(
                    time_var[i], tp, time_var[i] + tp,
                    active[i], f'iv_{pipe.value}_{i}')
            else:
                iv = model.new_interval_var(
                    time_var[i], tp, time_var[i] + tp,
                    f'iv_{pipe.value}_{i}')
            intervals.append(iv)
        if len(intervals) >= 2:
            model.add_no_overlap(intervals)

    # --- Objective: minimize makespan, then instruction count ---
    makespan = model.new_int_var(0, max_horizon, 'makespan')

    if joint:
        for i in range(N):
            model.add(makespan >= time_var[i]).only_enforce_if(active[i])
        total_active = model.new_int_var(0, N, 'total_active')
        model.add(total_active == sum(active))
        model.minimize(makespan * 1000 + total_active)
    else:
        model.add_max_equality(makespan, time_var)
        model.minimize(makespan)

    # --- Solve ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8

    callback = _ProgressCallback(verbose) if (joint and verbose) else None
    status = solver.solve(model, callback)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if verbose:
            print(f'  No solution: {solver.status_name(status)}')
        return None

    ms = int(solver.value(makespan))
    wall = solver.wall_time

    # --- Extract solution ---
    if joint:
        n_active = int(solver.value(total_active))
        active_items = [(solver.value(time_var[i]), i)
                        for i in range(N) if solver.value(active[i])]
        active_items.sort()
        sol = [i for _, i in active_items]
    else:
        n_active = N
        timed = [(solver.value(time_var[i]), i) for i in range(N)]
        timed.sort()
        sol = [i for _, i in timed]

    stalls = []
    for pos in range(len(sol)):
        if pos == 0:
            stalls.append(solver.value(time_var[sol[0]]))
        else:
            gap = solver.value(time_var[sol[pos]]) - solver.value(time_var[sol[pos - 1]])
            stalls.append(min(gap, 15))

    result = {
        'n_insns': n_active,
        'makespan': ms,
        'status': solver.status_name(status),
        'wall_time': wall,
        'order': sol,
        'stalls': stalls,
        'insns': insns,
    }

    if joint:
        alu_pairs = {}  # chunk_idx → [pair_indices]
        for key in pair_keys:
            chunk, pair = key
            if solver.value(use_alu[key]):
                alu_pairs.setdefault(chunk, []).append(pair)

        result.update({
            'k_alu': sum(len(v) for v in alu_pairs.values()),
            'alu_pairs': alu_pairs,
            'n_total': N,
            'branches': solver.num_branches,
            'conflicts': solver.num_conflicts,
        })

    return result


class _ProgressCallback(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions during joint solve."""
    def __init__(self, verbose):
        super().__init__()
        self._verbose = verbose
        self._count = 0
        self._start = time_module.time()

    def on_solution_callback(self):
        if self._verbose:
            self._count += 1
            elapsed = time_module.time() - self._start
            obj = self.objective_value
            ms = int(obj // 1000)
            insns = int(obj % 1000)
            if self._count <= 5 or self._count % 10 == 0:
                print(f'  sol #{self._count}: makespan={ms}, insns={insns} ({elapsed:.1f}s)')


# ---------------------------------------------------------------------------
# Solver entry points — Stage A (single chunk)
# ---------------------------------------------------------------------------

def solve_chunk_fixed_k(k_alu: int, time_limit: float = 60.0,
                        verbose: bool = True) -> Optional[Dict]:
    """Solve one 32-col chunk with fixed k_alu pairs on ALU path (Stage A)."""
    insns = generate_chunk_instructions(k_alu=k_alu)
    N = len(insns)

    if verbose:
        by_pipe = {}
        for insn in insns:
            by_pipe[insn.pipe.value] = by_pipe.get(insn.pipe.value, 0) + 1
        parts = ' '.join(f'{k}={v}' for k, v in sorted(by_pipe.items()))
        deps = len(build_deps(insns))
        print(f'k={k_alu}: {N} insns, {deps} deps | {parts}')

    result = _solve_model(insns, joint=False, time_limit=time_limit, verbose=verbose)
    if result:
        result['k_alu'] = k_alu
        result['n_chunks'] = 1
        if verbose:
            print(f'  \u2192 makespan={result["makespan"]} cyc, '
                  f'{result["status"]} ({result["wall_time"]:.1f}s)')
    return result


def solve_chunk_joint(time_limit: float = 300.0,
                      verbose: bool = True) -> Optional[Dict]:
    """Solve one 32-col chunk with joint path selection (Stage A)."""
    insns = generate_chunk_instructions(k_alu=None)
    N = len(insns)

    if verbose:
        print(f'Joint model: {N} instructions (both paths for all 16 pairs)')
        print(f'Solving (time limit {time_limit:.0f}s)...')

    result = _solve_model(insns, joint=True, time_limit=time_limit, verbose=verbose)
    if result:
        result['n_chunks'] = 1
        if verbose:
            ap = result['alu_pairs']
            ap_flat = ap.get(0, [])
            print(f'\n{result["status"]}: makespan={result["makespan"]} cyc, '
                  f'{result["n_insns"]} active insns, k_alu={result["k_alu"]} '
                  f'({result["wall_time"]:.1f}s, {result["branches"]} branches)')
            print(f'ALU pairs: {ap_flat}')
    return result


# ---------------------------------------------------------------------------
# Solver entry points — Stage B (multi-chunk sub-iteration)
# ---------------------------------------------------------------------------

def solve_subiter_fixed_k(n_chunks: int, k_alu,
                          time_limit: float = 60.0,
                          verbose: bool = True) -> Optional[Dict]:
    """Solve multi-chunk sub-iteration with fixed k_alu per chunk (Stage B)."""
    insns = generate_subiter_instructions(n_chunks=n_chunks, k_alu=k_alu)
    N = len(insns)

    if verbose:
        by_pipe = {}
        for insn in insns:
            by_pipe[insn.pipe.value] = by_pipe.get(insn.pipe.value, 0) + 1
        parts = ' '.join(f'{k}={v}' for k, v in sorted(by_pipe.items()))
        deps = len(build_deps(insns))
        k_str = k_alu if isinstance(k_alu, int) else str(k_alu)
        print(f'k={k_str}, {n_chunks} chunks: {N} insns, {deps} deps | {parts}')

    result = _solve_model(insns, joint=False, time_limit=time_limit, verbose=verbose)
    if result:
        result['k_alu'] = k_alu if isinstance(k_alu, int) else sum(k_alu)
        result['n_chunks'] = n_chunks
        if verbose:
            print(f'  \u2192 makespan={result["makespan"]} cyc, '
                  f'{result["status"]} ({result["wall_time"]:.1f}s)')
    return result


def solve_subiter_joint(n_chunks: int, time_limit: float = 600.0,
                        verbose: bool = True) -> Optional[Dict]:
    """Solve multi-chunk sub-iteration with joint path selection (Stage B)."""
    insns = generate_subiter_instructions(n_chunks=n_chunks, k_alu=None)
    N = len(insns)
    n_pairs = 16 * n_chunks

    if verbose:
        print(f'Joint model: {N} instructions, {n_chunks} chunks, '
              f'{n_pairs} pairs (both paths for all)')
        print(f'Solving (time limit {time_limit:.0f}s)...')

    result = _solve_model(insns, joint=True, time_limit=time_limit, verbose=verbose)
    if result:
        result['n_chunks'] = n_chunks
        if verbose:
            ap = result['alu_pairs']
            lines = []
            for c in range(n_chunks):
                ps = ap.get(c, [])
                lines.append(f'  Chunk {c}: k={len(ps)}, ALU pairs={ps}')
            print(f'\n{result["status"]}: makespan={result["makespan"]} cyc, '
                  f'{result["n_insns"]} active insns, k_alu={result["k_alu"]} '
                  f'({result["wall_time"]:.1f}s, {result["branches"]} branches)')
            print('\n'.join(lines))
    return result


# ---------------------------------------------------------------------------
# Solution printing
# ---------------------------------------------------------------------------

def print_schedule(result: Dict, detail: bool = False):
    """Print the solved schedule."""
    insns = result['insns']
    order = result['order']
    stalls = result['stalls']
    ms = result['makespan']
    n_chunks = result.get('n_chunks', 1)

    print(f'\n{"="*80}')
    k_str = f'k_alu={result["k_alu"]}' if 'k_alu' in result else ''
    chunk_str = f', {n_chunks} chunks' if n_chunks > 1 else ''
    print(f'Schedule: {k_str}{chunk_str}, makespan={ms} cyc, '
          f'{result["n_insns"]} insns')
    print(f'{"="*80}')

    if detail:
        print(f'\n{"Pos":>4s} {"Cyc":>4s} {"Stl":>3s} {"Pipe":>8s} {"Mnemonic":>8s}  '
              f'{"Comment"}')
        print('-' * 80)

        for pos, i in enumerate(order):
            insn = insns[i]
            cyc = sum(stalls[:pos + 1])
            pipe_s = insn.pipe.value
            print(f'{pos:4d} {cyc:4d} {stalls[pos]:3d} {pipe_s:>8s} '
                  f'{insn.mnemonic:>8s}  {insn.comment}')

    # Pipe summary
    pipe_insns: Dict[str, int] = {}
    pipe_cycles: Dict[str, int] = {}
    for i in order:
        insn = insns[i]
        pname = insn.pipe.value
        pipe_insns[pname] = pipe_insns.get(pname, 0) + 1
        pipe_cycles[pname] = pipe_cycles.get(pname, 0) + insn.throughput

    print(f'\nPipe utilization:')
    for pname in sorted(pipe_cycles.keys(), key=lambda x: -pipe_cycles[x]):
        cyc = pipe_cycles[pname]
        cnt = pipe_insns[pname]
        bar = '#' * min(cyc // 2, 40)
        print(f'  {pname:10s}: {cnt:3d} insns, {cyc:4d} tp-cyc  {bar}')

    print(f'\n  Makespan: {ms} cyc (includes latency hiding + interleaving)')

    # Compare to lower bounds
    bf16_cyc = pipe_cycles.get('BF16', 0)
    alu_cyc = pipe_cycles.get('ALU', 0)
    store_cyc = pipe_cycles.get('STORE', 0)
    load_cyc = pipe_cycles.get('LOAD', 0)
    lb = max(bf16_cyc, alu_cyc, store_cyc, load_cyc)
    print(f'  Throughput lower bound: {lb} cyc '
          f'(BF16={bf16_cyc}, ALU={alu_cyc}, STORE={store_cyc}, LOAD={load_cyc})')
    overhead = ms - lb
    print(f'  Scheduling overhead: {overhead} cyc ({overhead/max(lb, 1)*100:.0f}%)')

    if n_chunks > 1:
        # Per-chunk breakdown
        for c in range(n_chunks):
            c_insns = [i for i in order if insns[i].chunk_idx == c]
            c_store = sum(insns[i].throughput for i in c_insns
                         if insns[i].pipe == Pipe.STORE)
            c_bf16 = sum(insns[i].throughput for i in c_insns
                         if insns[i].pipe == Pipe.BF16)
            c_alu = sum(insns[i].throughput for i in c_insns
                        if insns[i].pipe == Pipe.ALU)
            print(f'  Chunk {c}: {len(c_insns)} insns '
                  f'(BF16={c_bf16}, ALU={c_alu}, STORE={c_store})')

        # Boundary tail (time from last STS to last boundary op)
        sts_positions = [pos for pos, i in enumerate(order)
                         if insns[i].mnemonic == 'STS.128']
        bnd_positions = [pos for pos, i in enumerate(order)
                         if insns[i].mnemonic in ('WARPSYNC', 'FENCE', 'UTMASTG', 'COMMIT')]
        if sts_positions and bnd_positions:
            last_sts_cyc = sum(stalls[:max(sts_positions) + 1])
            last_bnd_cyc = sum(stalls[:max(bnd_positions) + 1])
            print(f'  Boundary tail: {last_bnd_cyc - last_sts_cyc} cyc after last STS')


# ---------------------------------------------------------------------------
# Sweep modes
# ---------------------------------------------------------------------------

def sweep_k(n_chunks: int = 1, time_limit: float = 30.0,
            verbose: bool = True):
    """Sweep k=0..16 with fixed-k solver, report makespan curve."""
    results = []
    chunks_str = f', {n_chunks} chunks' if n_chunks > 1 else ''
    print(f'Sweeping k=0..16{chunks_str} (time limit {time_limit:.0f}s each)\n')

    for k in range(17):
        if n_chunks == 1:
            r = solve_chunk_fixed_k(k, time_limit=time_limit, verbose=verbose)
        else:
            r = solve_subiter_fixed_k(n_chunks, k, time_limit=time_limit, verbose=verbose)
        if r:
            results.append(r)

    if not results:
        print('No solutions found')
        return

    print(f'\n{"="*80}')
    print(f'SWEEP RESULTS')
    print(f'{"="*80}')
    print(f'\n{"k":>3s} {"Insns":>6s} {"Makespan":>9s} {"Status":>10s} '
          f'{"Time":>6s}')
    print('-' * 50)

    best = None
    for r in results:
        opt = '*' if r['status'] == 'OPTIMAL' else ' '
        print(f'{r["k_alu"]:3d} {r["n_insns"]:6d} {r["makespan"]:9d} '
              f'{r["status"]:>10s} {r["wall_time"]:5.1f}s{opt}')
        if best is None or r['makespan'] < best['makespan']:
            best = r

    print(f'\nBest: k={best["k_alu"]}, makespan={best["makespan"]} cyc '
          f'({best["n_insns"]} insns)')

    # Compare to pure strategies
    pure_bf16 = [r for r in results if r['k_alu'] == 0]
    if pure_bf16:
        reduction = (1 - best['makespan'] / pure_bf16[0]['makespan']) * 100
        print(f'  vs pure BF16 (k=0): {reduction:+.1f}%')

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Joint instruction selection + scheduling optimizer')
    parser.add_argument('--k', type=int, default=None,
                        help='Solve for specific k_alu (0-16)')
    parser.add_argument('--joint', action='store_true',
                        help='Full joint optimization (k as CP-SAT variable)')
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep k=0..16 with fixed-k solver (default)')
    parser.add_argument('--chunks', type=int, default=1,
                        help='Chunks per sub-iteration (1=Stage A, 2=Stage B)')
    parser.add_argument('--time', type=float, default=None,
                        help='Solver time limit in seconds (default: auto)')
    parser.add_argument('--detail', action='store_true',
                        help='Print detailed instruction schedule')
    args = parser.parse_args()

    nc = args.chunks

    # Auto time limits based on problem size
    if args.time is not None:
        tl = args.time
    elif args.joint:
        tl = 300.0 if nc == 1 else 600.0
    else:
        tl = 60.0

    if args.joint:
        if nc == 1:
            r = solve_chunk_joint(time_limit=tl, verbose=True)
        else:
            r = solve_subiter_joint(nc, time_limit=tl, verbose=True)
        if r:
            print_schedule(r, detail=args.detail)
    elif args.k is not None:
        if nc == 1:
            r = solve_chunk_fixed_k(args.k, time_limit=tl, verbose=True)
        else:
            r = solve_subiter_fixed_k(nc, args.k, time_limit=tl, verbose=True)
        if r:
            print_schedule(r, detail=args.detail)
    else:
        sweep_k(n_chunks=nc, time_limit=tl, verbose=True)


if __name__ == '__main__':
    main()
