#!/usr/bin/env python3
"""SASS scheduling analysis for SM 100a (Blackwell).

Parses cuobjdump --dump-sass output, decodes control words, builds register
dependency graphs, identifies scheduling slack.

Usage:
    # Parse and show annotated instruction listing
    python3 sass_analysis.py sass_dump.txt

    # Analyze specific address range (e.g., epilogue Phase 1)
    python3 sass_analysis.py sass_dump.txt --section 0x1300 0x1a70

    # Full dependency + slack analysis
    python3 sass_analysis.py sass_dump.txt --deps

    # Output JSON sidecar
    python3 sass_analysis.py sass_dump.txt --json analysis.json

    # Generate calibration kernels to verify control word bit layout
    python3 sass_analysis.py --calibrate

    # Analyze from cubin directly (runs cuobjdump internally)
    python3 sass_analysis.py --cubin siglip_vision

Prerequisites:
    Phase 0 (calibration): Compile & run calibration kernels on B200 to verify
    control word bit layout. Without calibration, stall counts use assumed
    Ampere-style layout — may be wrong for SM 100a.
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
# Control word decoder — configurable bit layout
# Default: Ampere-style (SM 80). Must be calibrated for SM 100a.
# ══════════════════════════════════════════════════════════════════════════════

CTRL_LAYOUT = {
    'stall':     (0, 4),    # bits [3:0]   — stall count (0-15 cycles)
    'yield':     (4, 1),    # bit  [4]     — yield hint
    'wr_bar':    (5, 5),    # bits [9:5]   — write barrier index (0-5, 31=none)
    'rd_bar':    (10, 5),   # bits [14:10] — read barrier index (0-5, 31=none)
    'wait_mask': (15, 6),   # bits [20:15] — wait barrier mask (6 bits)
    'reuse':     (21, 2),   # bits [22:21] — register reuse flags
}

def decode_control(ctrl_word: int, layout: dict = CTRL_LAYOUT) -> dict:
    """Extract scheduling fields from 64-bit control word."""
    result = {}
    for name, (start, width) in layout.items():
        mask = (1 << width) - 1
        result[name] = (ctrl_word >> start) & mask
    return result

# ══════════════════════════════════════════════════════════════════════════════
# Instruction latency table (initial estimates — refine with microbenchmarks)
# These are ISSUE latencies (cycles until result available for dependent op).
# Long-latency ops (LDTM, MMA, TMA) use barrier sync, not stall counts.
# ══════════════════════════════════════════════════════════════════════════════

LATENCY = {
    # Integer ALU
    'IMAD':     4,
    'IADD3':    4,
    'LEA':      4,
    'LOP3':     4,
    'SHF':      4,
    'PRMT':     4,
    'MOV':      2,
    'SEL':      4,
    'IMNMX':    4,
    'ISETP':    4,
    'ISET':     4,

    # Float ALU
    'FADD':     4,
    'FMUL':     4,
    'FFMA':     4,
    'MUFU':     8,    # special function unit — higher latency

    # Conversion
    'F2FP':     4,    # FP32→BF16 / precision changes
    'I2FP':     4,
    'F2IP':     4,
    'I2IP':     4,

    # Shared memory
    'STS':      4,    # fire-and-forget store (write port occupancy)
    'LDS':      20,   # shared load — ~20 cycle latency

    # TMEM (barrier-synchronized, stall count typically 0)
    'LDTM':     0,    # result via DEPBAR/wait, not stall
    'STTM':     0,

    # TMA (async, barrier-synchronized)
    'UTCBAR':   0,    # tensor core barrier

    # Special registers
    'S2R':      20,   # special reg read (often scoreboard-tracked)
    'S2UR':     20,
    'CS2R':     20,
    'R2UR':     8,    # cross-domain transfer

    # Constants
    'LDC':      0,    # constant load (scoreboard-tracked)
    'LDCU':     0,

    # Barriers / sync
    'BAR':      0,    # barrier sync
    'DEPBAR':   0,    # dependency barrier
    'BSYNC':    0,
    'WARPSYNC': 0,
    'NANOSLEEP': 0,
    'MEMBAR':   0,
    'ERRBAR':   0,
    'CGAERRBAR': 0,

    # Control flow
    'BRA':      0,
    'EXIT':     0,
    'RET':      0,
    'BREAK':    0,
    'CALL':     0,

    # MMA (barrier-synchronized)
    'TCGEN05':  0,

    # Default for unknown
    '_DEFAULT': 4,
}

def get_latency(opcode: str) -> int:
    """Get estimated issue latency for an opcode."""
    # Try exact match first
    base = opcode.split('.')[0]
    if base in LATENCY:
        return LATENCY[base]
    return LATENCY['_DEFAULT']

# ══════════════════════════════════════════════════════════════════════════════
# Instruction categories
# ══════════════════════════════════════════════════════════════════════════════

CATEGORIES = {
    'tmem_load':  {'LDTM'},
    'tmem_store': {'STTM'},
    'cvt':        {'F2FP', 'I2FP', 'F2IP', 'I2IP'},
    'sts':        {'STS'},
    'lds':        {'LDS'},
    'alu_int':    {'IMAD', 'IADD3', 'LEA', 'LOP3', 'SHF', 'PRMT', 'MOV', 'SEL', 'IMNMX'},
    'alu_fp':     {'FADD', 'FMUL', 'FFMA', 'MUFU'},
    'cmp':        {'ISETP', 'ISET', 'FSETP'},
    'mma':        {'TCGEN05'},
    'tma':        {'UTCBAR'},
    'barrier':    {'BAR', 'DEPBAR', 'BSYNC', 'WARPSYNC', 'NANOSLEEP', 'MEMBAR', 'ERRBAR', 'CGAERRBAR'},
    'const':      {'LDC', 'LDCU'},
    'special':    {'S2R', 'S2UR', 'CS2R', 'R2UR'},
    'control':    {'BRA', 'EXIT', 'RET', 'BREAK', 'CALL'},
}

def categorize(opcode: str) -> str:
    base = opcode.split('.')[0]
    for cat, ops in CATEGORIES.items():
        if base in ops:
            return cat
    return 'other'

# ══════════════════════════════════════════════════════════════════════════════
# Register read/write analysis
# ══════════════════════════════════════════════════════════════════════════════

# Number of consecutive registers written by LDTM variants
LDTM_WIDTHS = {'x8': 8, 'x16': 16, 'x32': 32, 'x64': 64}

# Regex patterns for operand parsing
RE_REG = re.compile(r'\bR(\d+)\b')
RE_UREG = re.compile(r'\bUR(\d+)\b')
RE_PRED = re.compile(r'\bP(\d+)\b')
RE_PRED_GUARD = re.compile(r'^@(!?)P(\d+)$')

def parse_reg_sets(opcode: str, operands: str, predicate: str) -> tuple[set, set]:
    """Parse registers read and written by an instruction.

    Returns (regs_read, regs_written) as sets of strings like 'R4', 'UR6', 'P0'.
    This is conservative: may over-report reads but tries to be accurate on writes.
    """
    reads = set()
    writes = set()
    base = opcode.split('.')[0]
    parts = [p.strip() for p in operands.split(',')]

    # Predicate guard — always a read
    if predicate:
        m = RE_PRED_GUARD.match(predicate)
        if m:
            reads.add(f'P{m.group(2)}')

    # --- LDTM: writes N consecutive regs from destination ---
    if base == 'LDTM':
        width = 1
        for suffix, w in LDTM_WIDTHS.items():
            if suffix in opcode:
                width = w
                break
        if parts:
            m = RE_REG.search(parts[0])
            if m:
                base_reg = int(m.group(1))
                for i in range(width):
                    writes.add(f'R{base_reg + i}')
        # tmem address is a read (uniform reg)
        for p in parts[1:]:
            for m in RE_UREG.finditer(p):
                reads.add(f'UR{m.group(1)}')
        return reads, writes

    # --- STS: store to shared memory, reads data + addr regs, writes nothing ---
    if base == 'STS':
        # STS [Raddr+offset], Rdata
        # STS.128 [Raddr], Rdata (4 consecutive regs)
        for p in parts:
            for m in RE_REG.finditer(p):
                reads.add(f'R{m.group(1)}')
            for m in RE_UREG.finditer(p):
                reads.add(f'UR{m.group(1)}')
        # Handle .128 = 4 regs, .64 = 2 regs from data operand
        if len(parts) >= 2:
            dm = RE_REG.search(parts[-1])
            if dm:
                base_reg = int(dm.group(1))
                if '.128' in opcode:
                    for i in range(1, 4):
                        reads.add(f'R{base_reg + i}')
                elif '.64' in opcode:
                    reads.add(f'R{base_reg + 1}')
        return reads, writes

    # --- LDS: load from shared memory, reads addr, writes dest ---
    if base == 'LDS':
        if parts:
            m = RE_REG.search(parts[0])
            if m:
                base_reg = int(m.group(1))
                writes.add(f'R{base_reg}')
                if '.128' in opcode:
                    for i in range(1, 4):
                        writes.add(f'R{base_reg + i}')
                elif '.64' in opcode:
                    writes.add(f'R{base_reg + 1}')
        for p in parts[1:]:
            for m in RE_REG.finditer(p):
                reads.add(f'R{m.group(1)}')
            for m in RE_UREG.finditer(p):
                reads.add(f'UR{m.group(1)}')
        return reads, writes

    # --- ISETP / FSETP: writes predicate, reads regs ---
    if base in ('ISETP', 'FSETP', 'ISET'):
        # ISETP.cond P0, PT, R23, 0x1f, PT
        for p in parts:
            pm = re.match(r'^\s*!?P(\d+)\s*$', p)
            if pm:
                # First predicate is write dest, second is write dest2
                if f'P{pm.group(1)}' not in writes and p.strip() != 'PT':
                    writes.add(f'P{pm.group(1)}')
                continue
            if p.strip() == 'PT':
                continue
            for m in RE_REG.finditer(p):
                reads.add(f'R{m.group(1)}')
            for m in RE_UREG.finditer(p):
                reads.add(f'UR{m.group(1)}')
        return reads, writes

    # --- Control flow: reads predicate (if guarded), no reg writes ---
    if base in ('BRA', 'EXIT', 'RET', 'BREAK', 'BSYNC', 'CALL'):
        return reads, writes

    # --- Barriers / sync: no reg read/write ---
    if base in ('BAR', 'DEPBAR', 'WARPSYNC', 'NANOSLEEP', 'MEMBAR', 'ERRBAR',
                'CGAERRBAR', 'UTCBAR'):
        # Some barrier ops reference registers
        for p in parts:
            for m in RE_REG.finditer(p):
                reads.add(f'R{m.group(1)}')
            for m in RE_UREG.finditer(p):
                reads.add(f'UR{m.group(1)}')
        return reads, writes

    # --- TCGEN05 (MMA): complex, but barrier-synchronized ---
    if base == 'TCGEN05':
        for p in parts:
            for m in RE_REG.finditer(p):
                reads.add(f'R{m.group(1)}')
            for m in RE_UREG.finditer(p):
                reads.add(f'UR{m.group(1)}')
        return reads, writes

    # --- S2R / CS2R: writes dest reg, no reg reads ---
    if base in ('S2R', 'CS2R'):
        if parts:
            m = RE_REG.search(parts[0])
            if m:
                writes.add(f'R{m.group(1)}')
                if '.32' not in opcode and '.64' in opcode:
                    writes.add(f'R{int(m.group(1)) + 1}')
        return reads, writes

    # --- S2UR: writes uniform dest ---
    if base == 'S2UR':
        if parts:
            m = RE_UREG.search(parts[0])
            if m:
                writes.add(f'UR{m.group(1)}')
        return reads, writes

    # --- R2UR: reads reg, writes uniform reg ---
    if base == 'R2UR':
        if len(parts) >= 2:
            m = RE_UREG.search(parts[0])
            if m:
                writes.add(f'UR{m.group(1)}')
            m = RE_REG.search(parts[1])
            if m:
                reads.add(f'R{m.group(1)}')
        return reads, writes

    # --- Default: first operand is dest (write), rest are sources (read) ---
    if parts:
        # Destination
        dm = RE_REG.search(parts[0])
        if dm:
            writes.add(f'R{dm.group(1)}')
        dm = RE_UREG.search(parts[0])
        if dm:
            writes.add(f'UR{dm.group(1)}')
        # Sources
        for p in parts[1:]:
            if p.strip() in ('RZ', 'URZ', 'PT', '!PT'):
                continue
            for m in RE_REG.finditer(p):
                reads.add(f'R{m.group(1)}')
            for m in RE_UREG.finditer(p):
                reads.add(f'UR{m.group(1)}')

    return reads, writes

# ══════════════════════════════════════════════════════════════════════════════
# SASS parser
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Instruction:
    addr: int
    opcode: str
    operands: str
    predicate: str             # '@P0', '@!P0', or ''
    encoding: int              # 64-bit instruction word
    control: int               # 64-bit control word
    ctrl_fields: dict          # decoded control fields
    regs_read: set = field(default_factory=set)
    regs_written: set = field(default_factory=set)
    category: str = ''
    latency: int = 0

    def __post_init__(self):
        if not self.category:
            self.category = categorize(self.opcode)
        if not self.latency:
            self.latency = get_latency(self.opcode)

    @property
    def stall(self) -> int:
        return self.ctrl_fields.get('stall', -1)

    def short(self) -> str:
        pred = f'{self.predicate} ' if self.predicate else ''
        return f'{pred}{self.opcode} {self.operands}'


def parse_sass(text: str, kernel_name: str = None) -> dict[str, list[Instruction]]:
    """Parse cuobjdump --dump-sass output into structured instructions.

    Returns dict mapping kernel function name → list of Instructions.
    """
    kernels = {}
    current_kernel = None
    instructions = []

    # State machine: collect pairs of hex values per instruction
    pending_addr = None
    pending_pred = ''
    pending_op = None
    pending_operands = ''
    pending_enc = None

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # Kernel function header
        fn_match = re.match(r'\s*Function\s*:\s*(\S+)', line)
        if fn_match:
            if current_kernel and instructions:
                kernels[current_kernel] = instructions
            current_kernel = fn_match.group(1)
            instructions = []
            pending_addr = None
            pending_enc = None
            i += 1
            continue

        # Instruction line: /*ADDR*/  [@pred] OPCODE operands ;  /* 0xENCODING */
        instr_match = re.match(
            r'\s*/\*([0-9a-fA-F]+)\*/\s+'
            r'(?:(@[!]?P\d+)\s+)?'
            r'(\S+?)(?:\s+(.*?))?\s*;\s*'
            r'/\*\s*(0x[0-9a-fA-F]+)\s*\*/',
            line
        )
        if instr_match:
            pending_addr = int(instr_match.group(1), 16)
            pending_pred = instr_match.group(2) or ''
            pending_op = instr_match.group(3)
            pending_operands = (instr_match.group(4) or '').strip()
            pending_enc = int(instr_match.group(5), 16)
            i += 1
            continue

        # Control word line (follows instruction): /* 0xCONTROL */
        ctrl_match = re.match(r'\s*/\*\s*(0x[0-9a-fA-F]+)\s*\*/', line)
        if ctrl_match and pending_enc is not None:
            ctrl = int(ctrl_match.group(1), 16)
            ctrl_fields = decode_control(ctrl)
            regs_r, regs_w = parse_reg_sets(pending_op, pending_operands, pending_pred)
            instr = Instruction(
                addr=pending_addr,
                opcode=pending_op,
                operands=pending_operands,
                predicate=pending_pred,
                encoding=pending_enc,
                control=ctrl,
                ctrl_fields=ctrl_fields,
                regs_read=regs_r,
                regs_written=regs_w,
            )
            instructions.append(instr)
            pending_enc = None
            pending_addr = None
            i += 1
            continue

        i += 1

    # Flush last kernel
    if current_kernel and instructions:
        kernels[current_kernel] = instructions

    # Filter to requested kernel
    if kernel_name:
        if kernel_name in kernels:
            return {kernel_name: kernels[kernel_name]}
        # Try substring match
        for k in kernels:
            if kernel_name in k:
                return {k: kernels[k]}
        print(f'Warning: kernel "{kernel_name}" not found. Available: {list(kernels.keys())}',
              file=sys.stderr)

    return kernels

# ══════════════════════════════════════════════════════════════════════════════
# Dependency graph
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DepEdge:
    src: int       # index of producer instruction
    dst: int       # index of consumer instruction
    reg: str       # register causing dependency
    dep_type: str  # 'RAW', 'WAR', 'WAW'
    latency: int   # expected latency of producer


def build_dep_graph(instrs: list[Instruction], raw_only: bool = True) -> list[DepEdge]:
    """Build dependency edges between instructions.

    For slack analysis, RAW (read-after-write) dependencies are primary.
    WAR/WAW matter for reordering feasibility but not for stall analysis.
    """
    edges = []
    # Map: register → index of last instruction that wrote it
    last_writer = {}
    # Map: register → index of last instruction that read it (for WAR)
    last_readers = defaultdict(list)

    for i, instr in enumerate(instrs):
        # RAW: I read a register that was written by an earlier instruction
        for reg in instr.regs_read:
            if reg in last_writer:
                j = last_writer[reg]
                edges.append(DepEdge(j, i, reg, 'RAW', instrs[j].latency))

        if not raw_only:
            # WAR: I write a register that was read by an earlier instruction
            for reg in instr.regs_written:
                for j in last_readers.get(reg, []):
                    edges.append(DepEdge(j, i, reg, 'WAR', 0))

            # WAW: I write a register that was written by an earlier instruction
            for reg in instr.regs_written:
                if reg in last_writer:
                    j = last_writer[reg]
                    edges.append(DepEdge(j, i, reg, 'WAW', 0))

        # Update tracking
        for reg in instr.regs_written:
            last_writer[reg] = i
            last_readers[reg] = []
        for reg in instr.regs_read:
            last_readers[reg].append(i)

    return edges


def compute_slack(instrs: list[Instruction], edges: list[DepEdge]) -> list[dict]:
    """Compute per-instruction slack: actual_stall - min_required_stall.

    min_required_stall = max over all RAW predecessors of:
        max(0, pred_latency - distance_in_instructions_between_them)

    This is a simplified model. Real hardware has:
    - Scoreboard tracking (DEPBAR) for long-latency ops
    - Multiple execution pipes (independent ops don't consume stall budget)
    - Warp-level scheduling (stall = warp stall, not thread stall)

    Instructions with non-zero wait_mask are flagged as 'barrier_wait' — their
    stall cycles are (partially) from scoreboard waits, not scheduling slack.
    """
    n = len(instrs)
    # Build predecessor map: for each instruction, list of (pred_idx, latency)
    preds = defaultdict(list)
    for e in edges:
        if e.dep_type == 'RAW':
            preds[e.dst].append((e.src, e.latency))

    results = []
    for i, instr in enumerate(instrs):
        actual = instr.stall
        wait_mask = instr.ctrl_fields.get('wait_mask', 0)
        has_barrier_wait = wait_mask != 0

        # Compute minimum stall needed to satisfy all RAW dependencies
        min_stall = 0
        limiting_dep = None
        for pred_idx, lat in preds[i]:
            # Distance = number of instructions between pred and this one
            # Each intervening instruction consumes at least 1 cycle (stall=0 → 1 cycle issue)
            distance = i - pred_idx - 1
            # Account for intervening stalls (they also consume cycles)
            intervening_stalls = sum(instrs[k].stall for k in range(pred_idx + 1, i))
            effective_distance = distance + intervening_stalls
            needed = max(0, lat - effective_distance)
            if needed > min_stall:
                min_stall = needed
                limiting_dep = (pred_idx, instrs[pred_idx].opcode)

        raw_slack = max(0, actual - min_stall) if actual >= 0 else -1

        # Classify slack source
        if has_barrier_wait and raw_slack > 0:
            # Stall is (at least partially) from scoreboard wait, not pure scheduling slack
            slack_type = 'barrier_wait'
        elif raw_slack > 0:
            slack_type = 'scheduling'
        else:
            slack_type = 'none'

        results.append({
            'idx': i,
            'addr': instr.addr,
            'opcode': instr.opcode,
            'category': instr.category,
            'actual_stall': actual,
            'min_stall': min_stall,
            'slack': raw_slack,
            'slack_type': slack_type,
            'wait_mask': wait_mask,
            'limiting_dep': limiting_dep,
        })

    return results

# ══════════════════════════════════════════════════════════════════════════════
# Critical path analysis
# ══════════════════════════════════════════════════════════════════════════════

def critical_path(instrs: list[Instruction], edges: list[DepEdge]) -> tuple[int, list[int]]:
    """Find the longest dependency chain (critical path) through the instruction stream.

    Returns (total_cycles, path_indices).
    """
    n = len(instrs)
    # dp[i] = longest path ending at instruction i (in cycles)
    dp = [0] * n
    parent = [-1] * n

    # Edges grouped by destination
    preds = defaultdict(list)
    for e in edges:
        if e.dep_type == 'RAW':
            preds[e.dst].append(e)

    for i in range(n):
        best = 0
        best_parent = -1
        for e in preds[i]:
            cost = dp[e.src] + e.latency
            if cost > best:
                best = cost
                best_parent = e.src
        dp[i] = best
        parent[i] = best_parent

    # Find the end of the critical path
    end = max(range(n), key=lambda i: dp[i])
    total = dp[end]

    # Trace back
    path = []
    cur = end
    while cur >= 0:
        path.append(cur)
        cur = parent[cur]
    path.reverse()

    return total, path

# ══════════════════════════════════════════════════════════════════════════════
# Output formatting
# ══════════════════════════════════════════════════════════════════════════════

def print_listing(instrs: list[Instruction], slack_data: list[dict] = None,
                  section: tuple[int, int] = None):
    """Print annotated instruction listing."""
    slack_map = {}
    if slack_data:
        for s in slack_data:
            slack_map[s['idx']] = s

    # Header
    if slack_data:
        print(f'{"ADDR":>6s}  {"STALL":>5s} {"MIN":>3s} {"SLACK":>5s}  '
              f'{"CAT":<12s}  INSTRUCTION')
        print('-' * 100)
    else:
        print(f'{"ADDR":>6s}  {"STALL":>5s} {"YIELD":>5s} {"WRBAR":>5s} '
              f'{"RDBAR":>5s} {"WAIT":>6s}  {"CAT":<12s}  INSTRUCTION')
        print('-' * 110)

    for i, instr in enumerate(instrs):
        if section:
            if instr.addr < section[0] or instr.addr > section[1]:
                continue

        addr_s = f'0x{instr.addr:04x}'
        cat = instr.category

        if slack_data and i in slack_map:
            s = slack_map[i]
            stall_s = f'{s["actual_stall"]:>5d}'
            min_s = f'{s["min_stall"]:>3d}'
            slack_s = f'{s["slack"]:>5d}' if s['slack'] > 0 else f'{"":>5s}'
            if s['slack'] >= 2 and s['slack_type'] == 'scheduling':
                marker = ' <<< SCHED'
            elif s['slack'] >= 2 and s['slack_type'] == 'barrier_wait':
                marker = f' [bar 0b{s["wait_mask"]:06b}]'
            else:
                marker = ''
            print(f'{addr_s}  {stall_s} {min_s} {slack_s}  {cat:<12s}  '
                  f'{instr.short()}{marker}')
        else:
            cf = instr.ctrl_fields
            stall_s = f'{cf["stall"]:>5d}'
            yield_s = f'{cf["yield"]:>5d}'
            wrbar_s = f'{cf["wr_bar"]:>5d}'
            rdbar_s = f'{cf["rd_bar"]:>5d}'
            wait_s = f'0b{cf["wait_mask"]:06b}'
            print(f'{addr_s}  {stall_s} {yield_s} {wrbar_s} {rdbar_s} {wait_s}  '
                  f'{cat:<12s}  {instr.short()}')


def print_summary(instrs: list[Instruction], slack_data: list[dict],
                  section: tuple[int, int] = None):
    """Print summary statistics."""
    # Filter to section if specified
    if section:
        indices = [i for i, ins in enumerate(instrs)
                   if section[0] <= ins.addr <= section[1]]
        filtered_slack = [s for s in slack_data if s['idx'] in set(indices)]
    else:
        filtered_slack = slack_data

    total_stalls = sum(s['actual_stall'] for s in filtered_slack if s['actual_stall'] >= 0)
    total_min = sum(s['min_stall'] for s in filtered_slack)
    total_slack = sum(s['slack'] for s in filtered_slack if s['slack'] > 0)
    sched_slack = sum(s['slack'] for s in filtered_slack
                      if s['slack'] > 0 and s['slack_type'] == 'scheduling')
    barrier_slack = sum(s['slack'] for s in filtered_slack
                        if s['slack'] > 0 and s['slack_type'] == 'barrier_wait')
    n_instructions = len(filtered_slack)

    print(f'\n{"=" * 60}')
    print('SCHEDULING ANALYSIS SUMMARY')
    print('  Decoder: assumed Ampere-style bit layout (NOT calibrated)')
    print('  bits [3:0]=stall appears correct; barrier fields unverified')
    print('  Run: python3 sass_analysis.py --calibrate > calibration.cu')
    if section:
        print(f'Section: 0x{section[0]:04x} - 0x{section[1]:04x}')
    print(f'{"=" * 60}')
    print(f'Instructions:       {n_instructions}')
    print(f'Total stall cycles: {total_stalls}')
    print(f'Min stall cycles:   {total_min} (RAW dependency-limited)')
    if total_stalls > 0:
        print(f'Total slack:        {total_slack} cycles ({100*total_slack/total_stalls:.1f}% of stalls)')
    else:
        print(f'Total slack:        {total_slack} cycles')
    print(f'  Scheduling slack: {sched_slack} cycles (ptxas conservative — recoverable)')
    print(f'  Barrier waits:    {barrier_slack} cycles (scoreboard waits — NOT recoverable)')

    # Category breakdown
    cat_counts = defaultdict(lambda: {'count': 0, 'stalls': 0, 'sched_slack': 0, 'bar_slack': 0})
    for s in filtered_slack:
        cat = s['category']
        cat_counts[cat]['count'] += 1
        cat_counts[cat]['stalls'] += s['actual_stall'] if s['actual_stall'] >= 0 else 0
        if s['slack'] > 0:
            if s['slack_type'] == 'scheduling':
                cat_counts[cat]['sched_slack'] += s['slack']
            else:
                cat_counts[cat]['bar_slack'] += s['slack']

    print(f'\n{"Category":<14s} {"Count":>6s} {"Stalls":>7s} {"SchedSlk":>8s} {"BarSlk":>6s}')
    print('-' * 44)
    for cat in sorted(cat_counts, key=lambda c: -cat_counts[c]['stalls']):
        cc = cat_counts[cat]
        print(f'{cat:<14s} {cc["count"]:>6d} {cc["stalls"]:>7d} '
              f'{cc["sched_slack"]:>8d} {cc["bar_slack"]:>6d}')

    # Top slack hotspots (scheduling only first, then barrier)
    sched_hotspots = sorted([s for s in filtered_slack
                             if s['slack'] > 0 and s['slack_type'] == 'scheduling'],
                            key=lambda s: -s['slack'])
    barrier_hotspots = sorted([s for s in filtered_slack
                               if s['slack'] > 0 and s['slack_type'] == 'barrier_wait'],
                              key=lambda s: -s['slack'])

    if sched_hotspots:
        print(f'\nScheduling slack hotspots (recoverable via reordering):')
        for h in sched_hotspots[:10]:
            dep_s = ''
            if h['limiting_dep']:
                dep_s = f'  (dep: 0x{instrs[h["limiting_dep"][0]].addr:04x} {h["limiting_dep"][1]})'
            print(f'  0x{h["addr"]:04x}  {h["opcode"]:<20s}  '
                  f'stall={h["actual_stall"]} min={h["min_stall"]} '
                  f'slack={h["slack"]}{dep_s}')

    if barrier_hotspots:
        print(f'\nBarrier wait stalls (scoreboard, not scheduling slack):')
        for h in barrier_hotspots[:10]:
            wm = h.get('wait_mask', 0)
            print(f'  0x{h["addr"]:04x}  {h["opcode"]:<20s}  '
                  f'stall={h["actual_stall"]} wait_mask=0b{wm:06b}')


def make_json(kernel_name: str, instrs: list[Instruction], slack_data: list[dict],
              cp_cycles: int, cp_path: list[int],
              section: tuple[int, int] = None) -> dict:
    """Build JSON output for machine consumption."""
    if section:
        indices = set(i for i, ins in enumerate(instrs)
                      if section[0] <= ins.addr <= section[1])
        filtered = [s for s in slack_data if s['idx'] in indices]
    else:
        filtered = slack_data

    total_stalls = sum(s['actual_stall'] for s in filtered if s['actual_stall'] >= 0)
    total_slack = sum(s['slack'] for s in filtered if s['slack'] > 0)

    cat_counts = defaultdict(int)
    for s in filtered:
        cat_counts[s['category']] += 1

    hotspots = sorted([s for s in filtered if s['slack'] > 0], key=lambda s: -s['slack'])

    result = {
        'kernel': kernel_name,
        'ctrl_layout': CTRL_LAYOUT,
        'ctrl_layout_calibrated': False,
        'latency_table_source': 'estimated_defaults',
        'n_instructions': len(filtered),
        'total_stall_cycles': total_stalls,
        'min_stall_cycles': sum(s['min_stall'] for s in filtered),
        'total_slack': total_slack,
        'slack_pct': round(100 * total_slack / total_stalls, 1) if total_stalls > 0 else 0,
        'critical_path_cycles': cp_cycles,
        'critical_path_length': len(cp_path),
        'category_counts': dict(cat_counts),
        'hotspots': [
            {
                'addr': f'0x{h["addr"]:04x}',
                'opcode': h['opcode'],
                'actual_stall': h['actual_stall'],
                'min_stall': h['min_stall'],
                'slack': h['slack'],
            }
            for h in hotspots[:20]
        ],
    }
    if section:
        result['section'] = {'start': f'0x{section[0]:04x}', 'end': f'0x{section[1]:04x}'}

    return result

# ══════════════════════════════════════════════════════════════════════════════
# Calibration kernel generator
# ══════════════════════════════════════════════════════════════════════════════

CALIBRATION_SOURCE = r'''// SASS control word calibration kernels for SM 100a
// Compile: nvcc -arch=sm_100a -O3 calibration.cu -o calibration
// Dump:    cuobjdump --dump-sass calibration > calibration_sass.txt
// Analyze: python3 sass_analysis.py calibration_sass.txt
//
// Purpose: Each kernel has a known dependency pattern. By examining the
// control words in the SASS output, we can identify which bits encode
// the stall count and verify the decoder.

#include <cstdio>
#include <cstdint>

// Kernel A: Independent operations — expect stall=0 between them
extern "C" __global__ void cal_independent(int* out) {
    int a = 1, b = 2, c = 3, d = 4;
    int e = 5, f = 6, g = 7, h = 8;
    // 8 independent adds — no data dependencies
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(b));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(c));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(d));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(e));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(f));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(g));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(h));
    out[threadIdx.x] = a + b + c + d + e + f + g + h;
}

// Kernel B: Tight RAW dependency chain — expect stall=latency between each
extern "C" __global__ void cal_dependent(int* out) {
    int a = threadIdx.x;
    // Each add depends on the previous result
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    out[threadIdx.x] = a;
}

// Kernel C: RAW with 1 independent op between — expect stall = max(0, latency-1)
extern "C" __global__ void cal_gap1(int* out) {
    int a = threadIdx.x, b = threadIdx.x + 1;
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));  // write a
    asm volatile("add.s32 %0, %0, 1;" : "+r"(b));  // independent (gap=1)
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));  // read a (stall = lat-1)
    asm volatile("add.s32 %0, %0, 1;" : "+r"(b));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(b));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(b));
    out[threadIdx.x] = a + b;
}

// Kernel D: RAW with 2 independent ops between — expect stall = max(0, latency-2)
extern "C" __global__ void cal_gap2(int* out) {
    int a = threadIdx.x, b = threadIdx.x + 1, c = threadIdx.x + 2;
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));  // write a
    asm volatile("add.s32 %0, %0, 1;" : "+r"(b));  // independent
    asm volatile("add.s32 %0, %0, 1;" : "+r"(c));  // independent (gap=2)
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));  // read a (stall = lat-2)
    asm volatile("add.s32 %0, %0, 1;" : "+r"(b));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(c));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(b));
    asm volatile("add.s32 %0, %0, 1;" : "+r"(c));
    out[threadIdx.x] = a + b + c;
}

// Kernel E: Shared memory store — measure STS issue latency
extern "C" __global__ void cal_sts(int* out) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    int a = tid, b = tid + 1;
    // STS is fire-and-forget — stall should be minimal
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(tid * 4), "r"(a));
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(tid * 4 + 1024), "r"(b));
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(tid * 4), "r"(a));
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(tid * 4 + 1024), "r"(b));
    __syncthreads();
    out[tid] = smem[tid];
}

// Kernel F: F2FP (float→bf16 conversion) latency
extern "C" __global__ void cal_f2fp(float* out) {
    float a = (float)threadIdx.x;
    unsigned r;
    // Dependent chain: convert, extract, use as input again
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(r) : "f"(a));
    asm volatile("mov.b32 %0, %1;" : "=f"(a) : "r"(r));
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(r) : "f"(a));
    asm volatile("mov.b32 %0, %1;" : "=f"(a) : "r"(r));
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(r) : "f"(a));
    asm volatile("mov.b32 %0, %1;" : "=f"(a) : "r"(r));
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(r) : "f"(a));
    asm volatile("mov.b32 %0, %1;" : "=f"(a) : "r"(r));
    out[threadIdx.x] = a;
}

// Kernel G: PRMT (permute) latency
extern "C" __global__ void cal_prmt(int* out) {
    unsigned a = threadIdx.x, b = threadIdx.x + 1;
    asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(a) : "r"(b));
    asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(a) : "r"(b));
    asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(a) : "r"(b));
    asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(a) : "r"(b));
    asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(a) : "r"(b));
    asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(a) : "r"(b));
    asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(a) : "r"(b));
    asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(a) : "r"(b));
    out[threadIdx.x] = a;
}

int main() {
    int *d_out;
    float *d_fout;
    cudaMalloc(&d_out, 256 * sizeof(int));
    cudaMalloc(&d_fout, 256 * sizeof(float));

    cal_independent<<<1, 32>>>(d_out);
    cal_dependent<<<1, 32>>>(d_out);
    cal_gap1<<<1, 32>>>(d_out);
    cal_gap2<<<1, 32>>>(d_out);
    cal_sts<<<1, 32>>>(d_out);
    cal_f2fp<<<1, 32>>>(d_fout);
    cal_prmt<<<1, 32>>>(d_out);

    cudaDeviceSynchronize();
    printf("Calibration kernels launched. Dump SASS with:\n");
    printf("  cuobjdump --dump-sass calibration > calibration_sass.txt\n");
    printf("  python3 sass_analysis.py calibration_sass.txt\n");

    cudaFree(d_out);
    cudaFree(d_fout);
    return 0;
}
'''


def print_calibration():
    """Print calibration CUDA source to stdout."""
    print(CALIBRATION_SOURCE)


# ══════════════════════════════════════════════════════════════════════════════
# Calibration comparison (SASS stall counts vs runtime measurements)
# ══════════════════════════════════════════════════════════════════════════════

CALIBRATION_KERNELS = [
    {'fn': 'k1_f2fp_throughput',     'label': 'K1: F2FP throughput',   'ops': ['F2FP'],                           'mode': 'throughput'},
    {'fn': 'k2_f2fp_latency',        'label': 'K2: F2FP latency',     'ops': ['F2FP'],                           'mode': 'latency'},
    {'fn': 'k3_f2fp_sts_conflict',   'label': 'K3: F2FP+STS',         'ops': ['F2FP', 'STS'],                    'mode': 'conflict'},
    {'fn': 'k4_hfma2_throughput',    'label': 'K4: HFMA2 throughput',  'ops': ['HFMA2', 'HMMA', 'HFMA'],         'mode': 'throughput'},
    {'fn': 'k5_hfma2_f2fp_conflict', 'label': 'K5: HFMA2+F2FP',       'ops': ['HFMA2', 'HMMA', 'HFMA', 'F2FP'], 'mode': 'conflict'},
    {'fn': 'k6_sts_throughput',      'label': 'K6: STS throughput',    'ops': ['STS'],                            'mode': 'throughput'},
    {'fn': 'k7a_iadd_independent',   'label': 'K7a: IADD indep',      'ops': ['IADD3', 'IADD', 'UIADD3'],       'mode': 'throughput'},
    {'fn': 'k7b_iadd_dependent',     'label': 'K7b: IADD dep',        'ops': ['IADD3', 'IADD', 'UIADD3'],       'mode': 'latency'},
    {'fn': 'k8_prmt_throughput',     'label': 'K8: PRMT throughput',   'ops': ['PRMT', 'UPRMT'],                 'mode': 'throughput'},
    {'fn': 'k9_f2fp_wide',           'label': 'K9: F2FP 32-wide',     'ops': ['F2FP'],                           'mode': 'throughput'},
]


def find_timed_region(instrs):
    """Find instruction indices between clock64() reads (CS2R)."""
    clock_idx = []
    for i, ins in enumerate(instrs):
        base = ins.opcode.split('.')[0]
        if base == 'CS2R':
            clock_idx.append(i)
        elif base == 'S2R' and 'CLOCK' in ins.operands.upper():
            clock_idx.append(i)
    if len(clock_idx) >= 2:
        return (clock_idx[-2] + 1, clock_idx[-1])
    return (-1, -1)


def extract_timed_stalls(instrs, target_ops, start, end):
    """Extract stall counts for target opcodes in a region."""
    by_op = defaultdict(list)
    for i in range(max(0, start), min(len(instrs), end)):
        base = instrs[i].opcode.split('.')[0]
        if base in target_ops:
            by_op[base].append(instrs[i].stall)
    result = {}
    for op, stalls in by_op.items():
        result[op] = {
            'count': len(stalls),
            'typical': max(set(stalls), key=stalls.count),
            'min': min(stalls),
            'max': max(stalls),
        }
    return result


def parse_runtime_output(text):
    """Parse calibration binary stdout into per-kernel measurements."""
    results = {}
    for line in text.splitlines():
        m = re.match(r'(K\d+[ab]?):\s+.*?\s{2,}(-?\d+)\s+([\d.]+)\s+([\d.]+)', line.strip())
        if m:
            results[m.group(1)] = {
                'total_cycles': int(m.group(2)),
                'cyc_per_instr': float(m.group(3)),
                'throughput': float(m.group(4)),
            }
    return results


def calibrate_compare(sass_text, runtime_text=None):
    """Compare SASS stall counts from calibration kernels vs runtime measurements."""
    kernels = parse_sass(sass_text)
    runtime = parse_runtime_output(runtime_text) if runtime_text else {}
    has_rt = bool(runtime)

    print()
    if has_rt:
        print("SASS vs RUNTIME CALIBRATION COMPARISON")
        print("=" * 78)
        print(f"{'Kernel':<25s} {'Op':<7s} {'SASS':>5s} {'Meas':>7s} {'Delta':>6s}  Verdict")
        print("-" * 78)
    else:
        print("SASS CALIBRATION STALL ANALYSIS (no runtime data)")
        print("=" * 65)
        print(f"{'Kernel':<25s} {'Op':<7s} {'Stall':>5s} {'N':>4s} {'Range':>7s}  Type")
        print("-" * 65)

    for kdef in CALIBRATION_KERNELS:
        # Find kernel in SASS (substring match on function name)
        found = None
        for kname in kernels:
            if kdef['fn'] in kname:
                found = kname
                break
        if not found:
            print(f"{kdef['label']:<25s}  (not found in SASS)")
            continue

        instrs = kernels[found]
        start, end = find_timed_region(instrs)
        if start < 0:
            print(f"{kdef['label']:<25s}  (no clock reads found)")
            continue

        stall_data = extract_timed_stalls(instrs, kdef['ops'], start, end)
        if not stall_data:
            found_ops = sorted(set(instrs[i].opcode.split('.')[0]
                                   for i in range(start, end)))
            print(f"{kdef['label']:<25s}  (target {kdef['ops']} not in timed region; "
                  f"found: {', '.join(found_ops[:8])})")
            continue

        prefix = re.match(r'K\d+[ab]?', kdef['label']).group(0)
        rt = runtime.get(prefix)

        for op, sd in stall_data.items():
            sass_stall = sd['typical']

            if has_rt and rt:
                measured = rt['cyc_per_instr']

                if kdef['mode'] == 'conflict':
                    # Conflict kernels: runtime is combined throughput, per-op delta is meaningless
                    print(f"{kdef['label']:<25s} {op:<7s} {sass_stall:>5d} {measured:>7.2f}         (conflict — see pipe analysis)")
                else:
                    delta = sass_stall - measured

                    if abs(delta) <= 0.5:
                        verdict = "MATCH"
                    elif delta > 0.5:
                        verdict = f"CONSERVATIVE (+{delta:.1f})"
                    else:
                        verdict = f"DECODER? ({delta:+.1f})"

                    print(f"{kdef['label']:<25s} {op:<7s} {sass_stall:>5d} {measured:>7.2f} {delta:>+6.1f}  {verdict}")
            else:
                rng = f"{sd['min']}-{sd['max']}" if sd['min'] != sd['max'] else str(sd['min'])
                print(f"{kdef['label']:<25s} {op:<7s} {sass_stall:>5d} {sd['count']:>4d} {rng:>7s}  {kdef['mode']}")

    # ── SASS-only mode: print guidance and exit ──
    if not has_rt:
        print()
        print("Expected patterns:")
        print("  Throughput kernels (K1,K4,K6,K7a,K8,K9): stall = pipe issue rate")
        print("  Latency kernels (K2,K7b): stall = result latency")
        print("  K7a stall << K7b stall: decoder reads stall field correctly")
        print("  K7a stall == K7b stall: decoder is WRONG (bits not at [3:0])")
        print()
        print("Next: run ./calibration on B200, then:")
        print("  python3 sass_analysis.py <sass> --calibrate-compare --runtime output.txt")
        return

    # ── Pipe conflict analysis ──
    print()
    print("=" * 78)
    print("PIPE CONFLICT ANALYSIS")
    print("-" * 78)

    k = {p: runtime.get(p, {}).get('cyc_per_instr') for p in
         ['K1', 'K3', 'K4', 'K5', 'K6', 'K7a', 'K7b', 'K9']}

    if k['K1'] and k['K6'] and k['K3']:
        overlap = max(k['K1'], k['K6'])
        serial = k['K1'] + k['K6']
        if k['K3'] <= overlap * 1.3:
            print(f"  F2FP vs STS:   DIFFERENT pipes  (K3={k['K3']:.2f} ~ max(K1,K6)={overlap:.2f})")
        elif k['K3'] >= serial * 0.8:
            print(f"  F2FP vs STS:   SAME pipe        (K3={k['K3']:.2f} ~ K1+K6={serial:.2f})")
        else:
            print(f"  F2FP vs STS:   PARTIAL overlap   (K3={k['K3']:.2f}, range [{overlap:.2f}, {serial:.2f}])")

    if k['K1'] and k['K4'] and k['K5']:
        overlap = max(k['K1'], k['K4'])
        serial = k['K1'] + k['K4']
        if k['K5'] <= overlap * 1.3:
            print(f"  HFMA2 vs F2FP: DIFFERENT pipes  (K5={k['K5']:.2f} ~ max(K1,K4)={overlap:.2f})")
        elif k['K5'] >= serial * 0.8:
            print(f"  HFMA2 vs F2FP: SAME pipe        (K5={k['K5']:.2f} ~ K1+K4={serial:.2f})")
        else:
            print(f"  HFMA2 vs F2FP: PARTIAL overlap   (K5={k['K5']:.2f}, range [{overlap:.2f}, {serial:.2f}])")

    if k['K1'] and k['K9']:
        ratio = k['K9'] / k['K1']
        if ratio > 1.3:
            print(f"  32-wide F2FP:  DEGRADED  (K9/K1 = {ratio:.2f}x)")
        else:
            print(f"  32-wide F2FP:  OK        (K9/K1 = {ratio:.2f}x)")

    # ── Decoder verification via K7a/K7b ──
    if k['K7a'] is not None and k['K7b'] is not None:
        lat = k['K7b'] - k['K7a']
        print()
        print("DECODER VERIFICATION")
        print("-" * 78)
        print(f"  IADD3 throughput (K7a) = {k['K7a']:.2f} cyc/instr")
        print(f"  IADD3 latency   (K7b) = {k['K7b']:.2f} cyc/instr")
        print(f"  Derived latency       = {lat:.2f} cycles")

        for kdef in CALIBRATION_KERNELS:
            if kdef['fn'] == 'k7b_iadd_dependent':
                for kname in kernels:
                    if kdef['fn'] in kname:
                        instrs = kernels[kname]
                        s, e = find_timed_region(instrs)
                        sd = extract_timed_stalls(instrs, kdef['ops'], s, e)
                        for op, d in sd.items():
                            if abs(d['typical'] - lat) <= 0.5:
                                print(f"  SASS dep stall={d['typical']} matches measured {lat:.1f}"
                                      f" -> DECODER CORRECT")
                            elif abs(d['typical'] - k['K7b']) <= 0.5:
                                print(f"  SASS dep stall={d['typical']} matches raw K7b {k['K7b']:.1f}"
                                      f" -> stall encodes total cost (not just latency)")
                            else:
                                print(f"  SASS dep stall={d['typical']} vs measured {lat:.1f}"
                                      f" -> MISMATCH (check bit layout)")

    # ── Production kernel implications ──
    if k['K1']:
        print()
        print("PRODUCTION KERNEL IMPLICATIONS")
        print("-" * 78)
        f2fp = k['K1']
        ptxas_stall = 15  # observed in production kernel SASS
        n_f2fp = 128      # F2FPs in production epilogue
        epi_budget = 6156  # cycles per epilogue tile
        print(f"  F2FP measured throughput = {f2fp:.2f} cyc/instr")
        print(f"  Production kernel: {n_f2fp} F2FPs with ptxas stall={ptxas_stall}")
        if f2fp < ptxas_stall - 0.5:
            headroom = ptxas_stall - round(f2fp)
            recoverable = headroom * n_f2fp
            pct = recoverable / epi_budget * 100
            print(f"  Headroom = ({ptxas_stall} - {round(f2fp)}) x {n_f2fp} = {recoverable} cycles")
            print(f"  Epilogue tile budget ~ {epi_budget} cycles -> {pct:.1f}% potential")
            if recoverable > 500:
                print(f"  VERDICT: Significant SASS scheduling headroom exists")
            else:
                print(f"  VERDICT: Modest headroom ({recoverable} cycles)")
        else:
            print(f"  ptxas stall={ptxas_stall} is correct -> NO scheduling headroom")

    # ── Suggested latency table updates ──
    suggestions = []
    for kdef in CALIBRATION_KERNELS:
        if kdef['mode'] != 'throughput':
            continue
        prefix = re.match(r'K\d+[ab]?', kdef['label']).group(0)
        rt = runtime.get(prefix)
        if not rt:
            continue
        measured = round(rt['cyc_per_instr'])
        for op in kdef['ops']:
            current = LATENCY.get(op)
            if current is not None and abs(current - measured) >= 1:
                suggestions.append((op, current, measured))
    if suggestions:
        print()
        print("LATENCY TABLE UPDATES")
        print("-" * 78)
        for op, cur, meas in suggestions:
            print(f"  '{op}': {cur} -> {meas}")
        print(f"  (edit LATENCY dict in sass_analysis.py to apply)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='SASS scheduling analysis for SM 100a (Blackwell)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s sass_dump.txt                           # annotated listing
  %(prog)s sass_dump.txt --deps                    # dependency + slack analysis
  %(prog)s sass_dump.txt --section 0x1300 0x1a70   # analyze address range
  %(prog)s sass_dump.txt --json out.json           # JSON output
  %(prog)s --calibrate > calibration.cu            # generate calibration kernels
  %(prog)s --cubin ./siglip_vision --deps          # analyze from cubin directly
        ''')

    parser.add_argument('input', nargs='?', help='SASS dump file (from cuobjdump --dump-sass)')
    parser.add_argument('--cubin', help='Path to cubin/executable (runs cuobjdump internally)')
    parser.add_argument('--kernel', '-k', help='Kernel function name (substring match)')
    parser.add_argument('--section', '-s', nargs=2, metavar=('START', 'END'),
                        help='Address range to analyze (hex, e.g., 0x1300 0x1a70)')
    parser.add_argument('--deps', '-d', action='store_true',
                        help='Run dependency + slack analysis')
    parser.add_argument('--json', '-j', metavar='FILE',
                        help='Write JSON output to file')
    parser.add_argument('--calibrate', action='store_true',
                        help='Print calibration CUDA source and exit')
    parser.add_argument('--list-kernels', action='store_true',
                        help='List kernel functions and exit')
    parser.add_argument('--calibrate-compare', action='store_true',
                        help='Calibration mode: compare SASS stall counts against runtime')
    parser.add_argument('--runtime', metavar='FILE',
                        help='Runtime output from ./calibration (for --calibrate-compare)')

    args = parser.parse_args()

    if args.calibrate:
        print_calibration()
        return

    if args.calibrate_compare:
        # Need SASS input (positional file or --cubin)
        if args.cubin:
            try:
                result = subprocess.run(
                    ['cuobjdump', '--dump-sass', args.cubin],
                    capture_output=True, text=True, check=True)
                sass_text_cc = result.stdout
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                print(f'Error getting SASS: {e}', file=sys.stderr)
                sys.exit(1)
        elif args.input:
            with open(args.input) as f:
                sass_text_cc = f.read()
        else:
            print('Error: --calibrate-compare needs a SASS input (positional file or --cubin)',
                  file=sys.stderr)
            sys.exit(1)
        runtime_text = None
        if args.runtime:
            with open(args.runtime) as f:
                runtime_text = f.read()
        calibrate_compare(sass_text_cc, runtime_text)
        return

    # Get SASS text
    if args.cubin:
        try:
            result = subprocess.run(
                ['cuobjdump', '--dump-sass', args.cubin],
                capture_output=True, text=True, check=True)
            sass_text = result.stdout
        except FileNotFoundError:
            print('Error: cuobjdump not found. Install CUDA toolkit or provide SASS dump file.',
                  file=sys.stderr)
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f'Error running cuobjdump: {e.stderr}', file=sys.stderr)
            sys.exit(1)
    elif args.input:
        with open(args.input) as f:
            sass_text = f.read()
    else:
        parser.print_help()
        sys.exit(1)

    # Parse
    kernels = parse_sass(sass_text, args.kernel)
    if not kernels:
        print('No kernels found in input.', file=sys.stderr)
        sys.exit(1)

    if args.list_kernels:
        for name, instrs in kernels.items():
            print(f'{name}: {len(instrs)} instructions')
        return

    # Parse section range
    section = None
    if args.section:
        section = (int(args.section[0], 16), int(args.section[1], 16))

    # Process each kernel
    for kname, instrs in kernels.items():
        print(f'\n{"=" * 60}')
        print(f'Kernel: {kname}  ({len(instrs)} instructions)')
        print(f'{"=" * 60}\n')

        if args.deps:
            edges = build_dep_graph(instrs)
            slack_data = compute_slack(instrs, edges)
            cp_cycles, cp_path = critical_path(instrs, edges)

            print_listing(instrs, slack_data, section)
            print_summary(instrs, slack_data, section)

            print(f'\nCritical path: {cp_cycles} cycles, {len(cp_path)} instructions')
            if len(cp_path) <= 20:
                for idx in cp_path:
                    print(f'  0x{instrs[idx].addr:04x}  {instrs[idx].short()}')

            if args.json:
                jdata = make_json(kname, instrs, slack_data, cp_cycles, cp_path, section)
                with open(args.json, 'w') as f:
                    json.dump(jdata, f, indent=2)
                print(f'\nJSON written to {args.json}')
        else:
            print_listing(instrs, section=section)

            # Basic stats even without --deps
            cat_counts = defaultdict(int)
            total_stall = 0
            for instr in instrs:
                if section and (instr.addr < section[0] or instr.addr > section[1]):
                    continue
                cat_counts[instr.category] += 1
                total_stall += instr.stall if instr.stall >= 0 else 0

            print(f'\nTotal stall cycles: {total_stall}')
            print(f'\nInstruction mix:')
            for cat in sorted(cat_counts, key=lambda c: -cat_counts[c]):
                print(f'  {cat:<14s} {cat_counts[cat]:>4d}')


if __name__ == '__main__':
    main()
