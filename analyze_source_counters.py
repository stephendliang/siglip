#!/usr/bin/env python3
"""Analyze ncu SourceCounters CSV for the SigLIP megakernel.

Usage:
    python analyze_source_counters.py source_counters_raw.csv
    python analyze_source_counters.py source_counters_raw.csv --top 20
    python analyze_source_counters.py source_counters_raw.csv --mma-detail
"""
import csv
import sys
import re
from collections import defaultdict

# --- CSV column mapping (ncu --set source --csv) ---
COL_ADDR = 0
COL_SOURCE = 1
COL_STALL_ALL = 2
COL_STALL_NI = 3
COL_SAMPLES = 4
COL_EXECS = 5
COL_THREAD_INSTR = 6
COL_PRED_ON = 7
COL_AVG_THREADS = 8
COL_ADDR_SPACE = 12
COL_ACCESS_OP = 13
# L1/L2 memory columns
COL_L1_TAG_GLOBAL = 15
COL_L1_CONFLICTS_SHARED = 16
COL_L1_WAVES_SHARED_EXCESS = 17
COL_L1_WAVES_SHARED = 18
COL_L1_WAVES_SHARED_IDEAL = 19
COL_L2_EXCESS = 20
COL_L2_SECTORS = 21
COL_L2_IDEAL = 22
# Stall columns (issued)
STALL_START = 31
STALL_NAMES = [
    "barrier", "branch_resolving", "dispatch", "drain", "lg",
    "long_sb", "math", "membar", "mio", "misc",
    "no_inst", "not_selected", "selected", "short_sb", "sleep",
    "tex", "wait"
]
# Stall columns (not-issued)
NI_STALL_START = 48
NI_STALL_NAMES = [
    "barrier", "branch_resolving", "dispatch", "drain", "lg",
    "long_sb", "math", "membar", "mio", "misc",
    "no_inst", "not_selected", "selected", "short_sb", "sleeping",
    "tex", "wait"
]


def parse_int(s):
    s = s.strip().strip('"')
    if not s or s == '-':
        return 0
    return int(s.replace(',', ''))


def extract_opcode(source):
    """Extract base opcode from SASS source line."""
    s = source.strip()
    # Handle predicated instructions: @P0 OPCODE or @!P0 OPCODE
    if s.startswith('@'):
        parts = s.split(None, 1)
        if len(parts) > 1:
            s = parts[1]
    parts = s.split(None, 1)
    return parts[0] if parts else '?'


def classify_opcode(opcode):
    """Classify SASS opcode into functional category."""
    op = opcode.upper().rstrip(',')
    if 'UTCQMMA' in op:
        return 'MMA'
    if op.startswith('R2UR') or op == 'R2UR.BROADCAST':
        return 'R2UR (reg→uniform)'
    if 'ELECT' in op:
        return 'ELECT'
    if op.startswith('PLOP3'):
        return 'PLOP3 (pred logic)'
    if op.startswith('SYNCS') or op.startswith('MBAR') or 'BAR' in op:
        return 'SYNC/MBAR'
    if op.startswith('UTCBAR'):
        return 'UTCBAR (commit)'
    if op.startswith('TCGEN05'):
        return 'TCGEN05 (TMEM ld)'
    if op.startswith('STG') or op.startswith('ST.'):
        return 'Global store'
    if op.startswith('STS') or op.startswith('STL'):
        return 'Shared/Local store'
    if op.startswith('LDG') or op.startswith('LD.'):
        return 'Global load'
    if op.startswith('LDS'):
        return 'Shared load'
    if op.startswith('IMAD') or op.startswith('IADD') or op.startswith('UIADD'):
        return 'Integer arith'
    if op.startswith('FADD') or op.startswith('FMUL') or op.startswith('FFMA'):
        return 'FP arith'
    if op.startswith('F2FP') or op.startswith('PRMT'):
        return 'CVT/PRMT'
    if op.startswith('SHF') or op.startswith('USHF') or op.startswith('LOP3'):
        return 'Shift/Logic'
    if op.startswith('MOV') or op.startswith('UMOV') or op.startswith('S2R') or op.startswith('S2UR'):
        return 'MOV/S2R'
    if op.startswith('BRA') or op.startswith('BSSY') or op.startswith('BSYNC'):
        return 'Branch/Sync'
    if op.startswith('ISETP') or op.startswith('SETP') or op.startswith('UISETP') or op.startswith('USETP'):
        return 'SETP (compare)'
    if op == 'NOP':
        return 'NOP'
    if op.startswith('NANOSLEEP') or op.startswith('WARPSYNC'):
        return 'WARPSYNC/Sleep'
    return 'Other'


def load_csv(path):
    """Load ncu source counters CSV, skip preamble."""
    with open(path) as f:
        lines = f.readlines()

    # Find header line starting with "Address"
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('"Address"'):
            header_idx = i
            break
    if header_idx is None:
        print("ERROR: Could not find header line starting with '\"Address\"'", file=sys.stderr)
        sys.exit(1)

    # Print kernel info from preamble
    info = {}
    for line in lines[:header_idx]:
        line = line.strip()
        if line.startswith('==PROF==') or line.startswith('"Kernel Name"'):
            continue
        if 'GEMM' in line or 'cta_group' in line:
            info['kernel'] = line
        if 'Custom kernel:' in line or 'TFLOPS' in line:
            info['perf'] = line

    reader = csv.reader(lines[header_idx:])
    header = next(reader)

    rows = []
    for row in reader:
        if len(row) < STALL_START + len(STALL_NAMES):
            continue
        execs = parse_int(row[COL_EXECS])
        if execs == 0:
            continue
        rows.append(row)

    return info, header, rows


def analyze(path, top_n=15, mma_detail=False):
    info, header, rows = load_csv(path)

    # --- Header ---
    print("=" * 72)
    print("  ncu SourceCounters Analysis")
    print("=" * 72)
    if 'kernel' in info:
        print(f"  {info['kernel']}")
    if 'perf' in info:
        print(f"  {info['perf']}")
    print(f"  {len(rows)} SASS instructions with nonzero executions")
    print()

    # --- Global stall distribution ---
    global_stalls = defaultdict(int)
    global_stalls_ni = defaultdict(int)
    total_samples = 0
    total_execs = 0
    total_thread_instr = 0

    for row in rows:
        execs = parse_int(row[COL_EXECS])
        total_execs += execs
        total_thread_instr += parse_int(row[COL_THREAD_INSTR])
        samples = parse_int(row[COL_SAMPLES])
        total_samples += samples
        for j, name in enumerate(STALL_NAMES):
            val = parse_int(row[STALL_START + j])
            global_stalls[name] += val
        for j, name in enumerate(NI_STALL_NAMES):
            val = parse_int(row[NI_STALL_START + j])
            global_stalls_ni[name] += val

    print("--- GLOBAL STALL DISTRIBUTION (All Samples) ---")
    print(f"  Total samples: {total_samples:,}   Total instr execs: {total_execs:,}   Thread instrs: {total_thread_instr:,}")
    print()
    sorted_stalls = sorted(global_stalls.items(), key=lambda x: -x[1])
    for name, count in sorted_stalls:
        if count == 0: continue
        pct = 100.0 * count / total_samples if total_samples else 0
        ni = global_stalls_ni.get(name, global_stalls_ni.get(name.replace('sleep', 'sleeping'), 0))
        ni_pct = 100.0 * ni / total_samples if total_samples else 0
        print(f"  {name:<20s} {count:>8,} ({pct:5.1f}%)   not-issued: {ni:>8,} ({ni_pct:5.1f}%)")
    print()

    # --- MMA analysis ---
    mma_rows = [r for r in rows if 'UTCQMMA' in r[COL_SOURCE]]
    mma_execs = sum(parse_int(r[COL_EXECS]) for r in mma_rows)
    mma_stalls = sum(parse_int(r[COL_STALL_ALL]) for r in mma_rows)
    mma_stall_detail = defaultdict(int)
    for r in mma_rows:
        for j, name in enumerate(STALL_NAMES):
            val = parse_int(r[STALL_START + j])
            mma_stall_detail[name] += val

    tiles = parse_int(mma_rows[0][COL_EXECS]) if mma_rows else 0

    print("--- MMA (UTCQMMA) ---")
    print(f"  {len(mma_rows)} instructions × {tiles:,} tiles = {mma_execs:,} total execs")
    print(f"  Stalls: {mma_stalls} ({100*mma_stalls/mma_execs:.3f}%)" if mma_execs else "  Stalls: 0")
    if mma_stalls > 0:
        detail = sorted(mma_stall_detail.items(), key=lambda x: -x[1])
        parts = [f"{n}={c}" for n, c in detail if c > 0]
        print(f"  Breakdown: {', '.join(parts)}")
    print()

    if mma_detail and mma_rows:
        print("  Per-MMA instruction detail:")
        for i, r in enumerate(mma_rows):
            s = parse_int(r[COL_STALL_ALL])
            dom = max(((STALL_NAMES[j], parse_int(r[STALL_START+j])) for j in range(len(STALL_NAMES))), key=lambda x: x[1])
            pred = "!UPT" if "!UPT" in r[COL_SOURCE] else "UPT"
            print(f"    MMA[{i:2d}] ({pred}): execs={parse_int(r[COL_EXECS]):,}  stalls={s}  dom={dom[0]}({dom[1]})")
        print()

    # --- W1 instruction budget ---
    # W1 = MMA warp. Identify by: same exec count as MMA (tiles), or in MMA address range
    if mma_rows:
        mma_addrs = [int(r[COL_ADDR].strip('"'), 16) for r in mma_rows]
        mma_min, mma_max = min(mma_addrs), max(mma_addrs)
        # W1 K-loop: instructions with exec count == tiles (exact match)
        w1_rows = [r for r in rows if parse_int(r[COL_EXECS]) == tiles]
        w1_total = len(w1_rows)
        w1_mma = len([r for r in w1_rows if 'UTCQMMA' in r[COL_SOURCE]])
        w1_stalls = defaultdict(int)
        for r in w1_rows:
            for j, name in enumerate(STALL_NAMES):
                w1_stalls[name] += parse_int(r[STALL_START + j])

        print(f"--- W1 INSTRUCTION BUDGET (exec_count = {tiles:,} = tiles) ---")
        print(f"  {w1_total} instructions/tile, {w1_mma} MMA ({100*w1_mma/w1_total:.1f}%), {w1_total-w1_mma} overhead ({100*(w1_total-w1_mma)/w1_total:.1f}%)")
        # Classify W1 overhead
        w1_cats = defaultdict(int)
        for r in w1_rows:
            op = extract_opcode(r[COL_SOURCE])
            cat = classify_opcode(op)
            w1_cats[cat] += 1
        print(f"  Overhead by category:")
        for cat, cnt in sorted(w1_cats.items(), key=lambda x: -x[1]):
            print(f"    {cat:<25s} {cnt:>4d} ({100*cnt/w1_total:5.1f}%)")
        print()
        w1_total_stalls = sum(w1_stalls.values())
        w1_total_execs = w1_total * tiles
        print(f"  W1 stalls: {w1_total_stalls:,} / {w1_total_execs:,} execs ({100*w1_total_stalls/w1_total_execs:.3f}%)")
        ws = sorted(w1_stalls.items(), key=lambda x: -x[1])
        for name, count in ws:
            if count == 0: continue
            print(f"    {name:<20s} {count:>6,} ({count/tiles:.3f}/tile)")
        print()

    # --- Mbar retry loops (spin-waits with NANOSLEEP) ---
    trywait_rows = [r for r in rows if 'TRYWAIT' in r[COL_SOURCE] and parse_int(r[COL_EXECS]) > 0]
    if trywait_rows and tiles:
        w1_mbars = [r for r in trywait_rows if parse_int(r[COL_EXECS]) == tiles]
        retry_loops = [r for r in trywait_rows if parse_int(r[COL_EXECS]) > tiles * 5]
        epi_mbars = [r for r in trywait_rows
                     if tiles < parse_int(r[COL_EXECS]) <= tiles * 5]

        if w1_mbars or retry_loops or epi_mbars:
            print("--- MBAR WAITS ---")
            if w1_mbars:
                w1_total_stalls = sum(parse_int(r[COL_STALL_ALL]) for r in w1_mbars)
                print(f"  W1 mbar_waits: {len(w1_mbars)} per tile ({w1_total_stalls} total stalls — near-zero = NANOSLEEP attribution)")
            if epi_mbars:
                epi_total_stalls = sum(parse_int(r[COL_STALL_ALL]) for r in epi_mbars)
                print(f"  Epilogue mbar_waits: {len(epi_mbars)} unique TRYWAIT ({epi_total_stalls} stalls)")
            if retry_loops:
                # Deduplicate by register pattern (same mbar, different addresses are same loop)
                seen = set()
                for r in retry_loops:
                    execs = parse_int(r[COL_EXECS])
                    if execs in seen:
                        continue
                    seen.add(execs)
                    ratio = execs / tiles
                    src = r[COL_SOURCE].strip()[:55]
                    print(f"  Retry loop: execs={execs:,} ({ratio:.0f}× tiles = avg {ratio:.0f} retries/tile)")
            print()

    # --- Top stalling instructions ---
    print(f"--- TOP {top_n} STALLING INSTRUCTIONS ---")
    stall_sorted = sorted(rows, key=lambda r: -parse_int(r[COL_STALL_ALL]))
    for i, r in enumerate(stall_sorted[:top_n]):
        stalls = parse_int(r[COL_STALL_ALL])
        if stalls == 0:
            break
        execs = parse_int(r[COL_EXECS])
        rate = 100.0 * stalls / execs if execs else 0
        source = r[COL_SOURCE].strip()[:55]
        # Find dominant stall
        dom_name, dom_val = max(
            ((STALL_NAMES[j], parse_int(r[STALL_START + j])) for j in range(len(STALL_NAMES))),
            key=lambda x: x[1]
        )
        print(f"  {i+1:2d}. [{stalls:>6,} stalls, {rate:5.1f}%] {source}")
        print(f"      execs={execs:>10,}  dominant: {dom_name}({dom_val})")
    print()

    # --- SMEM bank conflict summary ---
    total_shared_waves = 0
    total_shared_ideal = 0
    total_shared_conflicts = 0
    for r in rows:
        waves = parse_int(r[COL_L1_WAVES_SHARED])
        ideal = parse_int(r[COL_L1_WAVES_SHARED_IDEAL])
        conflicts = parse_int(r[COL_L1_CONFLICTS_SHARED])
        total_shared_waves += waves
        total_shared_ideal += ideal
        total_shared_conflicts += conflicts

    if total_shared_ideal > 0:
        excess_pct = 100.0 * (total_shared_waves - total_shared_ideal) / total_shared_ideal
        print(f"--- SMEM BANK CONFLICTS ---")
        print(f"  Wavefronts: {total_shared_waves:,} actual vs {total_shared_ideal:,} ideal ({excess_pct:+.1f}% excess)")
        print(f"  N-way conflicts: {total_shared_conflicts:,}")
        print()

    # --- Instruction mix by opcode category ---
    cat_execs = defaultdict(int)
    cat_count = defaultdict(int)
    for r in rows:
        op = extract_opcode(r[COL_SOURCE])
        cat = classify_opcode(op)
        cat_execs[cat] += parse_int(r[COL_EXECS])
        cat_count[cat] += 1

    print("--- INSTRUCTION MIX (by total executions, >=1%) ---")
    sorted_cats = sorted(cat_execs.items(), key=lambda x: -x[1])
    for cat, execs in sorted_cats:
        pct = 100.0 * execs / total_execs if total_execs else 0
        if pct < 1.0:
            continue
        n = cat_count[cat]
        print(f"  {cat:<25s} {execs:>12,} ({pct:5.1f}%)  [{n} unique]")
    print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    top_n = 10
    mma_detail = False
    for arg in sys.argv[2:]:
        if arg == '--mma-detail':
            mma_detail = True
        elif arg == '--top':
            idx = sys.argv.index('--top')
            top_n = int(sys.argv[idx + 1])

    analyze(path, top_n=top_n, mma_detail=mma_detail)
