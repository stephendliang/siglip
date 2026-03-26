#!/usr/bin/env python3
"""Phase 6.1+6.2: Round-trip fidelity tests for the SM100a SASS assembler.

6.1 — Disassemble cubin region → reassemble → compare per-field fidelity
6.2 — Patch cubin with assembled instructions → cuobjdump validity check

Usage:
    python3 tools/test_roundtrip.py                          # full report
    python3 tools/test_roundtrip.py --region 0x51c0 0x63b0   # epilogue only
    python3 tools/test_roundtrip.py --cuobjdump              # include 6.2 cuobjdump check
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)

from sass_edit import (
    CubinEditor, INSN_SIZE, FAMILY_REG_FIELDS, _DEFAULT_REG_FIELDS,
    disassemble_region, verify_asm_region, assemble, _load_asm_tables,
    insn_family, NOP_ENCODING,
)
from sm100a_opcodes import OPCODE_TABLE

PRED_MASK = 0xf000  # bits [15:12]


def compute_reg_mask(family):
    """Compute the bitmask of all register fields for a family."""
    fields = FAMILY_REG_FIELDS.get(family)
    if fields is None:
        fields = _DEFAULT_REG_FIELDS
    mask = 0
    for _, _, bit_offset, width in fields:
        mask |= ((1 << width) - 1) << bit_offset
    return mask


def analyze_instruction(asm_enc, asm_ctrl, bin_enc, bin_ctrl, mnemonic):
    """Compare assembled vs binary instruction, returning per-field results."""
    family = insn_family(mnemonic) if mnemonic else None

    # Look up opcode table entry
    opcode_bits = None
    var_mask = 0
    if mnemonic:
        # Strip guard prefix for opcode lookup
        m = mnemonic
        if m.startswith('@'):
            m = m.split(None, 1)[1] if ' ' in m else m
        try:
            # Try exact, then progressively shorter
            parts = m.split('.')
            for n in range(len(parts), 0, -1):
                key = '.'.join(parts[:n])
                if key in OPCODE_TABLE:
                    opcode_bits, var_mask = OPCODE_TABLE[key]
                    break
        except Exception:
            pass

    # Opcode match: bits outside var_mask and pred_mask
    opcode_mask = ~(var_mask | PRED_MASK) & 0xFFFFFFFFFFFFFFFF
    opcode_ok = (asm_enc & opcode_mask) == (bin_enc & opcode_mask)

    # Predicate match: bits [15:12]
    pred_ok = (asm_enc & PRED_MASK) == (bin_enc & PRED_MASK)

    # Register field match
    reg_mask = compute_reg_mask(family) if family else 0
    reg_ok = (asm_enc & reg_mask) == (bin_enc & reg_mask) if reg_mask else None

    # Control word match
    ctrl_ok = asm_ctrl == bin_ctrl

    # Full encoding match
    enc_ok = asm_enc == bin_enc

    return {
        'family': family or '?',
        'mnemonic': mnemonic or '?',
        'opcode_ok': opcode_ok,
        'pred_ok': pred_ok,
        'reg_ok': reg_ok,
        'ctrl_ok': ctrl_ok,
        'enc_ok': enc_ok,
        'asm_enc': asm_enc,
        'bin_enc': bin_enc,
        'asm_ctrl': asm_ctrl,
        'bin_ctrl': bin_ctrl,
    }


def run_roundtrip(cubin_path, sass_path, start, end, verbose=False):
    """Phase 6.1: Full round-trip fidelity analysis."""
    _load_asm_tables()

    ed = CubinEditor(cubin_path)
    k = ed.kernels[0]  # persistent_gemm
    n_xref = ed.xref(sass_path, k)

    # Disassemble
    asm_text = disassemble_region(k, start, end)

    # Reassemble
    binary, labels, instructions = assemble(asm_text, base_pc=start)

    # Per-instruction comparison
    n_insns = (end - start) // INSN_SIZE
    start_idx = start // INSN_SIZE
    results = []

    for i in range(min(len(instructions), n_insns)):
        asm_enc = struct.unpack_from('<Q', binary, i * INSN_SIZE)[0]
        asm_ctrl = struct.unpack_from('<Q', binary, i * INSN_SIZE + 8)[0]
        bin_insn = k.instructions[start_idx + i]

        r = analyze_instruction(
            asm_enc, asm_ctrl,
            bin_insn.encoding, bin_insn.control,
            bin_insn.mnemonic,
        )
        r['addr'] = start + i * INSN_SIZE
        results.append(r)

    return results


def print_report(results, verbose=False):
    """Print per-family fidelity report."""
    # Global counts
    n = len(results)
    n_opcode = sum(1 for r in results if r['opcode_ok'])
    n_pred = sum(1 for r in results if r['pred_ok'])
    n_reg = sum(1 for r in results if r['reg_ok'] is not False)  # None = no fields = ok
    n_ctrl = sum(1 for r in results if r['ctrl_ok'])
    n_enc = sum(1 for r in results if r['enc_ok'])

    print('=' * 72)
    print('Phase 6.1: Round-Trip Fidelity Report')
    print('=' * 72)
    print()
    print('Overall: %d instructions' % n)
    print('  Opcode bits:  %d/%d (%.1f%%)' % (n_opcode, n, 100 * n_opcode / n))
    print('  Predicate:    %d/%d (%.1f%%)' % (n_pred, n, 100 * n_pred / n))
    print('  Registers:    %d/%d (%.1f%%)' % (n_reg, n, 100 * n_reg / n))
    print('  Control word: %d/%d (%.1f%%)' % (n_ctrl, n, 100 * n_ctrl / n))
    print('  Full encoding:%d/%d (%.1f%%)' % (n_enc, n, 100 * n_enc / n))
    print()

    # Per-family breakdown
    families = defaultdict(lambda: {'total': 0, 'opcode': 0, 'pred': 0, 'reg': 0,
                                     'ctrl': 0, 'enc': 0})
    for r in results:
        f = families[r['family']]
        f['total'] += 1
        if r['opcode_ok']:
            f['opcode'] += 1
        if r['pred_ok']:
            f['pred'] += 1
        if r['reg_ok'] is not False:
            f['reg'] += 1
        if r['ctrl_ok']:
            f['ctrl'] += 1
        if r['enc_ok']:
            f['enc'] += 1

    print('Per-family breakdown:')
    print('  %-20s %5s %7s %7s %7s %7s %7s' % (
        'Family', 'Count', 'Opcode', 'Pred', 'Regs', 'Ctrl', 'Full'))
    print('  ' + '-' * 70)

    for fam in sorted(families.keys()):
        f = families[fam]
        t = f['total']
        print('  %-20s %5d %6d%% %6d%% %6d%% %6d%% %6d%%' % (
            fam, t,
            100 * f['opcode'] // t,
            100 * f['pred'] // t,
            100 * f['reg'] // t,
            100 * f['ctrl'] // t,
            100 * f['enc'] // t,
        ))

    # Detailed mismatches (if verbose)
    if verbose:
        print()
        print('Opcode mismatches:')
        for r in results:
            if not r['opcode_ok']:
                print('  [%04x] %s  asm=%016x bin=%016x' % (
                    r['addr'], r['mnemonic'], r['asm_enc'], r['bin_enc']))
        print()
        print('Register mismatches:')
        for r in results:
            if r['reg_ok'] is False:
                mask = compute_reg_mask(r['family'])
                print('  [%04x] %s  asm_regs=%016x bin_regs=%016x (mask=%016x)' % (
                    r['addr'], r['mnemonic'],
                    r['asm_enc'] & mask, r['bin_enc'] & mask, mask))
        print()
        print('Control mismatches:')
        for r in results:
            if not r['ctrl_ok']:
                print('  [%04x] %s  asm=%016x bin=%016x' % (
                    r['addr'], r['mnemonic'], r['asm_ctrl'], r['bin_ctrl']))

    return n_opcode, n_pred, n_reg, n_ctrl, n_enc


def run_cuobjdump_check(cubin_path, sass_path, start, end):
    """Phase 6.2: Patch cubin and verify cuobjdump can parse it."""
    _load_asm_tables()
    print()
    print('=' * 72)
    print('Phase 6.2: Cubin Validity Test (cuobjdump)')
    print('=' * 72)
    print()

    ed = CubinEditor(cubin_path)
    k = ed.kernels[0]
    n_xref = ed.xref(sass_path, k)

    # Test 1: Patch a small NOP region and verify cuobjdump parses it
    print('Test 1: NOP-fill a 16-instruction region...')
    test_start = start
    test_end = start + 16 * INSN_SIZE
    start_idx = test_start // INSN_SIZE
    end_idx = test_end // INSN_SIZE

    # Save originals
    originals = [(k.instructions[i].encoding, k.instructions[i].control)
                 for i in range(start_idx, end_idx)]

    # NOP-fill with valid NOP control word
    _load_asm_tables()
    from sm100a_encoding import CONTROL_DEFAULTS
    nop_ctrl = CONTROL_DEFAULTS.get('NOP', 0x000fe20000000000)
    for i in range(start_idx, end_idx):
        insn = k.instructions[i]
        insn.encoding = NOP_ENCODING
        insn.control = nop_ctrl
        insn.mnemonic = 'NOP'

    with tempfile.NamedTemporaryFile(suffix='.cubin', delete=False) as f:
        tmp_path = f.name
    try:
        ed.save(tmp_path)
        result = subprocess.run(
            ['cuobjdump', '-sass', tmp_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            # Check NOP appears in output
            nop_count = result.stdout.count('NOP')
            print('  PASS: cuobjdump parsed patched cubin (%d NOPs in output)' % nop_count)
        else:
            print('  FAIL: cuobjdump error: %s' % result.stderr[:200])
            return False
    finally:
        os.unlink(tmp_path)

    # Restore originals
    for i, (enc, ctrl) in enumerate(originals):
        k.instructions[start_idx + i].encoding = enc
        k.instructions[start_idx + i].control = ctrl

    # Test 2: Patch a small fully-matching region → cuobjdump must still parse
    # Find a contiguous run of fully-matching instructions
    print('Test 2: Patch contiguous matching region → cuobjdump...')
    asm_text = disassemble_region(k, start, end)
    binary, labels, instructions = assemble(asm_text, base_pc=start)
    n_insns = (end - start) // INSN_SIZE
    start_idx_base = start // INSN_SIZE

    # Find matching runs
    match_flags = []
    for i in range(min(len(instructions), n_insns)):
        asm_enc = struct.unpack_from('<Q', binary, i * INSN_SIZE)[0]
        asm_ctrl = struct.unpack_from('<Q', binary, i * INSN_SIZE + 8)[0]
        bin_insn = k.instructions[start_idx_base + i]
        match_flags.append(asm_enc == bin_insn.encoding and asm_ctrl == bin_insn.control)

    # Find longest contiguous run
    best_run_start = best_run_end = 0
    cur_run_start = 0
    for i, ok in enumerate(match_flags):
        if ok:
            if i == 0 or not match_flags[i - 1]:
                cur_run_start = i
            if (i - cur_run_start + 1) > (best_run_end - best_run_start):
                best_run_start = cur_run_start
                best_run_end = i + 1

    run_len = best_run_end - best_run_start
    if run_len < 2:
        print('  SKIP: no contiguous fully-matching region found')
    else:
        patch_start = start + best_run_start * INSN_SIZE
        patch_end = start + best_run_end * INSN_SIZE
        # Disassemble just this sub-region and patch it back
        ed_patch = CubinEditor(cubin_path)
        k_patch = ed_patch.kernels[0]
        ed_patch.xref(sass_path, k_patch)
        sub_asm = disassemble_region(k_patch, patch_start, patch_end)
        n_p, n_n = ed_patch.patch_region_asm(k_patch, patch_start, patch_end, sub_asm)
        print('  Patched %d instructions at [0x%x, 0x%x)' % (n_p, patch_start, patch_end))

        with tempfile.NamedTemporaryFile(suffix='.cubin', delete=False) as f:
            tmp_path = f.name
        try:
            ed_patch.save(tmp_path)
            result = subprocess.run(
                ['cuobjdump', '-sass', tmp_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                print('  PASS: cuobjdump parsed patched cubin')
            else:
                print('  FAIL: cuobjdump error: %s' % result.stderr[:200])
                return False
        finally:
            os.unlink(tmp_path)

    # Test 3: Verify ELF structure (size matches, sections intact)
    print('Test 3: ELF structure integrity...')
    ed2 = CubinEditor(cubin_path)
    k2 = ed2.kernels[0]
    with tempfile.NamedTemporaryFile(suffix='.cubin', delete=False) as f:
        tmp_path = f.name
    try:
        ed2.save(tmp_path)
        orig_size = os.path.getsize(cubin_path)
        new_size = os.path.getsize(tmp_path)
        if orig_size == new_size:
            print('  PASS: cubin size preserved (%d bytes)' % orig_size)
        else:
            print('  FAIL: size changed %d → %d' % (orig_size, new_size))
            return False
    finally:
        os.unlink(tmp_path)

    # Test 4: Full disasm→reassemble→patch → cuobjdump (ELF-valid, instruction decode may fail)
    print('Test 4: Full region reassemble → ELF validity (cuobjdump -elf)...')
    ed3 = CubinEditor(cubin_path)
    k3 = ed3.kernels[0]
    ed3.xref(sass_path, k3)
    full_asm = disassemble_region(k3, start, end)
    ed3.patch_region_asm(k3, start, end, full_asm)
    with tempfile.NamedTemporaryFile(suffix='.cubin', delete=False) as f:
        tmp_path = f.name
    try:
        ed3.save(tmp_path)
        result = subprocess.run(
            ['cuobjdump', '-elf', tmp_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and '.text.' in result.stdout:
            print('  PASS: ELF structure valid (.text section present)')
        else:
            print('  FAIL: cuobjdump -elf error: %s' % result.stderr[:200])
            return False
    finally:
        os.unlink(tmp_path)

    print()
    print('All cubin validity tests passed.')
    return True


def main():
    parser = argparse.ArgumentParser(description='Phase 6: SASS assembler verification')
    parser.add_argument('--cubin', default=os.path.join(ROOT_DIR, 'fc2.cubin'),
                       help='Cubin file (default: fc2.cubin)')
    parser.add_argument('--sass', default=os.path.join(ROOT_DIR, 'sass/fc2_cubin.txt'),
                       help='SASS xref file')
    parser.add_argument('--region', nargs=2, metavar=('START', 'END'),
                       help='Address range (hex). Default: full xref range.')
    parser.add_argument('--cuobjdump', action='store_true',
                       help='Also run Phase 6.2 cuobjdump validity tests')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Print detailed mismatch info')
    args = parser.parse_args()

    if not os.path.exists(args.cubin):
        print('Cubin not found: %s' % args.cubin)
        sys.exit(1)
    if not os.path.exists(args.sass):
        print('SASS file not found: %s' % args.sass)
        sys.exit(1)

    # Default region: FC2 epilogue
    if args.region:
        start = int(args.region[0], 0)
        end = int(args.region[1], 0)
    else:
        start = 0x51c0
        end = 0x63b0

    # Phase 6.1: Round-trip fidelity
    results = run_roundtrip(args.cubin, args.sass, start, end, verbose=args.verbose)
    n_opcode, n_pred, n_reg, n_ctrl, n_enc = print_report(results, verbose=args.verbose)

    # Phase 6.2: cuobjdump validity (optional)
    if args.cuobjdump:
        ok = run_cuobjdump_check(args.cubin, args.sass, start, end)
        if not ok:
            sys.exit(1)

    # Summary exit code: ctrl must be 100%, opcode/pred/reg should be high
    n = len(results)
    if n_ctrl < n:
        print('\nFAIL: control word round-trip < 100%%')
        sys.exit(1)
    print('\nPASS: ctrl=100%%, opcode=%.0f%%, pred=%.0f%%, reg=%.0f%%, full=%.0f%%' % (
        100 * n_opcode / n, 100 * n_pred / n, 100 * n_reg / n, 100 * n_enc / n))


if __name__ == '__main__':
    main()
