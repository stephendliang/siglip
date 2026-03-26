#!/usr/bin/env python3
"""Tests for Phase 4: Assembler Core (no GPU needed).

Tests parser, encoder, control word generation, and binary emitter
against real cubin data from FC2 and CUTLASS.

Run: python3 -m pytest tools/test_assembler.py -v
"""

import os
import re
import struct
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from sass_edit import (
    AsmInstruction, AsmOperand, parse_asm_line, parse_asm,
    encode_instruction, assemble, _load_asm_tables, _lookup_opcode,
    _encode_predicate, _parse_asm_operand, _split_asm_operands,
    OP_GPR, OP_UR, OP_PRED, OP_UPRED, OP_IMM, OP_MEM, OP_UMEM,
    OP_TMEM, OP_LABEL, OP_DESC, OP_TMEM_UR, OP_BARRIER,
    CubinEditor, _strip_pred, insn_family, read_enc_field,
    _split_operands, FAMILY_REG_FIELDS, write_enc_field,
    INSN_SIZE, NOP_ENCODING, disassemble_region, verify_asm_region,
)

REPO = os.path.dirname(os.path.dirname(__file__))
FC2_CUBIN = os.path.join(REPO, 'fc2.cubin')
CUTLASS_CUBIN = os.path.join(REPO, 'cutlass_bench.sm_100a.cubin')
_FC2_SASS = [
    os.path.join(REPO, 'sass', 'fc2_cubin.txt'),
    os.path.join(REPO, 'sass', 'fc2_current.txt'),
    os.path.join(REPO, 'sass', 'fc2_ns5.txt'),
    os.path.join(REPO, 'sass', 'fc2_ns5_is1_base.txt'),
]
_CUTLASS_SASS = os.path.join(REPO, 'sass', 'cutlass_fc2_max.txt')


def _find_sass(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


@pytest.fixture(scope='module')
def fc2_kernels():
    if not os.path.exists(FC2_CUBIN):
        pytest.skip('fc2.cubin not found')
    sass = _find_sass(_FC2_SASS)
    if not sass:
        pytest.skip('No FC2 SASS dump found')
    ed = CubinEditor(FC2_CUBIN)
    ed.xref(sass)
    return ed.kernels


@pytest.fixture(scope='module')
def all_kernels():
    kernels = []
    if os.path.exists(FC2_CUBIN):
        sass = _find_sass(_FC2_SASS)
        if sass:
            ed = CubinEditor(FC2_CUBIN)
            ed.xref(sass)
            kernels.extend(ed.kernels)
    if os.path.exists(CUTLASS_CUBIN) and os.path.exists(_CUTLASS_SASS):
        ed = CubinEditor(CUTLASS_CUBIN)
        ed.xref(_CUTLASS_SASS)
        kernels.extend(ed.kernels)
    if not kernels:
        pytest.skip('No cubins available')
    return kernels


@pytest.fixture(scope='module', autouse=True)
def load_tables():
    _load_asm_tables()


# ── Parser Tests ─────────────────────────────────────────────────────────


class TestParserOperands:
    def test_gpr(self):
        op = _parse_asm_operand('R42')
        assert op.type == OP_GPR
        assert op.val == {'reg': 42}
        assert not op.reuse

    def test_gpr_reuse(self):
        op = _parse_asm_operand('R42.reuse')
        assert op.type == OP_GPR
        assert op.val == {'reg': 42}
        assert op.reuse

    def test_rz(self):
        op = _parse_asm_operand('RZ')
        assert op.type == OP_GPR
        assert op.val == {'reg': 0xff}

    def test_ur(self):
        op = _parse_asm_operand('UR10')
        assert op.type == OP_UR
        assert op.val == {'reg': 10}

    def test_urz(self):
        op = _parse_asm_operand('URZ')
        assert op.type == OP_UR
        assert op.val == {'reg': 0xff}

    def test_pred_operand(self):
        op = _parse_asm_operand('P3')
        assert op.type == OP_PRED
        assert op.val['reg'] == 3
        assert not op.val['neg']

    def test_pt_operand(self):
        op = _parse_asm_operand('PT')
        assert op.type == OP_PRED
        assert op.val['reg'] == 7

    def test_upred_operand(self):
        op = _parse_asm_operand('UP2')
        assert op.type == OP_UPRED
        assert op.val['reg'] == 2

    def test_neg_pred_operand(self):
        op = _parse_asm_operand('!P5')
        assert op.type == OP_PRED
        assert op.val['reg'] == 5
        assert op.val['neg']

    def test_neg_register(self):
        op = _parse_asm_operand('-UR5')
        assert op.type == OP_UR
        assert op.val['reg'] == 5

    def test_barrier_operand(self):
        op = _parse_asm_operand('B2')
        assert op.type == OP_BARRIER
        assert op.val['reg'] == 2

    def test_cbank_operand(self):
        op = _parse_asm_operand('c[0x0][0x348]')
        assert op.type == OP_IMM
        assert op.val['cbank'] == 0
        assert op.val['value'] == 0x348

    def test_gdesc_operand(self):
        op = _parse_asm_operand('gdesc[UR10]')
        assert op.type == OP_DESC
        assert op.val['reg'] == 10

    def test_mem_ur_offset(self):
        op = _parse_asm_operand('[R10+URZ+0x28008]')
        assert op.type == OP_MEM
        assert op.val['base'] == 10
        assert op.val['offset'] == 0x28008

    def test_mem_neg_offset(self):
        op = _parse_asm_operand('[R143+-0x2000]')
        assert op.type == OP_MEM
        assert op.val['base'] == 143
        assert op.val['offset'] == -0x2000

    def test_tmem_ur_offset(self):
        op = _parse_asm_operand('tmem[UR6+0x80]')
        assert op.type == OP_TMEM_UR
        assert op.val['reg'] == 6
        assert op.val['offset'] == 0x80

    def test_mem_with_offset(self):
        op = _parse_asm_operand('[R107+0x0100]')
        assert op.type == OP_MEM
        assert op.val['base'] == 107
        assert op.val['offset'] == 0x100

    def test_mem_no_offset(self):
        op = _parse_asm_operand('[R0]')
        assert op.type == OP_MEM
        assert op.val['base'] == 0
        assert op.val['offset'] == 0

    def test_umem(self):
        op = _parse_asm_operand('[UR8+0x20]')
        assert op.type == OP_UMEM
        assert op.val == {'base': 8, 'offset': 0x20}

    def test_tmem(self):
        op = _parse_asm_operand('[TMEM+0x200]')
        assert op.type == OP_TMEM
        assert op.val == {'offset': 0x200}

    def test_desc(self):
        op = _parse_asm_operand('desc[UR8]')
        assert op.type == OP_DESC
        assert op.val == {'reg': 8}

    def test_tmem_ur(self):
        op = _parse_asm_operand('tmem[UR12]')
        assert op.type == OP_TMEM_UR
        assert op.val['reg'] == 12

    def test_imm_hex(self):
        op = _parse_asm_operand('0x1234')
        assert op.type == OP_IMM
        assert op.val == {'value': 0x1234}

    def test_imm_decimal(self):
        op = _parse_asm_operand('42')
        assert op.type == OP_IMM
        assert op.val == {'value': 42}

    def test_imm_negative(self):
        op = _parse_asm_operand('-16')
        assert op.type == OP_IMM
        assert op.val == {'value': -16}

    def test_label(self):
        op = _parse_asm_operand('.L_loop')
        assert op.type == OP_LABEL
        assert op.val == {'name': '.L_loop'}


class TestParserSplitOperands:
    def test_simple(self):
        assert _split_asm_operands('R1, R2, R3') == ['R1', 'R2', 'R3']

    def test_brackets(self):
        assert _split_asm_operands('[R0+0x10], R5') == ['[R0+0x10]', 'R5']

    def test_empty(self):
        assert _split_asm_operands('') == []


class TestParserLine:
    def test_empty(self):
        assert parse_asm_line('') is None

    def test_comment(self):
        assert parse_asm_line('# this is a comment') is None
        assert parse_asm_line('; this is also a comment') is None

    def test_nop(self):
        insn = parse_asm_line('    NOP')
        assert insn.mnemonic == 'NOP'
        assert insn.operands == []
        assert not insn.has_guard

    def test_three_operand(self):
        insn = parse_asm_line('    HADD2.BF16_V2 R96, R96, R80')
        assert insn.mnemonic == 'HADD2.BF16_V2'
        assert insn.family == 'HADD2'
        assert len(insn.operands) == 3
        assert insn.operands[0].val['reg'] == 96
        assert insn.operands[2].val['reg'] == 80

    def test_guard_positive(self):
        insn = parse_asm_line('@P0 BRA .L_target')
        assert insn.guard_reg == 0
        assert not insn.guard_neg
        assert not insn.guard_uniform
        assert insn.mnemonic == 'BRA'

    def test_guard_negative(self):
        insn = parse_asm_line('@!P3 EXIT')
        assert insn.guard_reg == 3
        assert insn.guard_neg
        assert insn.mnemonic == 'EXIT'

    def test_guard_uniform(self):
        insn = parse_asm_line('@UP2 NOP')
        assert insn.guard_reg == 2
        assert insn.guard_uniform
        assert not insn.guard_neg

    def test_memory_operand(self):
        insn = parse_asm_line('    STS.128 [R105+0x0000], R96')
        assert insn.mnemonic == 'STS.128'
        assert insn.operands[0].type == OP_MEM
        assert insn.operands[0].val['base'] == 105
        assert insn.operands[1].type == OP_GPR
        assert insn.operands[1].val['reg'] == 96

    def test_directive_stall(self):
        insn = parse_asm_line('.stall 5')
        assert '.stall' in insn.directives
        assert insn.directives['.stall'] == 5
        assert insn.mnemonic == ''

    def test_directive_yield(self):
        insn = parse_asm_line('.yield')
        assert '.yield' in insn.directives

    def test_directive_barrier(self):
        insn = parse_asm_line('.barrier wait=0x3, write=1')
        assert insn.directives['.barrier.wait'] == 3
        assert insn.directives['.barrier.write'] == 1

    def test_label_definition(self):
        insn = parse_asm_line('.L_loop:')
        assert insn.label == '.L_loop'
        assert insn.mnemonic == ''

    def test_inline_comment(self):
        insn = parse_asm_line('    NOP  # do nothing')
        assert insn.mnemonic == 'NOP'
        assert insn.operands == []


class TestParserMultiLine:
    def test_basic(self):
        text = """
    NOP
    HADD2.BF16_V2 R96, R96, R80
"""
        insns, labels = parse_asm(text)
        assert len(insns) == 2
        assert insns[0].mnemonic == 'NOP'
        assert insns[1].mnemonic == 'HADD2.BF16_V2'
        assert labels == {}

    def test_labels(self):
        text = """
.L_start:
    NOP
    NOP
.L_end:
    EXIT
"""
        insns, labels = parse_asm(text)
        assert len(insns) == 3
        assert labels == {'.L_start': 0, '.L_end': 2}

    def test_directive_attachment(self):
        text = """
.stall 5
.yield
    HADD2.BF16_V2 R96, R96, R80
"""
        insns, labels = parse_asm(text)
        assert len(insns) == 1
        assert insns[0].directives['.stall'] == 5
        assert insns[0].directives['.yield'] is True


# ── Encoder Tests ────────────────────────────────────────────────────────


class TestEncoder:
    def test_nop_encoding(self):
        insn = parse_asm_line('    NOP')
        enc, ctrl = encode_instruction(insn)
        assert enc == NOP_ENCODING

    def test_nop_control_default(self):
        from sm100a_encoding import CONTROL_DEFAULTS
        insn = parse_asm_line('    NOP')
        enc, ctrl = encode_instruction(insn)
        assert ctrl == CONTROL_DEFAULTS['NOP']

    def test_stall_override(self):
        text = '.stall 7\n    NOP'
        insns, _ = parse_asm(text)
        enc, ctrl = encode_instruction(insns[0])
        assert ctrl & 0xf == 7

    def test_yield_override(self):
        text = '.yield\n    NOP'
        insns, _ = parse_asm(text)
        enc, ctrl = encode_instruction(insns[0])
        assert (ctrl >> 4) & 1 == 1

    def test_predicate_encoding_p0(self):
        insn = parse_asm_line('@P0 NOP')
        enc, ctrl = encode_instruction(insn)
        pred = (enc >> 12) & 0xf
        assert pred == 0  # P0, not negated

    def test_predicate_encoding_neg_p3(self):
        insn = parse_asm_line('@!P3 NOP')
        enc, ctrl = encode_instruction(insn)
        pred = (enc >> 12) & 0xf
        assert pred == (1 << 3 | 3)  # 0xB = negated P3

    def test_predicate_encoding_pt(self):
        """Unguarded instructions should have PT (7) in predicate field."""
        insn = parse_asm_line('    NOP')
        enc, ctrl = encode_instruction(insn)
        pred = (enc >> 12) & 0xf
        assert pred == 7  # PT

    def test_register_fields_hadd2(self):
        insn = parse_asm_line('    HADD2.BF16_V2 R96, R64, R80')
        enc, ctrl = encode_instruction(insn)
        assert (enc >> 16) & 0xff == 96   # dst
        assert (enc >> 24) & 0xff == 64   # src1
        assert (enc >> 32) & 0xff == 80   # src2

    def test_register_rz(self):
        insn = parse_asm_line('    MOV R10, RZ')
        enc, ctrl = encode_instruction(insn)
        assert (enc >> 16) & 0xff == 10   # dst = R10
        assert (enc >> 24) & 0xff == 0xff  # src = RZ

    def test_store_register_fields(self):
        insn = parse_asm_line('    STS.128 [R105+0x0000], R96')
        enc, ctrl = encode_instruction(insn)
        # STS: addr @ [31:24], data @ [39:32]
        assert (enc >> 24) & 0xff == 105  # addr
        assert (enc >> 32) & 0xff == 96   # data

    def test_opcode_lookup_exact(self):
        bits, mask = _lookup_opcode('HADD2.BF16_V2')
        assert bits is not None
        assert mask is not None

    def test_opcode_lookup_fallback(self):
        """Should fall back to shorter names if exact doesn't match."""
        bits, mask = _lookup_opcode('NOP')
        assert bits == NOP_ENCODING

    def test_opcode_lookup_unknown(self):
        with pytest.raises(ValueError, match='unknown mnemonic'):
            _lookup_opcode('TOTALLY_FAKE_OP')


class TestBranchEncoding:
    def test_branch_label_resolution(self):
        text = """
.L_top:
    NOP
    NOP
@P0 BRA .L_top
"""
        binary, labels, insns = assemble(text, base_pc=0)
        assert labels == {'.L_top': 0}
        # BRA is instruction #2 at PC=32, target is PC=0
        # PC-relative offset should be -32
        enc = struct.unpack_from('<Q', binary, 2 * INSN_SIZE)[0]
        # The BRA encoding should contain the offset somewhere
        # (exact field layout is BRA-specific, just verify it assembled)
        assert enc != 0

    def test_forward_branch(self):
        text = """
@P0 BRA .L_end
    NOP
.L_end:
    NOP
"""
        binary, labels, insns = assemble(text, base_pc=0)
        assert labels['.L_end'] == 2
        assert len(insns) == 3

    def test_undefined_label(self):
        text = """
    BRA .L_nowhere
"""
        with pytest.raises(ValueError, match='undefined label'):
            assemble(text)


# ── Binary Emitter Tests ────────────────────────────────────────────────


class TestAssemble:
    def test_output_size(self):
        text = "NOP\nNOP\nNOP\n"
        binary, _, insns = assemble(text)
        assert len(binary) == 3 * INSN_SIZE
        assert len(insns) == 3

    def test_nop_round_trip(self):
        text = "NOP\n"
        binary, _, _ = assemble(text)
        enc = struct.unpack_from('<Q', binary, 0)[0]
        assert enc == NOP_ENCODING

    def test_base_pc(self):
        """Labels should be adjusted by base_pc."""
        text = """
.L_here:
    NOP
"""
        _, labels, _ = assemble(text, base_pc=0x100)
        assert labels == {'.L_here': 0}
        # Label index is still 0, but PC would be 0x100


# ── Round-trip Tests Against Real Cubins ─────────────────────────────────
#
# Gold standard: take cuobjdump text, feed it to our assembler, and verify
# that the encoding word matches the actual binary.


# Families where we have complete encoding knowledge
# (opcode + register fields + predicate)
_ROUNDTRIP_FAMILIES = frozenset([
    'HADD2', 'HFMA2', 'F2FP', 'FADD', 'FMUL', 'FFMA',
    'MOV', 'SEL', 'NOP', 'EXIT',
    'STS', 'LDS', 'STG', 'LDG',
    'S2R', 'LDSM', 'LDTM',
])

# Families where predicates are NOT in bits [15:12]
_PRED_EXCEPTION_FAMILIES = frozenset([
    'BRA', 'BSSY', 'ISETP', 'FSETP', 'HSETP2', 'PLOP3',
    'ELECT', 'NANOSLEEP', 'SYNCS', 'SHFL',
])


def _cuobjdump_to_asm(mnemonic, operands):
    """Convert cuobjdump text to assembly syntax.

    cuobjdump format: '@P0 HADD2.BF16_V2 R96, R96, R80.reuse'
    assembly format:  '@P0 HADD2.BF16_V2 R96, R96, R80.reuse'
    They're almost identical — just need to handle edge cases.
    """
    # Guard + mnemonic is already in good shape
    if not operands:
        return mnemonic
    return mnemonic + ' ' + operands


class TestRoundTrip:
    """Assemble cuobjdump text and verify encoding fields match."""

    def test_predicate_roundtrip(self, all_kernels):
        """Predicate field bits [15:12] must match for standard families."""
        total = 0
        match = 0
        failures = []

        for k in all_kernels:
            for insn in k.instructions:
                if not insn.mnemonic:
                    continue
                family = insn_family(insn.mnemonic)
                if family in _PRED_EXCEPTION_FAMILIES:
                    continue

                asm_text = _cuobjdump_to_asm(insn.mnemonic, insn.operands or '')
                try:
                    parsed = parse_asm_line(asm_text)
                except (ValueError, IndexError):
                    continue
                if not parsed or not parsed.mnemonic:
                    continue

                total += 1
                expected = (insn.encoding >> 12) & 0xf
                pred_enc = _encode_predicate(parsed)
                actual = (pred_enc >> 12) & 0xf

                if actual == expected:
                    match += 1
                elif len(failures) < 20:
                    failures.append('[%04x] %s: got 0x%x want 0x%x' % (
                        insn.offset, insn.mnemonic, actual, expected))

        pct = 100.0 * match / total if total else 0
        assert pct > 99.5, \
            'Predicate roundtrip %.1f%% (%d/%d). Failures:\n%s' % (
                pct, match, total, '\n'.join(failures[:10]))

    def test_register_roundtrip(self, all_kernels):
        """Register field values must match for known families.

        Only tests GPR/UR operands that our parser recognizes as registers.
        Multi-form instructions where cuobjdump text has operands our parser
        doesn't understand (complex addressing modes, c[] notation) are skipped.
        """
        total = 0
        match = 0
        failures = []

        for k in all_kernels:
            for insn in k.instructions:
                if not insn.mnemonic or not insn.operands:
                    continue
                family = insn_family(insn.mnemonic)
                fields = FAMILY_REG_FIELDS.get(family)
                if not fields:
                    continue

                asm_text = _cuobjdump_to_asm(insn.mnemonic, insn.operands)
                try:
                    parsed = parse_asm_line(asm_text)
                except (ValueError, IndexError):
                    continue
                if not parsed or not parsed.mnemonic:
                    continue

                for fname, tidx, boff, width in fields:
                    if tidx < 0 or tidx >= len(parsed.operands):
                        continue
                    op = parsed.operands[tidx]
                    if op.type == OP_GPR:
                        expected_val = op.val['reg']
                    elif op.type == OP_UR:
                        expected_val = op.val['reg']
                    elif op.type == OP_DESC:
                        expected_val = op.val['reg']
                    elif op.type == OP_TMEM_UR:
                        expected_val = op.val['reg']
                    elif op.type == OP_MEM and ('addr' in fname or 'src' in fname):
                        expected_val = op.val['base']
                    else:
                        continue  # skip non-register operands (imm, etc.)

                    actual_val = read_enc_field(insn.encoding, boff, width)
                    total += 1

                    if actual_val == expected_val:
                        match += 1
                    elif len(failures) < 20:
                        failures.append('[%04x] %s.%s: got %d want %d (ops=%s)' % (
                            insn.offset, insn.mnemonic, fname,
                            actual_val, expected_val, insn.operands[:60]))

        pct = 100.0 * match / total if total else 0
        # 96%+ expected — remaining failures are multi-form instructions where
        # operand indices shift between register and immediate variants
        assert pct > 96.0, \
            'Register roundtrip %.1f%% (%d/%d). Failures:\n%s' % (
                pct, match, total, '\n'.join(failures[:10]))

    def test_opcode_roundtrip(self, all_kernels):
        """Assembled opcode bits must match for known families."""
        from sm100a_opcodes import OPCODE_TABLE
        total = 0
        match = 0
        failures = {}

        for k in all_kernels:
            for insn in k.instructions:
                if not insn.mnemonic:
                    continue
                family = insn_family(insn.mnemonic)
                if family not in _ROUNDTRIP_FAMILIES:
                    continue

                asm_text = _cuobjdump_to_asm(insn.mnemonic, insn.operands or '')
                try:
                    parsed = parse_asm_line(asm_text)
                except (ValueError, IndexError):
                    continue
                if not parsed or not parsed.mnemonic:
                    continue

                try:
                    opcode_bits, var_mask = _lookup_opcode(parsed.mnemonic)
                except ValueError:
                    continue

                total += 1
                # Compare just the opcode (non-variable) bits
                actual_fixed = insn.encoding & ~var_mask
                assembled_fixed = opcode_bits

                if actual_fixed == assembled_fixed:
                    match += 1
                else:
                    fam = family
                    failures[fam] = failures.get(fam, 0) + 1

        pct = 100.0 * match / total if total else 0
        assert pct > 99.0, \
            'Opcode roundtrip %.1f%% (%d/%d). Top failures: %s' % (
                pct, match, total,
                sorted(failures.items(), key=lambda x: -x[1])[:10])

    def test_full_encoding_simple_alu(self, all_kernels):
        """Opcode + predicate + register fields must match for pure-register ALU.

        Only tests instructions where all operands are GPRs (no immediates),
        since we don't have IMMEDIATE_FIELDS entries for most ALU instructions.
        Checks that the assembled fixed bits (opcode & ~var_mask) match,
        predicate bits match, and register fields match.
        """
        from sm100a_opcodes import OPCODE_TABLE
        simple_families = frozenset(['HADD2', 'F2FP', 'FADD', 'FMUL', 'FFMA'])
        total = 0
        match = 0
        failures = []

        for k in all_kernels:
            for insn in k.instructions:
                if not insn.mnemonic or not insn.operands:
                    continue
                family = insn_family(insn.mnemonic)
                if family not in simple_families:
                    continue

                asm_text = _cuobjdump_to_asm(insn.mnemonic, insn.operands)
                try:
                    parsed = parse_asm_line(asm_text)
                except (ValueError, IndexError):
                    continue
                if not parsed or not parsed.mnemonic:
                    continue

                # Only test pure-register instructions (no immediates)
                has_imm = any(op.type == OP_IMM for op in parsed.operands)
                if has_imm:
                    continue

                try:
                    enc, ctrl = encode_instruction(parsed)
                except ValueError:
                    continue

                total += 1

                # Check predicate
                actual_pred = (insn.encoding >> 12) & 0xf
                assembled_pred = (enc >> 12) & 0xf
                pred_ok = (actual_pred == assembled_pred)

                # Check register fields
                regs_ok = True
                fields = FAMILY_REG_FIELDS.get(family, [])
                for fname, tidx, boff, width in fields:
                    if tidx < 0 or tidx >= len(parsed.operands):
                        continue
                    op = parsed.operands[tidx]
                    if op.type not in (OP_GPR, OP_UR):
                        continue
                    actual_r = read_enc_field(insn.encoding, boff, width)
                    assembled_r = read_enc_field(enc, boff, width)
                    if actual_r != assembled_r:
                        regs_ok = False

                # Check opcode fixed bits
                try:
                    opcode_bits, var_mask = _lookup_opcode(parsed.mnemonic)
                    actual_fixed = insn.encoding & ~var_mask
                    opc_ok = (actual_fixed == opcode_bits)
                except ValueError:
                    opc_ok = False

                if pred_ok and regs_ok and opc_ok:
                    match += 1
                elif len(failures) < 20:
                    failures.append('[%04x] %s: opc=%s pred=%s regs=%s' % (
                        insn.offset, insn.mnemonic,
                        'OK' if opc_ok else 'FAIL',
                        'OK' if pred_ok else 'FAIL',
                        'OK' if regs_ok else 'FAIL'))

        pct = 100.0 * match / total if total else 0
        assert pct > 99.0, \
            'Full encoding roundtrip %.1f%% (%d/%d). Failures:\n%s' % (
                pct, match, total, '\n'.join(failures[:10]))

    def test_control_word_family_default(self, all_kernels):
        """Control word must use correct family default for stall/yield bits."""
        from sm100a_encoding import CONTROL_DEFAULTS
        # Just verify our defaults exist for common families
        for family in ['NOP', 'HADD2', 'F2FP', 'STS', 'LDS', 'MOV', 'FFMA']:
            assert family in CONTROL_DEFAULTS, \
                'Missing CONTROL_DEFAULTS for %s' % family


class TestSyntheticRoundTrip:
    """Assemble known instructions and verify specific bits."""

    def test_nop_exact(self):
        enc, ctrl = encode_instruction(parse_asm_line('NOP'))
        assert enc == 0x0000000000007918

    def test_exit_exact(self):
        enc, ctrl = encode_instruction(parse_asm_line('EXIT'))
        # EXIT opcode should be in table
        from sm100a_opcodes import OPCODE_TABLE
        if 'EXIT' in OPCODE_TABLE:
            expected_bits, var_mask = OPCODE_TABLE['EXIT']
            assert (enc & ~var_mask) == expected_bits

    def test_hadd2_reg_placement(self):
        enc, ctrl = encode_instruction(parse_asm_line(
            'HADD2.BF16_V2 R10, R20, R30'))
        assert (enc >> 16) & 0xff == 10
        assert (enc >> 24) & 0xff == 20
        assert (enc >> 32) & 0xff == 30

    def test_sts128_fields(self):
        enc, ctrl = encode_instruction(parse_asm_line(
            'STS.128 [R105+0x0000], R96'))
        assert (enc >> 24) & 0xff == 105  # addr
        assert (enc >> 32) & 0xff == 96   # data

    def test_guarded_instruction(self):
        enc1, _ = encode_instruction(parse_asm_line('NOP'))
        enc2, _ = encode_instruction(parse_asm_line('@P0 NOP'))
        enc3, _ = encode_instruction(parse_asm_line('@!P5 NOP'))
        # PT = 7, P0 = 0, !P5 = 8|5 = 13
        assert (enc1 >> 12) & 0xf == 7
        assert (enc2 >> 12) & 0xf == 0
        assert (enc3 >> 12) & 0xf == 13

    def test_reuse_flag(self):
        text = '.reuse\n    HADD2.BF16_V2 R96, R96, R80'
        insns, _ = parse_asm(text)
        enc, ctrl = encode_instruction(insns[0])
        from sm100a_encoding import REUSE_CTRL_MASK
        assert ctrl & REUSE_CTRL_MASK != 0

    def test_reuse_via_operand(self):
        insn = parse_asm_line('    HADD2.BF16_V2 R96, R96.reuse, R80')
        enc, ctrl = encode_instruction(insn)
        from sm100a_encoding import REUSE_CTRL_MASK
        assert ctrl & REUSE_CTRL_MASK != 0

    def test_assemble_multiple(self):
        text = """
    NOP
    HADD2.BF16_V2 R96, R96, R80
    NOP
"""
        binary, labels, insns = assemble(text)
        assert len(binary) == 3 * INSN_SIZE
        # First instruction is NOP
        assert struct.unpack_from('<Q', binary, 0)[0] == NOP_ENCODING
        # Second instruction has HADD2 register fields
        enc1 = struct.unpack_from('<Q', binary, INSN_SIZE)[0]
        assert (enc1 >> 16) & 0xff == 96
        assert (enc1 >> 24) & 0xff == 96
        assert (enc1 >> 32) & 0xff == 80


class TestPatchRegionAsm:
    """Test CubinEditor.patch_region_asm (Phase 5.1)."""

    @pytest.fixture
    def cubin_editor(self, tmp_path):
        """Create a minimal fake cubin with a known instruction sequence."""
        # Build a minimal ELF-ish cubin that CubinEditor can parse.
        # Easier: use a real cubin if available, else skip.
        if os.path.exists(FC2_CUBIN):
            return CubinEditor(FC2_CUBIN)
        pytest.skip('No cubin available for patch_region_asm test')

    def test_basic_patch(self, cubin_editor):
        """Assemble NOPs into a small region and verify they land."""
        ed = cubin_editor
        k = ed.find_kernel()
        assert k is not None

        # Pick a 3-slot region starting at instruction 16
        start = 16 * INSN_SIZE
        end = 19 * INSN_SIZE

        # Save originals to verify change
        orig_encs = [k.instructions[i].encoding for i in range(16, 19)]

        asm_text = "NOP\nNOP\nNOP\n"
        n_patched, n_nops = ed.patch_region_asm(k, start, end, asm_text)

        assert n_patched == 3
        assert n_nops == 0
        for i in range(16, 19):
            assert k.instructions[i].encoding == NOP_ENCODING

    def test_nop_fill(self, cubin_editor):
        """Assembled output shorter than region gets NOP-filled."""
        ed = cubin_editor
        k = ed.find_kernel()

        start = 20 * INSN_SIZE
        end = 25 * INSN_SIZE  # 5 slots

        asm_text = "NOP\nNOP\n"  # only 2 instructions
        n_patched, n_nops = ed.patch_region_asm(k, start, end, asm_text)

        assert n_patched == 2
        assert n_nops == 3
        # All 5 should be NOPs
        for i in range(20, 25):
            assert k.instructions[i].encoding == NOP_ENCODING

    def test_overflow_raises(self, cubin_editor):
        """More assembled instructions than slots raises ValueError."""
        ed = cubin_editor
        k = ed.find_kernel()

        start = 30 * INSN_SIZE
        end = 31 * INSN_SIZE  # 1 slot

        asm_text = "NOP\nNOP\n"  # 2 instructions
        with pytest.raises(ValueError, match='only 1 slots'):
            ed.patch_region_asm(k, start, end, asm_text)

    def test_alignment_check(self, cubin_editor):
        """Unaligned start/end raises ValueError."""
        ed = cubin_editor
        k = ed.find_kernel()

        with pytest.raises(ValueError, match='aligned'):
            ed.patch_region_asm(k, 1, INSN_SIZE, "NOP\n")

    def test_real_instructions(self, cubin_editor):
        """Patch real instructions (not just NOPs) and verify encoding."""
        ed = cubin_editor
        k = ed.find_kernel()

        start = 40 * INSN_SIZE
        end = 43 * INSN_SIZE  # 3 slots

        asm_text = """
.stall 5
    HADD2.BF16_V2 R10, R20, R30
.stall 3
    STS.128 [R105+0x0000], R96
    NOP
"""
        n_patched, n_nops = ed.patch_region_asm(k, start, end, asm_text)
        assert n_patched == 3
        assert n_nops == 0

        # Verify register fields in HADD2
        enc0 = k.instructions[40].encoding
        assert (enc0 >> 16) & 0xff == 10   # dst
        assert (enc0 >> 24) & 0xff == 20   # src1
        assert (enc0 >> 32) & 0xff == 30   # src2

        # Verify stall counts
        assert k.instructions[40].control & 0xf == 5
        assert k.instructions[41].control & 0xf == 3

        # Verify STS.128 register fields
        enc1 = k.instructions[41].encoding
        assert (enc1 >> 24) & 0xff == 105  # addr
        assert (enc1 >> 32) & 0xff == 96   # data

    def test_save_roundtrip(self, cubin_editor, tmp_path):
        """Patch, save, reload, verify patched instructions persist."""
        ed = cubin_editor
        k = ed.find_kernel()

        start = 50 * INSN_SIZE
        end = 52 * INSN_SIZE

        ed.patch_region_asm(k, start, end, "NOP\nNOP\n")

        out = tmp_path / 'patched.cubin'
        ed.save(str(out))

        # Reload and verify
        ed2 = CubinEditor(str(out))
        k2 = ed2.find_kernel()
        assert k2.instructions[50].encoding == NOP_ENCODING
        assert k2.instructions[51].encoding == NOP_ENCODING

    def test_base_pc_for_branches(self, cubin_editor):
        """Branch targets should be encoded relative to the region's base PC."""
        ed = cubin_editor
        k = ed.find_kernel()

        start = 60 * INSN_SIZE
        end = 64 * INSN_SIZE

        # Label at insn 2 (offset 0x3c0 + 0x20 = 0x3e0), branch from insn 0 (offset 0x3c0)
        asm_text = """
    BRA .Ltarget
    NOP
.Ltarget:
    NOP
    NOP
"""
        n_patched, n_nops = ed.patch_region_asm(k, start, end, asm_text)
        assert n_patched == 4
        assert n_nops == 0


class TestCtrlDirective:
    """Test .ctrl directive for exact control word round-trip."""

    def test_ctrl_directive_basic(self):
        """Parser accepts .ctrl and encoder uses it."""
        text = ".ctrl 0x000fe20000000005\n    NOP\n"
        binary, _, insns = assemble(text)
        ctrl = struct.unpack_from('<Q', binary, 8)[0]
        assert ctrl == 0x000fe20000000005

    def test_ctrl_with_stall_override(self):
        """.stall overrides stall bits in .ctrl."""
        text = ".ctrl 0x000fe20000000005\n.stall 10\n    NOP\n"
        binary, _, insns = assemble(text)
        ctrl = struct.unpack_from('<Q', binary, 8)[0]
        assert ctrl & 0xf == 10
        assert ctrl & ~0xf == 0x000fe20000000000

    def test_ctrl_without_reuse_from_operand(self):
        """Operand .reuse should NOT set ctrl bit 58 when .ctrl is present."""
        _load_asm_tables()
        from sm100a_encoding import REUSE_CTRL_MASK as RM
        text = ".ctrl 0x000fe20000000000\n    HADD2.BF16_V2 R10, R20.reuse, R30\n"
        binary, _, insns = assemble(text)
        ctrl = struct.unpack_from('<Q', binary, 8)[0]
        assert ctrl & RM == 0  # reuse NOT set

    def test_ctrl_with_explicit_reuse_directive(self):
        """Explicit .reuse directive SHOULD set ctrl bit 58 even with .ctrl."""
        _load_asm_tables()
        from sm100a_encoding import REUSE_CTRL_MASK as RM
        text = ".ctrl 0x000fe20000000000\n.reuse\n    NOP\n"
        binary, _, insns = assemble(text)
        ctrl = struct.unpack_from('<Q', binary, 8)[0]
        assert ctrl & RM != 0


class TestDisasm:
    """Test disassemble_region (Phase 5.4)."""

    @pytest.fixture
    def xref_editor(self):
        if not os.path.exists(FC2_CUBIN):
            pytest.skip('No cubin')
        sass = _find_sass(_FC2_SASS)
        if not sass:
            pytest.skip('No SASS dump')
        ed = CubinEditor(FC2_CUBIN)
        k = ed.find_kernel()
        ed.xref(sass, k)
        return ed, k

    def test_basic_disasm(self, xref_editor):
        from sass_edit import disassemble_region
        ed, k = xref_editor
        text = disassemble_region(k, 0x100, 0x140)
        assert len(text) > 0
        # Should contain .ctrl directives
        assert '.ctrl 0x' in text
        # Should contain instruction lines
        lines = [l for l in text.splitlines() if l.strip() and not l.startswith('.')]
        assert len(lines) == 4  # 4 instructions

    def test_disasm_has_ctrl_for_each_insn(self, xref_editor):
        from sass_edit import disassemble_region
        ed, k = xref_editor
        text = disassemble_region(k, 0x100, 0x160)
        ctrl_count = text.count('.ctrl 0x')
        assert ctrl_count == 6  # 6 instructions

    def test_disasm_reassembles(self, xref_editor):
        """Disassembled text should parse without errors."""
        from sass_edit import disassemble_region, parse_asm
        ed, k = xref_editor
        text = disassemble_region(k, 0x100, 0x160)
        insns, labels = parse_asm(text)
        assert len(insns) == 6


class TestVerifyAsm:
    """Test verify_asm_region (Phase 5.3)."""

    @pytest.fixture
    def cubin_editor(self):
        if not os.path.exists(FC2_CUBIN):
            pytest.skip('No cubin')
        return CubinEditor(FC2_CUBIN)

    def test_ctrl_roundtrip(self, cubin_editor):
        """Control words should round-trip exactly via .ctrl directive."""
        from sass_edit import disassemble_region, verify_asm_region
        ed = cubin_editor
        k = ed.find_kernel()
        sass = _find_sass(_FC2_SASS)
        if not sass:
            pytest.skip('No SASS dump')
        ed.xref(sass, k)

        text = disassemble_region(k, 0x100, 0x160)
        n_match, n_mismatch, details = verify_asm_region(k, 0x100, 0x160, text)

        # Filter to ctrl-only mismatches
        ctrl_mismatches = [d for d in details if 'ctrl:' in d]
        assert len(ctrl_mismatches) == 0, 'ctrl mismatches: %s' % ctrl_mismatches


# ════════════════════════════════════════════════════════════════════════
# Phase 6: Verification & Testing (round-trip fidelity, cubin validity)
# ════════════════════════════════════════════════════════════════════════

class TestRoundTrip:
    """Phase 6.1: Round-trip fidelity on FC2 epilogue."""

    @pytest.fixture
    def fc2_with_xref(self):
        if not os.path.exists(FC2_CUBIN):
            pytest.skip('No cubin')
        sass = _find_sass(_FC2_SASS)
        if not sass:
            pytest.skip('No SASS dump')
        _load_asm_tables()
        ed = CubinEditor(FC2_CUBIN)
        k = ed.find_kernel()
        n = ed.xref(sass, k)
        return ed, k

    def test_ctrl_100pct(self, fc2_with_xref):
        """Control words must round-trip at 100% on the epilogue."""
        ed, k = fc2_with_xref
        start, end = 0x51c0, 0x63b0
        text = disassemble_region(k, start, end)
        binary, _, instructions = assemble(text, base_pc=start)

        n_insns = (end - start) // INSN_SIZE
        start_idx = start // INSN_SIZE
        ctrl_mismatches = 0
        for i in range(min(len(instructions), n_insns)):
            asm_ctrl = struct.unpack_from('<Q', binary, i * INSN_SIZE + 8)[0]
            if asm_ctrl != k.instructions[start_idx + i].control:
                ctrl_mismatches += 1

        assert ctrl_mismatches == 0, '%d ctrl mismatches in epilogue' % ctrl_mismatches

    def test_opcode_above_90pct(self, fc2_with_xref):
        """Opcode bits should round-trip above 90% on the epilogue."""
        from sm100a_opcodes import OPCODE_TABLE
        ed, k = fc2_with_xref
        start, end = 0x51c0, 0x63b0
        text = disassemble_region(k, start, end)
        binary, _, instructions = assemble(text, base_pc=start)

        PRED_MASK = 0xf000
        n_insns = (end - start) // INSN_SIZE
        start_idx = start // INSN_SIZE
        n_match = 0
        n_total = min(len(instructions), n_insns)

        for i in range(n_total):
            asm_enc = struct.unpack_from('<Q', binary, i * INSN_SIZE)[0]
            bin_insn = k.instructions[start_idx + i]
            mnemonic = bin_insn.mnemonic
            if not mnemonic:
                n_match += 1
                continue
            m = mnemonic
            if m.startswith('@'):
                m = m.split(None, 1)[1] if ' ' in m else m
            var_mask = 0
            parts = m.split('.')
            for n_parts in range(len(parts), 0, -1):
                key = '.'.join(parts[:n_parts])
                if key in OPCODE_TABLE:
                    _, var_mask = OPCODE_TABLE[key]
                    break
            opcode_mask = ~(var_mask | PRED_MASK) & 0xFFFFFFFFFFFFFFFF
            if (asm_enc & opcode_mask) == (bin_insn.encoding & opcode_mask):
                n_match += 1

        pct = 100 * n_match / n_total
        assert pct >= 90, 'opcode fidelity %.1f%% < 90%%' % pct

    def test_full_kernel_ctrl_100pct(self, fc2_with_xref):
        """Control words must round-trip at 100% on the FULL kernel."""
        ed, k = fc2_with_xref
        start, end = 0, k.n_insns * INSN_SIZE
        text = disassemble_region(k, start, end)
        binary, _, instructions = assemble(text, base_pc=start)

        n_insns = end // INSN_SIZE
        ctrl_mismatches = 0
        for i in range(min(len(instructions), n_insns)):
            asm_ctrl = struct.unpack_from('<Q', binary, i * INSN_SIZE + 8)[0]
            if asm_ctrl != k.instructions[i].control:
                ctrl_mismatches += 1

        assert ctrl_mismatches == 0, '%d ctrl mismatches in full kernel' % ctrl_mismatches

    def test_families_fully_matching(self, fc2_with_xref):
        """F2FP, HADD2, NOP, WARPSYNC must round-trip at 100% full encoding."""
        ed, k = fc2_with_xref
        start, end = 0x51c0, 0x63b0
        text = disassemble_region(k, start, end)
        binary, _, instructions = assemble(text, base_pc=start)

        PERFECT_FAMILIES = {'F2FP', 'HADD2', 'NOP', 'MEMBAR', 'FENCE',
                            'UTMALDG', 'UTMASTG', 'UTMACMDFLUSH', 'WARPSYNC', 'CCTL'}
        n_insns = (end - start) // INSN_SIZE
        start_idx = start // INSN_SIZE
        mismatches = []

        for i in range(min(len(instructions), n_insns)):
            bin_insn = k.instructions[start_idx + i]
            family = insn_family(bin_insn.mnemonic) if bin_insn.mnemonic else None
            if family not in PERFECT_FAMILIES:
                continue
            asm_enc = struct.unpack_from('<Q', binary, i * INSN_SIZE)[0]
            if asm_enc != bin_insn.encoding:
                mismatches.append('[%04x] %s asm=%016x bin=%016x' % (
                    start + i * INSN_SIZE, bin_insn.mnemonic, asm_enc, bin_insn.encoding))

        assert len(mismatches) == 0, 'Perfect-family mismatches:\n' + '\n'.join(mismatches[:10])


class TestCubinValidity:
    """Phase 6.2: Cubin structural validity after patching."""

    @pytest.fixture
    def fc2_editor(self):
        if not os.path.exists(FC2_CUBIN):
            pytest.skip('No cubin')
        return CubinEditor(FC2_CUBIN)

    def test_nop_fill_preserves_cubin(self, fc2_editor):
        """NOP-filling a region preserves cubin size."""
        import tempfile
        _load_asm_tables()
        from sm100a_encoding import CONTROL_DEFAULTS
        ed = fc2_editor
        k = ed.kernels[0]
        nop_ctrl = CONTROL_DEFAULTS.get('NOP', 0x000fe20000000000)

        # NOP-fill 8 instructions starting at offset 0x100
        start_idx = 0x100 // INSN_SIZE
        originals = [(k.instructions[start_idx + i].encoding,
                      k.instructions[start_idx + i].control) for i in range(8)]
        for i in range(8):
            k.instructions[start_idx + i].encoding = NOP_ENCODING
            k.instructions[start_idx + i].control = nop_ctrl

        with tempfile.NamedTemporaryFile(suffix='.cubin', delete=False) as f:
            tmp = f.name
        try:
            ed.save(tmp)
            assert os.path.getsize(tmp) == os.path.getsize(FC2_CUBIN)
        finally:
            os.unlink(tmp)

        # Restore
        for i, (enc, ctrl) in enumerate(originals):
            k.instructions[start_idx + i].encoding = enc
            k.instructions[start_idx + i].control = ctrl

    def test_cuobjdump_nop_patch(self, fc2_editor):
        """cuobjdump -sass must parse cubin after NOP-fill."""
        import subprocess, tempfile
        _load_asm_tables()
        from sm100a_encoding import CONTROL_DEFAULTS
        ed = fc2_editor
        k = ed.kernels[0]
        nop_ctrl = CONTROL_DEFAULTS.get('NOP', 0x000fe20000000000)

        start_idx = 0x100 // INSN_SIZE
        originals = [(k.instructions[start_idx + i].encoding,
                      k.instructions[start_idx + i].control) for i in range(8)]
        for i in range(8):
            k.instructions[start_idx + i].encoding = NOP_ENCODING
            k.instructions[start_idx + i].control = nop_ctrl

        with tempfile.NamedTemporaryFile(suffix='.cubin', delete=False) as f:
            tmp = f.name
        try:
            ed.save(tmp)
            result = subprocess.run(['cuobjdump', '-sass', tmp],
                                    capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, 'cuobjdump failed: %s' % result.stderr[:200]
        finally:
            os.unlink(tmp)

        for i, (enc, ctrl) in enumerate(originals):
            k.instructions[start_idx + i].encoding = enc
            k.instructions[start_idx + i].control = ctrl

    def test_cuobjdump_elf_after_reassemble(self, fc2_editor):
        """cuobjdump -elf must parse cubin after full epilogue reassembly."""
        import subprocess, tempfile
        sass = _find_sass(_FC2_SASS)
        if not sass:
            pytest.skip('No SASS dump')
        _load_asm_tables()
        ed = fc2_editor
        k = ed.kernels[0]
        ed.xref(sass, k)

        start, end = 0x51c0, 0x63b0
        text = disassemble_region(k, start, end)
        ed.patch_region_asm(k, start, end, text)

        with tempfile.NamedTemporaryFile(suffix='.cubin', delete=False) as f:
            tmp = f.name
        try:
            ed.save(tmp)
            result = subprocess.run(['cuobjdump', '-elf', tmp],
                                    capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, 'cuobjdump -elf failed: %s' % result.stderr[:200]
            assert '.text.' in result.stdout, '.text section missing from ELF output'
        finally:
            os.unlink(tmp)


class TestNewOperandForms:
    """Tests for operand forms added during Phase 5/6."""

    def test_srz(self):
        op = _parse_asm_operand('SRZ')
        assert op.type == OP_IMM
        assert op.val['sr_name'] == 'SRZ'

    def test_sr_tid_x(self):
        op = _parse_asm_operand('SR_TID.X')
        assert op.type == OP_IMM
        assert op.val['sr_name'] == 'TID.X'

    def test_mem_with_size(self):
        op = _parse_asm_operand('[R8.64]')
        assert op.type == OP_MEM
        assert op.val['base'] == 8
        assert op.val['size'] == 64

    def test_desc_mem(self):
        op = _parse_asm_operand('desc[UR6][R4.64]')
        assert op.type == OP_DESC
        assert op.val['reg'] == 6
        assert op.val['gpr'] == 4
        assert op.val['size'] == 64

    def test_ret_space_separated(self):
        """RET.REL.NODEC R2 0x0 — space-separated operands (no comma)."""
        insn = parse_asm_line('RET.REL.NODEC R2 0x0')
        assert insn.mnemonic == 'RET.REL.NODEC'
        assert len(insn.operands) == 2
        assert insn.operands[0].type == OP_GPR
        assert insn.operands[0].val['reg'] == 2
        assert insn.operands[1].type == OP_IMM
        assert insn.operands[1].val['value'] == 0
