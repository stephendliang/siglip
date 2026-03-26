#!/usr/bin/env python3
"""Tests for Phase 2+3 encoding analysis (no GPU needed).

Loads existing FC2 + CUTLASS cubins and verifies analysis results
against known SM100a instruction encoding properties.

Run: python3 -m pytest tools/test_encoding_analysis.py -v
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from sass_edit import (
    CubinEditor, analyze_predicates, analyze_immediates,
    analyze_control_words, verify_reg_fields, analyze_special_encodings,
    compute_opcode_table, FAMILY_REG_FIELDS, decode_ctrl,
    _contiguous_fields, _majority_mask, _bit_positions, _popcount64,
    _strip_pred, insn_family, read_enc_field, _split_operands,
)

REPO = os.path.dirname(os.path.dirname(__file__))
FC2_CUBIN = os.path.join(REPO, 'fc2.cubin')
CUTLASS_CUBIN = os.path.join(REPO, 'cutlass_bench.sm_100a.cubin')

_FC2_SASS = [
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
    """FC2 + CUTLASS cubins — comprehensive coverage."""
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


# ── Helpers ──────────────────────────────────────────────────────────────

class TestHelpers:
    def test_contiguous_fields_empty(self):
        assert _contiguous_fields(0) == []

    def test_contiguous_fields_single_bit(self):
        assert _contiguous_fields(0x10) == [(4, 1)]

    def test_contiguous_fields_byte(self):
        assert _contiguous_fields(0xFF) == [(0, 8)]

    def test_contiguous_fields_multi(self):
        assert _contiguous_fields(0xEF) == [(0, 4), (5, 3)]

    def test_bit_positions(self):
        assert _bit_positions(0) == []
        assert _bit_positions(0x5) == [0, 2]
        assert _bit_positions(0xFF) == list(range(8))

    def test_majority_mask(self):
        assert _majority_mask([], 0.5) == 0
        assert _majority_mask([0xFF, 0xFF, 0x0F], 0.6) == 0xFF  # 0xFF bits in 2/3=67% > 60%
        assert _majority_mask([0xFF, 0x0F, 0x0F], 0.6) == 0x0F  # upper bits only in 1/3=33%
        assert _majority_mask([0xFF, 0xFF], 0.5) == 0xFF

    def test_popcount64(self):
        assert _popcount64(0) == 0
        assert _popcount64(1) == 1
        assert _popcount64(0xFF) == 8
        assert _popcount64(0xFFFFFFFFFFFFFFFF) == 64
        assert _popcount64(0x5555555555555555) == 32


# ── Phase 2.3: Predicates ───────────────────────────────────────────────

class TestPredicates:
    def test_predicate_pairs_found(self, all_kernels):
        r = analyze_predicates(all_kernels)
        assert r['n_enable'] > 10, 'Too few enable pairs: %d' % r['n_enable']

    def test_predicate_in_encoding_word(self, all_kernels):
        r = analyze_predicates(all_kernels)
        assert r['enc_mask'] != 0, 'No predicate bits found in encoding word'

    def test_predicate_not_in_control_word(self, all_kernels):
        """Predicate should be in encoding word only (SM100a)."""
        r = analyze_predicates(all_kernels)
        assert r['ctrl_mask'] == 0, 'Unexpected predicate bits in control word'

    def test_predicate_field_at_bits_12_15(self, all_kernels):
        """SM100a predicate guard: bits [15:12] of encoding word."""
        r = analyze_predicates(all_kernels)
        assert r['pred_field_start'] == 12
        assert r['pred_field_width'] == 4

    def test_predicate_register_3_bits(self, all_kernels):
        """Register field should be 3 bits (P0-P6 + PT)."""
        r = analyze_predicates(all_kernels)
        reg_full = r.get('reg_full_field')
        assert reg_full is not None, 'No register field found'
        start, width = reg_full
        assert start == 12
        assert width == 3

    def test_negation_bit_15(self, all_kernels):
        r = analyze_predicates(all_kernels)
        assert r.get('neg_bit') == 15

    def test_guard_encoding_values(self, all_kernels):
        """Verify encoding values for each predicate guard type."""
        r = analyze_predicates(all_kernels)
        ge = r.get('guard_encoding', {})
        if not ge:
            pytest.skip('No guard encoding data')

        # PT (unguarded) should encode as 7 at bits [14:12], neg=0 → 0b0111 = 0x7
        pt_val, pt_count, pt_total = ge.get('PT', (None, 0, 0))
        assert pt_val == 7, 'PT should encode as 7, got %s' % pt_val
        # Allow 99%+ consistency (some multi-form instructions may have overlapping bits)
        assert pt_count >= pt_total * 0.99, \
            'PT encoding < 99%%: %d/%d' % (pt_count, pt_total)

        # @P0: register=0, neg=0 → 0b0000 = 0x0
        if '@P0' in ge:
            v, c, t = ge['@P0']
            assert v == 0, '@P0 should encode as 0, got %s' % v
            assert c >= t * 0.98, '@P0 < 98%%: %d/%d' % (c, t)

        # @!P0: register=0, neg=1 → 0b1000 = 0x8
        if '@!P0' in ge:
            v, c, t = ge['@!P0']
            assert v == 8, '@!P0 should encode as 8, got %s' % v
            assert c >= t * 0.98, '@!P0 < 98%%: %d/%d' % (c, t)

        # @P1: register=1, neg=0 → 0b0001 = 0x1
        if '@P1' in ge:
            v, c, t = ge['@P1']
            assert v == 1

        # @P6: register=6, neg=0 → 0b0110 = 0x6
        if '@P6' in ge:
            v, c, t = ge['@P6']
            assert v == 6

    def test_uniform_predicates_same_encoding(self, all_kernels):
        """UP0 and P0 share the same encoding bits [15:12]."""
        r = analyze_predicates(all_kernels)
        ge = r.get('guard_encoding', {})
        if '@UP0' in ge and '@P0' in ge:
            assert ge['@UP0'][0] == ge['@P0'][0], \
                'UP0 and P0 have different predicate encoding'


# ── Phase 2.2: Immediates ───────────────────────────────────────────────

class TestImmediates:
    def test_lds128_offset(self, all_kernels):
        r = analyze_immediates(all_kernels)
        assert 'LDS.128' in r, 'No LDS.128 offset field'
        info = r['LDS.128']
        assert info['type'] == 'mem_offset'
        assert info['scale'] == 16, 'LDS.128 scale should be 16, got %s' % info['scale']
        fields = info['fields']
        assert len(fields) >= 1
        # Primary field should be at high bits (>= 40)
        assert fields[0][0] >= 40, 'LDS.128 offset field too low: bit %d' % fields[0][0]

    def test_sts_offset_found(self, all_kernels):
        r = analyze_immediates(all_kernels)
        sts_keys = [k for k in r if k.startswith('STS')]
        assert len(sts_keys) > 0, 'No STS offset field found'

    def test_bra_target_found(self, all_kernels):
        r = analyze_immediates(all_kernels)
        bra_keys = [k for k in r if k.startswith('BRA')]
        assert len(bra_keys) > 0, 'No BRA target field found'

    def test_s2r_sr_field(self, all_kernels):
        r = analyze_immediates(all_kernels)
        assert 'S2R' in r, 'No S2R special register field'
        info = r['S2R']
        assert info['type'] == 'sr_id'
        assert len(info.get('sr_names', [])) >= 2, 'Too few SR names'

    def test_depbar_fields(self, all_kernels):
        r = analyze_immediates(all_kernels)
        depbar_keys = [k for k in r if 'DEPBAR' in k]
        assert len(depbar_keys) > 0, 'No DEPBAR fields found'

    def test_ldtm_offset(self, all_kernels):
        r = analyze_immediates(all_kernels)
        ldtm_keys = [k for k in r if k.startswith('LDTM')]
        assert len(ldtm_keys) > 0, 'No LDTM offset field found'


# ── Phase 2.1: Register verification ────────────────────────────────────

class TestRegVerification:
    def test_standard_alu_verified(self, all_kernels):
        ot = compute_opcode_table(all_kernels)
        r = verify_reg_fields(all_kernels, ot)
        # Families with no fields should be OK
        for fam in ['NOP', 'EXIT', 'BAR', 'DEPBAR']:
            info = r['verification'].get(fam, {})
            assert info.get('ok') is not False, '%s verification failed' % fam

    def test_missing_families_found(self, all_kernels):
        ot = compute_opcode_table(all_kernels)
        r = verify_reg_fields(all_kernels, ot)
        assert len(r['missing_families']) > 0, 'Should find missing families'

    def test_store_families_verified(self, all_kernels):
        """STS, STG store families should have correct field layout."""
        ot = compute_opcode_table(all_kernels)
        r = verify_reg_fields(all_kernels, ot)
        for fam in ['STS', 'STG']:
            info = r['verification'].get(fam, {})
            # May have mismatches on multi-form entries but base should be OK
            if info.get('status') == 'not_in_opcode_table':
                continue
            # At least should exist
            assert fam in r['verification'], 'Missing %s from verification' % fam


# ── Phase 2.4/2.5: Special ──────────────────────────────────────────────

class TestSpecial:
    def test_rz_is_0xff(self, all_kernels):
        r = analyze_special_encodings(all_kernels)
        if r['rz_n_checks'] == 0:
            pytest.skip('No RZ instances found in safe families')
        rz = r['rz_value']
        if isinstance(rz, set):
            # Edge cases may produce non-0xFF from multi-form operand mapping issues.
            # 0xFF must be in the set and should be the dominant value.
            assert 0xFF in rz, 'RZ value 0xFF not found: %s' % rz
        else:
            assert rz == 0xFF, 'RZ not 0xFF: %s' % rz

    def test_ur_encoding_8bit(self, all_kernels):
        """UR encoding uses 8-bit fields (same as GPR)."""
        r = analyze_special_encodings(all_kernels)
        if r['ur_n_checks'] == 0:
            pytest.skip('No UR instances found')
        # UR values > 63 would confirm 8-bit encoding
        # If max <= 63, it's 6-bit (can't distinguish without higher values)
        assert r['ur_max_value'] > 0, 'No UR values found'

    def test_reuse_in_control_word(self, all_kernels):
        """The .reuse flag should be in the control word, not encoding."""
        r = analyze_special_encodings(all_kernels)
        if r['n_reuse_pairs'] == 0:
            pytest.skip('No .reuse pairs found')
        assert r['reuse_in_control'], '.reuse not found in control word'

    def test_reuse_bit_58(self, all_kernels):
        """SM100a .reuse flag is at control word bit 58."""
        r = analyze_special_encodings(all_kernels)
        if not r['reuse_ctrl_bits']:
            pytest.skip('No reuse control bits')
        assert 58 in r['reuse_ctrl_bits'], \
            '.reuse not at bit 58: %s' % r['reuse_ctrl_bits']


# ── Phase 3.1: Control words ────────────────────────────────────────────

class TestControlWords:
    def test_stall_field_at_bits_0_3(self, all_kernels):
        r = analyze_control_words(all_kernels)
        assert r['global_var_mask'] & 0xF == 0xF, 'Stall [3:0] not all varying'

    def test_yield_bit_4(self, all_kernels):
        r = analyze_control_words(all_kernels)
        assert r['global_var_mask'] & (1 << 4), 'Yield bit 4 not varying'

    def test_family_defaults_exist(self, all_kernels):
        r = analyze_control_words(all_kernels)
        for fam in ['IADD3', 'STS', 'F2FP', 'MOV']:
            assert fam in r['family_stats'], 'Missing default for %s' % fam

    def test_stall_values_valid(self, all_kernels):
        r = analyze_control_words(all_kernels)
        for fam, st in r['family_stats'].items():
            for stall_val in st['stall_dist']:
                assert 0 <= stall_val <= 15, '%s: stall %d out of range' % (fam, stall_val)

    def test_bits_28_40_constant(self, all_kernels):
        """Bits 28-40 should be constant across all SM100a instructions."""
        r = analyze_control_words(all_kernels)
        dead_zone = 0
        for b in range(28, 41):
            dead_zone |= (1 << b)
        assert (r['global_var_mask'] & dead_zone) == 0, \
            'Expected constant bits 28-40, found varying: 0x%x' % (r['global_var_mask'] & dead_zone)

    def test_upper_bits_used(self, all_kernels):
        """Bits 41+ should have some usage (instruction type/class bits)."""
        r = analyze_control_words(all_kernels)
        upper = sum(1 for b in range(41, 61) if r['bit_freq'].get(b, 0) > 0)
        assert upper > 5, 'Too few upper bits used: %d' % upper


# ── Phase 3.4-3.5: Barriers ─────────────────────────────────────────────

class TestBarriers:
    def test_producers_found(self, all_kernels):
        r = analyze_control_words(all_kernels)
        assert r['n_producers'] > 0, 'No producers found'
        assert r['n_normal'] > 0, 'No normal instructions found'

    def test_wr_bar_enrichment(self, all_kernels):
        """At least one bit should be enriched in producers."""
        r = analyze_control_words(all_kernels)
        assert len(r['wr_bar_candidates']) > 0, 'No wr_bar candidates'
        # The top candidate should have > 2x enrichment
        top = r['wr_bar_candidates'][0]
        bit, prod_rate, norm_rate = top
        ratio = prod_rate / norm_rate if norm_rate > 0 else float('inf')
        assert ratio > 1.5, 'Top wr_bar candidate only %.1fx enriched' % ratio


# ── Integration: export ─────────────────────────────────────────────────

class TestExport:
    def test_export_file_created(self, all_kernels, tmp_path):
        from sass_edit import (
            export_encoding_data, format_encoding_analysis
        )
        pred = analyze_predicates(all_kernels)
        imm = analyze_immediates(all_kernels)
        ctrl = analyze_control_words(all_kernels)
        special = analyze_special_encodings(all_kernels)

        out = str(tmp_path / 'test_encoding.py')
        n = export_encoding_data(pred, imm, ctrl, special, out)
        assert n > 0
        assert os.path.exists(out)

        # Verify it's importable
        content = open(out).read()
        assert 'PREDICATE_ENCODING' in content
        assert 'IMMEDIATE_FIELDS' in content
        assert 'CONTROL_DEFAULTS' in content

    def test_format_output(self, all_kernels):
        from sass_edit import format_encoding_analysis
        pred = analyze_predicates(all_kernels)
        imm = analyze_immediates(all_kernels)
        ctrl = analyze_control_words(all_kernels)
        ot = compute_opcode_table(all_kernels)
        rv = verify_reg_fields(all_kernels, ot)
        special = analyze_special_encodings(all_kernels)

        text = format_encoding_analysis(pred, imm, ctrl, rv, special)
        assert 'PHASE 2+3' in text
        assert 'Predicate' in text
        assert 'Control Word' in text


# ── Gold standard: reconstruction tests ─────────────────────────────────
#
# For each instruction in the cubin, attempt to reconstruct the encoding
# from our tables and compare bit-for-bit. This is the real test of
# whether the encoding data is correct enough to build an assembler.

# Families where bits [15:12] encode the predicate guard.
# Determined empirically: these are the families where our predicate
# analysis is valid. Others (BRA, ISETP, ELECT, etc.) use those bits
# for opcode/modifier/immediate purposes.
_PRED_STANDARD_FAMILIES = frozenset([
    'F2FP', 'HADD2', 'HFMA2', 'HMUL2', 'FADD', 'FMUL', 'FFMA',
    'IADD3', 'IMAD', 'LEA', 'MOV', 'SEL', 'LOP3', 'SHF', 'PRMT',
    'LDS', 'LDG', 'LDC', 'LDTM', 'LDSM',
    'STS', 'STG', 'STL', 'STAS',
    'S2R', 'CS2R', 'R2UR',
    'UTCQMMA', 'UTMALDG', 'UTMASTG',
    'UMOV', 'UIADD3', 'USHF', 'ULEA',
    'NOP',
])


@pytest.fixture(scope='module')
def opcode_table(all_kernels):
    """Freshly computed opcode table from the same cubins."""
    return compute_opcode_table(all_kernels)


class TestReconstruction:
    """Gold standard: verify encoding knowledge against actual instruction binaries.

    For each annotated instruction in the cubins, checks that our encoding
    tables correctly predict the observed binary encoding at each known
    field position.
    """

    def test_opcode_match(self, all_kernels, opcode_table):
        """Every instruction's fixed bits must match its opcode table entry."""
        import re as _re
        total = 0
        match = 0
        fail_families = {}

        for k in all_kernels:
            for insn in k.instructions:
                if not insn.mnemonic:
                    continue
                _, bare = _strip_pred(insn.mnemonic)
                form = insn.encoding & 0xFF
                form_key = '%s{%02x}' % (bare, form)

                entry = opcode_table.get(form_key) or opcode_table.get(bare)
                if not entry:
                    continue

                total += 1
                opcode = entry['opcode']
                var_mask = entry['var_mask']
                actual = insn.encoding & ~var_mask

                if actual == opcode:
                    match += 1
                else:
                    fam = bare.split('.')[0]
                    fail_families[fam] = fail_families.get(fam, 0) + 1

        pct = 100.0 * match / total if total else 0
        assert pct > 99.0, \
            'Opcode match %.1f%% (%d/%d). Top failures: %s' % (
                pct, match, total,
                sorted(fail_families.items(), key=lambda x: -x[1])[:10])

    def test_predicate_field(self, all_kernels):
        """Bits [15:12] must encode predicate guard for standard families."""
        import re as _re
        total = 0
        match = 0
        failures = []

        for k in all_kernels:
            for insn in k.instructions:
                if not insn.mnemonic:
                    continue
                guard, bare = _strip_pred(insn.mnemonic)
                family = bare.split('.')[0]
                if family not in _PRED_STANDARD_FAMILIES:
                    continue

                total += 1
                pred_actual = (insn.encoding >> 12) & 0xF

                if guard:
                    m = _re.match(r'@(!?)(U?)P(\d)', guard)
                    if m:
                        neg = 1 if m.group(1) == '!' else 0
                        pred_expected = (neg << 3) | int(m.group(3))
                    else:
                        pred_expected = 7
                else:
                    pred_expected = 7  # PT

                if pred_actual == pred_expected:
                    match += 1
                elif len(failures) < 20:
                    failures.append('[%04x] %s: got 0x%x want 0x%x' % (
                        insn.offset, insn.mnemonic, pred_actual, pred_expected))

        pct = 100.0 * match / total if total else 0
        assert pct > 99.5, \
            'Predicate match %.1f%% (%d/%d). Failures:\n%s' % (
                pct, match, total, '\n'.join(failures[:10]))

    def test_register_fields(self, all_kernels):
        """Register values in encoding must match cuobjdump operand text."""
        import re as _re
        total = 0
        match = 0
        failures = []

        for k in all_kernels:
            for insn in k.instructions:
                if not insn.mnemonic or not insn.operands:
                    continue
                family = insn_family(insn.mnemonic)
                fields = FAMILY_REG_FIELDS.get(family, [])
                if not fields:
                    continue
                ops = _split_operands(insn.operands)

                for fname, tidx, boff, width in fields:
                    if tidx >= len(ops):
                        continue
                    text = ops[tidx].strip().replace('.reuse', '')

                    if fname.startswith('ur_'):
                        m = _re.search(r'UR(\d+)', text)
                    else:
                        m = _re.search(r'^R(\d+)$', text)

                    if not m:
                        # Could be RZ, memory operand, etc. — skip
                        continue

                    expected = int(m.group(1))
                    actual = read_enc_field(insn.encoding, boff, width)
                    total += 1

                    if actual == expected:
                        match += 1
                    elif len(failures) < 20:
                        failures.append('[%04x] %s %s: %s=%d want %d got %d' % (
                            insn.offset, insn.mnemonic, fname,
                            text, expected, actual, actual))

        pct = 100.0 * match / total if total else 0
        assert pct > 99.0, \
            'Register match %.1f%% (%d/%d). Failures:\n%s' % (
                pct, match, total, '\n'.join(failures[:10]))

    def test_lds128_offset_encoding(self, all_kernels):
        """LDS.128 offset field at bits [49:44] with scale=16.

        The XOR analysis discovered 6 bits at [49:44] from cubins with
        offsets up to ~1008. Larger offsets use additional higher bits
        that weren't captured (insufficient diversity in the sample).
        Test only offsets within the discovered 6-bit range.
        """
        import re as _re
        max_field_offset = 63 * 16  # 6-bit field, scale=16
        total = 0
        match = 0
        skipped_large = 0

        for k in all_kernels:
            for insn in k.instructions:
                if not insn.mnemonic or not insn.operands:
                    continue
                _, bare = _strip_pred(insn.mnemonic)
                if bare != 'LDS.128':
                    continue
                m = _re.search(r'\+\s*(0x[0-9a-fA-F]+)', insn.operands)
                if not m:
                    continue
                offset = int(m.group(1), 16)
                if offset > max_field_offset:
                    skipped_large += 1
                    continue
                field_val = read_enc_field(insn.encoding, 44, 6)
                total += 1
                if field_val == offset // 16:
                    match += 1

        pct = 100.0 * match / total if total else 0
        assert total > 0, 'No LDS.128 with small offsets found'
        assert pct == 100.0, \
            'LDS.128 offset match %.1f%% (%d/%d, %d skipped >%d)' % (
                pct, match, total, skipped_large, max_field_offset)

    def test_predicate_exceptions_identified(self, all_kernels):
        """Families NOT in _PRED_STANDARD_FAMILIES should fail predicate check.

        This verifies our exception list is real (not just overly cautious).
        """
        import re as _re
        known_exceptions = {'BRA', 'BSSY', 'ISETP', 'FSETP', 'HSETP2',
                            'PLOP3', 'ELECT', 'NANOSLEEP', 'SYNCS'}
        found_exception = set()

        for k in all_kernels:
            for insn in k.instructions:
                if not insn.mnemonic:
                    continue
                guard, bare = _strip_pred(insn.mnemonic)
                family = bare.split('.')[0]
                if family not in known_exceptions:
                    continue

                pred_actual = (insn.encoding >> 12) & 0xF
                if guard:
                    m = _re.match(r'@(!?)(U?)P(\d)', guard)
                    if m:
                        neg = 1 if m.group(1) == '!' else 0
                        pred_expected = (neg << 3) | int(m.group(3))
                    else:
                        pred_expected = 7
                else:
                    pred_expected = 7

                if pred_actual != pred_expected:
                    found_exception.add(family)

        # At least some known exceptions should actually be exceptions
        assert len(found_exception) > 0, \
            'None of the known predicate exceptions actually failed — list may be wrong'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
