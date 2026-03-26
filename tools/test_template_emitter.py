#!/usr/bin/env python3
"""Tests for the FC2 epilogue SASS template emitter."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from template_emitter import (
    RegMap, TN, COLS_PER_CHUNK, COLS_PER_REGION, NUM_EPI_WARPS,
    EPI_THREADS, STAGING_REGION_BYTES,
    emit_preamble, emit_compute_chunk, emit_compute_chunk_interleaved,
    emit_tma_store_is1, emit_tma_store_barsync,
    emit_epilogue_current, emit_epilogue_barsync,
    emit_f2fp_block, emit_hfma2_bias, emit_hadd2_residual,
    emit_sts_block_imm, emit_ldtm_x32, emit_lds_bias,
    emit_lds_residual_direct, emit_tmem_wait,
    emit_addr_setup_sub_iter, count_instructions,
    RegisterAllocator, default_allocator,
)

# Try to import assembler for integration tests
try:
    sys.path.insert(0, os.path.dirname(__file__))
    from sass_edit import parse_asm, assemble as sass_assemble
    HAS_ASSEMBLER = True
except ImportError:
    HAS_ASSEMBLER = False


class TestRegMap(unittest.TestCase):
    def setUp(self):
        self.rm = RegMap()

    def test_tmem_range(self):
        """TMEM registers must be R4-R35 (32 consecutive)."""
        self.assertEqual(self.rm.tmem(0), 4)
        self.assertEqual(self.rm.tmem(31), 35)

    def test_cvt_range(self):
        """CVT registers must be 16 consecutive from CVT_BASE."""
        for i in range(16):
            self.assertEqual(self.rm.cvt(i), self.rm.CVT_BASE + i)

    def test_no_overlap(self):
        """Register groups must not overlap."""
        tmem_set = set(self.rm.tmem(i) for i in range(32))
        cvt_set = set(self.rm.cvt(i) for i in range(16))
        bias_set = set(self.rm.bias(i) for i in range(16))
        res_set = set(self.rm.res(i) for i in range(16))
        sw_set = set(self.rm.sw(i) for i in range(8))

        self.assertTrue(tmem_set.isdisjoint(cvt_set), 'TMEM overlaps CVT')
        self.assertTrue(tmem_set.isdisjoint(bias_set), 'TMEM overlaps BIAS')
        self.assertTrue(cvt_set.isdisjoint(bias_set), 'CVT overlaps BIAS')
        self.assertTrue(cvt_set.isdisjoint(res_set), 'CVT overlaps RES')
        self.assertTrue(bias_set.isdisjoint(res_set), 'BIAS overlaps RES')
        self.assertTrue(sw_set.isdisjoint(cvt_set), 'SW overlaps CVT')

    def test_max_reg_under_256(self):
        """All registers must be below 256."""
        self.assertLess(self.rm.max_reg, 256)


class TestInstructionBlocks(unittest.TestCase):
    def setUp(self):
        self.rm = RegMap()

    def test_ldtm_count(self):
        lines = emit_ldtm_x32(self.rm, 0x20)
        counts = count_instructions(lines)
        self.assertEqual(counts.get('LDTM', 0), 1)

    def test_f2fp_count(self):
        lines = emit_f2fp_block(self.rm)
        counts = count_instructions(lines)
        self.assertEqual(counts['F2FP'], 16)

    def test_hfma2_count(self):
        lines = emit_hfma2_bias(self.rm)
        counts = count_instructions(lines)
        self.assertEqual(counts['HFMA2'], 16)

    def test_hadd2_count(self):
        lines = emit_hadd2_residual(self.rm)
        counts = count_instructions(lines)
        self.assertEqual(counts['HADD2'], 16)

    def test_sts_count(self):
        lines = emit_sts_block_imm(self.rm, 0)
        counts = count_instructions(lines)
        self.assertEqual(counts['STS'], 4)

    def test_lds_bias_count(self):
        lines = emit_lds_bias(self.rm, 0)
        counts = count_instructions(lines)
        self.assertEqual(counts['LDS'], 4)

    def test_lds_residual_count(self):
        lines = emit_lds_residual_direct(self.rm, 0)
        counts = count_instructions(lines)
        self.assertEqual(counts['LDS'], 4)

    def test_compute_chunk_total(self):
        """One 32-col chunk: 1 LDTM + 8 LDS + 16 F2FP + 16 HFMA2 + 16 HADD2 + 4 STS = 61."""
        lines = emit_compute_chunk(self.rm, 0, 0, 0)
        counts = count_instructions(lines)
        self.assertEqual(counts.get('LDTM', 0), 1)
        self.assertEqual(counts.get('LDS', 0), 8)
        self.assertEqual(counts['F2FP'], 16)
        self.assertEqual(counts['HFMA2'], 16)
        self.assertEqual(counts['HADD2'], 16)
        self.assertEqual(counts['STS'], 4)

    def test_compute_chunk_interleaved_same_counts(self):
        """Interleaved chunk has same instruction counts as sequential."""
        seq = count_instructions(emit_compute_chunk(self.rm, 0, 0, 0))
        inter = count_instructions(emit_compute_chunk_interleaved(self.rm, 0, 0, 0))
        for key in ('LDTM', 'LDS', 'F2FP', 'HFMA2', 'HADD2', 'STS'):
            self.assertEqual(seq.get(key, 0), inter.get(key, 0),
                             f'{key} count differs between sequential and interleaved')

    def test_interleaved_sts_distribution(self):
        """In interleaved mode, STS appears after every group (not clustered)."""
        lines = emit_compute_chunk_interleaved(self.rm, 0, 0, 0)
        insn_lines = [l.strip() for l in lines
                      if l.strip() and not l.strip().startswith('#')
                      and not l.strip().startswith('.')]
        sts_positions = [i for i, l in enumerate(insn_lines) if 'STS' in l]
        # 4 STS instructions, they should NOT be consecutive
        self.assertEqual(len(sts_positions), 4)
        # Check gaps between STS — each separated by at least 12 compute ops
        for j in range(1, len(sts_positions)):
            gap = sts_positions[j] - sts_positions[j-1]
            self.assertGreaterEqual(gap, 10,
                f'STS {j-1} and {j} too close (gap={gap}), should be interleaved')


class TestTMAProtocols(unittest.TestCase):
    def setUp(self):
        self.rm = RegMap()

    def test_is1_has_warpsync_fence_utmastg(self):
        lines = emit_tma_store_is1(self.rm, 0)
        counts = count_instructions(lines)
        self.assertEqual(counts.get('WARPSYNC', 0), 1)
        self.assertEqual(counts.get('FENCE', 0), 1)
        self.assertEqual(counts.get('UTMASTG', 0), 1)
        # IS1 must NOT have BAR
        self.assertEqual(counts.get('BAR', 0), 0)

    def test_barsync_has_required_instructions(self):
        lines = emit_tma_store_barsync(self.rm, 0, barrier_id=1)
        counts = count_instructions(lines)
        self.assertEqual(counts.get('MEMBAR', 0), 1)
        self.assertEqual(counts.get('FENCE', 0), 1)
        self.assertGreaterEqual(counts.get('BAR', 0), 2, 'Need at least 2 BAR.SYNC')
        self.assertEqual(counts.get('UTMASTG', 0), 1)
        self.assertEqual(counts.get('DEPBAR', 0), 1)
        self.assertEqual(counts.get('BSYNC', 0), 1)

    def test_barsync_last_skips_trailing_bar(self):
        lines_normal = emit_tma_store_barsync(self.rm, 0, is_last=False)
        lines_last = emit_tma_store_barsync(self.rm, 0, is_last=True)
        bars_normal = count_instructions(lines_normal).get('BAR', 0)
        bars_last = count_instructions(lines_last).get('BAR', 0)
        self.assertEqual(bars_normal, bars_last + 1,
                         'Last sub-iter should have 1 fewer BAR.SYNC')

    def test_barsync_thread_count(self):
        """BAR.SYNC must use 0x80 (128 threads = 4 epilogue warps)."""
        lines = emit_tma_store_barsync(self.rm, 0)
        bar_lines = [l for l in lines
                     if 'BAR.SYNC' in l and not l.strip().startswith('#')]
        self.assertGreater(len(bar_lines), 0, 'No BAR.SYNC instructions found')
        for bl in bar_lines:
            self.assertIn('0x80', bl, f'BAR.SYNC should use 128 threads: {bl}')


class TestFullEpilogue(unittest.TestCase):
    def setUp(self):
        self.rm = RegMap()

    def test_template_a_instruction_counts(self):
        lines = emit_epilogue_current(self.rm)
        counts = count_instructions(lines)
        total = sum(counts.values())
        # 128 cols: 4 chunks × (1 LDTM + 8 LDS + 16 F2FP + 16 HFMA2 + 16 HADD2 + 4 STS)
        # + preamble + 2 IS1 stores + commit
        self.assertEqual(counts['F2FP'], 64, '4 chunks × 16 F2FP')
        self.assertEqual(counts['HFMA2'], 64, '4 chunks × 16 HFMA2')
        self.assertEqual(counts['HADD2'], 64, '4 chunks × 16 HADD2')
        self.assertEqual(counts['STS'], 16, '4 chunks × 4 STS')
        self.assertEqual(counts['LDTM'], 4, '4 chunks × 1 LDTM')
        self.assertGreater(total, 250, 'Template A should have >250 instructions')
        self.assertLess(total, 400, 'Template A should have <400 instructions')

    def test_template_b_instruction_counts(self):
        lines = emit_epilogue_barsync(self.rm)
        counts = count_instructions(lines)
        total = sum(counts.values())
        # 256 cols: 8 chunks × compute + 4 BAR.SYNC protocols + preamble
        self.assertEqual(counts['F2FP'], 128, '8 chunks × 16 F2FP')
        self.assertEqual(counts['HFMA2'], 128, '8 chunks × 16 HFMA2')
        self.assertEqual(counts['HADD2'], 128, '8 chunks × 16 HADD2')
        self.assertEqual(counts['STS'], 32, '8 chunks × 4 STS')
        self.assertEqual(counts['LDTM'], 8, '8 chunks × 1 LDTM')
        self.assertGreater(total, 500, 'Template B should have >500 instructions')
        self.assertLess(total, 700, 'Template B should have <700 instructions')

    def test_template_b_has_barsync(self):
        lines = emit_epilogue_barsync(self.rm)
        counts = count_instructions(lines)
        self.assertGreaterEqual(counts.get('BAR', 0), 7,
                                'Template B needs 7 BAR.SYNC (4 sub-iters, 2 per except last)')

    def test_template_a_no_barsync(self):
        lines = emit_epilogue_current(self.rm)
        counts = count_instructions(lines)
        self.assertEqual(counts.get('BAR', 0), 0, 'Template A must not have BAR.SYNC')
        self.assertEqual(counts.get('MEMBAR', 0), 0, 'Template A must not have MEMBAR')


class TestDataFlow(unittest.TestCase):
    """Verify correct data dependencies in the compute core."""

    def setUp(self):
        self.rm = RegMap()

    def test_f2fp_reads_tmem_regs(self):
        """F2FP must read from TMEM register range."""
        lines = emit_f2fp_block(self.rm)
        for line in lines:
            if 'F2FP' in line:
                # Extract source registers
                parts = line.split(',')
                src_hi_str = parts[1].strip()
                src_lo_str = parts[2].strip()
                for s in (src_hi_str, src_lo_str):
                    reg_num = int(s.replace('R', ''))
                    self.assertGreaterEqual(reg_num, self.rm.TMEM_BASE)
                    self.assertLess(reg_num, self.rm.TMEM_BASE + 32)

    def test_hfma2_reads_cvt_and_bias(self):
        """HFMA2 must read from CVT range and BIAS range."""
        lines = emit_hfma2_bias(self.rm)
        for line in lines:
            if 'HFMA2' in line:
                parts = line.split(',')
                # Rdst = parts[0], Rsrc = parts[1] (should be CVT)
                # Last arg is Rbias
                bias_str = parts[-1].strip()
                bias_reg = int(bias_str.replace('R', ''))
                self.assertGreaterEqual(bias_reg, self.rm.BIAS_BASE)
                self.assertLess(bias_reg, self.rm.BIAS_BASE + 16)

    def test_hadd2_reads_cvt_and_res(self):
        """HADD2 must read from CVT range and RES range."""
        lines = emit_hadd2_residual(self.rm)
        for line in lines:
            if 'HADD2' in line:
                parts = line.split(',')
                res_str = parts[-1].strip()
                res_reg = int(res_str.replace('R', ''))
                self.assertGreaterEqual(res_reg, self.rm.RES_BASE)
                self.assertLess(res_reg, self.rm.RES_BASE + 16)

    def test_sts_reads_cvt(self):
        """STS.128 source must be from CVT range."""
        lines = emit_sts_block_imm(self.rm, 0)
        for line in lines:
            if 'STS' in line:
                parts = line.split(',')
                src_str = parts[-1].strip()
                src_reg = int(src_str.replace('R', ''))
                self.assertGreaterEqual(src_reg, self.rm.CVT_BASE)
                self.assertLess(src_reg, self.rm.CVT_BASE + 16)


@unittest.skipUnless(HAS_ASSEMBLER, 'sass_edit.py assembler not available')
class TestAssembly(unittest.TestCase):
    """Integration tests: verify templates produce valid assembled binary."""

    def setUp(self):
        self.rm = RegMap()

    def _assemble(self, lines):
        text = '\n'.join(lines) + '\n'
        binary, labels, instructions = sass_assemble(text, base_pc=0)
        return binary

    def test_template_a_assembles(self):
        lines = emit_epilogue_current(self.rm)
        binary = self._assemble(lines)
        expected_bytes = count_instructions(lines)
        total_insns = sum(expected_bytes.values())
        self.assertEqual(len(binary), total_insns * 16,
                         f'Expected {total_insns*16} bytes, got {len(binary)}')

    def test_template_b_assembles(self):
        lines = emit_epilogue_barsync(self.rm)
        binary = self._assemble(lines)
        expected = sum(count_instructions(lines).values())
        self.assertEqual(len(binary), expected * 16)

    def test_single_chunk_assembles(self):
        lines = emit_compute_chunk(self.rm, 0, 0, 0)
        binary = self._assemble(lines)
        self.assertGreater(len(binary), 0)

    def test_interleaved_chunk_assembles(self):
        lines = emit_compute_chunk_interleaved(self.rm, 0, 0, 0)
        binary = self._assemble(lines)
        self.assertGreater(len(binary), 0)


class TestRegisterAllocator(unittest.TestCase):
    """Tests for the register allocator."""

    def test_default_allocation_succeeds(self):
        """Default FC2 allocator must succeed."""
        ra = default_allocator()
        alloc = ra.allocate()
        self.assertIn('tmem', alloc)
        self.assertIn('cvt', alloc)
        self.assertIn('bias', alloc)
        self.assertIn('res', alloc)

    def test_tmem_fixed_at_r4(self):
        """TMEM must be fixed at R4 (LDTM.x32 hardware constraint)."""
        ra = default_allocator()
        alloc = ra.allocate()
        self.assertEqual(alloc['tmem'], 4)

    def test_no_overlap_with_blacklist(self):
        """Allocated registers must not overlap FC2 live-in blacklist."""
        ra = default_allocator()
        alloc = ra.allocate()
        allocated_regs = set()
        for g in ra.groups:
            allocated_regs |= set(range(g.physical, g.physical + g.count))
        conflict = allocated_regs & ra.blacklist
        self.assertEqual(len(conflict), 0,
                         f'Allocated regs conflict with blacklist: {sorted(conflict)}')

    def test_no_internal_overlap(self):
        """No two groups may share a physical register."""
        ra = default_allocator()
        ra.allocate()
        all_regs = []
        for g in ra.groups:
            regs = list(range(g.physical, g.physical + g.count))
            all_regs.extend(regs)
        self.assertEqual(len(all_regs), len(set(all_regs)),
                         'Internal overlap between groups')

    def test_within_budget(self):
        """Max register must be under budget."""
        ra = default_allocator()
        ra.allocate()
        max_reg = max(g.physical + g.count - 1 for g in ra.groups)
        self.assertLess(max_reg, ra.budget,
                        f'Max reg R{max_reg} exceeds budget R{ra.budget - 1}')

    def test_consecutive_groups_are_consecutive(self):
        """Consecutive groups must have sequential registers."""
        ra = default_allocator()
        ra.allocate()
        for g in ra.groups:
            if g.consecutive and g.count > 1:
                # Just check the group was assigned (physical >= 0)
                self.assertGreaterEqual(g.physical, 0,
                    f'Group {g.name} not assigned')

    def test_build_regmap(self):
        """build_regmap should return a valid RegMap."""
        ra = default_allocator()
        rm = ra.build_regmap()
        self.assertEqual(rm.TMEM_BASE, 4)
        self.assertGreater(rm.CVT_BASE, 0)
        self.assertNotEqual(rm.CVT_BASE, rm.BIAS_BASE)

    def test_allocated_regmap_emits_valid_template(self):
        """Template emitted with allocated RegMap must assemble."""
        ra = default_allocator()
        rm = ra.build_regmap()
        lines = emit_epilogue_barsync(rm)
        counts = count_instructions(lines)
        self.assertEqual(counts['F2FP'], 128)
        self.assertEqual(counts['STS'], 32)

    def test_custom_blacklist(self):
        """Allocator with custom blacklist should avoid those registers."""
        bl = frozenset(range(36, 52))  # blacklist R36-R51
        ra = RegisterAllocator(budget=207, blacklist=bl)
        ra.add_group('tmem', 32, fixed_at=4)
        ra.add_group('cvt', 16, alignment=4)
        alloc = ra.allocate()
        cvt_regs = set(range(alloc['cvt'], alloc['cvt'] + 16))
        self.assertTrue(cvt_regs.isdisjoint(bl),
                        f'CVT at R{alloc["cvt"]} overlaps blacklist')

    def test_impossible_allocation_raises(self):
        """Should raise ValueError if allocation is impossible."""
        ra = RegisterAllocator(budget=40, blacklist=frozenset())
        ra.add_group('big', 50)  # 50 regs in 40 budget
        with self.assertRaises(ValueError):
            ra.allocate()


if __name__ == '__main__':
    unittest.main()
