#!/usr/bin/env python3
"""Tests for lowering_rules.py."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from lowering_rules import (
    SemanticOp, DataType, Pipe, ConcreteInsn, LoweringRule,
    ConversionStrategy, STRATEGIES,
    ALL_LOWERINGS, get_lowerings, compatible_lowerings,
    compute_chunk_cost, analyze_hybrid, sweep_hybrid,
    BF16_SLOW, PIPE_THROUGHPUT, INSN_LATENCY,
)


def test_semantic_ops_complete():
    """All semantic ops have at least one lowering."""
    for op in SemanticOp:
        rules = get_lowerings(op)
        assert len(rules) > 0, f'{op.name} has no lowerings'


def test_add_bias_lowerings():
    """ADD_BIAS has 3 lowerings: HFMA2, FFMA+FP32bias, FFMA+BF16bias."""
    rules = get_lowerings(SemanticOp.ADD_BIAS)
    names = {r.name for r in rules}
    assert 'bias_hfma2' in names
    assert 'bias_ffma_fp32bias' in names
    assert 'bias_ffma_bf16bias' in names
    assert len(rules) == 3


def test_add_res_lowerings():
    """ADD_RES has 2 lowerings: HADD2, FFMA."""
    rules = get_lowerings(SemanticOp.ADD_RES)
    names = {r.name for r in rules}
    assert 'res_hadd2' in names
    assert 'res_ffma' in names
    assert len(rules) == 2


def test_convert_lowerings():
    """CONVERT has 2 lowerings: early and late. Both are 1× F2FP."""
    rules = get_lowerings(SemanticOp.CONVERT_BF16)
    assert len(rules) == 2
    for r in rules:
        assert len(r.instructions) == 1
        assert r.instructions[0].mnemonic == 'F2FP'
        assert r.instructions[0].pipe == Pipe.BF16


def test_fixed_ops_single_lowering():
    """Fixed ops (LDTM, TMEM_WAIT, LOAD_RES, STORE) have exactly 1 lowering."""
    for op in [SemanticOp.LOAD_TMEM, SemanticOp.TMEM_WAIT,
               SemanticOp.LOAD_RES, SemanticOp.STORE_SMEM]:
        rules = get_lowerings(op)
        assert len(rules) == 1, f'{op.name} has {len(rules)} lowerings, expected 1'


def test_load_bias_two_formats():
    """LOAD_BIAS has 2 lowerings: BF16 (4 LDS) and FP32 (8 LDS)."""
    rules = get_lowerings(SemanticOp.LOAD_BIAS)
    assert len(rules) == 2
    bf16 = [r for r in rules if r.name == 'load_bias_bf16'][0]
    fp32 = [r for r in rules if r.name == 'load_bias_fp32'][0]
    assert len(bf16.instructions) == 4
    assert len(fp32.instructions) == 8


def test_hfma2_is_bf16_pipe():
    """HFMA2 lowering is on BF16 pipe."""
    rules = get_lowerings(SemanticOp.ADD_BIAS)
    hfma2 = [r for r in rules if r.name == 'bias_hfma2'][0]
    assert hfma2.instructions[0].pipe == Pipe.BF16


def test_ffma_is_alu_pipe():
    """FFMA lowerings are on ALU pipe."""
    rules = get_lowerings(SemanticOp.ADD_BIAS)
    ffma = [r for r in rules if r.name == 'bias_ffma_fp32bias'][0]
    for insn in ffma.instructions:
        assert insn.pipe == Pipe.ALU


def test_bf16_slow_flag():
    """HFMA2 and HADD2 have the slow BF16 flag."""
    assert 'HFMA2' in BF16_SLOW
    assert 'HADD2' in BF16_SLOW
    assert 'F2FP' not in BF16_SLOW

    insn = ConcreteInsn('HFMA2', Pipe.BF16, 2, 4, [], [])
    assert insn.is_bf16_slow

    insn = ConcreteInsn('F2FP', Pipe.BF16, 2, 4, [], [])
    assert not insn.is_bf16_slow


def test_pipe_cost_computation():
    """LoweringRule __post_init__ computes pipe_cycles correctly."""
    rules = get_lowerings(SemanticOp.ADD_BIAS)
    hfma2 = [r for r in rules if r.name == 'bias_hfma2'][0]
    # HFMA2 is BF16 slow: 3 cyc throughput
    assert hfma2.pipe_cycles[Pipe.BF16] == 3

    ffma = [r for r in rules if r.name == 'bias_ffma_fp32bias'][0]
    # 2× FFMA at 2 cyc each
    assert ffma.pipe_cycles[Pipe.ALU] == 4


def test_compatibility_early_bf16():
    """Early BF16 strategy: bias→HFMA2, res→HADD2, convert→f2fp_early."""
    s = [x for x in STRATEGIES if x.name == 'early_bf16bias'][0]

    bias = compatible_lowerings(SemanticOp.ADD_BIAS, s)
    assert len(bias) == 1
    assert bias[0].name == 'bias_hfma2'

    res = compatible_lowerings(SemanticOp.ADD_RES, s)
    assert len(res) == 1
    assert res[0].name == 'res_hadd2'

    cvt = compatible_lowerings(SemanticOp.CONVERT_BF16, s)
    assert len(cvt) == 1
    assert cvt[0].name == 'f2fp_early'


def test_compatibility_late_fp32():
    """Late FP32 strategy: bias→FFMA_fp32bias, res→FFMA, convert→f2fp_late."""
    s = [x for x in STRATEGIES if x.name == 'late_fp32bias'][0]

    bias = compatible_lowerings(SemanticOp.ADD_BIAS, s)
    assert len(bias) == 1
    assert bias[0].name == 'bias_ffma_fp32bias'

    res = compatible_lowerings(SemanticOp.ADD_RES, s)
    assert len(res) == 1
    assert res[0].name == 'res_ffma'

    cvt = compatible_lowerings(SemanticOp.CONVERT_BF16, s)
    assert len(cvt) == 1
    assert cvt[0].name == 'f2fp_late'


def test_compatibility_late_bf16():
    """Late BF16bias strategy: bias→FFMA_bf16bias, res→FFMA, convert→f2fp_late."""
    s = [x for x in STRATEGIES if x.name == 'late_bf16bias'][0]

    bias = compatible_lowerings(SemanticOp.ADD_BIAS, s)
    assert len(bias) == 1
    assert bias[0].name == 'bias_ffma_bf16bias'

    res = compatible_lowerings(SemanticOp.ADD_RES, s)
    assert len(res) == 1
    assert res[0].name == 'res_ffma'


def test_load_bias_compatibility():
    """Bias format determines which LOAD_BIAS lowering is available."""
    s_bf16 = [x for x in STRATEGIES if x.name == 'early_bf16bias'][0]
    s_fp32 = [x for x in STRATEGIES if x.name == 'late_fp32bias'][0]

    bf16_loads = compatible_lowerings(SemanticOp.LOAD_BIAS, s_bf16)
    assert len(bf16_loads) == 1
    assert bf16_loads[0].name == 'load_bias_bf16'

    fp32_loads = compatible_lowerings(SemanticOp.LOAD_BIAS, s_fp32)
    assert len(fp32_loads) == 1
    assert fp32_loads[0].name == 'load_bias_fp32'


def test_chunk_cost_early_bf16():
    """Early BF16 chunk: 62 insns, BF16=128, ALU=0."""
    s = [x for x in STRATEGIES if x.name == 'early_bf16bias'][0]
    cost = compute_chunk_cost(s)
    assert cost.total_insns == 62
    assert cost.bf16_cycles == 128
    assert cost.alu_cycles == 0


def test_chunk_cost_late_fp32():
    """Late FP32 chunk: 130 insns, ALU dominates."""
    s = [x for x in STRATEGIES if x.name == 'late_fp32bias'][0]
    cost = compute_chunk_cost(s)
    assert cost.total_insns == 130
    assert cost.alu_cycles > cost.bf16_cycles


def test_hybrid_pure_bf16():
    """Hybrid k=0: same as pure BF16."""
    cost = analyze_hybrid(0, verbose=False)
    assert cost.total_insns == 62
    assert cost.bf16_cycles == 128
    assert cost.alu_cycles == 0


def test_hybrid_pure_alu():
    """Hybrid k=16: all ALU path."""
    cost = analyze_hybrid(16, verbose=False)
    assert cost.total_insns == 158
    assert cost.alu_cycles == 256  # 16 × 16 ALU cyc
    assert cost.bf16_cycles == 32   # 16 × 2 F2FP cyc


def test_hybrid_monotonic_bf16():
    """BF16 cycles decrease monotonically as k increases."""
    prev = 999
    for k in range(17):
        cost = analyze_hybrid(k, verbose=False)
        assert cost.bf16_cycles <= prev, \
            f'BF16 cycles increased at k={k}: {cost.bf16_cycles} > {prev}'
        prev = cost.bf16_cycles


def test_hybrid_monotonic_alu():
    """ALU cycles increase monotonically as k increases."""
    prev = -1
    for k in range(17):
        cost = analyze_hybrid(k, verbose=False)
        assert cost.alu_cycles >= prev, \
            f'ALU cycles decreased at k={k}: {cost.alu_cycles} < {prev}'
        prev = cost.alu_cycles


def test_hybrid_crossover():
    """BF16 > ALU at k=5, ALU > BF16 at k=7 (crossover at k=6)."""
    c5 = analyze_hybrid(5, verbose=False)
    assert c5.bf16_cycles > c5.alu_cycles

    c7 = analyze_hybrid(7, verbose=False)
    assert c7.alu_cycles > c7.bf16_cycles


def test_sweep_finds_optimal():
    """Sweep finds k=5 or k=6 as optimal (compute bottleneck minimized)."""
    best_k, best_cyc = sweep_hybrid(verbose=False)
    assert best_k in (5, 6), f'Expected k=5 or 6, got {best_k}'
    assert best_cyc < 100


def test_store_cycles_constant():
    """STORE cycles are 128 regardless of hybrid split."""
    for k in range(17):
        cost = analyze_hybrid(k, verbose=False)
        assert cost.store_cycles == 128


def test_insn_latencies_defined():
    """All instruction mnemonics in lowerings have defined latencies."""
    for op in SemanticOp:
        for rule in get_lowerings(op):
            for insn in rule.instructions:
                assert insn.mnemonic in INSN_LATENCY or insn.latency >= 0, \
                    f'{insn.mnemonic} has no latency defined'


def test_all_insns_have_pipe():
    """All instructions have a valid pipe assignment."""
    for op in SemanticOp:
        for rule in get_lowerings(op):
            for insn in rule.instructions:
                assert isinstance(insn.pipe, Pipe), \
                    f'{insn.mnemonic} in {rule.name} has invalid pipe {insn.pipe}'


def test_strategies_count():
    """We have exactly 3 conversion strategies."""
    assert len(STRATEGIES) == 3
    names = {s.name for s in STRATEGIES}
    assert 'early_bf16bias' in names
    assert 'late_fp32bias' in names
    assert 'late_bf16bias' in names


def test_unpack_insns_are_alu():
    """BF16 bias unpack (LOP3, SHF) in FFMA lowering are ALU pipe."""
    rules = get_lowerings(SemanticOp.ADD_BIAS)
    bf16bias = [r for r in rules if r.name == 'bias_ffma_bf16bias'][0]
    # First 2 insns are unpack (LOP3, SHF)
    assert bf16bias.instructions[0].mnemonic == 'LOP3'
    assert bf16bias.instructions[0].pipe == Pipe.ALU
    assert bf16bias.instructions[1].mnemonic == 'SHF'
    assert bf16bias.instructions[1].pipe == Pipe.ALU


def test_res_ffma_has_unpack():
    """res_ffma lowering includes LOP3+SHF unpack for BF16 residual."""
    rules = get_lowerings(SemanticOp.ADD_RES)
    ffma = [r for r in rules if r.name == 'res_ffma'][0]
    mnemonics = [i.mnemonic for i in ffma.instructions]
    assert 'LOP3' in mnemonics
    assert 'SHF' in mnemonics
    assert mnemonics.count('FFMA') == 2


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            print(f'FAIL {t.__name__}: {e}')
    print(f'\n{passed} passed, {failed} failed out of {len(tests)} tests')
    sys.exit(1 if failed else 0)
