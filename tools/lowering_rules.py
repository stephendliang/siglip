#!/usr/bin/env python3
"""
Lowering rules for FC2 epilogue joint optimization.

Defines semantic operation types and all valid concrete instruction sequences
(lowerings) for each. Each lowering specifies: instruction list, pipe assignments,
register pattern, latency chain, and cost model.

This is Phase 1 of the joint optimizer plan (docs/joint_optimizer_plan.md).
Consumed by the CP-SAT joint optimizer to enumerate instruction mix choices.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional, Set


# ---------------------------------------------------------------------------
# Data types flowing between operations
# ---------------------------------------------------------------------------

class DataType(Enum):
    """Types of data flowing between semantic operations."""
    FP32_PAIR = auto()     # 2× FP32 in consecutive regs (e.g., R4,R5)
    BF16_PACKED = auto()   # 1× reg holding 2 BF16 values (PACK_AB format)
    SMEM_ADDR = auto()     # SMEM address (for loads/stores)
    TMEM_ADDR = auto()     # TMEM address descriptor


# ---------------------------------------------------------------------------
# Execution pipes (matches sm100a_encoding.py PIPE_CLASS)
# ---------------------------------------------------------------------------

class Pipe(Enum):
    ALU = 'ALU'           # FFMA, LOP3, SHF, PRMT, IMAD, SEL
    BF16 = 'BF16'         # F2FP, HADD2, HFMA2
    FAST_INT = 'FAST_INT' # IADD3, LEA, MOV (free, overlaps everything)
    LOAD = 'LOAD'         # LDS.128
    STORE = 'STORE'       # STS.128
    TMEM = 'TMEM'         # LDTM
    BARRIER = 'BARRIER'   # DEPBAR, WARPSYNC, BAR, FENCE
    TMA = 'TMA'           # UTMASTG
    CONTROL = 'CONTROL'   # NOP, BRA


# Throughput: minimum cycles between consecutive issues on same pipe.
# From data/calib_tput.txt calibration.
PIPE_THROUGHPUT = {
    Pipe.ALU: 2,        # FFMA/PRMT/SHF all ~2.1 cyc @ILP=2
    Pipe.BF16: 2,       # F2FP=2.0@ILP=4 (HADD2/HFMA2=3.0@ILP=2, modeled per-insn)
    Pipe.FAST_INT: 0,   # free
    Pipe.LOAD: 4,       # LDS.128 = 4 cyc throughput
    Pipe.STORE: 32,     # STS.128 = 32 cyc throughput
    Pipe.TMEM: 20,      # LDTM throughput (~20 cyc)
    Pipe.BARRIER: 0,    # sync, not throughput
    Pipe.TMA: 0,        # async engine
    Pipe.CONTROL: 0,    # no pipe
}

# Latency: cycles from issue to result available.
# From data/calib_tput.txt + data/tma2.txt.
INSN_LATENCY = {
    'FFMA': 4,
    'FADD': 4,
    'F2FP': 4,
    'HFMA2': 4,
    'HADD2': 4,
    'PRMT': 4,
    'SHF': 4,
    'LOP3': 4,
    'MOV': 4,
    'IADD3': 4,
    'LDS.128': 20,
    'STS.128': 0,      # fire-and-forget (data must be ready, but no result)
    'LDTM': 20,        # to scoreboard wait
    'DEPBAR': 0,        # blocks until condition met
    'WARPSYNC': 0,
    'BAR': 0,
    'FENCE': 0,
    'UTMASTG': 0,      # async
}

# BF16 slow families: 3 cyc throughput at ILP<=2 (vs 2 cyc for F2FP)
BF16_SLOW = {'HADD2', 'HFMA2', 'HMUL2'}


# ---------------------------------------------------------------------------
# Semantic operation types
# ---------------------------------------------------------------------------

class SemanticOp(Enum):
    """Semantic operations in the FC2 epilogue dataflow."""
    LOAD_TMEM = auto()      # Load 32 FP32 values from TMEM
    TMEM_WAIT = auto()      # Wait for TMEM load completion
    LOAD_BIAS = auto()      # Load bias from SMEM (BF16 or FP32)
    LOAD_RES = auto()       # Load residual from SMEM (always BF16)
    ADD_BIAS = auto()       # Add bias to accumulator
    ADD_RES = auto()        # Add residual to accumulator
    CONVERT_BF16 = auto()   # Convert FP32 pair → BF16 packed
    STORE_SMEM = auto()     # Store BF16 packed to staging SMEM


# ---------------------------------------------------------------------------
# Concrete instruction in a lowering
# ---------------------------------------------------------------------------

@dataclass
class ConcreteInsn:
    """One concrete SASS instruction in a lowering expansion."""
    mnemonic: str           # e.g., 'FFMA', 'F2FP', 'PRMT', 'LDS.128'
    pipe: Pipe
    throughput: int         # min cycles until next same-pipe issue
    latency: int            # cycles until result available
    reads: List[str]        # symbolic register names read (e.g., ['tmem_lo', 'bias_lo'])
    writes: List[str]       # symbolic register names written (e.g., ['fp32_lo'])
    comment: str = ''       # human-readable note

    @property
    def is_bf16_slow(self) -> bool:
        """True if this instruction has the 3-cyc BF16 throughput penalty."""
        return self.mnemonic in BF16_SLOW


# ---------------------------------------------------------------------------
# Lowering rule: semantic op → concrete instruction sequence
# ---------------------------------------------------------------------------

@dataclass
class LoweringRule:
    """One way to implement a semantic operation as concrete instructions."""
    name: str                           # e.g., 'bias_hfma2', 'bias_ffma_fp32'
    semantic_op: SemanticOp
    instructions: List[ConcreteInsn]
    input_type: DataType                # what this lowering consumes
    output_type: DataType               # what this lowering produces

    # Cost summary (derived from instructions, but precomputed for fast access)
    insn_count: int = 0
    pipe_cycles: Dict[Pipe, int] = field(default_factory=dict)
    regs_acquired: int = 0             # new registers needed
    regs_freed: int = 0                # registers released after completion

    # Compatibility tags
    requires_fp32_context: bool = False  # needs FP32 input (from TMEM or prior F2FP_LATE)
    requires_bf16_context: bool = False  # needs BF16 input (from prior F2FP_EARLY)
    requires_fp32_bias: bool = False     # needs FP32 bias in SMEM
    requires_bf16_bias: bool = False     # needs BF16 bias in SMEM

    def __post_init__(self):
        self.insn_count = len(self.instructions)
        self.pipe_cycles = {}
        for insn in self.instructions:
            tp = insn.throughput
            if insn.is_bf16_slow:
                tp = 3  # HADD2/HFMA2 penalty at ILP<=2
            self.pipe_cycles[insn.pipe] = self.pipe_cycles.get(insn.pipe, 0) + tp


# ---------------------------------------------------------------------------
# Lowering rule definitions
# ---------------------------------------------------------------------------

def _make_lowerings_add_bias() -> List[LoweringRule]:
    """All valid lowerings for ADD_BIAS (per BF16 pair = 2 elements)."""
    rules = []

    # --- Strategy A: HFMA2 (BF16 context) ---
    # Input: BF16_PACKED accumulator + BF16_PACKED bias
    # 1× HFMA2.BF16_V2 Rd, Racc, 1, 1, Rbias  →  Rd = Racc * 1.0 + Rbias
    rules.append(LoweringRule(
        name='bias_hfma2',
        semantic_op=SemanticOp.ADD_BIAS,
        instructions=[
            ConcreteInsn(
                mnemonic='HFMA2', pipe=Pipe.BF16, throughput=2, latency=4,
                reads=['acc_bf16', 'bias_bf16'], writes=['acc_bf16'],
                comment='HFMA2.BF16_V2 Rd, Racc, 1, 1, Rbias',
            ),
        ],
        input_type=DataType.BF16_PACKED,
        output_type=DataType.BF16_PACKED,
        requires_bf16_context=True,
        requires_bf16_bias=True,
        regs_acquired=0,
        regs_freed=0,
    ))

    # --- Strategy B: 2×FFMA (FP32 context, FP32 bias in SMEM) ---
    # Input: FP32_PAIR accumulator + FP32 bias (2 regs from 2× LDS)
    # 2× FFMA Rd, Racc, 1.0, Rbias  →  Rd = Racc + Rbias (scalar FP32)
    rules.append(LoweringRule(
        name='bias_ffma_fp32bias',
        semantic_op=SemanticOp.ADD_BIAS,
        instructions=[
            ConcreteInsn(
                mnemonic='FFMA', pipe=Pipe.ALU, throughput=2, latency=4,
                reads=['acc_fp32_lo', 'bias_fp32_lo'], writes=['acc_fp32_lo'],
                comment='FFMA Rd, Racc_lo, 1.0, Rbias_lo',
            ),
            ConcreteInsn(
                mnemonic='FFMA', pipe=Pipe.ALU, throughput=2, latency=4,
                reads=['acc_fp32_hi', 'bias_fp32_hi'], writes=['acc_fp32_hi'],
                comment='FFMA Rd, Racc_hi, 1.0, Rbias_hi',
            ),
        ],
        input_type=DataType.FP32_PAIR,
        output_type=DataType.FP32_PAIR,
        requires_fp32_context=True,
        requires_fp32_bias=True,
        regs_acquired=0,
        regs_freed=0,
    ))

    # --- Strategy C: 2×FFMA (FP32 context, BF16 bias in SMEM — needs unpack) ---
    # Input: FP32_PAIR accumulator + BF16_PACKED bias
    # Must unpack BF16 bias → 2 FP32 scalars first.
    # BF16 is upper 16 bits of FP32. Unpack:
    #   PRMT R_lo, Rbias, RZ, 0x5410  → zero-extend low BF16 to FP32
    #   SHF.L R_lo, R_lo, 16, RZ      → shift to FP32 position
    # Actually simpler: BF16 in bits [15:0] and [31:16].
    #   LOP3 R_hi, Rbias, 0xFFFF0000, RZ, 0xC0  → mask high BF16 (already FP32-aligned)
    #   SHF.L.U32 R_lo, Rbias, 16, RZ            → shift low BF16 to FP32 position
    # 2 ops (FAST_INT + ALU) per pair for unpack, then 2× FFMA.
    rules.append(LoweringRule(
        name='bias_ffma_bf16bias',
        semantic_op=SemanticOp.ADD_BIAS,
        instructions=[
            # Unpack BF16 bias pair → 2 FP32 scalars
            ConcreteInsn(
                mnemonic='LOP3', pipe=Pipe.ALU, throughput=2, latency=4,
                reads=['bias_bf16'], writes=['bias_fp32_hi'],
                comment='LOP3 Rhi, Rbias, 0xFFFF0000, RZ, 0xC0 (mask high BF16)',
            ),
            ConcreteInsn(
                mnemonic='SHF', pipe=Pipe.ALU, throughput=2, latency=4,
                reads=['bias_bf16'], writes=['bias_fp32_lo'],
                comment='SHF.L.U32 Rlo, Rbias, 16, RZ (shift low BF16 → FP32)',
            ),
            # 2× FFMA add
            ConcreteInsn(
                mnemonic='FFMA', pipe=Pipe.ALU, throughput=2, latency=4,
                reads=['acc_fp32_lo', 'bias_fp32_lo'], writes=['acc_fp32_lo'],
                comment='FFMA Rd, Racc_lo, 1.0, Rbias_lo',
            ),
            ConcreteInsn(
                mnemonic='FFMA', pipe=Pipe.ALU, throughput=2, latency=4,
                reads=['acc_fp32_hi', 'bias_fp32_hi'], writes=['acc_fp32_hi'],
                comment='FFMA Rd, Racc_hi, 1.0, Rbias_hi',
            ),
        ],
        input_type=DataType.FP32_PAIR,
        output_type=DataType.FP32_PAIR,
        requires_fp32_context=True,
        requires_bf16_bias=True,
        regs_acquired=2,   # 2 temporary FP32 scalars for unpacked bias
        regs_freed=2,       # freed after FFMA consumes them
    ))

    return rules


def _make_lowerings_add_res() -> List[LoweringRule]:
    """All valid lowerings for ADD_RES (per BF16 pair = 2 elements).

    Residual is ALWAYS BF16 packed (from TMA load). FP32 path always needs unpack.
    """
    rules = []

    # --- Strategy A: HADD2 (BF16 context) ---
    # Input: BF16_PACKED accumulator + BF16_PACKED residual
    # 1× HADD2.BF16_V2 Rd, Racc, Rres
    rules.append(LoweringRule(
        name='res_hadd2',
        semantic_op=SemanticOp.ADD_RES,
        instructions=[
            ConcreteInsn(
                mnemonic='HADD2', pipe=Pipe.BF16, throughput=2, latency=4,
                reads=['acc_bf16', 'res_bf16'], writes=['acc_bf16'],
                comment='HADD2.BF16_V2 Rd, Racc, Rres',
            ),
        ],
        input_type=DataType.BF16_PACKED,
        output_type=DataType.BF16_PACKED,
        requires_bf16_context=True,
        regs_acquired=0,
        regs_freed=0,
    ))

    # --- Strategy B: 2×FFMA (FP32 context — must unpack BF16 residual) ---
    # Input: FP32_PAIR accumulator + BF16_PACKED residual
    # Unpack residual BF16 → 2 FP32, then 2× FFMA
    rules.append(LoweringRule(
        name='res_ffma',
        semantic_op=SemanticOp.ADD_RES,
        instructions=[
            # Unpack BF16 residual → 2 FP32 scalars
            ConcreteInsn(
                mnemonic='LOP3', pipe=Pipe.ALU, throughput=2, latency=4,
                reads=['res_bf16'], writes=['res_fp32_hi'],
                comment='LOP3 Rhi, Rres, 0xFFFF0000, RZ, 0xC0 (mask high BF16)',
            ),
            ConcreteInsn(
                mnemonic='SHF', pipe=Pipe.ALU, throughput=2, latency=4,
                reads=['res_bf16'], writes=['res_fp32_lo'],
                comment='SHF.L.U32 Rlo, Rres, 16, RZ (shift low BF16 → FP32)',
            ),
            # 2× FFMA add
            ConcreteInsn(
                mnemonic='FFMA', pipe=Pipe.ALU, throughput=2, latency=4,
                reads=['acc_fp32_lo', 'res_fp32_lo'], writes=['acc_fp32_lo'],
                comment='FFMA Rd, Racc_lo, 1.0, Rres_lo',
            ),
            ConcreteInsn(
                mnemonic='FFMA', pipe=Pipe.ALU, throughput=2, latency=4,
                reads=['acc_fp32_hi', 'res_fp32_hi'], writes=['acc_fp32_hi'],
                comment='FFMA Rd, Racc_hi, 1.0, Rres_hi',
            ),
        ],
        input_type=DataType.FP32_PAIR,
        output_type=DataType.FP32_PAIR,
        requires_fp32_context=True,
        regs_acquired=2,   # 2 temporary FP32 scalars for unpacked residual
        regs_freed=2,
    ))

    return rules


def _make_lowerings_convert() -> List[LoweringRule]:
    """All valid lowerings for CONVERT_BF16 (F2FP).

    F2FP is LOCKED to BF16 pipe. No alternative exists.
    The only choice is WHEN it fires (early vs late), which changes what
    surrounds it — modeled by compatibility tags, not by instruction alternatives.
    """
    rules = []

    # --- F2FP EARLY: immediately after TMEM load, before bias/res ---
    # Converts FP32 pair → BF16 packed. Everything downstream is BF16.
    rules.append(LoweringRule(
        name='f2fp_early',
        semantic_op=SemanticOp.CONVERT_BF16,
        instructions=[
            ConcreteInsn(
                mnemonic='F2FP', pipe=Pipe.BF16, throughput=2, latency=4,
                reads=['tmem_hi', 'tmem_lo'], writes=['acc_bf16'],
                comment='F2FP.BF16.F32.PACK_AB Rd, Rtmem_hi, Rtmem_lo (early)',
            ),
        ],
        input_type=DataType.FP32_PAIR,
        output_type=DataType.BF16_PACKED,
        requires_fp32_context=True,  # input is FP32 from TMEM
        regs_acquired=0,   # reuses 1 reg from the 2 FP32 inputs (net saves 1)
        regs_freed=1,       # frees 1 reg (2 FP32 → 1 BF16)
    ))

    # --- F2FP LATE: after bias+res in FP32, just before STS ---
    # Same instruction, different position in dataflow.
    rules.append(LoweringRule(
        name='f2fp_late',
        semantic_op=SemanticOp.CONVERT_BF16,
        instructions=[
            ConcreteInsn(
                mnemonic='F2FP', pipe=Pipe.BF16, throughput=2, latency=4,
                reads=['acc_fp32_hi', 'acc_fp32_lo'], writes=['acc_bf16'],
                comment='F2FP.BF16.F32.PACK_AB Rd, Racc_hi, Racc_lo (late)',
            ),
        ],
        input_type=DataType.FP32_PAIR,
        output_type=DataType.BF16_PACKED,
        requires_fp32_context=True,
        regs_acquired=0,
        regs_freed=1,
    ))

    return rules


def _make_lowerings_load_bias() -> List[LoweringRule]:
    """All valid lowerings for LOAD_BIAS.

    Per 32-col chunk: 16 BF16 pairs = 32 BF16 values = 64 bytes.
    BF16 format: 4× LDS.128 (16 bytes each) = 4 loads.
    FP32 format: 8× LDS.128 (16 bytes each) = 8 loads, 2× bandwidth.
    """
    rules = []

    # --- BF16 bias: 4× LDS.128 ---
    insns = []
    for g in range(4):
        insns.append(ConcreteInsn(
            mnemonic='LDS.128', pipe=Pipe.LOAD, throughput=4, latency=20,
            reads=['bias_saddr'], writes=[f'bias_bf16_{g*4}_{g*4+3}'],
            comment=f'LDS.128 R[bias+{g*4}], [bias_saddr+{g*16:#x}]',
        ))
    rules.append(LoweringRule(
        name='load_bias_bf16',
        semantic_op=SemanticOp.LOAD_BIAS,
        instructions=insns,
        input_type=DataType.SMEM_ADDR,
        output_type=DataType.BF16_PACKED,
        requires_bf16_bias=True,
        regs_acquired=16,
        regs_freed=0,
    ))

    # --- FP32 bias: 8× LDS.128 ---
    insns = []
    for g in range(8):
        insns.append(ConcreteInsn(
            mnemonic='LDS.128', pipe=Pipe.LOAD, throughput=4, latency=20,
            reads=['bias_saddr'], writes=[f'bias_fp32_{g*4}_{g*4+3}'],
            comment=f'LDS.128 R[bias+{g*4}], [bias_saddr+{g*16:#x}]',
        ))
    rules.append(LoweringRule(
        name='load_bias_fp32',
        semantic_op=SemanticOp.LOAD_BIAS,
        instructions=insns,
        input_type=DataType.SMEM_ADDR,
        output_type=DataType.FP32_PAIR,
        requires_fp32_bias=True,
        regs_acquired=32,
        regs_freed=0,
    ))

    return rules


def _make_lowerings_load_res() -> List[LoweringRule]:
    """Lowerings for LOAD_RES. Always BF16 from TMA. 4× LDS.128 per chunk."""
    insns = []
    for g in range(4):
        insns.append(ConcreteInsn(
            mnemonic='LDS.128', pipe=Pipe.LOAD, throughput=4, latency=20,
            reads=['res_saddr'], writes=[f'res_bf16_{g*4}_{g*4+3}'],
            comment=f'LDS.128 R[res+{g*4}], [res_saddr+{g*16:#x}]',
        ))
    return [LoweringRule(
        name='load_res_bf16',
        semantic_op=SemanticOp.LOAD_RES,
        instructions=insns,
        input_type=DataType.SMEM_ADDR,
        output_type=DataType.BF16_PACKED,
        regs_acquired=16,
        regs_freed=0,
    )]


def _make_lowerings_load_tmem() -> List[LoweringRule]:
    """LOAD_TMEM: 1× LDTM.x32. No alternatives."""
    return [LoweringRule(
        name='load_tmem',
        semantic_op=SemanticOp.LOAD_TMEM,
        instructions=[
            ConcreteInsn(
                mnemonic='LDTM', pipe=Pipe.TMEM, throughput=20, latency=20,
                reads=['tmem_addr'], writes=[f'tmem_{i}' for i in range(32)],
                comment='LDTM.x32 R[tmem_base], tmem[col_offset]',
            ),
        ],
        input_type=DataType.TMEM_ADDR,
        output_type=DataType.FP32_PAIR,
        regs_acquired=32,
        regs_freed=0,
    )]


def _make_lowerings_tmem_wait() -> List[LoweringRule]:
    """TMEM_WAIT: 1× DEPBAR. No alternatives."""
    return [LoweringRule(
        name='tmem_wait',
        semantic_op=SemanticOp.TMEM_WAIT,
        instructions=[
            ConcreteInsn(
                mnemonic='DEPBAR', pipe=Pipe.BARRIER, throughput=0, latency=0,
                reads=[], writes=[],
                comment='DEPBAR.LE SB0, 0x0 (wait for TMEM load)',
            ),
        ],
        input_type=DataType.FP32_PAIR,
        output_type=DataType.FP32_PAIR,
        regs_acquired=0,
        regs_freed=0,
    )]


def _make_lowerings_store() -> List[LoweringRule]:
    """STORE_SMEM: 4× STS.128 per chunk. No alternatives."""
    insns = []
    for g in range(4):
        insns.append(ConcreteInsn(
            mnemonic='STS.128', pipe=Pipe.STORE, throughput=32, latency=0,
            reads=[f'acc_bf16_{g*4}_{g*4+3}', f'sts_addr_{g}'],
            writes=[],
            comment=f'STS.128 [sts_addr_{g}], R[acc+{g*4}]',
        ))
    return [LoweringRule(
        name='store_smem',
        semantic_op=SemanticOp.STORE_SMEM,
        instructions=insns,
        input_type=DataType.BF16_PACKED,
        output_type=DataType.SMEM_ADDR,
        regs_acquired=0,
        regs_freed=16,  # BF16 accumulator regs freed after STS
    )]


# ---------------------------------------------------------------------------
# Lowering registry
# ---------------------------------------------------------------------------

ALL_LOWERINGS: Dict[SemanticOp, List[LoweringRule]] = {}

def _build_registry():
    builders = [
        _make_lowerings_add_bias,
        _make_lowerings_add_res,
        _make_lowerings_convert,
        _make_lowerings_load_bias,
        _make_lowerings_load_res,
        _make_lowerings_load_tmem,
        _make_lowerings_tmem_wait,
        _make_lowerings_store,
    ]
    for builder in builders:
        for rule in builder():
            ALL_LOWERINGS.setdefault(rule.semantic_op, []).append(rule)

_build_registry()


def get_lowerings(op: SemanticOp) -> List[LoweringRule]:
    """Get all valid lowering rules for a semantic operation."""
    return ALL_LOWERINGS.get(op, [])


# ---------------------------------------------------------------------------
# Compatibility checking
# ---------------------------------------------------------------------------

@dataclass
class ConversionStrategy:
    """F2FP placement strategy — determines which lowerings are compatible."""
    name: str
    f2fp_position: str  # 'early' or 'late'
    bias_format: str    # 'bf16' or 'fp32'

    @property
    def is_early(self) -> bool:
        return self.f2fp_position == 'early'

    @property
    def is_late(self) -> bool:
        return self.f2fp_position == 'late'


# All valid conversion strategies
STRATEGIES = [
    ConversionStrategy('early_bf16bias', 'early', 'bf16'),
    ConversionStrategy('late_fp32bias', 'late', 'fp32'),
    ConversionStrategy('late_bf16bias', 'late', 'bf16'),
    # early + fp32bias is wasteful: F2FP converts to BF16, then load FP32 bias
    # that can't be used with HFMA2 anyway. Not illegal but strictly worse.
]


def compatible_lowerings(op: SemanticOp,
                         strategy: ConversionStrategy) -> List[LoweringRule]:
    """Filter lowerings for a semantic op to those compatible with a strategy.

    F2FP_EARLY → accumulator is BF16 after convert → bias/res ops need BF16 context.
    F2FP_LATE  → accumulator stays FP32 until end → bias/res ops need FP32 context.
    """
    all_rules = get_lowerings(op)
    result = []

    for rule in all_rules:
        # F2FP placement compatibility
        if op == SemanticOp.ADD_BIAS:
            if strategy.is_early and not rule.requires_bf16_context:
                continue  # early → data is BF16, need BF16-context lowering
            if strategy.is_late and not rule.requires_fp32_context:
                continue  # late → data is FP32, need FP32-context lowering

        if op == SemanticOp.ADD_RES:
            if strategy.is_early and not rule.requires_bf16_context:
                continue
            if strategy.is_late and not rule.requires_fp32_context:
                continue

        # Bias format compatibility
        if op == SemanticOp.ADD_BIAS:
            if strategy.bias_format == 'bf16' and rule.requires_fp32_bias:
                continue
            if strategy.bias_format == 'fp32' and rule.requires_bf16_bias:
                continue

        if op == SemanticOp.LOAD_BIAS:
            if strategy.bias_format == 'bf16' and rule.requires_fp32_bias:
                continue
            if strategy.bias_format == 'fp32' and rule.requires_bf16_bias:
                continue

        # F2FP lowering selection
        if op == SemanticOp.CONVERT_BF16:
            if strategy.is_early and rule.name != 'f2fp_early':
                continue
            if strategy.is_late and rule.name != 'f2fp_late':
                continue

        result.append(rule)

    return result


# ---------------------------------------------------------------------------
# Cost model: per-chunk aggregate
# ---------------------------------------------------------------------------

@dataclass
class ChunkCost:
    """Aggregate cost for one 32-col chunk under a given strategy."""
    strategy: ConversionStrategy
    total_insns: int
    pipe_cycles: Dict[Pipe, int]
    regs_peak: int
    lowering_names: Dict[SemanticOp, str]   # which lowering was chosen per op

    @property
    def bf16_cycles(self) -> int:
        return self.pipe_cycles.get(Pipe.BF16, 0)

    @property
    def alu_cycles(self) -> int:
        return self.pipe_cycles.get(Pipe.ALU, 0)

    @property
    def store_cycles(self) -> int:
        return self.pipe_cycles.get(Pipe.STORE, 0)

    @property
    def load_cycles(self) -> int:
        return self.pipe_cycles.get(Pipe.LOAD, 0)

    @property
    def compute_bottleneck(self) -> Tuple[Pipe, int]:
        """Which compute pipe (ALU or BF16) is the bottleneck and its cost."""
        bf16 = self.bf16_cycles
        alu = self.alu_cycles
        if bf16 >= alu:
            return (Pipe.BF16, bf16)
        return (Pipe.ALU, alu)

    @property
    def balance_ratio(self) -> float:
        """BF16:ALU ratio. 1.0 = perfectly balanced. >1 = BF16 heavy."""
        alu = max(self.alu_cycles, 1)
        return self.bf16_cycles / alu


def compute_chunk_cost(strategy: ConversionStrategy,
                       bias_lowering: str = None,
                       res_lowering: str = None) -> ChunkCost:
    """Compute cost for one 32-col chunk (16 BF16 pairs) under a strategy.

    If bias_lowering/res_lowering not specified, uses the first compatible
    lowering (which is the natural one for the strategy).
    """
    pipe_cyc: Dict[Pipe, int] = {}
    total = 0
    regs = 0
    names: Dict[SemanticOp, str] = {}

    for op in SemanticOp:
        compat = compatible_lowerings(op, strategy)
        if not compat:
            continue

        # Select lowering
        chosen = compat[0]  # default: first compatible
        if op == SemanticOp.ADD_BIAS and bias_lowering:
            for r in compat:
                if r.name == bias_lowering:
                    chosen = r
                    break
        if op == SemanticOp.ADD_RES and res_lowering:
            for r in compat:
                if r.name == res_lowering:
                    chosen = r
                    break

        names[op] = chosen.name

        # Multiply per-pair ops by 16 pairs (except loads/stores which are per-chunk)
        multiplier = 1
        if op in (SemanticOp.ADD_BIAS, SemanticOp.ADD_RES, SemanticOp.CONVERT_BF16):
            multiplier = 16  # one per BF16 pair

        for insn in chosen.instructions:
            tp = insn.throughput
            if insn.is_bf16_slow:
                tp = 3
            pipe_cyc[insn.pipe] = pipe_cyc.get(insn.pipe, 0) + tp * multiplier
            total += multiplier

        regs = max(regs, regs + chosen.regs_acquired)

    return ChunkCost(
        strategy=strategy,
        total_insns=total,
        pipe_cycles=pipe_cyc,
        regs_peak=regs,
        lowering_names=names,
    )


# ---------------------------------------------------------------------------
# Analysis / reporting
# ---------------------------------------------------------------------------

def analyze_strategy(strategy: ConversionStrategy, verbose: bool = True) -> ChunkCost:
    """Analyze a conversion strategy, reporting pipe utilization and balance."""
    # Natural lowerings for this strategy
    cost = compute_chunk_cost(strategy)

    if verbose:
        print(f'\n{"="*70}')
        print(f'Strategy: {strategy.name}')
        print(f'  F2FP: {strategy.f2fp_position}, Bias: {strategy.bias_format}')
        print(f'{"="*70}')

        print(f'\nLowerings chosen:')
        for op, name in sorted(cost.lowering_names.items(), key=lambda x: x[0].name):
            lowers = get_lowerings(op)
            chosen = [l for l in lowers if l.name == name][0]
            print(f'  {op.name:20s} → {name:30s} ({chosen.insn_count} insn/pair)')

        print(f'\nPer-chunk cost (32 cols = 16 BF16 pairs):')
        print(f'  Total instructions: {cost.total_insns}')
        print(f'\n  Pipe breakdown (cycles):')
        for pipe in sorted(cost.pipe_cycles.keys(), key=lambda p: -cost.pipe_cycles[p]):
            cyc = cost.pipe_cycles[pipe]
            bar = '#' * min(cyc // 2, 40)
            print(f'    {pipe.value:10s}: {cyc:4d} cyc  {bar}')

        bn_pipe, bn_cyc = cost.compute_bottleneck
        print(f'\n  Compute bottleneck: {bn_pipe.value} ({bn_cyc} cyc)')
        print(f'  BF16:ALU ratio: {cost.balance_ratio:.1f}:1')

        # Per-256-col extrapolation (8 chunks)
        print(f'\n  Full tile (256 cols = 8 chunks):')
        print(f'    Instructions: {cost.total_insns * 8}')
        print(f'    Compute bottleneck: {bn_cyc * 8} cyc')
        print(f'    Store pipe: {cost.store_cycles * 8} cyc')

    return cost


def compare_all_strategies(verbose: bool = True) -> List[ChunkCost]:
    """Compare all conversion strategies side by side."""
    costs = []
    for s in STRATEGIES:
        cost = analyze_strategy(s, verbose=verbose)
        costs.append(cost)

    if verbose:
        print(f'\n{"="*70}')
        print('COMPARISON SUMMARY')
        print(f'{"="*70}')
        print(f'\n{"Strategy":30s} {"Insns":>6s} {"BF16":>6s} {"ALU":>6s} '
              f'{"LOAD":>6s} {"STORE":>6s} {"Ratio":>6s} {"Bottleneck":>12s}')
        print('-' * 100)
        for cost in costs:
            bn_pipe, bn_cyc = cost.compute_bottleneck
            print(f'{cost.strategy.name:30s} '
                  f'{cost.total_insns:6d} '
                  f'{cost.bf16_cycles:6d} '
                  f'{cost.alu_cycles:6d} '
                  f'{cost.load_cycles:6d} '
                  f'{cost.store_cycles:6d} '
                  f'{cost.balance_ratio:6.1f} '
                  f'{bn_pipe.value:>12s}')

        # Find best compute bottleneck
        best = min(costs, key=lambda c: c.compute_bottleneck[1])
        print(f'\nBest compute bottleneck: {best.strategy.name} '
              f'({best.compute_bottleneck[1]} cyc, '
              f'{best.total_insns} insns/chunk)')

    return costs


def analyze_hybrid(k_alu: int, verbose: bool = True) -> ChunkCost:
    """Analyze a hybrid per-pair path split for one 32-col chunk (16 pairs).

    k_alu: how many of 16 pairs take the ALU path (rest take BF16 path).

    Two complete per-pair dataflow paths:
      BF16 path (current): F2FP_early → HFMA2(bias) → HADD2(res)
        = 1 F2FP(2 cyc) + 1 HFMA2(3 cyc) + 1 HADD2(3 cyc) = 3 insns, 8 BF16 cyc
      ALU path (CUTLASS-style): unpack_bias(LOP3+SHF) → 2×FFMA_bias →
        unpack_res(LOP3+SHF) → 2×FFMA_res → F2FP_late
        = 2 LOP3(4 cyc) + 2 SHF(4 cyc) + 4 FFMA(8 cyc) + 1 F2FP(2 cyc)
        = 9 insns, 16 ALU + 2 BF16 cyc

    Both paths read the same BF16 bias and BF16 residual from SMEM.
    Both paths produce 1 BF16 packed result for STS.

    Fixed per-chunk ops (independent of k):
      1× LDTM (TMEM, 20 cyc)
      1× DEPBAR (BARRIER, 0 cyc)
      4× LDS.128 bias (LOAD, 16 cyc)
      4× LDS.128 residual (LOAD, 16 cyc)
      4× STS.128 (STORE, 128 cyc)
    """
    k_bf16 = 16 - k_alu

    pipe_cyc: Dict[Pipe, int] = {}
    total = 0

    # Fixed ops
    pipe_cyc[Pipe.TMEM] = 20    # 1 LDTM
    pipe_cyc[Pipe.BARRIER] = 0  # 1 DEPBAR
    pipe_cyc[Pipe.LOAD] = 32    # 8 LDS.128 (4 bias + 4 res)
    pipe_cyc[Pipe.STORE] = 128  # 4 STS.128
    total += 1 + 1 + 8 + 4      # 14 fixed insns

    # BF16 path pairs: 3 insns each (F2FP + HFMA2 + HADD2)
    bf16_from_bf16_path = k_bf16 * (2 + 3 + 3)  # F2FP=2, HFMA2=3, HADD2=3
    total += k_bf16 * 3

    # ALU path pairs: 9 insns each (2 LOP3 + 2 SHF + 4 FFMA + 1 F2FP)
    alu_from_alu_path = k_alu * (2 * 2 + 2 * 2 + 4 * 2)  # LOP3=2, SHF=2, FFMA=2
    bf16_from_alu_path = k_alu * 2                          # F2FP=2
    total += k_alu * 9

    pipe_cyc[Pipe.BF16] = bf16_from_bf16_path + bf16_from_alu_path
    pipe_cyc[Pipe.ALU] = alu_from_alu_path

    cost = ChunkCost(
        strategy=ConversionStrategy('hybrid', 'mixed', 'bf16'),
        total_insns=total,
        pipe_cycles=pipe_cyc,
        regs_peak=0,
        lowering_names={},
    )

    if verbose:
        bn_pipe, bn_cyc = cost.compute_bottleneck
        print(f'  k={k_alu:2d} ALU, {k_bf16:2d} BF16 | '
              f'insns={total:3d} | '
              f'BF16={cost.bf16_cycles:3d} ALU={cost.alu_cycles:3d} '
              f'STORE={cost.store_cycles:3d} LOAD={cost.load_cycles:2d} | '
              f'ratio={cost.balance_ratio:5.1f} | '
              f'compute_bn={bn_pipe.value} {bn_cyc}')

    return cost


def sweep_hybrid(verbose: bool = True) -> Tuple[int, int]:
    """Find optimal hybrid split.

    Returns (best_k_alu, best_bottleneck_cyc).
    """
    if verbose:
        print('Hybrid sweep: per 32-col chunk (16 BF16 pairs)')
        print('BF16 path: F2FP(2) + HFMA2(3) + HADD2(3) = 8 BF16 cyc/pair')
        print('ALU path:  LOP3(2) + SHF(2) + FFMA×4(8) + F2FP(2) = 16 ALU + 2 BF16 cyc/pair')
        print()

    best_k = 0
    best_cyc = 999999
    for k in range(17):
        cost = analyze_hybrid(k, verbose=verbose)
        bn_pipe, bn_cyc = cost.compute_bottleneck
        if bn_cyc < best_cyc:
            best_k = k
            best_cyc = bn_cyc

    if verbose:
        print(f'\nOptimal: k={best_k} pairs on ALU path, '
              f'compute bottleneck={best_cyc} cyc')
        # Compare to pure strategies
        pure_bf16 = analyze_hybrid(0, verbose=False)
        pure_alu = analyze_hybrid(16, verbose=False)
        print(f'  vs pure BF16: {pure_bf16.compute_bottleneck[1]} cyc '
              f'({pure_bf16.total_insns} insns)')
        print(f'  vs pure ALU:  {pure_alu.compute_bottleneck[1]} cyc '
              f'({pure_alu.total_insns} insns)')
        best_cost = analyze_hybrid(best_k, verbose=False)
        print(f'  hybrid:       {best_cyc} cyc ({best_cost.total_insns} insns)')
        reduction = (1 - best_cyc / pure_bf16.compute_bottleneck[1]) * 100
        print(f'  reduction:    {reduction:.0f}% vs pure BF16')

        # Caveat: STS throughput
        print(f'\n  CAVEAT: STORE pipe = {pure_bf16.store_cycles} cyc regardless of k.')
        print(f'  At k={best_k}, compute ({best_cyc}) < STORE ({pure_bf16.store_cycles}).')
        print(f'  STORE becomes the bottleneck. STS interleaving (hiding compute')
        print(f'  in 32-cyc STS shadow) is needed to make the balance pay off.')
        print(f'  → CP-SAT joint scheduling required for accurate makespan.')

    return best_k, best_cyc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='FC2 epilogue lowering analysis')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all strategies side by side')
    parser.add_argument('--strategy', type=str, default=None,
                        help='Analyze specific strategy (early_bf16bias, late_fp32bias, late_bf16bias)')
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep hybrid splits for all strategies')
    args = parser.parse_args()

    if args.sweep:
        sweep_hybrid(verbose=True)
    elif args.strategy:
        s = [x for x in STRATEGIES if x.name == args.strategy]
        if not s:
            print(f'Unknown strategy: {args.strategy}')
            print(f'Available: {", ".join(x.name for x in STRATEGIES)}')
            return
        analyze_strategy(s[0])
    elif args.compare:
        compare_all_strategies()
    else:
        compare_all_strategies()


if __name__ == '__main__':
    main()
