"""
SM100a (Blackwell) instruction database for SASS calibration benchmarks.

Each instruction is described by PTX syntax, operand types, anti-optimization
patterns, and hypothesized resource class. Used by gen_kernels.py to produce
throughput (ILP sweep), latency, and conflict matrix .cu files.

Instruction types:
  'alu'  — standard ALU: acc = OP(acc, srcs), simple generator
  'f2fp' — F2FP with mov.b32 bitcast wrapper
  'sel'  — SEL with predicate setup
  'sts'  — st.shared.v4.b32 (no accumulator)
  'stg'  — st.global.b32 (no accumulator)
  'lds'  — ld.volatile.shared + XOR for throughput, pointer chase for latency
  'ldg'  — ld.global + XOR for throughput, pointer chase for latency
"""

# Hypothesized resource classes (validated by conflict matrix measurements)
FP32 = 'fp32'
BF16 = 'bf16'
INT = 'int'
LSU_S = 'lsu_smem'
LSU_G = 'lsu_gmem'
SPEC = 'special'

INSTRUCTIONS = [
    # ══════════════════════════════════════════════════
    #  FP32 arithmetic
    # ══════════════════════════════════════════════════
    {
        'id': 'FADD', 'sass': 'FADD', 'cat': FP32, 'type': 'alu',
        'ptx': 'add.f32 %{A}, %{A}, %{S0};',
        'acc': ('float', 'f'),
        'srcs': [('float', 'f')],
        'init': 'gin[{i}]+({i}+1)',
        'si': ['gin[threadIdx.x]'],
        'li': 'gin[threadIdx.x]',
        'lsi': ['gin[threadIdx.x+32]'],
        'gin': 'volatile float*',
    },
    {
        'id': 'FMUL', 'sass': 'FMUL', 'cat': FP32, 'type': 'alu',
        'ptx': 'mul.f32 %{A}, %{A}, %{S0};',
        'acc': ('float', 'f'),
        'srcs': [('float', 'f')],
        'init': 'gin[{i}]+({i}+1)',
        'si': ['gin[threadIdx.x]+1.0001f'],
        'li': 'gin[threadIdx.x]+1.0001f',
        'lsi': ['gin[threadIdx.x+32]+1.0001f'],
        'gin': 'volatile float*',
    },
    {
        'id': 'FFMA', 'sass': 'FFMA', 'cat': FP32, 'type': 'alu',
        'ptx': 'fma.rn.f32 %{A}, %{A}, %{S0}, %{S1};',
        'acc': ('float', 'f'),
        'srcs': [('float', 'f'), ('float', 'f')],
        'init': 'gin[{i}]+({i}+1)',
        'si': ['gin[threadIdx.x]+1.0001f', 'gin[threadIdx.x+32]'],
        'li': 'gin[threadIdx.x]',
        'lsi': ['gin[threadIdx.x+32]+1.0001f', 'gin[threadIdx.x+64]'],
        'gin': 'volatile float*',
    },

    # ══════════════════════════════════════════════════
    #  BF16 arithmetic
    # ══════════════════════════════════════════════════
    {
        'id': 'HADD2', 'sass': 'HADD2', 'cat': BF16, 'type': 'alu',
        'ptx': 'add.rn.bf16x2 %{A}, %{A}, %{S0};',
        'acc': ('unsigned', 'r'),
        'srcs': [('unsigned', 'r')],
        'init': 'gin[{i}]|0x3c003c00',
        'si': ['gin[threadIdx.x]|0x00010001'],
        'li': 'gin[threadIdx.x]|0x3c003c00',
        'lsi': ['0x00010001u'],
        'gin': 'volatile unsigned*',
        'notes': 'Calibration K10 had ptxas HFMA2 contamination (8 HFMA2+8 HADD2).',
    },
    {
        'id': 'HFMA2', 'sass': 'HFMA2', 'cat': BF16, 'type': 'alu',
        'ptx': 'fma.rn.bf16x2 %{A}, %{S0}, %{S0}, %{A};',
        'acc': ('unsigned', 'r'),
        'srcs': [('unsigned', 'r')],
        'init': 'gin[{i}]',
        'si': ['gin[threadIdx.x]|0x3c003c00'],
        'li': 'gin[threadIdx.x]|0x3c003c00',
        'lsi': ['gin[threadIdx.x+32]|0x3c003c00'],
        'gin': 'volatile unsigned*',
    },
    {
        'id': 'F2FP', 'sass': 'F2FP', 'cat': BF16, 'type': 'f2fp',
        'notes': 'mov.b32 bitcast + cvt.rn.bf16x2.f32. Sources cycle mod 8.',
    },

    # ══════════════════════════════════════════════════
    #  Integer ALU
    # ══════════════════════════════════════════════════
    {
        'id': 'LOP3', 'sass': 'LOP3', 'cat': INT, 'type': 'alu',
        'ptx': 'lop3.b32 %{A}, %{A}, %{S0}, %{S1}, 0x96;',
        'acc': ('int', 'r'),
        'srcs': [('int', 'r'), ('int', 'r')],
        'init': 'gin[{i}]+({i}+1)',
        'si': ['gin[threadIdx.x]', 'gin[threadIdx.x+32]'],
        'li': 'gin[threadIdx.x]',
        'lsi': ['gin[threadIdx.x+32]', 'gin[threadIdx.x+64]'],
        'gin': 'volatile int*',
    },
    {
        'id': 'SHF', 'sass': 'SHF', 'cat': INT, 'type': 'alu',
        'ptx': 'shf.r.clamp.b32 %{A}, %{A}, %{S0}, %{A};',
        'acc': ('int', 'r'),
        'srcs': [('int', 'r')],
        'init': 'gin[{i}]+({i}+1)',
        'si': ['gin[threadIdx.x+32]&31'],
        'li': 'gin[threadIdx.x]',
        'lsi': ['gin[threadIdx.x+32]&31'],
        'gin': 'volatile int*',
    },
    {
        'id': 'PRMT', 'sass': 'PRMT', 'cat': INT, 'type': 'alu',
        'ptx': 'prmt.b32 %{A}, %{A}, %{S0}, 0x5410;',
        'acc': ('unsigned', 'r'),
        'srcs': [('unsigned', 'r')],
        'init': 'gin[{i}]',
        'si': ['gin[threadIdx.x]'],
        'li': 'gin[threadIdx.x]',
        'lsi': ['gin[threadIdx.x+32]'],
        'gin': 'volatile unsigned*',
    },
    {
        'id': 'VIADD', 'sass': 'VIADD', 'cat': INT, 'type': 'alu',
        'ptx': 'vadd.s32.s32.s32 %{A}, %{A}, %{S0};',
        'acc': ('int', 'r'),
        'srcs': [('int', 'r')],
        'init': 'gin[{i}]+({i}+1)',
        'si': ['gin[threadIdx.x]'],
        'li': 'gin[threadIdx.x]',
        'lsi': ['gin[threadIdx.x+32]'],
        'gin': 'volatile int*',
    },
    {
        'id': 'IADD3', 'sass': 'IADD3', 'cat': INT, 'type': 'alu',
        'ptx': 'add.s32 %{A}, %{A}, %{S0};',
        'acc': ('int', 'r'),
        'srcs': [('int', 'r')],
        'init': 'gin[{i}]+({i}+1)',
        'si': ['gin[threadIdx.x]'],
        'li': 'gin[threadIdx.x]',
        'lsi': ['gin[threadIdx.x+32]'],
        'gin': 'volatile int*',
        'sass_ratio': 0.5,
        'notes': 'ptxas merges pairs (a+=x; a+=x -> IADD3 a,a,x,x). SASS count = ILP/2.',
    },
    {
        'id': 'IMAD', 'sass': 'IMAD', 'cat': INT, 'type': 'alu',
        'ptx': 'mad.lo.s32 %{A}, %{A}, %{S0}, %{S1};',
        'acc': ('int', 'r'),
        'srcs': [('int', 'r'), ('int', 'r')],
        'init': 'gin[{i}]+({i}+1)',
        'si': ['gin[threadIdx.x]', 'gin[threadIdx.x+32]'],
        'li': 'gin[threadIdx.x]',
        'lsi': ['gin[threadIdx.x+32]', 'gin[threadIdx.x+64]'],
        'gin': 'volatile int*',
    },
    {
        'id': 'SEL', 'sass': 'SEL', 'cat': INT, 'type': 'sel',
        'notes': 'selp+add compound. Pred set once before loop.',
    },

    # ══════════════════════════════════════════════════
    #  Special
    # ══════════════════════════════════════════════════
    {
        'id': 'REDUX', 'sass': 'REDUX', 'cat': SPEC, 'type': 'alu',
        'ptx': 'redux.sync.add.s32 %{A}, %{A}, 0xffffffff;',
        'acc': ('int', 'r'),
        'srcs': [],
        'init': 'gin[{i}]+({i}+1)',
        'si': [],
        'li': 'gin[threadIdx.x]',
        'lsi': [],
        'gin': 'volatile int*',
        'notes': 'Warp-level reduction. Proxy for R2UR cost.',
    },

    # ══════════════════════════════════════════════════
    #  Memory (load/store)
    # ══════════════════════════════════════════════════
    {
        'id': 'STS', 'sass': 'STS', 'cat': LSU_S, 'type': 'sts',
        'notes': 'st.shared.v4.b32. Fire-and-forget, no accumulator. 32-cycle throughput.',
    },
    {
        'id': 'STG', 'sass': 'STG', 'cat': LSU_G, 'type': 'stg',
        'notes': 'st.global.b32. Fire-and-forget, no accumulator.',
    },
    {
        'id': 'LDS', 'sass': 'LDS', 'cat': LSU_S, 'type': 'lds',
        'notes': 'Throughput: ld.volatile.shared + XOR. Latency: pointer chase.',
    },
    {
        'id': 'LDG', 'sass': 'LDG', 'cat': LSU_G, 'type': 'ldg',
        'notes': 'Throughput: ld.global + XOR. Latency: pointer chase.',
    },
]

# Index by id for quick lookup
INSN_BY_ID = {insn['id']: insn for insn in INSTRUCTIONS}

# All instruction IDs in definition order
ALL_IDS = [insn['id'] for insn in INSTRUCTIONS]

# Instructions that support throughput benchmarks (all of them)
TPUT_IDS = ALL_IDS

# Instructions that support latency benchmarks
LAT_IDS = [i['id'] for i in INSTRUCTIONS if i['type'] not in ('sts', 'stg')]

# Instructions eligible for conflict matrix
CONFLICT_IDS = ALL_IDS
