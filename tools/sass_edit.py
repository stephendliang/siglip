#!/usr/bin/env python3
"""
SM100a SASS binary editor — parse, view, reorder, and patch cubin instructions.

Works on standalone .cubin files (ELF format). Cross-references with
cuobjdump --dump-sass text output for mnemonic annotation.

Usage:
    sass_edit.py info CUBIN
    sass_edit.py dump CUBIN [--kernel NAME] [--start ADDR] [--end ADDR] [--sass FILE]
    sass_edit.py swap CUBIN ADDR_A ADDR_B -o OUTPUT [--sass FILE] [--force]
    sass_edit.py reorder CUBIN START END ADDR,ADDR,... -o OUTPUT [--sass FILE] [--force]
    sass_edit.py patch CUBIN ADDR --stall N [--yield N] [--raw_lo HEX] -o OUTPUT
    sass_edit.py script CUBIN SCRIPT_FILE -o OUTPUT
    sass_edit.py verify CUBIN --sass FILE [--kernel NAME]
    sass_edit.py diff CUBIN_A CUBIN_B [--kernel NAME]
    sass_edit.py deps CUBIN --sass FILE --start ADDR --end ADDR [--reorder ADDR,ADDR,...]
    sass_edit.py probe-encoding CUBIN --sass FILE [--kernel NAME]
    sass_edit.py patch-reg CUBIN ADDR FIELD REG -o OUTPUT [--sass FILE]
    sass_edit.py copy-insn CUBIN SRC DST -o OUTPUT [--sass FILE]
    sass_edit.py pipeline CUBIN --sass FILE --start ADDR --end ADDR [--generate FILE]
    sass_edit.py schedule CUBIN --sass FILE --start ADDR --end ADDR [--recipe FILE] [-o OUTPUT]
    sass_edit.py find-donors CUBIN --sass FILE [--family NAME]
    sass_edit.py fatbin-patch HOST_BINARY --sass FILE [--script FILE | --stall ADDR VAL] -o OUTPUT
"""

import argparse
import collections
import io
import os
import re
import struct
import sys
import tempfile
from pathlib import Path

from elftools.elf.elffile import ELFFile

INSN_SIZE = 16  # 128-bit instructions = 16 bytes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Control word decode/encode
#
# Bit layout (tentative SM100a — stall field verified, others from SM89):
#   [3:0]   stall count (0-15)  ← VERIFIED
#   [4]     yield hint          ← plausible
#   [9:5]   write barrier       ← SM89 layout, may differ on SM100a
#   [14:10] read barrier        ← SM89 layout, may differ on SM100a
#   [20:15] barrier wait mask   ← SM89 layout, may differ on SM100a
#   [22:21] register reuse      ← SM89 layout, may differ on SM100a
#   [63:23] upper bits (purpose unknown, preserved on edits)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def decode_ctrl(ctrl):
    """Decode control word into field dict (SM89 layout, stall verified for SM100a)."""
    return {
        'stall':     ctrl & 0xf,
        'yield':     (ctrl >> 4) & 1,
        'wr_bar':    (ctrl >> 5) & 0x1f,
        'rd_bar':    (ctrl >> 10) & 0x1f,
        'wait_mask': (ctrl >> 15) & 0x3f,
        'reuse':     (ctrl >> 21) & 3,
    }


def encode_ctrl(fields, original_ctrl):
    """Encode control word fields, preserving bits above [22]."""
    mask = 0x7fffff  # bits [22:0]
    val = ((fields['stall'] & 0xf) |
           ((fields['yield'] & 1) << 4) |
           ((fields['wr_bar'] & 0x1f) << 5) |
           ((fields['rd_bar'] & 0x1f) << 10) |
           ((fields['wait_mask'] & 0x3f) << 15) |
           ((fields['reuse'] & 3) << 21))
    return (original_ctrl & ~mask) | val


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Latency table (calibrated on B200, 2026-03-21)
#
# Maps SASS mnemonic base → minimum cycles before a dependent consumer can
# read the register result. Stores have no register output (latency=0).
# For loads (LDS/LDG/LDTM), latency exceeds the 15-cycle stall cap —
# the hardware uses barrier mechanisms for these, not stall counts alone.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LATENCY = {
    'IADD3': 2, 'LEA': 2, 'MOV': 2, 'UMOV': 2, 'S2R': 2,
    'FADD': 4, 'FMUL': 4, 'FFMA': 4,
    'IMAD': 4, 'F2FP': 4, 'LOP3': 4, 'SHF': 4, 'PRMT': 4,
    'HADD2': 4, 'HFMA2': 4,
    'ISETP': 4, 'FSETP': 4, 'HSETP2': 4, 'PLOP3': 4,
    'SEL': 5, 'CSEL': 5,
    'R2UR': 4,
    'VIADD': 10, 'SHFL': 10,
    'REDUX': 44,
    'LDS': 20, 'LDSM': 20,
    'LDG': 40, 'LDC': 20,
    'LDTM': 20,
}
LATENCY_DEFAULT = 4
MAX_STALL = 15

STORE_NO_REG_OUTPUT = frozenset([
    'STS', 'STG', 'STL', 'ATOMS', 'ATOMG', 'RED',
])


def get_latency(mnemonic):
    """Get producer latency in cycles. Stores return 0 (no register output)."""
    if not mnemonic:
        return LATENCY_DEFAULT
    bare = mnemonic.split(None, 1)[-1] if mnemonic.startswith('@') else mnemonic
    base = bare.split('.')[0]
    if base in STORE_NO_REG_OUTPUT:
        return 0
    return LATENCY.get(base, LATENCY_DEFAULT)


def ctrl_str(ctrl_val):
    """Format control word for display."""
    c = decode_ctrl(ctrl_val)
    parts = ['st=%2d' % c['stall']]
    if c['yield']:
        parts.append('Y')
    # Show raw low 23 bits for full visibility
    parts.append('lo=%06x' % (ctrl_val & 0x7fffff))
    hi = ctrl_val >> 23
    if hi:
        parts.append('hi=%010x' % hi)
    return ' '.join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Instruction encoding — register field positions (SM100a)
#
# SM100a SASS instructions are 128 bits: encoding (64-bit) + control (64-bit).
# Register operands are encoded as 8-bit fields in the encoding word.
# Positions verified empirically by cross-referencing FC2 cuobjdump text
# (mnemonic + operands) with binary encoding values.
#
# Standard ALU layout:
#   bits[15:0]  = opcode (instruction type + modifiers)
#   bits[23:16] = destination register (R0-R255, 0xff = RZ)
#   bits[31:24] = source register 1
#   bits[39:32] = source register 2 (or immediate for some opcodes)
#   bits[63:40] = immediate extension / modifier bits
#
# Store layout (STS, STG):
#   bits[15:0]  = opcode
#   bits[23:16] = 0x00 (no destination)
#   bits[31:24] = address register
#   bits[39:32] = data register (base of vector for .128/.64)
#   bits[63:40] = offset immediate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NOP_ENCODING = 0x0000000000007918


def insn_family(mnemonic):
    """Get base instruction family (strip predicate guard + dot suffixes)."""
    if not mnemonic:
        return None
    bare = mnemonic.split(None, 1)[-1] if mnemonic.startswith('@') else mnemonic
    return bare.split('.')[0]


def read_enc_field(encoding, bit_offset, width):
    """Read a field from the 64-bit encoding word."""
    return (encoding >> bit_offset) & ((1 << width) - 1)


def write_enc_field(encoding, bit_offset, width, value):
    """Write a field in the 64-bit encoding word, preserving all other bits."""
    mask = ((1 << width) - 1) << bit_offset
    return (encoding & ~mask) | ((value & ((1 << width) - 1)) << bit_offset)


# Per-family register field layout.
# Each entry: list of (field_name, text_operand_index, bit_offset, width).
# text_operand_index = which cuobjdump text operand this field corresponds to.
# Field positions verified from FC2 NS5 SASS dump cross-reference.
FAMILY_REG_FIELDS = {
    # Standard 3-operand ALU: Rd @ [23:16], Rs1 @ [31:24], Rs2 @ [39:32]
    'F2FP':  [('dst', 0, 16, 8), ('src1', 1, 24, 8), ('src2', 2, 32, 8)],
    'HADD2': [('dst', 0, 16, 8), ('src1', 1, 24, 8), ('src2', 2, 32, 8)],
    'HFMA2': [('dst', 0, 16, 8), ('src1', 1, 24, 8), ('src2', 2, 32, 8)],
    'FADD':  [('dst', 0, 16, 8), ('src1', 1, 24, 8), ('src2', 2, 32, 8)],
    'FMUL':  [('dst', 0, 16, 8), ('src1', 1, 24, 8), ('src2', 2, 32, 8)],
    'FFMA':  [('dst', 0, 16, 8), ('src1', 1, 24, 8), ('src2', 2, 32, 8)],
    'SEL':   [('dst', 0, 16, 8), ('src1', 1, 24, 8), ('src2', 2, 32, 8)],
    'MOV':   [('dst', 0, 16, 8), ('src1', 1, 24, 8)],
    'LOP3':  [('dst', 0, 16, 8), ('src1', 1, 24, 8)],
    'SHF':   [('dst', 0, 16, 8), ('src1', 1, 24, 8)],
    'PRMT':  [('dst', 0, 16, 8), ('src1', 1, 24, 8)],

    # Integer ALU: dst @ [23:16], first GPR source @ [31:24]
    'IADD3': [('dst', 0, 16, 8), ('src1', 3, 24, 8)],
    'IMAD':  [('dst', 0, 16, 8), ('src1', 1, 24, 8)],
    'LEA':   [('dst', 0, 16, 8), ('src1', 1, 24, 8)],

    # Stores: no dst; addr @ [31:24], data @ [39:32]
    'STS':   [('addr', 0, 24, 8), ('data', 1, 32, 8)],
    'STG':   [('addr', 0, 24, 8), ('data', 1, 32, 8)],
    'STL':   [('addr', 0, 24, 8), ('data', 1, 32, 8)],

    # Loads: dst @ [23:16], addr @ [31:24]
    'LDG':   [('dst', 0, 16, 8), ('addr', 1, 24, 8)],
    'LDS':   [('dst', 0, 16, 8), ('addr', 1, 24, 8)],
    'LDC':   [('dst', 0, 16, 8)],
    'LDTM':  [('dst', 0, 16, 8)],
    'LDSM':  [('dst', 0, 16, 8)],

    # System: dst only
    'S2R':   [('dst', 0, 16, 8)],
    'R2UR':  [],

    # Tensor core + TMA (uniform register operands, not GPR)
    # UTCQMMA.2CTA gdesc[URa], gdesc[URb], tmem[URc], tmem[URd], idesc[URe]
    #   bits[31:24]=URa (gdesc1), [39:32]=URb (gdesc2), [47:40]=URd (tmem_src)
    #   URc (tmem_dst) and URe (idesc) packed elsewhere
    'UTCQMMA': [('ur_gdesc1', 0, 24, 8), ('ur_gdesc2', 1, 32, 8),
                ('ur_tmem_src', 3, 40, 8)],
    # UTMALDG.2D.2CTA [URa], [URb]
    #   bits[31:24]=URb (src desc), [39:32]=URa (dst desc)  (reversed from text!)
    'UTMALDG': [('ur_dst', 0, 32, 8), ('ur_src', 1, 24, 8)],
    # UTMASTG.2D [URa], [URb] — reversed: URb@[31:24], URa@[39:32]
    'UTMASTG': [('ur_dst', 0, 32, 8), ('ur_src', 1, 24, 8)],

    # Store-to-async-shared: same layout as STS
    'STAS':  [('addr', 0, 24, 8), ('data', 1, 32, 8)],

    # Shuffle: SHFL.IDX PT, Rd, Rs, Rclamp, imm
    # text op0=pred, op1=dst, op2=src, op3=clamp
    'SHFL':  [('dst', 1, 16, 8), ('src', 2, 24, 8), ('clamp', 3, 32, 8)],

    # Return: non-standard encoding
    'RET':   [],

    # Uniform find-leading-one: UFLO.U32 URd, URs → dst@[23:16], src@[39:32]
    'UFLO':  [('ur_dst', 0, 16, 8), ('ur_src', 1, 32, 8)],

    # Tensor core atomics:
    #   UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN UP, URa, URb → URa@[23:16], URb@[39:32]
    #   UTCATOMSWS.AND URZ, URa → URa@[39:32]
    # Only the FIND_AND_SET variant has op at [23:16]; AND variant has 0xff there.
    # Use the common field (op @ [39:32]) for both.
    'UTCATOMSWS': [('ur_op', -1, 32, 8)],

    # Predicates: complex encoding, not field-patchable
    'ISETP': [],
    'FSETP': [],
    'HSETP2': [],
    'PLOP3': [],

    # Control: no patchable register fields
    'NOP':   [],
    'EXIT':  [],
    'BRA':   [],
    'BSSY':  [],
    'BSYNC': [],
    'BAR':   [],
    'MEMBAR': [],
    'FENCE': [],
    'WARPSYNC': [],
    'YIELD': [],
    'NANOSLEEP': [],
    'DEPBAR': [],
    'UTCBAR': [],
}

_DEFAULT_REG_FIELDS = [('dst', 0, 16, 8), ('src1', 1, 24, 8), ('src2', 2, 32, 8)]


def get_reg_fields(mnemonic):
    """Get register field layout for an instruction.

    Returns list of (field_name, text_op_index, bit_offset, width).
    """
    family = insn_family(mnemonic)
    if family is None:
        return []
    return FAMILY_REG_FIELDS.get(family, _DEFAULT_REG_FIELDS)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Instruction / Kernel types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Instruction:
    __slots__ = ('offset', 'encoding', 'control', 'mnemonic', 'operands')

    def __init__(self, offset, encoding, control):
        self.offset = offset
        self.encoding = encoding
        self.control = control
        self.mnemonic = None
        self.operands = None

    @property
    def stall(self):
        return self.control & 0xf

    def clone(self):
        i = Instruction(self.offset, self.encoding, self.control)
        i.mnemonic = self.mnemonic
        i.operands = self.operands
        return i


class Kernel:
    def __init__(self, name, section_name, section_idx, file_offset, size):
        self.name = name
        self.section_name = section_name
        self.section_idx = section_idx
        self.file_offset = file_offset
        self.size = size
        self.instructions = []

    @property
    def short_name(self):
        """Demangled-ish short name."""
        # Try to extract template function name
        m = re.search(r'_Z\d+(\w+)', self.name)
        return m.group(1) if m else self.name[:60]

    @property
    def n_insns(self):
        return len(self.instructions)

    def parse_instructions(self, data):
        self.instructions = []
        for i in range(0, len(data) - 15, INSN_SIZE):
            enc = struct.unpack_from('<Q', data, i)[0]
            ctrl = struct.unpack_from('<Q', data, i + 8)[0]
            self.instructions.append(Instruction(i, enc, ctrl))

    def to_bytes(self):
        buf = bytearray(len(self.instructions) * INSN_SIZE)
        for i, insn in enumerate(self.instructions):
            struct.pack_into('<Q', buf, i * 16, insn.encoding)
            struct.pack_into('<Q', buf, i * 16 + 8, insn.control)
        return bytes(buf)

    def insn_at(self, addr):
        idx = addr // INSN_SIZE
        if 0 <= idx < len(self.instructions):
            return self.instructions[idx]
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Register def/use analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Regex patterns for operand parsing
_RE_GPR = re.compile(r'\bR(\d+)')           # R0-R255
_RE_URG = re.compile(r'\bUR(\d+)')          # UR0-UR63
_RE_PRED = re.compile(r'\bP(\d)\b')         # P0-P6 (not PT)
_RE_UPRED = re.compile(r'\bUP(\d)\b')       # UP0-UP6 (not UPT)

# Instructions that store to memory (no register def from the data operands)
_STORE_OPS = frozenset([
    'STS', 'STG', 'STL', 'STS.64', 'STS.128',
    'STG.64', 'STG.128', 'STL.64', 'STL.128',
    'STG.E', 'STG.E.64', 'STG.E.128',
    'STG.E.SYS', 'STG.E.SYS.128',
])

# Instructions that are pure control flow / barriers (no GPR defs)
_NO_DEF_OPS = frozenset([
    'NOP', 'EXIT', 'RET', 'BRA', 'BRA.U', 'BRA.DIV',
    'BSSY', 'BSYNC', 'BSYNC.RECONVERGENT',
    'BAR.SYNC', 'WARPSYNC', 'WARPSYNC.ALL',
    'MEMBAR', 'MEMBAR.ALL.CTA', 'MEMBAR.SC.GPU',
    'FENCE', 'FENCE.VIEW.ASYNC.S',
    'YIELD', 'NANOSLEEP',
])

# Instructions with predicate output in a non-first-operand position
_PRED_DEF_OPS = frozenset([
    'ISETP', 'FSETP', 'HSETP2', 'DSETP',
    'PLOP3', 'PLOP3.LUT', 'VOTEU', 'VOTEU.ALL',
])

# Vector width implied by instruction suffix
_VECTOR_WIDTHS = {
    '.128': 4,    # 4 x 32-bit regs
    '.64': 2,     # 2 x 32-bit regs
}


def _base_op(mnemonic):
    """Strip predicate guard and get base opcode (before first dot, or full name)."""
    op = mnemonic
    if op.startswith('@'):
        op = op.split(None, 1)[-1] if ' ' in op else op
    return op


def _strip_pred(mnemonic):
    """Return (guard_pred_str_or_None, bare_opcode)."""
    if not mnemonic.startswith('@'):
        return None, mnemonic
    parts = mnemonic.split(None, 1)
    return parts[0], parts[1] if len(parts) > 1 else ''


def _parse_guard_pred(mnemonic):
    """Extract guard predicate register from @P0 or @!P1 prefix. Returns set of pred reg strings."""
    if not mnemonic or not mnemonic.startswith('@'):
        return set()
    guard = mnemonic.split()[0]  # "@P0" or "@!P1" or "@!UP0"
    m = re.match(r'@!?U?P(\d)', guard)
    if m:
        if 'UP' in guard:
            return {'UP' + m.group(1)}
        return {'P' + m.group(1)}
    return set()


def _vector_width(opcode):
    """Determine how many consecutive registers a load/store uses."""
    for suffix, width in _VECTOR_WIDTHS.items():
        if suffix in opcode:
            return width
    return 1


def _ldtm_width(opcode, operands):
    """LDTM.xN defines N consecutive registers starting from the dest reg."""
    m = re.search(r'\.x(\d+)', opcode)
    if m:
        return int(m.group(1))
    return 1


def _reg_range(base_reg, width, prefix='R'):
    """Generate set of register names for a vector of width regs starting at base_reg."""
    return {prefix + str(base_reg + i) for i in range(width)}


def parse_reg_operands(mnemonic, operands):
    """Parse SASS instruction into (defs, uses) sets of register names.

    Register names: 'R0'-'R255', 'UR0'-'UR63', 'P0'-'P6', 'UP0'-'UP6'.
    Ignores RZ, URZ, PT, UPT (zero/true constants).

    Returns (defs: set[str], uses: set[str]).
    """
    if not mnemonic or not operands:
        return set(), set()

    defs = set()
    uses = set()

    guard_pred, bare_op = _strip_pred(mnemonic)

    # Guard predicate is always a USE
    uses |= _parse_guard_pred(mnemonic)

    # Split operands by comma, respecting brackets
    raw_ops = _split_operands(operands)

    # Determine instruction class from bare_op
    op_base = bare_op.split('.')[0]
    full_op = bare_op

    # --- Store instructions: all operands are uses ---
    # For vector stores (STS.128, STG.64, etc.), the data register
    # expands to consecutive regs based on vector width
    if _is_store(full_op):
        width = _vector_width(full_op)
        for i, op in enumerate(raw_ops):
            if i == len(raw_ops) - 1 and width > 1:
                # Last operand is the data source — expand to vector
                m = _RE_GPR.search(op.replace('.reuse', ''))
                if m:
                    base = int(m.group(1))
                    uses |= _reg_range(base, width)
                else:
                    uses |= _extract_regs(op)
            else:
                uses |= _extract_regs(op)
        return defs, uses

    # --- No-def instructions (branches, barriers, fences) ---
    if full_op in _NO_DEF_OPS or op_base in ('BRA', 'BSSY', 'BSYNC', 'EXIT',
                                               'RET', 'WARPSYNC', 'BAR',
                                               'MEMBAR', 'FENCE', 'NOP',
                                               'YIELD', 'NANOSLEEP', 'KILL'):
        for op in raw_ops:
            uses |= _extract_regs(op)
        return defs, uses

    # --- UTMASTG: TMA store, all operands are uses (uniform regs) ---
    if op_base == 'UTMASTG':
        for op in raw_ops:
            uses |= _extract_regs(op)
        return defs, uses

    # --- UTCBAR: barrier op, all operands are uses ---
    if op_base == 'UTCBAR':
        for op in raw_ops:
            uses |= _extract_regs(op)
        return defs, uses

    # --- Predicate-defining instructions (ISETP, PLOP3, VOTEU, etc.) ---
    if op_base in ('ISETP', 'FSETP', 'HSETP2', 'DSETP'):
        # Format: ISETP.cc.AND P0, PT, R36, RZ, PT
        # Defs: first two operands (pred, pred), Uses: rest
        for i, op in enumerate(raw_ops):
            if i < 2:
                defs |= _extract_preds(op)
            else:
                uses |= _extract_regs(op)
        return defs, uses

    if op_base == 'PLOP3':
        # PLOP3.LUT P2, PT, P1, PT, PT, 0x8, 0x80
        # Defs: first two operands, Uses: rest
        for i, op in enumerate(raw_ops):
            if i < 2:
                defs |= _extract_preds(op)
            else:
                uses |= _extract_regs(op)
        return defs, uses

    if op_base == 'VOTEU':
        # VOTEU.ALL UP0, P1
        # Defs: first operand (uniform pred), second is use
        if raw_ops:
            defs |= _extract_preds(raw_ops[0])
        for op in raw_ops[1:]:
            uses |= _extract_regs(op)
        return defs, uses

    # --- R2UR: move GPR to uniform reg ---
    if op_base == 'R2UR':
        # R2UR UR6, R4  or  R2UR P0, UR10, R99  or  R2UR.OR P0, UR9, R60
        # First operand(s) that are UR/P are defs, GPR sources are uses
        for i, op in enumerate(raw_ops):
            op_stripped = op.strip()
            if i == 0:
                # Could be pred def (P0) or UR def
                defs |= _extract_preds(op_stripped)
                defs |= _extract_uregs(op_stripped)
            elif _RE_URG.search(op_stripped) and i <= 1 and 'P' in raw_ops[0]:
                # Second operand is UR def when first was pred
                defs |= _extract_uregs(op_stripped)
            else:
                uses |= _extract_regs(op_stripped)
        return defs, uses

    # --- LDTM: load from TMEM, defines N consecutive GPRs ---
    if op_base == 'LDTM':
        width = _ldtm_width(full_op, operands)
        if raw_ops:
            m = _RE_GPR.search(raw_ops[0])
            if m:
                base = int(m.group(1))
                defs |= _reg_range(base, width)
        for op in raw_ops[1:]:
            uses |= _extract_regs(op)
        return defs, uses

    # --- LDC: load constant, defines dest reg(s) ---
    if op_base == 'LDC':
        width = 2 if '.64' in full_op else 1
        if raw_ops:
            m = _RE_GPR.search(raw_ops[0])
            if m:
                base = int(m.group(1))
                defs |= _reg_range(base, width)
        # c[x][y] operands have no register refs typically
        for op in raw_ops[1:]:
            uses |= _extract_regs(op)
        return defs, uses

    # --- LDG/LDS/LDL: load from memory, defines dest regs ---
    if op_base in ('LDG', 'LDS', 'LDL', 'LDSM'):
        width = _vector_width(full_op)
        if raw_ops:
            m = _RE_GPR.search(raw_ops[0])
            if m:
                base = int(m.group(1))
                defs |= _reg_range(base, width)
        for op in raw_ops[1:]:
            uses |= _extract_regs(op)
        return defs, uses

    # --- UMOV: move to/from uniform regs ---
    if op_base == 'UMOV':
        if raw_ops:
            defs |= _extract_uregs(raw_ops[0])
        for op in raw_ops[1:]:
            uses |= _extract_regs(op)
        return defs, uses

    # --- IMAD with predicate output ---
    # IMAD can write a carry predicate: IMAD.IADD R142, R115, 0x1, R142
    # But IADD3 has explicit pred outputs: IADD3 R130, P0, PT, R156, 0x2, RZ
    if op_base == 'IADD3':
        # IADD3 Rd, Pcarry, Pborrow, Ra, imm/Rb, Rc [, Pcarry_in, Pborrow_in]
        # First op = GPR def, next two = pred defs, rest = uses
        if raw_ops:
            m = _RE_GPR.search(raw_ops[0])
            if m:
                defs.add('R' + m.group(1))
        for i, op in enumerate(raw_ops[1:3], 1):
            defs |= _extract_preds(op)
        for op in raw_ops[3:]:
            uses |= _extract_regs(op)
        return defs, uses

    # --- LEA / LEA.HI ---
    if op_base == 'LEA':
        if raw_ops:
            m = _RE_GPR.search(raw_ops[0])
            if m:
                defs.add('R' + m.group(1))
        for op in raw_ops[1:]:
            uses |= _extract_regs(op)
        return defs, uses

    # --- Default: first operand is def, rest are uses ---
    if raw_ops:
        first = raw_ops[0].strip()
        # Check if first operand looks like a register (not a memory address)
        if first.startswith('[') or first.startswith('tmem'):
            # Unusual — treat everything as uses (safety)
            for op in raw_ops:
                uses |= _extract_regs(op)
        else:
            # First operand: defs
            m_gpr = _RE_GPR.search(first)
            if m_gpr:
                base = int(m_gpr.group(1))
                # Check for vector width on the instruction
                width = _vector_width(full_op)
                defs |= _reg_range(base, width)
            defs |= _extract_uregs(first)
            defs |= _extract_preds(first)

            # Remaining operands: uses
            for op in raw_ops[1:]:
                uses |= _extract_regs(op)

    return defs, uses


def _is_store(opcode):
    """Check if opcode is a store instruction (no register def)."""
    if opcode in _NO_DEF_OPS:
        return False
    base = opcode.split('.')[0]
    if base in ('STS', 'STG', 'STL', 'ATOMS', 'ATOMG', 'RED'):
        return True
    if opcode in _STORE_OPS:
        return True
    return False


def _split_operands(operands):
    """Split operands by comma, respecting brackets."""
    result = []
    depth = 0
    current = []
    for ch in operands:
        if ch in '([':
            depth += 1
            current.append(ch)
        elif ch in ')]':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            result.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        result.append(''.join(current).strip())
    return result


def _extract_regs(text):
    """Extract all register references (GPR, UR, pred) from an operand string.
    Ignores RZ, URZ, PT, UPT."""
    regs = set()
    # Strip .reuse suffix for matching
    text = text.replace('.reuse', '')
    for m in _RE_GPR.finditer(text):
        regs.add('R' + m.group(1))
    for m in _RE_URG.finditer(text):
        regs.add('UR' + m.group(1))
    for m in _RE_PRED.finditer(text):
        regs.add('P' + m.group(1))
    for m in _RE_UPRED.finditer(text):
        regs.add('UP' + m.group(1))
    return regs


def _extract_preds(text):
    """Extract only predicate registers from text. Ignores PT/UPT."""
    preds = set()
    for m in _RE_PRED.finditer(text):
        preds.add('P' + m.group(1))
    for m in _RE_UPRED.finditer(text):
        preds.add('UP' + m.group(1))
    return preds


def _extract_uregs(text):
    """Extract only uniform registers from text. Ignores URZ."""
    uregs = set()
    for m in _RE_URG.finditer(text):
        uregs.add('UR' + m.group(1))
    return uregs


# Instructions that act as reorder boundaries — no instruction may move
# across one of these in a reorder, even if register deps are satisfied.
_BARRIER_BASES = frozenset([
    'BRA', 'BSSY', 'BSYNC', 'EXIT', 'RET', 'CALL', 'BREAK', 'CONT',
    'BAR', 'WARPSYNC', 'MEMBAR', 'FENCE',
    'UTCBAR', 'DEPBAR',
    'YIELD', 'KILL',
])


def is_barrier(mnemonic):
    """Check if an instruction is a reorder boundary."""
    if not mnemonic:
        return False
    _, bare = _strip_pred(mnemonic)
    base = bare.split('.')[0]
    return base in _BARRIER_BASES


class DepViolation:
    """A dependency violation from a proposed reorder."""
    __slots__ = ('kind', 'producer_addr', 'consumer_addr', 'reg',
                 'producer_mnemonic', 'consumer_mnemonic',
                 'orig_order', 'new_order')

    def __init__(self, kind, producer_addr, consumer_addr, reg,
                 producer_mn='', consumer_mn=''):
        self.kind = kind  # 'RAW', 'WAW', 'WAR'
        self.producer_addr = producer_addr
        self.consumer_addr = consumer_addr
        self.reg = reg
        self.producer_mnemonic = producer_mn
        self.consumer_mnemonic = consumer_mn

    def __str__(self):
        return '%s [%04x] %s → [%04x] %s  via %s' % (
            self.kind,
            self.producer_addr, self.producer_mnemonic[:30],
            self.consumer_addr, self.consumer_mnemonic[:30],
            self.reg)


def check_deps(instructions, new_order_addrs=None):
    """Check for dependency and barrier violations in a proposed reorder.

    instructions: list of Instruction objects (must have mnemonic/operands set)
    new_order_addrs: if given, list of addresses in the proposed new order.
        If None, checks the existing order (useful for dumping deps).

    Returns list of DepViolation for any RAW/WAW/WAR or BARRIER violations.
    """
    # Build addr→instruction map
    addr_map = {insn.offset: insn for insn in instructions}

    if new_order_addrs is None:
        ordered = list(instructions)
    else:
        ordered = [addr_map[a] for a in new_order_addrs]

    # Parse defs/uses for each instruction
    du = {}
    for insn in ordered:
        d, u = parse_reg_operands(insn.mnemonic, insn.operands)
        du[insn.offset] = (d, u)

    orig_pos = {insn.offset: i for i, insn in enumerate(instructions)}
    new_pos = {insn.offset: i for i, insn in enumerate(ordered)}

    violations = []

    # --- Barrier boundary check ---
    # A barrier instruction pins all instructions on its original side.
    # No instruction from before the barrier may appear after it (or vice versa).
    if new_order_addrs is not None:
        for insn in instructions:
            if not is_barrier(insn.mnemonic):
                continue
            bar_orig = orig_pos[insn.offset]
            bar_new = new_pos[insn.offset]

            for other in instructions:
                if other.offset == insn.offset:
                    continue
                oth_orig = orig_pos[other.offset]
                oth_new = new_pos[other.offset]

                # Was before barrier, now after
                if oth_orig < bar_orig and oth_new > bar_new:
                    violations.append(DepViolation(
                        'BARRIER', other.offset, insn.offset, '-',
                        other.mnemonic or '', insn.mnemonic or ''))
                # Was after barrier, now before
                elif oth_orig > bar_orig and oth_new < bar_new:
                    violations.append(DepViolation(
                        'BARRIER', insn.offset, other.offset, '-',
                        insn.mnemonic or '', other.mnemonic or ''))

    # --- Register dependency check ---
    addrs = [insn.offset for insn in instructions]
    n = len(addrs)

    for i in range(n):
        addr_i = addrs[i]
        defs_i, uses_i = du[addr_i]
        mn_i = instructions[i].mnemonic or ''

        for j in range(i + 1, n):
            addr_j = addrs[j]
            defs_j, uses_j = du[addr_j]
            mn_j = instructions[j].mnemonic or ''

            # RAW: i defines R, j uses R → i must come before j
            raw_regs = defs_i & uses_j
            for r in raw_regs:
                if new_pos[addr_i] > new_pos[addr_j]:
                    violations.append(DepViolation(
                        'RAW', addr_i, addr_j, r, mn_i, mn_j))

            # WAW: i defines R, j defines R → i must come before j
            waw_regs = defs_i & defs_j
            for r in waw_regs:
                if new_pos[addr_i] > new_pos[addr_j]:
                    violations.append(DepViolation(
                        'WAW', addr_i, addr_j, r, mn_i, mn_j))

            # WAR: i uses R, j defines R → i must come before j
            war_regs = uses_i & defs_j
            for r in war_regs:
                if new_pos[addr_i] > new_pos[addr_j]:
                    violations.append(DepViolation(
                        'WAR', addr_i, addr_j, r, mn_i, mn_j))

    return violations


def dump_deps(instructions):
    """Print def/use analysis for a list of instructions."""
    for insn in instructions:
        defs, uses = parse_reg_operands(insn.mnemonic, insn.operands)
        def_str = ','.join(sorted(defs)) if defs else '-'
        use_str = ','.join(sorted(uses)) if uses else '-'
        mn = insn.mnemonic or '???'
        ops = insn.operands or ''
        bar = ' [BARRIER]' if is_barrier(insn.mnemonic) else ''
        print('[%04x] %-45s  def={%-20s}  use={%s}%s' % (
            insn.offset, (mn + ' ' + ops)[:45], def_str, use_str, bar))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stall recomputation + audit
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

StallChange = collections.namedtuple('StallChange', 'addr new_stall old_stall reason')
StallWarning = collections.namedtuple('StallWarning',
    'addr mnemonic producer_addr producer_mnemonic needed actual reg')


def compute_stalls(instructions, keep_first=True):
    """Compute minimum-safe stall counts for a (possibly reordered) instruction list.

    For each instruction i, finds its nearest RAW producer j (j < i), sums
    stall cycles of instructions between j and i, and sets the stall on i
    to cover any latency deficit.

    Returns list of StallChange for every instruction whose stall changed.
    """
    n = len(instructions)
    if n == 0:
        return []

    # parse defs/uses
    du = []
    for insn in instructions:
        d, u = parse_reg_operands(insn.mnemonic, insn.operands)
        du.append((d, u))

    changes = []
    start = 1 if keep_first else 0

    for i in range(start, n):
        insn = instructions[i]
        _, uses_i = du[i]
        if not uses_i:
            continue

        max_needed = 0
        worst_producer = None
        worst_reg = None

        # scan backwards for nearest producer of each used register
        for reg in uses_i:
            for j in range(i - 1, -1, -1):
                defs_j, _ = du[j]
                if reg in defs_j:
                    lat = get_latency(instructions[j].mnemonic)
                    if lat == 0:
                        break
                    # sum stall counts between producer and consumer (exclusive)
                    covered = sum(instructions[k].control & 0xf for k in range(j + 1, i))
                    needed = lat - covered
                    if needed > max_needed:
                        max_needed = needed
                        worst_producer = j
                        worst_reg = reg
                    break  # found nearest producer for this reg

        if max_needed <= 0:
            continue

        new_stall = min(max_needed, MAX_STALL)
        old_stall = insn.control & 0xf
        if new_stall == old_stall:
            continue

        reason = ''
        if worst_producer is not None:
            pmn = instructions[worst_producer].mnemonic or '?'
            reason = 'lat=%d from [%04x] %s via %s' % (
                get_latency(instructions[worst_producer].mnemonic),
                instructions[worst_producer].offset, pmn, worst_reg)
            if max_needed > MAX_STALL:
                reason += ' (NEEDS BARRIER: %d > %d)' % (max_needed, MAX_STALL)

        changes.append(StallChange(insn.offset, new_stall, old_stall, reason))

    return changes


def apply_stall_changes(instructions, changes):
    """Apply stall changes to instruction list. Returns number applied."""
    addr_map = {insn.offset: insn for insn in instructions}
    applied = 0
    for ch in changes:
        insn = addr_map.get(ch.addr)
        if insn is None:
            continue
        insn.control = (insn.control & ~0xf) | (ch.new_stall & 0xf)
        applied += 1
    return applied


def audit_stalls(instructions):
    """Check existing stall counts against minimum latency requirements.

    Returns list of StallWarning for any instruction whose stall count is
    insufficient to cover its producer's latency.
    """
    n = len(instructions)
    du = []
    for insn in instructions:
        d, u = parse_reg_operands(insn.mnemonic, insn.operands)
        du.append((d, u))

    warnings = []
    for i in range(1, n):
        insn = instructions[i]
        _, uses_i = du[i]
        if not uses_i:
            continue

        for reg in uses_i:
            for j in range(i - 1, -1, -1):
                defs_j, _ = du[j]
                if reg in defs_j:
                    lat = get_latency(instructions[j].mnemonic)
                    if lat == 0:
                        break
                    covered = sum(instructions[k].control & 0xf for k in range(j + 1, i))
                    if covered < lat:
                        warnings.append(StallWarning(
                            insn.offset,
                            insn.mnemonic or '?',
                            instructions[j].offset,
                            instructions[j].mnemonic or '?',
                            lat, covered, reg))
                    break

    return warnings


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Encoding verification + donor lookup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_RE_GPR_NUM = re.compile(r'\bR(\d+)')


def probe_reg_fields(kernel):
    """Verify register field positions by cross-referencing SASS text with binary.

    For each instruction with known text operands, checks that register numbers
    parsed from the cuobjdump text match the values at the expected bit positions
    in the encoding word.

    Returns dict: family → {'matches': n, 'mismatches': n, 'details': [...]}.
    """
    results = {}

    for insn in kernel.instructions:
        if not insn.mnemonic or not insn.operands:
            continue

        family = insn_family(insn.mnemonic)
        if family is None:
            continue

        fields = get_reg_fields(insn.mnemonic)
        if not fields:
            continue

        raw_ops = _split_operands(insn.operands)

        if family not in results:
            results[family] = {'matches': 0, 'mismatches': 0, 'details': []}

        for field_name, text_idx, bit_off, width in fields:
            if text_idx >= len(raw_ops):
                continue

            op_text = raw_ops[text_idx].replace('.reuse', '')
            # Match UR fields with UR regex, GPR fields with GPR regex
            if field_name.startswith('ur_'):
                m = _RE_URG.search(op_text)
            else:
                m = _RE_GPR_NUM.search(op_text)
            if not m:
                continue

            expected = int(m.group(1))
            actual = read_enc_field(insn.encoding, bit_off, width)

            if actual == expected:
                results[family]['matches'] += 1
            else:
                results[family]['mismatches'] += 1
                results[family]['details'].append(
                    '[%04x] %s: %s expected R%d at bits[%d:%d], got %d (0x%02x)' % (
                        insn.offset, insn.mnemonic, field_name,
                        expected, bit_off + width - 1, bit_off, actual, actual))

    return results


def find_donors(kernel):
    """Find donor instructions for each opcode family in the kernel.

    Returns dict: family → list of (addr, mnemonic, operands, encoding) for
    instructions that can serve as encoding templates.
    """
    donors = {}
    for insn in kernel.instructions:
        if not insn.mnemonic:
            continue
        family = insn_family(insn.mnemonic)
        if family is None:
            continue
        if family not in donors:
            donors[family] = []
        donors[family].append((insn.offset, insn.mnemonic, insn.operands or '',
                               insn.encoding))
    return donors


def find_nops(kernel):
    """Find all NOP instructions in the kernel (available as scratch space).

    Returns list of (addr, control_word).
    """
    nops = []
    for insn in kernel.instructions:
        if insn.encoding == NOP_ENCODING:
            nops.append((insn.offset, insn.control))
        elif insn.mnemonic and insn_family(insn.mnemonic) == 'NOP':
            nops.append((insn.offset, insn.control))
    return nops


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Epilogue analysis + software pipelining
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_VECTOR_WIDTHS_ENC = {'.128': 4, '.64': 2}


def _enc_vector_width(mnemonic):
    """Get vector width from mnemonic suffix (for encoding-level analysis)."""
    if not mnemonic:
        return 1
    bare = mnemonic.split(None, 1)[-1] if mnemonic.startswith('@') else mnemonic
    for suffix, w in _VECTOR_WIDTHS_ENC.items():
        if suffix in bare:
            return w
    return 1


class EpilogueGroup:
    """A compute+store group in the epilogue."""

    def __init__(self, group_idx, compute_insns, store_insns):
        self.group_idx = group_idx
        self.compute_insns = compute_insns
        self.store_insns = store_insns

    @property
    def compute_range(self):
        if not self.compute_insns:
            return (0, 0)
        return (self.compute_insns[0].offset,
                self.compute_insns[-1].offset + INSN_SIZE)

    @property
    def store_range(self):
        if not self.store_insns:
            return (0, 0)
        return (self.store_insns[0].offset,
                self.store_insns[-1].offset + INSN_SIZE)

    def compute_defs(self):
        """All registers defined by compute instructions."""
        defs = set()
        for insn in self.compute_insns:
            d, _ = parse_reg_operands(insn.mnemonic, insn.operands)
            defs |= d
        return defs

    def compute_uses(self):
        """All registers used by compute instructions."""
        uses = set()
        for insn in self.compute_insns:
            _, u = parse_reg_operands(insn.mnemonic, insn.operands)
            uses |= u
        return uses

    def store_uses(self):
        """All registers used by store instructions (addr + data vectors)."""
        uses = set()
        for insn in self.store_insns:
            _, u = parse_reg_operands(insn.mnemonic, insn.operands)
            uses |= u
        return uses

    def store_data_vectors(self):
        """Get (base_reg_num, width) for each store's data register.

        For STS.128, this returns the base register and vector width (4),
        meaning the store reads [base, base+1, base+2, base+3].
        """
        vectors = []
        for insn in self.store_insns:
            family = insn_family(insn.mnemonic)
            if family not in ('STS', 'STG', 'STL'):
                continue
            width = _enc_vector_width(insn.mnemonic)
            data_reg = read_enc_field(insn.encoding, 32, 8)
            vectors.append((data_reg, width))
        return vectors


def identify_epilogue_groups(instructions, store_family='STS'):
    """Parse instruction list into epilogue groups (compute + store blocks).

    A group is a block of non-store instructions followed by a cluster of
    store instructions. Groups are identified by finding store clusters and
    assigning preceding compute instructions to each.

    Returns list of EpilogueGroup.
    """
    n = len(instructions)

    # Find store clusters (runs of consecutive store instructions)
    clusters = []
    i = 0
    while i < n:
        insn = instructions[i]
        if insn.mnemonic and insn_family(insn.mnemonic) == store_family:
            start = i
            while i < n and instructions[i].mnemonic and \
                    insn_family(instructions[i].mnemonic) == store_family:
                i += 1
            clusters.append((start, i))
        else:
            i += 1

    if not clusters:
        return []

    # Build groups: compute block before each store cluster + the cluster
    groups = []
    prev_end = 0
    for gidx, (sstart, send) in enumerate(clusters):
        compute = instructions[prev_end:sstart]
        stores = instructions[sstart:send]
        groups.append(EpilogueGroup(gidx, compute, stores))
        prev_end = send

    return groups


def analyze_interleave(groups):
    """Analyze interleaving opportunities between adjacent epilogue groups.

    For each pair of adjacent groups (N, N+1), determines:
    - Register conflicts (group N store uses ∩ group N+1 compute defs)
    - Vector groups that need full remapping
    - Number of spare registers needed

    Returns list of dicts, one per adjacent pair.
    """
    analyses = []

    for i in range(len(groups) - 1):
        gn = groups[i]
        gn1 = groups[i + 1]

        n_store_uses = gn.store_uses()
        n1_compute_defs = gn1.compute_defs()
        conflicts = n_store_uses & n1_compute_defs

        # Expand conflicts to full vector groups for stores
        # If any register in a store's vector range conflicts, the whole vector
        # must be remapped (STS.128 reads 4 consecutive regs from base)
        expanded = set()
        for base_reg, width in gn.store_data_vectors():
            vec_regs = {'R%d' % (base_reg + j) for j in range(width)}
            if conflicts & vec_regs:
                expanded |= vec_regs

        # Also add any non-vector GPR conflicts
        expanded |= {r for r in conflicts if r.startswith('R')}

        # Compute how many spare GPRs needed for remapping
        gpr_conflicts = sorted(
            [r for r in expanded if r.startswith('R')],
            key=lambda r: int(r[1:]))

        analyses.append({
            'group_n': i,
            'group_n1': i + 1,
            'conflicts': conflicts,
            'expanded': expanded,
            'gpr_remap_needed': len(gpr_conflicts),
            'gpr_conflicts': gpr_conflicts,
            'n_store_count': len(gn.store_insns),
            'n1_compute_count': len(gn1.compute_insns),
        })

    return analyses


def plan_remap(conflicts_expanded, spare_start=208, spare_end=255):
    """Generate register remapping for conflict resolution.

    Maps conflicting GPR names to spare register numbers.
    Returns dict: reg_name (e.g., 'R89') → new_reg_number (e.g., 208).
    """
    remap = {}
    spare_next = spare_start
    for reg in sorted(conflicts_expanded,
                      key=lambda r: int(r[1:]) if r[1:].isdigit() else 999):
        if not reg.startswith('R'):
            continue
        if spare_next > spare_end:
            break
        remap[reg] = spare_next
        spare_next += 1
    return remap


def generate_interleave_recipe(groups, analyses, spare_start=208):
    """Generate an edit recipe for software-pipelined epilogue.

    Produces a list of edit commands (as strings) that can be written to a
    script file and applied with the script command.

    Strategy per group pair (N, N+1):
    1. Remap conflicting registers in group N+1's compute + stores
    2. Interleave group N+1's compute between group N's store instructions
    3. Recompute stall counts for the affected region
    """
    recipe_lines = []
    recipe_lines.append('# Auto-generated software pipeline recipe')
    recipe_lines.append('# Groups: %d, Pairs: %d' % (len(groups), len(analyses)))
    recipe_lines.append('')

    for analysis in analyses:
        gi = analysis['group_n']
        gn = groups[gi]
        gn1 = groups[gi + 1]

        recipe_lines.append('# === Groups %d-%d interleave ===' % (gi, gi + 1))

        # Register remapping
        if analysis['expanded']:
            remap = plan_remap(analysis['expanded'], spare_start)
            recipe_lines.append('# Register remap (%d registers):' % len(remap))
            for old_name, new_num in sorted(remap.items(),
                                            key=lambda x: int(x[0][1:])):
                recipe_lines.append('#   %s -> R%d' % (old_name, new_num))
            recipe_lines.append('')

            # Patch registers in group N+1's compute instructions
            for insn in gn1.compute_insns:
                if not insn.mnemonic:
                    continue
                fields = get_reg_fields(insn.mnemonic)
                defs, uses = parse_reg_operands(insn.mnemonic, insn.operands)

                # Check if any def or use needs remapping
                all_regs = defs | uses
                needs_patch = all_regs & set(remap.keys())
                if not needs_patch:
                    continue

                for field_name, text_idx, bit_off, width in fields:
                    current_val = read_enc_field(insn.encoding, bit_off, width)
                    current_name = 'R%d' % current_val
                    if current_name in remap:
                        recipe_lines.append(
                            'patch-reg 0x%04x %s %d  # %s -> R%d' % (
                                insn.offset, field_name, remap[current_name],
                                current_name, remap[current_name]))

            # Patch registers in group N+1's store instructions
            for insn in gn1.store_insns:
                if not insn.mnemonic:
                    continue
                fields = get_reg_fields(insn.mnemonic)
                for field_name, text_idx, bit_off, width in fields:
                    if field_name != 'data':
                        continue
                    current_val = read_enc_field(insn.encoding, bit_off, width)
                    current_name = 'R%d' % current_val
                    if current_name in remap:
                        recipe_lines.append(
                            'patch-reg 0x%04x %s %d  # %s -> R%d' % (
                                insn.offset, field_name, remap[current_name],
                                current_name, remap[current_name]))

            recipe_lines.append('')

        # Interleave schedule: distribute N+1 compute between N stores
        n_stores = len(gn.store_insns)
        n1_compute = list(gn1.compute_insns)

        if n_stores > 0 and n1_compute:
            # Split N+1 compute into chunks to fill between stores
            # Each STS has 32-cycle throughput → can fit ~15 compute ops in the gap
            chunk_size = max(1, len(n1_compute) // n_stores)
            chunks = []
            for ci in range(n_stores):
                start = ci * chunk_size
                end = start + chunk_size if ci < n_stores - 1 else len(n1_compute)
                chunks.append(n1_compute[start:end])

            recipe_lines.append('# Interleave schedule:')
            for si, store_insn in enumerate(gn.store_insns):
                recipe_lines.append('#   STS @ 0x%04x' % store_insn.offset)
                if si < len(chunks):
                    for ci_insn in chunks[si]:
                        recipe_lines.append(
                            '#     + 0x%04x %s' % (
                                ci_insn.offset,
                                (ci_insn.mnemonic or '?')[:40]))

            # Build the reorder address sequence
            reorder_addrs = []
            all_addrs = set()
            for si, store_insn in enumerate(gn.store_insns):
                reorder_addrs.append(store_insn.offset)
                all_addrs.add(store_insn.offset)
                if si < len(chunks):
                    for ci_insn in chunks[si]:
                        reorder_addrs.append(ci_insn.offset)
                        all_addrs.add(ci_insn.offset)

            # Only generate reorder if we have a valid contiguous region
            # This is complex because the instructions may not be contiguous
            # (group N stores and group N+1 compute are in different address ranges)
            recipe_lines.append('')
            recipe_lines.append(
                '# NOTE: Interleaving requires swapping instructions between')
            recipe_lines.append(
                '# non-contiguous regions. Use copy + nop commands to move')
            recipe_lines.append(
                '# instructions from group %d compute into NOP slots near' % (gi + 1))
            recipe_lines.append(
                '# group %d stores, then restall.' % gi)

        recipe_lines.append('')

    # Final restall for the whole region
    if groups:
        start_addr = groups[0].compute_range[0]
        end_addr = groups[-1].store_range[1]
        recipe_lines.append('# Restall the entire edited region')
        recipe_lines.append('restall 0x%04x 0x%04x' % (start_addr, end_addr))

    return recipe_lines


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CP-SAT optimal instruction scheduler
#
# Uses OR-Tools Constraint Programming to find the minimum-cycle instruction
# ordering for a SASS region, subject to:
#   - Data dependencies (RAW/WAW/WAR from register def/use analysis)
#   - Calibrated producer latencies (from B200 measurements)
#   - Per-pipe throughput constraints (from conflict matrix)
#   - Stall cap of 15 cycles (SM100a control word is 4 bits)
#   - Barrier boundaries (no crossing fences/syncs)
#
# The SM100a is in-order dispatch — the scheduler chooses an ordering, and
# the stall counts between consecutive instructions are derived from
# latency requirements. The key insight: STS.128 has 32-cycle throughput
# but max stall is 15, so the scheduler MUST interleave independent compute
# in the STS stall windows.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Execution pipe assignment from B200 conflict matrix.
# BF16 pipe: 7.5% overhead with STS (nearly independent)
# ALU pipe: 17% overhead with STS (some contention)
# STORE pipe: STS/STG/STAS share LSU, nearly additive
# FAST_INT: IADD3 is free everywhere (2-cycle latency = throughput)
# LOAD pipe: LDS/LDG/LDC/LDTM — shared memory/global load unit

PIPE_STORE = 'STORE'
PIPE_BF16 = 'BF16'
PIPE_ALU = 'ALU'
PIPE_FAST_INT = 'FAST_INT'
PIPE_LOAD = 'LOAD'
PIPE_SPECIAL = 'SPECIAL'
PIPE_NONE = 'NONE'

_PIPE_MAP = {
    'STS':    PIPE_STORE,
    'STG':    PIPE_STORE,
    'STAS':   PIPE_STORE,
    'STL':    PIPE_STORE,
    'HADD2':  PIPE_BF16,
    'HFMA2':  PIPE_BF16,
    'F2FP':   PIPE_BF16,
    'HSETP2': PIPE_BF16,
    'IADD3':  PIPE_FAST_INT,
    'LEA':    PIPE_FAST_INT,
    'MOV':    PIPE_FAST_INT,
    'FADD':   PIPE_ALU,
    'FMUL':   PIPE_ALU,
    'FFMA':   PIPE_ALU,
    'IMAD':   PIPE_ALU,
    'LOP3':   PIPE_ALU,
    'PRMT':   PIPE_ALU,
    'SHF':    PIPE_ALU,
    'SEL':    PIPE_ALU,
    'CSEL':   PIPE_ALU,
    'ISETP':  PIPE_ALU,
    'FSETP':  PIPE_ALU,
    'PLOP3':  PIPE_ALU,
    'LDS':    PIPE_LOAD,
    'LDSM':   PIPE_LOAD,
    'LDG':    PIPE_LOAD,
    'LDC':    PIPE_LOAD,
    'LDTM':   PIPE_LOAD,
    'REDUX':  PIPE_SPECIAL,
    'SHFL':   PIPE_SPECIAL,
    'VIADD':  PIPE_SPECIAL,
    'R2UR':   PIPE_SPECIAL,
    'S2R':    PIPE_FAST_INT,
    'UMOV':   PIPE_FAST_INT,
    'UFLO':   PIPE_SPECIAL,
}

# Per-pipe throughput: minimum cycles between two instructions on the same pipe.
# Store pipe throughput is per-instruction (STS.128=32, STG=26).
# ALU/BF16/FAST_INT all have throughput=2 at saturation.
_PIPE_THROUGHPUT = {
    PIPE_STORE:    32,  # STS.128 dominates, STG=26 but conservative
    PIPE_BF16:     2,
    PIPE_ALU:      2,
    PIPE_FAST_INT: 2,
    PIPE_LOAD:     4,   # LDS=4 throughput
    PIPE_SPECIAL:  10,  # REDUX=11, SHFL=10, conservative
    PIPE_NONE:     1,
}

# Cross-pipe conflict overhead (additive cycles when two pipes co-issue).
# From the conflict matrix: BF16×STS=7.5% (~2.4 cyc overhead per STS pair),
# ALU×STS=17% (~5.4 cyc). We model this as minimum gap when interleaving.
_CROSS_PIPE_MIN_GAP = {
    (PIPE_BF16, PIPE_STORE): 0,     # 7.5% overhead — nearly free, don't constrain
    (PIPE_STORE, PIPE_BF16): 0,
    (PIPE_ALU, PIPE_STORE): 1,      # 17% overhead — mild constraint
    (PIPE_STORE, PIPE_ALU): 1,
    (PIPE_FAST_INT, PIPE_STORE): 0,  # IADD3 is free everywhere
    (PIPE_STORE, PIPE_FAST_INT): 0,
}


def get_pipe(mnemonic):
    """Get execution pipe for an instruction."""
    family = insn_family(mnemonic)
    if family is None:
        return PIPE_NONE
    return _PIPE_MAP.get(family, PIPE_ALU)


def get_throughput(mnemonic):
    """Get per-instruction throughput (min cycles between same-type issues)."""
    family = insn_family(mnemonic)
    if family == 'STS':
        return 32
    if family == 'STG':
        return 26
    if family == 'STAS':
        return 32
    pipe = get_pipe(mnemonic)
    return _PIPE_THROUGHPUT.get(pipe, 2)


def schedule_cpsat(instructions, time_limit=60.0, verbose=True):
    """Find optimal instruction ordering using CP-SAT.

    instructions: list of Instruction objects (must have mnemonic/operands set).
    time_limit: solver time limit in seconds.
    verbose: print progress and solution details.

    Returns (ordered_addrs, stall_counts, stats) where:
      - ordered_addrs: list of addresses in optimal order
      - stall_counts: dict addr → stall count for each instruction
      - stats: dict with solver statistics
    Returns (None, None, stats) if no solution found.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        print('ERROR: ortools not installed. pip install ortools')
        return None, None, {'status': 'NO_ORTOOLS'}

    N = len(instructions)
    if N == 0:
        return [], {}, {'status': 'EMPTY'}

    # --- Parse register defs/uses ---
    du = []
    for insn in instructions:
        d, u = parse_reg_operands(insn.mnemonic, insn.operands)
        du.append((d, u))

    # --- Build dependency edges ---
    # edges: list of (src_idx, dst_idx, dep_type, reg, latency)
    edges = []
    for i in range(N):
        defs_i, uses_i = du[i]
        mn_i = instructions[i].mnemonic or ''
        lat_i = get_latency(mn_i)

        for j in range(i + 1, N):
            defs_j, uses_j = du[j]

            # RAW: i defines R, j uses R
            raw = defs_i & uses_j
            if raw:
                edges.append((i, j, 'RAW', raw, lat_i))

            # WAW: i defines R, j defines R
            waw = defs_i & defs_j
            if waw:
                edges.append((i, j, 'WAW', waw, 0))

            # WAR: i uses R, j defines R
            war = uses_i & defs_j
            if war:
                edges.append((i, j, 'WAR', war, 0))

    # --- Identify barriers ---
    barrier_indices = set()
    for i, insn in enumerate(instructions):
        if is_barrier(insn.mnemonic):
            barrier_indices.add(i)

    # --- Pipe classification ---
    pipes = {}
    for i, insn in enumerate(instructions):
        pipe = get_pipe(insn.mnemonic)
        pipes.setdefault(pipe, []).append(i)

    if verbose:
        print('CP-SAT scheduler: %d instructions, %d dep edges, %d barriers' % (
            N, len(edges), len(barrier_indices)))
        for pipe_name in sorted(pipes.keys()):
            tp = _PIPE_THROUGHPUT.get(pipe_name, 2)
            print('  %s: %d insns (throughput=%d)' % (
                pipe_name, len(pipes[pipe_name]), tp))

    # --- Build CP-SAT model ---
    model = cp_model.CpModel()

    # Upper bound on makespan: worst case = all instructions serial at max latency
    max_horizon = N * 40

    # Position in output sequence: pos[i] ∈ [0, N-1], all different
    pos = [model.new_int_var(0, N - 1, 'pos_%d' % i) for i in range(N)]
    model.add_all_different(pos)

    # Issue cycle for each instruction
    time = [model.new_int_var(0, max_horizon, 'time_%d' % i) for i in range(N)]

    # --- Data dependency constraints ---
    for src, dst, dep_type, regs, latency in edges:
        # All dep types require original order preserved
        model.add(pos[src] < pos[dst])

        # RAW dependencies need latency gap
        if dep_type == 'RAW' and latency > 0:
            model.add(time[dst] >= time[src] + latency)

    # --- Barrier constraints ---
    # Barriers pin relative order with all other instructions
    for bi in barrier_indices:
        for i in range(N):
            if i == bi:
                continue
            orig_before = (i < bi)
            if orig_before:
                model.add(pos[i] < pos[bi])
            else:
                model.add(pos[i] > pos[bi])

    # --- Stall cap: adjacent instructions ≤ 15 cycles apart ---
    # We model this efficiently using circuit/sequence constraints.
    # For each pair (i, j), if pos[j] = pos[i] + 1 (j is right after i),
    # then time[j] - time[i] ∈ [1, 15].
    #
    # Instead of O(N²) boolean indicators, we use a "next" formulation:
    # For each position p, exactly one instruction is at that position.
    # We use element constraints to link positions to times.

    # time_at_pos[p] = time of the instruction at position p
    time_at_pos = [model.new_int_var(0, max_horizon, 'tap_%d' % p) for p in range(N)]

    # Link: time_at_pos[pos[i]] == time[i]
    # Using element constraint: for each instruction i, time_at_pos[pos[i]] = time[i]
    for i in range(N):
        model.add_element(pos[i], time_at_pos, time[i])

    # Consecutive position constraint: monotonically increasing issue times.
    # No upper bound — the hardware can stall beyond 15 cycles via scoreboard
    # waits and pipe contention. The 15-cycle stall cap only limits what we
    # can encode; actual issue gaps may be larger (hardware blocks naturally).
    # We clamp to 15 when producing the stall output.
    for p in range(N - 1):
        model.add(time_at_pos[p + 1] >= time_at_pos[p] + 1)

    # --- Same-pipe throughput constraints ---
    # For instructions on the same pipe, enforce minimum gap.
    for pipe_name, indices in pipes.items():
        if pipe_name == PIPE_NONE:
            continue
        tp = _PIPE_THROUGHPUT.get(pipe_name, 2)
        if tp <= 1:
            continue

        # For each pair on this pipe, the one with smaller pos must have
        # enough time gap to the one with larger pos.
        for a_idx in range(len(indices)):
            for b_idx in range(a_idx + 1, len(indices)):
                i = indices[a_idx]
                j = indices[b_idx]
                # One of them comes first — use a boolean to model the disjunction
                b = model.new_bool_var('pipe_%s_%d_%d' % (pipe_name, i, j))
                # If b: pos[i] < pos[j] → time[j] >= time[i] + tp
                model.add(pos[i] < pos[j]).only_enforce_if(b)
                model.add(time[j] >= time[i] + tp).only_enforce_if(b)
                # If !b: pos[j] < pos[i] → time[i] >= time[j] + tp
                model.add(pos[j] < pos[i]).only_enforce_if(b.negated())
                model.add(time[i] >= time[j] + tp).only_enforce_if(b.negated())

    # --- Objective: minimize makespan ---
    makespan = model.new_int_var(0, max_horizon, 'makespan')
    model.add_max_equality(makespan, time)
    model.minimize(makespan)

    # --- Solve ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8

    if verbose:
        print('Solving (time limit %.0fs)...' % time_limit)

    status = solver.solve(model)
    status_name = solver.status_name(status)

    stats = {
        'status': status_name,
        'wall_time': solver.wall_time,
        'branches': solver.num_branches,
        'conflicts': solver.num_conflicts,
    }

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if verbose:
            print('No solution: %s (%.1fs)' % (status_name, solver.wall_time))
        return None, None, stats

    # --- Extract solution ---
    sol_pos = [(solver.value(pos[i]), i) for i in range(N)]
    sol_pos.sort()  # sort by position

    ordered_indices = [idx for _, idx in sol_pos]
    ordered_addrs = [instructions[idx].offset for idx in ordered_indices]

    # Compute stall counts from issue times
    stall_counts = {}
    for p in range(N):
        idx = ordered_indices[p]
        addr = instructions[idx].offset
        if p == 0:
            stall_counts[addr] = instructions[idx].control & 0xf  # keep original
        else:
            prev_idx = ordered_indices[p - 1]
            gap = solver.value(time[idx]) - solver.value(time[prev_idx])
            stall_counts[addr] = min(gap, MAX_STALL)

    obj_val = solver.objective_value
    stats['makespan'] = int(obj_val)

    if verbose:
        print('Solution: %s, makespan=%d cycles (%.1fs, %d branches, %d conflicts)' % (
            status_name, int(obj_val), solver.wall_time,
            solver.num_branches, solver.num_conflicts))

        # Print schedule
        print('\n%-4s  %-6s  %-4s  %-5s  %-8s  %s' % (
            'Pos', 'Addr', 'Cyc', 'Stall', 'Pipe', 'Instruction'))
        print('-' * 90)
        for p in range(N):
            idx = ordered_indices[p]
            insn = instructions[idx]
            t = solver.value(time[idx])
            stall = stall_counts[insn.offset]
            pipe = get_pipe(insn.mnemonic)
            mn = insn.mnemonic or 'NOP'
            ops = insn.operands or ''
            label = '%s %s' % (mn, ops)
            if len(label) > 55:
                label = label[:52] + '...'
            print('%4d  0x%04x  %4d  %5d  %-8s  %s' % (
                p, insn.offset, t, stall, pipe, label))

        # Summary by pipe
        print('\nPipe utilization:')
        pipe_cycles = {}
        for p in range(N):
            idx = ordered_indices[p]
            pipe = get_pipe(instructions[idx].mnemonic)
            t = solver.value(time[idx])
            pipe_cycles.setdefault(pipe, []).append(t)
        for pipe_name in sorted(pipe_cycles.keys()):
            cycles = sorted(pipe_cycles[pipe_name])
            if len(cycles) >= 2:
                span = cycles[-1] - cycles[0]
                avg_gap = span / (len(cycles) - 1) if len(cycles) > 1 else 0
                print('  %-10s: %d insns, span=%d cycles, avg gap=%.1f' % (
                    pipe_name, len(cycles), span, avg_gap))
            else:
                print('  %-10s: %d insns' % (pipe_name, len(cycles)))

    # --- Compare with original ---
    if verbose:
        orig_stalls = sum(insn.control & 0xf for insn in instructions)
        new_stalls = sum(stall_counts.values())
        print('\nOriginal total stalls: %d cycles' % orig_stalls)
        print('Optimized total stalls: %d cycles' % int(obj_val))
        delta = orig_stalls - int(obj_val)
        print('Delta: %d cycles (%+.1f%%)' % (
            delta, -100.0 * delta / orig_stalls if orig_stalls else 0))

    return ordered_addrs, stall_counts, stats


def schedule_to_recipe(instructions, ordered_addrs, stall_counts):
    """Convert CP-SAT schedule to an edit recipe (list of script command strings).

    The recipe uses reorder + stall patches to implement the schedule.
    """
    if not ordered_addrs:
        return []

    recipe = []
    recipe.append('# CP-SAT optimal schedule recipe')
    recipe.append('# Generated by sass_edit.py schedule command')
    recipe.append('# Instructions: %d' % len(ordered_addrs))
    recipe.append('')

    start_addr = min(insn.offset for insn in instructions)
    end_addr = max(insn.offset for insn in instructions) + INSN_SIZE

    # Check if reorder is needed (any instruction moved)
    orig_addrs = [insn.offset for insn in instructions]
    if ordered_addrs != orig_addrs:
        addr_str = ','.join('0x%04x' % a for a in ordered_addrs)
        recipe.append('reorder 0x%04x 0x%04x %s' % (start_addr, end_addr, addr_str))
        recipe.append('')

    # Stall patches
    recipe.append('# Stall count patches')
    for addr in ordered_addrs:
        stall = stall_counts.get(addr)
        if stall is not None:
            recipe.append('stall 0x%04x %d' % (addr, stall))

    recipe.append('')
    recipe.append('# Post-schedule audit')
    recipe.append('audit 0x%04x 0x%04x' % (start_addr, end_addr))

    return recipe


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# cuobjdump SASS cross-reference
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_sass_dump(sass_path, kernel_name_hint=None):
    """Parse cuobjdump --dump-sass output, return dict of {func_name: {addr: (opcode, operands)}}."""
    text = Path(sass_path).read_text()
    func_re = re.compile(r'Function\s*:\s*(\S+)')
    insn_re = re.compile(
        r'/\*([0-9a-fA-F]+)\*/\s+'
        r'(?:(@[!]?U?P\d+)\s+)?'
        r'(\S+?)\s+'
        r'(.*?)\s*;'
    )

    result = {}
    current_func = None

    for line in text.split('\n'):
        fm = func_re.search(line)
        if fm:
            current_func = fm.group(1)
            result[current_func] = {}
            continue

        if current_func is None:
            continue

        m = insn_re.search(line)
        if m:
            addr = int(m.group(1), 16)
            pred = m.group(2) or ''
            opcode = m.group(3)
            operands = m.group(4)
            if pred:
                opcode = pred + ' ' + opcode
            result[current_func][addr] = (opcode, operands)

    return result


def apply_sass_xref(kernel, sass_data):
    """Apply mnemonic annotations from SASS dump to kernel instructions."""
    # Find matching function in sass data
    best = None
    for func_name in sass_data:
        if func_name in kernel.name or kernel.name in func_name:
            best = func_name
            break

    if best is None:
        # Try partial match
        for func_name in sass_data:
            if kernel.short_name in func_name:
                best = func_name
                break

    if best is None:
        return 0

    addr_map = sass_data[best]
    count = 0
    for insn in kernel.instructions:
        if insn.offset in addr_map:
            insn.mnemonic, insn.operands = addr_map[insn.offset]
            count += 1

    return count


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CubinEditor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CubinEditor:
    def __init__(self, path):
        self.path = Path(path)
        self.data = bytearray(self.path.read_bytes())
        self.kernels = []
        self._parse_elf()
        self._modified = False

    def _parse_elf(self):
        f = io.BytesIO(bytes(self.data))
        elf = ELFFile(f)

        # Also grab ELF metadata we need for section manipulation
        self._ehdr_size = elf.header['e_ehsize']
        self._shoff = elf.header['e_shoff']
        self._shentsize = elf.header['e_shentsize']
        self._shnum = elf.header['e_shnum']

        for idx in range(elf.num_sections()):
            sec = elf.get_section(idx)
            name = sec.name
            if not name.startswith('.text.'):
                continue

            offset = sec.header['sh_offset']
            size = sec.header['sh_size']
            if size == 0:
                continue

            kernel_name = name[6:]
            k = Kernel(kernel_name, name, idx, offset, size)
            k.parse_instructions(self.data[offset:offset + size])
            self.kernels.append(k)

    def find_kernel(self, name=None):
        if not self.kernels:
            return None
        if name is None:
            return max(self.kernels, key=lambda k: k.size)
        for k in self.kernels:
            if name in k.name or name == k.short_name:
                return k
        # Try case-insensitive partial
        name_lower = name.lower()
        for k in self.kernels:
            if name_lower in k.name.lower():
                return k
        return None

    def xref(self, sass_path, kernel=None):
        sass_data = parse_sass_dump(sass_path)
        if kernel:
            return apply_sass_xref(kernel, sass_data)
        total = 0
        for k in self.kernels:
            total += apply_sass_xref(k, sass_data)
        return total

    def swap(self, kernel, addr_a, addr_b, force=False, no_restall=False):
        """Swap two instructions (full 128-bit swap including control words)."""
        idx_a = addr_a // INSN_SIZE
        idx_b = addr_b // INSN_SIZE
        instrs = kernel.instructions

        if not (0 <= idx_a < len(instrs) and 0 <= idx_b < len(instrs)):
            raise ValueError('Address out of range: 0x%x or 0x%x' % (addr_a, addr_b))

        # Dep check: treat as reorder of the range [min, max+1) with swapped positions
        lo, hi = sorted([idx_a, idx_b])
        region = instrs[lo:hi + 1]
        has_mnemonics = region[0].mnemonic is not None
        if has_mnemonics:
            addrs_orig = [insn.offset for insn in region]
            addrs_new = list(addrs_orig)
            ia = idx_a - lo
            ib = idx_b - lo
            addrs_new[ia], addrs_new[ib] = addrs_new[ib], addrs_new[ia]
            violations = check_deps(region, addrs_new)
            if violations:
                print('WARNING: %d dependency violation(s) in swap:' % len(violations))
                for v in violations:
                    print('  %s' % v)
                if not force:
                    raise ValueError(
                        'Swap blocked by dependency violations (use --force to override)')

        a, b = instrs[idx_a], instrs[idx_b]
        a.encoding, b.encoding = b.encoding, a.encoding
        a.control, b.control = b.control, a.control
        a.mnemonic, b.mnemonic = b.mnemonic, a.mnemonic
        a.operands, b.operands = b.operands, a.operands
        self._modified = True

        # auto-restall the affected region
        if not no_restall and has_mnemonics:
            region = instrs[lo:hi + 1]
            changes = compute_stalls(region, keep_first=True)
            if changes:
                apply_stall_changes(region, changes)
                for ch in changes:
                    print('  restall [%04x] %d -> %d  (%s)' % (
                        ch.addr, ch.old_stall, ch.new_stall, ch.reason))

    def reorder(self, kernel, start_addr, end_addr, new_order, force=False,
                no_restall=False):
        """Reorder instructions in [start, end) to the sequence given by new_order (list of addrs)."""
        start_idx = start_addr // INSN_SIZE
        end_idx = end_addr // INSN_SIZE
        instrs = kernel.instructions

        if end_idx > len(instrs):
            raise ValueError('End address 0x%x out of range (max 0x%x)' % (
                end_addr, len(instrs) * INSN_SIZE))

        count = end_idx - start_idx
        if len(new_order) != count:
            raise ValueError('new_order has %d entries, need %d' % (len(new_order), count))

        valid = set(range(start_addr, end_addr, INSN_SIZE))
        for addr in new_order:
            if addr not in valid:
                raise ValueError('Address 0x%x not in range [0x%x, 0x%x)' % (
                    addr, start_addr, end_addr))
        if len(set(new_order)) != count:
            raise ValueError('Duplicate addresses in new_order')

        # Dep check
        region = instrs[start_idx:end_idx]
        has_mnemonics = region and region[0].mnemonic is not None
        if has_mnemonics:
            violations = check_deps(region, new_order)
            if violations:
                print('WARNING: %d dependency violation(s) in reorder:' % len(violations))
                for v in violations:
                    print('  %s' % v)
                if not force:
                    raise ValueError(
                        'Reorder blocked by dependency violations (use --force to override)')

        # Snapshot originals
        orig = {a: instrs[a // INSN_SIZE].clone() for a in new_order}

        # Write in new order
        for i, addr in enumerate(new_order):
            src = orig[addr]
            dst = instrs[start_idx + i]
            dst.encoding = src.encoding
            dst.control = src.control
            dst.mnemonic = src.mnemonic
            dst.operands = src.operands

        self._modified = True

        # auto-restall the reordered region
        if not no_restall and has_mnemonics:
            region = instrs[start_idx:end_idx]
            changes = compute_stalls(region, keep_first=True)
            if changes:
                apply_stall_changes(region, changes)
                for ch in changes:
                    print('  restall [%04x] %d -> %d  (%s)' % (
                        ch.addr, ch.old_stall, ch.new_stall, ch.reason))

    def patch_ctrl_field(self, kernel, addr, **fields):
        """Modify specific control word fields at addr."""
        idx = addr // INSN_SIZE
        insn = kernel.instructions[idx]
        current = decode_ctrl(insn.control)
        for key, val in fields.items():
            if key not in current:
                raise ValueError('Unknown field: %s' % key)
            current[key] = val
        insn.control = encode_ctrl(current, insn.control)
        self._modified = True

    def patch_ctrl_raw(self, kernel, addr, raw_ctrl):
        """Replace entire control word at addr."""
        idx = addr // INSN_SIZE
        kernel.instructions[idx].control = raw_ctrl
        self._modified = True

    def patch_stall(self, kernel, addr, stall):
        """Set stall count at addr (most common operation)."""
        idx = addr // INSN_SIZE
        insn = kernel.instructions[idx]
        insn.control = (insn.control & ~0xf) | (stall & 0xf)
        self._modified = True

    def patch_reg(self, kernel, addr, field_name, new_reg_num):
        """Patch a register field in an instruction's encoding.

        field_name: 'dst', 'src1', 'src2', 'addr', 'data'
        new_reg_num: 0-255 (physical register number)
        """
        insn = kernel.insn_at(addr)
        if insn is None:
            raise ValueError('No instruction at 0x%x' % addr)

        fields = get_reg_fields(insn.mnemonic)
        for fname, text_idx, bit_off, width in fields:
            if fname == field_name:
                old_val = read_enc_field(insn.encoding, bit_off, width)
                insn.encoding = write_enc_field(
                    insn.encoding, bit_off, width, new_reg_num)
                self._modified = True

                # Update text operands to reflect the change
                if insn.operands:
                    insn.operands = insn.operands.replace(
                        'R%d' % old_val, 'R%d' % new_reg_num, 1)

                return old_val

        known = [f[0] for f in fields]
        raise ValueError('Field "%s" not found for %s (known: %s)' % (
            field_name, insn.mnemonic or '?', ', '.join(known) if known else 'none'))

    def copy_insn(self, kernel, src_addr, dst_addr, copy_ctrl=False):
        """Copy instruction encoding from src to dst.

        By default copies only the encoding word (preserving dst's control word).
        Set copy_ctrl=True to also copy the control word.
        """
        src = kernel.insn_at(src_addr)
        dst = kernel.insn_at(dst_addr)
        if src is None:
            raise ValueError('No instruction at src 0x%x' % src_addr)
        if dst is None:
            raise ValueError('No instruction at dst 0x%x' % dst_addr)

        dst.encoding = src.encoding
        dst.mnemonic = src.mnemonic
        dst.operands = src.operands
        if copy_ctrl:
            dst.control = src.control
        self._modified = True

    def nop_insn(self, kernel, addr):
        """Replace instruction at addr with a NOP (preserves control word)."""
        insn = kernel.insn_at(addr)
        if insn is None:
            raise ValueError('No instruction at 0x%x' % addr)
        insn.encoding = NOP_ENCODING
        insn.mnemonic = 'NOP'
        insn.operands = ''
        self._modified = True

    def save(self, output_path):
        """Write modified cubin. Never overwrites the original."""
        out_path = Path(output_path)
        if out_path.resolve() == self.path.resolve():
            raise ValueError('Cannot overwrite source cubin. Use a different output path.')

        out = bytearray(self.data)
        for k in self.kernels:
            new_bytes = k.to_bytes()
            if len(new_bytes) != k.size:
                raise ValueError(
                    'Kernel %s: instruction bytes (%d) != section size (%d). '
                    'Instruction count changed — not supported.' % (
                        k.short_name, len(new_bytes), k.size))
            out[k.file_offset:k.file_offset + k.size] = new_bytes

        out_path.write_bytes(bytes(out))
        return out_path

    def verify(self, sass_path, kernel=None):
        """Cross-reference binary with SASS dump to validate byte-level encoding match."""
        sass_data = parse_sass_dump(sass_path)
        k = kernel or self.find_kernel()
        if not k:
            print('No kernel found')
            return False

        # Parse the raw hex values from the SASS dump too
        text = Path(sass_path).read_text()
        hex_re = re.compile(r'/\*\s*0x([0-9a-fA-F]+)\s*\*/')
        func_re = re.compile(r'Function\s*:\s*(\S+)')

        in_target = False
        lines = text.split('\n')
        sass_insns = []  # (addr, encoding, control)

        i = 0
        while i < len(lines):
            line = lines[i]
            fm = func_re.search(line)
            if fm:
                fname = fm.group(1)
                in_target = (fname in k.name or k.name in fname)
                i += 1
                continue

            if not in_target:
                i += 1
                continue

            # Try to parse instruction line (has addr + ;)
            addr_m = re.search(r'/\*([0-9a-fA-F]+)\*/', line)
            if addr_m and ';' in line:
                addr = int(addr_m.group(1), 16)
                hexvals = hex_re.findall(line)
                enc = int(hexvals[-1], 16) if hexvals else 0

                # Next line: control word
                if i + 1 < len(lines):
                    ctrl_hexvals = hex_re.findall(lines[i + 1])
                    ctrl = int(ctrl_hexvals[0], 16) if ctrl_hexvals else 0
                    sass_insns.append((addr, enc, ctrl))
                    i += 2
                    continue

            i += 1

        # Compare
        mismatches = 0
        matched = 0
        for addr, sass_enc, sass_ctrl in sass_insns:
            idx = addr // INSN_SIZE
            if idx >= len(k.instructions):
                print('  WARN: SASS addr 0x%04x beyond binary (has %d insns)' % (
                    addr, len(k.instructions)))
                continue

            bin_insn = k.instructions[idx]
            enc_ok = bin_insn.encoding == sass_enc
            ctrl_ok = bin_insn.control == sass_ctrl

            if not enc_ok or not ctrl_ok:
                mismatches += 1
                if mismatches <= 20:
                    print('  MISMATCH at 0x%04x:' % addr)
                    if not enc_ok:
                        print('    enc:  bin=0x%016x  sass=0x%016x' % (bin_insn.encoding, sass_enc))
                    if not ctrl_ok:
                        print('    ctrl: bin=0x%016x  sass=0x%016x' % (bin_insn.control, sass_ctrl))
            else:
                matched += 1

        total = len(sass_insns)
        print('Verified %d/%d instructions: %d match, %d mismatch' % (
            total, k.n_insns, matched, mismatches))
        return mismatches == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Script parser — batch edit commands
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_script(script_path, editor, kernel):
    """Parse and execute a batch edit script.

    Script format (one command per line, # comments):
        swap 0xADDR_A 0xADDR_B
        stall 0xADDR VALUE
        ctrl 0xADDR 0xRAW_CTRL_HEX
        reorder 0xSTART 0xEND 0xA,0xB,0xC,...
    """
    lines = Path(script_path).read_text().strip().split('\n')
    n_ops = 0

    for lineno, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        cmd = parts[0].lower()

        try:
            if cmd == 'swap':
                a, b = int(parts[1], 0), int(parts[2], 0)
                editor.swap(kernel, a, b)
                n_ops += 1

            elif cmd == 'stall':
                addr, val = int(parts[1], 0), int(parts[2], 0)
                editor.patch_stall(kernel, addr, val)
                n_ops += 1

            elif cmd == 'ctrl':
                addr, raw = int(parts[1], 0), int(parts[2], 0)
                editor.patch_ctrl_raw(kernel, addr, raw)
                n_ops += 1

            elif cmd == 'reorder':
                start = int(parts[1], 0)
                end = int(parts[2], 0)
                addrs = [int(x, 0) for x in parts[3].split(',')]
                editor.reorder(kernel, start, end, addrs)
                n_ops += 1

            elif cmd == 'restall':
                start = int(parts[1], 0)
                end = int(parts[2], 0)
                start_idx = start // INSN_SIZE
                end_idx = end // INSN_SIZE
                region = kernel.instructions[start_idx:end_idx]
                changes = compute_stalls(region, keep_first=True)
                if changes:
                    apply_stall_changes(region, changes)
                    for ch in changes:
                        print('  restall [%04x] %d -> %d  (%s)' % (
                            ch.addr, ch.old_stall, ch.new_stall, ch.reason))
                n_ops += 1

            elif cmd == 'audit':
                start = int(parts[1], 0)
                end = int(parts[2], 0)
                start_idx = start // INSN_SIZE
                end_idx = end // INSN_SIZE
                region = kernel.instructions[start_idx:end_idx]
                warnings = audit_stalls(region)
                if warnings:
                    for w in warnings:
                        print('  STALL WARNING: [%04x] %s needs %d cycles after '
                              '[%04x] %s (via %s), has %d' % (
                              w.addr, w.mnemonic, w.needed,
                              w.producer_addr, w.producer_mnemonic,
                              w.reg, w.actual))
                else:
                    print('  audit [0x%x, 0x%x): all stalls OK' % (start, end))

            elif cmd == 'patch-reg':
                # patch-reg 0xADDR FIELD_NAME REG_NUM
                addr = int(parts[1], 0)
                field = parts[2]
                reg_num = int(parts[3], 0)
                old = editor.patch_reg(kernel, addr, field, reg_num)
                print('  patch-reg [%04x] %s: R%d -> R%d' % (
                    addr, field, old, reg_num))
                n_ops += 1

            elif cmd == 'copy':
                # copy 0xSRC 0xDST [ctrl]
                src = int(parts[1], 0)
                dst = int(parts[2], 0)
                copy_ctrl = len(parts) > 3 and parts[3].lower() == 'ctrl'
                editor.copy_insn(kernel, src, dst, copy_ctrl=copy_ctrl)
                print('  copy [%04x] -> [%04x]%s' % (
                    src, dst, ' +ctrl' if copy_ctrl else ''))
                n_ops += 1

            elif cmd == 'nop':
                # nop 0xADDR
                addr = int(parts[1], 0)
                editor.nop_insn(kernel, addr)
                print('  nop [%04x]' % addr)
                n_ops += 1

            else:
                print('WARN: unknown command "%s" at line %d' % (cmd, lineno))

        except Exception as e:
            print('ERROR at line %d: %s' % (lineno, e))
            raise

    return n_ops


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI commands
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_info(args):
    ed = CubinEditor(args.cubin)
    print('Cubin: %s (%d bytes)' % (args.cubin, len(ed.data)))
    print('Kernels: %d' % len(ed.kernels))
    print()
    for k in ed.kernels:
        print('  %-40s  %6d insns  %6d bytes  file_off=0x%x' % (
            k.short_name[:40], k.n_insns, k.size, k.file_offset))


def cmd_dump(args):
    ed = CubinEditor(args.cubin)
    k = ed.find_kernel(args.kernel)
    if not k:
        print('Kernel not found: %s' % args.kernel)
        print('Available: %s' % ', '.join(kk.short_name for kk in ed.kernels))
        sys.exit(1)

    if args.sass:
        n = ed.xref(args.sass, k)
        sys.stderr.write('Cross-referenced %d/%d instructions\n' % (n, k.n_insns))

    start = int(args.start, 0) if args.start else None
    end = int(args.end, 0) if args.end else None

    print('Kernel: %s  (%d insns, %d bytes)' % (k.short_name[:60], k.n_insns, k.size))
    print('=' * 120)

    for insn in k.instructions:
        if start is not None and insn.offset < start:
            continue
        if end is not None and insn.offset >= end:
            break

        stall = insn.control & 0xf
        stall_bar = '#' * stall if stall else '.'

        if insn.mnemonic:
            mnem_str = insn.mnemonic
            if insn.operands:
                mnem_str += ' ' + insn.operands[:50]
            print('[%04x] st=%2d %-15s  enc=%016x ctrl=%016x  %-60s' % (
                insn.offset, stall, stall_bar, insn.encoding, insn.control,
                mnem_str[:60]))
        else:
            print('[%04x] st=%2d %-15s  enc=%016x ctrl=%016x' % (
                insn.offset, stall, stall_bar, insn.encoding, insn.control))


def cmd_swap(args):
    ed = CubinEditor(args.cubin)
    k = ed.find_kernel(args.kernel)
    if not k:
        print('Kernel not found')
        sys.exit(1)

    addr_a = int(args.addr_a, 0)
    addr_b = int(args.addr_b, 0)

    if args.sass:
        ed.xref(args.sass, k)

    insn_a = k.insn_at(addr_a)
    insn_b = k.insn_at(addr_b)
    print('Swapping:')
    print('  [%04x] %s' % (addr_a, insn_a.mnemonic or ('enc=0x%016x' % insn_a.encoding)))
    print('  [%04x] %s' % (addr_b, insn_b.mnemonic or ('enc=0x%016x' % insn_b.encoding)))

    ed.swap(k, addr_a, addr_b,
            force=getattr(args, 'force', False),
            no_restall=getattr(args, 'no_restall', False))
    out = ed.save(args.output)
    print('Saved: %s' % out)


def cmd_reorder(args):
    ed = CubinEditor(args.cubin)
    k = ed.find_kernel(args.kernel)
    if not k:
        print('Kernel not found')
        sys.exit(1)

    start = int(args.start, 0)
    end = int(args.end, 0)
    new_order = [int(x, 0) for x in args.order.split(',')]

    if args.sass:
        ed.xref(args.sass, k)

    print('Reordering [0x%x, 0x%x): %d instructions' % (start, end, len(new_order)))
    for i, addr in enumerate(new_order):
        insn = k.insn_at(addr)
        tag = insn.mnemonic or ('enc=0x%016x' % insn.encoding)
        print('  %d: [%04x] %s' % (i, addr, tag))

    ed.reorder(k, start, end, new_order,
               force=getattr(args, 'force', False),
               no_restall=getattr(args, 'no_restall', False))
    out = ed.save(args.output)
    print('Saved: %s' % out)


def cmd_patch(args):
    ed = CubinEditor(args.cubin)
    k = ed.find_kernel(args.kernel)
    if not k:
        print('Kernel not found')
        sys.exit(1)

    addr = int(args.addr, 0)

    if args.raw_ctrl is not None:
        raw = int(args.raw_ctrl, 0)
        ed.patch_ctrl_raw(k, addr, raw)
        print('Patched [%04x] ctrl = 0x%016x' % (addr, raw))
    else:
        fields = {}
        if args.stall is not None:
            fields['stall'] = args.stall
        if args.yield_hint is not None:
            fields['yield'] = args.yield_hint
        if args.wr_bar is not None:
            fields['wr_bar'] = args.wr_bar
        if args.rd_bar is not None:
            fields['rd_bar'] = args.rd_bar
        if args.wait_mask is not None:
            fields['wait_mask'] = int(args.wait_mask, 0)
        if args.reuse is not None:
            fields['reuse'] = args.reuse

        if not fields:
            print('No fields specified')
            sys.exit(1)

        ed.patch_ctrl_field(k, addr, **fields)
        insn = k.insn_at(addr)
        print('Patched [%04x] ctrl = 0x%016x  (%s)' % (addr, insn.control, ctrl_str(insn.control)))

    out = ed.save(args.output)
    print('Saved: %s' % out)


def cmd_script(args):
    ed = CubinEditor(args.cubin)
    k = ed.find_kernel(args.kernel)
    if not k:
        print('Kernel not found')
        sys.exit(1)

    if args.sass:
        ed.xref(args.sass, k)

    n = parse_script(args.script, ed, k)
    print('Executed %d operations' % n)

    out = ed.save(args.output)
    print('Saved: %s' % out)


def cmd_verify(args):
    ed = CubinEditor(args.cubin)
    k = ed.find_kernel(args.kernel)
    if not k:
        print('Kernel not found')
        sys.exit(1)

    ok = ed.verify(args.sass, k)
    sys.exit(0 if ok else 1)


def cmd_sass(args):
    """Run cuobjdump on the cubin and optionally filter by kernel/address range."""
    import subprocess
    result = subprocess.run(
        ['cuobjdump', '--dump-sass', args.cubin],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print('cuobjdump failed: %s' % result.stderr.strip())
        sys.exit(1)

    # If no filter, print everything
    if not args.start and not args.end and not args.kernel:
        print(result.stdout, end='')
        return

    # Filter by kernel and/or address range
    func_re = re.compile(r'Function\s*:\s*(\S+)')
    insn_re = re.compile(r'/\*([0-9a-fA-F]+)\*/.*?;')  # instruction line has ; after addr
    ctrl_re = re.compile(r'^\s+/\*\s*0x[0-9a-fA-F]+\s*\*/\s*$')  # control-word-only line
    in_target = args.kernel is None
    start = int(args.start, 0) if args.start else None
    end = int(args.end, 0) if args.end else None
    show_next_ctrl = False

    for line in result.stdout.split('\n'):
        fm = func_re.search(line)
        if fm:
            fname = fm.group(1)
            if args.kernel:
                in_target = args.kernel in fname
            if in_target:
                print(line)
            show_next_ctrl = False
            continue

        if not in_target:
            continue

        # Control word line (follows instruction line)
        if ctrl_re.match(line):
            if show_next_ctrl:
                print(line)
            show_next_ctrl = False
            continue

        # Instruction line
        im = insn_re.search(line)
        if im:
            addr = int(im.group(1), 16)
            if (start is not None and addr < start) or (end is not None and addr >= end):
                show_next_ctrl = False
                continue
            print(line)
            show_next_ctrl = True
            continue

        # Non-instruction, non-control lines (headers, blank lines)
        if start is None and end is None:
            print(line)


def cmd_gen_loader(args):
    """Generate a minimal CUDA driver API loader for running patched cubins."""
    cubin_path = args.cubin
    ed = CubinEditor(cubin_path)
    k = ed.find_kernel(args.kernel)
    if not k:
        print('Kernel not found')
        sys.exit(1)

    # The loader needs to know the kernel's mangled name for cuModuleGetFunction
    mangled = k.name

    loader_src = '''\
/*
 Generated by sass_edit.py gen-loader
 Loads a patched cubin via CUDA driver API and launches the kernel.

 Build:  nvcc -O2 -std=c++17 -lcuda loader.cu -o loader
 Usage:  ./loader [patched.cubin]
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>

#define CU_CHECK(x) do {{ \\
    CUresult r = (x); \\
    if (r != CUDA_SUCCESS) {{ \\
        const char* es; cuGetErrorString(r, &es); \\
        fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, es); \\
        exit(1); \\
    }} \\
}} while(0)

int main(int argc, char** argv) {{
    const char* cubin_path = argc > 1 ? argv[1] : "{cubin_default}";

    CU_CHECK(cuInit(0));

    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, 0));

    CUcontext ctx;
    CU_CHECK(cuCtxCreate(&ctx, 0, dev));

    CUmodule mod;
    CUresult r = cuModuleLoad(&mod, cubin_path);
    if (r != CUDA_SUCCESS) {{
        const char* es;
        cuGetErrorString(r, &es);
        fprintf(stderr, "Failed to load cubin '%s': %s\\n", cubin_path, es);
        fprintf(stderr, "This cubin was compiled for sm_100a — requires B200 GPU\\n");
        return 1;
    }}

    CUfunction func;
    r = cuModuleGetFunction(&func, mod, "{mangled}");
    if (r != CUDA_SUCCESS) {{
        const char* es;
        cuGetErrorString(r, &es);
        fprintf(stderr, "Failed to find kernel: %s\\n", es);
        return 1;
    }}

    printf("Loaded cubin: %s\\n", cubin_path);
    printf("Kernel: {short_name}\\n");

    /* TODO: set up kernel arguments matching fc2.cu's main() */
    /* The kernel signature and launch config must match the original. */
    /* Copy the argument setup from fc2.cu's main() function. */
    printf("Kernel loaded successfully. Add launch code for your specific kernel.\\n");

    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    return 0;
}}
'''.format(
        cubin_default=cubin_path,
        mangled=mangled,
        short_name=k.short_name[:40]
    )

    out_path = Path(args.output)
    out_path.write_text(loader_src)
    print('Generated loader: %s' % out_path)
    print('Build: nvcc -O2 -std=c++17 -lcuda %s -o loader' % out_path)
    print('Usage: ./loader [patched.cubin]')


def cmd_diff(args):
    ed_a = CubinEditor(args.cubin_a)
    ed_b = CubinEditor(args.cubin_b)

    k_a = ed_a.find_kernel(args.kernel)
    k_b = ed_b.find_kernel(args.kernel)

    if not k_a or not k_b:
        print('Kernel not found in one or both cubins')
        sys.exit(1)

    if args.sass:
        ed_a.xref(args.sass, k_a)
        ed_b.xref(args.sass, k_b)

    print('Diff: %s vs %s' % (args.cubin_a, args.cubin_b))
    print('Kernel: %s' % k_a.short_name[:60])
    print('  A: %d insns, B: %d insns' % (k_a.n_insns, k_b.n_insns))

    max_insns = max(k_a.n_insns, k_b.n_insns)
    diffs = 0

    for i in range(max_insns):
        if i >= k_a.n_insns:
            print('  [%04x] A: <end>  B: enc=%016x ctrl=%016x' % (
                i * INSN_SIZE, k_b.instructions[i].encoding, k_b.instructions[i].control))
            diffs += 1
            continue
        if i >= k_b.n_insns:
            print('  [%04x] A: enc=%016x ctrl=%016x  B: <end>' % (
                i * INSN_SIZE, k_a.instructions[i].encoding, k_a.instructions[i].control))
            diffs += 1
            continue

        a = k_a.instructions[i]
        b = k_b.instructions[i]

        if a.encoding != b.encoding or a.control != b.control:
            diffs += 1
            mnem_a = a.mnemonic or '?'
            mnem_b = b.mnemonic or '?'

            enc_diff = '  enc' if a.encoding != b.encoding else '     '
            ctrl_diff = '  ctrl' if a.control != b.control else '      '

            stall_a = a.control & 0xf
            stall_b = b.control & 0xf
            stall_str = ''
            if stall_a != stall_b:
                stall_str = '  st: %d->%d' % (stall_a, stall_b)

            print('  [%04x]%s%s  %-20s -> %-20s%s' % (
                i * INSN_SIZE, enc_diff, ctrl_diff,
                mnem_a[:20], mnem_b[:20], stall_str))

    print('\n%d differences in %d instructions' % (diffs, max_insns))


def cmd_deps(args):
    """Show register def/use analysis and check dependencies in an address range."""
    ed = CubinEditor(args.cubin)
    k = ed.find_kernel(args.kernel)
    if not k:
        print('Kernel not found')
        sys.exit(1)

    if args.sass:
        n = ed.xref(args.sass, k)
        sys.stderr.write('Cross-referenced %d/%d instructions\n' % (n, k.n_insns))
    else:
        sys.stderr.write('WARNING: no --sass provided, mnemonics unavailable. '
                         'Dep analysis requires SASS cross-reference.\n')
        sys.exit(1)

    start = int(args.start, 0)
    end = int(args.end, 0)
    start_idx = start // INSN_SIZE
    end_idx = end // INSN_SIZE

    if end_idx > len(k.instructions):
        print('End address 0x%x out of range' % end)
        sys.exit(1)

    region = k.instructions[start_idx:end_idx]

    print('Def/use analysis [0x%x, 0x%x): %d instructions' % (start, end, len(region)))
    print('=' * 100)
    dump_deps(region)

    # Check current ordering for any issues
    violations = check_deps(region)
    if violations:
        print('\nDependency issues in current order: %d' % len(violations))
        for v in violations:
            print('  %s' % v)
    else:
        print('\nNo dependency issues in current order.')

    # If a proposed reorder is given via --reorder, check that too
    if args.reorder:
        new_order = [int(x, 0) for x in args.reorder.split(',')]
        print('\nProposed reorder: %s' % ', '.join('0x%x' % a for a in new_order))
        violations = check_deps(region, new_order)
        if violations:
            print('VIOLATIONS: %d' % len(violations))
            for v in violations:
                print('  %s' % v)
        else:
            print('OK: no dependency violations in proposed reorder.')

    # stall audit
    print('\nStall audit [0x%x, 0x%x):' % (start, end))
    warnings = audit_stalls(region)
    if warnings:
        for w in warnings:
            print('  WARNING: [%04x] %s needs %d cycles after [%04x] %s (via %s), has %d' % (
                w.addr, w.mnemonic, w.needed,
                w.producer_addr, w.producer_mnemonic, w.reg, w.actual))
    else:
        print('  All stalls OK.')


def cmd_probe_encoding(args):
    """Verify register field positions against SASS text."""
    ed = CubinEditor(args.cubin)
    k = ed.find_kernel(args.kernel)
    if not k:
        print('Kernel not found')
        sys.exit(1)

    n = ed.xref(args.sass, k)
    sys.stderr.write('Cross-referenced %d/%d instructions\n' % (n, k.n_insns))

    results = probe_reg_fields(k)

    print('Encoding field verification:')
    print('%-12s  %6s  %6s  %s' % ('Family', 'Match', 'Mismatch', 'Status'))
    print('-' * 50)

    total_match = 0
    total_mismatch = 0
    for family in sorted(results.keys()):
        r = results[family]
        total_match += r['matches']
        total_mismatch += r['mismatches']
        status = 'OK' if r['mismatches'] == 0 else 'FAIL'
        print('%-12s  %6d  %6d  %s' % (
            family, r['matches'], r['mismatches'], status))

    print('-' * 50)
    print('%-12s  %6d  %6d' % ('TOTAL', total_match, total_mismatch))

    if total_mismatch > 0:
        print('\nMismatches:')
        for family in sorted(results.keys()):
            for detail in results[family]['details']:
                print('  %s' % detail)


def cmd_patch_reg(args):
    """Patch a register field in an instruction."""
    ed = CubinEditor(args.cubin)
    k = ed.find_kernel(args.kernel)
    if not k:
        print('Kernel not found')
        sys.exit(1)

    if args.sass:
        ed.xref(args.sass, k)

    addr = int(args.addr, 0)
    reg_num = int(args.reg, 0)
    old = ed.patch_reg(k, addr, args.field, reg_num)

    insn = k.insn_at(addr)
    print('[%04x] %s %s: R%d -> R%d' % (
        addr, insn.mnemonic or '?', args.field, old, reg_num))

    ed.save(args.output)
    print('Saved to %s' % args.output)


def cmd_copy_insn(args):
    """Copy instruction encoding from one address to another."""
    ed = CubinEditor(args.cubin)
    k = ed.find_kernel(args.kernel)
    if not k:
        print('Kernel not found')
        sys.exit(1)

    if args.sass:
        ed.xref(args.sass, k)

    src = int(args.src, 0)
    dst = int(args.dst, 0)
    ed.copy_insn(k, src, dst, copy_ctrl=args.copy_ctrl)

    src_insn = k.insn_at(src)
    print('Copied [%04x] %s -> [%04x]' % (
        src, src_insn.mnemonic or '?', dst))

    ed.save(args.output)
    print('Saved to %s' % args.output)


def cmd_pipeline(args):
    """Analyze epilogue for software pipelining opportunities."""
    ed = CubinEditor(args.cubin)
    k = ed.find_kernel(args.kernel)
    if not k:
        print('Kernel not found')
        sys.exit(1)

    n = ed.xref(args.sass, k)
    sys.stderr.write('Cross-referenced %d/%d instructions\n' % (n, k.n_insns))

    start = int(args.start, 0)
    end = int(args.end, 0)
    start_idx = start // INSN_SIZE
    end_idx = end // INSN_SIZE
    region = k.instructions[start_idx:end_idx]

    # Identify groups
    groups = identify_epilogue_groups(region)
    if not groups:
        print('No store clusters found in [0x%x, 0x%x)' % (start, end))
        sys.exit(1)

    print('Epilogue groups: %d' % len(groups))
    print('=' * 80)

    for g in groups:
        cs, ce = g.compute_range
        ss, se = g.store_range
        print('Group %d: compute [0x%04x-0x%04x] (%d insns), '
              'store [0x%04x-0x%04x] (%d insns)' % (
                  g.group_idx, cs, ce, len(g.compute_insns),
                  ss, se, len(g.store_insns)))

        # Show store data vectors
        for base, width in g.store_data_vectors():
            regs = ', '.join('R%d' % (base + i) for i in range(width))
            print('  STS data: R%d × %d = [%s]' % (base, width, regs))

    # Interleave analysis
    analyses = analyze_interleave(groups)
    if analyses:
        print('\nInterleave analysis:')
        print('-' * 80)

        total_conflicts = 0
        total_spare = 0

        for a in analyses:
            gi, gi1 = a['group_n'], a['group_n1']
            nc = a['gpr_remap_needed']
            total_conflicts += len(a['conflicts'])
            total_spare += nc

            print('Groups %d→%d: %d stores, %d compute ops to interleave' % (
                gi, gi1, a['n_store_count'], a['n1_compute_count']))

            if a['conflicts']:
                print('  Conflicts: %s' % ', '.join(sorted(a['conflicts'])))
                if a['expanded'] != a['conflicts']:
                    print('  Expanded (vector): %s' % ', '.join(
                        sorted(a['expanded'])))
                print('  Spare GPRs needed: %d' % nc)

                # Show proposed remap
                remap = plan_remap(a['expanded'], args.spare_start)
                for old_name, new_num in sorted(remap.items(),
                                                key=lambda x: int(x[0][1:])):
                    print('    %s -> R%d' % (old_name, new_num))
            else:
                print('  No register conflicts — free to interleave')

            # Estimate benefit
            n_stores = a['n_store_count']
            n_compute = a['n1_compute_count']
            sts_cycles = n_stores * 32
            compute_cycles = n_compute * 3  # ~3 cycles avg per compute op
            overlap = min(sts_cycles, compute_cycles)
            print('  Estimated overlap: %d cycles (STS=%d, compute=%d)' % (
                overlap, sts_cycles, compute_cycles))
            print()

        print('Summary: %d total conflicts, %d spare GPRs needed (R%d-R255 = %d available)' % (
            total_conflicts, total_spare, args.spare_start,
            256 - args.spare_start))

    # Find NOP slots
    nops = find_nops(k)
    nops_in_range = [(addr, ctrl) for addr, ctrl in nops
                     if start <= addr < end]
    print('\nNOP slots in range: %d (of %d total in kernel)' % (
        len(nops_in_range), len(nops)))
    for addr, ctrl in nops_in_range[:20]:
        print('  0x%04x  ctrl=%s' % (addr, ctrl_str(ctrl)))
    if len(nops_in_range) > 20:
        print('  ... and %d more' % (len(nops_in_range) - 20))

    # Find donors
    if args.donors:
        donors = find_donors(k)
        print('\nDonor instructions available:')
        for family in sorted(donors.keys()):
            count = len(donors[family])
            if count > 0:
                first = donors[family][0]
                print('  %-10s: %d instances (e.g., [%04x] %s)' % (
                    family, count, first[0], first[1][:50]))

    # Generate recipe
    if args.generate:
        recipe = generate_interleave_recipe(groups, analyses, args.spare_start)
        out_path = Path(args.generate)
        out_path.write_text('\n'.join(recipe) + '\n')
        print('\nRecipe written to %s (%d lines)' % (out_path, len(recipe)))


def cmd_schedule(args):
    """Run CP-SAT optimal scheduler on a SASS region."""
    ed = CubinEditor(args.cubin)
    k = ed.find_kernel(args.kernel)
    if not k:
        print('Kernel not found')
        sys.exit(1)

    n = ed.xref(args.sass, k)
    sys.stderr.write('Cross-referenced %d/%d instructions\n' % (n, k.n_insns))

    start = int(args.start, 0)
    end = int(args.end, 0)
    start_idx = start // INSN_SIZE
    end_idx = end // INSN_SIZE

    if end_idx > len(k.instructions):
        print('End address 0x%x out of range' % end)
        sys.exit(1)

    region = k.instructions[start_idx:end_idx]

    # Filter out unannotated instructions (no mnemonic from SASS xref)
    annotated = [insn for insn in region if insn.mnemonic]
    if len(annotated) < len(region):
        sys.stderr.write('WARNING: %d/%d instructions have no mnemonic annotation\n' % (
            len(region) - len(annotated), len(region)))

    ordered_addrs, stall_counts, stats = schedule_cpsat(
        region,
        time_limit=args.time_limit,
        verbose=not args.quiet)

    if ordered_addrs is None:
        print('No solution found: %s' % stats.get('status', '?'))
        sys.exit(1)

    # Generate recipe
    if args.recipe:
        recipe = schedule_to_recipe(region, ordered_addrs, stall_counts)
        Path(args.recipe).write_text('\n'.join(recipe) + '\n')
        print('\nRecipe written to %s (%d lines)' % (args.recipe, len(recipe)))

    # Apply directly
    if args.output:
        # Apply reorder
        orig_addrs = [insn.offset for insn in region]
        if ordered_addrs != orig_addrs:
            ed.reorder(k, start, end, ordered_addrs, force=True)

        # Apply stall counts
        for addr, stall in stall_counts.items():
            ed.patch_stall(k, addr, stall)

        ed.save(args.output)
        print('Patched cubin written to %s' % args.output)


def cmd_find_donors(args):
    """Find donor instructions for each opcode family."""
    ed = CubinEditor(args.cubin)
    k = ed.find_kernel(args.kernel)
    if not k:
        print('Kernel not found')
        sys.exit(1)

    ed.xref(args.sass, k)
    donors = find_donors(k)

    family_filter = args.family.upper() if args.family else None

    for family in sorted(donors.keys()):
        if family_filter and family != family_filter:
            continue
        entries = donors[family]
        print('%s (%d):' % (family, len(entries)))
        limit = 5 if not family_filter else len(entries)
        for addr, mn, ops, enc in entries[:limit]:
            # Show register field values
            fields = FAMILY_REG_FIELDS.get(family, _DEFAULT_REG_FIELDS)
            field_strs = []
            for fname, _, bit_off, width in fields:
                val = read_enc_field(enc, bit_off, width)
                field_strs.append('%s=R%d' % (fname, val))
            field_info = ', '.join(field_strs) if field_strs else '-'
            print('  [%04x] %-40s  enc=0x%016x  %s' % (
                addr, ('%s %s' % (mn, ops))[:40], enc, field_info))
        if len(entries) > limit:
            print('  ... and %d more' % (len(entries) - limit))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fatbin patcher — patch embedded cubin inside compiled CUDA host binary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ELF_MAGIC = b'\x7fELF'


def find_cubin_in_fatbin(host_data):
    """Find embedded cubin(s) inside a compiled CUDA binary's .nv_fatbin section.

    Parses the host ELF to locate .nv_fatbin, then scans for embedded ELF
    images that contain .text. sections (cubins have kernel code, PTX ELFs don't).

    Returns list of (offset_in_host, size) sorted by size descending (largest first).
    """
    f = io.BytesIO(bytes(host_data))
    elf = ELFFile(f)

    # find .nv_fatbin section
    fatbin_sec = None
    for idx in range(elf.num_sections()):
        sec = elf.get_section(idx)
        if sec.name == '.nv_fatbin':
            fatbin_sec = sec
            break

    if fatbin_sec is None:
        raise ValueError('No .nv_fatbin section found in host binary')

    fb_offset = fatbin_sec.header['sh_offset']
    fb_size = fatbin_sec.header['sh_size']

    # scan for ELF magic within the fatbin section
    cubins = []
    pos = 0
    while pos < fb_size - 4:
        idx = host_data.find(ELF_MAGIC, fb_offset + pos, fb_offset + fb_size)
        if idx < 0:
            break
        pos = idx - fb_offset + 1

        # parse candidate ELF to get its size and check for .text. sections
        try:
            candidate = io.BytesIO(bytes(host_data[idx:fb_offset + fb_size]))
            celf = ELFFile(candidate)

            # total size = max of section end, segment end, shdr table end
            elf_size = celf.header['e_ehsize']

            # section headers table end
            if celf.header['e_shoff'] > 0:
                sh_end = celf.header['e_shoff'] + celf.header['e_shentsize'] * celf.header['e_shnum']
                elf_size = max(elf_size, sh_end)

            # section data ends
            has_text = False
            for si in range(celf.num_sections()):
                s = celf.get_section(si)
                s_end = s.header['sh_offset'] + s.header['sh_size']
                elf_size = max(elf_size, s_end)
                if s.name.startswith('.text.'):
                    has_text = True

            # program header table end
            if celf.header['e_phoff'] > 0 and celf.header['e_phnum'] > 0:
                ph_end = celf.header['e_phoff'] + celf.header['e_phentsize'] * celf.header['e_phnum']
                elf_size = max(elf_size, ph_end)

            # segment data ends
            for seg in celf.iter_segments():
                seg_end = seg.header['p_offset'] + seg.header['p_filesz']
                elf_size = max(elf_size, seg_end)

            if has_text:
                cubins.append((idx, elf_size))

        except Exception:
            continue

    # sort largest first
    cubins.sort(key=lambda x: -x[1])
    return cubins


def cmd_fatbin_patch(args):
    """Patch embedded cubin inside a compiled CUDA binary."""
    host_path = Path(args.binary)
    host_data = bytearray(host_path.read_bytes())

    cubins = find_cubin_in_fatbin(host_data)
    if not cubins:
        print('No embedded cubins found in %s' % host_path)
        sys.exit(1)

    print('Found %d embedded cubin(s):' % len(cubins))
    for i, (off, sz) in enumerate(cubins):
        print('  [%d] offset=0x%x  size=%d bytes' % (i, off, sz))

    ci = args.cubin_index
    if ci >= len(cubins):
        print('Cubin index %d out of range (have %d)' % (ci, len(cubins)))
        sys.exit(1)

    cubin_offset, cubin_size = cubins[ci]
    cubin_bytes = bytes(host_data[cubin_offset:cubin_offset + cubin_size])

    # write to temp file so CubinEditor can parse it
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.cubin')
    os.close(tmp_fd)
    tmp = Path(tmp_path)
    tmp.write_bytes(cubin_bytes)

    try:
        ed = CubinEditor(str(tmp))
        k = ed.find_kernel(args.kernel)
        if not k:
            print('Kernel not found')
            if ed.kernels:
                print('Available: %s' % ', '.join(kk.short_name for kk in ed.kernels))
            sys.exit(1)

        print('Kernel: %s (%d insns)' % (k.short_name[:60], k.n_insns))

        if args.sass:
            n = ed.xref(args.sass, k)
            sys.stderr.write('Cross-referenced %d/%d instructions\n' % (n, k.n_insns))

        n_ops = 0
        if args.script:
            n_ops += parse_script(args.script, ed, k)
        if args.stall:
            addr, val = int(args.stall[0], 0), int(args.stall[1], 0)
            ed.patch_stall(k, addr, val)
            n_ops += 1

        if n_ops == 0:
            print('No edits specified (use --script or --stall)')
            sys.exit(1)

        # save patched cubin to another temp file
        tmp2_fd, tmp2_path = tempfile.mkstemp(suffix='.cubin')
        os.close(tmp2_fd)
        tmp2 = Path(tmp2_path)
        ed.save(str(tmp2))
        patched_cubin = tmp2.read_bytes()
        tmp2.unlink()
    finally:
        tmp.unlink()

    if len(patched_cubin) != cubin_size:
        print('ERROR: cubin size changed (%d -> %d) — not supported' % (
            cubin_size, len(patched_cubin)))
        sys.exit(1)

    # splice patched cubin back into host binary
    host_data[cubin_offset:cubin_offset + cubin_size] = patched_cubin

    out_path = Path(args.output)
    out_path.write_bytes(bytes(host_data))
    os.chmod(str(out_path), 0o755)

    print('Applied %d edit(s). Patched %s -> %s (cubin at 0x%x, %d bytes)' % (
        n_ops, host_path, out_path, cubin_offset, cubin_size))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_addr(s):
    """Parse hex or decimal address."""
    return int(s, 0)


def main():
    p = argparse.ArgumentParser(
        description='SM100a SASS binary editor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    sub = p.add_subparsers(dest='cmd')

    # info
    s = sub.add_parser('info', help='List kernels in cubin')
    s.add_argument('cubin')

    # dump
    s = sub.add_parser('dump', help='Print instructions with control word decode')
    s.add_argument('cubin')
    s.add_argument('--kernel', '-k', default=None)
    s.add_argument('--start', '-s', default=None, help='Start address (hex)')
    s.add_argument('--end', '-e', default=None, help='End address (hex)')
    s.add_argument('--sass', default=None, help='cuobjdump SASS dump for mnemonic xref')

    # swap
    s = sub.add_parser('swap', help='Swap two instructions')
    s.add_argument('cubin')
    s.add_argument('addr_a', help='First address (hex)')
    s.add_argument('addr_b', help='Second address (hex)')
    s.add_argument('-o', '--output', required=True)
    s.add_argument('--kernel', '-k', default=None)
    s.add_argument('--sass', default=None)
    s.add_argument('--force', '-f', action='store_true',
                   help='Override dependency violation checks')
    s.add_argument('--no-restall', action='store_true',
                   help='Skip automatic stall recomputation after edit')

    # reorder
    s = sub.add_parser('reorder', help='Reorder instructions in a range')
    s.add_argument('cubin')
    s.add_argument('start', help='Start address (hex)')
    s.add_argument('end', help='End address (hex)')
    s.add_argument('order', help='Comma-separated new address order')
    s.add_argument('-o', '--output', required=True)
    s.add_argument('--kernel', '-k', default=None)
    s.add_argument('--sass', default=None)
    s.add_argument('--force', '-f', action='store_true',
                   help='Override dependency violation checks')
    s.add_argument('--no-restall', action='store_true',
                   help='Skip automatic stall recomputation after edit')

    # patch
    s = sub.add_parser('patch', help='Modify control word fields')
    s.add_argument('cubin')
    s.add_argument('addr', help='Instruction address (hex)')
    s.add_argument('-o', '--output', required=True)
    s.add_argument('--kernel', '-k', default=None)
    s.add_argument('--stall', type=int, default=None)
    s.add_argument('--yield-hint', type=int, default=None)
    s.add_argument('--wr-bar', type=int, default=None)
    s.add_argument('--rd-bar', type=int, default=None)
    s.add_argument('--wait-mask', default=None, help='Hex wait mask')
    s.add_argument('--reuse', type=int, default=None)
    s.add_argument('--raw-ctrl', default=None, help='Replace entire control word (hex)')

    # script
    s = sub.add_parser('script', help='Apply batch edits from script file')
    s.add_argument('cubin')
    s.add_argument('script', help='Script file path')
    s.add_argument('-o', '--output', required=True)
    s.add_argument('--kernel', '-k', default=None)
    s.add_argument('--sass', default=None)

    # verify
    s = sub.add_parser('verify', help='Verify binary matches SASS dump')
    s.add_argument('cubin')
    s.add_argument('--sass', required=True)
    s.add_argument('--kernel', '-k', default=None)

    # sass (cuobjdump wrapper)
    s = sub.add_parser('sass', help='Dump SASS via cuobjdump (with optional filter)')
    s.add_argument('cubin')
    s.add_argument('--kernel', '-k', default=None)
    s.add_argument('--start', '-s', default=None, help='Start address (hex)')
    s.add_argument('--end', '-e', default=None, help='End address (hex)')

    # gen-loader
    s = sub.add_parser('gen-loader', help='Generate CUDA driver API loader for patched cubins')
    s.add_argument('cubin')
    s.add_argument('-o', '--output', default='loader.cu')
    s.add_argument('--kernel', '-k', default=None)

    # diff
    s = sub.add_parser('diff', help='Compare two cubins')
    s.add_argument('cubin_a')
    s.add_argument('cubin_b')
    s.add_argument('--kernel', '-k', default=None)
    s.add_argument('--sass', default=None)

    # deps
    s = sub.add_parser('deps', help='Show register def/use analysis for an address range')
    s.add_argument('cubin')
    s.add_argument('--start', '-s', required=True, help='Start address (hex)')
    s.add_argument('--end', '-e', required=True, help='End address (hex)')
    s.add_argument('--sass', required=True, help='cuobjdump SASS dump for mnemonic xref')
    s.add_argument('--kernel', '-k', default=None)
    s.add_argument('--reorder', default=None,
                   help='Comma-separated proposed address order to check')

    # probe-encoding
    s = sub.add_parser('probe-encoding',
                       help='Verify register field positions against SASS text')
    s.add_argument('cubin')
    s.add_argument('--sass', required=True)
    s.add_argument('--kernel', '-k', default=None)

    # patch-reg
    s = sub.add_parser('patch-reg', help='Patch a register field in an instruction')
    s.add_argument('cubin')
    s.add_argument('addr', help='Instruction address (hex)')
    s.add_argument('field', help='Field name: dst, src1, src2, addr, data')
    s.add_argument('reg', help='New register number (0-255)')
    s.add_argument('-o', '--output', required=True)
    s.add_argument('--kernel', '-k', default=None)
    s.add_argument('--sass', default=None)

    # copy-insn
    s = sub.add_parser('copy-insn', help='Copy instruction encoding to another address')
    s.add_argument('cubin')
    s.add_argument('src', help='Source address (hex)')
    s.add_argument('dst', help='Destination address (hex)')
    s.add_argument('-o', '--output', required=True)
    s.add_argument('--kernel', '-k', default=None)
    s.add_argument('--sass', default=None)
    s.add_argument('--copy-ctrl', action='store_true',
                   help='Also copy control word (default: preserve destination ctrl)')

    # pipeline
    s = sub.add_parser('pipeline',
                       help='Analyze epilogue for software pipelining')
    s.add_argument('cubin')
    s.add_argument('--sass', required=True)
    s.add_argument('--start', '-s', required=True, help='Epilogue start address (hex)')
    s.add_argument('--end', '-e', required=True, help='Epilogue end address (hex)')
    s.add_argument('--kernel', '-k', default=None)
    s.add_argument('--spare-start', type=int, default=208,
                   help='First spare register number (default: 208)')
    s.add_argument('--donors', action='store_true',
                   help='Show donor instructions for each family')
    s.add_argument('--generate', default=None, metavar='RECIPE_FILE',
                   help='Generate interleave recipe script')

    # schedule (CP-SAT optimal scheduler)
    s = sub.add_parser('schedule',
                       help='CP-SAT optimal instruction scheduler')
    s.add_argument('cubin')
    s.add_argument('--sass', required=True)
    s.add_argument('--start', '-s', required=True, help='Region start address (hex)')
    s.add_argument('--end', '-e', required=True, help='Region end address (hex)')
    s.add_argument('--kernel', '-k', default=None)
    s.add_argument('--recipe', default=None, metavar='RECIPE_FILE',
                   help='Write edit recipe to file')
    s.add_argument('-o', '--output', default=None,
                   help='Apply schedule and write patched cubin')
    s.add_argument('--time-limit', type=float, default=60.0,
                   help='Solver time limit in seconds (default: 60)')
    s.add_argument('--quiet', '-q', action='store_true',
                   help='Suppress verbose output')

    # find-donors
    s = sub.add_parser('find-donors', help='Find donor instructions for each family')
    s.add_argument('cubin')
    s.add_argument('--sass', required=True)
    s.add_argument('--kernel', '-k', default=None)
    s.add_argument('--family', '-f', default=None,
                   help='Filter to specific family (e.g., HADD2)')

    # fatbin-patch
    s = sub.add_parser('fatbin-patch', help='Patch embedded cubin in compiled CUDA binary')
    s.add_argument('binary', help='Host binary (e.g., fc2)')
    s.add_argument('--sass', required=True, help='SASS dump for cross-reference')
    s.add_argument('--script', default=None, help='Edit script file')
    s.add_argument('--stall', nargs=2, default=None, metavar=('ADDR', 'VAL'),
                   help='Patch single stall count')
    s.add_argument('-o', '--output', required=True, help='Output patched binary')
    s.add_argument('--kernel', '-k', default=None)
    s.add_argument('--cubin-index', type=int, default=0, help='Which cubin (0=largest)')
    s.add_argument('--force', '-f', action='store_true')

    args = p.parse_args()

    if not args.cmd:
        p.print_help()
        sys.exit(1)

    cmds = {
        'info': cmd_info,
        'dump': cmd_dump,
        'swap': cmd_swap,
        'reorder': cmd_reorder,
        'patch': cmd_patch,
        'script': cmd_script,
        'verify': cmd_verify,
        'sass': cmd_sass,
        'gen-loader': cmd_gen_loader,
        'diff': cmd_diff,
        'deps': cmd_deps,
        'probe-encoding': cmd_probe_encoding,
        'patch-reg': cmd_patch_reg,
        'copy-insn': cmd_copy_insn,
        'pipeline': cmd_pipeline,
        'find-donors': cmd_find_donors,
        'schedule': cmd_schedule,
        'fatbin-patch': cmd_fatbin_patch,
    }
    cmds[args.cmd](args)


if __name__ == '__main__':
    main()
