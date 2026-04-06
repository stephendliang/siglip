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
    """Decode control word into field dict.

    SM100a layout (128-bit instructions, empirically verified):
      bits [7:0]    — extended operand (register for HFMA2/IMAD/LEA/LOP3, etc.)
      bits [52:8]   — unknown (instruction-format-dependent)
      bits [55:53]  — stall count (3 bits, range 0-7)
      bits [63:56]  — flags (bit 56 = .reuse in some formats)
    Only the stall field is verified. Other bit assignments are placeholders.
    """
    return {
        'stall':     (ctrl >> 53) & 0x7,
    }


def encode_ctrl(fields, original_ctrl):
    """Encode control word fields, preserving all bits except stall."""
    mask = 0x7 << 53  # bits [55:53]
    val = (fields['stall'] & 0x7) << 53
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
MAX_STALL = 7  # SM100a: 3-bit stall field at bits 53-55, range 0-7

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
    parts = ['st=%d' % c['stall']]
    parts.append('ctrl=%016x' % ctrl_val)
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
    'LDTM':  [('dst', 0, 16, 8), ('src_ur', 1, 32, 8)],
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
        return (self.control >> 53) & 0x7

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
                    covered = sum(instructions[k].stall for k in range(j + 1, i))
                    needed = lat - covered
                    if needed > max_needed:
                        max_needed = needed
                        worst_producer = j
                        worst_reg = reg
                    break  # found nearest producer for this reg

        if max_needed <= 0:
            continue

        new_stall = min(max_needed, MAX_STALL)
        old_stall = insn.stall
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
        mask = 0x7 << 53
        insn.control = (insn.control & ~mask) | ((ch.new_stall & 0x7) << 53)
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
                    covered = sum(instructions[k].stall for k in range(j + 1, i))
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
# Opcode table extraction (Phase 1 of SASS assembler)
#
# XOR analysis: collect multiple instances of the same instruction (with
# different register operands), XOR encodings pairwise. Bits that differ
# are operand fields; bits that stay constant are the opcode.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _xor_analysis(encodings):
    """Compute opcode and var_mask from a set of encodings via XOR.

    Returns (opcode, var_mask, verified).
    """
    unique = list(set(encodings))
    if len(unique) < 2:
        return unique[0], 0, True

    var_mask = 0
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            var_mask |= (unique[i] ^ unique[j])

    opcode = unique[0] & ~var_mask
    verified = all((enc & ~var_mask) == opcode for enc in unique)
    return opcode, var_mask, verified


def compute_opcode_table(kernels):
    """Extract opcode table from cross-referenced kernel instructions.

    Groups instructions by mnemonic (sans predicate guard), XORs encodings
    to separate fixed opcode bits from variable operand bits.

    When a mnemonic has multiple encoding forms (register vs immediate —
    detected when XOR analysis gives var_mask = all-1s), automatically
    sub-groups by bits [7:0] and produces separate entries per form.

    Returns dict: mnemonic → {
        'opcode': int,       # fixed bits
        'var_mask': int,     # bits that vary across instances (operand fields)
        'n_instances': int,
        'n_unique': int,     # unique encodings seen
        'verified': bool,    # all instances consistent
        'single': bool,      # only one unique encoding (var_mask incomplete)
        'form': int or None, # low8 form identifier if sub-grouped
    }
    """
    by_mnemonic = collections.defaultdict(list)
    for k in kernels:
        for insn in k.instructions:
            if not insn.mnemonic:
                continue
            _, bare = _strip_pred(insn.mnemonic)
            if not bare:
                continue
            by_mnemonic[bare].append(insn.encoding)

    table = {}
    for mnem, encodings in sorted(by_mnemonic.items()):
        unique = list(set(encodings))
        n_inst = len(encodings)

        if len(unique) < 2:
            table[mnem] = {
                'opcode': unique[0], 'var_mask': 0,
                'n_instances': n_inst, 'n_unique': 1,
                'verified': True, 'single': True, 'form': None,
            }
            continue

        # Always sub-group by low 8 bits to separate encoding forms
        # (e.g., register vs immediate source use different opcodes).
        by_low8 = collections.defaultdict(list)
        for enc in encodings:
            by_low8[enc & 0xFF].append(enc)

        if len(by_low8) == 1:
            # Single encoding form — use as-is
            opcode, var_mask, verified = _xor_analysis(encodings)
            table[mnem] = {
                'opcode': opcode, 'var_mask': var_mask,
                'n_instances': n_inst, 'n_unique': len(unique),
                'verified': verified, 'single': False, 'form': None,
            }
            continue

        # Multiple encoding forms — produce per-form entries.
        # Only suppress forms with < 2 unique encodings AND < 5 instances
        # (likely xref noise, not real forms).
        forms = sorted(by_low8.keys(), key=lambda k: -len(by_low8[k]))
        for low8 in forms:
            form_encs = by_low8[low8]
            form_unique = list(set(form_encs))
            form_key = '%s{%02x}' % (mnem, low8)

            if len(form_unique) < 2 and len(form_encs) < 5:
                continue

            if len(form_unique) < 2:
                table[form_key] = {
                    'opcode': form_unique[0], 'var_mask': 0,
                    'n_instances': len(form_encs), 'n_unique': 1,
                    'verified': True, 'single': True, 'form': low8,
                }
                continue

            fop, fvm, fv = _xor_analysis(form_encs)
            table[form_key] = {
                'opcode': fop, 'var_mask': fvm,
                'n_instances': len(form_encs), 'n_unique': len(form_unique),
                'verified': fv, 'single': False, 'form': low8,
            }

    return table


def format_opcode_table(table, show_all=False):
    """Format opcode table for display.

    Groups by instruction family. Shows opcode bits, variable mask,
    and flags single-instance entries that need more data.
    """
    lines = []
    prev_family = None
    n_verified = sum(1 for v in table.values() if v['verified'] and not v.get('single'))
    n_single = sum(1 for v in table.values() if v.get('single'))
    n_fail = sum(1 for v in table.values() if not v['verified'])
    lines.append('Opcode table: %d verified, %d single-instance, %d FAILED' % (
        n_verified, n_single, n_fail))
    lines.append('')
    lines.append('%-40s  %-18s  %-18s  %5s  %5s  %s' % (
        'Mnemonic', 'Opcode', 'VarMask', 'Inst', 'Uniq', 'Status'))
    lines.append('-' * 110)

    for mnem in sorted(table.keys()):
        entry = table[mnem]
        if not show_all and entry.get('single'):
            continue
        family = mnem.split('.')[0]
        if family != prev_family:
            if prev_family is not None:
                lines.append('')
            prev_family = family

        status = 'OK'
        if entry.get('single'):
            status = 'SINGLE'
        elif not entry['verified']:
            status = 'FAIL'

        lines.append('%-40s  0x%016x  0x%016x  %5d  %5d  %s' % (
            mnem, entry['opcode'], entry['var_mask'],
            entry['n_instances'], entry['n_unique'], status))

    if n_single > 0:
        lines.append('')
        lines.append('Single-instance mnemonics (need more data for var_mask):')
        for mnem in sorted(table.keys()):
            if table[mnem].get('single'):
                lines.append('  %-40s  enc=0x%016x  (%d instances, all identical)' % (
                    mnem, table[mnem]['opcode'], table[mnem]['n_instances']))

    return '\n'.join(lines)


def export_opcode_table(table, path):
    """Write opcode table as importable Python module."""
    lines = [
        '# SM100a opcode table — auto-generated by sass_edit.py opcode-table',
        '# Maps (mnemonic_with_modifiers) → (opcode_bits, var_mask)',
        '#',
        '# opcode_bits: fixed bits identifying the instruction + modifiers',
        '# var_mask: bits that encode operands (registers, immediates, etc.)',
        '# For any instance: (encoding & ~var_mask) == opcode_bits',
        '',
        'OPCODE_TABLE = {',
    ]
    for mnem in sorted(table.keys()):
        e = table[mnem]
        flag = ''
        if e.get('single'):
            flag = '  # single instance — var_mask incomplete'
        elif not e['verified']:
            flag = '  # VERIFICATION FAILED'
        lines.append("    %-44s (0x%016x, 0x%016x),%s" % (
            "'%s':" % mnem, e['opcode'], e['var_mask'], flag))
    lines.append('}')
    lines.append('')

    Path(path).write_text('\n'.join(lines))
    return len(table)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2+3: Operand Encoding + Control Word Analysis
#
# Phase 2.3: Predicate encoding — guard enable, register, negation bits
# Phase 2.2: Immediate fields — memory offsets, branch targets, SR IDs
# Phase 2.1: Register field verification against Phase 1 opcode table
# Phase 2.4/2.5: Uniform registers, RZ/PT, .reuse flag
# Phase 3.1: Control word catalog — per-family defaults, varying bits
# Phase 3.4-3.5: Barrier field identification via producer enrichment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_RE_MEM_OFFSET = re.compile(r'\[([^\]]*?)\+(0x[0-9a-fA-F]+|\d+)\]')
_RE_BARE_HEX = re.compile(r'(?<![RU\w])0x([0-9a-fA-F]+)\b')


def _normalize_ops(operands):
    """Normalize operand text for grouping (strip .reuse, collapse whitespace)."""
    return ' '.join(operands.replace('.reuse', '').split())


def _parse_guard(guard_str):
    """Parse '@P0', '@!P1', '@UP2' → (negated, uniform, reg_num) or None."""
    if not guard_str:
        return None
    m = re.match(r'@(!?)(U?)P(\d)', guard_str)
    return (m.group(1) == '!', m.group(2) == 'U', int(m.group(3))) if m else None


def _bit_positions(mask):
    """Sorted list of set bit positions in a 64-bit mask."""
    return [b for b in range(64) if mask & (1 << b)]


def _majority_mask(diffs, threshold=0.7):
    """Bits appearing in >= threshold fraction of diff values."""
    if not diffs:
        return 0
    counts = [0] * 64
    for d in diffs:
        for b in range(64):
            if d & (1 << b):
                counts[b] += 1
    n = len(diffs)
    return sum((1 << b) for b in range(64) if counts[b] >= n * threshold)


def _contiguous_fields(mask):
    """Decompose bit mask into contiguous (start_bit, width) fields."""
    fields = []
    b = 0
    while b < 64:
        if mask & (1 << b):
            start = b
            while b < 64 and (mask & (1 << b)):
                b += 1
            fields.append((start, b - start))
        else:
            b += 1
    return fields


def _popcount64(x):
    """Count set bits in a 64-bit integer."""
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f
    return ((x * 0x0101010101010101) & 0xFFFFFFFFFFFFFFFF) >> 56


def analyze_predicates(kernels):
    """Phase 2.3: Extract predicate guard encoding via exact-pair XOR.

    Groups instructions by (bare_mnemonic, normalized_operands, encoding_form).
    XORs pairs that differ only in predicate guard to isolate predicate bits.
    Filters by popcount (true predicate diffs should be <= 10 bits).
    Excludes BRA/BSSY (PC-dependent operands contaminate the XOR).
    """
    _skip_families = {'BRA', 'BSSY', 'CALL', 'RET'}

    by_key = collections.defaultdict(list)
    for k in kernels:
        for insn in k.instructions:
            if not insn.mnemonic or not insn.operands:
                continue
            guard, bare = _strip_pred(insn.mnemonic)
            family = bare.split('.')[0]
            if family in _skip_families:
                continue
            form = insn.encoding & 0xFF
            key = (bare, _normalize_ops(insn.operands), form)
            by_key[key].append((guard, insn.encoding, insn.control))

    enable_enc = []
    enable_ctrl = []
    reg_enc = []
    neg_enc = []
    max_popcount = 10

    for (bare_mn, ops, form), instances in by_key.items():
        by_guard = {}
        for g, enc, ctrl in instances:
            gk = g or ''
            if gk not in by_guard:
                by_guard[gk] = (enc, ctrl)

        if len(by_guard) < 2:
            continue

        guards = list(by_guard.keys())
        for i in range(len(guards)):
            for j in range(i + 1, len(guards)):
                ga, gb = guards[i], guards[j]
                ea, ca = by_guard[ga]
                eb, cb = by_guard[gb]
                ed = ea ^ eb
                cd = ca ^ cb

                if _popcount64(ed) + _popcount64(cd) > max_popcount:
                    continue

                pa = _parse_guard(ga) if ga else None
                pb = _parse_guard(gb) if gb else None

                if (pa is None) != (pb is None):
                    enable_enc.append(ed)
                    enable_ctrl.append(cd)
                elif pa and pb:
                    if pa[2] != pb[2] and pa[0] == pb[0] and pa[1] == pb[1]:
                        reg_enc.append(ed)
                    elif pa[2] == pb[2] and pa[0] != pb[0] and pa[1] == pb[1]:
                        neg_enc.append(ed)

    enc_enable = _majority_mask(enable_enc, 0.6)
    ctrl_enable = _majority_mask(enable_ctrl, 0.6)
    enc_reg = _majority_mask(reg_enc, 0.5) if reg_enc else 0
    enc_neg = (_majority_mask(neg_enc, 0.5) if neg_enc else 0) & ~enc_reg
    enc_enable_only = enc_enable & ~enc_reg & ~enc_neg

    # Merge adjacent enable+reg bits into a single predicate register field.
    # "Enable" bits are typically the MSB of the register field (PT=7 disables guard).
    all_pred_enc = enc_enable | enc_reg | enc_neg
    all_pred_bits = _bit_positions(all_pred_enc)

    # Identify predicate field bounds
    pred_field_start = min(all_pred_bits) if all_pred_bits else 0
    pred_field_width = (max(all_pred_bits) - min(all_pred_bits) + 1) if all_pred_bits else 0
    neg_bit_pos = _bit_positions(enc_neg)[0] if enc_neg else None

    # If the "enable" bits are adjacent to register bits, they form the reg field MSB
    reg_full = enc_reg | enc_enable_only
    reg_full_bits = _bit_positions(reg_full)

    result = {
        'enc_mask': enc_enable, 'ctrl_mask': ctrl_enable,
        'reg_mask': enc_reg, 'neg_mask': enc_neg, 'enable_mask': enc_enable_only,
        'n_enable': len(enable_enc), 'n_reg': len(reg_enc), 'n_neg': len(neg_enc),
        'pred_field_start': pred_field_start,
        'pred_field_width': pred_field_width,
    }

    # Full register field = reg_mask | enable_mask (enable is reg MSB)
    if reg_full_bits:
        result['reg_full_mask'] = reg_full
        result['reg_full_bits'] = reg_full_bits
        result['reg_full_field'] = (min(reg_full_bits), len(reg_full_bits))
    if neg_bit_pos is not None:
        result['neg_bit'] = neg_bit_pos

    for name, mask in [('reg', enc_reg), ('neg', enc_neg), ('enable', enc_enable_only)]:
        if mask:
            result[name + '_bits'] = _bit_positions(mask)
            fields = _contiguous_fields(mask)
            if fields:
                result[name + '_field'] = fields[0]

    if ctrl_enable:
        result['ctrl_pred_bits'] = _bit_positions(ctrl_enable)

    # Verify: extract actual encoding values per guard type
    if all_pred_enc:
        guard_values = collections.defaultdict(lambda: collections.Counter())
        for k in kernels:
            for insn in k.instructions:
                if not insn.mnemonic:
                    continue
                guard, bare = _strip_pred(insn.mnemonic)
                family = bare.split('.')[0]
                if family in _skip_families:
                    continue
                gkey = guard or 'PT'
                fval = (insn.encoding >> pred_field_start) & ((1 << pred_field_width) - 1)
                guard_values[gkey][fval] += 1

        guard_encoding = {}
        for gkey in sorted(guard_values.keys()):
            dominant_val, dominant_count = guard_values[gkey].most_common(1)[0]
            total = sum(guard_values[gkey].values())
            guard_encoding[gkey] = (dominant_val, dominant_count, total)
        result['guard_encoding'] = guard_encoding

    return result


def analyze_immediates(kernels):
    """Phase 2.2: Extract immediate field positions via XOR analysis.

    Identifies encoding bits that carry immediate values for:
    - Memory operations (STS, LDS, LDG, STG offsets)
    - Branches (BRA, BSSY PC-relative targets)
    - Special registers (S2R SR IDs)
    - Barrier/dependency (DEPBAR, BAR operands)
    """
    results = {}

    # --- Memory offsets: group by (mnemonic, ops_template), vary offset ---
    mem_groups = collections.defaultdict(list)
    for k in kernels:
        for insn in k.instructions:
            if not insn.mnemonic or not insn.operands:
                continue
            _, bare = _strip_pred(insn.mnemonic)
            m = _RE_MEM_OFFSET.search(insn.operands)
            if not m:
                continue
            offset_val = int(m.group(2), 0)
            template = insn.operands[:m.start(2)] + '<OFF>' + insn.operands[m.end(2):]
            template = _normalize_ops(template)
            mem_groups[(bare, template)].append((offset_val, insn.encoding))

    for (mnem, template), instances in mem_groups.items():
        by_off = {}
        for off, enc in instances:
            if off not in by_off:
                by_off[off] = enc
        if len(by_off) < 2:
            continue

        encs = list(by_off.values())
        mask = 0
        for i in range(len(encs)):
            for j in range(i + 1, len(encs)):
                mask |= encs[i] ^ encs[j]
        if not mask:
            continue

        fields = _contiguous_fields(mask)
        scale = None
        if fields:
            start, width = fields[0]
            off_items = sorted(by_off.items())
            for s in [1, 2, 4, 8, 16]:
                if all(off % s == 0 and
                       (enc >> start) & ((1 << width) - 1) == off // s
                       for off, enc in off_items):
                    scale = s
                    break

        if mnem not in results:
            results[mnem] = {
                'type': 'mem_offset', 'mask': mask, 'fields': fields,
                'scale': scale, 'n_values': len(by_off),
                'examples': sorted(by_off.keys())[:8],
            }

    # --- Branch targets: per-kernel PC-relative analysis ---
    # Group by (mnemonic, guard, encoding_form) to avoid mixing encoding forms
    bra_accumulated = collections.defaultdict(lambda: {'mask': 0, 'n': 0})
    for k in kernels:
        bra_by_key = collections.defaultdict(list)
        for insn in k.instructions:
            if not insn.mnemonic or not insn.operands:
                continue
            guard, bare = _strip_pred(insn.mnemonic)
            family = bare.split('.')[0]
            if family not in ('BRA', 'BSSY'):
                continue
            m = _RE_BARE_HEX.search(insn.operands)
            if not m:
                continue
            target = int(m.group(1), 16)
            rel = target - insn.offset
            form = insn.encoding & 0xFF
            bra_by_key[(bare, guard or '', form)].append((rel, insn.encoding))

        for (mnem, guard, form), instances in bra_by_key.items():
            by_rel = {}
            for rel, enc in instances:
                if rel not in by_rel:
                    by_rel[rel] = enc
            if len(by_rel) < 2:
                continue
            encs = list(by_rel.values())
            mask = 0
            for i in range(len(encs)):
                for j in range(i + 1, len(encs)):
                    mask |= encs[i] ^ encs[j]
            if mask:
                bra_accumulated[mnem]['mask'] |= mask
                bra_accumulated[mnem]['n'] += len(by_rel)

    for mnem, info in bra_accumulated.items():
        if info['mask']:
            results[mnem] = {
                'type': 'branch_target', 'mask': info['mask'],
                'fields': _contiguous_fields(info['mask']),
                'pc_relative': True, 'n_values': info['n'],
            }

    # --- S2R special register field ---
    s2r_by_sr = collections.defaultdict(list)
    for k in kernels:
        for insn in k.instructions:
            if not insn.mnemonic or not insn.operands:
                continue
            _, bare = _strip_pred(insn.mnemonic)
            if not bare.startswith('S2R'):
                continue
            ops = _split_operands(insn.operands)
            if len(ops) >= 2:
                s2r_by_sr[ops[1].strip()].append(insn.encoding)

    if len(s2r_by_sr) >= 2:
        sr_encs = {name: encs[0] for name, encs in s2r_by_sr.items()}
        sr_list = list(sr_encs.values())
        mask = 0
        for i in range(len(sr_list)):
            for j in range(i + 1, len(sr_list)):
                mask |= sr_list[i] ^ sr_list[j]
        mask &= ~(0xFF << 16)  # remove dst register field
        if mask:
            results['S2R'] = {
                'type': 'sr_id', 'mask': mask,
                'fields': _contiguous_fields(mask),
                'sr_names': sorted(s2r_by_sr.keys()),
            }

    # --- Generic operand fields (DEPBAR, BAR) ---
    for target_fam in ['DEPBAR', 'BAR']:
        fam_groups = collections.defaultdict(list)
        for k in kernels:
            for insn in k.instructions:
                if not insn.mnemonic or not insn.operands:
                    continue
                _, bare = _strip_pred(insn.mnemonic)
                if bare.split('.')[0] != target_fam:
                    continue
                fam_groups[bare].append((insn.operands, insn.encoding))

        for mnem, instances in fam_groups.items():
            by_ops = {}
            for ops, enc in instances:
                if ops not in by_ops:
                    by_ops[ops] = enc
            if len(by_ops) < 2:
                continue
            encs = list(by_ops.values())
            mask = 0
            for i in range(len(encs)):
                for j in range(i + 1, len(encs)):
                    mask |= encs[i] ^ encs[j]
            if mask and mnem not in results:
                results[mnem] = {
                    'type': 'operand_fields', 'mask': mask,
                    'fields': _contiguous_fields(mask),
                    'n_values': len(by_ops),
                    'operands': sorted(by_ops.keys())[:8],
                }

    return results


def analyze_control_words(kernels):
    """Phase 3.1 + 3.4-3.5: Control word analysis.

    Catalogs per-family control word patterns. Identifies varying bit fields
    and correlates with producer/consumer patterns for barrier identification.
    """
    by_family = collections.defaultdict(list)
    all_ctrls = []

    for k in kernels:
        for insn in k.instructions:
            if not insn.mnemonic:
                continue
            family = insn_family(insn.mnemonic)
            if family:
                by_family[family].append(insn.control)
                all_ctrls.append(insn.control)

    family_stats = {}
    for family, ctrls in sorted(by_family.items()):
        unique = list(set(ctrls))
        if not unique:
            continue
        counter = collections.Counter(ctrls)
        default_ctrl = counter.most_common(1)[0][0]
        var_mask = 0
        for c in unique:
            var_mask |= c ^ default_ctrl
        stall_dist = dict(collections.Counter((c >> 53) & 0x7 for c in ctrls).most_common())
        family_stats[family] = {
            'n_instances': len(ctrls), 'n_unique': len(unique),
            'default': default_ctrl, 'var_mask': var_mask,
            'stall_dist': stall_dist,
        }

    global_var = 0
    if all_ctrls:
        for c in all_ctrls:
            global_var |= c ^ all_ctrls[0]

    bit_freq = [0] * 64
    for c in all_ctrls:
        for b in range(64):
            if c & (1 << b):
                bit_freq[b] += 1

    n_total = len(all_ctrls)
    ctrl_fields = _contiguous_fields(global_var)

    # Barrier analysis: long-latency producers vs normal instructions
    long_lat = {'LDTM', 'LDS', 'LDG', 'LDSM', 'LDC', 'UTMALDG'}
    skip = {'NOP', 'EXIT', 'BRA', 'BSSY', 'BSYNC', 'RET'}
    producer_ctrls = []
    normal_ctrls = []
    for family, ctrls in by_family.items():
        if family in long_lat:
            producer_ctrls.extend(ctrls)
        elif family not in skip:
            normal_ctrls.extend(ctrls)

    np_ = len(producer_ctrls) or 1
    nn_ = len(normal_ctrls) or 1
    prod_freq = [0] * 64
    norm_freq = [0] * 64
    for c in producer_ctrls:
        for b in range(64):
            if c & (1 << b):
                prod_freq[b] += 1
    for c in normal_ctrls:
        for b in range(64):
            if c & (1 << b):
                norm_freq[b] += 1

    wr_bar_candidates = []
    for b in range(64):
        pr = prod_freq[b] / np_
        nr = norm_freq[b] / nn_
        if pr > 0.05 and nr > 0 and pr > nr * 1.5:
            wr_bar_candidates.append((b, pr, nr))

    return {
        'family_stats': family_stats,
        'global_var_mask': global_var,
        'ctrl_fields': ctrl_fields,
        'bit_freq': {b: bit_freq[b] for b in range(64) if bit_freq[b] > 0},
        'n_total': n_total,
        'wr_bar_candidates': wr_bar_candidates,
        'n_producers': len(producer_ctrls),
        'n_normal': len(normal_ctrls),
    }


def verify_reg_fields(kernels, opcode_table=None):
    """Phase 2.1: Verify FAMILY_REG_FIELDS against opcode table var_mask.

    Checks that register fields are a subset of var_mask. Identifies
    families in the opcode table that are missing from FAMILY_REG_FIELDS.
    """
    if opcode_table is None:
        opcode_table = compute_opcode_table(kernels)

    table_families = set()
    for mnem in opcode_table:
        base = re.sub(r'\{[0-9a-f]+\}', '', mnem)
        table_families.add(base.split('.')[0])

    verification = {}
    for family, fields in FAMILY_REG_FIELDS.items():
        if not fields:
            verification[family] = {'status': 'no_fields', 'ok': True}
            continue

        reg_mask = 0
        for _, _, bit_off, width in fields:
            reg_mask |= ((1 << width) - 1) << bit_off

        matched = {}
        for mnem, entry in opcode_table.items():
            base = re.sub(r'\{[0-9a-f]+\}', '', mnem)
            if base.split('.')[0] == family:
                matched[mnem] = entry

        if not matched:
            verification[family] = {'status': 'not_in_opcode_table', 'ok': None}
            continue

        issues = []
        for mnem, entry in matched.items():
            vm = entry['var_mask']
            if vm == 0:
                continue
            excess = reg_mask & ~vm
            if excess:
                issues.append((mnem, '0x%016x' % excess))

        verification[family] = {
            'status': 'verified' if not issues else 'mismatch',
            'ok': not issues, 'reg_mask': reg_mask,
            'issues': issues, 'n_entries': len(matched),
        }

    missing = sorted(f for f in table_families if f not in FAMILY_REG_FIELDS)
    return {'verification': verification, 'missing_families': missing}


def analyze_special_encodings(kernels):
    """Phase 2.4-2.5: Uniform registers, RZ/PT, .reuse flag analysis."""

    # RZ verification: check that RZ is always 0xFF in register fields
    # Only check families with verified standard register layouts
    # Only families with simple, well-verified operand layouts.
    # Excludes: HFMA2 (negation prefix operands), LOP3 (predicate output
    # shifts indices), multi-form instructions (IADD3, IMAD, MOV, etc.)
    _rz_safe = {'F2FP', 'HADD2', 'FADD', 'FMUL', 'FFMA',
                'STS', 'STG', 'LDG', 'LDS', 'S2R'}
    rz_vals = set()
    rz_n = 0
    for k in kernels:
        for insn in k.instructions:
            if not insn.mnemonic or not insn.operands or 'RZ' not in insn.operands:
                continue
            family = insn_family(insn.mnemonic)
            if family not in _rz_safe:
                continue
            fields = FAMILY_REG_FIELDS.get(family, [])
            ops = _split_operands(insn.operands)
            for _, tidx, boff, width in fields:
                if tidx >= len(ops):
                    continue
                if ops[tidx].strip().replace('.reuse', '') == 'RZ':
                    rz_vals.add(read_enc_field(insn.encoding, boff, width))
                    rz_n += 1

    # UR max value: determine encoding width
    ur_max = 0
    ur_n = 0
    for k in kernels:
        for insn in k.instructions:
            if not insn.mnemonic or not insn.operands or 'UR' not in insn.operands:
                continue
            family = insn_family(insn.mnemonic)
            for fname, _, boff, width in FAMILY_REG_FIELDS.get(family, []):
                if fname.startswith('ur_'):
                    val = read_enc_field(insn.encoding, boff, width)
                    if val > ur_max:
                        ur_max = val
                    ur_n += 1

    # .reuse: compare instances with/without .reuse, same everything else
    reuse_enc = []
    reuse_ctrl = []
    reuse_groups = collections.defaultdict(list)
    for k in kernels:
        for insn in k.instructions:
            if not insn.mnemonic or not insn.operands:
                continue
            _, bare = _strip_pred(insn.mnemonic)
            has_reuse = '.reuse' in insn.operands
            clean = _normalize_ops(insn.operands)
            reuse_groups[(bare, clean)].append((has_reuse, insn.encoding, insn.control))

    for key, instances in reuse_groups.items():
        with_r = [(e, c) for has, e, c in instances if has]
        without = [(e, c) for has, e, c in instances if not has]
        if with_r and without:
            ed = with_r[0][0] ^ without[0][0]
            cd = with_r[0][1] ^ without[0][1]
            if _popcount64(ed) + _popcount64(cd) > 8:
                continue
            if ed:
                reuse_enc.append(ed)
            if cd:
                reuse_ctrl.append(cd)

    reuse_enc_mask = _majority_mask(reuse_enc, 0.3) if reuse_enc else 0
    reuse_ctrl_mask = _majority_mask(reuse_ctrl, 0.3) if reuse_ctrl else 0

    rz_consistent = len(rz_vals) == 1
    rz_value = next(iter(rz_vals)) if rz_consistent else (rz_vals or {0})

    return {
        'rz_value': rz_value,
        'rz_consistent': rz_consistent,
        'rz_n_checks': rz_n,
        'ur_max_value': ur_max, 'ur_n_checks': ur_n,
        'reuse_in_encoding': reuse_enc_mask != 0,
        'reuse_in_control': reuse_ctrl_mask != 0,
        'reuse_enc_mask': reuse_enc_mask,
        'reuse_ctrl_mask': reuse_ctrl_mask,
        'reuse_enc_bits': _bit_positions(reuse_enc_mask),
        'reuse_ctrl_bits': _bit_positions(reuse_ctrl_mask),
        'n_reuse_pairs': len(reuse_enc) + len(reuse_ctrl),
    }


def format_encoding_analysis(pred, imm, ctrl, reg_verify, special):
    """Format all Phase 2+3 results for display."""
    lines = []
    lines.append('=' * 78)
    lines.append('PHASE 2+3: OPERAND ENCODING + CONTROL WORD ANALYSIS')
    lines.append('=' * 78)

    # Predicates
    lines.append('\n--- Phase 2.3: Predicate Encoding ---')
    lines.append('Evidence: %d enable pairs, %d reg pairs, %d neg pairs' % (
        pred['n_enable'], pred['n_reg'], pred['n_neg']))
    if pred['enc_mask']:
        lines.append('Predicate in ENCODING WORD:')
        lines.append('  All pred bits:  0x%016x  %s' % (
            pred['enc_mask'], _bit_positions(pred['enc_mask'])))
        if pred.get('enable_mask'):
            lines.append('  Enable bit(s):  0x%016x  %s' % (
                pred['enable_mask'], pred.get('enable_bits', [])))
        if pred.get('reg_mask'):
            lines.append('  Register field: 0x%016x  %s' % (
                pred['reg_mask'], pred.get('reg_bits', [])))
            if 'reg_field' in pred:
                s, w = pred['reg_field']
                lines.append('    -> bits [%d:%d] (width %d)' % (s + w - 1, s, w))
        if pred.get('neg_mask'):
            lines.append('  Negation bit:   0x%016x  %s' % (
                pred['neg_mask'], pred.get('neg_bits', [])))
    if pred.get('ctrl_mask'):
        lines.append('Predicate bits in CONTROL WORD:')
        lines.append('  0x%016x  %s' % (
            pred['ctrl_mask'], pred.get('ctrl_pred_bits', [])))
    if not pred['enc_mask'] and not pred.get('ctrl_mask'):
        lines.append('  WARNING: No predicate bits found')

    # Immediates
    lines.append('\n--- Phase 2.2: Immediate Field Encoding ---')
    for mnem in sorted(imm.keys()):
        info = imm[mnem]
        lines.append('%s:' % mnem)
        lines.append('  type=%s  mask=0x%016x  fields=%s' % (
            info.get('type', '?'), info.get('mask', 0), info.get('fields', [])))
        if info.get('scale') is not None:
            lines.append('  scale=%d' % info['scale'])
        if info.get('n_values'):
            lines.append('  %d distinct values' % info['n_values'])
        if info.get('examples'):
            lines.append('  examples: %s' % info['examples'])
        if info.get('sr_names'):
            lines.append('  SR names: %s' % info['sr_names'][:10])
        if info.get('operands'):
            lines.append('  operands: %s' % info['operands'][:5])

    # Register verification
    lines.append('\n--- Phase 2.1: Register Field Verification ---')
    ok = sum(1 for v in reg_verify['verification'].values() if v.get('ok'))
    tot = len(reg_verify['verification'])
    lines.append('Verified: %d/%d families OK' % (ok, tot))
    for fam, info in sorted(reg_verify['verification'].items()):
        if info.get('ok') is False:
            lines.append('  MISMATCH: %s %s' % (fam, info.get('issues', '')))
    if reg_verify['missing_families']:
        lines.append('Missing from FAMILY_REG_FIELDS (%d):' % len(reg_verify['missing_families']))
        for f in reg_verify['missing_families']:
            lines.append('  %s' % f)

    # Special encodings
    lines.append('\n--- Phase 2.4/2.5: Special Encodings ---')
    rz = special['rz_value']
    lines.append('RZ = 0x%02x (%s, %d checks)' % (
        rz if isinstance(rz, int) else 0,
        'consistent' if special['rz_consistent'] else 'INCONSISTENT %s' % rz,
        special['rz_n_checks']))
    lines.append('UR max value seen: %d -> %s' % (
        special['ur_max_value'],
        '8-bit' if special['ur_max_value'] > 63 else '6-bit' if special['ur_max_value'] > 0 else 'unknown'))
    if special['reuse_in_control']:
        lines.append('.reuse in CONTROL WORD: bits %s (mask 0x%016x)' % (
            special['reuse_ctrl_bits'], special['reuse_ctrl_mask']))
    elif special['reuse_in_encoding']:
        lines.append('.reuse in ENCODING WORD: bits %s (mask 0x%016x)' % (
            special['reuse_enc_bits'], special['reuse_enc_mask']))
    else:
        lines.append('.reuse: location undetermined (%d pairs)' % special['n_reuse_pairs'])

    # Control words
    lines.append('\n--- Phase 3.1: Control Word Catalog ---')
    lines.append('Total: %d instructions' % ctrl['n_total'])
    lines.append('Global varying: 0x%016x' % ctrl['global_var_mask'])
    lines.append('Contiguous fields: %s' % ctrl['ctrl_fields'])
    lines.append('\nBit usage (nonzero bits):')
    for b in sorted(ctrl['bit_freq'].keys()):
        freq = ctrl['bit_freq'][b]
        pct = freq * 100.0 / ctrl['n_total'] if ctrl['n_total'] else 0
        lines.append('  bit %2d: %7d (%5.1f%%)' % (b, freq, pct))

    lines.append('\n--- Phase 3.4-3.5: Barrier Field Analysis ---')
    lines.append('Producers: %d, Normal: %d' % (ctrl['n_producers'], ctrl['n_normal']))
    if ctrl['wr_bar_candidates']:
        lines.append('Bits enriched in producers (wr_bar candidates):')
        for b, pr, nr in ctrl['wr_bar_candidates']:
            ratio = pr / nr if nr > 0 else float('inf')
            lines.append('  bit %2d: %.1f%% producers, %.1f%% normal (%.1fx)' % (
                b, pr * 100, nr * 100, ratio))
    else:
        lines.append('No clear wr_bar candidates')

    lines.append('\nPer-family defaults (top 20):')
    sorted_f = sorted(ctrl['family_stats'].items(), key=lambda x: -x[1]['n_instances'])
    for fam, st in sorted_f[:20]:
        d = decode_ctrl(st['default'])
        lines.append('  %-12s ctrl=0x%016x  st=%d y=%d wr=%02x rd=%02x wm=%02x ru=%d  '
                      'var=0x%016x  (%d/%d)' % (
            fam, st['default'], d['stall'], d['yield'],
            d['wr_bar'], d['rd_bar'], d['wait_mask'], d['reuse'],
            st['var_mask'], st['n_unique'], st['n_instances']))

    return '\n'.join(lines)


def export_encoding_data(pred, imm, ctrl, special, path):
    """Export Phase 2+3 results as importable Python module."""
    lines = [
        '# SM100a encoding data — auto-generated by sass_edit.py analyze-encoding',
        '# Phase 2+3: Operand Encoding + Control Word Analysis',
        '',
        '# Phase 2.3: Predicate guard encoding',
        'PREDICATE_ENCODING = {',
    ]
    for key in ['enc_mask', 'reg_mask', 'neg_mask', 'enable_mask']:
        if pred.get(key):
            lines.append("    '%s': 0x%016x," % (key, pred[key]))
    for key in ['reg_field', 'neg_field', 'enable_field']:
        if key in pred:
            lines.append("    '%s': %s," % (key, pred[key]))
    lines.append('}')
    lines.append('')

    lines.append('# Phase 2.2: Immediate field encoding')
    lines.append('IMMEDIATE_FIELDS = {')
    for mnem in sorted(imm.keys()):
        info = imm[mnem]
        lines.append("    '%s': {" % mnem)
        for k in ['type', 'mask', 'fields', 'scale', 'pc_relative']:
            if k in info and info[k] is not None:
                val = info[k]
                if isinstance(val, int) and k == 'mask':
                    lines.append("        '%s': 0x%016x," % (k, val))
                else:
                    lines.append("        '%s': %r," % (k, val))
        lines.append('    },')
    lines.append('}')
    lines.append('')

    lines.append('# Phase 3.1: Control word field boundaries')
    lines.append('CONTROL_FIELDS = %r' % (ctrl['ctrl_fields'],))
    lines.append('')

    lines.append('# Per-family default control words')
    lines.append('CONTROL_DEFAULTS = {')
    for fam in sorted(ctrl['family_stats'].keys()):
        lines.append("    '%s': 0x%016x," % (fam, ctrl['family_stats'][fam]['default']))
    lines.append('}')
    lines.append('')

    if isinstance(special['rz_value'], int):
        lines.append('RZ_VALUE = 0x%02x' % special['rz_value'])
    if special['reuse_ctrl_mask']:
        lines.append('REUSE_CTRL_MASK = 0x%016x' % special['reuse_ctrl_mask'])
        lines.append('REUSE_CTRL_BITS = %s' % special['reuse_ctrl_bits'])
    if special['reuse_enc_mask']:
        lines.append('REUSE_ENC_MASK = 0x%016x' % special['reuse_enc_mask'])
        lines.append('REUSE_ENC_BITS = %s' % special['reuse_enc_bits'])
    lines.append('')

    Path(path).write_text('\n'.join(lines))
    return len(lines)


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
    # Upper bound = MAX_STALL (15) — since reorder invalidates scoreboard
    # barriers, stall counts must fully cover all latencies without relying
    # on hardware barrier waits.  Forces the solver to spread high-latency
    # deps across intermediate instructions rather than exceeding the stall cap.
    for p in range(N - 1):
        model.add(time_at_pos[p + 1] >= time_at_pos[p] + 1)
        model.add(time_at_pos[p + 1] <= time_at_pos[p] + MAX_STALL)

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
            stall_counts[addr] = instructions[idx].stall  # keep original
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
        orig_stalls = sum(insn.stall for insn in instructions)
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
    reordered = (ordered_addrs != orig_addrs)
    if reordered:
        addr_str = ','.join('0x%04x' % a for a in ordered_addrs)
        recipe.append('reorder 0x%04x 0x%04x %s' % (start_addr, end_addr, addr_str))
        recipe.append('')

    # Stall patches — after reorder, instruction originally at ordered_addrs[p]
    # is now at address start_addr + p * INSN_SIZE.  Stall commands must target
    # the NEW address, not the original.
    recipe.append('# Stall count patches')
    for p, orig_addr in enumerate(ordered_addrs):
        stall = stall_counts.get(orig_addr)
        if stall is not None:
            new_addr = (start_addr + p * INSN_SIZE) if reordered else orig_addr
            recipe.append('stall 0x%04x %d' % (new_addr, stall))

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

        # Write in new order.  Control words move with their instructions
        # unchanged — only stall counts (bits 0-3) are patched afterward.
        # SM100a control word layout above bit 4 is NOT verified, so we
        # must not modify any bits other than stall to avoid corruption.
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
        """Set stall count at addr (most common operation).
        SM100a: stall is 3 bits at bits 53-55 of the control word."""
        idx = addr // INSN_SIZE
        insn = kernel.instructions[idx]
        mask = 0x7 << 53
        insn.control = (insn.control & ~mask) | ((stall & 0x7) << 53)
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

    def patch_region_asm(self, kernel, start, end, asm_text):
        """Replace instructions in [start, end) with assembled SASS text.

        Assembles asm_text with base_pc=start, overwrites the instruction slots
        in that range, and NOP-fills any remaining slots. Raises if assembled
        output exceeds the available region.

        Returns (n_patched, n_nops) — number of assembled instructions and NOPs.
        """
        if start % INSN_SIZE != 0 or end % INSN_SIZE != 0:
            raise ValueError('start/end must be aligned to %d bytes' % INSN_SIZE)
        if start >= end:
            raise ValueError('start (0x%x) must be < end (0x%x)' % (start, end))

        n_slots = (end - start) // INSN_SIZE
        start_idx = start // INSN_SIZE
        end_idx = end // INSN_SIZE

        if end_idx > len(kernel.instructions):
            raise ValueError('end 0x%x beyond kernel (%d insns)' % (
                end, len(kernel.instructions)))

        binary, labels, instructions = assemble(asm_text, base_pc=start)
        n_asm = len(instructions)

        if n_asm > n_slots:
            raise ValueError(
                'Assembled %d instructions but region [0x%x, 0x%x) has only %d slots' % (
                    n_asm, start, end, n_slots))

        # Overwrite with assembled instructions
        for i in range(n_asm):
            enc = struct.unpack_from('<Q', binary, i * INSN_SIZE)[0]
            ctrl = struct.unpack_from('<Q', binary, i * INSN_SIZE + 8)[0]
            insn = kernel.instructions[start_idx + i]
            insn.encoding = enc
            insn.control = ctrl
            insn.mnemonic = instructions[i].mnemonic
            insn.operands = ''

        # NOP-fill remainder
        n_nops = n_slots - n_asm
        for i in range(n_asm, n_slots):
            insn = kernel.instructions[start_idx + i]
            insn.encoding = NOP_ENCODING
            insn.control = insn.control & ~(0x7 << 53)  # zero stall, keep other ctrl bits
            insn.mnemonic = 'NOP'
            insn.operands = ''

        self._modified = True
        return n_asm, n_nops

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

            elif cmd == 'asm':
                # asm 0xSTART 0xEND path/to/file.s
                start = int(parts[1], 0)
                end = int(parts[2], 0)
                asm_path = parts[3]
                asm_text = Path(asm_path).read_text()
                n_patched, n_nops = editor.patch_region_asm(
                    kernel, start, end, asm_text)
                print('  asm [%04x, %04x): %d instructions + %d NOPs' % (
                    start, end, n_patched, n_nops))
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

        stall = insn.stall
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

            stall_a = a.stall
            stall_b = b.stall
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


def cmd_opcode_table(args):
    """Extract opcode table from one or more cubins via XOR analysis."""
    all_kernels = []
    pairs = []
    if args.cubin_sass:
        for pair in args.cubin_sass:
            cubin_path, sass_path = pair.split(':')
            pairs.append((cubin_path, sass_path))
    else:
        pairs.append((args.cubin, args.sass))

    for cubin_path, sass_path in pairs:
        ed = CubinEditor(cubin_path)
        n = ed.xref(sass_path)
        total_insns = sum(k.n_insns for k in ed.kernels)
        sys.stderr.write('%s: xref %d/%d instructions across %d kernels\n' % (
            cubin_path, n, total_insns, len(ed.kernels)))
        all_kernels.extend(ed.kernels)

    table = compute_opcode_table(all_kernels)
    print(format_opcode_table(table, show_all=args.all))

    if args.export:
        n = export_opcode_table(table, args.export)
        sys.stderr.write('Exported %d entries to %s\n' % (n, args.export))


def cmd_analyze_encoding(args):
    """Run Phase 2+3 encoding analysis on one or more cubins."""
    all_kernels = []
    pairs = []
    if args.cubin_sass:
        for pair in args.cubin_sass:
            cubin_path, sass_path = pair.split(':')
            pairs.append((cubin_path, sass_path))
    else:
        pairs.append((args.cubin, args.sass))

    for cubin_path, sass_path in pairs:
        ed = CubinEditor(cubin_path)
        n = ed.xref(sass_path)
        total_insns = sum(k.n_insns for k in ed.kernels)
        sys.stderr.write('%s: xref %d/%d instructions across %d kernels\n' % (
            cubin_path, n, total_insns, len(ed.kernels)))
        all_kernels.extend(ed.kernels)

    sys.stderr.write('Running Phase 2+3 analysis on %d kernels...\n' % len(all_kernels))

    pred = analyze_predicates(all_kernels)
    imm = analyze_immediates(all_kernels)
    ctrl = analyze_control_words(all_kernels)
    opcode_table = compute_opcode_table(all_kernels)
    reg_verify = verify_reg_fields(all_kernels, opcode_table)
    special = analyze_special_encodings(all_kernels)

    print(format_encoding_analysis(pred, imm, ctrl, reg_verify, special))

    if args.export:
        n = export_encoding_data(pred, imm, ctrl, special, args.export)
        sys.stderr.write('Exported %d lines to %s\n' % (n, args.export))


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
        reordered = (ordered_addrs != orig_addrs)
        if reordered:
            ed.reorder(k, start, end, ordered_addrs, force=True)

        # Apply stall counts — after reorder, instruction originally at
        # ordered_addrs[p] is now at start + p * INSN_SIZE.
        for p, orig_addr in enumerate(ordered_addrs):
            stall = stall_counts.get(orig_addr)
            if stall is not None:
                new_addr = (start + p * INSN_SIZE) if reordered else orig_addr
                ed.patch_stall(k, new_addr, stall)

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
# Phase 4: Assembler Core
#
# Text-to-binary assembler for SM100a SASS. Uses opcode table, register
# field layout, immediate field encoding, and predicate encoding from
# Phases 1-3 to convert assembly text into 128-bit instruction words.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_asm_tables():
    """Lazy-load assembler data tables (sm100a_opcodes, sm100a_encoding)."""
    global OPCODE_TABLE, IMMEDIATE_FIELDS, CONTROL_DEFAULTS, REUSE_CTRL_MASK
    global _asm_tables_loaded
    if _asm_tables_loaded:
        return
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from sm100a_opcodes import OPCODE_TABLE as _OT
    from sm100a_encoding import (IMMEDIATE_FIELDS as _IF,
                                 CONTROL_DEFAULTS as _CD,
                                 REUSE_CTRL_MASK as _RM)
    OPCODE_TABLE = _OT
    IMMEDIATE_FIELDS = _IF
    CONTROL_DEFAULTS = _CD
    REUSE_CTRL_MASK = _RM
    _asm_tables_loaded = True

_asm_tables_loaded = False
OPCODE_TABLE = {}
IMMEDIATE_FIELDS = {}
CONTROL_DEFAULTS = {}
REUSE_CTRL_MASK = 0

# Operand types for parsed assembly
OP_GPR = 'gpr'          # R0-R255, RZ
OP_UR = 'ur'            # UR0-UR255, URZ
OP_PRED = 'pred'        # P0-P6, PT
OP_UPRED = 'upred'      # UP0-UP6, UPT
OP_IMM = 'imm'          # decimal or hex immediate
OP_MEM = 'mem'          # [Rbase+offset] or [Rbase]
OP_UMEM = 'umem'        # [URbase] or [URbase+offset]
OP_TMEM = 'tmem'        # [TMEM+offset]
OP_LABEL = 'label'      # .L_name (branch target)
OP_BARRIER = 'barrier'  # bare integer for BAR.SYNC barrier id
OP_DESC = 'desc'        # desc[URn] — TMA/TCQMMA descriptor
OP_TMEM_UR = 'tmem_ur'  # tmem[URn] — TMEM via uniform register

# Parsed operand: (type, value_dict)
# value_dict keys depend on type:
#   gpr:     {'reg': int}  (255 = RZ)
#   ur:      {'reg': int}  (255 = URZ)
#   pred:    {'reg': int}  (7 = PT)
#   upred:   {'reg': int}  (7 = UPT)
#   imm:     {'value': int}
#   mem:     {'base': int, 'offset': int}  (base = GPR num)
#   umem:    {'base': int, 'offset': int}  (base = UR num)
#   tmem:    {'offset': int}
#   label:   {'name': str}
#   barrier: {'id': int}
#   desc:    {'reg': int}   (UR num)
#   tmem_ur: {'reg': int}   (UR num)


class AsmOperand:
    __slots__ = ('type', 'val', 'reuse')
    def __init__(self, type, val, reuse=False):
        self.type = type
        self.val = val
        self.reuse = reuse
    def __repr__(self):
        r = '.reuse' if self.reuse else ''
        return 'AsmOperand(%s, %r%s)' % (self.type, self.val, r)


class AsmInstruction:
    __slots__ = ('line_num', 'guard_neg', 'guard_reg', 'guard_uniform',
                 'mnemonic', 'operands', 'directives', 'label')

    def __init__(self):
        self.line_num = 0
        self.guard_neg = False      # @!P0 → True
        self.guard_reg = 7          # PT (no guard) = 7
        self.guard_uniform = False  # @UP0 → True
        self.mnemonic = ''          # 'F2FP.BF16.F32.PACK_AB'
        self.operands = []          # list of AsmOperand
        self.directives = {}        # {'.stall': 5, '.yield': True, ...}
        self.label = None           # label defined at this line

    @property
    def family(self):
        return self.mnemonic.split('.')[0] if self.mnemonic else None

    @property
    def has_guard(self):
        return self.guard_reg != 7 or self.guard_neg


_RE_ASM_GPR = re.compile(r'^R(\d+)$')
_RE_ASM_RZ = re.compile(r'^RZ$')
_RE_ASM_UR = re.compile(r'^UR(\d+)$')
_RE_ASM_URZ = re.compile(r'^URZ$')
_RE_ASM_PRED_OP = re.compile(r'^(P[0-6]|PT)$')
_RE_ASM_UPRED_OP = re.compile(r'^(UP[0-6]|UPT)$')
_RE_ASM_BARRIER = re.compile(r'^B(\d+)$')
_RE_ASM_SB = re.compile(r'^SB(\d+)$')
_RE_ASM_SR = re.compile(r'^SR_([\w.]+)$')
_RE_ASM_SRZ = re.compile(r'^SRZ$')
_RE_ASM_MEM = re.compile(r'^\[R(\d+)(?:\.(\d+))?(?:\+(?:(0x[0-9a-fA-F]+)|(-?0x[0-9a-fA-F]+)|(\d+)))?\]$')
_RE_ASM_DESC_MEM = re.compile(r'^desc\[UR(\d+)\]\[R(\d+)(?:\.(\d+))?\]$', re.IGNORECASE)
_RE_ASM_MEM_UR = re.compile(
    r'^\[R(\d+)\+URZ(?:\+((?:0x[0-9a-fA-F]+)|\d+))?\]$')
_RE_ASM_MEM_NEG = re.compile(r'^\[R(\d+)\+(-0x[0-9a-fA-F]+|-\d+)\]$')
_RE_ASM_UMEM = re.compile(r'^\[UR(\d+)(?:\+(?:(0x[0-9a-fA-F]+)|(\d+)))?\]$')
_RE_ASM_TMEM = re.compile(r'^\[TMEM\+(?:(0x[0-9a-fA-F]+)|(\d+))\]$', re.IGNORECASE)
_RE_ASM_DESC = re.compile(r'^desc\[UR(\d+)\]$', re.IGNORECASE)
_RE_ASM_GDESC = re.compile(r'^gdesc\[UR(\d+)\]$', re.IGNORECASE)
_RE_ASM_IDESC = re.compile(r'^idesc\[UR(\d+)\]$', re.IGNORECASE)
_RE_ASM_TMEM_UR = re.compile(r'^tmem\[UR(\d+)(?:\+((?:0x[0-9a-fA-F]+)|\d+))?\]$', re.IGNORECASE)
_RE_ASM_CBANK = re.compile(r'^c\[(0x[0-9a-fA-F]+|\d+)\]\[(0x[0-9a-fA-F]+|\d+)\]$')
_RE_ASM_LABEL = re.compile(r'^\.L\w+$')
_RE_ASM_IMM = re.compile(r'^(-?(?:0x[0-9a-fA-F]+|\d+))$')
_RE_ASM_GUARD = re.compile(r'^@(!?)(U?P[0-6]|U?PT)$')
_RE_ASM_LABEL_DEF = re.compile(r'^(\.L\w+):$')
_RE_ASM_DIRECTIVE = re.compile(r'^\.(stall|yield|noyield|barrier|reuse|ctrl)\s*(.*)')


def _parse_asm_operand(text):
    """Parse a single operand token into an AsmOperand."""
    raw = text.strip()
    reuse = False
    if raw.endswith('.reuse'):
        reuse = True
        raw = raw[:-6].strip()

    # Strip modifiers (negation, abs, bitwise-NOT) for register operands.
    # Only strip if followed by a register name (R, UR, P), not for immediates.
    neg = False
    if raw.startswith('-') and len(raw) > 1 and raw[1:].lstrip('!').startswith(('R', 'UR', 'P', 'UP')):
        neg = True
        raw = raw[1:]
    elif raw.startswith('~') and len(raw) > 1:
        neg = True
        raw = raw[1:]
    elif raw.startswith('|') and raw.endswith('|') and len(raw) > 2:
        raw = raw[1:-1]

    # RZ
    if _RE_ASM_RZ.match(raw):
        return AsmOperand(OP_GPR, {'reg': 0xff}, reuse)
    # GPR
    m = _RE_ASM_GPR.match(raw)
    if m:
        return AsmOperand(OP_GPR, {'reg': int(m.group(1))}, reuse)
    # URZ
    if _RE_ASM_URZ.match(raw):
        return AsmOperand(OP_UR, {'reg': 0xff}, reuse)
    # UR
    m = _RE_ASM_UR.match(raw)
    if m:
        return AsmOperand(OP_UR, {'reg': int(m.group(1))}, reuse)
    # Predicate operand (including negated: !P0, !PT, !UP3)
    pred_raw = raw[1:] if raw.startswith('!') else raw
    m = _RE_ASM_PRED_OP.match(pred_raw)
    if m:
        val = 7 if pred_raw == 'PT' else int(pred_raw[1])
        return AsmOperand(OP_PRED, {'reg': val, 'neg': raw.startswith('!')}, reuse)
    m = _RE_ASM_UPRED_OP.match(pred_raw)
    if m:
        val = 7 if pred_raw == 'UPT' else int(pred_raw[2])
        return AsmOperand(OP_UPRED, {'reg': val, 'neg': raw.startswith('!')}, reuse)
    # Barrier register (B0-B6)
    m = _RE_ASM_BARRIER.match(raw)
    if m:
        return AsmOperand(OP_BARRIER, {'reg': int(m.group(1))}, reuse)
    # Scoreboard barrier (SB0-SB5)
    m = _RE_ASM_SB.match(raw)
    if m:
        return AsmOperand(OP_IMM, {'value': int(m.group(1))}, reuse)
    # Scoreboard register zero (SRZ)
    if _RE_ASM_SRZ.match(raw):
        return AsmOperand(OP_IMM, {'value': 0, 'sr_name': 'SRZ'}, reuse)
    # Special register (SR_xxx)
    m = _RE_ASM_SR.match(raw)
    if m:
        return AsmOperand(OP_IMM, {'value': 0, 'sr_name': m.group(1)}, reuse)
    # desc[URn]
    m = _RE_ASM_DESC.match(raw)
    if m:
        return AsmOperand(OP_DESC, {'reg': int(m.group(1))}, reuse)
    # gdesc[URn]
    m = _RE_ASM_GDESC.match(raw)
    if m:
        return AsmOperand(OP_DESC, {'reg': int(m.group(1)), 'kind': 'gdesc'}, reuse)
    # idesc[URn]
    m = _RE_ASM_IDESC.match(raw)
    if m:
        return AsmOperand(OP_DESC, {'reg': int(m.group(1)), 'kind': 'idesc'}, reuse)
    # desc[URn][Rn.64] — TMA descriptor with GPR address
    m = _RE_ASM_DESC_MEM.match(raw)
    if m:
        ur_reg = int(m.group(1))
        gpr_reg = int(m.group(2))
        size = int(m.group(3)) if m.group(3) else 0
        return AsmOperand(OP_DESC, {'reg': ur_reg, 'gpr': gpr_reg, 'size': size, 'kind': 'desc_mem'}, reuse)
    # tmem[URn+offset] or tmem[URn]
    m = _RE_ASM_TMEM_UR.match(raw)
    if m:
        off = int(m.group(2), 0) if m.group(2) else 0
        return AsmOperand(OP_TMEM_UR, {'reg': int(m.group(1)), 'offset': off}, reuse)
    # [TMEM+offset]
    m = _RE_ASM_TMEM.match(raw)
    if m:
        off = int(m.group(1) or m.group(2), 0)
        return AsmOperand(OP_TMEM, {'offset': off}, reuse)
    # [Rn+URZ+offset] — memory with uniform register offset
    m = _RE_ASM_MEM_UR.match(raw)
    if m:
        base = int(m.group(1))
        off = int(m.group(2), 0) if m.group(2) else 0
        return AsmOperand(OP_MEM, {'base': base, 'offset': off, 'ur': True}, reuse)
    # [Rn+-offset] — memory with negative offset
    m = _RE_ASM_MEM_NEG.match(raw)
    if m:
        base = int(m.group(1))
        off = int(m.group(2), 0)
        return AsmOperand(OP_MEM, {'base': base, 'offset': off}, reuse)
    # [Rn+offset] or [Rn] or [Rn.64]
    m = _RE_ASM_MEM.match(raw)
    if m:
        base = int(m.group(1))
        size = int(m.group(2)) if m.group(2) else 0
        off = int(m.group(3) or m.group(4) or m.group(5) or '0', 0)
        return AsmOperand(OP_MEM, {'base': base, 'offset': off, 'size': size}, reuse)
    # [URn+offset] or [URn]
    m = _RE_ASM_UMEM.match(raw)
    if m:
        base = int(m.group(1))
        off = int(m.group(2) or m.group(3) or '0', 0)
        return AsmOperand(OP_UMEM, {'base': base, 'offset': off}, reuse)
    # Constant bank c[bank][offset]
    m = _RE_ASM_CBANK.match(raw)
    if m:
        bank = int(m.group(1), 0)
        off = int(m.group(2), 0)
        return AsmOperand(OP_IMM, {'value': off, 'cbank': bank}, reuse)
    # Label
    if _RE_ASM_LABEL.match(raw):
        return AsmOperand(OP_LABEL, {'name': raw}, reuse)
    # Immediate (must be last — catches bare numbers)
    m = _RE_ASM_IMM.match(raw)
    if m:
        return AsmOperand(OP_IMM, {'value': int(m.group(1), 0)}, reuse)

    raise ValueError('cannot parse operand: %r' % text)


def _split_asm_operands(text):
    """Split operand string respecting brackets: '[R0+0x10], R5' → ['[R0+0x10]', 'R5']
    If no commas present, split on spaces (for RET-style: 'R2 0x0')."""
    has_comma = ',' in text
    parts = []
    depth = 0
    cur = []
    for ch in text:
        if ch == '[':
            depth += 1
            cur.append(ch)
        elif ch == ']':
            depth -= 1
            cur.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(cur).strip())
            cur = []
        elif ch == ' ' and depth == 0 and not has_comma:
            s = ''.join(cur).strip()
            if s:
                parts.append(s)
            cur = []
        else:
            cur.append(ch)
    if cur:
        s = ''.join(cur).strip()
        if s:
            parts.append(s)
    return parts


def parse_asm_line(line, line_num=0):
    """Parse a single assembly line into an AsmInstruction (or None for empty/comment)."""
    stripped = line.split('#')[0].split(';')[0].strip()
    if not stripped:
        return None

    insn = AsmInstruction()
    insn.line_num = line_num

    # Directive?
    m = _RE_ASM_DIRECTIVE.match(stripped)
    if m:
        insn.mnemonic = ''  # directive-only, no instruction
        name = '.' + m.group(1)
        arg = m.group(2).strip()
        if name == '.stall':
            insn.directives['.stall'] = int(arg, 0)
        elif name == '.yield':
            insn.directives['.yield'] = True
        elif name == '.noyield':
            insn.directives['.noyield'] = True
        elif name == '.reuse':
            insn.directives['.reuse'] = True
        elif name == '.ctrl':
            insn.directives['.ctrl'] = int(arg, 0)
        elif name == '.barrier':
            # .barrier wait=0x3 or .barrier write=1
            for kv in arg.split(','):
                kv = kv.strip()
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    insn.directives['.barrier.' + k.strip()] = int(v.strip(), 0)
        return insn

    # Label definition?
    m = _RE_ASM_LABEL_DEF.match(stripped)
    if m:
        insn.label = m.group(1)
        insn.mnemonic = ''
        return insn

    # Guard predicate?
    tokens = stripped.split()
    idx = 0
    m = _RE_ASM_GUARD.match(tokens[0])
    if m:
        insn.guard_neg = (m.group(1) == '!')
        pred_str = m.group(2)
        if pred_str.startswith('U'):
            insn.guard_uniform = True
            pred_str = pred_str[1:]  # UP0 → P0
        if pred_str == 'PT':
            insn.guard_reg = 7
        else:
            insn.guard_reg = int(pred_str[1])
        idx = 1

    if idx >= len(tokens):
        return insn

    # Mnemonic
    insn.mnemonic = tokens[idx]
    idx += 1

    # Operands (everything after mnemonic, joined)
    if idx < len(tokens):
        ops_text = ' '.join(tokens[idx:])
        for op_str in _split_asm_operands(ops_text):
            insn.operands.append(_parse_asm_operand(op_str))

    return insn


def parse_asm(text):
    """Parse multi-line assembly text into list of AsmInstruction.

    Returns (instructions, labels) where labels is {name: insn_index}.
    Directive-only lines attach to the next instruction.
    """
    instructions = []
    labels = {}
    pending_directives = {}
    pending_label = None

    for line_num, line in enumerate(text.splitlines(), 1):
        insn = parse_asm_line(line, line_num)
        if insn is None:
            continue

        # Label definition
        if insn.label:
            pending_label = insn.label
            continue

        # Directive-only (no mnemonic)
        if not insn.mnemonic:
            pending_directives.update(insn.directives)
            continue

        # Real instruction — attach pending directives and label
        if pending_directives:
            for k, v in pending_directives.items():
                if k not in insn.directives:
                    insn.directives[k] = v
            pending_directives = {}

        if pending_label:
            labels[pending_label] = len(instructions)
            insn.label = pending_label
            pending_label = None

        instructions.append(insn)

    return instructions, labels


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Encoder: AsmInstruction → 128-bit binary (encoding + control word)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _lookup_opcode(mnemonic):
    """Look up opcode_bits and var_mask for a mnemonic.

    Tries exact match first, then strips trailing modifiers progressively.
    Returns (opcode_bits, var_mask) or raises ValueError.
    """
    if mnemonic in OPCODE_TABLE:
        return OPCODE_TABLE[mnemonic]

    # Try progressively shorter modifier chains
    parts = mnemonic.split('.')
    for n in range(len(parts) - 1, 0, -1):
        key = '.'.join(parts[:n])
        if key in OPCODE_TABLE:
            return OPCODE_TABLE[key]

    raise ValueError('unknown mnemonic: %r (not in opcode table)' % mnemonic)


def _encode_predicate(insn):
    """Encode predicate guard into encoding word bits [15:12]."""
    reg = insn.guard_reg & 0x7
    neg = 1 if insn.guard_neg else 0
    return (neg << 3 | reg) << 12


def _encode_registers(insn, encoding):
    """Fill register fields in encoding word based on FAMILY_REG_FIELDS."""
    family = insn.family
    fields = FAMILY_REG_FIELDS.get(family)
    if fields is None:
        fields = _DEFAULT_REG_FIELDS

    for field_name, text_op_idx, bit_offset, width in fields:
        # Find the operand — text_op_idx maps to operand position in cuobjdump
        # but our AsmOperand list is 0-indexed from the assembly text.
        # For negative indices, use a heuristic
        if text_op_idx < 0:
            continue  # skip special handling (e.g., UTCATOMSWS)

        if text_op_idx >= len(insn.operands):
            continue

        op = insn.operands[text_op_idx]
        if op.type == OP_GPR:
            encoding = write_enc_field(encoding, bit_offset, width, op.val['reg'])
        elif op.type == OP_UR:
            encoding = write_enc_field(encoding, bit_offset, width, op.val['reg'])
        elif op.type == OP_DESC:
            encoding = write_enc_field(encoding, bit_offset, width, op.val['reg'])
        elif op.type == OP_TMEM_UR:
            encoding = write_enc_field(encoding, bit_offset, width, op.val['reg'])
        elif op.type == OP_MEM:
            if 'addr' in field_name or 'src' in field_name:
                encoding = write_enc_field(encoding, bit_offset, width, op.val['base'])
        elif op.type == OP_UMEM:
            encoding = write_enc_field(encoding, bit_offset, width, op.val['base'])
        elif op.type == OP_PRED:
            encoding = write_enc_field(encoding, bit_offset, width, op.val['reg'])
        elif op.type == OP_UPRED:
            encoding = write_enc_field(encoding, bit_offset, width, op.val['reg'])

    return encoding


def _encode_immediate(insn, encoding, pc, labels):
    """Fill immediate fields based on IMMEDIATE_FIELDS."""
    key = insn.mnemonic
    if key not in IMMEDIATE_FIELDS:
        # Try family-level match
        key = insn.family
    if key not in IMMEDIATE_FIELDS:
        return encoding

    info = IMMEDIATE_FIELDS[key]
    itype = info.get('type', '')
    fields = info['fields']
    scale = info.get('scale', 1)
    is_pc_relative = info.get('pc_relative', False)
    mode_bits = info.get('mode_bits', 0)

    # DEPBAR: first OP_IMM = SB index, second OP_IMM = count
    if itype == 'depbar':
        imm_ops = [op for op in insn.operands if op.type == OP_IMM]
        if len(imm_ops) >= 1:
            sb_val = imm_ops[0].val['value']
            sb_off, sb_w = info['sb_field']
            encoding = write_enc_field(encoding, sb_off, sb_w, sb_val & ((1 << sb_w) - 1))
        if len(imm_ops) >= 2:
            count_val = imm_ops[1].val['value']
            for bit_offset, width in fields:
                encoding = write_enc_field(encoding, bit_offset, width, count_val & ((1 << width) - 1))
                count_val >>= width
        return encoding

    # Constant pair (HFMA2.BF16_V2): one field per OP_IMM operand
    if itype == 'constant_pair':
        cmap = info.get('constant_map', {})
        imm_ops = [op for op in insn.operands if op.type == OP_IMM]
        for i, (bit_offset, width) in enumerate(fields):
            if i < len(imm_ops):
                raw = imm_ops[i].val['value']
                val = cmap.get(raw, raw)
                encoding = write_enc_field(encoding, bit_offset, width, val & ((1 << width) - 1))
        if imm_ops:
            encoding |= mode_bits
        return encoding

    # Standard: find single immediate value from operands
    imm_val = None
    for op in insn.operands:
        if op.type == OP_IMM:
            imm_val = op.val['value']
            break
        elif op.type == OP_LABEL and is_pc_relative and labels is not None:
            target_pc = labels.get(op.val['name'])
            if target_pc is None:
                raise ValueError('line %d: undefined label %r' % (insn.line_num, op.val['name']))
            if info.get('next_pc', False):
                imm_val = target_pc - pc - INSN_SIZE
            else:
                imm_val = target_pc - pc
            break
        elif op.type == OP_MEM:
            imm_val = op.val['offset']
            break
        elif op.type == OP_UMEM:
            imm_val = op.val['offset']
            break
        elif op.type == OP_TMEM:
            imm_val = op.val['offset']
            break
        elif op.type == OP_TMEM_UR:
            imm_val = op.val['offset']
            break

    if imm_val is None:
        return encoding

    # Set mode bits when immediate is present
    if mode_bits and imm_val is not None:
        encoding |= mode_bits

    # Apply scale
    if scale > 1:
        imm_val = imm_val // scale

    # Distribute value across fields (LSB-first into field list)
    val = imm_val & ((1 << 64) - 1)  # unsigned for bit manipulation
    for bit_offset, width in fields:
        chunk = val & ((1 << width) - 1)
        encoding = write_enc_field(encoding, bit_offset, width, chunk)
        val >>= width

    return encoding


def encode_instruction(insn, pc=0, labels=None):
    """Encode an AsmInstruction into (encoding_word, control_word).

    pc: byte offset of this instruction (for PC-relative branches)
    labels: {name: pc_address} for branch target resolution
    """
    # Start with opcode bits
    opcode_bits, var_mask = _lookup_opcode(insn.mnemonic)
    encoding = opcode_bits

    # Predicate guard — bits [15:12]
    pred_bits = _encode_predicate(insn)
    # Clear predicate field in encoding, then set
    encoding = (encoding & ~0xf000) | pred_bits

    # Register fields
    encoding = _encode_registers(insn, encoding)

    # Immediate fields
    encoding = _encode_immediate(insn, encoding, pc, labels)

    # Control word
    if '.ctrl' in insn.directives:
        # Exact control word — use as-is, then apply explicit directive overrides.
        # Operand .reuse flags are NOT applied (they're in encoding bits, not ctrl).
        ctrl = insn.directives['.ctrl']
        if '.stall' in insn.directives:
            mask = 0x7 << 53
            ctrl = (ctrl & ~mask) | ((insn.directives['.stall'] & 0x7) << 53)
        if '.yield' in insn.directives:
            ctrl = ctrl | (1 << 4)
        elif '.noyield' in insn.directives:
            ctrl = ctrl & ~(1 << 4)
        if '.reuse' in insn.directives:
            ctrl = ctrl | REUSE_CTRL_MASK
    else:
        # Compute from family default
        family = insn.family
        ctrl = CONTROL_DEFAULTS.get(family, 0x000fe20000000000)
        if '.stall' in insn.directives:
            mask = 0x7 << 53
            ctrl = (ctrl & ~mask) | ((insn.directives['.stall'] & 0x7) << 53)
        if '.yield' in insn.directives:
            ctrl = ctrl | (1 << 4)
        if '.barrier.wait' in insn.directives:
            wait_val = insn.directives['.barrier.wait']
            ctrl = (ctrl & ~(0x3f << 11)) | ((wait_val & 0x3f) << 11)
        if '.barrier.write' in insn.directives:
            wr_val = insn.directives['.barrier.write']
            ctrl = (ctrl & ~(0x7 << 5)) | ((wr_val & 0x7) << 5)
        if '.barrier.read' in insn.directives:
            rd_val = insn.directives['.barrier.read']
            ctrl = (ctrl & ~(0x7 << 8)) | ((rd_val & 0x7) << 8)
        if '.reuse' in insn.directives or any(op.reuse for op in insn.operands):
            ctrl = ctrl | REUSE_CTRL_MASK

    return encoding, ctrl


def disassemble_region(kernel, start, end):
    """Disassemble instructions in [start, end) to assembler-format text.

    Requires xref to have been called first (instructions need mnemonic/operands).
    Returns a string of assembly text that can be fed back to assemble().
    """
    _load_asm_tables()
    if start % INSN_SIZE != 0 or end % INSN_SIZE != 0:
        raise ValueError('start/end must be aligned to %d bytes' % INSN_SIZE)

    start_idx = start // INSN_SIZE
    end_idx = end // INSN_SIZE
    if end_idx > len(kernel.instructions):
        raise ValueError('end 0x%x beyond kernel (%d insns)' % (
            end, len(kernel.instructions)))

    lines = []
    n_raw = 0
    for idx in range(start_idx, end_idx):
        insn = kernel.instructions[idx]
        ctrl = insn.control
        stall = (ctrl >> 53) & 0x7

        # Emit control word as raw hex (exact round-trip),
        # plus human-readable stall/yield/reuse for editability
        lines.append('.ctrl 0x%016x' % ctrl)
        if stall != 0:
            lines.append('.stall %d' % stall)
        if ctrl & (1 << 4):
            lines.append('.yield')
        if REUSE_CTRL_MASK and (ctrl & REUSE_CTRL_MASK):
            lines.append('.reuse')

        if insn.mnemonic is None:
            # No xref — emit as raw hex comment + NOP placeholder
            lines.append('    NOP  # RAW: enc=%016x ctrl=%016x @0x%x' % (
                insn.encoding, ctrl, insn.offset))
            n_raw += 1
            continue

        # Mnemonic already has guard prefix from xref (e.g., "@P0 HADD2.BF16_V2")
        mnemonic = insn.mnemonic
        operands = insn.operands or ''

        if operands:
            lines.append('    %s %s' % (mnemonic, operands))
        else:
            lines.append('    %s' % mnemonic)

    if n_raw:
        sys.stderr.write('WARNING: %d/%d instructions without xref (emitted as NOP + RAW comment)\n' % (
            n_raw, end_idx - start_idx))

    return '\n'.join(lines) + '\n'


def verify_asm_region(kernel, start, end, asm_text):
    """Assemble text and compare against binary region [start, end).

    Returns (n_match, n_mismatch, details) where details is a list of
    mismatch descriptions.
    """
    binary, labels, instructions = assemble(asm_text, base_pc=start)
    n_asm = len(instructions)
    n_slots = (end - start) // INSN_SIZE
    start_idx = start // INSN_SIZE

    n_match = 0
    n_mismatch = 0
    details = []

    # Compare instruction-by-instruction (up to min of assembled and region)
    n_compare = min(n_asm, n_slots)
    for i in range(n_compare):
        asm_enc = struct.unpack_from('<Q', binary, i * INSN_SIZE)[0]
        asm_ctrl = struct.unpack_from('<Q', binary, i * INSN_SIZE + 8)[0]
        bin_insn = kernel.instructions[start_idx + i]

        enc_ok = asm_enc == bin_insn.encoding
        ctrl_ok = asm_ctrl == bin_insn.control

        if enc_ok and ctrl_ok:
            n_match += 1
        else:
            n_mismatch += 1
            addr = start + i * INSN_SIZE
            d = '[%04x]' % addr
            if not enc_ok:
                d += ' enc: asm=%016x bin=%016x' % (asm_enc, bin_insn.encoding)
            if not ctrl_ok:
                d += ' ctrl: asm=%016x bin=%016x' % (asm_ctrl, bin_insn.control)
            mnem = bin_insn.mnemonic or '?'
            d += '  %s' % mnem
            details.append(d)

    # Extra assembled instructions beyond region
    if n_asm > n_slots:
        details.append('assembled %d instructions but region has %d slots' % (
            n_asm, n_slots))
        n_mismatch += n_asm - n_slots

    # Remaining slots in region not covered by assembly
    if n_asm < n_slots:
        for i in range(n_asm, n_slots):
            bin_insn = kernel.instructions[start_idx + i]
            if bin_insn.encoding != NOP_ENCODING:
                n_mismatch += 1
                addr = start + i * INSN_SIZE
                details.append('[%04x] extra binary insn (not NOP): %s' % (
                    addr, bin_insn.mnemonic or '?'))
            else:
                n_match += 1

    return n_match, n_mismatch, details


def assemble(text, base_pc=0):
    """Assemble multi-line SASS text into bytes.

    Returns (bytes, labels, instructions) where bytes is the packed binary,
    labels is the label dict, and instructions is the parsed instruction list.
    """
    _load_asm_tables()
    instructions, labels = parse_asm(text)

    # Two-pass: first pass to resolve labels, second to encode
    # Convert label indices to PC addresses for branch encoding
    label_pcs = {name: base_pc + idx * INSN_SIZE for name, idx in labels.items()}

    # Encode all instructions
    encoded = []
    for i, insn in enumerate(instructions):
        pc = base_pc + i * INSN_SIZE
        enc, ctrl = encode_instruction(insn, pc, label_pcs)
        encoded.append((enc, ctrl))

    # Pack into bytes
    buf = bytearray(len(encoded) * INSN_SIZE)
    for i, (enc, ctrl) in enumerate(encoded):
        struct.pack_into('<Q', buf, i * INSN_SIZE, enc)
        struct.pack_into('<Q', buf, i * INSN_SIZE + 8, ctrl)

    return bytes(buf), labels, instructions


def cmd_assemble(args):
    """Assemble SASS text file into binary or patch into cubin."""
    with open(args.asm, 'r') as f:
        text = f.read()

    base_pc = int(args.base_pc, 0) if args.base_pc else 0
    binary, labels, instructions = assemble(text, base_pc)

    print('Assembled %d instructions (%d bytes)' % (len(instructions), len(binary)))
    if labels:
        print('Labels:')
        for name, idx in sorted(labels.items()):
            print('  %s → insn %d (pc=0x%x)' % (name, idx, base_pc + idx * INSN_SIZE))

    # Dump each instruction
    for i, insn in enumerate(instructions):
        pc = base_pc + i * INSN_SIZE
        enc = struct.unpack_from('<Q', binary, i * INSN_SIZE)[0]
        ctrl = struct.unpack_from('<Q', binary, i * INSN_SIZE + 8)[0]
        guard = ''
        if insn.has_guard:
            u = 'U' if insn.guard_uniform else ''
            neg = '!' if insn.guard_neg else ''
            if insn.guard_reg == 7:
                guard = '@%s%s%sPT ' % (neg, u, '')
            else:
                guard = '@%s%sP%d ' % (neg, u, insn.guard_reg)
        ops = ', '.join(repr(op) for op in insn.operands)
        stall = (ctrl >> 53) & 0x7
        print('  [%04x] %016x %016x  stall=%d  %s%s %s' % (
            pc, enc, ctrl, stall, guard, insn.mnemonic, ops))

    if args.output:
        with open(args.output, 'wb') as f:
            f.write(binary)
        print('Written to %s' % args.output)


def cmd_disasm(args):
    """Disassemble cubin region to assembler-format text."""
    ed = CubinEditor(args.cubin)
    k = ed.find_kernel(args.kernel)
    if not k:
        print('Kernel not found')
        if ed.kernels:
            print('Available: %s' % ', '.join(kk.short_name for kk in ed.kernels))
        sys.exit(1)

    if not args.sass:
        print('--sass required for disassembly (provides mnemonic cross-reference)')
        sys.exit(1)

    n = ed.xref(args.sass, k)
    sys.stderr.write('Cross-referenced %d/%d instructions\n' % (n, k.n_insns))

    start = int(args.start, 0) if args.start else 0
    end = int(args.end, 0) if args.end else k.n_insns * INSN_SIZE

    text = disassemble_region(k, start, end)

    if args.output:
        Path(args.output).write_text(text)
        n_insns = (end - start) // INSN_SIZE
        print('Disassembled %d instructions [0x%x, 0x%x) -> %s' % (
            n_insns, start, end, args.output))
    else:
        print(text, end='')


def cmd_verify_asm(args):
    """Verify assembly text matches cubin binary region."""
    ed = CubinEditor(args.cubin)
    k = ed.find_kernel(args.kernel)
    if not k:
        print('Kernel not found')
        if ed.kernels:
            print('Available: %s' % ', '.join(kk.short_name for kk in ed.kernels))
        sys.exit(1)

    if args.sass:
        n = ed.xref(args.sass, k)
        sys.stderr.write('Cross-referenced %d/%d instructions\n' % (n, k.n_insns))

    start = int(args.start, 0)
    end = int(args.end, 0)
    asm_text = Path(args.asm).read_text()

    n_match, n_mismatch, details = verify_asm_region(k, start, end, asm_text)
    total = n_match + n_mismatch

    print('Verified [0x%x, 0x%x): %d/%d match (%.1f%%)' % (
        start, end, n_match, total, 100 * n_match / total if total else 0))

    if details:
        print('Mismatches:')
        for d in details[:50]:
            print('  %s' % d)
        if len(details) > 50:
            print('  ... and %d more' % (len(details) - 50))

    sys.exit(0 if n_mismatch == 0 else 1)


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
        if args.asm:
            if not args.start or not args.end:
                print('--asm requires --start and --end')
                sys.exit(1)
            asm_start = int(args.start, 0)
            asm_end = int(args.end, 0)
            asm_text = Path(args.asm).read_text()
            n_patched, n_nops = ed.patch_region_asm(k, asm_start, asm_end, asm_text)
            print('Assembled %d instructions + %d NOPs into [0x%x, 0x%x)' % (
                n_patched, n_nops, asm_start, asm_end))
            n_ops += 1
        if args.script:
            n_ops += parse_script(args.script, ed, k)
        if args.stall:
            addr, val = int(args.stall[0], 0), int(args.stall[1], 0)
            ed.patch_stall(k, addr, val)
            n_ops += 1

        if n_ops == 0 and not args.identity:
            print('No edits specified (use --script, --stall, --asm, or --identity)')
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

    # opcode-table
    s = sub.add_parser('opcode-table',
                       help='Extract opcode table via XOR analysis across cubins')
    s.add_argument('--cubin', default=None, help='Single cubin file')
    s.add_argument('--sass', default=None, help='SASS dump for single cubin')
    s.add_argument('cubin_sass', nargs='*', metavar='CUBIN:SASS',
                   help='cubin:sass pairs (e.g., fc2.cubin:sass/fc2.txt)')
    s.add_argument('--export', '-e', default=None,
                   help='Export as Python module (e.g., tools/sm100a_opcodes.py)')
    s.add_argument('--all', '-a', action='store_true',
                   help='Show single-instance entries too')

    # analyze-encoding
    s = sub.add_parser('analyze-encoding',
                       help='Phase 2+3: Operand encoding + control word analysis')
    s.add_argument('--cubin', default=None, help='Single cubin file')
    s.add_argument('--sass', default=None, help='SASS dump for single cubin')
    s.add_argument('cubin_sass', nargs='*', metavar='CUBIN:SASS',
                   help='cubin:sass pairs (e.g., fc2.cubin:sass/fc2.txt)')
    s.add_argument('--export', '-e', default=None,
                   help='Export as Python module (e.g., tools/sm100a_encoding.py)')

    # assemble
    s = sub.add_parser('assemble', help='Assemble SASS text to binary')
    s.add_argument('asm', help='Assembly source file')
    s.add_argument('-o', '--output', default=None, help='Output binary file')
    s.add_argument('--base-pc', default='0', help='Base PC offset (hex, default 0)')

    # disasm
    s = sub.add_parser('disasm', help='Disassemble cubin region to assembler text')
    s.add_argument('cubin')
    s.add_argument('--sass', required=True, help='SASS dump for mnemonic cross-reference')
    s.add_argument('--start', '-s', default=None, help='Start address (hex)')
    s.add_argument('--end', '-e', default=None, help='End address (hex)')
    s.add_argument('--kernel', '-k', default=None)
    s.add_argument('-o', '--output', default=None, help='Output assembly file')

    # verify-asm
    s = sub.add_parser('verify-asm',
                       help='Verify assembly text matches cubin binary region')
    s.add_argument('cubin')
    s.add_argument('--asm', required=True, help='Assembly source file')
    s.add_argument('--start', '-s', required=True, help='Region start address (hex)')
    s.add_argument('--end', '-e', required=True, help='Region end address (hex)')
    s.add_argument('--sass', default=None, help='SASS dump for mnemonic xref in output')
    s.add_argument('--kernel', '-k', default=None)

    # fatbin-patch
    s = sub.add_parser('fatbin-patch', help='Patch embedded cubin in compiled CUDA binary')
    s.add_argument('binary', help='Host binary (e.g., fc2)')
    s.add_argument('--sass', required=True, help='SASS dump for cross-reference')
    s.add_argument('--script', default=None, help='Edit script file')
    s.add_argument('--stall', nargs=2, default=None, metavar=('ADDR', 'VAL'),
                   help='Patch single stall count')
    s.add_argument('--asm', default=None, help='Assembly source file to patch into region')
    s.add_argument('--start', default=None, help='Region start address (hex) for --asm')
    s.add_argument('--end', default=None, help='Region end address (hex) for --asm')
    s.add_argument('-o', '--output', required=True, help='Output patched binary')
    s.add_argument('--kernel', '-k', default=None)
    s.add_argument('--cubin-index', type=int, default=0, help='Which cubin (0=largest)')
    s.add_argument('--force', '-f', action='store_true')
    s.add_argument('--identity', action='store_true',
                   help='Identity round-trip (zero edits) — tests pipeline only')

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
        'opcode-table': cmd_opcode_table,
        'analyze-encoding': cmd_analyze_encoding,
        'schedule': cmd_schedule,
        'assemble': cmd_assemble,
        'disasm': cmd_disasm,
        'verify-asm': cmd_verify_asm,
        'fatbin-patch': cmd_fatbin_patch,
    }
    cmds[args.cmd](args)


if __name__ == '__main__':
    main()
