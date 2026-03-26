#!/usr/bin/env python3
"""
SM100a SASS template emitter for FC2 epilogue.

Generates SASS assembly text for two FC2 epilogue variants:
  A) 'current' — IS1 inline TMA stores, per-warp staging, WARPSYNC
  B) 'barsync' — CUTLASS-style BAR.SYNC sub-iteration pipelining

Output: .s assembly files processable by sass_edit.py assembler.

Usage:
  python3 tools/template_emitter.py --variant current -o templates/fc2_epi_current.s
  python3 tools/template_emitter.py --variant barsync -o templates/fc2_epi_barsync.s
  python3 tools/template_emitter.py --variant both    # emit both to templates/
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# FC2 kernel constants (base config: NS5, IS1, BIAS_SMEM, TMA_RESIDUAL=1)
# ---------------------------------------------------------------------------

TN = 256                          # N-dimension tile size
COLS_PER_CHUNK = 32               # TMEM load width (LDTM.x32)
COLS_PER_REGION = 64              # SWIZZLE_128B region = 64 BF16 cols
STAGING_REGION_BYTES = 4096       # 32 rows × 128 bytes
STAGING_REGION_ROW_BYTES = 128    # 64 BF16 cols = 128 bytes per row
NUM_EPI_WARPS = 4                 # epilogue warps (W2-W5)
EPI_THREADS = NUM_EPI_WARPS * 32  # 128 threads total for BAR.SYNC

# Number of 32-col chunks and 64-col sub-iterations per half (128 cols)
CHUNKS_PER_HALF = 4               # 128 / 32
SUB_ITERS_PER_HALF = 2            # 128 / 64
HALVES = 2                        # 256 / 128

# Default control words (from sm100a_encoding.py CONTROL_DEFAULTS)
CTRL = {
    'LDTM':    '0x000ea200082c0000',
    'LDS':     '0x000fe20000000c00',
    'F2FP':    '0x000fe200000010ff',
    'HFMA2':   '0x000fe20000200000',
    'HADD2':   '0x000fe20000200000',
    'STS':     '0x0003e20000000c00',
    'NOP':     '0x000fe20000000000',
    'WARPSYNC': '0x000fea0003800000',
    'MEMBAR':  '0x0005ec0000008000',
    'FENCE':   '0x004ea20000000000',
    'DEPBAR':  '0x000fc80000000000',
    'BAR':     '0x000fe20000010000',
    'BSSY':    '0x000fe20003800200',
    'BSYNC':   '0x004fea0003800200',
    'BRA':     '0x000fea0003800000',
    'ISETP':   '0x000fe20003f04070',
    'IMAD':    '0x000fe200078e00ff',
    'IADD3':   '0x000fe200078e00ff',
    'R2UR':    '0x000fe200000e0000',
    'PLOP3':   '0x000fe20003f4f070',
    'S2R':     '0x000e620000008800',
    'UTMASTG': '0x0005e20008008000',
    'UTMALDG': '0x0003e40008008000',
    'SYNCS':   '0x0007e200080004ff',
    'ELECT':   '0x000fda0003840000',
    'MOV':     '0x000fe20000000f00',
    'VIADD':   '0x000fe20000000000',
    'LOP3':    '0x000fe200078ec0ff',
    'SHF':     '0x000fce00000006ff',
    'LDCU':    '0x008ea20008000a00',
    'UTMACMDFLUSH': '0x0001e20000000000',
    'UIADD3':  '0x000fc8000ff1e0ff',
    'CCTL':    '0x000fe20002000000',
}

def ctrl(family, stall=None, yld=False):
    """Return .ctrl directive string, optionally overriding stall/yield."""
    base = int(CTRL.get(family, CTRL['NOP']), 16)
    if stall is not None:
        base = (base & ~0xF) | (stall & 0xF)
    if yld:
        base |= 0x10
    return f'.ctrl 0x{base:016x}'


# ---------------------------------------------------------------------------
# Register allocator — assigns physical registers to virtual groups
# ---------------------------------------------------------------------------

@dataclass
class RegGroup:
    """A virtual register group requiring allocation."""
    name: str                    # e.g., 'tmem', 'cvt', 'bias'
    count: int                   # number of registers
    consecutive: bool = True     # must be consecutive
    alignment: int = 1           # start must be aligned (e.g., 4 for LDS.128)
    physical: int = -1           # assigned start register (-1 = unassigned)
    fixed: bool = False          # True = don't move (e.g., TMEM at R4)


class RegisterAllocator:
    """Linear-scan register allocator for epilogue templates.

    Assigns physical registers to virtual groups, respecting:
    - Consecutive constraints (LDTM=32, LDS.128=4, STS.128=4)
    - Blacklist (live-in registers from kernel boundary)
    - Budget (max register number must stay under kernel limit)

    FC2 epilogue live-in registers (from SASS analysis):
      R0, R53, R54, R79, R81-R83, R88-R91         — tile/loop control + pre-loaded data
      R114-R115, R117-R122, R126-R135, R147, R149  — SMEM/TMA addresses
      UR6, UR8-UR10                                 — TMEM column, TMA descriptors
      P0, P3, P4, P5                                — predicates
    """

    # Default blacklist: registers live-in at FC2 epilogue entry.
    # The allocator won't assign these to any virtual group.
    FC2_LIVE_IN = frozenset([
        0,                          # tile counter
        53, 54,                     # STS column offsets
        79,                         # TMA descriptor
        81, 82, 83,                 # pre-loaded bias
        88, 89, 90, 91,             # pre-loaded residual
        114, 115, 117, 118, 119,    # LDS source addresses
        120, 121, 122,              # LDS source addresses
        126, 127, 128, 129, 130,    # tile state / SMEM base
        131, 132, 134, 135,         # mbar / TMA addresses
        147, 149,                   # TMEM offsets
    ])

    def __init__(self, budget: int = 207, blacklist: Optional[frozenset] = None):
        self.budget = budget
        self.blacklist = blacklist if blacklist is not None else self.FC2_LIVE_IN
        self.groups: List[RegGroup] = []

    def add_group(self, name: str, count: int, consecutive: bool = True,
                  alignment: int = 1, fixed_at: int = -1) -> 'RegisterAllocator':
        """Add a virtual register group to allocate."""
        g = RegGroup(name=name, count=count, consecutive=consecutive,
                     alignment=alignment,
                     physical=fixed_at, fixed=(fixed_at >= 0))
        self.groups.append(g)
        return self

    def allocate(self) -> dict:
        """Run allocation. Returns {group_name: start_register}.

        Strategy: allocate largest groups first (greedy).
        Fixed groups are placed first, then free groups fill remaining space.
        """
        # Collect already-occupied registers
        occupied = set(self.blacklist)

        # Place fixed groups first
        result = {}
        for g in self.groups:
            if g.fixed:
                regs = set(range(g.physical, g.physical + g.count))
                conflict = regs & occupied
                if conflict:
                    raise ValueError(
                        f'Fixed group {g.name} at R{g.physical} conflicts with '
                        f'occupied regs: {sorted(conflict)}')
                occupied |= regs
                result[g.name] = g.physical

        # Sort remaining groups by size (largest first for best packing)
        free_groups = sorted(
            [g for g in self.groups if not g.fixed],
            key=lambda g: -g.count
        )

        for g in free_groups:
            start = self._find_slot(g.count, g.alignment, occupied)
            if start is None:
                raise ValueError(
                    f'Cannot allocate {g.name} ({g.count} regs): '
                    f'no contiguous slot available under budget {self.budget}')
            regs = set(range(start, start + g.count))
            occupied |= regs
            g.physical = start
            result[g.name] = start

        return result

    def _find_slot(self, count: int, alignment: int, occupied: set) -> Optional[int]:
        """Find first available contiguous slot of `count` registers."""
        for start in range(0, self.budget - count + 1, alignment):
            slot = set(range(start, start + count))
            if not (slot & occupied):
                return start
        return None

    def build_regmap(self) -> 'RegMap':
        """Run allocation and return a RegMap with physical assignments."""
        alloc = self.allocate()
        rm = RegMap()
        # Map group names to RegMap fields
        field_map = {
            'tmem': 'TMEM_BASE',
            'cvt': 'CVT_BASE',
            'bias': 'BIAS_BASE',
            'res': 'RES_BASE',
            'sw': 'SW_BASE',
            'srow': 'SROW',
            'rs_base': 'RS_BASE',
            'lane': 'LANE',
            'xor_val': 'XOR_VAL',
            'bias_saddr': 'BIAS_SADDR',
            'stg_saddr': 'STG_SADDR',
            'res_stg_saddr': 'RES_STG_SADDR',
            'n_start': 'N_START',
            'gm_base': 'GM_BASE',
            'tmem_col_lo': 'TMEM_COL_LO',
            'tmem_col_hi': 'TMEM_COL_HI',
            'tma_src': 'TMA_SRC',
            'tma_n': 'TMA_N',
            'temp0': 'TEMP0',
            'temp1': 'TEMP1',
            'temp2': 'TEMP2',
            'sts_addr': 'STS_ADDR_BASE',
            'epi_mbar': 'EPI_MBAR',
            'res_mbar': 'RES_MBAR',
            'res_consumed_mbar': 'RES_CONSUMED_MBAR',
            'chunk_counter': 'CHUNK_COUNTER',
            'pass_counter': 'PASS_COUNTER',
        }
        for gname, start in alloc.items():
            if gname in field_map:
                setattr(rm, field_map[gname], start)
        return rm

    def print_allocation(self):
        """Print the register allocation map."""
        alloc = {g.name: (g.physical, g.count) for g in self.groups}
        print('Register allocation:')
        for name, (start, count) in sorted(alloc.items(), key=lambda x: x[1][0]):
            end = start + count - 1
            if count == 1:
                print(f'  R{start:3d}         {name}')
            else:
                print(f'  R{start:3d}-R{end:<3d}    {name} ({count} regs)')
        max_reg = max(s + c - 1 for s, c in alloc.values())
        print(f'  Max register: R{max_reg} (budget: R{self.budget - 1})')
        used = sum(c for _, c in alloc.values())
        print(f'  Used: {used} regs, Blacklisted: {len(self.blacklist)} regs')


def default_allocator(budget: int = 207) -> RegisterAllocator:
    """Create allocator with FC2 epilogue register groups."""
    ra = RegisterAllocator(budget=budget)
    # Large consecutive groups
    ra.add_group('tmem', 32, consecutive=True, fixed_at=4)  # LDTM.x32 R4
    ra.add_group('cvt', 16, consecutive=True, alignment=4)
    ra.add_group('bias', 16, consecutive=True, alignment=4)
    ra.add_group('res', 16, consecutive=True, alignment=4)
    ra.add_group('sw', 8, consecutive=True)
    ra.add_group('sts_addr', 4, consecutive=True, alignment=4)
    # Scalar groups (1 reg each)
    for name in ('srow', 'rs_base', 'lane', 'xor_val', 'bias_saddr',
                 'stg_saddr', 'res_stg_saddr', 'n_start', 'gm_base',
                 'tmem_col_lo', 'tmem_col_hi', 'tma_src', 'tma_n',
                 'temp0', 'temp1', 'temp2',
                 'epi_mbar', 'res_mbar', 'res_consumed_mbar',
                 'chunk_counter', 'pass_counter'):
        ra.add_group(name, 1, consecutive=False)
    return ra


# ---------------------------------------------------------------------------
# Register map — physical register assignments (filled by allocator or defaults)
# ---------------------------------------------------------------------------

@dataclass
class RegMap:
    """Physical register assignment for the FC2 epilogue template.

    Matches the interface registers from the compiled kernel at the
    epilogue entry point, but uses a clean internal allocation for
    compute temporaries.
    """
    # TMEM load destination (32 consecutive, hardware constraint)
    TMEM_BASE: int = 4           # R4-R35

    # Compute temporaries per chunk (reused each chunk)
    CVT_BASE: int = 36           # R36-R51: 16 F2FP outputs / HFMA2+HADD2 accumulators
    BIAS_BASE: int = 52          # R52-R67: 4×LDS.128 bias (16 regs)
    RES_BASE: int = 68           # R68-R83: 4×LDS.128 residual (16 regs)

    # Swizzle offsets (precomputed in preamble, persistent)
    SW_BASE: int = 84            # R84-R91: sw0..sw7 (8 swizzle offsets)

    # Per-lane staging addresses (persistent)
    SROW: int = 92               # output staging row base (srow_base)
    RS_BASE: int = 93            # residual staging row base (rs)

    # Scalar params (set by preamble or passed from caller)
    LANE: int = 94               # lane ID
    XOR_VAL: int = 95            # swizzle XOR value = (lane & 7) << 4
    BIAS_SADDR: int = 96         # bias SMEM base address
    STG_SADDR: int = 97          # staging SMEM base for this warp
    RES_STG_SADDR: int = 98      # residual staging SMEM base
    N_START: int = 99            # global N column offset
    GM_BASE: int = 100           # global M row offset
    TMEM_COL_LO: int = 101       # TMEM column base (low 32 bits of taddr_base)
    TMEM_COL_HI: int = 102       # TMEM column base (high, with row info)

    # TMA store address temporaries
    TMA_SRC: int = 103           # SMEM source for TMA store
    TMA_N: int = 104             # N coordinate for TMA store
    TEMP0: int = 105             # general scratch
    TEMP1: int = 106
    TEMP2: int = 107

    # STS destination addresses (precomputed per sub-iter)
    STS_ADDR_BASE: int = 108     # R108-R111: 4 STS dest addrs per chunk

    # Additional persistent addresses
    EPI_MBAR: int = 112          # epilogue mbar address
    RES_MBAR: int = 113          # residual TMA mbar address
    RES_CONSUMED_MBAR: int = 114 # residual consumed mbar address
    CHUNK_COUNTER: int = 115     # loop counter for sub-iterations
    PASS_COUNTER: int = 116      # pass counter

    # Uniform registers
    UR_TMEM_COL: int = 6         # UR6: TMEM column offset
    UR_TMEM_COL_HI: int = 7     # UR7: TMEM column high
    UR_TMA_C0: int = 8          # UR8-UR9: TMA output descriptor
    UR_TMA_C1: int = 9
    UR_TMA_RES0: int = 10       # UR10-UR11: TMA residual descriptor
    UR_TMA_RES1: int = 11

    def tmem(self, i):
        """TMEM register i (0-31)."""
        return self.TMEM_BASE + i

    def cvt(self, i):
        """Compute register i (0-15)."""
        return self.CVT_BASE + i

    def bias(self, i):
        """Bias register i (0-15)."""
        return self.BIAS_BASE + i

    def res(self, i):
        """Residual register i (0-15)."""
        return self.RES_BASE + i

    def sw(self, i):
        """Swizzle offset register i (0-7)."""
        return self.SW_BASE + i

    def sts_addr(self, i):
        """STS destination address register i (0-3)."""
        return self.STS_ADDR_BASE + i

    @property
    def max_reg(self):
        """Highest register number used."""
        return max(self.PASS_COUNTER, self.sts_addr(3))


# ---------------------------------------------------------------------------
# Instruction emitters — each returns list of assembly lines
# ---------------------------------------------------------------------------

def emit_ldtm_x32(rm: RegMap, col_offset: int) -> List[str]:
    """LDTM.x32: load 32 FP32 cols from TMEM into R[TMEM_BASE:+32]."""
    return [
        ctrl('LDTM'),
        f'    LDTM.x32 R{rm.tmem(0)}, tmem[UR{rm.UR_TMEM_COL}+0x{col_offset:x}]',
    ]


def emit_tmem_wait() -> List[str]:
    """tcgen05.wait — wait for TMEM load to complete.
    Encoded as DEPBAR in SASS with specific scoreboard."""
    return [
        ctrl('DEPBAR', stall=15, yld=True),
        '    DEPBAR.LE SB0, 0x0',
    ]


def emit_lds_bias(rm: RegMap, chunk_col: int) -> List[str]:
    """4× LDS.128: load 32 BF16 bias values (64 bytes) from SMEM.

    chunk_col: column offset within the tile (0, 32, 64, ..., 224)
    Bias is linear in SMEM (no swizzle): bias_saddr + col * 2.
    """
    lines = []
    byte_offset = chunk_col * 2  # BF16 = 2 bytes per element
    for g in range(4):
        dst = rm.bias(g * 4)
        off = byte_offset + g * 16  # 16 bytes = 8 BF16 values per LDS.128
        lines.append(ctrl('LDS'))
        lines.append(f'    LDS.128 R{dst}, [R{rm.BIAS_SADDR}+0x{off:x}]')
    return lines


def emit_lds_residual(rm: RegMap, half: int) -> List[str]:
    """4× LDS.128: load 32 BF16 residual values from staging SMEM.

    half: 0 = first 32 cols of region (sw0-sw3), 1 = second 32 cols (sw4-sw7)
    Uses precomputed swizzle offsets in R[SW_BASE+half*4 : +4].
    """
    lines = []
    sw_start = half * 4  # sw0-sw3 for half=0, sw4-sw7 for half=1
    for g in range(4):
        dst = rm.res(g * 4)
        sw_reg = rm.sw(sw_start + g)
        lines.append(ctrl('LDS'))
        lines.append(f'    LDS.128 R{dst}, [R{rm.RS_BASE}+R{sw_reg}]')
    return lines


def emit_lds_residual_direct(rm: RegMap, half: int) -> List[str]:
    """4× LDS.128 with precomputed absolute addresses (no [Rb+Roff] form).

    Since LDS.128 uses [Rbase+imm], we need the swizzle offset baked into
    the base register. Use STS_ADDR registers as precomputed residual addrs.
    """
    lines = []
    sw_start = half * 4
    for g in range(4):
        dst = rm.res(g * 4)
        # We'll precompute rs+sw[g] into separate address regs
        # For now, use the swizzle register directly as an offset placeholder
        sw_off = f'0x{(g + sw_start) * 16:x}'  # placeholder
        lines.append(ctrl('LDS'))
        lines.append(f'    LDS.128 R{dst}, [R{rm.RS_BASE}+{sw_off}]')
    return lines


def emit_f2fp_block(rm: RegMap) -> List[str]:
    """16× F2FP.BF16.F32.PACK_AB: convert 32 FP32 TMEM values → 16 BF16x2.

    F2FP.BF16.F32.PACK_AB Rdst, Rsrc_hi, Rsrc_lo
    Packs two FP32 → one BF16x2.
    """
    lines = []
    for i in range(16):
        dst = rm.cvt(i)
        src_hi = rm.tmem(i * 2 + 1)
        src_lo = rm.tmem(i * 2)
        lines.append(ctrl('F2FP', stall=15, yld=True))
        lines.append(f'    F2FP.BF16.F32.PACK_AB R{dst}, R{src_hi}, R{src_lo}')
    return lines


def emit_hfma2_bias(rm: RegMap) -> List[str]:
    """16× HFMA2.BF16_V2: add bias to converted values.

    HFMA2.BF16_V2 Rd, Rsrc, 1, 1, Rbias  →  Rd = Rsrc * 1.0 + Rbias
    """
    lines = []
    for i in range(16):
        cvt = rm.cvt(i)
        b = rm.bias(i)
        lines.append(ctrl('HFMA2'))
        lines.append(f'    HFMA2.BF16_V2 R{cvt}, R{cvt}, 1, 1, R{b}')
    return lines


def emit_hadd2_residual(rm: RegMap) -> List[str]:
    """16× HADD2.BF16_V2: add residual to bias-adjusted values.

    HADD2.BF16_V2 Rd, Ra, Rb  →  Rd = Ra + Rb
    """
    lines = []
    for i in range(16):
        cvt = rm.cvt(i)
        r = rm.res(i)
        lines.append(ctrl('HADD2'))
        lines.append(f'    HADD2.BF16_V2 R{cvt}, R{cvt}, R{r}')
    return lines


def emit_sts_block(rm: RegMap, half: int) -> List[str]:
    """4× STS.128: store 16 BF16x2 values (64 bytes) to staging SMEM.

    Stores to 4 swizzled addresses within the staging region.
    half: 0 = first 32 cols (sw0-sw3), 1 = second (sw4-sw7).
    """
    lines = []
    sw_start = half * 4
    for g in range(4):
        src = rm.cvt(g * 4)
        sw_reg = rm.sw(sw_start + g)
        lines.append(ctrl('STS'))
        lines.append(f'    STS.128 [R{rm.SROW}+R{sw_reg}], R{src}')
    return lines


def emit_sts_block_imm(rm: RegMap, half: int) -> List[str]:
    """4× STS.128 with immediate offsets (since STS uses [Rbase+imm]).

    Uses precomputed STS address registers.
    """
    lines = []
    for g in range(4):
        src = rm.cvt(g * 4)
        addr = rm.sts_addr(g)
        lines.append(ctrl('STS'))
        lines.append(f'    STS.128 [R{addr}], R{src}')
    return lines


# ---------------------------------------------------------------------------
# Compute chunk: LDTM → LDS(bias) → LDS(res) → TMEM_WAIT → F2FP → HFMA2 → HADD2 → STS
# ---------------------------------------------------------------------------

def emit_compute_chunk(rm: RegMap, chunk_col: int, half: int,
                       tmem_col_offset: int, include_ldtm: bool = True) -> List[str]:
    """Emit one 32-col compute chunk (sequential: all F2FP, all HFMA2, all HADD2, all STS).

    chunk_col: absolute column within the tile (0..224 in steps of 32)
    half: 0 or 1 (which half of the 64-col region — determines swizzle set)
    tmem_col_offset: TMEM column offset for LDTM
    include_ldtm: False if LDTM was prefetched by the previous chunk
    """
    lines = []
    lines.append(f'# --- chunk: cols {chunk_col}-{chunk_col+31}, half={half} ---')

    if include_ldtm:
        lines.extend(emit_ldtm_x32(rm, tmem_col_offset))

    lines.extend(emit_lds_bias(rm, chunk_col))
    lines.extend(emit_lds_residual_direct(rm, half))
    lines.extend(emit_tmem_wait())
    lines.extend(emit_f2fp_block(rm))
    lines.extend(emit_hfma2_bias(rm))
    lines.extend(emit_hadd2_residual(rm))
    lines.extend(emit_sts_block_imm(rm, half))

    return lines


def emit_compute_chunk_interleaved(rm: RegMap, chunk_col: int, half: int,
                                   tmem_col_offset: int, include_ldtm: bool = True) -> List[str]:
    """Emit one 32-col compute chunk with INTERLEAVED STS.

    CUTLASS pattern: process 4 groups of 8 cols (4 BF16x2 each).
    Per group: 4 F2FP → 4 HFMA2(bias) → 4 HADD2(res) → 1 STS.128.
    STS fires after every 12 compute ops, hiding in the shadow of the next group.

    This is the key scheduling difference from Template A (where ptxas
    clusters all 4 STS at the end, creating a 4×32=128 cycle stall bubble).
    """
    lines = []
    lines.append(f'# --- chunk (interleaved): cols {chunk_col}-{chunk_col+31}, half={half} ---')

    if include_ldtm:
        lines.extend(emit_ldtm_x32(rm, tmem_col_offset))

    lines.extend(emit_lds_bias(rm, chunk_col))
    lines.extend(emit_lds_residual_direct(rm, half))
    lines.extend(emit_tmem_wait())

    # Process 4 groups of 4 BF16x2 (8 cols), interleave STS after each group
    for g in range(4):
        base = g * 4
        lines.append(f'# group {g}: cols {chunk_col + g*8}-{chunk_col + g*8 + 7}')

        # 4× F2FP for this group
        for j in range(4):
            i = base + j
            dst = rm.cvt(i)
            src_hi = rm.tmem(i * 2 + 1)
            src_lo = rm.tmem(i * 2)
            lines.append(ctrl('F2FP', stall=15, yld=True))
            lines.append(f'    F2FP.BF16.F32.PACK_AB R{dst}, R{src_hi}, R{src_lo}')

        # 4× HFMA2 (add bias)
        for j in range(4):
            i = base + j
            cvt_r = rm.cvt(i)
            b = rm.bias(i)
            lines.append(ctrl('HFMA2'))
            lines.append(f'    HFMA2.BF16_V2 R{cvt_r}, R{cvt_r}, 1, 1, R{b}')

        # 4× HADD2 (add residual)
        for j in range(4):
            i = base + j
            cvt_r = rm.cvt(i)
            r = rm.res(i)
            lines.append(ctrl('HADD2'))
            lines.append(f'    HADD2.BF16_V2 R{cvt_r}, R{cvt_r}, R{r}')

        # 1× STS.128 — store this group's 4 regs immediately
        src = rm.cvt(base)
        addr = rm.sts_addr(g)
        lines.append(ctrl('STS'))
        lines.append(f'    STS.128 [R{addr}], R{src}')

    return lines


# ---------------------------------------------------------------------------
# TMA store protocols
# ---------------------------------------------------------------------------

def emit_tma_store_is1(rm: RegMap, sub_iter_col: int) -> List[str]:
    """Template A: IS1 inline TMA store per 64-col region.

    WARPSYNC → FENCE → UTMASTG.2D (lane 0 only)
    """
    lines = [
        f'# --- IS1 TMA store: cols {sub_iter_col}-{sub_iter_col+63} ---',
        ctrl('WARPSYNC'),
        '    WARPSYNC.ALL',
        ctrl('FENCE'),
        '    FENCE.VIEW.ASYNC.S',
    ]
    # Lane 0 issues TMA store (guarded by @!P0 where P0 = lane != 0)
    lines.extend([
        ctrl('UTMASTG'),
        f'    @!UP0 UTMASTG.2D [UR{rm.UR_TMA_C0}], [UR{rm.UR_TMEM_COL}]',
    ])
    return lines


def emit_tma_store_barsync(rm: RegMap, sub_iter_col: int,
                           barrier_id: int = 1,
                           is_last: bool = False) -> List[str]:
    """Template B: CUTLASS-style BAR.SYNC pipelined TMA store.

    MEMBAR → FENCE → BAR.SYNC(epi_threads) → BSSY B0 → @P0 BRA skip
    → UTMASTG → DEPBAR → BSYNC → BAR.SYNC(epi_threads)

    BSSY sets reconvergence target so BSYNC can reconverge the divergent
    branch.  Target = instruction after BSYNC (the trailing BAR.SYNC or
    the next section), matching the CUTLASS pattern.

    barrier_id: hardware barrier ID (must not conflict with mainloop)
    is_last: if True, skip the trailing BAR.SYNC (no next sub-iter to protect)
    """
    thread_count = EPI_THREADS  # 128
    tc_hex = f'0x{thread_count:x}'

    # Label for BSSY reconvergence target = instruction after BSYNC
    reconverge_label = f'.L_barsync_done_{sub_iter_col}'

    lines = [
        f'# --- BAR.SYNC TMA store: cols {sub_iter_col}-{sub_iter_col+63} ---',
        # 1. Fence STS → visible to TMA engine
        ctrl('MEMBAR'),
        '    MEMBAR.ALL.CTA',
        ctrl('FENCE'),
        '    FENCE.VIEW.ASYNC.S',
        # 2. All epilogue warps sync (STS complete, staging readable by TMA)
        ctrl('BAR', stall=15),
        f'    BAR.SYNC.DEFER_BLOCKING 0x{barrier_id:x}, {tc_hex}',
        # 3. Set reconvergence point, then diverge: lane 0 → TMA, others → skip
        ctrl('BSSY'),
        f'    BSSY.RECONVERGENT B0, {reconverge_label}',
        ctrl('BRA'),
        f'    @P0 BRA .L_barsync_skip_{sub_iter_col}',
    ]

    # TMA store (lane 0 only — P0 is "lane != 0" set by preamble)
    lines.extend([
        ctrl('UTMASTG'),
        f'    UTMASTG.2D [UR{rm.UR_TMA_C0}], [UR{rm.UR_TMEM_COL}]',
        ctrl('UTMACMDFLUSH'),
        '    UTMACMDFLUSH',
    ])

    # 4. Wait for TMA to finish reading staging SMEM
    lines.extend([
        ctrl('DEPBAR'),
        '    DEPBAR.LE SB0, 0x1',
    ])

    # Reconverge — BSYNC pops the convergence barrier pushed by BSSY
    lines.extend([
        f'.L_barsync_skip_{sub_iter_col}:',
        ctrl('BSYNC'),
        '    BSYNC.RECONVERGENT B0',
    ])

    # 5. Staging SMEM now free — next sub-iter can overwrite
    # This is the BSSY reconvergence target
    lines.append(f'{reconverge_label}:')
    if not is_last:
        lines.extend([
            ctrl('BAR'),
            f'    BAR.SYNC.DEFER_BLOCKING 0x{barrier_id:x}, {tc_hex}',
        ])

    return lines


# ---------------------------------------------------------------------------
# Pass commit (end of 128-col pass)
# ---------------------------------------------------------------------------

def emit_pass_commit() -> List[str]:
    """Commit group for all TMA stores in this pass."""
    return [
        '# --- commit TMA stores ---',
        ctrl('NOP'),
        '    NOP',  # placeholder for cp.async.bulk.commit_group (inline PTX)
    ]


# ---------------------------------------------------------------------------
# Mbarrier protocol for residual self-load (Template A only)
# ---------------------------------------------------------------------------

def emit_mbar_self_load(rm: RegMap, pass_col: int) -> List[str]:
    """Self-load residual via TMA for the current pass.

    Lane 0: mbar_arrive_expect_tx → TMA load residual into staging SMEM.
    Then: mbar_wait for load completion.
    """
    lines = [
        f'# --- self-load residual for pass starting at col {pass_col} ---',
        ctrl('SYNCS'),
        f'    SYNCS.ARRIVE.TRANS64.RED RZ, [R{rm.RES_MBAR}+URZ], R{rm.TEMP0}',
        ctrl('UTMALDG'),
        f'    UTMALDG.2D [UR{rm.UR_TMA_RES0}], [UR{rm.UR_TMEM_COL}]',
    ]
    # For 128-col pass with 2 regions, issue second TMA load
    lines.extend([
        ctrl('UTMALDG'),
        f'    UTMALDG.2D [UR{rm.UR_TMA_RES0}], [UR{rm.UR_TMEM_COL}]',
    ])
    # mbar_wait
    lines.extend([
        ctrl('SYNCS', stall=15, yld=True),
        f'    SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [R{rm.RES_MBAR}+URZ], R{rm.TEMP1}',
    ])
    return lines


# ---------------------------------------------------------------------------
# Preamble: address computation (shared by both templates)
# ---------------------------------------------------------------------------

def emit_preamble(rm: RegMap) -> List[str]:
    """Epilogue preamble: compute swizzle offsets, staging addresses, lane ID.

    This runs once at the start of the epilogue. All address registers are
    persistent across chunks/sub-iters.
    """
    lines = [
        '# ===== EPILOGUE PREAMBLE =====',
        '# Compute swizzle offsets and staging addresses.',
        '# Inputs: R_lane, R_stg_saddr, R_bias_saddr, R_res_stg_saddr',
        '',
        '# xor_val = (lane & 7) << 4',
        ctrl('IMAD'),
        f'    IMAD.MOV.U32 R{rm.TEMP0}, RZ, RZ, 0x7',
        ctrl('LOP3'),
        f'    LOP3.LUT R{rm.XOR_VAL}, R{rm.LANE}, R{rm.TEMP0}, RZ, 0xc0, !PT',
        ctrl('SHF'),
        f'    SHF.L.U32 R{rm.XOR_VAL}, R{rm.XOR_VAL}, 0x4, RZ',
        '',
        '# Precompute 8 swizzle offsets: sw[i] = (i*16) ^ xor_val',
    ]
    for i in range(8):
        base_off = i * 16
        lines.extend([
            ctrl('IMAD'),
            f'    IMAD.MOV.U32 R{rm.TEMP0}, RZ, RZ, 0x{base_off:x}',
            ctrl('LOP3'),
            f'    LOP3.LUT R{rm.sw(i)}, R{rm.TEMP0}, R{rm.XOR_VAL}, RZ, 0x60, !PT',
        ])

    lines.extend([
        '',
        '# srow_base = staging_saddr + lane * 128',
        ctrl('IMAD'),
        f'    IMAD R{rm.SROW}, R{rm.LANE}, 0x80, R{rm.STG_SADDR}',
        '',
        '# rs_base = res_staging_saddr + lane * 128',
        ctrl('IMAD'),
        f'    IMAD R{rm.RS_BASE}, R{rm.LANE}, 0x80, R{rm.RES_STG_SADDR}',
        '',
        '# Wait for previous tile TMA stores',
        ctrl('DEPBAR', stall=15),
        '    DEPBAR.LE SB0, 0x0',
        ctrl('WARPSYNC'),
        '    WARPSYNC.ALL',
    ])
    return lines


# ---------------------------------------------------------------------------
# Precompute STS/LDS addresses for a sub-iteration
# ---------------------------------------------------------------------------

def emit_addr_setup_sub_iter(rm: RegMap, region_idx: int, half: int) -> List[str]:
    """Precompute STS destination and LDS residual source addresses.

    STS dest: srow_base + region_idx * STAGING_REGION_BYTES + sw[half*4+g]
    LDS res:  rs_base + region_idx * STAGING_REGION_BYTES + sw[half*4+g]
    """
    region_offset = region_idx * STAGING_REGION_BYTES
    sw_start = half * 4
    lines = [
        f'# --- address setup: region={region_idx}, half={half} ---',
    ]

    # STS addresses
    for g in range(4):
        sw_reg = rm.sw(sw_start + g)
        sts = rm.sts_addr(g)
        lines.extend([
            ctrl('IADD3'),
            f'    IADD3 R{sts}, PT, PT, R{rm.SROW}, 0x{region_offset:x}, R{sw_reg}',
        ])

    return lines


# ---------------------------------------------------------------------------
# Full epilogue emission
# ---------------------------------------------------------------------------

def emit_epilogue_current(rm: RegMap) -> List[str]:
    """Template A: current FC2 architecture.

    Structure: 2 passes × 128 cols. Within each pass:
      - Self-load residual via TMA (mbar protocol)
      - 4 chunks × 32 cols with IS1 inline TMA stores after each 64-col pair
      - commit_group at end of pass

    For simplicity, this emits one half (128 cols). The full epilogue
    calls this twice (NC_START=0 and NC_START=128).
    """
    lines = [
        '# =============================================================',
        '# Template A: FC2 epilogue — current architecture (IS1)',
        '# 128 cols: 4 chunks of 32, IS1 TMA store per 64-col region',
        '# =============================================================',
        '',
    ]
    lines.extend(emit_preamble(rm))
    lines.append('')

    # Process 4 chunks (128 cols), 2 sub-iterations of 64 cols
    for si in range(SUB_ITERS_PER_HALF):
        lines.append(f'')
        lines.append(f'# ===== SUB-ITERATION {si}: cols {si*64}-{si*64+63} =====')

        for chunk in range(2):  # 2 chunks per sub-iter
            half = chunk  # 0=first 32 cols, 1=second 32 cols of region
            chunk_col = si * 64 + chunk * 32
            tmem_offset = chunk_col  # TMEM column offset

            # Precompute addresses for this chunk
            lines.extend(emit_addr_setup_sub_iter(rm, si, half))
            lines.append('')

            # Emit compute chunk
            lines.extend(emit_compute_chunk(rm, chunk_col, half, tmem_offset))
            lines.append('')

        # IS1 TMA store after 64-col sub-iteration
        lines.extend(emit_tma_store_is1(rm, si * 64))
        lines.append('')

    # Commit all TMA stores
    lines.extend(emit_pass_commit())
    lines.append('')

    return lines


def emit_epilogue_barsync(rm: RegMap) -> List[str]:
    """Template B: BAR.SYNC pipelined epilogue.

    Structure: 4 sub-iterations × 64 cols. Each sub-iter:
      - 2 chunks × 32 cols compute
      - BAR.SYNC protocol: MEMBAR → FENCE → BAR.SYNC → UTMASTG → DEPBAR → BAR.SYNC
      - Staging buffer REUSED (not 4 separate regions)

    Key differences from Template A:
      - Cross-warp BAR.SYNC (128 threads) instead of per-warp WARPSYNC
      - Single staging region reused via BAR.SYNC ownership protocol
      - Creates synchronized idle gaps → reduces K-loop warp contention
    """
    lines = [
        '# =============================================================',
        '# Template B: FC2 epilogue — BAR.SYNC pipelined',
        '# 256 cols: 4 sub-iters of 64, BAR.SYNC staging ownership',
        '# Same compute core as Template A, different control flow.',
        '# =============================================================',
        '',
    ]
    lines.extend(emit_preamble(rm))
    lines.append('')

    num_sub_iters = TN // COLS_PER_REGION  # 4

    for si in range(num_sub_iters):
        is_last = (si == num_sub_iters - 1)
        lines.append(f'')
        lines.append(f'# ===== SUB-ITERATION {si}: cols {si*64}-{si*64+63} =====')

        for chunk in range(2):  # 2 chunks per 64-col sub-iter
            half = chunk
            chunk_col = si * 64 + chunk * 32
            tmem_offset = chunk_col

            # Region index = 0 always (single staging buffer, reused via BAR.SYNC)
            lines.extend(emit_addr_setup_sub_iter(rm, 0, half))
            lines.append('')

            # Use interleaved STS — the key difference from Template A
            lines.extend(emit_compute_chunk_interleaved(rm, chunk_col, half, tmem_offset))
            lines.append('')

        # BAR.SYNC TMA store protocol
        lines.extend(emit_tma_store_barsync(rm, si * 64,
                                            barrier_id=1, is_last=is_last))
        lines.append('')

    # Final commit
    lines.extend(emit_pass_commit())
    lines.append('')

    return lines


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def count_instructions(lines: List[str]) -> dict:
    """Count instruction types in assembly output."""
    counts = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or stripped.startswith('.'):
            continue
        # Remove predicate guard
        if stripped.startswith('@'):
            stripped = stripped.split(None, 1)[1] if ' ' in stripped else stripped
        # Get mnemonic
        mnemonic = stripped.split()[0] if stripped else ''
        # Normalize
        base = mnemonic.split('.')[0]
        counts[base] = counts.get(base, 0) + 1
    return counts


def print_stats(name: str, lines: List[str]):
    """Print instruction statistics."""
    counts = count_instructions(lines)
    total = sum(counts.values())
    print(f'\n{name}: {total} instructions')
    for mnemonic in sorted(counts, key=counts.get, reverse=True):
        print(f'  {mnemonic:20s} {counts[mnemonic]:4d}')


# ---------------------------------------------------------------------------
# CP-SAT scheduling integration
# ---------------------------------------------------------------------------

def schedule_template(lines: List[str], time_limit: float = 60.0,
                      verbose: bool = True) -> List[str]:
    """Schedule a template using CP-SAT within each barrier-delimited block.

    Takes emitted .s lines, assembles them, creates Instruction objects,
    partitions into barrier blocks, schedules each block, then reconstructs
    the .s text with optimized instruction order and stall counts.

    Returns: new list of .s lines with optimal scheduling.
    """
    from sass_edit import (
        assemble, Instruction, parse_reg_operands, schedule_cpsat,
        is_barrier, get_pipe, get_latency,
        INSN_SIZE, encode_instruction, decode_ctrl,
    )
    import struct

    text = '\n'.join(lines) + '\n'
    binary, labels, asm_insns = assemble(text, base_pc=0)

    if verbose:
        print(f'Assembled {len(asm_insns)} instructions for scheduling')

    # Build Instruction objects that the scheduler understands
    insns = []
    for i, ai in enumerate(asm_insns):
        pc = i * INSN_SIZE
        enc = struct.unpack_from('<Q', binary, pc)[0]
        ctl = struct.unpack_from('<Q', binary, pc + 8)[0]

        insn = Instruction(pc, enc, ctl)
        # Reconstruct mnemonic with predicate guard
        mn = ''
        if ai.guard_reg != 7 or ai.guard_neg:
            prefix = '@'
            if ai.guard_neg:
                prefix += '!'
            if ai.guard_uniform:
                prefix += 'UP'
            else:
                prefix += 'P'
            prefix += str(ai.guard_reg)
            mn = prefix + ' '
        mn += ai.mnemonic
        insn.mnemonic = mn

        # Reconstruct operands as comma-separated string
        op_strs = []
        for op in ai.operands:
            op_strs.append(_format_asm_operand(op))
        insn.operands = ', '.join(op_strs)

        insns.append(insn)

    # Build reverse label map: PC offset → label name
    label_pcs = {}
    for name, idx in labels.items():
        pc = idx * INSN_SIZE
        label_pcs[pc] = name

    # Partition into barrier-delimited blocks
    blocks = []
    current_block = []
    for i, insn in enumerate(insns):
        if is_barrier(insn.mnemonic):
            if current_block:
                blocks.append(('compute', current_block))
                current_block = []
            blocks.append(('barrier', [insn]))
        else:
            current_block.append(insn)
    if current_block:
        blocks.append(('compute', current_block))

    if verbose:
        compute_blocks = [(j, b) for j, (t, b) in enumerate(blocks) if t == 'compute']
        print(f'{len(blocks)} blocks: {len(compute_blocks)} compute, '
              f'{len(blocks) - len(compute_blocks)} barriers')
        for j, b in compute_blocks:
            print(f'  block {j}: {len(b)} insns')

    # Schedule each compute block independently
    scheduled_insns = []
    total_orig_cycles = 0
    total_opt_cycles = 0

    for block_type, block_insns in blocks:
        if block_type == 'barrier':
            scheduled_insns.extend(block_insns)
            continue

        if len(block_insns) <= 2:
            # Too small to schedule
            scheduled_insns.extend(block_insns)
            continue

        if verbose:
            print(f'\nScheduling block ({len(block_insns)} insns)...')

        orig_cycles = sum(i.control & 0xf for i in block_insns)
        total_orig_cycles += orig_cycles

        ordered_addrs, stall_counts, stats = schedule_cpsat(
            block_insns, time_limit=time_limit / max(1, len([b for t, b in blocks if t == 'compute'])),
            verbose=verbose
        )

        if ordered_addrs is None:
            # Scheduling failed — keep original order
            if verbose:
                print(f'  (keeping original order)')
            scheduled_insns.extend(block_insns)
            total_opt_cycles += orig_cycles
            continue

        # Reorder instructions
        addr_to_insn = {insn.offset: insn for insn in block_insns}
        for addr in ordered_addrs:
            insn = addr_to_insn[addr].clone()
            insn.control = (insn.control & ~0xF) | min(stall_counts.get(addr, 1), 15)
            scheduled_insns.append(insn)

        opt_cycles = stats.get('makespan', orig_cycles)
        total_opt_cycles += opt_cycles

    if verbose:
        print(f'\nTotal: {total_orig_cycles} → {total_opt_cycles} cycles '
              f'({total_orig_cycles - total_opt_cycles:+d})')

    # Reconstruct .s text from scheduled instructions
    out_lines = []
    for insn in scheduled_insns:
        # Emit label if this instruction had one
        if insn.offset in label_pcs:
            out_lines.append(f'{label_pcs[insn.offset]}:')
        ctl_val = insn.control
        out_lines.append(f'.ctrl 0x{ctl_val:016x}')
        mn = insn.mnemonic or 'NOP'
        ops = insn.operands or ''
        if ops:
            out_lines.append(f'    {mn} {ops}')
        else:
            out_lines.append(f'    {mn}')

    return out_lines


def _format_asm_operand(op) -> str:
    """Format an AsmOperand back to its text representation.

    AsmOperand has .type (str), .val (dict or int), .reuse (bool).
    """
    from sass_edit import (OP_GPR, OP_UR, OP_PRED, OP_UPRED, OP_IMM,
                           OP_MEM, OP_UMEM, OP_TMEM, OP_TMEM_UR,
                           OP_LABEL, OP_BARRIER, OP_DESC)

    kind = op.type
    val = op.val

    if kind == OP_GPR:
        reg = val if isinstance(val, int) else val.get('reg', 255)
        if reg == 255:
            return 'RZ'
        return f'R{reg}'

    if kind == OP_UR:
        reg = val if isinstance(val, int) else val.get('reg', 63)
        if reg == 63:
            return 'URZ'
        return f'UR{reg}'

    if kind == OP_PRED:
        reg = val if isinstance(val, int) else val.get('reg', 7)
        if reg == 7:
            return 'PT'
        return f'P{reg}'

    if kind == OP_UPRED:
        reg = val if isinstance(val, int) else val.get('reg', 7)
        if reg == 7:
            return 'UPT'
        return f'UP{reg}'

    if kind == OP_IMM:
        v = val if isinstance(val, int) else val.get('value', 0)
        if v < 0:
            return f'-0x{-v:x}'
        return f'0x{v:x}'

    if kind == OP_MEM:
        base = val.get('base', 255)
        offset = val.get('offset', 0)
        base_str = f'R{base}' if base != 255 else 'RZ'
        if offset > 0:
            return f'[{base_str}+0x{offset:x}]'
        elif offset < 0:
            return f'[{base_str}+-0x{-offset:x}]'
        return f'[{base_str}]'

    if kind == OP_UMEM:
        base = val.get('base', 63)
        offset = val.get('offset', 0)
        base_str = f'UR{base}' if base != 63 else 'URZ'
        if offset:
            return f'[{base_str}+0x{offset:x}]'
        return f'[{base_str}]'

    if kind == OP_TMEM:
        off = val.get('offset', 0)
        return f'[TMEM+0x{off:x}]'

    if kind == OP_TMEM_UR:
        reg = val.get('reg', 0)
        off = val.get('offset', 0)
        if off:
            return f'tmem[UR{reg}+0x{off:x}]'
        return f'tmem[UR{reg}]'

    if kind == OP_LABEL:
        name = val if isinstance(val, str) else val.get('name', '')
        return name

    if kind == OP_BARRIER:
        bid = val if isinstance(val, int) else val.get('id', 0)
        return f'B{bid}'

    if kind == OP_DESC:
        reg = val if isinstance(val, int) else val.get('reg', 0)
        return f'desc[UR{reg}]'

    # Fallback
    return str(val)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='FC2 epilogue SASS template emitter')
    parser.add_argument('--variant', choices=['current', 'barsync', 'both'],
                        default='both', help='Which variant to emit')
    parser.add_argument('-o', '--output', help='Output file (for single variant)')
    parser.add_argument('--stats', action='store_true', help='Print instruction statistics')
    parser.add_argument('--schedule', action='store_true',
                        help='Run CP-SAT scheduler on output (requires ortools)')
    parser.add_argument('--time-limit', type=float, default=60.0,
                        help='CP-SAT solver time limit in seconds (default 60)')
    parser.add_argument('--allocate', action='store_true',
                        help='Use register allocator (respects FC2 live-in blacklist)')
    parser.add_argument('--print-alloc', action='store_true',
                        help='Print register allocation map')
    parser.add_argument('--outdir', default='templates', help='Output directory (for --variant both)')
    args = parser.parse_args()

    if args.allocate:
        ra = default_allocator()
        rm = ra.build_regmap()
        if args.print_alloc:
            ra.print_allocation()
    else:
        rm = RegMap()

    variants = []
    if args.variant in ('current', 'both'):
        variants.append(('current', emit_epilogue_current(rm), 'fc2_epi_current.s'))
    if args.variant in ('barsync', 'both'):
        variants.append(('barsync', emit_epilogue_barsync(rm), 'fc2_epi_barsync.s'))

    for name, lines, default_fname in variants:
        if args.stats:
            print_stats(name, lines)

        if args.schedule:
            suffix = default_fname.replace('.s', '_scheduled.s')
            sched_lines = schedule_template(lines, time_limit=args.time_limit, verbose=True)
            sched_path = os.path.join(args.outdir, suffix)
            os.makedirs(args.outdir, exist_ok=True)
            with open(sched_path, 'w') as f:
                f.write('\n'.join(sched_lines) + '\n')
            print(f'Wrote {sched_path} ({len(sched_lines)} lines, scheduled)')

        if args.output and len(variants) == 1:
            outpath = args.output
        else:
            os.makedirs(args.outdir, exist_ok=True)
            outpath = os.path.join(args.outdir, default_fname)

        text = '\n'.join(lines) + '\n'
        with open(outpath, 'w') as f:
            f.write(text)
        print(f'Wrote {outpath} ({len(lines)} lines)')


if __name__ == '__main__':
    main()
