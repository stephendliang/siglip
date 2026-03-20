#!/usr/bin/env python3
"""SASS analysis for SM100a persistent GEMM kernels.

Parses cuobjdump --dump-sass output, segments kernels into functional regions
(prologue, K-loop, epilogue), produces compact analysis for LLM consumption.

Modes:
  list:     List GEMM kernels in a multi-kernel dump with tile/stage info
  analyze:  Section-level breakdown of a single kernel
  compare:  Side-by-side of two kernels (e.g. ours vs CUTLASS)

Usage:
  python3 sass_analysis.py list sass/cutlass_fc2_max.txt
  python3 sass_analysis.py analyze sass/fc2_ns5_is1_base.txt
  python3 sass_analysis.py analyze sass/cutlass_fc2_max.txt -k 0
  python3 sass_analysis.py compare sass/fc2_ns5_is1_base.txt sass/cutlass_fc2_max.txt
  python3 sass_analysis.py compare sass/fc2_best.txt sass/cutlass_fc2_max.txt -k 3
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# ═══════════════════════════════════════════════════════════════════════════════
# Opcode classification
# ═══════════════════════════════════════════════════════════════════════════════

OPCODE_CATS = {
    'UTCQMMA':   'mma',
    'UTCBAR':    'mma_commit',
    'LDTM':      'tmem_load',
    'STTM':      'tmem_store',
    'F2FP':      'cvt',
    'I2FP':      'cvt',
    'F2IP':      'cvt',
    'I2IP':      'cvt',
    'STS':       'sts',
    'LDS':       'lds',
    'STG':       'stg',
    'LDG':       'ldg',
    'R2UR':      'r2ur',
    'ELECT':     'elect',
    'PLOP3':     'plop3',
    'UPLOP3':    'plop3',
    'HFMA2':     'bf16_fma',
    'HADD2':     'bf16_add',
    'HMUL2':     'bf16_mul',
    'FMUL':      'fp32_mul',
    'FFMA':      'fp32_fma',
    'FADD':      'fp32_add',
    'MUFU':      'fp32_mufu',
    'IMAD':      'int_alu',
    'IADD3':     'int_alu',
    'LEA':       'int_alu',
    'ULEA':      'int_alu',
    'LOP3':      'int_alu',
    'SHF':       'int_alu',
    'PRMT':      'int_alu',
    'MOV':       'int_alu',
    'UMOV':      'int_alu',
    'SEL':       'int_alu',
    'USEL':      'int_alu',
    'ISETP':     'int_cmp',
    'ISET':      'int_cmp',
    'UISETP':    'int_cmp',
    'SYNCS':     'barrier',
    'BAR':       'barrier',
    'DEPBAR':    'barrier',
    'BSYNC':     'barrier',
    'BSSY':      'barrier',
    'WARPSYNC':  'barrier',
    'NANOSLEEP': 'nanosleep',
    'MEMBAR':    'barrier',
    'ERRBAR':    'barrier',
    'CGAERRBAR': 'barrier',
    'FENCE':     'fence',
    'BRA':       'branch',
    'EXIT':      'branch',
    'RET':       'branch',
    'BREAK':     'branch',
    'CALL':      'branch',
    'NOP':       'nop',
    'S2R':       'special_reg',
    'S2UR':      'special_reg',
    'CS2R':      'special_reg',
    'UIADD3':    'int_alu',
    'USHF':      'int_alu',
    'ULOP3':     'int_alu',
    'UIMAD':     'int_alu',
    'STL':       'local_store',
    'LDL':       'local_load',
    'LDSM':      'lds',
    'STSM':      'sts',
}

# For display: group fine categories into summary categories
DISPLAY_GROUPS = [
    ('MMA',         ['mma']),
    ('MMA commit',  ['mma_commit']),
    ('TMEM load',   ['tmem_load']),
    ('R2UR',        ['r2ur']),
    ('ELECT',       ['elect']),
    ('BF16 compute',['bf16_fma', 'bf16_add', 'bf16_mul']),
    ('FP32 compute',['fp32_mul', 'fp32_fma', 'fp32_add', 'fp32_mufu']),
    ('CVT (F2FP)',  ['cvt']),
    ('STS',         ['sts']),
    ('LDS',         ['lds']),
    ('LDG',         ['ldg']),
    ('Barrier/sync',['barrier', 'nanosleep', 'fence']),
    ('Int ALU',     ['int_alu', 'int_cmp']),
    ('Branch/ctrl', ['branch', 'nop']),
    ('Other',       ['plop3', 'stg', 'special_reg', 'local_store',
                     'local_load', 'tmem_store', 'other']),
]


def cat_of(opcode):
    """Classify an opcode string into a category."""
    base = opcode.split('.')[0]
    return OPCODE_CATS.get(base, 'other')

# ═══════════════════════════════════════════════════════════════════════════════
# Instruction / Kernel data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Instr:
    addr: int
    opcode: str
    full_line: str
    cat: str

@dataclass
class Kernel:
    name: str
    instrs: list
    meta: dict = field(default_factory=dict)  # parsed from mangled name

    @property
    def n_instrs(self):
        return len(self.instrs)

    def count(self, cat):
        return sum(1 for i in self.instrs if i.cat == cat)

    def count_opcode(self, pattern):
        return sum(1 for i in self.instrs if re.search(pattern, i.opcode))

    def cat_counts(self):
        c = Counter()
        for i in self.instrs:
            c[i.cat] += 1
        return c

# ═══════════════════════════════════════════════════════════════════════════════
# SASS parser (fast, minimal — just opcode + address + full line)
# ═══════════════════════════════════════════════════════════════════════════════

RE_FUNC = re.compile(r'\s*Function\s*:\s*(\S+)')
RE_INSTR = re.compile(r'\s*/\*([0-9a-fA-F]+)\*/\s+(?:@[!]?P\d+\s+)?(\S+?)(?:\s+.*?)?\s*;')

def parse_sass_file(filename, max_kernels=None):
    """Parse cuobjdump --dump-sass output. Returns list of Kernel objects.

    Only parses functions that contain actual instructions. For huge dumps,
    max_kernels limits how many kernel functions to keep (largest by instr count).
    """
    kernels = []
    current_name = None
    current_instrs = []

    with open(filename) as f:
        for line in f:
            fm = RE_FUNC.match(line)
            if fm:
                if current_name and current_instrs:
                    kernels.append(Kernel(current_name, current_instrs))
                current_name = fm.group(1)
                current_instrs = []
                continue

            im = RE_INSTR.match(line)
            if im and current_name:
                addr = int(im.group(1), 16)
                opcode = im.group(2)
                current_instrs.append(Instr(addr, opcode, line.rstrip(), cat_of(opcode)))

    if current_name and current_instrs:
        kernels.append(Kernel(current_name, current_instrs))

    # Parse metadata from mangled names
    for k in kernels:
        k.meta = parse_kernel_name(k.name)

    if max_kernels and len(kernels) > max_kernels:
        kernels.sort(key=lambda k: k.n_instrs, reverse=True)
        kernels = kernels[:max_kernels]

    return kernels


def filter_gemm_kernels(kernels):
    """Return only kernels containing UTCQMMA (i.e. GEMM kernels)."""
    return [k for k in kernels if k.count('mma') > 0]

# ═══════════════════════════════════════════════════════════════════════════════
# CUTLASS mangled name parser
# ═══════════════════════════════════════════════════════════════════════════════

def parse_kernel_name(name):
    """Extract tile/stage/epilogue info from CUTLASS mangled name."""
    meta = {}

    # Our kernel: persistent_gemm<EpilogueOp::XXX>
    if 'persistent_gemm' in name:
        meta['source'] = 'custom'
        if 'BIAS_RESIDUAL' in name or 'EpilogueOp2' in name:
            meta['epilogue'] = 'bias+residual'
        elif 'BIAS_GELU' in name or 'EpilogueOp1' in name:
            meta['epilogue'] = 'bias+gelu'
        elif 'BIAS_POS_EMBED' in name or 'EpilogueOp0' in name:
            meta['epilogue'] = 'bias+pos_embed'
        return meta

    # CUTLASS kernel
    if 'cutlass' not in name and 'GemmUniversal' not in name:
        return meta

    meta['source'] = 'cutlass'

    # Stages: MainloopSm100TmaUmmaWarpSpecializedILi(\d+)E
    m = re.search(r'WarpSpecializedILi(\d+)E', name)
    if m:
        meta['stages'] = int(m.group(1))

    # CTA group size: ILi2E usually appears for cta_group::2
    # Cluster: look for cluster config ILi(\d+)E after the first tuple

    # Tile: sequence of ILi(\d+)EE values after tile config
    # The tile shape appears as N4cute5tupleIJ... ILi(M)EE ILi(N)EE ILi(K)EE
    # For 2x1SM tiles: the 2nd set of ILi values has the tile shape
    tile_ints = re.findall(r'ILi(\d+)EE', name)
    if len(tile_ints) >= 3:
        # Heuristic: find the set of 3 values that look like tile dims
        # Tile dims are typically 64-256 range
        candidates = [int(x) for x in tile_ints]
        for i in range(len(candidates) - 2):
            a, b, c = candidates[i], candidates[i+1], candidates[i+2]
            if 32 <= a <= 512 and 32 <= b <= 512 and 32 <= c <= 512:
                meta['tile_m'] = a
                meta['tile_n'] = b
                meta['tile_k'] = c
                break

    # Epilogue type
    if 'LinCombPerColBiasEltAct' in name:
        if 'Identity' in name:
            meta['epilogue'] = 'bias+residual'
        elif 'Sm80_GeLU' in name or 'GeLU' in name:
            meta['epilogue'] = 'bias+gelu'
        else:
            meta['epilogue'] = 'bias+act'
    elif 'LinearCombination' in name:
        meta['epilogue'] = 'linear_comb'
    elif 'PerColBias' in name:
        meta['epilogue'] = 'bias_only'

    # TMEM load type (extract just the meaningful suffix)
    m = re.search(r'TMEM_LOAD_(\d+dp\d+b\d+x)', name)
    if m:
        meta['tmem_load'] = m.group(1)

    # Epi tile: look for Sm100TmaWarpSpecializedILi(\d+)ELi(\d+)ELi(\d+)E
    m = re.search(r'Sm100TmaWarpSpecializedILi(\d+)ELi(\d+)ELi(\d+)E', name)
    if m:
        meta['epi_warps'] = int(m.group(1))
        meta['epi_ctas'] = int(m.group(2))
        meta['epi_frag'] = int(m.group(3))

    return meta


def kernel_short_name(k):
    """Generate a short descriptive name for a kernel."""
    m = k.meta
    if m.get('source') == 'custom':
        return f"custom ({m.get('epilogue', '?')})"
    if m.get('source') == 'cutlass':
        tile = ''
        if 'tile_m' in m:
            tile = f"{m['tile_m']}x{m['tile_n']}x{m['tile_k']}"
        stages = m.get('stages', '?')
        epi = m.get('epilogue', '?')
        tmem = m.get('tmem_load', '?')
        parts = [f"CUTLASS {tile}", f"stg={stages}", epi]
        if tmem != '?':
            parts.append(f"tmem={tmem}")
        return ' '.join(parts)
    # Unknown — demangle if possible, otherwise truncate
    short = k.name
    # Try to extract a meaningful suffix
    for prefix in ['_ZN7cutlass13device_kernelI', '_Z']:
        if short.startswith(prefix):
            short = short[len(prefix):]
            break
    if len(short) > 50:
        short = short[:50] + '...'
    return short

# ═══════════════════════════════════════════════════════════════════════════════
# Section detection
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Section:
    name: str
    start: int  # instruction index
    end: int    # instruction index (exclusive)
    addr_lo: int
    addr_hi: int

def _cluster_indices(indices, max_gap=40):
    """Group instruction indices into clusters (allowing gaps of max_gap)."""
    if not indices:
        return []
    clusters = []
    start = indices[0]
    end = indices[0]
    for i in indices[1:]:
        if i - end <= max_gap:
            end = i
        else:
            clusters.append((start, end))
            start = i
            end = i
    clusters.append((start, end))
    return clusters


def detect_sections(instrs):
    """Detect K-loop and epilogue regions by instruction clustering.

    Warp-specialized kernels interleave W0/W1/W2-W5 code paths in the SASS,
    so address-range sections can overlap. Instead, we cluster instruction
    indices near MMA and LDTM instructions independently.

    Returns list of Section objects (may be non-contiguous, may not cover all instrs).
    """
    n = len(instrs)
    if n == 0:
        return []

    mma_indices = [i for i, x in enumerate(instrs) if x.cat == 'mma']
    ldtm_indices = [i for i, x in enumerate(instrs) if x.cat == 'tmem_load']

    sections = []

    if not mma_indices:
        sections.append(Section('full_kernel', 0, n, instrs[0].addr, instrs[-1].addr))
        return sections

    # K-loop: cluster MMA instructions, expand each cluster to include
    # surrounding overhead (R2UR, ELECT, SYNCS, UTCBAR)
    mma_clusters = _cluster_indices(mma_indices)
    kloop_start = mma_clusters[0][0]
    kloop_end = mma_clusters[-1][1] + 1

    # Expand backward from first MMA cluster to include mbar_wait
    for i in range(kloop_start - 1, max(0, kloop_start - 60), -1):
        if instrs[i].cat == 'barrier' and 'SYNCS' in instrs[i].opcode:
            kloop_start = i
            break

    # Expand forward from last MMA to include UTCBAR
    for i in range(kloop_end, min(n, kloop_end + 30)):
        if instrs[i].cat == 'mma_commit':
            kloop_end = i + 1
            break

    sections.append(Section('kloop', kloop_start, kloop_end,
                            instrs[kloop_start].addr, instrs[kloop_end-1].addr))

    # Epilogue: neighborhood of ALL LDTM instructions.
    # In warp-specialized kernels, epilogue code is replicated across W2-W5
    # at scattered address ranges. Use neighborhood union instead of clustering.
    if ldtm_indices:
        # Mark all instructions within RADIUS of any LDTM as epilogue
        RADIUS = 60
        epi_mask = [False] * n
        for li in ldtm_indices:
            for i in range(max(0, li - RADIUS), min(n, li + RADIUS + 1)):
                if instrs[i].cat != 'mma':  # don't pull in K-loop
                    epi_mask[i] = True

        # Find contiguous runs for reporting, but analyze all marked instructions
        epi_indices = [i for i in range(n) if epi_mask[i]]
        if epi_indices:
            epi_start = epi_indices[0]
            epi_end = epi_indices[-1] + 1
            sections.append(Section('epilogue', epi_start, epi_end,
                                    instrs[epi_start].addr, instrs[epi_end-1].addr))
            # Store the mask for analysis (not all instructions in range are epilogue)
            sections[-1]._epi_indices = epi_indices

    return sections


def section_counts(instrs, section):
    """Count instructions per fine category within a section."""
    c = Counter()
    for i in range(section.start, section.end):
        c[instrs[i].cat] += 1
    return c

# ═══════════════════════════════════════════════════════════════════════════════
# K-loop structure analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_kloop(instrs, section):
    """Analyze K-loop structure: sub-MMA pattern, overhead per MMA."""
    sl = instrs[section.start:section.end]
    mma_positions = [i for i, x in enumerate(sl) if x.cat == 'mma']
    n_mma = len(mma_positions)

    if n_mma == 0:
        return {'n_mma': 0}

    # Count overhead per MMA: look at instructions between consecutive MMAs
    gaps = []
    for j in range(1, len(mma_positions)):
        gap_start = mma_positions[j-1] + 1
        gap_end = mma_positions[j]
        gap_instrs = sl[gap_start:gap_end]
        gap_cats = Counter(x.cat for x in gap_instrs)
        gaps.append({'size': len(gap_instrs), 'cats': dict(gap_cats)})

    # Per-MMA overhead pattern (from the most common gap)
    avg_gap = sum(g['size'] for g in gaps) / len(gaps) if gaps else 0

    # Count key opcodes in K-loop
    cats = Counter(x.cat for x in sl)

    # Check if descriptors use UR (CUTLASS-style) or R→UR transfers
    r2ur_per_mma = cats.get('r2ur', 0) / n_mma if n_mma > 0 else 0
    elect_per_mma = cats.get('elect', 0) / n_mma if n_mma > 0 else 0

    # Identify MMA groups (consecutive MMAs with small gaps)
    groups = []
    current_group = [mma_positions[0]]
    for j in range(1, len(mma_positions)):
        if mma_positions[j] - mma_positions[j-1] < 30:
            current_group.append(mma_positions[j])
        else:
            groups.append(current_group)
            current_group = [mma_positions[j]]
    groups.append(current_group)

    return {
        'n_mma': n_mma,
        'n_groups': len(groups),
        'mma_per_group': [len(g) for g in groups],
        'avg_gap': avg_gap,
        'r2ur_per_mma': r2ur_per_mma,
        'elect_per_mma': elect_per_mma,
        'total_instrs': len(sl),
        'cats': dict(cats),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# Epilogue structure analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_epilogue(instrs, section):
    """Analyze epilogue: TMEM width, compute precision, store count."""
    # Use neighborhood mask if available (warp-specialized kernels)
    if hasattr(section, '_epi_indices') and section._epi_indices:
        sl = [instrs[i] for i in section._epi_indices]
    else:
        sl = instrs[section.start:section.end]
    cats = Counter(x.cat for x in sl)

    # TMEM load width
    tmem_widths = Counter()
    for x in sl:
        if x.cat == 'tmem_load':
            for w in ('x64', 'x32', 'x16', 'x8'):
                if w in x.opcode:
                    tmem_widths[w] += 1
                    break

    # Compute precision
    bf16_ops = cats.get('bf16_fma', 0) + cats.get('bf16_add', 0) + cats.get('bf16_mul', 0)
    fp32_ops = cats.get('fp32_mul', 0) + cats.get('fp32_fma', 0) + cats.get('fp32_add', 0)
    precision = 'BF16' if bf16_ops > fp32_ops else 'FP32' if fp32_ops > 0 else 'unknown'

    return {
        'total_instrs': len(sl),
        'tmem_loads': dict(tmem_widths),
        'n_tmem_total': sum(tmem_widths.values()),
        'precision': precision,
        'bf16_ops': bf16_ops,
        'fp32_ops': fp32_ops,
        'cvt': cats.get('cvt', 0),
        'sts': cats.get('sts', 0),
        'lds': cats.get('lds', 0),
        'ldg': cats.get('ldg', 0),
        'barriers': cats.get('barrier', 0) + cats.get('nanosleep', 0),
        'cats': dict(cats),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# Output: list mode
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_list(args):
    """List all GEMM kernels in a dump."""
    kernels = parse_sass_file(args.file)
    epi_filter = getattr(args, 'epilogue', None)
    unique = dedup_gemm_kernels(kernels, epi_filter)

    if not unique:
        print(f"No GEMM kernels found in {args.file}")
        return

    all_gemm = filter_gemm_kernels(kernels)
    print(f"=== {args.file} — {len(unique)} unique GEMM configs"
          f" ({len(all_gemm)} total, {len(kernels)} functions) ===\n")
    print(f"  {'#':>3s}  {'Instrs':>6s}  {'MMAs':>4s}  {'LDTM':>4s}  {'STS':>4s}  {'R2UR':>5s}  Description")
    print(f"  {'---':>3s}  {'------':>6s}  {'----':>4s}  {'----':>4s}  {'---':>4s}  {'-----':>5s}  -----------")

    for idx, k in enumerate(unique):
        n_mma = k.count('mma')
        n_ldtm = k.count('tmem_load')
        n_sts = k.count('sts')
        n_r2ur = k.count('r2ur')
        desc = kernel_short_name(k)
        print(f"  {idx:>3d}  {k.n_instrs:>6d}  {n_mma:>4d}  {n_ldtm:>4d}  {n_sts:>4d}  {n_r2ur:>5d}  {desc}")

# ═══════════════════════════════════════════════════════════════════════════════
# Output: analyze mode
# ═══════════════════════════════════════════════════════════════════════════════

def dedup_gemm_kernels(kernels, epi_filter=None):
    """Filter to GEMM kernels, optionally filter by epilogue, deduplicate by config."""
    gemm = filter_gemm_kernels(kernels)
    if epi_filter:
        gemm = [k for k in gemm if epi_filter in k.meta.get('epilogue', '')]
    # Deduplicate by (tile, stages, epilogue)
    seen = {}
    unique = []
    for k in gemm:
        m = k.meta
        key = (m.get('tile_m'), m.get('tile_n'), m.get('tile_k'),
               m.get('stages'), m.get('epilogue'))
        if key not in seen:
            seen[key] = len(unique)
            unique.append(k)
    return unique


def select_kernel(kernels, index=None, epi_filter=None):
    """Select a GEMM kernel by index, or auto-select the largest."""
    gemm = dedup_gemm_kernels(kernels, epi_filter)
    if not gemm:
        return None
    if index is not None:
        if 0 <= index < len(gemm):
            return gemm[index]
        print(f"Error: kernel index {index} out of range (0-{len(gemm)-1})", file=sys.stderr)
        return None
    # Auto-select: largest GEMM kernel
    return max(gemm, key=lambda k: k.n_instrs)


def print_instruction_mix(cats, label="INSTRUCTION MIX"):
    """Print compact instruction mix from a category counter."""
    print(f"\n  {label}:")
    for group_name, group_cats in DISPLAY_GROUPS:
        total = sum(cats.get(c, 0) for c in group_cats)
        if total > 0:
            # Show sub-breakdown for groups with multiple nonzero entries
            subs = [(c, cats.get(c, 0)) for c in group_cats if cats.get(c, 0) > 0]
            if len(subs) == 1:
                print(f"    {group_name:<16s} {total:>5d}")
            else:
                sub_str = ', '.join(f"{c}={n}" for c, n in subs)
                print(f"    {group_name:<16s} {total:>5d}  ({sub_str})")


def cmd_analyze(args):
    """Analyze a single GEMM kernel with section breakdown."""
    kernels = parse_sass_file(args.file)
    epi_filter = getattr(args, 'epilogue', None)
    k = select_kernel(kernels, args.kernel, epi_filter)
    if not k:
        print(f"No GEMM kernel found in {args.file}")
        return

    name = kernel_short_name(k)
    n_mma = k.count('mma')
    n_commit = k.count('mma_commit')

    print(f"=== {args.file} ===")
    print(f"  Kernel: {name}")
    print(f"  {k.n_instrs} instructions | {n_mma} UTCQMMA | {n_commit} UTCBAR")

    # Detect sections
    sections = detect_sections(k.instrs)

    if sections:
        print(f"\n  SECTIONS:")
        for s in sections:
            n = s.end - s.start
            print(f"    {s.name:<14s}  {n:>5d} instr  0x{s.addr_lo:04x}-0x{s.addr_hi:04x}")

    # Global instruction mix
    print_instruction_mix(k.cat_counts())

    # Per-section analysis
    for s in sections:
        if s.name == 'kloop':
            kl = analyze_kloop(k.instrs, s)
            print(f"\n  K-LOOP ({kl['n_mma']} static MMAs in {kl['n_groups']} groups"
                  f" of {'/'.join(str(x) for x in set(kl['mma_per_group']))} sub-MMAs):")
            print(f"    Per-MMA overhead:  R2UR={kl['r2ur_per_mma']:.1f}  ELECT={kl['elect_per_mma']:.1f}")
            print(f"    Avg gap between MMAs: {kl['avg_gap']:.1f} instructions")
            if kl['r2ur_per_mma'] > 2:
                print(f"    NOTE: high R2UR/MMA = inline PTX desc transfer overhead")
            elif kl['r2ur_per_mma'] < 1:
                print(f"    NOTE: low R2UR/MMA = descs pre-computed in UR space (CUTLASS-style)")

        elif s.name == 'epilogue':
            ep = analyze_epilogue(k.instrs, s)
            tmem_str = ', '.join(f"{n}x {w}" for w, n in sorted(ep['tmem_loads'].items()))
            print(f"\n  EPILOGUE ({ep['total_instrs']} instructions, {ep['precision']} precision):")
            print(f"    TMEM loads:   {ep['n_tmem_total']}  ({tmem_str})")
            if ep['bf16_ops'] > 0:
                print(f"    BF16 compute: {ep['bf16_ops']}  (HFMA2/HADD2/HMUL2)")
            if ep['fp32_ops'] > 0:
                print(f"    FP32 compute: {ep['fp32_ops']}  (FMUL/FFMA/FADD)")
            print(f"    CVT (F2FP):   {ep['cvt']}")
            print(f"    STS:          {ep['sts']}")
            print(f"    LDS:          {ep['lds']}")
            if ep['ldg'] > 0:
                print(f"    LDG:          {ep['ldg']}")
            print(f"    Barriers:     {ep['barriers']}")

# ═══════════════════════════════════════════════════════════════════════════════
# Output: compare mode
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_compare(args):
    """Side-by-side comparison of two GEMM kernels."""
    kernels_a = parse_sass_file(args.file)
    kernels_b = parse_sass_file(args.file2)

    epi_filter = getattr(args, 'epilogue', None)
    ka = select_kernel(kernels_a, args.kernel, epi_filter)
    kb = select_kernel(kernels_b, args.kernel2, epi_filter)

    if not ka:
        print(f"No GEMM kernel found in {args.file}")
        return
    if not kb:
        print(f"No GEMM kernel found in {args.file2}")
        return

    name_a = kernel_short_name(ka)
    name_b = kernel_short_name(kb)

    print(f"{'=' * 78}")
    print(f"  SASS COMPARISON")
    print(f"{'=' * 78}")
    print(f"  A: {args.file}")
    print(f"     {name_a}")
    print(f"  B: {args.file2}")
    print(f"     {name_b}")

    # High-level metrics
    print(f"\n  {'METRIC':<24s}  {'A':>8s}  {'B':>8s}  {'Delta':>8s}")
    print(f"  {'-'*24}  {'-'*8}  {'-'*8}  {'-'*8}")

    metrics = [
        ('Total instructions', ka.n_instrs, kb.n_instrs),
        ('Static MMAs', ka.count('mma'), kb.count('mma')),
        ('MMA commits', ka.count('mma_commit'), kb.count('mma_commit')),
    ]
    for label, a, b in metrics:
        d = a - b
        sign = '+' if d > 0 else ''
        print(f"  {label:<24s}  {a:>8d}  {b:>8d}  {sign}{d:>7d}")

    # Instruction mix comparison
    cats_a = ka.cat_counts()
    cats_b = kb.cat_counts()
    all_cats = set(cats_a.keys()) | set(cats_b.keys())

    print(f"\n  {'CATEGORY':<24s}  {'A':>8s}  {'B':>8s}  {'Delta':>8s}")
    print(f"  {'-'*24}  {'-'*8}  {'-'*8}  {'-'*8}")

    for group_name, group_cats in DISPLAY_GROUPS:
        a = sum(cats_a.get(c, 0) for c in group_cats)
        b = sum(cats_b.get(c, 0) for c in group_cats)
        if a == 0 and b == 0:
            continue
        d = a - b
        sign = '+' if d > 0 else ''
        # Annotation for significant differences
        note = ''
        if group_name == 'R2UR' and abs(d) > 50:
            note = '  ← inline PTX desc overhead' if d > 0 else '  ← UR pre-computed'
        elif group_name == 'BF16 compute' and a > 0 and b == 0:
            note = '  ← BF16 epilogue (A only)'
        elif group_name == 'FP32 compute' and b > 0 and a == 0:
            note = '  ← FP32 epilogue (B only)'
        print(f"  {group_name:<24s}  {a:>8d}  {b:>8d}  {sign}{d:>7d}{note}")

    # Section-level comparison
    secs_a = detect_sections(ka.instrs)
    secs_b = detect_sections(kb.instrs)

    kloop_a = next((s for s in secs_a if s.name == 'kloop'), None)
    kloop_b = next((s for s in secs_b if s.name == 'kloop'), None)
    epi_a = next((s for s in secs_a if s.name == 'epilogue'), None)
    epi_b = next((s for s in secs_b if s.name == 'epilogue'), None)

    if kloop_a and kloop_b:
        kla = analyze_kloop(ka.instrs, kloop_a)
        klb = analyze_kloop(kb.instrs, kloop_b)
        print(f"\n  K-LOOP COMPARISON:")
        print(f"    {'':24s}  {'A':>8s}  {'B':>8s}")
        print(f"    {'Static MMAs':<24s}  {kla['n_mma']:>8d}  {klb['n_mma']:>8d}")
        print(f"    {'MMA groups':<24s}  {kla['n_groups']:>8d}  {klb['n_groups']:>8d}")
        grp_a = '/'.join(str(x) for x in set(kla['mma_per_group']))
        grp_b = '/'.join(str(x) for x in set(klb['mma_per_group']))
        print(f"    {'MMAs/group':<24s}  {grp_a:>8s}  {grp_b:>8s}")
        print(f"    {'Total instructions':<24s}  {kla['total_instrs']:>8d}  {klb['total_instrs']:>8d}")
        print(f"    {'R2UR/MMA':<24s}  {kla['r2ur_per_mma']:>8.1f}  {klb['r2ur_per_mma']:>8.1f}")
        print(f"    {'ELECT/MMA':<24s}  {kla['elect_per_mma']:>8.1f}  {klb['elect_per_mma']:>8.1f}")
        print(f"    {'Avg gap/MMA':<24s}  {kla['avg_gap']:>8.1f}  {klb['avg_gap']:>8.1f}")

    if epi_a and epi_b:
        epa = analyze_epilogue(ka.instrs, epi_a)
        epb = analyze_epilogue(kb.instrs, epi_b)
        tmem_a = ', '.join(f"{n}x{w}" for w, n in sorted(epa['tmem_loads'].items()))
        tmem_b = ', '.join(f"{n}x{w}" for w, n in sorted(epb['tmem_loads'].items()))
        print(f"\n  EPILOGUE COMPARISON:")
        print(f"    {'':24s}  {'A':>8s}  {'B':>8s}")
        print(f"    {'Total instructions':<24s}  {epa['total_instrs']:>8d}  {epb['total_instrs']:>8d}")
        print(f"    {'Precision':<24s}  {epa['precision']:>8s}  {epb['precision']:>8s}")
        print(f"    {'TMEM loads':<24s}  {tmem_a:>8s}  {tmem_b:>8s}")
        print(f"    {'BF16 compute':<24s}  {epa['bf16_ops']:>8d}  {epb['bf16_ops']:>8d}")
        print(f"    {'FP32 compute':<24s}  {epa['fp32_ops']:>8d}  {epb['fp32_ops']:>8d}")
        print(f"    {'CVT (F2FP)':<24s}  {epa['cvt']:>8d}  {epb['cvt']:>8d}")
        print(f"    {'STS':<24s}  {epa['sts']:>8d}  {epb['sts']:>8d}")
        print(f"    {'LDS':<24s}  {epa['lds']:>8d}  {epb['lds']:>8d}")
        if epa['ldg'] > 0 or epb['ldg'] > 0:
            print(f"    {'LDG':<24s}  {epa['ldg']:>8d}  {epb['ldg']:>8d}")
        print(f"    {'Barriers':<24s}  {epa['barriers']:>8d}  {epb['barriers']:>8d}")

    # Key structural differences
    diffs = []
    if epi_a and epi_b:
        epa = analyze_epilogue(ka.instrs, epi_a)
        epb = analyze_epilogue(kb.instrs, epi_b)
        if epa['precision'] != epb['precision']:
            diffs.append(f"Epilogue precision: A={epa['precision']}, B={epb['precision']}")
        tmem_keys_a = set(epa['tmem_loads'].keys())
        tmem_keys_b = set(epb['tmem_loads'].keys())
        if tmem_keys_a != tmem_keys_b:
            ta = '/'.join(sorted(tmem_keys_a)) or 'none'
            tb = '/'.join(sorted(tmem_keys_b)) or 'none'
            diffs.append(f"TMEM load width: A={ta}, B={tb}")

    if kloop_a and kloop_b:
        kla = analyze_kloop(ka.instrs, kloop_a)
        klb = analyze_kloop(kb.instrs, kloop_b)
        if abs(kla['r2ur_per_mma'] - klb['r2ur_per_mma']) > 1:
            if kla['r2ur_per_mma'] > klb['r2ur_per_mma']:
                diffs.append(f"A has {kla['r2ur_per_mma']:.0f} R2UR/MMA (inline PTX overhead) vs B's {klb['r2ur_per_mma']:.0f}")
            else:
                diffs.append(f"B has {klb['r2ur_per_mma']:.0f} R2UR/MMA (inline PTX overhead) vs A's {kla['r2ur_per_mma']:.0f}")

    if diffs:
        print(f"\n  KEY DIFFERENCES:")
        for i, d in enumerate(diffs, 1):
            print(f"    {i}. {d}")

# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='SASS analysis for SM100a persistent GEMM kernels')
    sub = parser.add_subparsers(dest='cmd')

    # list
    p_list = sub.add_parser('list', help='List GEMM kernels in a dump')
    p_list.add_argument('file', help='cuobjdump --dump-sass output file')
    p_list.add_argument('--epilogue', '-e', default=None,
                       help='Filter by epilogue type (e.g. "residual", "gelu")')

    # analyze
    p_analyze = sub.add_parser('analyze', help='Section-level kernel breakdown')
    p_analyze.add_argument('file', help='SASS dump file')
    p_analyze.add_argument('-k', '--kernel', type=int, default=None,
                          help='GEMM kernel index (from list mode, default: largest)')
    p_analyze.add_argument('-e', '--epilogue', default=None,
                          help='Filter by epilogue type')

    # compare
    p_compare = sub.add_parser('compare', help='Side-by-side kernel comparison')
    p_compare.add_argument('file', help='First SASS dump (A)')
    p_compare.add_argument('file2', help='Second SASS dump (B)')
    p_compare.add_argument('-k', '--kernel', type=int, default=None,
                          help='Kernel index for file A')
    p_compare.add_argument('-k2', '--kernel2', type=int, default=None,
                          help='Kernel index for file B')
    p_compare.add_argument('-e', '--epilogue', default=None,
                          help='Filter by epilogue type')

    args = parser.parse_args()

    if args.cmd == 'list':
        cmd_list(args)
    elif args.cmd == 'analyze':
        cmd_analyze(args)
    elif args.cmd == 'compare':
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
