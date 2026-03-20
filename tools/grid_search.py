#!/usr/bin/env python3
"""Compile-time grid search for SigLIP kernel parameters.

Enumerates parameter combos, prunes invalid configs (SMEM budget, thread count),
compiles with -D flags, runs with timeout + hang detection, collects results
into a sorted table + CSV.

Supports all three kernels: patch_embed (default), fc1_gelu, fc2.

Usage:
    python3 grid_search.py                   # per-kernel tiered search (default)
    python3 grid_search.py --kernel fc1_gelu # FC1 per-kernel tiers (GV+ST first)
    python3 grid_search.py --kernel fc2      # FC2 per-kernel tiers (NS+KLU first)
    python3 grid_search.py --tier 1          # structure: N_STAGES x NUM_EPI_WARPS (~5 configs)
    python3 grid_search.py --tier all        # per-kernel tiered + interactions, dynamic k, η²
    python3 grid_search.py --full-cross      # all parameters crossed (~3000 configs)
    python3 grid_search.py --only N_STAGES STAGGER_CYCLES --fix MBAR_EARLY=1
    python3 grid_search.py --kernel fc2 --tier all   # sweep FC2 kernel
    python3 grid_search.py --interact epilogue --kernel fc1_gelu   # cross-tier interaction sweep
    python3 grid_search.py --interact all --kernel fc2             # all applicable interactions
    python3 grid_search.py --only BATCH_EPILOGUE INTERLEAVE_STRATEGY --base MBAR_EARLY=1 STAGGER_CYCLES=160
"""

import argparse
import csv
import glob as glob_mod
import itertools
import os
import re
import subprocess
import sys
import tempfile
import time

# Force unbuffered stdout so log files get flushed per-line
sys.stdout.reconfigure(line_buffering=True)

# ── Defaults (match kernel_common.cuh #ifndef values) ──
DEFAULTS = {
    'N_STAGES': 4,
    'NUM_EPI_WARPS': 4,
    'TMEM_LOAD_WIDTH': 32,
    'INTERLEAVE_STRATEGY': 2,
    'MBAR_EARLY': 0,
    'STAGGER_CYCLES': 80,
    'PHASE1_UNROLL': 2,
    'SNAKE_ORDER': 1,
    'CVT_ADD_FUSED': 1,
    'K_LOOP_UNROLL': 4,  # default = N_STAGES
    'W0_LOOP_UNROLL': 0,  # 0=no pragma (compiler decides)
    'SUB_MMA_UNROLL': 0,  # 0=no pragma (compiler decides)
    'PRELOAD_MODE': 1,
    'PREFETCH_BEFORE_STORE': 0,
    'GELU_VARIANT': 0,
    'TMA_RESIDUAL': 0,
    'W0_RES_PREFETCH': 0,
    'W0_RES_FULL': 0,
    'PREFETCH_MBAR': 0,
    'EPI_LOAD_WARP': 0,
    'OVERLAP_EPI_WAIT': 0,
    'BATCH_MMA': 0,
    'BATCH_EPILOGUE': 0,
    'GELU_VECTOR_WIDTH': 32,
    'STORE_TIMING': 0,
    'EPILOGUE_LOOP': 0,
    'STS_WIDTH': 16,
    'EPI_SYNC': 0,
    'NUM_PASSES_PARAM': 0,
    'BIAS_SMEM': 0,
    'FP32_EPILOGUE': 0,
}

# ── Parameter ranges ──
RANGES = {
    'N_STAGES': [3, 4, 5],
    'NUM_EPI_WARPS': [4, 5],
    'TMEM_LOAD_WIDTH': [32, 64],
    'INTERLEAVE_STRATEGY': [0, 1, 2, 3],
    'MBAR_EARLY': [0, 1],
    'STAGGER_CYCLES': [0, 40, 60, 80, 100, 120, 160, 200],
    'PHASE1_UNROLL': [1, 2, 4],
    'SNAKE_ORDER': [1],  # pinned — SNAKE=0 is catastrophic (+49μs PE, never wins)
    'CVT_ADD_FUSED': [0, 1],
    'K_LOOP_UNROLL': [1, 2, 4, 6, 8],
    'W0_LOOP_UNROLL': [0, 1, 4],
    'SUB_MMA_UNROLL': [0, 1, 3],
    'PRELOAD_MODE': [0, 1, 2],
    'PREFETCH_BEFORE_STORE': [0, 1],
    'GELU_VARIANT': [0, 4, 5],  # V1,2,3,6 use software tanhf() → 9.1ms catastrophic (4x slower)
    'TMA_RESIDUAL': [0, 1, 2],
    'W0_RES_PREFETCH': [0, 1],
    'W0_RES_FULL': [0, 1],
    'PREFETCH_MBAR': [0, 1],
    'EPI_LOAD_WARP': [0, 1],
    'OVERLAP_EPI_WAIT': [0, 1],
    'BATCH_MMA': [0, 1],
    'BATCH_EPILOGUE': [0, 1],
    'GELU_VECTOR_WIDTH': [8, 16, 32],
    'STORE_TIMING': [0, 1],
    'EPILOGUE_LOOP': [0, 1],
    'STS_WIDTH': [16, 32],
    'EPI_SYNC': [0, 1],
    'NUM_PASSES_PARAM': [0, 4],
    'BIAS_SMEM': [0, 1],
    'DEFERRED_WAIT': [0, 1],
    'FP32_EPILOGUE': [0, 1, 2],
}

# ── Tier definitions (generic, used by --tier 1/2/3/4/5) ──
TIER_PARAMS = {
    1: ['N_STAGES', 'NUM_EPI_WARPS'],
    2: ['INTERLEAVE_STRATEGY', 'MBAR_EARLY', 'STAGGER_CYCLES', 'TMEM_LOAD_WIDTH'],
    3: ['PHASE1_UNROLL', 'CVT_ADD_FUSED', 'K_LOOP_UNROLL',
        'W0_LOOP_UNROLL', 'SUB_MMA_UNROLL'],
    4: ['PRELOAD_MODE', 'PREFETCH_BEFORE_STORE', 'BATCH_EPILOGUE', 'TMA_RESIDUAL', 'STORE_TIMING',
        'EPILOGUE_LOOP', 'STS_WIDTH', 'EPI_SYNC', 'NUM_PASSES_PARAM'],
    5: ['GELU_VARIANT', 'GELU_VECTOR_WIDTH'],
}

# ── Per-kernel tiers (used by --tier all) ──
# Ordered by balanced-η² from existing sweep data. Noise params (η²<0.01 with
# adequate N_bal) are omitted — pinned at KERNEL_BASES values and never swept.
# Tier 1: known dominant. Tier 2: known secondary. Tier 3: untested, need data.
# Source: session_20260315 (FC1, FC2), session_20260314 (PE). See docs/SEARCH_ANALYSIS.md.
# Invalidate after structural kernel changes (new epilogue ops, tile config changes).
KERNEL_TIERS = {
    'fc1_gelu': {
        1: ['GELU_VARIANT', 'INTERLEAVE_STRATEGY'],
        2: ['PHASE1_UNROLL', 'STORE_TIMING', 'PRELOAD_MODE', 'BATCH_EPILOGUE'],
        3: ['EPILOGUE_LOOP', 'STS_WIDTH', 'EPI_SYNC', 'GELU_VECTOR_WIDTH'],
    },
    'fc2': {
        1: ['N_STAGES', 'K_LOOP_UNROLL', 'TMA_RESIDUAL', 'W0_RES_PREFETCH', 'W0_RES_FULL', 'BATCH_MMA', 'FP32_EPILOGUE'],
        2: ['INTERLEAVE_STRATEGY', 'PHASE1_UNROLL', 'BIAS_SMEM', 'TMEM_LOAD_WIDTH'],
        3: ['BATCH_EPILOGUE', 'STORE_TIMING', 'STS_WIDTH', 'PRELOAD_MODE', 'DEFERRED_WAIT'],
        4: ['EPILOGUE_LOOP', 'EPI_SYNC', 'NUM_PASSES_PARAM'],
    },
    'patch_embed': {
        1: ['N_STAGES', 'K_LOOP_UNROLL', 'W0_LOOP_UNROLL'],
        2: ['PHASE1_UNROLL'],
        3: ['EPILOGUE_LOOP', 'EPI_SYNC', 'STORE_TIMING'],
    },
}

# Proven-meaningful params per kernel for --cross mode (full cross-product).
# Tiers 1-2 only — high η² params that interact. Tier 3+ are noise, pinned at KERNEL_BASES.
KERNEL_CROSS_PARAMS = {
    'fc1_gelu': [
        'GELU_VARIANT', 'INTERLEAVE_STRATEGY',                   # tier 1
        'PHASE1_UNROLL', 'STORE_TIMING', 'PRELOAD_MODE', 'BATCH_EPILOGUE',  # tier 2
    ],
    'fc2': [
        'N_STAGES', 'TMA_RESIDUAL', 'W0_RES_PREFETCH', 'W0_RES_FULL', 'BATCH_MMA', 'FP32_EPILOGUE',  # tier 1
        'INTERLEAVE_STRATEGY', 'PHASE1_UNROLL', 'BIAS_SMEM', 'BATCH_EPILOGUE',       # tier 2
        'STORE_TIMING', 'DEFERRED_WAIT',                                               # tier 3
    ],
    'patch_embed': [
        'N_STAGES', 'K_LOOP_UNROLL', 'W0_LOOP_UNROLL',          # tier 1
        'PHASE1_UNROLL',                                          # tier 2
    ],
}

# Structural params: always branch both values, never prune by dynamic-k.
# These change pipeline depth or fundamental kernel structure — later tiers
# may interact differently with each value, so both must survive.
BRANCH_PARAMS = {
    'fc2': {'N_STAGES', 'BATCH_MMA', 'FP32_EPILOGUE'},
}

# Best-known configs per kernel (pin non-swept params here).
# Derived from sweep winners — noise params locked at their winning values.
# FC1: 2.267ms winner from session_20260315. FC2: 1.471ms from session_20260315.
# PE: 0.525ms from session_20260314 (effectively exhausted).
KERNEL_BASES = {
    'fc1_gelu': {
        'N_STAGES': 5, 'K_LOOP_UNROLL': 5, 'MBAR_EARLY': 1,
        'STAGGER_CYCLES': 0, 'W0_LOOP_UNROLL': 0, 'SUB_MMA_UNROLL': 3,
        'PREFETCH_BEFORE_STORE': 0, 'TMEM_LOAD_WIDTH': 32,
        'CVT_ADD_FUSED': 1, 'NUM_EPI_WARPS': 4,
    },
    'fc2': {
        'MBAR_EARLY': 0, 'STAGGER_CYCLES': 100,
        'W0_LOOP_UNROLL': 0, 'SUB_MMA_UNROLL': 0,
        'PREFETCH_BEFORE_STORE': 0, 'TMEM_LOAD_WIDTH': 32,
        'CVT_ADD_FUSED': 1, 'NUM_EPI_WARPS': 4,
        'DEFERRED_WAIT': 0,
    },
    'patch_embed': {
        'MBAR_EARLY': 1, 'STAGGER_CYCLES': 160,
        'INTERLEAVE_STRATEGY': 2, 'PRELOAD_MODE': 1,
        'PREFETCH_BEFORE_STORE': 0, 'SUB_MMA_UNROLL': 0,
        'TMEM_LOAD_WIDTH': 32, 'CVT_ADD_FUSED': 1,
        'NUM_EPI_WARPS': 4,
    },
}

# ── Kernel source files ──
KERNELS = {
    'patch_embed': 'patch_embed.cu',
    'fc1_gelu': 'fc1_gelu.cu',
    'fc2': 'fc2.cu',
}

# ── Cross-tier interaction groups ──
INTERACTIONS = {
    'epilogue': {
        'params': ['BATCH_EPILOGUE', 'INTERLEAVE_STRATEGY', 'PRELOAD_MODE', 'STORE_TIMING'],
        'kernels': ['fc1_gelu', 'fc2'],
    },
    'gelu': {
        'params': ['BATCH_EPILOGUE', 'GELU_VARIANT'],
        'kernels': ['fc1_gelu'],
    },
    'residual': {
        'params': ['TMA_RESIDUAL', 'PRELOAD_MODE', 'INTERLEAVE_STRATEGY', 'BIAS_SMEM', 'DEFERRED_WAIT'],
        'kernels': ['fc2'],
    },
    'reg_pressure': {
        'params': ['TMEM_LOAD_WIDTH', 'BATCH_EPILOGUE', 'BIAS_SMEM'],
        'kernels': ['fc1_gelu', 'fc2'],
    },
    'unroll_batch': {
        'params': ['PHASE1_UNROLL', 'BATCH_EPILOGUE'],
        'kernels': ['fc1_gelu', 'fc2'],
    },
    'gelu_interleave': {
        'params': ['GELU_VARIANT', 'INTERLEAVE_STRATEGY', 'STORE_TIMING'],
        'kernels': ['fc1_gelu'],
    },
    'gelu_width': {
        'params': ['GELU_VECTOR_WIDTH', 'GELU_VARIANT', 'BATCH_EPILOGUE'],
        'kernels': ['fc1_gelu'],
    },
    'store_batch': {
        'params': ['STORE_TIMING', 'BATCH_EPILOGUE'],
        'kernels': ['fc1_gelu', 'fc2'],
    },
    'prefetch_store': {
        'params': ['PREFETCH_BEFORE_STORE', 'STORE_TIMING'],
        'kernels': ['fc1_gelu', 'fc2'],
    },
    'loop_store': {
        'params': ['EPILOGUE_LOOP', 'STORE_TIMING'],
        'kernels': ['fc1_gelu', 'fc2', 'patch_embed'],
    },
    'sts_batch': {
        'params': ['STS_WIDTH', 'BATCH_EPILOGUE'],
        'kernels': ['fc1_gelu', 'fc2'],
    },
    'sync_stagger': {
        'params': ['EPI_SYNC', 'STAGGER_CYCLES'],
        'kernels': ['fc1_gelu', 'fc2', 'patch_embed'],
    },
    'passes_residual': {
        'params': ['NUM_PASSES_PARAM', 'TMA_RESIDUAL'],
        'kernels': ['fc2'],
    },
    'w0_prefetch': {
        'params': ['W0_RES_PREFETCH', 'W0_RES_FULL', 'TMA_RESIDUAL', 'N_STAGES'],
        'kernels': ['fc2'],
    },
}

# ── Build config ──
NVCC = 'nvcc'
ARCH = 'sm_100a'
CFLAGS = f'-gencode arch=compute_100a,code={ARCH} -O3 -std=c++17 -lineinfo --ptxas-options=-v'
LDFLAGS = '-lcurand -lcuda'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
COMPILE_TIMEOUT = 120
RUN_TIMEOUT = 30
SMEM_LIMIT = 233472  # 228 KB


def is_valid(cfg, kernel='patch_embed'):
    """Pre-compile constraint check. Returns (valid, reason)."""
    n_stages = cfg['N_STAGES']
    num_epi = cfg['NUM_EPI_WARPS']

    # Thread count
    threads = 32 * (2 + num_epi)
    if threads > 1024:
        return False, f'threads {threads} > 1024'

    # Row group coverage
    if num_epi < 4:
        return False, 'NUM_EPI_WARPS < 4'

    # SMEM budget
    stage_bytes = 32768  # TK=128: 128*128 + 128*128 = 32KB
    off_tmem = n_stages * stage_bytes
    off_tma_mbar = off_tmem + 8
    off_mma_mbar = off_tma_mbar + n_stages * 8
    off_mainloop_mbar = off_mma_mbar + n_stages * 8
    off_epilogue_mbar = off_mainloop_mbar + 16
    tma_res = cfg.get('TMA_RESIDUAL', 0)
    w0_res_pf = cfg.get('W0_RES_PREFETCH', 0)
    w0_res_full = cfg.get('W0_RES_FULL', 0)
    epi_load_warp_ = cfg.get('EPI_LOAD_WARP', 0)
    if tma_res:
        off_res_mbar = off_epilogue_mbar + 16
        if w0_res_full or epi_load_warp_:
            off_res_consumed_mbar = off_res_mbar + num_epi * 8
            off_res_pass_mbar = off_res_consumed_mbar + 8
            mbar_end = off_res_pass_mbar + 8
        elif w0_res_pf:
            off_res_consumed_mbar = off_res_mbar + num_epi * 8
            mbar_end = off_res_consumed_mbar + 8
        else:
            mbar_end = off_res_mbar + num_epi * 8
    else:
        mbar_end = off_epilogue_mbar + 16
    bias_smem_bytes = (256 * 2 if kernel == 'fc2' else 256 * 4) if cfg.get('BIAS_SMEM', 0) else 0
    off_staging = (mbar_end + bias_smem_bytes + 1023) & ~1023
    staging_warp_bytes = 4 * 32 * 128  # 16384
    smem_total = (off_staging + num_epi * staging_warp_bytes + 127) & ~127
    if smem_total > SMEM_LIMIT:
        return False, f'SMEM {smem_total} > {SMEM_LIMIT}'

    # GELU_VARIANT only meaningful for fc1_gelu
    if cfg.get('GELU_VARIANT', 0) != 0 and kernel != 'fc1_gelu':
        return False, 'GELU_VARIANT only for fc1_gelu'

    # TMA_RESIDUAL only meaningful for fc2
    if cfg.get('TMA_RESIDUAL', 0) != 0 and kernel != 'fc2':
        return False, 'TMA_RESIDUAL only for fc2'

    # W0_RES_PREFETCH only for fc2, requires TMA_RESIDUAL>=1
    if cfg.get('W0_RES_PREFETCH', 0) == 1:
        if kernel != 'fc2':
            return False, 'W0_RES_PREFETCH only for fc2'
        if cfg.get('TMA_RESIDUAL', 0) < 1:
            return False, 'W0_RES_PREFETCH requires TMA_RESIDUAL>=1'

    # W0_RES_FULL only for fc2, requires TMA_RESIDUAL>=1, mutually exclusive with W0_RES_PREFETCH
    if cfg.get('W0_RES_FULL', 0) == 1:
        if kernel != 'fc2':
            return False, 'W0_RES_FULL only for fc2'
        if cfg.get('TMA_RESIDUAL', 0) < 1:
            return False, 'W0_RES_FULL requires TMA_RESIDUAL>=1'
        if cfg.get('W0_RES_PREFETCH', 0) == 1:
            return False, 'W0_RES_FULL and W0_RES_PREFETCH are mutually exclusive'

    # EPI_LOAD_WARP: fc2 only, requires TMA_RESIDUAL>=1, exclusive with W0_RES_*
    if cfg.get('EPI_LOAD_WARP', 0) == 1:
        if kernel != 'fc2':
            return False, 'EPI_LOAD_WARP only for fc2'
        if cfg.get('TMA_RESIDUAL', 0) < 1:
            return False, 'EPI_LOAD_WARP requires TMA_RESIDUAL>=1'
        if cfg.get('W0_RES_FULL', 0) == 1 or cfg.get('W0_RES_PREFETCH', 0) == 1:
            return False, 'EPI_LOAD_WARP mutually exclusive with W0_RES_FULL/W0_RES_PREFETCH'

    # CVT_ADD_FUSED only meaningful for patch_embed (dead code on fc1/fc2)
    if cfg.get('CVT_ADD_FUSED', 1) != 1 and kernel != 'patch_embed':
        return False, 'CVT_ADD_FUSED only for patch_embed'

    # NUM_EPI_WARPS=5 disabled — split warp creates 2 extra template instantiations
    # that double FC1 SASS (9227 vs ~5000 lines), hurting icache and register allocation.
    # EPI=4 eliminates split-warp paths entirely (1 instantiation vs 3).
    if num_epi == 5:
        return False, 'NUM_EPI_WARPS=5 disabled (template bloat)'

    # BATCH_EPILOGUE only meaningful for fc1_gelu and fc2 (BIAS_ADD unaffected)
    if cfg.get('BATCH_EPILOGUE', 0) != 0 and kernel == 'patch_embed':
        return False, 'BATCH_EPILOGUE only for fc1_gelu/fc2'

    # BATCH_EPILOGUE with GELU_VARIANT >= 4 not supported (no standalone gelu_approx)
    if cfg.get('BATCH_EPILOGUE', 0) != 0 and kernel == 'fc1_gelu' and cfg.get('GELU_VARIANT', 0) >= 4:
        return False, 'BATCH_EPILOGUE requires GELU_VARIANT <= 3'

    # TMEM_LOAD_WIDTH=64 causes COMPILE_FAIL on FC1 (register pressure)
    if cfg.get('TMEM_LOAD_WIDTH', 32) == 64 and kernel == 'fc1_gelu':
        return False, 'TMEM_LOAD_WIDTH=64 COMPILE_FAIL on fc1_gelu'

    # GELU_VECTOR_WIDTH only meaningful for fc1_gelu with BATCH_EPILOGUE=1
    gvw = cfg.get('GELU_VECTOR_WIDTH', 32)
    if gvw != 32:
        if kernel != 'fc1_gelu':
            return False, 'GELU_VECTOR_WIDTH only for fc1_gelu'
        if cfg.get('BATCH_EPILOGUE', 0) != 1:
            return False, 'GELU_VECTOR_WIDTH requires BATCH_EPILOGUE=1'
        if cfg.get('GELU_VARIANT', 0) >= 4:
            return False, 'GELU_VECTOR_WIDTH requires GELU_VARIANT <= 3'

    # STORE_TIMING=1 with INTERLEAVE_STRATEGY=0 is redundant
    # (IS=0 already puts all stores at end, same as STORE_TIMING=1)
    if cfg.get('STORE_TIMING', 0) == 1 and cfg.get('INTERLEAVE_STRATEGY', 2) == 0:
        return False, 'STORE_TIMING=1 redundant with INTERLEAVE_STRATEGY=0'

    # EPILOGUE_LOOP constraints
    if cfg.get('EPILOGUE_LOOP', 0) == 1:
        if cfg.get('STORE_TIMING', 0) != 1:
            return False, 'EPILOGUE_LOOP=1 requires STORE_TIMING=1'
        if cfg.get('BATCH_EPILOGUE', 0) != 0:
            return False, 'EPILOGUE_LOOP=1 incompatible with BATCH_EPILOGUE'
        if cfg.get('PRELOAD_MODE', 1) > 1:
            return False, 'EPILOGUE_LOOP=1 requires PRELOAD_MODE<=1'
        if cfg.get('GELU_VECTOR_WIDTH', 32) != 32:
            return False, 'EPILOGUE_LOOP=1 requires GELU_VECTOR_WIDTH=32'

    # STS_WIDTH constraints
    if cfg.get('STS_WIDTH', 16) == 32:
        if kernel == 'patch_embed':
            return False, 'STS_WIDTH=32 not for patch_embed'
        if cfg.get('BATCH_EPILOGUE', 0) != 1:
            return False, 'STS_WIDTH=32 requires BATCH_EPILOGUE=1'
        if kernel == 'fc1_gelu' and cfg.get('GELU_VECTOR_WIDTH', 32) == 8:
            return False, 'STS_WIDTH=32 incompatible with GELU_VECTOR_WIDTH=8'

    # EPI_SYNC + STAGGER prune
    if cfg.get('EPI_SYNC', 0) == 1 and cfg.get('STAGGER_CYCLES', 80) > 0:
        return False, 'EPI_SYNC=1 makes STAGGER_CYCLES redundant'

    # NUM_PASSES_PARAM constraints
    npp = cfg.get('NUM_PASSES_PARAM', 0)
    if npp != 0:
        if kernel != 'fc2':
            return False, 'NUM_PASSES_PARAM only for fc2'
        if cfg.get('TMA_RESIDUAL', 0) == 0:
            return False, 'NUM_PASSES_PARAM requires TMA_RESIDUAL>0'
        if npp == 4 and cfg.get('W0_RES_FULL', 0) == 1:
            return False, 'W0_RES_FULL hardcodes 2 passes, deadlocks with NUM_PASSES_PARAM=4'

    # DEFERRED_WAIT only for fc2, requires TMA_RESIDUAL>=1
    if cfg.get('DEFERRED_WAIT', 0) == 1:
        if kernel != 'fc2':
            return False, 'DEFERRED_WAIT only for fc2'
        if cfg.get('TMA_RESIDUAL', 0) < 1:
            return False, 'DEFERRED_WAIT requires TMA_RESIDUAL>=1'

    # BIAS_SMEM only meaningful for fc1_gelu and fc2 (BIAS_ADD has combined table, not bias vector)
    if cfg.get('BIAS_SMEM', 0) != 0 and kernel == 'patch_embed':
        return False, 'BIAS_SMEM only for fc1_gelu/fc2'

    return True, 'ok'


def smem_kb(cfg):
    """Compute SMEM usage in KB."""
    n_stages = cfg['N_STAGES']
    num_epi = cfg['NUM_EPI_WARPS']
    stage_bytes = 32768
    off_tmem = n_stages * stage_bytes
    off_tma_mbar = off_tmem + 8
    off_mma_mbar = off_tma_mbar + n_stages * 8
    off_mainloop_mbar = off_mma_mbar + n_stages * 8
    off_epilogue_mbar = off_mainloop_mbar + 16
    tma_res = cfg.get('TMA_RESIDUAL', 0)
    w0_res_pf = cfg.get('W0_RES_PREFETCH', 0)
    w0_res_full = cfg.get('W0_RES_FULL', 0)
    epi_load_warp = cfg.get('EPI_LOAD_WARP', 0)
    if tma_res:
        off_res_mbar = off_epilogue_mbar + 16
        if w0_res_full or epi_load_warp:
            off_res_consumed_mbar = off_res_mbar + num_epi * 8
            off_res_pass_mbar = off_res_consumed_mbar + 8
            mbar_end = off_res_pass_mbar + 8
        elif w0_res_pf:
            off_res_consumed_mbar = off_res_mbar + num_epi * 8
            mbar_end = off_res_consumed_mbar + 8
        else:
            mbar_end = off_res_mbar + num_epi * 8
    else:
        mbar_end = off_epilogue_mbar + 16
    is_fc2 = 'TMA_RESIDUAL' in cfg  # FC2-only param → BIAS_BF16 active
    bias_smem_bytes = (256 * 2 if is_fc2 else 256 * 4) if cfg.get('BIAS_SMEM', 0) else 0
    off_staging = (mbar_end + bias_smem_bytes + 1023) & ~1023
    staging_warp_bytes = 4 * 32 * 128
    smem_total = (off_staging + num_epi * staging_warp_bytes + 127) & ~127
    return smem_total / 1024


def parse_ptxas(stderr):
    """Parse ptxas output for registers and spills."""
    regs = 0
    spills_store = 0
    spills_load = 0
    for line in stderr.splitlines():
        m = re.search(r'Used (\d+) registers', line)
        if m:
            regs = max(regs, int(m.group(1)))
        m = re.search(r'(\d+) bytes spill stores', line)
        if m:
            spills_store = max(spills_store, int(m.group(1)))
        m = re.search(r'(\d+) bytes spill loads', line)
        if m:
            spills_load = max(spills_load, int(m.group(1)))
    return regs, spills_store + spills_load


def parse_result(line):
    """Parse @@RESULT line. Returns (ms, tflops, checksum, valid, c0) or None."""
    m = re.match(r'@@RESULT ms=([\d.]+) tflops=([\d.]+) checksum=([\d.]+) valid=([01]) c0=([\d.]+)', line)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4)), float(m.group(5))


def make_dflags(cfg):
    """Build -D flags string, only for values that differ from defaults."""
    parts = []
    for k, v in sorted(cfg.items()):
        if k in DEFAULTS and v != DEFAULTS[k]:
            # STORE_TIMING=1 suppresses inline stores, making IS dead code
            if k == 'INTERLEAVE_STRATEGY' and cfg.get('STORE_TIMING', 0) == 1:
                continue
            # K_LOOP_UNROLL = N_STAGES is the kernel default; don't emit redundantly
            if k == 'K_LOOP_UNROLL' and v == cfg.get('N_STAGES', DEFAULTS['N_STAGES']):
                continue
            # EPILOGUE_LOOP=1 forces PHASE1_UNROLL=1 in kernel_common.cuh
            if k == 'PHASE1_UNROLL' and cfg.get('EPILOGUE_LOOP', 0) == 1:
                continue
            parts.append(f'-D{k}={v}')
    return ' '.join(parts)


def run_config(cfg, binary_path, src_path, repeat=1):
    """Compile and run a single config. Returns result dict."""
    dflags = make_dflags(cfg)
    cmd = f'{NVCC} {CFLAGS} {dflags} {src_path} -o {binary_path} {LDFLAGS}'

    result = {**cfg, 'status': 'UNKNOWN', 'ms': float('inf'), 'tflops': 0.0,
              'regs': 0, 'spills': 0, 'smem_kb': smem_kb(cfg), 'dflags': dflags}

    # K_LOOP_UNROLL defaults to N_STAGES in the kernel (#define K_LOOP_UNROLL N_STAGES).
    # When N_STAGES differs from default but K_LOOP_UNROLL wasn't explicitly varied,
    # the result dict should reflect the actual binary value (N_STAGES, not 4).
    n_stages = result.get('N_STAGES', DEFAULTS['N_STAGES'])
    klu = result.get('K_LOOP_UNROLL', DEFAULTS['K_LOOP_UNROLL'])
    if klu == DEFAULTS['K_LOOP_UNROLL'] and n_stages != DEFAULTS['N_STAGES']:
        result['K_LOOP_UNROLL'] = n_stages

    # Compile
    try:
        comp = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=COMPILE_TIMEOUT)
    except subprocess.TimeoutExpired:
        result['status'] = 'COMPILE_TIMEOUT'
        return result

    if comp.returncode != 0:
        result['status'] = 'COMPILE_FAIL'
        result['error'] = comp.stderr[-500:] if comp.stderr else ''
        return result

    regs, spills = parse_ptxas(comp.stderr)
    result['regs'] = regs
    result['spills'] = spills

    if regs > 255:
        result['status'] = 'SKIP_REGS'
        return result
    if spills > 0:
        result['status'] = 'SKIP_SPILLS'
        return result

    # Run (potentially multiple times)
    best_ms = float('inf')
    best_tflops = 0.0
    checksum = 0.0
    c0 = 0.0

    for rep in range(repeat):
        try:
            proc = subprocess.Popen([binary_path], stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
            stdout, stderr = proc.communicate(timeout=RUN_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            result['status'] = 'HANG'
            return result

        class _Run:
            pass
        run = _Run()
        run.returncode = proc.returncode
        run.stdout = stdout
        run.stderr = stderr

        if run.returncode != 0:
            result['status'] = 'RUNTIME_ERROR'
            result['error'] = run.stderr[-500:] if run.stderr else ''
            return result

        # Parse @@RESULT
        parsed = None
        for line in run.stdout.splitlines():
            if line.startswith('@@RESULT'):
                parsed = parse_result(line)
                break

        if parsed is None:
            result['status'] = 'NO_RESULT'
            result['error'] = run.stdout[-500:]
            return result

        ms, tflops, cksum, valid_flag, c0_val = parsed

        # Binary self-validates against CPU reference (32 spot checks)
        if not valid_flag:
            result['status'] = 'VALIDATION_FAILED'
            result['ms'] = ms
            result['tflops'] = tflops
            return result

        if ms < best_ms:
            best_ms = ms
            best_tflops = tflops
        checksum = cksum
        c0 = c0_val

    result['status'] = 'OK'
    result['ms'] = best_ms
    result['tflops'] = best_tflops
    return result


def enumerate_configs(sweep_params, fixed):
    """Generate all configs from sweep_params × fixed values."""
    param_names = sorted(sweep_params.keys())
    param_values = [sweep_params[p] for p in param_names]

    for combo in itertools.product(*param_values):
        cfg = dict(fixed)
        cfg.update(zip(param_names, combo))
        yield cfg


def print_table(results, file=sys.stdout):
    """Print sorted results table."""
    # Sort by ms (ascending), with non-OK at the end
    ok_results = [r for r in results if r['status'] == 'OK']
    other_results = [r for r in results if r['status'] != 'OK']
    ok_results.sort(key=lambda r: r['ms'])

    if ok_results:
        best_ms = ok_results[0]['ms']
    else:
        best_ms = None

    header = (f'{"#":>3}  {"STG":>3}  {"EPI":>3}  {"INTER":>5}  {"MBAR":>4}  {"TMEM":>4}  '
              f'{"STAG":>4}  {"PH1U":>4}  {"SNAKE":>5}  {"FUSE":>4}  {"KLU":>3}  '
              f'{"W0U":>3}  {"SMU":>3}  {"PRLD":>4}  {"PFBS":>4}  {"GELU":>4}  '
              f'{"TMAR":>4}  {"BTCH":>4}  {"GVW":>3}  {"STIM":>4}  '
              f'{"ELOP":>4}  {"STSW":>4}  {"ESYN":>4}  {"NPP":>3}  '
              f'{"REGS":>4}  {"SMEM":>7}  '
              f'{"MS":>7}  {"TFLOPS":>7}  {"STATUS"}')
    print(header, file=file)
    print('-' * len(header), file=file)

    for i, r in enumerate(ok_results + other_results, 1):
        status = r['status']
        if status == 'OK' and best_ms is not None and r['ms'] == best_ms:
            status = 'BEST'

        ms_str = f'{r["ms"]:.3f}' if r['ms'] < float('inf') else '  -'
        tflops_str = f'{r["tflops"]:.0f}' if r['tflops'] > 0 else '  -'

        print(f'{i:>3}  {r["N_STAGES"]:>3}  {r["NUM_EPI_WARPS"]:>3}  '
              f'{r["INTERLEAVE_STRATEGY"]:>5}  {r["MBAR_EARLY"]:>4}  '
              f'{r["TMEM_LOAD_WIDTH"]:>4}  {r["STAGGER_CYCLES"]:>4}  '
              f'{r["PHASE1_UNROLL"]:>4}  {r["SNAKE_ORDER"]:>5}  '
              f'{r["CVT_ADD_FUSED"]:>4}  {r["K_LOOP_UNROLL"]:>3}  '
              f'{r["W0_LOOP_UNROLL"]:>3}  {r["SUB_MMA_UNROLL"]:>3}  '
              f'{r["PRELOAD_MODE"]:>4}  {r["PREFETCH_BEFORE_STORE"]:>4}  '
              f'{r["GELU_VARIANT"]:>4}  '
              f'{r.get("TMA_RESIDUAL", 0):>4}  '
              f'{r.get("BATCH_EPILOGUE", 0):>4}  '
              f'{r.get("GELU_VECTOR_WIDTH", 32):>3}  '
              f'{r.get("STORE_TIMING", 0):>4}  '
              f'{r.get("EPILOGUE_LOOP", 0):>4}  '
              f'{r.get("STS_WIDTH", 16):>4}  '
              f'{r.get("EPI_SYNC", 0):>4}  '
              f'{r.get("NUM_PASSES_PARAM", 0):>3}  '
              f'{r["regs"]:>4}  {r["smem_kb"]:>6.1f}K  '
              f'{ms_str:>7}  {tflops_str:>7}  {status}',
              file=file)


def write_csv(results, path):
    """Write results to CSV."""
    ok_results = [r for r in results if r['status'] == 'OK']
    other_results = [r for r in results if r['status'] != 'OK']
    ok_results.sort(key=lambda r: r['ms'])
    all_sorted = ok_results + other_results

    fields = ['N_STAGES', 'NUM_EPI_WARPS', 'INTERLEAVE_STRATEGY', 'MBAR_EARLY',
              'TMEM_LOAD_WIDTH', 'STAGGER_CYCLES', 'PHASE1_UNROLL', 'SNAKE_ORDER',
              'CVT_ADD_FUSED', 'K_LOOP_UNROLL', 'W0_LOOP_UNROLL', 'SUB_MMA_UNROLL',
              'PRELOAD_MODE', 'PREFETCH_BEFORE_STORE', 'GELU_VARIANT', 'TMA_RESIDUAL',
              'BATCH_EPILOGUE', 'GELU_VECTOR_WIDTH', 'STORE_TIMING',
              'EPILOGUE_LOOP', 'STS_WIDTH', 'EPI_SYNC', 'NUM_PASSES_PARAM',
              'regs', 'spills', 'smem_kb', 'ms', 'tflops', 'status', 'dflags']

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in all_sorted:
            writer.writerow(r)


def run_sweep(sweep_params, fixed, src_path, repeat=1, kernel='patch_embed', seen_cfgs=None):
    """Run a sweep over sweep_params with fixed values pinned.

    seen_cfgs: optional set of dflags strings already evaluated (for top-k dedup).
    """
    configs = list(enumerate_configs(sweep_params, fixed))

    # Pre-compile pruning
    valid = []
    pruned = 0
    deduped = 0
    for cfg in configs:
        ok, reason = is_valid(cfg, kernel=kernel)
        if not ok:
            pruned += 1
            continue
        # Dedup across branches (top-k mode)
        if seen_cfgs is not None:
            sig = make_dflags(cfg)
            if sig in seen_cfgs:
                deduped += 1
                continue
            seen_cfgs.add(sig)
        valid.append(cfg)

    total = len(configs)
    dedup_msg = f', {deduped} deduped' if deduped else ''
    print(f'Sweep: {total} total, {len(valid)} valid, {pruned} pruned{dedup_msg}')
    if not valid:
        print('No valid configs!')
        return []

    # Estimate time
    est_min = len(valid) * 5 / 60  # ~5s compile+run per config
    print(f'Estimated time: ~{est_min:.0f} min ({len(valid)} configs × ~5s each)')
    print()

    results = []
    binary_path = os.path.join(tempfile.gettempdir(), 'sweep_bin')

    for i, cfg in enumerate(valid):
        dflags = make_dflags(cfg)
        label = dflags if dflags else '(defaults)'
        print(f'[{i+1}/{len(valid)}] {label} ...', end=' ', flush=True)

        t0 = time.time()
        result = run_config(cfg, binary_path, src_path, repeat=repeat)
        dt = time.time() - t0

        if result['status'] == 'OK':
            print(f'{result["ms"]:.3f} ms / {result["tflops"]:.0f} TFLOPS  '
                  f'({result["regs"]} regs, {dt:.1f}s)')
        else:
            print(f'{result["status"]}  ({dt:.1f}s)')

        results.append(result)

    # Cleanup
    if os.path.exists(binary_path):
        os.unlink(binary_path)

    return results


def get_best(results):
    """Return the best OK config, or None."""
    ok = [r for r in results if r['status'] == 'OK']
    if not ok:
        return None
    return min(ok, key=lambda r: r['ms'])


def get_top_k(results, tier_params, k=3):
    """Return up to k distinct param-value combos from the best OK configs.

    "Distinct" means configs that differ in at least one tier_param value.
    This avoids running the next tier with k copies of the same pinned state.
    """
    ok = sorted([r for r in results if r['status'] == 'OK'], key=lambda r: r['ms'])
    if not ok:
        return []
    seen = set()
    top = []
    for r in ok:
        sig = tuple(r[p] for p in tier_params)
        if sig not in seen:
            seen.add(sig)
            top.append(r)
            if len(top) >= k:
                break
    return top


def compute_eta_sq_inline(results, param):
    """One-way ANOVA eta-squared from result dicts. No scipy needed."""
    ok = [r for r in results if r['status'] == 'OK']
    if len(ok) < 3:
        return None
    ms_vals = [r['ms'] for r in ok]
    grand_mean = sum(ms_vals) / len(ms_vals)
    ss_total = sum((m - grand_mean) ** 2 for m in ms_vals)
    if ss_total == 0:
        return 0.0
    levels = {}
    for r in ok:
        levels.setdefault(r[param], []).append(r['ms'])
    if len(levels) < 2:
        return None
    ss_between = sum(len(v) * (sum(v)/len(v) - grand_mean)**2 for v in levels.values())
    return ss_between / ss_total


def print_eta_summary(results, params):
    """Print inline eta-squared for each swept param."""
    parts = []
    for p in params:
        eta = compute_eta_sq_inline(results, p)
        if eta is not None:
            parts.append(f'{p}={eta:.3f}')
    if parts:
        print(f'  η²: {", ".join(parts)}')


def print_top_lock_summary(results, all_params, top_ns=(5, 10, 20)):
    """Print top-k universality analysis. Returns locks at the widest top-N."""
    ok = [r for r in results if r['status'] == 'OK']
    if len(ok) < 5:
        return []
    best_locks = []
    parts = []
    for p in all_params:
        # Find the widest top-N where this param is still locked
        max_n = 0
        locked_val = None
        base_rate = 1.0
        ok_sorted = sorted(ok, key=lambda r: r['ms'])
        for n in top_ns:
            if n > len(ok_sorted):
                break
            top_vals = set(r[p] for r in ok_sorted[:n])
            if len(top_vals) == 1:
                max_n = n
                locked_val = top_vals.pop()
                n_with = sum(1 for r in ok_sorted if r[p] == locked_val)
                base_rate = n_with / len(ok_sorted)
            else:
                break
        if max_n >= 5 and base_rate < 0.70:
            parts.append(f'{p}={locked_val}(top{max_n},{base_rate:.0%})')
            best_locks.append((p, locked_val, base_rate, max_n))
    if parts:
        print(f'  top-lock: {", ".join(parts)}')
    return best_locks


def load_best_from_csv(kernel):
    """Load the best config from the most recent sweep CSV for this kernel."""
    csv_dir = os.path.join(ROOT_DIR, 'data')
    pattern = os.path.join(csv_dir, f'sweep_{kernel}*.csv')
    files = sorted(glob_mod.glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        reader = csv.DictReader(f)
        best_ms = float('inf')
        best_cfg = None
        for row in reader:
            if row.get('status') != 'OK':
                continue
            ms = float(row['ms'])
            if ms < best_ms:
                best_ms = ms
                best_cfg = {k: int(row[k]) for k in DEFAULTS if k in row}
    if best_cfg:
        print(f'Loaded baseline from {os.path.basename(files[0])} ({best_ms:.3f} ms)')
    return best_cfg


def main():
    parser = argparse.ArgumentParser(description='Grid search for SigLIP kernel parameters')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--tier', choices=['1', '2', '3', '4', '5', 'all'],
                      help='Tiered search (1=structure, 2=epilogue, 3=tuning, 4=scheduling, 5=GELU variant, all=sequential)')
    mode.add_argument('--full-cross', action='store_true',
                      help='Full cross-product of all parameters')
    mode.add_argument('--cross', action='store_true',
                      help='Full cross-product of proven-meaningful params (KERNEL_CROSS_PARAMS), noise pinned at KERNEL_BASES')
    mode.add_argument('--interact', choices=list(INTERACTIONS.keys()) + ['all'],
                      help='Cross-tier interaction sweep (named group or all)')
    parser.add_argument('--kernel', choices=list(KERNELS.keys()), default='patch_embed',
                        help='Kernel to sweep (default: patch_embed)')
    parser.add_argument('--fix', nargs='*', default=[], metavar='PARAM=VAL',
                        help='Pin specific parameters (e.g. --fix MBAR_EARLY=1 N_STAGES=4)')
    parser.add_argument('--base', nargs='*', default=[], metavar='PARAM=VAL',
                        help='Base config to inherit from (e.g. from @@GRID_WINNER output)')
    parser.add_argument('--only', nargs='*', default=[], metavar='PARAM',
                        help='Sweep only these parameters, pin rest at best-known config')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Run each config N times, report best (default: 1)')
    parser.add_argument('--top-k', type=int, default=3, dest='top_k',
                        help='Carry top-k branches between tiers (default: 3, 1=greedy)')
    parser.add_argument('--no-interact', action='store_true', dest='no_interact',
                        help='Skip interaction sweeps after --tier all')
    parser.add_argument('--csv', default=None,
                        help='Output CSV path (default: data/sweep_<kernel>.csv)')
    args = parser.parse_args()

    # Default mode: --cross when no explicit mode given
    if not args.only and not args.tier and not args.full_cross and not args.cross and not args.interact:
        args.cross = True

    src_path = os.path.join(ROOT_DIR, KERNELS[args.kernel])
    if args.csv is None:
        args.csv = os.path.join(ROOT_DIR, 'data', f'sweep_{args.kernel}.csv')
    print(f'Kernel: {args.kernel} ({KERNELS[args.kernel]})')

    # Parse --fix
    fixed_overrides = {}
    for f in args.fix:
        if '=' not in f:
            print(f'Error: --fix arg must be PARAM=VAL, got: {f}', file=sys.stderr)
            sys.exit(1)
        k, v = f.split('=', 1)
        if k not in DEFAULTS:
            print(f'Error: unknown parameter: {k}', file=sys.stderr)
            sys.exit(1)
        fixed_overrides[k] = int(v)

    # Parse --base
    base_overrides = {}
    for b in args.base:
        if '=' not in b:
            print(f'Error: --base arg must be PARAM=VAL, got: {b}', file=sys.stderr)
            sys.exit(1)
        k, v = b.split('=', 1)
        if k not in DEFAULTS:
            print(f'Error: unknown parameter: {k}', file=sys.stderr)
            sys.exit(1)
        base_overrides[k] = int(v)

    # Validate --only
    for p in args.only:
        if p not in RANGES:
            print(f'Error: unknown parameter: {p}', file=sys.stderr)
            sys.exit(1)

    all_results = []

    if args.only:
        # Custom sweep: only specified params
        # Baseline: --base > CSV best > DEFAULTS
        sweep_params = {p: RANGES[p] for p in args.only}
        if base_overrides:
            fixed = dict(DEFAULTS)
            fixed.update(base_overrides)
        else:
            csv_best = load_best_from_csv(args.kernel)
            fixed = csv_best if csv_best else dict(DEFAULTS)
        fixed.update(fixed_overrides)
        for p in args.only:
            fixed.pop(p, None)

        print(f'=== Custom sweep: {", ".join(args.only)} ===')
        results = run_sweep(sweep_params, fixed, src_path, repeat=args.repeat, kernel=args.kernel)
        all_results.extend(results)
        print_eta_summary(results, list(sweep_params.keys()))

    elif args.cross:
        # Full cross-product of proven-meaningful params, noise pinned at KERNEL_BASES
        if args.kernel not in KERNEL_CROSS_PARAMS:
            print(f'Error: no KERNEL_CROSS_PARAMS for {args.kernel}', file=sys.stderr)
            sys.exit(1)
        cross_params = KERNEL_CROSS_PARAMS[args.kernel]
        sweep_params = {p: RANGES[p] for p in cross_params if p not in fixed_overrides}
        fixed = dict(DEFAULTS)
        fixed.update(KERNEL_BASES.get(args.kernel, {}))
        fixed.update(base_overrides)
        fixed.update(fixed_overrides)
        for p in sweep_params:
            fixed.pop(p, None)

        n_cross = 1
        for v in sweep_params.values():
            n_cross *= len(v)
        pinned = {p: fixed[p] for p in sorted(fixed) if p in DEFAULTS and fixed[p] != DEFAULTS[p]
                  and p not in sweep_params}
        print(f'=== Cross-product: {", ".join(sweep_params.keys())} ({n_cross} raw configs) ===')
        if pinned:
            print(f'Pinned: {" ".join(f"{p}={v}" for p, v in sorted(pinned.items()))}')
        results = run_sweep(sweep_params, fixed, src_path, repeat=args.repeat, kernel=args.kernel)
        all_results.extend(results)
        print_eta_summary(results, list(sweep_params.keys()))

        best = get_best(results)
        if best:
            composite_dflags = make_dflags(best)
            print(f'\n@@GRID_WINNER {composite_dflags or "(defaults)"}')

    elif args.full_cross:
        # Full cross-product of ALL parameters (huge — use --cross instead)
        sweep_params = dict(RANGES)
        fixed = dict(fixed_overrides)
        for p in sweep_params:
            fixed.pop(p, None)

        print('=== Full cross-product sweep ===')
        results = run_sweep(sweep_params, fixed, src_path, repeat=args.repeat, kernel=args.kernel)
        all_results.extend(results)

    elif args.interact:
        # Cross-tier interaction sweep
        groups_to_run = []
        if args.interact == 'all':
            for name, group in INTERACTIONS.items():
                if args.kernel in group['kernels']:
                    groups_to_run.append((name, group))
        else:
            group = INTERACTIONS[args.interact]
            if args.kernel not in group['kernels']:
                print(f'Error: interaction {args.interact!r} not applicable to {args.kernel}',
                      file=sys.stderr)
                sys.exit(1)
            groups_to_run.append((args.interact, group))

        if not groups_to_run:
            print(f'No applicable interactions for {args.kernel}')
            sys.exit(0)

        # Baseline: --base > CSV best > DEFAULTS
        if base_overrides:
            baseline = dict(DEFAULTS)
            baseline.update(base_overrides)
        else:
            csv_best = load_best_from_csv(args.kernel)
            baseline = csv_best if csv_best else dict(DEFAULTS)
        baseline.update(fixed_overrides)

        for name, group in groups_to_run:
            sweep_params = {p: RANGES[p] for p in group['params']}
            fixed = dict(baseline)
            for p in sweep_params:
                fixed.pop(p, None)

            print(f'\n=== Interaction: {name} ({", ".join(sweep_params.keys())}) ===')
            results = run_sweep(sweep_params, fixed, src_path, repeat=args.repeat, kernel=args.kernel)
            all_results.extend(results)
            print_eta_summary(results, group['params'])

            int_best = get_best(results)
            if int_best:
                print(f'  Best: {make_dflags(int_best) or "(defaults)"} '
                      f'→ {int_best["ms"]:.3f} ms / {int_best["tflops"]:.0f} TFLOPS')

    elif args.tier == 'all':
        # Sequential tiers, top-k pinning (explores k branches per tier)
        # Uses per-kernel tiers + bases when available (balanced-η² ordering).
        k = args.top_k
        use_kernel_tiers = args.kernel in KERNEL_TIERS
        if use_kernel_tiers:
            ktiers = KERNEL_TIERS[args.kernel]
            tier_nums = sorted(ktiers.keys())
            base = dict(DEFAULTS)
            base.update(KERNEL_BASES.get(args.kernel, {}))
            swept_params = set()
            for tn in tier_nums:
                swept_params.update(ktiers[tn])
            skipped = [p for p in RANGES if p not in swept_params
                       and p not in KERNEL_BASES.get(args.kernel, {})]
            if skipped:
                print(f'Per-kernel tiers for {args.kernel}: skipping {", ".join(skipped)}')
            pinned = {p: base[p] for p in DEFAULTS if p not in swept_params and p in base}
            if pinned:
                pinned_str = ' '.join(f'{p}={v}' for p, v in sorted(pinned.items()) if v != DEFAULTS[p])
                if pinned_str:
                    print(f'Pinned at winner values: {pinned_str}')
        else:
            ktiers = TIER_PARAMS
            tier_nums = [1, 2, 3, 4, 5]
            base = dict(DEFAULTS)

        branches = [dict(base)]
        branches[0].update(fixed_overrides)
        auto_pinned = {}  # params locked at top across tiers (informative base rate)

        for tier_num in tier_nums:
            tier_params = ktiers[tier_num]
            sweep_params = {p: RANGES[p] for p in tier_params if p not in fixed_overrides}
            if not sweep_params:
                print(f'\n=== Tier {tier_num}: all params pinned by --fix, skipping ===')
                continue

            print(f'\n=== Tier {tier_num}: {", ".join(sweep_params.keys())} ===')

            # Run sweep for each branch, dedup configs to avoid re-running
            tier_results = []
            seen_cfgs = set()
            for bi, branch in enumerate(branches):
                fixed = {k_: v for k_, v in branch.items() if k_ not in sweep_params}
                if len(branches) > 1:
                    print(f'  Branch {bi+1}/{len(branches)}: {make_dflags(branch) or "(defaults)"}')
                results = run_sweep(sweep_params, fixed, src_path, repeat=args.repeat, kernel=args.kernel,
                                    seen_cfgs=seen_cfgs)
                for r in results:
                    r['_branch_idx'] = bi
                tier_results.extend(results)

            all_results.extend(tier_results)
            print_eta_summary(tier_results, list(sweep_params.keys()))

            # Top-lock analysis: find params universally locked at the top
            # Uses all params seen so far (not just this tier) to catch cross-tier locks
            all_swept_so_far = [p for tn in tier_nums if tn <= tier_num
                                for p in ktiers.get(tn, [])]
            tier_locks = print_top_lock_summary(tier_results, all_swept_so_far)
            for p, v, _br, _n in tier_locks:
                auto_pinned[p] = v

            top = get_top_k(tier_results, tier_params, k=k)
            if top:
                # Dynamic k: reduce branching when winner is clear
                # Conservative: never drop below 2 branches — interactions
                # may flip rankings when combined with later-tier params
                k_eff = len(top)
                if len(top) >= 2:
                    gap_pct = (top[1]['ms'] - top[0]['ms']) / top[0]['ms'] * 100
                    if gap_pct > 5.0:
                        k_eff = 2
                    elif gap_pct > 2.0:
                        k_eff = min(k_eff, 2)

                # Structural params: ensure each distinct value survives
                branch_ps = BRANCH_PARAMS.get(args.kernel, set()) & set(tier_params)
                if branch_ps:
                    needed = set()
                    for t in top:
                        needed.add(tuple(t[p] for p in sorted(branch_ps)))
                    k_min = len(needed)
                    if k_eff < k_min:
                        # Find minimum k_eff that covers all structural values
                        covered = set()
                        for i, t in enumerate(top):
                            covered.add(tuple(t[p] for p in sorted(branch_ps)))
                            if len(covered) >= k_min:
                                k_eff = i + 1
                                break
                        bp_str = ', '.join(sorted(branch_ps))
                        print(f'\n  Branch params ({bp_str}): k_eff raised to {k_eff} '
                              f'(preserving {k_min} distinct values)')

                if k_eff < len(top) and not branch_ps:
                    print(f'\n  Gap: {gap_pct:.2f}% → k_eff={k_eff} (from {len(top)})')

                # Build new branches from top-k winners
                new_branches = []
                for i, t in enumerate(top[:k_eff]):
                    parent_idx = t.get('_branch_idx', 0)
                    b = dict(branches[parent_idx])
                    for p in tier_params:
                        b[p] = t[p]
                    # Pin auto-locked params into branches
                    for p, v in auto_pinned.items():
                        b[p] = v
                    new_branches.append(b)

                branches = new_branches
                best = top[0]
                tier_label = f'Tier {tier_num} winner'
                if len(top) > 1:
                    tier_label += f' (top-{len(top)})'
                print(f'\n{tier_label}: {make_dflags(best) or "(defaults)"} '
                      f'→ {best["ms"]:.3f} ms / {best["tflops"]:.0f} TFLOPS')
                for i, t in enumerate(top[1:], 2):
                    print(f'  #{i}: {make_dflags(t) or "(defaults)"} '
                          f'→ {t["ms"]:.3f} ms / {t["tflops"]:.0f} TFLOPS')
            else:
                print(f'\nTier {tier_num}: no valid results!')

        # Run applicable cross-tier interaction groups
        if not args.no_interact:
            best_so_far = get_best(all_results)
            any_interaction = False
            for name, group in INTERACTIONS.items():
                if args.kernel not in group['kernels']:
                    continue
                # Skip interactions involving params not in any tier (noise params)
                if use_kernel_tiers:
                    if not all(p in swept_params for p in group['params']):
                        skipped_in = [p for p in group['params'] if p not in swept_params]
                        print(f'\n=== Interaction: {name} — skipped (noise params: {", ".join(skipped_in)}) ===')
                        continue
                any_interaction = True
                sweep_params = {p: RANGES[p] for p in group['params']}

                print(f'\n=== Interaction: {name} ({", ".join(sweep_params.keys())}) ===')

                group_results = []
                seen_cfgs = set()
                for bi, branch in enumerate(branches):
                    fixed = dict(branch)
                    for p in sweep_params:
                        fixed.pop(p, None)
                    if len(branches) > 1:
                        print(f'  Branch {bi+1}/{len(branches)}: '
                              f'{make_dflags(branch) or "(defaults)"}')
                    results = run_sweep(sweep_params, fixed, src_path,
                                        repeat=args.repeat, kernel=args.kernel,
                                        seen_cfgs=seen_cfgs)
                    group_results.extend(results)

                all_results.extend(group_results)
                print_eta_summary(group_results, group['params'])

                int_best = get_best(group_results)
                if int_best:
                    print(f'  Best: {make_dflags(int_best) or "(defaults)"} '
                          f'→ {int_best["ms"]:.3f} ms / {int_best["tflops"]:.0f} TFLOPS')
                    if best_so_far is None or int_best['ms'] < best_so_far['ms']:
                        for p in group['params']:
                            branches[0][p] = int_best[p]
                        best_so_far = int_best
                        print(f'  ** New overall best from interaction {name!r} **')

            if not any_interaction:
                print(f'\nNo applicable interactions for {args.kernel}')

        # Print auto-pinned params from top-lock analysis
        if auto_pinned:
            pin_parts = [f'{p}={v}' for p, v in sorted(auto_pinned.items())]
            print(f'\nTop-locked params: {", ".join(pin_parts)}')

        # Print the composite pinned winner (best branch)
        composite_dflags = make_dflags(branches[0])
        print(f'\n@@GRID_WINNER {composite_dflags or "(defaults)"}')

    else:
        # Single tier
        tier_num = int(args.tier)
        tier_params = TIER_PARAMS[tier_num]
        sweep_params = {p: RANGES[p] for p in tier_params if p not in fixed_overrides}
        fixed = dict(DEFAULTS)
        fixed.update(fixed_overrides)
        for p in sweep_params:
            fixed.pop(p, None)

        print(f'=== Tier {tier_num}: {", ".join(sweep_params.keys())} ===')
        results = run_sweep(sweep_params, fixed, src_path, repeat=args.repeat, kernel=args.kernel)
        all_results.extend(results)
        print_eta_summary(results, list(sweep_params.keys()))

    # Summary
    if all_results:
        print('\n' + '=' * 80)
        print('RESULTS (sorted by ms)')
        print('=' * 80)
        print_table(all_results)

        write_csv(all_results, args.csv)
        print(f'\nCSV written to {args.csv}')

        best = get_best(all_results)
        if best:
            dflags = make_dflags(best)
            print(f'\nBest: {dflags or "(defaults)"} → {best["ms"]:.3f} ms / '
                  f'{best["tflops"]:.0f} TFLOPS ({best["regs"]} regs)')


if __name__ == '__main__':
    main()
