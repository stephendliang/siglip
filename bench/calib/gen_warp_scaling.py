#!/usr/bin/env python3
"""
Generate multi-warp scheduling calibration benchmarks for SM100a.

Measures how the warp scheduler arbitrates across multiple warps sharing
the same SM. Single-warp calibration (gen_tput/gen_lat/gen_conflict) tells
us per-warp pipe throughputs. These tests reveal:

  S-tests: Same-pipe throughput scaling (1-8 warps all doing STS/FFMA/etc.)
  X-tests: Cross-pipe independence (N warps on different pipes)
  F-tests: Foreground/background interference (FC2-like: epi + kloop warps)
  B-tests: BAR.SYNC contention effect (Template B hypothesis)

All tests launch a single CTA so every warp lands on the same SM.
Per-warp clock64() timing for individual throughput measurement.

Build:
  nvcc -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 \
       bench/calib/gen_warp_scaling.cu -o warp-scaling

Run:
  ./warp-scaling
"""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTFILE = os.path.join(SCRIPT_DIR, 'gen_warp_scaling.cu')

REPS = 512
WARMUP = 32
WARP_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8]
MAX_WARPS = max(WARP_COUNTS)
WARMUP_LAUNCHES = 3
MEASURE_LAUNCHES = 10

# Pipe workloads — each is a (name, setup, loop_body, sink) tuple.
# setup: per-warp variable declarations (uses {tid}, {warp}, {lane})
# loop_body: one iteration's work (repeated REPS times)
# sink: prevents dead-code elimination (writes to out[])

PIPE_WORKLOADS = {
    'STS': {
        'desc': 'STS.128 (st.shared.v4.b32)',
        'smem_per_warp': 1024,  # 256 ints = 1KB, enough for 4 STS.128 offsets
        'setup': (
            '    uint32_t _base = _smem_base + {lane} * 16;\n'
            '    unsigned _v0=0xdead, _v1=0xcafe, _v2=0x1234, _v3=0x9abc;\n'
        ),
        'loop_body': (
            '        uint32_t _a = _base + ((_i & 3) << 6);\n'
            '        asm volatile(\n'
            '            "st.shared.v4.b32 [%0], {%1,%2,%3,%4};\\n\\t"\n'
            '            "st.shared.v4.b32 [%0+16], {%1,%2,%3,%4};\\n\\t"\n'
            '            "st.shared.v4.b32 [%0+32], {%1,%2,%3,%4};\\n\\t"\n'
            '            "st.shared.v4.b32 [%0+48], {%1,%2,%3,%4};\\n\\t"\n'
            '            :: "r"(_a), "r"(_v0), "r"(_v1), "r"(_v2), "r"(_v3)\n'
            '            : "memory"\n'
            '        );\n'
        ),
        'insns_per_iter': 4,
        'sink': '    ((volatile unsigned*)(_dyn_smem + _warp * 1024))[{lane}] = _v0;',
    },
    'FFMA': {
        'desc': 'FFMA (fma.rn.f32)',
        'smem_per_warp': 0,
        'setup': (
            '    float _a0=1.001f+{lane}, _a1=1.002f+{lane}, _a2=1.003f+{lane}, _a3=1.004f+{lane};\n'
            '    float _s0=1.0001f, _s1=0.9999f;\n'
        ),
        'loop_body': (
            '        asm volatile(\n'
            '            "fma.rn.f32 %0, %0, %4, %5;\\n\\t"\n'
            '            "fma.rn.f32 %1, %1, %4, %5;\\n\\t"\n'
            '            "fma.rn.f32 %2, %2, %4, %5;\\n\\t"\n'
            '            "fma.rn.f32 %3, %3, %4, %5;\\n\\t"\n'
            '            : "+f"(_a0), "+f"(_a1), "+f"(_a2), "+f"(_a3)\n'
            '            : "f"(_s0), "f"(_s1)\n'
            '        );\n'
        ),
        'insns_per_iter': 4,
        'sink': '    ((volatile float*)(out + 64))[{tid}] = _a0 + _a1 + _a2 + _a3;',
    },
    'HFMA2': {
        'desc': 'HFMA2 (fma.rn.bf16x2)',
        'smem_per_warp': 0,
        'setup': (
            '    unsigned _h0=0x3f80+{lane}, _h1=0x3f81+{lane}, _h2=0x3f82+{lane}, _h3=0x3f83+{lane};\n'
            '    unsigned _hs=0x3f803f80;\n'
        ),
        'loop_body': (
            '        asm volatile(\n'
            '            "fma.rn.bf16x2 %0, %0, %4, %0;\\n\\t"\n'
            '            "fma.rn.bf16x2 %1, %1, %4, %1;\\n\\t"\n'
            '            "fma.rn.bf16x2 %2, %2, %4, %2;\\n\\t"\n'
            '            "fma.rn.bf16x2 %3, %3, %4, %3;\\n\\t"\n'
            '            : "+r"(_h0), "+r"(_h1), "+r"(_h2), "+r"(_h3)\n'
            '            : "r"(_hs)\n'
            '        );\n'
        ),
        'insns_per_iter': 4,
        'sink': '    out[64 + {tid}] = (long long)(_h0 + _h1 + _h2 + _h3);',
    },
    'LDS': {
        'desc': 'LDS.128 (ld.shared.v4.b32)',
        'smem_per_warp': 2048,  # reads up to _base+768+16=1520 bytes; 1024 was too small
        'setup': (
            '    uint32_t _base = _smem_base + {lane} * 16;\n'
            '    unsigned _r0=0, _r1=0, _r2=0, _r3=0;\n'
            '    unsigned _ra0=0, _ra1=0, _ra2=0, _ra3=0;\n'
        ),
        'loop_body': (
            '        uint32_t _a = _base + ((_i & 0xf) << 4);\n'
            '        asm volatile(\n'
            '            "ld.shared.v4.b32 {%0,%1,%2,%3}, [%8];\\n\\t"\n'
            '            "add.u32 %4, %4, %0;\\n\\t"\n'
            '            "ld.shared.v4.b32 {%0,%1,%2,%3}, [%8+256];\\n\\t"\n'
            '            "add.u32 %5, %5, %0;\\n\\t"\n'
            '            "ld.shared.v4.b32 {%0,%1,%2,%3}, [%8+512];\\n\\t"\n'
            '            "add.u32 %6, %6, %0;\\n\\t"\n'
            '            "ld.shared.v4.b32 {%0,%1,%2,%3}, [%8+768];\\n\\t"\n'
            '            "add.u32 %7, %7, %0;\\n\\t"\n'
            '            : "=r"(_r0), "=r"(_r1), "=r"(_r2), "=r"(_r3),\n'
            '              "+r"(_ra0), "+r"(_ra1), "+r"(_ra2), "+r"(_ra3)\n'
            '            : "r"(_a)\n'
            '            : "memory"\n'
            '        );\n'
        ),
        'insns_per_iter': 4,
        'sink': '    out[64 + {tid}] = (long long)(_ra0 + _ra1 + _ra2 + _ra3);',
    },
    'F2FP': {
        'desc': 'F2FP (cvt.rn.bf16x2.f32)',
        'smem_per_warp': 0,
        'setup': (
            '    float _f0=1.0f+{lane}, _f1=2.0f+{lane};\n'
            '    unsigned _cv=0;\n'
        ),
        'loop_body': (
            '        asm volatile(\n'
            '            "cvt.rn.bf16x2.f32 %0, %2, %1;\\n\\t"\n'
            '            "cvt.rn.bf16x2.f32 %0, %1, %2;\\n\\t"\n'
            '            "cvt.rn.bf16x2.f32 %0, %2, %1;\\n\\t"\n'
            '            "cvt.rn.bf16x2.f32 %0, %1, %2;\\n\\t"\n'
            '            : "+r"(_cv)\n'
            '            : "f"(_f0), "f"(_f1)\n'
            '        );\n'
        ),
        'insns_per_iter': 4,
        'sink': '    out[64 + {tid}] = (long long)_cv;',
    },
    'HADD2': {
        'desc': 'HADD2 (add.rn.bf16x2)',
        'smem_per_warp': 0,
        'setup': (
            '    unsigned _h0=0x3f80+{lane}, _h1=0x3f81+{lane}, _h2=0x3f82+{lane}, _h3=0x3f83+{lane};\n'
            '    unsigned _hs=0x00010001;\n'
        ),
        'loop_body': (
            '        asm volatile(\n'
            '            "add.rn.bf16x2 %0, %0, %4;\\n\\t"\n'
            '            "add.rn.bf16x2 %1, %1, %4;\\n\\t"\n'
            '            "add.rn.bf16x2 %2, %2, %4;\\n\\t"\n'
            '            "add.rn.bf16x2 %3, %3, %4;\\n\\t"\n'
            '            : "+r"(_h0), "+r"(_h1), "+r"(_h2), "+r"(_h3)\n'
            '            : "r"(_hs)\n'
            '        );\n'
        ),
        'insns_per_iter': 4,
        'sink': '    out[64 + {tid}] = (long long)(_h0 + _h1 + _h2 + _h3);',
    },
    'NOP': {
        'desc': 'NOP (baseline — scheduler overhead only)',
        'smem_per_warp': 0,
        'setup': '    unsigned _nop = {lane};\n',
        'loop_body': (
            '        asm volatile(\n'
            '            "mov.b32 %0, %0;\\n\\t"\n'
            '            "mov.b32 %0, %0;\\n\\t"\n'
            '            "mov.b32 %0, %0;\\n\\t"\n'
            '            "mov.b32 %0, %0;\\n\\t"\n'
            '            : "+r"(_nop)\n'
            '        );\n'
        ),
        'insns_per_iter': 4,
        'sink': '    out[64 + {tid}] = (long long)_nop;',
    },
    'LDG': {
        'desc': 'LDG.128 (ld.global.v4.b32) — TEX/L1 pipe',
        'smem_per_warp': 0,
        'setup': (
            '    const char* _gbase = (const char*)results + {lane} * 64;\n'
            '    unsigned _g0=0, _g1=0, _g2=0, _g3=0;\n'
            '    unsigned _ga0=0, _ga1=0, _ga2=0, _ga3=0;\n'
        ),
        'loop_body': (
            '        {\n'
            '        const unsigned* _lp = (const unsigned*)(_gbase + ((_i & 0xf) << 4));\n'
            '        asm volatile(\n'
            '            "ld.global.v4.b32 {%0,%1,%2,%3}, [%8];\\n\\t"\n'
            '            "add.u32 %4, %4, %0;\\n\\t"\n'
            '            "ld.global.v4.b32 {%0,%1,%2,%3}, [%8+256];\\n\\t"\n'
            '            "add.u32 %5, %5, %0;\\n\\t"\n'
            '            "ld.global.v4.b32 {%0,%1,%2,%3}, [%8+512];\\n\\t"\n'
            '            "add.u32 %6, %6, %0;\\n\\t"\n'
            '            "ld.global.v4.b32 {%0,%1,%2,%3}, [%8+768];\\n\\t"\n'
            '            "add.u32 %7, %7, %0;\\n\\t"\n'
            '            : "=r"(_g0), "=r"(_g1), "=r"(_g2), "=r"(_g3),\n'
            '              "+r"(_ga0), "+r"(_ga1), "+r"(_ga2), "+r"(_ga3)\n'
            '            : "l"(_lp)\n'
            '            : "memory"\n'
            '        );\n'
            '        }\n'
        ),
        'insns_per_iter': 4,
        'sink': '    out[64 + {tid}] = (long long)(_ga0 + _ga1 + _ga2 + _ga3);',
    },
    'STS_ILP1': {
        'desc': 'STS.128 ILP=1 (sub-partition probe)',
        'smem_per_warp': 1024,
        'setup': (
            '    uint32_t _base = _smem_base + {lane} * 16;\n'
            '    unsigned _v0=0xdead, _v1=0xcafe, _v2=0x1234, _v3=0x9abc;\n'
        ),
        'loop_body': (
            '        uint32_t _a = _base + ((_i & 3) << 6);\n'
            '        asm volatile(\n'
            '            "st.shared.v4.b32 [%0], {%1,%2,%3,%4};\\n\\t"\n'
            '            :: "r"(_a), "r"(_v0), "r"(_v1), "r"(_v2), "r"(_v3)\n'
            '            : "memory"\n'
            '        );\n'
        ),
        'insns_per_iter': 1,
        'sink': '    ((volatile unsigned*)(_dyn_smem + _warp * 1024))[{lane}] = _v0;',
    },
    'LDG_L2': {
        'desc': 'LDG.128 (ld.global.v4.b32) — forced L2 miss',
        'smem_per_warp': 0,
        'setup': (
            '    /* Stride through large buffer — forces L2, not L1 */\n'
            '    const char* _lbuf = g_large_buf + (size_t){warp} * LARGE_BUF_WARP_STRIDE;\n'
            '    unsigned _g0=0, _g1=0, _g2=0, _g3=0;\n'
            '    unsigned _ga0=0, _ga1=0, _ga2=0, _ga3=0;\n'
        ),
        'loop_body': (
            '        {\n'
            '        const unsigned* _lp = (const unsigned*)(_lbuf + (size_t)_i * 4096 + _lane * 16);\n'
            '        asm volatile(\n'
            '            "ld.global.v4.b32 {%0,%1,%2,%3}, [%8];\\n\\t"\n'
            '            "add.u32 %4, %4, %0;\\n\\t"\n'
            '            "ld.global.v4.b32 {%0,%1,%2,%3}, [%8+512];\\n\\t"\n'
            '            "add.u32 %5, %5, %0;\\n\\t"\n'
            '            "ld.global.v4.b32 {%0,%1,%2,%3}, [%8+1024];\\n\\t"\n'
            '            "add.u32 %6, %6, %0;\\n\\t"\n'
            '            "ld.global.v4.b32 {%0,%1,%2,%3}, [%8+1536];\\n\\t"\n'
            '            "add.u32 %7, %7, %0;\\n\\t"\n'
            '            : "=r"(_g0), "=r"(_g1), "=r"(_g2), "=r"(_g3),\n'
            '              "+r"(_ga0), "+r"(_ga1), "+r"(_ga2), "+r"(_ga3)\n'
            '            : "l"(_lp)\n'
            '            : "memory"\n'
            '        );\n'
            '        }\n'
        ),
        'insns_per_iter': 4,
        'sink': '    out[64 + {tid}] = (long long)(_ga0 + _ga1 + _ga2 + _ga3);',
    },
}

# Which pipes to include in same-pipe scaling tests
S_PIPES = ['STS', 'FFMA', 'HFMA2', 'LDS', 'F2FP', 'HADD2', 'NOP', 'LDG', 'STS_ILP1', 'LDG_L2']

# Cross-pipe configs: list of (name, warp→pipe assignments)
# Each assignment maps warp_id → pipe_name. Unassigned warps idle.
X_CONFIGS = [
    ('X_4pipe', {0: 'STS', 1: 'FFMA', 2: 'HFMA2', 3: 'LDS'}),
    ('X_sts_ffma', {0: 'STS', 1: 'FFMA'}),
    ('X_sts_hfma2', {0: 'STS', 1: 'HFMA2'}),
    ('X_sts_lds', {0: 'STS', 1: 'LDS'}),
    ('X_ffma_hfma2', {0: 'FFMA', 1: 'HFMA2'}),
    ('X_sts_f2fp', {0: 'STS', 1: 'F2FP'}),
    # STS+LDS scaling — SMEM port contention at scale
    ('X_2sts_2lds', {0: 'STS', 1: 'STS', 2: 'LDS', 3: 'LDS'}),
    ('X_3sts_3lds', {0: 'STS', 1: 'STS', 2: 'STS', 3: 'LDS', 4: 'LDS', 5: 'LDS'}),
    ('X_4sts_4lds', {0: 'STS', 1: 'STS', 2: 'STS', 3: 'STS', 4: 'LDS', 5: 'LDS', 6: 'LDS', 7: 'LDS'}),
    # STS+LDG cross-pipe (LSU vs TEX — FC2 epilogue does both)
    ('X_sts_ldg', {0: 'STS', 1: 'LDG'}),
    ('X_2sts_2ldg', {0: 'STS', 1: 'STS', 2: 'LDG', 3: 'LDG'}),
]

# Foreground/background: (name, fg_pipe, fg_count, bg_pipe, bg_count)
# FC2-like scenarios: epi warps (STS/BF16-heavy) + kloop warps (compute)
F_CONFIGS = [
    # FC2 model: 4 epi warps (STS), 1-4 kloop warps (FFMA)
    ('F_4sts_1ffma', 'STS', 4, 'FFMA', 1),
    ('F_4sts_2ffma', 'STS', 4, 'FFMA', 2),
    ('F_4sts_3ffma', 'STS', 4, 'FFMA', 3),
    ('F_4sts_4ffma', 'STS', 4, 'FFMA', 4),
    # Reverse: what if kloop warps are foreground?
    ('F_2ffma_1sts', 'FFMA', 2, 'STS', 1),
    ('F_2ffma_2sts', 'FFMA', 2, 'STS', 2),
    ('F_2ffma_4sts', 'FFMA', 2, 'STS', 4),
    # BF16 epilogue variant
    ('F_4hfma_2ffma', 'HFMA2', 4, 'FFMA', 2),
    # Mixed epi (STS+BF16 interleaved) — closer to real FC2
    ('F_4mix_2ffma', 'STS', 4, 'FFMA', 2),  # will customize loop_body
    # FC2-realistic: LDG+BF16+STS (actual epilogue instruction mix per warp)
    ('F_4ldgmix_0ffma', 'STS', 4, 'FFMA', 0),  # will customize — just epi, no kloop
    ('F_4ldgmix_1ffma', 'STS', 4, 'FFMA', 1),  # will customize
    ('F_4ldgmix_2ffma', 'STS', 4, 'FFMA', 2),  # will customize
    # 6-warp FC2-exact heterogeneous: 4 ldgmix epi + 1 LDG (W0) + 1 FFMA (W1)
    ('F_fc2_exact_6w', 'STS', 4, 'FFMA', 1),   # custom routing → gen_fc2_exact_kernel
    # L2 variants: forced L2-miss LDG in epi warps
    ('F_4ldg_l2_0ffma', 'STS', 4, 'FFMA', 0),  # custom routing → gen_ldgmix_epi_kernel(l2=True)
    ('F_4ldg_l2_2ffma', 'STS', 4, 'FFMA', 2),  # custom routing
]

# Producer-consumer configs (CUTLASS W3 pattern):
# (name, n_epi_warps, load_mode, n_kloop_warps, has_load_warp)
# load_mode: 'ldg' = self-load from global (our arch), 'lds' = pre-loaded SMEM (CUTLASS result)
# has_load_warp: True = add 1 dedicated warp that does brief periodic work then idles (CUTLASS W3)
P_CONFIGS = [
    # Core comparison: self-load vs pre-loaded, 4 epi warps
    ('P_4epi_ldg',       4, 'ldg', 0, False),
    ('P_4epi_lds',       4, 'lds', 0, False),
    ('P_4epi_lds_1lw',   4, 'lds', 0, True),
    # With 2 K-loop warps (full FC2 scenario)
    ('P_4epi_ldg_2kl',   4, 'ldg', 2, False),
    ('P_4epi_lds_2kl',   4, 'lds', 2, False),
    ('P_4epi_lds_1lw_2kl', 4, 'lds', 2, True),
    # With 1 K-loop warp
    ('P_4epi_ldg_1kl',   4, 'ldg', 1, False),
    ('P_4epi_lds_1kl',   4, 'lds', 1, False),
    ('P_4epi_lds_1lw_1kl', 4, 'lds', 1, True),
    # 3 epi warps (if we freed one for loading)
    ('P_3epi_ldg_2kl',   3, 'ldg', 2, False),
    ('P_3epi_lds_1lw_2kl', 3, 'lds', 2, True),
]

# BAR.SYNC tests: measure whether synchronized idle gaps help "background" warps
B_CONFIGS = [
    # (name, n_bar_warps, n_compute_warps, bar_interval)
    # bar_interval = how many compute iters between BAR.SYNCs
    ('B_4bar_2cmp_i4', 4, 2, 4),
    ('B_4bar_2cmp_i8', 4, 2, 8),
    ('B_4bar_2cmp_i16', 4, 2, 16),
    ('B_4bar_2cmp_i32', 4, 2, 32),
    # Baseline: no BAR.SYNC, same total work
    ('B_4nobar_2cmp', 4, 2, 0),
    # Different warp ratios
    ('B_2bar_2cmp_i8', 2, 2, 8),
    ('B_6bar_2cmp_i8', 6, 2, 8),
]

# B-test mixed-epi variants: bar warps do LDG+BF16+STS (realistic epilogue mix)
B_MIXED_CONFIGS = [
    # (name, n_bar_warps, n_compute_warps, bar_interval)
    ('B_4mixbar_2cmp_i8', 4, 2, 8),
    ('B_4mixbar_2cmp_nobar', 4, 2, 0),
]

# Nanosleep calibration: measure actual sleep duration for different values
# (name, nanosleep_value)
N_CONFIGS = [
    ('N_sleep_10', 10),
    ('N_sleep_50', 50),
    ('N_sleep_100', 100),
    ('N_sleep_500', 500),
    ('N_sleep_1000', 1000),
    ('N_sleep_5000', 5000),
]

# Asymmetric duration: epi warps exit early, compute warps run full REPS.
# Tests dynamic dispatch recovery after epi warps finish.
# (name, n_epi, n_compute, epi_divisor) — epi runs REPS/divisor iters
A_CONFIGS = [
    ('A_4ldgmix_2ffma_half', 4, 2, 2),      # epi does REPS/2 then exits
    ('A_4ldgmix_2ffma_quarter', 4, 2, 4),    # epi does REPS/4 then exits
]


def max_smem_needed():
    """Compute max SMEM across all tests."""
    max_s = 0
    for pipe in S_PIPES:
        spw = PIPE_WORKLOADS[pipe]['smem_per_warp']
        max_s = max(max_s, spw * MAX_WARPS)
    return max(max_s, MAX_WARPS * 1024)  # at least 1KB/warp for SMEM tests


def gen_header():
    return f"""/* Auto-generated by gen_warp_scaling.py — do not edit */
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_bf16.h>
#include <algorithm>

#define REPS {REPS}
#define WARMUP {WARMUP}
#define MAX_WARPS {MAX_WARPS}

#define CUDA_CHECK(call) do {{ \\
    cudaError_t _e = (call); \\
    if (_e != cudaSuccess) {{ \\
        fprintf(stderr, "CUDA error at %s:%d: %s (%d)\\n", __FILE__, __LINE__, cudaGetErrorString(_e), _e); \\
        exit(1); \\
    }} \\
}} while(0)

__device__ __forceinline__ long long asm_clock64() {{
    long long r;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(r) :: "memory");
    return r;
}}

/* Per-warp timing: out[warp*2] = start, out[warp*2+1] = end */
struct WarpResult {{
    long long start;
    long long end;
}};

/* Large buffer for L2-miss LDG tests — pointer set by host before launch */
__device__ const char* g_large_buf;
#define LARGE_BUF_SIZE         (32u << 20)  /* 32 MB total */
#define LARGE_BUF_WARP_STRIDE  (4u << 20)   /* 4 MB per warp */
"""


def gen_same_pipe_kernel(pipe, n_warps):
    """S-test: all N warps run the same pipe workload."""
    w = PIPE_WORKLOADS[pipe]
    name = f'S_{pipe}_w{n_warps}'
    n_threads = n_warps * 32
    smem_total = w['smem_per_warp'] * n_warps

    lines = []
    lines.append(f'extern "C" __global__ void {name}(WarpResult* results) {{')

    if smem_total > 0:
        # Dynamic SMEM
        lines.append(f'    extern __shared__ __align__(128) char _dyn_smem[];')
        lines.append(f'    const int _warp = threadIdx.x / 32;')
        lines.append(f'    const int _lane = threadIdx.x % 32;')
        lines.append(f'    const int _tid = threadIdx.x;')
        lines.append(f'    uint32_t _smem_base = (uint32_t)(uint64_t)(_dyn_smem + _warp * {w["smem_per_warp"]});')
    else:
        lines.append(f'    const int _warp = threadIdx.x / 32;')
        lines.append(f'    const int _lane = threadIdx.x % 32;')
        lines.append(f'    const int _tid = threadIdx.x;')
        lines.append(f'    uint32_t _smem_base = 0; (void)_smem_base;')

    lines.append(f'    (void)_tid;')
    lines.append('')

    # Setup
    setup = w['setup'].format(lane='_lane', warp='_warp', tid='_tid')
    lines.append(setup)

    # Warmup
    lines.append(f'    __syncthreads();')
    lines.append(f'    for (int _i = 0; _i < WARMUP; _i++) {{')
    lines.append(w['loop_body'])
    lines.append(f'    }}')

    # Synchronized start
    lines.append(f'    __syncthreads();')
    lines.append(f'    long long _t0 = asm_clock64();')
    lines.append('')

    # Timed loop
    lines.append(f'    for (int _i = 0; _i < REPS; _i++) {{')
    lines.append(w['loop_body'])
    lines.append(f'    }}')
    lines.append('')

    lines.append(f'    long long _t1 = asm_clock64();')
    lines.append('')

    # Sink + timing
    lines.append(f'    long long* out = (long long*)results;')
    lines.append(w['sink'].format(lane='_lane', tid='_tid'))
    lines.append(f'    if (_lane == 0) {{')
    lines.append(f'        results[_warp].start = _t0;')
    lines.append(f'        results[_warp].end = _t1;')
    lines.append(f'    }}')
    lines.append(f'}}')

    return name, n_threads, smem_total, w['insns_per_iter'], '\n'.join(lines)


def gen_cross_pipe_kernel(config_name, assignments):
    """X-test: different warps on different pipes."""
    n_warps = max(assignments.keys()) + 1
    n_threads = n_warps * 32

    # Calculate SMEM
    smem_total = 0
    for wid, pipe in assignments.items():
        smem_total = max(smem_total, (wid + 1) * PIPE_WORKLOADS[pipe]['smem_per_warp'])
    smem_total = max(smem_total, n_warps * 1024)  # ensure enough

    lines = []
    lines.append(f'extern "C" __global__ void {config_name}(WarpResult* results) {{')
    lines.append(f'    extern __shared__ __align__(128) char _dyn_smem[];')
    lines.append(f'    const int _warp = threadIdx.x / 32;')
    lines.append(f'    const int _lane = threadIdx.x % 32;')
    lines.append(f'    const int _tid = threadIdx.x;')
    lines.append(f'    uint32_t _smem_base = (uint32_t)(uint64_t)(_dyn_smem + _warp * 1024);')
    lines.append(f'    long long* out = (long long*)results;')
    lines.append(f'    (void)_tid; (void)_smem_base; (void)out;')
    lines.append('')

    # Per-warp setup + loop via switch
    lines.append(f'    __syncthreads();')
    lines.append(f'    long long _t0, _t1;')
    lines.append('')
    lines.append(f'    switch (_warp) {{')

    for wid in range(n_warps):
        pipe = assignments.get(wid, 'NOP')
        w = PIPE_WORKLOADS[pipe]
        lines.append(f'    case {wid}: {{ /* {pipe} */')
        setup = w['setup'].format(lane='_lane', warp='_warp', tid='_tid')
        lines.append(setup)
        lines.append(f'        for (int _i = 0; _i < WARMUP; _i++) {{')
        lines.append(w['loop_body'])
        lines.append(f'        }}')
        lines.append(f'        __syncthreads();')
        lines.append(f'        _t0 = asm_clock64();')
        lines.append(f'        for (int _i = 0; _i < REPS; _i++) {{')
        lines.append(w['loop_body'])
        lines.append(f'        }}')
        lines.append(f'        _t1 = asm_clock64();')
        lines.append(w['sink'].format(lane='_lane', tid='_tid'))
        lines.append(f'        if (_lane == 0) {{ results[_warp].start = _t0; results[_warp].end = _t1; }}')
        lines.append(f'        break;')
        lines.append(f'    }}')

    lines.append(f'    default: __syncthreads(); _t0 = _t1 = 0;')
    lines.append(f'        if (_lane == 0) {{ results[_warp].start = _t0; results[_warp].end = _t1; }}')
    lines.append(f'        break;')
    lines.append(f'    }}')
    lines.append(f'}}')

    insns = max(PIPE_WORKLOADS[assignments.get(wid, 'NOP')]['insns_per_iter']
                for wid in range(n_warps))
    return config_name, n_threads, smem_total, insns, '\n'.join(lines)


def gen_fg_bg_kernel(name, fg_pipe, fg_count, bg_pipe, bg_count):
    """F-test: fg_count warps on fg_pipe + bg_count warps on bg_pipe."""
    assignments = {}
    for i in range(fg_count):
        assignments[i] = fg_pipe
    for i in range(bg_count):
        assignments[fg_count + i] = bg_pipe
    return gen_cross_pipe_kernel(name, assignments)


def gen_barsync_kernel(name, n_bar_warps, n_compute_warps, bar_interval):
    """B-test: n_bar_warps do STS+BAR.SYNC, n_compute_warps do FFMA continuously.

    When bar_interval > 0: bar warps do `bar_interval` STS iterations, then BAR.SYNC
    among themselves (only the bar warps participate). Compute warps run free.

    When bar_interval == 0: bar warps do plain STS, no BAR.SYNC (baseline).

    The BAR.SYNC only synchronizes bar warps with each other (thread count =
    n_bar_warps * 32). Compute warps never participate in the barrier.
    """
    n_warps = n_bar_warps + n_compute_warps
    n_threads = n_warps * 32
    bar_thread_count = n_bar_warps * 32
    smem_total = n_warps * 1024  # enough for STS

    lines = []
    lines.append(f'extern "C" __global__ void {name}(WarpResult* results) {{')
    lines.append(f'    extern __shared__ __align__(128) char _dyn_smem[];')
    lines.append(f'    const int _warp = threadIdx.x / 32;')
    lines.append(f'    const int _lane = threadIdx.x % 32;')
    lines.append(f'    const int _tid = threadIdx.x;')
    lines.append(f'    uint32_t _smem_base = (uint32_t)(uint64_t)(_dyn_smem + _warp * 1024);')
    lines.append(f'    long long* out = (long long*)results;')
    lines.append(f'    (void)_tid; (void)_smem_base; (void)out;')
    lines.append('')
    lines.append(f'    __syncthreads();')
    lines.append(f'    long long _t0, _t1;')
    lines.append('')

    # Bar warps: STS with periodic BAR.SYNC
    lines.append(f'    if (_warp < {n_bar_warps}) {{')
    lines.append(f'        /* BAR warp: STS + periodic BAR.SYNC (interval={bar_interval}) */')
    lines.append(f'        uint32_t _base = _smem_base + _lane * 16;')
    lines.append(f'        unsigned _v0=0xdead, _v1=0xcafe, _v2=0x1234, _v3=0x9abc;')
    lines.append(f'        for (int _i = 0; _i < WARMUP; _i++) {{')
    lines.append(f'            uint32_t _a = _base + ((_i & 3) << 6);')
    lines.append(f'            asm volatile("st.shared.v4.b32 [%0], {{%1,%2,%3,%4}};"')
    lines.append(f'                :: "r"(_a), "r"(_v0), "r"(_v1), "r"(_v2), "r"(_v3) : "memory");')
    lines.append(f'        }}')
    lines.append(f'        __syncthreads();')
    lines.append(f'        _t0 = asm_clock64();')

    if bar_interval > 0:
        # Total iterations = REPS, with BAR.SYNC every bar_interval iters
        lines.append(f'        for (int _outer = 0; _outer < REPS / {bar_interval}; _outer++) {{')
        lines.append(f'            for (int _inner = 0; _inner < {bar_interval}; _inner++) {{')
        lines.append(f'                uint32_t _a = _base + (((_outer * {bar_interval} + _inner) & 3) << 6);')
        lines.append(f'                asm volatile(')
        lines.append(f'                    "st.shared.v4.b32 [%0], {{%1,%2,%3,%4}};\\n\\t"')
        lines.append(f'                    "st.shared.v4.b32 [%0+16], {{%1,%2,%3,%4}};\\n\\t"')
        lines.append(f'                    "st.shared.v4.b32 [%0+32], {{%1,%2,%3,%4}};\\n\\t"')
        lines.append(f'                    "st.shared.v4.b32 [%0+48], {{%1,%2,%3,%4}};\\n\\t"')
        lines.append(f'                    :: "r"(_a), "r"(_v0), "r"(_v1), "r"(_v2), "r"(_v3)')
        lines.append(f'                    : "memory"')
        lines.append(f'                );')
        lines.append(f'            }}')
        # BAR.SYNC among bar warps only
        lines.append(f'            asm volatile("bar.sync 1, {bar_thread_count};" ::: "memory");')
        lines.append(f'        }}')
    else:
        # No BAR.SYNC baseline
        lines.append(f'        for (int _i = 0; _i < REPS; _i++) {{')
        lines.append(f'            uint32_t _a = _base + ((_i & 3) << 6);')
        lines.append(f'            asm volatile(')
        lines.append(f'                "st.shared.v4.b32 [%0], {{%1,%2,%3,%4}};\\n\\t"')
        lines.append(f'                "st.shared.v4.b32 [%0+16], {{%1,%2,%3,%4}};\\n\\t"')
        lines.append(f'                "st.shared.v4.b32 [%0+32], {{%1,%2,%3,%4}};\\n\\t"')
        lines.append(f'                "st.shared.v4.b32 [%0+48], {{%1,%2,%3,%4}};\\n\\t"')
        lines.append(f'                :: "r"(_a), "r"(_v0), "r"(_v1), "r"(_v2), "r"(_v3)')
        lines.append(f'                : "memory"')
        lines.append(f'            );')
        lines.append(f'        }}')

    lines.append(f'        _t1 = asm_clock64();')
    lines.append(f'        ((volatile unsigned*)(_dyn_smem + _warp * 1024))[_lane] = _v0;')

    # Compute warps: continuous FFMA
    lines.append(f'    }} else {{')
    lines.append(f'        /* Compute warp: continuous FFMA */')
    lines.append(f'        float _a0=1.001f+_lane, _a1=1.002f+_lane, _a2=1.003f+_lane, _a3=1.004f+_lane;')
    lines.append(f'        float _s0=1.0001f, _s1=0.9999f;')
    lines.append(f'        for (int _i = 0; _i < WARMUP; _i++)')
    lines.append(f'            asm volatile("fma.rn.f32 %0, %0, %2, %3;" : "+f"(_a0) : "f"(_a0), "f"(_s0), "f"(_s1));')
    lines.append(f'        __syncthreads();')
    lines.append(f'        _t0 = asm_clock64();')
    lines.append(f'        for (int _i = 0; _i < REPS; _i++) {{')
    lines.append(f'            asm volatile(')
    lines.append(f'                "fma.rn.f32 %0, %0, %4, %5;\\n\\t"')
    lines.append(f'                "fma.rn.f32 %1, %1, %4, %5;\\n\\t"')
    lines.append(f'                "fma.rn.f32 %2, %2, %4, %5;\\n\\t"')
    lines.append(f'                "fma.rn.f32 %3, %3, %4, %5;\\n\\t"')
    lines.append(f'                : "+f"(_a0), "+f"(_a1), "+f"(_a2), "+f"(_a3)')
    lines.append(f'                : "f"(_s0), "f"(_s1)')
    lines.append(f'            );')
    lines.append(f'        }}')
    lines.append(f'        _t1 = asm_clock64();')
    lines.append(f'        ((volatile float*)(out + 64))[_tid] = _a0 + _a1 + _a2 + _a3;')
    lines.append(f'    }}')
    lines.append('')

    lines.append(f'    if (_lane == 0) {{')
    lines.append(f'        results[_warp].start = _t0;')
    lines.append(f'        results[_warp].end = _t1;')
    lines.append(f'    }}')
    lines.append(f'}}')

    return name, n_threads, smem_total, 4, '\n'.join(lines)


def gen_mixed_epi_kernel(name, n_epi_warps, n_compute_warps):
    """F_mix test: epi warps do alternating STS+HFMA2 (closer to real FC2 epilogue).
    Compute warps do continuous FFMA."""
    n_warps = n_epi_warps + n_compute_warps
    n_threads = n_warps * 32
    smem_total = n_warps * 1024

    lines = []
    lines.append(f'extern "C" __global__ void {name}(WarpResult* results) {{')
    lines.append(f'    extern __shared__ __align__(128) char _dyn_smem[];')
    lines.append(f'    const int _warp = threadIdx.x / 32;')
    lines.append(f'    const int _lane = threadIdx.x % 32;')
    lines.append(f'    const int _tid = threadIdx.x;')
    lines.append(f'    uint32_t _smem_base = (uint32_t)(uint64_t)(_dyn_smem + _warp * 1024);')
    lines.append(f'    long long* out = (long long*)results;')
    lines.append(f'    (void)_tid; (void)_smem_base; (void)out;')
    lines.append('')
    lines.append(f'    __syncthreads();')
    lines.append(f'    long long _t0, _t1;')
    lines.append('')

    # Epi warps: interleaved BF16 compute + STS (matches FC2 epilogue pattern)
    lines.append(f'    if (_warp < {n_epi_warps}) {{')
    lines.append(f'        uint32_t _base = _smem_base + _lane * 16;')
    lines.append(f'        unsigned _v0=0xdead, _v1=0xcafe, _v2=0x1234, _v3=0x9abc;')
    lines.append(f'        unsigned _h0=0x3f80+_lane, _h1=0x3f81+_lane;')
    lines.append(f'        unsigned _hs=0x3f803f80;')
    lines.append(f'        float _f0=1.0f+_lane, _f1=2.0f+_lane;')
    lines.append(f'        unsigned _cv=0;')
    lines.append(f'        for (int _i = 0; _i < WARMUP; _i++) {{')
    lines.append(f'            asm volatile("st.shared.v4.b32 [%0], {{%1,%2,%3,%4}};"')
    lines.append(f'                :: "r"(_base), "r"(_v0), "r"(_v1), "r"(_v2), "r"(_v3) : "memory");')
    lines.append(f'        }}')
    lines.append(f'        __syncthreads();')
    lines.append(f'        _t0 = asm_clock64();')
    lines.append(f'        for (int _i = 0; _i < REPS; _i++) {{')
    lines.append(f'            uint32_t _a = _base + ((_i & 3) << 6);')
    lines.append(f'            /* F2FP + HFMA2 + HADD2 (BF16 compute) then STS */')
    lines.append(f'            asm volatile(')
    lines.append(f'                "cvt.rn.bf16x2.f32 %0, %5, %4;\\n\\t"')
    lines.append(f'                "fma.rn.bf16x2 %1, %1, %3, %1;\\n\\t"')
    lines.append(f'                "add.rn.bf16x2 %2, %2, %3;\\n\\t"')
    lines.append(f'                : "+r"(_cv), "+r"(_h0), "+r"(_h1)')
    lines.append(f'                : "r"(_hs), "f"(_f0), "f"(_f1)')
    lines.append(f'            );')
    lines.append(f'            asm volatile(')
    lines.append(f'                "st.shared.v4.b32 [%0], {{%1,%2,%3,%4}};\\n\\t"')
    lines.append(f'                :: "r"(_a), "r"(_v0), "r"(_v1), "r"(_v2), "r"(_v3)')
    lines.append(f'                : "memory"')
    lines.append(f'            );')
    lines.append(f'        }}')
    lines.append(f'        _t1 = asm_clock64();')
    lines.append(f'        ((volatile unsigned*)(_dyn_smem + _warp * 1024))[_lane] = _v0 + _cv + _h0;')

    # Compute warps: continuous FFMA
    lines.append(f'    }} else {{')
    lines.append(f'        float _a0=1.001f+_lane, _a1=1.002f+_lane, _a2=1.003f+_lane, _a3=1.004f+_lane;')
    lines.append(f'        float _s0=1.0001f, _s1=0.9999f;')
    lines.append(f'        for (int _i = 0; _i < WARMUP; _i++)')
    lines.append(f'            asm volatile("fma.rn.f32 %0, %0, %2, %3;" : "+f"(_a0) : "f"(_a0), "f"(_s0), "f"(_s1));')
    lines.append(f'        __syncthreads();')
    lines.append(f'        _t0 = asm_clock64();')
    lines.append(f'        for (int _i = 0; _i < REPS; _i++) {{')
    lines.append(f'            asm volatile(')
    lines.append(f'                "fma.rn.f32 %0, %0, %4, %5;\\n\\t"')
    lines.append(f'                "fma.rn.f32 %1, %1, %4, %5;\\n\\t"')
    lines.append(f'                "fma.rn.f32 %2, %2, %4, %5;\\n\\t"')
    lines.append(f'                "fma.rn.f32 %3, %3, %4, %5;\\n\\t"')
    lines.append(f'                : "+f"(_a0), "+f"(_a1), "+f"(_a2), "+f"(_a3)')
    lines.append(f'                : "f"(_s0), "f"(_s1)')
    lines.append(f'            );')
    lines.append(f'        }}')
    lines.append(f'        _t1 = asm_clock64();')
    lines.append(f'        ((volatile float*)(out + 64))[_tid] = _a0 + _a1 + _a2 + _a3;')
    lines.append(f'    }}')
    lines.append('')

    lines.append(f'    if (_lane == 0) {{')
    lines.append(f'        results[_warp].start = _t0;')
    lines.append(f'        results[_warp].end = _t1;')
    lines.append(f'    }}')
    lines.append(f'}}')

    return name, n_threads, smem_total, 4, '\n'.join(lines)


def gen_prodcons_kernel(name, n_epi, load_mode, n_kloop, has_load_warp):
    """P-test: producer-consumer pattern modeling CUTLASS W3 architecture.

    Epilogue warps do FC2-like work: load data + BF16 compute (F2FP+HFMA2+HADD2) + STS.
    load_mode='ldg': warps self-load from global memory (16 LDG — our architecture)
    load_mode='lds': warps load from SMEM (4 LDS — CUTLASS, data pre-loaded by W3)

    If has_load_warp: add 1 warp that does brief periodic work every `interval` iters
    (simulating TMA issue + mbar arrive — mostly idle, like CUTLASS W3).

    If n_kloop > 0: add compute warps doing continuous FFMA (K-loop simulation).
    """
    n_warps = n_epi + n_kloop + (1 if has_load_warp else 0)
    n_threads = n_warps * 32
    # Each epi warp needs 1KB for STS targets; load warp needs 2KB for LDS source data
    smem_total = n_warps * 2048

    # Load warp does brief work every this many outer iters
    lw_interval = 8

    lines = []
    lines.append(f'extern "C" __global__ void {name}(WarpResult* results) {{')
    lines.append(f'    extern __shared__ __align__(128) char _dyn_smem[];')
    lines.append(f'    const int _warp = threadIdx.x / 32;')
    lines.append(f'    const int _lane = threadIdx.x % 32;')
    lines.append(f'    const int _tid = threadIdx.x;')
    lines.append(f'    uint32_t _smem_base = (uint32_t)(uint64_t)(_dyn_smem + _warp * 2048);')
    lines.append(f'    long long* out = (long long*)results;')
    lines.append(f'    (void)_tid; (void)_smem_base; (void)out;')
    lines.append('')

    # Pre-fill SMEM (for LDS mode and load warp source)
    lines.append(f'    /* Pre-fill SMEM so LDS reads valid data */')
    lines.append(f'    ((volatile int*)(_dyn_smem))[_tid] = _tid;')
    lines.append(f'    __syncthreads();')
    lines.append(f'    long long _t0, _t1;')
    lines.append('')

    # --- Epilogue warps ---
    lines.append(f'    if (_warp < {n_epi}) {{')
    lines.append(f'        /* Epilogue warp: {"LDG (self-load)" if load_mode == "ldg" else "LDS (pre-loaded)"} + BF16 compute + STS */')
    lines.append(f'        uint32_t _sts_base = _smem_base + _lane * 16;')
    lines.append(f'        unsigned _v0=0xdead, _v1=0xcafe, _v2=0x1234, _v3=0x9abc;')
    lines.append(f'        unsigned _h0=0x3f80+_lane, _h1=0x3f81+_lane;')
    lines.append(f'        unsigned _hs=0x3f803f80;')
    lines.append(f'        float _f0=1.0f+_lane, _f1=2.0f+_lane;')
    lines.append(f'        unsigned _cv=0;')

    if load_mode == 'ldg':
        # LDG setup — read from global (the results buffer, doesn't matter what)
        lines.append(f'        const unsigned* _gptr = (const unsigned*)results + _lane * 4;')
    else:
        # LDS setup — read from SMEM (pre-filled above)
        lines.append(f'        uint32_t _lds_base = (uint32_t)(uint64_t)(_dyn_smem) + _lane * 16;')

    lines.append(f'        for (int _w = 0; _w < WARMUP; _w++) {{')
    lines.append(f'            asm volatile("st.shared.v4.b32 [%0], {{%1,%2,%3,%4}};"')
    lines.append(f'                :: "r"(_sts_base), "r"(_v0), "r"(_v1), "r"(_v2), "r"(_v3) : "memory");')
    lines.append(f'        }}')
    lines.append(f'        __syncthreads();')
    lines.append(f'        unsigned _ld0=0, _ld1=0, _ld2=0, _ld3=0, _lda0=0;')
    lines.append(f'        _t0 = asm_clock64();')
    lines.append(f'        for (int _i = 0; _i < REPS; _i++) {{')

    if load_mode == 'ldg':
        # 4× LDG.128 from global (simulating 16 LDG in real FC2, scaled down to match iters)
        lines.append(f'            asm volatile(')
        lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4];\\n\\t"')
        lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4+16];\\n\\t"')
        lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4+32];\\n\\t"')
        lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4+48];\\n\\t"')
        lines.append(f'                : "+r"(_ld0), "+r"(_ld1), "+r"(_ld2), "+r"(_ld3)')
        lines.append(f'                : "l"(_gptr + ((_i & 0x3) << 2))')
        lines.append(f'                : "memory"')
        lines.append(f'            );')
        lines.append(f'            _lda0 += _ld0;')
    else:
        # 4× LDS.128 from SMEM (data pre-loaded, like CUTLASS after W3)
        lines.append(f'            uint32_t _la = _lds_base + ((_i & 3) << 6);')
        lines.append(f'            asm volatile(')
        lines.append(f'                "ld.shared.v4.b32 {{%0,%1,%2,%3}}, [%4];\\n\\t"')
        lines.append(f'                "ld.shared.v4.b32 {{%0,%1,%2,%3}}, [%4+16];\\n\\t"')
        lines.append(f'                "ld.shared.v4.b32 {{%0,%1,%2,%3}}, [%4+32];\\n\\t"')
        lines.append(f'                "ld.shared.v4.b32 {{%0,%1,%2,%3}}, [%4+48];\\n\\t"')
        lines.append(f'                : "+r"(_ld0), "+r"(_ld1), "+r"(_ld2), "+r"(_ld3)')
        lines.append(f'                : "r"(_la)')
        lines.append(f'                : "memory"')
        lines.append(f'            );')
        lines.append(f'            _lda0 += _ld0;')

    # BF16 compute: F2FP + HFMA2 + HADD2 (same as real FC2 epilogue)
    lines.append(f'            asm volatile(')
    lines.append(f'                "cvt.rn.bf16x2.f32 %0, %5, %4;\\n\\t"')
    lines.append(f'                "fma.rn.bf16x2 %1, %1, %3, %1;\\n\\t"')
    lines.append(f'                "add.rn.bf16x2 %2, %2, %3;\\n\\t"')
    lines.append(f'                : "+r"(_cv), "+r"(_h0), "+r"(_h1)')
    lines.append(f'                : "r"(_hs), "f"(_f0), "f"(_f1)')
    lines.append(f'            );')

    # STS.128 (the bottleneck pipe)
    lines.append(f'            uint32_t _sa = _sts_base + ((_i & 3) << 6);')
    lines.append(f'            asm volatile(')
    lines.append(f'                "st.shared.v4.b32 [%0], {{%1,%2,%3,%4}};\\n\\t"')
    lines.append(f'                :: "r"(_sa), "r"(_ld0), "r"(_ld1), "r"(_ld2), "r"(_ld3)')
    lines.append(f'                : "memory"')
    lines.append(f'            );')

    lines.append(f'        }}')
    lines.append(f'        _t1 = asm_clock64();')
    lines.append(f'        ((volatile unsigned*)(_dyn_smem + _warp * 1024))[_lane] = _v0 + _cv + _h0 + _lda0;')

    # --- Load warp (CUTLASS W3 simulation) ---
    if has_load_warp:
        lw_id = n_epi  # load warp is right after epi warps
        lines.append(f'    }} else if (_warp == {lw_id}) {{')
        lines.append(f'        /* Load warp (W3 sim): brief periodic work, mostly idle */')
        lines.append(f'        /* Real W3: TMA issue (~1 cyc) + mbar arrive (~2 cyc) per sub-iter */')
        lines.append(f'        /* Then idle until next sub-iter needs loading */')
        lines.append(f'        uint32_t _lw_base = _smem_base;')
        lines.append(f'        unsigned _lw_v = _lane;')
        lines.append(f'        __syncthreads();')
        lines.append(f'        _t0 = asm_clock64();')
        lines.append(f'        for (int _i = 0; _i < REPS / {lw_interval}; _i++) {{')
        # Brief burst: 2 STS (simulating TMA descriptor setup + issue) + bar arrive
        # Then idle for lw_interval iterations worth of time
        lines.append(f'            /* Brief work: simulate TMA issue + mbar arrive */')
        lines.append(f'            asm volatile(')
        lines.append(f'                "st.shared.b32 [%0], %1;\\n\\t"')
        lines.append(f'                "st.shared.b32 [%0+4], %1;\\n\\t"')
        lines.append(f'                :: "r"(_lw_base), "r"(_lw_v) : "memory"')
        lines.append(f'            );')
        # Nanosleep to simulate being idle — warp yields to scheduler
        # nanosleep puts the warp to sleep, removing it from eligible pool
        # This is the key: CUTLASS W3 is idle most of the time
        lines.append(f'            /* Idle period: warp sleeps, freeing scheduler for others */')
        lines.append(f'            asm volatile("nanosleep.u32 100;" ::: "memory");')
        lines.append(f'        }}')
        lines.append(f'        _t1 = asm_clock64();')
        lines.append(f'        ((volatile unsigned*)(_dyn_smem + _warp * 1024))[_lane] = _lw_v;')

    # --- K-loop warps ---
    if n_kloop > 0:
        kl_start = n_epi + (1 if has_load_warp else 0)
        lines.append(f'    }} else if (_warp >= {kl_start}) {{')
        lines.append(f'        /* K-loop warp: continuous FFMA compute */')
        lines.append(f'        float _a0=1.001f+_lane, _a1=1.002f+_lane;')
        lines.append(f'        float _a2=1.003f+_lane, _a3=1.004f+_lane;')
        lines.append(f'        float _s0=1.0001f, _s1=0.9999f;')
        lines.append(f'        for (int _w = 0; _w < WARMUP; _w++)')
        lines.append(f'            asm volatile("fma.rn.f32 %0, %0, %2, %3;"')
        lines.append(f'                : "+f"(_a0) : "f"(_a0), "f"(_s0), "f"(_s1));')
        lines.append(f'        __syncthreads();')
        lines.append(f'        _t0 = asm_clock64();')
        lines.append(f'        for (int _i = 0; _i < REPS; _i++) {{')
        lines.append(f'            asm volatile(')
        lines.append(f'                "fma.rn.f32 %0, %0, %4, %5;\\n\\t"')
        lines.append(f'                "fma.rn.f32 %1, %1, %4, %5;\\n\\t"')
        lines.append(f'                "fma.rn.f32 %2, %2, %4, %5;\\n\\t"')
        lines.append(f'                "fma.rn.f32 %3, %3, %4, %5;\\n\\t"')
        lines.append(f'                : "+f"(_a0), "+f"(_a1), "+f"(_a2), "+f"(_a3)')
        lines.append(f'                : "f"(_s0), "f"(_s1)')
        lines.append(f'            );')
        lines.append(f'        }}')
        lines.append(f'        _t1 = asm_clock64();')
        lines.append(f'        ((volatile float*)(out + 64))[_tid] = _a0 + _a1 + _a2 + _a3;')

    # Close warp selection
    lines.append(f'    }} else {{')
    lines.append(f'        __syncthreads();')
    lines.append(f'        _t0 = _t1 = 0;')
    lines.append(f'    }}')
    lines.append('')

    lines.append(f'    if (_lane == 0) {{')
    lines.append(f'        results[_warp].start = _t0;')
    lines.append(f'        results[_warp].end = _t1;')
    lines.append(f'    }}')
    lines.append(f'}}')

    return name, n_threads, smem_total, 8, '\n'.join(lines)


def gen_ldgmix_epi_kernel(name, n_epi_warps, n_compute_warps, l2=False):
    """FC2-realistic F-test: epi warps do LDG→F2FP→HFMA2→HADD2→STS (actual FC2 epilogue mix).
    Compute warps do continuous FFMA.
    l2=True: stride through large buffer to force L2 miss (not L1-cached)."""
    n_warps = n_epi_warps + n_compute_warps
    n_threads = n_warps * 32
    smem_total = n_warps * 1024

    lines = []
    lines.append(f'extern "C" __global__ void {name}(WarpResult* results) {{')
    lines.append(f'    extern __shared__ __align__(128) char _dyn_smem[];')
    lines.append(f'    const int _warp = threadIdx.x / 32;')
    lines.append(f'    const int _lane = threadIdx.x % 32;')
    lines.append(f'    const int _tid = threadIdx.x;')
    lines.append(f'    uint32_t _smem_base = (uint32_t)(uint64_t)(_dyn_smem + _warp * 1024);')
    lines.append(f'    long long* out = (long long*)results;')
    lines.append(f'    (void)_tid; (void)_smem_base; (void)out;')
    lines.append('')
    lines.append(f'    __syncthreads();')
    lines.append(f'    long long _t0, _t1;')
    lines.append('')

    # Epi warps: LDG → F2FP → HFMA2 → HADD2 → STS (actual FC2 epilogue)
    lines.append(f'    if (_warp < {n_epi_warps}) {{')
    lines.append(f'        uint32_t _base = _smem_base + _lane * 16;')
    if l2:
        lines.append(f'        const char* _lbuf = g_large_buf + (size_t)_warp * LARGE_BUF_WARP_STRIDE;')
    else:
        lines.append(f'        const unsigned* _gptr = (const unsigned*)results + _lane * 4;')
    lines.append(f'        unsigned _h0=0x3f80+_lane, _h1=0x3f81+_lane;')
    lines.append(f'        unsigned _hs=0x3f803f80;')
    lines.append(f'        float _f0=1.0f+_lane, _f1=2.0f+_lane;')
    lines.append(f'        unsigned _cv=0;')
    lines.append(f'        for (int _i = 0; _i < WARMUP; _i++) {{')
    lines.append(f'            asm volatile("st.shared.v4.b32 [%0], {{%1,%2,%3,%4}};"')
    lines.append(f'                :: "r"(_base), "r"(0u), "r"(0u), "r"(0u), "r"(0u) : "memory");')
    lines.append(f'        }}')
    lines.append(f'        __syncthreads();')
    lines.append(f'        unsigned _ld0=0, _ld1=0, _ld2=0, _ld3=0, _lda0=0;')
    lines.append(f'        _t0 = asm_clock64();')
    lines.append(f'        for (int _i = 0; _i < REPS; _i++) {{')
    # LDG.128 — load from global (or large buffer for L2)
    if l2:
        lines.append(f'            {{')
        lines.append(f'            const unsigned* _lp = (const unsigned*)(_lbuf + (size_t)_i * 4096 + _lane * 16);')
        lines.append(f'            asm volatile(')
        lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4];\\n\\t"')
        lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4+512];\\n\\t"')
        lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4+1024];\\n\\t"')
        lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4+1536];\\n\\t"')
        lines.append(f'                : "+r"(_ld0), "+r"(_ld1), "+r"(_ld2), "+r"(_ld3)')
        lines.append(f'                : "l"(_lp)')
        lines.append(f'                : "memory"')
        lines.append(f'            );')
        lines.append(f'            }}')
        lines.append(f'            _lda0 += _ld0;')
    else:
        lines.append(f'            asm volatile(')
        lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4];\\n\\t"')
        lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4+16];\\n\\t"')
        lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4+32];\\n\\t"')
        lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4+48];\\n\\t"')
        lines.append(f'                : "+r"(_ld0), "+r"(_ld1), "+r"(_ld2), "+r"(_ld3)')
        lines.append(f'                : "l"(_gptr + ((_i & 0x3) << 2))')
        lines.append(f'                : "memory"')
        lines.append(f'            );')
        lines.append(f'            _lda0 += _ld0;')
    # F2FP + HFMA2 + HADD2 (BF16 compute — FC2 epilogue math)
    lines.append(f'            asm volatile(')
    lines.append(f'                "cvt.rn.bf16x2.f32 %0, %5, %4;\\n\\t"')
    lines.append(f'                "fma.rn.bf16x2 %1, %1, %3, %1;\\n\\t"')
    lines.append(f'                "add.rn.bf16x2 %2, %2, %3;\\n\\t"')
    lines.append(f'                : "+r"(_cv), "+r"(_h0), "+r"(_h1)')
    lines.append(f'                : "r"(_hs), "f"(_f0), "f"(_f1)')
    lines.append(f'            );')
    # STS.128 — store to SMEM staging
    lines.append(f'            uint32_t _sa = _base + ((_i & 3) << 6);')
    lines.append(f'            asm volatile(')
    lines.append(f'                "st.shared.v4.b32 [%0], {{%1,%2,%3,%4}};\\n\\t"')
    lines.append(f'                :: "r"(_sa), "r"(_ld0), "r"(_ld1), "r"(_ld2), "r"(_ld3)')
    lines.append(f'                : "memory"')
    lines.append(f'            );')
    lines.append(f'        }}')
    lines.append(f'        _t1 = asm_clock64();')
    lines.append(f'        ((volatile unsigned*)(_dyn_smem + _warp * 1024))[_lane] = _cv + _h0 + _lda0;')

    # Compute warps
    if n_compute_warps > 0:
        lines.append(f'    }} else {{')
        lines.append(f'        float _a0=1.001f+_lane, _a1=1.002f+_lane, _a2=1.003f+_lane, _a3=1.004f+_lane;')
        lines.append(f'        float _s0=1.0001f, _s1=0.9999f;')
        lines.append(f'        for (int _i = 0; _i < WARMUP; _i++)')
        lines.append(f'            asm volatile("fma.rn.f32 %0, %0, %2, %3;" : "+f"(_a0) : "f"(_a0), "f"(_s0), "f"(_s1));')
        lines.append(f'        __syncthreads();')
        lines.append(f'        _t0 = asm_clock64();')
        lines.append(f'        for (int _i = 0; _i < REPS; _i++) {{')
        lines.append(f'            asm volatile(')
        lines.append(f'                "fma.rn.f32 %0, %0, %4, %5;\\n\\t"')
        lines.append(f'                "fma.rn.f32 %1, %1, %4, %5;\\n\\t"')
        lines.append(f'                "fma.rn.f32 %2, %2, %4, %5;\\n\\t"')
        lines.append(f'                "fma.rn.f32 %3, %3, %4, %5;\\n\\t"')
        lines.append(f'                : "+f"(_a0), "+f"(_a1), "+f"(_a2), "+f"(_a3)')
        lines.append(f'                : "f"(_s0), "f"(_s1)')
        lines.append(f'            );')
        lines.append(f'        }}')
        lines.append(f'        _t1 = asm_clock64();')
        lines.append(f'        ((volatile float*)(out + 64))[_tid] = _a0 + _a1 + _a2 + _a3;')
    lines.append(f'    }}')
    lines.append('')

    lines.append(f'    if (_lane == 0) {{')
    lines.append(f'        results[_warp].start = _t0;')
    lines.append(f'        results[_warp].end = _t1;')
    lines.append(f'    }}')
    lines.append(f'}}')

    return name, n_threads, smem_total, 8, '\n'.join(lines)


def gen_nanosleep_kernel(name, sleep_value):
    """N-test: measure actual nanosleep duration. 1 warp, REPS calls."""
    lines = []
    lines.append(f'extern "C" __global__ void {name}(WarpResult* results) {{')
    lines.append(f'    const int _lane = threadIdx.x % 32;')
    lines.append(f'    long long* out = (long long*)results;')
    lines.append(f'    (void)out;')
    lines.append('')
    lines.append(f'    /* Warmup */')
    lines.append(f'    for (int _i = 0; _i < WARMUP; _i++)')
    lines.append(f'        asm volatile("nanosleep.u32 {sleep_value};" ::: "memory");')
    lines.append('')
    lines.append(f'    __syncthreads();')
    lines.append(f'    long long _t0 = asm_clock64();')
    lines.append(f'    for (int _i = 0; _i < REPS; _i++)')
    lines.append(f'        asm volatile("nanosleep.u32 {sleep_value};" ::: "memory");')
    lines.append(f'    long long _t1 = asm_clock64();')
    lines.append('')
    lines.append(f'    out[64 + _lane] = _t1;  /* sink */')
    lines.append(f'    if (_lane == 0) {{')
    lines.append(f'        results[0].start = _t0;')
    lines.append(f'        results[0].end = _t1;')
    lines.append(f'    }}')
    lines.append(f'}}')

    return name, 32, 0, 1, '\n'.join(lines)


def gen_fc2_exact_kernel(name):
    """F-test: exact FC2 6-warp config — 4 ldgmix epi + 1 LDG (W0) + 1 FFMA (W1).
    Heterogeneous 3-role warp assignment matching real FC2 architecture."""
    n_warps = 6
    n_threads = 192
    smem_total = n_warps * 1024

    lines = []
    lines.append(f'extern "C" __global__ void {name}(WarpResult* results) {{')
    lines.append(f'    extern __shared__ __align__(128) char _dyn_smem[];')
    lines.append(f'    const int _warp = threadIdx.x / 32;')
    lines.append(f'    const int _lane = threadIdx.x % 32;')
    lines.append(f'    const int _tid = threadIdx.x;')
    lines.append(f'    uint32_t _smem_base = (uint32_t)(uint64_t)(_dyn_smem + _warp * 1024);')
    lines.append(f'    long long* out = (long long*)results;')
    lines.append(f'    (void)_tid; (void)_smem_base; (void)out;')
    lines.append('')
    lines.append(f'    __syncthreads();')
    lines.append(f'    long long _t0, _t1;')
    lines.append('')

    # Warps 0-3: ldgmix epi (LDG→F2FP→HFMA2→HADD2→STS)
    lines.append(f'    if (_warp < 4) {{')
    lines.append(f'        /* Epi warp: LDG→F2FP→HFMA2→HADD2→STS (FC2 epilogue) */')
    lines.append(f'        uint32_t _base = _smem_base + _lane * 16;')
    lines.append(f'        const unsigned* _gptr = (const unsigned*)results + _lane * 4;')
    lines.append(f'        unsigned _h0=0x3f80+_lane, _h1=0x3f81+_lane;')
    lines.append(f'        unsigned _hs=0x3f803f80;')
    lines.append(f'        float _f0=1.0f+_lane, _f1=2.0f+_lane;')
    lines.append(f'        unsigned _cv=0;')
    lines.append(f'        for (int _i = 0; _i < WARMUP; _i++)')
    lines.append(f'            asm volatile("st.shared.v4.b32 [%0], {{%1,%2,%3,%4}};"')
    lines.append(f'                :: "r"(_base), "r"(0u), "r"(0u), "r"(0u), "r"(0u) : "memory");')
    lines.append(f'        __syncthreads();')
    lines.append(f'        unsigned _ld0=0, _ld1=0, _ld2=0, _ld3=0, _lda0=0;')
    lines.append(f'        _t0 = asm_clock64();')
    lines.append(f'        for (int _i = 0; _i < REPS; _i++) {{')
    lines.append(f'            asm volatile(')
    lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4];\\n\\t"')
    lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4+16];\\n\\t"')
    lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4+32];\\n\\t"')
    lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4+48];\\n\\t"')
    lines.append(f'                : "+r"(_ld0), "+r"(_ld1), "+r"(_ld2), "+r"(_ld3)')
    lines.append(f'                : "l"(_gptr + ((_i & 0x3) << 2))')
    lines.append(f'                : "memory"')
    lines.append(f'            );')
    lines.append(f'            _lda0 += _ld0;')
    lines.append(f'            asm volatile(')
    lines.append(f'                "cvt.rn.bf16x2.f32 %0, %5, %4;\\n\\t"')
    lines.append(f'                "fma.rn.bf16x2 %1, %1, %3, %1;\\n\\t"')
    lines.append(f'                "add.rn.bf16x2 %2, %2, %3;\\n\\t"')
    lines.append(f'                : "+r"(_cv), "+r"(_h0), "+r"(_h1)')
    lines.append(f'                : "r"(_hs), "f"(_f0), "f"(_f1)')
    lines.append(f'            );')
    lines.append(f'            uint32_t _sa = _base + ((_i & 3) << 6);')
    lines.append(f'            asm volatile(')
    lines.append(f'                "st.shared.v4.b32 [%0], {{%1,%2,%3,%4}};\\n\\t"')
    lines.append(f'                :: "r"(_sa), "r"(_ld0), "r"(_ld1), "r"(_ld2), "r"(_ld3)')
    lines.append(f'                : "memory"')
    lines.append(f'            );')
    lines.append(f'        }}')
    lines.append(f'        _t1 = asm_clock64();')
    lines.append(f'        ((volatile unsigned*)(_dyn_smem + _warp * 1024))[_lane] = _cv + _h0 + _lda0;')

    # Warp 4: pure LDG (W0 TMA load warp proxy)
    lines.append(f'    }} else if (_warp == 4) {{')
    lines.append(f'        /* W0 proxy: continuous LDG (simulates TMA load warp) */')
    lines.append(f'        const unsigned* _gptr = (const unsigned*)results + _lane * 4;')
    lines.append(f'        unsigned _g0, _g1, _g2, _g3;')
    lines.append(f'        __syncthreads();')
    lines.append(f'        _t0 = asm_clock64();')
    lines.append(f'        for (int _i = 0; _i < REPS; _i++) {{')
    lines.append(f'            asm volatile(')
    lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4];\\n\\t"')
    lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4+16];\\n\\t"')
    lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4+32];\\n\\t"')
    lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4+48];\\n\\t"')
    lines.append(f'                : "+r"(_g0), "+r"(_g1), "+r"(_g2), "+r"(_g3)')
    lines.append(f'                : "l"(_gptr + ((_i & 0x3) << 2))')
    lines.append(f'                : "memory"')
    lines.append(f'            );')
    lines.append(f'        }}')
    lines.append(f'        _t1 = asm_clock64();')
    lines.append(f'        out[64 + _tid] = (long long)(_g0 + _g1 + _g2 + _g3);')

    # Warp 5: pure FFMA (W1 MMA warp proxy)
    lines.append(f'    }} else {{')
    lines.append(f'        /* W1 proxy: continuous FFMA (simulates MMA warp) */')
    lines.append(f'        float _a0=1.001f+_lane, _a1=1.002f+_lane;')
    lines.append(f'        float _a2=1.003f+_lane, _a3=1.004f+_lane;')
    lines.append(f'        float _s0=1.0001f, _s1=0.9999f;')
    lines.append(f'        __syncthreads();')
    lines.append(f'        _t0 = asm_clock64();')
    lines.append(f'        for (int _i = 0; _i < REPS; _i++) {{')
    lines.append(f'            asm volatile(')
    lines.append(f'                "fma.rn.f32 %0, %0, %4, %5;\\n\\t"')
    lines.append(f'                "fma.rn.f32 %1, %1, %4, %5;\\n\\t"')
    lines.append(f'                "fma.rn.f32 %2, %2, %4, %5;\\n\\t"')
    lines.append(f'                "fma.rn.f32 %3, %3, %4, %5;\\n\\t"')
    lines.append(f'                : "+f"(_a0), "+f"(_a1), "+f"(_a2), "+f"(_a3)')
    lines.append(f'                : "f"(_s0), "f"(_s1)')
    lines.append(f'            );')
    lines.append(f'        }}')
    lines.append(f'        _t1 = asm_clock64();')
    lines.append(f'        ((volatile float*)(out + 64))[_tid] = _a0 + _a1 + _a2 + _a3;')
    lines.append(f'    }}')
    lines.append('')
    lines.append(f'    if (_lane == 0) {{')
    lines.append(f'        results[_warp].start = _t0;')
    lines.append(f'        results[_warp].end = _t1;')
    lines.append(f'    }}')
    lines.append(f'}}')

    return name, n_threads, smem_total, 8, '\n'.join(lines)


def gen_asymmetric_kernel(name, n_epi, n_compute, epi_divisor):
    """A-test: epi warps run REPS/divisor, compute warps run full REPS.
    Tests dynamic dispatch recovery — does compute warp CPI improve after
    epi warps exit and free their dispatch slots?"""
    n_warps = n_epi + n_compute
    n_threads = n_warps * 32
    smem_total = n_warps * 1024
    epi_reps = REPS // epi_divisor

    lines = []
    lines.append(f'extern "C" __global__ void {name}(WarpResult* results) {{')
    lines.append(f'    extern __shared__ __align__(128) char _dyn_smem[];')
    lines.append(f'    const int _warp = threadIdx.x / 32;')
    lines.append(f'    const int _lane = threadIdx.x % 32;')
    lines.append(f'    const int _tid = threadIdx.x;')
    lines.append(f'    uint32_t _smem_base = (uint32_t)(uint64_t)(_dyn_smem + _warp * 1024);')
    lines.append(f'    long long* out = (long long*)results;')
    lines.append(f'    (void)_tid; (void)_smem_base; (void)out;')
    lines.append('')
    lines.append(f'    __syncthreads();')
    lines.append(f'    long long _t0, _t1;')
    lines.append('')

    # Epi warps: short-lived ldgmix (REPS/divisor iterations, then exit)
    lines.append(f'    if (_warp < {n_epi}) {{')
    lines.append(f'        /* Epi warp: LDG+BF16+STS for {epi_reps} iters (REPS/{epi_divisor}), then exit */')
    lines.append(f'        uint32_t _base = _smem_base + _lane * 16;')
    lines.append(f'        const unsigned* _gptr = (const unsigned*)results + _lane * 4;')
    lines.append(f'        unsigned _h0=0x3f80+_lane, _h1=0x3f81+_lane;')
    lines.append(f'        unsigned _hs=0x3f803f80;')
    lines.append(f'        float _f0=1.0f+_lane, _f1=2.0f+_lane;')
    lines.append(f'        unsigned _cv=0;')
    lines.append(f'        for (int _i = 0; _i < WARMUP; _i++)')
    lines.append(f'            asm volatile("st.shared.v4.b32 [%0], {{%1,%2,%3,%4}};"')
    lines.append(f'                :: "r"(_base), "r"(0u), "r"(0u), "r"(0u), "r"(0u) : "memory");')
    lines.append(f'        __syncthreads();')
    lines.append(f'        unsigned _ld0=0, _ld1=0, _ld2=0, _ld3=0, _lda0=0;')
    lines.append(f'        _t0 = asm_clock64();')
    lines.append(f'        for (int _i = 0; _i < {epi_reps}; _i++) {{')
    lines.append(f'            asm volatile(')
    lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4];\\n\\t"')
    lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4+16];\\n\\t"')
    lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4+32];\\n\\t"')
    lines.append(f'                "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4+48];\\n\\t"')
    lines.append(f'                : "+r"(_ld0), "+r"(_ld1), "+r"(_ld2), "+r"(_ld3)')
    lines.append(f'                : "l"(_gptr + ((_i & 0x3) << 2))')
    lines.append(f'                : "memory"')
    lines.append(f'            );')
    lines.append(f'            _lda0 += _ld0;')
    lines.append(f'            asm volatile(')
    lines.append(f'                "cvt.rn.bf16x2.f32 %0, %5, %4;\\n\\t"')
    lines.append(f'                "fma.rn.bf16x2 %1, %1, %3, %1;\\n\\t"')
    lines.append(f'                "add.rn.bf16x2 %2, %2, %3;\\n\\t"')
    lines.append(f'                : "+r"(_cv), "+r"(_h0), "+r"(_h1)')
    lines.append(f'                : "r"(_hs), "f"(_f0), "f"(_f1)')
    lines.append(f'            );')
    lines.append(f'            uint32_t _sa = _base + ((_i & 3) << 6);')
    lines.append(f'            asm volatile(')
    lines.append(f'                "st.shared.v4.b32 [%0], {{%1,%2,%3,%4}};\\n\\t"')
    lines.append(f'                :: "r"(_sa), "r"(_ld0), "r"(_ld1), "r"(_ld2), "r"(_ld3)')
    lines.append(f'                : "memory"')
    lines.append(f'            );')
    lines.append(f'        }}')
    lines.append(f'        _t1 = asm_clock64();')
    lines.append(f'        ((volatile unsigned*)(_dyn_smem + _warp * 1024))[_lane] = _cv + _h0 + _lda0;')
    lines.append(f'        /* Epi warp exits — frees dispatch slot for compute warps */')

    # Compute warps: full REPS FFMA
    lines.append(f'    }} else {{')
    lines.append(f'        /* Compute warp: FFMA for full {REPS} iters (measures dynamic recovery) */')
    lines.append(f'        float _a0=1.001f+_lane, _a1=1.002f+_lane;')
    lines.append(f'        float _a2=1.003f+_lane, _a3=1.004f+_lane;')
    lines.append(f'        float _s0=1.0001f, _s1=0.9999f;')
    lines.append(f'        for (int _w = 0; _w < WARMUP; _w++)')
    lines.append(f'            asm volatile("fma.rn.f32 %0, %0, %2, %3;"')
    lines.append(f'                : "+f"(_a0) : "f"(_a0), "f"(_s0), "f"(_s1));')
    lines.append(f'        __syncthreads();')
    lines.append(f'        _t0 = asm_clock64();')
    lines.append(f'        for (int _i = 0; _i < REPS; _i++) {{')
    lines.append(f'            asm volatile(')
    lines.append(f'                "fma.rn.f32 %0, %0, %4, %5;\\n\\t"')
    lines.append(f'                "fma.rn.f32 %1, %1, %4, %5;\\n\\t"')
    lines.append(f'                "fma.rn.f32 %2, %2, %4, %5;\\n\\t"')
    lines.append(f'                "fma.rn.f32 %3, %3, %4, %5;\\n\\t"')
    lines.append(f'                : "+f"(_a0), "+f"(_a1), "+f"(_a2), "+f"(_a3)')
    lines.append(f'                : "f"(_s0), "f"(_s1)')
    lines.append(f'            );')
    lines.append(f'        }}')
    lines.append(f'        _t1 = asm_clock64();')
    lines.append(f'        ((volatile float*)(out + 64))[_tid] = _a0 + _a1 + _a2 + _a3;')
    lines.append(f'    }}')
    lines.append('')
    lines.append(f'    if (_lane == 0) {{')
    lines.append(f'        results[_warp].start = _t0;')
    lines.append(f'        results[_warp].end = _t1;')
    lines.append(f'    }}')
    lines.append(f'}}')

    return name, n_threads, smem_total, 8, '\n'.join(lines)


def gen_barsync_mixed_kernel(name, n_bar_warps, n_compute_warps, bar_interval):
    """B-test (mixed): bar warps do LDG+BF16+STS + periodic BAR.SYNC.
    More realistic than pure-STS B-tests — matches actual FC2 epilogue pipe mix.
    Compute warps run continuous FFMA."""
    n_warps = n_bar_warps + n_compute_warps
    n_threads = n_warps * 32
    bar_thread_count = n_bar_warps * 32
    smem_total = n_warps * 1024

    lines = []
    lines.append(f'extern "C" __global__ void {name}(WarpResult* results) {{')
    lines.append(f'    extern __shared__ __align__(128) char _dyn_smem[];')
    lines.append(f'    const int _warp = threadIdx.x / 32;')
    lines.append(f'    const int _lane = threadIdx.x % 32;')
    lines.append(f'    const int _tid = threadIdx.x;')
    lines.append(f'    uint32_t _smem_base = (uint32_t)(uint64_t)(_dyn_smem + _warp * 1024);')
    lines.append(f'    long long* out = (long long*)results;')
    lines.append(f'    (void)_tid; (void)_smem_base; (void)out;')
    lines.append('')
    lines.append(f'    __syncthreads();')
    lines.append(f'    long long _t0, _t1;')
    lines.append('')

    # Bar warps: LDG+BF16+STS with periodic BAR.SYNC
    lines.append(f'    if (_warp < {n_bar_warps}) {{')
    lines.append(f'        /* BAR warp: LDG+BF16+STS + periodic BAR.SYNC (interval={bar_interval}) */')
    lines.append(f'        uint32_t _base = _smem_base + _lane * 16;')
    lines.append(f'        const unsigned* _gptr = (const unsigned*)results + _lane * 4;')
    lines.append(f'        unsigned _h0=0x3f80+_lane, _h1=0x3f81+_lane;')
    lines.append(f'        unsigned _hs=0x3f803f80;')
    lines.append(f'        float _f0=1.0f+_lane, _f1=2.0f+_lane;')
    lines.append(f'        unsigned _cv=0;')
    lines.append(f'        for (int _i = 0; _i < WARMUP; _i++)')
    lines.append(f'            asm volatile("st.shared.v4.b32 [%0], {{%1,%2,%3,%4}};"')
    lines.append(f'                :: "r"(_base), "r"(0u), "r"(0u), "r"(0u), "r"(0u) : "memory");')
    lines.append(f'        __syncthreads();')
    lines.append(f'        unsigned _ld0=0, _ld1=0, _ld2=0, _ld3=0, _lda0=0;')
    lines.append(f'        _t0 = asm_clock64();')

    # Mixed epi body (LDG+BF16+STS per iter)
    mixed_iter = '\n'.join([
        f'                asm volatile(',
        f'                    "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4];\\n\\t"',
        f'                    "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4+16];\\n\\t"',
        f'                    "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4+32];\\n\\t"',
        f'                    "ld.global.v4.b32 {{%0,%1,%2,%3}}, [%4+48];\\n\\t"',
        f'                    : "+r"(_ld0), "+r"(_ld1), "+r"(_ld2), "+r"(_ld3)',
        f'                    : "l"(_gptr + ((_inner & 0x3) << 2))',
        f'                    : "memory"',
        f'                );',
        f'                _lda0 += _ld0;',
        f'                asm volatile(',
        f'                    "cvt.rn.bf16x2.f32 %0, %5, %4;\\n\\t"',
        f'                    "fma.rn.bf16x2 %1, %1, %3, %1;\\n\\t"',
        f'                    "add.rn.bf16x2 %2, %2, %3;\\n\\t"',
        f'                    : "+r"(_cv), "+r"(_h0), "+r"(_h1)',
        f'                    : "r"(_hs), "f"(_f0), "f"(_f1)',
        f'                );',
        f'                uint32_t _sa = _base + ((_inner & 3) << 6);',
        f'                asm volatile(',
        f'                    "st.shared.v4.b32 [%0], {{%1,%2,%3,%4}};\\n\\t"',
        f'                    :: "r"(_sa), "r"(_ld0), "r"(_ld1), "r"(_ld2), "r"(_ld3)',
        f'                    : "memory"',
        f'                );',
    ])

    if bar_interval > 0:
        lines.append(f'        for (int _outer = 0; _outer < REPS / {bar_interval}; _outer++) {{')
        lines.append(f'            for (int _inner = 0; _inner < {bar_interval}; _inner++) {{')
        lines.append(mixed_iter)
        lines.append(f'            }}')
        lines.append(f'            asm volatile("bar.sync 1, {bar_thread_count};" ::: "memory");')
        lines.append(f'        }}')
    else:
        lines.append(f'        for (int _inner = 0; _inner < REPS; _inner++) {{')
        lines.append(mixed_iter)
        lines.append(f'        }}')

    lines.append(f'        _t1 = asm_clock64();')
    lines.append(f'        ((volatile unsigned*)(_dyn_smem + _warp * 1024))[_lane] = _cv + _h0 + _lda0;')

    # Compute warps: continuous FFMA
    lines.append(f'    }} else {{')
    lines.append(f'        /* Compute warp: continuous FFMA */')
    lines.append(f'        float _a0=1.001f+_lane, _a1=1.002f+_lane, _a2=1.003f+_lane, _a3=1.004f+_lane;')
    lines.append(f'        float _s0=1.0001f, _s1=0.9999f;')
    lines.append(f'        for (int _i = 0; _i < WARMUP; _i++)')
    lines.append(f'            asm volatile("fma.rn.f32 %0, %0, %2, %3;" : "+f"(_a0) : "f"(_a0), "f"(_s0), "f"(_s1));')
    lines.append(f'        __syncthreads();')
    lines.append(f'        _t0 = asm_clock64();')
    lines.append(f'        for (int _i = 0; _i < REPS; _i++) {{')
    lines.append(f'            asm volatile(')
    lines.append(f'                "fma.rn.f32 %0, %0, %4, %5;\\n\\t"')
    lines.append(f'                "fma.rn.f32 %1, %1, %4, %5;\\n\\t"')
    lines.append(f'                "fma.rn.f32 %2, %2, %4, %5;\\n\\t"')
    lines.append(f'                "fma.rn.f32 %3, %3, %4, %5;\\n\\t"')
    lines.append(f'                : "+f"(_a0), "+f"(_a1), "+f"(_a2), "+f"(_a3)')
    lines.append(f'                : "f"(_s0), "f"(_s1)')
    lines.append(f'            );')
    lines.append(f'        }}')
    lines.append(f'        _t1 = asm_clock64();')
    lines.append(f'        ((volatile float*)(out + 64))[_tid] = _a0 + _a1 + _a2 + _a3;')
    lines.append(f'    }}')
    lines.append('')
    lines.append(f'    if (_lane == 0) {{')
    lines.append(f'        results[_warp].start = _t0;')
    lines.append(f'        results[_warp].end = _t1;')
    lines.append(f'    }}')
    lines.append(f'}}')

    return name, n_threads, smem_total, 8, '\n'.join(lines)


def gen_main(all_tests):
    """Generate main() with dual measurement:
    - S-tests: per-warp clock64 CPI (accurate for same-pipe scaling)
    - X/F/P/B/N/A tests: cudaEvent wall time (avoids broken per-warp clocks)
    """
    lines = []
    lines.append('')
    lines.append(f'#define WARMUP_LAUNCHES {WARMUP_LAUNCHES}')
    lines.append(f'#define MEASURE_LAUNCHES {MEASURE_LAUNCHES}')
    lines.append('')
    lines.append('struct Test {')
    lines.append('    const char* name;')
    lines.append('    const char* category;')
    lines.append('    int n_warps;')
    lines.append('    int insns_per_iter;')
    lines.append('    int smem_bytes;')
    lines.append('    int use_events;  /* 0=per-warp clock64, 1=cudaEvent wall time */')
    lines.append('    void (*fn)(WarpResult*);')
    lines.append('};')
    lines.append('')

    lines.append('Test tests[] = {')
    for name, category, n_warps, insns_per_iter, smem_bytes in all_tests:
        use_events = 0 if category.startswith('S:') else 1
        lines.append(f'    {{"{name}", "{category}", {n_warps}, {insns_per_iter}, {smem_bytes}, {use_events}, {name}}},')
    lines.append('};')
    lines.append('')

    lines.append('int main() {')
    lines.append('    WarpResult *d_results, h_results[MAX_WARPS];')
    lines.append('    cudaMalloc(&d_results, MAX_WARPS * sizeof(WarpResult) + 65536);')
    lines.append('    cudaMemset(d_results, 0, MAX_WARPS * sizeof(WarpResult) + 65536);')
    lines.append('')
    lines.append('    /* Large buffer for L2-miss LDG tests */')
    lines.append('    char* h_large_buf;')
    lines.append('    cudaMalloc(&h_large_buf, LARGE_BUF_SIZE);')
    lines.append('    cudaMemset(h_large_buf, 0x42, LARGE_BUF_SIZE);')
    lines.append('    cudaMemcpyToSymbol(g_large_buf, &h_large_buf, sizeof(char*));')
    lines.append('')
    lines.append('    /* cudaEvent for wall-time measurement */')
    lines.append('    cudaEvent_t ev_start, ev_stop;')
    lines.append('    cudaEventCreate(&ev_start);')
    lines.append('    cudaEventCreate(&ev_stop);')
    lines.append('')
    lines.append('    /* Per-warp min elapsed across all measured launches */')
    lines.append('    long long min_elapsed[MAX_WARPS];')
    lines.append('    /* Wall-time measurements (us) */')
    lines.append('    float wall_times[MEASURE_LAUNCHES];')
    lines.append('')
    lines.append('    int n = sizeof(tests) / sizeof(tests[0]);')
    lines.append(f'    printf("Warp Scaling Calibration — SM100a\\n");')
    lines.append(f'    printf("REPS=%d, %d warmup + %d measured launches\\n",')
    lines.append(f'           REPS, WARMUP_LAUNCHES, MEASURE_LAUNCHES);')
    lines.append(f'    printf("S-tests: per-warp clock64 CPI | X/F/P/B/N/A: cudaEvent wall time (us)\\n\\n");')
    lines.append('')
    lines.append('    const char* prev_cat = "";')
    lines.append('    for (int t = 0; t < n; t++) {')
    lines.append('        if (strcmp(tests[t].category, prev_cat) != 0) {')
    lines.append('            printf("\\n=== %s ===\\n", tests[t].category);')
    lines.append('            if (!tests[t].use_events) {')
    lines.append('                printf("%-30s %6s", "Test", "Warps");')
    lines.append(f'                for (int w = 0; w < MAX_WARPS; w++)')
    lines.append(f'                    printf("  W%d_cyc/i", w);')
    lines.append(f'                printf("  Max_cyc/i  Min_cyc/i\\n");')
    lines.append('            } else {')
    lines.append('                printf("%-30s %6s  %10s  %10s  %10s\\n",')
    lines.append('                       "Test", "Warps", "min_us", "med_us", "max_us");')
    lines.append('            }')
    lines.append('            prev_cat = tests[t].category;')
    lines.append('        }')
    lines.append('')
    lines.append('        if (tests[t].smem_bytes > 0)')
    lines.append('            cudaFuncSetAttribute(tests[t].fn, cudaFuncAttributeMaxDynamicSharedMemorySize, tests[t].smem_bytes);')
    lines.append('')
    lines.append('        /* Warmup launches */')
    lines.append('        fflush(stdout);')
    lines.append('        for (int r = 0; r < WARMUP_LAUNCHES; r++) {')
    lines.append('            cudaMemset(d_results, 0, MAX_WARPS * sizeof(WarpResult) + 65536);')
    lines.append('            tests[t].fn<<<1, tests[t].n_warps * 32, tests[t].smem_bytes>>>(d_results);')
    lines.append('            CUDA_CHECK(cudaGetLastError());')
    lines.append('            CUDA_CHECK(cudaDeviceSynchronize());')
    lines.append('        }')
    lines.append('')
    lines.append('        if (!tests[t].use_events) {')
    lines.append('            /* ── S-tests: per-warp clock64 measurement ── */')
    lines.append('            for (int w = 0; w < MAX_WARPS; w++) min_elapsed[w] = 0x7FFFFFFFFFFFFFFFLL;')
    lines.append('            for (int r = 0; r < MEASURE_LAUNCHES; r++) {')
    lines.append('                cudaMemset(d_results, 0, MAX_WARPS * sizeof(WarpResult) + 65536);')
    lines.append('                tests[t].fn<<<1, tests[t].n_warps * 32, tests[t].smem_bytes>>>(d_results);')
    lines.append('                CUDA_CHECK(cudaGetLastError());')
    lines.append('                CUDA_CHECK(cudaDeviceSynchronize());')
    lines.append('                cudaMemcpy(h_results, d_results, MAX_WARPS * sizeof(WarpResult), cudaMemcpyDeviceToHost);')
    lines.append('                for (int w = 0; w < tests[t].n_warps; w++) {')
    lines.append('                    long long e = h_results[w].end - h_results[w].start;')
    lines.append('                    if (e < min_elapsed[w]) min_elapsed[w] = e;')
    lines.append('                }')
    lines.append('            }')
    lines.append('            printf("%-30s %6d", tests[t].name, tests[t].n_warps);')
    lines.append('            double max_cpi = 0, min_cpi_all = 1e18;')
    lines.append('            for (int w = 0; w < tests[t].n_warps; w++) {')
    lines.append(f'                double cpi = (double)min_elapsed[w] / (REPS * tests[t].insns_per_iter);')
    lines.append('                printf("  %9.2f", cpi);')
    lines.append('                if (cpi > max_cpi) max_cpi = cpi;')
    lines.append('                if (cpi < min_cpi_all) min_cpi_all = cpi;')
    lines.append('            }')
    lines.append('            for (int w = tests[t].n_warps; w < MAX_WARPS; w++)')
    lines.append('                printf("          -");')
    lines.append('            printf("  %9.2f  %9.2f\\n", max_cpi, min_cpi_all);')
    lines.append('')
    lines.append('        } else {')
    lines.append('            /* ── X/F/P/B/N/A tests: cudaEvent wall-time measurement ── */')
    lines.append('            const int BATCH = 100; /* launches per event pair for resolution */')
    lines.append('            for (int r = 0; r < MEASURE_LAUNCHES; r++) {')
    lines.append('                cudaEventRecord(ev_start);')
    lines.append('                for (int b = 0; b < BATCH; b++)')
    lines.append('                    tests[t].fn<<<1, tests[t].n_warps * 32, tests[t].smem_bytes>>>(d_results);')
    lines.append('                cudaEventRecord(ev_stop);')
    lines.append('                cudaEventSynchronize(ev_stop);')
    lines.append('                float ms = 0;')
    lines.append('                cudaEventElapsedTime(&ms, ev_start, ev_stop);')
    lines.append('                wall_times[r] = ms * 1000.0f / BATCH; /* per-launch us */')
    lines.append('            }')
    lines.append('            /* Sort for min/median/max */')
    lines.append('            std::sort(wall_times, wall_times + MEASURE_LAUNCHES);')
    lines.append('            float min_us = wall_times[0];')
    lines.append('            float med_us = wall_times[MEASURE_LAUNCHES / 2];')
    lines.append('            float max_us = wall_times[MEASURE_LAUNCHES - 1];')
    lines.append('            printf("%-30s %6d  %10.3f  %10.3f  %10.3f",')
    lines.append('                   tests[t].name, tests[t].n_warps, min_us, med_us, max_us);')
    lines.append('            /* Debug: verify events work on first event-based test */')
    lines.append('            if (t < n && strcmp(tests[t].name, "X_4pipe") == 0) {')
    lines.append('                cudaError_t e1 = cudaEventRecord(ev_start);')
    lines.append('                for (int b = 0; b < 1000; b++)')
    lines.append('                    tests[t].fn<<<1, tests[t].n_warps * 32, tests[t].smem_bytes>>>(d_results);')
    lines.append('                cudaError_t e2 = cudaGetLastError();')
    lines.append('                cudaError_t e3 = cudaEventRecord(ev_stop);')
    lines.append('                cudaError_t e4 = cudaEventSynchronize(ev_stop);')
    lines.append('                float raw_ms = 0;')
    lines.append('                cudaError_t e5 = cudaEventElapsedTime(&raw_ms, ev_start, ev_stop);')
    lines.append('                printf("  [evRec=%d kern=%d evStop=%d sync=%d elapsed=%d ms=%.6f]",')
    lines.append('                       e1, e2, e3, e4, e5, raw_ms);')
    lines.append('            }')
    lines.append('            printf("\\n");')
    lines.append('        }')
    lines.append('    }')
    lines.append('')
    lines.append('    printf("\\n@@WARP_SCALING\\n");')
    lines.append('    cudaEventDestroy(ev_start);')
    lines.append('    cudaEventDestroy(ev_stop);')
    lines.append('    cudaFree(h_large_buf);')
    lines.append('    cudaFree(d_results);')
    lines.append('    return 0;')
    lines.append('}')

    return '\n'.join(lines)


def main():
    all_kernels = []  # (code_string, ...)
    all_tests = []    # (name, category, n_warps, insns_per_iter, smem_bytes)

    # ─── S-tests: same-pipe scaling 1-8 warps ───
    for pipe in S_PIPES:
        for nw in WARP_COUNTS:
            name, n_threads, smem, insns, code = gen_same_pipe_kernel(pipe, nw)
            all_kernels.append(code)
            all_tests.append((name, f'S: {PIPE_WORKLOADS[pipe]["desc"]} scaling', nw, insns, smem))

    # ─── X-tests: cross-pipe independence ───
    for config_name, assignments in X_CONFIGS:
        name, n_threads, smem, insns, code = gen_cross_pipe_kernel(config_name, assignments)
        all_kernels.append(code)
        n_warps = max(assignments.keys()) + 1
        all_tests.append((name, 'X: Cross-pipe independence', n_warps, insns, smem))

    # ─── F-tests: foreground/background interference ───
    for fname, fg, fg_n, bg, bg_n in F_CONFIGS:
        if fname == 'F_fc2_exact_6w':
            name, n_threads, smem, insns, code = gen_fc2_exact_kernel(fname)
        elif fname == 'F_4mix_2ffma':
            name, n_threads, smem, insns, code = gen_mixed_epi_kernel(fname, fg_n, bg_n)
        elif fname.startswith('F_4ldg_l2'):
            name, n_threads, smem, insns, code = gen_ldgmix_epi_kernel(fname, fg_n, bg_n, l2=True)
        elif fname.startswith('F_4ldgmix'):
            name, n_threads, smem, insns, code = gen_ldgmix_epi_kernel(fname, fg_n, bg_n)
        else:
            name, n_threads, smem, insns, code = gen_fg_bg_kernel(fname, fg, fg_n, bg, bg_n)
        all_kernels.append(code)
        all_tests.append((name, 'F: FG/BG interference', n_threads // 32, insns, smem))

    # ─── P-tests: producer-consumer (CUTLASS W3 pattern) ───
    for pname, n_epi, load_mode, n_kloop, has_lw in P_CONFIGS:
        name, n_threads, smem, insns, code = gen_prodcons_kernel(
            pname, n_epi, load_mode, n_kloop, has_lw)
        all_kernels.append(code)
        nw = n_epi + n_kloop + (1 if has_lw else 0)
        all_tests.append((name, 'P: Producer-consumer (W3)', nw, insns, smem))

    # ─── B-tests: BAR.SYNC contention effect ───
    for bname, n_bar, n_cmp, interval in B_CONFIGS:
        name, n_threads, smem, insns, code = gen_barsync_kernel(bname, n_bar, n_cmp, interval)
        all_kernels.append(code)
        all_tests.append((name, 'B: BAR.SYNC effect', n_bar + n_cmp, insns, smem))

    # ─── N-tests: nanosleep calibration ───
    for nname, sleep_val in N_CONFIGS:
        name, n_threads, smem, insns, code = gen_nanosleep_kernel(nname, sleep_val)
        all_kernels.append(code)
        all_tests.append((name, 'N: Nanosleep calibration', 1, insns, smem))

    # ─── A-tests: asymmetric duration (dispatch recovery) ───
    for aname, n_epi, n_compute, divisor in A_CONFIGS:
        name, n_threads, smem, insns, code = gen_asymmetric_kernel(
            aname, n_epi, n_compute, divisor)
        all_kernels.append(code)
        all_tests.append((name, 'A: Asymmetric duration', n_threads // 32, insns, smem))

    # ─── B-tests (mixed): BAR.SYNC with realistic epi workload ───
    for bname, n_bar, n_cmp, interval in B_MIXED_CONFIGS:
        name, n_threads, smem, insns, code = gen_barsync_mixed_kernel(
            bname, n_bar, n_cmp, interval)
        all_kernels.append(code)
        all_tests.append((name, 'B: BAR.SYNC mixed-epi', n_bar + n_cmp, insns, smem))

    # Count
    n_kernels = len(all_kernels)
    n_s = sum(1 for t in all_tests if t[1].startswith('S:'))
    n_x = sum(1 for t in all_tests if t[1].startswith('X:'))
    n_f = sum(1 for t in all_tests if t[1].startswith('F:'))
    n_p = sum(1 for t in all_tests if t[1].startswith('P:'))
    n_b = sum(1 for t in all_tests if t[1].startswith('B:'))
    n_n = sum(1 for t in all_tests if t[1].startswith('N:'))
    n_a = sum(1 for t in all_tests if t[1].startswith('A:'))

    # Write .cu file
    with open(OUTFILE, 'w') as f:
        f.write(gen_header())
        f.write('\n\n')
        for code in all_kernels:
            f.write(code)
            f.write('\n\n')
        f.write(gen_main(all_tests))
        f.write('\n')

    print(f'Generated {OUTFILE}')
    print(f'  {n_kernels} kernels total: {n_s} S, {n_x} X, {n_f} F, {n_p} P, {n_b} B, {n_n} N, {n_a} A tests')
    print(f'  S-tests: {len(S_PIPES)} pipes × {len(WARP_COUNTS)} warp counts = {n_s}')
    print(f'  Measurement: {WARMUP_LAUNCHES} warmup + {MEASURE_LAUNCHES} measured launches, min CPI')
    print(f'  Warp counts: {WARP_COUNTS}')


if __name__ == '__main__':
    main()
