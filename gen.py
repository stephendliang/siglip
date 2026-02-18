#!/usr/bin/env python3
"""
gen.py — SigLIP2 patch embed GEMM with warp-specialized pipelining.

Emits megakernel.cu with:
  - 4 warps (128 threads), TMEM double-buffered:
    Warp 0:   Load  (TMA async bulk copy, 1 thread active) + TMEM alloc/dealloc
    Warp 1:   MMA   (tcgen05.mma.cta_group::1, 1 thread active)
    Warps 2-3: Epilogue for PREVIOUS tile (overlapped, 2 warps × 2 passes)
    Warps 0-3: Drain epilogue for LAST tile (4×32 = 128 rows)

  cta_group::1 uses one warpgroup (warps 0-3) — no cluster launch needed.
  Warp 0 (WG0 leader) allocates TMEM via cta_group::1.

  N-stage circular buffer for Load↔MMA pipelining.
  Producer-consumer mbarrier protocol:
    tma_mbar[s]:  Load→MMA  (data ready, via TMA complete_tx::bytes)
    mma_mbar[s]:  MMA→Load  (buffer free, via tcgen05.commit)

  TMA descriptors for A and B (cuTensorMapEncodeTiled on host).
  No __syncthreads in the K-loop hot path — only targeted mbarrier waits.

  tcgen05.fence::before_thread_sync + __syncthreads + fence::after at
  bottom of each tile iteration: drains MMA writes AND blocks until
  warps 2-3 finish TMEM reads (safe to overwrite buffer next iteration).

Hardened:
  - M/N/K need NOT be divisible by tile dims (TMA OOB zero-fill).
  - --no-coop disables cooperative launch / grid sync.
  - --cluster-size overrides target default (1 = no clustering).

Usage:
    python3 gen.py                        # B200, 3-stage pipeline
    python3 gen.py --pipeline-stages 5    # deeper pipeline
    python3 gen.py --target b300          # B300
    python3 gen.py --no-coop              # no grid sync
    python3 gen.py --dry-run              # analysis only
"""

import argparse
from math import ceil

# ═══════════════════════════════════════════════════════════════════
# Root constants
# ═══════════════════════════════════════════════════════════════════

TARGETS = {
    "b200": dict(sm_count=148, cluster_size=4,  smem_bytes=228 * 1024),
    "b300": dict(sm_count=160, cluster_size=16, smem_bytes=228 * 1024),
}

D_MODEL   = 768
SEQ_LEN   = 196
PATCH_DIM = 768

# ═══════════════════════════════════════════════════════════════════
# Derived computation
# ═══════════════════════════════════════════════════════════════════

def align_up(x, a=128):
    return (x + a - 1) & ~(a - 1)


def compute(args):
    t = TARGETS[args.target]
    c = {}

    # ── Hardware ──
    c["target"]       = args.target.upper()
    c["sm_count"]     = t["sm_count"]
    c["cluster_size"] = args.cluster_size if args.cluster_size is not None else t["cluster_size"]
    c["smem_bytes"]   = t["smem_bytes"]
    c["use_coop"]     = not args.no_coop
    c["snake"]        = not args.no_snake

    # ── Warp config ──
    c["n_warps"]   = 4     # 1 Load + 1 MMA + 2 idle (1 warpgroup)
    c["threads"]   = c["n_warps"] * 32  # 128
    c["cta_group"] = 1     # 1 warpgroup, no cluster launch needed

    # ── Pipeline ──
    c["n_stages"] = args.pipeline_stages

    # ── Tile dims ──
    c["TM"] = args.tile_m
    c["TN"] = args.tile_n
    c["TK"] = args.tile_k

    # ── Batch ──
    c["imgs_per_sm"] = args.imgs_per_sm
    c["B"] = c["sm_count"] * c["imgs_per_sm"]

    # ── GEMM dimensions ──
    c["M"] = c["B"] * SEQ_LEN
    c["N"] = D_MODEL
    c["K"] = PATCH_DIM

    # ── Tile counts ──
    assert c["TM"] in (64, 128, 256), "TM must be 64, 128, or 256"
    assert c["TN"] % 8 == 0 and 8 <= c["TN"] <= 256, "TN must be 8..256 step 8"
    assert c["TK"] in (32, 128), "TK must be 32 or 128 for FP8"

    # ── MMA sub-iterations (hardware MMA K=32, TK may be larger) ──
    c["mma_k"] = 32
    c["mma_per_ki"] = c["TK"] // c["mma_k"]

    c["tiles_m"]     = ceil(c["M"] / c["TM"])
    c["tiles_n"]     = ceil(c["N"] / c["TN"])
    c["k_iters"]     = ceil(c["K"] / c["TK"])
    c["total_tiles"] = c["tiles_m"] * c["tiles_n"]

    c["m_pad"] = c["tiles_m"] * c["TM"] - c["M"]
    c["n_pad"] = c["tiles_n"] * c["TN"] - c["N"]
    c["k_pad"] = c["k_iters"] * c["TK"] - c["K"]

    # Balanced SM assignment
    c["tiles_floor"]    = c["total_tiles"] // c["sm_count"]
    c["tiles_ceil"]     = c["tiles_floor"] + 1
    c["sms_with_extra"] = c["total_tiles"] % c["sm_count"]
    c["sms_with_floor"] = c["sm_count"] - c["sms_with_extra"]
    c["max_tiles_per_sm"] = c["tiles_ceil"] if c["sms_with_extra"] > 0 else c["tiles_floor"]
    c["wgmma_per_sm"]    = c["max_tiles_per_sm"] * c["k_iters"] * c["mma_per_ki"]

    # ── Per-tile sizes (bytes, FP8 = 1 byte/element) ──
    c["sz_a"] = c["TM"] * c["TK"]
    c["sz_b"] = c["TN"] * c["TK"]

    # ── SMEM layout: N_STAGES × (A_tile + B_tile) + metadata ──
    off = 0
    c["off_a"] = []
    c["off_b"] = []
    for s in range(c["n_stages"]):
        c["off_a"].append(off);  off += align_up(c["sz_a"], 128)
        c["off_b"].append(off);  off += align_up(c["sz_b"], 128)

    c["off_tmem_0"] = off;  off += 4
    c["off_tmem_1"] = off;  off = align_up(off + 4, 16)

    # Mbarriers: tma_mbar[N] + mma_mbar[N]  (each 8 bytes, 8B aligned)
    off = align_up(off, 8)
    c["off_tma_mbar"] = off;  off += 8 * c["n_stages"]
    off = align_up(off, 8)
    c["off_mma_mbar"] = off;  off += 8 * c["n_stages"]
    off = align_up(off, 8)

    c["smem_used"] = off
    assert c["smem_used"] <= c["smem_bytes"], \
        f"SMEM: {c['smem_used']} > {c['smem_bytes']}"

    # TMA bytes per stage
    c["tma_bytes"] = c["sz_a"] + c["sz_b"]

    # ── SMEM descriptor params (swizzle mode derived from TK) ──
    c["sbo"] = 8 * c["TK"]
    if c["TK"] >= 128:
        c["swizzle_name"] = "SWIZZLE_128B"
        c["swizzle_desc_bits"] = "(1ULL << 62)"   # mode 2 in bits [63:61]
        c["swizzle_cu_enum"] = "CU_TENSOR_MAP_SWIZZLE_128B"
    else:
        c["swizzle_name"] = "SWIZZLE_32B"
        c["swizzle_desc_bits"] = "(1ULL << 61)"   # mode 1 in bits [63:61]
        c["swizzle_cu_enum"] = "CU_TENSOR_MAP_SWIZZLE_32B"

    # ── TMEM ──
    c["tmem_cols"] = c["TN"]
    c["tmem_cols_total"] = 2 * c["TN"]

    # ── Instruction descriptor (idesc) ──
    idesc  = 0
    idesc |= (1 << 4)
    idesc |= (0 << 7)
    idesc |= (0 << 10)
    idesc |= (0 << 15)
    idesc |= (0 << 16)
    idesc |= ((c["TN"] // 8) << 17)
    idesc |= ((c["TM"] // 16) << 24)
    c["idesc"]     = idesc
    c["idesc_hex"] = f"0x{idesc:08X}"

    c["seq_len"]  = SEQ_LEN

    return c


# ═══════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════

def print_analysis(c):
    pct = 100 * c["smem_used"] / c["smem_bytes"]
    pad_parts = []
    if c["m_pad"]: pad_parts.append(f"M+{c['m_pad']}")
    if c["n_pad"]: pad_parts.append(f"N+{c['n_pad']}")
    if c["k_pad"]: pad_parts.append(f"K+{c['k_pad']}")
    pad_str = ", ".join(pad_parts) if pad_parts else "none"
    coop_str = "cooperative" if c["use_coop"] else "non-cooperative"
    snake_str = "snake (zigzag)" if c["snake"] else "row-major"
    cg = c["cta_group"]
    nw = c["n_warps"]
    print(f"""
{'='*66}
  SigLIP2 Patch Embed GEMM — warp-specialized tcgen05 pipeline
{'='*66}
  Target:    {c['target']} ({c['sm_count']} SMs, cluster-of-{c['cluster_size']})
  Launch:    {coop_str}
  Tile order: {snake_str}
  Pipeline:  {c['n_stages']}-stage circular buffer (TMA + mbarrier)
  Batch:     {c['B']} images ({c['imgs_per_sm']} per SM)
  Warps:     {nw} ({nw * 32} threads)  cta_group::{cg}

  GEMM:  A[{c['M']:,}, {c['K']}] × B[{c['N']}, {c['K']}]^T → C[{c['M']:,}, {c['N']}]
  Padding: {pad_str}

  Tile dims:  M={c['TM']}  N={c['TN']}  K={c['TK']}
  Tile grid:
    M tiles: {c['tiles_m']}   N tiles: {c['tiles_n']}   K iters: {c['k_iters']}
    Total output tiles: {c['total_tiles']:,}
    Per SM: {c['sms_with_extra']} × {c['tiles_ceil']} + {c['sms_with_floor']} × {c['tiles_floor']}
    WGMMA / SM: {c['wgmma_per_sm']:,}

  Warp roles ({nw} warps = {nw*32} threads, TMEM double-buffered):
    Warp 0:   Load  (TMA async bulk copy, 1 thread active) + TMEM alloc/dealloc
    Warp 1:   MMA   (tcgen05.mma.cta_group::{cg}, 1 thread active) → tmem_base[buf]
    Warps 2-{nw-1}: Overlapped epilogue for PREVIOUS tile from tmem_base[buf^1]
              (2 warps × 2 passes, skipped on first tile)
    Drain:    All {nw} warps epilogue for LAST tile ({nw} × 32 = {nw*32} rows)

  idesc: {c['idesc_hex']}  (E4M3 × E4M3 → FP32, M={c['TM']}, N={c['TN']})
  SMEM desc: SBO={c['sbo']}B  {c['swizzle_name']}  (LBO implicit)  MMA/iter: {c['mma_per_ki']}
  TMEM: 2 × {c['tmem_cols']} = {c['tmem_cols_total']} columns ({100*c['tmem_cols_total']/512:.0f}% of 512) [double-buffered]

  SMEM: {c['smem_used']:,} / {c['smem_bytes']:,} bytes ({pct:.1f}%)
    Circular buffer: {c['n_stages']} stages × (A[{c['TM']},{c['TK']}] + B[{c['TN']},{c['TK']}])
    Per stage: {c['sz_a'] + c['sz_b']:,} B   Total: {c['n_stages'] * (c['sz_a'] + c['sz_b']):,} B
    tma_mbar[{c['n_stages']}]:  @{c['off_tma_mbar']:#06x}
    mma_mbar[{c['n_stages']}]:  @{c['off_mma_mbar']:#06x}
    TMEM addr[0]:   @{c['off_tmem_0']:#06x}
    TMEM addr[1]:   @{c['off_tmem_1']:#06x}
{'='*66}
""")


# ═══════════════════════════════════════════════════════════════════
# Code generation
# ═══════════════════════════════════════════════════════════════════

def gen_code(c):
    L = []
    def w(s=""):
        if '\n' in s:
            L.extend(s.split('\n'))
        else:
            L.append(s)

    NS = c["n_stages"]
    CG = c["cta_group"]

    # ────────────────────────────────────────────────────────────
    # Header
    # ────────────────────────────────────────────────────────────
    w(f"""// AUTO-GENERATED by gen.py — do not hand-edit
// Target: {c['target']}  Batch: {c['B']}  GEMM: [{c['M']},{c['K']}]×[{c['N']},{c['K']}]^T
// Pipeline: {NS}-stage  K-iters: {c['k_iters']}  MMA/iter: {c['mma_per_ki']}  idesc: {c['idesc_hex']}
// Warps: {c['n_warps']} ({c['threads']} threads)  cta_group::{CG}
// Warp-specialized: Load(W0) | MMA(W1,cta_group::{CG}) | Epilogue(W0-{c['n_warps']-1})
// tcgen05.mma.cta_group::{CG}.kind::f8f6f4  (E4M3 × E4M3 → FP32)""")
    w()
    if c["use_coop"]:
        w(f"#include <cooperative_groups.h>")
    w(f"""#include <cuda.h>
#include <curand.h>
#include <cstdint>
#include <cstdio>""")
    w()

    # ────────────────────────────────────────────────────────────
    # Defines
    # ────────────────────────────────────────────────────────────
    w(f"""#define SM_COUNT       {c['sm_count']}
#define THREADS        {c['threads']}
#define BATCH_SIZE     {c['B']}
#define SEQ_LEN        {c['seq_len']}
#define M_TOTAL        {c['M']}
#define N_DIM          {c['N']}
#define K_DIM          {c['K']}
#define TM             {c['TM']}
#define TN             {c['TN']}
#define TK             {c['TK']}
#define TILES_M        {c['tiles_m']}
#define TILES_N        {c['tiles_n']}
#define K_ITERS        {c['k_iters']}
#define TOTAL_TILES    {c['total_tiles']}
#define N_STAGES       {NS}""")
    a_offs = ", ".join(str(c["off_a"][s]) for s in range(NS))
    b_offs = ", ".join(str(c["off_b"][s]) for s in range(NS))
    w(f"""#define OFF_TMEM_0     {c['off_tmem_0']}
#define OFF_TMEM_1     {c['off_tmem_1']}
#define OFF_TMA_MBAR   {c['off_tma_mbar']}
#define OFF_MMA_MBAR   {c['off_mma_mbar']}
#define SMEM_BYTES     {c['smem_used']}
#define TMEM_COLS      {c['tmem_cols']}
#define IDESC          {c['idesc_hex']}U
#define SBO            {c['sbo']}
#define TMA_BYTES      {c['tma_bytes']}
#define MMA_K          {c['mma_k']}
#define MMA_PER_KI     {c['mma_per_ki']}""")
    w()
    w(f"""#define CUDA_CHECK(x) do {{ \\
    cudaError_t e_ = (x); \\
    if (e_ != cudaSuccess) {{ \\
        fprintf(stderr, "CUDA %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \\
        exit(1); \\
    }} \\
}} while(0)""")
    w()
    w(f"""#define CU_CHECK(x) do {{ \\
    CUresult r_ = (x); \\
    if (r_ != CUDA_SUCCESS) {{ \\
        const char* s_; cuGetErrorString(r_, &s_); \\
        fprintf(stderr, "CU %s:%d: %s\\n", __FILE__, __LINE__, s_); \\
        exit(1); \\
    }} \\
}} while(0)""")
    w()

    # ────────────────────────────────────────────────────────────
    # Device helpers
    # ────────────────────────────────────────────────────────────
    w(f"""// ── Device helpers ──────────────────────────────────────────

static __device__ __forceinline__
uint32_t smem_to_uint(const void* p) {{
    uint32_t r;
    asm volatile("{{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }}"
        : "=r"(r) : "l"(p));
    return r;
}}""")
    w()
    w(f"""static __device__ __forceinline__
uint64_t make_smem_desc(uint32_t addr, uint32_t sbo) {{
    uint64_t d = 0;
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= (uint64_t)((sbo  & 0x3FFFF) >> 4) << 32;
    d |= (1ULL << 46);
    d |= {c['swizzle_desc_bits']};  // {c['swizzle_name']}
    return d;
}}""")
    w()
    w(f"""static __device__ __forceinline__
void mbar_init(uint32_t addr, uint32_t count) {{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(addr), "r"(count));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}}""")
    w()
    w(f"""static __device__ __forceinline__
void mbar_wait(uint32_t addr, uint32_t phase) {{
    uint32_t done;
    do {{
        asm volatile(
            "{{\\n\\t"
            ".reg .pred p;\\n\\t"
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2, 0x989680;\\n\\t"
            "selp.b32 %0, 1, 0, p;\\n\\t"
            "}}"
            : "=r"(done) : "r"(addr), "r"(phase));
    }} while (!done);
}}""")
    w()
    w(f"""static __device__ __forceinline__
void mbar_arrive_expect_tx(uint32_t addr, uint32_t tx_count) {{
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(addr), "r"(tx_count) : "memory");
}}""")
    w()
    w(f"""static __device__ __forceinline__
void tma_load_2d(uint32_t smem_dst, const void* tma_desc,
                  int32_t c0, int32_t c1, uint32_t mbar) {{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {{%2, %3}}], [%4];"
        :: "r"(smem_dst), "l"(tma_desc), "r"(c0), "r"(c1), "r"(mbar)
        : "memory");
}}""")
    w()

    # ────────────────────────────────────────────────────────────
    # Kernel
    # ────────────────────────────────────────────────────────────
    w(f"""// ═════════════════════════════════════════════════════════════
// Patch embed GEMM — warp-specialized tcgen05 (cta_group::{CG})
// ═════════════════════════════════════════════════════════════

__global__ void __launch_bounds__(THREADS, 1)
patch_embed_gemm(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    const float*   __restrict__ bias,
    const float*   __restrict__ pos_embed,
    float*         __restrict__ C
) {{""")
    if c["use_coop"]:
        w(f"""    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();""")
    w()
    w(f"""    extern __shared__ __align__(128) char smem[];
    const int sm_id = blockIdx.x;
    const int tid   = threadIdx.x;
    const int warp  = tid / 32;
    const int lane  = tid % 32;""")
    w()
    w(f"""    // Per-stage SMEM offsets (codegen constants)
    static constexpr int off_a[N_STAGES] = {{{a_offs}}};
    static constexpr int off_b[N_STAGES] = {{{b_offs}}};""")
    w()

    # ── TMEM allocation (cta_group::CG) ──
    w(f"""    // ── TMEM allocation: 2 buffers (cta_group::{CG}) ──
    // Warp 0 (WG0 leader) allocates both; all others relinquish twice
    if (warp == 0) {{
        asm volatile("tcgen05.alloc.cta_group::{CG}.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(smem_to_uint(smem + OFF_TMEM_0)), "r"(TMEM_COLS));
        asm volatile("tcgen05.alloc.cta_group::{CG}.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(smem_to_uint(smem + OFF_TMEM_1)), "r"(TMEM_COLS));
    }} else {{
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::{CG}.sync.aligned;");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::{CG}.sync.aligned;");
    }}
    __syncthreads();
    const uint32_t tmem_base0 = *(volatile uint32_t*)(smem + OFF_TMEM_0);
    const uint32_t tmem_base1 = *(volatile uint32_t*)(smem + OFF_TMEM_1);
    const uint32_t tmem_base[2] = {{tmem_base0, tmem_base1}};""")
    w()

    # ── Mbarrier init ──
    w(f"""    // ── Mbarrier init ──
    if (tid == 0) {{
        for (int s = 0; s < N_STAGES; s++) {{
            mbar_init(smem_to_uint(smem + OFF_TMA_MBAR + s * 8), 1);
            mbar_init(smem_to_uint(smem + OFF_MMA_MBAR + s * 8), 1);
        }}
    }}
    __syncthreads();""")
    w()

    # Precompute mbarrier addresses
    w(f"""    uint32_t tma_mbar[N_STAGES], mma_mbar[N_STAGES];
    uint32_t smem_a[N_STAGES], smem_b[N_STAGES];
    for (int s = 0; s < N_STAGES; s++) {{
        tma_mbar[s] = smem_to_uint(smem + OFF_TMA_MBAR + s * 8);
        mma_mbar[s] = smem_to_uint(smem + OFF_MMA_MBAR + s * 8);
        smem_a[s]   = smem_to_uint(smem + off_a[s]);
        smem_b[s]   = smem_to_uint(smem + off_b[s]);
    }}""")
    w()

    # ── Tile assignment ──
    w(f"""    const int tile_start = (int)((long long)sm_id * TOTAL_TILES / SM_COUNT);
    const int tile_end   = (int)((long long)(sm_id + 1) * TOTAL_TILES / SM_COUNT);""")
    w()

    # ────────────────────────────────────────────────────────────
    # Tile loop (TMEM double-buffered, overlapped epilogue)
    # ────────────────────────────────────────────────────────────
    # Precompute MMA mask (Python-level constants)
    n_mask = CG * 4
    mask_refs = ", ".join(f"%{5+i}" for i in range(n_mask))
    mask_args = ", ".join('"r"(0)' for _ in range(n_mask))

    w(f"""    int tma_phase[N_STAGES] = {{0}};
    int mma_phase[N_STAGES] = {{0}};
    int first_tile = 1;

    for (int tile_idx = tile_start; tile_idx < tile_end; tile_idx++) {{
        const int buf = tile_idx & 1;
        const int tm = tile_idx / TILES_N;""")
    if c["snake"]:
        w(f"""        int tn = tile_idx % TILES_N;
        if (tm & 1) tn = TILES_N - 1 - tn;  // snake: reverse odd M-rows""")
    else:
        w(f"        const int tn = tile_idx % TILES_N;")
    w(f"""        const int m_start = tm * TM;
        const int n_start = tn * TN;""")
    w()

    # ═══════════════════════════════════════════════════════════
    # K-loop (warps 0-1) + overlapped epilogue (warps 2-3)
    # ═══════════════════════════════════════════════════════════
    w(f"""        // ═══ K-LOOP (W0-1) + OVERLAPPED EPILOGUE (W2-3) ═══
        if (warp == 0) {{
            // ── LOAD WARP (W0): TMA async bulk copies ──
            if (lane == 0) {{
                for (int ki = 0; ki < K_ITERS; ki++) {{
                    const int s = ki % N_STAGES;
                    const int k_start = ki * TK;

                    if (!(first_tile && ki < N_STAGES)) {{
                        mbar_wait(mma_mbar[s], mma_phase[s]);
                        mma_phase[s] ^= 1;
                    }}

                    mbar_arrive_expect_tx(tma_mbar[s], TMA_BYTES);
                    tma_load_2d(smem_a[s], &tma_a, k_start, m_start, tma_mbar[s]);
                    tma_load_2d(smem_b[s], &tma_b, k_start, n_start, tma_mbar[s]);
                }}
            }}
        }} else if (warp == 1) {{
            // ── MMA WARP (W1): tcgen05.mma.cta_group::{CG} → tmem_base[buf] ──
            if (lane == 0) {{
                for (int ki = 0; ki < K_ITERS; ki++) {{
                    const int s = ki % N_STAGES;

                    mbar_wait(tma_mbar[s], tma_phase[s]);
                    tma_phase[s] ^= 1;

                    for (int sub = 0; sub < MMA_PER_KI; sub++) {{
                        uint64_t desc_a = make_smem_desc(smem_a[s] + sub * MMA_K, SBO);
                        uint64_t desc_b = make_smem_desc(smem_b[s] + sub * MMA_K, SBO);
                        uint32_t accumulate = (ki == 0 && sub == 0) ? 0 : 1;

                        // MMA: cta_group::{CG}, {n_mask}-reg enable mask
                        asm volatile(
                            "{{\\n\\t"
                            ".reg .pred p;\\n\\t"
                            "setp.ne.b32 p, %4, 0;\\n\\t"
                            "tcgen05.mma.cta_group::{CG}.kind::f8f6f4 "
                            "[%0], %1, %2, %3, {{{mask_refs}}}, p;\\n\\t"
                            "}}"
                            :
                            : "r"(tmem_base[buf]), "l"(desc_a), "l"(desc_b), "r"(IDESC),
                              "r"(accumulate),
                              {mask_args});
                    }}

                    asm volatile(
                        "tcgen05.commit.cta_group::{CG}.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                        :: "r"(mma_mbar[s]) : "memory");
                }}
            }}
        }} else {{
            // ── OVERLAPPED EPILOGUE (W2-3): previous tile from tmem_base[buf^1] ──
            if (!first_tile) {{
                const int prev_idx = tile_idx - 1;
                const int ptm = prev_idx / TILES_N;""")
    if c["snake"]:
        w(f"""                int ptn = prev_idx % TILES_N;
                if (ptm & 1) ptn = TILES_N - 1 - ptn;  // snake""")
    else:
        w(f"                const int ptn = prev_idx % TILES_N;")
    w(f"""                const int prev_m = ptm * TM;
                const int prev_n = ptn * TN;
                const uint32_t prev_tmem = tmem_base[buf ^ 1];

                for (int pass = 0; pass < 2; pass++) {{
                    const int ew = (warp - 2) + pass * 2;  // 0,1,2,3
                    for (int nc = 0; nc < TN; nc += 8) {{
                        float v0, v1, v2, v3, v4, v5, v6, v7;
                        int addr = prev_tmem + (ew * 32 << 16) + nc;

                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                            "{{%0, %1, %2, %3, %4, %5, %6, %7}}, [%8];"
                            : "=f"(v0), "=f"(v1), "=f"(v2), "=f"(v3),
                              "=f"(v4), "=f"(v5), "=f"(v6), "=f"(v7)
                            : "r"(addr));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

                        int gm = prev_m + ew * 32 + lane;
                        int gn = prev_n + nc;
                        if (gm < M_TOTAL) {{
                            int pos_row = gm % SEQ_LEN;
                            float* out  = C + (long long)gm * N_DIM;
                            const float* b = bias;
                            const float* p = pos_embed + (long long)pos_row * N_DIM;""")
    for j, vn in enumerate(["v0","v1","v2","v3","v4","v5","v6","v7"]):
        w(f"                            if (gn + {j} < N_DIM) out[gn+{j}] = {vn} + b[gn+{j}] + p[gn+{j}];")
    w(f"""                        }}
                    }}
                }}
            }}
        }}""")
    w()

    # ═══════════════════════════════════════════════════════════
    # Fence + sync at bottom of each tile iteration
    # ═══════════════════════════════════════════════════════════
    w(f"""        // ── Fence + sync: drain MMA writes + block until epilogue done ──
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        __syncthreads();
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        first_tile = 0;
    }}  // tile loop""")
    w()

    # ═══════════════════════════════════════════════════════════
    # Drain epilogue: all 4 warps, last tile
    # ═══════════════════════════════════════════════════════════
    w(f"""    // ── Drain epilogue: all {c['n_warps']} warps, last tile ──
    if (tile_start < tile_end) {{
        const int last_idx = tile_end - 1;
        const int last_buf = last_idx & 1;
        const int ltm = last_idx / TILES_N;""")
    if c["snake"]:
        w(f"""        int ltn = last_idx % TILES_N;
        if (ltm & 1) ltn = TILES_N - 1 - ltn;  // snake""")
    else:
        w(f"        const int ltn = last_idx % TILES_N;")
    w(f"""        const int last_m = ltm * TM;
        const int last_n = ltn * TN;
        const uint32_t drain_tmem = tmem_base[last_buf];

        for (int nc = 0; nc < TN; nc += 8) {{
            float v0, v1, v2, v3, v4, v5, v6, v7;
            int addr = drain_tmem + (warp * 32 << 16) + nc;

            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                "{{%0, %1, %2, %3, %4, %5, %6, %7}}, [%8];"
                : "=f"(v0), "=f"(v1), "=f"(v2), "=f"(v3),
                  "=f"(v4), "=f"(v5), "=f"(v6), "=f"(v7)
                : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

            int gm = last_m + warp * 32 + lane;
            int gn = last_n + nc;
            if (gm < M_TOTAL) {{
                int pos_row = gm % SEQ_LEN;
                float* out  = C + (long long)gm * N_DIM;
                const float* b = bias;
                const float* p = pos_embed + (long long)pos_row * N_DIM;""")
    for j, vn in enumerate(["v0","v1","v2","v3","v4","v5","v6","v7"]):
        w(f"                if (gn + {j} < N_DIM) out[gn+{j}] = {vn} + b[gn+{j}] + p[gn+{j}];")
    w(f"""            }}
        }}
    }}""")
    w()
    w(f"    __syncthreads();  // all warps done before dealloc")
    w()

    # ── Grid sync + TMEM dealloc ──
    if c["use_coop"]:
        w(f"    grid.sync();")
    w()
    w(f"""    // ── TMEM dealloc: 2 buffers (warp 0, cta_group::{CG}) ──
    if (warp == 0) {{
        asm volatile("tcgen05.dealloc.cta_group::{CG}.sync.aligned.b32 %0, %1;"
            :: "r"(tmem_base[0]), "r"(TMEM_COLS));
        asm volatile("tcgen05.dealloc.cta_group::{CG}.sync.aligned.b32 %0, %1;"
            :: "r"(tmem_base[1]), "r"(TMEM_COLS));
    }}
}}""")
    w()

    # ────────────────────────────────────────────────────────────
    # Host main
    # ────────────────────────────────────────────────────────────
    w(f"""// ═════════════════════════════════════════════════════════════
// Host
// ═════════════════════════════════════════════════════════════

int main() {{
    printf("SigLIP2 patch embed GEMM — tcgen05 cta_group::{CG} ({c["n_warps"]} warps)\\n");
    printf("  GEMM: [%d,%d] x [%d,%d]^T  {NS}-stage pipeline  idesc: 0x%08X\\n",
           M_TOTAL, K_DIM, N_DIM, K_DIM, IDESC);

    uint8_t *d_A, *d_B;
    float *d_bias, *d_pos, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A,    (size_t)M_TOTAL * K_DIM));
    CUDA_CHECK(cudaMalloc(&d_B,    (size_t)N_DIM   * K_DIM));
    CUDA_CHECK(cudaMalloc(&d_bias,  (size_t)N_DIM  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos,   (size_t)SEQ_LEN * N_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C,    (size_t)M_TOTAL * N_DIM * sizeof(float)));

    curandGenerator_t rng;
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(rng, 42);
    curandGenerate(rng, (unsigned int*)d_A, ((size_t)M_TOTAL * K_DIM + 3) / 4);
    curandGenerate(rng, (unsigned int*)d_B, ((size_t)N_DIM * K_DIM + 3) / 4);
    curandGenerateUniform(rng, d_bias, N_DIM);
    curandGenerateUniform(rng, d_pos,  SEQ_LEN * N_DIM);
    curandDestroyGenerator(rng);

    CUtensorMap h_tma_a, h_tma_b;

    {{
        uint64_t dims[2]    = {{K_DIM, M_TOTAL}};
        uint64_t strides[1] = {{(uint64_t)K_DIM}};
        uint32_t box[2]     = {{TK, TM}};
        uint32_t estrides[2]= {{1, 1}};
        CU_CHECK(cuTensorMapEncodeTiled(&h_tma_a,
            CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)d_A,
            dims, strides, box, estrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            {c['swizzle_cu_enum']},
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }}

    {{
        uint64_t dims[2]    = {{K_DIM, N_DIM}};
        uint64_t strides[1] = {{(uint64_t)K_DIM}};
        uint32_t box[2]     = {{TK, TN}};
        uint32_t estrides[2]= {{1, 1}};
        CU_CHECK(cuTensorMapEncodeTiled(&h_tma_b,
            CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)d_B,
            dims, strides, box, estrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            {c['swizzle_cu_enum']},
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }}

    CUDA_CHECK(cudaFuncSetAttribute(patch_embed_gemm,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));""")
    w()
    if c["use_coop"]:
        w(f"""    void* args[] = {{ &h_tma_a, &h_tma_b, &d_bias, &d_pos, &d_C }};
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)patch_embed_gemm,
        dim3(SM_COUNT), dim3(THREADS), args, SMEM_BYTES));""")
    else:
        w(f"    patch_embed_gemm<<<SM_COUNT, THREADS, SMEM_BYTES>>>(h_tma_a, h_tma_b, d_bias, d_pos, d_C);")
    w()
    w(f"""    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Kernel completed.\\n");

    float* h_C = (float*)malloc((size_t)M_TOTAL * N_DIM * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, (size_t)M_TOTAL * N_DIM * sizeof(float), cudaMemcpyDeviceToHost));

    double cksum = 0;
    for (int i = 0; i < 1024 && i < M_TOTAL * N_DIM; i++) cksum += h_C[i];
    printf("Checksum (first 1024): %f\\n", cksum);
    printf("C[0,0..3] = %f %f %f %f\\n", h_C[0], h_C[1], h_C[2], h_C[3]);

    free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_bias); cudaFree(d_pos); cudaFree(d_C);
    return 0;
}}""")

    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="SigLIP2 warp-specialized patch embed megakernel codegen")
    p.add_argument("--target", choices=TARGETS, default="b200")
    p.add_argument("--imgs-per-sm", type=int, default=4)
    p.add_argument("--tile-m", type=int, default=128)
    p.add_argument("--tile-n", type=int, default=128)
    p.add_argument("--tile-k", type=int, default=32)
    p.add_argument("--pipeline-stages", type=int, default=3)
    p.add_argument("--no-coop", action="store_true",
                   help="Disable cooperative launch and grid sync")
    p.add_argument("--no-snake", action="store_true",
                   help="Disable snake (zigzag) tile traversal")
    p.add_argument("--cluster-size", type=int, default=None,
                   help="Override cluster size (1 = no clustering)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-o", default="megakernel.cu")
    args = p.parse_args()

    c = compute(args)
    print_analysis(c)

    if args.dry_run:
        return

    code = gen_code(c)
    with open(args.o, "w") as f:
        f.write(code)
    print(f"Wrote {args.o}  ({len(code):,} chars, {code.count(chr(10))+1} lines)")


if __name__ == "__main__":
    main()
