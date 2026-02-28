# Grid Search Design

Python sweep script that iterates over kernel parameter configurations, computes constraints (skip invalid combos before compiling), builds, runs, and collects results into a sorted table + CSV.

**Why Python:** Constraint checking (SMEM budget, TMEM cols, tile divisibility, register estimates) is arithmetic that's natural in Python and brittle in bash. The same infrastructure later becomes the objective function for Bayesian optimization when the parameter space grows (FC1/FC2 shapes).

---

## Parameter injection

Wrap existing `#define` values in `#ifndef` guards in `megakernel.cu`:

```c
#ifndef TN
#define TN 256
#endif
```

The sweep script passes `-DTN=128 -DNUM_EPI_WARPS=4 -DN_STAGES=3` etc. on the `nvcc` command line. Manual `make` still works with defaults. No file patching, no sed.

For code-path switches, use `#ifdef` guards in the kernel with a define like `-DUSE_SNAKE_ORDER` or `-DPHASE2_UNROLL=16`. The sweep script only needs to control which `-D` flags are set.

---

## Parameter tiers

Not all parameters are equal. Some are define-only (change a number, recompile). Others require maintaining parallel code paths (`#ifdef` blocks). Others require deep structural rewrites that cannot be toggle-switched at all. The sweep script handles the first two tiers; the third tier is manual experimentation.

### Tier 1: Pure define sweeps (change a number, everything recomputes)

These are the core sweep parameters. Each is a single `#define` value, and the entire SMEM layout, tile counts, loop bounds etc. recompute via the existing `#define` chain.

| Parameter | Values | Notes |
|-----------|--------|-------|
| **TN** | 128, 256 | **NOT 192** — Phase 2 requires `NC_COLS / 32` to be 4 or 8 (V2/V4 store widths). TN=192 → COLS_PER_THREAD=6, no valid vector store. TN=384 → 12, same problem. Only {128, 256} work with the current Phase 2 store logic. **Prerequisite:** several hardcoded constants must be converted to derived expressions before TN can be swept — see "TN sweep prerequisite" below. |
| **N_STAGES** | 3, 4, 5 | 6 tested and rejected (F6: too many regs/SMEM). 2 is too shallow for K_ITERS=6. |
| **NUM_EPI_WARPS** | 4, 5 | **NOT 3** — warp assignment uses `row_group = ew % 4`, so 4 row_groups require at least 4 epilogue warps. With 3 warps, row_group 3 (rows 96-127) is never written — silent data corruption. 6 tested and rejected (F3: TMEM contention). |
| **PHASE2_UNROLL** | 4, 8, 16 | `#pragma unroll N` on the Phase 2 loop. Currently 8. |
| **STAGING_ROW_PAD** | 0, 16, 32 | Alignment padding per staging row. Affects bank conflicts and SMEM usage. |

**Grid size:** 2 × 3 × 2 × 3 × 3 = **108 configs.** After constraint pruning (SMEM > 228 KB, TMEM > 512 cols, invalid COLS_PER_THREAD), probably ~60 valid. At ~5 seconds each = ~5 minutes.

#### TN sweep prerequisite

TN is listed as Tier 1 but currently **cannot be swept** without first converting these hardcoded constants to `#define` expressions:

| Constant | Current | Must become | Why it breaks at TN=128 |
|----------|---------|-------------|-------------------------|
| `TMEM_COLS` | 512 | `TN * 2` | Over-allocates TMEM (256 needed). Wastes but doesn't crash. |
| `TILES_N` | 3 | `N_DIM / TN` | Should be 6. Half the output tiles never processed. |
| `TOTAL_TILES` | 10878 | `TILES_M * TILES_N` | Wrong work distribution across clusters. |
| `STAGE_BYTES` | 32768 | `TK * TM + TK * (TN / 2)` | B tile is 8 KB not 16 KB. Wastes SMEM, descriptor reads garbage. |
| `TMA_BYTES` | 32768 | `TK * TM + TK * (TN / 2)` | **HANG.** mbar_arrive_expect_tx reports 32 KB but only 24 KB arrives. Mbarrier never completes. Infinite spin at line 455. |

These conversions are straightforward `#define` arithmetic — no code logic changes. But they MUST be done before the first TN=128 run, or the kernel hangs on a never-completing mbarrier.

```c
// Convert these hardcoded values to derived expressions:
#define STAGE_BYTES    (TK * TM + TK * (TN / 2))  // was 32768
#define TMEM_COLS      (TN * 2)                     // was 512
#define TILES_N        (N_DIM / TN)                 // was 3
#define TILES_M        (M_TOTAL / (TM * 2))         // was 3626
#define TOTAL_TILES    (TILES_M * TILES_N)           // was 10878
#define TMA_BYTES      (STAGE_BYTES)                 // was 32768
```

### Tier 2: Code-path switches (`#ifdef`-gated variants)

These require maintaining `#ifdef` blocks in the kernel — two or more code paths selected at compile time. More maintenance burden but still automatable.

| Parameter | Values | Notes |
|-----------|--------|-------|
| **SNAKE_ORDER** | on, off | Wrap the snake-reversal line in `#ifdef`. |
| **TMA_L2_PROMOTION** | none, promoted | Change the `CU_TENSOR_MAP_L2_PROMOTION_*` flag in TMA descriptor setup. |
| **ADD_SOURCE** | precomputed_bf16, precomputed_fp32 | FP32 combined skips BF16→FP32 conversion in epilogue (8 fewer CVT per nc iter) but doubles combined tensor size (294 KB → 588 KB). |
| **TMEM_LOAD_WIDTH** | x16, x32 | Both macros already exist. Known perf-neutral (F8) but free to include. |
| **COMBINED_LOAD_VEC** | uint2, uint4 | Requires `#ifdef` in Phase 1 loop body — different load counts, offsets, conversion sequences. |

**Grid multiplier:** × 2 × 2 × 2 × 2 × 2 = ×32. Combined with Tier 1: 150 × 16 = ~2400. Too slow for full cross. Run Tier 2 as an outer sweep on the Tier 1 winner only (~16 configs, ~80 seconds).

### Tier 3: Structural changes (NOT sweepable, manual experiments only)

These require deep code rewrites — different warp roles, synchronization protocols, or instruction sequences. They cannot be `#ifdef`-toggled without maintaining essentially separate kernels.

| Parameter | Why not sweepable |
|-----------|-------------------|
| **TM** (64, 128) | Changes row_group count (TM/32), hardcoded in warp assignment (`ew % 4`), epilogue mbarrier thread counts, and staging layout. Requires rewriting warp distribution logic. |
| **TK** (64, 128) | Changes STAGE_BYTES, SWIZZLE pattern (64B vs 128B), IDESC (MMA instruction descriptor bitfield encoding), MMA_PER_KI, K_ITERS, `make_smem_desc` SBO. The IDESC is a hardware-specific bitfield — changing it requires understanding the tcgen05 encoding. |
| **CTA_GROUP** (1, 2) | Baked into MMA PTX (`tcgen05.mma.cta_group::2`), TMEM alloc/dealloc PTX, multicast commit PTX, W1 control flow (CTA0-only MMA). Switching to cta_group::1 rewrites the entire MMA warp. |
| **CLUSTER_X** (1, 2) | Affects `__cluster_dims__`, TMA mbarrier protocol (both CTAs at CTA0's mbar vs independent), multicast mask (`0x3` vs `0x1`), epilogue mbarrier counts, `cta_rank` computation. |
| **TMEM_DOUBLE_BUFFER** | The overlapped epilogue IS the architecture. Disabling it removes the main performance feature. Not a parameter — it's the design. |
| **EPILOGUE_SPLIT_POLICY** | Current col_split logic is embedded in warp assignment, template instantiation (`<0,TN/2>`, `<TN/2,TN>`, `<0,TN>`), and Phase 2 store width selection. Alternative splits require rewriting the assignment logic. |
| **MAINLOOP_SYNC_MODE** | Independent poll (current, committed) vs broadcast barrier (F7, rejected). The code was restructured for independent polling. Reverting requires restoring the bar.sync path. |
| **COMBINED_PREFETCH_MODE** | register_prefetch = proposal 1a (loop restructuring). smem_prefetch = F5 (rejected). W0_TMA_prefetch = proposal 5 (new mbarrier protocol). Each is a different kernel variant. |
| **ADD_PIPELINE_DISTANCE** | Distance 1 = proposal 1a. Requires restructuring Phase 1 loop body (prologue loads, register renaming, if-guarded next-iteration loads). |
| **MAXRREGCOUNT** | Broken / silently ignored on SM100a. Exclude entirely. |
| **LAUNCH_BOUNDS_MIN_BLOCKS** | Setting to 2 requires regs/thread ≤ ~146 (65536 / 448). At 222 regs, this would cause massive spills. Only viable after major register reduction. Include in sweep but expect 100% prune rate at current code size. |

### Derived parameters (not independent axes)

| Parameter | Derived from |
|-----------|-------------|
| THREADS | `32 * (2 + NUM_EPI_WARPS)` |
| TILES_M, TILES_N, TOTAL_TILES | `M_TOTAL / (TM * 2)`, `N_DIM / TN` |
| K_ITERS | `K_DIM / TK` |
| STAGE_BYTES | `TK * TM` (A) + `TK * (TN / CTA_GROUP)` (B) |
| TMEM_COLS | `TN * 2` (double buffer) |
| SMEM_BYTES | pipeline + mbarriers + staging |
| TMA_BOX_A, TMA_BOX_B | `(TK, TM)`, `(TK, TN / CTA_GROUP)` |
| TMA_SWIZZLE | Determined by TK (64B → SWIZZLE_64B, 128B → SWIZZLE_128B) |
| IDESC | Encodes MMA tile shape, tied to TK/TM/TN |
| SBO | SMEM stride, tied to SWIZZLE and tile dimensions |

The sweep script computes all derived values from the independent parameters. Used for constraint checking, not as sweep axes.

### Addition-specific parameters: assessment

| Parameter | Tier | Notes |
|-----------|------|-------|
| **ADD_SOURCE** | Tier 2 | precomputed_bf16 (current) vs precomputed_fp32. Straightforward `#ifdef`. |
| **ADD_TIMING** | Tier 3 | Moving the add before/after TMEM_WAIT or into Phase 2 requires restructuring the loop body. High-value experiment but not a define toggle. |
| **ADD_PRECISION_PATH** | Tier 2-3 | fp32_accum (current) vs bf16_early. Simple if just changing CVT placement; structural if it changes the FADD sequence. |
| **COMBINED_LOAD_VEC** | Tier 2 | uint2 vs uint4. **Not a pure define** — the Phase 1 loop body (lines 269-304) hardcodes uint4 load patterns, BF16X2_TO_F32 call counts, and stride offsets (+0/+8/+16/+24). uint2 needs a different load/convert sequence with different offsets and iteration counts. Requires `#ifdef`-gated code path. |
| **COMBINED_PREFETCH_MODE** | Tier 3 | Each mode is a different kernel variant (see above). |
| **ADD_PIPELINE_DISTANCE** | Tier 3 | Proposal 1a. Code restructuring, not a define. |
| **POS_INDEX_MODE** | Tier 1-2 | modulo (current `% SEQ_LEN`) vs precomputed pointer. Minor — the modulo is cheap. Low priority. |

---

## Constraint system (pre-compile pruning)

Before invoking `nvcc`, the script computes derived values and checks:

```python
TK, TM = 128, 128  # fixed for current kernel (Tier 3 to change)

def is_valid(tn, n_stages, num_epi_warps, staging_pad):
    threads = 32 * (2 + num_epi_warps)
    if threads > 1024:
        return False, "threads > 1024"

    # Row group coverage: ew % 4 must cover all 4 row_groups
    if num_epi_warps < 4:
        return False, f"NUM_EPI_WARPS={num_epi_warps} < 4: row_group 3 never written"

    # SMEM budget — mirrors exact offset chain from megakernel.cu lines 32-41
    stage_bytes = TK * TM + TK * (tn // 2)   # A tile + B tile per CTA
    off_tmem = n_stages * stage_bytes
    off_tma_mbar = off_tmem + 8
    off_mma_mbar = off_tma_mbar + n_stages * 8
    off_mainloop_mbar = off_mma_mbar + n_stages * 8
    off_epilogue_mbar = off_mainloop_mbar + 16
    off_staging = (off_epilogue_mbar + 16 + 127) & ~127
    staging_row_bytes = tn * 2 + staging_pad
    staging_warp_bytes = 32 * staging_row_bytes
    smem_total = (off_staging + num_epi_warps * staging_warp_bytes + 127) & ~127
    if smem_total > 233472:
        return False, f"SMEM {smem_total} > 228KB"

    # TMEM budget
    tmem_cols = tn * 2
    if tmem_cols > 512:
        return False, f"TMEM {tmem_cols} > 512"

    # Tile divisibility
    if 768 % tn != 0:
        return False, f"N_DIM % TN != 0"

    # Phase 2 store validity (COLS_PER_THREAD must be 4 or 8)
    cpt_full = tn // 32
    if cpt_full not in (4, 8):
        return False, f"COLS_PER_THREAD={cpt_full} invalid"
    if num_epi_warps > 4:
        cpt_split = (tn // 2) // 32
        if cpt_split not in (4, 8):
            return False, f"split COLS_PER_THREAD={cpt_split} invalid"

    return True, "ok"
```

Post-compile, parse `ptxas` output for register count and spills:

```python
# Parse: "ptxas info : Used 222 registers, ..."
# Parse: "ptxas info : 0 bytes spill stores, 0 bytes spill loads"
if regs > 255: skip("regs > 255")
if spills > 0: skip("spills detected")
```

---

## Hang detection and timeout

Barrier-heavy kernels can hang permanently on misconfigured mbarrier expected counts or wrong TMA_BYTES. The sweep script must kill hung processes.

```python
import subprocess, signal

TIMEOUT_SEC = 30  # kernel should complete in < 2 seconds

try:
    result = subprocess.run(
        ["./siglip_vision"],
        capture_output=True, text=True,
        timeout=TIMEOUT_SEC
    )
except subprocess.TimeoutExpired:
    # GPU driver may not release cleanly — SIGKILL the process
    log_result(config, status="HANG", ms=None, tflops=None)
    # May need: subprocess.run(["nvidia-smi", "--gpu-reset"]) in extreme cases
    continue
```

This is especially critical for the TN sweep: if TMA_BYTES doesn't track TN, `mbar_arrive_expect_tx` reports the wrong TX count and `mbar_wait` spins forever.

---

## Kernel output protocol

Add one machine-readable line to `megakernel.cu`:

```c
printf("@@RESULT ms=%.3f tflops=%.2f checksum=%f c0=%.1f\n",
       _ms, 2.0 * M_TOTAL * N_DIM * K_DIM / _ms / 1e9, cksum,
       __bfloat162float(h_C[0]));
```

The `@@` prefix never collides with other output. Existing human-readable printfs stay unchanged.

```python
for line in stdout.splitlines():
    if line.startswith("@@RESULT"):
        result = dict(kv.split("=") for kv in line.split()[1:])
```

---

## Output format

Sorted table printed at the end, plus CSV for archival:

```
TN   EPI  STG  PAD  UNR  VEC   REGS  SPILL  SMEM_KB  MS      TFLOPS  CHECKSUM     STATUS
256  5    4    16   8    uint4 222   0      211      0.630   1739    1769472.0    BEST
256  4    4    16   8    uint4 216   0      131      0.700   1564    1769472.0    ok
128  4    4    16   8    uint4 ???   0      ???      0.681   1609    1769472.0    ok
256  5    5    16   8    uint4 -     -      243      -       -       -            SMEM>228K
...
```

Checksum column catches correctness regressions — any config with wrong checksum is flagged `CHECKSUM_MISMATCH` and excluded from ranking.

---

## Execution strategy

0. **Phase 0: Convert hardcoded constants.** Before any TN sweep, convert `TMEM_COLS`, `TILES_N`, `TOTAL_TILES`, `STAGE_BYTES`, `TMA_BYTES` from hardcoded values to `#define` expressions derived from TN/TK/TM (see "TN sweep prerequisite" above). Verify TN=256 still produces the same binary (regression test).

1. **Phase 1: Tier 1 full sweep** (~60 valid configs, ~5 min). Find the best `(TN, N_STAGES, NUM_EPI_WARPS, PHASE2_UNROLL, STAGING_ROW_PAD)`.

2. **Phase 2: Tier 2 sweep on the Tier 1 winner** (~32 configs, ~3 min). Toggle `SNAKE_ORDER`, `TMA_L2_PROMOTION`, `ADD_SOURCE`, `TMEM_LOAD_WIDTH`, `COMBINED_LOAD_VEC` on the best Tier 1 config.

3. **Phase 3: Manual Tier 3 experiments.** Structural changes (CTA_GROUP, TM, TK, EPILOGUE_SPLIT_POLICY, etc.) require code changes — run the sweep on each structural variant separately.

---

## FC1/FC2 expansion (future)

The same sweep infrastructure works for different GEMM shapes by parameterizing the problem dimensions:

```python
GEMM_SHAPES = {
    "patch_embed": {"M": 928256, "N": 768,  "K": 768,  "epilogue": "bias+pos"},
    "fc1":         {"M": 928256, "N": 3072, "K": 768,  "epilogue": "bias+gelu"},
    "fc2":         {"M": 928256, "N": 768,  "K": 3072, "epilogue": "bias+residual"},
}
```

Additional parameters for non-square GEMMs:

| Parameter | Values | Applies to |
|-----------|--------|------------|
| SPLIT_K | 1, 2, 4 | FC2 (tall-K: K=3072) |
| K_TILE_SCHEDULING | serial, parallel | With SPLIT_K > 1 |
| EPILOGUE_FUSION | bias, bias+gelu, bias+residual | Per-shape |

TN sweep space grows for FC1 (N=3072): valid TN values are {128, 256} (must satisfy TMEM ≤ 512 and COLS_PER_THREAD ∈ {4, 8}). With SPLIT_K: ~50-100 configs per shape. Still under 10 minutes each.

Separate best-config tables per GEMM shape. The optimal `(TN, NUM_EPI_WARPS, N_STAGES)` for patch_embed (square, N=768) will differ from FC1 (wide, N=3072) and FC2 (tall-K, K=3072).

---

## Future: Bayesian optimization

When the parameter space exceeds ~500 valid configs (FC1/FC2 with SPLIT_K and expanded Tier 2), replace the exhaustive `for` loop with an optimizer (e.g., `optuna`). The constraint checker and evaluation function stay identical — only the search strategy changes. The sweep script's `evaluate(config) -> ms` function becomes the objective.

Not needed now. The current grid (150 Tier 1 configs) is small enough for exhaustive search.
