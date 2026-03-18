# SigLIP2 Vision Encoder — Persistent GEMM Kernels

Hand-tuned Blackwell (SM100a) persistent kernels for `google/siglip2-base-patch16-224`.
FP8 (E4M3) precision, tcgen05 WGMMA, TMA, `cta_group::2` with 2-CTA clusters. Cross-compiled on CPU VPS, runs on B200.

## Current state

Three fused GEMM kernels for the vision encoder MLP:

| Kernel | Shape | Epilogue | Best ms | TFLOPS | Regs | vs CUTLASS fused |
|--------|-------|----------|---------|--------|------|------------------|
| **patch_embed** | [928256,768]×[768,768]^T | bias + pos_embed | 0.525 | 2085 | 174-214 | **2% faster** (0.536) |
| **fc1_gelu** | [928256,768]×[768,3072]^T | bias + GELU | 2.267 | 1932 | 244 | **3% faster** (2.323) |
| **fc2** | [928256,3072]×[3072,768]^T | bias + residual | 1.466 | 2988 | 184-217 | **20% slower** (1.225) |

Batch = 4736 images × 196 patches = 928256 rows. BF16 output, FP8 inputs.

The kernels' value is **fusion**: overlapped epilogues eliminate unfused overhead entirely. All three are correct (non-uniform validation, checksum validated) and stable.

**PE is exhausted** — 145 balanced configs all within 0.001ms. No parameter moves the needle.
**FC1** — won via interaction sweeps finding GV=4+IS=0+PH1U=4. Prior best 2.247ms (IS=1+ST=1).
**FC2** — biggest opportunity. 20% gap to CUTLASS. W0 restructured (all threads compute addrs, lane 0 issues TMA), `W0_RES_FULL` added, `DEFERRED_WAIT` added. New params untested on B200.

## Grid search

The primary optimization tool. Per-kernel tiered parameter sweep with top-lock analysis, interaction sweeps, and dynamic branching.

### How it works

1. **Per-kernel tiers**: Each kernel has ordered tiers of params (most impactful first). Sweep tier 1, carry top-k winners as branches into tier 2, etc.
2. **Top-lock analysis**: After each tier, checks if any param is universally locked at the top (single value in all top-5/10/20 results, base rate <70%). Auto-pins into subsequent tier branches.
3. **Dynamic k**: Reduces branching when the gap is clear (>2% → k=1, >0.5% → k≤2). Structural params (`BRANCH_PARAMS`) override this — N_STAGES always branches both values for FC2.
4. **Interaction sweeps**: After all tiers, tests cross-tier param combinations (e.g., epilogue: BATCH_EPILOGUE × IS × PRELOAD_MODE). Skipped only if ALL params in the group are noise (not in any tier).
5. **Inline η²**: Per-param eta-squared printed after each tier/interaction.

### Per-kernel tier ordering

Based on balanced-η² from session_20260315. Params not in any tier are pinned at `KERNEL_BASES` values.

**FC1** (K=768, N=3072, 12 N-tiles):
- Tier 1: `GELU_VARIANT`, `INTERLEAVE_STRATEGY` — IS is dominant (η²=0.533)
- Tier 2: `PHASE1_UNROLL`, `STORE_TIMING`, `PRELOAD_MODE`, `BATCH_EPILOGUE` — PH1U=4 is critical (η²=0.842)
- Tier 3: `EPILOGUE_LOOP`, `STS_WIDTH`, `EPI_SYNC`, `GELU_VECTOR_WIDTH`
- Pinned: N_STAGES=5 (mandatory, 23% slower at 4), KLU=5, SMU=3

**FC2** (K=3072, N=768, 3 N-tiles, 24 K-iterations):
- Tier 1: `N_STAGES`, `K_LOOP_UNROLL`, `TMA_RESIDUAL`, `W0_RES_PREFETCH`, `W0_RES_FULL` — TMAR=1 is key (η²=0.435)
- Tier 2: `INTERLEAVE_STRATEGY`, `PHASE1_UNROLL`, `BIAS_SMEM`, `TMEM_LOAD_WIDTH`
- Tier 3: `BATCH_EPILOGUE`, `STORE_TIMING`, `STS_WIDTH`, `PRELOAD_MODE`, `DEFERRED_WAIT`
- Tier 4: `EPILOGUE_LOOP`, `EPI_SYNC`, `NUM_PASSES_PARAM`
- N_STAGES is a `BRANCH_PARAM` — both 4 and 5 always survive dynamic-k

**PE** (K=768, N=768, 3 N-tiles): Exhausted. Tiers exist but return same 0.525ms.

### Key findings

- **SNAKE_ORDER=1**: Mandatory (pinned). SNAKE=0 is catastrophic (+49μs PE, never wins anywhere).
- **FC1 breakthrough**: GV=4+IS=0+PH1U=4 found via interaction sweeps, not tier search. IS=3 won tier 1 but IS=0 was the real winner.
- **FC2 regs**: 184-217 depending on config (down from 255 after W0 restructure). No longer at allocator ceiling.
- **Top-lock vs η²**: η² catches params that create spread. Top-lock catches params required for peak. Complementary — η² says what to avoid, top-lock says what to pin.
- **Tiered data is confounded**: Raw η², RF, Pearson/Spearman all give wrong rankings. Only balanced-subset η² works.

## Kernel structure

Warp-specialized, 6 warps (192 threads), `cta_group::2`, `__cluster_dims__(2,1,1)`:

- **W0**: TMA async bulk loads (A + B tiles, all threads compute addrs, lane 0 issues TMA). FC2: optionally prefetches (`W0_RES_PREFETCH`) or fully loads (`W0_RES_FULL`) residual via TMA after K-loop.
- **W1**: TMEM alloc (512 cols, single alloc for double buffering) + `tcgen05.mma.cta_group::2` accumulation into TMEM (CTA0 lane-0 only, multicast commit to both CTAs)
- **W2-W5**: Overlapped epilogue (4 warps) — each warp independently polls mainloop mbarrier, then runs unified SMEM-staged store: Phase 1 (all 256 cols through 4 SWIZZLE_128B regions: x32 `tcgen05.ld` → epilogue op → CVT → `st.shared`, with **interleaved TMA stores** hiding in TMEM stall windows), Phase 2 (`cp.async.bulk.commit_group` only — all TMA stores already issued inline), **mbar_arrive** signals TMEM free for W1.

Epilogue ops: bias+pos_embed (PE), bias+GELU (FC1), bias+residual (FC2). FC2's residual loads a full [M,N] matrix — heaviest epilogue, memory-bound. FC1's GELU is compute-bound. PE's add is trivial.

The overlapped epilogue for tile N-1 runs concurrently with the K-loop for tile N (double-buffered TMEM, mbarrier-protected).

### Tile config
- Tile: 256×256×128 (M=2×128 from cta_group::2, N=256, K=128)
- TMEM: single alloc of 512 cols (TN*2), double-buffered via column offset
- SMEM: 4-stage pipeline (131 KB) + epilogue staging (64 KB, SWIZZLE_128B) = ~197 KB of 228 KB
- Tiles: 3626 M-tiles × N_TILES (3 for PE/FC2, 12 for FC1), snake ordering
- K-iterations: 6 (PE/FC1, K=768) or 24 (FC2, K=3072)

## B200 sessions

All execution, profiling, and benchmarking requires B200 hardware. Cross-compilation works anywhere.

```bash
./tools/b200_session.sh                    # grid search + benchmark (no CUTLASS, default)
./tools/b200_session.sh --cutlass          # include CUTLASS build + comparison
python3 tools/analyze_session.py data/session_*/
```

Phases: machine snapshot → grid search + benchmark → ncu profiling → SASS dumps. Autocommits session data to git. All outputs to timestamped `data/session_*/`.

## Development workflow

All kernels share `kernel_common.cuh` (pipeline, TMEM, TMA, mbarriers, tuning params) and `kernel_body.cuh` (epilogue_store template, persistent_gemm template). Each `.cu` file `#define N_DIM` and `K_DIM`, defines its epilogue macro, then includes both headers.

```
edit kernel_common.cuh / kernel_body.cuh / patch_embed.cu / fc1_gelu.cu / fc2.cu -> make -> ./patch_embed (or ./fc1_gelu, ./fc2)
```

### Compile-time tuning params

Key params (controlled via `-D` flags, swept by grid_search.py):

| Param | Values | Effect |
|-------|--------|--------|
| `N_STAGES` | 3,4,5 | Pipeline depth. FC1 needs 5. FC2 tests both 4 and 5. |
| `K_LOOP_UNROLL` | 1,2,4,6,8 | K-loop unroll factor |
| `PHASE1_UNROLL` | 1,2,4 | Epilogue unroll. PH1U=4 critical for FC1, neutral for PE. |
| `INTERLEAVE_STRATEGY` | 0,1,2,3 | TMA store interleaving. IS=0 wins FC1, IS=2 wins PE. |
| `GELU_VARIANT` | 0,4,5 | FC1 only. V4 (batched asm) is best. |
| `TMA_RESIDUAL` | 0,1,2 | FC2 only. TMAR=1 (TMA coalesced) is key. |
| `W0_RES_PREFETCH` | 0,1 | FC2 only. W0 prefetches pass-0 residual after K-loop. Requires TMA_RESIDUAL≥1. |
| `W0_RES_FULL` | 0,1 | FC2 only. W0 loads ALL residual (both passes) with pass handshake. Requires TMA_RESIDUAL≥1, mutually exclusive with W0_RES_PREFETCH. |
| `DEFERRED_WAIT` | 0,1 | FC2 only. Defers `wait_group 0` until after TMEM load + residual mbar_wait. Requires TMA_RESIDUAL≥1. |
| `BIAS_SMEM` | 0,1 | FC1/FC2 only. Load bias vector into SMEM (vs global `__ldg`). |
| `STORE_TIMING` | 0,1 | 0=inline stores, 1=all deferred after Phase 1. |
| `BATCH_EPILOGUE` | 0,1 | 1=separate compute/store phases (FC1/FC2 only). |

See grid_search.py `RANGES` dict for full parameter list and valid values.

## Code style

Names say what, comments say why. No single-line `/**/`. No multi-line `//`.
No decorated block comments (no leading `*` per line). Bare `/*` open, undecorated lines, `*/` close.
No restating code. Mark ownership/lifetime when types don't show it.

## Context efficiency

Don't narrate tool calls ("Let me read the file..."). Just do it.
Don't echo back file contents — the user can see them.
Keep explanations proportional to complexity. Simple changes need one sentence.
Parallelize independent tool calls (e.g., multiple reads/greps) in one message.
Use Grep to locate relevant sections before reading entire large files.
For files over 500 lines, use offset/limit to read only the relevant section.
When using subagents, include output rules: "Final response under 2000 characters. List outcomes, not process."

## Repository structure

```
kernel_common.cuh       # Shared infrastructure (pipeline, TMEM, TMA, mbarriers, tuning params)
kernel_body.cuh         # Shared kernel body (epilogue_store template, persistent_gemm template)
patch_embed.cu          # Patch embed GEMM — [928256,768]×[768,768]^T + bias + pos_embed
fc1_gelu.cu             # FC1+GELU GEMM — [928256,768]×[768,3072]^T + bias + GELU
fc2.cu                  # FC2 GEMM — [928256,3072]×[3072,768]^T + bias + residual
Makefile                # Build rules (sm_100a, nvcc flags)
CLAUDE.md               # This file
TASKS.md                # B200 session playbook — automated benchmarking workflow

tools/                  # Analysis & sweep scripts
  grid_search.py        # Per-kernel tiered parameter sweep (top-lock, interactions, CSV output)
  b200_session.sh       # Automated B200 session (grid search, benchmark, ncu, SASS, autocommit)
  compare_all.py        # Benchmark: CUTLASS vs ours (--no-cutlass default in sessions)
  analyze_session.py    # Summarize session output for Claude Code interpretation
  analyze_sweep.py      # Grid search analysis: eta-squared, balanced subsets, RF importance
  balanced_eta.py       # Standalone balanced-subset eta-squared tool
  sass_analysis.py      # SASS scheduling analyzer (control words, dep graphs, slack)
  analyze_timing.py     # clock64 timing → equilibrium analysis
  analyze_source_counters.py  # ncu SourceCounters CSV → stall breakdown
  simulate_lhs.py       # Bootstrap convergence simulation (no GPU)
  analyze_gelu_variants.py  # Static GELU variant scheduling analysis (no GPU)
  remote.py             # Remote B200 provisioning + sweep runner
  ncu_diff.py           # ncu CSV diff tool
  compare_sass.py       # SASS dump diff tool
  cutlass_sweep.sh      # Build + run all CUTLASS bench variants

bench/                  # Benchmark & calibration kernels
  cutlass_bench.cu      # CUTLASS tile/policy sweep (compile-time N/K/epilogue via -D flags)
  siglip_periodic_add.hpp # Custom EVT visitor (Sm100PeriodicAddNode)
  cublas_bench.cu       # cuBLAS baseline benchmark
  calibration.cu        # SASS latency microbenchmarks (K1-K26)
  common.h              # Shared PTX helpers (mbarrier, TMA, tcgen05)
  profiler.h            # globaltimer-based kernel profiler

data/                   # Sweep results, ncu profiles, session outputs (data/session_*/)
docs/                   # Experiments (F1-F40), proposals, grid search, SASS notes, ncu analysis
```

## Build and run

```bash
make                    # compile patch_embed.cu -> patch_embed
./patch_embed           # run on B200, prints timing + TFLOPS + checksum
make fc1-gelu           # compile fc1_gelu.cu -> fc1-gelu
./fc1-gelu              # run FC1+GELU kernel
make fc2                # compile fc2.cu -> fc2
./fc2                   # run FC2 kernel

# Parameter grid search — per-kernel tiered (default)
python3 tools/grid_search.py                                    # per-kernel tiered (default)
python3 tools/grid_search.py --kernel fc1_gelu                  # FC1 tiers
python3 tools/grid_search.py --kernel fc2                       # FC2 tiers
python3 tools/grid_search.py --tier all --no-interact           # tiers only, skip interactions
python3 tools/grid_search.py --full-cross                       # all parameters crossed
python3 tools/grid_search.py --only K_LOOP_UNROLL TMA_RESIDUAL  # sweep specific params
python3 tools/grid_search.py --interact epilogue --kernel fc2   # single interaction group
python3 tools/grid_search.py --only BATCH_EPILOGUE --base MBAR_EARLY=1  # --base sets baseline

# CUTLASS comparison
make cutlass-bench-max && ./cutlass-bench-max         # PE
make cutlass-bench-fc1-max && ./cutlass-bench-fc1-max # FC1
make cutlass-bench-fc2-max && ./cutlass-bench-fc2-max # FC2
python3 tools/compare_all.py --runs 20 --csv data/compare.csv   # full ANOVA comparison

# Analysis (all run locally, no GPU)
python3 tools/analyze_sweep.py data/session_*/sweep_*.csv       # param importance rankings
python3 tools/balanced_eta.py data/session_*/sweep_fc2.csv      # balanced η² for FC2
python3 tools/sass_analysis.py --cubin patch_embed --deps       # SASS dependency analysis
```

## Key constraints

- Target: `sm_100a` (B200, 148 SMs)
- `cta_group::2` with `__cluster_dims__(2,1,1)` — 74 clusters of 2 CTAs
- TMEM: 512 cols/SM total, single alloc for double buffering (two separate allocs deadlock)
- SMEM: 228 KB/SM — current usage ~192-225 KB depending on N_STAGES
- All inline PTX — no CUTLASS dependency for the kernels themselves
- Validation: non-uniform B, non-uniform bias/pos_embed, 1024 strided checksum + 32 CPU reference spot checks
- **OFF_STAGING must be 1024-byte aligned** for SWIZZLE_128B correctness
- **`fence.proxy.async.shared::cta` required** before every TMA store that reads from SMEM written by `st.shared`
- FC2 regs: 184-217 depending on config (down from 255 after W0 restructure + `cta_rank` uniformity fix).
