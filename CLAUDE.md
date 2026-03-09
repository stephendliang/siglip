# SigLIP2 Vision Encoder — Persistent GEMM Kernels

Hand-tuned Blackwell (SM100a) persistent kernels for `google/siglip2-base-patch16-224`.
FP8 (E4M3) precision, tcgen05 WGMMA, TMA, `cta_group::2` with 2-CTA clusters. Cross-compiled on CPU VPS, runs on B200.

## Current state

**0.519 ms / 2108 TFLOPS** fused (GEMM + bias + pos_embed) with `MBAR_EARLY=1 STAGGER_CYCLES=160` — **38% faster** than cuBLAS end-to-end (0.835 ms = best GEMM + unfused pos_embed). Defaults produce 0.524 ms / 2090 TFLOPS.

The kernel's value is **fusion**: the overlapped epilogue eliminates the 0.470 ms unfused pos_embed overhead entirely.

cuBLAS pure GEMM is faster: 0.365 ms / 3001 TFLOPS (per-tensor FP8, best-of-128 heuristics, 256MB workspace).
Our effective TFLOPS (2090-2108) counts fused epilogue time in the denominator — not a fair GEMM-only comparison.

GEMM: `[928256, 768] x [768, 768]^T` with fused bias + positional embedding add, BF16 output.
Batch = 4736 images x 196 patches = 928256 rows. Square weight matrix (768x768).

The kernel is correct (non-uniform validation, checksum validated) and stable.

## Current bottlenecks (post-F40)

The kernel is **epilogue-bound** in a balanced producer-consumer equilibrium. F38 unified the epilogue (all 256 cols through SWIZZLE_128B + TMA stores, eliminating the legacy Phase 2A manual LDS+STG path). F40 recovered the stall-window utilization by interleaving TMA stores during TMEM readback.

**Key facts:**
- Phase 1 TMEM readback = binding constraint. Interleaved TMA stores (strategy 2, half-batch) fill TMEM stall windows.
- K-loop: 4,059 cycles. Precomputed descriptors, unroll controlled by `K_LOOP_UNROLL` (default `N_STAGES`).
- TMA multicast not applicable (B is N-split across CTAs).
- ~161-254 regs/thread (varies with `K_LOOP_UNROLL`, `W0_LOOP_UNROLL`, `SUB_MMA_UNROLL`, `CVT_ADD_FUSED`, `PHASE1_UNROLL`, `MBAR_EARLY`, `PREFETCH_BEFORE_STORE`; defaults: PE=205, FC1=~242, FC2=254), 0 spills. Limits occupancy to 1 CTA/SM.
- Timing build uses ~245 regs (distorts cycles vs production). Wall clock is ground truth.

Run `python3 tools/analyze_timing.py data/clock64_timing.txt` for full equilibrium analysis and what-if projections.
Run `python3 tools/analyze_source_counters.py data/source_counters_raw.csv` for per-instruction stall breakdown.
See `docs/EXPERIMENTS.md` for experiments (F1-F40) with hypotheses, results, and analysis. See `docs/FUTURE_PROPOSALS.md` for optimization roadmap.

## Kernel structure

Warp-specialized, 6 warps (192 threads), `cta_group::2`, `__cluster_dims__(2,1,1)`:

- **W0**: TMA async bulk loads (A + B tiles, both CTAs load independently)
- **W1**: TMEM alloc (512 cols, single alloc for double buffering) + `tcgen05.mma.cta_group::2` accumulation into TMEM (CTA0 lane-0 only, multicast commit to both CTAs)
- **W2-W5**: Overlapped epilogue (4 warps) — each warp independently polls mainloop mbarrier, then runs unified SMEM-staged store: Phase 1 (all 256 cols through 4 SWIZZLE_128B regions: x32 `tcgen05.ld` → epilogue op → CVT → `st.shared`, with **interleaved TMA stores** every 2 regions hiding in TMEM stall windows), Phase 2 (`cp.async.bulk.commit_group` only — all TMA stores already issued inline), **mbar_arrive** signals TMEM free for W1. The epilogue op is kernel-specific: bias+pos_embed add (patch_embed), bias+GELU (fc1_gelu), or bias+residual (fc2). Side-data LDG placement controlled by `PRELOAD_MODE` (0=after TMEM_WAIT, 1=partial preload before, 2=full preload for BIAS_ADD only). Next-tile TMEM prefetch timing controlled by `PREFETCH_BEFORE_STORE` (0=after TMA stores, 1=before).

TM=128 rows / 32 rows per warp = 4 row groups. With 4 epi warps (default), each warp owns a full row group (256 cols, `is_split=0`). With 5 epi warps, warp 4 shares row_group 0 via `ew % 4`, creating split warps (`is_split=1`) that each handle 128 cols (2 regions). `epilogue_store` is templated on `<NC_START, NC_END>` so the compiler sees constant loop bounds and `N_REGIONS = (NC_END - NC_START) / 64` is constexpr; unroll depth controlled by `PHASE1_UNROLL` (default 2).

The overlapped epilogue for tile N-1 runs concurrently with the K-loop for tile N (double-buffered TMEM, mbarrier-protected). After the tile loop, W2-5 run a drain epilogue for the last tile.

### Tile config
- Tile: 256x256x128 (M=2x128 from cta_group::2, N=256, K=128)
- TMEM: single alloc of 512 cols (TN*2), double-buffered via column offset (buf*TN)
- SMEM: 4-stage pipeline (131 KB) + epilogue staging (4 warps x 16,384 = 64 KB, SWIZZLE_128B) = ~197 KB total of 228 KB
- Tiles: 3626 M-tiles x 3 N-tiles = 10,878 total, snake ordering
- ~161-254 registers/thread, 0 spills (varies with unroll params and epilogue type; PE defaults=205, FC2=254)
- `NUM_EPI_WARPS` controls epilogue warp count (currently 4); `THREADS` derived as `32*(2+NUM_EPI_WARPS)`

## B200 sessions

All execution, profiling, and benchmarking requires B200 hardware. Cross-compilation works anywhere. Never debug interactively on the server — run the automated session, download data, analyze locally.

```bash
./tools/b200_session.sh                              # full session (~45 min), always run full
python3 tools/analyze_session.py data/session_*/     # structured summary (uses latest session dir)
```

Phases: machine snapshot → `compare_all.py --runs 20 --grid-search` (ANOVA) → ncu profiling → SASS dumps (our kernels + CUTLASS). All outputs to timestamped `data/session_*/`. See **`TASKS.md`** for manual follow-ups.

## Development workflow

All kernels share `kernel_common.cuh` (pipeline, TMEM loads, TMA helpers, mbarrier ops, tuning parameters) and `kernel_body.cuh` (epilogue_store template, persistent_gemm kernel template). Each `.cu` file `#define N_DIM` and `K_DIM`, defines its epilogue macro (e.g., `CVT_ADD_STS_V4`), then includes both headers — tile counts and K-iter constants are derived automatically.

```
edit kernel_common.cuh / kernel_body.cuh / patch_embed.cu / fc1_gelu.cu / fc2.cu -> make -> ./patch_embed (or ./fc1_gelu, ./fc2)
```

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
  sass_analysis.py      # SASS scheduling analyzer (decodes control words, dep graphs, slack)
  grid_search.py        # Compile-time parameter sweep (tiered search, CSV output)
  analyze_timing.py     # clock64 timing → equilibrium analysis
  analyze_source_counters.py  # ncu SourceCounters CSV → stall breakdown
  compare_all.py        # Unified benchmark: CUTLASS vs ours + ANOVA
  analyze_sweep.py      # Grid search analysis: eta-squared, balanced subsets, RF importance
  remote.py             # Remote B200 provisioning + sweep runner
  ncu_diff.py           # ncu CSV diff tool
  compare_sass.py       # SASS dump diff tool
  cutlass_sweep.sh      # Build + run all CUTLASS bench variants (max/standard)
  b200_session.sh       # Automated B200 rental session (builds, benchmarks, profiles, SASS)
  analyze_session.py    # Summarize session output for Claude Code interpretation

bench/                  # Benchmark & calibration kernels
  cutlass_bench.cu      # CUTLASS tile/policy sweep (compile-time N/K/epilogue via -D flags)
  siglip_periodic_add.hpp # Custom EVT visitor (Sm100PeriodicAddNode)
  cublas_bench.cu       # cuBLAS baseline benchmark (compile-time N/K/epilogue via -D flags)
  calibration.cu        # SASS latency microbenchmarks (K1-K26)
  common.h              # Shared PTX helpers (mbarrier, TMA, tcgen05)
  profiler.h            # globaltimer-based kernel profiler
  CALIBRATION_LACKING   # Calibration status audit (measured/pending/unfixable)

data/                   # Sweep results, ncu profiles, session outputs (data/session_*/)
docs/                   # Experiments (F1-F40), proposals, grid search, SASS notes, ncu analysis
```

## Build and run

```bash
make                    # compile patch_embed.cu -> patch_embed
./patch_embed         # run on B200, prints timing + TFLOPS + checksum
make fc1-gelu           # compile fc1_gelu.cu -> fc1-gelu
./fc1-gelu              # run FC1+GELU kernel
make fc2                # compile fc2.cu -> fc2
./fc2                   # run FC2 kernel
make timing && ./patch_embed_timing | tee data/clock64_timing.txt | python3 tools/analyze_timing.py
ncu --set source --csv ./patch_embed > data/source_counters_raw.csv && python3 tools/analyze_source_counters.py data/source_counters_raw.csv

# CUTLASS tile/policy sweep (compile-time N/K/epilogue via -D flags, one binary per epilogue)
make cutlass-bench          # patch embed: N=768 K=768 PERIODIC_ADD (default)
make cutlass-bench-fc1      # FC1: N=3072 K=768 GELU_BIAS
make cutlass-bench-fc2      # FC2: N=768 K=3072 BIAS_RESIDUAL
./cutlass-bench [imgs_per_sm]  # default 32 → M=928256

# CUTLASS extended sweep (more tile/cluster configs)
make cutlass-bench-max          # extended patch embed
make cutlass-bench-fc1-max      # extended FC1
make cutlass-bench-fc2-max      # extended FC2

# Full CUTLASS sweep across all three layers (builds + runs)
./tools/cutlass_sweep.sh              # max mode, 32 imgs/SM
./tools/cutlass_sweep.sh 1            # quick test (1 img/SM = 148*196 rows)
./tools/cutlass_sweep.sh 32 standard  # standard tile list only

# Parameter grid search (any kernel)
python3 tools/grid_search.py --tier all                    # sequential 1→2→3→4, pinning winners
python3 tools/grid_search.py --full-cross                  # all parameters crossed
python3 tools/grid_search.py --kernel fc2 --tier all       # FC2 sweep
python3 tools/grid_search.py --kernel fc1_gelu --tier 3    # FC1 tier 3 only
python3 tools/grid_search.py --only K_LOOP_UNROLL W0_LOOP_UNROLL SUB_MMA_UNROLL  # sweep specific params

# Grid search analysis (eta-squared, balanced subsets, random forest)
python3 tools/analyze_sweep.py data/sweep_results_run4.csv             # balanced PE data
python3 tools/analyze_sweep.py data/session_*/sweep_*.csv              # all layers from session

# SASS scheduling analysis
cuobjdump --dump-sass patch_embed > sass_dump.txt
python3 tools/sass_analysis.py sass_dump.txt                          # annotated listing
python3 tools/sass_analysis.py sass_dump.txt --section 0x1300 0x1a70  # address range (e.g., epilogue)
python3 tools/sass_analysis.py sass_dump.txt --deps                   # dependency + slack analysis
python3 tools/sass_analysis.py --cubin patch_embed                  # runs cuobjdump internally

# Calibration: verify SASS control word decoder on B200
make calibration          # compile bench/calibration.cu
./calibration > cal_output.txt
cuobjdump --dump-sass calibration > cal_sass.txt
python3 tools/sass_analysis.py cal_sass.txt --calibrate-compare                          # SASS-only
python3 tools/sass_analysis.py cal_sass.txt --calibrate-compare --runtime cal_output.txt # compare vs runtime

# Unified comparison (CUTLASS vs ours, ANOVA statistical analysis)
make compare                                              # all layers, 10 runs, CSV output
python3 tools/compare_all.py --runs 20 --csv data/compare.csv   # more runs for tighter CI
python3 tools/compare_all.py --layer patch_embed --runs 5        # single layer, quick
python3 tools/compare_all.py --grid-search                      # run grid search first
```

### Bench details

Both `cutlass_bench.cu` and `cublas_bench.cu` are compile-time parameterized via `-D` flags: `BENCH_N`, `BENCH_K`, `BENCH_EPILOGUE` (`1`=PERIODIC_ADD, `2`=GELU_BIAS, `3`=BIAS_RESIDUAL, `0`=NONE). CUTLASS adds `CUTLASS_EXTENDED_SWEEP=1` for more tile/cluster configs. Configs exceeding SMEM capacity return sentinel `-3.0f`. cuBLAS tests both MXFP8 and per-tensor FP8 with up to 128 algo heuristics.

## Key constraints

- Target: `sm_100a` (B200, 148 SMs)
- `cta_group::2` with `__cluster_dims__(2,1,1)` — 74 clusters of 2 CTAs
- TMEM: 512 cols/SM total, single alloc of TN*2 for double buffering (learned from matmul_v7: two separate allocs deadlock, single alloc works)
- SMEM: 228 KB/SM — current usage ~192 KB (4-stage pipeline + unified SWIZZLE_128B epilogue staging, ~36 KB free)
- All inline PTX — no CUTLASS dependency for the kernels themselves
- Kernels are hand-edited directly; shared infrastructure lives in `kernel_common.cuh`
- Validation: non-uniform B (alternating FP8 rows: 1.5/1.0 → distinct even/odd col accumulators), non-uniform bias/pos_embed, 1024 strided checksum + 32 CPU reference spot checks (valid=1 in @@RESULT)
- **OFF_STAGING must be 1024-byte aligned** for SWIZZLE_128B correctness — TMA swizzle operates on absolute SMEM address bits `addr[6:4] ^= addr[9:7]`; misalignment causes systematic 8-col (16-byte) group swaps in output. Swizzle period = 8 rows x 128 bytes = 1024 bytes.
- **`fence.proxy.async.shared::cta` required** before every TMA store that reads from SMEM written by `st.shared` — bridges sync→async memory proxies. Without it, TMA may read stale data (sporadic corruption).
- ml_phase init must account for odd tile_start (start_buf-dependent)

## Grid search findings

Sweep data analyzed via `python3 tools/analyze_sweep.py` using eta-squared (ANOVA), balanced-subset eta-squared (controls for tiered search confounds), and random forest permutation importance. Tiered search data is heavily imbalanced (pinned params get 90%+ of configs) — only balanced η² and the run4 balanced sweep (145 configs) give trustworthy importance rankings.

**Patch embed** (run4, 145 balanced configs): parameter search exhausted — all top configs within 0.001 ms. Best: `MBAR_EARLY=1 STAGGER_CYCLES=160` → 0.519 ms / 2108 TFLOPS. Defaults → 0.524 ms / 2090 TFLOPS.

**Cross-kernel universal defaults**: `SNAKE_ORDER=1`, `PHASE1_UNROLL=2`, `K_LOOP_UNROLL=4`, `W0_LOOP_UNROLL=0`, `TMEM_LOAD_WIDTH=32`, `PRELOAD_MODE=1`, `PREFETCH_BEFORE_STORE=0`.
**Per-kernel tuning**: `INTERLEAVE_STRATEGY=2` (PE, FC2) vs `=1` (FC1) — N=3072 changes epilogue pattern.
**Catastrophic values**: `PHASE1_UNROLL=4` (+2.4 ms on FC1), `NUM_EPI_WARPS=5` (+6 ms on FC1), `SNAKE_ORDER=0` (+49 us on PE).
**Epilogue scheduling** (tier 4): `PRELOAD_MODE` (0/1/2) × `PREFETCH_BEFORE_STORE` (0/1) = 6 configs. PRELOAD_MODE=2 is BIAS_ADD only (full preload of all 8 uint4 before TMEM_WAIT); GELU/RESIDUAL silently fall back to mode 1. PREFETCH_BEFORE_STORE=1 adds ~14 regs on PE (205→219); FC2 stays at 254 (allocator at ceiling).

