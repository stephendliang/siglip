# SigLIP2 Vision Encoder — Persistent GEMM Kernels

Hand-tuned Blackwell (SM100a) persistent kernels for `google/siglip2-base-patch16-224`.
FP8 (E4M3) precision, tcgen05 WGMMA, TMA, `cta_group::2` with 2-CTA clusters. Cross-compiled on CPU VPS, runs on B200.

## Current state

Three fused GEMM kernels for the vision encoder MLP. **Only FC2 is being actively optimized** — PE and FC1 beat CUTLASS and are done. Do NOT suggest or run grid search / B200 sessions on PE or FC1.

| Kernel | Shape | Epilogue | Best ms | TFLOPS | Regs | vs CUTLASS fused | Status |
|--------|-------|----------|---------|--------|------|------------------|--------|
| **patch_embed** | [928256,768]×[768,768]^T | bias + pos_embed | 0.525 | 2085 | 174-214 | **2% faster** (0.536) | **DONE** |
| **fc1_gelu** | [928256,768]×[768,3072]^T | bias + GELU | 2.267 | 1932 | 244 | **3% faster** (2.323) | **DONE** |
| **fc2** | [928256,3072]×[3072,768]^T | bias + residual | 1.452 | 3016 | 207 | **19% slower** (1.225) | **ACTIVE** |

Batch = 4736 images × 196 patches = 928256 rows. BF16 output, FP8 inputs.

### FC2 gap analysis (session 2026-03-19, updated 2026-03-21)

Best: **1.452ms / 3016 TFLOPS** (NS5, IS1, base config). CUTLASS: 1.225ms. Gap = **227μs (19%)**.

**CUTLASS uses identical tile/cluster config**: 256×256×128, 2×1 cluster, cta_group::2. Same 96 UTCQMMA per tile. K-loop compute is byte-identical — the **entire 19% gap is from Phase 1 epilogue contention**, not compute throughput. CUTLASS achieves **99.6% of the K-loop throughput bound** (1.225ms vs 1.220ms theoretical) — its epilogue is invisible.

Cycle breakdown (NS5.IS1.base, timing build):
- **K-loop: 16705 cycles** — identical MMA work to CUTLASS
- **Epilogue Phase 1: 8430 cycles** — creates SMEM contention that slows K-loop by ~3300 cyc/tile
- Epilogue Phase 2: 63 cycles (trivial)
- epi_wait: 107 cycles (balanced producer-consumer)
- tma0_wait: 721 cycles

Per-tile: our 20,743 cyc vs CUTLASS 17,500 cyc. K-loop alone = 17,426 cyc. The 3,317 cyc gap is Phase 1 overhead — W2-W5 running Phase 1 competes with W0/W1 K-loop for dispatch slots. A faster Phase 1 reduces contention for ALL 147 tiles, not just the last tile.

**Root cause — instruction-level scheduling (corrected by TMA bench 2026-03-21):**

TMA bench proved TMA and LSU have **separate SMEM ports** on SM100a (zero contention). The original "SMEM port contention" hypothesis was wrong. The real problem is 5 structural PTX mistakes:

1. **Monolithic asm blocks**: each `BIAS_RES_CVT_STS_V4` = 12 BF16/CVT + 1 STS in one `asm volatile`. STS shadow fits only 4 BF16 free (section P); 12 overflows at ~+100% cost. **~216 wasted cyc/group.**
2. **16 LDS at ILP=16**: 8 bias + 8 residual LDS back-to-back. ILP=16 regresses to 10 cyc/op (section O); sweet spot is ILP=7 at 3.54 cyc/op. **~104 wasted cyc.**
3. **No LDS↔STS temporal overlap**: all LDS before all STS. Section E shows LDS+STS interleaved = 82.3% faster than sequential.
4. **Bias+residual not pre-combined**: two separate HADD2 rounds. Pre-adding reduces BF16/STS from 12→8.
5. **ptxas clustering**: source-level STS scheduling is a dead end (5 approaches tried, identical SASS).

**Two attack vectors:**
1. **SASS scheduling** (built): CP-SAT optimal scheduler spreads STS at 27-32 cyc intervals, packs ≤4 BF16 per window, clusters LDS at ILP=7. 287-insn: 686 cyc (was 1417, -51.6%). Ready for B200 via fatbin-patch.
2. **Pre-combine bias+residual** (source change): `add.bf16x2(bias, residual)` before TMEM_WAIT reduces BF16/STS from 12 to 8. Makes SASS scheduling more effective.

**SM100a hardware data** (from `bench/tma_bench.cu`, raw: `data/tma2.txt`):
- STS.128 throughput: 27 cyc | LDS.128: 25 cyc @ILP=1, 3.5 cyc @ILP=7
- STS shadow: ≤4 BF16 free, 8=+55%, 15=+161% | LDS+STS overlap: 82.3%
- TMA load: 419 cyc (L2-warm) | TMA store: 197 cyc | TMA↔LSU: independent
- mbarrier arrive: 2 cyc | wait: 47 cyc | fence.proxy.async: 10 cyc
- CUTLASS epilogue source: `third_party/cutlass/.../sm100_epilogue_tma_warpspecialized.hpp`

## Grid search (FC2 only)

Per-kernel tiered parameter sweep with top-lock analysis, interaction sweeps, and dynamic branching. **Only run on FC2** — PE and FC1 are exhausted/won.

### How it works

1. **Per-kernel tiers**: Ordered tiers of params (most impactful first). Sweep tier 1, carry top-k winners as branches into tier 2, etc.
2. **Top-lock analysis**: After each tier, checks if any param is universally locked at the top (single value in all top-5/10/20 results, base rate <70%). Auto-pins into subsequent tier branches.
3. **Dynamic k**: Reduces branching when the gap is clear (>2% → k=1, >0.5% → k≤2). Structural params (`BRANCH_PARAMS`) override this.
4. **Interaction sweeps**: After all tiers, tests cross-tier param combinations. Skipped only if ALL params in the group are noise.
5. **Inline η²**: Per-param eta-squared printed after each tier/interaction.

### FC2 tier ordering (K=3072, N=768, 3 N-tiles, 24 K-iterations)

- Tier 1: `N_STAGES`, `K_LOOP_UNROLL`, `TMA_RESIDUAL`, `W0_RES_PREFETCH`, `W0_RES_FULL`, `BATCH_MMA`
- Tier 2: `INTERLEAVE_STRATEGY`, `PHASE1_UNROLL`, `BIAS_SMEM`, `TMEM_LOAD_WIDTH`
- Tier 3: `BATCH_EPILOGUE`, `STORE_TIMING`, `STS_WIDTH`, `PRELOAD_MODE`, `DEFERRED_WAIT`
- Tier 4: `EPILOGUE_LOOP`, `EPI_SYNC`, `NUM_PASSES_PARAM`
- N_STAGES and BATCH_MMA are `BRANCH_PARAMS` — always branch both values

### FC2 confirmed results (session 2026-03-19)

- **N_STAGES=5 mandatory**: 1.452ms vs 1.594ms (NS4) — 10% gap, never test NS4 again
- **IS1 ≈ IS2**: Noise for FC2 (both within ±0.003ms)
- **BATCH_MMA**: Zero effect — K-loop overhead hidden by pipeline overlap
- **PREFETCH_MBAR, OVERLAP_EPI_WAIT**: Noise
- **W0_RES_FULL**: Catastrophic (+15%), epi_wait 107→5400 cycles
- **W0_RES_PREFETCH**: Neutral
- **EPI_LOAD_WARP**: Runtime hang (mbarrier bug)
- Regs: 207 (NS5 base), down from 255 after W0 restructure

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

### Compile-time tuning params (FC2-relevant)

Key params (controlled via `-D` flags, swept by grid_search.py):

| Param | Values | FC2 best | Effect |
|-------|--------|----------|--------|
| `N_STAGES` | 3,4,5 | **5** | Pipeline depth. NS5 mandatory (10% gap vs NS4). |
| `K_LOOP_UNROLL` | 1,2,4,6,8 | 4 | K-loop unroll factor |
| `TMA_RESIDUAL` | 0,1,2 | **1** | TMAR=1 (TMA coalesced) is key. |
| `BIAS_SMEM` | 0,1 | 1 | Load bias vector into SMEM (vs global `__ldg`). |
| `INTERLEAVE_STRATEGY` | 0,1,2,3 | 1 | TMA store interleaving. IS1≈IS2 for FC2. |
| `PHASE1_UNROLL` | 1,2,4 | 1 | Epilogue unroll. |
| `BATCH_MMA` | 0,1 | 0 | Single asm block for 4 sub-MMAs. **Noise** — hidden by overlap. |
| `BATCH_EPILOGUE` | 0,1 | 0 | 1=separate compute/store phases. |
| `STORE_TIMING` | 0,1 | 0 | 0=inline stores, 1=all deferred after Phase 1. |
| `DEFERRED_WAIT` | 0,1 | 0 | Defers `wait_group 0` after TMEM load. |

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
  sass_edit.py          # SASS binary editor + CP-SAT scheduler + fatbin patcher (see docs/sass_binary_editing.md)
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
  calib/                # Generated calibration benchmarks (instruction DB + codegen)
    instruction_db.py   # 18 SM100a instruction families, resource class hypotheses
    gen_kernels.py      # Generates tput/lat/conflict .cu files from instruction_db.py
    run.sh              # Generate → build → SASS verify → run all calibration suites

data/                   # Sweep results, ncu profiles, session outputs (data/session_*/)
docs/                   # Experiments (F1-F40), proposals, grid search, SASS notes, ncu analysis
  sass_binary_editing.md # SASS binary editor: capabilities, workflow, FC2 patching plan
  sass_editor_roadmap.md # SASS editor improvement checklist — deps, barriers, latency, loader
```

## Build and run

```bash
# FC2 — the active target
make fc2                # compile fc2.cu -> fc2
./fc2                   # run on B200, prints timing + TFLOPS + checksum
make fc2-timing         # compile with -DTIMING for cycle breakdown

# FC2 architecture search (combinatorial sweep on B200)
./tools/fc2_arch_search.sh              # full: 48 configs (32 combinatorial + 16 W0_RES)
./tools/fc2_arch_search.sh --quick      # 32 combinatorial configs only

# FC2 grid search
python3 tools/grid_search.py --kernel fc2 --tier all            # FC2 tiered sweep
python3 tools/grid_search.py --kernel fc2 --full-cross          # FC2 full cross-product
python3 tools/grid_search.py --kernel fc2 --only TMA_RESIDUAL   # sweep specific params
python3 tools/grid_search.py --kernel fc2 --interact residual   # residual interaction group

# CUTLASS FC2 comparison
make cutlass-bench-fc2-max && ./cutlass-bench-fc2-max
python3 tools/compare_all.py --runs 20 --csv data/compare.csv

# Analysis (local, no GPU)
python3 tools/analyze_sweep.py data/session_*/sweep_fc2.csv
python3 tools/balanced_eta.py data/session_*/sweep_fc2.csv
python3 tools/sass_analysis.py --cubin fc2 --deps

# SASS binary editing + CP-SAT scheduler (local, no GPU needed)
nvcc --cubin -arch=sm_100a -O3 fc2.cu -o fc2.cubin       # produce standalone cubin
python3 tools/sass_edit.py info fc2.cubin                  # list kernels
python3 tools/sass_edit.py dump fc2.cubin --sass sass/fc2.txt --start 0x50f0 --end 0x5160
python3 tools/sass_edit.py schedule fc2.cubin --sass sass/fc2.txt -s 0x51c0 -e 0x5620 --recipe recipe.txt
python3 tools/sass_edit.py schedule fc2.cubin --sass sass/fc2.txt -s 0x51c0 -e 0x63b0 --time-limit 3600  # full epilogue
python3 tools/sass_edit.py script fc2.cubin recipe.txt -o fc2_patched.cubin
python3 tools/sass_edit.py fatbin-patch fc2 --sass sass/fc2.txt --script recipe.txt -o fc2_patched
python3 tools/sass_edit.py diff fc2.cubin fc2_patched.cubin
# see docs/sass_binary_editing.md for full workflow + FC2 patching plan

# Calibration microbenchmarks (generated from instruction DB)
./bench/calib/run.sh              # generate + build + SASS verify + run (all 3 suites)
./bench/calib/run.sh tput         # throughput ILP sweep only (90 kernels)
./bench/calib/run.sh conflict     # NxN conflict matrix (153 pairwise tests)
make calib-all                    # just build (Makefile targets)

# PE and FC1 (done — do NOT sweep or optimize)
make                    # compile patch_embed.cu -> patch_embed
make fc1-gelu           # compile fc1_gelu.cu -> fc1-gelu
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
