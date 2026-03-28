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

### FC2 gap analysis (SASS-verified 2026-03-22)

Best: **1.452ms / 3016 TFLOPS** (NS5, IS1, base config). CUTLASS: 1.225ms fused. Gap = **227μs (19%)**.
At identical locked 1770 MHz: 1.482ms vs 1.263ms = **17.3% gap = ~1,940 cycles/tile**.
Clock throttling disproved — both kernels run at ~1.85 GHz unlocked on the same Verda B200.

**CUTLASS uses identical tile/cluster config**: 256×256×128, 2×1 cluster, cta_group::2. But has 2,864 SASS instructions vs our 2,248 (27% MORE code, 17% FASTER). **The entire gap is epilogue scheduling quality.**

### CUTLASS vs our architecture (SASS-verified)

**CUTLASS uses 8 warps (256 threads) — 2 extra warps we don't have:**
- **Warp 3: Dedicated EpilogueLoad** — pre-loads residual+bias into SMEM via TMA, independently from compute. Epilogue warps just LDS from SMEM. **This is the single biggest architectural difference. DO NOT FORGET THIS.**
- Warp 0: Tile scheduler via `UTCBAR.2CTA.MULTICAST` (NOT CLC — no CLCX/SETCTAID in SASS)
- Warp 1: MMA, Warp 2: MainloopLoad, Warps 4-7: Epilogue (128 threads)

**Our 6 warps (192 threads) — epilogue warps must self-load:**
- W0: TMA A/B loads. W1: MMA. W2-W5: Epilogue.
- **No dedicated epilogue load warp. Epilogue warps issue 16× LDG.E.128.CONSTANT (global loads) themselves.**

### SASS-verified epilogue differences

| | **Our FC2** | **CUTLASS** |
|---|---|---|
| Epilogue math | **BF16** (HFMA2/HADD2 after F2FP) | **FP32** (FFMA, F2FP only at end) |
| Bias/res source | LDG from global IN epilogue warps | LDS from SMEM (**W3 pre-loaded**) |
| STS placement | **4 consecutive** at block end (serialized) | **Interleaved** with FFMA/F2FP, 2-4 ops/shadow |
| Intra-epi pipeline | None — sequential | **2× BAR.SYNC + DEPBAR** per sub-iter |
| Sub-iterations | 3 (×64 cols, 8 STS each) | 4 (×64 cols, 4 STS each) |
| Instruction flow | 16 LDG → 16 F2FP → 64 BF16 → 8 STS | 12 LDS → LDTM → 64 FFMA ↔ F2FP ↔ STS |

### What's actually wrong — STRIP_EPILOGUE breakthrough (2026-03-26)

**The epilogue is NOT overlapped. It inflates the K-loop by 36%.** STRIP_EPILOGUE (commit 33a13eb) proves this definitively:

| | GEMM-only | Fused | Epilogue overhead |
|---|-----------|-------|-------------------|
| **Ours** | 1.161ms | 1.636ms | **+475μs (+41%)** |
| **CUTLASS** | 1.147ms | 1.225ms | **+78μs (+6.8%)** |

K-loops are identical (1.161 vs 1.147ms = 1.2% difference). **The entire gap is epilogue-induced K-loop inflation.** Our epilogue causes **6x more K-loop overhead** than CUTLASS's.

Per-tile W1 timing confirms this is uniform across all tiles, not outlier tiles:
- Stripped: kloop avg=13,773 cyc, total p50=13,645
- Full: kloop avg=18,714 cyc, total p50=19,944
- **K-loop inflates +4,941 cyc (+36%) with epilogue running**

**Why microbenchmarks (mma_bench, warp_scaling) were misleading:** They tested single tiles in isolation. The contention only manifests in the persistent kernel with 10,878 tiles, cross-tile mbarrier handoffs, and sustained concurrent memory traffic from all 74 clusters.

See "Confirmed dead" and "Strip bench decomposition" sections below for full dead-end list.

### Strip bench decomposition (2026-03-27) — NO single operation is the cause

30-experiment sweep (`tools/fc2_strip_bench.sh`, data: `data/strip_bench_20260328_055725/`). **Devastating result: ALL variants ≈ 1.635ms regardless of which epilogue operation is active.** Only STRIP_EPILOGUE (no epilogue at all) is fast at 1.160ms.

| Variant | ms | Delta | What it keeps |
|---|---|---|---|
| baseline | 1.637 | +477μs | Full epilogue |
| strip_all | 1.160 | 0 | Nothing (STRIP_EPILOGUE) |
| only_tma_load | 1.634 | +474μs | TMA loads only |
| only_tma_store | 1.636 | +476μs | TMA stores only |
| only_compute | 1.633 | +473μs | STS/BF16 only |
| bar_chunk | 1.635 | +475μs | BAR.SYNC serialization |
| fp32_epi | 1.638 | +478μs | FP32 epilogue math |

**Conclusion: the overhead is from epilogue warps BEING ACTIVE, not from what they do.** Even the lightest possible epilogue (a single TMA load per pass) causes the full 477μs.

**SASS diff (base 186-reg vs strip 78-reg):** K-loop is 347 structurally identical opcodes. Only register indices differ (base R104-R169, strip R2-R61). The compiler eliminates all dead epilogue code when STRIP_EPILOGUE=1, reducing regs from 186→78. Register file occupancy: 54.5% (base) vs 22.9% (strip).

### Confirmed dead — do NOT retry (updated 2026-03-27)

ALL of these have been benchmarked on B200 with zero effect on the 477μs gap:
- **BAR.SYNC serialization** (EPI_SYNC, EPI_BAR_PASS, EPI_BAR_CHUNK): ≈1.635ms. Does not help.
- **Combinatorial strips** (only_tma_load, only_tma_store, only_compute): ≈1.635ms. No single operation is the cause.
- **BAR.SYNC + strip combos** (bar_chunk_no_load, bar_chunk_no_store): ≈1.635ms.
- **FP32_EPILOGUE**: 1.638ms. ALU type doesn't matter.
- **Stagger** (500/2000 cycles): ≈1.633ms. Temporal offset doesn't help.
- **Pass count** (NUM_PASSES_PARAM=4): 1.635ms.
- **Store deferred** (STORE_TIMING=1): 1.642ms.
- EPI_LOAD_WARP (serial): +13% regression
- EPI_PIPELINE (fire-and-forget): +0.8% noise
- PRE_COMBINE, EPI_NOINLINE: 0μs wall time
- Source-level STS scheduling: ptxas immutable (byte-identical SASS)
- W0_RES_FULL: +15% catastrophic

### Remaining hypotheses (priority order)

The gap is NOT from specific epilogue operations. It's from epilogue warps being active at all. Three mechanisms:

1. **Register file hardware contention** — 186 regs × 192 threads = 54.5% RF occupancy vs 78 regs × 22.9%. RF bank conflicts may slow MMA warp's R2UR/UTCQMMA. **Test: REG_PAD flag — force STRIP_EPILOGUE to allocate 186 regs via dummy asm volatile declarations.**
2. **TMEM release timing** — Epilogue warps delay mbar_arrive that frees TMEM for W1 by ~400 instructions. Strip arrives almost immediately. **Test: EPI_DELAY flag — add nanosleep before arrive in strip path.**
3. **Warp scheduler pressure** — 6 active warps vs 2 effectively-active. MMA warp (sub-partition 1) shares with W5. **If A and B both fail, this is the mechanism — and CUTLASS solves it with 8 warps (more sub-partition parallelism, not less).**

**SM100a hardware data** (from `bench/tma_bench.cu` raw: `data/tma0-3.txt`, `bench/mma_bench.cu` raw: `data/mma0-1.txt`):
- STS.128 throughput: 27 cyc | LDS.128: 25 cyc @ILP=1, 3.5 cyc @ILP=7
- STS shadow: ≤4 BF16 free, 8=+55%, 15=+161% | LDS+STS overlap: 82.3%
- TMA load: 419 cyc (L2-warm) | TMA store: 197 cyc | TMA↔LSU: independent
- mbarrier arrive: 2 cyc | wait: 47 cyc | fence.proxy.async: 10 cyc
- **TMEM load (tcgen05.ld.sync): 2 cyc total** regardless of x16/x32/x64 width or ILP (bandwidth-limited)
- **TMEM double-buffer: zero contention** — read buf0 while MMA writes buf1 = +0.0 cyc
- **MMA K-iteration: 665 cyc** (fence + 4×MMA + commit + wait). Pipelined K=24: **525.6 cyc/iter**
- **MMA shadow budget: HUGE** — 16 STS or 64 BF16 or 8 full epilogue chunks = 100% hidden
- **Epilogue warps add +0.7 cyc (0.1%) to K-iter IN ISOLATION** — single-tile microbenchmark only. Real persistent kernel: **+36% K-loop inflation** (STRIP_EPILOGUE proof). Microbenchmarks miss cross-tile/cross-cluster contention.
- **Cross-warp handoff: 725 cyc** (W1 commit → W2 mbar_wait + TMEM_LD). +60 cyc vs same-warp K-iter.
- **Multi-SM spread: 0.2%** — perfectly uniform across 74 clusters
- CUTLASS epilogue source: `third_party/cutlass/.../sm100_epilogue_tma_warpspecialized.hpp`

**Multi-warp pipe scaling** (from `bench/calib/gen_warp_scaling.py`, raw: `data/warp_scaling.txt`):
- **4 sub-partitions**: warp i → sub-partition i%4. Visible in STS bimodal per-warp data.
- **STS**: 10→37 cyc (3.65× at 8 warps) — heaviest contention, SMEM store port shared
- **LDS**: 4.5→16 cyc (3.56×) — nearly as bad as STS
- **HFMA2/HADD2**: flat 2 cyc for 1-4 warps, doubles to 4 cyc at 5+ (2-partition BF16 ALU)
- **FFMA**: 1.5→2.0 cyc (1.36×) — nearly free
- **F2FP**: flat 2.0 cyc for ALL warp counts — zero contention, dedicated units
- **LDG L1**: 2→5.5 cyc (2.77×) | **LDG L2**: 23.6→35.6 cyc (1.50× — latency-dominated)
- **FFMA compute is free alongside any epilogue**: F_4ldgmix_0ffma = F_4ldgmix_2ffma = 8.25 us
- **BAR.SYNC hurts mixed epilogue by +25%**: B_4mixbar_i8=10.3 vs nobar=8.25 us
- **Dedicated load warp always adds overhead** in P-tests

### Multi-warp scheduling calibration (`bench/calib/gen_warp_scaling.py`)

134 generated kernels across 8 test suites, all single-CTA with per-warp `clock64()` timing.
3 warmup + 10 measured launches per kernel, reports min CPI across launches.
**Data collected 2026-03-26**: `data/warp_scaling.txt`. Key findings integrated into hardware data above.

| Suite | Count | What it measures |
|-------|-------|-----------------|
| **S** | 80 | Same-pipe throughput scaling: 10 pipes (STS, FFMA, HFMA2, LDS, F2FP, HADD2, NOP, **LDG**, **STS_ILP1**, **LDG_L2**) × 8 warp counts (1–8). Key: is STS throughput per-warp or per-SM? LDG pipe (TEX/L1) scaling for FC2 epilogue. STS_ILP1 reveals sub-partition structure. **LDG_L2** uses 32 MB buffer to force L2 miss — real FC2 residual loads hit L2, not L1. |
| **X** | 11 | Cross-pipe independence: different warps on different pipes simultaneously. STS+LDS scaling at 2+2, 3+3, 4+4 warps (SMEM port contention). **STS+LDG** cross-pipe (LSU vs TEX — FC2 epilogue does both). |
| **F** | 15 | Foreground/background interference: FC2-like configs (4 STS warps + 1–4 FFMA warps, mixed epi, reverse). FC2-realistic LDG+BF16+STS epi mix. **6-warp FC2-exact** (4 ldgmix epi + 1 LDG W0 proxy + 1 FFMA W1 proxy). **L2-miss LDG** variants force realistic memory latency. |
| **P** | 11 | Producer-consumer (CUTLASS W3 pattern): 4 epi warps self-loading via LDG (our arch) vs LDS from pre-loaded SMEM (CUTLASS result) vs with dedicated load warp (nanosleep-idle, like W3). With/without 1–2 K-loop FFMA warps. **Directly tests whether a dedicated idle load warp reduces dispatch contention.** |
| **B** | 9 | BAR.SYNC contention effect: 4 STS warps with periodic BAR.SYNC (intervals 4/8/16/32) + 2 FFMA compute warps, plus no-BAR baseline. **Mixed-epi variants** (LDG+BF16+STS bar warps for realistic epilogue pipe mix). |
| **N** | 6 | Nanosleep calibration: actual sleep duration for values 10/50/100/500/1000/5000. Validates P-test load warp idle simulation. |
| **A** | 2 | Asymmetric duration: epi warps run REPS/2 or REPS/4 then exit, compute warps run full REPS. **Tests dynamic dispatch recovery** — does compute CPI improve after epi warps finish? |

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
- **No dedicated EpilogueLoad warp** — epilogue warps (W2-W5) load bias/residual themselves via LDG from global memory. CUTLASS has a dedicated W3 that pre-loads into SMEM via TMA.

Epilogue ops: bias+pos_embed (PE), bias+GELU (FC1), bias+residual (FC2). FC2's residual loads a full [M,N] matrix — heaviest epilogue, memory-bound. FC1's GELU is compute-bound. PE's add is trivial. **FC2 epilogue uses BF16 math (HFMA2/HADD2); CUTLASS uses FP32 (FFMA) and converts at the end.**

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
  fc2_strip_bench.sh     # Epilogue contention decomposition (30 experiments, ~10 min on B200)
  analyze_warp_scaling.py  # Warp-scaling benchmark analysis (scaling curves, BAR.SYNC effect)
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
    gen_warp_scaling.py # Multi-warp scheduling calibration (134 kernels: S/X/F/P/B/N/A tests)
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

# FC2 strip bench (epilogue contention decomposition, ~30 experiments)
bash tools/fc2_strip_bench.sh              # run everything (~10 min)
bash tools/fc2_strip_bench.sh --batch 1    # run only batch 1
bash tools/fc2_strip_bench.sh --dry-run    # print commands without running

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

# Multi-warp scheduling calibration (134 kernels, requires B200)
make calib-warp                                              # build
./calib-warp > data/warp_scaling.txt                         # run on B200
python3 tools/analyze_warp_scaling.py data/warp_scaling.txt  # analyze

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
