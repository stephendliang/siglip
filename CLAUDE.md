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
- ~161-254 regs/thread (varies with `K_LOOP_UNROLL`, `W0_LOOP_UNROLL`, `SUB_MMA_UNROLL`, `CVT_ADD_FUSED`, `PHASE1_UNROLL`, `MBAR_EARLY`; defaults: PE=205, FC1=~209, FC2=254), 0 spills. Limits occupancy to 1 CTA/SM.
- Timing build uses ~245 regs (distorts cycles vs production). Wall clock is ground truth.

Run `python3 tools/analyze_timing.py data/clock64_timing.txt` for full equilibrium analysis and what-if projections.
Run `python3 tools/analyze_source_counters.py data/source_counters_raw.csv` for per-instruction stall breakdown.
See `docs/EXPERIMENTS.md` for experiments (F1-F40) with hypotheses, results, and analysis. See `docs/FUTURE_PROPOSALS.md` for optimization roadmap.

## Kernel structure

Warp-specialized, 6 warps (192 threads), `cta_group::2`, `__cluster_dims__(2,1,1)`:

- **W0**: TMA async bulk loads (A + B tiles, both CTAs load independently)
- **W1**: TMEM alloc (512 cols, single alloc for double buffering) + `tcgen05.mma.cta_group::2` accumulation into TMEM (CTA0 lane-0 only, multicast commit to both CTAs)
- **W2-W5**: Overlapped epilogue (4 warps) — each warp independently polls mainloop mbarrier, then runs unified SMEM-staged store: Phase 1 (all 256 cols through 4 SWIZZLE_128B regions: x32 `tcgen05.ld` → epilogue op → CVT → `st.shared`, with **interleaved TMA stores** every 2 regions hiding in TMEM stall windows), Phase 2 (`cp.async.bulk.commit_group` only — all TMA stores already issued inline), **mbar_arrive** signals TMEM free for W1. The epilogue op is kernel-specific: bias+pos_embed add (patch_embed), bias+GELU (fc1_gelu), or bias+residual (fc2).

TM=128 rows / 32 rows per warp = 4 row groups. With 4 epi warps (default), each warp owns a full row group (256 cols, `is_split=0`). With 5 epi warps, warp 4 shares row_group 0 via `ew % 4`, creating split warps (`is_split=1`) that each handle 128 cols (2 regions). `epilogue_store` is templated on `<NC_START, NC_END>` so the compiler sees constant loop bounds and `N_REGIONS = (NC_END - NC_START) / 64` is constexpr; unroll depth controlled by `PHASE1_UNROLL` (default 2).

The overlapped epilogue for tile N-1 runs concurrently with the K-loop for tile N (double-buffered TMEM, mbarrier-protected). After the tile loop, W2-5 run a drain epilogue for the last tile.

### Tile config
- Tile: 256x256x128 (M=2x128 from cta_group::2, N=256, K=128)
- TMEM: single alloc of 512 cols (TN*2), double-buffered via column offset (buf*TN)
- SMEM: 4-stage pipeline (131 KB) + epilogue staging (4 warps x 16,384 = 64 KB, SWIZZLE_128B) = ~197 KB total of 228 KB
- Tiles: 3626 M-tiles x 3 N-tiles = 10,878 total, snake ordering
- ~161-254 registers/thread, 0 spills (varies with unroll params and epilogue type; PE defaults=205, FC2=254)
- `NUM_EPI_WARPS` controls epilogue warp count (currently 4); `THREADS` derived as `32*(2+NUM_EPI_WARPS)`

## Development workflow

All kernels share `kernel_common.cuh` (pipeline, TMEM loads, TMA helpers, mbarrier ops, tuning parameters) and `kernel_body.cuh` (epilogue_store template, persistent_gemm kernel template). Each `.cu` file `#define N_DIM` and `K_DIM`, defines its epilogue macro (e.g., `CVT_ADD_STS_V4`), then includes both headers — tile counts and K-iter constants are derived automatically.

```
edit kernel_common.cuh / kernel_body.cuh / patch_embed.cu / fc1_gelu.cu / fc2.cu -> make -> ./siglip_vision (or ./fc1_gelu, ./fc2)
```

## Repository structure

```
kernel_common.cuh       # Shared infrastructure (pipeline, TMEM, TMA, mbarriers, tuning params)
kernel_body.cuh         # Shared kernel body (epilogue_store template, persistent_gemm template)
patch_embed.cu          # Patch embed GEMM — [928256,768]×[768,768]^T + bias + pos_embed
fc1_gelu.cu             # FC1+GELU GEMM — [928256,768]×[768,3072]^T + bias + GELU
fc2.cu                  # FC2 GEMM — [928256,3072]×[3072,768]^T + bias + residual
Makefile                # Build rules (sm_100a, nvcc flags)
CLAUDE.md               # This file

tools/                  # Analysis & sweep scripts
  sass_analysis.py      # SASS scheduling analyzer (decodes control words, dep graphs, slack)
  grid_search.py        # Compile-time parameter sweep (tiered search, CSV output)
  analyze_timing.py     # clock64 timing → equilibrium analysis
  analyze_source_counters.py  # ncu SourceCounters CSV → stall breakdown
  remote.py             # Remote B200 provisioning + sweep runner
  compare.py            # ncu CSV diff tool
  compare_sass.py       # SASS dump diff tool
  stat_test.py          # Statistical significance testing
  bench_sweep.sh        # Shell-based sweep (legacy)
  cutlass_sweep.sh      # Build + run all CUTLASS bench variants (max/standard)

bench/                  # Benchmark & calibration kernels
  cutlass_bench.cu      # CUTLASS tile/policy sweep (compile-time N/K/epilogue via -D flags)
  siglip_periodic_add.hpp # Custom EVT visitor (Sm100PeriodicAddNode)
  cublas_bench.cu       # cuBLAS baseline benchmark (compile-time N/K/epilogue via -D flags)
  calibration.cu        # SASS latency microbenchmarks (K1-K26)
  common.h              # Shared PTX helpers (mbarrier, TMA, tcgen05)
  profiler.h            # globaltimer-based kernel profiler
  CALIBRATION_LACKING   # Calibration status audit (measured/pending/unfixable)
  test_tmem.cu          # TMEM test kernel
  SUPERIOR.cu           # Reference kernel
  leetgpu_b200.cu       # LeetGPU benchmark

data/                   # Profiling data & sweep results
  sweep_results.csv     # Run 1 grid search (145 configs). Best: 0.519 ms
  sweep_results_run2.csv  # Run 2 (pre is_split fix). Best: 0.522 ms
  sweep_results_run3.csv  # Run 3 (post is_split fix). 141/145 valid
  sweep_results_run4.csv  # Run 4 (post fence.proxy.async). 145/145 valid
  top5_profile.csv      # ncu profiling of top 5 configs
  source_counters_raw.csv  # Raw ncu --set source --csv data
  clock64_timing*.txt   # Kernel printf from timing builds
  after.csv, baseline.csv  # ncu comparison data

ISSUES_CALIBRATION      # Calibration expansion design (K13-K26 kernel specs)

docs/                   # Documentation & analysis
  EXPERIMENTS.md        # Experiment log (F1-F40)
  FUTURE_PROPOSALS.md   # Optimization roadmap
  GRID_SEARCH.md        # Grid search design & findings
  top5_analysis.md      # Top 5 config ncu analysis
  sass_optimizer.md     # SASS post-compile optimizer design
  architecture.md       # Kernel architecture notes
  make_better.md        # Deep ncu profiling analysis
  sass.md               # SASS reference notes
  reference/            # Model dump, SASS disassembly
```

## Build and run

```bash
make                    # compile patch_embed.cu -> siglip_vision
./siglip_vision         # run on B200, prints timing + TFLOPS + checksum
make fc1-gelu           # compile fc1_gelu.cu -> fc1_gelu
./fc1_gelu              # run FC1+GELU kernel
make fc2                # compile fc2.cu -> fc2
./fc2                   # run FC2 kernel
make timing && ./siglip_timing | tee data/clock64_timing.txt | python3 tools/analyze_timing.py
ncu --set source --csv ./siglip_vision > data/source_counters_raw.csv && python3 tools/analyze_source_counters.py data/source_counters_raw.csv

# CUTLASS tile/policy sweep (compile-time N/K/epilogue via -D flags, one binary per epilogue)
make cutlass-bench          # patch embed: N=768 K=768 PERIODIC_ADD (default)
make cutlass-bench-fc1      # FC1: N=3072 K=768 GELU_BIAS
make cutlass-bench-fc2      # FC2: N=768 K=3072 BIAS_ONLY
./cutlass-bench [imgs_per_sm]  # default 32 → M=928256

# CUTLASS extended sweep (more tile/cluster configs)
make cutlass-bench-max          # extended patch embed
make cutlass-bench-fc1-max      # extended FC1
make cutlass-bench-fc2-max      # extended FC2

# Full CUTLASS sweep across all three layers (builds + runs)
./tools/cutlass_sweep.sh              # max mode, 32 imgs/SM
./tools/cutlass_sweep.sh 1            # quick test (1 img/SM = 148*196 rows)
./tools/cutlass_sweep.sh 32 standard  # standard tile list only

# cuBLAS baseline (compile-time N/K/epilogue, best-of-heuristics)
make cublas-bench           # patch embed: N=768 K=768 PERIODIC_ADD
make cublas-bench-fc1       # FC1: N=3072 K=768 GELU_BIAS
make cublas-bench-fc2       # FC2: N=768 K=3072 BIAS_ONLY
./cublas-bench [imgs_per_sm]  # default 32 → M=928256

# Parameter grid search (any kernel)
python3 tools/grid_search.py --tier all                    # sequential 1→2→3, pinning winners
python3 tools/grid_search.py --full-cross                  # all parameters crossed
python3 tools/grid_search.py --kernel fc2 --tier all       # FC2 sweep
python3 tools/grid_search.py --kernel fc1_gelu --tier 3    # FC1 tier 3 only
python3 tools/grid_search.py --only K_LOOP_UNROLL W0_LOOP_UNROLL SUB_MMA_UNROLL  # sweep specific params

# SASS scheduling analysis
cuobjdump --dump-sass siglip_vision > sass_dump.txt
python3 tools/sass_analysis.py sass_dump.txt                          # annotated listing
python3 tools/sass_analysis.py sass_dump.txt --section 0x1300 0x1a70  # address range (e.g., epilogue)
python3 tools/sass_analysis.py sass_dump.txt --deps                   # dependency + slack analysis
python3 tools/sass_analysis.py --cubin siglip_vision                  # runs cuobjdump internally

# Calibration: verify SASS control word decoder on B200
make calibration          # compile bench/calibration.cu
./calibration > cal_output.txt
cuobjdump --dump-sass calibration > cal_sass.txt
python3 tools/sass_analysis.py cal_sass.txt --calibrate-compare                          # SASS-only
python3 tools/sass_analysis.py cal_sass.txt --calibrate-compare --runtime cal_output.txt # compare vs runtime
```

### CUTLASS bench details

`bench/cutlass_bench.cu` is compile-time parameterized via `-D` flags (mirrors `cublas_bench.cu`):
- `BENCH_N` (output dim, default 768), `BENCH_K` (reduction dim, default 768)
- `BENCH_EPILOGUE`: `1`=PERIODIC_ADD (patch embed), `2`=GELU_BIAS (FC1), `3`=BIAS_ONLY (FC2)
- `CUTLASS_EXTENDED_SWEEP=1` adds more tile/cluster/policy configs

Each binary sweeps tile shapes × cluster configs × scheduling policies (auto, 1SM, 2SM with explicit stage counts S3/S4/S5). Configs that exceed SM100a SMEM capacity (232448 bytes) are detected at compile time via `sizeof(SharedStorage)` and reported as ">SMEM" without crashing — `if constexpr` guards prevent the CUTLASS `static_assert` from firing.

**SMEM overflow recovery**: Each Instance struct exposes `static constexpr size_t kSmemBytes = sizeof(typename GemmKernel_::SharedStorage)` — a host-side constexpr giving exact SMEM usage. Each `run()` method checks `if constexpr (kSmemBytes > cutlass::arch::sm100_smem_capacity_bytes)` and returns `-3.0f` (sentinel) in the true branch, while the `else` branch contains `gemm.run()`. Since `if constexpr` discards the false branch entirely, `GemmKernel_::operator()` (which contains the `static_assert`) is never instantiated. Result: always compiles, never crashes, prints ">SMEM" or "SKIP — SMEM exceeded".

**PERIODIC_ADD** (patch embed): 6 measurements per config — GEMM-only (beta=0), fused FP32/BF16 (beta=1), fused EVT (`SigLipPeriodicAdd` custom visitor), PostAdd-only, GEMM+PostAdd. EVT validation checks spot-check values.

**GELU_BIAS** (FC1): `BiasActGemmInstance<GELU_taylor>` using `LinCombPerColBiasEltAct`. 3 measurements — GEMM-only, fused bias+GELU, GEMM+unfused bias+GELU.

**BIAS_ONLY** (FC2): `BiasActGemmInstance<Identity>` — same dispatch path as FC1 with identity activation. 3 measurements — GEMM-only, fused bias, GEMM+unfused bias.

Data init: A=0x3C (FP8 1.5), B alternating columns (even=1.5, odd=1.0), non-uniform bias. Timing: 2 warmup, 10 timed. Sentinel values: -1.0f=can't implement, -2.0f=validation fail, -3.0f=SMEM exceeded.

### cuBLAS bench details

`bench/cublas_bench.cu` is compile-time parameterized via the same `-D` flags as CUTLASS (`BENCH_N`, `BENCH_K`, `BENCH_EPILOGUE`). Uses cublasLt with up to 128 algo heuristics and 256MB workspace.

Tests both **MXFP8** (block-scaled Vec32 UE8M0) and **per-tensor FP8** paths. For each scaling mode, measures:
1. GEMM-only (beta=0)
2. Fused epilogue (beta=1 add for PERIODIC_ADD; cuBLAS GELU_BIAS/BIAS epilogue for FC1/FC2)
3. Unfused (GEMM + post-kernel timed together)
4. Post-kernel only (standalone unfused kernel)

Reports best-of-all-heuristics for each measurement, plus overhead analysis (fused vs unfused delta). Timing: 3 warmup, 20 timed iters.

`BENCH_EPILOGUE=0` (NONE) is also supported — GEMM-only benchmarking without any epilogue.

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

## Grid search findings (4 runs × 145 configs)

Full parameter sweep results in `data/sweep_results.csv` (run 1), `data/sweep_results_run2.csv` (run 2), `data/sweep_results_run3.csv` (run 3), `data/sweep_results_run4.csv` (run 4). Run 4 is the definitive run after all correctness fixes (fence.proxy.async + is_split + OFF_STAGING alignment) — **145/145 valid, zero failures**. Key findings:

### Performance (valid configs only, sorted by impact)

| Parameter | Best | Default | Effect |
|-----------|------|---------|--------|
| SNAKE_ORDER | 1 | 1 | **Critical**: 0=1 costs ~50-200 TFLOPS (10-40%). Snake ordering essential. |
| PHASE1_UNROLL | 2 | 2 | **Critical**: 1 is catastrophic (-30%, ~1500 TFLOPS). 4 slightly slower + 255 regs. |
| INTERLEAVE_STRATEGY | 2 | 2 | **Best perf.** 0 = ~30 TFLOPS slower. 3 = ~50 TFLOPS slower. All strategies pass 145/145 after fence.proxy.async fix. |
| MBAR_EARLY | 1 | 0 | +5-18 TFLOPS consistently. 0.519 vs 0.524 ms at optimal stagger. |
| STAGGER_CYCLES | 120-160 | 80 | Sweet spot 80-160; 0 costs ~5 TFLOPS; 200 slightly worse. |
| N_STAGES | 4 | 4 | 3 too few (0.557 ms). 5 works but +20 regs for marginal gain. |
| TMEM_LOAD_WIDTH | 32 | 32 | 64 burns 255 regs, ~10-20 TFLOPS slower than 32 (209 regs). |
| CVT_ADD_FUSED | 0 or 1 | 1 | Equivalent within noise. |
| K_LOOP_UNROLL | N_STAGES | N_STAGES | Controls W1 K-loop unroll. 1,2=runtime modulo (+stack), 4=compile-time stages. |
| W0_LOOP_UNROLL | 0 | 0 | Controls W0 load warp K-iter loop. 0=compiler decides, 1=no unroll, N=unroll by N. |
| SUB_MMA_UNROLL | 0 | 0 | Controls sub-MMA inner loop (3 iters). 0=compiler decides, 1=no unroll, 3=full. |
| NUM_EPI_WARPS | 4 or 5 | 4 | 5 matches 4 in perf (~0.523 ms). All strategies work with both 4 and 5 warps. |

### Top configs (cross-run)

```
Run 1 best: MBAR_EARLY=1 STAGGER_CYCLES=160              → 0.519 ms / 2108 TFLOPS (211 regs, 4 epi warps)
Run 1 #2:   MBAR_EARLY=1 STAGGER_CYCLES=120              → 0.520 ms / 2104 TFLOPS (211 regs)
Run 2 best: INTER=1 MBAR=1 EPI=5 PH1U=4 STAG=160 TLW=64 → 0.522 ms / 2096 TFLOPS (235 regs)
Defaults:                                                  → 0.524 ms / 2090 TFLOPS (209 regs)
```

### Validation history and is_split fix

**Bug 1 — OFF_STAGING alignment** (fixed): SWIZZLE_128B requires 1024-byte aligned SMEM base. Was 128-aligned → systematic 8-col group swaps. Fix: `(... + 1023) & ~1023` rounding.

**Bug 2 — Strategy 0,3 hardcoded region counts** (fixed): With 5 epi warps, `row_group = ew % 4` wraps, creating split warps (`is_split=1`) that handle 128 cols (2 regions) instead of 256 cols (4 regions). Strategy 0 hardcoded 4 TMA stores (wrote garbage from uninitialized regions 2-3). Strategy 3's inline trigger `((nc - NC_START) >> 6) == 2` never fired for 2-region split, and Phase 2 was guarded by `N_REGIONS > 3`. Fix: derive `constexpr N_REGIONS = (NC_END - NC_START) / 64` at function scope; loop over N_REGIONS in strategy 0; adjust trigger/loop bounds in strategy 3.

**Bug 3 — Missing `fence.proxy.async.shared::cta`** (fixed): TMA stores read staging SMEM via the async proxy, but `st.shared` writes use the sync proxy. `__syncwarp()` does NOT bridge proxies — `fence.proxy.async.shared::cta` is required (per PTX ISA). Without it, TMA reads stale data. Strategy 1 was most affected (inline stores fire immediately after writes); strategies 0,3 had enough latency to mask the bug. Fix: added fence after every `__syncwarp()` preceding TMA stores (8 sites).

| Config | Run 1 (4w) | Run 2 (5w, pre-fix) | Run 3 (5w, +is_split) | Run 4 (+fence) |
|--------|------------|---------------------|-----------------------|----------------|
| INTERLEAVE_STRATEGY=0 | 32/32 | 0/32 | **32/32** | 32/32 |
| INTERLEAVE_STRATEGY=1 | ~17/32 | ~22/32 | ~29/32 | **32/32** |
| INTERLEAVE_STRATEGY=2 | 32/32 | 31/32 | 31/32 | **32/32** |
| INTERLEAVE_STRATEGY=3 | 32/32 | 0/32 | **32/32** | 32/32 |
| **Total** | **130/145** | **74/145** | **141/145** | **145/145** |
