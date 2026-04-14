# FC2 Optimization — SigLIP2 Vision Encoder

Hand-tuned Blackwell (SM100a) persistent GEMM kernel for FC2 layer of `google/siglip2-base-patch16-224`.
FP8 (E4M3) inputs, BF16 output, tcgen05 MMA, TMA, `cta_group::2` with 2-CTA clusters.
Cross-compiled on CPU VPS, runs on B200. PE and FC1 kernels are done — see `CLAUDE.md.mothballed` for their docs.

## Current status (B200, 2026-04-14)

Default FC2 shape: [928256, 3072] x [3072, 768]^T + bias + residual. Batch = 4736 images x 196 patches.

| Variant | ms | TFLOPS | vs CUTLASS fused |
|---|---|---|---|
| **w3_fused (NS6+PREFILL)** | **1.111** | **3943** | **-9.4%** |
| w3_gemm (GEMM-only) | 1.100 | 3981 | — |
| w3_sched (TD=4 dispatch) | 1.147 | 3819 | -6.4% |
| CUTLASS fused | 1.226 | 3573 | baseline |
| CUTLASS strip | 1.152 | 3801 | — |

Key: N_STAGES=6 (6-stage pipeline, 227KB of 228KB SMEM) + PREFILL (skip epilogue_mbar wait, rely on TMEM double-buffering). Tile/cluster: 256x256x128, 2x1 cluster, cta_group::2, 74 clusters.

## Dimension sweep (B200-verified, 2026-04-09)

Infrastructure: `tools/dim_sweep.sh`. Dims overridable via `-DM_TOTAL=X -DN_DIM=Y -DK_DIM=Z`. Constraints: M%256==0, N%256==0, K%128==0. Must use `make -B` (Make doesn't track DFLAGS).

### M scaling (N=768, K=3072) — advantage stable 8-10%

| M | w3_fused | cutlass_fused | Delta | TFLOPS |
|---|---|---|---|---|
| 116032 | 0.154 | 0.167 | -7.8% | 3557 |
| 232064 | 0.289 | 0.317 | -8.8% | 3787 |
| 464128 | 0.563 | 0.621 | -9.3% | 3893 |
| 928256 | 1.110 | 1.226 | -9.5% | 3947 |
| 1856512 | 2.206 | 2.433 | -9.3% | 3971 |

Near-perfect linear scaling. Advantage robust across batch sizes.

### K scaling (M=928256, N=768) — advantage vanishes at large K

| K | K_ITERS | w3_fused | cutlass_fused | Delta | w3_gemm | cutlass_strip | gemm Delta |
|---|---|---|---|---|---|---|---|
| 1536 | 12 | needs NO_PREFILL | 0.729 | — | 0.634 | 0.657 | -3.5% |
| 2048 | 16 | needs NO_PREFILL | 0.914 | — | 0.842 | 0.827 | +1.8% |
| 3072 | 24 | 1.110 | 1.226 | **-9.5%** | 1.100 | 1.152 | -4.5% |
| 4096 | 32 | 1.538 | 1.551 | -0.8% | 1.511 | 1.474 | +2.5% |
| 6144 | 48 | 2.257 | 2.263 | -0.3% | 2.150 | 2.151 | 0.0% |

K=1536/2048 deadlocked in early runs (PREFILL unsafe at K_ITERS<20). dim_sweep.sh auto-adds `-DNO_PREFILL` for K_ITERS<20. Fused with NO_PREFILL not yet re-tested.

Why advantage vanishes: NS6 pipeline depth is 6/K_ITERS of the K-loop. At K=3072 (25%) it hides significant DRAM latency. At K=6144 (12.5%) the hiding is negligible and compute dominates.

### TD=4 (w3_sched) wins at large K

| K | w3_fused | w3_sched (TD=4) | Delta |
|---|---|---|---|
| 3072 | 1.110 | 1.134 | sched 2.2% slower |
| 4096 | 1.538 | 1.545 | tied |
| 6144 | 2.257 | **2.098** | **sched 7% faster** |

TD=4 (dedicated W7 scheduler warp, dynamic tile assignment via atomicAdd) eliminates DRAM amplification (ncu: 4.28GB = 1.00x vs default dispatch 4.85GB = 1.13x). Has ~34us scheduler overhead (300 cyc/tile tile_ready_mbar on W3 critical path). At K<=3072, overhead > savings. K=4096 is the battleground (tied). At K>=5120, savings win clearly.

### N=256 — advantage shrinks

N=256 (1 N-tile): w3_fused=0.562ms vs cutlass_fused=0.576ms = -2.4%. TFLOPS drops to 2598 (vs 3947 at N=768). Memory-bandwidth-limited: compute proportional to N but A traffic constant.

Large N (>1536) not yet tested. NS5 required for N>1536 (SMEM per stage grows with N).

### Adaptive tuning knobs

| Knob | Rule | Why |
|---|---|---|
| N_STAGES | NS6 for N<=1536, NS5 for N>1536 | SMEM per stage grows with N; NS6 needs <=228KB |
| PREFILL | On for K_ITERS>=20, off otherwise | Short K-loop can't drain epilogue before TMEM reuse |
| Tile dispatch | Default for K<=4096, TD=4 for K>=5120 | TD=4 34us overhead vs amplification crossover; K=4096 is battleground |
| Tile shape | 256x256x128 fixed | Not yet explored for different N |

## Kernel structure (fc2_w3.cu)

Warp-specialized, 7 warps (224 threads), `cta_group::2`, `__cluster_dims__(2,1,1)`:

- **W0**: TMA async bulk loads (A + B tiles)
- **W1**: TMEM alloc + tcgen05.mma.cta_group::2 accumulation
- **W2**: EpilogueLoad — TMA loads residual into SMEM (circular 2-stage pipe, previous tile)
- **W3-W6**: Overlapped epilogue (4 warps) — LDS residual+bias from SMEM, TMEM ld, epilogue math, CVT, STS, TMA store
- **W7**: Only in TD=4 (scheduler warp) or 8-warp mode (NUM_IDLE_WARPS=1)

Tile: 256x256x128, TMEM 512 cols double-buffered, 6-stage SMEM pipeline (default), K_ITERS=K_DIM/128.

### Tile dispatch modes

| Mode | Flag | Status |
|---|---|---|
| Default (Group-3) | none | Active. Fixed tn per cluster, stride M. Best at K<=4096 |
| Sched (TD=4) | TILE_DISPATCH=4 | Active. Dedicated W7, atomicAdd. Best at K>=5120, 0 DRAM amplification |
| Atomic (TD=1) | TILE_DISPATCH=1 | Dead (1.370ms overhead) |
| Inline atomic (TD=6) | TILE_DISPATCH=6 | Dead (W0 blocked at tile boundary) |
| Inline K-loop (TD=7) | TILE_DISPATCH=7 | Dead (atomicAdd in K-loop disrupts TMA pipeline, +41% tma_issue) |
| CLC (TD=5) | TILE_DISPATCH=5 | Dead (deadlocks, incompatible with persistent kernel) |
| Spin (TD=2), Grid (TD=3) | TILE_DISPATCH=2/3 | Experimental |

## Compute floor

From bench/mma_bench.cu (data/mma1.txt):
- MMA K-iteration: 665 cyc raw, 525.6 cyc/iter pipelined
- Per tile (K=3072, 24 iters): ~12,614 cyc
- Theoretical strip floor at 1.813 GHz matches observed ~1.048ms
- **Epilogue is 100% hidden in MMA shadow.** The 58us fused overhead above strip is entirely memory-side (DRAM amplification + TMA store contention), not compute or instruction scheduling.

## DRAM read amplification

ncu data: `data/ncu_20260408_032714/` (8 variants) and `data/ncu_20260409_015856/` (7 variants).

Theoretical minimum: strip = A+B = 2.85GB, fused = A+B+residual+bias = 4.28GB.

| Variant | DRAM read | vs theoretical | Write |
|---|---|---|---|
| w3_fused (default) | 4.85GB | 1.13x | 1.42GB |
| w3_gemm | 3.80GB | 1.33x (vs 2.85) | 1.42GB |
| w3_strip | 4.55GB | 1.60x (vs 2.85) | 0.01GB |
| cutlass_fused | 4.28GB | 1.00x | 1.40GB |
| cutlass_strip | 2.85GB | 1.00x | 1.40GB |
| **w3_sched (TD=4)** | **4.28GB** | **1.00x** | 1.43GB |
| **w3_atomic (TD=1)** | **4.28GB** | **1.00x** | 1.41GB |

Key finding: TD=4 and TD=1 dispatch eliminate amplification entirely. Default (Group-3) dispatch has L2 capacity misses from tile ordering. NS6 deeper pipeline hides the extra latency (faster despite more DRAM traffic).

Investigated and failed: tile reordering (N-batch, phase-offset, Group-3 all failed/neutral), L2 cache hints (EVICT_FIRST/LAST/NORMAL all zero effect), CLC vs static (+7us noise). Root cause is L2 capacity, not eviction policy.

## Epilogue analysis

Our fusion (residual+bias) costs ~44us vs CUTLASS's ~72us. BF16 math (HFMA2/HADD2) faster than CUTLASS FP32 (FFMA+F2FP). Cross-warp STS clustering is real (barrier stalls +753% in ncu) but net cost is cheaper.

STS clustering proven immutable from source level: 6+ approaches all produce identical SASS. ptxas controls STS scheduling. Not worth pursuing further — epilogue is already hidden in MMA shadow.

## Dead ends — do NOT retry

**Source-level epilogue (fc2_w3):** CUTLASS_LOOP, FP32_EPILOGUE, CUTLASS_EPILOGUE, CPP_EPILOGUE, CUTE_STORE, @!PT LDS, cvta.shared, NO_PRE/POST_STORE_BAR, NUM_EPI_STAGES, stmatrix, EPI_REORDER, NUM_EPI_WARPS=1/2. All identical SASS — ptxas immutable.

**CUTLASS hybrid (fc2_hybrid.cu):** Phase 2 (=Phase1 speed), Phase 3b (2.76ms, CUTLASS breaks at non-8-warp), Phase 4 (2.77ms, static tile loop 2.3x slower). All dead.

**Old fc2.cu (4 warps):** BAR.SYNC serialization, FP32, NOP_EPILOGUE, EPI_DELAY, REG_PAD, EPI_PIPELINE, W0_RES_FULL/PREFETCH, EPI_REUSE_SMEM (worse than NS5). All dead.

**Cross-warp temporal:** SELF_LOAD (per-warp TMA, no sync), SELF_STAGGER (nanosleep 50-200ns). Zero effect.

**Dispatch dead ends:** TD=1 atomic (1.370ms timing, though 0 amplification — overhead kills it), TD=5 CLC (deadlocks), TD=6 inline atomic (1.370ms, W0 blocked). COL_LOCK (column-locked dynamic dispatch, 1.137ms fused — TMA penalty is inherent to W7 mbarrier path, not tile ordering; collock_strip 1.060ms = 62us slower than sched due to load imbalance across columns). TD=7 inline atomic in K-loop (1.257ms fused, 1.040ms strip — atomicAdd at ki=0 disrupts TMA pipeline, +41% tma_issue in strip, +77% in fused; W0's K-loop is memory-pipeline-sensitive, ANY global atomic degrades TMA throughput).

**LDG/STG kernel (fc2_ldg.cu):** STG goes through L1TEX (128B/thread), TMA bypasses it. Fundamentally bandwidth-limited, even GEMM-only slower than fc2_w3.

**L2 cache hints:** All EVICT combos on TMA loads. Zero effect — capacity, not policy.

**SASS patching:** Intra-warp reorder targets wrong bottleneck (memory, not instructions). Inter-warp YIELD stagger untested but irrelevant since epilogue is hidden in MMA shadow.

**PREFILL at NS5:** Zero effect (K-loop >> epilogue at NS5). Only helps at NS6.

## Build and run

```bash
# Primary kernel (default: NS6, PREFILL, Group-3 dispatch)
make fc2-w3 && ./fc2-w3                          # fused 1.110ms
make fc2-w3-gemm && ./fc2-w3-gemm                # GEMM-only (valid=1)
make fc2-w3-strip && ./fc2-w3-strip               # MMA-only (valid=0)
make fc2-w3-sched && ./fc2-w3-sched               # TD=4 scheduler dispatch

# Custom dims (MUST use -B: Make doesn't track DFLAGS)
make -B fc2-w3 DFLAGS='-DM_TOTAL=464128 -DN_DIM=1024 -DK_DIM=2048 -DN_STAGES=6'

# Dimension sweep (B200)
./tools/dim_sweep.sh --fast                       # 80 configs, ~1.8hr
./tools/dim_sweep.sh                              # 252 configs, ~5.6hr
./tools/dim_sweep.sh --custom 928256 768,1024 2048,3072,4096

# CUTLASS reference
make fc2-cutlass && ./fc2-cutlass                 # 1.226ms
make fc2-cutlass-strip && ./fc2-cutlass-strip     # 1.152ms

# Comparison and profiling
bash tools/fc2_cutlass_vs_w3.sh
bash tools/ncu_bench.sh && python3 tools/ncu_anova.py
```

## Key files

```
fc2_w3.cu                       # Hand-tuned PTX kernel (ACTIVE)
fc2_cutlass.cu                  # CUTLASS GemmUniversal wrapper (reference)
fc2_hybrid.cu                   # CUTLASS integration experiments (ALL DEAD)
fc2_ldg.cu                      # LDG/STG epilogue experiment (DEAD)
fc2.cu                          # Old 4-warp FC2 kernel (DEAD)
kernel_common.cuh               # Shared infra (pipeline, TMEM, TMA, mbarriers)
kernel_body.cuh                 # Shared kernel body (epilogue_store, persistent_gemm)
Makefile                        # Build rules (sm_100a, DFLAGS for dim override)
tools/dim_sweep.sh              # M/N/K grid search benchmark
tools/fc2_cutlass_vs_w3.sh      # Head-to-head comparison
tools/ncu_bench.sh              # ncu profiling all variants
tools/ncu_anova.py              # ncu data analysis
tools/sass_edit.py              # SASS binary editor + CP-SAT scheduler (~5500 lines)
bench/                          # Microbenchmarks (TMA, MMA, stmatrix, warp scaling)
data/                           # All benchmark + ncu results
```

## ncu profiling datasets

| Dataset | Variants | Notes |
|---|---|---|
| ncu_20260409_015856 | 7 (incl sched/atomic) | Most recent |
| ncu_20260408_032714 | 8 (incl epi1) | Most comprehensive, consistent amplification numbers |
| ncu_20260407_074815 | 5 (no sched/atomic) | Strided dispatch era |
| ncu_20260405_202913 | — | Old contiguous dispatch, obsolete |

## SM100a hardware data (B200-measured)

- STS.128: 27 cyc | LDS.128: 25 cyc @ILP=1, 3.5 cyc @ILP=7
- TMA load: 419 cyc (L2-warm) | TMA store: 197 cyc
- TMEM load (tcgen05.ld.sync): 2 cyc regardless of width/ILP
- MMA K-iteration: 665 cyc (pipelined: 525.6 cyc/iter)
- STS scaling: 10->37 cyc at 8 warps (3.65x contention)
- LDS scaling: 4.5->16 cyc (3.56x)
- FFMA: nearly free (1.36x at 8 warps)
- F2FP: zero contention (flat 2.0 cyc all warp counts)

## Code style

Names say what, comments say why. No single-line `/**/`. No multi-line `//`.
No decorated block comments. Bare `/*` open, undecorated lines, `*/` close.

## Context efficiency

Don't narrate tool calls. Don't echo file contents. Keep explanations proportional.
Parallelize independent tool calls. Use offset/limit for large files.

## Key constraints

- Target: sm_100a (B200, 148 SMs), cta_group::2, 74 clusters
- TMEM: 512 cols, single alloc for double buffering
- SMEM: 228 KB/SM
- All inline PTX in fc2_w3.cu (no CUTLASS dependency)
- OFF_STAGING must be 1024-byte aligned for SWIZZLE_128B
- fence.proxy.async.shared::cta required before TMA store after st.shared
- N_STAGES=6 default (NS5 for N>1536, NS7 doesn't fit)
- PREFILL default on (auto-disabled for K_ITERS<20 in dim_sweep.sh)
- BIAS_SMEM=1 default (-15us free)
- Custom dims require `make -B` (Make doesn't track DFLAGS)
