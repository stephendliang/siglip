# FC2 Optimization — SigLIP2 Vision Encoder

Hand-tuned Blackwell (SM100a) persistent GEMM kernel for FC2 layer of `google/siglip2-base-patch16-224`.
FP8 (E4M3) inputs, BF16 output, tcgen05 MMA, TMA, `cta_group::2` with 2-CTA clusters.
Cross-compiled on CPU VPS, runs on B200. PE and FC1 kernels are done — see `CLAUDE.md.mothballed` for their docs.

## The problem

FC2 shape: [928256, 3072] x [3072, 768]^T + bias + residual. Batch = 4736 images x 196 patches.

| | Fused (D=A×B+bias+res) | vs CUTLASS |
|---|---|---|
| **fc2_w3 NS6+PREFILL (ours)** | **1.109ms** | **-10.1%** |
| **CUTLASS** | 1.233ms | baseline |

**We beat CUTLASS by 124us (10.1%).** Our fused kernel (1.109ms) is faster than CUTLASS's *strip/GEMM-only* (1.157ms).

Key breakthrough: N_STAGES=6 (6-stage mainloop pipeline, 227KB of 228KB SMEM) + PREFILL (skip epilogue_mbar wait, rely on TMEM double-buffering). N_STAGES=6 alone gives 1.119ms; PREFILL adds another 10us.

### Previous gap analysis (NS5, pre-2026-04-08)

At N_STAGES=5: w3_fused=1.242ms vs CUTLASS=1.233ms = 9us gap (0.7%). GEMM-only was 46us slower (1.198 vs 1.152ms) due to DRAM read amplification (1.52x). The deeper pipeline (NS6) hides this latency by letting W0 run further ahead of W1.

### Previous flawed analysis (corrected 2026-04-07)

Old strip comparison was apples-to-oranges: our STRIP_EPILOGUE skips ALL output (valid=0, 0.01GB write, 1.016ms) while CUTLASS strip writes full output (valid=1, 1.40GB write, 1.152ms). The "135us mainloop advantage" and "226us vs 72us epilogue overhead" were artifacts of comparing different workloads. GEMM_ONLY mode (commit 9965903) fixes this by writing D=BF16(A×B).

Both use identical tile/cluster config: 256x256x128, 2x1 cluster, cta_group::2, 74 clusters.

## DRAM read amplification (OPEN PROBLEM)

### The data (ncu B200-verified, data/ncu_20260407_074815/)

```
                    Theoretical    w3_strip    w3_fused    cutlass_strip    cutlass_fused
dram_read              —            4.33GB      4.99GB       2.85GB           4.28GB
amplification          —            1.52x       1.17x        1.00x            1.00x
dram_write             —            0.01GB      1.42GB       1.40GB           1.40GB
dram_pct               —            31.81       43.50        30.59            40.40
lts_hit_rate           —            62.13       55.59        62.84            55.85
```

Theoretical minimum DRAM read: strip = A + B = 2.85GB, fused = A + B + residual + bias = 4.28GB.

### Matrix sizes and access patterns

```
A: [928256, 3072] FP8 = 2.85GB — dominates. 3626 M-tile-rows × 768KB each. FAR exceeds 96MB L2.
B: [768, 3072]    FP8 = 2.25MB — fits entirely in L2. Zero amplification concern.
Residual: [928256, 768] BF16 = 1.43GB — per-tile slice is 128KB (256×256×2B).
Bias: [768] BF16 = 1.5KB — trivial.

Tile grid: TILES_M=3626, TILES_N=3, TOTAL_TILES=10878, K_ITERS=24.
Per tile: A load = 768KB (256 rows × 3072 cols × 1B, across 24 K-iters).
          B load = 768KB (256 cols × 3072 rows × 1B, across 24 K-iters).
L2 cache: 96MB = ~125 M-tile-rows of A.
Wavefront: 74 clusters active → ~25 M-rows simultaneous = 19.2MB of A (fits in L2).
```

**A reuse opportunity**: tiles (tm,0), (tm,1), (tm,2) share identical A data (768KB). Load once, use 3x. B is trivially cached (2.25MB total, all tiles at same tn share it).

### Strip excess: 4.33GB - 2.85GB = 1.48GB (52% of A re-read from DRAM)

STRIP_EPILOGUE skips W2 residual loads and ALL output writes (valid=0, 0.01GB write). The 4.33GB is purely A+B reads with 1.52x amplification. CUTLASS strip reads exactly 2.85GB but also writes 1.40GB output — total DRAM traffic is similar (4.34 vs 4.25GB), but ours is all reads while theirs is reads+writes.

**GEMM_ONLY** (1.198ms, valid=1) is the correct apples-to-apples comparison vs CUTLASS strip (1.152ms). It writes full output, confirming the 46us GEMM+output gap is real, not a strip measurement artifact.

### What does NOT explain the amplification

**Tile dispatch ordering (3 experiments, 2026-04-07, all B200-verified):**

| Dispatch | Strip | Fused | Description |
|---|---|---|---|
| Strided (baseline) | 1.016ms | 1.242ms | stride=74, wavefront ~25 rows |
| N-batch (reverted) | — | 1.395ms (+12%) | all clusters sync on same tn |
| Phase-offset N-batch (reverted) | 1.079ms (+6%) | 1.374ms (+11%) | per-cluster A reuse, 74-row wavefront too wide |
| Group-3 (current) | 1.028ms (neutral) | 1.242ms (neutral) | fixed tn/cluster, 25-row wavefront |

**CLC vs static dispatch:** CUTLASS StaticPersistentScheduler = +7us noise vs CLC. Strict ordering is NOT why CUTLASS gets 1.00x.

### What does NOT explain it (INVESTIGATED)

- **TMA descriptor cache policy hints (2026-04-08):** Tested EVICT_FIRST/EVICT_LAST/EVICT_NORMAL on A, B, and residual TMA loads. Zero effect (strategies 1/3 neutral, strategy 2 regression). CUTLASS also uses EVICT_NORMAL default. Amplification is L2 capacity, not eviction policy.

### What might explain it (UNINVESTIGATED)

- **TMA load scheduling** — CUTLASS W2 loads A/B (separate warp from scheduler), our W0 loads A/B (also does mbarrier waits). Different TMA issue timing could affect L2 hit patterns
- **Pipeline depth interaction** — 6-stage pipeline means 6 K-iters of A in flight (6 × 32KB = 192KB/cluster, 27.7MB total across 148 CTAs). L2 sector conflicts from concurrent TMA streams?
- **8 vs 7 warp effect on TMA scheduling** — more warps = different SM scheduling of TMA load warp

### Why amplification matters for epilogue too

Extra DRAM traffic competes for memory bandwidth with epilogue TMA stores and residual loads. Reducing amplification would directly improve fused performance even if epilogue code is unchanged.

## Epilogue: cross-warp STS clustering (NOT the dominant problem)

### Corrected understanding (2026-04-07)

Our epilogue fusion (residual+bias) costs **44us** vs CUTLASS's **72us** — ours is 39% cheaper. The old "226us vs 72us" comparison was invalid because our old strip (1.016ms) skipped output writes that CUTLASS strip (1.152ms) included. The 182us difference was mostly output write cost, not epilogue quality.

Despite costing less overall, our epilogue has worse STS scheduling — it just doesn't matter much because BF16 compute is fast enough to compensate. Fixing STS clustering could still yield gains, but the primary optimization target is the **46us GEMM+output gap** (DRAM amplification).

### Architecture comparison

| | **fc2_w3 (ours)** | **CUTLASS** |
|---|---|---|
| Warps | 7 (W0-W6, 6 active + 1 idle) | 8 (W0-W7) |
| W0 | TMA Load (A/B) | Scheduler (tile dispatch) |
| W1 | MMA (tcgen05.mma) | MMA |
| W2 | **EpilogueLoad (TMA residual→SMEM)** | MainloopLoad (TMA A/B) |
| W3 / W3 | Epilogue (part of W3-W6) | **EpilogueLoad (TMA residual+bias→SMEM)** |
| W3-W6 / W4-W7 | Epilogue (4 warps, LDS from SMEM) | Epilogue (4 warps, LDS from SMEM) |
| Epilogue math | BF16 (HFMA2/HADD2) | FP32 (FFMA, F2FP at end) |
| STS scheduling | **4-8 consecutive at block end** | **Interleaved with FFMA/F2FP (6-12 ops between)** |
| Fusion cost | 44us | 72us |

### STS clustering (real but not dominant)

All 4 epilogue warps (W3-W6) hit STS stores simultaneously because BF16 compute (~1/4 of CUTLASS's FP32) is too short. ncu fused-vs-fused shows barrier stalls +753% (273K vs 32K). But our net fusion cost is still cheaper (44us vs 72us) because BF16 math is faster than CUTLASS's FP32 math.

**What does NOT matter (all proven on B200):**
- Intra-warp STS ordering (CP-SAT finds 75-83% better, CUTLASS has same gap, doesn't care)
- Instruction class (stmatrix = STS.128, identical throughput at all warp counts)
- Barrier count (removing all = zero effect, stalls are a symptom)
- Source-level anything (6+ approaches, all identical SASS — ptxas controls STS scheduling)
- SELF_LOAD + SELF_STAGGER (eliminated cross-warp sync + nanosleep, ZERO effect)

### Remaining epilogue approaches (lower priority than DRAM amplification)

**Inter-warp stagger (tools/interwarp_stagger.py):** YIELD patching for warp phase offset. NEVER TESTED on B200.

**LDS_DRAIN (fc2_w3.cu ifdef):** 4x drain loads after STS. NEVER TESTED on B200.

## fc2_hybrid.cu (ALL PHASES DEAD)

All attempts to combine our PTX mainloop with CUTLASS's epilogue have failed.

- **Phase 1/3a — Pure CUTLASS (WORKS, 1.224ms):** `make fc2-hybrid && ./fc2-hybrid`. Reference baseline.
- **Phase 2 — HybridMainloop (DEAD, 1.220ms = Phase 1):** K-loop unrolling is NOT the mainloop gap.
- **Phase 3b — 7-warp custom (DEAD, 2.76ms):** CUTLASS mainloop breaks at non-8-warp counts.
- **Phase 4 — 8-warp our-dispatch (DEAD, 2.77ms):** Static tile loop = 2.3x slower, root cause unknown.
- **8-warp strip test (PASSED):** 1.086ms vs 1.088ms (old MMA-only strip). 8 warps does NOT hurt MMA loop.

## Exhaustive dead-end list

### Source-level epilogue attempts on fc2_w3 (all ~1.48ms pre-strided-dispatch)

CUTLASS_LOOP=1/2/3, FP32_EPILOGUE, CUTLASS_EPILOGUE, CPP_EPILOGUE, CUTE_STORE, @!PT LDS fences, cvta.shared, NO_PRE/POST_STORE_BAR, NUM_EPI_STAGES=3/4, stmatrix, all combinations. **ptxas STS scheduling is immutable from any source-level approach.**

### Tile dispatch reordering (2026-04-07, all on fc2_w3)

- N-batch: all clusters sync on same tn → +12% regression (TMA store contention)
- Phase-offset N-batch: per-cluster A reuse, 74-row wavefront → +6-11% regression (L2 too wide)
- Group-3: fixed tn/cluster, 25-row wavefront → delta neutral (cross-cluster L2 timing unchanged)
- **Tile dispatch ordering cannot fix DRAM amplification.** CLC vs static also irrelevant (+7us noise).

### Architectural attempts on old fc2.cu (all ~1.635ms at 4 warps)

BAR.SYNC serialization, FP32, NOP_EPILOGUE, EPI_DELAY, REG_PAD, EPI_LOAD_WARP, EPI_PIPELINE, W0_RES_FULL/PREFETCH, STAGES_C+EPI_REUSE_SMEM (broken), stmatrix.

### CUTLASS integration (fc2_hybrid.cu)

Phase 2 (=Phase1), Phase 3b (2.76ms), Phase 4 (2.77ms), stage sweep (3-7 identical), StaticPersistentScheduler (+7us noise).

### Cross-warp temporal

SELF_LOAD (per-warp TMA, no sync), SELF_STAGGER (nanosleep 50-200ns). ZERO effect.

### TD=5 CLC hardware dispatch (2026-04-07)

`clusterlaunchcontrol.try_cancel` — two attempts, both deadlocked on B200. Attempt 1: dedicated W7 scheduler (phase desync). Attempt 2: inline in main loop (unknown cause, compute-sanitizer also hangs). CLC execution model incompatible with persistent kernel loop. Fully reverted.

### TD=6 inline atomic dispatch (2026-04-08)

W0 does atomicAdd at tile boundary instead of dedicated W7 scheduler warp. 1.370ms — dead. W0 blocked doing dispatch (atomicAdd + CTA1 epoch spin + mbar broadcast) delays TMA loads for next tile.

### PREFILL at N_STAGES=5 (2026-04-08)

Skip epilogue_mbar wait in W1, rely on TMEM double-buffering. 1.242ms = zero effect at NS5 (K-loop >> epilogue, mbar wait was never the bottleneck). Only effective at NS6 where deeper pipeline exposes the 10us overlap.

### L2 cache hints (2026-04-08)

Added `.L2::cache_hint` to TMA loads (`-DL2_HINTS=N`). Three strategies: (1) residual EVICT_FIRST = neutral, (2) A EVICT_FIRST + B EVICT_LAST = regression, (3) A EVICT_LAST + residual EVICT_FIRST = neutral. DRAM amplification is a tile-ordering capacity problem, not L2 eviction policy. Hints can't fix capacity misses.

### Universally dead

Clock freq, SASS intra-warp reorder (wrong target), multicast epilogue, R2UR reduction, BATCH_MMA.

## ncu profiling summary

Most recent: `data/ncu_20260407_074815/` (strided dispatch, current code).
Older: `data/ncu_20260405_202913/` and `data/ncu_20260407_005837/` (old contiguous dispatch, 1.088ms strip / 1.479ms fused — obsolete numbers but has phase4/hybrid data).

### w3 fused - strip delta (FLAWED — different workloads, ncu_20260407_074815)

**WARNING**: w3_strip has valid=0 (no output, 0.01GB write) while w3_fused writes 1.42GB. The delta includes output write cost, not just epilogue compute. Do NOT interpret as "epilogue overhead."

| Metric | Strip | Fused | Δ% | Note |
|---|---|---|---|---|
| short_scoreboard | 1.2K | 138.1K | +11,408% | Includes output STS+TMA store |
| dram_write | 0.01GB | 1.42GB | — | Strip writes nothing |
| dram_read | 4.33GB | 4.99GB | +15% | Fused adds residual reads |

### CUTLASS fused - strip delta (valid comparison, ncu_20260407_074815)

CUTLASS strip writes full output (1.40GB, valid=1), so this delta isolates residual+bias fusion cost.

| Metric | Strip | Fused | Δ% |
|---|---|---|---|
| short_scoreboard | 30.3K | 134.8K | +345% |
| barrier | 93.1K | 32.0K | **-66%** |
| long_scoreboard | 2.41M | 2.70M | +12% |
| dram_read | 2.85GB | 4.28GB | +50% |

CUTLASS epilogue DECREASES barrier stalls — warps fill time that would otherwise be idle.

### w3 fused vs cutlass fused (head-to-head, ncu_20260407_074815)

| Stall | w3_fused | cutlass_fused | Δ% |
|---|---|---|---|
| **barrier** | **273.1K** | **32.0K** | **+753%** |
| **wait** | **379.3K** | **228.1K** | **+66%** |
| short_scoreboard | 138.1K | 134.8K | +2.4% |
| long_scoreboard | 2.25M | 2.70M | -17% |
| dram_read | 4.99GB | 4.28GB | +17% |
| dram_pct | 43.5% | 40.4% | — |
| sm_pct | 90.6% | 94.9% | — |

Barrier stalls (+241K, +753%) dominate. Symptom of STS clustering, not a cause (removing barriers = zero effect). We have 17% more DRAM reads AND 753% more barrier stalls.

### Strip comparison (FLAWED — asymmetric workloads, ncu_20260407_074815)

**WARNING**: w3_strip = MMA-only (no output, valid=0, 48 regs), cutlass_strip = full GEMM pipeline (output written, valid=1, 186 regs, 141M instructions vs 44M). NOT comparable. w3_strip runs 2 active warps; cutlass_strip runs 8.

| Metric | w3_strip | cutlass_strip | Note |
|---|---|---|---|
| dram_read | **4.33GB (1.52x)** | **2.85GB (1.00x)** | A amplification is real |
| dram_write | 0.01GB | **1.40GB** | w3 skips all output |
| inst_executed | 43.7M | 141.7M | 3.2x fewer (no epilogue) |
| regs_per_thread | 48 | 186 | No epilogue regs |

The A read amplification (1.52x) is real regardless of the output asymmetry. Use GEMM_ONLY (1.198ms) for timing comparison vs cutlass_strip (1.152ms).

### DRAM amplification across code versions

| Dataset | Code version | w3_strip read | w3_fused read |
|---|---|---|---|
| ncu_20260405_202913 | Old contiguous dispatch | 3.33GB (1.17x) | 6.30GB (1.47x) |
| ncu_20260407_074815 | Strided dispatch | 4.33GB (1.52x) | 4.99GB (1.17x) |
| (both datasets) | CUTLASS | 2.85GB (1.00x) | 4.28GB (1.00x) |

Strided dispatch traded strip amplification (1.17x→1.52x) for massive fused improvement (1.47x→1.17x). Net result: strip 1.088→1.016ms, fused 1.479→1.242ms.

## Kernel structure (fc2_w3.cu)

Warp-specialized, 7 warps (224 threads), `cta_group::2`, `__cluster_dims__(2,1,1)`:

- **W0**: TMA async bulk loads (A + B tiles)
- **W1**: TMEM alloc + tcgen05.mma.cta_group::2 accumulation
- **W2**: EpilogueLoad — TMA loads residual into SMEM (circular 2-stage pipe, previous tile)
- **W3-W6**: Overlapped epilogue (4 warps) — LDS residual+bias from SMEM, TMEM ld, epilogue math, CVT, STS, TMA store
- **W7**: Idle (8-warp mode via NUM_IDLE_WARPS=1, only exists when flag set)

Tile: 256x256x128, TMEM 512 cols double-buffered, 5-stage SMEM pipeline, 24 K-iterations.

### Tile dispatch (current: Group-3, delta neutral vs strided baseline)

Each cluster handles a fixed N-tile (cluster_id % 3), striding through M-rows. 25/25/24 clusters per tn. Wavefront = ~25 M-rows = 19.2MB. Code at lines 824-829 (init) and 965-970 (loop).

## SM100a hardware data (B200-measured)

- STS.128: 27 cyc | LDS.128: 25 cyc @ILP=1, 3.5 cyc @ILP=7
- TMA load: 419 cyc (L2-warm) | TMA store: 197 cyc
- TMEM load (tcgen05.ld.sync): 2 cyc regardless of width/ILP
- MMA K-iteration: 665 cyc (pipelined: 525.6 cyc/iter)
- STS scaling: 10→37 cyc at 8 warps (3.65x contention)
- LDS scaling: 4.5→16 cyc (3.56x)
- FFMA: nearly free (1.36x at 8 warps)
- F2FP: zero contention (flat 2.0 cyc all warp counts)

## Build and run

```bash
# Primary kernel (default: N_STAGES=6, PREFILL)
make fc2-w3 && ./fc2-w3                                    # fused (1.109ms, beats CUTLASS by 10%)
make fc2-w3-gemm && ./fc2-w3-gemm                          # GEMM-only D=A×B (valid=1)
make fc2-w3-strip && ./fc2-w3-strip                        # MMA-only, no output (valid=0)
make fc2-w3 DFLAGS='-DN_STAGES=5' && ./fc2-w3              # old NS5 for comparison (1.119ms)
make fc2-w3 DFLAGS=-DNO_PREFILL && ./fc2-w3                # disable PREFILL (1.119ms)

# Hybrid (CUTLASS integration)
make fc2-hybrid && ./fc2-hybrid                            # Phase 1/3a (1.224ms) WORKS
make fc2-hybrid-mma && ./fc2-hybrid-mma                    # Phase 2 (1.220ms) = Phase 1
make fc2-hybrid-phase3 && ./fc2-hybrid-phase3              # Phase 3b (2.76ms) BROKEN
make fc2-hybrid-phase4 && ./fc2-hybrid-phase4              # Phase 4 (2.77ms) BROKEN

# CUTLASS reference
make fc2-cutlass && ./fc2-cutlass                          # CUTLASS GemmUniversal (1.224ms)
make fc2-cutlass-strip && ./fc2-cutlass-strip              # CUTLASS strip (1.152ms)

# Head-to-head comparison
bash tools/fc2_cutlass_vs_w3.sh                            # all variants

# ncu profiling
bash tools/ncu_bench.sh                                    # run ncu on all variants
python3 tools/ncu_anova.py                                 # analyze ncu data

# Analysis (no GPU)
python3 tools/analyze_sweep.py data/session_*/sweep_fc2.csv
python3 tools/sass_analysis.py --cubin fc2 --deps
```

## Key files

```
fc2_w3.cu                       # Our hand-tuned PTX kernel (ACTIVE)
fc2_hybrid.cu                   # CUTLASS integration experiments (Phases 1-4, ALL DEAD)
fc2_cutlass.cu                  # Pure CUTLASS GemmUniversal wrapper (1.224ms reference)
fc2.cu                          # Old FC2 kernel (predecessor to fc2_w3)
kernel_common.cuh               # Shared infrastructure (pipeline, TMEM, TMA, mbarriers)
kernel_body.cuh                 # Shared kernel body (epilogue_store, persistent_gemm)
Makefile                        # Build rules (sm_100a)
tools/sass_edit.py              # SASS binary editor + CP-SAT intra-warp scheduler (~5500 lines)
tools/interwarp_stagger.py      # Inter-warp YIELD patching (UNTESTED on B200)
tools/test_interwarp.sh         # Inter-warp B200 test harness (UNTESTED)
tools/test_sass_patch.sh        # Intra-warp SASS patch bisection (levels 0-5 pass, 6 crashes)
tools/ncu_bench.sh              # ncu profiling all variants
tools/ncu_anova.py              # ncu data analysis
tools/                          # Sweep scripts, SASS tools, benchmarks
bench/                          # Microbenchmarks (TMA, MMA, stmatrix, warp scaling, calibration)
data/                           # All benchmark results
docs/                           # Experiment logs, proposals
```

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
- N_STAGES=6 default (1.109ms, 10% faster than CUTLASS; NS5=1.242ms, NS7 doesn't fit in 228KB SMEM)
- BIAS_SMEM=1 default (-15us free)
