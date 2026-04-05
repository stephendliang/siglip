# FC2 Optimization — SigLIP2 Vision Encoder

Hand-tuned Blackwell (SM100a) persistent GEMM kernel for FC2 layer of `google/siglip2-base-patch16-224`.
FP8 (E4M3) inputs, BF16 output, tcgen05 MMA, TMA, `cta_group::2` with 2-CTA clusters.
Cross-compiled on CPU VPS, runs on B200. PE and FC1 kernels are done — see `CLAUDE.md.mothballed` for their docs.

## The problem

FC2 shape: [928256, 3072] x [3072, 768]^T + bias + residual. Batch = 4736 images x 196 patches.

| | Strip (GEMM-only) | Fused | Epilogue overhead |
|---|---|---|---|
| **fc2_w3 (ours)** | **1.089ms** | 1.477ms | **388us (36%)** |
| **CUTLASS** | 1.152ms | **1.224ms** | **72us (6.3%)** |

**Our mainloop is 63us FASTER. The entire gap is epilogue overhead: 388us vs 72us = 5.4x worse.**
Theoretical ceiling if we had CUTLASS-quality epilogue: 1.089 + 0.072 = **1.161ms** (5% faster than CUTLASS).

Both use identical tile/cluster config: 256x256x128, 2x1 cluster, cta_group::2, 74 clusters.

## Architecture comparison

| | **fc2_w3 (ours)** | **CUTLASS** |
|---|---|---|
| Warps | 7 (W0-W6, 6 active + 1 idle) | 8 (W0-W7) |
| W0 | TMA Load (A/B) | Scheduler (tile dispatch) |
| W1 | MMA (tcgen05.mma) | MMA |
| W2 | **EpilogueLoad (TMA residual→SMEM)** | MainloopLoad (TMA A/B) |
| W3 / W3 | Epilogue (part of W3-W6) | **EpilogueLoad (TMA residual+bias→SMEM)** |
| W3-W6 / W4-W7 | Epilogue (4 warps, LDS from SMEM) | Epilogue (4 warps, LDS from SMEM) |
| Epilogue source | **LDS from SMEM (W2 pre-loaded via TMA)** | **LDS from SMEM (W3 pre-loaded via TMA)** |
| Epilogue math | BF16 (HFMA2/HADD2) | FP32 (FFMA, F2FP at end) |
| STS scheduling | **4-8 consecutive at block end** | **Interleaved with FFMA/F2FP (6-12 ops between)** |
| @!PT LDS fences | **None (ptxas DCE's all attempts)** | **4 per sub-iter (pipeline drain)** |
| BAR.SYNC.DEFER | **2 per sub-iter (same as CUTLASS)** | **2 per sub-iter (SMEM coherence)** |
| Per-warp epi cost | ~97us/warp | ~18us/warp |

**Both kernels have a dedicated EpilogueLoad warp and LDS from SMEM.** The architecture is structurally the same. Barriers are identical (2 BAR.SYNC.DEFER_BLOCKING per sub-iter each; removing ours = zero effect). The 5.4x per-warp epilogue cost difference is dominated by **STS clustering**:

**STS clustering is the #1 suspect for the 388us epilogue overhead.** Both kernels issue exactly 32 STS.128 per tile per thread — same total work. But the temporal distribution is completely different:
- **CUTLASS**: FP32 math (64 FFMA per 32-col sub-iter) creates a long compute chain. ptxas naturally interleaves STS into this chain with 6-12 ops (12-24 cyc) between each STS. Each STS's SMEM write latency (27 cyc) is fully hidden by the next compute burst.
- **Ours**: BF16 math (HFMA2/HADD2) finishes in ~1/4 the cycles. By the time compute ends, all 4 STS fire back-to-back with 0-2 ops between them. This creates a ~108 cyc burst of SMEM write pressure (4×27 cyc) per chunk, 8 chunks per sub-iter, across 4 warps simultaneously.

**Why this might cause K-loop inflation**: The K-loop's W0 TMA loads and W1 MMA both need SMEM ports. Clustered STS bursts from 4 epilogue warps could saturate SMEM write ports, stalling W0's TMA loads (which also write SMEM) or W1's MMA (which reads SMEM). CUTLASS's spread-out STS avoids this burst pattern.

**Why we can't fix it from source**: ptxas controls STS scheduling. We've tried 6+ source-level approaches (FP32 epilogue, CUTLASS-style C++, per-group STS, different loop structures) — all produce identical STS clustering in SASS. The BF16 compute chain is simply too short to give ptxas room to spread STS apart. The only known way to get CUTLASS-quality STS scheduling is to use CUTLASS's actual FP32 epilogue code path.

The other known difference is @!PT LDS pipeline drain fences (CUTLASS has 4 per sub-iter, we have 0, ptxas DCE's all attempts to add them).

## Current approach: fc2_hybrid.cu

Trying to combine our fast PTX mainloop with CUTLASS's efficient epilogue.

### Phase 1 / Phase 3a — Pure CUTLASS (WORKS, 1.222ms)

`make fc2-hybrid && ./fc2-hybrid`

Wraps CUTLASS `GemmUniversal` with our host setup (FP8 scaling, validation). Uses CUTLASS's full operator() with CLC scheduler, 8 warps, all pipelines. **This is our working reference baseline.** 1.222ms fused, 1.152ms strip.

### Phase 2 — HybridMainloop override (DEAD, same as Phase 1)

`make fc2-hybrid-mma && ./fc2-hybrid-mma`

Inherits `CollectiveMainloop`, overrides `mma()` with unrolled K-loop. Result: 1.220ms = identical to Phase 1. **K-loop unrolling is NOT the 63us mainloop gap.** The gap is architectural (6 vs 8 warps, TMA load pattern), not codegen.

### Phase 3b — 7-warp custom kernel (DEAD, 2.76ms)

`make fc2-hybrid-phase3 && ./fc2-hybrid-phase3`

Custom 7-warp kernel calling CUTLASS's `collective_mainloop.load()/mma()` + `collective_epilogue`. Static tile scheduling (no CLC). Result: **2.763ms fused, 2.629ms strip = 2.3x slower than CUTLASS.** Correct output (valid=1).

Root cause: CUTLASS's `collective_mainloop.mma()` internal pipeline ops assume 8-warp threading. `threadIdx.x % ThreadCount` rotation, pipeline arrival counts, umma_arrive multicast — all technically correct but catastrophic warp scheduling at 7 warps.

### Phase 4 — 8-warp our-dispatch + CUTLASS mainloop (DEAD, 2.77ms)

`make fc2-hybrid-phase4 && ./fc2-hybrid-phase4`

8-warp kernel matching CUTLASS's exact warp mapping (W0=Sched, W1=MMA, W2=Load, W3=EpiLoad, W4-7=Epilogue). All 7 pipelines initialized (MainloopPipeline, EpiLoadPipeline, EpiStorePipeline, LoadOrderBarrier, AccumulatorPipeline, CLCPipeline, CLCThrottlePipeline). Static tile scheduling replaces CLC queries. Result: **2.764ms fused, 2.627ms strip = same as Phase 3b.**

Two versions tried:
- v1: Custom warp mapping, 5 pipelines -> 2.770ms
- v2: CUTLASS-exact warp mapping, all 7 pipelines -> 2.764ms (no improvement)

**Root cause unknown.** Something about static `for ti=t0..t1` loop vs CUTLASS's `do { ... scheduler.fetch_next_work() ... } while (valid)` pattern. Both produce correct output.

### 8-warp strip test (PASSED)

`make fc2-w3-8w && ./fc2-w3-8w` (adds idle 8th warp to fc2_w3)

fc2-w3-8w strip=1.086ms vs fc2-w3 strip=1.088ms. **8 warps does NOT hurt mainloop speed.** The theoretical ceiling of 1.161ms remains viable.

### Next step: ncu profiling to identify the contended hardware resource

We know WHAT: epilogue warps inflate the K-loop by 36% (388us). We know it's NOT any single operation (strip decomposition proved TMA-load-only, TMA-store-only, and compute-only ALL produce identical overhead). We know it's NOT dispatch pressure, TMEM timing, or register file occupancy (all individually ruled out via busy-wait and REG_PAD experiments). The overhead is from epilogue warps actively executing instructions, creating aggregate pressure on shared hardware. But we don't know WHICH resource.

The root cause is the same across all our kernels (old fc2 at 120us/warp, fc2_w3 at 98us/warp) because they share the same structural deficiencies vs CUTLASS:

**What our epilogue does wrong (SASS-verified):**
Both kernels have the same architecture (EpilogueLoad warp + LDS from SMEM) and identical barriers (2 BAR.SYNC.DEFER_BLOCKING per sub-iter). The differences:
1. **STS clustering (primary suspect)**: Our STS.128 are clustered 4-8 back-to-back with 0-2 ops between them. CUTLASS interleaves 6-12 FFMA/F2FP ops between each STS. Same 32 STS/tile, but our burst pattern creates ~108 cyc SMEM write storms per chunk across 4 warps simultaneously — likely saturating SMEM ports and stalling the K-loop's W0 TMA loads and W1 MMA reads. See architecture comparison above for full analysis.
2. **Missing @!PT LDS pipeline drain fences**: CUTLASS has 4 per sub-iter after STS, before MEMBAR+FENCE, ensuring STS are visible in SMEM before TMA store reads it. ptxas DCE's all our attempts to add them (3 approaches tried).

**ncu must answer: which of these creates the K-loop contention?** Candidates:
- **SMEM port contention**: clustered STS create burst write pressure on shared memory ports, interfering with K-loop SMEM activity. ncu metrics: `shared_pct`, `smem_st_wavefronts`, `smem_ld_wavefronts`.
- **Warp scheduler starvation**: 6 active warps (vs 2 in strip) competing for issue slots, starving the MMA warp of execution bandwidth. ncu metrics: `warps_eligible`, `not_selected`.
- **TMA unit saturation**: epilogue TMA stores + W2 residual TMA loads compete with W0's TMA loads for MIO queue slots. ncu metrics: `mio_throttle`, `mio_pq_read/write_cycles`.
- **DRAM/L2 bandwidth saturation**: W2's TMA residual loads add ~1.36GB of reads across 74 clusters, competing with W0's TMA loads for A/B tiles. ncu metrics: `long_scoreboard`, `dram_pct`, `global_ld_wavefronts`.

These are NOT mutually exclusive — it's likely a combination, which is why no single strip variant showed improvement. The ncu data will reveal the proportions.

Run `./tools/fc2_ncu_bench.sh` on the B200. See `docs/ncu_diagnosis_guide.txt` for detailed interpretation of each metric and what to look for in each comparison pair (Q1: strip vs fused, Q2: w3 vs CUTLASS, Q3: Phase 4 broken).

## What we know works

- Our PTX mainloop is fastest (1.089ms strip vs CUTLASS 1.152ms)
- 8 warps doesn't hurt mainloop (1.086ms with idle W7)
- CUTLASS's full operator() with CLC produces 1.222ms (Phase 1/3a)
- Theoretical ceiling: 1.089 + 0.072 = 1.161ms

## What doesn't work

Calling CUTLASS's `collective_mainloop.load()/mma()` from our own dispatch loop = 2.3x slower, regardless of:
- Warp count (7 or 8)
- Pipeline init (5 or 7 pipelines)
- Warp-to-role mapping (custom or CUTLASS-exact)
- Grid linearization approach

## Exhaustive dead-end list

### Source-level epilogue attempts on fc2_w3 (all ~1.48ms)

- CUTLASS_LOOP=1/2/3: changes SASS interleaving, zero perf
- FP32_EPILOGUE: zero perf
- CUTLASS_EPILOGUE (FP32 per-group STS, 113 regs): identical STS clustering in SASS
- CPP_EPILOGUE: byte-identical SASS to asm volatile
- CUTE_STORE (C++ pointer stores): byte-identical SASS
- @!PT LDS fences (3 approaches): all DCE'd by ptxas
- cvta.shared + generic store: ptxas DCEs all 64 stores
- NO_PRE_STORE_BAR + NO_POST_STORE_BAR: 17->1 BAR.SYNC, zero perf (we already HAVE 2 BAR.SYNC.DEFER_BLOCKING per sub-iter, same as CUTLASS — removing them changes nothing)
- NUM_EPI_STAGES=3/4: -6 to -23us (noise)
- All combinations of above: noise
- **ptxas STS scheduling is immutable from any source-level approach**

### Architectural attempts on fc2 (old kernel, all ~1.635ms at 4 warps)

- BAR.SYNC serialization (EPI_SYNC, EPI_BAR_PASS, EPI_BAR_CHUNK)
- Combinatorial strips (only_tma_load, only_tma_store, only_compute)
- FP32_EPILOGUE, stagger (500/2000 cyc), pass count, store deferred
- EPI_LOAD_WARP (+13% regression), EPI_PIPELINE (+0.8% noise)
- W0_RES_FULL (+15% catastrophic), W0_RES_PREFETCH (neutral)
- STAGES_C=2 (neutral), STAGES_C+EPI_REUSE_SMEM (broken), STAGES_C+PRE_COMBINE (broken)
- NOP_EPILOGUE, EPI_DELAY, REG_PAD: all ruled out individual causes
- SASS fatbin-patch: CP-SAT schedules fine, patched binaries crash (illegal instruction)

### Hypothesis isolation (all ruled out)

- Warp scheduler dispatch pressure: 10k-cycle busy-wait = zero effect
- TMEM release timing: 10k-cycle delay = zero effect
- Register file occupancy: 186 regs (54.5% RF) allocated but unused = strip speed
- **Overhead is from epilogue warps actively executing memory/compute instructions**
- Linear at ~98us/warp (fc2_w3) / ~120us/warp (fc2 old). No threshold, no saturation.

### CUTLASS integration attempts (fc2_hybrid.cu)

- Phase 2 HybridMainloop: K-loop unrolling = zero effect (1.220 vs 1.222ms)
- Phase 3b 7-warp custom: CUTLASS mainloop dies at 7 warps (2.76ms)
- Phase 4 8-warp static dispatch: same 2.77ms despite matching all init
- CUTLASS stage count sweep (3-7): all identical. Do NOT repeat.
- StaticPersistentScheduler: CLC is NOT the mainloop gap (+7us noise)

## Kernel structure (fc2_w3.cu)

Warp-specialized, 7 warps (224 threads), `cta_group::2`, `__cluster_dims__(2,1,1)`:

- **W0**: TMA async bulk loads (A + B tiles)
- **W1**: TMEM alloc + tcgen05.mma.cta_group::2 accumulation
- **W2**: EpilogueLoad — TMA loads residual into SMEM (circular 2-stage pipe, previous tile)
- **W3-W6**: Overlapped epilogue (4 warps) — LDS residual+bias from SMEM, TMEM ld, epilogue math, CVT, STS, TMA store
- **W7**: Idle (8-warp mode via NUM_IDLE_WARPS=1, only exists when flag set)

Tile: 256x256x128, TMEM 512 cols double-buffered, 5-stage SMEM pipeline, 24 K-iterations.

## SM100a hardware data (B200-measured)

- STS.128: 27 cyc | LDS.128: 25 cyc @ILP=1, 3.5 cyc @ILP=7
- TMA load: 419 cyc (L2-warm) | TMA store: 197 cyc
- TMEM load (tcgen05.ld.sync): 2 cyc regardless of width/ILP
- MMA K-iteration: 665 cyc (pipelined: 525.6 cyc/iter)
- STS scaling: 10->37 cyc at 8 warps (3.65x contention)
- LDS scaling: 4.5->16 cyc (3.56x)
- FFMA: nearly free (1.36x at 8 warps)
- F2FP: zero contention (flat 2.0 cyc all warp counts)

## Build and run

```bash
# Primary kernel
make fc2-w3 && ./fc2-w3                                    # fused (1.477ms)
make fc2-w3 DFLAGS=-DSTRIP_EPILOGUE && ./fc2-w3            # strip (1.089ms)
make fc2-w3-8w && ./fc2-w3-8w                              # 8-warp strip test

# Hybrid (CUTLASS integration)
make fc2-hybrid && ./fc2-hybrid                            # Phase 1/3a (1.222ms) WORKS
make fc2-hybrid-mma && ./fc2-hybrid-mma                    # Phase 2 (1.220ms) = Phase 1
make fc2-hybrid-phase3 && ./fc2-hybrid-phase3              # Phase 3b (2.76ms) BROKEN
make fc2-hybrid-phase4 && ./fc2-hybrid-phase4              # Phase 4 (2.77ms) BROKEN

# CUTLASS reference
make fc2-cutlass && ./fc2-cutlass                          # CUTLASS GemmUniversal (1.224ms)
make fc2-cutlass-strip && ./fc2-cutlass-strip              # CUTLASS strip (1.152ms)

# Head-to-head comparison
bash tools/fc2_cutlass_vs_w3.sh                            # all variants

# Analysis (no GPU)
python3 tools/analyze_sweep.py data/session_*/sweep_fc2.csv
python3 tools/sass_analysis.py --cubin fc2 --deps
```

## Key files

```
fc2_w3.cu               # Our hand-tuned PTX kernel (ACTIVE)
fc2_hybrid.cu           # CUTLASS integration experiments (Phases 1-4)
fc2_cutlass.cu          # Pure CUTLASS GemmUniversal wrapper
fc2.cu                  # Old FC2 kernel (predecessor to fc2_w3)
kernel_common.cuh       # Shared infrastructure (pipeline, TMEM, TMA, mbarriers)
kernel_body.cuh         # Shared kernel body (epilogue_store, persistent_gemm)
Makefile                # Build rules (sm_100a)
tools/                  # Sweep scripts, SASS tools, benchmarks
bench/                  # Microbenchmarks (TMA, MMA, warp scaling, calibration)
data/                   # All benchmark results
docs/                   # Experiment logs, proposals
CLAUDE.md.mothballed    # Full docs for PE/FC1 (done), grid search, calibration suites
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
- N_STAGES=5 mandatory (10% gap vs NS4)
- BIAS_SMEM=1 default (-15us free)
