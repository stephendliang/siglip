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

**CROSS-WARP STS clustering is the proven root cause of the 388us epilogue overhead.** Both kernels issue exactly 32 STS.128 per tile per thread — same total work. The problem is NOT how each warp schedules its own STS instructions internally. The problem is that all 4 epilogue warps (W3-W6) blast STS at the same time, creating synchronized SMEM write pressure bursts.

**Why warps synchronize (the real problem):**
- Our BF16 math (HFMA2/HADD2) finishes in ~1/4 the cycles of CUTLASS's FP32 (FFMA).
- All 4 warps start each chunk at roughly the same time (after BAR.SYNC).
- With so little compute, all 4 warps reach their STS stores nearly simultaneously.
- Result: 4 warps × 4 STS.128 = 16 concurrent SMEM writes per chunk, repeated 8 chunks per sub-iter.
- CUTLASS: FP32 compute is 4x longer, so warps naturally drift apart. STS stores from different warps are temporally spread across the compute window.

**Why this kills K-loop throughput:** W0 TMA loads and W1 MMA both need SMEM ports. Synchronized STS bursts from 4 epilogue warps saturate SMEM write ports, stalling W0's TMA loads (which also write SMEM). ncu confirms: barrier stalls +860%, short_scoreboard +11,261% — all from SMEM port contention across warps.

**What does NOT matter (proven):**
- Intra-warp STS ordering: CP-SAT finds 75-83% better schedules per-warp, but CUTLASS has the same 5x intra-warp scheduling gap and doesn't care. The issue is inter-warp temporal alignment.
- Instruction class: stmatrix (STSM) has identical throughput to STS.128 at all warp counts (B200-verified 2026-04-05). Contention is architectural, not instruction-dependent.
- Barrier count: removing barriers = zero perf effect. Barrier stalls are a SYMPTOM (uneven warp progress from STS bursts), not a cause.
- @!PT LDS pipeline drain fences: CUTLASS has 4 per sub-iter, we have 0, ptxas DCE's all attempts to add them.

**Why we can't fix it from source**: ptxas controls STS scheduling. We've tried 6+ source-level approaches (FP32 epilogue, CUTLASS-style C++, per-group STS, different loop structures) — all produce identical STS clustering in SASS. The BF16 compute chain is simply too short to give ptxas room to spread STS apart. The only known way to get CUTLASS-quality STS scheduling is to use CUTLASS's actual FP32 epilogue code path.

## Remaining approaches

### SASS binary patching (tools/sass_edit.py, tools/interwarp_stagger.py)

The only remaining path to beat CUTLASS. Directly attacks STS clustering at the binary level.

**Intra-warp scheduling (tools/sass_edit.py CP-SAT):** Works locally (75-83% stall reduction), bisection test levels 0-5 pass on B200, level 6 (full epilogue) crashes. But **intra-warp scheduling is futile** — the problem is cross-warp STS temporal alignment, not per-warp instruction ordering. CUTLASS has the same 5x intra-warp scheduling gap and doesn't care.

**Inter-warp stagger (tools/interwarp_stagger.py):** Replaces epilogue NOPs with YIELD instructions to create temporal phase offset between warp pairs {W3,W5} and {W4,W6}. Two modes: yield-only (all warps) and predicated (@P6 = warp parity). Test harness at tools/test_interwarp.sh. **Never tested on B200** — tools committed but `data/interwarp_*` doesn't exist.

**LDS_DRAIN (fc2_w3.cu ifdef):** 4x LDS drain loads after STS to keep LSU pipeline busy and prevent STS clustering. Committed (bc121f9). **Never tested on B200.**

### stmatrix migration (DEAD, 2026-04-05)

`bench/stmatrix_bench.cu` proved stmatrix has identical throughput to STS.128 at all warp counts. Contention is architectural (SMEM ports), not instruction-class-dependent. See `docs/stmatrix_migration.md`.

## fc2_hybrid.cu (ALL PHASES DEAD)

All attempts to combine our PTX mainloop with CUTLASS's epilogue have failed.

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

### ncu profiling results (B200-verified 2026-04-05)

**STS clustering CONFIRMED as root cause. Barrier stalls are the dominant visible symptom.**

Data: `data/ncu_20260405_202913/`, analysis: `python3 tools/ncu_anova.py`

#### Q1: What our epilogue adds (w3 fused - strip)

| Metric | Strip | Fused | Δ% | Verdict |
|---|---|---|---|---|
| short_scoreboard | 1,222 | 138,859 | **+11,261%** | STS WAR hazards from clustering |
| SMEM wavefronts | 102K | 39.9M | +39,035% | LDS/STS from epilogue (expected) |
| barrier | 227,465 | 298,055 | +31% | Uneven warp progress from STS bursts |
| wait | 254,519 | 376,622 | +48% | General wait increase |
| **long_scoreboard** | 2,540,781 | 2,549,600 | **+0.3%** | **DRAM is NOT the bottleneck** |
| **mio_throttle** | 16 | 16 | **+0.3%** | **TMA is NOT the bottleneck** |

#### Q1b: What CUTLASS's epilogue adds (cutlass fused - strip)

| Metric | Strip | Fused | Δ% |
|---|---|---|---|
| short_scoreboard | 30,309 | 135,873 | +348% (from higher base) |
| barrier | 91,410 | 31,069 | **-66%** (epilogue fills idle time!) |
| long_scoreboard | 2,439,589 | 2,741,875 | +12.4% |
| mio_throttle | 2,176 | 3,785 | +74% |

CUTLASS's epilogue DECREASES barrier stalls — warps fill time that would otherwise be idle. Our epilogue INCREASES them because STS clustering causes uneven warp progress.

#### Q2: w3 fused vs cutlass fused — head-to-head

| Stall | w3_fused | cutlass_fused | Δ% | Notes |
|---|---|---|---|---|
| short_scoreboard | 138,859 | 135,873 | +2.2% | Basically same! |
| long_scoreboard | 2,549,600 | 2,741,875 | -7.0% | We're slightly better |
| **barrier** | **298,055** | **31,069** | **+860%** | THE dominant difference |
| **wait** | **376,622** | **228,165** | **+65%** | Significant |
| sleeping | 37,566 | 0 | w3 only | |

**Barrier stalls (+267K, +860%) account for 110% of the total stall delta** (other stalls partially cancel). But removing barriers = zero perf effect, so barrier stalls are a SYMPTOM: STS clustering → SMEM port contention across 4 warps → uneven STS completion → earlier-finishing warps wait at barriers → inflated barrier stall count.

#### Q2b: Strip comparison (mainloop only)

| Metric | w3_strip | cutlass_strip |
|---|---|---|
| SMEM wavefronts | 102K | 20.0M |
| short_scoreboard | 1,222 | 30,309 |
| cycles_active | 1,824,165 | 1,914,702 (-4.7%) |

Our mainloop is cleaner — nearly zero SMEM activity. Confirms our strip speed advantage.

#### Q3b: Phase 4 — why it's 2.5x broken

| Metric | phase4_fused | cutlass_fused | Δ% |
|---|---|---|---|
| cycles_active | 4,897,540 | 1,944,987 | +152% |
| inst_executed | 396M | 163M | +143% |
| long_scoreboard | 6.87M | 2.74M | +151% |
| wait | 905K | 228K | +297% |
| LTS read sectors | 1.18B | 618M | +91% |

Phase 4 does ~2x the work. Static `for ti=t0..t1` dispatch causes either redundant work, L2 thrashing from bad tile ordering, or failure to overlap mainloop/epilogue across tiles.

#### Ruled out by ncu data

- **DRAM bandwidth**: long_scoreboard +0.3% fused vs strip. Not a factor.
- **TMA saturation**: mio_throttle flat at 16. Not a factor.
- **Warp scheduler starvation**: not_selected +5K (+739%) but absolute is tiny (5.8K vs 22K for CUTLASS). Not dominant.

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
- CUTLASS_EPILOGUE (FP32 per-group STS, 113 regs): identical STS clustering
- CPP_EPILOGUE: byte-identical SASS to asm volatile
- CUTE_STORE (C++ pointer stores): byte-identical SASS
- @!PT LDS fences (3 approaches): all DCE'd by ptxas
- cvta.shared + generic store: ptxas DCEs all 64 stores
- NO_PRE_STORE_BAR + NO_POST_STORE_BAR: 17->1 BAR.SYNC, zero perf
- NUM_EPI_STAGES=3/4: -6 to -23us (noise)
- stmatrix (STSM.16.M88.4): identical throughput to STS.128 at all warp counts (B200-verified)
- All combinations of above: noise
- **ptxas STS scheduling is immutable from any source-level approach**
- **Intra-warp STS reordering is futile** — problem is cross-warp temporal alignment

### Architectural attempts on fc2 (old kernel, all ~1.635ms at 4 warps)

- BAR.SYNC serialization (EPI_SYNC, EPI_BAR_PASS, EPI_BAR_CHUNK)
- Combinatorial strips (only_tma_load, only_tma_store, only_compute)
- FP32_EPILOGUE, stagger (500/2000 cyc), pass count, store deferred
- EPI_LOAD_WARP (+13% regression), EPI_PIPELINE (+0.8% noise)
- W0_RES_FULL (+15% catastrophic), W0_RES_PREFETCH (neutral)
- STAGES_C=2 (neutral), STAGES_C+EPI_REUSE_SMEM (broken), STAGES_C+PRE_COMBINE (broken)
- NOP_EPILOGUE, EPI_DELAY, REG_PAD: all ruled out individual causes
- SASS intra-warp reorder: level 5 (single chunk) passes B200, level 6 (full epilogue) crashes. But intra-warp scheduling is the wrong target anyway.
- stmatrix (STSM): identical throughput to STS.128 at all warp counts (B200 2026-04-05)

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

## Key Learnings

### The root cause: cross-warp STS synchronization
- **The problem is INTER-WARP, not INTRA-WARP.** All 4 epilogue warps hit their STS stores at the same time because BF16 compute is too short to create natural phase drift. CUTLASS's FP32 compute is 4x longer, so warps drift apart naturally.
- **Intra-warp instruction scheduling is irrelevant.** CP-SAT finds 75-83% better per-warp schedules, but CUTLASS has the same 5x intra-warp gap and doesn't care. Reordering STS within one warp doesn't fix 4 warps hitting SMEM ports simultaneously.
- **stmatrix = STS.128 at the SMEM port level.** Instruction class doesn't matter.
- **Barrier stalls are a SYMPTOM, not a cause.** Removing barriers = zero effect.

### Architecture
- **fc2_w3 ALREADY has CUTLASS-style architecture**: W2=EpilogueLoad, W3-W6 LDS from SMEM.
- **STRIP_EPILOGUE is THE diagnostic** — K-loop inflates +36%
- **Overhead linear at ~98us/warp** — no single epilogue operation matters individually
- **Microbenchmarks MISLEADING** — single-tile +0.1%, persistent kernel +36%
- **ptxas STS scheduling immutable from source** — 6+ approaches, all identical

### Hardware
- **N_STAGES=5 mandatory**: 10% gap vs NS4. **BIAS_SMEM=1 default**: -15us free.
- **4 sub-partitions on SM100a** — warp i%4. **MAX_STALL = 7** (3-bit, bits 53-55).
- **STS scaling: 10→37 cyc at 8 warps (3.65x)** — SMEM port contention is real and measured.

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
- N_STAGES=5 mandatory (10% gap vs NS4)
- BIAS_SMEM=1 default (-15us free)
