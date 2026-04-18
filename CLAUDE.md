# SigLIP2 Vision Encoder — Hand-tuned Blackwell GEMM Kernels

Hand-tuned SM100a persistent GEMM kernels for FC1 and FC2 layers of `google/siglip2-base-patch16-224`.
FP8 (E4M3) inputs, BF16 output, tcgen05 MMA, TMA, `cta_group::2` with 2-CTA clusters.
Cross-compiled on CPU VPS, runs on B200 (148 SMs, 74 clusters). PE kernel is done — see `CLAUDE.md.mothballed`.

## Current status (B200, 2026-04-17, PACKED_TILES parity)

All runs below use `-DPACKED_TILES` for static and dynamic variants alike. PREFILL no
longer needs to be disabled for static swizzles (TD=8-16) — they work under the
default pipeline settings. Numbers come from `./tools/bench.sh --comprehensive`.

### FC2: [928256, 3072] x [3072, 768]^T + bias + residual

| Variant | fused | gemm | strip | f-g | g-s |
|---|---|---|---|---|---|
| **default (stride)** | **1.071** | 1.073 | 1.026 | -0.002 | 0.047 |
| rowmajor | 1.071 | 1.074 | 0.988 | -0.003 | 0.086 |
| zigzag (TD=11) | 1.073 | 1.073 | 0.988 | 0.000 | 0.085 |
| zorder (TD=9) | 1.082 | 1.071 | 0.988 | 0.011 | 0.083 |
| dgswizzle (TD=8) | 1.065 | 1.053 | 0.989 | 0.012 | 0.064 |
| hilbert (TD=10) | 1.089 | 1.067 | 0.989 | 0.022 | 0.078 |
| sched (TD=4) | 1.101 | 1.083 | 0.994 | 0.018 | 0.089 |
| tail | 1.102 | 1.086 | 0.993 | 0.016 | 0.093 |
| tail-lean | 1.106 | 1.093 | 0.993 | 0.013 | 0.100 |
| lean (LEAN_DISPATCH) | 1.107 | 1.093 | 0.994 | 0.014 | 0.099 |
| ncycle / nsnake | 1.226 | 1.200 | 1.024 | 0.026 | 0.175 |
| nflat | 1.205 | 1.168 | 1.021 | 0.037 | 0.147 |
| rowsteal | 1.242 | 1.213 | 1.037 | 0.029 | 0.176 |

Static swizzles with PACKED_TILES (default/rowmajor/zigzag/zorder/dgswizzle) now
beat LEAN by ~30us. Work-stealing (sched/lean/tail/tail-lean) all cluster around
1.10ms. Strip floor for the fast group is 0.988ms; default's 1.026ms strip is an
outlier worth investigating.

### FC1: [928256, 768] x [768, 3072]^T + bias + GELU

| Variant | fused | gemm | strip | f-g | g-s |
|---|---|---|---|---|---|
| zigzag (TD=11) | 2.024 | 1.894 | 1.382 | 0.130 | 0.512 |
| rowmajor | 2.027 | 1.894 | 1.383 | 0.133 | 0.511 |
| nflat | 2.035 | 1.721 | 1.339 | 0.314 | 0.382 |
| nsnake | 2.035 | 2.042 | 1.336 | -0.007 | 0.706 |
| ncycle | 2.033 | 2.039 | 1.339 | -0.006 | 0.700 |
| sched | 2.076 | 1.854 | 1.411 | 0.222 | 0.443 |
| lean | 2.075 | 1.832 | 1.410 | 0.243 | 0.422 |
| default | 2.094 | 1.875 | 1.380 | 0.219 | 0.495 |
| dgswizzle | 2.093 | 1.659 | 1.378 | 0.434 | 0.281 |
| zorder | 2.089 | 1.740 | 1.362 | 0.349 | 0.378 |
| hilbert | 2.257 | 1.694 | 1.435 | 0.563 | 0.259 |

FC1 dispatch matters more than FC2: dgswizzle's gemm 1.659ms vs default 1.875ms is
a 216us gap. zigzag/rowmajor have the fastest fused (2.024ms) despite mediocre gemm
— they overlap mainloop and epilogue differently. ncycle/nsnake show f-g ≈ 0
(fused == gemm), meaning those variants get no epilogue/mainloop overlap at all.
Strip floor spread is 100us (1.336 nsnake → 1.435 hilbert), so tile order affects
even the GEMM compute ceiling for FC1.

## Tile dispatch: static swizzles win (updated 2026-04-18)

**The old thesis was wrong.** For a year we believed LEAN won by eliminating
DRAM read amplification from 1.13× to 1.00×. The 2026-04-18 ncu run under
PACKED_TILES parity (76 configs, full 35-metric capture, see
`data/bench_20260418_034637/anova.txt`) inverts this. Static swizzles lead FC2
fused by ~30us despite reading 20-59% *more* DRAM bytes — because those extra
bytes don't translate into MMA stalls, while work-stealing's tight wavefront
does.

### The new metric: `long_scoreboard`, not DRAM amplification

| FC2 fused (p) | ms | long_sb | barrier | DRAM rd | amp |
|---|---|---|---|---|---|
| default (stride) | 1.071 | 2.12M | 272K | 6.79 GB | 1.59× |
| **zigzag** (TD=11) | **1.073** | 2.12M | 271K | 6.04 GB | 1.41× |
| dgswizzle (TD=8) | 1.065 | 2.02M | 267K | 5.44 GB | 1.27× |
| sched (TD=4) | 1.101 | 2.66M | 45K | 4.28 GB | 1.00× |
| **lean** (LEAN_DISPATCH) | 1.107 | 2.66M | 44K | 4.28 GB | 1.00× |
| rowsteal | 1.242 | 2.76M | 45K | 4.28 GB | 1.00× |

LEAN trades **540K more long_scoreboard stalls for 230K fewer barrier stalls**.
Net: slower. The mbarrier-removal trick was real and measurable (barrier 44K
vs 272K), but the barrier wasn't the bottleneck — DRAM-load serialization
was, and work-stealing made it *worse*.

### Why static beats work-stealing

- **Static TMA streaming is pipeline-friendly.** With striding or zigzag, 74
  clusters hit B-columns at *offset* K-phases. Even if the L2 working set
  exceeds 96MB and forces DRAM refills, those refills overlap each other and
  the TMA load pipeline stays full. The "amplification" is bandwidth the HBM
  already has spare.
- **Work-stealing creates a synchronous wavefront.** All clusters march
  through tiles in the same global order. Each tile's A+B must arrive
  *just-in-time* from a hot L2, and when it doesn't, every cluster stalls
  on the same `long_scoreboard` at once. Zero amplification, but every
  miss is on the MMA critical path.
- **PACKED_TILES changed the game.** Pre-2026-04-17 data compared static
  swizzles without PACKED_TILES against dynamic variants with implicit
  good-order atomics. Not apples-to-apples. Under parity, static wins.

### Within the static family: zigzag gets +4.2 points L2 hit rate free

Zigzag (TD=11) matches default's stall profile *exactly* (long_sb 2.12M,
barrier 271K, wait 387K) but reads 750MB less DRAM and has 50.1% L2 hit rate
vs default's 45.9% — +4.2 points. Same fused ms, better cache efficiency, so
it's the new recommended default for FC2. dgswizzle has the lowest long_sb
(2.02M) and lowest DRAM amp of the static group (1.27×), leading fused at
1.065ms — but at the cost of a register-count bump that matters on FC1.

### FC1 is a different story

FC1's f-g gap is a bigger lever than FC2's dispatch choice:

- dgswizzle gemm 1.659ms (best) → fused 2.093ms. 434us f-g gap.
- zigzag gemm 1.894ms → fused 2.024ms. 130us f-g gap — best fused.
- ncycle/nsnake: f-g ≈ 0 — zero epilogue/mainloop overlap, pathological.

FC1 wants a dispatch that preserves dgswizzle's mainloop locality *without*
its epilogue overlap penalty. Phase-offset DG (static N band, dynamic M-row)
is the open experiment.

### Why LEAN's barrier reduction still matters (and doesn't)

LEAN_DISPATCH does eliminate the ~300 cyc/tile `tile_ready_mbar` broadcast
from W3's critical path — that's real, measurable, and shows up as a 227K
drop in barrier stalls. It just lands on a non-bottleneck under PACKED_TILES
parity. Keep LEAN in the tree for large-K regimes where DRAM amplification
dominates (K≥5120 data pre-parity suggested work-stealing wins), but at
K=3072 it's outclassed.

### Old variants (all dead)

- **Contiguous [begin:end]**: never implemented. Adjacent CTAs touch disjoint
  M-row ranges, catastrophic DRAM amplification.
- **TD=1 atomic**: every warp does atomicAdd. 1.370ms — overhead swamps any
  amplification savings.
- **TD=5 CLC**: hardware dispatch via `clusterlaunchcontrol`. Deadlocks —
  CLC's one-block-per-tile model is incompatible with persistent loops.
- **TD=6 inline atomic (W0 at tile boundary)**: 1.370ms — blocks W0, delays
  TMA loads.
- **TD=7 inline atomic in K-loop**: 1.257ms — +41% TMA issue overhead. W0's
  K-loop is memory-pipeline-sensitive; ANY global memory op degrades TMA.
- **COL_LOCK**: 1.137ms. TMA penalty inherent to W7 mbarrier path, not tile
  order. Strip 62us slower due to load imbalance (74 clusters / 3 cols).
- **N-batch striding**: +12% regression. Static dispatch can't match LEAN's
  L2 efficiency at the N-batch wavefront width.
- **ncycle / nsnake / nflat**: column-first. Under PACKED_TILES still slow
  (1.20-1.23ms) because 74 clusters hammer the same N-column → TMA store
  contention.
- **rowsteal**: 1.242ms. Work-stealing variant that tried to add N-column
  locality. Worst of both worlds.
- **L2 cache hints (EVICT_FIRST/LAST/NORMAL)**: zero effect. long_scoreboard
  stalls are an arrival-pattern problem, not an eviction problem.

### K-crossover (needs re-verification under parity)

Pre-2026-04-17 data (without PACKED_TILES parity) showed work-stealing winning
at K=6144 by 7%. Whether this reverses under the new thesis is an open
question — if long_scoreboard stalls dominate regardless of K, static may lead
at all K. Re-run `./tools/bench.sh --comprehensive --ncu` with `-DK_DIM=6144`
and `K_DIM=4096` to confirm.

### Other dispatch variants tried (all dead)

- **TD=1 atomic**: Every warp does atomicAdd. 1.370ms — overhead kills any amplification savings.
- **TD=5 CLC**: Hardware dispatch via `clusterlaunchcontrol`. Deadlocks — CLC's one-block-per-tile model is incompatible with persistent kernel loops.
- **TD=6 inline atomic**: W0 does atomicAdd at tile boundary. 1.370ms — blocks W0, delays TMA loads.
- **TD=7 inline atomic in K-loop**: atomicAdd at ki=0. 1.257ms — disrupts W0's TMA pipeline (+41% tma_issue). Proves W0's K-loop is memory-pipeline-sensitive; ANY global memory op degrades TMA throughput.
- **COL_LOCK**: Column-locked dispatch (fixed tn, dynamic M-row). 1.137ms — TMA penalty is inherent to the W7 mbarrier path, not tile ordering. Strip 62us slower than sched (load imbalance: 74 clusters / 3 cols = 25/25/24).
- **Tile reordering (striding variants)**: N-batch (+12% regression), phase-offset N-batch (+6-11%), Group-3 (neutral). Static dispatch can't match work-stealing's L2 efficiency.
- **Space-filling curves (TD=9-12)**: Z-order/Morton, Hilbert, zigzag-N, column-first (1.707ms DEAD, catastrophic: all clusters hit same N-column → TMA store contention + enormous A-tile L2 working set). Pre-2026-04-17 numbers (no PACKED_TILES) put these at 1.11-1.13ms. **Under PACKED_TILES parity (2026-04-17) dgswizzle landed at 1.074ms — tied with stride, ~30us under LEAN — so the "tile ordering doesn't matter" conclusion is under review.** hilbert/zorder/zigzag under PACKED_TILES not yet measured head-to-head.
- **L2 cache hints**: EVICT_FIRST/LAST/NORMAL on TMA loads. Zero effect — amplification is a capacity problem, not eviction policy.

## Pipeline depth: why NS6 matters

6-stage mainloop pipeline uses 227KB of 228KB SMEM. Each stage holds one 256x128 A-tile (FP8) + one 256x128 B-tile (FP8) in SMEM.

NS6 does two things:
1. Hides TMA load latency (6 stages in flight vs 5) — marginal at L2-warm, significant at L2-miss.
2. Enables **PREFILL**: overlaps previous tile's epilogue drain with the first 6 K-iterations of the next tile's MMA. W1 skips the epilogue_mbar check for the first 6 iters, allowing epilogue warps to finish while MMA is already running. Saves ~10us at K=3072.

PREFILL is unsafe at K_ITERS<20 (W1 races ahead, mainloop_mbar parity wraps → deadlock). Auto-guarded: `#if K_DIM/128 < 20` enables NO_PREFILL. FC1 (K_ITERS=6) always uses NO_PREFILL.

NS5 is required for N>1536 (SMEM per stage grows with N). NS7 doesn't fit in 228KB.

## Kernel structure (fc2_w3.cu / fc1_w3.cu)

Warp-specialized, 7 warps (224 threads), `cta_group::2`, `__cluster_dims__(2,1,1)`:

| Warp | Role | Notes |
|---|---|---|
| W0 | TMA Load (A+B) | Memory-pipeline-sensitive — no global ops in K-loop |
| W1 | tcgen05.mma K-loop | TMEM 512 cols double-buffered |
| W2 | EpilogueLoad | TMA loads residual (FC2) or nothing (FC1) into SMEM, circular 2-stage pipe |
| W3-W6 | Epilogue compute | LDS+TMEM ld+math+CVT+STS+TMA store (4 warps) |
| W7 | Scheduler (TD=4/LEAN only) | atomicAdd tile counter, mbarrier broadcast |

Tile: 256x256x128. K_ITERS=K_DIM/128. FC2: K=3072 (24 iters), FC1: K=768 (6 iters).

## Compute floor and epilogue

From bench/mma_bench.cu: MMA K-iteration 665 cyc raw, 525.6 cyc/iter pipelined. Per FC2 tile (24 iters): ~12,614 cyc. Theoretical strip floor at 1.813 GHz = ~1.048ms (matches observed).

**Epilogue is 100% hidden in MMA shadow.** The fused-strip gap (~45us across
all variants under PACKED_TILES) is entirely memory-side: `long_scoreboard`
bubbles on A+B arrival + TMA store contention. NOT compute, NOT instruction
scheduling, NOT cross-warp STS clustering. Note: older text in git history
attributed this gap to "DRAM amplification" — 2026-04-18 data shows
amplification is a red herring; what matters is the *pattern* of arrivals,
not the byte count.

Our BF16 epilogue (HFMA2/HADD2) costs ~44us vs CUTLASS's FP32 (FFMA+F2FP) ~72us. Cross-warp STS clustering is real (barrier stalls +753% in ncu) but is a symptom, not a bottleneck — proven by STRIP_EPILOGUE isolating the gap to memory traffic.

## DRAM read amplification (historical note — NOT the bottleneck)

Theoretical minimum: fused = A+B+residual+bias = 4.28GB, strip = A+B = 2.85GB.

| Variant (FC2 fused, PACKED_TILES) | DRAM read | vs theoretical | ms |
|---|---|---|---|
| default (striding) | 6.79 GB | 1.59× | 1.071 |
| zigzag (TD=11) | 6.04 GB | 1.41× | 1.073 |
| dgswizzle (TD=8) | 5.44 GB | 1.27× | 1.065 |
| sched (work-stealing) | 4.28 GB | 1.00× | 1.101 |
| lean (LEAN_DISPATCH) | 4.28 GB | 1.00× | 1.107 |
| cutlass_fused (CLC) | 4.28 GB | 1.00× | 1.226 |

Work-stealing achieves 1.00× amplification and is *slower* than 1.59× striding.
Amplification is not the bottleneck — MMA stalls from synchronous A+B arrivals
are. See tile dispatch section above.

Root cause of striding's 1.13x: 74 clusters striding through M-rows with different N-tile phases creates a working set that exceeds L2 capacity. Work-stealing processes tiles in global order, keeping the wavefront narrow.

## Dimension sweep (B200-verified)

Infrastructure: `tools/dim_sweep.sh`. Dims: `-DM_TOTAL=X -DN_DIM=Y -DK_DIM=Z`. Constraints: M%256==0, N%256==0, K%128==0. Must use `make -B`.

### M scaling (N=768, K=3072) — advantage stable 8-10%

| M | w3_fused | cutlass_fused | Delta |
|---|---|---|---|
| 116032 | 0.154 | 0.167 | -7.8% |
| 232064 | 0.289 | 0.317 | -8.8% |
| 464128 | 0.563 | 0.621 | -9.3% |
| 928256 | 1.110 | 1.226 | -9.5% |
| 1856512 | 2.206 | 2.433 | -9.3% |

### K scaling — NS6 advantage fades, work-stealing advantage grows

| K | K_ITERS | w3_fused (stride) | w3_sched (TD=4) | cutlass_fused |
|---|---|---|---|---|
| 3072 | 24 | **1.110** | 1.134 | 1.226 |
| 4096 | 32 | 1.538 | 1.545 | 1.551 |
| 6144 | 48 | 2.257 | **2.098** | 2.263 |

At K=3072, NS6 pipeline depth = 25% of K-loop, hiding DRAM latency effectively. At K=6144, it's 12.5% — compute dominates, amplification penalty is fully exposed. Work-stealing's zero amplification wins.

### Adaptive tuning knobs

| Knob | Rule | Why |
|---|---|---|
| N_STAGES | NS6 for N<=1536, NS5 for N>1536 | SMEM per stage grows with N |
| PREFILL | On for K_ITERS>=20, off otherwise | Short K-loop deadlocks (parity wrap) |
| Dispatch | **UNSETTLED (2026-04-17)** | LEAN previously claimed best; one PACKED_TILES ncu run had stride/dgswizzle ~30us ahead of LEAN. Needs reps. |

## Dead ends — do NOT retry

### Epilogue optimization (all futile — epilogue is hidden in MMA shadow)

**Source-level SASS immutability**: CUTLASS_LOOP, FP32_EPILOGUE, CUTLASS_EPILOGUE, CPP_EPILOGUE, CUTE_STORE, @!PT LDS, cvta.shared, NO_PRE/POST_STORE_BAR, NUM_EPI_STAGES, stmatrix, EPI_REORDER, NUM_EPI_WARPS=1/2. ptxas generates identical STS clustering regardless of source — the compiler controls it, not us. 6+ approaches tried, all produce the same SASS.

**Cross-warp STS clustering**: The real cause of barrier stalls (+753%), but irrelevant because epilogue is hidden in MMA shadow. SELF_LOAD (per-warp TMA), SELF_STAGGER (nanosleep 50-200ns) — zero effect. SASS intra-warp reorder (CP-SAT scheduler) — targets wrong bottleneck. All proven on B200.

### CUTLASS hybrid (fc2_hybrid.cu)

Attempted to combine our PTX mainloop with CUTLASS's C++ epilogue (for better STS scheduling). Phase 2 (=Phase 1 speed), Phase 3b (2.76ms — CUTLASS mainloop breaks at non-8-warp), Phase 4 (2.77ms — 8-warp static tile loop 2.3x slower than our persistent loop). All dead.

### Other dead approaches

- **fc2_ldg.cu (LDG/STG kernel)**: STG goes through L1TEX (128B/thread), TMA bypasses it. Fundamentally bandwidth-limited.
- **Old fc2.cu (4 warps)**: BAR.SYNC serialization, FP32, all epilogue variants. Superseded by 7-warp architecture.
- **SASS patching**: Intra-warp reorder wrong target. Inter-warp YIELD stagger irrelevant (epilogue hidden).
- **FC1 FORCE_PREFILL**: Deadlocks at K_ITERS=6. NO_PREFILL guard is necessary.

## Build and run

```bash
# FC2 (best: LEAN)
make fc2-w3-lean && ./fc2-w3-lean                # fused 1.074ms
make fc2-w3 && ./fc2-w3                          # striding 1.113ms
make fc2-w3-sched && ./fc2-w3-sched              # work-stealing 1.147ms
make fc2-w3-gemm && ./fc2-w3-gemm                # GEMM-only
make fc2-w3-strip && ./fc2-w3-strip              # MMA-only

# FC1 (best: LEAN)
make fc1-w3-lean && ./fc1-w3-lean                # fused 2.037ms
make fc1-w3 && ./fc1-w3                          # default
make fc1-w3-sched && ./fc1-w3-sched              # work-stealing

# Custom dims (MUST use -B: Make doesn't track DFLAGS)
make -B fc2-w3 DFLAGS='-DM_TOTAL=464128 -DN_DIM=1024 -DK_DIM=2048 -DN_STAGES=6'
# FC1/FC2 strip/gemm via DFLAGS: -DSTRIP_EPILOGUE / -DGEMM_ONLY

# CUTLASS reference
make fc2-cutlass && ./fc2-cutlass                # 1.226ms
make fc2-cutlass-strip && ./fc2-cutlass-strip    # 1.152ms

# Profiling and comparison
bash tools/fc2_cutlass_vs_w3.sh
bash tools/ncu_bench.sh && python3 tools/ncu_anova.py
./tools/dim_sweep.sh --fast                      # 80 configs
```

## Key files

```
fc2_w3.cu                       # FC2 hand-tuned PTX kernel (ACTIVE — best)
fc1_w3.cu                       # FC1 hand-tuned PTX kernel (ACTIVE)
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
- All inline PTX in fc2_w3.cu/fc1_w3.cu (no CUTLASS dependency)
- OFF_STAGING must be 1024-byte aligned for SWIZZLE_128B
- fence.proxy.async.shared::cta required before TMA store after st.shared
- N_STAGES=6 default (NS5 for N>1536, NS7 doesn't fit)
- PREFILL default on for FC2 (auto-disabled for K_ITERS<20)
- NO_PREFILL always on for FC1 (K_ITERS=6)
- BIAS_SMEM=1 default (-15us free)
- Custom dims require `make -B` (Make doesn't track DFLAGS)
- W0's K-loop is memory-pipeline-sensitive — no global memory ops allowed
