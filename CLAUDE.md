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
| **zigzag + K_STAGGER=1** (TD=11) | **1.998** | — | — | — | — |
| dgswizzle + K_STAGGER=1 (TD=8) | 2.023 | — | — | — | — |
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

**K_STAGGER compose (2026-04-18)**: FC1 is sensitive to odd K_STAGGER (ks=1 or 3
help, ks=2 hurts). zigzag+ks=1 → 1.998 (−14 µs vs zigzag alone); dgswizzle+ks=1 →
2.023 (−115 µs vs dgswizzle baseline 2.138); default+ks=1 → 2.051 (−49 µs). The
ks=1/ks=3 parity on FC1 (K_ITERS=6) suggests phase alignment at the DRAM/L2
boundary, not raw phase count, is the mechanism. FC2 is near-wash across all
(disp, ks); zigzag+ks=2 = 1.065 ties dgswizzle but no new ceiling break.

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
- **Tile reordering (pre-PACKED_TILES)**: N-batch (+12%), phase-offset N-batch (+6–11%), Group-3 (neutral). Old framing "static can't match work-stealing's L2 efficiency" is **reversed under parity** — see status table; re-test any of these before declaring dead.
- **Space-filling curves (TD=9–12) under PACKED_TILES**: zigzag (TD=11), zorder (TD=9), hilbert (TD=10), dgswizzle (TD=8) are all in the **winning cluster** (FC2 1.065–1.085 ms). Pre-parity "catastrophic" label was wrong. Only TD=12 (pure column-first) is consistently slow on FC2 fused; on FC1 strip it's actually the fastest known (1.337 ms ncycle) — it just pays at the store phase.
- **L2 cache hints (EVICT_FIRST/LAST/NORMAL)**: zero effect on `long_scoreboard` arrival-pattern stalls. Accurate conclusion; the old rationale ("amplification is a capacity problem") is superseded — amplification isn't the bottleneck, but L2 hints still don't move the needle.

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

## Compute floor and decomposition (not a ceiling)

Hard MMA compute floor: 525.6 cyc/iter pipelined × K_ITERS × tiles / (74 clusters × 2 CTAs × 1.813 GHz). FC2 = ~1.048 ms. FC1 strip floor is NOT compute-bound (K=6 is so short that TMA load latency dominates).

**Strip itself is ordering-sensitive, not a floor.** The 2026-04-18 sweep shows
FC2 strip spans 0.987–1.031 ms (44 µs spread) and FC1 strip spans 1.336–1.501 ms
(165 µs spread) purely from tile ordering. "Strip ≈ physics floor" is wrong —
strip includes TMA load + L2 state + mbarrier handoff, all ordering-addressable.
ncycle/nsnake minimize FC1 strip by maximizing within-cluster B-tile reuse.

**fused = strip + (g-s) + (f-g).** Each gap is a separate, partially-decoupled
axis:

| gap | what it measures | moves with |
|---|---|---|
| g-s | store phase (bias load, residual TMA, STS+TMA store) | **cluster-wavefront N-column diversity** (dgswizzle minimizes, column-sync maximizes) |
| f-g | epilogue overlap with next-tile mainloop | **K_ITERS** (FC1 K=6 can't hide), PREFILL, epilogue duration |

The FC1 ncycle anomaly proves the axes separate: f-g ≈ 0 (epilogue fully hidden)
but g-s = 0.681 ms (store contention). Best dispatch on g-s (dgswizzle) costs
0.315 ms for FC1 — 366 µs of ordering-addressable reducible cost between
ncycle's mainloop and dgswizzle's store pattern. No dispatch today combines both.

BF16 epilogue compute cost (HFMA2/HADD2) ~44 µs. Cross-warp STS clustering
(barrier stalls +753%) is a symptom that changes with ordering, not a fixed
bottleneck. Source-level STS layout is ptxas-immutable (proven), but the
*inter-cluster* arrival pattern into STS/TMA-store is fully ordering-controlled.

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

## cuBLASLt rank-1 comparison (2026-04-20)

`tools/probe_cublaslt.sh` (probe 1 = `bench/cublaslt_introspect.cu`) enumerates
every heuristic cuBLASLt returns, times each, reports rank-1. This is the true
ceiling to beat. Earlier comparisons against `cublas-bench-fc2` measured
cuBLASLt's **default heuristic pick**, not rank-1.

### FC2 K-sweep vs our static-swizzle/lean variants (data/cublaslt_probe_20260420_043659)

| K    | cuBLASLt  | zigzag    | dgsw      | lean      | gap (best ours − cuBLASLt) |
|------|-----------|-----------|-----------|-----------|----------------------------|
| 1024 | ERR       | 0.862     | 0.866     | **0.859** | n/a                        |
| 2048 | ERR       | **0.922** | 0.922     | 0.925     | n/a                        |
| 3072 | **1.045** | 1.073     | **1.064** | 1.113     | +19 µs                     |
| 4096 | **1.360** | 1.498     | **1.476** | 1.503     | +116 µs                    |
| 6144 | **1.997** | 2.015     | **2.007** | 2.052     | +10 µs                     |
| 8192 | **2.682** | 2.742     | 2.734     | **2.731** | +49 µs                     |

**FC1 rank-1 (K=768, M=928256, N=3072, gelu_bias): cuBLASLt 1.894 ms** vs best
ours 1.998 (zigzag+ks=1) → **+104 µs (+5.5%) — wider gap than FC2**.

### Enum decode — authoritative (do NOT re-derive)

From `/opt/cuda/targets/x86_64-linux/include/cublasLt.h`:
- **`CUBLASLT_MATMUL_TILE_*`** — per-CTA tile shape. Lookups: `23=128x256`,
  `24=256x128`, `32=128x192`, `197=168x128`, `201=176x128`, `495=256x96`,
  `535=320x192`. Enum is per-CTA; cluster output = tile × clusterShape.
- **`stages=36 = CUBLASLT_MATMUL_STAGES_128xAUTO`** — K-stage=128, pipeline
  depth is AUTO, **resolved at kernel-compile time per variant** (not runtime).
  cuBLASLt ships multiple prebuilt variants with different NS; heuristic picks.

### Full heuristic decode via nsys (FC2 K=3072 on B200, 2026-04-20)

Kernel names: `nvjet_sm100_qqtst_<M>x<N>_128x<NS>_<CM>x<CN>_[2cta_]<h|v>_<...>_T<A><B>`.
Token `2cta` = **cta_group::2** present, absent = cta_group::1. `h/v` = TMA
multicast axis (h=along N, v=along M). `bz_bias` = bias-only epilogue.

| listed  | tile (enum)    | NS | cluster | cta_grp | ms     |
|---------|----------------|----|---------|---------|--------|
| L1      | 176x128 (201)  | 8  | 1x2     | **2**   | 1.0454 |
| **L2†** | 128x256 (23)   | 6  | 2x1     | **2**   | 1.0457 |
| L3      | 128x192 (32)   | 7  | 2x1     | 2       | 1.094  |
| L4      | 256x256 (513)  | 4  | 2x1     | 2       | 1.192  |
| L5      | 320x192 (535)  | 4  | 2x1     | 2       | 1.196  |
| L6      | 256x128 (24)   | 4  | 1x2     | 1       | 1.267  |
| L7      | 168x128 (197)  | 5  | 1x2     | 1       | 1.358  |
| L8      | 256x96  (495)  | 4  | 1x2     | 1       | 1.440  |

† **L2 is what we call "rank-1" from 2026-04-20 onward.** The listed L1
(176x128 NS=8 1x2) edges L2 by 0.3 µs — pure run-to-run noise — and uses NS=8
which isn't SMEM-feasible at our 256x256 output (we jam at NS=6), so L1 is
neither reproducibly faster nor architecturally reachable. L2 (128x256 NS=6
2x1 2cta) is our exact per-CTA MMA geometry + cluster layout, and is the real
1.046 ms ceiling. "Rank-1" in all prose below = the L2 kernel:
`nvjet_sm100_qqtst_128x256_128x6_2x1_2cta_v_bz_bias_TNT`. Dumped SASS is at
`rank1.sass`.

### Hard-won conclusions

1. **cuBLASLt's top 5 listings all use `cta_group::2`**. Our 2sm-MMA architecture
   is aligned with cuBLASLt's winning designs; cta_group::1 (L6–L8) is uniformly
   slower here.
2. **Rank-1 (= listed L2) is OUR exact geometry**: 128x256 per-CTA, 2x1 cluster,
   cta_group::2, NS=6, v-mcast (along M). Times at **1.046 ms**. Our best (dgsw)
   is 1.064. **We're +18 µs behind our architectural twin** — the gap is NOT
   architectural, it's pure epilogue/dispatch/scheduling. Most important
   comparison point going forward.
3. **Listed L1 is a fluke**. NS=8 at 176x128 wins by 0.3 µs — within noise, and
   NS=8 at our 256x256 isn't SMEM-feasible. Not reachable, not meaningful.
4. **cuBLASLt also ships a `256x256_128x4_2x1_2cta` variant** (listed L4 at 1.192).
   Same cluster-output as ours but NS=4 — we beat it by ~125 µs. The 256x256
   variant is a fallback in cuBLASLt's grid.
5. **The cuBLASLt advantage is NOT split-K** (`splitk=1` throughout) and NOT
   a CUTLASS-style tile swizzle (`swizzle=0` throughout) — meaning our 18 µs
   gap is not a tile-ordering lever, and matching rank-1 via a new dispatch
   is unlikely to close it. The remaining levers are epilogue-local
   (LDTM/STSM shape, TMA store box) and K-phase scheduling.
6. FC1 K=768 cuBLASLt heuristics have **no 2x2x1 (4-CTA) entry** — 4-CTA ruled
   out at short K. Our `fc2_w3_c4*` multicast deadlocks are not the FC1 lever.
   4-CTA remains interesting only for FC2 per-tensor FP8 (`tile=128x192
   clusterShape=2x2x1` is in the FC2 list).

### Live levers (2026-04-20)

- **Lever A (TMA_STORE_WIDE) — DEAD** (2026-04-20). Wide-rows box {64,64}
  halves dynamic UTMASTG 16→8 per CTA per tile, matching rank-1's 8
  UTMASTG exactly. B200 measured `fc2-w3-epi-100` (wide+dgswizzle) 1.064 ms
  vs `fc2-w3-epi-000` (narrow+dgswizzle) 1.066 ms — 2 µs is run-to-run
  noise. Closes zero of the 18 µs gap. Confirms **TMA store issue count
  is not the bottleneck.** {128,32} unreachable under SWIZZLE_128B
  (boxDim[0]*esize cap 128 B); SWIZZLE_NONE legalizes it but would
  reintroduce 4-way STS bank conflicts. Don't retry.
- **Lever C (USE_STMATRIX)** — live but broken. Diagnostic (commit 909da64)
  confirmed reg-layout + address-collision bugs. Fix requires swapping TMEM load
  from `tcgen05.ld.sync.aligned.32x32b.x32.b32` to an stmatrix-native variant
  (likely `16x128b.x4` or `16x256b.x4`) + reworking CVT/STS macros + fixing the
  `(lane & 7)` address swizzle to use full lane ID. ~80-line refactor.
- **Lever B (EPI_SINGLE_PASS) — DEAD** (2026-04-20). Single-pass restructure
  doesn't address the 18 µs gap. Not the bottleneck.

### Next action

With Lever A + B dead and dispatch axis exhausted (dgphase/dgnrot), the
remaining levers are (1) fix Lever C — swap TMEM load to
`LDTM.16dp256bit.x4`, halves SMEM store instruction count 64 STS.128 → 32
STSM; or (2) match rank-1's 4× UTCQMMA grouping in the K-loop.
SASS-diff rank-1 against our fc2-w3-dgswizzle to localize further:
```bash
cuobjdump --dump-sass \
    --function 'nvjet_sm100_qqtst_128x256_128x6_2x1_2cta_v_bz_bias_TNT' \
    /opt/cuda/lib64/libcublasLt.so.13 > rank1.sass
cuobjdump --dump-sass ./fc2-w3-dgswizzle > ours.sass
```

Known caveat: K=1024/2048 still report ERR — one heuristic IMAs on the device
(`cublasLtMatmul` returns SUCCESS but `cudaDeviceSynchronize()` aborts). A
resilient mode (skip past IMA, report what worked) is pending.

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

### K scaling under PACKED_TILES + best-of-dispatch (2026-04-20)

| K    | K_ITERS | w3 best (dispatch) | cuBLASLt rank-1 | gap to cuBLASLt |
|------|---------|--------------------|-----------------|-----------------|
| 1024 | 8       | 0.859 (lean)       | ERR             | n/a             |
| 2048 | 16      | 0.922 (zigzag)     | ERR             | n/a             |
| 3072 | 24      | 1.064 (dgswizzle)  | 1.045           | +19 µs          |
| 4096 | 32      | 1.476 (dgswizzle)  | 1.360           | +116 µs         |
| 6144 | 48      | 2.007 (dgswizzle)  | 1.997           | +10 µs          |
| 8192 | 64      | 2.731 (lean)       | 2.682           | +49 µs          |

dgswizzle leads on K=3072–6144, lean takes over at K=8192 (0.003 ms under dgsw)
and at the short-K end where NO_PREFILL kicks in. The earlier "sched wins at
K=6144" thesis (from pre-PACKED_TILES, pre-dgswizzle data) is superseded.

### Adaptive tuning knobs

| Knob | Rule | Why |
|---|---|---|
| N_STAGES | NS6 for N<=1536, NS5 for N>1536 | SMEM per stage grows with N |
| PREFILL | On for K_ITERS>=20, off otherwise | Short K-loop deadlocks (parity wrap) |
| Dispatch | FC2: zigzag or dgswizzle (1.065 ms). FC1: **zigzag + K_STAGGER=1** (1.998 ms). | PACKED_TILES + odd K_STAGGER on FC1. FC2 is near-wash on K_STAGGER. Open axes: mainloop/store decoupling, cluster-heterogeneous dispatch. |

## Dead ends — do NOT retry

### Source-level epilogue tuning (all futile — ptxas controls STS layout)

**Source-level SASS immutability**: CUTLASS_LOOP, FP32_EPILOGUE, CUTLASS_EPILOGUE, CPP_EPILOGUE, CUTE_STORE, @!PT LDS, cvta.shared, NO_PRE/POST_STORE_BAR, NUM_EPI_STAGES, stmatrix, EPI_REORDER, NUM_EPI_WARPS=1/2. ptxas generates identical STS clustering regardless of source — the compiler controls it, not us. 6+ approaches tried, all produce the same SASS.

**Cross-warp STS clustering (intra-warp-only attempts)**: The real cause of barrier stalls (+753%). SELF_LOAD (per-warp TMA), SELF_STAGGER (nanosleep 50–200 ns) — zero effect on FC2. SASS intra-warp reorder (CP-SAT scheduler) — targets wrong axis. Note: these attempts only tried to stagger warps *within a CTA*; they did not touch the much larger *inter-cluster* store-arrival pattern which IS ordering-addressable (see `g-s` gap on FC1).

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
tools/tile_regress.py           # simulates TDs in Python, regresses ms on tile-sequence features
tools/stagger_sweep.sh          # K_STAGGER × N_STAGGER × dispatch × mode sweep (WIP)
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

## Open frontiers (2026-04-18) — axes not yet decoupled

Current dispatches couple mainloop-order with store-order. The data shows these
are separable:

1. **mainloop-order ≠ store-order**: ncycle has best strip (max B-reuse) but
   worst g-s (max store contention). Modify TD=14 so cluster `c` rotates its
   `tn` by `c × TILES_N / NC` — preserves within-cluster B-reuse, breaks
   cross-cluster store synchrony. Predicted FC1 gemm 2.02 → ~1.65 if the
   hypothesis holds.
2. **K_STAGGER × tight-locality dispatch** (TESTED 2026-04-18): swept
   {default, dgswizzle, zigzag, checkered, dg16} × ks={0,1,2,3} × {fused, gemm,
   strip} on both layers. **FC1 wins measurably**: zigzag+ks=1 → 1.998 (new
   best), dgswizzle+ks=1 → 2.023, default+ks=1 → 2.051. Odd ks helps, ks=2
   hurts (K_ITERS=6 phase-alignment effect). **FC2 is near-wash** (±0.013 ms);
   zigzag+ks=2 = 1.065 ties dgswizzle, no ceiling break. Open: strip+gemm
   decomp on (zigzag, ks=1) to diagnose whether the FC1 gain lives on strip
   (L2/TMA arrival) or g-s (store phase).
3. **Cluster-heterogeneous dispatch**: first half dgswizzle, second half
   column-first. Tests whether cluster-wavefront diversity is an independent
   axis from the single-cluster ordering.

Structural ideas (not pure-ordering; defer until axes 1–3 are explored):
- Streaming epilogue: start storing TMEM columns as they retire instead of after
  full tile. Breaks FC1's K=6 pathology (~400 µs epilogue that can't hide in 6
  K-iters). Removes the f-g constraint that forces zigzag over ncycle.
- Bigger N-tile: 256 → 512. Doubles K-iters-per-tile effectively; gives FC1 the
  overlap budget FC2 has.

## Tile-feature regression (v2, WIP)

`tools/tile_regress.py` replicates TD=0,8–21 in pure Python, emits per-cluster
tile visit sequences, extracts 51-feature vectors covering: within-cluster
1-hop and 2-hop locality, cross-cluster synchronous contention, drift-windowed
contention at w∈{2,4} (SMs don't march in lockstep), L2 carry-over at
w∈{1,2,4,8,16,32}, `collide_K` (mainloop/epilogue asymmetry at lag=K_ITERS),
`tm/tn_autocorr_{4,8,16}`, `path_curve` (2-hop direction continuity), K-phase
stats, and K_ITERS interactions.  Fits against
`data/bench_20260418_034637/*_wall_r1.txt` with Ridge, LassoCV, GBR, and
exhaustive best-subset k=3 under LOOCV.

### Findings (2026-04-19, v2)

| layer | mode  | best model      | rmse_loo (ms) | R²_loo  | (v1 R²_loo) |
|---|---|---|---|---|---|
| fc2   | strip | ridge_top5      | **0.0022**    | +0.99   | —      |
| fc2   | gemm  | ridge_all       | **0.0111**    | +0.96   | +0.95  |
| fc2   | fused | ridge_top3      | **0.0125**    | +0.97   | +0.80  |
| fc1   | strip | ridge_top3      | 0.0212        | +0.56   | +0.10  |
| fc1   | gemm  | best_subset k=3 | 0.0677        | +0.75   | +0.47  |
| fc1   | fused | best_subset k=3 | 0.0606        | +0.22   | −0.39  |

**FC2 fused/gemm.** `a_reuse` (synchronous-A-wavefront) is univariately
dominant (r=+0.99).  ncycle/nsnake/nflat share `a_reuse=0.671` and land at
1.20–1.23 ms; dgswizzle/zigzag/rowmajor have `a_reuse=0` and land at 1.07 ms.
The `long_scoreboard` mechanism is now predictive.

**FC1 strip.** `a_reuse` flips sign (r=−0.83): FC1 K=6 strip is TMA-bound, so
same-tm packing is a cache win, not a stall cause.  Confirms no universal
dispatch exists.

**FC1 gemm.** Best triple `{path_curve, unique_tn_w2, tm_carry_w16}`.  The 16-
step window matches 4×4 block traversal timescale — picks up that hilbert
retires each block before moving on while zorder bit-interleaves.

**FC1 fused — the hilbert problem.** Univariate-filtered Ridge (top-k by |r|)
never picks `b_reuse_w2` because only hilbert has non-zero value (0.25 vs 0.00
for all others).  Exhaustive best-subset search over C(51,3) triples finds
`{tn_jump, b_reuse_w2, tm_carry_w1}` — rmse_loo 0.061 ms vs 0.077 for
univariate.  `b_reuse_w2` is the mechanism: hilbert holds the same N-column
across 2 consecutive steps 25% of the time, and with FC1 K_ITERS=6 the 4
epilogue warps keep hammering the same store port → contention.  R²_loo still
only +0.22 because with n=9 we have one sample (hilbert) lighting up the
feature; predictions extrapolate badly beyond the training feature range.

### Using it to predict a new curve

Workflow for a new dispatch you can implement in Python `static_swizzle()`:
add the TD to `DISPATCH_TD`, call `features()`, plug into the best-subset fit
for each (layer, mode).  Expected accuracy: ±0.2–1.5% for FC2 across all
modes, ±1.4% for FC1 strip, ±3–4% for FC1 gemm/fused.  Flag: any feature
value outside training range is pure extrapolation — n=9 gives no cover for
OOD dispatches.

### Next iterations

- Merge `stagger_sweep.sh` CSV via `--extra` once B200-tested — 75×3 rows
  gives `ks`/`ns` columns variance and quadruples n.
- Regress against ncu metrics (`dram__bytes_read`, `long_scoreboard`,
  `mbarrier_wait`) directly as targets.
- Add `--predict <dispatch>` CLI for quick what-if evaluation.

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
- W0's K-loop is TMA-sensitive: atomicAdd at ki=0 (TD=7) costs +41–77% tma_issue. Non-critical-path global ops (W7 scheduler TD=4, tile-boundary) are fine.
