# SigLIP2 Vision Encoder — Hand-tuned Blackwell GEMM Kernels

Hand-tuned SM100a persistent GEMM for FC1/FC2 of `google/siglip2-base-patch16-224`.
FP8 (E4M3) → BF16, tcgen05 MMA, TMA, `cta_group::2`, 2-CTA clusters. Cross-compiled
on CPU VPS, runs B200 (148 SMs, 74 clusters). PE kernel done — see
`CLAUDE.md.mothballed`.

## Current best (B200, 2026-05-06)

| target | ms | kernel | dispatch | vs cuBLASLt fused rank-1 |
|---|---|---|---|---|
| FC2 K=3072 BIAS_ONLY (strip floor) | 0.98502 | `fc2_w3x` `-DSTRIP_EPILOGUE` | n/a | NS=6+PREFILL structural floor (1814685 cyc) |
| **FC2 K=3072 BIAS_ONLY (full)** | **1.00092** | `fc2_w3x` (bias-preload, STSM-only) | basin floor (default `gflip_blkswap` TD=54) | **−27 µs** vs 1.028 PerTensor / **−116 µs** vs 1.117 MXFP8; +16 µs exposed epi vs strip |
| FC2 K=3072 fused (+residual) | ~1.060 | `fc2_w3` | **gflip_blkswap TD=54** PACKED (now default) | (no apples-to-apples ref) |
| FC1 K=768 fused (+GELU+bias) | 1.998 | `fc1_w3` | zigzag TD=11 + K_STAGGER=1 | **−416 µs** vs 2.414 PerTensor / **+47 µs** vs 1.951 MXFP8 |

`fc2_w3x` = clean-sheet 6-warp persistent bias-only, beats per-tensor and
MXFP8 cuBLASLt rank-1.
`fc2_w3` = legacy 7-warp fused, production for residual path.
`fc1_w3` beats per-tensor cuBLASLt by 416 µs but trails MXFP8 by 47 µs —
MXFP8 has different (better) kernels for FC1's small-K geometry.

cuBLASLt fused-bias / fused-GELU+bias algorithms **do exist** on sm_100a /
CUDA 13.0 — but only with **BF16 bias dtype**. FP32 bias dtype gets you 0
algos and `CUBLAS_STATUS_NOT_SUPPORTED` regardless of layout. 4-cell probe
({[N,M],[M,N]} × {BF16,FP32}) showed bias dtype is the lever. Layout
orientation [N,M] vs [M,N] doesn't change algo enumeration but DOES change
the kernels' runtime — the prior 1.894 ms / 1.046 ms references on [M,N]
(transposed) layout aren't directly comparable to production [N,M] numbers.
Default in `bench/fc_problem.cuh` is now BF16 bias on [N,M] layout, which
mirrors cublas-bench-fc1/fc2 and reproduces the rank-1 references above.

## Kernel structure

7-warp persistent (224 threads), `cta_group::2`, `__cluster_dims__(2,1,1)`. Tile
256x256x128. K_ITERS = K_DIM/128 (FC2: 24, FC1: 6).

| Warp | Role | Notes |
|---|---|---|
| W0 | TMA Load (A+B) | TMA-sensitive — no global ops in K-loop |
| W1 | tcgen05.mma K-loop | TMEM 512 cols double-buffered |
| W2 | EpilogueLoad | TMA loads residual (FC2), 2-stage |
| W3-W6 | Epilogue compute | LDS + TMEM ld + math + CVT + STS + TMA store |
| W7 | Scheduler (TD=4/LEAN) | atomicAdd tile counter, mbarrier broadcast |

`fc2_w3x` differs: 6 warps (W0-W3 epi, W4 TMA, W5 MMA CTA0-only). No W7. `buf = tt & 1`.

## Adaptive tuning + pipeline depth

| Knob | Rule | Why |
|---|---|---|
| N_STAGES | auto: `min(NS_BY_N, max(2, K_ITERS−3))`; NS_BY_N = 6/5/4/3 for N≤1536/2048/4096/larger | SMEM ceiling (228 KB; 32 KB A+B FP8/stage; NS7 too big). Pipeline-fill margin gap≥3 (gap=2 FAILs at K=1024 NS=6). |
| PREFILL | auto: K_ITERS≥20 → on, else NO_PREFILL | Short K-loop deadlocks (parity wrap). NO_PREFILL caps eff ~0.77; PREFILL ~0.91. fc2_w3 auto-guards via `#if K_DIM/128 < 20`; fc2_w3x kernel-side macro guard (3d6c1cb). |
| Dispatch | FC2 fused: `gflip_blkswap` (TD=54) **default** (−6018 cyc/−3.4 µs vs dgsw TD=8, STRONG d=−1.33 — basin transfers from bias-only; `make fc2-w3` now builds TD=54). FC2 bias-only: any `gflip_*` basin-floor — `gflip_blkswap` (TD=54) default. FC1: zigzag + K_STAGGER=1. | PACKED_TILES + odd ks helps FC1; FC2 wash on ks. |

PREFILL overlaps prev tile's epi drain with first 6 K-iters of next tile's MMA;
W1 skips epilogue_mbar check for first 6 iters. 6-stage pipeline = 227 KB / 228 KB SMEM.

## FC2 fused status (PACKED_TILES, M=928256, K=3072, N=768)

| Variant | fused | gemm | strip | f-g | g-s |
|---|---|---|---|---|---|
| **default (stride)** | **1.071** | 1.073 | 1.026 | -0.002 | 0.047 |
| zigzag (TD=11) | 1.073 | 1.073 | 0.988 | 0.000 | 0.085 |
| dgswizzle (TD=8) | 1.065 | 1.053 | 0.989 | 0.012 | 0.064 |
| rowmajor / zorder / hilbert | 1.07–1.09 | ~1.07 | ~0.988 | — | — |
| sched (TD=4) | 1.101 | 1.083 | 0.994 | 0.018 | 0.089 |
| lean (LEAN_DISPATCH) | 1.107 | 1.093 | 0.994 | 0.014 | 0.099 |
| ncycle / nsnake / nflat | 1.20–1.23 | — | — | — | — |
| rowsteal | 1.242 | 1.213 | 1.037 | 0.029 | 0.176 |

Static swizzles > work-stealing (~30 µs). Strip ~0.988 ms.
`fused = strip + (g-s) + (f-g)`. g-s: store contention, cluster-wavefront N-column
diversity. f-g: epi/next-tile-mainloop overlap, K_ITERS-limited.

**gflip basin sweep on fused fc2_w3 (2026-06-30, cyc, n=134 trimmed passes, η²=0.99).**
fc2_w3 is now templated `<TD,DGG>` like fc2_w3x; the whole gflip basin compiles
into one `-DCOMBO_QUICK` binary (`make fc2-w3-swizzle-sweep`, cyc via CLOCK_TOTAL).
**The bias-only basin floor transfers: `gflip_blkswap` (TD=54) wins the fused path
too — −6018 cyc (−3.4 µs/−0.32%) vs dgsw TD=8, STRONG (d=−1.33, 34% win).** Top
tier blkswap/dgsnake/snrot2/gflip_snrot/gflip all STRONG faster than dgsw (−4.2 to
−6.0k cyc). **But the lmrev family DECISIVELY *hurts* fused (+12 to +14k cyc) —
opposite of bias-only, where lmrev tied for first; the residual consumer is
m-axis-traversal sensitive in a way bias-only isn't.** `stride` (old TD=0 default)
is +156k cyc (+8.4%) — never ship it; TD=54 is now the `make fc2-w3` default.
**Caveat:** fc2_w3 has a **confirmed sparse output RACE** (~0.04%, the
run-to-run nondeterminism). Root cause PROVEN 2026-06-30 (Modal B200,
barrier-knockout + depth battery, 20×/cfg) = **epilogue-staging recycle DEPTH**
(`NUM_EPI_STAGES`), NOT the MMA/TMEM handshake and NOT the store barriers:
BIAS_ONLY ES1→ES2 fixes both the race AND the deterministic +64 (20/20 valid, 1
checksum); removing wait_group/bar.sync/proxy-fence does NOT worsen FULL.
Production FULL already runs ES2 so the severe ES1 collision doesn't apply; its
remaining 1/20 spot collapse (e.g. 290080,470 → 780) is the W2-residual-ring ↔
consumed_mbar handshake, still marginal at depth 2. Deeper ring needs NS6→5
(= resadd-port path, +71 µs). Kernel SASS byte-identical to pre-template HEAD →
not the swizzle refactor; doesn't affect cyc so the basin sweep stands. Full
analysis: `memory/project-fc2-w3-epilogue-race.md`. Sweep: `tools/sweep_fc2_w3_swizzle.sh` /
`modal run dummy_modal.py --target fc2-w3-swizzle-sweep --run-args "SWEEP=front REPS=200"`.

## FC1 fused status (PACKED_TILES, M=928256, K=768, N=3072)

| Variant | fused | gemm | strip | f-g | g-s |
|---|---|---|---|---|---|
| **zigzag + K_STAGGER=1** (TD=11) | **1.998** | — | — | — | — |
| dgswizzle + K_STAGGER=1 (TD=8) | 2.023 | — | — | — | — |
| zigzag (TD=11) | 2.024 | 1.894 | 1.382 | 0.130 | 0.512 |
| nflat | 2.035 | 1.721 | 1.339 | 0.314 | 0.382 |
| nsnake / ncycle | ~2.034 | ~2.04 | 1.337 | ~0 | ~0.703 |
| sched / lean | ~2.075 | ~1.84 | 1.41 | ~0.23 | ~0.43 |
| dgswizzle (no ks) | 2.093 | 1.659 | 1.378 | 0.434 | 0.281 |
| hilbert | 2.257 | 1.694 | 1.435 | 0.563 | 0.259 |

FC1 dispatch lever > FC2's. Odd K_STAGGER (1/3) helps; ks=2 hurts.
ncycle/nsnake have f-g≈0 — pathological.

## Tile dispatch — mechanism

Static swizzles win under PACKED_TILES parity. Pre-2026-04-17 "work-stealing
wins via 1.00× DRAM amplification" thesis dead — static reads 20–59% MORE bytes,
runs faster. Real metric: **`long_scoreboard` stalls** (synchronous-A-wavefront),
not DRAM amp.

| FC2 fused | ms | long_sb | barrier | DRAM rd | amp |
|---|---|---|---|---|---|
| default | 1.071 | 2.12M | 272K | 6.79 GB | 1.59× |
| zigzag TD=11 | 1.073 | 2.12M | 271K | 6.04 GB | 1.41× |
| dgswizzle TD=8 | 1.065 | 2.02M | 267K | 5.44 GB | 1.27× |
| sched | 1.101 | 2.66M | 45K | 4.28 GB | 1.00× |
| lean | 1.107 | 2.66M | 44K | 4.28 GB | 1.00× |

LEAN trades 540K more long_sb for 230K fewer barrier — slower net.

**"DRAM amp ≠ bottleneck" cleanest proof (cutlass-static, 2026-04-23)** — same
tile/cluster/2SM-schedule/PACKED, only scheduler/epi differ:

| variant | wall µs | tensor% | long_sb | L2 hit% | DRAM rd | amp |
|---|---|---|---|---|---|---|
| cutlass-static (fused) | 1244 | 81.92 | 10.22 | 59.53 | 4.280 GB | **1.000×** |
| fc2_w3x (bias-only)    | 1059 | **97.94** | **6.70** | **67.65** | 2.978 GB | 1.043× |

CUTLASS at 1.000× amp floor, 185 µs slower at 21% more instructions
(169.9M vs 140.2M → 16-pt tensor-pipe gap). **Tensor-pipe utilization is the
lever.** fc2_w3x reads 1.3 GB MORE per launch, runs faster.

## fc2_w3x basin (n=29420) — settled

m-axis dispatch is a **wide tied basin**. Top 7 cells (blkx5/6/7, blk_qrt0/2/3,
blkswap) within ~135 cyc, η²=0.0075 NEGLIGIBLE. **Default `gflip_blkswap`
(TD=54) stays — middle of basin, zero churn.**

Mechanism: once gflip's XOR=1 group pairing is in place (`cluster_tm_corr ↓`
0.94→0.65), any m-axis perturbation that decorrelates paired CTAs' tm-traversal
saturates the gain. Three sub-tiers (Stage 1 n=2048): floor (XK=1 + m-axis
perturb) Δ −600 to −770 cyc; mid (weaker perturb) Δ −400 to −500; shallow
(gflip alone or wrong axis) Δ −100 to −300.

**Catastrophic gflip failures stay relevant:** `gflip_cidperm` (TD=55, +1718
DECISIVE — `*15 mod 74` cluster perm breaks SM→L2 contiguity, cluster_tm_corr
0.16 vs 0.65), `gflip_xk2/3/5/7_blkswap` (non-XK=1 pairs non-adjacent groups),
bare `gflip` (~80% of gain is m-axis perturb). **`lmrev` demoted:** prior
n=43910 "blkswap+lmrev TIE" sweep artifact; σ=768 in n=2048 fluke produced
misleading DECISIVE call.

`tools/bloom_filter.py` validated conservative-let-through, zero false
negatives across n=43910+n=29420. WORTHY/MARGINAL = build, OVERSHOOT-RISK =
build expecting regression, STUPID = skip. `adj_tn_diff` empirical channel
may be stale (snrot2 12th at n=10978 vs "2nd at n=32768" prior).

Swizzle metric pipeline (Stage 1 simulate `lin_tile = cluster_id + tt*NC`,
Stage 2 PCA+KMeans+Ward against baked WALL_NS200):
```bash
python3 tools/analyze_swizzle.py --csv /tmp/swizzle_metrics.csv
python3 tools/cluster_swizzle.py /tmp/swizzle_metrics.csv
```
Verdict (2026-04-28): PARTIALLY CAPTURES. dgsnake/gflip/lmrev within ~150 cyc
metric-indistinguishable; blkswap/lmrev captured by sign-stable τ axes
(`adj_tm_diff`, `tm_extent_mean`).

Full table + tier mechanics + bloom scorecard: `memory/project_w3x_n29420_basin.md`.

## Compute floor

`tcgen05.mma.cta_group::2` = cluster-wide work per insn. Per-cluster cycles =
`147 tiles × 24 K_iters × cyc/iter`:

| source | cyc/iter | wall (B200) | notes |
|---|---|---|---|
| hardware MMA retirement | 460 | 0.896 ms | absolute ceiling, no staging — **unreachable** |
| bench NS=4 + W0-TMA overlap | 520.8 | 1.014 ms | published microbench |
| bench NS=4, no TMA overlap | 525.6 | 1.023 ms | published microbench |
| **fc2-w3x-strip** (NS=6+PREFILL) | **493** | **0.98502 ms** | **structural staging floor** |
| **fc2-w3x** (full) | **502** | **1.00092 ms** | production |
| cuBLASLt fused rank-1 PerTensor | ~515 | 1.028 ms | reference (BF16 bias, [N,M] layout) |
| cuBLASLt fused rank-1 MXFP8 | ~559 | 1.117 ms | block-scaled reference |

**Strip vs full = 15.9 µs / 31460 cyc / 214 cyc/tile** (1.7% of 12482 cyc/tile
MMA budget) — exposed epi coupling W0-W3 epi-end and W5's next-tile MMA via
`bar.sync` / `mbar_wait`. ~98% of epi hidden in MMA shadow; the 2% that isn't
is real headroom.

**Gap decomposition** (vs structural floor):
- **89 µs** (pure-MMA → strip): NS=6 staging bubble, unreachable without removing staging.
- **16 µs** (strip → fc2-w3x): exposed epi, real headroom.
- **43 µs** (strip → cuBLASLt PerTensor rank-1): rank-1 has more exposed epi.

**Realistic recoverable: ~1-3 µs.**

Per-SASS stall hotspots (`docs/fc2_w3x_ncu_sass.txt`): mbar spin-wait dominates
at 5.0M samples, 7× next category (epi compute body ~696K each); TMA store
scaffolding ~174K each. TC pipe 98.49% → strip IS the MMA-staging structural
floor. **Final-tile drain FALSE** (per-tt PROFILE_W5: tt=146 = 11872 cyc
fastest); 12 µs gap between ncu cyc_avg and wall cyc_max is cross-CTA workload
variance (~26800 cyc tail), not final-tile artifact. LAST_TILE_FAST_PATH would
save <1 µs — not implemented.

**16 µs gap = ~4 µs steady-state mbar/cluster bar.sync + ~12 µs cross-CTA tail
variance.**

**Ignore ncu warnings:** `"13398-way bank conflict"` (STSM mis-attribution; ncu
doesn't model `stmatrix`'s bank-routed datapath); `"21.3 active threads/warp"`
(warp-specialization by design). FC1 strip is TMA-load-dominated.

NANOSLEEP_CYC sweep (n=5489): ns32 lone WEAK-faster (-0.06 µs); ns0 busy-spin
WEAK slower (+170 cyc) — spin runs on idle warps, removing nap doesn't unlock
budget. Spread ~281 cyc; anything in [4..32] equivalent. ns20 default stays.
Calibration: n=1373→n=5489 demoted 3 of 4 "WEAK faster" cells to TIE — same
canonical pattern as gflip basin. See `memory/project_w3x_nanosleep_basin.md`.

## FC2 N×K dim sweep (pow2 grid, 2026-05-06, 16 cells)

`tools/dim_sweep_w3x.py` default = `N ∈ {256,512,1024,2048} × K ∈
{1024,2048,4096,8192}`. Cycles paired with cuBLASLt rank-1 via
stream-serialized `clock64()` sentinels (same SM-clock domain). Both
`cb_bias` (EPI=3 BIAS_ONLY) and `cb_none` (EPI=0 GEMM-only) are now valid —
the prior table had `cb_bias` blocked by FP32-bias enumeration failure;
fixed in `bench/fc_problem.cuh` (BF16-bias default).

| N | K | K_it | NS | ours cyc | cb_bias cyc | Δb% | cb_b tile | cb_none cyc | Δn% | cb_n tile |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 1024 | 8  | 5\* | 398.1k  | 372.5k   | +6.85  | 128x160 | 378.7k  | +5.13  | 128x128 |
| 256 | 2048 | 16 | 6   | 634.9k  | 668.0k   | −4.96  | 128x256 | 669.6k  | −5.17  | 128x256 |
| 256 | 4096 | 32 | 6   | 1152.7k | 1180.7k  | −2.38  | 128x256 | 1179.9k | −2.31  | 128x256 |
| 256 | 8192 | 64 | 6   | 2151.7k | 2417.8k  | −11.01 | **128x384** | 2416.8k | −10.97 | 128x256 |
| 512 | 1024 | 8  | 5   | 661.9k  | 618.8k   | +6.97  | 128x256 | 582.0k  | +13.72 | 128x256 |
| 512 | 2048 | 16 | 6   | 952.5k  | 959.1k   | −0.69  | 128x256 | 949.2k  | +0.35  | 128x256 |
| 512 | 4096 | 32 | 6   | 1779.6k | 1743.9k  | +2.05  | 128x256 | 1736.0k | +2.51  | 128x256 |
| 512 | 8192 | 64 | 6   | 3384.3k | 3574.2k  | −5.31  | 128x256 | 3542.7k | −4.47  | id=513  |
| 1024| 1024 | 8  | 5   | 1333.9k | 1090.6k  | +22.30 | 128x256 | 1011.8k | +31.83 | 128x256 |
| 1024| 2048 | 16 | 6   | 1691.2k | 1729.5k  | −2.21  | 128x256 | 1719.6k | −1.65  | 128x256 |
| 1024| 4096 | 32 | 6   | 3409.6k | 3280.5k  | +3.94  | 128x256 | 3277.5k | +4.03  | 128x256 |
| 1024| 8192 | 64 | 6   | 6512.0k | 6593.9k  | −1.24  | **128x192** | 6580.9k | −1.05  | 128x192 |
| 2048| 1024 | 8  | 5   | 2691.4k | 2092.9k  | +28.60 | 128x256 | 1964.3k | +37.02 | 128x256 |
| 2048| 2048 | 16 | 5   | 3761.1k | 3303.0k  | +13.87 | 128x256 | 3295.7k | +14.12 | 128x256 |
| 2048| 4096 | 32 | 5   | 6738.3k | 6483.4k  | +3.93  | 128x256 | 6487.1k | +3.87  | 128x256 |
| 2048| 8192 | 64 | 5   | 12963.5k| 12974.9k | −0.09  | 128x256 | 13006.1k| −0.33  | 128x256 |

cluster_id=2x1x1 (id=3) on every cell, both columns. tile=23 (128x256) is
the default; **bold** = cuBLASLt picks a non-default tile. id=513 is a
vendor-private tile id beyond the standard `cublasLtMatmulTile_t` enum.

\*N=256 K=1024 picks NS=5 via `min(NS_BY_N=6, K_ITERS−3=5)`. K=1024 N∈{512,1024}
was FAIL@NS=6; auto-picker now NS=5.

cb_bias is generally within 1-2% of cb_none — cuBLASLt's algoId=66 fuses
bias nearly free into the same family of kernels.

**cuBLASLt rank-1 picks tile=128x256 cluster=2x1x1 in 13 of 16 cells**
for the BIAS_ONLY column. Outliers: N=256 K=1024 → tile=29 (128x160);
N=256 K=8192 → tile=177 (128x384); N=1024 K=8192 → tile=32 (128x192).
noBIAS column has 14 of 16 at 128x256, with N=256 K=1024 → 128x128 and
N=512 K=8192 → vendor-private tile id=513. At small N (=256) cuBLASLt
narrows the N-tile dimension; at long K with mid N it picks 128x192.
Cluster_id=3 (2x1x1) holds across every cell — exactly fc2_w3x's geometry.

**Production point K=3072 N=768: fc2_w3x 1.0043 ms vs cuBLASLt
BIAS_ONLY rank-1 1.0270 ms = −22.7 µs / −2.2% in cycles** (apples-to-apples
fused-bias-on-bias). Confirmed via separate K-sweep below.

**Loss patterns (sharper now that cb_bias is visible):**
1. **K=1024 universal loss (+5 to +29% across N).** cuBLASLt's K=1024 BIAS_ONLY
   kernel (algoId=66 tile=23 NS=36 cluster=2x1x1, same family as K=3072) is much
   tighter at small K — our NS=5 + NO_PREFILL + gap=3 stack pays heavily here.
   N=2048 K=1024 catastrophe (+28.59% bias / +36.96% none, eff=0.54) is the
   extreme.
2. **K=4096 systematic loss N≥512 (+2.07 to +3.96% bias).** Persistent across
   reruns. cuBLASLt heuristic switches efficiency tier; ours flat. Suspects:
   256×96 tile (TILE_ID=495), deeper NS, split-K.
3. **N=2048 NS=5 SMEM tax.** Now decisively visible: K=2048 +13.72% bias
   (was −3.92% in prior n=1 sample — old number was a fluke), K=4096 +3.94%.
   Gap closes by K=8192 (−0.14%). 1-stage latency-hide loss confirms
   LDTM_X64's NUM_EPI_STAGES=1 costs ~14 µs.

**Sweet spots:** K=8192 across all N (−0.14 to −11.10%, the longer K-loop
amortizes our pipeline depth advantage); K=2048 at small N (N=256/512/1024:
−5 to −1%, K_ITERS=16 past PREFILL gap=10); K=4096 at smallest N (N=256:
−2.38%, cuBLASLt floor degrades fastest at lowest tile/cluster).

See `memory/project_w3x_dim_sweep_vs_cublas.md` for cuBLASLt sparse-heuristic
gaps at non-pow2 K.

## FC1 N×K dim sweep (2026-05-06, 16 cells)

`tools/dim_sweep_fc1.py` default = `N ∈ {1024,2048,3072,4096} × K ∈
{512,768,1024,1536}` — centered on FC1 production (N=3072, K=768).
fc1_w3 with production tune (zigzag TILE_DISPATCH=11 + K_STAGGER=1).
cuBLASLt rank-1 via `cublaslt-introspect` at EPI=2 (GELU+BIAS) and
EPI=0 (GEMM-only). Comparison in **ms** — fc1_w3 doesn't emit per-CTA
clock64 cyc.

| N | K | K_it | NS | ms | cb_gelu | Δg µs | Δg% | cb_g cl | cb_none | Δn% |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 512  | 4  | 3 | 0.7120 | 0.8049 | −93   | −11.54 | 2x2x1 | 0.4079 | +74.55 |
| 1024 | 768  | 6  | 5 | 0.6700 | 0.8130 | −143  | −17.59 | 2x2x1 | 0.4665 | +43.62 |
| 1024 | 1024 | 8  | 5 | 0.6970 | 0.8298 | −133  | −16.00 | 2x2x1 | 0.5476 | +27.28 |
| 1024 | 1536 | 12 | 5 | 0.8960 | 0.8481 | +48   | +5.65  | 2x2x1 | 0.7335 | +22.15 |
| 2048 | 512  | 4  | 3 | 1.4140 | 1.5991 | −185  | −11.58 | 2x2x1 | 0.8083 | +74.94 |
| 2048 | 768  | 6  | 5 | 1.3300 | 1.6126 | −283  | −17.52 | 2x2x1 | 0.9267 | +43.52 |
| 2048 | 1024 | 8  | 5 | 1.3860 | 1.6474 | −261  | −15.87 | 2x2x1 | 1.0647 | +30.18 |
| 2048 | 1536 | 12 | 5 | 1.7050 | 1.6805 | +24   | +1.46  | 2x2x1 | 1.3685 | +24.59 |
| 3072 | 512  | 4  | 3 | 2.1210 | 2.3943 | −273  | −11.41 | **4x4x1** | 1.2097 | +75.33 |
| **3072**| **768**| 6 | 5 |**2.0260**| **2.4135**| **−388** | **−16.06** | 2x2x1 | 1.3642 | +48.51 |
| 3072 | 1024 | 8  | 5 | 2.0810 | 2.4649 | −384  | −15.57 | 2x2x1 | 1.5997 | +30.09 |
| 3072 | 1536 | 12 | 5 | 2.5310 | 2.5118 | +19   | +0.76  | 2x2x1 | 2.0520 | +23.34 |
| 4096 | 512  | 4  | 3 | 2.8080 | 3.1902 | −382  | −11.98 | 2x2x1 | 1.6100 | +74.41 |
| 4096 | 768  | 6  | 5 | 2.6490 | 3.2137 | **−565** | **−17.57** | 2x2x1 | 1.8253 | +45.13 |
| 4096 | 1024 | 8  | 5 | 2.7690 | 3.2821 | −513  | −15.63 | 2x2x1 | 2.1331 | +29.81 |
| 4096 | 1536 | 12 | 5 | 3.2840 | 3.3433 | −59   | −1.77  | **2x4x1** | 2.7720 | +18.47 |

GELU+BIAS column: tile=128x256 in every cell; cluster mostly 2x2x1
(id=6), two outliers in **bold**. cb_none column: tile=128x256 cluster
2x1x1 (id=3) uniformly across all 16 cells — same family fc2_w3x targets.

**Production point N=3072 K=768: −16.06% / −388 µs vs PerTensor rank-1.**
Reproduces 2.026 ms (within run-variance of the published 1.998 reference).

**Three K regions, consistent across all N tested:**
1. **K∈{512,768,1024} — ours dominates by 11.4 to 17.6%.** fc1_w3 was tuned
   exactly for this regime (K=768 gets a flat −17.5% across all N from
   1024 to 4096). cuBLASLt's algoId=71 GELU+BIAS family doesn't have an
   efficient short-K kernel.
2. **K=1536 — flip to near-tie or slight loss.** Ranges +5.61% (N=1024)
   to −1.79% (N=4096). Crossover where cuBLASLt's K-amortization catches
   up — K_iters=12 lets the algoId=71 family hide GELU.
3. **K≥2048** (from FC1 K-sweep, not in this grid): ours decisively
   loses vs PerTensor (+177 to +653 µs at K=2048), still beats MXFP8 at
   K=3072/4096.

**Best absolute Δ%: K=768 at every N** (−17.4 to −17.6%). **Best absolute
µs: N=4096 K=768 = −565 µs.** Production N=3072 K=768 leaves ~50 µs vs
MXFP8 (per separate K-sweep) — real headroom signal at FC1 small-K geometry.

`cb_none` (GEMM-only) is a noBIAS reference; ours is +18 to +75% over it
because we're doing GELU+BIAS the cb_none entry isn't. Useful only as a
GEMM-floor anchor.

**Cluster shape across cells:** GELU+BIAS rank-1 is `tile=128x256 cl=2x2x1`
(id=6) in 14 of 16 cells. Two outliers: N=3072 K=512 → `cl=4x4x1` (id=10),
N=4096 K=1536 → `cl=2x4x1` (id=9). noBIAS rank-1 is `tile=128x256
cl=2x1x1` (id=3) uniformly — same family fc2_w3x targets. fc1_w3 uses
2-CTA cluster (2x1x1) like fc2_w3x; cuBLASLt's GELU+BIAS family runs on
4-CTA clusters (2x2x1). The 50 µs MXFP8 gap may be a function of cluster
choice as much as algo family.


## cuBLASLt reference

`tools/probe_cublaslt.sh` (probe 1) enumerates every heuristic, times each,
reports rank-1. `bench/fc_problem.cuh` is the single source of truth for
descriptor / layout / bias-dtype across `cublaslt_introspect` and
`cublas_bench`.

**Critical knob: bias dtype = BF16.** cuBLASLt's fused-FP8-bias kernels on
sm_100a / CUDA 13.0 only enumerate when `BIAS_DATA_TYPE = CUDA_R_16BF`. FP32
bias gets you 0 algos. Layout orientation `[N,M]` vs `[M,N]` doesn't change
algo enumeration — but DOES change kernel runtime, especially at FC1 dims
(see below).

History (now fully resolved): pre-fix introspect ran [M,N] (transposed) with
BF16 bias → enumerated algos but on a different problem geometry, producing
the 1.046 ms FC2 / 1.894 ms FC1 references that ended up in CLAUDE.md. The
"FP32 bias to match cublas-bench" commit (`be6198c`) blocked enumeration
without us realizing — cublas-bench had been silently returning 0 fused
algos for the same FP32-bias reason. `rank1.sass` is real (real cuBLASLt
FP8 BIAS_ONLY kernel, BF16 bias) but its 1.046 ms timing was for the
transposed [M,N] geometry.

### FC2 K=3072 (production [N,M], BF16 bias)

| variant                                       | ms     | algo     |
|-----------------------------------------------|--------|----------|
| cuBLASLt fused BIAS_ONLY rank-1 (PerTensor)   | 1.028  | algoId=66 tile=128x256 NS=36 cluster=2x1x1 (id=3) |
| cuBLASLt fused BIAS_ONLY rank-1 (MXFP8)       | 1.117  | algoId=66 tile=128x256 NS=36 cluster=2x1x1 (id=3) |
| cuBLASLt GEMM-only rank-1                     | 1.043  | per-tensor, no epilogue |
| cuBLASLt unfused (GEMM + post-kernel bias)    | 1.546  | sequential |
| **fc2_w3x** (bias-only, fully fused)          | **1.001** | gflip_blkswap TD=54 |

fc2_w3x beats both fused rank-1 paths (-27 µs PerTensor, -116 µs MXFP8) and
the GEMM-only path (-42 µs while fused).

### FC1 K=768 (production [N,M], BF16 bias)

| variant                                       | ms     | algo     |
|-----------------------------------------------|--------|----------|
| cuBLASLt fused GELU+BIAS rank-1 (PerTensor)   | 2.414  | algoId=71 tile=128x256 NS=36 cluster=2x2x1 (id=6) |
| cuBLASLt fused GELU+BIAS rank-1 (MXFP8)       | 1.951  | algoId=66 tile=128x256 NS=36 cluster=2x1x1 (id=3) |
| cuBLASLt fused BIAS_ONLY (no GELU, hypothetical) | 1.520 | algoId=66 — much faster without GELU |
| cuBLASLt GEMM-only rank-1                     | 1.363  | per-tensor |
| cuBLASLt unfused (GEMM + post-kernel GELU+bias) | 4.320 | sequential |
| **fc1_w3** (zigzag TD=11 + ks=1, fully fused) | **1.998** | |

fc1_w3 beats per-tensor fused (-416 µs) but trails MXFP8 fused (+47 µs).
**MXFP8 wins by switching algo families:** algoId=66 (the BIAS_ONLY family)
+ cluster=2x1x1 instead of PT's algoId=71 (GELU+BIAS family) + cluster=2x2x1.
The MXFP8 codepath effectively runs the BIAS_ONLY kernel and folds GELU into
the same kernel-internal apply pass that MXFP8 already needs for VEC32_UE8M0
scales — so GELU is "free" piggyback while PT pays the algoId=71 GELU pass.
fc1_w3 leaves ~50 µs vs MXFP8 at FC1's K=768 geometry; the lever is matching
this 2x1x1 cluster choice + algoId=66 family discipline.

The +894 µs jump from BIAS_ONLY (1.520) to GELU+BIAS (2.414) on the
PerTensor path tells you GELU is expensive in cuBLASLt's algoId=71 family.
fc1_w3's fused-GELU path doesn't pay this — likely because we vectorize
GELU directly in the epilogue compute warps without an extra pass.

**MXFP8 algo enumeration is uniform** across every (FC1, FC2) × every K
we've measured: algoId=66 tile=23 (128x256) NS=36 cluster=3 (2x1x1)
splitk=1 swizzle=0. PerTensor varies cluster_id by problem (FC2 BIAS_ONLY
always 2x1x1; FC1 GELU+BIAS shifts 4x4x1/2x2x1/2x4x1 by K). MXFP8 only ever
uses the BIAS_ONLY-family kernel — even when the epilogue requests
GELU+BIAS — apparently because the GELU+BIAS algoId=71 family hasn't been
ported to the VEC32_UE8M0 codepath. Source: `data/mxfp8_introspect_20260506/`.

### FC2 K-sweep (cuBLASLt fused BIAS_ONLY rank-1, PerTensor; N=768)

apples-to-apples both columns are BIAS_ONLY (fc2_w3x is bias-only fused;
cuBLASLt is fused BIAS_ONLY rank-1). Prior table mixed kernels and had
"heur ERR" for K∈{1024,2048} from FP32-bias enumeration failure — fixed.

| K    | cuBLASLt fused | fc2_w3x (bias-only) | gap     |
|------|----------------|---------------------|---------|
| 1024 | 0.4548         | 0.5466              | +91.8 µs |
| 2048 | 0.7254         | 0.7230              | −2.4 µs (tie) |
| 3072 | **1.0270**     | **1.0043**          | **−22.7 µs** |
| 4096 | 1.3462         | 1.4055              | +59.3 µs |
| 6144 | 1.9996         | 1.9762              | −23.4 µs |
| 8192 | 2.7378         | 2.6711              | −66.7 µs |

All cuBLASLt K values pick algoId=66 tile=23 (128x256) NS=36 cluster=2x1x1
(id=3). MXFP8 picks the same kernel identity at every K (just slower).

**Two new losses surfaced:**
1. **K=1024 +92 µs** — cuBLASLt's same algoId=66 family runs much tighter at
   small K. Matches the dim-sweep K=1024 universal loss (+5 to +29% across N).
2. **K=4096 +59 µs** — surprising; we're winning at K=3072 and K=6144 but
   losing at K=4096. Likely N=768/K=4096 is in cuBLASLt's sweet spot (32
   K-iters with 2-CTA cluster waves of 147 tiles at perfectly tuned tile=23).
   Probably actionable — tune fc2_w3x basin at K=4096.

K=2048 is a tie. Production point K=3072 holds at −23 µs. Long-K wins (K=6144,
K=8192) hold.

### FC1 K-sweep (cuBLASLt fused GELU+BIAS rank-1; N=3072)

apples-to-apples GELU+BIAS both sides. cuBLASLt PerTensor via introspect
rank-1; cuBLASLt MXFP8 via `cublas-bench-fc1` (the only path that hits the
MXFP8 codepath). fc1_w3 with the production tune (zigzag TD=11 +
K_STAGGER=1 + auto NO_PREFILL).

| K    | cuBLASLt PT | cuBLASLt MXFP8 | fc1_w3 | Δ vs PT  | Δ vs MXFP8 |
|------|-------------|----------------|--------|----------|------------|
| 512  | 2.392       | 1.923          | 2.123  | −269 µs  | +200 µs    |
| 768  | **2.413**   | **1.951**      | **2.025** | **−388 µs** | **+74 µs** |
| 1024 | 2.465       | 2.055          | 2.081  | −384 µs  | +26 µs     |
| 1536 | 2.512       | 2.445          | 2.532  | +20 µs   | +87 µs     |
| 2048 | 2.687       | 3.071          | 3.34   | +653 µs  | +269 µs    |
| 3072 | 3.952       | 4.470          | 4.129  | +177 µs  | −341 µs    |
| 4096 | 5.249       | 5.838          | 5.709  | +460 µs  | −129 µs    |

cuBLASLt PT picks algoId=71 tile=23 (128x256) NS=36 across K. cluster shape
varies: 4x4x1 / 2x2x1 / 2x2x1 / 2x2x1 / 2x4x1 / 2x2x1 / 2x2x1 from K=512→4096
(cluster_id 10/6/6/6/9/6/6). algoId=71 = GELU+BIAS family; FC2 BIAS_ONLY uses
algoId=66 with cluster=2x1x1 (id=3).

**Three regions:**
1. **Small K (≤1024)** — fc1_w3 tracks MXFP8 within ~25-200 µs, decisively
   beats PT (-269 to -388 µs). Production K=768 lands here. fc1_w3 was
   tuned for this.
2. **Tie zone (K=1536)** — all three within 87 µs; ours +20 µs vs PT.
3. **Large K (≥2048)** — ours decisively LOSES vs PT at K=2048 (+653 µs)
   and stays bad through K=4096. cuBLASLt's PerTensor switches algorithm
   tier at K=2048 (cluster=9 instead of 6); fc1_w3 doesn't have that gear.
   At K∈{3072,4096} ours BEATS MXFP8 by 129-341 µs but trails PT —
   indicates cuBLASLt's PT path has a better large-K kernel that MXFP8
   doesn't get.

The K=768 production point reproduces 2.025 ms in 4-rep min — within run
variance of the published 1.998 reference. PT/MXFP8 references match
CLAUDE.md (2.413 / 1.951).

### Rank-1 decode (FC2 K=3072 BIAS_ONLY, [N,M] BF16 bias, PerTensor)

Kernel name pattern: `nvjet_sm100_qqtst_<M>x<N>_128x<NS>_<CM>x<CN>_[2cta_]<h|v>_<...>_T<A><B>`.
`2cta` = `cta_group::2`. Algo enumeration:

| rank | algoId | tile_id | tile     | NS | cluster_id | cluster | ms     |
|------|--------|---------|----------|----|------------|---------|--------|
| 1    | 66     | 23      | 128x256  | 36 | 3          | 2x1x1   | 1.0277 |
| 2    | 66     | 29      | 128x160  | 36 | 3          | 2x1x1   | 1.1128 |
| 3    | 66     | 31      | 192x128  | 36 | 3          | 2x1x1   | 1.2252 |

Tile 23 = 128x256 = our exact geometry. NS=36 in the algoConfig encodes
"AUTO" (NS resolved per kernel; reads as 36 = 0x24 = AUTO marker). cluster=3
is the cuBLASLt enum value for `(2,1,1)` 2-CTA cluster
(`CUBLASLT_CLUSTER_SHAPE_2x1x1`). `rank1.sass` (dumped from pre-fix [M,N]
runs) is the same algo family — the SASS opcodes are real, the ms timing
was for the transposed problem.

**Cluster_id → shape map** (subset of `cublasLtClusterShape_t`):

| id | shape  | id | shape  | id | shape  | id | shape  | id | shape   |
|----|--------|----|--------|----|--------|----|--------|----|---------|
| 2  | 1x1x1  | 3  | 2x1x1  | 4  | 4x1x1  | 5  | 1x2x1  | 6  | 2x2x1   |
| 7  | 4x2x1  | 8  | 1x4x1  | 9  | 2x4x1  | 10 | 4x4x1  | 11 | 8x1x1   |
| 12 | 1x8x1  | 13 | 8x2x1  | 14 | 2x8x1  | 15 | 16x1x1 | 16 | 1x16x1  |

Full map encoded in `tools/dim_sweep_w3x.py:CLUSTER_SHAPE_NAME` and
`tools/dim_sweep_fc1.py:CLUSTER_SHAPE_NAME`. The introspect tool now also
runs MXFP8 (`./cublaslt-introspect <M> <N> <K> <epi> 1`) — see
`data/mxfp8_introspect_20260506/`.

## Status (2026-04-30)

fc2_w3x bias-only at 1.00092 ms with `gflip_blkswap` (TD=54), beats cuBLASLt
fused PerTensor rank-1 (1.028 ms) by 27 µs and MXFP8 rank-1 (1.117 ms) by
116 µs; W5 MMA-ceiling-bound (~12482 cyc/tile ≈ 24×520 cyc/iter), tensor
pipe 95.84% active. Tree state: bias-preload default (Δ=−1.73 µs at
z=−23.23 STRONG, n=128), STSM default (matches FP8-bias-only SASS pattern
from rank1.sass; LDTM_X32 ties at MMA floor — Δ=−0.021 µs), 4D packed-tile
output ABI (`[TILES_M, TILES_N, TM*2, TN]`; host `pack_idx_C(m,n)`);
SASS-level epi tuning exhausted.
Histories: `memory/project_w3x_bias_preload_win.md`, `project_w3x_packed_c_abi.md`.

**Realistic remaining headroom ~1-3 µs** — largest single target by past-win
standards (bias-preload 1.7 µs, STSM 0.4 µs); probably needs new lever class.

**OPEN GOAL (2026-06-30): a 100%-valid residual-carrying NS=6 kernel.**
fc2_w3's fused residual path at NS=6/ES2 is NOT race-free — ~0.04% sparse output
corruption (~1/20 spot-check fail), root-caused to the epilogue-staging recycle
(W2-residual-ring ↔ `consumed_mbar` handshake, marginal at ES depth 2). Proven
2026-06-30: GEMM_ONLY (NS6/ES1) and BIAS_ONLY (NS6/ES2) are both 20/20 valid +
deterministic, but **no residual-carrying config is provably clean at NS=6**.
This is now a correctness REQUIREMENT, not a perf nicety — residual is the
production path. Constraint: must hold NS=6 (the 1.063 ms floor); dropping to
NS=5 for a deeper ring is the `fc2_w3y` path (+71 µs) and does NOT meet this goal.
Full analysis + fix paths: `memory/project-fc2-w3-epilogue-race.md`.

**Next:** close the residual NS=6 race. The fused-residual *port* to fc2_w3x
(`fc2_w3y`) is a DEAD END — see Dead ends below — but the race lives in the
legacy `fc2_w3` residual path itself and must be fixed there. Residual's home
stays `fc2_w3` (NS=6, 1.063 ms) — but it must become bit-exact across launches.

## Dead ends — do NOT retry

Full chronological log + per-item memory files: `memory/MEMORY.md`. Headlines:

- **fc2_w3y (residual on fc2_w3x) — DEAD END (2026-06-28).** w3x's 1.001 floor is
  a *zero-slack* MMA: under PREFILL it free-runs with NO consumer back-pressure
  (`MBAR_TMEM_CONSUMED` is `#ifdef NO_PREFILL` only), so correctness is pure
  rate-ordering on the 2-deep TMEM. Residual slows the consumer → the marginless
  MMA laps the TMEM. Severity scales with NS: **NS=6 deadlock / NS=5 sparse
  accumulator corruption (2/32) / NS=4 valid but 1.243 ms** (regression). The
  1.001 floor and a fused residual are mutually exclusive in one kernel —
  residual needs the slack the floor removed. Tried across Rounds 4-6: dedicated
  residual ring + decoupled handshake; LDSM gather (NS=5 valid 1.134, still >
  legacy 1.063); X32+uint4 "cheap" read (bank-conflicted, no win). bias-only
  `LDTM_X32` PASSES at NS=6 1.004 → proves the wedge is consumer cost, not the
  store. **Residual stays in `fc2_w3` (NS=6, 1.063, full-tile prefetch +
  xor-swizzled uint4 read).** Reverted to f8b70b5, fc2_w3y.cu deleted.
  `memory/project-fc2-resadd-port.md`.
- **Source-level epi tuning** — ptxas owns STS layout. CUTLASS_LOOP, FP32_EPILOGUE,
  cvta.shared, NUM_EPI_STAGES, stmatrix variants — identical SASS.
- **Cross-warp STS clustering (intra-warp)** — SELF_LOAD/STAGGER, SASS reorder
  zero effect (wrong axis). *Inter-cluster* arrival into STS/TMA-store IS
  ordering-controlled (g-s).
- **Hand-written PTX `fc2_w3x.ptx`** — byte-identical SASS. PTX has no UR type;
  ptxas owns R-vs-UR. Frozen at `fc2-w3x-ptx`.
- **K_UNROLL** — u1/u2/u3/u4/u8 regress 87–197 µs (UR datapath collapses on
  non-N_STAGES-multiples). u6/u12/u24 tie default; `K_UNROLL=24` shrinks SASS
  39% — free cleanup. `memory/project_k_unroll_sweep.md`.
- **Cluster-axis swap** — B200 hard-rejects `(1,2,1)`/`(1,1,2)` cluster_dims.
  cuBLASLt `2x1`/`1x2` is logical labeling inside X-axis grid (`rank1.sass`
  encodes cluster (2,1,1)). 2-CTA clusters X-axis-only.
- **KERN_3WARP merge** — W4 TMA into W5 MMA regresses 1.006→1.172 ms (+166 µs).
  W4 empty_wait is hw-sleep, not free issue. Structural warp floor = 4.
- **EPI_2WARP + DROP_LEAD_BARSYNC marginal opt-in (n=128):** v10011 −0.27 µs at
  z=−3.49; v10110 (no EPI_2WARP) +2.54 µs at z=+34.9. Stays opt-in — DROP_LEAD
  ships cross-warp STS-before-TMA race; EPI_2WARP fc2_w3x-only.
- **LDTM_X32 ties STSM at MMA floor (n=512):** Δ=−0.021 µs. STSM stays default
  by rank-1 SASS parity (rank1.sass is the real cuBLASLt FP8 BIAS_ONLY
  algoId=66 kernel — opcodes apply). **LDTM_X64 forces NUM_EPI_STAGES=2→1 →
  +14 µs STRONG** (n=16) — confirms NS_EPI=2 is worth ~14 µs.
- **fc2_w3x post-WIN levers (all ±3 µs or regression):** subpass 8→4, cross-tile
  TMA carry, SWIZZLE_64B, DROP_TRAIL_BARSYNC, WAIT_GROUP_READ, **XPF_A/B**
  (Bonferroni-confirmed regressions; macros removed), CHET/PMIX/INGH,
  **gflip_cidperm** (TD=55, +1568 cyc DECISIVE; bloom-filter overshoot caught
  this), **STAGGER=2 split-mbar** (+3 µs across 11 dispatches; macro removed),
  DG sweep, native BF16 epi (kept ±0, cleaner).
- **Older dead variants:** TD=1/5/6/7, COL_LOCK, 4-CTA TMA multicast (deadlock),
  mbar→SMEM polling, L2 cache hints, dgphase/dgnrot, fc2_ldg, fc2_hybrid,
  N-batch / phase-offset / Group-3 (pre-PACKED_TILES — re-test before citing).
- **Workstealing dispatch STRIPPED from fc1_w3.cu/fc2_w3.cu (2026-06-27):** all
  dynamic-dispatch code removed — fc2: TD∈{1,2,3,4,6,7} (atomic + grid-nonpersistent);
  fc1: TD=4 — plus helper flags ATOMIC_TILES / ROW_STEAL / TAIL_STEAL / COL_LOCK /
  LEAN_DISPATCH and the dedicated scheduler warp (W7 fc2 / SCHED_WARP fc1) + g_tile_ctr.
  Static-only now: TD=0 (Group-3 strided / BIDIR_SNAKE) + TD≥8 (swizzles in
  tile_dispatch.cuh). fc2_w3.cu 4117→3426 lines, fc1_w3.cu 1888→1445.
  **Provably SASS-identical** for production (TD=0/8/11): pp-diff + cubin SASS
  byte-match (only dynamic branches removed, already `#if`-false at those TDs).
  Workstealing always lost anyway (static > stealing ~30 µs; sched 1.101 / lean
  1.107 / rowsteal 1.242 vs dgswizzle 1.065 — bottleneck is long_scoreboard, not
  the warp). Removed Makefile targets fc2-w3-sched / fc2-w3-lean. The historical
  perf tables above are retained as the *why*. Dispatch is now compile-time only.
- **FC1 FORCE_PREFILL** — deadlocks at K_ITERS=6. NO_PREFILL guard mandatory.
- **fc1_w3x PER_WARP_STORE** — per-warp TMA store (private double-buffers, drop
  the cross-warp store bar.sync) to fix the 3.11 ms FC1 epilogue-exposed
  regression. CONCLUSIVELY DEAD (B200, 2026-06, via Modal): barrier-free
  (`__syncwarp` only) **crashes** — Xid 13 CGA "CTA Not Present", one
  `cta_group::2` CTA drains its private buffers and exits the persistent loop
  while its cluster peer still issues cluster ops. Adding back a full `bar.sync`
  (`PW_STORE_BARSYNC`) is correct but **3.177 ms = +63 µs vs the 3.114 ms serial
  default** — reintroduces the serialization it meant to remove. Hard proof the
  tid==0 serial store was never the FC1 bottleneck; the regression is exposed
  epilogue COMPUTE (K_ITERS=6 MMA shadow too short — identical store hides fine
  in fc2_w3x at K_ITERS=24/1.001 ms). Reverted to HEAD. NOT the store path. See
  `memory/project-fc1-w3x-epilogue-exposed.md`.
- **fc1_w3x WIDE_SUBPASS** — 64-col subpasses (4 instead of 8) via a
  `SUBPASS_CHUNKS` inner loop, grouping 2 of the fixed 32-col TMEM-load/STSM
  units into one staging buffer + ONE bar.sync pair + ONE TMA store (halves
  per-subpass sync/store-dispatch overhead). TESTED, NOT a production win (B200,
  2026-06-26, Modal): GELU production 3.116→3.133 ms = **wash** (repeats span
  3.114–3.137, ~23 µs run-noise > the gap); bias-only (`-DNO_GELU`) 2.317→2.270
  = **−47 µs real**. So the per-subpass overhead IS a genuine cost, but GELU's
  SFU-bound compute (808 µs, see below) masks it entirely at the production
  point — and FC1 always needs GELU, so the bias-only win has no home. Macro
  kept opt-in (default off, SUBPASS_CHUNKS=1 = byte-clean). **GELU/bias split
  (NO_GELU switch):** strip 1.353 / bias-only 2.317 / full GELU 3.125 — GELU =
  +808 µs but *cheaper* than cuBLASLt's 894 µs; the +797 µs deficit vs cuBLASLt
  bias-only (1.520) is the exposed bias/CVT/STS structure, not GELU. **Real FC1
  lever = GELU compute throughput (SFU/`MUFU.tanh` scheduling), not epilogue
  structure.** See `memory/project-fc1-w3x-epilogue-exposed.md`.

## Build and run

```bash
make fc2-w3x && ./fc2-w3x                   # ~1.001 ms (BEST, beats cuBLASLt fused rank-1 PerTensor 1.028 / MXFP8 1.117)
make fc2-w3x-strip && ./fc2-w3x-strip       # NS=6+PREFILL floor (~0.985 ms)
make fc2-w3x-ptx                            # hand-written PTX, byte-identical SASS

make fc2-w3x-tile-sweep                     # TILE_DISPATCH variants
./tools/sweep_fc2_w3x_swizzle.sh            # SWEEP=front for top tier
./tools/sweep_fc2_w3x_nanosleep.sh          # NS_CYC sweep
./tools/sweep_fc2_w3x_dg.sh                 # DG_GROUP_SIZE × INNER_T × STAGGER
./tools/sweep_fc2_w3x_prof.sh               # per-warp clock64
python3 tools/aggregate_prof.py data/<dir>
make fc2-w3x DFLAGS='-DPROFILE_CYCLES'      # |-DPROFILE_KI|-DPROFILE_TILE|-DPROFILE_W5

make fc1-w3x && ./fc1-w3x                   # FC1 GELU+BIAS (clean-sheet, target ~2.025 ms ± basin)
make fc1-w3x-tile-sweep                     # TILE_DISPATCH variants for FC1
make fc1-w3x-ks-sweep                       # K_STAGGER sweep (default ks=1)
make fc2-w3 && ./fc2-w3                     # fused ~1.060 (gflip_blkswap TD=54 default)
make fc2-w3-swizzle-sweep && ./fc2-w3-swizzle-sweep SWEEP=front REPS=200  # basin sweep (cyc)
make fc1-w3 && ./fc1-w3                     # FC1 legacy 1.998 (zigzag+ks=1) — fc1_w3x supersedes for non-residual

make -B fc2-w3 DFLAGS='-DM_TOTAL=464128 -DN_DIM=1024 -DK_DIM=2048'  # custom dims need -B
# Decomp: -DSTRIP_EPILOGUE / -DGEMM_ONLY

make fc2-cutlass && ./fc2-cutlass           # 1.226 reference
./tools/probe_cublaslt.sh                   # cuBLASLt rank-1 (BF16 bias, [N,M] layout)
bash tools/bench.sh --comprehensive         # cuBLASLt-rank-1-baselined
bash tools/ncu_bench.sh && python3 tools/ncu_anova.py
bash tools/ncu_fc2_w3x.sh --max --reps 3
bash tools/ncu_fc2_pipes.sh                 # dodges --set full deadlock
./tools/dim_sweep.sh --fast                 # fc2_w3 80 configs (M×N×K)
./tools/dim_sweep_w3x.py                    # fc2_w3x N×K pow2 grid (vs cuBLASLt BIAS_ONLY)
./tools/dim_sweep_fc1.py                    # fc1_w3 N×K grid (vs cuBLASLt GELU+BIAS, prod tune)
```

## Remote B200 via Modal (timing runs, not ncu)

`dummy_modal.py` builds one Makefile target on a Modal B200 and runs it —
faster turnaround than spinning up vast/verda for `clock64` cycle-timing and
`-DPROFILE_*` decomposition. **Replaces vast/verda for timing only; Modal's
CUDA image has no Nsight Compute and shared GPUs block perf counters, so
`ncu --set full` SASS-stall work still needs vast.ai.**

```bash
pip install modal && modal token new                                  # one-time
modal run dummy_modal.py                                              # fc1-w3x default
modal run dummy_modal.py --target fc2-w3x
modal run dummy_modal.py --target fc1-w3x --dflags "-DPER_WARP_STORE"
modal run dummy_modal.py --target fc1-w3x --dflags "-DPROFILE_CYCLES"
modal run dummy_modal.py --target fc2-w3 \
    --dflags "-DM_TOTAL=464128 -DN_DIM=1024 -DK_DIM=2048"
```

Mechanics: `nvidia/cuda:13.2.0-devel-ubuntu24.04` + `add_python="3.14"`
(needed for `gen/bias_switch_inc_*.cuh` codegen) + `apt_install("make")`.
Repo root mounted via `image.add_local_dir(".", "/root/src", ignore=[...])`
— `data/`, `*.log`, `.git`, `.claude`, `third_party`, CSVs excluded so each
run doesn't re-upload GB of benchmark artifacts. Binary name == target name
(in `/root/src`). `--rebuild` (default True) forces `make -B` — mandatory
because DFLAGS changes don't touch `.cu` mtime, same reason custom dims need
`-B`. Output is line-streamed so `@@SAMPLE`/`@@RESULT` appear live. **Never pipe
a Modal run through `tail`/`grep`/`head` — block-buffering swallows the
`@@RESULT`/`PASS`/Xid lines you actually need; redirect the FULL output to a log
(`modal run ... > run.log 2>&1`) and read the log.** Build and
run share one B200-attached container, so `-lcuda` links against the real
driver. Aggregation (`aggregate_prof.py`, `anova_1way.py`) stays local
against streamed stdout — don't ship it to the container. The deprecated
pre-1.0 `modal.Mount` / `mounts=` API does NOT work; use image-folded
`add_local_dir`.

## Key files

```
fc2_w3x.cu         FC2 bias-only (ACTIVE — beats cuBLASLt fused PerTensor & MXFP8 rank-1)
fc1_w3x.cu         FC1 GELU+BIAS (ACTIVE — clean-sheet port of fc2_w3x architecture)
fc2_w3x.ptx        Hand-written PTX, byte-identical SASS (frozen)
fc2_w3.cu          FC2 fused-residual (legacy, retained for residual path)
fc1_w3.cu          FC1 (legacy; superseded by fc1_w3x for non-residual)
swizzle_w3x.cuh    Shared 48 swizzle templates (TD=11..99) for fc1_w3x/fc2_w3x
epilogue_ops.cuh   Shared CVT_ADD/CVT_GELU_ADD macros + gelu_approx + pack_idx_C
gen/bias_switch_inc_<N>.cuh  Build-time codegen — see tools/gen_bias_switch.py
tile_dispatch.cuh  Legacy TD=8..58 used by fc1_w3 / fc2_w3 (NOT w3x family)
fc2_cutlass.cu     CUTLASS reference
kernel_common.cuh, kernel_body.cuh  Legacy w3 infra (NOT used by w3x family)
docs/STSM_STATUS.md, PURE_PTX_REWRITE_STRATEGY.md, BENCHMARKING.md
docs/fc2_w3x_ncu_sass.txt    ncu --set full per-SASS stalls (mbar 7×)
docs/fc2_w3x_ncu_details.txt 98.5% TC pipe (ignore STSM bank-conflict warns)
rank1.sass         Dumped cuBLASLt FP8 BIAS_ONLY kernel (real algoId=66 tile=128x256; SASS opcodes apply, but its 1.046 ms timing was on transposed [M,N] geometry — see "cuBLASLt reference" for correct [N,M] numbers)
bench/fc_problem.cuh       Shared cuBLASLt FP8 problem definition (BF16 bias [N,M]) — single source of truth for cublas_bench + cublaslt_introspect
tools/bench.sh, probe_cublaslt.sh, dim_sweep.sh, dim_sweep_w3x.py, dim_sweep_fc1.py
tools/gen_bias_switch.py   Codegen for bias-load switch chain (avoids local-mem spill)
tools/sweep_fc2_w3x_*.sh   tiles / dg / nanosleep / prof
tools/ncu_*.sh, ncu_anova.py, aggregate_prof.py
tools/analyze_swizzle.py, cluster_swizzle.py    structural metric + verdict
tools/anova_1way.py        paired ANOVA + AUC + d + η² + rank/win%
tools/sass_edit.py         SASS binary editor + CP-SAT scheduler
dummy_modal.py             Remote B200 build+run on Modal (timing/PROFILE_*, not ncu)
token_count.py             tiktoken budgeting
bench/                     TMA / MMA / stmatrix / cublaslt_introspect
data/                      Benchmark + ncu results
```

## SM100a hardware (B200-measured)

- STS.128: 27 cyc | LDS.128: 25 cyc @ILP=1, 3.5 cyc @ILP=7
- TMA load: 419 cyc (L2-warm) | TMA store: 197 cyc
- TMEM load (tcgen05.ld.sync): 2 cyc regardless of width/ILP
- MMA K-iter: 665 cyc (pipelined: 525.6 cyc/iter)
- STS scaling: 10→37 cyc at 8 warps (3.65×); LDS 4.5→16 cyc (3.56×)
- FFMA: ~free (1.36× at 8 warps); F2FP: zero contention (flat 2.0 cyc)

## Key constraints

- Target: sm_100a (B200, 148 SMs), `cta_group::2`, 74 clusters
- TMEM: 512 cols, single alloc for double buffering. SMEM: 228 KB/SM
- Inline PTX in fc2_w3.cu/fc1_w3.cu (no CUTLASS dependency)
- OFF_STAGING must be 1024-byte aligned for SWIZZLE_128B
- `fence.proxy.async.shared::cta` required before TMA store after st.shared
- N_STAGES + PREFILL kernel-side auto-picked from N_DIM + K_ITERS (3d6c1cb)
- BIAS_SMEM=1 default (-15 µs free); custom dims require `make -B`
- W0 K-loop TMA-sensitive: any global op (atomicAdd) costs +41–77% tma_issue.
  Non-critical-path global ops (W7 scheduler at tile-boundary) fine.

## Benchmarking

`docs/BENCHMARKING.md`. TL;DR:

- **Cycles, not ms.** `clock64()` per-CTA, `max_over_CTAs / N_TIMED`. Clock-freq
  invariant — required on vast.ai (no locked clocks).
- **Pass-major** (randomized block): outer pass p, inner variant v.
  `@@SAMPLE pass=p variant=v cyc=Y` per launch.
- **Trim** first 33–50% of passes (cold L2 + thermal ramp).
- **Paired analysis** by pass; report **AUC**, **Cohen's d**, **η²**, **mean
  rank**, **win%**. **No p-values** at large n.
- `tools/anova_1way.py --metric cyc --paired rep --trim 0.33` is canonical.
- **n thresholds (σ_residual ~1400 cyc):** n<5000 unreliable for sub-σ effects
  (Stage 1 lmrev DECISIVE n=2048 → Stage 2 mid-pack n=29420). MOD-band ~600 cyc:
  n≥10978; TIE-band ~150 cyc: n≥43910 — within-basin may be sub-resolution.
  Default 2-stage: REPS=2048 filter → REPS=43910 survivors.

## Working in this repo

Names say what, comments say why. No single-line `/**/`, no multi-line `//`, no
decorated block comments. Bare `/*`, undecorated lines, `*/`.

Don't narrate tool calls; don't echo file contents; parallelize independent
tool calls; offset/limit for large files.

**w3x shared-header structure (2026-05-07):** fc1_w3x.cu and fc2_w3x.cu share
three pieces of infrastructure — edit once, both kernels rebuild:
  - `swizzle_w3x.cuh`  — 48 swizzle templates + tile_swizzle_t / tile_in_group_t
  - `epilogue_ops.cuh` — CVT_ADD / CVT_GELU_ADD macros + gelu_approx + pack_idx_C
  - `gen/bias_switch_inc_<N>.cuh` — build-time codegen via tools/gen_bias_switch.py
    (Makefile rule `gen/bias_switch_inc_%.cuh`; new BIAS_REG_COUNT just needs the
    Makefile prereq line updated, .cu uses `#include "gen/bias_switch_inc_<N>.cuh"`)
SASS-verified zero codegen change vs prior in-line copies (cuobjdump diff = 0).
**Real fc1↔fc2 lever surface is now ~150 lines** (header / dim defines / NS
picker / GELU vs BIAS-only macro at the subpass site / K_STAGGER / golden ref).
Legacy w3 family (fc1_w3, fc2_w3) still uses kernel_common.cuh / kernel_body.cuh
/ tile_dispatch.cuh — do NOT cross-include between w3 and w3x families.

LLM context is the binding constraint — treat CLAUDE.md, docs, memory as a
token budget. Prefer brief pointers to topic files. `./token_count.py <file>`
is coarse proxy (tiktoken o200k_base); Claude tokenizer reports ~1.5× higher
on table-heavy markdown.
