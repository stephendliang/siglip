# SigLIP2 Vision Encoder — Hand-tuned Blackwell GEMM Kernels

Hand-tuned SM100a persistent GEMM for FC1/FC2 of `google/siglip2-base-patch16-224`.
FP8 (E4M3) → BF16, tcgen05 MMA, TMA, `cta_group::2`, 2-CTA clusters. Cross-compiled
on CPU VPS, runs B200 (148 SMs, 74 clusters). PE kernel done — see
`CLAUDE.md.mothballed`.

## Current best (B200, 2026-05-01)

| target | ms | kernel | dispatch | vs cuBLASLt rank-1 |
|---|---|---|---|---|
| FC2 K=3072 BIAS_ONLY (strip floor) | 0.98502 | `fc2_w3x` `-DSTRIP_EPILOGUE` | n/a | NS=6+PREFILL structural floor (1814685 cyc) |
| **FC2 K=3072 BIAS_ONLY (full)** | **1.00092** | `fc2_w3x` (bias-preload, STSM-only) | basin floor (default `gflip_blkswap` TD=54) | **−45 µs** vs 1.046; +16 µs exposed epi vs strip |
| FC2 K=3072 fused (+residual) | 1.063 | `fc2_w3` | dgswizzle TD=8 PACKED | (no apples-to-apples ref) |
| FC1 K=768 fused (+GELU+bias) | 1.998 | `fc1_w3` | zigzag TD=11 + K_STAGGER=1 | +104 µs vs 1.894 |

`fc2_w3x` = clean-sheet 6-warp persistent bias-only, beat rank-1.
`fc2_w3` = legacy 7-warp fused, production for residual path.

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
| Dispatch | FC2 fused: zigzag (TD=11) or dgswizzle (TD=8). FC2 bias-only: any `gflip_*` basin-floor — `gflip_blkswap` (TD=54) default. FC1: zigzag + K_STAGGER=1. | PACKED_TILES + odd ks helps FC1; FC2 wash on ks. |

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
| cuBLASLt rank-1 (L2) | ~520 | 1.046 ms | reference |

**Strip vs full = 15.9 µs / 31460 cyc / 214 cyc/tile** (1.7% of 12482 cyc/tile
MMA budget) — exposed epi coupling W0-W3 epi-end and W5's next-tile MMA via
`bar.sync` / `mbar_wait`. ~98% of epi hidden in MMA shadow; the 2% that isn't
is real headroom.

**Gap decomposition** (vs structural floor):
- **89 µs** (pure-MMA → strip): NS=6 staging bubble, unreachable without removing staging.
- **16 µs** (strip → fc2-w3x): exposed epi, real headroom.
- **61 µs** (strip → cuBLASLt rank-1): rank-1 has 4× more exposed epi.

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

## N×K dim sweep (pow2 grid, 2026-05-01, 16 cells)

`tools/dim_sweep_w3x.py` default = `N ∈ {256,512,1024,2048} × K ∈
{1024,2048,4096,8192}`. Cycles paired with cuBLASLt rank-1 via stream-serialized
`clock64()` sentinels (same SM-clock domain). Both EPI=3 BIAS_ONLY (cb_bias)
and EPI=0 plain GEMM (cb_none).

| N | K | K_it | NS | ours cyc | cb_bias cyc | cb_none cyc | Δb% | Δn% |
|---|---|---|---|---|---|---|---|---|
| 256 | 1024 | 8  | 5\* | 397.6k  | n/a       | 390.3k    | n/a   | +1.88 |
| 256 | 2048 | 16 | 6   | 634.5k  | n/a       | 664.9k    | n/a   | −4.57 |
| 256 | 4096 | 32 | 6   | 1152.4k | n/a       | 1172.1k   | n/a   | −1.68 |
| 256 | 8192 | 64 | 6   | 2150.6k | 2453.4k   | 2451.9k   | −12.34| −12.29|
| 512 | 2048 | 16 | 6   | 951.2k  | n/a       | 991.9k    | n/a   | −4.11 |
| 512 | 4096 | 32 | 6   | 1783.1k | 1767.7k   | 1753.7k   | +0.87 | +1.67 |
| 512 | 8192 | 64 | 6   | 3386.9k | 3451.5k   | 3446.9k   | −1.87 | −1.74 |
| 1024| 2048 | 16 | 6   | 1692.5k | 1817.1k   | 1773.3k   | −6.86 | −4.55 |
| 1024| 4096 | 32 | 6   | 3405.1k | 3302.0k   | 3292.6k   | +3.12 | +3.42 |
| 1024| 8192 | 64 | 6   | 6510.4k | 6525.2k   | 6532.7k   | −0.23 | −0.34 |
| 2048| 1024 | 8  | 5   | 2717.4k | 2289.9k   | 2135.3k   |+18.67 |+27.26 |
| 2048| 2048 | 16 | 5   | 3288.5k | 3434.0k   | 3422.8k   | −4.24 | −3.92 |
| 2048| 4096 | 32 | 5   | 6727.9k | 6510.7k   | 6497.5k   | +3.34 | +3.55 |
| 2048| 8192 | 64 | 5   | 12940.7k| 12908.1k  | 12908.6k  | +0.25 | +0.25 |

\*N=256 K=1024 picks NS=5 via `min(NS_BY_N=6, K_ITERS−3=5)`. K=1024 N∈{512,1024}
was FAIL@NS=6; auto-picker now NS=5. n/a = sparse cuBLASLt heuristic at small-N
+ medium-K (per-tensor FP8 BIAS_ONLY); EPI=0 covers all but FAIL.

**Production point K=3072 N=768: −4.15% in cycles** (1845.4k vs 1925.3k) —
bigger margin than 1.046→1.001 ms gap suggests. Earlier non-pow2 sweep had one
loss (N=K=1536 K_iters=12 NO_PREFILL, +1.94%) — square × NO_PREFILL cap.

**Three loss patterns:**
1. **N=2048 K=1024 catastrophe (+27%, eff=0.54).** NS=5 + NO_PREFILL + gap=3
   stack. Probably needs different short-K kernel (no PREFILL ever, smaller tile).
2. **K=4096 systematic loss N≥512 (+1.7 to +3.5%).** Most actionable. cuBLASLt
   eff jumps 0.81→0.88 K=2048→4096; ours flat 0.86. Pending: harness now parses
   `# Winner:` from `cublaslt-introspect`. Suspects: 256×96 tile (TILE_ID=495),
   deeper NS, split-K.
3. **N=2048 NS=5 SMEM tax (~1-3.5%).** 1-stage latency-hide loss; gap shrinks
   as K grows. LDTM_X64 dead-end says NUM_EPI_STAGES=1 costs +14 µs.

**Sweet spots:** K=2048 across all N (−4 to −7%, K_ITERS=16 past PREFILL gap=10);
small N (N=256, −1 to −12%, cuBLASLt heuristic floor degrades faster at low
tile/cluster).

See `memory/project_w3x_dim_sweep_vs_cublas.md` for cuBLASLt sparse-heuristic
gaps (K=1536 zero algos almost everywhere; EPI=0 recovers most).

## cuBLASLt rank-1 reference

`tools/probe_cublaslt.sh` (probe 1) enumerates every heuristic, times each,
reports rank-1. Reference target — we beat it at FC2 K=3072 BIAS_ONLY. Earlier
comparisons against `cublas-bench-fc2` measured the **default heuristic pick**,
not rank-1.

### FC2 K-sweep

| K    | cuBLASLt  | best ours | gap |
|------|-----------|-----------|-----|
| 1024 | ERR       | 0.859 (lean)        | n/a |
| 2048 | ERR       | 0.922 (zigzag)      | n/a |
| 3072 | **1.046** | **1.008** (w3x bias-only, blkswap TD=54) / 1.064 (dgsw fused) | **−38 µs** / +18 µs |
| 4096 | **1.360** | 1.476 (dgsw)        | +116 µs |
| 6144 | **1.997** | 2.007 (dgsw)        | +10 µs |
| 8192 | **2.682** | 2.731 (lean)        | +49 µs |

FC1 K=768: cuBLASLt **1.894** vs ours 1.998 (zigzag+ks=1) → +104 µs (+5.5%).

### Rank-1 decode (B200, 2026-04-20, FC2 K=3072)

Kernel name: `nvjet_sm100_qqtst_<M>x<N>_128x<NS>_<CM>x<CN>_[2cta_]<h|v>_<...>_T<A><B>`.
`2cta` = `cta_group::2`. `h/v` = TMA multicast axis. `bz_bias` = bias-only epi.

| listed | tile | NS | cluster | cta_grp | ms |
|---|---|---|---|---|---|
| L1 | 176x128 | 8 | 1x2 | 2 | 1.0454 |
| **L2†** | **128x256** | **6** | **2x1** | **2** | **1.0457** |
| L3 | 128x192 | 7 | 2x1 | 2 | 1.094 |
| L4 | 256x256 | 4 | 2x1 | 2 | 1.192 |

† **"Rank-1" = L2** (our exact geometry). L1 wins by 0.3 µs noise + uses NS=8
(not SMEM-feasible at our tile). Top 5 listings all `cta_group::2`. Not split-K
(`splitk=1`), not CUTLASS-style swizzle. SASS at `rank1.sass`.

Tile enum (from `cublasLt.h`): `23=128x256, 24=256x128, 32=128x192, 197=168x128,
201=176x128, 495=256x96, 535=320x192`. `stages=36 = 128xAUTO`, NS resolved per
kernel variant. K=1024/2048 ERR — one heuristic IMAs on device.

## Status (2026-04-30)

fc2_w3x bias-only at 1.00092 ms with `gflip_blkswap` (TD=54); W5 MMA-ceiling-bound
(~12482 cyc/tile ≈ 24×520 cyc/iter), tensor pipe 95.84% active. Tree state:
bias-preload default (Δ=−1.73 µs at z=−23.23 STRONG, n=128), STSM mandatory
(rank-1 SASS opcode parity), 4D packed-tile output ABI (`[TILES_M, TILES_N,
TM*2, TN]`; host `pack_idx_C(m,n)`); SASS-level epi tuning exhausted.
Histories: `memory/project_w3x_bias_preload_win.md`, `project_w3x_packed_c_abi.md`.

**Realistic remaining headroom ~1-3 µs** — largest single target by past-win
standards (bias-preload 1.7 µs, STSM 0.4 µs); probably needs new lever class.

**Next:** port `fc2_w3x` from bias-only to fused-residual.

## Dead ends — do NOT retry

Full chronological log + per-item memory files: `memory/MEMORY.md`. Headlines:

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
  cuBLASLt `2x1`/`1x2` is logical labeling inside X-axis grid (rank1.sass:
  cluster (2,1,1)). 2-CTA clusters X-axis-only.
- **KERN_3WARP merge** — W4 TMA into W5 MMA regresses 1.006→1.172 ms (+166 µs).
  W4 empty_wait is hw-sleep, not free issue. Structural warp floor = 4.
- **EPI_2WARP + DROP_LEAD_BARSYNC marginal opt-in (n=128):** v10011 −0.27 µs at
  z=−3.49; v10110 (no EPI_2WARP) +2.54 µs at z=+34.9. Stays opt-in — DROP_LEAD
  ships cross-warp STS-before-TMA race; EPI_2WARP fc2_w3x-only.
- **LDTM_X32 ties STSM at MMA floor (n=512):** Δ=−0.021 µs. STSM stays default
  by rank-1 SASS parity. **LDTM_X64 forces NUM_EPI_STAGES=2→1 → +14 µs STRONG**
  (n=16) — confirms NS_EPI=2 is worth ~14 µs.
- **fc2_w3x post-WIN levers (all ±3 µs or regression):** subpass 8→4, cross-tile
  TMA carry, SWIZZLE_64B, DROP_TRAIL_BARSYNC, WAIT_GROUP_READ, **XPF_A/B**
  (Bonferroni-confirmed regressions; macros removed), CHET/PMIX/INGH,
  **gflip_cidperm** (TD=55, +1568 cyc DECISIVE; bloom-filter overshoot caught
  this), **STAGGER=2 split-mbar** (+3 µs across 11 dispatches; macro removed),
  DG sweep, native BF16 epi (kept ±0, cleaner).
- **Older dead variants:** TD=1/5/6/7, COL_LOCK, 4-CTA TMA multicast (deadlock),
  mbar→SMEM polling, L2 cache hints, dgphase/dgnrot, fc2_ldg, fc2_hybrid,
  N-batch / phase-offset / Group-3 (pre-PACKED_TILES — re-test before citing).
- **FC1 FORCE_PREFILL** — deadlocks at K_ITERS=6. NO_PREFILL guard mandatory.

## Build and run

```bash
make fc2-w3x && ./fc2-w3x                   # ~1.001 ms (BEST, beats rank-1)
make fc2-w3x-strip && ./fc2-w3x-strip       # NS=6+PREFILL floor (~0.985 ms)
make fc2-w3x-ptx                            # hand-written PTX, byte-identical SASS

make fc2-w3x-tile-sweep                     # TILE_DISPATCH variants
./tools/sweep_fc2_w3x_swizzle.sh            # SWEEP=front for top tier
./tools/sweep_fc2_w3x_nanosleep.sh          # NS_CYC sweep
./tools/sweep_fc2_w3x_dg.sh                 # DG_GROUP_SIZE × INNER_T × STAGGER
./tools/sweep_fc2_w3x_prof.sh               # per-warp clock64
python3 tools/aggregate_prof.py data/<dir>
make fc2-w3x DFLAGS='-DPROFILE_CYCLES'      # |-DPROFILE_KI|-DPROFILE_TILE|-DPROFILE_W5

make fc2-w3 && ./fc2-w3                     # fused 1.063 (dgswizzle TD=8)
make fc1-w3 && ./fc1-w3                     # FC1 1.998 (zigzag+ks=1)

make -B fc2-w3 DFLAGS='-DM_TOTAL=464128 -DN_DIM=1024 -DK_DIM=2048'  # custom dims need -B
# Decomp: -DSTRIP_EPILOGUE / -DGEMM_ONLY

make fc2-cutlass && ./fc2-cutlass           # 1.226 reference
./tools/probe_cublaslt.sh                   # cuBLASLt rank-1 (TRUE ceiling)
bash tools/bench.sh --comprehensive         # rank-1-baselined
bash tools/ncu_bench.sh && python3 tools/ncu_anova.py
bash tools/ncu_fc2_w3x.sh --max --reps 3
bash tools/ncu_fc2_pipes.sh                 # dodges --set full deadlock
./tools/dim_sweep.sh --fast                 # fc2_w3 80 configs (M×N×K)
./tools/dim_sweep_w3x.py                    # fc2_w3x N×K pow2 grid (vs cuBLASLt)
```

## Key files

```
fc2_w3x.cu         FC2 bias-only (ACTIVE — beats rank-1)
fc2_w3x.ptx        Hand-written PTX, byte-identical SASS (frozen)
fc2_w3.cu          FC2 fused-residual (ACTIVE)
fc1_w3.cu          FC1 (ACTIVE)
fc2_ws.cu          FC2 warp-specialized w/ rank-1 warp retirement
tile_dispatch.cuh  Shared TD=8..58 (CHET/PMIX/INGH, gflip family)
fc2_cutlass.cu     CUTLASS reference
fc2_hybrid.cu, fc2_ldg.cu, fc2.cu  DEAD
kernel_common.cuh, kernel_body.cuh  Shared infra
docs/STSM_STATUS.md, PURE_PTX_REWRITE_STRATEGY.md, BENCHMARKING.md
docs/fc2_w3x_ncu_sass.txt    ncu --set full per-SASS stalls (mbar 7×)
docs/fc2_w3x_ncu_details.txt 98.5% TC pipe (ignore STSM bank-conflict warns)
rank1.sass         Dumped cuBLASLt rank-1
tools/bench.sh, probe_cublaslt.sh, dim_sweep.sh, dim_sweep_w3x.py
tools/sweep_fc2_w3x_*.sh   tiles / dg / nanosleep / prof
tools/ncu_*.sh, ncu_anova.py, aggregate_prof.py
tools/analyze_swizzle.py, cluster_swizzle.py    structural metric + verdict
tools/anova_1way.py        paired ANOVA + AUC + d + η² + rank/win%
tools/sass_edit.py         SASS binary editor + CP-SAT scheduler
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

LLM context is the binding constraint — treat CLAUDE.md, docs, memory as a
token budget. Prefer brief pointers to topic files. `./token_count.py <file>`
is coarse proxy (tiktoken o200k_base); Claude tokenizer reports ~1.5× higher
on table-heavy markdown.
