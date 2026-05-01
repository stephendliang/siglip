# SigLIP2 Vision Encoder — Hand-tuned Blackwell GEMM Kernels

Hand-tuned SM100a persistent GEMM kernels for FC1 and FC2 layers of `google/siglip2-base-patch16-224`.
FP8 (E4M3) inputs, BF16 output, tcgen05 MMA, TMA, `cta_group::2` with 2-CTA clusters.
Cross-compiled on CPU VPS, runs on B200 (148 SMs, 74 clusters). PE kernel is done — see `CLAUDE.md.mothballed`.

## Current best (B200, 2026-04-30)

| target | ms | kernel | dispatch | vs cuBLASLt rank-1 |
|---|---|---|---|---|
| FC2 K=3072 BIAS_ONLY (strip floor) | 0.98502 | `fc2_w3x` `-DSTRIP_EPILOGUE` | n/a | NS=6+PREFILL structural floor (1814685 cyc) |
| FC2 K=3072 BIAS_ONLY (full) | **1.00092** | `fc2_w3x` (bias-preload default, STSM-only) | any of `{blkswap, blkx5/6/7, blk_qrt0/2/3}` (basin floor) | **−45 µs** (rank-1: 1.046); +16 µs exposed epi vs strip |
| FC2 K=3072 fused (+residual) | 1.063 | `fc2_w3` | dgswizzle TD=8 PACKED | (no apples-to-apples ref) |
| FC1 K=768 fused (+GELU+bias) | 1.998 | `fc1_w3` | zigzag TD=11 + K_STAGGER=1 | +104 µs (rank-1: 1.894) |

`fc2_w3x` = clean-sheet 6-warp persistent bias-only kernel that beat rank-1.
`fc2_w3` = legacy 7-warp fused kernel still used for the production residual path.

## Status tables (PACKED_TILES parity, 2026-04-17/18)

All `-DPACKED_TILES`. Static swizzles work under default pipeline settings.

### FC2: [928256, 3072] x [3072, 768]^T + bias + residual

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

Static swizzles beat work-stealing by ~30µs. Strip floor ~0.988ms.

### FC1: [928256, 768] x [768, 3072]^T + bias + GELU

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

FC1 dispatch lever is bigger than FC2's. Odd K_STAGGER (1 or 3) helps FC1; ks=2 hurts.
ncycle/nsnake have f-g≈0 (zero epi/mainloop overlap) — pathological dispatch.

`fused = strip + (g-s) + (f-g)`. Each gap is a separate axis: g-s = store contention
(cluster-wavefront N-column diversity), f-g = epilogue overlap with next-tile mainloop
(K_ITERS-limited).

## Kernel structure

Warp-specialized, 7 warps (224 threads), `cta_group::2`, `__cluster_dims__(2,1,1)`:

| Warp | Role | Notes |
|---|---|---|
| W0 | TMA Load (A+B) | Memory-pipeline-sensitive — no global ops in K-loop |
| W1 | tcgen05.mma K-loop | TMEM 512 cols double-buffered |
| W2 | EpilogueLoad | TMA loads residual (FC2) into SMEM, circular 2-stage pipe |
| W3-W6 | Epilogue compute | LDS + TMEM ld + math + CVT + STS + TMA store |
| W7 | Scheduler (TD=4/LEAN) | atomicAdd tile counter, mbarrier broadcast |

`fc2_w3x` differs: 6 warps (W0-W3 epi, W4 TMA, W5 MMA CTA0-only). No W7. `buf = tt & 1`.

Tile: 256x256x128. K_ITERS=K_DIM/128. FC2: K=3072 (24), FC1: K=768 (6).

## Pipeline depth (NS6)

6-stage mainloop pipeline uses 227KB of 228KB SMEM. Each stage holds 256x128 A+B FP8.

PREFILL overlaps previous tile's epilogue drain with the first 6 K-iters of the next
tile's MMA. W1 skips epilogue_mbar check for first 6 iters. Saves ~10µs at K=3072.
**Unsafe at K_ITERS<20** (parity wrap → deadlock). fc2_w3 auto-guards via
`#if K_DIM/128 < 20`; **fc2_w3x does NOT auto-guard** — caller must pass
`-DNO_PREFILL` explicitly for short K. FC1 (K_ITERS=6) always uses NO_PREFILL.

NS5 required for N>1536. NS7 doesn't fit in 228KB.

## Tile dispatch — what wins now

Static swizzles beat work-stealing under PACKED_TILES parity. The pre-2026-04-17
"work-stealing wins via 1.00× DRAM amplification" thesis is dead — static reads
20–59% MORE bytes and runs faster. The actual metric is **`long_scoreboard` stalls**
(synchronous-A-wavefront), not DRAM amp.

| FC2 fused | ms | long_sb | barrier | DRAM rd | amp |
|---|---|---|---|---|---|
| default | 1.071 | 2.12M | 272K | 6.79 GB | 1.59× |
| zigzag TD=11 | 1.073 | 2.12M | 271K | 6.04 GB | 1.41× |
| dgswizzle TD=8 | 1.065 | 2.02M | 267K | 5.44 GB | 1.27× |
| sched | 1.101 | 2.66M | 45K | 4.28 GB | 1.00× |
| lean | 1.107 | 2.66M | 44K | 4.28 GB | 1.00× |

LEAN trades 540K more long_sb stalls for 230K fewer barrier stalls — slower net.
Static TMA streaming (offset K-phases across 74 clusters) keeps the load pipeline
full; work-stealing's tight wavefront makes every L2 miss land on the MMA critical
path.

**Recommended:** zigzag (TD=11) for FC2 — same stall profile as default, +4.2 pts
L2 hit rate, 750MB less DRAM. dgswizzle (TD=8) lowest fused at 1.065 but bumps
register count. LEAN remains in tree for large-K (re-verification under parity open).

### fc2_w3x dispatch — basin floor, not a peak (n=29420, 2026-04-29)

The `fc2_w3` table above is fused-with-residual. On `fc2_w3x` (bias-only,
production for that path), the m-axis dispatch lever is **a wide tied
basin, not a sharp peak**. A 2-stage coord-descent sweep over 20 byte-cheap
neighborhood probes (TD=80..99) + the 3 prior round-7 probes converged on
this conclusion.

Two paired-pass sweeps via `tools/sweep_fc2_w3x_swizzle.sh` +
`tools/anova_1way.py --metric cyc --paired rep --trim 0.33`. AUC bands
< 0.55 TIE / < 0.65 WEAK / < 0.75 MODERATE / < 0.85 STRONG / ≥ 0.85 DECISIVE.

**Stage 2 sweep (11 cells, n=29420):**

| variant | mean_rank | Δ̄ resid | σ resid | Δ vs dgsw | AUC vs dgsw |
|---|---|---|---|---|---|
| gflip_blkx7 | **5.25** | −205 | 6761 ⚠ | −738 | 0.315 MODERATE |
| gflip_blk_qrt3 | 5.31 | −234 | 1415 | −767 | 0.318 MODERATE |
| gflip_blkx6 | 5.32 | −237 | 1351 | −770 | 0.315 MODERATE |
| gflip_blk_qrt2 | 5.43 | −211 | 1362 | −744 | 0.321 MODERATE |
| gflip_blk_qrt0 | 5.59 | −100 | 6928 ⚠ | −634 | 0.341 MODERATE |
| gflip_blkswap | 5.63 | −119 | 1443 | −652 | 0.349 MODERATE |
| gflip_blkx5 | 5.69 | −122 | 1352 | −655 | 0.341 MODERATE |
| gflip_bitrev_xor1_alt1 | 6.06 | −16 | 1379 | −550 | 0.365 WEAK |
| gflip_lmrev | 6.54 | +172 | 1506 | −362 | 0.412 WEAK |
| dgsw | 7.47 | +533 | 1576 | 0 | — |
| dg_snlmrev | 7.73 | +539 | 1420 | +5 | 0.501 TIE |

**Top 7 cells form a tied basin** (mean_rank 5.25–5.69, residual Δ̄ within
~135 cyc). η²=0.0075 NEGLIGIBLE — the swizzle factor barely explains
within-basin variance at this resolution. blkx6 and blk_qrt3 are the
cleanest leaders (low σ, MODERATE faster than dgsw); blkx7/blk_qrt0 have
better mean_rank but σ_residual=~7000 indicates outlier-tail in their
point estimates.

**lmrev demoted: prior "blkswap+lmrev TIE for first" was n=43910 sweep
artifact.** With the wider candidate set lmrev lands ~5th-9th depending
on resolution. The bloom filter pre-flagged it MARGINAL (not WORTHY) — so
the demotion is consistent with the model. lmrev's σ=768 in the earlier
n=2048 stage-1 was a fluke that produced a misleading DECISIVE call;
Stage 2 corrected it.

**Why the basin exists.** Once gflip's XOR=1 group pairing is in place
(pair-axis: `cluster_tm_corr ↓` from 0.94 → 0.65), *any* m-axis
perturbation that decorrelates paired CTAs' tm-traversal saturates the
gain. blkswap (`lm^4` on alt groups), lmrev (uniform bit-rev), blkx5/6/7
(varying alt-mask), blk_qrt0/2/3 (qtr-density mask) all produce the same
~600-770 cyc improvement. The sub-tier within the basin (~135 cyc spread)
is below paired-pass resolution at n=29420.

**Three sub-tiers within the gflip family (Stage 1 n=2048, 28 variants):**

| sub-tier | members | Δ vs dgsw | mean_rank |
|---|---|---|---|
| **floor** (XK=1 + m-axis perturbation) | blkx5/6/7, blk_qrt0/2/3, blkswap, lmrev, bitrev_xor1_alt1 | −600 to −770 | 5–17 |
| **mid** (weaker m-axis perturbation) | blklmrev, blkmul3, blkx1/2, mul5 | −400 to −500 | 19–21 |
| **shallow** (gflip alone or wrong pairing axis) | bare gflip, xk2/3/5/7_blkswap, snrot | −100 to −300 | 22–27 |

Three catastrophic gflip failures stay relevant: `gflip_cidperm` (TD=55,
+1718 DECISIVE — `*15 mod 74` cluster permutation breaks SM→L2
contiguity, cluster_tm_corr 0.16 vs gflip 0.65), `gflip_xk2/3/5/7_blkswap`
(non-XK=1 pairing pairs non-adjacent groups, slower than blkswap), and
bare `gflip` (does pair-axis only, ~80% of gain comes from the m-axis
perturbation).

**Default dispatch:** `gflip_blkswap` (TD=54) stays — middle of the basin,
zero churn. blkx6/blk_qrt3 are ~115 cyc faster but at TIE-band resolution
(~0.06 µs wall, below thermal-drift noise on B200). The basin shape is
the load-bearing finding, not the choice within it. See
`memory/project_w3x_n29420_basin.md` (write-up pending).

### Bloom filter validation (n=24 wall test across 2 sweeps, 2026-04-29)

`tools/bloom_filter.py` is **conservative-let-through, not predictive** —
its purpose is zero false negatives on wall winners, and that bar is
satisfied. Cumulative scorecard across the n=43910 + n=29420 sweeps:

| bloom verdict | count | wall outcome | match? |
|---|---|---|---|
| WORTHY (s ≥ +0.10) | 4 | 4 in basin floor | ✅ |
| MARGINAL (−0.10 < s < +0.10) | 16 | 14 basin/MODERATE faster, 2 WEAK | ✅ no false reject |
| STUPID / OVERSHOOT-RISK | 5 | all in TIE/WEAK or DECISIVE LOSER (cidperm) | ✅ no false promote |

**False negatives: 0/7 floor-tier winners.** The model can't discriminate
within the basin (16 MARGINAL cells include known basin-floor members
like blkx6 and blk_qrt3 — bloom let them through, wall confirmed). Use
this pipeline before paying CUDA build + B200 sweep cost: WORTHY = build,
MARGINAL = build (treat as unknown), OVERSHOOT-RISK = build but expect
possible regression, STUPID = skip unless you want negative-control data.

**Calibration data point — Stage 1 lmrev DECISIVE collapsed at Stage 2.**
n=2048 paired-pass mean_rank had lmrev at 6.18 win% 36.2% (DECISIVE Δ
−1605 vs dgsw); n=29420 demoted it to mean_rank 6.54, win% 6.6% (WEAK
Δ −362). The bloom filter pre-flagged it MARGINAL (not WORTHY) — so
the model was correctly skeptical even when small-cohort wall data
appeared decisive. Don't trust mean_rank-based ordering at n<5000 for
sub-σ_residual (~600 cyc) effects.

**Caveat — adj_tn_diff empirical channel may be stale.** snrot2 was
originally labeled "empirical 2nd at n=32768" — at the new n=10978 with 32
cells, snrot2 is 12th at +1750 cyc DECISIVE behind blkswap. The empirical
channel correctly let gflip_snrot through (it landed mid-tier, not
catastrophic), but the wording "snrot2-class, empirical 2nd" should be
updated when next reviewed.

### Cleanest "DRAM amp ≠ bottleneck" proof (cutlass-static, 2026-04-23)

Same tile shape (256x256x128), same cluster (2x1), same 2SM schedule, same
PACKED_TILES — only scheduler/epilogue differ:

| variant | wall µs | tensor% | long_sb | L2 hit% | DRAM rd | amp |
|---|---|---|---|---|---|---|
| cutlass-static (fused) | 1244 | 81.92 | 10.22 | 59.53 | 4.280 GB | **1.000×** |
| fc2_w3x (bias-only)    | 1059 | **97.94** | **6.70** | **67.65** | 2.978 GB | 1.043× |

cutlass-static hits the optimal 1.000× amp floor and runs **185 µs slower**
than fc2_w3x at 1.043× amp. CUTLASS uses 21% more instructions (169.9M vs
140.2M) → 16-pt tensor-pipe gap. Tensor-pipe utilization is the lever, not
DRAM traffic. fc2_w3x reads 1.3 GB MORE per launch and is faster.

### Swizzle metric pipeline (predicting wall from m/n structure)

Two-stage Python pipeline that simulates each TILE_DISPATCH variant in pure
host code, extracts per-cluster m/n visit-sequence features, and asks
whether those features explain the measured wall ranking.

```bash
python3 tools/analyze_swizzle.py --csv /tmp/swizzle_metrics.csv  # simulate
python3 tools/cluster_swizzle.py /tmp/swizzle_metrics.csv        # cluster + regress
```

**Stage 1 — `analyze_swizzle.py`.** Replays `lin_tile = cluster_id + tt*NC`
(NC=74, TILES_M=3626, TILES_N=3, TICKS=147) for every variant
(dgsw G∈{2,4,8,16,32}, dgsnake, checkered, zigzag, rowmajor, ncycle/ncyrot/
nflat/nsnake/nlock, gflip, tn2br, INGH/CHET/PMIX, plus propose_*). Per
variant emits 8 structural metrics covering A-strip reuse (the only
DRAM-bound axis since B is L2-resident at 2.3 MB):
intra-tn / intra-tm run length, wavefront unique-tm count + multiplicity,
cross-cluster L2-window A reuse, fresh-A influx per tick, peak concurrent
unique tm, group-locality score.

**Stage 2 — `cluster_swizzle.py`.** Joins to a baked-in `WALL_NS200` table
(n=200 paired-pass cycles, 13 wall-labeled variants), standardizes,
runs PCA + KMeans(k=4) + Ward (k=2..5). Closes with a "front-tier
separation" diagnosis: feature-space distance between {dgsnake, gflip,
tn2br} centroid and dgsw_G8 baseline, top features by |Δ|, verdict band:

| max |Δ| (std-units) | Verdict |
|---|---|
| < 0.3 | **ANALYZER BLIND** — lever lives outside metric set (likely SM→L2-partition cache affinity, ncu-only) |
| < 1.0 | WEAK SIGNAL — partly captured |
| ≥ 1.0 | CAPTURES LEVER — top feature names it |

**Current verdict (2026-04-28): PARTIALLY CAPTURES.** The earlier n=200
"BLIND" call was on dgsnake/gflip/lmrev, which sit within ~150 cyc of each
other and ARE metric-indistinguishable. But the bigger lever — `gflip_blkswap`
(TD=54) and `gflip_lmrev` (TD=53), now front-tier — IS captured by the
sign-stable τ axes (`adj_tm_diff`, `tm_extent_mean`). Both predicted WORTHY
by `bloom_filter.py` (s=+0.31 and +1.46) before the wall test that confirmed
them as winners. Use this pipeline before paying CUDA build + B200 sweep
cost: WORTHY = build, OVERSHOOT-RISK = build but expect possible big
regression (cidperm caught here), STUPID = don't.

Why clustering, not regression: with 13 labeled points × 8 features the
βs are unstable and an OLS/Lasso prediction is false-precision noise.
Centroid-distance assignment generalises gracefully to unseen variants
(places them in the nearest known tier) where regression would
extrapolate confidently and wrongly. The retired `tile_regress.py` /
`td22_sweep.py` pair predicted ms (not cyc) on older fc2_w3
fused-residual data — pre-paired-analysis, pre-thermal-defense; deleted
2026-04-27.

### Adaptive tuning knobs

| Knob | Rule | Why |
|---|---|---|
| N_STAGES | NS6 for N≤1536, NS5 for N>1536 | SMEM per stage grows with N |
| PREFILL | On for K_ITERS≥20, off otherwise | Short K-loop deadlocks (parity wrap). Also: NO_PREFILL caps eff at ~0.77 vs MMA-staging ceiling, PREFILL pushes to ~0.91 — the eff jump confirmed in n=20 dim sweep (2026-04-30, see below). |
| Dispatch | FC2 fused: zigzag or dgswizzle. FC2 bias-only: any basin-floor variant — `gflip_blkswap` (TD=54) is default, `blkx6`/`blk_qrt3` ~115 cyc faster at TIE-band. FC1: zigzag + K_STAGGER=1. | PACKED_TILES + odd ks on FC1; FC2 wash on ks. |

## Compute floor

`tcgen05.mma.cta_group::2` produces cluster-wide work per instruction (no extra ×2
CTA factor). Per-cluster cycles = `147 tiles × 24 K_iters × cyc/iter`:

| source | cyc/iter | wall (B200) | notes |
|---|---|---|---|
| hardware MMA retirement | 460 | 0.896 ms | absolute ceiling, no staging — **unreachable** |
| bench NS=4 + W0-TMA overlap | 520.8 | 1.014 ms | published microbench |
| bench NS=4, no TMA overlap | 525.6 | 1.023 ms | published microbench |
| **fc2-w3x-strip** (NS=6 + PREFILL) | **493** | **0.98502 ms** | **measured staging floor (1814685 cyc, 2026-04-30)** |
| **fc2-w3x** (full kernel) | **502** | **1.00092 ms** | measured production (1846145 cyc) |
| cuBLASLt rank-1 (L2) | ~520 | 1.046 ms | reference ceiling |

**Strip vs full = 15.9 µs / 31460 cyc / 214 cyc/tile** (1.7% of 12482 cyc MMA
budget per tile) — exposed-on-critical-path epilogue work, what `bar.sync` /
`mbar_wait` couples between W0-W3 epi-end and W5's next-tile MMA. **~98% of the
epilogue body IS hidden in MMA shadow**; the 2% that isn't is our real headroom.
This contradicts the earlier "epi 100% in MMA shadow" framing from PROFILE_W5.

**Gap decomposition** (vs structural floor, not the unreachable 460-cyc ceiling):
- 89 µs (pure-MMA → strip) = NS=6 staging bubble. **Unreachable** without
  removing staging.
- 16 µs (strip → fc2-w3x) = exposed epi. **Real headroom**, addressable.
- 61 µs (strip → cuBLASLt rank-1) = rank-1 has 4× more exposed-epi than us.
  We've already cleaned up most of the W0-W3/W5 coupling.

The earlier "~100 µs headroom" framing referenced the unreachable pure-MMA
ceiling. **Real headroom on fc2-w3x is ~16 µs** vs the structural NS=6+PREFILL
floor — but per the SASS-attribution + per-tt PROFILE_W5 analysis below,
**realistic recoverable is ~1-3 µs, not the earlier ~5-10 µs estimate**.

### Strip-vs-full localization (ncu --set full + per-tt PROFILE_W5, 2026-05-01)

**SASS-attributed stall hotspots** — see `docs/fc2_w3x_ncu_sass.txt`
(per-line stall sampling) + `docs/fc2_w3x_ncu_details.txt` (Speed-of-Light /
Memory Workload / Scheduler / Warp State sections):

| stall samples | instruction | what it is |
|---|---|---|
| **5,024,622** | `SYNCS.PHASECHK.TRANS64.TRYWAIT` (mbar phase check) | **mbar spin-wait body — 7× the next category** |
| ~696K each | `F2FP.BF16.F32.PACK_AB`, `HADD2.BF16_V2`, `SHFL.IDX`, `IMAD`, `VIADD` | epi compute body (8 subpasses × 147 tiles) |
| ~174K each | `UTMACMDFLUSH`, `UTMASTG.4D`, `R2UR.OR`, `BSYNC.RECONVERGENT` | TMA store + scaffolding |

Confirms the +0.85 pt barrier delta from focused metrics is mbar-dominated
steady-state work, not mis-attribution. TC pipeline 98.49% → strip 0.985 ms
IS the MMA-staging structural floor.

**Final-tile drain is FALSE.** Per-tt PROFILE_W5 (`make -B fc2-w3x
DFLAGS='-DPROFILE_W5 -DPROFILE_TILE'`, 2026-05-01): **tt=146 = 11872 cyc =
the *fastest* tile** (mean 12098). Slowest is tt=0 (13723 cyc, cold-start)
and a periodic +200-840 cyc bump at tt=4/20/36/52/68/84/100/116/132
(~16-tile spacing — looks like L2-replacement-cycle hits). The 12 µs gap
between ncu `cyc_avg` (Δ +4664) and wall `cyc_max` (Δ +31460) is
**cross-CTA workload variance** (one CTA finishes ~26800 cyc later than
the average), not a final-tile artifact. LAST_TILE_FAST_PATH would have
saved <1 µs — not implemented.

**16 µs gap decomposition (revised):**
- **~4 µs steady-state mbar / cluster bar.sync** (sass-attributed at 5.02M
  samples; symptomatic of the 6-warp design + `cta_group::2` cluster sync —
  not addressable without rewriting the warp-specialization or cluster shape)
- **~12 µs cross-CTA tail variance** (structural; basin-floor dispatch
  already minimizes it, within-basin spread is at TIE-band)

**ncu warnings to IGNORE in `docs/fc2_w3x_ncu_details.txt`:**
- `"13398-way bank conflict, 40.22% Est. Speedup"` — **STSM mis-attribution**.
  ncu doesn't model `stmatrix`'s dedicated bank-routed datapath; reports
  the lane→bank fan-out as generic shared-store conflicts. Not real.
- `"35.01% Est. Speedup, 21.3 active threads/warp"` — warp-specialization by
  design (W4/W5 single-warp issue, lane-split epi). Not a lever.

FC1 strip is TMA-load-dominated, not compute-bound.

### N×K dim-sweep findings (2026-04-30, n=20 cells)

`tools/dim_sweep_w3x.py` swept fc2_w3x across N∈{256,512,768,1024,1536} ×
K∈{768,1536,3072,6144}. Headline pattern (eff = pure-MMA-floor / measured ms;
note this overstates headroom — vs realistic NS=4 ceiling 525 cyc/iter most
cells are ≥1.0):

| | K=1536 NO_PREFILL | K=3072 PREFILL | K=6144 PREFILL |
|---|---|---|---|
| **eff at N=768** | 0.77 | **0.89** (production) | 0.91 |
| **eff at N=1536** | 0.77 | 0.91 | 0.91 |
| **eff at N=256** | 0.54 | 0.62 | 0.67 |

- **PREFILL crossover is the big lever**: K_iters=12 → K_iters=24 jumps eff
  +0.12 (NO_PREFILL caps you at ~0.77 of pure-MMA regardless of N). Confirms
  K_ITERS≥20 threshold is real, not just safety guard.
- **K=3072 sweet spot**: K=6144 gains only +0.02 eff → diminishing returns.
- **N≥512 K≥3072 plateaus at ~0.91** — that's the asymptotic staged ceiling
  on this geometry, matches strip-vs-full 16 µs gap analysis.
- **N=256 starved**: only 49 tiles/cluster — pipeline never fully saturates
  (eff caps at 0.67 even at K=6144).

**Known issue: K=768 (K_iters=6) + N≥512 fails.** N=256 K=768 passes; N=512+
fails (return code 1, undiagnosed). Suspect: K_iters==N_STAGES==6 + multi-column
dispatch (TILES_N≥2). Either kernel-side bug at full-pipeline-fill K-loops with
multi-tile transitions, or host-side validation/ABI issue. Diagnose via
`grep -A20 "FAIL" data/dim_sweep_w3x_<ts>/sweep.log` before relying on short-K
results.

## cuBLASLt rank-1

`tools/probe_cublaslt.sh` (probe 1) enumerates every heuristic, times each, reports
rank-1. This is the true ceiling. Earlier comparisons against `cublas-bench-fc2`
measured the **default heuristic pick**, not rank-1.

### FC2 K-sweep

| K    | cuBLASLt  | best ours | gap |
|------|-----------|-----------|-----|
| 1024 | ERR       | 0.859 (lean)        | n/a |
| 2048 | ERR       | 0.922 (zigzag)      | n/a |
| 3072 | **1.046** | 1.064 (dgsw fused) / **1.008 (w3x bias-only, blkswap TD=54)** | +18 µs / **−38 µs** |
| 4096 | **1.360** | 1.476 (dgsw)        | +116 µs |
| 6144 | **1.997** | 2.007 (dgsw)        | +10 µs |
| 8192 | **2.682** | 2.731 (lean)        | +49 µs |

FC1 K=768: cuBLASLt **1.894 ms** vs ours 1.998 (zigzag+ks=1) → +104 µs (+5.5%).

### Rank-1 decode (B200, 2026-04-20, FC2 K=3072)

Kernel name: `nvjet_sm100_qqtst_<M>x<N>_128x<NS>_<CM>x<CN>_[2cta_]<h|v>_<...>_T<A><B>`.
`2cta` = `cta_group::2`. `h/v` = TMA multicast axis. `bz_bias` = bias-only epilogue.

| listed | tile | NS | cluster | cta_grp | ms |
|---|---|---|---|---|---|
| L1 | 176x128 | 8 | 1x2 | 2 | 1.0454 |
| **L2†** | **128x256** | **6** | **2x1** | **2** | **1.0457** |
| L3 | 128x192 | 7 | 2x1 | 2 | 1.094 |
| L4 | 256x256 | 4 | 2x1 | 2 | 1.192 |

† **"Rank-1" = L2** (our exact geometry). L1 wins by 0.3 µs noise and uses NS=8 which
isn't SMEM-feasible at our 256x256 tile. Dumped SASS at `rank1.sass`. Top 5 listings
all use `cta_group::2`. Not split-K (`splitk=1`), not CUTLASS-style swizzle (`swizzle=0`).

Tile enum (from `cublasLt.h`): `23=128x256, 24=256x128, 32=128x192, 197=168x128,
201=176x128, 495=256x96, 535=320x192`. `stages=36 = 128xAUTO`, NS resolved per kernel
variant at compile time.

K=1024/2048 still report ERR — one heuristic IMAs on the device.

### Status (2026-04-30): strip-measured floor 0.985 ms, full kernel 1.001 ms (16 µs exposed epi)

`fc2_w3x` bias-only at 1.00092 ms with **gflip_blkswap (TD=54)** dispatch
(prior dgsw_G8 default ~1.009 ms, +0.33 µs slower); W5 is MMA-ceiling-bound
(~12482 cyc/tile ≈ 24 × 520 cyc/iter, per `PROFILE_W5`). Tensor pipe 95.84%
active. The 9-grievance SASS delta list vs rank-1 is exhausted (STSM
mandatory, R2UR/ELECT confirmed orthogonal-to-W5, ptxas-owned descriptor
operand class — see dead-end log).

**Strip-measured floor (2026-04-30):** `fc2-w3x-strip` at **0.98502 ms**
(1814685 cyc) = NS=6 + PREFILL + W4-TMA structural floor on this geometry.
Full at 1.00092 ms → **15.9 µs / 31460 cyc / 214 cyc/tile exposed epi** =
real remaining headroom. ~98% of epi body is MMA-shadowed; remaining 2% is
cluster-barrier coupling between W0-W3 epi-end and W5's next-tile MMA.
The earlier "epi 100% in MMA shadow" framing from PROFILE_W5 was directionally
right but quantitatively wrong. Realistically recoverable: ~5-10 µs (some
fraction is inevitable: final-tile drain, proxy fence, cluster bar.sync).
By past-win standards (bias-preload 1.7 µs, STSM 0.4 µs) this is the largest
single remaining target — but probably needs new lever class, not more
SASS-level epi tuning.

**bias-preload (default, 2026-04-26)** — pre-loads full bias [768 bf16] into
per-lane registers at kernel start; subpass-level shfl×4 replaces the per-rh
LDS×4 in the epilogue. B200 n=128 pass-major interleaved (commit `ef62fed`):
vbase 1010.70 µs → vpreload 1008.97 µs, **Δ = −1.73 µs at z=−23.23 STRONG**
(SE 0.075, df=243.5, p ≪ 0.001). 6× the EPI_2WARP marginal effect with zero
race risk (same SMEM data, just read once vs N times). Regs 56→65 / stack
16 B / 0 spills. Promoted to default + `!BIAS_PRELOAD` path stripped from the
tree. See `memory/project_w3x_bias_preload_win.md`.

**STSM (mandatory)** — `bcce329` layout fix matches rank-1 SASS
opcode mix (STS.128 4→0, STSM.16.M88.4 0→4). B200 n=10 (2026-04-25): PASS,
mean 1.0039 ± 0.0004 ms vs legacy STS 1.0043 ± 0.0002 ms. The legacy
STS.128 path was retired with bias-preload (the bias-broadcast pattern only
fits the STSM lane mapping); `USE_STMATRIX` is no longer a flag.

**Output ABI (2026-04-26):** `d_C` is now stored in a 4D packed-tile layout
`[TILES_M, TILES_N, TM*2, TN]` (each (tile_m, tile_n) is one contiguous
TM*2 × TN bf16 block), matching A/B's PACKED_TILES convention so the next
kernel in the SigLIP pipeline reads it pre-packed. Host-side index helper
`pack_idx_C(m, n)` mirrors the 4D TMA descriptor; verify path uses it.
Requires `M_TOTAL % (TM*2) == 0` and `N_DIM % TN == 0` (static_assert).
Regs 53→56 default / 78→80 PROFILE_CYCLES (4D TMA op needs 2 extra
register operands); 0 spills, 1 barrier unchanged. Wall impact in this
kernel expected ±3 µs (TMA store is in MMA shadow); win is downstream.

**Next:** port `fc2_w3x` from bias-only to fused-residual (production target).

## Dead ends — do NOT retry

See `memory/MEMORY.md` for full chronological dead-end log. Highlights:

- **Source-level epilogue tuning:** ptxas owns STS layout. CUTLASS_LOOP, FP32_EPILOGUE,
  cvta.shared, NUM_EPI_STAGES, stmatrix variants — all generate identical SASS.
- **Cross-warp STS clustering (intra-warp attempts):** SELF_LOAD, SELF_STAGGER (nanosleep),
  SASS intra-warp reorder. Zero effect — wrong axis. *Inter-cluster* arrival into
  STS/TMA-store IS ordering-controlled (see g-s).
- **Hand-written PTX `fc2_w3x.ptx`:** byte-identical SASS to nvcc emission. PTX ISA
  has no uniform-register type; ptxas owns R-vs-UR placement. PTX escape hatch does
  not exist. Frozen at `fc2-w3x-ptx`.
- **K_UNROLL partial-unroll** (Apr 25, B200 n=3 sweep): u1/u2/u3/u4/u8 all regress
  87–197 µs (UIADD3=7, regs=66 — UR datapath collapses on non-N_STAGES-multiples).
  u6/u12/u24 tie default within 0.4 µs (UR on, regs=64). Explicit `K_UNROLL=24`
  shrinks SASS 39% (10029→6077 lines) at parity wall — free cleanup. Default
  variance anomalously wide (max−min=7.9 µs on n=3) — n=10 re-test open.
  See `memory/project_k_unroll_sweep.md`.
- **Cluster-axis swap** (Apr 25, 2 attempts): B200 runtime hard-rejects
  `__cluster_dims__(1,2,1)` and `(1,1,2)` with "cluster misconfiguration",
  even with `cudaLaunchKernelEx` + explicit `cudaLaunchAttributeClusterDimension`
  + `mapa.shared::cluster` peer-addressing. cuBLASLt rank-1's `2x1`/`1x2`
  notation is logical-shape labeling inside an X-axis grid (verified via
  `rank1.sass`: grid (21756,1,1), cluster (2,1,1)). 2-CTA clusters are
  X-axis-only on B200. See `memory/dead_cluster_axis_swap.md`.
- **KERN_3WARP merge** (Apr 25): merging W4 TMA-issuer into W5 MMA-issuer
  (single combined warp on CTA0, software-pipelined offset N_STAGES−1)
  regresses 1.006→1.172 ms (+166 µs, +16.5%) on B200, valid=1. K_UNROLL=24
  byte-identical SASS to no-pragma (ptxas already fully unrolls); K_UNROLL=6
  partial-unroll regresses further to 1.2064 ms — unrolling ruled out as
  the lever. Per-tile ~12482→~14700 cyc (+90 cyc/iter on top of 525 cyc/iter
  MMA-throughput floor). W4's measured `empty_wait` slack is mbar_wait
  sleep, not free issue cycles — TMA-issue + wait_tma_empty serialize onto
  W5's instruction stream. Structural warp-count floor is 4 (2 epi + W4 +
  W5). Commit `e292107` kept as referenced dead-end. See
  `memory/dead_kern_3warp.md`.
- **EPI_2WARP+DROP_LEAD_BARSYNC marginal opt-in win** (Apr 26, n=128 4-cell
  interleaved with baseline): v10011 (NO_BULK + DROP_LEAD + EPI_2WARP) beats
  baseline by 0.27 µs at z=-3.49 (STRONG α≈0.001). v10110 (same combo minus
  EPI_2WARP) regresses +2.54 µs at z=+34.9 — EPI_2WARP is the only
  load-bearing lever; under 4-warp epi the lead bar.sync elision replaces
  a coalescing barrier with 4 disjoint proxy fences. NO_BULK_MEMCLBR is
  byte-identical SASS (zero contribution). **Stays opt-in:** DROP_LEAD
  ships a cross-warp STS-before-TMA race (verify-passing is timing-luck),
  EPI_2WARP's 2-warp restructure is fc2_w3x-only (FC1's 7-warp GELU epi
  can't take it), 0.27 µs is ~0.026% wall. See
  `memory/project_w3x_epi2warp_marginal.md`.
- **LDTM_X32 (widest TMEM-load + STS.128) ties STSM at MMA floor** (Apr 26, B200
  n=512 pass-major): Δ=−0.021 µs, p=0.6523. `tcgen05.ld.32x32b.x32` (lane t = row
  t cols 0..31) + 4× `st.shared.v4.b32` interchangeable with rank-1's
  `LDTM.16dp256bit.x4 ×2` + `STSM.16.MT88.4 ×4`; LDTM cycle savings live in W0-W3
  MMA shadow. STSM stays default by symmetry with rank-1. LDTM_X64 (.32x32b.x64)
  forces `NUM_EPI_STAGES=2 → 1` (SMEM budget) and regresses +14 µs at n=16
  STRONG — confirms NS_EPI=2 is worth ~14 µs (correcting the prior "NS_EPI sweep"
  ±3 µs footnote, which tested NS_EPI=3,4 vs 2 not NS_EPI=1 vs 2). See
  `memory/dead_ldtm_x32_tie.md`.
- **fc2_w3x post-WIN levers (all ±3 µs or regression):** subpass 8→4, cross-tile TMA
  carry, SWIZZLE_64B, DROP_TRAIL_BARSYNC, WAIT_GROUP_READ,
  XPF_A/B prefetch (Bonferroni-confirmed regression in
  the 2026-04-26 128-cell combo sweep: XPF_A +3.05 µs at z=+6.30, XPF_B
  +1.27 µs at z=+2.62; **macros removed from tree**), CHET/PMIX/INGH hybrid dispatches,
  gflip_cidperm (TD=55) +1568 cyc DECISIVE at n=43910 — `c*15 mod 74` cluster
  permutation breaks SM→L2 contiguity (cluster_tm_corr 0.16 vs gflip 0.65,
  bloom-filter overshoot threshold caught this). The earlier n=5489 "3-way
  front TIE" (dgsnake/lmrev/gflip) is MODERATE behind the n=29420 7-cell
  basin (blkswap, blkx5/6/7, blk_qrt0/2/3); the n=43910 "blkswap+lmrev
  TIE" is also superseded — lmrev demoted to mid-pack with the wider
  candidate set. See basin section above.
  STAGGER=2 split-mbar
  (uniformly +3 µs across all 11 dispatches, zero stagger×dispatch interaction —
  +36 cyc on each of W4/W5 from extra arrive + extra mbar_wait; **macro removed
  from tree 2026-04-26**, postmortem in `memory/dead_fc2_w3x_stagger.md`), DG
  sweep, native BF16 epilogue (kept ±0 wall, cleaner).
- **Older dead variants:** TD=1 atomic, TD=5 CLC, TD=6/7 inline atomic, COL_LOCK,
  4-CTA TMA multicast (silent deadlock), mbar→SMEM polling, L2 cache hints,
  dgphase/dgnrot (TD=23/24), fc2_ldg (LDG/STG), fc2_hybrid (CUTLASS phases 2/3b/4),
  N-batch / phase-offset / Group-3 (pre-PACKED_TILES — re-test before citing).
- **FC1 FORCE_PREFILL:** deadlocks at K_ITERS=6. NO_PREFILL guard is necessary.

## Build and run

```bash
# FC2 BIAS_ONLY (BEST — beats cuBLASLt rank-1)
make fc2-w3x && ./fc2-w3x                        # ~1.001 ms (bias-preload + STSM mandatory)
make fc2-w3x-strip && ./fc2-w3x-strip            # NS=6+PREFILL staging floor (~0.985 ms)
make fc2-w3x-ptx                                 # hand-written PTX, byte-identical SASS

# fc2_w3x sweeps + diagnostics
make fc2-w3x-tile-sweep                          # TILE_DISPATCH macro variants
./tools/sweep_fc2_w3x_tiles.sh                   # full tile sweep
./tools/sweep_fc2_w3x_dg.sh                      # DG_GROUP_SIZE × INNER_T × STAGGER
./tools/sweep_fc2_w3x_prof.sh                    # per-warp clock64 phases
python3 tools/aggregate_prof.py data/<dir>
make fc2-w3x DFLAGS='-DPROFILE_CYCLES'           # per-warp phases
make fc2-w3x DFLAGS='-DPROFILE_KI|-DPROFILE_TILE|-DPROFILE_W5'

# FC2 fused-with-residual (uses fc2_w3.cu)
make fc2-w3-lean && ./fc2-w3-lean                # fused 1.074
make fc2-w3 && ./fc2-w3                          # striding 1.113
make fc2-w3-sched && ./fc2-w3-sched              # work-stealing
make fc2-w3-gemm && ./fc2-w3-gemm                # GEMM-only
make fc2-w3-strip && ./fc2-w3-strip              # MMA-only

# FC1
make fc1-w3-lean && ./fc1-w3-lean                # fused 2.037
make fc1-w3 && ./fc1-w3
make fc1-w3-sched && ./fc1-w3-sched

# Custom dims (MUST use -B: Make doesn't track DFLAGS)
make -B fc2-w3 DFLAGS='-DM_TOTAL=464128 -DN_DIM=1024 -DK_DIM=2048 -DN_STAGES=6'
# Decomp via DFLAGS: -DSTRIP_EPILOGUE / -DGEMM_ONLY

# References
make fc2-cutlass && ./fc2-cutlass                # 1.226
./tools/probe_cublaslt.sh                        # cuBLASLt rank-1 (TRUE ceiling)

# Profiling
bash tools/bench.sh --comprehensive              # rank-1-baselined
bash tools/ncu_bench.sh && python3 tools/ncu_anova.py
bash tools/ncu_fc2_w3x.sh --max --reps 3
bash tools/ncu_fc2_pipes.sh                      # dodges --set full deadlock
./tools/dim_sweep.sh --fast                      # fc2_w3 80 configs (M×N×K)
./tools/dim_sweep_w3x.py                         # fc2_w3x N×K grid (default 20 cells, N≤1536)
```

## Key files

```
fc2_w3x.cu                      # FC2 bias-only (ACTIVE, 1.007 ms — beats rank-1)
fc2_w3x.ptx                     # Hand-written PTX, byte-identical SASS (frozen)
fc2_w3.cu                       # FC2 fused-residual (ACTIVE for fused path)
fc1_w3.cu                       # FC1 (ACTIVE)
fc2_ws.cu                       # FC2 warp-specialized w/ rank-1 warp retirement
tile_dispatch.cuh               # Shared TD=8..16, 21..32 (incl. CHET/PMIX/INGH)
fc2_cutlass.cu                  # CUTLASS reference
fc2_hybrid.cu, fc2_ldg.cu, fc2.cu  # DEAD
kernel_common.cuh, kernel_body.cuh # Shared infra
Makefile                        # sm_100a, DFLAGS for dim override
docs/STSM_STATUS.md             # STSM layout playbook
docs/PURE_PTX_REWRITE_STRATEGY.md
docs/BENCHMARKING.md            # cycles/AUC/η²/rank study guide — read before benchmarking
docs/fc2_w3x_ncu_sass.txt       # ncu --set full per-SASS stall sampling (2026-05-01) — mbar spin-wait dominates 7×
docs/fc2_w3x_ncu_details.txt    # ncu --page details (SoL/Memory/Scheduler/Warp State) — 98.5% TC pipe, ignore STSM bank-conflict warnings
rank1.sass                      # Dumped cuBLASLt rank-1 for diffing
tools/bench.sh                  # FC1/FC2 × dispatch × packed × decomp (rank-1 baseline)
tools/probe_cublaslt.sh         # cuBLASLt rank-1 timing
tools/dim_sweep.sh              # fc2_w3 M/N/K grid (bash)
tools/dim_sweep_w3x.py          # fc2_w3x N×K grid (Python; per-cell binaries for cross-machine)
tools/ncu_bench.sh, ncu_fc2_w3x.sh, ncu_fc2_pipes.sh   # ncu profiling
tools/sweep_fc2_w3x_*.sh        # tiles / dg / prof sweeps
tools/aggregate_prof.py         # PROFILE_* aggregator
tools/ncu_anova.py
tools/analyze_swizzle.py        # per-swizzle structural metric simulator (A/B reuse, group locality)
tools/cluster_swizzle.py        # PCA + KMeans + Ward — wall-vs-metric blind/captures verdict (centroid-distance)
tools/anova_1way.py             # paired ANOVA + AUC + Cohen's d + η² + mean-rank/win% (no p-values)
tools/sass_edit.py              # SASS binary editor + CP-SAT scheduler
token_count.py                  # tiktoken-based token budgeting for CLAUDE.md / docs / memory
bench/                          # Microbenchmarks (TMA, MMA, stmatrix, cublaslt_introspect)
data/                           # Benchmark + ncu results
```

## SM100a hardware data (B200-measured)

- STS.128: 27 cyc | LDS.128: 25 cyc @ILP=1, 3.5 cyc @ILP=7
- TMA load: 419 cyc (L2-warm) | TMA store: 197 cyc
- TMEM load (tcgen05.ld.sync): 2 cyc regardless of width/ILP
- MMA K-iter: 665 cyc (pipelined: 525.6 cyc/iter)
- STS scaling: 10→37 cyc at 8 warps (3.65× contention)
- LDS scaling: 4.5→16 cyc (3.56×)
- FFMA: ~free (1.36× at 8 warps)
- F2FP: zero contention (flat 2.0 cyc all warp counts)

## Key constraints

- Target: sm_100a (B200, 148 SMs), `cta_group::2`, 74 clusters
- TMEM: 512 cols, single alloc for double buffering
- SMEM: 228 KB/SM
- All inline PTX in fc2_w3.cu/fc1_w3.cu (no CUTLASS dependency)
- OFF_STAGING must be 1024-byte aligned for SWIZZLE_128B
- `fence.proxy.async.shared::cta` required before TMA store after st.shared
- N_STAGES=6 default (NS5 for N>1536, NS7 doesn't fit)
- PREFILL on for K_ITERS≥20, off otherwise (auto-guarded)
- BIAS_SMEM=1 default (-15 µs free)
- Custom dims require `make -B`
- W0's K-loop is TMA-sensitive: any global op (atomicAdd, etc.) costs +41–77% tma_issue.
  Non-critical-path global ops (W7 scheduler at tile-boundary) are fine.

## Benchmarking

See `docs/BENCHMARKING.md` for the full playbook. TL;DR for any timing-claim
work in this repo:

- **Cycles, not ms.** `clock64()` per-CTA in the kernel, `max_over_CTAs / N_TIMED`.
  Clock-frequency invariant — required on vast.ai (no locked clocks).
- **Pass-major** (randomized block): outer pass p, inner variant v.
  `@@SAMPLE pass=p variant=v cyc=Y` per launch.
- **Trim** first 33–50% of passes (cold L2 + thermal ramp).
- **Paired analysis** by pass; report **AUC**, **Cohen's d**, **η²**, **mean
  rank**, **win%**. **No p-values** — they're meaningless at large n.
- `tools/anova_1way.py --metric cyc --paired rep --trim 0.33` is the canonical
  invocation; `tools/sweep_fc2_w3x_*.sh` uses it by default.
- **n thresholds for paired-pass (σ_residual ~1400 cyc on fc2_w3x):**
  n<5000 unreliable for sub-σ effects (Stage 1 of coord-descend had lmrev
  DECISIVE at n=2048; Stage 2 demoted it to mid-pack at n=29420). For
  ~600 cyc Δ across MODERATE-band cells use **n≥10978**; to crack TIE band
  (~150 cyc) use **n≥43910** — but expect that within-basin separations
  may sit below the resolution floor regardless of n. Default 2-stage flow:
  Stage 1 REPS=2048 across all candidates to filter to top ~7, Stage 2
  REPS=43910 on survivors + anchors.

## Code style

Names say what, comments say why. No single-line `/**/`. No multi-line `//`.
No decorated block comments. Bare `/*` open, undecorated lines, `*/` close.

## Context efficiency

Don't narrate tool calls. Don't echo file contents. Keep explanations proportional.
Parallelize independent tool calls. Use offset/limit for large files.

### Token budgeting

LLM context is the binding constraint on how much of this codebase a single
session can reason over coherently. Every kilobyte spent on stale narrative,
duplicated dead-end logs, or verbose status prose is a kilobyte unavailable
for actual code, SASS, ncu CSVs, or chain-of-thought. Treat CLAUDE.md, docs,
and memory files as a token budget — when bloat creeps in, prefer a brief
pointer to a topic file over inlining the full story.

`./token_count.py <file>` reports tiktoken counts (o200k_base for GPT-4o-class,
cl100k_base for legacy GPT-4) plus three heuristics. GPT tokenizers approximate
Claude's tokenizer to within ~10% — fine for budgeting, not for billing.

```bash
python3 token_count.py CLAUDE.md          # baseline
python3 token_count.py docs/STSM_STATUS.md
find docs/ memory/ -name '*.md' | xargs -I{} python3 token_count.py {} | grep o200k_base
```
