# SigLIP2 Vision Encoder — Hand-tuned Blackwell GEMM Kernels

Hand-tuned SM100a persistent GEMM kernels for FC1/FC2 of `google/siglip2-base-patch16-224`.
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
`fc2_w3` = legacy 7-warp fused, still production for residual path.

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
| N_STAGES | auto: `min(NS_BY_N, max(2, K_ITERS−3))`; NS_BY_N = 6/5/4/3 for N≤1536/2048/4096/larger | SMEM ceiling (228 KB; each NS stage = 32 KB A+B FP8; NS7 doesn't fit). Pipeline-fill margin gap≥3 (gap=2 FAILs at K=1024 NS=6). |
| PREFILL | auto: K_ITERS≥20 → on, else NO_PREFILL | Short K-loop deadlocks (parity wrap). NO_PREFILL caps eff at ~0.77; PREFILL pushes ~0.91. fc2_w3 auto-guards via `#if K_DIM/128 < 20`; fc2_w3x kernel-side macro guard (commit 3d6c1cb). |
| Dispatch | FC2 fused: zigzag (TD=11) or dgswizzle (TD=8). FC2 bias-only: any `gflip_*` basin-floor — `gflip_blkswap` (TD=54) default, `blkx6/blk_qrt3` ~115 cyc faster at TIE-band. FC1: zigzag + K_STAGGER=1. | PACKED_TILES + odd ks helps FC1; FC2 wash on ks. |

PREFILL overlaps prev tile's epilogue drain with first 6 K-iters of next tile's
MMA; W1 skips epilogue_mbar check for first 6 iters (eff lift quantified in
N×K table below). 6-stage pipeline uses 227 KB of 228 KB SMEM.

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

Static swizzles beat work-stealing by ~30 µs. Strip floor ~0.988 ms.
`fused = strip + (g-s) + (f-g)`. g-s = store contention (cluster-wavefront N-column
diversity); f-g = epilogue/next-tile-mainloop overlap (K_ITERS-limited).

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

FC1 dispatch lever bigger than FC2's. Odd K_STAGGER (1/3) helps; ks=2 hurts.
ncycle/nsnake have f-g≈0 (zero overlap) — pathological dispatch.

## Tile dispatch — mechanism

Static swizzles beat work-stealing under PACKED_TILES parity. The pre-2026-04-17
"work-stealing wins via 1.00× DRAM amplification" thesis is dead — static reads
20–59% MORE bytes and runs faster. Actual metric is **`long_scoreboard` stalls**
(synchronous-A-wavefront), not DRAM amp.

| FC2 fused | ms | long_sb | barrier | DRAM rd | amp |
|---|---|---|---|---|---|
| default | 1.071 | 2.12M | 272K | 6.79 GB | 1.59× |
| zigzag TD=11 | 1.073 | 2.12M | 271K | 6.04 GB | 1.41× |
| dgswizzle TD=8 | 1.065 | 2.02M | 267K | 5.44 GB | 1.27× |
| sched | 1.101 | 2.66M | 45K | 4.28 GB | 1.00× |
| lean | 1.107 | 2.66M | 44K | 4.28 GB | 1.00× |

LEAN trades 540K more long_sb for 230K fewer barrier — slower net.

**Cleanest "DRAM amp ≠ bottleneck" proof (cutlass-static, 2026-04-23)** — same
tile/cluster/2SM-schedule/PACKED, only scheduler/epilogue differ:

| variant | wall µs | tensor% | long_sb | L2 hit% | DRAM rd | amp |
|---|---|---|---|---|---|---|
| cutlass-static (fused) | 1244 | 81.92 | 10.22 | 59.53 | 4.280 GB | **1.000×** |
| fc2_w3x (bias-only)    | 1059 | **97.94** | **6.70** | **67.65** | 2.978 GB | 1.043× |

CUTLASS hits 1.000× amp floor and runs **185 µs slower** at 21% more instructions
(169.9M vs 140.2M → 16-pt tensor-pipe gap). **Tensor-pipe utilization is the
lever.** fc2_w3x reads 1.3 GB MORE per launch and is faster.

## fc2_w3x basin (n=29420, 2026-04-29)

m-axis dispatch is **a wide tied basin, not a sharp peak**. Top 7 cells (mean_rank
5.25–5.69, residual Δ̄ within ~135 cyc) form the floor; η²=0.0075 NEGLIGIBLE.

| variant | mean_rank | Δ̄ resid | σ resid | Δ vs dgsw | AUC |
|---|---|---|---|---|---|
| gflip_blkx7 | **5.25** | −205 | 6761 ⚠ | −738 | 0.315 MOD |
| gflip_blk_qrt3 | 5.31 | −234 | 1415 | −767 | 0.318 MOD |
| gflip_blkx6 | 5.32 | −237 | 1351 | −770 | 0.315 MOD |
| gflip_blk_qrt2 | 5.43 | −211 | 1362 | −744 | 0.321 MOD |
| gflip_blk_qrt0 | 5.59 | −100 | 6928 ⚠ | −634 | 0.341 MOD |
| gflip_blkswap | 5.63 | −119 | 1443 | −652 | 0.349 MOD |
| gflip_blkx5 | 5.69 | −122 | 1352 | −655 | 0.341 MOD |
| gflip_bitrev_xor1_alt1 | 6.06 | −16 | 1379 | −550 | 0.365 WEAK |
| gflip_lmrev | 6.54 | +172 | 1506 | −362 | 0.412 WEAK |
| dgsw | 7.47 | +533 | 1576 | 0 | — |
| dg_snlmrev | 7.73 | +539 | 1420 | +5 | 0.501 TIE |

**Three sub-tiers (Stage 1 n=2048, 28 variants):** floor (XK=1 + m-axis perturbation:
blkx5/6/7, blk_qrt0/2/3, blkswap, lmrev, bitrev_xor1_alt1) Δ −600 to −770; mid
(weaker m-perturbation: blklmrev, blkmul3, blkx1/2, mul5) Δ −400 to −500; shallow
(gflip alone or wrong pairing axis: bare gflip, xk2/3/5/7_blkswap, snrot) Δ −100 to −300.

**Why basin exists:** once gflip's XOR=1 group pairing is in place
(`cluster_tm_corr ↓` 0.94→0.65), *any* m-axis perturbation that decorrelates
paired CTAs' tm-traversal saturates the gain. **Default `gflip_blkswap` (TD=54)
stays — middle of basin, zero churn**; blkx6/blk_qrt3 ~115 cyc faster at
TIE-band (~0.06 µs wall, below thermal-drift noise on B200). **Basin shape is
the load-bearing finding, not the choice within it.**

**Catastrophic gflip failures stay relevant:** `gflip_cidperm` (TD=55, +1718
DECISIVE — `*15 mod 74` cluster perm breaks SM→L2 contiguity, cluster_tm_corr
0.16 vs gflip 0.65), `gflip_xk2/3/5/7_blkswap` (non-XK=1 pairs non-adjacent
groups), bare `gflip` alone (~80% of gain comes from m-axis perturbation).
**`lmrev` demoted: prior "blkswap+lmrev TIE" was n=43910 sweep artifact** —
σ=768 in n=2048 was a fluke producing misleading DECISIVE call; Stage 2
corrected. **Don't trust mean_rank ordering at n<5000 for sub-σ_residual
(~600 cyc) effects.**

See `memory/project_w3x_n29420_basin.md`.

### Bloom filter validation (n=24 wall test, 2026-04-29)

`tools/bloom_filter.py` is **conservative-let-through, not predictive** — zero
false negatives on wall winners. Cumulative scorecard (n=43910 + n=29420):

| bloom verdict | count | wall outcome | match? |
|---|---|---|---|
| WORTHY (s ≥ +0.10) | 4 | 4 in basin floor | ✅ |
| MARGINAL (−0.10 < s < +0.10) | 16 | 14 basin/MOD-faster, 2 WEAK | ✅ no false reject |
| STUPID / OVERSHOOT-RISK | 5 | all in TIE/WEAK or DECISIVE LOSER (cidperm) | ✅ no false promote |

Use before paying CUDA build + B200 sweep cost: WORTHY = build, MARGINAL = build
(treat as unknown), OVERSHOOT-RISK = build but expect possible regression,
STUPID = skip unless you want negative-control data.

**Caveat — `adj_tn_diff` empirical channel may be stale.** snrot2 was originally
"empirical 2nd at n=32768" — at n=10978 with 32 cells it's 12th at +1750 cyc
DECISIVE behind blkswap.

### Swizzle metric pipeline

```bash
python3 tools/analyze_swizzle.py --csv /tmp/swizzle_metrics.csv  # simulate
python3 tools/cluster_swizzle.py /tmp/swizzle_metrics.csv        # cluster + verdict
```

Stage 1 replays `lin_tile = cluster_id + tt*NC` (NC=74, TILES_M=3626, TILES_N=3,
TICKS=147) per variant, emits 8 structural metrics covering A-strip reuse
(intra-tn/tm run, wavefront unique-tm count + multiplicity, cross-cluster
L2-window A reuse, fresh-A influx, peak unique tm, group-locality). Stage 2
PCA+KMeans+Ward against baked-in `WALL_NS200` (n=200 paired-pass cycles, 13
wall-labeled). Verdict: < 0.3 ANALYZER BLIND / < 1.0 WEAK / ≥ 1.0 CAPTURES LEVER.

**Current verdict (2026-04-28): PARTIALLY CAPTURES.** dgsnake/gflip/lmrev
within ~150 cyc are metric-indistinguishable; gflip_blkswap (TD=54) and
gflip_lmrev (TD=53) ARE captured by sign-stable τ axes (`adj_tm_diff`,
`tm_extent_mean`). Both predicted WORTHY by bloom filter before wall test
confirmed. Clustering not regression: 13 labeled × 8 features, βs unstable;
centroid-distance generalises gracefully to unseen variants.

## Compute floor

`tcgen05.mma.cta_group::2` produces cluster-wide work per instruction. Per-cluster
cycles = `147 tiles × 24 K_iters × cyc/iter`:

| source | cyc/iter | wall (B200) | notes |
|---|---|---|---|
| hardware MMA retirement | 460 | 0.896 ms | absolute ceiling, no staging — **unreachable** |
| bench NS=4 + W0-TMA overlap | 520.8 | 1.014 ms | published microbench |
| bench NS=4, no TMA overlap | 525.6 | 1.023 ms | published microbench |
| **fc2-w3x-strip** (NS=6+PREFILL) | **493** | **0.98502 ms** | **structural staging floor (1814685 cyc)** |
| **fc2-w3x** (full) | **502** | **1.00092 ms** | production (1846145 cyc) |
| cuBLASLt rank-1 (L2) | ~520 | 1.046 ms | reference |

**Strip vs full = 15.9 µs / 31460 cyc / 214 cyc/tile** (1.7% of 12482 cyc/tile MMA
budget) — exposed-on-critical-path epi work coupling W0-W3 epi-end and W5's
next-tile MMA via `bar.sync` / `mbar_wait`. ~98% of epi body IS hidden in MMA
shadow; the 2% that isn't is real headroom (contradicts earlier "100% hidden"
framing from PROFILE_W5).

**Gap decomposition** (vs structural floor, not unreachable 460-cyc ceiling):
- **89 µs** (pure-MMA → strip): NS=6 staging bubble. Unreachable without removing staging.
- **16 µs** (strip → fc2-w3x): exposed epi. Real headroom.
- **61 µs** (strip → cuBLASLt rank-1): rank-1 has 4× more exposed-epi than us.

**Realistic recoverable: ~1-3 µs** (per SASS-attribution + per-tt PROFILE_W5),
not the earlier ~5-10 µs estimate.

### Strip-vs-full localization (ncu --set full + per-tt PROFILE_W5, 2026-05-01)

SASS-attributed stall hotspots (`docs/fc2_w3x_ncu_sass.txt`,
`docs/fc2_w3x_ncu_details.txt`):

| stall samples | instruction | what it is |
|---|---|---|
| **5,024,622** | `SYNCS.PHASECHK.TRANS64.TRYWAIT` | **mbar spin-wait — 7× next category** |
| ~696K each | `F2FP.BF16.F32.PACK_AB`, `HADD2.BF16_V2`, `SHFL.IDX`, `IMAD`, `VIADD` | epi compute body (8 subpasses × 147 tiles) |
| ~174K each | `UTMACMDFLUSH`, `UTMASTG.4D`, `R2UR.OR`, `BSYNC.RECONVERGENT` | TMA store + scaffolding |

TC pipeline 98.49% → strip 0.985 ms IS the MMA-staging structural floor.

**Final-tile drain is FALSE.** Per-tt PROFILE_W5: tt=146 = 11872 cyc = *fastest*
tile (mean 12098). Slowest tt=0 (13723 cold-start) + periodic +200-840 cyc bumps
at tt=4/20/36/52/68/84/100/116/132 (~16-tile L2-replacement-cycle hits). 12 µs
gap between ncu `cyc_avg` (Δ +4664) and wall `cyc_max` (Δ +31460) is **cross-CTA
workload variance** (one CTA finishes ~26800 cyc later than mean), not final-tile
artifact. LAST_TILE_FAST_PATH would save <1 µs — not implemented.

**16 µs gap revised:** ~4 µs steady-state mbar / cluster bar.sync (NANOSLEEP
sweep below proves only ~0.06 µs tunable) + ~12 µs cross-CTA tail variance
(basin-floor dispatch already minimizes; within-basin spread at TIE-band).

**ncu warnings to IGNORE:**
- `"13398-way bank conflict, 40.22% Est. Speedup"` — STSM mis-attribution; ncu
  doesn't model `stmatrix`'s bank-routed datapath.
- `"35.01% Est. Speedup, 21.3 active threads/warp"` — warp-specialization by design.

FC1 strip is TMA-load-dominated, not compute-bound.

### NANOSLEEP_CYC sweep (n=5489, 2026-05-01)

10-cell sweep over `mbar_wait`'s `nanosleep.u32 N` immediate, all pinned to
`gflip_blkswap`. NS_CYC threaded as template parameter (zero runtime cost).

| variant | mean_rank | Δ vs ns20 | AUC | verdict |
|---|---|---|---|---|
| **ns32** | **4.90** | **−111 cyc** | 0.441 | WEAK faster |
| ns8/16/4 | 5.00–5.11 | −62 to −95 cyc | 0.451–0.466 | TIE |
| **ns20** (default) | 5.53 | 0 (anchor) | — | — |
| ns64/12/48 | 5.67–5.72 | +27 to +44 cyc | 0.517–0.522 | TIE |
| ns24 / ns0 (busy-spin) | 6.15 / 6.21 | +122 / +170 cyc | 0.564 / 0.577 | WEAK slower |

Total spread ~281 cyc / 0.15 µs end-to-end. **ns0 only WEAK slower, not DECISIVE
— mbar_wait runs on idle warps (W0-W3 post-STSM, W5 post-MMA-arrive); removing
nap doesn't unlock productive work, only burns issue slots already idle.** ns20
default stays at ~0.06 µs / ~0.006% wall addressable headroom — below promotion
bar. Anything in [4..32] functionally equivalent.

**Calibration: n=1373→n=5489 demoted 3 of 4 "WEAK faster" cells to TIE** (same
canonical pattern as gflip Stage 1 lmrev DECISIVE → Stage 2 mid-pack). σ_residual
≈ 1400 cyc dominates a 100-cyc effect even at n=5489. See
`memory/project_w3x_nanosleep_basin.md`.

## N×K dim sweep (pow2 grid, 2026-05-01, 16 cells)

`tools/dim_sweep_w3x.py` default = `N ∈ {256,512,1024,2048} × K ∈
{1024,2048,4096,8192}`. Cycles paired with cuBLASLt rank-1 via stream-serialized
`clock64()` sentinels (same SM-clock domain, throttling-invariant). Both EPI=3
BIAS_ONLY (cb_bias) and EPI=0 plain GEMM (cb_none).

| N | K | K_it | NS | ours cyc | cb_bias cyc | cb_none cyc | Δb% | Δn% |
|---|---|---|---|---|---|---|---|---|
| 256 | 1024 | 8  | 5\* | 397.6k  | n/a       | 390.3k    | n/a   | +1.88 |
| 256 | 2048 | 16 | 6   | 634.5k  | n/a       | 664.9k    | n/a   | −4.57 |
| 256 | 4096 | 32 | 6   | 1152.4k | n/a       | 1172.1k   | n/a   | −1.68 |
| 256 | 8192 | 64 | 6   | 2150.6k | 2453.4k   | 2451.9k   | −12.34| −12.29|
| 512 | 1024 | 8  | 5   | (was FAIL@NS=6 — auto-picker now NS=5)            |
| 512 | 2048 | 16 | 6   | 951.2k  | n/a       | 991.9k    | n/a   | −4.11 |
| 512 | 4096 | 32 | 6   | 1783.1k | 1767.7k   | 1753.7k   | +0.87 | +1.67 |
| 512 | 8192 | 64 | 6   | 3386.9k | 3451.5k   | 3446.9k   | −1.87 | −1.74 |
| 1024| 1024 | 8  | 5   | (was FAIL@NS=6 — auto-picker now NS=5)            |
| 1024| 2048 | 16 | 6   | 1692.5k | 1817.1k   | 1773.3k   | −6.86 | −4.55 |
| 1024| 4096 | 32 | 6   | 3405.1k | 3302.0k   | 3292.6k   | +3.12 | +3.42 |
| 1024| 8192 | 64 | 6   | 6510.4k | 6525.2k   | 6532.7k   | −0.23 | −0.34 |
| 2048| 1024 | 8  | 5   | 2717.4k | 2289.9k   | 2135.3k   |+18.67 |+27.26 |
| 2048| 2048 | 16 | 5   | 3288.5k | 3434.0k   | 3422.8k   | −4.24 | −3.92 |
| 2048| 4096 | 32 | 5   | 6727.9k | 6510.7k   | 6497.5k   | +3.34 | +3.55 |
| 2048| 8192 | 64 | 5   | 12940.7k| 12908.1k  | 12908.6k  | +0.25 | +0.25 |

\*N=256 K=1024 picks NS=5 via `min(NS_BY_N=6, K_ITERS−3=5)`. n/a = sparse
cuBLASLt heuristic table at small-N + medium-K (per-tensor FP8 BIAS_ONLY); EPI=0
covers all but FAIL cells.

**Production point K=3072 N=768: −4.15% in cycles** (1845.4k vs 1925.3k) —
bigger margin than the 1.046→1.001 ms gap suggests. Earlier non-pow2 sweep had
one loss (N=K=1536 K_iters=12 NO_PREFILL, +1.94%) — square-ish × NO_PREFILL cap.

**Three loss patterns in pow2 sweep:**
1. **N=2048 K=1024 catastrophe (+27%, eff=0.54).** NS=5 + NO_PREFILL + gap=3
   stack. Probably needs a different short-K kernel (no PREFILL ever, smaller tile).
2. **K=4096 systematic loss N≥512 (+1.7 to +3.5%).** Most actionable. cuBLASLt
   eff jumps 0.81→0.88 K=2048→4096; ours stays flat 0.86. Pending: harness now
   parses `# Winner:` from `cublaslt-introspect` and emits tile/stages/cluster/
   splitk per cell — re-run on B200 to fill in. Suspects: 256×96 tile
   (TILE_ID=495) for narrower N, deeper NS, or split-K.
3. **N=2048 NS=5 SMEM tax (~1-3.5%).** 1-stage-of-latency-hiding loss. Gap
   shrinks as K grows (3.5% K=4096 vs 0.25% K=8192). Only fixable by shrinking
   something else; LDTM_X64 dead-end says NUM_EPI_STAGES=1 costs +14 µs.

**Sweet spots:** K=2048 across all N (−4 to −7%, K_ITERS=16 past PREFILL,
gap=10); small N (N=256, −1 to −12%, cuBLASLt heuristic floor degrades faster
than ours at low tile/cluster).

See `memory/project_w3x_dim_sweep_vs_cublas.md` for cuBLASLt sparse-heuristic
gaps (K=1536 zero algos almost everywhere; EPI=0 recovers most).

## cuBLASLt rank-1 reference

`tools/probe_cublaslt.sh` (probe 1) enumerates every heuristic, times each,
reports rank-1. Reference target — we now beat it at FC2 K=3072 BIAS_ONLY.
Earlier comparisons against `cublas-bench-fc2` measured the **default
heuristic pick**, not rank-1.

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
(not SMEM-feasible at our 256x256). Top 5 listings all `cta_group::2`. Not
split-K (`splitk=1`), not CUTLASS-style swizzle (`swizzle=0`). SASS at
`rank1.sass`.

Tile enum (from `cublasLt.h`): `23=128x256, 24=256x128, 32=128x192, 197=168x128,
201=176x128, 495=256x96, 535=320x192`. `stages=36 = 128xAUTO`, NS resolved per
kernel variant at compile time. K=1024/2048 ERR — one heuristic IMAs on device.

## Status (2026-04-30)

fc2_w3x bias-only at 1.00092 ms with `gflip_blkswap` (TD=54); W5 MMA-ceiling-bound
(~12482 cyc/tile ≈ 24×520 cyc/iter), tensor pipe 95.84% active. Tree state:
bias-preload default (Δ=−1.73 µs at z=−23.23 STRONG, n=128), STSM mandatory
(rank-1 SASS opcode parity), 4D packed-tile output ABI (`[TILES_M, TILES_N, TM*2,
TN]`; host `pack_idx_C(m,n)`); SASS-level epi tuning exhausted. Histories:
`memory/project_w3x_bias_preload_win.md`, `project_w3x_packed_c_abi.md`.

**Realistic remaining headroom ~1-3 µs** — largest single target by past-win
standards (bias-preload 1.7 µs, STSM 0.4 µs) but probably needs new lever class.

**Next:** port `fc2_w3x` from bias-only to fused-residual (production target).

## Dead ends — do NOT retry

See `memory/MEMORY.md` for full chronological log. Each item below has a memory file.

- **Source-level epi tuning** — ptxas owns STS layout. CUTLASS_LOOP, FP32_EPILOGUE,
  cvta.shared, NUM_EPI_STAGES, stmatrix variants all generate identical SASS.
- **Cross-warp STS clustering (intra-warp)** — SELF_LOAD, SELF_STAGGER, SASS
  intra-warp reorder all zero effect (wrong axis). *Inter-cluster* arrival into
  STS/TMA-store IS ordering-controlled (g-s).
- **Hand-written PTX `fc2_w3x.ptx`** — byte-identical SASS to nvcc emission.
  PTX has no UR type; ptxas owns R-vs-UR. PTX escape hatch does not exist.
  Frozen at `fc2-w3x-ptx`.
- **K_UNROLL** — u1/u2/u3/u4/u8 regress 87–197 µs (UR datapath collapses on
  non-N_STAGES-multiples; UIADD3=7, regs=66). u6/u12/u24 tie default within
  0.4 µs (UR on, regs=64). `K_UNROLL=24` shrinks SASS 39% (10029→6077 lines)
  at parity wall (free cleanup). `memory/project_k_unroll_sweep.md`.
- **Cluster-axis swap** — B200 hard-rejects `(1,2,1)`/`(1,1,2)` cluster_dims
  even with explicit launch attrs + peer-addressing. cuBLASLt's `2x1`/`1x2`
  notation is logical labeling inside an X-axis grid (`rank1.sass`: cluster
  (2,1,1)). 2-CTA clusters X-axis-only on B200.
  `memory/dead_cluster_axis_swap.md`.
- **KERN_3WARP merge** — merging W4 TMA into W5 MMA (single combined warp on
  CTA0) regresses 1.006→1.172 ms (+166 µs). W4 empty_wait is hw-sleep, not free
  issue; merging serializes TMA+wait onto W5 (~+90 cyc/iter). Structural warp
  floor = 4. `memory/dead_kern_3warp.md`.
- **EPI_2WARP + DROP_LEAD_BARSYNC marginal opt-in win (n=128):** v10011
  (NO_BULK + DROP_LEAD + EPI_2WARP) −0.27 µs at z=−3.49 STRONG; v10110 (no
  EPI_2WARP) regresses +2.54 µs at z=+34.9. Stays opt-in — DROP_LEAD ships
  cross-warp STS-before-TMA race; EPI_2WARP fc2_w3x-only.
  `memory/project_w3x_epi2warp_marginal.md`.
- **LDTM_X32 ties STSM at MMA floor (n=512):** Δ=−0.021 µs, p=0.6523. Cycle
  savings live in W0-W3 MMA shadow. STSM stays default by symmetry with rank-1.
  **LDTM_X64 forces NUM_EPI_STAGES=2→1 (SMEM budget) and regresses +14 µs
  STRONG** (n=16) — confirms NS_EPI=2 is worth ~14 µs (corrects prior
  "±3 µs NS_EPI" footnote). `memory/dead_ldtm_x32_tie.md`.
- **fc2_w3x post-WIN levers (all ±3 µs or regression):** subpass 8→4, cross-tile
  TMA carry, SWIZZLE_64B, DROP_TRAIL_BARSYNC, WAIT_GROUP_READ, **XPF_A/B**
  (Bonferroni-confirmed: XPF_A +3.05 µs at z=+6.30, XPF_B +1.27 µs at z=+2.62
  in 2026-04-26 128-cell combo; macros removed), CHET/PMIX/INGH hybrid
  dispatches, **gflip_cidperm** (TD=55, +1568 cyc DECISIVE at n=43910 — `c*15
  mod 74` cluster perm breaks SM→L2 contiguity, bloom-filter overshoot caught
  this), **STAGGER=2 split-mbar** (uniformly +3 µs across 11 dispatches; +36
  cyc on each W4/W5; macro removed `memory/dead_fc2_w3x_stagger.md`), DG sweep,
  native BF16 epilogue (kept ±0 wall, cleaner). The earlier n=5489 "3-way front
  TIE" (dgsnake/lmrev/gflip) is MOD behind n=29420 7-cell basin; n=43910
  "blkswap+lmrev TIE" superseded.
- **Older dead variants:** TD=1 atomic, TD=5 CLC, TD=6/7 inline atomic,
  COL_LOCK, 4-CTA TMA multicast (silent deadlock), mbar→SMEM polling, L2 cache
  hints, dgphase/dgnrot (TD=23/24), fc2_ldg (LDG/STG), fc2_hybrid (CUTLASS
  phases 2/3b/4), N-batch / phase-offset / Group-3 (pre-PACKED_TILES — re-test
  before citing).
- **FC1 FORCE_PREFILL** — deadlocks at K_ITERS=6. NO_PREFILL guard mandatory.

## Build and run

```bash
# FC2 BIAS_ONLY (BEST — beats cuBLASLt rank-1)
make fc2-w3x && ./fc2-w3x                        # ~1.001 ms (bias-preload + STSM mandatory)
make fc2-w3x-strip && ./fc2-w3x-strip            # NS=6+PREFILL staging floor (~0.985 ms)
make fc2-w3x-ptx                                 # hand-written PTX, byte-identical SASS

# fc2_w3x sweeps + diagnostics
make fc2-w3x-tile-sweep                          # TILE_DISPATCH macro variants
./tools/sweep_fc2_w3x_swizzle.sh                 # dispatch sweep (SWEEP=front for top tier)
./tools/sweep_fc2_w3x_nanosleep.sh               # NS_CYC sweep
./tools/sweep_fc2_w3x_dg.sh                      # DG_GROUP_SIZE × INNER_T × STAGGER
./tools/sweep_fc2_w3x_prof.sh                    # per-warp clock64 phases
python3 tools/aggregate_prof.py data/<dir>
make fc2-w3x DFLAGS='-DPROFILE_CYCLES'           # per-warp phases
make fc2-w3x DFLAGS='-DPROFILE_KI|-DPROFILE_TILE|-DPROFILE_W5'

# FC2 fused-residual + FC1
make fc2-w3 && ./fc2-w3                          # fused 1.063 (dgswizzle TD=8)
make fc1-w3 && ./fc1-w3                          # FC1 fused 1.998 (zigzag+ks=1)

# Custom dims (MUST use -B: Make doesn't track DFLAGS)
make -B fc2-w3 DFLAGS='-DM_TOTAL=464128 -DN_DIM=1024 -DK_DIM=2048'
# Decomp via DFLAGS: -DSTRIP_EPILOGUE / -DGEMM_ONLY

# References + profiling
make fc2-cutlass && ./fc2-cutlass                # 1.226
./tools/probe_cublaslt.sh                        # cuBLASLt rank-1 (TRUE ceiling)
bash tools/bench.sh --comprehensive              # rank-1-baselined
bash tools/ncu_bench.sh && python3 tools/ncu_anova.py
bash tools/ncu_fc2_w3x.sh --max --reps 3
bash tools/ncu_fc2_pipes.sh                      # dodges --set full deadlock
./tools/dim_sweep.sh --fast                      # fc2_w3 80 configs (M×N×K)
./tools/dim_sweep_w3x.py                         # fc2_w3x N×K pow2 grid (16 cells, vs cuBLASLt)
```

## Key files

```
fc2_w3x.cu               FC2 bias-only (ACTIVE — beats rank-1)
fc2_w3x.ptx              Hand-written PTX, byte-identical SASS (frozen)
fc2_w3.cu                FC2 fused-residual (ACTIVE for fused path)
fc1_w3.cu                FC1 (ACTIVE)
fc2_ws.cu                FC2 warp-specialized w/ rank-1 warp retirement
tile_dispatch.cuh        Shared TD=8..58 (CHET/PMIX/INGH, gflip family)
fc2_cutlass.cu           CUTLASS reference
fc2_hybrid.cu, fc2_ldg.cu, fc2.cu  DEAD
kernel_common.cuh, kernel_body.cuh  Shared infra
Makefile                 sm_100a, DFLAGS for dim override
docs/STSM_STATUS.md
docs/PURE_PTX_REWRITE_STRATEGY.md
docs/BENCHMARKING.md     cycles/AUC/η²/rank study guide — READ before benchmarking
docs/fc2_w3x_ncu_sass.txt    ncu --set full per-SASS stall sampling (mbar 7×)
docs/fc2_w3x_ncu_details.txt ncu --page details (98.5% TC pipe, ignore STSM bank-conflict warns)
rank1.sass               Dumped cuBLASLt rank-1 for diffing
tools/bench.sh, probe_cublaslt.sh, dim_sweep.sh, dim_sweep_w3x.py
tools/sweep_fc2_w3x_*.sh tiles / dg / nanosleep / prof
tools/ncu_*.sh, ncu_anova.py
tools/aggregate_prof.py  PROFILE_* aggregator
tools/analyze_swizzle.py, cluster_swizzle.py    structural metric simulator + verdict
tools/anova_1way.py      paired ANOVA + AUC + Cohen's d + η² + mean-rank/win%
tools/sass_edit.py       SASS binary editor + CP-SAT scheduler
token_count.py           tiktoken budgeting
bench/                   Microbenchmarks (TMA, MMA, stmatrix, cublaslt_introspect)
data/                    Benchmark + ncu results
```

## SM100a hardware (B200-measured)

- STS.128: 27 cyc | LDS.128: 25 cyc @ILP=1, 3.5 cyc @ILP=7
- TMA load: 419 cyc (L2-warm) | TMA store: 197 cyc
- TMEM load (tcgen05.ld.sync): 2 cyc regardless of width/ILP
- MMA K-iter: 665 cyc (pipelined: 525.6 cyc/iter)
- STS scaling: 10→37 cyc at 8 warps (3.65× contention)
- LDS scaling: 4.5→16 cyc (3.56×)
- FFMA: ~free (1.36× at 8 warps); F2FP: zero contention (flat 2.0 cyc)

## Key constraints

- Target: sm_100a (B200, 148 SMs), `cta_group::2`, 74 clusters
- TMEM: 512 cols, single alloc for double buffering. SMEM: 228 KB/SM
- All inline PTX in fc2_w3.cu/fc1_w3.cu (no CUTLASS dependency)
- OFF_STAGING must be 1024-byte aligned for SWIZZLE_128B
- `fence.proxy.async.shared::cta` required before TMA store after st.shared
- N_STAGES + PREFILL kernel-side auto-picked from N_DIM + K_ITERS (commit 3d6c1cb)
- BIAS_SMEM=1 default (-15 µs free); custom dims require `make -B`
- W0's K-loop is TMA-sensitive: any global op (atomicAdd, etc.) costs +41–77%
  tma_issue. Non-critical-path global ops (W7 scheduler at tile-boundary) fine.

## Benchmarking

See `docs/BENCHMARKING.md`. TL;DR for any timing-claim work:

- **Cycles, not ms.** `clock64()` per-CTA, `max_over_CTAs / N_TIMED`. Clock-freq
  invariant — required on vast.ai (no locked clocks).
- **Pass-major** (randomized block): outer pass p, inner variant v.
  `@@SAMPLE pass=p variant=v cyc=Y` per launch.
- **Trim** first 33–50% of passes (cold L2 + thermal ramp).
- **Paired analysis** by pass; report **AUC**, **Cohen's d**, **η²**, **mean
  rank**, **win%**. **No p-values** — meaningless at large n.
- `tools/anova_1way.py --metric cyc --paired rep --trim 0.33` is canonical.
- **n thresholds (σ_residual ~1400 cyc on fc2_w3x):** n<5000 unreliable for
  sub-σ effects (Stage 1 lmrev DECISIVE n=2048 → Stage 2 mid-pack n=29420).
  ~600 cyc Δ MOD-band: n≥10978; TIE-band (~150 cyc): n≥43910 — but within-basin
  may sit below resolution floor regardless. Default 2-stage: REPS=2048 filter
  → REPS=43910 survivors.

## Working in this repo

Names say what, comments say why. No single-line `/**/`, no multi-line `//`, no
decorated block comments. Bare `/*` open, undecorated lines, `*/` close.

Don't narrate tool calls; don't echo file contents; parallelize independent tool
calls; use offset/limit for large files.

LLM context is the binding constraint — treat CLAUDE.md, docs, memory as a token
budget. Prefer brief pointers to topic files over inlining. `./token_count.py
<file>` is a coarse proxy (tiktoken o200k_base); Claude's tokenizer typically
reports ~2× higher on table-heavy markdown (see `/context` for actual cost).
