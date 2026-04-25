# SigLIP2 Vision Encoder — Hand-tuned Blackwell GEMM Kernels

Hand-tuned SM100a persistent GEMM kernels for FC1 and FC2 layers of `google/siglip2-base-patch16-224`.
FP8 (E4M3) inputs, BF16 output, tcgen05 MMA, TMA, `cta_group::2` with 2-CTA clusters.
Cross-compiled on CPU VPS, runs on B200 (148 SMs, 74 clusters). PE kernel is done — see `CLAUDE.md.mothballed`.

## Current best (B200, 2026-04-21)

| target | ms | kernel | dispatch | vs cuBLASLt rank-1 |
|---|---|---|---|---|
| FC2 K=3072 BIAS_ONLY | **1.007** | `fc2_w3x` | dgswizzle PACKED | **−39 µs** (rank-1: 1.046) |
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
**Unsafe at K_ITERS<20** (parity wrap → deadlock); auto-guarded `#if K_DIM/128 < 20`.
FC1 (K_ITERS=6) always uses NO_PREFILL.

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

### fc2_w3x dispatch is near-flat (locked-clock ncu, 2026-04-23)

The `fc2_w3` table above is fused-with-residual. On `fc2_w3x` (bias-only,
production for that path), the dispatch lever has compressed: top 8 variants
land within 1.7 µs of dgsw (n=10 wall, σ=0.2–0.7 µs each — sub-noise).

| fc2_w3x dispatch | Δ vs dgsw | long_sb | L2 hit% | DRAM rd | amp |
|---|---|---|---|---|---|
| dgswizzle G=8 (default)        | —        | 6.71 | 67.44 | 2.988 GB | 1.048× |
| checkered / dgsnake / dg4 / hilbert / zigzag / rowmajor | ±2 µs | 6.64–7.21 | 65.1–68.7 | 2.94–3.03 | 1.03–1.06× |
| nlock                          | +52 µs   | — | — | — | — |
| nflat / ncyrot / nsnake / ncycle | +100–186 µs | — | — | — | — |
| pmix hybrid (TD=31)            | +181 µs  | 8.51 | 51.82 | **5.724 GB** | **1.99×** |

The 6-warp persistent structure (no W7, no W2 EpilogueLoad) is less dispatch-
sensitive than fc2_w3. dgsw stays default. PMIX is the cautionary case: a
dgsw+rowmajor per-cluster mix that passes a tile-bijection check still
destroys L2 staggering (51.82% hit) and doubles DRAM traffic.

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

### Adaptive tuning knobs

| Knob | Rule | Why |
|---|---|---|
| N_STAGES | NS6 for N≤1536, NS5 for N>1536 | SMEM per stage grows with N |
| PREFILL | On for K_ITERS≥20, off otherwise | Short K-loop deadlocks (parity wrap) |
| Dispatch | FC2: zigzag or dgswizzle. FC1: zigzag + K_STAGGER=1. | PACKED_TILES + odd ks on FC1; FC2 wash on ks. |

## Compute floor

`tcgen05.mma.cta_group::2` produces cluster-wide work per instruction (no extra ×2 CTA
factor). Per-cluster floor = `147 tiles × 24 K_iters × cyc/iter / clock`:

| cyc/iter | what | @ 1.813 GHz | @ 1.965 GHz |
|---|---|---|---|
| 460 | hardware MMA retirement (no staging) | 0.896 ms | 0.827 ms |
| 520.8 | bench NS=4 + W0-TMA overlap | 1.014 ms | 0.935 ms |
| 525.6 | bench NS=4, no TMA overlap | 1.023 ms | 0.944 ms |

`fc2_w3x` at 1.007 ms back-solves to 517 cyc/iter (base) or 561 (boost). ~100 µs
headroom remains to pure-MMA floor — staging + pipeline bubbles, not compute-bound.
FC1 strip is TMA-load-dominated, not compute-bound.

## cuBLASLt rank-1

`tools/probe_cublaslt.sh` (probe 1) enumerates every heuristic, times each, reports
rank-1. This is the true ceiling. Earlier comparisons against `cublas-bench-fc2`
measured the **default heuristic pick**, not rank-1.

### FC2 K-sweep

| K    | cuBLASLt  | best ours | gap |
|------|-----------|-----------|-----|
| 1024 | ERR       | 0.859 (lean)        | n/a |
| 2048 | ERR       | 0.922 (zigzag)      | n/a |
| 3072 | **1.046** | 1.064 (dgsw fused) / **1.007 (w3x bias-only)** | +18 µs / **−39 µs** |
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

### Status (2026-04-24): wall stuck at 1.007 ms

`fc2_w3x` bias-only at 1.007 ms; W5 is MMA-ceiling-bound (~12482 cyc/tile ≈ 24 × 520
cyc/iter, per `PROFILE_W5`). Tensor pipe 95.84% active. Every store-side / epilogue /
dispatch lever tried since lands ±3 µs noise. See `docs/W3X_GRIEVANCES_VS_RANK1.md`
for 9 remaining SASS deltas (15-25 µs total upside).

**Active:** Lever C (USE_STMATRIX=1) — `bcce329` layout fix matches rank-1 SASS opcode
mix (STS.128 4→0, STSM.16.M88.4 0→4, regs 64→54). UNVERIFIED on B200; pure
diagnostic value if it passes (no perf upside expected).

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
- **K-loop `#pragma unroll 1`:** 6× SASS shrink (9805→1645 lines) but +200 µs wall.
  Full-unroll gives ptxas cross-ki scheduling freedom. Reverted in `31ad6cb`. Don't re-roll.
- **fc2_w3x post-WIN levers (all ±3 µs or regression):** subpass 8→4, cross-tile TMA
  carry, SWIZZLE_64B, NS_EPI sweep, EPI_2WARP, DROP_TRAIL_BARSYNC, WAIT_GROUP_READ,
  DROP_LEAD_BARSYNC, XPF_A/B prefetch, CHET/PMIX/INGH hybrid dispatches, 11 non-dgsw
  TILE_DISPATCH variants, DG sweep, native BF16 epilogue (kept ±0 wall, cleaner).
- **Older dead variants:** TD=1 atomic, TD=5 CLC, TD=6/7 inline atomic, COL_LOCK,
  4-CTA TMA multicast (silent deadlock), mbar→SMEM polling, L2 cache hints,
  dgphase/dgnrot (TD=23/24), fc2_ldg (LDG/STG), fc2_hybrid (CUTLASS phases 2/3b/4),
  N-batch / phase-offset / Group-3 (pre-PACKED_TILES — re-test before citing).
- **FC1 FORCE_PREFILL:** deadlocks at K_ITERS=6. NO_PREFILL guard is necessary.

## Build and run

```bash
# FC2 BIAS_ONLY (BEST — beats cuBLASLt rank-1)
make fc2-w3x && ./fc2-w3x                        # 1.007 ms
make fc2-w3x-stsm && ./fc2-w3x-stsm              # Lever C USE_STMATRIX — UNVERIFIED
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
./tools/dim_sweep.sh --fast                      # 80 configs
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
docs/W3X_GRIEVANCES_VS_RANK1.md # 9 SASS-level deltas vs rank-1
docs/LEVER_C_STATUS.md          # STSM layout playbook
docs/PURE_PTX_REWRITE_STRATEGY.md
rank1.sass                      # Dumped cuBLASLt rank-1 for diffing
tools/bench.sh                  # FC1/FC2 × dispatch × packed × decomp (rank-1 baseline)
tools/probe_cublaslt.sh         # cuBLASLt rank-1 timing
tools/dim_sweep.sh              # M/N/K grid
tools/ncu_bench.sh, ncu_fc2_w3x.sh, ncu_fc2_pipes.sh   # ncu profiling
tools/sweep_fc2_w3x_*.sh        # tiles / dg / prof sweeps
tools/aggregate_prof.py         # PROFILE_* aggregator
tools/ncu_anova.py
tools/tile_regress.py           # Python TD simulation + regression on tile-sequence features
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
python3 token_count.py docs/W3X_GRIEVANCES_VS_RANK1.md
find docs/ memory/ -name '*.md' | xargs -I{} python3 token_count.py {} | grep o200k_base
```
