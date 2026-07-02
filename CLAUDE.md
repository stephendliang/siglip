# SigLIP Vision Encoder — Hand-tuned Blackwell GEMM Kernels

SM100a (tcgen05) persistent GEMM for SigLIP FC1/FC2; FP8 (E4M3) → BF16.
Production points: FC2 M=928256 N=768 K=3072 (+residual); FC1 M=928256 N=3072
K=768 (GELU+bias).

## Current best (B200, cycles-first)

| target | result | kernel (dispatch) | vs references |
|---|---|---|---|
| FC2 bias-only strip floor | 0.98502 ms | `fc2_w3x -DSTRIP_EPILOGUE` | NS=6+PREFILL structural floor |
| **FC2 bias-only** | **1.00092 ms** | `fc2_w3x` (gflip_blkswap TD=54) | **−27 µs** vs cuBLASLt PT 1.028 / −116 µs vs MXFP8 1.117 |
| **FC2 fused +residual** | **1902.5 kcyc** (wall 1.06–1.13, clock-dep) | `fc2_w3` (TD=54, back-pressured NS=6) | **−14.8%** vs CUTLASS 2232.0k / −16.7% vs cuBLASLt beta=1 2283.6k |
| FC1 GELU+bias | 3487.7 kcyc (wall 1.89; matrix cell, base anchor 3620.8) | `fc1_w3` (TD=11+ks=1, EPI_DECOUPLE+ES=3+BIAS_PER_TILE ring defaults) | **+2.8%** vs cuBLASLt 13.3 PT 3391.2k / +0.3% vs MXFP8 3476.7k / −18.7% vs CUTLASS 4288.1k |

`fc2_w3x` = clean-sheet 6-warp bias-only. `fc2_w3` = legacy 7-warp, production
residual path. `fc1_w3` = FC1 production (legacy family); `fc1_w3x` (clean-sheet
port) sits at ~3.11 ms — exposed-epilogue problem, does NOT supersede fc1_w3 yet.

**FC1 is the open front.** libcublasLt 13.3.0.5 (algoId=66 family, GELU
verified via epi_ok witness) beats us by 96.5 kcyc at our exact geometry.
fc1_w3 is **epilogue-RATE-bound** (1-tile overlap exists; E = 5.93 kcyc/tile
vs vendor 5.77, loader 4.64@NS5 / 3.09@NS6, MMA 3.15 — MMA-side levers can't
move the wall). Shipped in-file defaults: EPI_DECOUPLE+ES=3 (2026-07-02,
−336.2 kcyc — per-subiter cross-warp barriers dropped, warps self-pace +
3-deep store ring; `-DNO_EPI_DECOUPLE -DNUM_EPI_STAGES=2` reverts
SASS-identically) and BIAS_PER_TILE cp.async 3-slot ring at TD≥8
(2026-07-02 evening, −133.1 kcyc; `-DNO_BIAS_PER_TILE` reverts; mechanism
PROVEN = compact 512 B slots: `-DSTAGING_PAD=4608` restores the old
OFF_STAGING=171008 and the win survives).
**Remaining gap is bare store RATE, not math or opcode:** vendor GEMM-only
rank-1 = 2510.4 kcyc (4.27 kcyc/tile bare store, full GELU only +1.5/tile on
top) vs our GEMM_ONLY 3227.3 (5.49/tile — our GELU-on-top +0.44/tile is
CHEAPER than theirs); packed f32x2 GELU (4.5 slots/elem, denser than vendor
5.5) REGRESSED +31 — not FP-issue-bound; STSM stores REGRESSED +45..52
GELU / TIE GEMM_ONLY even at vendor's exact STSM.16.MT88.4
(`-DSTSM_STORE[=1|2]` timing-only probe — 16x256b LDTM rewrite not worth
it). Left: wait_group→UTMACMDFLUSH-style pacing, LDTM width, subpass shape.
`memory/project-fc1-gelu-x2-bias-ring.md`.
NS=6 matrix 2026-07-02: TIE at production (and pays the ES=2 tax vs ring);
strip NS=6 hits the MMA floor (3.09 kcyc/tile, −33% — NS=5's "loader floor"
was the within-tile stage-0 recycle). Epi-rate runway to ~1.85 Mcyc; ring
makes NS6+ES2 / NS5+ES4 fit at 231424 B (ES=4 tested: +169 at normal rate).
`memory/project-fc1-ns6-matrix.md`.

## Kernel structure

7-warp persistent (224 threads), `cta_group::2`, `__cluster_dims__(2,1,1)`,
tile 256x256x128, K_ITERS = K_DIM/128 (FC2 24, FC1 6).
W0 TMA A+B (TMA-sensitive: any global op in K-loop costs +41–77% tma_issue) ·
W1 tcgen05.mma K-loop (TMEM 512 cols double-buffered) · W2 epi TMA load
(residual) · W3-W6 epi compute (LDS + TMEM ld + CVT + STS + TMA store).
`fc2_w3x`: 6 warps — W0-W3 epi, W4 TMA, W5 MMA CTA0-only, `buf = tt & 1`.
w3x output ABI: 4D packed tiles `[TILES_M, TILES_N, TM*2, TN]`, host `pack_idx_C(m,n)`.
PREFILL overlaps prev tile's epi drain with first 6 K-iters of next tile's MMA.

## Tuning rules

| knob | rule |
|---|---|
| N_STAGES | auto `min(NS_BY_N, max(2, K_ITERS−3))`; NS_BY_N=6/5/4/3 for N≤1536/2048/4096/larger. 228 KB SMEM ceiling (32 KB A+B/stage); fill-gap≥3 (gap=2 FAILs K=1024 NS=6) |
| PREFILL | auto K_ITERS≥20, else NO_PREFILL (short K deadlocks on parity wrap). PREFILL eff ~0.91 vs ~0.77. fc2_w3 residual FULL auto-forces back-pressure (= NO_PREFILL path, free at K_ITERS=24; `-DFORCE_PREFILL` opts out) |
| Dispatch | compile-time only (workstealing stripped). FC2 both kernels: gflip_blkswap TD=54. FC1: zigzag TD=11 + K_STAGGER=1 (odd ks helps FC1, wash on FC2). stride TD=0 on fused fc2_w3 = +8.4% — never ship |

Dispatch mechanism (settled): static swizzles > workstealing ~30 µs; bottleneck
is `long_scoreboard` stalls, NOT DRAM amplification — CUTLASS at 1.000× amp is
185 µs slower; **tensor-pipe utilization is the lever**. fc2_w3x m-axis basin
is wide and tied (η²=0.0075): TD=54 is mid-basin, don't churn. Hurts: bare
gflip, non-XK=1 pairings, gflip_cidperm (breaks SM→L2 contiguity). lmrev: fine
bias-only, +12–14k cyc on fused — residual consumer is m-traversal-sensitive.
Tables: `memory/project-perf-table-archive.md`.

**fc2_w3 residual race — FIXED 2026-07-01 (170e543), goal CLOSED.** Root cause
= buffer/rate deficit, not handshake: free-running NS=6 MMA under PREFILL laps
the 2-slot ReuseSmemC staging (~0.25% elements dirty/run). Fix = the auto
back-pressure above; dirty=0/1850 cumulative, 1.077 ms (vs 1.079 PREFILL).
Gate ANY residual change with `-DSELF_DIFF=N` (double-launch bitwise self-diff,
`@@SELFDIFF_SUMMARY`; 32-spot checks miss sparse corruption).
Full analysis: `memory/project-fc2-w3-epilogue-race.md`.

## Compute floor (FC2)

Per-cluster cyc = 147 tiles × 24 K_iters × cyc/iter:

| source | cyc/iter | wall |
|---|---|---|
| HW MMA retirement (no staging) | 460 | 0.896 ms — unreachable |
| **fc2-w3x-strip** (NS=6+PREFILL) | **493** | **0.98502** — structural staging floor |
| **fc2-w3x** (full) | **502** | **1.00092** — production |
| cuBLASLt PT fused rank-1 | ~515 | 1.028 |
| cuBLASLt MXFP8 fused rank-1 | ~559 | 1.117 |

Gap: 89 µs (MMA→strip, staging bubble, unreachable) + 16 µs (strip→full,
exposed epi ≈ 4 µs steady-state mbar/bar.sync + 12 µs cross-CTA tail variance).
TC pipe 98.5%; mbar spin-wait dominates SASS stalls 7× next category
(regen `tools/ncu_fc2_w3x.sh`). Realistic recoverable ~1–3 µs — needs a new
lever class (past wins: bias-preload 1.7 µs, STSM 0.4 µs; SASS-level epi
tuning exhausted). NANOSLEEP: [4..32] cyc equivalent, ns20 stays.
**Ignore ncu warnings:** "13398-way bank conflict" (STSM mis-attribution) and
"21.3 active threads/warp" (warp specialization). FC1 strip is TMA-load-dominated.

## cuBLASLt reference

`bench/fc_problem.cuh` = single source of truth (FP8, **BF16 bias**, [N,M]
layout) for `cublas_bench` + `cublaslt_introspect`. **BF16 bias is mandatory**
— FP32 bias enumerates 0 fused algos on sm_100a; layout orientation changes
runtime, not enumeration.
`./cublaslt-introspect <M> <N> <K> <epi> [scale] [beta]` enumerates + times all
heuristics, reports rank-1; `res_ok` col guards silent C-skip, `epi_ok` guards
silent GELU/bias-skip (beta=0, bias=−0.5 witness). Rank-1 at both production
points = algoId=66 tile=23 (128x256) NS=36(AUTO) cluster 2x1x1 — exactly our
geometry. Decode + cluster-id map: `memory/project-perf-table-archive.md` and
`tools/dim_sweep_w3x.py:CLUSTER_SHAPE_NAME`.

### FC2 K=3072 (wall ms, [N,M] BF16 bias)

| variant | ms |
|---|---|
| **fc2_w3x** bias-only fused | **1.001** |
| cuBLASLt fused BIAS_ONLY PT rank-1 | 1.028 |
| cuBLASLt GEMM-only rank-1 | 1.043 |
| cuBLASLt fused BIAS_ONLY MXFP8 rank-1 | 1.117 |
| cuBLASLt BIAS + residual beta=1 (beta=0 control 1.0346) | 1.237 |
| cuBLASLt unfused (GEMM + bias kernel) | 1.546 |

### FC2 fused-residual paired cyc (2026-07-01, one container, 3 interleaved rounds)

| impl | kcyc/launch | wall ms | GHz |
|---|---|---|---|
| **fc2_w3** (TD=54, back-pressured NS=6) | **1902.5** (±2.7k) | 1.130 | 1.68 |
| CUTLASS fc2-cutlass | 2232.0 (±2.5k) | 1.209 | 1.85 |
| cuBLASLt rank-1 beta=1 | 2283.6 (±0.8k) | 1.237 | 1.85 |

fc2_w3 beats even cuBLASLt's residual-free beta=0 control (1910.6k). cuBLASLt
CAN fuse residual (all 8 algoId=66 kernels pass res_ok) but pays +19.5% cyc
over beta=0; ours pays +3.2% over w3x. **Wall understates the lead:** fc2_w3's
denser draw hits the 1000 W SW power cap (1.68 vs 1.85 GHz; not preventable,
`-lgc`/`-pl` denied on Modal) — cycles-first reporting is the mitigation. cyc
is container-portable (+0.15%); wall is not. Protocol: stream-serialized
clock64 sentinels, avg over 10 launches; fc2_w3/fc2_cutlass/fc1_w3/fc1_cutlass
emit `@@CYC` unconditionally. Detail: `memory/project-perf-table-archive.md`,
logs `data/residual_introspect_20260701/`.

### FC1 K=768 paired cyc (2026-07-02, libcublasLt 13.3.0.5)

| impl | kcyc/launch | wall ms |
|---|---|---|
| cuBLASLt GEMM-only rank-1 (epi=0, context) | 2510.4 | 1.360 |
| fc1_w3 GEMM_ONLY (store-only, context) | 3227.3 | 1.748 |
| cuBLASLt PT rank-1 (algoId=66 t23 cl 2x1x1) | **3391.2** (±0.3k) | 1.838 |
| **fc1_w3** (production; matrix cell, base anchor 3620.8) | **3487.7** | 1.891 |
| cuBLASLt MXFP8 rank-1 (same identity) | 3476.7 (±0.4k) | 1.885 |
| fc1_w3 pre-ring (EPI_DECOUPLE+ES=3, paired) | 3619.4 (±1.8k) | 1.961 |
| fc1_cutlass (GELU_taylor) | 4288.1 (±0.1k) | 2.324 |

EPI_DECOUPLE+ES=3 = −336.2 kcyc (−8.5%) vs the 2026-07-01 lockstep build
(3955.6, itself −43.1 over TD=0): both per-subiter 128-thread barriers
dropped (staging regions are warp-private → warps self-pace; single-flag
split: decouple −310, ES=3 −123, combined −326) + 3-deep store ring.
SELF_DIFF dirty=0/100 ×4 runs; `-DNUM_EPI_STAGES=2 -DNO_EPI_DECOUPLE`
reproduces the old build SASS byte-identically. No differential throttle at
FC1 (~1.85 GHz all impls). Logs `data/fc1_epi_decouple_20260702/`.
BIAS_PER_TILE ring = −133.1 further (2026-07-02 evening 9-cell matrix, all
valid=1 + dirty=0/100; ES=4 +169, NS=6 +120 vs ring; paired rerun pending).
Logs `data/fc1_x2_ring_20260702/`. Store probe (same day, logs
`data/fc1_store_probe_20260702/`): base 3484.5 reproduced ring cross-container
(+13.2 base2 drift); STSM/pad/GEMM_ONLY verdicts in the open-front paragraph.

### Dim/K sweep conclusions (full tables: `memory/project-perf-table-archive.md`)

- **FC2 vs cuBLASLt BIAS_ONLY (cyc-paired pow2 grid + K-sweep at N=768, both
  columns valid):** we win long K (K=8192: −0.1..−11%; K=6144 −23 µs; K=3072
  −22.7 µs), tie K=2048, **lose K=1024 universally** (+5..+29%; NS=5 +
  NO_PREFILL + gap=3 stack pays) and **K=4096 at N≥512** (+2..+4%, +59 µs at
  N=768 — possibly actionable: tune basin at K=4096). N=2048 pays NS=5 SMEM tax.
  cuBLASLt picks algoId=66 tile 128x256 cluster 2x1x1 almost everywhere.
- **FC1 grid (CUDA-13.0-era cuBLASLt columns — STALE, algoId=71 era):** ours
  dominated K≤1024 across N (−11..−18%), tie K=1536, lost K≥2048. Our-kernel
  column still directional across dims; re-run on 13.3 before citing cuBLASLt.

## Dead ends — do NOT retry

Per-item files: `memory/MEMORY.md`. One-liners:

- **fc2_w3y (residual on fc2_w3x):** w3x's 1.001 floor is a zero-slack MMA;
  residual needs the slack the floor removed. NS=6 deadlock / NS=5 corrupt /
  NS=4 valid but 1.243 ms. Residual stays in fc2_w3. `memory/project-fc2-resadd-port.md`.
- **Source-level epi tuning / hand-written PTX / cross-warp STS clustering:**
  ptxas owns SASS — byte-identical or noise (CUTLASS_LOOP, FP32_EPILOGUE,
  cvta.shared, NUM_EPI_STAGES, stmatrix variants, SELF_LOAD/STAGGER).
- **K_UNROLL:** non-N_STAGES multiples regress 87–197 µs; u6/u12/u24 tie.
- **Cluster-axis swap:** B200 hard-rejects (1,2,1)/(1,1,2); 2-CTA clusters
  X-axis only (cuBLASLt "2x1/1x2" is logical labeling).
- **KERN_3WARP merge (W4→W5):** +166 µs; W4 empty_wait is hw-sleep, not free
  issue. Structural warp floor = 4.
- **EPI_2WARP + DROP_LEAD_BARSYNC:** −0.27 µs, opt-in only (DROP_LEAD ships a
  cross-warp STS-before-TMA race); not applicable to GELU ops.
- **LDTM_X32 ties STSM** at MMA floor; STSM stays (rank1.sass parity).
  **LDTM_X64 forces NS_EPI 2→1 = +14 µs** — NS_EPI=2 is worth ~14 µs.
- **fc2_w3x post-WIN levers all ±3 µs or regressions:** subpass 8→4, cross-tile
  TMA carry, SWIZZLE_64B, DROP_TRAIL_BARSYNC, WAIT_GROUP_READ, XPF_A/B (macros
  removed), CHET/PMIX/INGH, gflip_cidperm, STAGGER=2 split-mbar (removed), DG
  sweep, native BF16 epi (kept, ±0).
- **Older:** TD=1/5/6/7, COL_LOCK, 4-CTA TMA multicast (deadlock), mbar→SMEM
  polling, L2 hints, dgphase/dgnrot, fc2_ldg, fc2_hybrid, N-batch/phase-offset/
  Group-3 (pre-PACKED_TILES — re-test before citing).
- **Workstealing stripped from fc1_w3/fc2_w3 (2026-06-27):** static-only now
  (TD=0 + TD≥8); provably SASS-identical at production TDs via `nvcc -E`
  pp-diff + cubin byte-match (reuse that gate for any `#ifdef` strip). Static
  always won anyway. `memory/project-w3-workstealing-strip.md`.
- **FC1 FORCE_PREFILL:** deadlocks at K_ITERS=6. NO_PREFILL guard mandatory.
- **FC1 LDG_BIAS:** +2.0 Mcyc (+54%) — L1 bias in the epi hot loop is dead;
  SMEM bias mandatory (only layout-parity use in no-bias builds).
- **FC1 GELU_F32X2 (packed f32x2 GELU):** perfect FADD2/FMUL2/FFMA2 packing
  (4.5 slots/elem < vendor 5.5, zero scalar residue) is +31 kcyc alone and
  +200–340 in combos — epi is not FP-issue-bound; kept opt-in only.
  `memory/project-fc1-gelu-x2-bias-ring.md`.
- **FC1 STSM_STORE (stmatrix epi store):** +45..52 kcyc GELU / TIE
  GEMM_ONLY, incl. vendor's exact STSM.16.MT88.4 — warp-wide gather stalls
  on GELU lane skew. Timing-only probe (our LDTM is lane-per-row → bytes
  permuted, valid=0); do NOT build the 16x256b LDTM rewrite to feed it.
- **fc1_w3x PER_WARP_STORE:** barrier-free crashes (Xid 13 CGA "CTA Not
  Present" — one CTA exits persistent loop while cluster peer issues cluster
  ops); with bar.sync back, +63 µs. Serial tid==0 store was never the FC1
  bottleneck — the 3.11 ms regression is exposed epi COMPUTE (K_ITERS=6 MMA
  shadow too short; same store hides fine at K_ITERS=24).
  `memory/project-fc1-w3x-epilogue-exposed.md`.
- **fc1_w3x WIDE_SUBPASS (64-col subpasses):** wash on GELU production
  (SFU-bound compute masks it), −47 µs bias-only (no home; FC1 always needs
  GELU). Kept opt-in. GELU/bias split: strip 1.353 / bias-only 2.317 / full
  3.125 — our GELU +808 µs is cheaper than cuBLASLt's 894 µs; the deficit vs
  their bias-only is exposed bias/CVT/STS structure. **Real FC1 lever = GELU
  compute throughput (SFU/`MUFU.tanh` scheduling), not epilogue structure.**

## Build and run

```bash
make fc2-w3x && ./fc2-w3x                   # 1.001 ms production bias-only
make fc2-w3x-strip && ./fc2-w3x-strip       # 0.985 floor
make fc2-w3 && ./fc2-w3                     # fused residual (TD=54 default)
make fc1-w3 && ./fc1-w3                     # FC1 production (TD=11+ks=1, ES=3, BIAS_PER_TILE ring)
make fc1-w3x && ./fc1-w3x                   # FC1 clean-sheet (~3.11 ms, WIP)
make fc1-cutlass && ./fc1-cutlass           # FC1 CUTLASS reference
make fc2-cutlass && ./fc2-cutlass           # FC2 CUTLASS reference
make fc2-w3-swizzle-sweep && ./fc2-w3-swizzle-sweep SWEEP=front REPS=200  # basin sweep (cyc)
./tools/probe_cublaslt.sh                   # cuBLASLt rank-1 enumeration
./tools/dim_sweep_w3x.py                    # fc2_w3x N×K grid vs cuBLASLt
./tools/dim_sweep_fc1.py                    # fc1_w3 N×K grid vs cuBLASLt
modal run gpu_interface/cubin_dump.py       # CUPTI nvjet SASS dump (no ncu needed)
bash tools/ncu_fc2_w3x.sh --max --reps 3    # SASS stalls (vast.ai only)
bash tools/ncu_fc2_pipes.sh                 # dodges --set full deadlock

make -B fc2-w3 DFLAGS='-DM_TOTAL=464128 -DN_DIM=1024 -DK_DIM=2048'  # custom dims need -B
# Decomp: -DSTRIP_EPILOGUE / -DGEMM_ONLY; profiling: -DPROFILE_CYCLES|_KI|_TILE|_W5
# Sweeps: tools/sweep_fc2_w3x_{swizzle,nanosleep,dg,prof}.sh; aggregate_prof.py
```

## Remote B200 via Modal (timing only — no ncu; perf counters blocked)

```bash
modal run gpu_interface/runner.py --target fc2-w3x [--dflags "-DPROFILE_CYCLES"]
```

Image `nvidia/cuda:13.2.0-devel-ubuntu24.04` + python 3.14 (codegen) + make;
repo mounted via image-folded `add_local_dir` (data/, logs, .git, third_party
excluded; pre-1.0 `modal.Mount` API does NOT work). `--rebuild` default forces
`make -B` (DFLAGS don't touch mtime). Build+run share one B200 container so
`-lcuda` links the real driver. **Never pipe a Modal stream through
tail/grep/head — block-buffering swallows `@@RESULT`/PASS/Xid; redirect FULL
output to a log (`modal run ... > run.log 2>&1`) and Read the log.**
Aggregation stays local. `ncu --set full` SASS-stall work still needs vast.ai.

## Key files

```
fc2_w3x.cu / fc1_w3x.cu    clean-sheet family (fc2 ACTIVE-best; fc1 WIP, exposed-epi)
fc2_w3.cu / fc1_w3.cu      legacy family — fc2 production residual, fc1 production GELU
swizzle_w3x.cuh            48 swizzle templates (TD=11..99), w3x family
epilogue_ops.cuh           CVT_ADD/CVT_GELU_ADD + gelu_approx + pack_idx_C (shared w3x)
gen/bias_switch_inc_<N>.cuh  codegen via tools/gen_bias_switch.py (Makefile rule)
tile_dispatch.cuh          legacy TD=8..58 for fc1_w3/fc2_w3 (NOT w3x)
kernel_common.cuh, kernel_body.cuh  legacy w3 infra (NOT w3x)
fc2_cutlass.cu / fc1_cutlass.cu     CUTLASS references
bench/fc_problem.cuh       shared cuBLASLt problem def (BF16 bias [N,M])
bench/cublaslt_cubindump.cu + gpu_interface/cubin_dump.py  CUPTI nvjet SASS dump
bench/                     TMA/MMA/stmatrix microbench + cublaslt_introspect
sass/root_dumps/rank1.sass real cuBLASLt algoId=66 kernel dump (opcodes valid)
tools/anova_1way.py        paired ANOVA + AUC + d + η² (canonical analysis)
tools/                     bench/probe/dim_sweep/ncu/sweep/sass_edit/analyze_swizzle
gpu_interface/runner.py     remote B200 build+run
token_count.py             tiktoken budgeting
data/                      benchmark + ncu results
```

Do NOT cross-include between w3 (kernel_common/kernel_body/tile_dispatch) and
w3x (swizzle_w3x/epilogue_ops/gen) families. w3x shared headers: edit once,
both kernels rebuild; real fc1↔fc2 lever surface ~150 lines (header / dims /
NS picker / GELU-vs-BIAS macro at subpass site / K_STAGGER / golden ref).

## SM100a hardware (B200-measured)

- STS.128: 27 cyc | LDS.128: 25 cyc @ILP=1, 3.5 @ILP=7
- TMA load: 419 cyc (L2-warm) | TMA store: 197 cyc
- TMEM load (tcgen05.ld.sync): 2 cyc regardless of width/ILP
- MMA K-iter: 665 cyc (pipelined 525.6)
- STS scaling 10→37 cyc at 8 warps; LDS 4.5→16; FFMA ~free; F2FP flat 2.0

## Key constraints

- sm_100a (B200, 148 SMs), `cta_group::2`, 74 clusters; TMEM 512 cols single
  alloc; SMEM 228 KB/SM
- Inline PTX in the w3 family (no CUTLASS dependency)
- OFF_STAGING 1024-byte aligned for SWIZZLE_128B
- `fence.proxy.async.shared::cta` before TMA store after st.shared
- N_STAGES + PREFILL auto-picked kernel-side from N_DIM + K_ITERS
- BIAS_SMEM=1 default; custom dims require `make -B`

## Benchmarking

- **Cycles, not ms.** clock64 per-CTA `max/N_TIMED` or stream-serialized
  sentinels. Wall lies: boost→base ~8%, thermal ramp, cold L2, power cap.
- **Pass-major randomized blocks**, `@@SAMPLE pass=p variant=v cyc=Y`; never
  batch variants. **Trim** first 33–50% of passes.
- **Paired analysis** (residual = sample − per-pass mean): report AUC, Cohen's
  d, η², mean rank, win%. **No p-values** at large n.
- **Verdict bands:** AUC <0.55 TIE · <0.65 WEAK · <0.75 MOD · <0.85 STRONG ·
  ≥0.85 DECISIVE. η² <0.01 negligible · <0.06 small · <0.14 medium · ≥ large.
  |d| <0.2 trivial · <0.5 small · <0.8 medium · ≥ large. Kendall τ cross-metric:
  low cyc-vs-ms τ ⇒ the ms is clock noise.
- Canonical: `tools/anova_1way.py --metric cyc --paired rep --trim 0.33`.
- **n thresholds (σ_res ~1400 cyc):** n<5000 unreliable sub-σ; MOD ~600 cyc
  needs n≥10978; TIE ~150 cyc needs n≥43910. Two-stage: REPS=2048 filter →
  43910 survivors. Small-n DECISIVE calls routinely demote on rerun.

## Working in this repo

- Names say what, comments say why. Bare `/*`, undecorated lines, `*/` — no
  single-line `/**/`, no multi-line `//`.
- Use `rtk` before every standard bash command (A MUST).
- Token terseness: grep/Explore then Read offset/limit, never whole kernels
  (fc2_w3.cu ~56 K tokens); `nvcc -E` to collapse `#ifdef` (also the
  SASS-identity gate); `cuobjdump -symbols` first, full `-sass` to disk only;
  ncu `--metrics --csv` only; `git show` over re-reading; Explore agent for
  multi-file fan-out. CLAUDE.md/docs/memory are a token budget — brief
  pointers to topic files; `./token_count.py` (Claude ≈1.5× on tables).
- Local bench output ~3 lines — tail is fine; Modal streams NEVER (see Modal).
- Don't narrate tool calls; don't echo file contents; parallelize independent
  calls.
