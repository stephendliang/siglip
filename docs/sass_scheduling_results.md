# SASS Scheduling Analysis — FC2 Epilogue vs CUTLASS

CP-SAT optimal scheduler results for FC2 epilogue variants and CUTLASS.
All analysis local (no GPU). Scheduler: `tools/sass_edit.py schedule`.

## Key finding

**ptxas generates ~5× worse schedules than optimal on epilogue compute blocks — uniformly across both our kernel and CUTLASS.** The scheduler consistently finds 75-83% reductions in stall cycles. CUTLASS gets away with it because StagesC hides the epilogue behind the K-loop; we can't (yet).

## Methodology

The CP-SAT scheduler operates on **basic blocks** — maximal straight-line instruction sequences between branches. It models:
- Per-pipe throughput constraints (STS=32 cyc, BF16=2 cyc, ALU=2 cyc, LDS=4 cyc)
- Register data dependencies (WAR, RAW, WAW)
- Barrier dependencies (WARPSYNC, etc.)

It does NOT model: memory latency (LDTM, TMA), cross-warp contention, mbarrier wait times. These are runtime properties invisible to static analysis.

Cycle counts = issue cycle of last instruction + 1. "ptxas cyc" = sum(stalls) + num_insns. "optimal cyc" = best achievable via reorder + restall.

## Instruction count comparison (current code)

| Config | Insns | Regs | vs CUTLASS |
|--------|------:|-----:|:-----------|
| CUTLASS (4-stage, N=256, cta_group::2, bias+res) | 2688 | ~170 | reference |
| Baseline (NS5, forceinline) | 2768 | 164 | +80 (+3.0%) |
| Noinline (NS5) | 2616 | 168 | -72 (-2.7%) |
| StagesC (NS4, noinline) | 2696 | 172 | +8 (+0.3%) |
| **StagesC+PC** (NS4, noinline, precombine) | **2440** | **144** | **-248 (-9.2%)** |
| Branchless (NS4, StagesC, noinline, full unroll) | 2832 | 157 | +144 (+5.4%) |

StagesC+PC has 9.2% fewer instructions than CUTLASS at 26 fewer registers. The gap was never instruction count — it was always scheduling quality + epilogue architecture.

## Epilogue scheduling: our kernel

### StagesC+PC (non-branchless, PHASE1_UNROLL=1)

4 inner loop iterations per pass, each producing a ~67-insn basic block. Only 2 passes (128 cols each) because PRE_COMBINE + BIAS_BF16 halves the BF16/STS work.

| Block | Insns | ptxas cyc | optimal cyc | Delta |
|-------|------:|----------:|------------:|------:|
| Pass 0 body (0x7bd0-0x7ff0) | 67 | 614 | 120 | -80.5% |
| Pass 1 body (0x8340-0x87d0) | 74 | 667 | 160 | -76.0% |

Each block covers ONE 32-col chunk (4 LDS + 16 F2FP + 16 HADD2/HFMA2 + 4 STS.128).

Per-pass total (4 chunks): ~2576 ptxas → ~520 optimal. Plus ~120 cyc branch/mbar overhead between chunks.
**Per-pass effective: ~2696 ptxas → ~671 optimal.**

### Noinline (NS5, no StagesC)

Same epilogue structure but with TMA residual loads blocking inside each pass.

| Block | Insns | ptxas cyc | optimal cyc | Delta |
|-------|------:|----------:|------------:|------:|
| Group 0 (0x7570-0x7b10) | 91 | 644 | 130 | -79.8% |
| Group 2 (0x8020-0x8640) | 99 | 767 | 141 | -81.6% |

Per-pass total (4 groups): ~2820 ptxas → ~540 optimal. Plus ~1600 cyc of blocking TMA residual waits.
**Per-pass effective: ~4420 ptxas → ~2140 optimal** (TMA waits dominate).

### Branchless (full unroll + predicated TMA + deferred stores)

Eliminated: loop back-edge, IS1 interleave branches, `if (lane==0)` BSSY/BSYNC.
Result: **one 317-insn basic block per pass** covering all 4 TMEM chunks + TMA stores.

| Block | Insns | ptxas cyc | optimal cyc | Delta |
|-------|------:|----------:|------------:|------:|
| Pass 0 (0x7bf0-0x8fb0) | 317 | 2011 | 935 | -53.5% |
| Pass 1 (0x90f0-0xa2d0) | 287 | 1730 | 902 | -47.8% |

Fewer internal branches means ptxas produces better code out of the box (2011 vs 2696 per-pass). But the optimal floor is higher (935 vs 671) because 16 STS.128 in one block bottleneck on STS throughput (16 × 32 = 512 cyc minimum). The non-branchless version exploits TMEM_WAIT gaps between blocks as "free" STS spacing.

**Trade-off**: branchless is better unpatched (+25%), non-branchless is better SASS-patched (-28%).

## Epilogue scheduling: CUTLASS

The winning CUTLASS FC2 kernel (4-stage, 256×128×64 per-CTA, cta_group::2, LinCombPerColBiasEltAct, 2688 insns, 1.225ms) has the same 5× scheduling gap.

| Block | Insns | ptxas cyc | optimal cyc | Delta |
|-------|------:|----------:|------------:|------:|
| Epilogue body 1 (0x6820-0x6fd0) | 124 | 1204 | 204 | -83.1% |
| Epilogue body 2 (0x75f0-0x7da0) | 124 | 1222 | 212 | -82.7% |
| Drain epilogue (0x83a0-0x8b90) | 128 | 1166 | 225 | -80.7% |

CUTLASS's epilogue is simpler (F2FP + STS, no HADD2 — bias is fused differently) but ptxas still generates 5× suboptimal stall counts. **This is a ptxas limitation, not a CUTLASS or our-kernel limitation.**

CUTLASS achieves 99.6% of the K-loop throughput bound despite this because StagesC pipelining hides the entire epilogue behind the next tile's K-loop. The bad scheduling only affects the last tile and drain path — <1% of total runtime for 147 tiles.

## Why ptxas schedules badly

ptxas consistently spaces STS.128 instructions 0-2 cycles apart instead of the 27-32 cycle throughput interval. The TMA bench (section P, `data/tma2.txt`) proved STS.128 throughput is 32 cycles, and stuffing BF16 ops into the STS shadow region overflows at >4 ops (+55% at 8, +161% at 15).

ptxas appears to:
1. Cluster all F2FP conversions together (create all BF16 values)
2. Cluster all HADD2/HFMA2 additions together (apply bias+residual)
3. Cluster all STS together (store to staging SMEM)

The optimal schedule interleaves them: F2FP → HADD2 → STS → F2FP → HADD2 → STS, maintaining exactly 32-cycle STS spacing with ≤4 BF16 ops per STS shadow window.

## What blocks can be scheduled

Only straight-line basic blocks. Branch boundaries come from:

1. **`if (lane == 0)` for TMA ops** → BSSY/BSYNC + R2UR (biggest source, 12+ per epilogue)
2. **TMEM_WAIT()** → SYNCS.PHASECHK.TRYWAIT + BRA (unavoidable, one per 32-col chunk)
3. **Inner loop back-edge** → BRA (eliminated by full unroll)
4. **IS1 TMA store interleaving** → conditional BRA (eliminated by deferred stores)

The branchless rewrite eliminates #1, #3, #4 via predicated TMA helpers (`pred_tma_store_2d`, `pred_commit_group`, `pred_mbar_arrive`), full `#pragma unroll`, and deferred stores. Only #2 remains — ptxas always emits TRYWAIT+BRA for TMEM waits regardless of source.

## Recipes generated

All recipes are in `sass/recipes/`. They are **address-dependent** — if the cubin changes (different compile flags, code changes), new recipes must be generated.

| Recipe | Target | Insns | ptxas → optimal |
|--------|--------|------:|:---------------|
| `stages_c_pc_pass0.recipe` | StagesC+PC pass 0 | 66 | 614 → 120 cyc |
| `stages_c_pc_pass1.recipe` | StagesC+PC pass 1 | 73 | 667 → 160 cyc |
| `noinline_g0.recipe` | Noinline group 0 | 91 | 644 → 130 cyc |
| `noinline_g2.recipe` | Noinline group 2 | 99 | 767 → 141 cyc |

B200 application via `fatbin-patch`:
```bash
nvcc ... -DSTAGES_C=2 -DEPI_NOINLINE=1 -DPRE_COMBINE=1 fc2.cu -o fc2_sc_pc
cuobjdump --dump-sass fc2_sc_pc > sass/fc2_sc_pc.sass
python3 tools/sass_edit.py fatbin-patch fc2_sc_pc --sass sass/fc2_sc_pc.sass \
    --script sass/recipes/stages_c_pc_pass0.recipe -o fc2_tmp
python3 tools/sass_edit.py fatbin-patch fc2_tmp --sass sass/fc2_sc_pc.sass \
    --script sass/recipes/stages_c_pc_pass1.recipe -o fc2_sc_pc_opt
```

Note: recipes must be regenerated on B200 if the SASS layout differs (likely — different nvcc version, different register allocation).

## Implications

1. **StagesC is the architectural fix.** If it works, epilogue scheduling becomes irrelevant (hidden behind K-loop), same as CUTLASS. The scheduler is a diagnostic tool, not the primary optimization.

2. **SASS patching is a fallback.** If StagesC doesn't fully hide the epilogue (last tile, drain, W0 starvation), SASS patching recovers 75-83% of the scheduling gap. Worth 200-800 cycles per tile.

3. **ptxas scheduling is uniformly 5× suboptimal on epilogue blocks.** This applies to CUTLASS too. A SASS-patched CUTLASS would be marginally faster (<1% wall time because StagesC already hides the epilogue).

4. **The branchless rewrite is better without patching, worse with it.** Ship branchless if no SASS patching; ship non-branchless + recipe if SASS patching is part of the build.
