# Top 5 Grid Search Configs — ncu Profiling Analysis

Data: `top5_profile.csv` (21 ncu metrics + 2 derived, per config).
Source: `sweep_results_run4.csv` run 4 (145/145 valid, post fence.proxy.async fix).

## The configs

All share: N_STAGES=4, NUM_EPI_WARPS=4, PHASE1_UNROLL=2, SNAKE_ORDER=1, CVT_ADD_FUSED=1.

| Rank | Strategy | MBAR | TLW | Stagger | Regs | ms    | TFLOPS |
|------|----------|------|-----|---------|------|-------|--------|
| 1    | 1        | 1    | 64  | 80      | 239  | 0.522 | 2096   |
| 2    | 2        | 1    | 32  | 100     | 211  | 0.522 | 2098   |
| 3    | 1        | 1    | 64  | 160     | 239  | 0.523 | 2093   |
| 4    | 2        | 0    | 32  | 120     | 209  | 0.523 | 2092   |
| 5    | 2        | 1    | 32  | 60      | 211  | 0.523 | 2092   |

Two archetypes: **A** = Strategy 1 + TMEM_LOAD_WIDTH=64 (ranks 1, 3); **B** = Strategy 2 + TMEM_LOAD_WIDTH=32 (ranks 2, 4, 5).

## Key profiling results

### 1. TC utilization — the discriminating metric

| Rank | TC active % | Tensor pipe % | Cycles elapsed | Cycles active |
|------|------------|---------------|----------------|---------------|
| 1    | **60.1%**  | **57.4%**     | **786K**       | 746K          |
| 3    | 58.7%      | 56.4%         | 801K           | 760K          |
| 5    | 58.0%      | 55.5%         | 813K           | 770K          |
| 2    | 57.3%      | 54.9%         | 822K           | 776K          |
| 4    | **56.8%**  | **54.6%**     | **828K**       | 786K          |

Rank 1 has 3.3% more TC utilization and 42K fewer cycles than rank 4.
The TC utilization gap maps directly to the wall time difference.

### 2. Instruction counts — why Archetype A is faster

| Pipe | Archetype A (S1+TLW64) | Archetype B (S2+TLW32) | Difference |
|------|------------------------|------------------------|------------|
| TMEM | **348K**               | 696K                   | **-50%** (half the tcgen05.ld instructions) |
| FMA  | **18.8M**              | 21.0M                  | **-10%** (fewer loop iterations) |
| ALU  | **32.5-33.5M**         | 35.9-36.3M             | **-8%** (less index math/loop overhead) |
| LSU  | 6.56M                  | **6.21M**              | +6% (wider stores per iteration) |
| TC   | 337K                   | 337K                   | identical (same MMA count) |

TMEM_LOAD_WIDTH=64 loads 64 cols per `tcgen05.ld` instead of 32, halving TMEM instructions.
The x64 loop body is larger but runs half as many iterations, eliminating loop overhead.
TC and Tensor instruction counts are identical across all 5 configs — same GEMM workload.

### 3. Stall breakdown (cycle-weighted, avg per warp)

| Stall                | Rank 1   | Rank 2   | Rank 3   | Rank 4   | Rank 5   |
|----------------------|----------|----------|----------|----------|----------|
| selected (productive)| 77,962   | 87,789   | 79,452   | 87,226   | 87,084   |
| long_scoreboard      | **314K** | 299K     | **308K** | 308K     | 300K     |
| wait (TMA pipeline)  | **152K** | 175K     | 161K     | **181K** | 170K     |
| sleeping             | 64K      | 70K      | 67K      | **73K**  | 69K      |
| barrier              | 63K      | 66K      | 65K      | 67K      | 65K      |
| short_scoreboard     | **21K**  | 16K      | **21K**  | **10K**  | 16K      |
| math_throttle        | 2.0K     | 2.1K     | 1.9K     | 2.1K     | 2.0K     |

Key patterns:

- **long_scoreboard** (TMEM readback latency): Archetype A has ~10-15K *more* stall cycles.
  Wider 64-col TMEM loads have longer dependency chains, so each `tcgen05.ld` stalls more.
  But fewer total loads means the wall-clock epilogue is still shorter.

- **wait** (TMA pipeline): Archetype A has **13-16% less** wait. Strategy 1 fires one TMA store
  per region (inline, immediately after each 64-col region completes). Strategy 2 batches in
  pairs (2 stores after every 2nd region). Per-region stores let the warp return to compute
  faster, reducing pipeline backpressure.

- **short_scoreboard** (SMEM dependency): Archetype A has ~2x more (21K vs 10-16K). The wider
  TMEM loads create more st.shared → data-ready chains within each loop iteration.

- **sleeping**: Rank 4 (MBAR_EARLY=0) has the most sleeping time (73K vs 64-70K). Without
  early mbarrier signaling, W1 waits longer for TMEM release, and epilogue warps park longer
  between tiles.

- **selected** (productive instructions): Archetype B shows ~10K more "selected" cycles despite
  being slower. This reflects the *extra* instructions (more TMEM, FMA, ALU) that archetype B
  must execute — more work per tile, not more useful progress.

### 4. SMEM bank conflicts

| Metric                | All 5 configs |
|-----------------------|---------------|
| SMEM load conflicts   | **0**         |
| SMEM store conflicts  | ~4.1M (26-27% of store wavefronts) |
| SMEM load wavefronts  | 1,850 (TMA pipeline loads only) |
| SMEM store wavefronts | ~15.2M |

SMEM load bank conflicts are **zero** across all configs. The original baseline (docs/make_better.md)
had 5.37M load conflicts / 32.5% rate. The SWIZZLE_128B + TMA-store architecture eliminated all
SMEM ld.shared conflicts — the epilogue uses `cp.async.bulk` to read staging SMEM directly via the
TMA unit, bypassing the LSU pipe entirely. The 1,850 load wavefronts are W0 TMA pipeline loads.

Store conflicts (~27%) come from Phase 1 `st.shared` writes to staging. All 32 lanes write to the
same SWIZZLE_128B region, and the XOR-based swizzle doesn't fully eliminate bank conflicts for
the CVT_ADD_STS pattern. This is in the epilogue shadow and does not affect wall time at current
equilibrium.

### 5. Effect of MBAR_EARLY

Rank 4 is the only config with MBAR_EARLY=0. Compared to rank 2 (same archetype, MBAR_EARLY=1):

| Metric           | Rank 2 (MBAR=1) | Rank 4 (MBAR=0) | Delta    |
|------------------|-----------------|-----------------|----------|
| TC active %      | 57.3%           | 56.8%           | -0.5%    |
| wait stalls      | 175K            | 181K            | +6K      |
| sleeping         | 70K             | 73K             | +3K      |
| cycles elapsed   | 822K            | 828K            | +6K      |
| short_scoreboard | 16K             | 10K             | **-6K**  |

MBAR_EARLY=1 signals "TMEM is free" at the end of Phase 1 (after `tcgen05.ld` copies data to
registers), before Phase 2 TMA stores. This lets W1 start the next K-loop sooner. The tradeoff:
slightly more short_scoreboard stalls (16K vs 10K) because Phase 2 stores now overlap with W1's
TMEM writes, creating more SMEM pressure. Net effect: +0.5% TC utilization, -6K elapsed cycles.

### 6. Effect of STAGGER_CYCLES

Within Archetype A (S1+TLW64):

| Stagger | Rank | Cycles elapsed | long_scoreboard | wait   |
|---------|------|----------------|-----------------|--------|
| 80      | 1    | **786K**       | 315K            | 152K   |
| 160     | 3    | 801K           | 308K            | 161K   |

Lower stagger (80) is better for Archetype A: warps start closer together. The fast S1 epilogue
(fewer instructions) handles TMEM contention well, so spreading warps further (160) just adds
dead time. The 7K fewer long_scoreboard stalls at STG=160 don't compensate for the 15K extra
elapsed cycles from the stagger delay itself.

Within Archetype B (S2+TLW32), the stagger sweet spot is broader (60-100 are within noise)
because the longer S2 epilogue naturally spaces warps more. STG=120 with MBAR=0 (rank 4) is
slightly worse due to the MBAR effect, not the stagger itself.

## Summary

The top configs win through two orthogonal mechanisms:

1. **TMEM_LOAD_WIDTH=64** cuts epilogue instruction count by 10-50% across pipes (half the
   tcgen05.ld, fewer loop iterations → less FMA/ALU). This frees TC time despite using 30
   more registers and creating more SMEM dependency stalls. The instruction savings outweigh
   the register pressure cost.

2. **MBAR_EARLY=1** reduces W1 idle time by signaling TMEM-free at end of Phase 1 instead of
   Phase 2. Net +0.5% TC utilization.

3. **Strategy 1 vs 2** is a secondary effect: per-region inline TMA stores (S1) have 13-16%
   less TMA pipeline wait than batched pairs (S2), but S2 has fewer short_scoreboard stalls.

The configs are within ~0.001 ms of each other because the kernel is in producer-consumer
equilibrium — epilogue savings are absorbed by the balanced pipeline. To break out of this
plateau, the K-loop itself (currently ~4,059 cycles) must be shortened, not just the epilogue.

## Reproducing

```bash
# Profile any config:
nvcc -gencode arch=compute_100a,code=sm_100a -O3 <DFLAGS> megakernel.cu -o tmp -lcurand -lcuda
ncu --metrics sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed,... -k patch_embed_gemm -c 1 ./tmp

# Parse the CSV:
python3 -c "import csv; [print(r) for r in csv.DictReader(open('top5_profile.csv'))]"
```

Full dflags for each config are in the `dflags` column of `sweep_results_run4.csv`.
