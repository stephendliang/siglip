# clock64() Timing Analysis — Current Kernel State

**Kernel version:** Post-F19 (4 epilogue warps, double-buffered SMEM staging, early mbar signal)
**Baseline:** 0.542 ms / 2020 TFLOPS, 223 regs, 0 spills
**Timing build:** 0.535 ms / 2045 TFLOPS, 254 regs, 0 spills (16B stack)

## Setup

**W1 timestamps** (CTA0, lane 0) at 4 points per tile:
- `t_tile_start`: After previous tile's mainloop_mbar commit
- `t_after_epi`: After epilogue_mbar wait (W1 has TMEM buffer)
- `t_after_tma0`: After first TMA mbar wait (ki=0 data ready)
- `t_kloop_end`: After last MMA commit + mainloop_mbar commit

**W3 timestamps** (ew=1, CTA0, lane 0) at 4 points per tile:
- `epi_t_before_ml`: Before mainloop_mbar wait
- `epi_t0`: After mainloop_mbar wait (epilogue work starts)
- `epi_t1`: After Phase 1A + Phase 1B/2A interleaved (before syncwarp + early mbar arrive)
- `epi_t2`: After Phase 2B completes

## Results (5-run average, last run of checksum iteration)

```
Kernel: 0.535 ms / 2045 TFLOPS (baseline: 0.542 ms / 2020 TFLOPS)
Checksum: 1769472.0

=== W1 PER-TILE ===
  Epilogue mbar wait:     1,052 cycles /  501 ns   (17.3%)
  TMA stage-0 wait:         856 cycles /  408 ns   (14.1%)
  K-loop (6 ki x 4 MMA):  4,155 cycles / 1979 ns   (68.5%)
  Total tile:              6,063 cycles / 2887 ns
  Overhead (epi+tma0):     1,908 cycles /  909 ns   (31.5%)

  K-loop range:  min=2,425  max=11,100  (4.6x spread)
  Total range:   min=3,120  max=13,400  (4.3x spread)

=== EPILOGUE WARP (W3/ew=1) PER-TILE ===
  Mainloop mbar wait:      1,123 cycles /  535 ns   (18.2%)
  Phase 1 (TMEM->SMEM):    4,140 cycles / 1971 ns   (67.2%)
  Phase 2B (SMEM->global):   899 cycles /  428 ns   (14.6%)
  Total (wait+work):       6,162 cycles / 2934 ns
  Work only (P1+P2):       5,039 cycles / 2400 ns

  Mainloop wait range: min=329  max=10,000 (30x spread)
  Phase 1 range:       min=2,960  max=8,300 (2.8x spread)
  Phase 2 range:       min=470  max=1,300  (2.8x spread)
```

## System Equilibrium

The kernel is in a balanced producer-consumer steady state. W1 and the epilogue warps wait for each other in roughly equal measure:

```
W1 cycle:      epi_wait(1,052) + TMA0(856) + K-loop(4,155) = 6,063
W3 cycle:      ml_wait(1,123)  + Phase1(4,140) + Phase2B(899) = 6,162
                                                     (close match — same throughput)
```

### Timeline (steady state, one tile period)

```
W1:  ████████░░░░░░░░░░░░░░░░░░░░░░████████████████████████████████████████████████████████████████████
     epi_wait(17%)    TMA0(14%)                    K-loop (69%)
     ←─ WAITING ─→                           ←── PRODUCTIVE ──→
                                                               ↓ commit mainloop_mbar

W3:  ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░██████████████░░░░░░░░░░░░░░░░
     ml_wait(18%)           Phase 1: TMEM readback (67%)      ↑mbar      Phase 2B (15%)
     ←─ WAITING ─→         ←── TMEM + combined add + CVT → SMEM ──→    ←─ SMEM → global ─→
                                                                arrive epi_mbar
                                                                (early — before Phase 2B)
```

### Why both warps wait ~1,100 cycles

The double-buffered TMEM pipeline creates a 2-tile lag:
- W1 commits mainloop_mbar[buf] at end of tile N
- W3 reads tmem[buf] during tile N+1
- W3 arrives epilogue_mbar[buf] during tile N+1 (after Phase 1, before Phase 2B)
- W1 needs epilogue_mbar[buf] at start of tile N+2

The "effective epilogue time" (what W1 cares about) = ml_wait + Phase 1 + syncwarp ≈ 1,123 + 4,140 + 20 ≈ 5,283 cycles. This exceeds W1's productive time (856 + 4,155 = 5,011) by ~272 cycles. With double-buffer amplification, this small ~272-cycle imbalance creates the ~1,052-cycle W1 wait.

Additionally, W1 waits on epilogue_mbar which requires ALL 4 warps to arrive. We measure W3, but the slowest warp determines the mbar completion. Inter-warp TMEM contention likely makes some warps slower than W3.

## Key Findings

### 1. Epilogue still the bottleneck (67% of epilogue time is Phase 1 TMEM readback)

F18 (double-buffered SMEM) and F19 (early mbar signal) reduced W1's epi_wait from 2,466 → 1,052 cycles (57% reduction). But Phase 1 TMEM readback still dominates at 4,140 cycles per tile — 81% of the epilogue's work time.

### 2. TMA stage-0 wait is NOT negligible

TMA0_wait = 856 cycles (14% of tile), up from 292 cycles in the 5-warp baseline. This is not a scheduling issue — F20 (next-tile TMA prefetch) showed ~0% improvement. It's DRAM bandwidth: A matrix loads need ~16KB from DRAM per tile, and the 4-stage pipeline can't always hide the ~1,940-cycle DRAM latency.

### 3. K-loop jitter is significant

The 4.6x spread in K-loop time (min=2,425, max=11,100) reflects DRAM latency variation. The max K-loop (11,100) is 1.8x the average (6,063 total tile). Individual tiles occasionally suffer data starvation.

### 4. Phase 2B is small and overlapped

Phase 2B (SMEM→global stores) takes only 899 cycles (15% of epilogue cycle). It runs AFTER the early mbar signal, so it overlaps with W1's next K-loop. This is free — F19's early mbar successfully hides Phase 2B.

### What SourceCounters told us vs what clock64 reveals

| Finding | SourceCounters (stale, 5-warp) | clock64 (current, 4-warp) |
|---------|-------------------------------|---------------------------|
| W1 stalled on tma_mbar? | No (0 stalls) | 856 cycles avg (14%) |
| W1 stalled on epilogue_mbar? | No (0 stalls)* | 1,052 cycles (17%) |
| W1 stalled on anything? | 0.026% | 31.5% of tile is waiting |

*SourceCounters paradox: mbar_wait spin-loop uses NANOSLEEP. SourceCounters attributes stalls to NANOSLEEP (as "sleeping"), not to the try_wait instruction. clock64 correctly captures total wait time regardless of attribution.

## Impact Analysis

### Eliminating epi_wait entirely (perfect overlap)

- Tile time: 856 + 4,155 = **5,011 cycles** (vs 6,063)
- Speedup: 6,063 / 5,011 = **1.21x**
- Projected TFLOPS: 2,020 x 1.21 = **2,444 TFLOPS**

### Also eliminating TMA0_wait

- Tile time: 4,155 cycles
- Speedup: 6,063 / 4,155 = **1.46x**
- Projected TFLOPS: 2,020 x 1.46 = **2,949 TFLOPS** (≈ cuBLAS's 3,001)

### Current ceiling analysis

cuBLAS achieves 3,001 TFLOPS on this shape. Our gap: 3,001 / 2,020 = 1.49x.
Eliminating ALL overhead (epi_wait + TMA0_wait) would give 1.46x — almost closes the gap.
The remaining 0.03x is K-loop efficiency (W1 instruction overhead between MMAs).

## Optimization Paths (current priority)

### 1. Faster Phase 1 TMEM readback — PRIMARY TARGET

Phase 1 = 4,140 cycles (67% of epilogue cycle). TMEM readback (`tcgen05.ld.x32`) dominates.

Options tried and rejected:
- F17: Direct global stores (skip SMEM staging) — 19% slower Phase 1 due to L1 contention
- F19: TMA bulk stores for Phase 2B — 3x slower per-instruction than manual stores
- F20: Next-tile TMA prefetch — ~0% (DRAM bandwidth, not scheduling)

Remaining options:
- **Reduce TMEM contention**: 4 warps reading TMEM simultaneously. If Phase 1 scales sub-linearly with warps, 3 warps might be faster per-warp (each doing 2 row_groups). Trade-off: more work per warp vs less contention.
- **Deeper Phase 1 pipelining**: Currently prefetches chunk N+1 while processing chunk N. Could try x16 loads for finer interleaving.

### 2. Larger TM (256) — amortizes overhead

With TM=256 (each CTA processes 256 rows): 1,813 M-tiles instead of 3,626. Halves tile count → halves per-tile overhead. K-loop stays the same (same K). But needs 2x more TMEM cols (1024 — exceeds 512-col limit) or creative sub-tiling.

### 3. Accept current performance

At 2,020 TFLOPS for a FUSED kernel (GEMM + bias + pos_embed), we're already 35% faster than the cuBLAS end-to-end path (0.835 ms). The remaining gap to cuBLAS pure GEMM (3,001 TFLOPS) is the cost of fusion — the epilogue does real work (BF16 add + CVT) that cuBLAS doesn't.
