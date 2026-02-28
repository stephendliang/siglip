# clock64() Timing Analysis — W1 Per-Tile Breakdown

## Setup

Added `clock64()` timestamps to W1 (CTA0, lane 0) at 4 points per tile:
- `t_tile_start`: After previous tile's mainloop_mbar commit (before loop increment)
- `t_after_epi`: After epilogue_mbar wait (W1 has TMEM buffer)
- `t_after_tma0`: After first TMA mbar wait (ki=0 data ready)
- `t_kloop_end`: After last MMA commit + mainloop_mbar commit

Instrumentation: 237 regs (vs 223 baseline), 0 spills, no performance impact (0.576 ms vs 0.579 ms baseline).

## Results

```
Kernel: 0.576 ms / 1902 TFLOPS (baseline: 0.579 ms / 1892 TFLOPS)
Checksum: 1769472.0 ✓

Per-tile averages (10,878 tiles across 74 clusters):
  Epilogue mbar wait:     2,466 cycles / 1,175 ns   (36.2%)
  TMA stage-0 wait:         292 cycles /   139 ns   ( 4.3%)
  K-loop (6 ki × 4 MMA):  4,046 cycles / 1,927 ns  (59.5%)
  Total tile:              6,805 cycles / 3,241 ns
  Overhead (epi+tma0):     2,759 cycles / 1,314 ns  (40.5% of tile)

K-loop range:  min=2,343  max=9,464  (4.0× spread)
Total range:   min=3,114  max=15,375 (4.9× spread)
```

## Critical Finding: EPILOGUE IS THE BOTTLENECK

**The epilogue mbar wait is 36% of W1's tile time.** This directly contradicts the earlier F10 conclusion that "the kernel is K-loop-bound." The epilogue takes longer than one tile's worth of time to drain, causing W1 to stall waiting for TMEM buffer availability.

### Timing decomposition

```
┌──────────────────────────────────────────────────────────────┐
│                    W1 Tile Timeline                          │
│                                                              │
│  ███████████████         ██         ████████████████████████  │
│  epi_mbar wait          TMA0       K-loop (6ki × 4mma)      │
│     36.2%               4.3%            59.5%                │
│                                                              │
│  ├── OVERHEAD (40.5%) ──┤├──── PRODUCTIVE (59.5%) ─────────┤ │
└──────────────────────────────────────────────────────────────┘
```

### Why the epilogue is slow

With double-buffered TMEM, the epilogue for tile N has exactly one tile time (tile N+1) to complete before W1 needs tmem[buf] again for tile N+2:

```
Tile N:    W1 writes tmem[0] → signals mainloop_mbar[0]
Tile N+1:  Epilogue reads tmem[0]. W1 writes tmem[1].
Tile N+2:  W1 needs tmem[0]. Waits on epilogue_mbar[0].
           Wait = max(0, epilogue_time - tile_N+1_time)
```

Measured: epilogue_mbar wait = 2,466 cycles
Therefore: epilogue_time ≈ tile_time + 2,466 ≈ 6,805 + 2,466 ≈ **9,271 cycles**

The epilogue takes **2.3× the K-loop time** (9,271 vs 4,046). It cannot hide in the K-loop shadow. This is because:
- 5 warps do TMEM readback (tcgen05.ld) → 390K cycles/warp of long_scoreboard stalls
- Each warp processes 4-8 nc iterations with: TMEM load → BF16 add → CVT → SMEM store → syncwarp → Phase 2 coalesced stores
- TMEM readback latency dominates (long_scoreboard at 3.3× productive time in average_warp_latency)

### Impact analysis

If we eliminated the epilogue mbar wait entirely (perfect overlap):
- Tile time: 4,046 + 292 = **4,338 cycles** (vs 6,805)
- Speedup: 6,805 / 4,338 = **1.57×**
- Projected TFLOPS: 1,900 × 1.57 = **2,983 TFLOPS** (≈ cuBLAS's 3,001)

Even halving the epilogue mbar wait would give:
- Tile time: 4,338 + 1,233 = **5,571 cycles**
- Speedup: 6,805 / 5,571 = **1.22×**
- Projected TFLOPS: **2,318 TFLOPS**

### K-loop variability

The 4.0× spread in K-loop time (min=2,343, max=9,464) is significant:
- **min=2,343**: TC pipeline perfectly primed, all TMA data ready, zero inter-K-iter waits
- **max=9,464**: Likely DRAM latency spikes on TMA loads, possibly first tile of cluster
- The max K-loop (9,464) exceeds the average total tile time (6,805) — some tiles are dominated by K-loop
- This suggests occasional data starvation on individual tiles, even though the AVERAGE tma0 wait is only 292 cycles

### What SourceCounters told us vs what clock64 reveals

| Finding | SourceCounters | clock64 |
|---------|---------------|---------|
| W1 stalled on tma_mbar? | No (0 stalls) | Correct — only 292 cycles avg |
| W1 stalled on epilogue_mbar? | No (0 stalls)* | **YES — 2,466 cycles (36%)** |
| W1 stalled on anything? | 1,329 total (0.026%) | 40% of tile is overhead |

*The SourceCounters paradox: W1's epilogue_mbar SYNCS.PHASECHK shows 0 stalls because the mbar_wait is a spin-loop with NANOSLEEP. The warp "sleeps" (which SourceCounters counts as sleep, not as a stall at the SYNCS instruction). The stall attribution goes to the NANOSLEEP, not the TRYWAIT. clock64 correctly captures the total wait time regardless of how it's attributed.

### SourceCounters found W1 has 475 instructions/tile, only 24 are MMA (5.1%)

This remains true but is NOT the bottleneck explanation. The 437 non-MMA instructions take ~437 cycles on W1's SMSP (if no competition). That's only 6.4% of the 6,805 cycle tile time. The real issue is the 2,466 cycle epilogue wait.

## Optimization Paths (re-evaluated)

### 1. Faster epilogue — HIGHEST PRIORITY

The epilogue takes ~9,271 cycles (2.3× K-loop). Reducing it to ≤ 6,805 cycles (1 tile time) would eliminate the mbar wait entirely.

Options:
- **Reduce TMEM readback latency**: tcgen05.ld.x32 takes ~390K cycles per warp in long_scoreboard. If we could overlap TMEM reads better (prefetch next chunk while processing current), or use x16 loads with better interleaving.
- **Fewer epilogue warps doing more each**: Counter-intuitive — F3 showed more warps regress (TMEM contention). But 4 warps instead of 5 might reduce contention while keeping enough parallelism.
- **Reduce Phase 2 work**: Phase 2 (SMEM → global stores) is ~3.5M instructions. TMA stores could eliminate these entirely but have staging reuse problems (Proposal 6).
- **Eliminate __syncwarp**: The syncwarp between Phase 1 and Phase 2 serializes within each warp. If we could overlap Phase 1 of chunk N+1 with Phase 2 of chunk N (double-buffered within-warp), the epilogue could pipeline better.

### 2. Larger TK (256) — reduces K-loop time, may expose epilogue MORE

With TK=256: 3 K-iterations instead of 6. K-loop time roughly halves (~2,023 cycles). But if the epilogue doesn't speed up, the mbar wait INCREASES:
- New epi_wait ≈ 9,271 - (2,023 + 146) ≈ 7,102 cycles
- Total tile: 2,023 + 146 + 7,102 = 9,271 cycles → SAME total because the epilogue is the bottleneck regardless
- This would help ONLY if combined with epilogue improvements

### 3. TMA multicast for B — reduces L2/DRAM pressure

Halves B matrix bandwidth. May indirectly speed up both K-loop and epilogue (less L2 contention for combined loads and Phase 2 stores). Worth trying but unlikely to be transformative alone.

## Next Steps

1. **Profile the epilogue in detail**: Which phase (Phase 1 TMEM readback vs Phase 2 SMEM→global) dominates? Add clock64 to epilogue warps at syncwarp boundary.
2. **Try 4 epilogue warps** (`NUM_EPI_WARPS=4`): Reduces TMEM contention. Each warp does more work but with less inter-warp interference.
3. **Try 3 epilogue warps**: Even less contention, but each warp processes 128 cols (no column splitting needed).
4. **Investigate within-warp pipelining**: Can Phase 1 and Phase 2 overlap using double-buffered SMEM staging?
