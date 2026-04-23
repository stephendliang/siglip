## fc2_w3x dgswizzle vs rowmajor — locked-clock ncu (1800 MHz, --clock-control none)

Test: does CUTLASS-style row-major static dispatch (TILE_DISPATCH=13) give us
1.00x DRAM amp AND beat dgswizzle on fc2_w3x?

Build: `make -B fc2-w3x` vs `DFLAGS='-DTILE_DISPATCH=13' make -B fc2-w3x`. Same
PACKED_TILES, same NS=6, same 6-warp persistent kernel. Only the tile order
differs.

### Wall (host bench, real packed FP8 inputs, 3 reps each, locked 1800 MHz)

| dispatch  | wall (ms)  | TFLOPS  |
|-----------|------------|---------|
| dgswizzle | 1.034      | 4237    |
| rowmajor  | 1.037      | 4224    |

**Rowmajor is +3 µs slower**, perfectly consistent across reps.

### NCU under NCU_PROFILE (memset'd inputs, single launch, 23 metrics)

| metric                                | dgsw          | row           | delta        |
|---------------------------------------|---------------|---------------|--------------|
| sm_cycles_elapsed.max                 | 1,904,234     | 1,898,993     | −5,241       |
| sm_cycles_elapsed.avg                 | 1,883,333     | 1,880,676     | −2,657       |
| sm_pipe_tensor_cycles_active.avg.pct  | 97.99 %       | 97.83 %       | −0.16        |
| smsp_inst_executed.sum                | 140.15 M      | 135.13 M      | −5.03 M      |
| smsp_long_scoreboard.ratio            | 6.70          | **7.21**      | **+0.51**    |
| smsp_short_scoreboard.ratio           | 0.27          | 0.24          | −0.03        |
| smsp_barrier.ratio                    | 0.68          | 0.71          | +0.03        |
| smsp_wait.ratio                       | 1.55          | 1.50          | −0.05        |
| dram_bytes_read                       | 2.974 GB      | 2.935 GB      | −39 MB       |
| dram_bytes_write                      | 1.403 GB      | 1.404 GB      | +1 MB        |
| dram_throughput.pct                   | 52.17 %       | 51.79 %       | −0.38        |
| lts_t_sectors                         | 600.2 M       | 640.7 M       | +40.5 M      |
| lts_t_sector_hit_rate                 | 67.85 %       | 68.02 %       | +0.17        |

### DRAM amplification calc

bias-only theoretical read = A(M·K) + B(N·K) + bias = 2.852 GB + 2.36 MB + 1.5 KB ≈ 2.852 GB

| dispatch  | actual read   | amp     |
|-----------|---------------|---------|
| dgswizzle | 2.974 GB      | **1.043×** |
| rowmajor  | 2.935 GB      | **1.029×** |

Rowmajor IS closer to 1.00× amp (1.4 pts better), but neither is AT 1.00×. To
get true 1.00× amp on fc2_w3x we'd need the synchronous-wavefront variants
(ncycle/nsnake/nflat) — which are documented as 50–180 µs SLOWER in CLAUDE.md.

### Verdict

**The premise inverts on fc2_w3x too.** Rowmajor:
- reads 39 MB less DRAM (1.029× vs 1.043× amp — closer to perfect)
- has 5 M fewer instructions (simpler address arithmetic)
- has 0.17 pts higher L2 hit rate
- has 5 K fewer sm_cycles.max in ncu (memset'd inputs)
- ...but has **+0.51 long_scoreboard stalls per issue cycle** (7.21 vs 6.70)

The long_sb increase wins: real-data wall is +3 µs slower (1.037 vs 1.034 ms).
This is the same direction of effect as the FC2 fc2_w3 sweep (rowmajor 1.071 vs
dgswizzle 1.065 = +6 µs); on fc2_w3x the gap shrinks to +3 µs because the
6-warp persistent structure is less sensitive to dispatch (W7 scheduler is
gone, no W2 EpilogueLoad to fight).

dgswizzle's "extra" 39 MB of DRAM reads (1.04× amp) come from clusters
asymmetrically warming L2 — the 8 clusters per group all hit the same A-tiles,
the L2 lines stay hot, but adjacent groups partially refetch when working set
exceeds 96 MB. Those refetches are NOT on the critical path (they overlap
with MMA), while rowmajor's tighter wavefront makes its L2 misses serial.

**Conclusion: don't ship rowmajor.** dgswizzle's 1.04× DRAM amp is the cost
of the L2 staggering that dgswizzle uses to hide misses behind MMA work.
Trading that for 1.029× amp + +0.51 long_sb is a net loss. The "1.00× amp
must be better" intuition is the same trap we documented at length on
2026-04-18 — confirmed once more on fc2_w3x with this clean A/B test.

Files: `dgsw.csv` `row.csv` (raw ncu CSV), `dgsw.stdout` `row.stdout`
(program output, garbage under NCU_PROFILE memset'd inputs).
