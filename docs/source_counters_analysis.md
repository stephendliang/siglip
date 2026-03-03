# SourceCounters Analysis — W1 Stall Isolation

**Status:** Collected on the 5-warp baseline (0.579 ms). Kernel has since changed to 4 warps (F16), double-buffered SMEM (F18), early mbar signal (F19). W1's K-loop is unchanged — the SourceCounters findings below remain valid for W1. Epilogue warp data (W2-W6 → now W2-W5) may differ due to reduced contention and different Phase 1B/2A overlap pattern.

## TL;DR

**W1 executes 475 instructions per tile, only 24 are MMA (5.1%).** The TC idle time comes from W1's non-MMA instruction overhead — descriptor building, uniform register moves, mbarrier management — stretching the gap between MMA issues.

**However**, clock64 timing (see `clock64_timing_analysis.md`) reveals the TRUE bottleneck is the **epilogue**. W1 spends ~17% of each tile waiting on epilogue_mbar (invisible to SourceCounters due to NANOSLEEP attribution). The instruction overhead is secondary.

## W1 Instruction Budget (per tile)

| Category | Instructions/tile | % |
|----------|-------------------|---|
| **UTCQMMA.2CTA (MMA)** | **24** | **5.1%** |
| SYNCS (mbar_wait) | 7 | 1.5% |
| UTCBAR (commit) | 7 | 1.5% |
| **Other (ALU/ctrl/desc)** | **437** | **91.9%** |
| **Total** | **475** | |

The 437 "other" instructions per tile include:
- `R2UR` / `R2UR.BROADCAST`: converting registers to uniform registers (required by UTCQMMA operands)
- `ELECT`: CTA0-lane-0 election for uniform operations
- `PLOP3.LUT`: predicate logic for election/accumulate control
- `IMAD` / `IMAD.WIDE` / `IMAD.MOV`: descriptor building (make_smem_desc), SMEM address computation
- `SHF`, `LOP3.LUT`: shift and mask for SMEM descriptor encoding (SWIZZLE_128B)
- `STL`: local store for phase tracking variables

Per K-iteration: ~79 instructions for 4 MMAs = **~20 overhead instructions per MMA issue**.

## W1 Stall Breakdown (total across all tiles)

| Stall | Samples | Per tile |
|-------|---------|----------|
| wait | 675 | 0.062 |
| long_scoreboard | 265 | 0.024 |
| selected | 161 | 0.015 |
| branch_resolving | 89 | 0.008 |
| not_selected | 58 | 0.005 |
| no_instruction | 53 | 0.005 |
| math | 25 | 0.002 |

**Total: 1,329 stalls across 5,172,603 instruction executions = 0.026% stall rate.**

The near-zero stall rate is misleading. SourceCounters attributes mbar_wait time to NANOSLEEP (counted as "sleeping"), not as a stall on the try_wait instruction. clock64 reveals W1 spends 1,052 cycles/tile (17%) waiting on epilogue_mbar — a stall invisible to SourceCounters.

## MMA Instruction Detail

24 UTCQMMA.2CTA instructions per tile, fully unrolled (6 K-iters x 4 sub-MMAs):

- Total executions: 261,072 (= 10,878 tiles x 24 MMA)
- Total stalls: 92 (0.035%) — essentially zero
- Dominant stall at MMA: `wait` (75 out of 92) — TC pipeline not ready yet
- Zero `math_pipe_throttle` — TC pipe never backs up

**The TC pipe is NOT being saturated.** It idles waiting for the next MMA instruction.

## Key mbar_wait Detail (5-warp baseline)

W1's 7 mbar_waits per tile:

| Wait type | Exec count | Stalls | Stall rate |
|-----------|------------|--------|------------|
| epilogue_mbar[buf] | 10,878 | 0 | 0.000% |
| tma_mbar[s=0] ki=0 | 10,878 | 0 | 0.000% |
| tma_mbar[s=1] ki=1 | 10,878 | 1 | 0.009% |
| tma_mbar[s=2] ki=2 | 10,878 | 2 | 0.018% |
| tma_mbar[s=3] ki=3 | 10,878 | 1 | 0.009% |
| tma_mbar[s=0] ki=4 | 10,878 | 0 | 0.000% |
| tma_mbar[s=1] ki=5 | 10,878 | 2 | 0.018% |

All stall rates ~0% because:
- TMA data is always ready (4-stage pipeline works perfectly)
- epilogue_mbar wait uses NANOSLEEP → attributed as "sleeping" not "stalled"

## Epilogue Warp Stalls (5-warp baseline — partially stale)

The top stall instructions were ALL epilogue warps (W2-W6, now W2-W5):

| Instruction | Exec | Stalls | Dominant stall | Role |
|-------------|------|--------|----------------|------|
| IMAD.U32 R109, R76, 0x10000, RZ | 259,296 | 4,103 | long_sb (4,080) | TMEM addr compute after tcgen05.ld |
| @!P0 BRA (mainloop_mbar loop) | 108,780 | 3,414 | long_sb (3,404) | mainloop_mbar fail-then-branch |
| WARPSYNC.ALL | 1,258 | 2,261 | sleep (2,225) | End-of-kernel cluster sync |
| UCGABAR_WAIT | 1,036 | 2,081 | barrier (2,081) | Cluster barrier wait |

Mainloop_mbar retry behavior (5-warp):
- First try: 108,780 execs
- Retry loop: 2,193,504 total iterations → avg 20 retries per wait

With 4 warps (current): fewer retries expected since epilogue finishes faster. clock64 shows mainloop_mbar_wait = 1,123 cycles avg (18% of epilogue cycle).

## What This Means

The TC idle time (~50%) has two causes:

1. **Epilogue bottleneck (primary):** W1 spends 17% of each tile waiting for TMEM buffer release. Invisible to SourceCounters. Confirmed by clock64.

2. **W1 instruction overhead (secondary):** 20 non-MMA instructions between each MMA issue. With 4,155 cycles for the K-loop and 24 MMAs, each MMA takes ~173 cycles on average. The ~20 overhead instructions take ~20 cycles (single-issue) → 12% of per-MMA time. This is unavoidable given `cta_group::2` MMA semantics.

## Reconciliation with clock64

| Metric | SourceCounters | clock64 |
|--------|---------------|---------|
| W1 epi_mbar stalls | 0 (NANOSLEEP) | 1,052 cycles (17%) |
| W1 tma_mbar stalls | 0 | 856 cycles (14%) |
| W1 K-loop | — | 4,155 cycles (69%) |
| Epilogue Phase 1 | 4,080 long_sb stalls | 4,140 cycles (67% of epi) |
| Epilogue Phase 2B | — | 899 cycles (15% of epi) |
| Epilogue ml_wait | 20 retries avg | 1,123 cycles (18% of epi) |

SourceCounters excels at per-instruction attribution (which SASS instructions stall) but misses aggregate time spent in spin-wait loops with NANOSLEEP. clock64 gives absolute cycle counts but can't attribute to specific instructions. Together they give the complete picture.
