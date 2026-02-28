# SourceCounters Analysis — W1 Stall Isolation

## TL;DR

**W1 is NOT stalled on anything.** It executes 475 instructions per tile, of which only 24 are MMA (5.1%). The TC idle time comes from W1's non-MMA instruction overhead — descriptor building, uniform register moves, mbarrier management — stretching the gap between MMA issues.

All three hypothesized causes are ruled out:

| Cause | Evidence | Verdict |
|-------|----------|---------|
| Data starvation (tma_mbar wait) | 6 tma_mbar waits, 0 stalls each (out of 10,878 execs) | **Eliminated** |
| TMEM buffer pressure (epilogue_mbar wait) | 1 epilogue_mbar wait, 0 stalls | **Eliminated** |
| Tile transition overhead | W1 total stalls = 1,329 out of 5,172,603 execs (0.026%) | **Not dominant** |

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
W1 is issue-bound, not stall-bound.

## MMA Instruction Detail

24 UTCQMMA.2CTA instructions per tile, fully unrolled (6 K-iters × 4 sub-MMAs):

- Total executions: 261,072 (= 10,878 tiles × 24 MMA) ✓
- Total stalls: 92 (0.035%) — essentially zero
- Dominant stall at MMA: `wait` (75 out of 92) — TC pipeline not ready yet
- Zero `math_pipe_throttle` — TC pipe never backs up

**The TC pipe is NOT being saturated.** It idles waiting for the next MMA instruction.

## Key mbar_wait Detail

W1's 7 mbar_waits (addresses 0x70efc95b2f30 through 0x70efc95b43d0):

| Wait type | Register | Exec count | Stalls | Stall rate |
|-----------|----------|------------|--------|------------|
| epilogue_mbar[buf] | [R5+URZ], R4 | 10,878 | 0 | 0.000% |
| tma_mbar[s=0] ki=0 | [R50+URZ], R9 | 10,878 | 0 | 0.000% |
| tma_mbar[s=1] ki=1 | [R48+URZ], R9 | 10,878 | 1 | 0.009% |
| tma_mbar[s=2] ki=2 | [R46+URZ], R7 | 10,878 | 2 | 0.018% |
| tma_mbar[s=3] ki=3 | [R44+URZ], R7 | 10,878 | 1 | 0.009% |
| tma_mbar[s=0] ki=4 | [R50+URZ], R7 | 10,878 | 0 | 0.000% |
| tma_mbar[s=1] ki=5 | [R48+URZ], R7 | 10,878 | 2 | 0.018% |

TMA data is ALWAYS ready when W1 needs it. The 4-stage pipeline works perfectly.
Epilogue completes well before W1 needs the TMEM buffer. F10 confirmed.

## Epilogue Warp Stalls (for context)

The top stall instructions are ALL epilogue warps (W2-W6):

| Instruction | Exec | Stalls | Dominant stall | Role |
|-------------|------|--------|----------------|------|
| IMAD.U32 R109, R76, 0x10000, RZ | 259,296 | 4,103 | long_sb (4,080) | TMEM addr compute after tcgen05.ld |
| @!P0 BRA 0x70efc95bab30 | 108,780 | 3,414 | long_sb (3,404) | mainloop_mbar fail-then-branch |
| WARPSYNC.ALL | 1,258 | 2,261 | sleep (2,225) | End-of-kernel cluster sync |
| UCGABAR_WAIT | 1,036 | 2,081 | barrier (2,081) | Cluster barrier wait |

W2-W6 mainloop_mbar retry behavior:
- First try (in tile loop): 108,780 execs
- Retry loop: 2,193,504 total iterations → avg 20 retries per wait
- This confirms epilogue finishes AFTER K-loop starts → epilogue is in K-loop shadow

## What This Means

The TC idle time (50%) is caused by **W1 instruction overhead**, not by any stall. W1 issues 1 MMA every ~20 instructions on average. Between MMA issues, the TC pipeline has no new work and eventually drains.

The fix is to **reduce W1's instruction count between MMA issues**:

1. **Larger TK** (e.g., 256): Halves K-iters (3 instead of 6), halving the per-tile overhead of tma_mbar waits, commits, and descriptor rebuilding. Same 24 MMAs but fewer inter-K-iter transitions. Needs more SMEM per stage (32 KB × 2 = 64 KB, vs current 32 KB) — may require reducing N_STAGES to 3.

2. **Fewer descriptor rebuilds**: The `make_smem_desc()` function is rebuilt fresh each K-iteration. Could precompute all 4 stage descriptors once and index them.

3. **Fewer R2UR/ELECT instructions**: These uniform register operations are forced by the cta_group::2 MMA semantics. May be unavoidable without architectural changes.

4. **TMA multicast for B**: Doesn't directly reduce W1 overhead but frees L2/DRAM bandwidth, potentially allowing TMA to deliver faster for larger TK tiles.

## Next Step: clock64() Instrumentation

SourceCounters gives stall attribution but not absolute timing. clock64() timestamps at tile boundaries will directly measure:
- K-loop time per tile (first MMA to last MMA commit)
- Tile transition overhead (last MMA commit to next first MMA)
- Total per-tile budget validation
