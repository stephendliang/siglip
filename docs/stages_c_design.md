# StagesC Pipeline — Architectural Fix for FC2 Epilogue

## Problem Statement

FC2's 19% gap vs CUTLASS (1.452ms vs 1.225ms) is entirely in the epilogue.
CUTLASS achieves near-zero epilogue overhead by pipelining C-tile (residual) loads
and D-tile (output) stores across multiple SMEM buffer stages. Our epilogue is
synchronous — W2-W5 issue TMA loads, wait, compute, store, all in sequence.

SASS instruction count: 2936 (ours) vs 2352 (CUTLASS). The +584 gap splits:
- +361 in setup/K-loop (pipeline-hidden, timing-irrelevant)
- +223 in epilogue (critical path)

Biggest category deltas: R2UR +292 (313 vs 21), PLOP3 +287 (311 vs 24),
BSSY/BSYNC +144 (174 vs 30). These are infrastructure overhead, not useful work.

## How CUTLASS Does It

Reference: `third_party/cutlass/include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp`

### Two independent pipelines

```
LoadPipeline  = PipelineTransactionAsync<StagesC=3>   // C-tile: GMEM → SMEM
StorePipeline = PipelineTmaStore<StagesD=2>            // D-tile: SMEM → GMEM
```

Each pipeline has its own mbarrier array and rotating SMEM buffer pool.

### W0 (TMA producer): loads C upfront

```
// Lines 515-544 of sm100_epilogue_tma_warpspecialized.hpp
// W0 runs this ONCE per tile, before any epilogue consumer starts
for epi_m in [0, EPI_M):
    for epi_n in [0, EPI_N):
        producer_acquire(load_state)           // wait for free SMEM slot
        tma_load(smem_C[buffer_idx], gmem_C)   // non-blocking TMA issue
        producer_commit(load_state)             // signal mbarrier, advance
```

Key: all C-tile loads are issued in a tight loop BEFORE the consumer warp group
even starts. W0 doesn't wait for any of them to complete — it issues all TMA
descriptors and moves on. The TMA engine handles data movement asynchronously.

W0 then continues issuing K-loop TMA loads for the next tile. The C loads for
the current tile and K loads for the next tile are all in flight simultaneously.

### W2-W5 (consumer): compute with pre-loaded C

```
// Lines 820-945
for epi_m in [0, EPI_M):
    for epi_n in [0, EPI_N):
        // C data already in SMEM — just wait on mbarrier
        consumer_wait(load_state)              // typically instant (data pre-loaded)
        copy(smem_C → reg_C)                   // SMEM → registers

        // TMEM accumulator load
        consumer_wait(acc_pipeline)            // wait for K-loop TMEM ready
        copy(tmem → reg_acc)                   // LDTM

        // Epilogue compute
        reg_D = visit(reg_acc, reg_C)          // bias + residual + CVT

        // Store to SMEM staging
        copy(reg_D → smem_D[buffer_idx])       // STS to output staging

        // Issue TMA store (pipelined — previous store completes in background)
        fence_view_async_shared()
        tma_store(smem_D[buffer_idx] → gmem_D)
        producer_commit(store_state)           // signal store mbarrier
```

### Why this achieves zero epilogue overhead

The TMEM double-buffering (512 TMEM cols, 2 × 256) means W1's MMA for tile N+1
writes to the "other" TMEM buffer while W2-W5 read from the current buffer.
The C-tile is pre-loaded by W0, so `consumer_wait(load_state)` is instant.
The TMA store pipeline means SMEM→GMEM for the previous subtile overlaps with
compute for the current subtile.

Result: epilogue compute is completely hidden under the K-loop of the next tile.
CUTLASS achieves 1.225ms — 99.6% of the 1.220ms theoretical K-loop throughput
bound. The epilogue adds < 5μs.

## Why Our Previous Attempts Failed

### W0_RES_PREFETCH (neutral — no timing improvement)

```
// Our W0 after K-loop completes for tile N:
if (W0_RES_PREFETCH) {
    mbar_arrive_expect_tx(res_mbar, size)
    tma_load(res_staging, tma_res_desc, tile_N_cols, tile_N_row, res_mbar)
    // Only loads PASS 0 (64 cols out of 256)
    // Pass 1 still loaded by epilogue warps themselves
}
```

**Why it failed**: Only prefetches 64 of 256 cols. The other 192 cols are still
loaded by W2-W5 on the critical path. The 64-col prefetch saves ~200 cycles of
TMA wait, but the total Phase 1 is 8430 cycles — saving 2.4% is noise.

### W0_RES_FULL (catastrophic — +15%, epi_wait 107→5400 cycles)

```
// Our W0 loads ALL residual for tile N:
for pass in [0, LOCAL_PASSES):
    mbar_arrive_expect_tx(res_mbar, pass_size)
    tma_load(res_staging, tma_res_desc, pass_cols, ...)
    // THEN WAITS for epilogue to consume pass before issuing next
    mbar_wait(res_consumed_mbar)   // ← BLOCKS W0
```

**Why it failed**: Circular dependency.

```
W0 wants to load pass-1 → waits for res_consumed_mbar (epilogue consumed pass-0)
W2-W5 consuming pass-0 → blocked on Phase 1 compute (STS contention)
Phase 1 contention → slows K-loop by 3300 cyc/tile
K-loop slow → W0 delayed on K-loop mbarrier for next tile
→ W0 can't load pass-1 promptly → epilogue starved → more contention
```

This is a textbook priority inversion. W0 (high-priority TMA issuer) blocks on
W2-W5 (slow epilogue consumer), which blocks on W0 (K-loop TMA). The epi_wait
going from 107→5400 cycles is the system entering a near-deadlock state where
every warp group is waiting on another.

### EPI_LOAD_WARP (runtime hang — mbarrier bug)

A separate warp dedicated to residual loading. Hung on mbarrier — never debugged.
The concept is sound (decouple loading from K-loop warp), but the implementation
had a synchronization bug.

### Why CUTLASS avoids all of this

1. **No circular dependency**: W0 issues ALL C loads upfront, then moves to
   K-loop for the next tile. No waiting for consumer to consume. StagesC=3
   means 3 SMEM buffers — W0 can be 3 subtiles ahead of the consumer.

2. **No pass-by-pass handshake**: CUTLASS loads C for the entire tile in one
   burst (all epi_m × epi_n subtiles). No pass-0/pass-1 dependency chain.

3. **Separate pipeline objects**: LoadPipeline and StorePipeline have independent
   mbarrier arrays. No sharing between C-loads and D-stores.

4. **W0 never blocks on consumer**: `producer_acquire()` only blocks if ALL
   StagesC buffers are full — which requires the consumer to be 3 subtiles
   behind. With 4 epilogue warps and fast compute, this never happens.

## Design: StagesC for Our Kernel

### Required changes

#### 1. SMEM layout: add rotating C-tile buffer (StagesC=2)

Current SMEM budget:
- Mainloop pipeline: 4 stages × ~32KB = ~131KB
- Epilogue staging: 4 regions × 16KB = 64KB
- Bias SMEM: 512B
- Total: ~196KB of 228KB available

New allocation:
- Mainloop pipeline: 4 stages × ~32KB = ~131KB
- Epilogue staging (output): 2 regions × 16KB = 32KB (StagesD=2, halved)
- C-tile buffer: 2 stages × 16KB = 32KB (StagesC=2, 64 cols × 256 rows × 2B)
- Bias SMEM: 512B
- Total: ~196KB ← fits!

With StagesC=2, we can have 2 C-tile subtiles in flight. W0 loads subtile N
while W2-W5 consume subtile N-1.

#### 2. New mbarrier set for C-tile pipeline

```c
// In SMEM mbarrier layout
OFF_RES_LOAD_MBAR    // 2 mbarriers (StagesC=2), W0 arrives, W2-W5 wait
OFF_RES_RELEASE_MBAR // 2 mbarriers (StagesC=2), W2-W5 arrive, W0 waits (backpressure only)
```

#### 3. W0 mainloop restructure

```
// TILE LOOP (W0)
for tile_idx in assigned_tiles:
    // === K-loop: issue TMA loads for A/B tiles ===
    for k in [0, K_ITERS):
        tma_load_A(stage[k % N_STAGES])
        tma_load_B(stage[k % N_STAGES])
        mbar_arrive(mainloop_mbar[k % N_STAGES])

    // === C-tile pre-load: issue TMA loads for residual ===
    // NON-BLOCKING — issue all subtile loads, don't wait for any
    for subtile in [0, N_SUBTILES):
        res_acquire(subtile % StagesC)          // backpressure if all buffers full
        tma_load_C(res_smem[subtile % StagesC], ...)
        res_commit(subtile % StagesC)           // signal consumer mbarrier
    // W0 immediately proceeds to next tile's K-loop
```

Key difference from W0_RES_FULL: NO waiting for consumer between subtiles.
The `res_acquire()` only blocks if BOTH StagesC buffers are full — with StagesC=2,
W0 can be 1 subtile ahead. With StagesC=3 (if SMEM allows), 2 subtiles ahead.

#### 4. Epilogue consumer restructure

```
// EPILOGUE (W2-W5, runs for previous tile while K-loop for current tile executes)
for subtile in [0, N_SUBTILES):
    // C data pre-loaded by W0 — just wait on mbarrier
    res_consumer_wait(subtile % StagesC)    // typically instant

    // Load pre-combined bias+residual from SMEM (already in rotating buffer)
    LDS bias+residual from res_smem[subtile % StagesC]

    // Signal W0 that this buffer slot is free
    res_consumer_release(subtile % StagesC)

    // TMEM load + epilogue compute + STS (unchanged from current Phase 1)
    TMEM_WAIT()
    LDTM → F2FP → HADD2 → STS

    // TMA store (pipelined via StagesD)
    store_acquire(subtile % StagesD)
    fence_proxy_async()
    UTMASTG
    store_commit(subtile % StagesD)
```

#### 5. SMEM buffer management

The C-tile rotating buffer needs SWIZZLE_128B layout (same as current residual
staging) so LDS can read with the same swizzle pattern. Each buffer slot holds
64 cols × 32 rows × 2B = 4KB per warp-row, × 8 warp-rows = 32KB per stage.

Wait — 32KB × 2 stages = 64KB for C-tile. That's the same as current residual
staging. We can REUSE the existing residual staging SMEM for the C-tile pipeline
by converting from single-use to double-buffered.

### What changes in source files

```
kernel_common.cuh:
  - Add STAGES_C param (default 2)
  - Add OFF_RES_LOAD_MBAR, OFF_RES_RELEASE_MBAR offsets
  - Add res_smem rotating buffer calculation

kernel_body.cuh:
  - W0 mainloop: add C-tile load loop after K-loop (lines ~1600)
  - epilogue_store: replace TMA_RESIDUAL wait-and-load with consumer_wait
  - Add store pipeline (StagesD) rotation for TMA stores
  - Remove W0_RES_PREFETCH, W0_RES_FULL code paths (subsumed)

fc2.cu:
  - No changes needed (epilogue macro unchanged)
```

Estimated: ~200 lines changed, ~50 lines added, ~100 lines removed (old W0_RES code).

### Risks

1. **SMEM pressure**: StagesC=2 requires 64KB for C-tile buffers. Combined with
   131KB mainloop + 32KB output staging + bias = 228KB. That's exactly the 228KB
   limit. Any SMEM increase breaks it. StagesC=3 won't fit.

2. **Register pressure**: Epilogue warps need registers for both C data (from
   SMEM) and accumulator (from TMEM) simultaneously. Current regs: 207. Adding
   C data registers could push past 256, causing spills.

3. **W0 starvation**: If K-loop TMA loads take longer than expected, W0 may not
   issue C-tile loads in time. Need to verify that K-loop TMA + C-tile TMA fit
   within the pipeline window.

4. **mbarrier complexity**: 3 independent mbarrier sets (mainloop, C-load,
   D-store) each with phase tracking. Debugging is hard — EPI_LOAD_WARP already
   hung on an mbarrier bug.

### Verification plan

1. Identity test: StagesC with no-op epilogue (just TMEM load + STS, no bias/res)
   Should match baseline timing.
2. Correctness: Same checksum as current FC2.
3. Timing: Compare Phase 1 cycles vs current 8430.
4. ncu: Verify SMEM utilization, mbarrier wait times.
5. Regression: Run full grid search with STAGES_C=2 as new param.
