# FC2 Epilogue Rewrite — Expert Briefing + Implementation Checklist

## The Problem

We have a hand-tuned persistent GEMM kernel for Blackwell (SM100a / B200) that runs the FC2
layer of a SigLIP2 vision encoder.  Shape: [928256, 3072] × [3072, 768]^T + bias + residual.
FP8 inputs, BF16 output, 128-row tiles via `cta_group::2` with 2×1 clusters (74 clusters on
148 SMs).

**Our kernel: 1.47ms.  CUTLASS: 1.21ms.  Gap: 20% (260μs).**

Both kernels use the **exact same tile config**: 256×256×128, 2×1 cluster, `cta_group::2`.
CUTLASS has 2,864 SASS instructions vs our 2,248 (27% more code, 20% faster).  Our GEMM core
is actually **10% faster** than CUTLASS fused (1.088ms stripped vs 1.211ms CUTLASS).  The gap
is entirely that our epilogue destroys the GEMM advantage.

---

## The Proof: STRIP_EPILOGUE

`STRIP_EPILOGUE=1` compiles out all epilogue code.  Epilogue warps just signal "TMEM free"
and idle.

| Metric | GEMM-only (strip) | Fused (full epilogue) | Epilogue overhead |
|--------|--------------------|-----------------------|-------------------|
| **Ours** | 1.088ms | 1.472ms | **+384μs (+35%)** |
| **CUTLASS** | — | 1.211ms | **~78μs (estimated)** |

Our GEMM core (1.088ms) is **10% faster than CUTLASS fused** (1.211ms).  We have a faster
engine but our epilogue adds 384μs (+35%), turning a 10% lead into a 20% deficit.  CUTLASS's
epilogue adds ~78μs.  **Our epilogue overhead is ~5× higher.**

### Linear scaling proof (EPI_WARP_LIMIT)

`EPI_WARP_LIMIT=N` restricts how many of the 4 epilogue warps run real epilogue code.

| Active warps | ms | Overhead | Per-warp |
|---|---|---|---|
| 0 (strip) | 1.088 | 0 | — |
| 1 | 1.159 | 71μs | 71μs |
| 2 | 1.248 | 160μs | 80μs |
| 3 | — | — | — |
| 4 (baseline) | 1.472 | 384μs | 96μs |

~80-96μs per active warp.  CUTLASS total epilogue overhead is ~78μs across 4 warps (~20μs
each).  **Our per-warp overhead is ~5× higher.**

### What causes the overhead

It's NOT any single operation.  A 30-experiment strip bench proved that removing any individual
epilogue operation (TMA load, TMA store, compute, BAR.SYNC) from the 4-warp baseline still
produces ~1.635ms.  The overhead is the **aggregate resource pressure** from epilogue warps
actively executing memory and compute instructions — SMEM ports, L2 bandwidth, TMA units — all
contending with the K-loop across 74 clusters and 10,878 tiles.

The following individual hypotheses were all ruled out by isolation experiments (earlier session,
different clock conditions — absolute values differ from above, but all match their session's
strip baseline, proving each hypothesis contributes zero overhead in isolation):
- Warp scheduler dispatch pressure (NOP_EPILOGUE): matched strip speed at 10k cycles busy-wait
- TMEM release timing (EPI_DELAY): matched strip speed at 10k cycles delayed arrive
- Register file occupancy (REG_PAD): matched strip speed at 186 regs (54.5% RF)

**The only path to reducing the overhead is making each warp finish its epilogue faster** —
fewer cycles of active memory/compute work per warp per tile.

---

## What We Currently Do (6 warps)

### Warp layout

| Warp | Role | Threads |
|------|------|---------|
| W0 | TMA A/B loads (lane 0 issues, all threads compute addrs) | 32 |
| W1 | MMA — `tcgen05.mma.cta_group::2`, accumulates into TMEM | 32 |
| W2 | Epilogue warp 0 — row_group 0 (rows 0-31 of 128-row CTA tile) | 32 |
| W3 | Epilogue warp 1 — row_group 1 (rows 32-63) | 32 |
| W4 | Epilogue warp 2 — row_group 2 (rows 64-95) | 32 |
| W5 | Epilogue warp 3 — row_group 3 (rows 96-127) | 32 |
| | **Total: 192 threads** | |

### Per-warp epilogue flow (per tile, x32 TMEM path)

Each epilogue warp operates **independently** — no coordination with other epilogue warps.
All 4 warps run the identical code with different `row_group` and SMEM staging addresses.

```
1. mbar_wait(mainloop_mbar[prev_buf])     // block until W1 finishes MMA for this tile

2. for pass = 0..1:                       // 2 passes × 128 cols = 256 cols total
                                          // (NUM_PASSES_PARAM=2, PASS_COLS=128)

   a. SELF-LOAD RESIDUAL — each warp loads its own 32-row × 128-col slice:
      lane 0 only:
        mbar_arrive_expect_tx(res_mbar, 8192)     // 2 regions × 4096 B
        tma_load_2d_cta(res_staging_region_0, &tma_res, col_0, gm_base, res_mbar)
        tma_load_2d_cta(res_staging_region_1, &tma_res, col_1, gm_base, res_mbar)
      // These are global memory TMA loads — each hits L2 or DRAM

   b. for chunk = 0..3:                   // 4 chunks × 32 cols = 128 cols per pass
      // TMEM LOAD — read 32 FP32 accumulators from tensor memory:
      tcgen05.ld.sync.aligned.32x32b.x32.b32  taddr[chunk_col]

      // WAIT for residual TMA load to land in SMEM:
      mbar_wait(res_mbar, res_phase)               // only on first chunk of pass

      // LOAD BIAS — 4× LDS.128 from SMEM (BIAS_SMEM=1) or 4× LDG from global:
      ld.shared.v4.b32 × 4                         // 32 BF16 bias values

      // LOAD RESIDUAL — 4× LDS.128 from per-warp staging SMEM:
      ld.shared.v4.b32 × 4                         // 32 BF16 residual values
      // (swizzled addressing via precomputed offsets)

      // COMPUTE — BF16 math:
      F2FP × 4:    cvt.rn.bf16x2.f32 (FP32 acc → BF16 pairs)
      HFMA2 × 8:   fma.rn.bf16x2 (bias add, BF16 precision)
      HADD2 × 4:   add.bf16x2 (residual add, BF16 precision)
      // 16 ALU ops per chunk

      // STS — 4× st.shared.v4.b32 to output staging SMEM:
      st.shared.v4.b32 × 4                         // CLUSTERED at end of chunk
      // 27 cyc each with zero shadow — ptxas refuses to interleave

      // FENCE + TMA STORE (interleaved per INTERLEAVE_STRATEGY):
      __syncwarp()
      fence.proxy.async.shared::cta
      cp.async.bulk.tensor.2d  (output to global)
      cp.async.bulk.commit_group

3. mbar_arrive(epi_mbar)                 // signal W1: TMEM buffer is free for next tile
```

### SMEM layout (per-warp independent staging)

```
Each epilogue warp ew gets its own 16 KB region:
OFF_STAGING + ew * 16384
  ├── [+0]     output region 0:   4096 B  (32 rows × 64 cols × 2B BF16, SWIZZLE_128B)
  ├── [+4096]  output region 1:   4096 B
  ├── [+8192]  residual region 0: 4096 B  (TMA load target)
  └── [+12288] residual region 1: 4096 B
Total 4 warps: 65,536 B (64 KB)

Full SMEM breakdown (NS5 config):
  A/B pipeline stages: 5 × 32,768 = 163,840 B
  Mbarriers:           ~400 B (TMA×5, MMA×5, mainloop×2, epilogue×2, residual×4)
  Bias SMEM:           512 B (256 BF16)
  Epilogue staging:    4 × 16,384 = 65,536 B
  ─────────────────────────────────
  TOTAL:               ~230 KB (of 228 KB limit — barely fits NS5)
```

### Why this architecture is slow

1. **16 TMA global loads per tile for residual alone.**  4 warps × 2 passes × 2 regions = 16
   TMA loads, all hitting L2/DRAM.  Each is 419 cyc (L2-warm).  They compete with W0's A/B
   TMA loads for the next tile's K-loop.

2. **STS is clustered at end of each chunk** — 4 consecutive `st.shared.v4.b32` with no
   intervening compute to shadow.  27 cyc throughput each.  ptxas emits identical SASS
   regardless of source-level scheduling attempts (5 approaches tested, all byte-identical).

3. **BF16 math in epilogue.**  We convert FP32→BF16 first (F2FP), then add bias and residual
   in BF16 (HFMA2/HADD2).  HFMA2 has 7.5% conflict with STS (calibrated).  FFMA has ~0%.

4. **Per-warp SMEM staging costs 64 KB** — forces NS5 to barely fit.  No room for prefetch
   buffers or larger tiles.

5. **No inter-warp coordination** — warps can't share loaded data.  Each warp independently
   loads, computes, and stores its own 32-row slice.

---

## What CUTLASS Does (8 warps)

### Warp layout

| Warp | Role | Threads |
|------|------|---------|
| W0 | Tile scheduler (`UTCBAR.2CTA.MULTICAST`) | 32 |
| W1 | MMA — `tcgen05.mma.cta_group::2` | 32 |
| W2 | MainloopLoad — TMA A/B | 32 |
| **W3** | **EpilogueLoad — TMA-loads residual into SHARED SMEM** | 32 |
| W4 | Epilogue warp 0 | 32 |
| W5 | Epilogue warp 1 | 32 |
| W6 | Epilogue warp 2 | 32 |
| W7 | Epilogue warp 3 | 32 |
| | **Total: 256 threads** | |

Key structural difference: W0 is a dedicated scheduler (no TMA loads), W2 is a dedicated TMA
loader, and **W3 is a dedicated epilogue data loader**.  Our kernel combines scheduler+TMA in
W0 (no W3 equivalent at all).

### Epilogue flow (coordinated, shared SMEM)

CUTLASS uses `ReuseSmemC=true`: residual load staging and output store staging **share the same
SMEM region**.  W3 TMA-loads residual into the region, all 4 epilogue warps LDS from it, then
overwrite it with STS output data, then TMA-store to global.  Two BAR.SYNCs per sub-iteration
gate the transitions.

```
W3 (EpilogueLoad):                          W4-W7 (Epilogue warps):
──────────────────                           ───────────────────────
                                             mbar_wait(mainloop_mbar)
for sub_iter = 0..3:                         for sub_iter = 0..3:

  load_pipe.producer_acquire()                 load_pipe.consumer_wait()
    // mbar_wait: SMEM stage free                // mbar_wait: W3 finished loading
    // (first iter: immediately free)

  TMA LOAD residual into shared SMEM:          TMEM LOAD — tcgen05.ld.sync x32
    128 rows × 64 cols × 2B = 16 KB              (32 FP32 accumulators per warp)
    One TMA load covers ALL 4 warps' rows
                                               LDS residual — from SHARED SMEM:
  load_pipe.producer_commit()                    4× ld.shared.v4.b32 per warp
    // mbar_arrive: signal load complete          (each warp reads its 32-row slice)
                                                  NOT LDG — zero global memory traffic!

                                               LDS bias — from SMEM (pre-loaded once)

                                               FP32 COMPUTE — FFMA, not HFMA2:
                                                 32× FFMA (bias add in FP32)
                                                 32× FFMA (residual add in FP32)
                                                 (residual needs LOP3/SHF to unpack BF16)
                                                 16× F2FP (FP32→BF16, interleaved w/ FFMA)

                                               STS — INTERLEAVED with compute in SASS:
                                                 4× STS.128 (each has 2-4 FFMA shadow)
                                                 Total STS wall time ≈ 0 (fully hidden)

                                               FENCE.VIEW.ASYNC.S
                                               BAR.SYNC 1, 128t — STS visible to TMA unit

                                               load_pipe.consumer_release()
                                                 // signal W3: SMEM free for next load

                                               TMA STORE — lane 0:
                                                 UTMASTG.3D + UTMACMDFLUSH
                                                 (output → global, async)

                                               DEPBAR.LE — wait for TMA store to drain
                                               BAR.SYNC 1, 128t — SMEM free for next STS

epi_mbar_arrive()                            epi_mbar_arrive()  // TMEM free for W1
```

### SMEM layout (shared, not per-warp)

```
With ReuseSmemC=true and StagesC=2 (double-buffered):
  ├── stage 0: 16,384 B  (128 rows × 64 cols × 2B, SWIZZLE)
  └── stage 1: 16,384 B
  C-load and D-store SHARE the same buffer (sequential use within each sub-iter).
  Total epilogue SMEM: 32 KB (vs our 64 KB)

With StagesC=4 (quad-buffered, what CUTLASS actually configures):
  Stages 0-3: 4 × 16,384 = 65,536 B
  But with ReuseSmemC, C and D share — so it's still 64 KB max.
  Allows W3 to prefetch up to 3 sub-iters ahead of epilogue warps.
```

### Why this is fast (SASS-verified differences)

| Aspect | **Ours** | **CUTLASS** | Impact |
|--------|----------|-------------|--------|
| Residual source | LDG (global, ~40 cyc L2) | **LDS (SMEM, ~4-25 cyc)** | Fewer active cycles/warp |
| Residual loads/tile | 16 TMA loads (4 warps × 4) | **4 TMA loads (W3 only)** | 4× less L2/DRAM traffic |
| STS scheduling | 4 clustered (27 cyc each) | **4 interleaved (0 cyc visible)** | ~108 cyc saved/chunk |
| Epilogue math | BF16 (HFMA2, 7.5% STS conflict) | **FP32 (FFMA, ~0% STS conflict)** | Less pipe contention |
| SMEM budget | 64 KB (per-warp) | **32 KB (shared, 2-stage)** | Room for NS5+ easily |
| Per-warp overhead | ~96μs | **~20μs** | ~5× faster epilogue |

The combination of LDS-from-SMEM (instead of LDG), 4× fewer TMA loads, and interleaved STS
means each epilogue warp is active for far fewer cycles.  Less active time = less K-loop
inflation = ~78μs total vs our 384μs.

---

## Why Every Previous Attempt Failed

### Attempt 1: EPI_LOAD_WARP (+13% regression)

**What it did:** Added W2 as a dedicated loader warp.  W2 serialized TMA loads across ALL 4
epilogue warps: loaded residual for warp 0, then warp 1, then warp 2, then warp 3, each into
that warp's PRIVATE staging region.  Epilogue warps (now W3-W6, 7 warps total) waited on
per-warp mbarriers.

**Why it failed:** Per-warp loading means W2 does 4 warps × 2 passes × 2 regions = 16 TMA
loads SERIALLY.  That's slower than 4 warps self-loading in parallel.  W2 also blocked on
`mbar_wait(res_consumed_mbar)` between tiles (had to wait for epilogue to finish before
loading the next tile's residual).  The serialization + blocking made total epilogue time
WORSE.

**What it got wrong:** It preserved per-warp staging.  The CUTLASS approach loads into a
SHARED region — one TMA load serves all 4 warps.  4 TMA loads total per tile, not 16.

### Attempt 2: W0_RES_FULL (+15% regression)

**What it did:** W0 loads ALL residual data (all 4 warps × both passes) after the K-loop
finishes, before signaling epilogue warps.

**Why it failed:** W0 had to complete 8 TMA loads before arriving on `epi_mbar`.  This delayed
the epilogue start by ~5,400 cycles (measured via `clock64()` timing).  The K-loop for the NEXT
tile couldn't start until the epilogue finished (TMEM double-buffer protocol).  So the delay
serialized into the critical path.  Also still per-warp staging (loaded into each warp's
private SMEM).

**What it got wrong:** Loading ALL data upfront defeats pipelining.  CUTLASS's W3 loads ONE
sub-iteration at a time, overlapping with epilogue compute on the previous sub-iteration.

### Attempt 3: W0_RES_PREFETCH (+10-38μs, neutral to negative)

**What it did:** W0 pre-loads residual for the first pass during the K-loop (while W0 is
otherwise idle between A/B loads).  Epilogue warps still self-load remaining passes.

**Why it failed:** W0's TMA loads for residual competed with W0's TMA loads for A/B.  The
prefetch stole bandwidth from the mainloop pipeline, slightly inflating K-loop iterations.
Net effect: small regression.  Also: only covered 1 of 2 passes — warps still self-loaded
the second pass.

### Attempt 4: STAGES_C=2 (neutral)

**What it did:** W0 pre-loads residual into 2 dedicated SMEM stages per warp (separate from
A/B pipeline stages).  Epilogue warps LDS from pre-loaded SMEM instead of self-loading.

**Why it failed two ways:**

- **STAGES_C + EPI_REUSE_SMEM (broken):** Tried to overlap residual staging with A/B pipeline
  stages to save SMEM.  But A/B loads at ki=2 overwrite the pre-loaded residual before the
  epilogue reads it.  Unfixable without blocking W0 (deadlock).  `#error` guard added.

- **STAGES_C without SMEM reuse (neutral):** Requires extra SMEM: 4 warps × 2 stages × 4KB
  = 32 KB extra at NEPI=4.  Couldn't fit with NS5 at 4 warps.  Only testable at NEPI=2
  (2 warps), where it was 1.244ms ≈ baseline 1.246ms.  Neutral because 2 warps doing 2×
  work each = same total memory traffic = same K-loop contention.

**What it got wrong:** Still per-warp staging (each warp's residual in a separate SMEM
region).  With 4 warps, the SMEM budget explodes.  CUTLASS uses SHARED staging — one 16 KB
region serves all 4 warps, gated by BAR.SYNC.

### Attempt 5: BAR.SYNC variants (EPI_SYNC, EPI_BAR_PASS, EPI_BAR_CHUNK) — all ~1.635ms

**What they did:** Added BAR.SYNC between epilogue warps at various granularities — between
passes, between chunks, after each STS.

**Why they failed:** BAR.SYNC without shared SMEM is pure overhead.  It serializes warps
(trading parallelism for nothing) because each warp still uses its own SMEM and its own data.
There's nothing to coordinate.  In CUTLASS, BAR.SYNC is the mechanism that gates shared SMEM
reuse — it has a PURPOSE.  Our BAR.SYNCs were empty synchronization with no data sharing.

### Attempt 6: SASS binary patching (crash, then correct but incomplete)

**What it did:** CP-SAT constraint solver optimally reordered epilogue instructions within
each LDTM boundary (scheduling chunks).  Then patched the cubin embedded in the host binary
via `fatbin-patch`.

**Why it crashed:** Two bugs.

1. **Wrong stall field position (fixed):** SM100a control word has stall at bits 53-55 (3-bit,
   range 0-7), NOT bits 0-3 like SM89.  `patch_stall` was overwriting register operands.
   E.g., HFMA2's last register in byte 0 of control word: RZ (0xff) became R241 (0xf1) →
   "illegal instruction."  Fixed to `(ctrl >> 53) & 0x7`.

2. **Instruction reorder (still broken):** Even with correct stall patching, reordering
   128-bit instruction words to new addresses causes "illegal instruction."  Cause unknown —
   may be position-dependent encoding constraints on SM100a that we don't understand.
   Stall-only patches (no reorder) remain untested.

**Even if fixed:** SASS patching can improve STS interleaving (maybe 2× shadow improvement)
but can't bridge the ~5× per-warp gap.  The gap is architectural (LDG vs LDS, 16 vs 4 TMA
loads), not scheduling.

---

## The Missing Experiment

**Nobody has ever tried:** a SHARED residual staging region with BAR.SYNC-coordinated
sub-iteration pipelining.

Every attempt preserved per-warp independent staging.  They varied WHO loads (self vs W0 vs
W2) and WHEN (prefetch vs upfront vs pipelined), but the data always went into per-warp
private SMEM.  The fundamental architectural difference in CUTLASS — one shared buffer,
one TMA load serving all warps, BAR.SYNC gating reuse — was never implemented.

---

## Implementation Plan

### Goal

Build `fc2_w3.cu` — a standalone kernel with identical GEMM core (W0+W1) and a completely new
epilogue matching CUTLASS's shared-SMEM architecture.  Independent from `kernel_body.cuh` and
`kernel_common.cuh`.

### File structure

```
fc2_w3.cu              # Standalone: host setup + kernel + validation
fc2_w3_common.cuh      # SMEM layout, mbar/TMA helpers (extracted + simplified)
fc2_w3_epilogue.cuh    # New epilogue: W2 loader + W3-W6 consumer
Makefile               # New target: fc2-w3
```

### Warp layout (7 warps = 224 threads)

| Warp | Role | Source |
|------|------|--------|
| W0 | TMA A/B loads + tile scheduling | Copy from current W0 |
| W1 | MMA — `tcgen05.mma.cta_group::2` | Copy from current W1 |
| **W2** | **EpilogueLoad** — TMA residual into shared SMEM | **NEW** |
| W3 | Epilogue warp 0 (rows 0-31) | **NEW** |
| W4 | Epilogue warp 1 (rows 32-63) | **NEW** |
| W5 | Epilogue warp 2 (rows 64-95) | **NEW** |
| W6 | Epilogue warp 3 (rows 96-127) | **NEW** |

7 warps (not CUTLASS's 8) because our tile scheduler is combined into W0.

### New SMEM layout

```
A/B pipeline:  5 × 32,768 = 163,840 B          (same as current NS5)
Mbarriers:     ~464 B
  TMA mbar:      5 × 8 = 40 B                   (W0→W1 per-stage)
  MMA mbar:      5 × 8 = 40 B                   (W1→W0 per-stage)
  mainloop mbar: 2 × 8 = 16 B                   (W1→epilogue, double-buffered)
  epilogue mbar: 2 × 8 = 16 B                   (epilogue→W1, double-buffered)
  load mbar:     2 × 8 = 16 B                   (W2→epilogue, double-buffered)
  load_consumed: 2 × 8 = 16 B                   (epilogue→W2, double-buffered)
Bias SMEM:     512 B                             (256 BF16, loaded once at kernel start)
Epi staging:   2 × 16,384 = 32,768 B            (double-buffered shared region)
  Each stage: 128 rows × 64 cols × 2B = 16,384 B (SWIZZLE_128B)
  Used for BOTH residual load AND output store (ReuseSmemC pattern)
──────────────────────────────────────
TOTAL:         ≈ 193 KB  (well within 228 KB, saves 37 KB vs current)
```

### Barrier protocol

```
mainloop_mbar[buf]:    existing — W1 arrives after MMA tile complete → epilogue warps wait
epi_mbar[buf]:         existing — epilogue arrives after all TMEM consumed → W1 waits
load_mbar[s]:          NEW — W2 arrives (with expect_tx) after TMA load → epilogue warps wait
load_consumed_mbar[s]: NEW — epilogue arrives after LDS + STS done → W2 waits before next load

s = sub_iter % 2 (double-buffered: stage 0 and stage 1)
buf = tile_idx % 2 (TMEM double-buffer)
```

### Per-sub-iteration flow

```
W2 (EpilogueLoad):                             W3-W6 (Epilogue compute):
──────────────────                              ────────────────────────
mbar_wait(mainloop_mbar[prev_buf])              mbar_wait(mainloop_mbar[prev_buf])

for si = 0..3:     (4 sub-iters × 64 cols)      for si = 0..3:

  if si >= 2:                                      // WAIT for W2's TMA load:
    mbar_wait(load_consumed[si%2])                 mbar_wait(load_mbar[si%2])
      // wait: previous user freed stage

  // TMA LOAD residual into shared stage:          // TMEM LOAD (2× for 64 cols):
  //   128 rows × 64 cols = 16 KB                 tcgen05.ld.sync x32 chunk[0]
  mbar_arrive_expect_tx(                           tcgen05.ld.sync x32 chunk[1]
    load_mbar[si%2], 16384)
  tma_load_2d_cta(                                 // LDS RESIDUAL from shared SMEM:
    staging[si%2],                                 //   each warp reads its 32-row slice
    &tma_res, col, row,                            ld.shared.v4.b32 × 8
    load_mbar[si%2])                               //   (4 per 32-col chunk × 2 chunks)

                                                   // LDS BIAS from SMEM:
                                                   ld.shared.v4.b32 × 8

                                                   // FP32 COMPUTE:
                                                   FFMA × 64  (32 bias + 32 res adds)
                                                   F2FP × 16  (FP32 → BF16, interleaved)

                                                   // STS to output staging:
                                                   //   (same region if ReuseSmemC,
                                                   //    else separate output region)
                                                   st.shared.v4.b32 × 8 (interleaved)

                                                   // FENCE + SYNC:
                                                   fence.proxy.async.shared::cta
                                                   bar.sync 1, 128  (STS visible)

                                                   // SIGNAL: shared stage free for W2:
                                                   mbar_arrive(load_consumed[si%2])

                                                   // TMA STORE output → global:
                                                   cp.async.bulk.tensor.2d × 1
                                                   cp.async.bulk.commit_group

                                                   // WAIT for TMA store to drain:
                                                   cp.async.bulk.wait_group 0
                                                   bar.sync 1, 128  (SMEM free for STS)

                                                   // last sub_iter only:
                                                   mbar_arrive(epi_mbar[prev_buf])
```

### ReuseSmemC decision

**Option A — separate output staging:** simpler, needs 2 × 16 KB extra = +32 KB.
Total: ~225 KB.  Fits but tight.

**Option B — ReuseSmemC (CUTLASS style):** residual and output share the same 16 KB stage.
Within each sub-iter: W2 loads residual → epilogue LDS → epilogue overwrites with STS output →
TMA store reads output → stage free.  Needs careful ordering: LDS must complete before STS
begins (no extra barrier needed — same warp, sequential code).  Saves 32 KB.
Total: ~193 KB.

**Recommendation:** start with Option A (simpler, still fits).  Switch to B if SMEM pressure
becomes an issue.

---

## Checklist

### Phase 0: Preparation
- [ ] Read and understand W0 loop (`kernel_body.cuh:1806-1842`)
- [ ] Read and understand W1 MMA loop (`kernel_body.cuh:1995-2102`)
- [ ] Read and understand tile scheduling (`kernel_body.cuh` top of `persistent_gemm`)
- [ ] Read and understand host setup in `fc2.cu` (TMA descriptors, launch config, validation)
- [ ] Read CUTLASS epilogue source: `third_party/cutlass/.../sm100_epilogue_tma_warpspecialized.hpp`
- [ ] Verify SMEM budget math against `kernel_common.cuh` macros

### Phase 1: Skeleton — compiles and runs stripped
- [ ] Create `fc2_w3_common.cuh` — extract from `kernel_common.cuh`:
  - [ ] SMEM layout macros (new layout with shared staging)
  - [ ] Mbarrier helpers (mbar_init, mbar_wait, mbar_arrive, mbar_arrive_expect_tx)
  - [ ] TMA helpers (tma_load_2d_cta, smem_to_uint)
  - [ ] Pipeline constants (STAGE_BYTES, N_STAGES, etc.)
  - [ ] TMEM macros (TMEM_LOAD_X32, LOAD_32_COLS, TMEM_WAIT)
  - [ ] Swizzle address computation
- [ ] Create `fc2_w3.cu` — host setup + kernel launch:
  - [ ] Copy host-side TMA descriptor creation from `fc2.cu`
  - [ ] Copy validation / checksum logic
  - [ ] Kernel function with 7 warps (224 threads)
  - [ ] Copy W0 body (TMA A/B loads + tile scheduling)
  - [ ] Copy W1 body (MMA loop)
  - [ ] Stub epilogue: warps 2-6 just `mbar_arrive(epi_mbar)` immediately
  - [ ] New SMEM layout: shared staging region (no per-warp)
  - [ ] Mbarrier init for new barriers (load_mbar, load_consumed_mbar)
- [ ] Add Makefile target: `fc2-w3`
- [ ] Test: compiles with `make fc2-w3`
- [ ] Test on B200: runs, valid=0, no crash, timing ≈ 1.09ms (matches STRIP_EPILOGUE)

### Phase 2: W2 loader warp
- [ ] Implement W2 (EpilogueLoad):
  - [ ] Wait on mainloop_mbar (same as epilogue warps)
  - [ ] For each sub_iter (0..3):
    - [ ] Wait on load_consumed_mbar (stage free) — skip for first 2 sub_iters
    - [ ] TMA load: 128 rows × 64 cols residual into shared staging[si%2]
    - [ ] mbar_arrive(load_mbar[si%2]) with expect_tx
  - [ ] After all 4 sub_iters: participate in epi_mbar_arrive
- [ ] Test: compiles.  W2 loads but epilogue ignores → still valid=0, no crash.

### Phase 3: Epilogue compute warps (W3-W6)
- [ ] Implement epilogue warps:
  - [ ] Wait on mainloop_mbar
  - [ ] For each sub_iter (0..3):
    - [ ] mbar_wait(load_mbar[si%2]) — wait for W2's load
    - [ ] TMEM load: 2× tcgen05.ld.sync x32 (32 cols each, 64 cols per sub_iter)
    - [ ] LDS residual from shared staging (4× ld.shared.v4.b32 per 32-col chunk)
    - [ ] LDS bias from SMEM (4× ld.shared.v4.b32 per chunk, already loaded at kernel start)
    - [ ] FP32 compute: FFMA (bias+residual add) + F2FP (→BF16)
    - [ ] STS to output region (st.shared.v4.b32 × 8)
    - [ ] fence.proxy.async + BAR.SYNC (128 threads)
    - [ ] mbar_arrive(load_consumed[si%2]) — signal W2: stage free
    - [ ] TMA store: cp.async.bulk.tensor.2d + commit_group
    - [ ] cp.async.bulk.wait_group 0 + BAR.SYNC (SMEM free for next STS)
  - [ ] After last sub_iter: mbar_arrive(epi_mbar)
- [ ] SWIZZLE_128B addressing for both residual LDS and output STS
- [ ] Handle lane 0 vs all-thread operations correctly
- [ ] Handle cta_rank (cta_group::2 means 2 CTAs share TMEM)
- [ ] Test on B200: valid=1, checksum=7315328.0

### Phase 4: Drain path
- [ ] Implement drain (last tile) for W2 and W3-W6
  - [ ] Same structure as main path but no next-tile overlap
  - [ ] All warps must complete before kernel exit
- [ ] Test: valid=1 with full tile count

### Phase 5: Measure + compare
- [ ] Benchmark: `./fc2-w3` timing vs `./fc2` vs `./cutlass-bench-fc2-max`
- [ ] Target: < 1.30ms (within 7% of CUTLASS's 1.21ms)
- [ ] If > 1.40ms: SASS dump + compare STS placement vs CUTLASS
- [ ] If STS still clustered: SASS binary patch to interleave

### Phase 6: Optimize (if needed)
- [ ] Try FP32 compute (FFMA) if not already default — measure vs BF16 (HFMA2)
- [ ] Increase to StagesC=3 or 4 (W2 prefetches further ahead)
- [ ] Try removing one BAR.SYNC per sub_iter (combine STS gate + SMEM free gate)
- [ ] Tune TMA store timing (delayed vs inline)
- [ ] SASS analysis: compare instruction scheduling vs CUTLASS
- [ ] If ptxas still clusters STS: SASS binary patch with CP-SAT scheduler

### Phase 7: Integration (only if fc2_w3 wins)
- [ ] Merge new epilogue back into kernel_body.cuh as a compile-time path
- [ ] Parameterize: shared-SMEM epilogue vs per-warp epilogue
- [ ] Verify PE and FC1 still work (they use different epilogue ops)

---

## Key Risks

1. **ptxas STS scheduling.** Even with the new architecture, if ptxas clusters STS at the end
   of each chunk (as it does now), the per-warp active time won't drop as much as CUTLASS.
   CUTLASS's SASS has interleaved STS.  We may need SASS binary patching as a follow-up.

2. **Register pressure.** 7 warps × max_regs.  If max stays at 186, that's 41,664 RF entries
   (63.6% of 65,536).  Should be fine.  But if the new epilogue code inflates regs beyond
   ~220, we may hit occupancy limits.  Monitor with `--ptxas-options=-v`.

3. **BAR.SYNC contention.** Our warp_scaling calibration showed BAR.SYNC adds +25% to mixed
   epilogue microbenchmarks.  But CUTLASS uses it and is fast — the overhead from BAR.SYNC is
   outweighed by the benefit of shared SMEM (fewer TMA loads, LDS instead of LDG).  Still,
   minimize BAR.SYNC count (2 per sub_iter = 8 per tile).

4. **TMA load granularity.** Each W2 TMA load covers 128 rows × 64 cols = 16 KB.  The TMA
   unit may need this to be aligned and fit within tensor map constraints.  Verify that the
   existing `tma_res` descriptor supports this access pattern (it currently loads 32 rows ×
   64 cols = 4 KB per warp — the new load is 4× larger per invocation).

5. **Drain path complexity.** The persistent kernel's drain (last tile for each cluster) has
   special mbarrier handling.  Must replicate correctly for the new barrier protocol.

---

## Hardware Reference (SM100a / B200)

All numbers from calibrated microbenchmarks (`bench/calib/`, `data/warp_scaling.txt`):

| Operation | Throughput | Notes |
|-----------|-----------|-------|
| STS.128 | 27 cyc | st.shared.v4.b32, ILP=1 |
| STS shadow | ≤4 ops free | 8=+55%, 15=+161% |
| LDS.128 | 25 cyc @ILP=1, 3.5 cyc @ILP=7 | ld.shared.v4.b32 |
| HFMA2 | 2 cyc | BF16 fused multiply-add |
| FFMA | 1.5-2 cyc | FP32 fused multiply-add, ~0% STS conflict |
| F2FP | 2 cyc flat | Zero contention at any warp count |
| HADD2 | 2 cyc | BF16 add |
| TMA load | 419 cyc (L2-warm) | cp.async.bulk.tensor |
| TMA store | 197 cyc | cp.async.bulk.tensor |
| TMEM load (x32) | 2 cyc | tcgen05.ld.sync, bandwidth-limited |
| mbar arrive | 2 cyc | mbarrier.arrive |
| mbar wait | 47 cyc | mbarrier.try_wait |
| fence.proxy.async | 10 cyc | Required before TMA store |
| MMA K-iter (pipelined) | 525.6 cyc | With fence+4×MMA+commit+wait |

Multi-warp contention (STS, most relevant):
- 1 warp: 10 cyc, 2 warps: 17 cyc, 4 warps: 28 cyc, 8 warps: 37 cyc
- 4 sub-partitions: warp i%4 maps to sub-partition i%4

SMEM: 228 KB / SM.  RF: 65,536 entries / SM.  148 SMs, 74 clusters.
