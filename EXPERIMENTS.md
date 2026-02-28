# Experiment Log

Kernel: `patch_embed_gemm` — persistent megakernel for SigLIP2 patch embed GEMM.
GEMM: `[928256, 768] x [768, 768]^T`, FP8 E4M3 inputs, BF16 output with fused bias + positional embedding.
Target: B200 (SM100a), 148 SMs, `cta_group::2`, `__cluster_dims__(2,1,1)`.

**Current best: 0.579 ms / 1892 TFLOPS** (fused end-to-end). 223 regs, 0 spills.

---

## Optimization history (committed)

| Commit | Change | Time (ms) | TFLOPS | Regs | Notes |
|--------|--------|-----------|--------|------|-------|
| 6c327fe | Replace TMA stores with st.global.v4 | — | ~300 | — | |
| c5f6c8b | Software-pipeline TMEM loads | — | ~400 | — | |
| 4ff9644 | Prefetch bias+pos_embed into SMEM during K-loop | — | ~500 | — | |
| 4760f3b | Unified epilogue | — | 727 | 147 | 0 spills |
| 892766c | Replace SMEM prefetch with inline BF16 loads | — | 1043 | — | |
| fca9178 | Upgrade cta_group::1 → cta_group::2 | — | 1190 | — | |
| 9557e0b | TN=128→256, single TMEM alloc of 512 cols | — | 1433 | — | |
| abf04a5 | x16→x32 TMEM loads (no double-buffer) | 0.764 | 1433 | — | perf-neutral |
| 6319928 | Remove centralized mainloop_mbar bar.sync | 0.743 | — | — | |
| 521ad55 | 5th epilogue warp via column-split | 0.722 | 1517 | — | |
| cefc59d | Reduce pipeline stages 6→4 | 0.700 | 1564 | 216 | |
| c32ab7a | Epilogue SMEM staging → coalesced stores | 0.633 | 1729 | 222 | |
| d882aba | Phase 2 unroll 4→8 | 0.630 | 1739 | 222 | |
| 9bf658a | Blocked combined relayout | 0.579 | 1892 | 223 | current best |

Reference: cuBLAS per-tensor FP8 (best-of-8 algos, 256MB workspace) = 0.365 ms / 3001 TFLOPS (GEMM only, no fusion).
cuBLAS + unfused pos_embed = 0.835 ms end-to-end.

---

## Experiments — failed or neutral

### Experiment F1: Split x32→2x16 TMEM prefetch

**Date:** 2027-02-27
**Baseline:** 0.630 ms / 222 regs
**Result:** 0.663-0.665 ms (5% regression), 213 regs, 0 spills
**Verdict:** REJECTED

**Hypothesis:** Splitting the x32 TMEM prefetch into two x16 loads and issuing the first half earlier (after CVT_STS consumes a0..a15) would increase overlap with TMEM latency.

**Changes:**
1. Pre-loop: replaced `TMEM_LOAD_X32` with two consecutive `TMEM_LOAD` (x16) calls
2. After first CVT_STS: inserted early prefetch of first x16 (a0..a15 are dead at that point)
3. End-of-loop: replaced x32 prefetch with second-half x16 only (a16..a31)

**Why it failed:** `tcgen05.wait::ld` is a **global fence** — it waits for ALL outstanding TMEM loads to complete. The total latency is dominated by the LAST issued load (the second x16), which sits at roughly the same position as the original x32. The "early" first x16 completes before the fence but doesn't reduce the fence stall. Meanwhile, two separate x16 instructions have more overhead than one x32.

**Observation:** Register count dropped 222→213, suggesting the compiler could schedule better with smaller loads, but the instruction overhead negated this.

---

### Experiment F2: Phase 0 coalesced combined loads (SMEM pre-staging)

**Date:** 2027-02-27
**Baseline:** 0.630 ms / 222 regs
**Result:** 0.858 ms (36% regression), 246 regs, 0 spills
**Verdict:** REJECTED

**Hypothesis:** The 4 global `uint4` combined loads per loop iteration in Phase 1 are uncoalesced (32 lanes each load from a different row, 1536B stride → each generates a separate 128B cache line fill). A "Phase 0" cooperative coalesced pre-load would reduce L1 traffic by ~8x: all 32 lanes load consecutive 16B chunks from the same row, writing to the existing SMEM staging buffer. Phase 1 then reads combined from SMEM instead of global.

**Changes:**
1. Inserted Phase 0 loop (32 iterations, one per row) before TMEM prefetch: each iteration has 32 lanes cooperatively load one row's combined data (coalesced `uint4`/`uint2` loads) and `st.shared.v4` to staging buffer, followed by `__syncwarp()`
2. Replaced 4 global `uint4` loads in Phase 1 with `ld.shared.v4` from staging buffer
3. Removed dead `comb_row` pointer

**SMEM budget:** Zero additional — reused existing staging buffer (16,896 bytes/warp > 16,384 bytes needed for 256 cols x 32 rows x 2B). Read-before-write within each Phase 1 nc iteration ensures correctness.

**Why it failed (badly):**
- The Phase 0 loop (32 iterations x global load + SMEM store, fully unrolled by `#pragma unroll`) generates massive code and register pressure (222→246 regs)
- The `__syncwarp()` barrier adds latency before TMEM prefetch can begin
- The original inline global loads were hitting L1 cache effectively — the combined tensor is only 196x768x2B = 294 KB, small enough for good L1 residency
- Net: Phase 0 cost far exceeded savings from coalesced reads in Phase 1

**Lesson:** L1 cache hit rate on the combined tensor was already adequate. The "uncoalesced" loads were generating extra L2 sectors but not actually stalling — L1 was absorbing most of them. Optimizing L2 sector efficiency is not worthwhile when L1 is the bottleneck.

---

### Experiment F3: 6 epilogue warps

**Baseline:** ~0.722 ms (at 4 epilogue warps)
**Result:** 0.747 ms (regression)
**Verdict:** REJECTED

**Hypothesis:** More epilogue warps = more parallelism in the overlapped epilogue.

**Why it failed:** TMEM bandwidth contention. 6 warps issuing concurrent `tcgen05.ld` saturates TMEM read bandwidth. 5 warps is the sweet spot — enough parallelism without bandwidth saturation.

---

### Experiment F4: Runtime loop bounds for 5th epilogue warp

**Baseline:** ~0.826 ms (at 4 epilogue warps)
**Result:** 0.826 ms (no improvement from the 5th warp)
**Verdict:** REJECTED (fixed by different approach)

**Hypothesis:** Add a 5th epilogue warp by passing `nc_start`/`nc_end` as runtime function arguments.

**Why it failed:** Runtime loop bounds prevented the compiler from unrolling the epilogue loop. Fixed by templating `epilogue_store<NC_START, NC_END>` so the compiler sees constant bounds and fully unrolls. The templated version (committed as 521ad55) gave 0.722 ms.

---

### Experiment F5: SMEM prefetch of combined tensor during K-loop

**Commits:** 4ff9644 → 892766c (committed, then replaced)
**Result:** ~500 TFLOPS → replaced by inline BF16 loads at 1043 TFLOPS
**Verdict:** REPLACED

**Hypothesis:** Prefetching bias+pos_embed into SMEM during the K-loop would hide global memory latency.

**Why it was replaced:** Inline BF16 loads from global were faster, likely because L1 cache hits on the small combined tensor (294 KB) provided sufficient bandwidth without SMEM staging overhead. Removed SMEM prefetch, freeing SMEM capacity.

---

### Experiment F6: 6 pipeline stages (original default)

**Result:** 0.722 ms, 247 regs, 192 KB SMEM
**After reducing to 4 stages:** 0.700 ms, 216 regs, 131 KB SMEM
**Verdict:** 6 stages REJECTED; 4 stages committed (cefc59d)

**Findings:** 6 stages used too many registers (247) and SMEM (192 KB). Reducing to 4 stages freed 64 KB SMEM and dropped to 216 regs, giving 3.1% speedup. 5 stages also tested (0.715 ms, 236 regs) — marginal vs 4.

---

### Experiment F7: Centralized bar.sync for mainloop mbar broadcast

**Baseline:** 0.764 ms
**After (independent polling):** 0.743 ms
**Verdict:** Independent polling WINS (committed 6319928)

**Original approach:** Warp 2 polls the mainloop mbarrier, then uses `bar.sync` to broadcast completion to W3-W5. Cost 10.3% stall from the bar.sync.

**Replacement:** All epilogue warps poll the mainloop mbarrier independently. Eliminates the centralized synchronization bottleneck.

---

### Experiment F8: x32 TMEM loads (vs x16)

**Commit:** abf04a5
**Result:** Performance-neutral (0.764 ms with both x16 and x32)
**Verdict:** Committed (neutral, simpler code)

**Hypothesis:** x32 loads would reduce instruction count by issuing one TMEM load instead of two.

**Finding:** The kernel is bandwidth-bound on TMEM reads, not instruction-bound. Halving instruction count doesn't help when each load takes the same total TMEM bandwidth.

---

### Experiment F9: cta_group::4

**Result:** Not attempted
**Verdict:** IMPOSSIBLE

**Reason:** Would require 1024 TMEM columns (256 cols x 4 CTAs), exceeding the hardware limit of 512 TMEM cols per SM.

---

### Experiment F10: Software-pipeline combined tensor loads

**Date:** 2026-02-28
**Baseline:** 0.630 ms / 1739 TFLOPS / 222 regs
**Result:** 0.630 ms / 1738 TFLOPS (neutral), 224 regs, 0 spills
**Verdict:** NEUTRAL — no improvement, reverted

**Hypothesis:** The 4 uncoalesced global `uint4` loads for the combined tensor in Phase 1 are issued and immediately consumed by `BF16X2_TO_F32`, causing ~100-200 cycle stalls per load before the warp even reaches `TMEM_WAIT()`. Software-pipelining — issuing iteration N+1's loads at the start of iteration N — would hide these stalls behind the previous iteration's TMEM_WAIT + FADD + CVT_STS work.

**Changes:**
1. Prologue: declared `uint4 next_c0..next_c3`, issued 4 global loads for the first iteration's combined data before `TMEM_LOAD_X32`
2. Loop body: `c0..c3 = next_c0..c3` (register rename in unrolled code), then issue NEXT iteration's 4 loads guarded by `if (nc + 32 < NC_END)`, then convert from `c0..c3` (no stall — data arrived during previous iteration)
3. Both halves of the combined data (cols 0-15 from `c0,c1` and cols 16-31 from `c2,c3`) use the same pipeline — all 4 loads prefetched together, eliminating the original code's reuse of `craw0/craw1` across halves

**Register cost:** +2 regs (222→224). The proposal estimated +16 to +24, but the compiler performed effective register renaming across the fully-unrolled iterations (8 iters for `<0,256>`, 4 for `<0,128>`/`<128,256>`). No spills, 31 regs below the 255 ISA ceiling.

**Why it didn't help:** The combined load stalls are **not on the critical path**. The epilogue is overlapped with the K-loop via double-buffered TMEM and mainloop mbarriers. The K-loop (6 iterations × 4 MMAs × TMA loads) takes longer than the epilogue (8 nc iterations of TMEM readback + add + store). Epilogue warps finish and wait at `mainloop_mbar` regardless — the combined load stalls were occurring inside this slack time, so eliminating them saves nothing on the wall clock.

**Key insight:** This confirms the kernel is **K-loop-bound, not epilogue-bound**. Epilogue optimizations (proposals 2, 3, 4) targeting combined load efficiency or warp balance are unlikely to improve wall time unless they also reduce K-loop latency or eliminate the epilogue entirely. The only epilogue improvement that would help is one that makes the epilogue so much faster that it fits within a single tile's K-loop with room to spare — currently there's slack, but the epilogue is already inside it.

---

### Experiment F11: Combined loads bypass L1 via `.cg` (on top of F10 pipelining)

**Date:** 2026-02-28
**Baseline:** 0.630 ms / 1739 TFLOPS / 222 regs (F10 pipelined: 0.630 ms / 224 regs)
**Result:** 0.908 ms / 1206 TFLOPS (44% regression), 216 regs, 0 spills
**Verdict:** REJECTED

**Hypothesis:** L1 tex throughput at 85% is the primary bottleneck. The combined tensor (196×768×2B = 294 KB) slightly exceeds L1 capacity. Replacing the 4 uncoalesced `uint4` combined loads with `ld.global.cg.v4.b32` (cache-global, bypass L1) would route them through L2 (at 54% utilization) instead, relieving L1 pressure for Phase 2's coalesced stores. Software pipelining from F10 would hide the longer L2 latency (~200 cycles vs ~50 for L1 hits).

**Changes (on top of F10):**
1. Added `LDG_CG_V4(reg, ptr)` macro wrapping `ld.global.cg.v4.b32` inline PTX
2. Replaced all 8 combined load sites (4 in prologue + 4 in loop body) with `LDG_CG_V4`

**Register note:** Regs dropped from 224 (F10) to 216 — the `.cg` qualifier gives the compiler more scheduling freedom since these loads don't interact with L1 cache state, allowing tighter register live ranges. Compile time also dropped 210ms → 143ms.

**Why it failed catastrophically:** The proposal's own risk analysis was exactly right. `.cg` does not fix the uncoalesced access pattern — it moves 128 scattered cache line fetches per warp from L1 to L2. With 148 SMs × 5 epilogue warps × 4 loads/iteration × 8 iterations, the aggregate L2 scatter traffic is enormous: each warp's 4 `uint4` loads generate 128 scattered L2 accesses (16 KB of traffic for 2 KB of useful data, 8× amplification). The 54% L2 headroom was completely inadequate for this traffic pattern. The combined tensor was getting good L1 cache hits in the baseline (294 KB fits marginally, and access patterns have temporal locality within a tile), so bypassing L1 traded fast cache hits for slow L2 scatter — and the pipelining couldn't hide the aggregate L2 contention across all SMs.

**Compounding factor:** Since F10 showed the epilogue isn't on the critical path anyway, the `.cg` regression doesn't just slow the epilogue — it makes the epilogue **longer than the K-loop**, flipping the overlap balance. The epilogue becomes the new critical path, and the 44% regression reflects the full exposed epilogue cost.

---

### Experiment F12: TN=128 revisit with current epilogue stack

**Date:** 2026-02-28
**Baseline:** 0.630 ms / 1739 TFLOPS / 222 regs
**Result:** Best = 0.702 ms / 1560 TFLOPS (N_STAGES=6, NUM_EPI_WARPS=4) — 11.4% regression
**Verdict:** REJECTED

**Hypothesis:** TN=128 was last tested at 1190 TFLOPS under a completely different kernel (x16 TMEM loads, no SMEM staging, no 5th warp). The current epilogue (SMEM-staged coalesced stores, x32 TMEM loads) might tip the balance differently. TN=128 halves TMEM cols (512→256), potentially reducing `long_scoreboard` stalls (4.4%). Freed SMEM (~80 KB) enables deeper pipelines (up to N_STAGES=6 with 4 epilogue warps).

**Changes:**
- TN: 256→128, TILES_N: 3→6, TOTAL_TILES: 10,878→21,756
- TMEM_COLS: 512→256 (single alloc, double-buffered)
- IDESC: 0x10400010→0x10200010 (N=128 encoding, from old commit fca9178)
- STAGE_BYTES: 32,768→24,576 (A=16KB, B=8KB)
- NUM_EPI_WARPS: 5→4 (4 row_groups map perfectly to 4 warps, no column-split)

**Sweep results (NUM_EPI_WARPS=4):**

| N_STAGES | ms | TFLOPS | Regs | vs baseline |
|---|---|---|---|---|
| 4 | 0.763 | 1436 | 215 | -21% |
| 5 | 0.765 | 1432 | 235 | -21% |
| 6 | 0.702 | 1560 | 242 | -11.4% |

NUM_EPI_WARPS=5 was not viable: with TN=128, the column-split gives NC_COLS=64 → COLS_PER_THREAD=2, which breaks the V2 store path (ld.shared.v2.b32 reads 8 bytes but each thread owns only 4 bytes → overlapping writes). NUM_EPI_WARPS=3 also not viable: only 3 warps for 4 row_groups leaves row_group 3 unprocessed.

**Why it failed:** The regression is caused by **tile transition overhead dominance**, not registers (regs actually dropped to 215 at 4-stage):

1. **2× more tiles** (21,756 vs 10,878) with the **same per-tile overhead** (mbarrier wait + phase flip + tile index computation + snake reorder + TMEM prefetch setup + epilogue mbar signal)
2. **Half the compute per tile** — each MMA accumulates into 128 TMEM cols instead of 256. Same 24 MMA instructions per tile (6 K-iters × 4 MMAs), but each does less useful work
3. **Compute-to-overhead ratio ~2× worse** — the total FLOP count is identical, just spread across 2× more tiles with fixed per-tile cost
4. Deeper pipelines recovered some loss (6-stage: 0.763→0.702 ms) by pre-buffering all 6 K-iterations, but N_STAGES=6=K_ITERS is the ceiling — no further hiding possible
5. SMEM/TMEM savings (80 KB freed, TMEM 512→256) provided no benefit because the bottleneck is tile-level scheduling overhead, not memory capacity

**Key insight:** TN=256 is fundamentally superior for this GEMM shape. The 256-wide tile amortizes per-tile overhead over 2× more compute. TN=128 is now ruled out definitively — tested under both old (1190 TFLOPS) and current (1560 TFLOPS peak) architectures, and loses both times for the same structural reason.

---

### Experiment F13: Blocked combined relayout (COMMITTED)

**Date:** 2026-02-28
**Baseline:** 0.630 ms / 1739 TFLOPS / 222 regs
**Result:** 0.579 ms / 1892 TFLOPS (+8.1%), 223 regs, 0 spills
**Verdict:** COMMITTED (9bf658a)

**Hypothesis:** The combined tensor `[196, 768]` row-major forces each lane's `uint4` load to stride 1536 bytes across rows, scattering L1 cache lines. Relaying into a blocked `[7, 24, 32, 32]` format (32×32 blocks, rows padded to 224) makes each nc iteration read from a contiguous 2 KB block, improving L1 spatial locality. Since L1 tex is at 85%, reducing L1 pressure from combined loads should free bandwidth for K-loop TMA/MMA traffic.

**Changes:**
1. `precompute_combined` (host-side, outside timed region): outputs blocked layout. Rows padded to 224 (ceil(196/32)*32), padding rows wrap via `row % 196`.
2. `epilogue_store`: blocked addressing replaces `comb_row = combined + pos_row * N_DIM + n_start`. New: `comb_base` points into the block row, `comb_ptr` per-nc steps by `COMB_BLOCK_ELEMS` (1024). Within each nc iteration, offsets +0/+8/+16/+24 read 32 contiguous BF16 values.
3. Allocation: `SEQ_LEN * N_DIM` → `COMB_PADDED_ROWS * N_DIM` (+43 KB device memory).

**ncu profiling (vs 0.630 ms baseline):**

| Metric | Baseline | F13 | Delta |
|---|---:|---:|:---:|
| **L1 tex** | **85.0%** | **67.8%** | **-17.2%** |
| L2 | 54.0% | 49.3% | -4.7% |
| DRAM | 27.0% | 31.7% | +4.7% |
| selected (issuing) | 19.1% | 22.9% | **+3.8%** |
| long_scoreboard (TMEM) | 4.4% | 4.7% | +0.3% |
| short_scoreboard (SMEM) | 1.1% | 0.57% | **-0.53%** |
| wait (TMA) | 1.2% | 1.5% | +0.3% |
| sleeping | 1.1% | 0.96% | -0.14% |
| barrier | 0.8% | 0.81% | unchanged |

**Why it worked:** L1 tex dropped 85% → 68% — the single largest L1 reduction from any change on this kernel. The blocked layout eliminated L1 cache line scatter: each nc iteration now touches 2 contiguous cache lines (64 bytes in a 2 KB block) instead of 32 scattered lines across a 49 KB address range. The freed L1 bandwidth directly accelerated the K-loop (shared L1 with TMA/MMA traffic), raising productive issue rate from 19.1% → 22.9%. SMEM stalls halved (1.1% → 0.57%) due to less L1 backpressure. DRAM up slightly because the kernel runs faster (same data, less time).

---

### Experiment F14: Combined load L1 bypass (3 variants, all failed)

**Date:** 2026-02-28
**Baseline:** 0.579 ms / 1892 TFLOPS / 223 regs (F13)
**Verdict:** ALL REJECTED

**Hypothesis:** F13 reduced L1 from 85% → 68%. Eliminating combined loads from L1 entirely (via SMEM staging or cache hints) would free additional L1 bandwidth for K-loop traffic.

**Variant A — cp.async.cg SMEM staging:**
- Preloaded combined data into the existing staging buffer via `cp.async.cg.shared.global` (bypasses L1, goes L2→SMEM). Issued all preloads before Phase 1, waited with `cp.async.wait_all`, then Phase 1 reads combined from SMEM via `ld.shared.v4`.
- **Result:** 0.809 ms / 1353 TFLOPS (**+40% regression**), 184 regs, 0 spills.

**Variant B — ld.global.cg (bypass L1, cache in L2 only):**
- Replaced the 4 `reinterpret_cast<const uint4*>` combined loads with `ld.global.cg.v4.u32` inline PTX. Same code structure as F13, just a cache hint change.
- **Result:** 0.615 ms / 1780 TFLOPS (**+6% regression**), 197 regs, 0 spills.

**Variant C — ld.global.cs (streaming/evict-first in L1):**
- Same as B but with `.cs` hint (keeps L1 access but uses evict-first policy).
- **Result:** 0.743 ms / 1474 TFLOPS (**+28% regression**), 197 regs, 0 spills.

**Why ALL variants failed — the same root cause:**

F13's blocked layout already solved the combined load problem. After relayout, each nc iteration reads 64 contiguous bytes from a 2 KB block — near-perfect L1 locality with ~100% hit rate at ~30 cycle latency. These loads overlap with TMEM latency (~200 cycles) in the existing code: they're issued before `TMEM_WAIT`, so their 30-cycle L1 latency is fully hidden.

Every L1-bypass approach breaks this overlap:

- **Variant A:** The `cp.async.wait_all` serialized the preload — the epilogue couldn't start Phase 1 until all combined data arrived in SMEM (~300+ cycles). This pushed the epilogue past the K-loop shadow, making it the new critical path. Additionally, SMEM reads (~20 cycles) are *too fast* to fill the TMEM overlap window — `TMEM_WAIT` now stalls for the full ~200 cycles with nothing useful hiding the latency.
- **Variant B:** Going from L1 (~30 cycles) to L2 (~100-200 cycles) increased per-load latency. The loads still overlap with TMEM, but the longer latency slightly extended the epilogue. Since the epilogue was already near the K-loop boundary, even a small extension spilled into wall clock.
- **Variant C:** The streaming buffer has worse hit rates than the main L1 cache on Blackwell. Combined data that previously hit L1 at 30 cycles now thrashed in the streaming buffer, causing frequent L2 fallbacks. Worst of both worlds: still in L1 path (no bandwidth relief) but with worse latency.

**Meta-lesson:** Once the blocked layout gives near-perfect L1 locality, there is nothing left to optimize on the combined load path. The L1 utilization drop (85% → 68%) was the blocked layout eliminating L1 *capacity pressure*, not leaving headroom for further improvement. L1 at 68% is the new equilibrium — the remaining L1 traffic is from the K-loop itself (TMA fills, MMA traffic), not from combined loads.

---

## Profiling data — SMEM staging A/B comparison

Profiled via `ncu --set detailed`, single kernel instance.
Raw data: `baseline.csv` (pre-staging), `after.csv` (post-staging). Run `python compare.py baseline.csv after.csv`.

### Build stats (ptxas)

| Metric | Baseline (st.global.v8) | SMEM staging |
|--------|:-----------------------:|:------------:|
| Registers/thread | 216 | 222 |
| Spills | 0 | 0 |
| Stack | 16 bytes | 16 bytes |
| SMEM (dynamic) | ~131 KB | ~211 KB |

### Warp stall breakdown (% of peak sustained active)

| Stall reason | Baseline | Staging | Delta | Notes |
|---|---:|---:|---:|---|
| **selected (issuing)** | **14.1%** | **19.1%** | **+35%** | Warps doing productive work |
| long_scoreboard (TMEM) | 6.4% | 4.4% | -30% | Less L1 store backpressure |
| short_scoreboard (SMEM) | 0.1% | 1.1% | +1.0pp | SMEM staging ld/st chains |
| sleeping | 1.3% | 1.1% | -15% | Less idle between tiles |
| wait (TMA) | 0.9% | 1.2% | +0.3pp | Slightly more TMA pressure |
| barrier | 0.8% | 0.8% | unchanged | |
| mio_throttle | 0.0% | 0.03% | negligible | |

### Memory throughput

| Subsystem | Baseline | Staging | Delta |
|---|:---:|:---:|:---:|
| **L1 tex** | **82%** | **85%** | +3% (primary ceiling) |
| L2 | 60% | 54% | -11% |
| DRAM | 24% | 27% | +13% |

### Uncoalesced access analysis

| Metric | Baseline | Staging |
|---|---:|---:|
| Excessive L2 sectors | 44.56M | 44.56M |
| % of total L2 sectors | **50%** | **33%** |
| Global store instructions | 1,392,384 | 3,480,960 |

Absolute excess sector count unchanged (44.56M — from split warps' V2 stores and combined BF16 global loads). Fraction dropped 50% → 33% because coalesced V4/V2 stores generate more "good" sectors. Store instruction count increased 2.5x (V4 16B replaces V8 32B), but each is fully coalesced.

### Key compare.py findings (>5% relative change)

| Metric | Baseline | Staging | Change |
|--------|----------|---------|--------|
| Inst executed per cycle (active) | 82.9 | 112.3 | **+35%** |
| SMEM shared ld instructions | 666 | 3,481,626 | +523K% (Phase 2) |
| SMEM shared st instructions | 148 | 2,784,916 | +1.9M% (Phase 1) |
| Shared memory L1 wavefronts | 1,554 | 22,279,698 | +1.4M% |
| SMEM L1 pipe utilization | 0.1% | 20.6% | new workload |
| L1 hit rate | 61.7% | 33.6% | -46% (fewer RMW) |
| Instruction cache requests | 849K | 3,510K | +314% (larger code) |

### SMEM budget

| Component | Bytes |
|-----------|-------|
| Pipeline (4 stages x 32KB) | 131,072 |
| TMEM addr + mbarriers | 128 |
| Staging (5 warps x 16,896) | 84,480 |
| **Total (SMEM_BYTES)** | **215,808** |
| SM limit | 233,472 (228 KB) |
| **Headroom** | **~17 KB** |

### Why SMEM staging won (+9.6%)

The uncoalesced `st.global.v8` stores caused:
1. **100% excess L2 sectors** — 32 sectors per request vs 16 ideal
2. **L1 read-modify-write** — each 128B cache line received only 32 bytes (25% fill)

SMEM staging fixes #2 completely and makes #1 neutral. L1 hit rate drop (61.7% → 33.6%) confirms RMW elimination. Cost: SMEM traffic (20.6% L1 shared pipe), 1.1% short_scoreboard stalls, 2.5x store instructions. Net: **warp issue rate +35%, TMEM stalls -30%**.
