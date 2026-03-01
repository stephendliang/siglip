# Experiment Log

Kernel: `patch_embed_gemm` — persistent megakernel for SigLIP2 patch embed GEMM.
GEMM: `[928256, 768] x [768, 768]^T`, FP8 E4M3 inputs, BF16 output with fused bias + positional embedding.
Target: B200 (SM100a), 148 SMs, `cta_group::2`, `__cluster_dims__(2,1,1)`.

**Current best: 0.530 ms / 2067 TFLOPS** (fused end-to-end). 236 regs, 0 spills.

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
| 9bf658a | Blocked combined relayout | 0.579 | 1892 | 223 | |
| dbe8c72 | F15+F16: epilogue phase timing + 4 epilogue warps | 0.560 | 1955 | 219 | |
| d1fc103 | F18: double-buffered SMEM epilogue overlap | 0.543 | 2018 | 223 | current best |
| — | F19: early epilogue mbar + TMA Phase 2B stores | 0.542 | 2020 | 223 | early mbar neutral, TMA stores rejected |
| — | F20: next-tile TMA prefetch | 0.544 | 2014 | 221 | rejected — ~0% improvement |
| b5da9e8 | F22: BF16 epilogue arithmetic | 0.536 | 2041 | 229 | `#pragma unroll 2`, -1480 epilogue SASS |
| 2bb3675 | F28: K-loop restructuring | 0.536 | 2041 | 229 | perf-neutral, -76 cyc K-loop, cleaner baseline |
| — | F24: Swizzled staging + TMA tensor stores | 0.532 | 2059 | 235 | asymmetric layout, Phase 2B -626 cyc |
| — | F30: Staging_b swizzle address precomputation | 0.532 | 2059 | 235 | no-op — compiler already hoisted; source cleanup only |
| ef2aa9c | F31: Per-warp Phase 1 stagger | 0.530 | 2067 | 236 | clock64 spin, STAGGER=80, contention-based |
| — | F34: Parallel TMEM load diagnostic | 0.531 | 2064 | 248 | 2×x16 before single WAIT — loads pipeline |

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

---

### Experiment F15: Epilogue phase profiling (DIAGNOSTIC)

**Date:** 2026-02-28
**Baseline:** 0.579 ms / 1892 TFLOPS / 223 regs (5 epilogue warps)
**Build:** `make timing` (`-DTIMING`), 240 regs, 0 spills — timing build only, no baseline change
**Verdict:** DIAGNOSTIC — data used to guide F16 and F17

**Goal:** Determine whether Phase 1 (TMEM readback + BF16 add + CVT → SMEM) or Phase 2 (SMEM transpose → coalesced global store) dominates the 9,271-cycle epilogue overrun identified by clock64 W1 timing.

**Instrumentation:** Added `clock64()` timestamps to W3 (ew=1, full 256-col warp, representative of the bottleneck warps W3-W5). Three points:
- `t0`: After `mainloop_mbar` wait (start of epilogue work)
- `t1`: Just before `__syncwarp()` inside `epilogue_store` (end of Phase 1)
- `t2`: After `epilogue_store` returns (end of Phase 2)

Timing parameter `long long& t_phase1_end` added to `epilogue_store` under `#ifdef TIMING`. Accumulates min/max/sum per cluster for both phases. Timing buffer expanded from [74×8] to [74×16] values.

**Results (with F16 applied — NUM_EPI_WARPS=4):**

```
=== EPILOGUE PHASE TIMING (W3/ew=1, 10804 tiles across 74 clusters) ===
  Per-tile averages:
    Phase 1 (TMEM→SMEM):   4,762 cycles / 2,268 ns  (81.6%)
    Phase 2 (SMEM→global): 1,071 cycles /   510 ns  (18.4%)
    Total epilogue:         5,833 cycles / 2,778 ns
  Phase 1 range: min=3,115 max=10,542 (3.4× spread)
  Phase 2 range: min=669   max=2,276  (3.4× spread)
```

**Conclusion:** **Phase 1 dominates at 81.6%.** TMEM readback (`tcgen05.ld.x32` + `tcgen05.wait::ld`) is the epilogue bottleneck. Phase 2 (coalesced global stores via SMEM) is only 18.4% — not worth optimizing in isolation. This triggered Path A (TMEM contention reduction → F16) and ruled out Phase 2-focused optimizations.

---

### Experiment F16: NUM_EPI_WARPS 5→4 (COMMITTED)

**Date:** 2026-02-28
**Baseline:** 0.579 ms / 1892 TFLOPS / 223 regs (5 epilogue warps)
**Result:** 0.560 ms / 1955 TFLOPS (+3.3%), 219 regs, 0 spills
**Verdict:** COMMITTED

**Hypothesis:** With 5 warps, W2 and W6 split row_group 0 (128 cols each), while W3-W5 each handle a full row_group (256 cols). The split warps finish early, but the epilogue mbar only fires when the LAST warp arrives. The 5th warp adds TMEM read contention without helping the critical path. With 4 warps, each handles exactly one row_group with full 256 cols — equal work, no wasted early-finish, less TMEM contention.

**Changes:** Single-line: `#define NUM_EPI_WARPS 4` (was 5). Everything auto-derives:
- `THREADS`: 32 × (2+4) = 192 (was 224)
- `SMEM_BYTES`: saves 16,896 bytes (one fewer staging buffer)
- `epilogue_mbar` expected arrivals: 4 × 2 × 32 = 256 (was 320)
- `is_split`: `(row_group < (4-4))` = always 0 → all warps call `epilogue_store<0, TN>` (full 256 cols, no column splitting)

**W1 timing (F16 vs baseline):**

| Metric | Baseline (5 warps) | F16 (4 warps) | Delta |
|---|---:|---:|:---:|
| Epilogue mbar wait | 2,466 cycles | 2,381 cycles | -3.4% |
| TMA stage-0 wait | 292 cycles | 318 cycles | +8.9% |
| K-loop | 4,046 cycles | 4,073 cycles | +0.7% |
| Total tile | 6,805 cycles | 6,773 cycles | -0.5% |
| Wall clock | 0.579 ms | 0.560 ms | **-3.3%** |

**Why it worked:** Removing the 5th warp reduced TMEM read port contention during Phase 1. The epilogue mbar wait dropped 85 cycles (-3.4%), and register pressure dropped (223→219, freeing 4 regs). The wall clock improvement (3.3%) exceeds the tile-level improvement (0.5%) because reduced TMEM contention also improves tail-latency tiles (max K-loop dropped from 9,464 to the same range, and max tile time spread tightened from 4.9× to 4.8×).

---

### Experiment F17: Direct CVT_STG stores (eliminate Phase 2)

**Date:** 2026-02-28
**Baseline:** 0.560 ms / 1955 TFLOPS / 219 regs (F16, 4 epilogue warps)
**Result:** 0.568 ms / 1929 TFLOPS (-1.4% regression), 215 regs, 0 spills
**Verdict:** REJECTED

**Hypothesis:** F15 showed Phase 2 (SMEM→global) takes only 1,071 cycles (18.4%). Eliminating it entirely by reverting to direct `CVT_STG` (`st.global.v8`) stores would cut the epilogue from 5,833 to ~4,762 cycles — potentially fitting within the K-loop shadow (tile time = 6,773 cycles). The L1 headroom from F13's blocked layout (68% vs old 85%) should absorb the uncoalesced store pressure.

**Changes:**
1. Replaced `CVT_STS` (st.shared) with `CVT_STG` (st.global.v8) in `epilogue_store`
2. Added `row_ptr = C + (gm_base + lane) * N_DIM + n_start` for direct addressing
3. Removed `__syncwarp()` and Phase 2 loop entirely
4. Removed `staging_saddr` parameter from all call sites

**Timing comparison:**

| Metric | F16 (SMEM staging) | F17 (direct stores) | Delta |
|---|---:|---:|:---:|
| Phase 1 (TMEM→store) | 4,762 cycles | 5,688 cycles | **+19.4%** |
| Phase 2 (SMEM→global) | 1,071 cycles | 18 cycles | -98.3% |
| Total epilogue | 5,833 cycles | 5,706 cycles | -2.2% |
| Epilogue mbar wait | 2,381 cycles | 2,503 cycles | +5.1% |
| Wall clock | 0.560 ms | 0.568 ms | **+1.4%** |

**Why it failed:** The uncoalesced `st.global.v8` stores compete with TMEM reads for L1 bandwidth. Each warp's 32-byte store scatters across 32 different cache lines (one per thread, each thread writes to a different row with 1536-byte stride). This generates 4KB of L1 traffic for 1KB of useful data — 4× amplification. The L1 pressure backs up into the TMEM readback pipeline, inflating Phase 1 from 4,762 to 5,688 cycles (+926). This nearly cancels the Phase 2 elimination (1,071 cycles saved), netting only a 127-cycle (2.2%) epilogue reduction — but the increased L1 contention also slows the K-loop's TMA traffic, causing a net regression.

**Key insight:** SMEM staging is not just an optimization for store coalescing — it isolates TMEM reads from global store traffic on the L1 bus. Phase 1's `st.shared` stores use the SMEM path (no L1 contention), while Phase 2's coalesced `st.global.v4` stores are bandwidth-efficient when they do hit L1. Replacing both with uncoalesced `st.global.v8` removes this isolation, causing TMEM and store traffic to contend on L1.

**Lessons for future experiments:** Any epilogue optimization must preserve the SMEM staging architecture. Approaches that might still help:
- Double-buffered SMEM staging (overlap Phase 2 of first half with Phase 1 of second half) → **implemented as F18**
- TMA multicast for B matrix (frees L2/DRAM bandwidth, indirectly reduces contention)
- Larger TM (e.g., 192 or 256) to amortize per-tile overhead over more compute

---

### Experiment F18: Double-buffered SMEM epilogue overlap (COMMITTED)

**Date:** 2026-02-28
**Baseline:** 0.560 ms / 1955 TFLOPS / 219 regs (F16, 4 epilogue warps)
**Result:** 0.543 ms / 2018 TFLOPS (+3.2%), 223 regs, 0 spills
**Verdict:** COMMITTED

**Hypothesis:** F15 showed Phase 1 (TMEM→SMEM) takes 4,762 cycles (81.6%) and Phase 2 (SMEM→global) takes 1,071 cycles (18.4%). During Phase 1, each TMEM_WAIT stalls ~200 cycles waiting for `tcgen05.ld.x32` to complete. If we split the 256-col epilogue into two 128-col halves with separate staging buffers, we can interleave Phase 2 stores from the first half into Phase 1's TMEM_WAIT stall windows during the second half — hiding Phase 2A latency behind TMEM readback latency.

**Changes:**

SMEM layout:
- `STAGING_HALF_ROW_BYTES = 128 * 2 + 16 = 272` (was `STAGING_ROW_BYTES = 528`)
- Two half-buffers per warp: `staging_a` and `staging_b = staging_a + 8704`
- `STAGING_WARP_BYTES = 2 × 8704 = 17,408` (was 16,896)
- Total epilogue staging: 4 × 17,408 = 69,632 bytes (+2,048 vs baseline)

Three-phase pipeline in `epilogue_store`:
1. **Phase 1A**: TMEM readback cols 0–127 → staging_a (4 iterations, same code as before but half-width)
2. **Overlap**: Issue TMEM_LOAD for second half before `__syncwarp()`. Then interleave: each of 4 Phase 1B iterations processes 8 Phase 2A rows (ld.shared.v2 + st.global.v2 from staging_a) before TMEM_WAIT, filling the TMEM latency window. Phase 1B writes to staging_b.
3. **Phase 2B**: 32 rows coalesced stores from staging_b (same as old Phase 2 but half-width, v2 instead of v4)

Phase 2 stores use `COALESCED_STORE_V2` (128 cols / 32 lanes = 4 BF16 per thread = 8 bytes = v2) instead of `COALESCED_STORE_V4` (256 cols → 8 per thread → v4).

**Timing comparison:**

| Metric | F16 baseline | F18 overlap | Delta |
|---|---:|---:|:---:|
| **Wall clock** | **0.560 ms** | **0.543 ms** | **-3.0%** |
| **TFLOPS** | **1955** | **2018** | **+3.2%** |
| Epilogue mbar wait | 2,381 cycles | 1,530 cycles | **-35.7%** |
| TMA stage-0 wait | 318 cycles | 576 cycles | +81% |
| K-loop | 4,073 cycles | 4,017 cycles | -1.4% |
| Total tile | 6,773 cycles | 6,124 cycles | **-9.6%** |
| Epilogue Phase 1 | 4,762 cycles | 4,102 cycles | **-13.9%** |
| Epilogue Phase 2 | 1,071 cycles | 824 cycles | -23.1% |
| **Total epilogue** | **5,833 cycles** | **4,926 cycles** | **-15.6%** |
| Regs | 219 | 223 | +4 |

**Why it worked:** The overlap hides Phase 2A's coalesced global stores (ld.shared + st.global, using SMEM port + LSU) inside Phase 1B's TMEM_WAIT stalls (waiting on TMEM port — different hardware path). 8 Phase 2A rows (~264 cycles) inserted per TMEM iteration fill the ~200-cycle TMEM stall window. Total epilogue dropped 907 cycles (15.6%). The epi_mbar_wait — the time W1 blocks waiting for epilogue to release TMEM — dropped 851 cycles (35.7%), directly reducing tile time by 649 cycles (9.6%).

The increased TMA stage-0 wait (318→576) is expected: with faster tiles, W1 reaches the TMA wait sooner, occasionally before TMA data arrives. This is a secondary effect, not a regression — it means the K-loop's compute budget is better utilized.

**SMEM budget:**

| Component | Bytes |
|---|---|
| Pipeline (4 stages × 32 KB) | 131,072 |
| TMEM addr + mbarriers | 128 |
| Staging (4 warps × 17,408) | 69,632 |
| **Total (SMEM_BYTES)** | **~201 KB** |
| SM limit | 228 KB |
| **Headroom** | **~27 KB** |

---

### Experiment F19: Early epilogue mbar signal + TMA Phase 2B stores

**Date:** 2026-02-28
**Baseline:** 0.543 ms / 2018 TFLOPS / 223 regs (F18)
**Result:** Early mbar alone: 0.542 ms (~0% change). TMA stores: 0.573 ms (+5.5% regression, reverted).
**Verdict:** Early mbar KEPT (structurally sound, neutral performance). TMA stores REJECTED.

#### Motivation

The epilogue overruns the K-loop by ~909 cycles (epilogue 4,926 vs K-loop 4,017). Phase 2B (staging_b → global via `ld.shared.v2 + st.global.v2`, 824 cycles) doesn't touch TMEM — only SMEM reads and global stores. Moving the `epilogue_mbar` arrive from after Phase 2B to after Phase 1B's `__syncwarp()` should let W1 reclaim TMEM sooner, with Phase 2B running overlapped with the next K-loop.

Predicted speedup: ~7% (fixed-point analysis: epi_mbar_wait 1,530 → 1,119 → tile time 6,124 → 5,712).

#### F19a: Early epilogue mbar signal (KEPT)

**Change:** Pass `epi_mbar_addr` into `epilogue_store`. Call `mbar_arrive()` after Phase 1B's `__syncwarp()` (after all TMEM reads, before Phase 2B). Drain epilogue passes 0 (no signal needed). Removed standalone `mbar_arrive` from caller.

**Timing:**

| Metric | F18 baseline | F19a early mbar | Delta |
|---|---:|---:|:---:|
| **Wall clock** | **0.543 ms** | **0.542 ms** | **~0%** |
| Epilogue mbar wait | 1,530 cycles | 1,122 cycles | **-26.7%** |
| TMA stage-0 wait | 576 cycles | 818 cycles | +42.0% |
| K-loop | 4,017 cycles | 4,187 cycles | +4.2% |
| Total tile | 6,124 cycles | 6,128 cycles | ~0% |
| Overhead (epi+tma0) | 2,107 cycles | 1,941 cycles | -7.9% |
| Phase 1 | 4,102 cycles | 4,133 cycles | +0.8% |
| Phase 2B | 824 cycles | 825 cycles | +0.1% |
| Total epilogue | 4,926 cycles | 4,958 cycles | +0.6% |

**Why it's neutral:** The early mbar signal works exactly as designed — epi_mbar_wait dropped 408 cycles (26.7%). But Phase 2B's 128-thread parallel `ld.shared.v2 + st.global.v2` stores now overlap with the K-loop and add L1/LSU contention (+170 cycles to K-loop). The remaining 242-cycle improvement shifted to TMA stage-0 wait (W1 arrives sooner, TMA data isn't ready yet — moving idle time from one mbar to another). Net: overhead dropped 166 cycles, K-loop grew 170 cycles. Perfect cancellation.

**Kept because:** Structurally correct, no performance cost, and would pay off if Phase 2B's LSU contention could be eliminated (motivating F19b).

#### F19b: TMA bulk stores for Phase 2B (REJECTED)

**Hypothesis:** Replace Phase 2B's 32-thread manual stores with `cp.async.bulk.global.shared::cta.bulk_group` TMA stores (lane 0 only, 32 × 256B per warp). The TMA/DMA engine uses an independent hardware path from the SMSP's LSU, eliminating the K-loop contention that made F19a neutral.

**Changes:**
- Replaced Phase 2B's `COALESCED_STORE_V2` loop with 32 × `cp.async.bulk.global.shared::cta.bulk_group` stores (lane 0 only, 256 bytes each = 128 BF16 cols per row)
- Added `cp.async.bulk.commit_group` after the 32 stores
- Added `cp.async.bulk.wait_group 0` before Phase 1B+2A loop (protects staging_b from being overwritten while TMA reads are in flight)
- Added `cp.async.bulk.wait_group 0` after drain epilogue (ensures completion before kernel exit)
- 226 regs (+3), 0 spills

**Timing:**

| Metric | F18 baseline | F19b TMA stores | Delta |
|---|---:|---:|:---:|
| **Wall clock** | **0.543 ms** | **0.573 ms** | **+5.5%** |
| Epilogue mbar wait | 1,530 cycles | 2,714 cycles | +77.4% |
| TMA stage-0 wait | 576 cycles | 160 cycles | -72.2% |
| K-loop | 4,017 cycles | 3,509 cycles | **-12.6%** |
| Total tile | 6,124 cycles | 6,384 cycles | +4.2% |
| Phase 1 | 4,102 cycles | 3,590 cycles | -12.5% |
| Phase 2B | 824 cycles | 2,455 cycles | **+198%** |
| Total epilogue | 4,926 cycles | 6,045 cycles | **+22.7%** |

**Why it regressed:** `cp.async.bulk.global.shared::cta` has ~70 cycles per-instruction overhead. With 32 stores × ~70 cycles = ~2,240 cycles, the TMA approach is 3× slower than 32 threads doing parallel v2 stores (~825 cycles). TMA bulk copies are designed for large transfers (128 KB tile loads), not many small 256-byte row copies. The K-loop DID improve (-12.6%), confirming the LSU contention hypothesis — but the epilogue regression more than offset it.

The padded SMEM layout (STAGING_HALF_ROW_BYTES = 272 = 256 useful + 16 pad, needed for bank-conflict-free Phase 1B writes where consecutive lanes must map to different banks: lane 0 → bank 0, lane 1 → bank 4, lane 2 → bank 8 with 272-byte stride; without pad, 256-byte stride maps ALL lanes to bank 0) prevents using a single `cp.async.bulk.tensor.2d` TMA tensor store, which would amortize the overhead across the full 8 KB transfer. The tensor store requires SMEM layout to match the descriptor's swizzle mode (contiguous 256-byte rows with `CU_TENSOR_MAP_SWIZZLE_NONE`), incompatible with our 272-byte padded rows.

**Key insight:** Phase 2B's L1/LSU contention with the K-loop is real (confirmed by K-loop improving 12.6% when Phase 2B uses TMA), but the only way to exploit this is a SMEM layout that's both bank-conflict-free for Phase 1B writes AND contiguous for TMA tensor stores. This requires swizzled SMEM (e.g., `CU_TENSOR_MAP_SWIZZLE_128B`), which would be a major restructuring of the staging buffer architecture.

---

### Experiment F20: Next-tile TMA prefetch

**Date:** 2026-02-28
**Baseline:** 0.542 ms / 2020 TFLOPS (F19a early mbar), 223 regs, 0 spills
**Result:** 0.544 ms / 2014 TFLOPS, 221 regs, 0 spills (16-byte stack frame)
**Verdict:** REJECTED — zero meaningful improvement

**Hypothesis:** W1's TMA stage-0 wait (818 cycles) exists because the DRAM latency for loading tile N+1's A matrix exceeds the epilogue_mbar overlap window. By having W0 issue TMA loads for tile N+1's ki=0 at the END of tile N (after the K-loop), the TMA gets a head start. By the time W1 finishes epilogue_mbar wait for tile N+1, stage 0 data should already be in SMEM.

**Changes:**
- Separated W0's K-loop: ki=0 handled standalone (conditionally skipped if prefetched), ki=1..5 in a constant-bounds loop (preserves compiler unrolling)
- After ki=5, W0 computes next tile's coordinates (snake ordering), waits for `mma_mbar[0]` (W1 consumed ki=4), issues TMA loads for next tile's ki=0 into stage 0
- mbarrier phase tracking verified across multiple tiles: prefetch consumes `mma_mbar[0]` at end of tile N, so tile N+1 skips ki=0's mbar wait + TMA load; `tma_phase` unaffected (W1 consumer, not W0 producer)

**Implementation note:** First attempt used `for (int ki = ki_start; ...)` with runtime `ki_start = prefetched`. This prevented loop unrolling: 196 regs (+96-byte stack frame) → 0.784 ms (44% regression). Fixed by splitting ki=0 from ki=1..5 with constant loop bounds: 221 regs, 16-byte stack frame.

**Timing:**

| Metric | F19a baseline | F20 prefetch | Delta |
|---|---:|---:|:---:|
| **Wall clock** | **0.542 ms** | **0.544 ms** | **~0%** |
| Epilogue mbar wait | 1,122 cycles | 1,146 cycles | +24 (noise) |
| TMA stage-0 wait | 818 cycles | 797 cycles | **-21 (noise)** |
| K-loop | 4,187 cycles | 4,175 cycles | -12 (noise) |
| Total tile | 6,128 cycles | 6,120 cycles | -8 (noise) |
| Phase 1+2A | 4,133 cycles | 4,107 cycles | -26 (noise) |
| Phase 2B | 825 cycles | 817 cycles | -8 (noise) |
| Total epilogue | 4,958 cycles | 4,924 cycles | -34 (noise) |

**Why it failed:** The prefetch provides essentially zero head start. The key insight is understanding the concurrent timeline:

```
Without prefetch (normal flow):
  W0 enters tile N+1 → ~20 cycles overhead → mma_mbar[0] check (already passed) → issues TMA
  W1 enters tile N+1 → epilogue_mbar wait (1,122 cycles) → tma_mbar[0] wait (818 cycles)
  TMA had ~1,122 cycles of flight time during epi_mbar_wait
  DRAM latency ≈ 1,122 + 818 ≈ 1,940 cycles

With prefetch:
  W0 end of tile N → mma_mbar[0] wait → issues TMA → ~20 cycles loop transition → enters tile N+1
  The prefetch issues TMA only ~20-30 cycles before normal flow would have
  TMA had ~1,142 cycles of flight time → TMA0_wait ≈ 1,940 - 1,142 = 798 (vs 818)
```

W0 already issues ki=0 TMA loads at the very start of each tile body, concurrently with W1's epilogue_mbar wait. The ~1,122-cycle overlap window is substantial but still insufficient to cover the ~1,940-cycle DRAM latency for the A matrix. The prefetch only shifts the TMA issue point by the tile loop transition overhead (~20-30 cycles) — negligible against the 818-cycle shortfall.

**What would actually help TMA0_wait:** The 818-cycle gap is a DRAM bandwidth limitation, not a scheduling problem. Reducing it requires either: (1) TMA multicast for B matrix (halves B bandwidth, frees DRAM for A), (2) larger tile dimensions to amortize per-tile DRAM access, or (3) L2 residency controls to keep B cached.

---

### Experiment F21: L2 promotion for B matrix

**Date:** 2026-03-01
**Baseline:** 0.543 ms / 2018 TFLOPS / 223 regs (F18+F19a)
**Result:** No change (TMA0_wait: 787 avg vs 794 avg baseline, within noise)
**Verdict:** REJECTED — B is already L2-resident

**Hypothesis:** B matrix is 576 KB (768×768 FP8) with high reuse (only 3 N-tiles, all 74 clusters read the same B data). A matrix is 680 MB streaming with zero reuse. A's streaming traffic may evict B from L2. Setting `CU_TENSOR_MAP_L2_PROMOTION_L2_128B` on the B TMA descriptor should hint the hardware to keep B lines in L2, reducing TMA0_wait.

**Change:** Single line in B TMA descriptor setup:
```c
CU_TENSOR_MAP_L2_PROMOTION_NONE  →  CU_TENSOR_MAP_L2_PROMOTION_L2_128B
```
A descriptor unchanged (`_NONE` — promoting 680 MB streaming A would evict B).

**Build verification:** 223 regs, 0 spills (production). 252 regs, 0 spills (timing). Checksum 1769472.0 correct.

**Timing (3 runs each, timing build):**

| Run | Baseline TMA0 | L2_128B TMA0 |
|-----|---:|---:|
| 1 | 787 | 766 |
| 2 | 780 | 804 |
| 3 | 814 | 791 |
| **avg** | **794** | **787** |

Difference: 7 cycles — pure noise. Wall clock identical (0.535 ms timing build both variants).

**Why it failed:** B is already fully L2-resident without any hint. B200 has 48 MB L2 cache; B matrix is 576 KB (1.2% of L2). Despite A's 680 MB streaming traffic, the hardware LRU policy keeps B hot — B's high reuse rate (every cluster reads it every 3 tiles) ensures it's always recently-accessed and never evicted.

TMA0_wait (~790 cycles) is pure A-matrix DRAM latency: each tile loads 16 KB of A from a 680 MB streaming dataset with zero reuse. No L2 policy can help — A misses are mandatory.

**Implications:**
- **F26 (L2 persistence window): DEAD** — per fail-fast rule, if F21 shows zero change, B is already L2-hot and reserving L2 capacity via `cudaAccessPropertyPersisting` adds nothing.
- TMA0_wait is confirmed as a pure DRAM bandwidth cost for A, not an L2 eviction problem.

---

### Experiment F22: BF16 Epilogue Arithmetic

**Date:** 2026-03-01
**Baseline:** 0.543 ms / 2018 TFLOPS / 223 regs (F18+F19a)
**Result:** 0.536 ms / 2041 TFLOPS / 229 regs — **1.3% faster**
**Verdict:** ACCEPTED (marginal)

**Hypothesis:** Phase 1 post-TMEM_WAIT path does 76 instructions per 32-col chunk: 8 BF16X2_TO_F32 unpacks → 16 FP32 adds → CVT_STS, repeated for second half. Replacing this with BF16-native arithmetic (`cvt.rn.bf16x2.f32` + `add.bf16x2` + `st.shared.v4`) eliminates the unpack+add chain, reducing to ~36 ops per chunk.

**Changes:**
1. Added `cvt_add_bf16x2()` inline helper using C++ intrinsics (`__floats2bfloat162_rn` + `__hadd2`)
2. Added `STS_V4` macro for `st.shared.v4.b32`
3. Rewrote Phase 1A and Phase 1B loop bodies — eliminated `BF16X2_TO_F32`, `float s0..s15` temps, and FP32 scalar adds
4. Added `#pragma unroll 2` to both Phase 1A and Phase 1B loops (critical for register pressure — see below)

**Precision:** Checksum 1769472.0 — **exact match**. The double-rounding (`round_bf16(round_bf16(acc) + combined)` vs `round_bf16(acc + combined_as_fp32)`) happened to produce identical results at the test values.

**SASS analysis (epilogue only, K-loop is structurally identical):**

| Instruction | Baseline | F22 | Delta |
|-------------|----------|-----|-------|
| PRMT (bf16 unpack) | 411 | 27 | **-384** |
| FADD (fp32 scalar add) | 769 | 1 | **-768** |
| HADD2/HFMA2 (bf16x2 add) | 1 | 385 | +384 |
| F2FP (cvt f32→bf16x2) | 385 | 385 | 0 |
| Epilogue SASS total | 4,502 | 3,022 | **-1,480** |
| K-loop SASS total | 394 | 394 | **0** |

**Register pressure investigation — the main complication:**

The naive implementation (full unroll, default) hit **255 registers** despite eliminating 1,480 instructions. Three approaches were tried — all hit 255:

| Approach | Regs | Why 255 |
|----------|------|---------|
| Full-width asm macro (25 operands) | 255 | Too many simultaneous inputs |
| Half-width asm macro (13 operands) | 255 | Still too many cross-barrier live values |
| C++ intrinsics + STS_V4 | 255 | Intermediates become global physical regs |

**Root cause:** In the baseline, `CVT_STS` uses asm-local `.reg .b32 b0..b7` — the 385 F2FP outputs are invisible to ptxas's global allocator. With C++ intrinsics, every `F2FP` and `HADD2` output becomes a global physical register. The fully-unrolled loop (4 iters × 2 phases × 4 template instantiations) keeps multiple iterations' intermediates alive across `asm volatile` barriers.

PTX virtual register comparison confirms this:

| | Baseline | F22 (full unroll) |
|---|---------|-------------------|
| `%r` (int b32) | 1,307 | **2,891** (+1,584) |
| `%f` (float) | 2,305 | 1,537 (-768) |

The BF16 math shifted work from float to int register space, and the asm-local → global exposure added ~816 net virtual registers.

**The fix: `#pragma unroll 2`** on both Phase 1 loops. This limits cross-iteration register liveness while still allowing the compiler to interleave 2 iterations for ILP:

| Unroll factor | Regs | Production ms | TFLOPS |
|---------------|------|--------------|--------|
| Full (default) | 255 | 0.547 | 2000 |
| 1 (no unroll) | 198 | 0.614 | 1783 |
| **2** | **229** | **0.536** | **2041** |

Full unroll has lowest instruction count but worst register pressure — the register bloat negates the instruction savings. No unroll has best register pressure but too much loop overhead. Unroll 2 is the sweet spot.

**Production timing (5 runs):** 0.535–0.538 ms, median 0.537 ms / 2040 TFLOPS.

**Timing build caveat:** The timing build (`-DTIMING`) hits 255 regs (clock64 instrumentation pushes over the edge), which distorts the cycle breakdown. The production build at 229 regs is the authoritative performance measurement.

**Lessons learned:**
- Exposing asm-local intermediates to the global register allocator (via C++ intrinsics or smaller asm blocks) can cause severe register inflation even when peak liveness is lower.
- Full loop unrolling is not always optimal — register pressure from cross-iteration liveness can outweigh the ILP benefits. The compiler's default full-unroll heuristic is tuned for occupancy-limited kernels; for register-constrained kernels at 1 CTA/SM, partial unrolling is superior.
- Register count is a better predictor of performance than instruction count for this epilogue-bound kernel.

---

## F23C: 2-warp epilogue contention test — ✗ REJECTED (contention <10%)

**Hypothesis:** TMEM contention between 4 simultaneous epilogue warps accounts for a significant fraction of Phase 1 time. Reducing to 2 warps (each processing 2 row_groups sequentially) should reduce per-rg Phase 1 time enough to compensate for doubled work.

**Changes:** `NUM_EPI_WARPS=2`, THREADS=128 (was 192). Each warp loops over 2 row_groups sequentially, reusing the same staging buffer. Mbar arrive only after second rg (both TMEM reads complete). Drain epilogue same pattern. Epilogue mbar expected arrivals: 2×2×32=128 (was 256).

**Build:** 210 regs (was 229), 0 spills, 16-byte stack frame. SMEM staging 35 KB (was 70 KB).

**Result: 42% REGRESSION.** 0.759 ms / 1443 TFLOPS (was 0.536 ms / 2041 TFLOPS). Checksum correct.

**Cycle breakdown (timing build, 250 regs):**

| Metric | 4 warps (F22 baseline) | 2 warps (F23C) | Delta |
|--------|----------------------|----------------|-------|
| Phase 1 | 4,524 | 8,357 | **1.85x** |
| K-loop | 4,066 | 3,513 | -14% (better!) |
| TMA0 | 585 | 81 | -86% (much better) |
| epi_wait | 1,570 | 5,579 | +4,009 |
| Wall clock | 0.536 ms | 0.759 ms | +42% |

**Fail-fast triggered:** Per-warp Phase 1 = 1.85x (≥1.8x threshold). TMEM contention at 4 warps is **<10% of Phase 1 time**. The binding constraint is raw TMEM readback bandwidth per request, not queueing between concurrent warps.

**Notable side effects of 2 warps:**
- K-loop improved 14% (3,513 vs 4,066): lower register pressure (210 vs 229) and reduced scheduling contention with fewer threads
- TMA0 dropped 86% (81 vs 585): with 128 threads vs 192, W0/W1 have more scheduling bandwidth
- ml_wait dropped from 1,149 to 404: epilogue barely waits for W1 because it's so much slower

These improvements prove that register pressure and thread count significantly impact W1 performance — but the doubled epilogue work overwhelms them.

**Implication:** All warp-count variants (3, 2, or 1 epilogue warps) are ruled out. The TMEM contention hypothesis is dead. 4 warps is optimal for this tile geometry.

**Key insight:** Phase 1 is bandwidth-limited, not contention-limited. The TMEM read port serves 4 concurrent requesters with <10% overhead. Optimization efforts must focus on reducing per-warp Phase 1 work (instruction count, scheduling) rather than reducing the number of concurrent TMEM readers.

---

## F28: K-loop restructuring — descriptor precomputation + manual unroll — PERF-NEUTRAL

**Hypothesis:** Eliminating `make_smem_desc` recomputation (12 calls/tile × 5-8 instructions), `ki % N_STAGES` modulo, `ki == 0` conditional, and runtime accumulate predicate reduces K-loop overhead enough to be measurable.

**Changes:**
1. Precomputed `desc_a_base[N_STAGES]` and `desc_b_base[N_STAGES]` before tile loop — descriptors are constant across all tiles (SMEM base addresses never change).
2. Replaced `for (int ki = 0; ki < K_ITERS; ki++)` with 6 explicit blocks using `K_ITER_ACCUM(S)` macro. Stage index S is a compile-time constant per block.
3. ki=0: constant-false predicate (`setp.ne.b32 p, 0, 0`) clears accumulator. ki=1-5: constant-true predicate (`setp.ne.b32 p, 1, 0`) accumulates. Eliminates runtime `accumulate` variable.
4. Sub-MMA inner loop kept as `for (int sub = 1; sub < MMA_PER_KI; sub++)` — compiler auto-unrolls (MMA_PER_KI=4 is constant).

**Build:** 229 regs (unchanged from baseline), 0 spills. Timing build: 255 regs, 0 spills.

**Result: PERF-NEUTRAL.** 0.536 ms / 2041 TFLOPS (identical to F22 baseline). Checksum correct: 1769472.0.

**Cycle breakdown (timing build, 255 regs):**

| Metric | F22 baseline | F28 | Delta |
|--------|-------------|-----|-------|
| K-loop | 4,154 | 4,078 | **-76 (-1.8%)** |
| TMA0 | 857 | 600 | -257 (noise/run variance) |
| epi_wait | 1,056 | 1,532 | +476 |
| Phase 1 | 4,129 | 4,524 | +395 |
| ml_wait | 1,139 | 1,141 | +2 (noise) |
| Phase 2B | 899 | 641 | -258 |
| W1 total | 6,067 | 6,211 | +144 |
| Epilogue total | 6,167 | 6,306 | +139 |

**Analysis:** K-loop saved 76 cycles (modest). However, the timing build uses 255 regs (maxed out) vs 229 for production — cycle data is distorted by register pressure and should not be taken at face value. The unchanged wall clock (0.536 ms) at 229 regs is the ground truth.

The kernel remains epilogue-bound. K-loop savings alone cannot shift the equilibrium — as predicted by the timing model. The change is retained as a cleaner K-loop baseline (eliminates descriptor recomputation, modulo, conditional) for potential future pairing with epilogue optimizations.

**What the compiler did:** Despite adding 8 `uint64_t` precomputed descriptors (expected +16 regs), register count stayed at 229. The compiler likely kept the descriptors in the same registers previously used for `make_smem_desc` results — the precomputation hoisted the computation without inflating live ranges.

---

## F24: Swizzled SMEM staging + TMA tensor stores for Phase 2B — ACCEPTED (+0.7%)

**Date:** 2026-03-01
**Baseline:** 0.536 ms / 2041 TFLOPS / 229 regs (F22+F28)
**Result:** 0.532 ms / 2059 TFLOPS / 235 regs — **+0.7% faster** (5 runs: 0.531–0.533 ms)
**Verdict:** ACCEPTED (marginal)

**Hypothesis:** F19a proved Phase 2B's manual `ld.shared.v2 + st.global.v2` stores contend with the concurrent K-loop on L1/LSU (~170 cycles). F19b proved TMA bulk stores eliminate the contention but 32 individual `cp.async.bulk` stores at ~70 cycles each = 2,240 cycles (3x slower). Using `cp.async.bulk.tensor.2d` TMA tensor stores, a single instruction moves an entire 64-col × 32-row region (4 KB), amortizing per-instruction overhead. Requires SWIZZLE_128B-compatible SMEM layout in staging_b.

**Changes (3 iterations to reach final form):**

*Iteration 1 — full swizzle (both staging_a and staging_b):*
1. Replaced `STAGING_ROW_PAD` / `STAGING_HALF_ROW_BYTES` (272-byte rows) with two 64-col SWIZZLE_128B regions per 128-col staging half. Each region: 32 rows × 128 bytes = 4,096 bytes.
2. Phase 1A/1B STS_V4 writes use swizzle XOR addressing: `byte_offset ^ ((lane & 7) << 4)`.
3. Phase 2A reads use matching swizzle XOR for `ld.shared.v2`.
4. Phase 2B replaced with 2 × `cp.async.bulk.tensor.2d.global.shared::cta.bulk_group` (64 cols × 32 rows each) + `commit_group`.
5. Added `const __grid_constant__ CUtensorMap tma_c` kernel parameter with SWIZZLE_128B descriptor for output matrix C.
6. TMA completion fence (`cp.async.bulk.wait_group 0`) between Phase 1A and Phase 1B to protect staging_b from overwrite while TMA reads it.

**Result: 0.538 ms / 2036 TFLOPS, 242 regs.** Perf-neutral. Phase 2B dropped -627 cycles (-70%) but Phase 1 regressed +466 cycles (+11.3%), nearly cancelling the benefit.

*Root cause analysis of Phase 1 regression:*

The +466 cycle Phase 1 regression was initially attributed to bank conflicts from the swizzle XOR pattern. Analysis proved this wrong — the old 272-byte stride layout had **identical** 4-way write bank conflicts (`gcd(272/4, 32) = gcd(68, 32) = 4`). Phase 2A reads were actually *better* (conflict-free with swizzle vs 2-way before). The real culprits:

| Source | Est. cycles | Mechanism |
|--------|-------------|-----------|
| Swizzle XOR computation (Phase 1A+1B) | ~100-150 | Region select, byte_base, XOR per loop iteration × 8 iterations |
| Phase 2A swizzle reads | ~100-150 | XOR + ternary region branch per read × 32 reads |
| TMA `wait_group` + extra `__syncwarp()` | ~30-50 | Inserted between Phase 1A and Phase 1B |
| `memory` clobber on `wait_group` asm | ~100-200 | Compiler can't keep registers live across barrier |

Key insight: **staging_a doesn't need swizzle.** Only staging_b feeds the TMA tensor stores. Phase 2A uses manual `COALESCED_STORE_V2`. The swizzle was imposed on staging_a for no benefit.

*Iteration 2 — asymmetric layout (Remedy A) + relocated TMA wait (Remedy B):*

**Remedy A:** Reverted staging_a to linear 272-byte rows (original layout, 4-way bank conflicts, simple addressing). Kept staging_b in SWIZZLE_128B layout for TMA stores. Asymmetric per-warp layout: 8,704 bytes (staging_a, linear) + 8,192 bytes (staging_b, swizzled) = 16,896 bytes.

**Remedy B:** Moved TMA `wait_group 0` from between Phase 1A/Phase 1B to the start of `epilogue_store` (before Phase 1A). After `ml_wait` (~1,342 cycles), TMA stores from the previous tile are long completed — the wait is a true no-op. This eliminates the extra `__syncwarp()` and moves the `memory` clobber away from the critical Phase 1A→1B register optimization window.

**Result: 0.532 ms / 2059 TFLOPS, 235 regs.** This is the accepted form.

**Final cycle breakdown (timing build, 255 regs):**

| Metric | F22+F28 baseline | F24 initial (all swizzled) | F24 final (A+B) | Delta (final vs baseline) |
|--------|:-:|:-:|:-:|:-:|
| **Wall clock** | **0.536 ms** | **0.538 ms** | **0.532 ms** | **-0.7%** |
| K-loop | 4,154 | 4,118 | 4,107 | **-47 (-1.1%)** |
| Phase 1 | 4,129 | 4,595 | 4,569 | **+440 (+10.7%)** |
| Phase 2B | 899 | 272 | 273 | **-626 (-69.6%)** |
| TMA0 | 857 | 762 | 658 | **-199 (-23.2%)** |
| epi_wait | 1,056 | 1,244 | 1,352 | +296 |
| ml_wait | 1,139 | 1,342 | 1,360 | +221 |

**What worked:**
- **Phase 2B: -626 cycles.** TMA tensor stores are dramatically more efficient than 32 manual `ld.shared.v2 + st.global.v2` iterations. Two 4KB TMA stores + commit = 273 cycles vs 899 cycles of LSU-issued stores.
- **K-loop: -47 cycles.** Modest improvement from eliminating LSU contention between Phase 2B stores and K-loop memory traffic. Smaller than the expected -170 from F19a's measurement — TMA stores use a DMA path that has some residual L1 interaction.
- **TMA0: -199 cycles.** Unexpected bonus. Possibly from reduced SMEM/L1 pressure (smaller staging footprint, fewer LSU stores) freeing memory subsystem bandwidth for TMA loads.

**What didn't work — Phase 1: +440 cycles.** The staging_b swizzle XOR computation adds ~100-150 cycles of instruction overhead (region select, byte_base, XOR per STS_V4). This is the cost of SWIZZLE_128B-compatible writes. The asymmetric layout (Remedy A) eliminated the staging_a half of this overhead, but staging_b's swizzle cost remains. The 272-byte baseline was essentially free (simple `lane * stride + offset`).

**Why wall clock improved despite wider deficit:**

The cycle budget now shows a wider epilogue deficit (1,164 cycles vs baseline 257), yet wall clock improved. This seems paradoxical. The explanation: the timing build uses 255 regs (maxed out), which inflates Phase 1 disproportionately relative to the production build at 235 regs. The production build's equilibrium is tighter than the timing data suggests. The ground truth is wall clock at production register count: 0.532 ms vs 0.536 ms = real improvement.

**Per-warp Phase 1 (timing build):**
```
W2 (ew=0, rg=0):  avg=4,533  p95=5,856
W3 (ew=1, rg=1):  avg=4,569  p95=5,842
W4 (ew=2, rg=2):  avg=4,863  p95=6,027
W5 (ew=3, rg=3):  avg=4,779  p95=6,002
Spread of averages: 330 cycles
```

Spread increased from F25's baseline 172 to 330 cycles. Now asymmetric (above 200-cycle threshold). This may reopen F27 (dephasing) as a viable optimization, though the asymmetry could be a timing-build artifact.

**SMEM budget:**
- Previous: 4 warps × 17,408 = 69,632 bytes staging. ~201 KB total.
- F24: 4 warps × 16,896 = 67,584 bytes staging. ~199 KB total of 228 KB.

**Production timing (5 runs):** 0.531, 0.532, 0.533, 0.533, 0.533 ms. Median 0.533 ms / 2055 TFLOPS.

**Lessons learned:**
- Asymmetric staging layouts (linear for manual stores, swizzled for TMA stores) are superior to uniform swizzle. Only swizzle what TMA touches.
- Placement of `memory` clobber asm barriers matters. A `cp.async.bulk.wait_group 0` between Phase 1A and Phase 1B prevented register optimization across the critical boundary. Moving it before Phase 1A (where it's a no-op after ml_wait) eliminated this cost.
- STS_V4 (`st.shared.v4.b32`) requires 16-byte alignment, which constrains staging_a row stride to multiples of 16. The optimal bank-conflict-minimizing stride with 16-byte alignment is 272 bytes (gcd(68, 32) = 4 → 4-way conflicts). Stride 260 (zero conflicts) is only 4-byte aligned — causes misaligned address trap.
- The +440 cycle Phase 1 regression from staging_b swizzle is a genuine cost of TMA tensor stores. The trade is acceptable here (Phase 2B savings + TMA0 bonus > Phase 1 cost), but future work should explore ways to reduce swizzle addressing overhead.

---

## F30: Staging_b swizzle address precomputation (no-op — compiler already hoisted)

**Hypothesis:** The +440 cycle Phase 1 regression from F24 includes ~40 instructions of redundant per-iteration swizzle address computation. Hoisting loop-invariant values (`xor_val_b`, row base addresses) before the Phase 1B loop should save ~50-100 cycles.

**Changes:**
1. Precomputed `xor_val_b = (lane & 7) << 4`, `saddr_row_b_lo`, `saddr_row_b_hi` before the `#pragma unroll 2` loop
2. Replaced 5-line per-iteration address block (region_base_b ternary + lane multiply + xor recompute) with 3-line version (ternary selects between precomputed row bases)

**Result: 0.532 ms / 2059 TFLOPS, 235 regs.** Performance-neutral. Identical SASS (modulo register renumbering).

**SASS verification:** `cuobjdump --dump-sass` diff between old and new binaries shows:
- Same total instruction count (2864)
- Same opcode histogram (LOP3: 256, SHF: 69, STS: 67, ISETP: 44)
- Same total SASS lines (5760)
- Only register number differences (e.g., R168→R164, R174→R172)

**Conclusion:** nvcc at `-O3` was already hoisting all three loop-invariant computations. The source change makes the invariance explicit (fewer lines, clearer intent) but produces identical machine code. The +440 cycle Phase 1 regression from F24 is NOT from redundant swizzle address computation — the overhead comes from the swizzle XOR operations themselves within the STS_V4 address operands and/or register pressure from staging_b pointers displacing other scheduling-critical values.

**Kept as source cleanup** — shorter, clearer Phase 1B loop body (3 lines vs 5 lines per iteration).

---

## F29: PACK::16b TMEM Load Mode (rejected — produces zeros)

**Hypothesis:** Replace `tcgen05.ld.sync.aligned.32x32b.x32.b32` (32 FP32 regs) with `tcgen05.ld.sync.aligned.32x32b.x16.pack::16b.b32` (16 BF16x2 regs). PACK converts FP32→BF16 during TMEM readback and packs pairs, eliminating 128 CVT instructions and freeing 16 registers.

**Changes:**
1. New `TMEM_LOAD_PACK` macro: `.x16.pack::16b.b32` with `"=r"` constraints (16 regs)
2. New `pack_add_bf16x2` helper: `__hadd2` on packed BF16x2 (no FP32→BF16 CVT)
3. Variable declarations: `uint32_t p0..p15` (16 regs) instead of `float a0..a31` (32 regs)
4. All 4 `TMEM_LOAD_X32` → `TMEM_LOAD_PACK`, all `cvt_add_bf16x2` → `pack_add_bf16x2`

**Build result:** 222 regs (down from 235), 0 spills. PTX and SASS assembled successfully. SASS showed correct `LDTM.x16.PACK16BIT` instructions and 126 `HADD2.BF16_V2` instructions (vs 128 CVT instructions in baseline). Only 1 `F2FP` instruction remaining.

**Result: Checksum = 0.0 (FAIL).** Output is all zeros. Raw PACK output (bypassing combined add) also all zeros. The PACK instruction reads zeros from TMEM with FP32 accumulators.

**Root cause:** CUTLASS's copy traits reveal that `pack::16b` is ONLY defined for `16dp` tile shapes (`16x64b`, `16x128b`, `16x256b`) — there are NO `SM100_TMEM_LOAD_32dp*_16b` variants. The `32x32b.pack::16b` combination assembles in PTX/SASS but does not correctly read FP32 accumulator data. The hardware semantics appear undefined or incompatible for `32x32b` + `pack::16b`.

**Why `16x64b` won't work for us:** The `16x64b` shape operates on a half-warp (16 threads), interleaving column pairs. Our per-thread TMEM layout from `32x32b` MMA uses all 32 threads. Switching to `16x64b` would require a complete epilogue restructure with only half the threads active.

**Conclusion:** `pack::16b` is not viable for our `32x32b` TMEM layout. Phase 1 CVT elimination requires a different approach (inline PTX `cvt.rn.bf16x2.f32` + fused add, or TMEM layout change to `16x64b` with warp restructure).

---

## F31: Per-warp Phase 1 stagger — 0.530 ms / 2067 TFLOPS (+0.4%)

**Hypothesis:** The 330-cycle step function between W2/W3 (fast, ~4,550 cycles) and W4/W5 (slow, ~4,820 cycles) in Phase 1 is caused by warp scheduling contention — lower warp IDs win TMEM read port arbitration. Staggering Phase 1 start times across warps should spread out the initial burst of `tcgen05.ld` instructions and reduce queueing.

### Phase 1: Diagnostic (rg-swap)

Swapped W2↔W4 row_group assignments (`row_group = (ew==0)?2:(ew==2)?0:ew`) to determine causality:

| Warp | Baseline (rg=ew) | Swap (rg swapped) | Follows |
|------|---|---|---|
| W2 (ew=0) | 4,533 (fast, rg=0) | 4,451 (fast, rg=2) | warp ID |
| W3 (ew=1) | 4,569 (fast, rg=1) | 4,553 (fast, rg=1) | unchanged |
| W4 (ew=2) | 4,863 (slow, rg=2) | 4,842 (slow, rg=0) | warp ID |
| W5 (ew=3) | 4,779 (slow, rg=3) | 4,799 (slow, rg=3) | unchanged |

**Verdict: CONTENTION.** Timing follows warp ID, not row_group. W2 stays fast even when processing rg=2 (previously slow). W4 stays slow even when processing rg=0 (previously fast). The asymmetry is scheduling arbitration, not structural TMEM column or SMEM address effects.

### Phase 2: Stagger sweep

After `mbar_wait` + fence, each warp spins for `ew * STAGGER_CYCLES` via `clock64()` before Phase 1. Lane 0 only, followed by `__syncwarp()`.

| STAGGER_CYCLES | Wall clock | TFLOPS | vs baseline |
|---|---|---|---|
| 0 (baseline) | 0.532 ms | 2059 | — |
| 50 | 0.531 ms | 2063 | -0.2% |
| **80** | **0.530 ms** | **2067** | **-0.4%** |
| 100 | 0.530 ms | 2065 | -0.4% |
| 200 | 0.532 ms | 2060 | 0% |

Selected STAGGER_CYCLES=80 (theoretical optimum = 330 / 4 warps = 82.5).

### clock64 timing (STAGGER=80 vs baseline)

```
Baseline:                                   F31 (STAGGER=80):
  W2: avg=4,533  p95=5,856                   W2: avg=4,284  p95=5,581  (-249)
  W3: avg=4,569  p95=5,842                   W3: avg=4,345  p95=5,567  (-224)
  W4: avg=4,863  p95=6,027                   W4: avg=4,553  p95=5,740  (-310)
  W5: avg=4,779  p95=6,002                   W5: avg=4,553  p95=5,617  (-226)
  Spread: 330 cycles                         Spread: 269 cycles (-18%)

  Phase 1 avg: 4,569 → 4,345  (-224, -4.9%)
  ml_wait:     1,369 → 1,538  (+169) — stagger shifts epilogue start later
  epi_wait:    1,352 → 1,344  (-8)
```

All warps improved, not just W4/W5 — confirms pervasive contention across all 4 warps, with lower warp IDs suffering less. The ml_wait increase partially offsets the Phase 1 gain (warps start Phase 1 later relative to W1's commit).

### Changes

1. `#define STAGGER_CYCLES 80` — tunable define at top of file
2. Overlapped epilogue (line 763): `clock64()` spin after mbar_wait + fence, before Phase 1
3. Drain epilogue (line 884): identical stagger

236 regs (+1 for clock64 spin variable), 0 spills. Checksum = 1769472.0.

---

## F32: TMEM x16 granularity re-evaluation — REJECTED (10.6% regression)

**Date:** 2026-03-01
**Hypothesis:** `tcgen05.ld.x16` (16 cols per load) could improve Phase 1 by creating finer-grained TMEM_WAIT windows for interleaving compute. The original x16-vs-x32 comparison (F8, commit `abf04a5`) was done at 1433 TFLOPS with no double-buffered staging, no Phase 2A interleaving, no BF16 math — a fundamentally different operating point.

**Key question:** Is TMEM load latency proportional to data volume (x16 ≈ 100 cycles → total unchanged) or fixed per instruction (x16 ≈ 200 cycles → 2× worse)?

### Changes

Single file `megakernel.cu`, `epilogue_store` function only:

1. Removed `a16..a31` register declarations (32→16 float regs)
2. Phase 1A: `TMEM_LOAD_X32` → `TMEM_LOAD` (x16), loop stride 32→16, 2 STS_V4 per iter instead of 4
3. Phase 1A→1B transition: `TMEM_LOAD_X32` → `TMEM_LOAD`
4. Phase 1B + Phase 2A: loop stride 32→16, Phase 2A interleaving 8→4 rows per iter (8 iters × 4 rows = 32 total), 2 STS_V4 per iter instead of 4
5. Combined data addressing: `+ (nc & 31)` to select correct half of 32-col COMB blocks when stepping by 16

### Results

| Metric | Baseline (x32) | F32 (x16) | Delta |
|--------|----------------|-----------|-------|
| Wall clock | 0.530 ms | 0.586 ms | **+10.6% regression** |
| TFLOPS | 2067 | 1868 | -9.6% |
| Registers | 236 | 174 | -62 |
| Spills | 0 | 0 | — |
| Checksum | 1769472.0 | 1769472.0 | Match |

### Analysis

**TMEM load latency is fixed per instruction, not proportional to data volume.** Doubling the number of loads (8→16 per phase) adds ~56 µs wall clock despite halving data per load. The 62-register savings (236→174) from eliminating `a16..a31` do not compensate — in fact, the massive register drop suggests the compiler fully unrolled the simpler loop body but couldn't hide the additional TMEM latency.

This definitively answers the key question from F8: the `tcgen05.ld` ~200-cycle latency is a fixed cost per instruction. x32 is optimal — it maximizes data per fixed-cost load. x16 pays the same ~200-cycle penalty per load but moves half the data.

**Implications:**
- x32 is the correct TMEM load granularity. This is now confirmed at two very different operating points (1433 TFLOPS and 2067 TFLOPS).
- Phase 1 optimization cannot come from finer TMEM load granularity. The binding constraint is the number of `tcgen05.ld` instructions × fixed latency per instruction.
- The 62-register savings are real but useless — no register-gated optimization exists that could overcome the 10.6% regression.

Rolled back to baseline. No code changes committed.

---

## F33: tcgen05.cp TMEM→SMEM async copy — RULED OUT

**Date:** 2026-03-01
**Hypothesis:** `tcgen05.cp` could bypass the register file with async TMEM→SMEM copy, replacing the `tcgen05.ld` (TMEM→registers) → CVT → add → `st.shared` path and dramatically reducing Phase 1 latency.

**Result: KILLED BY ISA RESEARCH.** `tcgen05.cp` is architecturally SMEM→TMEM only. No hardware path exists for TMEM→SMEM direction.

**Evidence (3 independent sources):**
1. **CUTLASS**: Has `make_s2t_copy()` (SMEM-to-TMEM) but NO `make_t2s_copy()`. All epilogue code uses `tcgen05.ld`.
2. **Colfax tutorial**: "data gets _into_ TMEM via UMMA operations, and is explicitly moved _out_ to registers using `tcgen05.ld`."
3. **JAX/Pallas docs**: "only way to move data out from tensor memory is through `tcgen05.ld`"

**Conclusion:** The only way to read accumulators out of TMEM is `tcgen05.ld` (TMEM→registers). Phase 1 optimization must work within this constraint — no DMA shortcut exists.

No code changes. No timing data.

---

## F34: Parallel TMEM load diagnostic — loads pipeline (confirmed)

**Date:** 2026-03-01
**Baseline:** 0.530 ms / 2067 TFLOPS / 236 regs (F31)
**Result:** 0.531 ms / 2064 TFLOPS / 248 regs — **within noise of baseline**
**Verdict:** DIAGNOSTIC SUCCESS — 2 `tcgen05.ld.x16` loads pipeline before a single `TMEM_WAIT()`

**Hypothesis:** F32 showed `tcgen05.ld` has a fixed ~200-cycle latency per instruction, but tested only sequential loads (load→wait→process→load→wait). F1 tested split x32→2×x16, but with a WAIT between each pair (the two loads were never simultaneously outstanding). **Neither tested whether 2 `tcgen05.ld` instructions issued before a single `tcgen05.wait::ld` execute in parallel on the TMEM read port.**

**How this differs from F1 and F32:**
- **F1** (split x32→2×x16): Issued first x16, then WAIT, then second x16, then WAIT. Two loads never simultaneously in-flight. Tested earlier issue time — no benefit.
- **F32** (pure x16): Changed loop stride to 16, one load + one WAIT per iteration. Tested x16 latency — same ~200 cycles as x32.
- **F34** (this): Issues 2 x16 loads back-to-back before a single WAIT. Tests concurrent outstanding loads.

**Changes:**

Replaced all 4 `TMEM_LOAD_X32` call sites with pairs of `TMEM_LOAD` (x16):

```c
// Before (1 x32 load):
TMEM_LOAD_X32(a0,...,a31, addr);

// After (2 x16 loads, same WAIT):
TMEM_LOAD(a0,...,a15, addr);
TMEM_LOAD(a16,...,a31, addr + 16);
```

4 sites modified: Phase 1A initial load, Phase 1A prefetch, Phase 1A→1B transition, Phase 1B prefetch. Same `a0..a31` declarations, same `TMEM_WAIT()` positions, same processing. The ONLY difference is whether 32 columns arrive via 1 instruction or 2 instructions before the same fence.

**Build:** 248 regs (+12 from baseline), 0 spills. The 12 extra registers come from having both x16 loads' destination registers simultaneously live (compiler can't reuse across two `asm volatile` blocks). Still fits 1 CTA/SM (248 × 192 = 47,616 < 65,536 register file).

**Result:**

| Metric | Baseline (x32) | F34 (2×x16, same WAIT) | Delta |
|--------|----------------|------------------------|-------|
| Wall clock | 0.530 ms | 0.531 ms | +0.2% (noise) |
| TFLOPS | 2067 | 2064 | -0.1% (noise) |
| Registers | 236 | 248 | +12 |
| Checksum | 1769472.0 | 1769472.0 | Match |

**Conclusion: Loads pipeline.** Two x16 loads issued before a single WAIT complete in essentially the same time as one x32 load. The TMEM read port can process 2 outstanding loads concurrently — they overlap rather than serializing.

If loads had serialized, wall clock would have been ~0.586 ms (matching F32's regression from doubling load count). Instead, 0.531 ms = baseline, proving the hardware pipelines concurrent loads.

**Implications:**
- Opens the door to software-pipelined TMEM readback: issue next iteration's TMEM loads early (before current WAIT), overlapping them with CVT+add+STS compute from the current iteration.
- The +12 register cost (248 vs 236) is the price of parallelism — both loads' destinations must be live simultaneously. A software-pipelined design would need careful register budgeting.
- Combined with F32's finding (fixed ~200-cycle latency per load), this means x32 remains optimal for sequential access, but parallel x16 loads are viable when register budget allows overlapping compute with TMEM reads.
