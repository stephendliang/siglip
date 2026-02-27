# Experiment Log

Kernel: `patch_embed_gemm` — persistent megakernel for SigLIP2 patch embed GEMM.
GEMM: `[928256, 768] x [768, 768]^T`, FP8 E4M3 inputs, BF16 output with fused bias + positional embedding.
Target: B200 (SM100a), 148 SMs, `cta_group::2`, `__cluster_dims__(2,1,1)`.

**Current best: 0.630 ms / 1739 TFLOPS** (fused end-to-end). 222 regs, 0 spills.

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
| d882aba | Phase 2 unroll 4→8 | 0.630 | 1739 | 222 | current best |

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
