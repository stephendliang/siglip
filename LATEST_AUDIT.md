# Performance Audit — 2026-02-27

Kernel: `patch_embed_gemm` (SMEM-staged coalesced stores, 4-stage pipeline, 5 epilogue warps)
GEMM: [928256, 768] x [768, 768]^T, FP8 E4M3, BF16 output with fused bias+pos_embed

## Performance summary

**End-to-end** (GEMM + bias + pos_embed — the actual workload):

| Kernel | Time (ms) | TFLOPS | Notes |
|--------|-----------|--------|-------|
| **This kernel (fused)** | **0.633** | **1729** | GEMM + bias + pos in overlapped epilogue |
| Previous (st.global.v8) | 0.700 | 1564 | Same fused approach, uncoalesced stores |
| cuBLAS best + unfused pos | 0.835 | — | Per-tensor FP8 GEMM + separate pos_embed kernel |

**24% faster** than cuBLAS end-to-end (0.835 ms). **9.6% faster** than previous kernel (0.700 ms).

**GEMM only** (cuBLAS benchmark, best-of-N algos, 256MB workspace):

| Mode | Time (ms) | TFLOPS |
|------|-----------|--------|
| Per-tensor FP8 (best-of-8) | 0.365 | 3001 |
| MXFP8 block-scaled (best-of-3) | 0.375 | 2920 |

### Optimization history

| Commit | Change | Time (ms) | TFLOPS |
|--------|--------|-----------|--------|
| 6c327fe | Replace TMA stores with st.global.v4 | — | ~300 |
| c5f6c8b | Software-pipeline TMEM loads (double-buffer A/B) | — | ~400 |
| 4ff9644 | Prefetch bias+pos_embed into SMEM during K-loop | — | ~500 |
| 4760f3b | Unified epilogue, 147 regs, 0 spills | — | 727 |
| 892766c | Replace SMEM prefetch with inline BF16 loads | — | 1043 |
| fca9178 | Upgrade cta_group::1 to cta_group::2 | — | 1190 |
| 9557e0b | TN=128→256, single TMEM alloc of 512 cols | — | 1433 |
| abf04a5 | x16→x32 TMEM loads (no double-buffer) | 0.764 | 1433 |
| 6319928 | Remove centralized mainloop_mbar bar.sync | 0.743 | — |
| cefc59d | Reduce pipeline stages 6→4 | 0.700 | 1564 |
| **current** | **Epilogue SMEM staging → coalesced stores** | **0.633** | **1729** |

## Change: Epilogue SMEM staging

### Problem

Epilogue `st.global.v8.b32` stores were **uncoalesced**: each warp's 32 threads wrote to 32 different rows (1536B stride between lanes), producing 32 L1 sectors per store request (vs 16 ideal = 100% excess). This caused L1 read-modify-write amplification (each 128B cache line received only 32 bytes = 25% fill).

### Solution

Two-phase epilogue via per-warp SMEM staging buffers:

1. **Phase 1**: Same TMEM + combined BF16 loop, but `CVT_STS` stores to SMEM (row-per-thread layout). Each thread writes its own row — no bank conflicts with 16-byte row padding.

2. **`__syncwarp()`** barrier between phases.

3. **Phase 2**: Transposed coalesced store. Loop over 32 rows; all 32 threads read from the same staging row (different column chunks) and write to the same global row. `st.global.v4.b32` for 256-col warps (W3-W5), `st.global.v2.b32` for 128-col split warps (W2/W6).

### SMEM budget

| Component | Bytes |
|-----------|-------|
| Pipeline (4 stages x 32KB) | 131,072 |
| TMEM addr + mbarriers | 128 |
| Staging (5 warps x 16,896) | 84,480 |
| **Total (SMEM_BYTES)** | **215,808** |
| SM limit | 233,472 (228 KB) |
| **Headroom** | **~17 KB** |

## A/B comparison: baseline (st.global.v8) vs SMEM staging

Profiled via `ncu --set detailed`, single kernel instance, same GPU.

Raw data: `baseline.csv` (pre-staging), `after.csv` (post-staging). Run `python compare.py baseline.csv after.csv` for full diff.

### Build stats (ptxas)

| Metric | Baseline | SMEM staging |
|--------|:--------:|:------------:|
| Registers/thread | 216 | 222 |
| Spills | 0 | 0 |
| Stack | 16 bytes | 16 bytes |
| SMEM (dynamic) | ~131 KB | ~211 KB |

### Warp stall breakdown (% of peak sustained active)

| Stall reason | Baseline | Staging | Delta | Notes |
|---|---:|---:|---:|---|
| **selected (issuing)** | **14.1%** | **19.1%** | **+35%** | **Major win.** Warps spend 35% more time doing productive work. |
| long_scoreboard (TMEM) | 6.4% | 4.4% | -30% | TMEM stalls reduced — less time wasted on L1 store backpressure. |
| short_scoreboard (SMEM) | 0.1% | 1.1% | +1.0pp | New: SMEM staging ld/st dependency chains. Expected cost. |
| sleeping | 1.3% | 1.1% | -15% | Less idle time between tiles. |
| wait (TMA) | 0.9% | 1.2% | +0.3pp | Slightly more TMA pipeline pressure. |
| barrier | 0.8% | 0.8% | unchanged | |
| mio_throttle | 0.0% | 0.03% | negligible | Shared memory I/O backpressure, minimal. |

### Memory throughput

| Subsystem | Baseline | Staging | Delta |
|---|:---:|:---:|:---:|
| **L1 tex** | **82%** | **85%** | +3% (still primary ceiling) |
| L2 | 60% | 54% | -11% (less L2 pressure) |
| DRAM | 24% | 27% | +13% |

### Uncoalesced access analysis

| Metric | Baseline | Staging |
|---|---:|---:|
| Excessive L2 sectors | 44.56M | 44.56M |
| % of total L2 sectors | **50%** | **33%** |
| Global store instructions | 1,392,384 | 3,480,960 |

The absolute excess sector count is unchanged (44.56M — from split warps' V2 stores and combined BF16 global loads). But excess as a fraction of total dropped from 50% to 33% because the coalesced V4/V2 stores generate more "good" sectors. Global store instruction count increased 2.5x because V4 stores (16 bytes) replace V8 stores (32 bytes), but each V4 store is fully coalesced.

### Key compare.py findings (>5% relative change)

| Metric | Baseline | Staging | Change |
|--------|----------|---------|--------|
| Inst executed per cycle (active) | 82.9 | 112.3 | **+35%** |
| SMEM shared ld instructions | 666 | 3,481,626 | +523K% (Phase 2 reads) |
| SMEM shared st instructions | 148 | 2,784,916 | +1.9M% (Phase 1 writes) |
| Shared memory L1 wavefronts | 1,554 | 22,279,698 | +1.4M% (staging traffic) |
| SMEM L1 pipe utilization | 0.1% | 20.6% | (new workload) |
| L1 hit rate | 61.7% | 33.6% | -46% (fewer read-modify-write) |
| Instruction cache requests | 849K | 3,510K | +314% (larger code) |

## Conclusions

### Why SMEM staging wins 9.6%

The uncoalesced `st.global.v8` stores were causing two problems:
1. **100% excess L2 sectors** — 32 sectors per store request vs 16 ideal
2. **L1 read-modify-write** — each 128B cache line received only 32 bytes (25% fill), forcing L1 to read the line before writing

SMEM staging fixes #2 completely (full 128B lines written in Phase 2) and makes #1 neutral (same absolute excess but lower fraction). The L1 hit rate drop (61.7% → 33.6%) confirms the read-modify-write elimination.

The cost is SMEM traffic (20.6% L1 shared pipe utilization, 1.1% short_scoreboard stalls) and more global store instructions (2.5x). But the net effect is strongly positive: **warp issue rate up 35%, TMEM stalls down 30%**.

### New bottleneck profile

The kernel is still **L1 throughput-bound** (85% of peak), but now with a healthier instruction mix:
- More productive execution (19.1% selected vs 14.1%)
- Less wasted time on TMEM stalls (4.4% vs 6.4%)
- Small new SMEM cost (1.1% short_scoreboard)
- TC pipeline utilization improved: 36.3% → 41.8%
