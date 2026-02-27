# Performance Audit — 2026-02-27

Kernel: `patch_embed_gemm` (commit abf04a5, TN=256, x32 TMEM loads)
GEMM: [928256, 768] x [768, 768]^T, FP8 E4M3, BF16 output with fused bias+pos_embed

## Performance summary

**End-to-end** (GEMM + bias + pos_embed — the actual workload):

| Kernel | Time (ms) | Notes |
|--------|-----------|-------|
| **This kernel (fused)** | **0.764** | GEMM + bias + pos in overlapped epilogue |
| cuBLAS best + unfused pos | 0.835 | Per-tensor FP8 GEMM + separate pos_embed kernel |
| cuBLAS MXFP8 + unfused pos | 0.846 | MXFP8 GEMM + separate pos_embed kernel |

**GEMM only** (cuBLAS benchmark, best-of-N algos, 256MB workspace):

| Mode | Time (ms) | TFLOPS |
|------|-----------|--------|
| Per-tensor FP8 (best-of-8) | 0.365 | 3001 |
| MXFP8 block-scaled (best-of-3) | 0.375 | 2920 |

The kernel's advantage is **fusion**: the 0.470 ms unfused pos_embed overhead is eliminated by the overlapped epilogue. The "1433 TFLOPS" effective rate includes fused epilogue time in the denominator — it is not a GEMM-only metric.

### Optimization history

| Commit | Change | TFLOPS |
|--------|--------|--------|
| 6c327fe | Replace TMA stores with st.global.v4 | ~300 |
| c5f6c8b | Software-pipeline TMEM loads (double-buffer A/B) | ~400 |
| 4ff9644 | Prefetch bias+pos_embed into SMEM during K-loop | ~500 |
| 4760f3b | Unified epilogue, 147 regs, 0 spills | 727 |
| 892766c | Replace SMEM prefetch with inline BF16 loads | 1043 |
| fca9178 | Upgrade cta_group::1 to cta_group::2 | 1190 |
| 9557e0b | TN=128→256, single TMEM alloc of 512 cols | 1433 |
| **abf04a5** | **x16→x32 TMEM loads (no double-buffer)** | **1433 (perf-neutral)** |

## A/B comparison: x16 vs x32 TMEM loads

This section compares the previous epilogue (double-buffered x16 TMEM loads, 216 regs) with the current epilogue (single-buffer x32 TMEM loads, 248 regs). Both profiled via `ncu --set detailed`, single kernel instance (skip 2 warmup), GPU reset between runs.

Raw data: `profile_x16.csv`, `profile_x32.csv` (1203 ncu metrics each).

### Duration / cycles

| Metric | x16 | x32 | Delta |
|--------|:---:|:---:|:-----:|
| Duration (ncu) | 1.243 ms | 1.240 ms | -0.2% |
| Active cycles | 1,353,248 | 1,349,029 | -0.3% |

ncu duration (~1.24 ms) is higher than wall-clock timing (0.764 ms) due to profiling overhead — relative comparison is valid.

### Build stats (ptxas)

| Metric | x16 | x32 |
|--------|:---:|:---:|
| Registers/thread | 216 | 248 |
| Spills | 0 | 0 |
| Stack | 16 bytes | 16 bytes |
| SMEM | 196,864 bytes | 196,864 bytes |
| Barriers | 2 | 2 |
| Achieved occupancy | 9.37% | 9.38% |

248 regs still fits in the 256-reg budget for 1 CTA/SM. No occupancy change.

### Warp stall breakdown (pcsamp sampling)

SM100a uses PC-sampling-based stall attribution (`pcsamp`). Values are sample counts; percentages are share of total samples.

| Stall reason | x16 | x16% | x32 | x32% | Delta | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| **long_scoreboard** | **27,024** | **55.3%** | **27,495** | **56.4%** | **+1.7%** | **Dominant.** TMEM load latency. x32 loads don't help — same bandwidth, fewer but equally slow loads. |
| sleeping | 5,432 | 11.1% | 5,391 | 11.1% | -0.8% | Warp-specialized warps waiting for their role (TMA/MMA warps idle during epilogue). Unchanged. |
| barrier | 5,171 | 10.6% | 5,008 | 10.3% | -3.2% | `bar.sync` between warp roles at tile boundaries. Slight improvement. |
| wait | 4,961 | 10.1% | 4,146 | 8.5% | **-16.4%** | `cp.async.bulk.wait` in TMA warp. **Significant drop** — fewer TMEM loads means less TMA/TMEM contention. |
| selected | 4,650 | 9.5% | 4,673 | 9.6% | +0.5% | Warp was selected to issue. Stable. |
| branch_resolving | 882 | 1.8% | 874 | 1.8% | -0.9% | Warp-specialization branch dispatch. Stable. |
| no_instructions | 276 | 0.6% | 285 | 0.6% | +3.3% | Instruction cache misses. Negligible. |
| short_scoreboard | 232 | 0.5% | 582 | 1.2% | **+150.9%** | Register dependencies. **Notable increase** — x32 loads produce 32 output regs with longer dep chains. |
| dispatch_stall | 79 | 0.2% | 91 | 0.2% | +15.2% | Scheduler dispatch contention. Negligible. |
| math_pipe_throttle | 73 | 0.1% | 61 | 0.1% | -16.4% | MMA pipe not a bottleneck. |
| not_selected | 71 | 0.1% | 94 | 0.2% | +32.4% | More warps eligible but not picked — consistent with shorter epilogue code having more scheduling opportunities. |
| misc | 45 | 0.1% | 43 | 0.1% | -4.4% | Uncategorized. |
| drain | 1 | 0.0% | 1 | 0.0% | +0.0% | End-of-kernel drain. |

**Key finding**: `long_scoreboard` is 55-56% of all stall samples in both versions — completely dominant. The x32 change traded away `wait` stalls (-16.4%, from reduced TMA/TMEM contention) but gained `short_scoreboard` stalls (+150.9%, from longer register dependency chains in the 32-wide load). These cancel out, yielding identical wall-clock time.

### Throughput

| Subsystem | x16 | x32 | Delta |
|---|:---:|:---:|:---:|
| Memory (overall) | 84.0% | 84.2% | +0.3% |
| L1/TEX (% active) | 85.6% | 85.9% | +0.4% |
| L2 (LTS) | 58.2% | 58.3% | +0.1% |
| DRAM | 25.3% | 25.3% | +0.2% |
| SM Compute | 33.6% | 33.7% | +0.1% |

All throughput metrics are identical within noise. The kernel remains **memory-throughput bound** (84% memory vs 34% compute).

### Instruction counts

| Metric | x16 | x32 | Delta |
|---|---:|---:|:---:|
| Instructions executed | 105,921,306 | 102,427,219 | **-3.3%** |
| Global loads (LDG) | 2,784,768 | 2,784,768 | 0.0% |
| Global stores (STG) | 1,392,384 | 1,392,384 | 0.0% |

x32 saves 3.5M instructions (-3.3%) from halving TMEM load count (8 x32 loads vs 16 x16 loads per warp per tile). But instruction count was never the bottleneck — TMEM bandwidth is. Global load/store counts are identical — output size doesn't change.

### Memory traffic

| Metric | x16 | x32 | Delta |
|---|---:|---:|:---:|
| DRAM read | 715.4 MB | 715.2 MB | -0.0% |
| DRAM write | 1.376 GB | 1.376 GB | +0.0% |

Byte-identical memory traffic. Expected — same data, same access pattern.

### Uncoalesced access analysis

| Metric | x16 | x32 |
|---|---:|---:|
| Excessive sectors | 44.56 MB | 44.56 MB |
| % of total sectors | 49% | 50% |

Identical uncoalesced pattern. The 49→50% change is because x32 issues slightly fewer total sectors (89.5M vs 90.5M — from fewer TMEM load instructions generating L1 traffic) while the same absolute number of store sectors remain uncoalesced.

## Conclusions

### Why x32 TMEM loads are perf-neutral

The epilogue bottleneck is **TMEM bandwidth**, not instruction issue rate. `tcgen05.ld.sync.aligned.32x32b.x32.b32` reads 32 columns in one instruction, but the hardware still transfers the same 128 bytes per warp and the ~200-cycle latency is unchanged. Halving the instruction count saves 3.3% of instructions but buys zero wall-clock time because the scheduler was already stalled waiting for TMEM data, not waiting to issue the next load.

The stall profile confirms this: `long_scoreboard` (TMEM latency) is 55%+ of all samples in both versions. The x32 version slightly reduced `wait` stalls (TMA contention) but introduced `short_scoreboard` stalls (register dependency chains), canceling out.

### What would actually help

1. **More epilogue warps**: long_scoreboard at 55% means 4 warps can't hide 200-cycle TMEM latency. Need 6-8 epilogue warps to fully interleave. Blocked by register pressure (248 regs × 256 threads = 63.5K, approaching 64K limit).

2. **Fix uncoalesced stores (49% excess sectors)**: 44.6 MB of wasted L2 sector traffic per invocation. Would require SMEM staging (cooperative store to SMEM → coalesced write to global) or output transposition.

3. **Reduce epilogue work**: The fundamental ratio problem — 256×256×4 bytes of TMEM readback vs 256×128 FP8 MMA per K-iter. The epilogue is the long pole by 2x. Smaller tiles reduce epilogue per-tile cost but add more tile transitions.

### What we learned about TMEM

- **512 columns/SM total** (not 256 as previously believed)
- Two separate `tcgen05.alloc` calls of 256 each deadlock — the hardware cannot satisfy the second alloc while the first holds its allocation
- A single `tcgen05.alloc` of 512 works — like `mmap`, one large allocation succeeds where two half-size allocations fail
- `tcgen05.alloc` requires power-of-2 column counts (192 → illegal instruction)
- First alloc always returns column 0; double-buffering is just `buf * TN` as a column offset
- Alloc should be placed in the MMA warp to avoid blocking TMA startup
- **x16 vs x32 TMEM loads are bandwidth-equivalent** — instruction count doesn't matter, only total bytes transferred and latency per load
