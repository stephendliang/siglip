# Performance Audit — 2026-02-27

Kernel: `patch_embed_gemm` (commit 9557e0b, TN=256, single TMEM alloc)
GEMM: [928256, 768] x [768, 768]^T, MXFP8 E4M3, BF16 output with fused bias+pos_embed

## Performance summary

| Kernel | Time (ms) | TFLOPS | vs cuBLAS |
|--------|-----------|--------|-----------|
| **This kernel** | **0.764** | **1433** | **110%** |
| cuBLAS (reference) | 0.845 | 1295 | 100% |

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

## Build stats (ptxas)

```
Registers:    216/thread (0 spills)
Stack:        16 bytes
SMEM:         196,864 bytes (192.25 KB of 228 KB)
Barriers:     2
```

## ncu: GPU Speed of Light

| Metric | Old (TN=128) | New (TN=256) |
|--------|:---:|:---:|
| Duration | 917 µs | **770 µs** |
| Elapsed Cycles | ~1,655,000 | **1,407,154** |
| Memory Throughput | 68.5% | **82.2%** |
| L1/TEX Throughput | 68.4% | **84.0%** |
| L2 Throughput | 53.2% | **55.4%** |
| DRAM Throughput | 18.3% | **35.4%** |
| SM Compute | 34.0% | **33.2%** |

The kernel is **memory-throughput bound** (82% vs 33% compute). Wider tiles nearly doubled DRAM utilization (18→35%) by reducing per-tile mbar/sync overhead and amortizing TMA setup across more output columns.

## ncu: Warp stall breakdown

| Stall reason | Old | New | Interpretation |
|---|:---:|:---:|---|
| **long_scoreboard** | 5.12% | **4.69%** | **Dominant.** TMEM load latency (~200 cycles). 4 epilogue warps can't fully hide it. Slightly improved with wider tiles (better amortization). |
| barrier | 1.08% | **1.14%** | `bar.sync` between epilogue warps. Stable — warp specialization working. |
| wait | 1.02% | **0.95%** | `cp.async.bulk.wait` in TMA warp. Slightly better with fewer tiles. |
| short_scoreboard | 0.02% | **0.04%** | Negligible. No SMEM dependency issues. |
| math_pipe_throttle | 0.02% | **0.01%** | MMA pipe not a bottleneck. |
| not_selected | 0.03% | **0.02%** | Near-zero — scheduler has few eligible warps competing. |
| mio_throttle | 0.00% | **0.00%** | No store backpressure. |
| lg_throttle | 0.00% | **0.00%** | No global load throttling. |
| membar | 0.00% | **0.00%** | mbarrier waits not significant. |

### Analysis

The stall profile is clean and similar to TN=128. `long_scoreboard` dropped slightly (5.1→4.7%) — fewer tiles means fewer per-tile transitions, giving the epilogue more uninterrupted time to pipeline TMEM loads. All other stalls remain <1.2%.

## ncu: Throughput breakdown

| Subsystem | Old | New | Notes |
|---|:---:|:---:|---|
| L1/TEX | 68.4% | **84.0%** | Epilogue-dominated: TMEM loads + global stores flow through L1. Near saturation. |
| L2 (LTS) | 53.2% | **55.4%** | BF16 combined loads + output stores. Moderate. |
| DRAM | 18.3% | **35.4%** | Nearly doubled — wider tiles reduce overhead, more sustained memory streaming. |
| SM Compute | 34.0% | **33.2%** | Underutilized — epilogue overhead limits MMA pipe utilization. |

## ncu: Uncoalesced access analysis

ncu flags **49% uncoalesced global accesses**: 44.6M excessive sectors out of 90.5M total. This is the epilogue's `st.global.v8` scatter pattern. Each warp writes 32 rows — adjacent threads store to addresses 1536 bytes apart (N_DIM × sizeof(BF16) = 768×2). Every lane hits a different 128-byte L2 sector.

This is **inherent to row-major output** with per-row TMEM readback. Cannot be fixed without output transposition or fundamentally different output tiling. The same pattern exists in cuBLAS.

## ncu: Instruction counts (per kernel invocation)

| Metric | Old | New |
|---|:---:|:---:|
| Total instructions executed | 108.2M | **105.9M** |
| Total thread-instructions | 2,837M | **3,058M** |
| Global loads (LDG) | 2,784,768 | **2,784,768** |
| Global stores (STG) | 1,392,384 | **1,392,384** |

Fewer total instructions (-2.1%) due to fewer tiles (10,878 vs 21,756) reducing per-tile overhead. More thread-instructions (+7.8%) because the epilogue inner loop does 8 column iterations (was 4 with TN=128).

Global load/store counts are identical — output size is the same regardless of tile shape. Store count: 1,392,384 = 10,878 tiles × 16 stores/warp × 4 warps × 2 CTAs. Each `st.global.v8.b32` writes 32 bytes (16 BF16). Total: 1,392,384 × 32 = 44.6 MB = 928,256 × 768 × 2 bytes.

## ncu: Occupancy

| Metric | Value |
|---|---|
| Theoretical occupancy | 9.38% |
| Achieved occupancy | 9.51% |
| Registers/thread | 216 |
| Waves per SM | 1 |
| Block limit: registers | 1 |
| Block limit: shared mem | 1 |

Occupancy is 9.4% (1 CTA per SM, 192 of 2048 max threads). By design — persistent kernel with 1 wave. Both registers (216) and shared memory (192 KB) independently limit to 1 block per SM.

## Remaining performance ceiling

We are now **10% faster than cuBLAS**. The profile shows we are in diminishing returns territory:

### What's left on the table

1. **Uncoalesced stores (49% excess sectors)**: The dominant source of wasted memory bandwidth. Inherent to row-major output — each lane in a warp writes to a different row. Would require output transposition or a two-pass scheme (write coalesced to scratch, then transpose). Not practical for a fused kernel.

2. **TMEM load latency (long_scoreboard 4.7%)**: Still the #1 stall. 4 epilogue warps interleave loads but ~200-cycle latency needs ~6-8 warps to fully hide. Adding epilogue warps requires either reducing TMA/MMA warps (hurts pipeline) or increasing total thread count (hits register limits at 216 regs/thread).

3. **SM compute at 33%**: The MMA pipe is idle 2/3 of the time. The epilogue (TMEM→add→convert→store) takes ~2x longer than the K-loop MMA. With overlapped execution, the epilogue is the long pole. This is the fundamental ratio problem: 256×256 output requires 256×256×4 bytes of TMEM readback but only 256×128×2 FLOPs of MMA per K-iteration.

4. **Register pressure (216 regs)**: Up from 168 with TN=128. The wider epilogue loop keeps more BF16 combined data and TMEM values alive. Could try `--maxrregcount=192` but may cause spills that hurt more than they help.

### What we learned about TMEM

- **512 columns/SM total** (not 256 as previously believed)
- Two separate `tcgen05.alloc` calls of 256 each deadlock — the hardware cannot satisfy the second alloc while the first holds its allocation
- A single `tcgen05.alloc` of 512 works — like `mmap`, one large allocation succeeds where two half-size allocations fail
- `tcgen05.alloc` requires power-of-2 column counts (192 → illegal instruction)
- First alloc always returns column 0; double-buffering is just `buf * TN` as a column offset
- Alloc should be placed in the MMA warp to avoid blocking TMA startup
