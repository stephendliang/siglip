# Performance Audit — 2026-02-27

Kernel: `patch_embed_gemm` (commit fca9178, cta_group::2)
GEMM: [928256, 768] x [768, 768]^T, MXFP8 E4M3, BF16 output with fused bias+pos_embed

## Performance summary

| Kernel | Time (ms) | TFLOPS | % of cuBLAS |
|--------|-----------|--------|-------------|
| **This kernel** | **0.917** | **1190** | **92%** |
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

## Build stats (ptxas)

```
Registers:    168/thread (0 spills)
Stack:        24 bytes
SMEM:         147,712 bytes (144.25 KB of 228 KB)
Barriers:     2
```

## ncu: GPU Speed of Light

```
SM Frequency:           1111 MHz
DRAM Frequency:         3996 MHz
Elapsed Cycles:         ~1,655,000
Compute (SM) Throughput: 34%
Memory Throughput:       68.5%
```

The kernel is **memory-throughput bound** (68.5% vs 34% compute). The memory subsystem is 2x more utilized than the SM compute path.

## ncu: Warp stall breakdown

| Stall reason | % of peak | Interpretation |
|---|---|---|
| **long_scoreboard** | **5.12%** | **Dominant.** TMEM load latency (~200 cycles). The double-buffered epilogue helps but 4 warps can't fully hide it. |
| barrier | 1.08% | `bar.sync` between epilogue warps (W2-5 named barrier). Low — warp specialization is working. |
| wait | 1.02% | `cp.async.bulk.wait` in TMA load warp (W0). Expected for TMA pipeline. |
| math_pipe_throttle | 0.02% | MMA pipe not a bottleneck (tcgen05 throughput is sufficient). |
| not_selected | 0.03% | Almost zero — scheduler has very few eligible warps competing. |
| short_scoreboard | 0.02% | No SMEM dependency issues (inline BF16 loads eliminated SMEM staging). |
| mio_throttle | 0.00% | No store backpressure (st.global.v8 scatter pattern is not saturating L2). |
| lg_throttle | 0.00% | No global load throttling. |
| membar | 0.00% | mbarrier waits are not a significant stall source. |

### Analysis

The stall profile is clean. `long_scoreboard` at 5.1% is the only significant stall — this is TMEM load latency that the 4 epilogue warps can't fully hide. All other stall sources are <1.1%.

The old bottlenecks are gone:
- `barrier` was dominant before warp specialization + mbarrier upgrade (now 1.08%)
- `wait` was dominant before TMA store removal (now 1.02%)
- `short_scoreboard` was significant before SMEM staging removal (now 0.02%)

## ncu: Throughput breakdown

| Subsystem | % of peak | Notes |
|---|---|---|
| L1/TEX | 68.4% | Epilogue-dominated (TMEM loads + global stores flow through L1) |
| L2 (LTS) | 53.2% | Moderate — BF16 combined loads + output stores |
| DRAM | 18.3% | Low — good L2 hit rate for bias+pos_embed (reused across M-tiles) |
| SM Compute | 34% | Underutilized — epilogue overhead limits MMA pipe utilization |

## ncu: Instruction counts (per kernel invocation)

| Metric | Count |
|---|---|
| Total instructions executed | 108.2M |
| Total thread-instructions | 2,837M |
| Global loads (LDG) | 2,784,768 |
| Global stores (STG) | 1,392,384 |

Global store count: 1,392,384 = 21,756 tiles x 8 stores/warp x 4 warps x 2 CTAs. Each `st.global.v8.b32` writes 32 bytes (16 BF16 values). Total output: 1,392,384 x 32 = 44.6 MB = exactly 928,256 x 768 x 2 bytes.

## ncu: Occupancy

| Metric | Value |
|---|---|
| Theoretical occupancy | 9.38% |
| Achieved occupancy | 9.37% |
| Registers/thread | 168 |
| Waves per SM | 1 |

Occupancy is 9.4% (1 CTA per SM, 192 of 2048 max threads). This is by design — persistent kernel with 1 wave, all SMs active. Low occupancy is expected and not a problem.

## Remaining gap to cuBLAS (8%)

cuBLAS achieves 0.845ms / 1295 TFLOPS with 256x192x128 tiles (confirmed by benchmarks in commit a0bfad4).

### What cuBLAS likely does differently
1. **Larger N tile (192 vs 128)**: Fewer tiles (14504 vs 21756), better per-tile amortization. However, we confirmed TMEM capacity limits us to N=128 with double-buffered TMEM on SM100 (2x128=256 cols = max capacity). TN=192 requires power-of-2 TMEM alloc (would need 256), and TN=256 with 2 buffers = 512 cols exceeds capacity (deadlocks).
2. **Different TMEM strategy**: cuBLAS may use single-buffered TMEM (256 cols, no overlap) with a faster serialized epilogue, or split MMA passes targeting different TMEM column ranges.
3. **Better epilogue pipelining**: May have more aggressive TMEM latency hiding via different warp scheduling or instruction ordering.

### Constraints discovered
- **TMEM capacity**: 256 columns/SM total. With double-buffered alloc, max N=128.
- **tcgen05.alloc**: Must be power-of-2 in [32, 256]. 192 causes illegal instruction.
- **tcgen05.alloc with 2x256**: Deadlocks — second alloc blocks forever (no TMEM available).
