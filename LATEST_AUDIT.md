## Correctness: no bugs found

I walked through every nontrivial invariant — mbarrier phases, TMEM double-buffer sync, SMEM OOB, store alignment, race conditions, edge cases, tile coverage — and everything checks out. The 4-byte margin between the max sums access (byte 229372) and OFF_TMEM_0 (byte 229376) is tight but correct.

## Severe issues: one real concern, two things to watch

### 1. Scattered `st.global.v8` writes — the biggest unknown

This is the one thing I'd actually worry about. Every `st.global.v8.b32` from a warp scatters across 32 different L2 cache lines (1536-byte stride between lanes, each hitting a different row). That's 32 cache lines × 32 bytes = 1024 bytes written to 32 × 128-byte = 4096 bytes of L2 footprint per warp per store. 8 stores per tile per warp = 256 cache lines touched.

With 4 epilogue warps firing simultaneously, that's 1024 cache line writes per tile's epilogue. The L2 write-back coalescing buffers on Blackwell can absorb this, but if they saturate, you'll see backpressure stalling the store instructions themselves, which chains back to stalling the entire epilogue pipeline you just worked hard to optimize.

The old TMA store path had the same fundamental scatter pattern, but the TMA engine goes through a separate DMA path in the memory controller that can batch 2D writes more efficiently than individual store instructions. The tradeoff was correct (lane-0 serialization was worse), but the scatter efficiency might not be a free win.

**What to profile:**
```bash
# Tells you if store backpressure is a problem
ncu --metrics \
    smsp__warps_issue_stalled_mio_throttle.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_drain.avg.pct_of_peak_sustained_active,\
    l1tex__m_xbar2l1tex_write_bytes.sum,\
    lts__throughput.avg.pct_of_peak_sustained_elapsed \
    -k patch_embed_gemm ./siglip_vision
```

If `stalled_mio_throttle` or `stalled_drain` is high, the scattered stores are the bottleneck and you'd need to rethink — either transpose through SMEM before storing (adds SMEM pressure and sync), or batch stores differently.

### 2. TMEM latency might not be fully hidden

Back-of-napkin math: each chunk between TMEM_LOAD and TMEM_WAIT has ~33 instructions (16 ld.shared + 16 FADD + CVT_STG). At ~1 IPC with 4-warp interleaving, that's ~132 cycles. TMEM load latency is ~200 cycles. There's a ~70-cycle gap where all 4 warps may be stalled on TMEM_WAIT simultaneously.

The double-buffering helps but doesn't fully close the gap because the compute between a TMEM_LOAD and its corresponding TMEM_WAIT is split across two phases (A's compute, then B's load+compute), not all of it between the load and wait.

**What to profile:**
```bash
ncu --metrics \
    smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_not_selected.avg.pct_of_peak_sustained_active \
    -k patch_embed_gemm ./siglip_vision
```

If `stalled_long_scoreboard` is the dominant stall, TMEM latency is still the bottleneck. If `not_selected` is high alongside it, the scheduler has eligible warps — meaning the interleaving is working and you just need more compute per chunk. If both are low, you've won and the bottleneck is elsewhere.

### 3. Bias+pos prefetch loop might be slower than expected

Line 420-421:
```c
for (int j = 0; j < TN; j++)
    sums_out[j * 32] = bp_cur[j] + pp_cur[j];
```

This does 128 iterations with 2 LDG (bias + pos_embed) and 1 STS (store to SMEM) per iteration. The LDG for `bias` is a broadcast (all 32 lanes in a warp load the same `bias[n_start + j]`), which coalesces into a single L2 read. But `pos_embed` access is `pos_embed[pos_row * 768 + n_start + j]` where pos_row varies per lane — 32 different rows, same column. That's 32 scattered L2 reads per j-iteration, same cache line scatter pattern as the stores.

128 iterations × 32 scattered reads = 4096 L2 reads per warp during the prefetch. This happens during the K-loop when W2-5 are "idle," but they're not truly idle — they're competing with W0's TMA loads for memory bandwidth. If the prefetch takes too long, it might not complete before the next epilogue needs the sums, introducing a hidden stall.

**What to profile:**
```bash
# Source-level profiling to see if the prefetch loop is a hotspot
ncu --set source --source-level all \
    -k patch_embed_gemm -o source.ncu-rep ./siglip_vision
```

Look at per-instruction stall cycles in the prefetch loop vs. the epilogue proper.

## The exact profiling sequence to run when you get a B200

```bash
# Build with lineinfo
nvcc -gencode arch=compute_100a,code=sm_100a -O3 -lineinfo \
    megakernel.cu -o siglip_vision -lcurand -lcuda

# 1. Correctness first (catches mbarrier protocol bugs that would hang ncu)
compute-sanitizer --tool synccheck ./siglip_vision
compute-sanitizer --tool racecheck ./siglip_vision
compute-sanitizer --tool memcheck ./siglip_vision

# 2. SASS dump — verify compiler output
cuobjdump --dump-sass siglip_vision > /tmp/sass_new.txt
# Check epilogue loop body: should see TCGEN05.LD → TCGEN05.WAIT → 
# FADD×16 → CVT×8 → STG.E.V8 per chunk. No FENCE.PROXY.ASYNC, 
# no BAR.SYNC.WARP, no CP.ASYNC.BULK.
grep -c 'STG.E' /tmp/sass_new.txt      # expect 8 per epilogue invocation
grep -c 'FENCE.PROXY' /tmp/sass_new.txt  # expect 0
grep -c 'BAR.SYNC' /tmp/sass_new.txt     # expect only the __syncthreads ones

# 3. Full stall breakdown — THE key measurement
ncu --set detailed \
    -k patch_embed_gemm \
    -o detailed.ncu-rep \
    ./siglip_vision

# 4. Targeted: the three concerns above
ncu --metrics \
    smsp__warps_issue_stalled_barrier.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_membar.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_wait.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_not_selected.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_mio_throttle.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_drain.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_lg_throttle.avg.pct_of_peak_sustained_active \
    -k patch_embed_gemm ./siglip_vision

# 5. Memory throughput — is L2 the bottleneck?
ncu --metrics \
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    lts__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
    smsp__inst_executed_op_global_st.sum,\
    smsp__inst_executed_op_global_ld.sum,\
    smsp__inst_executed_op_shared_ld.sum,\
    smsp__inst_executed_op_shared_st.sum \
    -k patch_embed_gemm ./siglip_vision

# 6. Source-level hotspots (open in Nsight Compute GUI)
ncu --set source --source-level all \
    -k patch_embed_gemm -o source.ncu-rep ./siglip_vision
```
The answer to "is this code fast" is entirely in step 3 and 4. 
The stall breakdown tells you what to fix next.
