# Making It Faster — Advanced Profiling

Profiling regime for a high-performance kernel (1892 TFLOPS, 48% TC utilization).
Not "what's broken" — "where are the remaining cycles going?"

See `docs/whats_wrong.md` for the old triage-level profiling (barrier stalls, TMA serialization, etc.).

**Chip**: GB100 (B200), `ncu --list-chips` → `gb100`, **NOT** `sm_100a`.

---

## 1. Tensor core utilization — THE metric

The single most important measurement at this performance level. Tells you what fraction of cycles the tensor core is actually computing.

```bash
ncu --metrics \
    sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed,\
    sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
    sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
    sm__inst_executed_pipe_tc.sum,\
    sm__inst_executed_pipe_tc_scope_2cta.sum,\
    sm__inst_executed_pipe_tensor.sum,\
    smsp__cycles_active.avg,\
    smsp__cycles_elapsed.avg \
    -k patch_embed_gemm -c 1 ./siglip_vision
```

**Current values (F13 baseline, 0.579 ms / 1892 TFLOPS):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| `sm__pipe_tc_cycles_active` | **50.1%** | TC control unit active half the time |
| `sm__pipe_tensor_cycles_active` | **48.2%** | Tensor execution pipeline |
| `sm__mem_tensor_cycles_active` | **49.3%** | TMEM active (reads + accumulation) |
| `sm__inst_executed_pipe_tc` | 337,440 | TC micro-ops (includes UTCCP, UTCBAR, etc.) |
| `sm__inst_executed_pipe_tc_scope_2cta` | 337,292 | 99.96% are cta_group::2 — correct |
| `sm__inst_executed_pipe_tensor` | 261,072 | Tensor pipe warp instructions = MMA count |
| `smsp__cycles_active.avg` | 908,593 | Active cycles per SMSP |
| `smsp__cycles_elapsed.avg` | 935,850 | Elapsed cycles per SMSP |

**What this means:**
- TC is idle **50% of the time**. That's where the gap to cuBLAS (3001 TFLOPS, ~79% TC utilization estimated) lives.
- The other 50%: tile transitions, pipeline fill, W1 bookkeeping between MMA instructions, and epilogue interference.
- `pipe_tensor` (261,072) = exact MMA instruction count: 882 per cluster × 148 SMs / 74 clusters × 2 SMs/cluster = 261,072. Matches 147 tiles × 6 K-iters = 882 per cluster. Accounting is clean.
- SM utilization: cycles_active / cycles_elapsed = 97.1%. SMs are almost never idle — the issue is TC idle time *within* active SMs.

**Reference calculation:**
- B200 FP8 dense peak: 4500 TFLOPS
- At 48.2% TC utilization: 4500 × 0.482 = 2169 TFLOPS (theoretical)
- We measure 1892 TFLOPS → 87% of what the active TC time should yield
- The 13% gap within active time = pipeline bubbles within MMA instructions

---

## 2. Per-pipe instruction mix

Decomposes the 122.6M total instructions into functional pipes. Shows where instruction bandwidth goes.

```bash
ncu --metrics \
    sm__inst_executed_pipe_alu.sum,\
    sm__inst_executed_pipe_fma.sum,\
    sm__inst_executed_pipe_fmalite.sum,\
    sm__inst_executed_pipe_fmaheavy.sum,\
    sm__inst_executed_pipe_lsu.sum,\
    sm__inst_executed_pipe_tc.sum,\
    sm__inst_executed_pipe_tensor.sum,\
    sm__inst_executed_pipe_tma.sum,\
    sm__inst_executed_pipe_tmem.sum,\
    sm__inst_executed_pipe_cbu.sum,\
    sm__inst_executed_pipe_adu.sum \
    -k patch_embed_gemm -c 1 ./siglip_vision
```

**Current values:**

| Pipe | Instructions | % of 122.6M | What |
|------|-------------|-------------|------|
| FMA total | 52,584,884 | 42.9% | Epilogue FADD (bias+pos), CVT (FP32→BF16) |
| — fmaheavy | 35,578,191 | 29.0% | FADD, F2FP (FP32→BF16 conversion) |
| — fmalite | 17,006,693 | 13.9% | PRMT, MOV, light ops |
| ALU | 47,497,571 | 38.7% | Integer math (addressing, indexing, loop control) |
| LSU | 12,780,392 | 10.4% | ld.shared, st.shared, ld.global, st.global |
| CBU | 2,439,867 | 2.0% | Branches, predicates, BAR.SYNC |
| ADU | 1,214,594 | 1.0% | Address divergence unit |
| TMEM | 696,192 | 0.6% | tcgen05.ld (epilogue TMEM readback) |
| TC | 337,440 | 0.3% | tcgen05.mma + UTCBAR + UTCCP |
| TMA | 261,072 | 0.2% | cp.async.bulk.tensor loads |

**Key insight:** 99.4% of instructions are non-TC. The TC pipe fires 337K instructions that do all the FLOPs, while 122M instructions handle epilogue compute + addressing + control flow. The instruction mix is overwhelmingly epilogue-dominated, but this doesn't matter if the epilogue runs in the K-loop's shadow (which it does).

---

## 3. SMEM bank conflicts

Phase 1 writes staging (st.shared), Phase 2 reads with transposition (ld.shared). Bank conflicts cause replayed wavefronts.

```bash
ncu --metrics \
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum \
    -k patch_embed_gemm -c 1 ./siglip_vision
```

**Current values:**

| Metric | Value | Rate |
|--------|-------|------|
| SMEM load bank conflicts | 5,370,403 | **32.5%** of load wavefronts |
| SMEM store bank conflicts | 3,209,846 | **22.4%** of store wavefronts |
| SMEM load wavefronts | 16,511,473 | |
| SMEM store wavefronts | 14,349,510 | |

**Interpretation:** 32.5% load conflict rate is high. The STAGING_ROW_PAD (16 bytes = 8 BF16 elements) was meant to prevent this, but the Phase 2 transposed read pattern still hits bank conflicts. Each conflict replays the wavefront, adding 1 extra cycle.

Impact estimate: 5.37M extra cycles from load conflicts. At ~936K cycles per SM: 5.37M / 148 SMs = 36K extra cycles per SM = 3.9% overhead. **Not dominant but not negligible.**

**However:** These conflicts occur in the epilogue, which runs in the K-loop's shadow. Unless the epilogue becomes the longer leg, bank conflicts don't affect wall time. They're a concern only if we reduce K-loop time enough to expose the epilogue.

---

## 4. TMA multicast utilization

With `cta_group::2` and `__cluster_dims__(2,1,1)`, the B matrix tile is the same for both CTAs in the cluster. Multicast would let one TMA load serve both CTAs.

```bash
ncu --metrics \
    l1tex__m_xbar2l1tex_read_sectors_mem_global_op_tma_ld.sum,\
    l1tex__m_xbar2l1tex_read_sectors_mem_global_op_tma_ld_dest_multicast.sum,\
    l1tex__m_xbar2l1tex_read_sectors_mem_global_op_tma_ld_dest_self.sum \
    -k patch_embed_gemm -c 1 ./siglip_vision
```

**Current values:**

| Metric | Value |
|--------|-------|
| TMA load sectors (total) | 133,668,864 |
| TMA load sectors (multicast) | **0** |
| TMA load sectors (self) | 133,668,864 |

**Every single TMA load is self — zero multicast.** Both CTAs load their own copy of the B matrix independently. Enabling multicast for B loads would halve the B matrix bandwidth, reducing L2 and DRAM pressure.

This is potentially actionable even though TMA `wait` stalls are only 1.5% — the bandwidth savings could indirectly speed up TMA loads for A.

---

## 5. TMEM and MMA breakdown

```bash
ncu --metrics \
    sm__mem_tensor_reads.sum,\
    sm__mem_tensor_reads_op_ldt.sum,\
    sm__mem_tensor_reads_op_utcmma_matrix_c.sum,\
    l1tex__data_pipe_tc_wavefronts.sum,\
    l1tex__data_pipe_tc_wavefronts_mem_shared_op_utcmma_matrix_a.sum,\
    l1tex__data_pipe_tc_wavefronts_mem_shared_op_utcmma_matrix_b_scope_2cta.sum \
    -k patch_embed_gemm -c 1 ./siglip_vision
```

**Current values:**

| Metric | Value | Notes |
|--------|-------|-------|
| TMEM reads total | 267.34 MB | |
| TMEM reads (MMA accumulator, matrix_c) | 256.20 MB | 95.8% — MMA read-back |
| TMEM reads (tcgen05.ld, epilogue) | 11.14 MB | 4.2% — epilogue readout |
| TC wavefronts total | 33,417,216 | |
| UTCMMA matrix_a wavefronts | 16,708,608 | 50% — balanced |
| UTCMMA matrix_b scope_2cta | 16,708,608 | 50% — balanced |

**Interpretation:** A and B matrix wavefronts are perfectly balanced (50/50). The SMEM-to-TC data path is symmetric. TMEM traffic is 96% MMA accumulator (expected — this is where partial sums live).

---

## 6. Instruction cache

```bash
ncu --metrics \
    smsp__warps_issue_stalled_no_instruction.avg.pct_of_peak_sustained_active \
    -k patch_embed_gemm -c 1 ./siglip_vision
```

**Current value: 0.22%** — negligible. Despite the large unrolled epilogue, the instruction cache is not a bottleneck. Adding more code (e.g., software pipelining) should be safe from an icache perspective.

---

## 7. Complete stall breakdown

### Percentage-based (scheduler perspective)

What the warp scheduler sees each cycle. `selected` = productive issue.

```bash
ncu --metrics \
    smsp__warps_issue_stalled_selected.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_wait.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_sleeping.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_barrier.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_branch_resolving.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_no_instruction.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_not_selected.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_math_pipe_throttle.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_dispatch_stall.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_mio_throttle.avg.pct_of_peak_sustained_active \
    -k patch_embed_gemm -c 1 ./siglip_vision
```

| Stall | % of peak | Role |
|-------|-----------|------|
| **selected** | **22.82%** | Productive |
| long_scoreboard | 4.71% | TMEM/global latency |
| wait | 1.52% | TMA pipeline |
| sleeping | 0.94% | Parked warps |
| barrier | 0.80% | Cluster sync |
| short_scoreboard | 0.57% | SMEM deps |
| branch_resolving | 0.35% | Branch targets |
| no_instruction | 0.22% | Icache miss |
| not_selected | 0.20% | Eligible, not picked |
| math_pipe_throttle | 0.10% | FMA backpressure |
| dispatch_stall | 0.06% | Dispatch |
| mio_throttle | 0.04% | Memory I/O |
| drain | 0.00% | Post-exit cleanup |
| membar | 0.00% | Memory barriers |

**These percentages are misleading for warp-specialized kernels.** The scheduler picks one warp per cycle. With 7 warps, `selected` = 22.8% means roughly 1.6 warps are ready each cycle. The stall percentages represent the *scheduled warp's* state — they undercount stalls on warps the scheduler doesn't pick.

### Cycle-weighted latency (per-warp truth)

Total accumulated stall cycles across ALL warps, divided by warp count. This correctly weights by how many warps are simultaneously stalled.

```bash
ncu --metrics \
    smsp__average_warp_latency_issue_stalled_selected.ratio,\
    smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio,\
    smsp__average_warp_latency_issue_stalled_wait.ratio,\
    smsp__average_warp_latency_issue_stalled_sleeping.ratio,\
    smsp__average_warp_latency_issue_stalled_barrier.ratio,\
    smsp__average_warp_latency_issue_stalled_short_scoreboard.ratio,\
    smsp__average_warp_latency_issue_stalled_not_selected.ratio,\
    smsp__average_warp_latency_issue_stalled_branch_resolving.ratio,\
    smsp__average_warp_latency_issue_stalled_no_instruction.ratio,\
    smsp__average_warp_latency_issue_stalled_math_pipe_throttle.ratio,\
    smsp__average_warp_latency_issue_stalled_dispatch_stall.ratio \
    -k patch_embed_gemm -c 1 ./siglip_vision
```

| Stall | Avg cycles/warp | Dominant warp role |
|-------|-----------------|-------------------|
| **long_scoreboard** | **390,153** | W2-W6 epilogue (tcgen05.ld TMEM readback) |
| wait | 126,949 | W0 TMA + W2-W6 pipeline waits |
| **selected** | **118,918** | All warps (productive) |
| sleeping | 78,840 | Warps parked between tiles |
| barrier | 68,312 | Cluster sync (bar.sync) |
| short_scoreboard | 47,227 | SMEM ld/st chains |
| branch_resolving | 28,717 | Warp-specialized branch divergence |
| no_instruction | 18,123 | Icache (minor) |
| not_selected | 15,241 | Eligible but not scheduled |
| math_pipe_throttle | 8,251 | FMA pipe full |
| dispatch_stall | 4,782 | Dispatch overhead |

**This is the real picture.** `long_scoreboard` at 390K cycles/warp is **3.3× the productive `selected` time**. Warps spend 3.3× more time waiting for TMEM readback than doing useful work. The scheduler hides this by always picking the ready warp, making `selected` look dominant in the percentage view.

**Critical nuance:** This TMEM stall is overwhelmingly W2-W6 (epilogue warps doing tcgen05.ld). Since the epilogue runs in the K-loop's shadow, these stalls don't directly affect wall time. They only matter if we reduce K-loop time enough to expose the epilogue.

---

## 8. Global access efficiency

```bash
ncu --metrics \
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld,\
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st \
    -k patch_embed_gemm -c 1 ./siglip_vision
```

| Metric | Sectors/request | % of max |
|--------|-----------------|----------|
| Global loads | 32 | 100% (TMA loads, perfect) |
| Global stores | 12.80 | 40% (misleading — see below) |

**The 40% store "efficiency" is NOT a coalescing problem.** Per-instruction analysis (SourceCounters) shows **zero excessive sectors** for every STG instruction. The 40% ratio comes from mixing STG.E.128 (16 sectors ideal) and STG.E.64 (8 sectors ideal); the max is 32 for STG.E.SYS. All stores are perfectly coalesced.

The ncu "32.93% estimated speedup from uncoalesced accesses" warning is a **false positive** — it's comparing total sector traffic against a theoretical minimum that doesn't account for the instruction mix.

---

## 9. Source-level stall attribution

Since ncu can't filter by warp ID, use source-level profiling to attribute stalls to SASS address ranges. Each warp role executes distinct code paths (W0 = TMA loads, W1 = MMA, W2-W6 = epilogue), so per-instruction stall counts effectively give per-role attribution.

```bash
ncu --section SourceCounters --page source --csv \
    -k patch_embed_gemm -c 1 ./siglip_vision > source_counters.csv
```

Then filter for hot instructions:
```python
import csv
with open('source_counters.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        stalls = int(row.get('Warp Stall Sampling (All Samples)', 0))
        if stalls > 20:
            print(f"{row['Address']} {row['Source'][:50]:50s} stalls={stalls}")
```

**Known limitation (NCU 2025.3.1.0):** `--set source` (which enables InstructionStats + WarpStateStats) may hang on B200 for kernels using `cp.async.bulk` + `mbarrier.try_wait.parity`. Use `--section SourceCounters` alone (safe).

---

## Summary: what we know and what to attack

### The gap breakdown (1892 vs cuBLAS 3001 TFLOPS)

| Component | Evidence | Est. impact |
|-----------|----------|-------------|
| TC idle 50% of the time | pipe_tc 50.1% | **Primary** — this IS the gap |
| TC only 87% efficient when active | 1892 vs 2169 theoretical | ~13% within-active loss |
| Zero TMA multicast for B | dest_multicast = 0 | Wastes L2/DRAM bandwidth |
| SMEM bank conflicts 32% | epilogue-only, hidden | 0% (in K-loop shadow) |
| Instruction cache | 0.22% | Negligible |

### What would actually help

1. **Reduce TC idle time** — the 50% idle is tile transitions + pipeline fill + W1 overhead. Measure per-tile overhead with instrumentation (clock64 timestamps at MMA start/end).

2. **Enable TMA multicast for B loads** — halves B matrix bandwidth, may reduce TMA `wait` stalls and L2 contention.

3. **Profile W1 specifically** — the aggregate stall data is dominated by W2-W6. Need source-level attribution to isolate what W1 (the sole MMA issuer) is stalled on between MMA instructions.
