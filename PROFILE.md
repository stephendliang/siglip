## Rectified Profiling Playbook

Your original notes have a few issues:

1. **`sm__pipe_tensor_op_tcgen05_stalled_reason`** — not a real metric name. The stall reasons live under `smsp__warps_issue_stalled_*`.
2. **`smsp__sass_warp_efficiency.avg`** — deprecated in recent NCU versions. Replaced by warp stall reason breakdown.
3. **`smsp__inst_executed_not_predicated_off.avg`** — real and useful, but only for one specific sub-problem (lane-0 divergence in TMA stores).
4. **`nvprune`** — strips architectures from fatbinaries. Irrelevant for perf diagnosis.
5. **`nm`** — lists symbols. Marginal use here (verifying kernel name for ncu `-k` filter, that's it).

Here's what you actually need, mapped to each TODOLIST item:

---

### Phase 0: Baseline (run before any changes)

**Get the full warp stall breakdown** — this is the single most important measurement:

```bash
ncu --set detailed \
    -k patch_embed_gemm \
    -o baseline.ncu-rep \
    ./siglip_vision
```

Then inspect the **WarpStateStatistics** section. The key metrics:

| Metric | What it reveals | Expected finding |
|--------|----------------|------------------|
| `smsp__warps_issue_stalled_barrier` | `__syncthreads` stalls | **High** — all 6 warps rendezvous every tile (line 396) |
| `smsp__warps_issue_stalled_wait` | `cp.async.bulk.wait_group` stalls | **High** — lane-0 `tma_store_wait_1` serializes epilogue |
| `smsp__warps_issue_stalled_long_scoreboard` | Global/TMEM memory latency | Moderate — `tcgen05.ld` ~200 cycles, only 4-way interleaving |
| `smsp__warps_issue_stalled_not_selected` | Warps eligible but not picked | Low-moderate — indicates enough warps exist but they're all stuck |
| `smsp__warps_issue_stalled_mio_throttle` | Memory I/O backpressure | Indicates TMA store path congestion if high |
| `smsp__warps_issue_stalled_short_scoreboard` | Shared memory / register deps | Shows CVT_STS → SMEM → TMA store dependency chain |

**Get the SASS dump** — verify the compiler output matches your intent:

```bash
cuobjdump --dump-sass siglip_vision > sass_new.txt
```

Count instructions in the epilogue inner loop. In current code, look for this sequence per column-chunk iteration:
```
TCGEN05.LD  (TMEM load)
TCGEN05.WAIT
FADD × 16   (bias+pos)
CVT × 8     (FP32→BF16)
STS × 2     (st.shared)
FENCE.PROXY.ASYNC
BAR.SYNC.WARP
BRANCH (lane==0)
  CP.ASYNC.BULK.WAIT
  CP.ASYNC.BULK.TENSOR.2D (TMA store)
  CP.ASYNC.BULK.COMMIT
```

That's ~30+ instructions with heavy serialization. After item 2, should be ~27 instructions with zero serialization.

**Get instruction predication stats** — quantifies lane-0 divergence:

```bash
ncu --metrics \
    smsp__inst_executed.sum,\
    smsp__inst_executed_not_predicated_off.sum,\
    smsp__thread_inst_executed.sum,\
    smsp__thread_inst_executed_not_predicated_off.sum \
    -k patch_embed_gemm \
    ./siglip_vision
```

The ratio `thread_inst_executed_not_predicated_off / thread_inst_executed` tells you what fraction of lane-cycles are wasted on predicated-off instructions (the `lane==0` branches at lines 380-384, 456-460).

**Memory throughput** — confirm the epilogue is not DRAM-bound:

```bash
ncu --metrics \
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    lts__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__throughput.avg.pct_of_peak_sustained_elapsed \
    -k patch_embed_gemm \
    ./siglip_vision
```

---

### Per-Item Diagnosis

#### Item 1: `__syncthreads` → mbarriers

**Before**: Confirm barrier stall is dominant:
```bash
ncu --metrics \
    smsp__warps_issue_stalled_barrier.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_membar.avg.pct_of_peak_sustained_active \
    -k patch_embed_gemm \
    ./siglip_vision
```

`stalled_barrier` should be the top stall reason. After the fix, it should drop to near-zero. `stalled_membar` may increase slightly (mbarrier waits), but will be far lower because only the dependent warp group waits, not all 6.

**After**: Run correctness checks:
```bash
compute-sanitizer --tool synccheck ./siglip_vision
compute-sanitizer --tool racecheck ./siglip_vision
```

`synccheck` catches incorrect barrier usage (missing arrives, wrong thread counts). `racecheck` catches shared memory races from removed `__syncthreads`.

#### Item 2: TMA stores → `st.global.v8.b32`

**Before**: Confirm TMA store serialization:
```bash
ncu --metrics \
    smsp__warps_issue_stalled_wait.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_active \
    -k patch_embed_gemm \
    ./siglip_vision
```

`stalled_wait` = `cp.async.bulk.wait_group` stalls. `stalled_short_scoreboard` = SMEM staging dependencies.

**After**: Verify via SASS that the epilogue loop body is clean:
```bash
cuobjdump --dump-sass siglip_vision | grep -A 200 "TCGEN05.LD"
```

Should see: `TCGEN05.LD` → `TCGEN05.WAIT` → `FADD`s → `CVT`s → `STG.E.V8` (the `st.global.v8.b32`). No `FENCE.PROXY.ASYNC`, no `BAR.SYNC.WARP`, no `CP.ASYNC.BULK`.

Also verify lane utilization improved:
```bash
ncu --metrics \
    smsp__thread_inst_executed_not_predicated_off.avg.pct_of_peak_sustained_active \
    -k patch_embed_gemm \
    ./siglip_vision
```

Should increase (no more lane==0 branch = all 32 lanes active every cycle).

#### Item 3: 6-way TMEM interleaving

**Before**: Confirm TMEM latency is the remaining bottleneck:
```bash
ncu --metrics \
    smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_not_selected.avg.pct_of_peak_sustained_active \
    -k patch_embed_gemm \
    ./siglip_vision
```

If `stalled_long_scoreboard` is high and `stalled_not_selected` is low after items 1+2, that means you don't have enough warps to hide the TMEM latency — confirming the 6-way interleaving thesis.

#### Item 4: Bias/pos_embed SMEM prefetch

**Before**: Confirm global loads are visible:
```bash
ncu --metrics \
    lts__throughput.avg.pct_of_peak_sustained_elapsed,\
    smsp__warps_issue_stalled_lg_throttle.avg.pct_of_peak_sustained_active,\
    smsp__inst_executed_op_global_ld.sum \
    -k patch_embed_gemm \
    ./siglip_vision
```

`inst_executed_op_global_ld` counts LDG instructions. After prefetch, this should drop significantly (bias+pos loads become LDS instead).

---

### Source-Level Analysis (for pinpointing exact hotspots)

```bash
ncu --set source \
    --source-level all \
    -k patch_embed_gemm \
    -o source_analysis.ncu-rep \
    ./siglip_vision
```

Open in Nsight Compute GUI. This gives per-SASS-instruction stall cycles, showing exactly which instruction in the epilogue loop is the bottleneck (the `tma_store_wait`, the `__syncwarp`, etc.).

Requires compiling with line info:
```bash
nvcc -gencode arch=compute_100a,code=sm_100a -O3 -lineinfo megakernel.cu -o siglip_vision -lcurand -lcuda
```

---

### SASS Deep Dive (nvdisasm for control flow)

```bash
cuobjdump --dump-cubin siglip_vision > kernel.cubin
nvdisasm -cfg kernel.cubin > cfg.dot     # control flow graph
nvdisasm -bbcfg kernel.cubin > bbcfg.dot # basic-block level CFG
nvdisasm -playout kernel.cubin           # physical layout with addresses
```

The CFG is especially useful for verifying the warp-specialized branch structure: you should see 3 distinct subgraphs (W0 load path, W1 MMA path, W2-5 epilogue path) connected only at the entry and exit `__syncthreads`/mbarrier points. After item 1, the subgraphs should be completely disconnected except at tile boundaries via mbarrier.

---

### Correctness Validation (run after every change)

```bash
# Memory errors (OOB access from wrong st.global addressing)
compute-sanitizer --tool memcheck ./siglip_vision

# Shared memory races (after removing __syncthreads)
compute-sanitizer --tool racecheck ./siglip_vision

# Barrier correctness (after mbarrier changes)
compute-sanitizer --tool synccheck ./siglip_vision

# Uninitialized reads (after SMEM buffer deletion)
compute-sanitizer --tool initcheck ./siglip_vision
```

Run order: `synccheck` first (catches barrier bugs that cause hangs), then `racecheck` (catches data races), then `memcheck` (catches addressing bugs from st.global conversion).

---

### Final Comparison

After all items are done, A/B the two kernels:

```bash
# Side-by-side ncu comparison
ncu --set detailed -k patch_embed_gemm -o after.ncu-rep ./siglip_vision
ncu --page raw --csv baseline.ncu-rep > baseline.csv
ncu --page raw --csv after.ncu-rep > after.csv
diff baseline.csv after.csv
```

---

### Discarded from your original notes

| Tool | Why discarded |
|------|---------------|
| `nvprune` | Strips fat binary architectures. You only target `sm_100a`. No use. |
| `nm` | Lists ELF symbols. At most useful for `nm siglip_vision \| grep patch_embed` to confirm the kernel name for `-k` filters. One-time use, not a diagnosis tool. |

---

### Summary: the minimum set

For the iterative optimize→measure→validate loop on this kernel, the tools you actually need are:

1. **`ncu`** — primary perf diagnosis (warp stalls, instruction mix, memory throughput)
2. **`cuobjdump --dump-sass`** — verify compiler output matches intent
3. **`compute-sanitizer`** (synccheck + racecheck + memcheck) — correctness after each barrier/store change
4. **`nvdisasm -cfg`** — verify warp-specialized control flow structure.
