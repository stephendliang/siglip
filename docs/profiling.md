# Profiling Playbook

Profiling workflow for the patch embed GEMM epilogue on SM100 (Blackwell).
Run each section **before and after** every optimization item in `docs/tasks.md`.

## Tools

| Tool | Purpose | When to use |
|------|---------|-------------|
| `ncu` (Nsight Compute) | Kernel-level metrics: warp stalls, memory throughput, instruction mix | Every change — this is the primary tool |
| `cuobjdump --dump-sass` | Dump SASS assembly from compiled binary | Every change — verify compiler output matches intent |
| `nvdisasm -cfg` | Control flow graph from cubin | After structural changes (mbarriers, warp-specialization) |
| `compute-sanitizer` | Correctness: race detection, sync errors, OOB access | After every barrier or store change |
| `nsys` (Nsight Systems) | System timeline: kernel duration, launch overhead, GPU idle gaps | Occasional — for end-to-end timing validation |

### Discarded tools

| Tool | Why not needed |
|------|----------------|
| `nvprune` | Strips fat binary architectures. We only target `sm_100a`. No use. |
| `nm` | Lists ELF symbols. At most useful for `nm siglip_vision \| grep patch_embed` to confirm the kernel name for `-k` filters. One-time use, not a diagnosis tool. |

## Build with line info

Source-level profiling requires `-lineinfo`:

```bash
nvcc -gencode arch=compute_100a,code=sm_100a -O3 -lineinfo megakernel.cu -o siglip_vision -lcurand -lcuda
```

This adds debug line mappings without affecting optimization. Use for `ncu --set source` runs.

## Phase 0: Baseline (run before any changes)

### Full warp stall breakdown

This is the single most important measurement.

```bash
ncu --set detailed \
    -k patch_embed_gemm \
    -o baseline.ncu-rep \
    ./siglip_vision
```

The **WarpStateStatistics** section shows why warps are stalled:

| Metric | What it reveals | Expected finding |
|--------|----------------|------------------|
| `smsp__warps_issue_stalled_barrier` | `__syncthreads` / `bar.sync` stalls | **High** — all 6 warps rendezvous every tile (line 396) |
| `smsp__warps_issue_stalled_wait` | `cp.async.bulk.wait_group` stalls | **High** — lane-0 `tma_store_wait_1` serializes epilogue |
| `smsp__warps_issue_stalled_long_scoreboard` | Global/TMEM memory latency | Moderate — `tcgen05.ld` ~200 cycles, only 4-way interleaving |
| `smsp__warps_issue_stalled_not_selected` | Warps eligible but not picked | Low-moderate — indicates enough warps exist but they're all stuck |
| `smsp__warps_issue_stalled_mio_throttle` | Memory I/O backpressure | Indicates TMA store path congestion if high |
| `smsp__warps_issue_stalled_short_scoreboard` | Shared memory / register deps | Shows CVT_STS → SMEM → TMA store dependency chain |

### SASS dump

Verify the compiler output matches intent:

```bash
cuobjdump --dump-sass siglip_vision > docs/reference/sass_dump.txt
```

Count instructions in the epilogue inner loop. Current (pre-optimization) per column-chunk iteration:
```
TCGEN05.LD       (TMEM load)
TCGEN05.WAIT
FADD × 16       (bias+pos)
CVT × 8         (FP32→BF16)
STS × 2         (st.shared)
FENCE.PROXY.ASYNC
BAR.SYNC.WARP
BRANCH (lane==0)
  CP.ASYNC.BULK.WAIT
  CP.ASYNC.BULK.TENSOR.2D (TMA store)
  CP.ASYNC.BULK.COMMIT
```

That's ~30+ instructions with heavy serialization. After task #2, should be:
```
TCGEN05.LD       (TMEM load)
TCGEN05.WAIT
FADD × 16       (bias+pos)
CVT × 8         (FP32→BF16)
STG.E.V8 × 1    (direct global store)
```

### Instruction predication stats

Quantifies lane-0 divergence from the `lane==0` branches (lines 380-384, 456-460):

```bash
ncu --metrics \
    smsp__inst_executed.sum,\
    smsp__inst_executed_not_predicated_off.sum,\
    smsp__thread_inst_executed.sum,\
    smsp__thread_inst_executed_not_predicated_off.sum \
    -k patch_embed_gemm \
    ./siglip_vision
```

The ratio `thread_inst_executed_not_predicated_off / thread_inst_executed` tells you what fraction of lane-cycles are wasted on predicated-off instructions. Currently low because 31 of 32 lanes diverge off during TMA store issue.

### Memory throughput

Confirm the epilogue is not DRAM-bound (it shouldn't be — the bottleneck is instruction-level):

```bash
ncu --metrics \
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    lts__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__throughput.avg.pct_of_peak_sustained_elapsed \
    -k patch_embed_gemm \
    ./siglip_vision
```

## Per-task profiling

### Task #1: `__syncthreads` → mbarriers

**Before** — confirm barrier stall is dominant:
```bash
ncu --metrics \
    smsp__warps_issue_stalled_barrier.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_membar.avg.pct_of_peak_sustained_active \
    -k patch_embed_gemm ./siglip_vision
```

`stalled_barrier` should be the top stall reason. After the fix, it should drop to near-zero. `stalled_membar` may increase slightly (mbarrier waits), but will be far lower because only the dependent warp group waits, not all 6.

**Correctness** — must run after changing any barrier:
```bash
compute-sanitizer --tool synccheck ./siglip_vision   # barrier protocol errors
compute-sanitizer --tool racecheck ./siglip_vision   # shared memory races
```

`synccheck` catches incorrect barrier usage (missing arrives, wrong thread counts). `racecheck` catches shared memory races from removed `__syncthreads`.

### Task #2: TMA stores → `st.global.v8.b32`

**Before** — confirm TMA store serialization:
```bash
ncu --metrics \
    smsp__warps_issue_stalled_wait.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_active \
    -k patch_embed_gemm ./siglip_vision
```

`stalled_wait` = `cp.async.bulk.wait_group` stalls. `stalled_short_scoreboard` = SMEM staging dependencies.

**After** — verify SASS epilogue is clean:
```bash
cuobjdump --dump-sass siglip_vision | grep -A 200 "TCGEN05.LD"
```

Should see: `TCGEN05.LD` → `TCGEN05.WAIT` → `FADD`s → `CVT`s → `STG.E.V8`. No `FENCE.PROXY.ASYNC`, no `BAR.SYNC.WARP`, no `CP.ASYNC.BULK`.

**After** — confirm lane utilization improved (no more lane==0 divergence):
```bash
ncu --metrics \
    smsp__thread_inst_executed_not_predicated_off.avg.pct_of_peak_sustained_active \
    -k patch_embed_gemm ./siglip_vision
```

Should increase (no more lane==0 branch = all 32 lanes active every cycle).

### Task #3: Software-pipeline TMEM loads

**Before** — confirm TMEM latency is the remaining bottleneck:
```bash
ncu --metrics \
    smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_not_selected.avg.pct_of_peak_sustained_active \
    -k patch_embed_gemm ./siglip_vision
```

If `stalled_long_scoreboard` is high and `stalled_not_selected` is low after tasks #1+#2, that means 4 warps can't hide the TMEM latency — confirming the software-pipeline thesis.

**After** — `stalled_long_scoreboard` should drop significantly because the next `tcgen05.ld` is issued before the current compute phase, doubling the overlap window.

### Task #4: Bias/pos_embed SMEM prefetch

**Before** — confirm global loads are visible:
```bash
ncu --metrics \
    lts__throughput.avg.pct_of_peak_sustained_elapsed,\
    smsp__warps_issue_stalled_lg_throttle.avg.pct_of_peak_sustained_active,\
    smsp__inst_executed_op_global_ld.sum \
    -k patch_embed_gemm ./siglip_vision
```

`inst_executed_op_global_ld` counts LDG instructions. After prefetch, this should drop significantly (bias+pos loads become LDS instead).

## Control flow graph

Verify warp-specialized branch structure after structural changes:

```bash
cuobjdump --dump-cubin siglip_vision > /tmp/kernel.cubin
nvdisasm -cfg /tmp/kernel.cubin > /tmp/cfg.dot
nvdisasm -bbcfg /tmp/kernel.cubin > /tmp/bbcfg.dot
nvdisasm -playout /tmp/kernel.cubin           # physical layout with addresses
```

The CFG is especially useful for verifying the warp-specialized branch structure: you should see 3 distinct subgraphs (W0 load path, W1 MMA path, W2-5 epilogue path) connected only at the entry and exit `__syncthreads`/mbarrier points. After task #1, the subgraphs should be completely disconnected except at tile boundaries via mbarrier.

## Source-level hotspot analysis

Pinpoint exact stall cycles per SASS instruction:

```bash
ncu --set source --source-level all \
    -k patch_embed_gemm -o source.ncu-rep ./siglip_vision
```

Open in Nsight Compute GUI. This gives per-SASS-instruction stall cycles, showing exactly which instruction in the epilogue loop is the bottleneck (the `tma_store_wait`, the `__syncwarp`, etc.).

Requires the `-lineinfo` build (see above).

## Correctness validation

Run after **every** barrier or store change, in this order:

```bash
# 1. Barrier correctness (catches hangs from wrong mbarrier protocol)
compute-sanitizer --tool synccheck ./siglip_vision

# 2. Shared memory races (catches races from removed __syncthreads)
compute-sanitizer --tool racecheck ./siglip_vision

# 3. Memory errors (catches OOB from wrong st.global addressing)
compute-sanitizer --tool memcheck ./siglip_vision

# 4. Uninitialized reads (catches stale data after SMEM buffer deletion)
compute-sanitizer --tool initcheck ./siglip_vision
```

Run order: `synccheck` first (catches barrier bugs that cause hangs), then `racecheck` (catches data races), then `memcheck` (catches addressing bugs from st.global conversion).

## A/B comparison

After all tasks are done, compare baseline vs optimized:

```bash
ncu --set detailed -k patch_embed_gemm -o after.ncu-rep ./siglip_vision
ncu --page raw --csv baseline.ncu-rep > /tmp/baseline.csv
ncu --page raw --csv after.ncu-rep > /tmp/after.csv
diff /tmp/baseline.csv /tmp/after.csv
```

## Summary: the minimum set

For the iterative optimize → measure → validate loop on this kernel:

1. **`ncu`** — primary perf diagnosis (warp stalls, instruction mix, memory throughput)
2. **`cuobjdump --dump-sass`** — verify compiler output matches intent
3. **`compute-sanitizer`** (synccheck + racecheck + memcheck) — correctness after each barrier/store change
4. **`nvdisasm -cfg`** — verify warp-specialized control flow structure
