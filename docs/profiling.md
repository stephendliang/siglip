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

**Not needed**: `nvprune` (strips fat binary architectures, irrelevant for single-target sm_100a), `nm` (symbol listing, marginal use).

## Build with line info

Source-level profiling requires `-lineinfo`:

```bash
nvcc -gencode arch=compute_100a,code=sm_100a -O3 -lineinfo megakernel.cu -o siglip_vision -lcurand -lcuda
```

This adds debug line mappings without affecting optimization. Use for `ncu --set source` runs.

## Baseline: full warp stall breakdown

This is the single most important measurement. Run before any changes.

```bash
ncu --set detailed -k patch_embed_gemm -o baseline.ncu-rep ./siglip_vision
```

The **WarpStateStatistics** section shows why warps are stalled:

| Metric | What it reveals | Expected before optimization |
|--------|----------------|------------------------------|
| `smsp__warps_issue_stalled_barrier` | `__syncthreads` / `bar.sync` stalls | **High** — all 6 warps rendezvous every tile |
| `smsp__warps_issue_stalled_wait` | `cp.async.bulk.wait_group` stalls | **High** — lane-0 `tma_store_wait` serializes epilogue |
| `smsp__warps_issue_stalled_long_scoreboard` | Global/TMEM memory latency | Moderate — `tcgen05.ld` ~200 cycles, 4-way interleaving |
| `smsp__warps_issue_stalled_short_scoreboard` | Shared memory / register deps | CVT_STS → SMEM → TMA store dependency chain |
| `smsp__warps_issue_stalled_not_selected` | Warp eligible but not picked | Low-moderate |
| `smsp__warps_issue_stalled_mio_throttle` | Memory I/O backpressure | TMA store path congestion if high |

## Per-task profiling

### Task #1: `__syncthreads` → mbarriers

**Before** — confirm barrier stall is dominant:
```bash
ncu --metrics \
    smsp__warps_issue_stalled_barrier.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_membar.avg.pct_of_peak_sustained_active \
    -k patch_embed_gemm ./siglip_vision
```

**After** — `stalled_barrier` should drop to near-zero. `stalled_membar` may increase slightly (mbarrier waits are lighter but still show up here).

**Correctness** — must run after changing any barrier:
```bash
compute-sanitizer --tool synccheck ./siglip_vision   # barrier protocol errors
compute-sanitizer --tool racecheck ./siglip_vision   # shared memory races
```

### Task #2: TMA stores → `st.global.v8.b32`

**Before** — confirm TMA store serialization:
```bash
ncu --metrics \
    smsp__warps_issue_stalled_wait.avg.pct_of_peak_sustained_active,\
    smsp__warps_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_active \
    -k patch_embed_gemm ./siglip_vision
```

**After** — verify SASS epilogue is clean:
```bash
cuobjdump --dump-sass siglip_vision | grep -A 200 "TCGEN05.LD"
```

Should see: `TCGEN05.LD` → `FADD`s → `CVT`s → `STG.E.V8`. No `FENCE.PROXY.ASYNC`, no `BAR.SYNC.WARP`, no `CP.ASYNC.BULK`.

**After** — confirm lane utilization improved (no more lane==0 divergence):
```bash
ncu --metrics \
    smsp__thread_inst_executed.sum,\
    smsp__thread_inst_executed_not_predicated_off.sum \
    -k patch_embed_gemm ./siglip_vision
```

Ratio `not_predicated_off / total` should increase (all 32 lanes active every cycle).

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

**Before** — count global loads in epilogue:
```bash
ncu --metrics \
    smsp__inst_executed_op_global_ld.sum,\
    lts__throughput.avg.pct_of_peak_sustained_elapsed \
    -k patch_embed_gemm ./siglip_vision
```

**After** — `inst_executed_op_global_ld` should drop (bias+pos loads become LDS instead of LDG). L2 throughput should decrease (less global traffic).

## Memory throughput

Confirm the epilogue is not DRAM-bound (it shouldn't be — the bottleneck is instruction-level):

```bash
ncu --metrics \
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    lts__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__throughput.avg.pct_of_peak_sustained_elapsed \
    -k patch_embed_gemm ./siglip_vision
```

## SASS dump and analysis

Dump the full SASS to verify compiler output:
```bash
cuobjdump --dump-sass siglip_vision > docs/reference/sass_dump.txt
```

Count epilogue loop body instructions. Current (pre-optimization) inner loop per column chunk:
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
  CP.ASYNC.BULK.TENSOR.2D
  CP.ASYNC.BULK.COMMIT
```

After task #2, should be:
```
TCGEN05.LD       (TMEM load)
TCGEN05.WAIT
FADD × 16       (bias+pos)
CVT × 8         (FP32→BF16)
STG.E.V8 × 1    (direct global store)
```

## Control flow graph

Verify warp-specialized branch structure after structural changes:
```bash
cuobjdump --dump-cubin siglip_vision > /tmp/kernel.cubin
nvdisasm -cfg /tmp/kernel.cubin > /tmp/cfg.dot
nvdisasm -bbcfg /tmp/kernel.cubin > /tmp/bbcfg.dot
```

After task #1 (mbarriers), the CFG should show 3 disconnected subgraphs (W0 load, W1 MMA, W2-5 epilogue) connected only at entry/exit via mbarrier arrive/wait — not by `__syncthreads` edges.

## Source-level hotspot analysis

Pinpoint exact stall cycles per SASS instruction:
```bash
ncu --set source --source-level all \
    -k patch_embed_gemm -o source.ncu-rep ./siglip_vision
```

Open in Nsight Compute GUI. Shows per-instruction stall cycles — identifies whether the bottleneck is the `tma_store_wait`, the `__syncwarp`, or the `tcgen05.ld.sync` wait.

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

## A/B comparison

After all tasks are done, compare baseline vs optimized:
```bash
ncu --set detailed -k patch_embed_gemm -o after.ncu-rep ./siglip_vision
ncu --page raw --csv baseline.ncu-rep > /tmp/baseline.csv
ncu --page raw --csv after.ncu-rep > /tmp/after.csv
diff /tmp/baseline.csv /tmp/after.csv
```
