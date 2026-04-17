#!/bin/bash
# FC2 c4 ncu diagnosis — find the 2x slowdown of fc2_w3_c4 vs fc2_w3.
#
# Head-to-head: c2 baseline (fc2-w3) vs c4 A-multicast (fc2-w3-c4) at strip,
# gemm, and full-fused levels.  Separate script so its outputs stay isolated
# from fc2_ncu_bench.sh (which compares dispatch variants on c2).
#
# Hypothesis buckets:
#   H1  Cluster placement: cluster_dims(4,1,1) forces tight GPC packing →
#       worse TMA/DRAM locality than 74 independent c2 clusters.
#   H2  cta_group::2 multicast mbar overhead (peer-bit routing, 2 pair
#       leaders arrived per stage) vs c2's simple pair-leader mbar.
#   H3  Cluster-wide barrier.cluster at init/dealloc costs more with 4 CTAs
#       vs 2 CTAs (6 cross-CTA sync pairs vs 1).
#   H4  Epilogue mainloop_mbar / epilogue_mbar contention: both pairs in
#       the same cluster racing for TMA store ports.
#   H5  pair1 (CTA2,CTA3) in SHARED mode has no A-issue work — idle W0
#       stalls, dispatch imbalance.
#
# Diff targets:
#   Q1  c2 strip vs c4 strip       — pure GEMM core: dispatch cost
#   Q2  c2 gemm  vs c4 gemm        — + TMA store: epilogue store contention
#   Q3  c2 fused vs c4 fused       — + residual load: cluster-wide mbar cost
#   Q4  c2 lean  vs c4 fused       — our best c2 vs c4
#
# Usage:
#   ./tools/fc2_c4_ncu.sh                # full run
#   ./tools/fc2_c4_ncu.sh --dry-run      # print commands
#
# Output: data/c4_ncu_YYYYMMDD_HHMMSS/

set -uo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="data/c4_ncu_${TIMESTAMP}"
mkdir -p "$OUTDIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/session.log"; }

log "========================================"
log "  FC2 c4 NCU DIAGNOSIS  $TIMESTAMP"
log "  Output: $OUTDIR"
log "========================================"

if [ "$DRY_RUN" = "0" ]; then
    nvidia-smi > /dev/null 2>&1 || { log "FATAL: no GPU"; exit 1; }
    nvidia-smi --query-gpu=gpu_name,clocks.sm --format=csv,noheader | tee -a "$OUTDIR/session.log"
fi

# ── Metrics ──

METRICS_STALL="\
smsp__warps_issue_stalled_long_scoreboard.avg,\
smsp__warps_issue_stalled_short_scoreboard.avg,\
smsp__warps_issue_stalled_wait.avg,\
smsp__warps_issue_stalled_barrier.avg,\
smsp__warps_issue_stalled_sleeping.avg,\
smsp__warps_issue_stalled_not_selected.avg,\
smsp__warps_issue_stalled_mio_throttle.avg,\
smsp__warps_issue_stalled_math_pipe_throttle.avg,\
smsp__warps_issue_stalled_membar.avg,\
smsp__warps_issue_stalled_imc_miss.avg"

METRICS_MEM="\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
dram__sectors_read.sum,\
dram__sectors_write.sum,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sectors.sum,\
lts__t_sectors_op_read.sum,\
lts__t_sectors_op_write.sum,\
lts__t_sector_hit_rate.pct,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed"

METRICS_PIPE="\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed,\
sm__mio_pq_read_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__mio_pq_write_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.per_cycle_active,\
smsp__cycles_active.avg,\
smsp__inst_executed.sum"

METRICS_SCHED="\
smsp__warps_eligible.avg.per_cycle_active,\
launch__registers_per_thread,\
launch__occupancy,\
launch__block_size,\
launch__cluster_size,\
launch__grid_size"

METRICS_SMEM="\
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_global_op_ld.sum"

METRICS_TMA="\
l1tex__t_requests_pipe_tma_opc_read.sum,\
l1tex__t_requests_pipe_tma_opc_write.sum,\
l1tex__t_bytes_pipe_tma_opc_read.sum,\
l1tex__t_bytes_pipe_tma_opc_write.sum"

METRICS_ALL="${METRICS_STALL},${METRICS_MEM},${METRICS_PIPE},${METRICS_SCHED},${METRICS_SMEM},${METRICS_TMA}"

# ── Binaries: label|build_cmd|binary|kernel_filter ──
# Both sides use PACKED_TILES (tile-contiguous DRAM) for an apples-to-apples
# comparison.  c2 uses fc2-w3-packed / fc2-w3-packed-lean; c4 is packed-only.
BINARIES=(
    "c2_strip|make -B fc2-w3-packed DFLAGS=-DSTRIP_EPILOGUE|COPY:fc2-w3-packed:fc2-w3-packed-strip|fc2_w3_kernel"
    "c2_gemm|make -B fc2-w3-packed DFLAGS=-DGEMM_ONLY|COPY:fc2-w3-packed:fc2-w3-packed-gemm|fc2_w3_kernel"
    "c2_fused|make -B fc2-w3-packed|./fc2-w3-packed|fc2_w3_kernel"
    "c2_lean|make -B fc2-w3-packed-lean|./fc2-w3-packed-lean|fc2_w3_kernel"
    "c4_strip|make -B fc2-w3-c4-strip|./fc2-w3-c4-strip|fc2_w3_c4_kernel"
    "c4_gemm|make -B fc2-w3-c4-gemm|./fc2-w3-c4-gemm|fc2_w3_c4_kernel"
    "c4_fused|make -B fc2-w3-c4|./fc2-w3-c4|fc2_w3_c4_kernel"
)

# ══════════════════════════════════════════════════════════════════
#  PHASE 1: Build
# ══════════════════════════════════════════════════════════════════

log ""
log "── Phase 1: Building binaries ──"

for entry in "${BINARIES[@]}"; do
    IFS='|' read -r label build_cmd binary kfilter <<< "$entry"
    log "  [$label] $build_cmd"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  $build_cmd"
        continue
    fi

    if ! eval "$build_cmd" > "$OUTDIR/${label}_build.log" 2>&1; then
        log "  [$label] BUILD FAILED"
        cat "$OUTDIR/${label}_build.log" >> "$OUTDIR/session.log"
        continue
    fi

    if [[ "$binary" == COPY:* ]]; then
        src="${binary#COPY:}"
        src_bin="${src%%:*}"
        dst_bin="${src#*:}"
        cp "$src_bin" "$dst_bin"
        log "  [$label] copied $src_bin -> $dst_bin"
    fi

    regs=$(grep -o '[0-9]* registers' "$OUTDIR/${label}_build.log" | tail -1 | grep -o '[0-9]*')
    bars=$(grep -o 'used [0-9]* barriers' "$OUTDIR/${label}_build.log" | tail -1 | grep -o '[0-9]*')
    log "  [$label] built (regs=$regs, bars=$bars)"
done

# ══════════════════════════════════════════════════════════════════
#  PHASE 2: Wall time
# ══════════════════════════════════════════════════════════════════

log ""
log "── Phase 2: Wall time sanity ──"

for entry in "${BINARIES[@]}"; do
    IFS='|' read -r label build_cmd binary kfilter <<< "$entry"

    if [[ "$binary" == COPY:* ]]; then
        binary="./${binary##*:}"
    fi

    if [ "$DRY_RUN" = "1" ]; then
        echo "  $binary"
        continue
    fi

    if [ ! -x "$binary" ]; then
        log "  [$label] MISSING: $binary"
        echo "@@RESULT ms=ERR label=$label" >> "$OUTDIR/results.txt"
        continue
    fi

    output=$(timeout 30 $binary 2>&1) || true
    echo "$output" > "$OUTDIR/${label}_wall.txt"

    result_line=$(echo "$output" | grep '@@RESULT' | head -1)
    if [ -n "$result_line" ]; then
        ms=$(echo "$result_line" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
        valid=$(echo "$result_line" | grep -o 'valid=[01]' | cut -d= -f2)
        tflops=$(echo "$result_line" | grep -o 'tflops=[0-9.]*' | cut -d= -f2)
        echo "${result_line} label=${label}" >> "$OUTDIR/results.txt"
        log "  [$label] ${ms}ms  ${tflops} TFLOPS  valid=${valid}"
    else
        log "  [$label] NO @@RESULT LINE"
        echo "@@RESULT ms=ERR label=${label}" >> "$OUTDIR/results.txt"
    fi
done

# ══════════════════════════════════════════════════════════════════
#  PHASE 3: ncu metrics
# ══════════════════════════════════════════════════════════════════

log ""
log "── Phase 3: ncu metric collection ──"

for entry in "${BINARIES[@]}"; do
    IFS='|' read -r label build_cmd binary kfilter <<< "$entry"

    if [[ "$binary" == COPY:* ]]; then
        binary="./${binary##*:}"
    fi

    if [ "$DRY_RUN" = "1" ]; then
        echo "  ncu --metrics ... --kernel-name $kfilter -o $OUTDIR/$label $binary"
        continue
    fi

    if [ ! -x "$binary" ]; then
        log "  [$label] SKIP (no binary)"
        continue
    fi

    log "  [$label] collecting ncu metrics..."
    if ncu --metrics "$METRICS_ALL" \
           --kernel-name "$kfilter" \
           -o "$OUTDIR/${label}" \
           "$binary" \
           > "$OUTDIR/${label}_ncu_stdout.txt" 2>"$OUTDIR/${label}_ncu_stderr.txt"; then

        ncu --import "$OUTDIR/${label}.ncu-rep" --csv \
            > "$OUTDIR/${label}.csv" 2>/dev/null

        lines=$(wc -l < "$OUTDIR/${label}.csv")
        log "  [$label] ncu done ($lines CSV lines)"
    else
        log "  [$label] NCU FAILED"
        cat "$OUTDIR/${label}_ncu_stderr.txt" >> "$OUTDIR/session.log"
    fi
done

# ══════════════════════════════════════════════════════════════════
#  PHASE 4: Targeted diffs — c4 vs c2 at each layer
# ══════════════════════════════════════════════════════════════════

log ""
log "── Phase 4: c4 vs c2 diffs ──"

run_diff() {
    local label="$1" csv_a="$2" csv_b="$3"
    if [ -f "$csv_a" ] && [ -f "$csv_b" ]; then
        log "  $label: $(basename "$csv_a") vs $(basename "$csv_b")"
        python3 tools/ncu_diff.py "$csv_a" "$csv_b" > "$OUTDIR/${label}.txt" 2>&1 || true
    else
        log "  $label: SKIP (missing CSV)"
    fi
}

if [ "$DRY_RUN" = "0" ]; then
    # Q1: strip — pure GEMM core (no epilogue)
    run_diff "diff_q1_strip"  "$OUTDIR/c2_strip.csv"  "$OUTDIR/c4_strip.csv"
    # Q2: gemm — + TMA store
    run_diff "diff_q2_gemm"   "$OUTDIR/c2_gemm.csv"   "$OUTDIR/c4_gemm.csv"
    # Q3: fused — + residual load + bias
    run_diff "diff_q3_fused"  "$OUTDIR/c2_fused.csv"  "$OUTDIR/c4_fused.csv"
    # Q4: c2-lean (best c2) vs c4-fused
    run_diff "diff_q4_lean_vs_c4" "$OUTDIR/c2_lean.csv" "$OUTDIR/c4_fused.csv"
fi

# ══════════════════════════════════════════════════════════════════
#  PHASE 5: Summary
# ══════════════════════════════════════════════════════════════════

log ""
log "── Phase 5: Summary ──"

if [ "$DRY_RUN" = "0" ]; then
    python3 - "$OUTDIR" "${BINARIES[@]}" << 'PYEOF' 2>&1 | tee "$OUTDIR/summary.txt" | tee -a "$OUTDIR/session.log"
import sys, csv, os

outdir = sys.argv[1]
entries = sys.argv[2:]

KEY_METRICS = [
    # Stall reasons
    ("long_scoreboard",   "smsp__warps_issue_stalled_long_scoreboard.avg"),
    ("short_scoreboard",  "smsp__warps_issue_stalled_short_scoreboard.avg"),
    ("wait",              "smsp__warps_issue_stalled_wait.avg"),
    ("barrier",           "smsp__warps_issue_stalled_barrier.avg"),
    ("membar",            "smsp__warps_issue_stalled_membar.avg"),
    ("imc_miss",          "smsp__warps_issue_stalled_imc_miss.avg"),
    ("sleeping",          "smsp__warps_issue_stalled_sleeping.avg"),
    ("not_selected",      "smsp__warps_issue_stalled_not_selected.avg"),
    ("mio_throttle",      "smsp__warps_issue_stalled_mio_throttle.avg"),
    ("math_throttle",     "smsp__warps_issue_stalled_math_pipe_throttle.avg"),
    # DRAM
    ("dram_read_bytes",   "dram__bytes_read.sum"),
    ("dram_write_bytes",  "dram__bytes_write.sum"),
    ("dram_read_sectors", "dram__sectors_read.sum"),
    ("dram_pct",          "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
    # L2
    ("lts_pct",           "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("lts_sectors",       "lts__t_sectors.sum"),
    ("lts_hit_rate",      "lts__t_sector_hit_rate.pct"),
    ("l1tex_pct",         "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"),
    # Pipe
    ("sm_pct",            "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("shared_pct",        "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("warps_active",      "sm__warps_active.avg.per_cycle_active"),
    ("cycles_active",     "smsp__cycles_active.avg"),
    ("inst_executed",     "smsp__inst_executed.sum"),
    # Scheduler
    ("warps_eligible",    "smsp__warps_eligible.avg.per_cycle_active"),
    ("regs_per_thread",   "launch__registers_per_thread"),
    ("occupancy",         "launch__occupancy"),
    ("block_size",        "launch__block_size"),
    ("cluster_size",      "launch__cluster_size"),
    ("grid_size",         "launch__grid_size"),
    # SMEM
    ("smem_wavefronts",   "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"),
    ("smem_ld_wf",        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"),
    ("smem_st_wf",        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"),
    ("global_ld_wf",      "l1tex__data_pipe_lsu_wavefronts_mem_global_op_ld.sum"),
    # TMA
    ("tma_read_reqs",     "l1tex__t_requests_pipe_tma_opc_read.sum"),
    ("tma_write_reqs",    "l1tex__t_requests_pipe_tma_opc_write.sum"),
    ("tma_read_bytes",    "l1tex__t_bytes_pipe_tma_opc_read.sum"),
    ("tma_write_bytes",   "l1tex__t_bytes_pipe_tma_opc_write.sum"),
]

labels = []
data = {}

for entry in entries:
    parts = entry.split('|')
    label = parts[0]
    csv_path = os.path.join(outdir, f"{label}.csv")
    if not os.path.exists(csv_path):
        continue
    labels.append(label)
    data[label] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Metric Name", "")
            val = row.get("Metric Value", "")
            unit = row.get("Metric Unit", "")
            data[label][name] = (val, unit)

if not labels:
    print("No CSV data found.")
    sys.exit(0)

col_w = 14
hdr = f"{'metric':<22}"
for l in labels:
    hdr += f"  {l:>{col_w}}"
print(hdr)
print("-" * len(hdr))

UNIT_SCALE = {"Gbyte": 1e9, "Mbyte": 1e6, "Kbyte": 1e3, "byte": 1,
              "Gsector": 1e9, "Msector": 1e6, "Ksector": 1e3, "sector": 1}

def to_base(val_str, unit_str):
    try:
        v = float(val_str.replace(",", ""))
    except (ValueError, AttributeError):
        return None
    scale = UNIT_SCALE.get(unit_str, 1.0)
    return v * scale

def fmt_val(val_str, unit_str):
    bv = to_base(val_str, unit_str)
    if bv is None:
        return val_str
    if abs(bv) >= 1e9:
        return f"{bv/1e9:.2f}G"
    elif abs(bv) >= 1e6:
        return f"{bv/1e6:.2f}M"
    elif abs(bv) >= 1e3:
        return f"{bv/1e3:.1f}K"
    else:
        return f"{bv:.2f}"

for short, full in KEY_METRICS:
    row = f"{short:<22}"
    for l in labels:
        entry = data[l].get(full, ("—", ""))
        if isinstance(entry, tuple):
            v = fmt_val(entry[0], entry[1])
        else:
            v = entry
        row += f"  {v:>{col_w}}"
    print(row)

print()
print("=" * 80)
print("c4 vs c2 AMPLIFICATION (c4 / c2)")
print("=" * 80)

pairs = [("strip", "c2_strip", "c4_strip"),
         ("gemm",  "c2_gemm",  "c4_gemm"),
         ("fused", "c2_fused", "c4_fused")]

key_names = [
    ("dram_read",     "dram__bytes_read.sum"),
    ("dram_write",    "dram__bytes_write.sum"),
    ("lts_sectors",   "lts__t_sectors.sum"),
    ("lts_hit_rate",  "lts__t_sector_hit_rate.pct"),
    ("tma_rd_reqs",   "l1tex__t_requests_pipe_tma_opc_read.sum"),
    ("tma_wr_reqs",   "l1tex__t_requests_pipe_tma_opc_write.sum"),
    ("cycles_active", "smsp__cycles_active.avg"),
    ("inst_executed", "smsp__inst_executed.sum"),
    ("wait_stall",    "smsp__warps_issue_stalled_wait.avg"),
    ("barrier_stall", "smsp__warps_issue_stalled_barrier.avg"),
    ("membar_stall",  "smsp__warps_issue_stalled_membar.avg"),
    ("long_sb_stall", "smsp__warps_issue_stalled_long_scoreboard.avg"),
]

for ptitle, ca, cb in pairs:
    print(f"\n  [{ptitle}]  {ca} -> {cb}")
    for short, full in key_names:
        va = data.get(ca, {}).get(full, ("", ""))
        vb = data.get(cb, {}).get(full, ("", ""))
        bva = to_base(va[0], va[1]) if isinstance(va, tuple) else None
        bvb = to_base(vb[0], vb[1]) if isinstance(vb, tuple) else None
        if bva and bvb and bva != 0:
            ratio = bvb / bva
            mark = " ⚠" if abs(ratio - 1.0) > 0.10 else ""
            print(f"    {short:<18} c2={fmt_val(va[0],va[1]):<10} c4={fmt_val(vb[0],vb[1]):<10}  ratio={ratio:.2f}x{mark}")

PYEOF
fi

# ══════════════════════════════════════════════════════════════════
#  Done
# ══════════════════════════════════════════════════════════════════

log ""
log "════════════════════════════════════════"
log "  DONE"
log "════════════════════════════════════════"
log ""
log "Key outputs:"
log "  $OUTDIR/results.txt          — wall times"
log "  $OUTDIR/summary.txt          — all metrics side-by-side + c4/c2 ratios"
log "  $OUTDIR/diff_q1_strip.txt    — strip: pure GEMM dispatch cost"
log "  $OUTDIR/diff_q2_gemm.txt     — gemm:  + TMA store"
log "  $OUTDIR/diff_q3_fused.txt    — fused: + residual+bias"
log "  $OUTDIR/diff_q4_lean_vs_c4.txt — best c2 vs c4"
log ""
