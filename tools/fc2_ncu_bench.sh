#!/bin/bash
# FC2 ncu diagnosis — targeted metric collection for comparison pairs.
#
# Profiles: w3 (static), atomic (cluster barrier), spin (flag spin),
#           grid (non-persistent), CUTLASS, hybrid, Phase 4
# Each in fused + strip variants.
#
# Q1: Epilogue overhead (fused vs strip) for each dispatch mode
# Q2: Architecture gap (w3 vs CUTLASS) for both fused and strip
# Q3: Phase 4 broken (2.77ms vs 1.22ms)
# Q4: L2 locality (w3 vs atomic, atomic vs CUTLASS)
# Dispatch: spin vs atomic, grid vs CUTLASS (dispatch overhead isolation)
#
# Usage:
#   ./tools/fc2_ncu_bench.sh                # full run
#   ./tools/fc2_ncu_bench.sh --dry-run      # print commands
#   ./tools/fc2_ncu_bench.sh --quick        # skip Phase 4 (Q1+Q2 only)
#   ./tools/fc2_ncu_bench.sh --full         # also collect --set full profiles
#   ./tools/fc2_ncu_bench.sh --only-phase4  # Phase 4 vs Phase 1 only (Q3)
#
# Output: data/ncu_YYYYMMDD_HHMMSS/
#
# After running, key outputs:
#   results.txt            — wall time sanity checks
#   diff_q1.txt            — Q1: w3 strip vs fused (our epilogue overhead)
#   diff_q1_cutlass.txt    — Q1: CUTLASS strip vs fused (CUTLASS epilogue overhead)
#   diff_q2.txt            — Q2: w3 fused vs CUTLASS fused (architecture gap)
#   diff_q2_strip.txt      — Q2: w3 strip vs CUTLASS strip (mainloop gap)
#   diff_q3.txt            — Q3: hybrid fused vs phase4 fused (Phase 4 broken)
#   diff_q3_strip.txt      — Q3: hybrid strip vs phase4 strip
#   summary.txt            — all stall metrics side-by-side

set -uo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=0
QUICK=0
FULL=0
ONLY_PHASE4=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)      DRY_RUN=1; shift ;;
        --quick)        QUICK=1; shift ;;
        --full)         FULL=1; shift ;;
        --only-phase4)  ONLY_PHASE4=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="data/ncu_${TIMESTAMP}"
mkdir -p "$OUTDIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/session.log"; }

log "========================================"
log "  FC2 NCU DIAGNOSIS  $TIMESTAMP"
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
smsp__warps_issue_stalled_math_pipe_throttle.avg"

METRICS_MEM="\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sectors_op_read.sum,\
lts__t_sectors_op_write.sum,\
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
launch__occupancy"

METRICS_SMEM="\
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_global_op_ld.sum"

METRICS_ALL="${METRICS_STALL},${METRICS_MEM},${METRICS_PIPE},${METRICS_SCHED},${METRICS_SMEM}"

# ── Binary definitions: label|build_cmd|binary|kernel_filter ──
BINARIES=()

if [ "$ONLY_PHASE4" = "0" ]; then
    # Strip builds FIRST (they overwrite the base binary), then fused rebuilds restore it
    BINARIES+=(
        "w3_strip|make -B fc2-w3 DFLAGS=-DSTRIP_EPILOGUE|COPY:fc2-w3:fc2-w3-strip|fc2_w3_kernel"
        "w3_fused|make -B fc2-w3|./fc2-w3|fc2_w3_kernel"
        "atomic_strip|make -B fc2-w3-atomic DFLAGS=-DSTRIP_EPILOGUE|COPY:fc2-w3-atomic:fc2-w3-atomic-strip|fc2_w3_kernel"
        "atomic_fused|make -B fc2-w3-atomic|./fc2-w3-atomic|fc2_w3_kernel"
        "spin_strip|make -B fc2-w3-spin DFLAGS=-DSTRIP_EPILOGUE|COPY:fc2-w3-spin:fc2-w3-spin-strip|fc2_w3_kernel"
        "spin_fused|make -B fc2-w3-spin|./fc2-w3-spin|fc2_w3_kernel"
        "grid_strip|make -B fc2-w3-grid DFLAGS=-DSTRIP_EPILOGUE|COPY:fc2-w3-grid:fc2-w3-grid-strip|fc2_w3_kernel"
        "grid_fused|make -B fc2-w3-grid|./fc2-w3-grid|fc2_w3_kernel"
        "cutlass_fused|make fc2-cutlass|./fc2-cutlass|regex:^(?!init)"
        "cutlass_strip|make fc2-cutlass-strip|./fc2-cutlass-strip|regex:^(?!init)"
        "hybrid_fused|make fc2-hybrid|./fc2-hybrid|regex:fc2_hybrid_kernel"
        "hybrid_strip|make fc2-hybrid-strip|./fc2-hybrid-strip|regex:fc2_hybrid_kernel"
    )
fi

if [ "$QUICK" = "0" ]; then
    BINARIES+=(
        "phase4_strip|make -B fc2-hybrid-phase4 DFLAGS=-DSTRIP_EPILOGUE|COPY:fc2-hybrid-phase4:fc2-hybrid-phase4-strip|regex:fc2_phase4"
        "phase4_fused|make -B fc2-hybrid-phase4|./fc2-hybrid-phase4|regex:fc2_phase4"
    )
fi

# Always include hybrid as Phase 4 reference point
if [ "$ONLY_PHASE4" = "1" ]; then
    BINARIES+=(
        "hybrid_fused|make fc2-hybrid|./fc2-hybrid|regex:fc2_hybrid_kernel"
        "hybrid_strip|make fc2-hybrid-strip|./fc2-hybrid-strip|regex:fc2_hybrid_kernel"
    )
fi

# ══════════════════════════════════════════════════════════════════
#  PHASE 1: Build all binaries
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

    # Handle COPY: prefix (for strip builds that overwrite the binary)
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
#  PHASE 2: Wall time sanity check
# ══════════════════════════════════════════════════════════════════

log ""
log "── Phase 2: Wall time sanity checks ──"

for entry in "${BINARIES[@]}"; do
    IFS='|' read -r label build_cmd binary kfilter <<< "$entry"

    # Resolve binary path
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
#  PHASE 3: Targeted ncu metrics
# ══════════════════════════════════════════════════════════════════

log ""
log "── Phase 3: Targeted ncu metric collection ──"

for entry in "${BINARIES[@]}"; do
    IFS='|' read -r label build_cmd binary kfilter <<< "$entry"

    # Resolve binary path
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
#  PHASE 4: Diffs (the actual diagnosis)
# ══════════════════════════════════════════════════════════════════

log ""
log "── Phase 4: Metric diffs ──"

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
    if [ "$ONLY_PHASE4" = "0" ]; then
        run_diff "diff_q1"            "$OUTDIR/w3_strip.csv"      "$OUTDIR/w3_fused.csv"
        run_diff "diff_q1_cutlass"    "$OUTDIR/cutlass_strip.csv" "$OUTDIR/cutlass_fused.csv"
        run_diff "diff_q1_atomic"     "$OUTDIR/atomic_strip.csv"  "$OUTDIR/atomic_fused.csv"
        run_diff "diff_q1_spin"       "$OUTDIR/spin_strip.csv"    "$OUTDIR/spin_fused.csv"
        run_diff "diff_q1_grid"       "$OUTDIR/grid_strip.csv"    "$OUTDIR/grid_fused.csv"
        run_diff "diff_q2"            "$OUTDIR/w3_fused.csv"      "$OUTDIR/cutlass_fused.csv"
        run_diff "diff_q2_strip"      "$OUTDIR/w3_strip.csv"      "$OUTDIR/cutlass_strip.csv"
        run_diff "diff_q2_hybrid"     "$OUTDIR/w3_fused.csv"      "$OUTDIR/hybrid_fused.csv"
        run_diff "diff_q4_fused"      "$OUTDIR/w3_fused.csv"      "$OUTDIR/atomic_fused.csv"
        run_diff "diff_q4_strip"      "$OUTDIR/w3_strip.csv"      "$OUTDIR/atomic_strip.csv"
        run_diff "diff_q4_vs_cutlass" "$OUTDIR/atomic_fused.csv"  "$OUTDIR/cutlass_fused.csv"
        # Cross-dispatch comparison: all dispatch modes strip + fused
        run_diff "diff_disp_spin_vs_atomic_fused"  "$OUTDIR/atomic_fused.csv"  "$OUTDIR/spin_fused.csv"
        run_diff "diff_disp_spin_vs_atomic_strip"  "$OUTDIR/atomic_strip.csv"  "$OUTDIR/spin_strip.csv"
        run_diff "diff_disp_grid_vs_cutlass_fused" "$OUTDIR/grid_fused.csv"    "$OUTDIR/cutlass_fused.csv"
        run_diff "diff_disp_grid_vs_cutlass_strip" "$OUTDIR/grid_strip.csv"    "$OUTDIR/cutlass_strip.csv"
    fi
    if [ "$QUICK" = "0" ]; then
        run_diff "diff_q3"       "$OUTDIR/hybrid_fused.csv" "$OUTDIR/phase4_fused.csv"
        run_diff "diff_q3_strip" "$OUTDIR/hybrid_strip.csv" "$OUTDIR/phase4_strip.csv"
    fi
fi

# ══════════════════════════════════════════════════════════════════
#  PHASE 5: Side-by-side summary table
# ══════════════════════════════════════════════════════════════════

log ""
log "── Phase 5: Summary table ──"

if [ "$DRY_RUN" = "0" ]; then
    # Extract key stall metrics from each CSV into a summary
    python3 - "$OUTDIR" "${BINARIES[@]}" << 'PYEOF' 2>&1 | tee "$OUTDIR/summary.txt" | tee -a "$OUTDIR/session.log"
import sys, csv, os

outdir = sys.argv[1]
entries = sys.argv[2:]

# Key metrics to extract (short names for display)
KEY_METRICS = [
    # ── Stall reasons ──
    ("long_scoreboard",   "smsp__warps_issue_stalled_long_scoreboard.avg"),
    ("short_scoreboard",  "smsp__warps_issue_stalled_short_scoreboard.avg"),
    ("wait",              "smsp__warps_issue_stalled_wait.avg"),
    ("barrier",           "smsp__warps_issue_stalled_barrier.avg"),
    ("sleeping",          "smsp__warps_issue_stalled_sleeping.avg"),
    ("not_selected",      "smsp__warps_issue_stalled_not_selected.avg"),
    ("mio_throttle",      "smsp__warps_issue_stalled_mio_throttle.avg"),
    ("math_throttle",     "smsp__warps_issue_stalled_math_pipe_throttle.avg"),
    # ── Memory throughput ──
    ("dram_read_bytes",   "dram__bytes_read.sum"),
    ("dram_write_bytes",  "dram__bytes_write.sum"),
    ("dram_pct",          "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("lts_pct",           "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("l1tex_pct",         "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"),
    # ── Pipe utilization ──
    ("sm_pct",            "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("shared_pct",        "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("warps_active",      "sm__warps_active.avg.per_cycle_active"),
    ("cycles_active",     "smsp__cycles_active.avg"),
    ("inst_executed",     "smsp__inst_executed.sum"),
    # ── Scheduler + registers ──
    ("warps_eligible",    "smsp__warps_eligible.avg.per_cycle_active"),
    ("regs_per_thread",   "launch__registers_per_thread"),
    ("occupancy",         "launch__occupancy"),
    # ── SMEM + global wavefronts ──
    ("smem_wavefronts",   "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"),
    ("smem_ld_wf",        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"),
    ("smem_st_wf",        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"),
    ("global_ld_wf",      "l1tex__data_pipe_lsu_wavefronts_mem_global_op_ld.sum"),
]

labels = []
data = {}  # label -> {metric_name: value}

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
            data[label][name] = val

if not labels:
    print("No CSV data found.")
    sys.exit(0)

# Print table
col_w = 16
hdr = f"{'metric':<22}"
for l in labels:
    hdr += f"  {l:>{col_w}}"
print(hdr)
print("-" * len(hdr))

for short, full in KEY_METRICS:
    row = f"{short:<22}"
    for l in labels:
        v = data[l].get(full, "—")
        # Truncate long numbers
        try:
            fv = float(v.replace(",", ""))
            if fv > 1e9:
                v = f"{fv/1e9:.2f}G"
            elif fv > 1e6:
                v = f"{fv/1e6:.2f}M"
            elif fv > 1e3:
                v = f"{fv/1e3:.1f}K"
            else:
                v = f"{fv:.2f}"
        except (ValueError, AttributeError):
            pass
        row += f"  {v:>{col_w}}"
    print(row)

PYEOF
fi

# ══════════════════════════════════════════════════════════════════
#  PHASE 6: Full profiles (optional, --full flag)
# ══════════════════════════════════════════════════════════════════

if [ "$FULL" = "1" ]; then
    log ""
    log "── Phase 6: Full ncu profiles (--set full) ──"

    FULL_TARGETS=()
    if [ "$ONLY_PHASE4" = "0" ]; then
        FULL_TARGETS+=("w3_fused|./fc2-w3|fc2_w3_kernel")
        FULL_TARGETS+=("cutlass_fused|./fc2-cutlass|regex:^(?!init)")
    fi
    FULL_TARGETS+=("hybrid_fused|./fc2-hybrid|regex:fc2_hybrid_kernel")
    if [ "$QUICK" = "0" ]; then
        FULL_TARGETS+=("phase4_fused|./fc2-hybrid-phase4|regex:fc2_phase4")
    fi

    for entry in "${FULL_TARGETS[@]}"; do
        IFS='|' read -r label binary kfilter <<< "$entry"

        if [ "$DRY_RUN" = "1" ]; then
            echo "  ncu --set full --kernel-name $kfilter -o $OUTDIR/full_${label} $binary"
            continue
        fi

        if [ ! -x "$binary" ]; then
            log "  [full/$label] SKIP (no binary)"
            continue
        fi

        log "  [full/$label] collecting --set full..."
        if ncu --set full \
               --kernel-name "$kfilter" \
               -o "$OUTDIR/full_${label}" \
               "$binary" \
               > "$OUTDIR/full_${label}_stdout.txt" 2>"$OUTDIR/full_${label}_stderr.txt"; then

            # Export source counters
            ncu --import "$OUTDIR/full_${label}.ncu-rep" --csv --page source \
                > "$OUTDIR/full_${label}_source.csv" 2>/dev/null
            # Export all metrics
            ncu --import "$OUTDIR/full_${label}.ncu-rep" --csv \
                > "$OUTDIR/full_${label}.csv" 2>/dev/null

            log "  [full/$label] done"
        else
            log "  [full/$label] FAILED"
            cat "$OUTDIR/full_${label}_stderr.txt" >> "$OUTDIR/session.log"
        fi
    done

    # Source counter analysis
    if [ "$DRY_RUN" = "0" ]; then
        log ""
        log "── Source counter analysis ──"
        for f in "$OUTDIR"/full_*_source.csv; do
            [ -f "$f" ] || continue
            label=$(basename "$f" _source.csv)
            log "  $label:"
            python3 tools/analyze_source_counters.py "$f" 2>&1 | head -40 | tee -a "$OUTDIR/session.log"
        done
    fi
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
log "  $OUTDIR/results.txt       — wall times"
log "  $OUTDIR/summary.txt       — stall metrics side-by-side"
if [ "$ONLY_PHASE4" = "0" ]; then
    log "  $OUTDIR/diff_q1*.txt      — Q1: epilogue overhead per dispatch mode"
    log "  $OUTDIR/diff_q2.txt       — Q2: w3 vs CUTLASS"
    log "  $OUTDIR/diff_q4*.txt      — Q4: L2 locality (w3 vs atomic vs CUTLASS)"
    log "  $OUTDIR/diff_disp*.txt    — Dispatch overhead (spin vs atomic, grid vs CUTLASS)"
fi
if [ "$QUICK" = "0" ]; then
    log "  $OUTDIR/diff_q3.txt       — Q3: Phase 1 vs Phase 4"
    log "  $OUTDIR/diff_q3_strip.txt — Q3: strip comparison"
fi
if [ "$FULL" = "1" ]; then
    log "  $OUTDIR/full_*.csv        — full profiles + source counters"
fi
log ""
log "To interpret: read summary.txt first, then diff_q*.txt for specifics."
log "See docs/ncu_diagnosis_guide.txt for what each metric means."
