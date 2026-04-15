#!/bin/bash
# FC2 ncu diagnosis v4 — dispatch comparison + DRAM amplification.
#
# Profiles: w3 (strided), w3-lean (LEAN_DISPATCH), w3-dgswizzle (DeepGEMM 2D),
#           CUTLASS fused/strip. Each in fused + gemm/strip variants.
#
# Q1: Fusion cost — w3 fused vs w3 gemm, cutlass fused vs cutlass strip
# Q2: Head-to-head fused — w3_lean vs cutlass_fused
# Q3: GEMM comparison — w3 gemm vs cutlass strip
# Q4: DRAM amplification — dispatch method vs theoretical minimum
# Q5: Dispatch comparison — lean vs striding vs dgswizzle
# Q6: MMA-only baseline — w3 strip (no output, valid=0)
#
# Usage:
#   ./tools/fc2_ncu_bench.sh                # full run (7 variants)
#   ./tools/fc2_ncu_bench.sh --dry-run      # print commands
#   ./tools/fc2_ncu_bench.sh --full         # also collect --set full profiles
#
# Output: data/ncu_YYYYMMDD_HHMMSS/

set -uo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=0
FULL=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift ;;
        --full)     FULL=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="data/ncu_${TIMESTAMP}"
mkdir -p "$OUTDIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/session.log"; }

log "========================================"
log "  FC2 NCU DIAGNOSIS v4  $TIMESTAMP"
log "  Output: $OUTDIR"
log "========================================"

if [ "$DRY_RUN" = "0" ]; then
    nvidia-smi > /dev/null 2>&1 || { log "FATAL: no GPU"; exit 1; }
    nvidia-smi --query-gpu=gpu_name,clocks.sm --format=csv,noheader | tee -a "$OUTDIR/session.log"
fi

# ── Metrics ──

# Stall reasons
METRICS_STALL="\
smsp__warps_issue_stalled_long_scoreboard.avg,\
smsp__warps_issue_stalled_short_scoreboard.avg,\
smsp__warps_issue_stalled_wait.avg,\
smsp__warps_issue_stalled_barrier.avg,\
smsp__warps_issue_stalled_sleeping.avg,\
smsp__warps_issue_stalled_not_selected.avg,\
smsp__warps_issue_stalled_mio_throttle.avg,\
smsp__warps_issue_stalled_math_pipe_throttle.avg"

# Memory throughput + DRAM amplification
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

# Pipe utilization
METRICS_PIPE="\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed,\
sm__mio_pq_read_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__mio_pq_write_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.per_cycle_active,\
smsp__cycles_active.avg,\
smsp__inst_executed.sum"

# Scheduler + occupancy
METRICS_SCHED="\
smsp__warps_eligible.avg.per_cycle_active,\
launch__registers_per_thread,\
launch__occupancy"

# SMEM + global wavefronts
METRICS_SMEM="\
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_global_op_ld.sum"

# TMA activity
METRICS_TMA="\
l1tex__t_requests_pipe_tma_opc_read.sum,\
l1tex__t_requests_pipe_tma_opc_write.sum,\
l1tex__t_bytes_pipe_tma_opc_read.sum,\
l1tex__t_bytes_pipe_tma_opc_write.sum"

METRICS_ALL="${METRICS_STALL},${METRICS_MEM},${METRICS_PIPE},${METRICS_SCHED},${METRICS_SMEM},${METRICS_TMA}"

# ── Binary definitions: label|build_cmd|binary|kernel_filter ──
BINARIES=(
    "w3_strip|make -B fc2-w3 DFLAGS=-DSTRIP_EPILOGUE|COPY:fc2-w3:fc2-w3-strip|fc2_w3_kernel"
    "w3_gemm|make -B fc2-w3-gemm|./fc2-w3-gemm|fc2_w3_kernel"
    "w3_fused|make -B fc2-w3|./fc2-w3|fc2_w3_kernel"
    "w3_lean|make -B fc2-w3-lean|./fc2-w3-lean|fc2_w3_kernel"
    "w3_dgswizzle|make -B fc2-w3-dgswizzle DFLAGS=-DNO_PREFILL|./fc2-w3-dgswizzle|fc2_w3_kernel"
    "cutlass_strip|make fc2-cutlass-strip|./fc2-cutlass-strip|regex:^(?!init)"
    "cutlass_fused|make fc2-cutlass|./fc2-cutlass|regex:^(?!init)"
)


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
#  PHASE 4: Diffs
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
    # Q1: Fusion cost (fused - gemm/strip)
    run_diff "diff_q1_w3"         "$OUTDIR/w3_gemm.csv"       "$OUTDIR/w3_fused.csv"
    run_diff "diff_q1_cutlass"    "$OUTDIR/cutlass_strip.csv" "$OUTDIR/cutlass_fused.csv"

    # Q2: Head-to-head fused (best variant vs CUTLASS)
    run_diff "diff_q2_lean_vs_cutlass" "$OUTDIR/w3_lean.csv"  "$OUTDIR/cutlass_fused.csv"

    # Q3: GEMM comparison (apples-to-apples: both write output, no residual/bias)
    run_diff "diff_q3_gemm"       "$OUTDIR/w3_gemm.csv"       "$OUTDIR/cutlass_strip.csv"

    # Q5: Dispatch comparison — lean vs striding vs dgswizzle
    run_diff "diff_q5_lean_vs_fused"      "$OUTDIR/w3_lean.csv"      "$OUTDIR/w3_fused.csv"
    run_diff "diff_q5_dgswizzle_vs_fused" "$OUTDIR/w3_dgswizzle.csv" "$OUTDIR/w3_fused.csv"
    run_diff "diff_q5_dgswizzle_vs_lean"  "$OUTDIR/w3_dgswizzle.csv" "$OUTDIR/w3_lean.csv"

    # Q6: MMA-only baseline (w3 strip vs w3 gemm = output write cost)
    run_diff "diff_q6_output_cost" "$OUTDIR/w3_strip.csv"     "$OUTDIR/w3_gemm.csv"
fi

# ══════════════════════════════════════════════════════════════════
#  PHASE 5: Summary table
# ══════════════════════════════════════════════════════════════════

log ""
log "── Phase 5: Summary table ──"

if [ "$DRY_RUN" = "0" ]; then
    python3 - "$OUTDIR" "${BINARIES[@]}" << 'PYEOF' 2>&1 | tee "$OUTDIR/summary.txt" | tee -a "$OUTDIR/session.log"
import sys, csv, os

outdir = sys.argv[1]
entries = sys.argv[2:]

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
    # ── DRAM (amplification check) ──
    ("dram_read_bytes",   "dram__bytes_read.sum"),
    ("dram_write_bytes",  "dram__bytes_write.sum"),
    ("dram_read_sectors", "dram__sectors_read.sum"),
    ("dram_write_sectors","dram__sectors_write.sum"),
    ("dram_pct",          "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
    # ── L2 cache ──
    ("lts_pct",           "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("lts_sectors",       "lts__t_sectors.sum"),
    ("lts_read_sectors",  "lts__t_sectors_op_read.sum"),
    ("lts_write_sectors", "lts__t_sectors_op_write.sum"),
    ("lts_hit_rate",      "lts__t_sector_hit_rate.pct"),
    ("l1tex_pct",         "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"),
    # ── Pipe utilization ──
    ("sm_pct",            "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("shared_pct",        "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("warps_active",      "sm__warps_active.avg.per_cycle_active"),
    ("cycles_active",     "smsp__cycles_active.avg"),
    ("inst_executed",     "smsp__inst_executed.sum"),
    # ── Scheduler ──
    ("warps_eligible",    "smsp__warps_eligible.avg.per_cycle_active"),
    ("regs_per_thread",   "launch__registers_per_thread"),
    ("occupancy",         "launch__occupancy"),
    # ── SMEM wavefronts ──
    ("smem_wavefronts",   "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"),
    ("smem_ld_wf",        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"),
    ("smem_st_wf",        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"),
    ("global_ld_wf",      "l1tex__data_pipe_lsu_wavefronts_mem_global_op_ld.sum"),
    # ── TMA ──
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

col_w = 16
hdr = f"{'metric':<22}"
for l in labels:
    hdr += f"  {l:>{col_w}}"
print(hdr)
print("-" * len(hdr))

UNIT_SCALE = {"Gbyte": 1e9, "Mbyte": 1e6, "Kbyte": 1e3, "byte": 1,
              "Gsector": 1e9, "Msector": 1e6, "Ksector": 1e3, "sector": 1}

def to_base(val_str, unit_str):
    """Convert ncu value+unit to base units (bytes, sectors, etc)."""
    try:
        v = float(val_str.replace(",", ""))
    except (ValueError, AttributeError):
        return None
    scale = UNIT_SCALE.get(unit_str, 1.0)
    return v * scale

def fmt_val(val_str, unit_str):
    """Convert ncu value to base units, then auto-format with G/M/K suffix."""
    bv = to_base(val_str, unit_str)
    if bv is None:
        return val_str  # "—" or "n/a"
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

# ── DRAM amplification analysis ──
print()
print("=" * 80)
print("DRAM AMPLIFICATION ANALYSIS")
print("=" * 80)

# Theoretical minimum bytes for FC2
# Read: A[M,K] + B[K,N] + residual[M,N] + bias[N] (all in appropriate types)
M, K, N = 928256, 3072, 768
theoretical_read = M * K * 1 + K * N * 1 + M * N * 2 + N * 2  # FP8 + BF16
theoretical_write = M * N * 2  # BF16 output
print(f"Theoretical minimum DRAM (FC2 {M}x{K}x{N}):")
print(f"  Read:  {theoretical_read/1e9:.3f} GB (A:FP8 + B:FP8 + residual:BF16 + bias:BF16)")
print(f"  Write: {theoretical_write/1e9:.3f} GB (output:BF16)")
print()

for l in labels:
    dr_entry = data[l].get("dram__bytes_read.sum", ("", ""))
    dw_entry = data[l].get("dram__bytes_write.sum", ("", ""))
    dr_val = to_base(dr_entry[0], dr_entry[1]) if isinstance(dr_entry, tuple) else None
    dw_val = to_base(dw_entry[0], dw_entry[1]) if isinstance(dw_entry, tuple) else None
    if dr_val is not None and dw_val is not None:
        read_amp = dr_val / theoretical_read if theoretical_read > 0 else 0
        write_amp = dw_val / theoretical_write if theoretical_write > 0 else 0
        print(f"  {l:20s}: read={dr_val/1e9:.3f}GB ({read_amp:.2f}x)  write={dw_val/1e9:.3f}GB ({write_amp:.2f}x)")
    else:
        print(f"  {l:20s}: read=n/a  write=n/a")

PYEOF
fi

# ══════════════════════════════════════════════════════════════════
#  PHASE 6: Full profiles (optional, --full flag)
# ══════════════════════════════════════════════════════════════════

if [ "$FULL" = "1" ]; then
    log ""
    log "── Phase 6: Full ncu profiles (--set full) ──"

    FULL_TARGETS=(
        "w3_lean|./fc2-w3-lean|fc2_w3_kernel"
        "cutlass_fused|./fc2-cutlass|regex:^(?!init)"
    )

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

            ncu --import "$OUTDIR/full_${label}.ncu-rep" --csv --page source \
                > "$OUTDIR/full_${label}_source.csv" 2>/dev/null
            ncu --import "$OUTDIR/full_${label}.ncu-rep" --csv \
                > "$OUTDIR/full_${label}.csv" 2>/dev/null

            log "  [full/$label] done"
        else
            log "  [full/$label] FAILED"
            cat "$OUTDIR/full_${label}_stderr.txt" >> "$OUTDIR/session.log"
        fi
    done
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
log "  $OUTDIR/summary.txt       — all metrics side-by-side + DRAM amplification"
log "  $OUTDIR/diff_q1*.txt      — Q1: fusion cost (fused - gemm/strip)"
log "  $OUTDIR/diff_q2*.txt      — Q2: lean vs CUTLASS fused"
log "  $OUTDIR/diff_q3*.txt      — Q3: GEMM comparison (w3_gemm vs cutlass_strip)"
log "  $OUTDIR/diff_q5*.txt      — Q5: dispatch comparison (lean vs striding vs dgswizzle)"
log "  $OUTDIR/diff_q6*.txt      — Q6: output write cost (w3_strip vs w3_gemm)"
if [ "$FULL" = "1" ]; then
    log "  $OUTDIR/full_*.csv        — full profiles + source counters"
fi
log ""
log "Analysis: python3 tools/ncu_anova.py $OUTDIR"
