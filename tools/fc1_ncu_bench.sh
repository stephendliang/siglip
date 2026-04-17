#!/bin/bash
# FC1 ncu diagnosis — tile-dispatch sweep and amplification analysis.
#
# Profiles FC1 under every TD variant (default striding, TD=4 sched/LEAN,
# TD=8..16 static swizzles) in fused/gemm/strip decomposition.  Designed
# for batch B200 runs: timeouts guard against deadlock.
#
# FC1 shape: [928256, 768] × [768, 3072]^T + bias + GELU
# 43512 tiles (8× FC2), K_ITERS=6 (4× shorter than FC2)
# NO_PREFILL auto-forced for K_ITERS<20 — static dispatch safe by default.
#
# Usage:
#   ./tools/fc1_ncu_bench.sh                # fused dispatch sweep (default)
#   ./tools/fc1_ncu_bench.sh --decomp       # also gemm+strip for every TD
#   ./tools/fc1_ncu_bench.sh --bias-sweep   # legacy bias-strategy sweep
#   ./tools/fc1_ncu_bench.sh --quick        # only fused/lean/sched
#   ./tools/fc1_ncu_bench.sh --dry-run      # print commands
#   ./tools/fc1_ncu_bench.sh --full         # also collect --set full profiles
#
# Output: data/fc1_ncu_YYYYMMDD_HHMMSS/

set -uo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=0
QUICK=0
FULL=0
DECOMP=0
BIAS_SWEEP=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)     DRY_RUN=1;    shift ;;
        --quick)       QUICK=1;      shift ;;
        --full)        FULL=1;       shift ;;
        --decomp)      DECOMP=1;     shift ;;
        --bias-sweep)  BIAS_SWEEP=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="data/fc1_ncu_${TIMESTAMP}"
mkdir -p "$OUTDIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/session.log"; }

log "========================================"
log "  FC1 NCU DIAGNOSIS  $TIMESTAMP"
log "  Output: $OUTDIR"
log "  mode: quick=$QUICK decomp=$DECOMP bias=$BIAS_SWEEP full=$FULL"
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
launch__occupancy"

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

# ── Binary definitions: label|build_cmd|binary|kernel_filter ──
# Every target copies its output to a label-unique binary so back-to-back
# variants don't clobber each other.

BINARIES=()

# Dispatch sweep (fused). Every static TD gets NO_PREFILL explicitly for parity.
add_fused() {
    local label="$1" target="$2" extra="$3"
    local bin
    if [ -z "$target" ]; then bin="fc1-w3"; else bin="fc1-w3-${target}"; fi
    local out="fc1-w3-${label}"
    BINARIES+=("${label}|make -B ${bin} DFLAGS='${extra}'|COPY:${bin}:${out}|fc1_w3_kernel")
}

add_fused fused     ""           ""
add_fused sched     "sched"      ""
add_fused lean      "lean"       ""
if [ "$QUICK" = "0" ]; then
    add_fused dgswizzle "dgswizzle" "-DNO_PREFILL"
    add_fused zorder    "zorder"    "-DNO_PREFILL"
    add_fused hilbert   "hilbert"   "-DNO_PREFILL"
    add_fused zigzag    "zigzag"    "-DNO_PREFILL"
    add_fused rowmajor  "rowmajor"  "-DNO_PREFILL"
    add_fused ncycle    "ncycle"    "-DNO_PREFILL"
    add_fused nflat     "nflat"     "-DNO_PREFILL"
    add_fused nsnake    "nsnake"    "-DNO_PREFILL"
fi

# gemm+strip decomposition for every dispatch variant.
if [ "$DECOMP" = "1" ]; then
    add_decomp() {
        local label="$1" target="$2" extra_base="$3"
        local bin
        if [ -z "$target" ]; then bin="fc1-w3"; else bin="fc1-w3-${target}"; fi
        for mode in gemm strip; do
            local flag
            case "$mode" in
                gemm)  flag="-DGEMM_ONLY" ;;
                strip) flag="-DSTRIP_EPILOGUE" ;;
            esac
            local extra
            if [ -n "$extra_base" ]; then extra="$extra_base $flag"; else extra="$flag"; fi
            local out="fc1-w3-${label}-${mode}"
            BINARIES+=("${label}_${mode}|make -B ${bin} DFLAGS='${extra}'|COPY:${bin}:${out}|fc1_w3_kernel")
        done
    }
    add_decomp fused     ""           ""
    add_decomp sched     "sched"      ""
    add_decomp lean      "lean"       ""
    add_decomp dgswizzle "dgswizzle"  "-DNO_PREFILL"
    add_decomp zorder    "zorder"     "-DNO_PREFILL"
    add_decomp hilbert   "hilbert"    "-DNO_PREFILL"
    add_decomp zigzag    "zigzag"     "-DNO_PREFILL"
    add_decomp rowmajor  "rowmajor"   "-DNO_PREFILL"
    add_decomp ncycle    "ncycle"     "-DNO_PREFILL"
    add_decomp nflat     "nflat"      "-DNO_PREFILL"
    add_decomp nsnake    "nsnake"     "-DNO_PREFILL"
fi

# Bias-strategy sweep (legacy Q6 — only when requested).  Uses ns6+TD=4 base.
if [ "$BIAS_SWEEP" = "1" ]; then
    BINARIES+=(
        "bias_ns6_sched|make -B fc1-w3 DFLAGS='-DN_STAGES=6 -DTILE_DISPATCH=4'|COPY:fc1-w3:fc1-w3-bias-ns6-sched|fc1_w3_kernel"
        "bias_ns6_ldg|make -B fc1-w3 DFLAGS='-DN_STAGES=6 -DTILE_DISPATCH=4 -DLDG_BIAS'|COPY:fc1-w3:fc1-w3-bias-ns6-ldg|fc1_w3_kernel"
        "bias_ns6_preload|make -B fc1-w3 DFLAGS='-DN_STAGES=6 -DTILE_DISPATCH=4 -DBIAS_PRELOAD -DBIAS_BATCH=2'|COPY:fc1-w3:fc1-w3-bias-ns6-preload|fc1_w3_kernel"
        "bias_ns6_bb6|make -B fc1-w3 DFLAGS='-DN_STAGES=6 -DTILE_DISPATCH=4 -DBIAS_BATCH=6'|COPY:fc1-w3:fc1-w3-bias-ns6-bb6|fc1_w3_kernel"
        "bias_sched_ldg|make -B fc1-w3 DFLAGS='-DTILE_DISPATCH=4 -DLDG_BIAS'|COPY:fc1-w3:fc1-w3-bias-sched-ldg|fc1_w3_kernel"
    )
fi


# ══════════════════════════════════════════════════════════════════
#  PHASE 1: Build all binaries
# ══════════════════════════════════════════════════════════════════

log ""
log "── Phase 1: Building binaries (${#BINARIES[@]} variants) ──"

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

    # Handle COPY: prefix (builds that overwrite their source binary)
    if [[ "$binary" == COPY:* ]]; then
        src="${binary#COPY:}"
        src_bin="${src%%:*}"
        dst_bin="${src#*:}"
        cp "$src_bin" "$dst_bin"
        log "  [$label] copied $src_bin -> $dst_bin"
    fi

    regs=$(grep -o '[0-9]* registers' "$OUTDIR/${label}_build.log" | tail -1 | grep -o '[0-9]*')
    bars=$(grep -o 'used [0-9]* barriers' "$OUTDIR/${label}_build.log" | tail -1 | grep -o '[0-9]*')
    smem=$(grep -o '[0-9]* bytes of shared memory' "$OUTDIR/${label}_build.log" | tail -1 | grep -o '^[0-9]*')
    log "  [$label] built (regs=$regs, bars=$bars, smem=${smem:-?})"
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
        log "  [$label] NO @@RESULT LINE (possible deadlock/crash)"
        echo "@@RESULT ms=ERR label=${label}" >> "$OUTDIR/results.txt"
    fi
done

# ══════════════════════════════════════════════════════════════════
#  PHASE 3: Targeted ncu metrics (with deadlock timeout guard)
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
    if timeout 180 ncu --metrics "$METRICS_ALL" \
           --kernel-name "$kfilter" \
           -o "$OUTDIR/${label}" \
           "$binary" \
           > "$OUTDIR/${label}_ncu_stdout.txt" 2>"$OUTDIR/${label}_ncu_stderr.txt"; then

        ncu --import "$OUTDIR/${label}.ncu-rep" --csv \
            > "$OUTDIR/${label}.csv" 2>/dev/null

        lines=$(wc -l < "$OUTDIR/${label}.csv")
        log "  [$label] ncu done ($lines CSV lines)"
    else
        rc=$?
        if [ $rc -eq 124 ]; then
            log "  [$label] NCU TIMEOUT (likely deadlock — skipping)"
        else
            log "  [$label] NCU FAILED (rc=$rc)"
        fi
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
    # Each static dispatch vs fused baseline (stride) and vs sched (TD=4).
    for v in sched lean dgswizzle zorder hilbert zigzag rowmajor ncycle nflat nsnake; do
        run_diff "diff_${v}_vs_fused" "$OUTDIR/${v}.csv"  "$OUTDIR/fused.csv"
    done
    run_diff "diff_lean_vs_sched"     "$OUTDIR/lean.csv"     "$OUTDIR/sched.csv"
    run_diff "diff_ncycle_vs_nflat"   "$OUTDIR/ncycle.csv"   "$OUTDIR/nflat.csv"
    run_diff "diff_ncycle_vs_nsnake"  "$OUTDIR/ncycle.csv"   "$OUTDIR/nsnake.csv"

    if [ "$DECOMP" = "1" ]; then
        for v in fused sched lean dgswizzle ncycle nflat nsnake; do
            run_diff "diff_${v}_gemm_vs_strip" "$OUTDIR/${v}_gemm.csv" "$OUTDIR/${v}_strip.csv"
            run_diff "diff_${v}_fused_vs_gemm" "$OUTDIR/${v}.csv"      "$OUTDIR/${v}_gemm.csv"
        done
    fi

    if [ "$BIAS_SWEEP" = "1" ]; then
        for v in bias_ns6_ldg bias_ns6_preload bias_ns6_bb6 bias_sched_ldg; do
            run_diff "diff_${v}_vs_sched" "$OUTDIR/${v}.csv" "$OUTDIR/bias_ns6_sched.csv"
        done
    fi
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
    ("long_scoreboard",   "smsp__warps_issue_stalled_long_scoreboard.avg"),
    ("short_scoreboard",  "smsp__warps_issue_stalled_short_scoreboard.avg"),
    ("wait",              "smsp__warps_issue_stalled_wait.avg"),
    ("barrier",           "smsp__warps_issue_stalled_barrier.avg"),
    ("sleeping",          "smsp__warps_issue_stalled_sleeping.avg"),
    ("not_selected",      "smsp__warps_issue_stalled_not_selected.avg"),
    ("mio_throttle",      "smsp__warps_issue_stalled_mio_throttle.avg"),
    ("math_throttle",     "smsp__warps_issue_stalled_math_pipe_throttle.avg"),
    ("dram_read_bytes",   "dram__bytes_read.sum"),
    ("dram_write_bytes",  "dram__bytes_write.sum"),
    ("dram_read_sectors", "dram__sectors_read.sum"),
    ("dram_write_sectors","dram__sectors_write.sum"),
    ("dram_pct",          "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("lts_pct",           "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("lts_sectors",       "lts__t_sectors.sum"),
    ("lts_read_sectors",  "lts__t_sectors_op_read.sum"),
    ("lts_write_sectors", "lts__t_sectors_op_write.sum"),
    ("lts_hit_rate",      "lts__t_sector_hit_rate.pct"),
    ("l1tex_pct",         "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("sm_pct",            "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("shared_pct",        "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("warps_active",      "sm__warps_active.avg.per_cycle_active"),
    ("cycles_active",     "smsp__cycles_active.avg"),
    ("inst_executed",     "smsp__inst_executed.sum"),
    ("warps_eligible",    "smsp__warps_eligible.avg.per_cycle_active"),
    ("regs_per_thread",   "launch__registers_per_thread"),
    ("occupancy",         "launch__occupancy"),
    ("smem_wavefronts",   "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"),
    ("smem_ld_wf",        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"),
    ("smem_st_wf",        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"),
    ("global_ld_wf",      "l1tex__data_pipe_lsu_wavefronts_mem_global_op_ld.sum"),
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
print("DRAM AMPLIFICATION ANALYSIS")
print("=" * 80)

M, K, N = 928256, 768, 3072
theoretical_read_fused = M * K * 1 + K * N * 1 + N * 2
theoretical_read_strip = M * K * 1 + K * N * 1
theoretical_write = M * N * 2

print(f"Theoretical minimum DRAM (FC1 {M}x{K}x{N}):")
print(f"  Read (fused/gemm): {theoretical_read_fused/1e9:.3f} GB (A:FP8 + B:FP8 + bias)")
print(f"  Read (strip):      {theoretical_read_strip/1e9:.3f} GB (A:FP8 + B:FP8)")
print(f"  Write (fused/gemm):{theoretical_write/1e9:.3f} GB (output:BF16)")
print(f"  Total (fused):     {(theoretical_read_fused+theoretical_write)/1e9:.3f} GB")
print(f"  NOTE: B matrix = {K*N/1e6:.1f}MB — fits entirely in L2")
print()

for l in labels:
    dr_entry = data[l].get("dram__bytes_read.sum", ("", ""))
    dw_entry = data[l].get("dram__bytes_write.sum", ("", ""))
    dr_val = to_base(dr_entry[0], dr_entry[1]) if isinstance(dr_entry, tuple) else None
    dw_val = to_base(dw_entry[0], dw_entry[1]) if isinstance(dw_entry, tuple) else None
    if dr_val is not None and dw_val is not None:
        th_read = theoretical_read_strip if "strip" in l else theoretical_read_fused
        th_write = theoretical_write if "strip" not in l else 1
        read_amp = dr_val / th_read if th_read > 0 else 0
        write_amp = dw_val / th_write if th_write > 0 else 0
        print(f"  {l:22s}: read={dr_val/1e9:.3f}GB ({read_amp:.2f}x)  write={dw_val/1e9:.3f}GB ({write_amp:.2f}x)")
    else:
        print(f"  {l:22s}: read=n/a  write=n/a")

PYEOF
fi

# ══════════════════════════════════════════════════════════════════
#  PHASE 6: Full profiles (optional, --full flag)
# ══════════════════════════════════════════════════════════════════

if [ "$FULL" = "1" ]; then
    log ""
    log "── Phase 6: Full ncu profiles (--set full) ──"

    FULL_TARGETS=(
        "fused|./fc1-w3-fused|fc1_w3_kernel"
        "lean|./fc1-w3-lean|fc1_w3_kernel"
    )
    if [ "$QUICK" = "0" ]; then
        FULL_TARGETS+=(
            "ncycle|./fc1-w3-ncycle|fc1_w3_kernel"
            "nflat|./fc1-w3-nflat|fc1_w3_kernel"
            "nsnake|./fc1-w3-nsnake|fc1_w3_kernel"
        )
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
        if timeout 300 ncu --set full \
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
            rc=$?
            if [ $rc -eq 124 ]; then
                log "  [full/$label] NCU TIMEOUT (likely deadlock — skipping)"
            else
                log "  [full/$label] NCU FAILED (rc=$rc)"
            fi
            cat "$OUTDIR/full_${label}_stderr.txt" >> "$OUTDIR/session.log"
        fi
    done
fi

# ══════════════════════════════════════════════════════════════════
#  Done
# ══════════════════════════════════════════════════════════════════

log ""
log "════════════════════════════════════════"
log "  DONE  ${#BINARIES[@]} variants profiled"
log "════════════════════════════════════════"
log ""
log "Key outputs:"
log "  $OUTDIR/results.txt       — wall times"
log "  $OUTDIR/summary.txt       — all metrics side-by-side + DRAM amplification"
log "  $OUTDIR/diff_*.txt        — pairwise diffs"
if [ "$FULL" = "1" ]; then
    log "  $OUTDIR/full_*.csv        — full profiles + source counters"
fi
log ""
log "Analysis: python3 tools/ncu_anova.py $OUTDIR"
