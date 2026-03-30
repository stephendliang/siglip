#!/bin/bash
# FC2 memory contention measurement — wall time + ncu stall/throughput analysis.
# Measures per-operation overhead below saturation (1 warp) and hardware resource
# utilization scaling across warp counts.
#
# Usage:
#   ./tools/fc2_contention_bench.sh              # run everything
#   ./tools/fc2_contention_bench.sh --wall-only   # skip ncu (fast, ~1 min)
#   ./tools/fc2_contention_bench.sh --ncu-only    # skip wall time experiments
#   ./tools/fc2_contention_bench.sh --dry-run     # print commands
#
# Output:  data/contention_YYYYMMDD_HHMMSS/
#   wall_results.txt     — wall time @@RESULT lines
#   wall_summary.txt     — wall time comparison table
#   ncu/                 — per-variant ncu CSV exports
#   ncu_summary.txt      — stall/throughput scaling table
#   session.log          — full log

set -uo pipefail
cd "$(dirname "$0")/.."

RUN_WALL=1
RUN_NCU=1
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --wall-only) RUN_NCU=0; shift ;;
        --ncu-only)  RUN_WALL=0; shift ;;
        --dry-run)   DRY_RUN=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="data/contention_${TIMESTAMP}"
mkdir -p "$OUTDIR/ncu"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/session.log"; }

log "========================================"
log "  FC2 CONTENTION BENCH  $TIMESTAMP"
log "  Output: $OUTDIR"
log "========================================"

if [ "$DRY_RUN" = "0" ]; then
    nvidia-smi > /dev/null 2>&1 || { log "FATAL: no GPU"; exit 1; }
    nvidia-smi --query-gpu=gpu_name,clocks.sm --format=csv,noheader | tee -a "$OUTDIR/session.log"
fi

# ── ncu metric list ──
# Warp stall reasons — both naming conventions (ncu version-dependent)
STALL_METRICS=(
    # Standard stall cycle counts (absolute, per-SM)
    "smsp__warps_issue_stalled_long_scoreboard.avg"
    "smsp__warps_issue_stalled_short_scoreboard.avg"
    "smsp__warps_issue_stalled_mio_throttle.avg"
    "smsp__warps_issue_stalled_math_pipe_throttle.avg"
    "smsp__warps_issue_stalled_not_selected.avg"
    "smsp__warps_issue_stalled_barrier.avg"
    "smsp__warps_issue_stalled_membar.avg"
    "smsp__warps_issue_stalled_wait.avg"
    "smsp__warps_issue_stalled_drain.avg"
    "smsp__warps_issue_stalled_sleeping.avg"
    "smsp__warps_issue_stalled_misc.avg"
    # Percentage variants
    "smsp__warps_issue_stalled_long_scoreboard_per_warp_active.pct"
    "smsp__warps_issue_stalled_short_scoreboard_per_warp_active.pct"
    "smsp__warps_issue_stalled_mio_throttle_per_warp_active.pct"
    "smsp__warps_issue_stalled_math_pipe_throttle_per_warp_active.pct"
    "smsp__warps_issue_stalled_not_selected_per_warp_active.pct"
    "smsp__warps_issue_stalled_barrier_per_warp_active.pct"
    "smsp__warps_issue_stalled_wait_per_warp_active.pct"
    # Active cycle counts
    "smsp__cycles_active.avg"
    "smsp__warps_active.avg.per_cycle_active"
    "sm__warps_active.avg.per_cycle_active"
    "sm__cycles_active.avg"
)

# Memory throughput and utilization
MEM_METRICS=(
    "lts__throughput.avg.pct_of_peak_sustained_elapsed"
    "dram__throughput.avg.pct_of_peak_sustained_elapsed"
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"
    "lts__t_sectors_op_read.sum"
    "lts__t_sectors_op_write.sum"
    "dram__bytes_read.sum"
    "dram__bytes_write.sum"
    "lts__t_sectors.sum"
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"
)

# SM pipe utilization
PIPE_METRICS=(
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed"
    "sm__inst_executed_pipe_tex.avg.pct_of_peak_sustained_elapsed"
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed"
    "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_elapsed"
    "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed"
    "smsp__inst_executed.sum"
    "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    "sm__inst_executed.avg.per_cycle_active"
)

# TMA / MIO (if available)
TMA_METRICS=(
    "sm__mio_pq_read_cycles_active.avg.pct_of_peak_sustained_elapsed"
    "sm__mio_pq_write_cycles_active.avg.pct_of_peak_sustained_elapsed"
)

# Join all metrics
ALL_METRICS=("${STALL_METRICS[@]}" "${MEM_METRICS[@]}" "${PIPE_METRICS[@]}" "${TMA_METRICS[@]}")
METRICS_CSV=$(IFS=,; echo "${ALL_METRICS[*]}")

# ── Experiment definitions ──
# Format: "label|dflags"

# Part 1: Warp scaling (0,1,2,3,4 active warps) — for ncu + wall
WARP_SCALING=(
    "strip|-DSTRIP_EPILOGUE=1"
    "w1|-DEPI_WARP_LIMIT=1"
    "w2|-DEPI_WARP_LIMIT=2"
    "w3|-DEPI_WARP_LIMIT=3"
    "baseline|"
)

# Part 2: Per-operation at 1 warp (below saturation) — wall only
W1_STRIP=(
    "w1_full|-DEPI_WARP_LIMIT=1"
    "w1_only_tma_load|-DEPI_WARP_LIMIT=1 -DSTRIP_EPI_TMA_STORE=1 -DSTRIP_EPI_COMPUTE=1"
    "w1_only_tma_store|-DEPI_WARP_LIMIT=1 -DSTRIP_EPI_TMA_LOAD=1 -DSTRIP_EPI_COMPUTE=1"
    "w1_only_compute|-DEPI_WARP_LIMIT=1 -DSTRIP_EPI_TMA_LOAD=1 -DSTRIP_EPI_TMA_STORE=1"
    "w1_no_tma_load|-DEPI_WARP_LIMIT=1 -DSTRIP_EPI_TMA_LOAD=1"
    "w1_no_tma_store|-DEPI_WARP_LIMIT=1 -DSTRIP_EPI_TMA_STORE=1"
    "w1_no_compute|-DEPI_WARP_LIMIT=1 -DSTRIP_EPI_COMPUTE=1"
)

# Part 3: Per-operation at 2 warps (test additivity) — wall only
W2_STRIP=(
    "w2_full|-DEPI_WARP_LIMIT=2"
    "w2_only_tma_load|-DEPI_WARP_LIMIT=2 -DSTRIP_EPI_TMA_STORE=1 -DSTRIP_EPI_COMPUTE=1"
    "w2_only_tma_store|-DEPI_WARP_LIMIT=2 -DSTRIP_EPI_TMA_LOAD=1 -DSTRIP_EPI_COMPUTE=1"
    "w2_only_compute|-DEPI_WARP_LIMIT=2 -DSTRIP_EPI_TMA_LOAD=1 -DSTRIP_EPI_TMA_STORE=1"
)

# Part 4: ncu on per-operation at 1 warp — stall attribution per operation
W1_STRIP_NCU=(
    "w1_only_tma_load|-DEPI_WARP_LIMIT=1 -DSTRIP_EPI_TMA_STORE=1 -DSTRIP_EPI_COMPUTE=1"
    "w1_only_tma_store|-DEPI_WARP_LIMIT=1 -DSTRIP_EPI_TMA_LOAD=1 -DSTRIP_EPI_COMPUTE=1"
    "w1_only_compute|-DEPI_WARP_LIMIT=1 -DSTRIP_EPI_TMA_LOAD=1 -DSTRIP_EPI_TMA_STORE=1"
)

run_wall() {
    local label="$1" dflags="$2"
    log "  [wall] $label  DFLAGS='$dflags'"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  make fc2 DFLAGS=\"$dflags\" && ./fc2"
        return 0
    fi

    if ! make fc2 DFLAGS="$dflags" > "$OUTDIR/${label}_build.log" 2>&1; then
        log "  [$label] BUILD FAILED"
        cat "$OUTDIR/${label}_build.log" >> "$OUTDIR/session.log"
        return 1
    fi

    local regs
    regs=$(grep -o '[0-9]* registers' "$OUTDIR/${label}_build.log" | head -1 | grep -o '[0-9]*')

    local output
    if ! output=$(./fc2 2>&1); then
        log "  [$label] RUN FAILED"
        echo "$output" >> "$OUTDIR/session.log"
        return 1
    fi

    echo "$output" > "$OUTDIR/${label}_output.txt"

    local result_line
    result_line=$(echo "$output" | grep '@@RESULT' | head -1)
    if [ -n "$result_line" ]; then
        echo "${result_line} label=${label} regs=${regs} dflags=\"${dflags}\"" >> "$OUTDIR/wall_results.txt"
        local ms valid
        ms=$(echo "$result_line" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
        valid=$(echo "$result_line" | grep -o 'valid=[01]' | cut -d= -f2)
        log "  [$label] ${ms}ms  valid=${valid}  regs=${regs}"
    else
        log "  [$label] NO @@RESULT LINE"
        echo "@@RESULT ms=ERR label=${label}" >> "$OUTDIR/wall_results.txt"
    fi
}

run_ncu() {
    local label="$1" dflags="$2"
    log "  [ncu] $label  DFLAGS='$dflags'"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  make fc2 DFLAGS=\"$dflags\" && ncu --metrics ... ./fc2"
        return 0
    fi

    if ! make fc2 DFLAGS="$dflags" > "$OUTDIR/${label}_build.log" 2>&1; then
        log "  [$label] NCU BUILD FAILED"
        cat "$OUTDIR/${label}_build.log" >> "$OUTDIR/session.log"
        return 1
    fi

    local regs
    regs=$(grep -o '[0-9]* registers' "$OUTDIR/${label}_build.log" | head -1 | grep -o '[0-9]*')
    log "  [$label] regs=${regs}, collecting ncu metrics..."

    # Collect ncu metrics as CSV.
    # --csv gives machine-readable output. Do NOT use --page raw (overrides --metrics).
    # ncu-rep first, then export CSV (separates kernel stdout from metric data).
    if ncu --metrics "$METRICS_CSV" \
           -o "$OUTDIR/ncu/${label}" \
           ./fc2 \
           > "$OUTDIR/ncu/${label}_stdout.txt" 2>"$OUTDIR/ncu/${label}_stderr.txt"; then
        # Export clean CSV from the .ncu-rep (no kernel stdout contamination)
        ncu --import "$OUTDIR/ncu/${label}.ncu-rep" \
            --csv \
            > "$OUTDIR/ncu/${label}.csv" 2>/dev/null
        rm -f "$OUTDIR/ncu/${label}.ncu-rep"
        local lines
        lines=$(wc -l < "$OUTDIR/ncu/${label}.csv")
        log "  [$label] ncu: ${lines} CSV lines"
    else
        log "  [$label] NCU FAILED (see ncu/${label}_stderr.txt)"
    fi
}

START_TIME=$(date +%s)

# ════════════════════════════════════════
# PART A: Wall time experiments (~2 min)
# ════════════════════════════════════════
if [ "$RUN_WALL" = "1" ]; then
    log ""
    log "==== PART A: WALL TIME — WARP SCALING ===="
    for exp in "${WARP_SCALING[@]}"; do
        IFS='|' read -r label dflags <<< "$exp"
        run_wall "$label" "$dflags"
    done

    log ""
    log "==== PART A: WALL TIME — PER-OPERATION @ 1 WARP ===="
    for exp in "${W1_STRIP[@]}"; do
        IFS='|' read -r label dflags <<< "$exp"
        run_wall "$label" "$dflags"
    done

    log ""
    log "==== PART A: WALL TIME — PER-OPERATION @ 2 WARPS ===="
    for exp in "${W2_STRIP[@]}"; do
        IFS='|' read -r label dflags <<< "$exp"
        run_wall "$label" "$dflags"
    done

    # Thermal drift check
    log ""
    log "==== PART A: THERMAL DRIFT CHECK ===="
    run_wall "strip_end" "-DSTRIP_EPILOGUE=1"

    # ── Wall time summary ──
    if [ -f "$OUTDIR/wall_results.txt" ] && [ "$DRY_RUN" = "0" ]; then
        log ""
        log "==== WALL TIME SUMMARY ===="

        strip_ms=$(grep 'label=strip ' "$OUTDIR/wall_results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)

        {
            printf "%-25s %8s %8s %8s  %s\n" "LABEL" "MS" "DELTA" "REGS" "DFLAGS"
            printf "%-25s %8s %8s %8s  %s\n" "-------------------------" "--------" "--------" "--------" "------"
            while IFS= read -r line; do
                ms=$(echo "$line" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
                label=$(echo "$line" | grep -o 'label=[^ ]*' | cut -d= -f2)
                regs=$(echo "$line" | grep -o 'regs=[0-9]*' | cut -d= -f2)
                dflags=$(echo "$line" | sed -n 's/.*dflags="\([^"]*\)".*/\1/p')
                valid=$(echo "$line" | grep -o 'valid=[01]' | cut -d= -f2)
                mark=""
                [ "$valid" = "0" ] && mark=" INVALID"

                delta="—"
                if [ -n "$strip_ms" ] && [ -n "$ms" ]; then
                    delta=$(echo "$ms - $strip_ms" | bc -l 2>/dev/null || echo "?")
                fi

                printf "%-25s %8s %8s %5s  %s%s\n" "$label" "$ms" "$delta" "$regs" "$dflags" "$mark"
            done < "$OUTDIR/wall_results.txt"
        } | tee "$OUTDIR/wall_summary.txt"

        # Decomposition at 1 warp
        w1_full_ms=$(grep 'label=w1_full ' "$OUTDIR/wall_results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
        if [ -n "$w1_full_ms" ] && [ -n "$strip_ms" ]; then
            w1_overhead=$(echo "$w1_full_ms - $strip_ms" | bc -l 2>/dev/null || echo "?")
            echo "" | tee -a "$OUTDIR/wall_summary.txt"
            echo "=== 1-WARP DECOMPOSITION (overhead=${w1_overhead}ms) ===" | tee -a "$OUTDIR/wall_summary.txt"
            printf "%-25s %8s %8s %8s\n" "OPERATION" "MS" "DELTA" "% OF 1W OVERHEAD" | tee -a "$OUTDIR/wall_summary.txt"
            printf "%-25s %8s %8s %8s\n" "-------------------------" "--------" "--------" "----------------" | tee -a "$OUTDIR/wall_summary.txt"
            for exp in "${W1_STRIP[@]}"; do
                IFS='|' read -r label dflags <<< "$exp"
                vms=$(grep "label=${label} " "$OUTDIR/wall_results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
                if [ -n "$vms" ] && [ "$w1_overhead" != "?" ] && [ "$w1_overhead" != "0" ]; then
                    vdelta=$(echo "$vms - $strip_ms" | bc -l 2>/dev/null || echo "?")
                    pct=$(echo "scale=1; $vdelta / $w1_overhead * 100" | bc -l 2>/dev/null || echo "?")
                    printf "%-25s %8s %8.3f %7s%%\n" "$label" "$vms" "$vdelta" "$pct" | tee -a "$OUTDIR/wall_summary.txt"
                fi
            done
        fi

        # Decomposition at 2 warps
        w2_full_ms=$(grep 'label=w2_full ' "$OUTDIR/wall_results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
        if [ -n "$w2_full_ms" ] && [ -n "$strip_ms" ]; then
            w2_overhead=$(echo "$w2_full_ms - $strip_ms" | bc -l 2>/dev/null || echo "?")
            echo "" | tee -a "$OUTDIR/wall_summary.txt"
            echo "=== 2-WARP DECOMPOSITION (overhead=${w2_overhead}ms) ===" | tee -a "$OUTDIR/wall_summary.txt"
            printf "%-25s %8s %8s %8s\n" "OPERATION" "MS" "DELTA" "% OF 2W OVERHEAD" | tee -a "$OUTDIR/wall_summary.txt"
            printf "%-25s %8s %8s %8s\n" "-------------------------" "--------" "--------" "----------------" | tee -a "$OUTDIR/wall_summary.txt"
            for exp in "${W2_STRIP[@]}"; do
                IFS='|' read -r label dflags <<< "$exp"
                vms=$(grep "label=${label} " "$OUTDIR/wall_results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
                if [ -n "$vms" ] && [ "$w2_overhead" != "?" ] && [ "$w2_overhead" != "0" ]; then
                    vdelta=$(echo "$vms - $strip_ms" | bc -l 2>/dev/null || echo "?")
                    pct=$(echo "scale=1; $vdelta / $w2_overhead * 100" | bc -l 2>/dev/null || echo "?")
                    printf "%-25s %8s %8.3f %7s%%\n" "$label" "$vms" "$vdelta" "$pct" | tee -a "$OUTDIR/wall_summary.txt"
                fi
            done
        fi
    fi
fi

# ════════════════════════════════════════
# PART B: NCU profiling (~15 min)
# ════════════════════════════════════════
if [ "$RUN_NCU" = "1" ]; then
    log ""
    log "==== PART B: NCU — WARP SCALING (0,1,2,3,4 warps) ===="
    for exp in "${WARP_SCALING[@]}"; do
        IFS='|' read -r label dflags <<< "$exp"
        run_ncu "ncu_${label}" "$dflags"
    done

    log ""
    log "==== PART B: NCU — PER-OPERATION @ 1 WARP ===="
    for exp in "${W1_STRIP_NCU[@]}"; do
        IFS='|' read -r label dflags <<< "$exp"
        run_ncu "ncu_${label}" "$dflags"
    done
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
log ""
log "Done in ${ELAPSED}s. Results: $OUTDIR/"
log "  wall_summary.txt  — wall time decomposition"
log "  ncu/              — per-variant ncu CSV"
log "  session.log       — full log"
