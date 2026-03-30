#!/bin/bash
# FC2 prefetch + warp reduction benchmark.
# Tests: (1) ncu CUTLASS DRAM profile, (2) W0_RES_PREFETCH × EPI_WARP_LIMIT,
#        (3) NUM_EPI_WARPS=2 proper build.
#
# Usage:
#   ./tools/fc2_prefetch_bench.sh              # full run (~5 min)
#   ./tools/fc2_prefetch_bench.sh --dry-run    # print commands
#   ./tools/fc2_prefetch_bench.sh --no-cutlass # skip CUTLASS ncu (saves ~3 min)
#
# Output:  data/prefetch_YYYYMMDD_HHMMSS/

set -uo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=0
RUN_CUTLASS=1
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)    DRY_RUN=1; shift ;;
        --no-cutlass) RUN_CUTLASS=0; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="data/prefetch_${TIMESTAMP}"
mkdir -p "$OUTDIR/ncu"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/session.log"; }

log "========================================"
log "  FC2 PREFETCH BENCH  $TIMESTAMP"
log "  Output: $OUTDIR"
log "========================================"

if [ "$DRY_RUN" = "0" ]; then
    nvidia-smi > /dev/null 2>&1 || { log "FATAL: no GPU"; exit 1; }
    nvidia-smi --query-gpu=gpu_name,clocks.sm --format=csv,noheader | tee -a "$OUTDIR/session.log"
fi

# ── Best known FC2 base config ──
# N_STAGES=5 mandatory, TMA_RESIDUAL=1 (required for W0_RES_PREFETCH)
BASE="-DN_STAGES=5 -DTMA_RESIDUAL=1"

# ── ncu metrics (same as contention bench, fixed) ──
NCU_METRICS=(
    "smsp__warps_issue_stalled_long_scoreboard.avg"
    "smsp__warps_issue_stalled_short_scoreboard.avg"
    "smsp__warps_issue_stalled_wait.avg"
    "smsp__warps_issue_stalled_barrier.avg"
    "smsp__warps_issue_stalled_sleeping.avg"
    "smsp__warps_issue_stalled_not_selected.avg"
    "smsp__warps_issue_stalled_mio_throttle.avg"
    "smsp__warps_issue_stalled_math_pipe_throttle.avg"
    "smsp__cycles_active.avg"
    "sm__warps_active.avg.per_cycle_active"
    "dram__throughput.avg.pct_of_peak_sustained_elapsed"
    "dram__bytes_read.sum"
    "dram__bytes_write.sum"
    "lts__throughput.avg.pct_of_peak_sustained_elapsed"
    "lts__t_sectors.sum"
    "lts__t_sectors_op_read.sum"
    "lts__t_sectors_op_write.sum"
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"
    "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed"
    "sm__mio_pq_read_cycles_active.avg.pct_of_peak_sustained_elapsed"
    "sm__mio_pq_write_cycles_active.avg.pct_of_peak_sustained_elapsed"
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed"
    "smsp__inst_executed.sum"
)
NCU_METRICS_CSV=$(IFS=,; echo "${NCU_METRICS[*]}")

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
    local label="$1" binary="$2" dflags="${3:-}"
    log "  [ncu] $label"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  ncu --metrics ... $binary"
        return 0
    fi

    # Build if dflags provided (our kernel), otherwise assume binary exists (CUTLASS)
    if [ -n "$dflags" ]; then
        if ! make fc2 DFLAGS="$dflags" > "$OUTDIR/${label}_build.log" 2>&1; then
            log "  [$label] NCU BUILD FAILED"
            return 1
        fi
    fi

    if [ ! -x "$binary" ]; then
        log "  [$label] binary not found: $binary"
        return 1
    fi

    log "  [$label] collecting ncu metrics..."
    if ncu --metrics "$NCU_METRICS_CSV" \
           -o "$OUTDIR/ncu/${label}" \
           "$binary" \
           > "$OUTDIR/ncu/${label}_stdout.txt" 2>"$OUTDIR/ncu/${label}_stderr.txt"; then
        ncu --import "$OUTDIR/ncu/${label}.ncu-rep" \
            --csv \
            > "$OUTDIR/ncu/${label}.csv" 2>/dev/null
        rm -f "$OUTDIR/ncu/${label}.ncu-rep"
        local lines
        lines=$(wc -l < "$OUTDIR/ncu/${label}.csv")
        log "  [$label] ncu: ${lines} CSV lines"
    else
        log "  [$label] NCU FAILED"
        cat "$OUTDIR/ncu/${label}_stderr.txt" >> "$OUTDIR/session.log"
    fi
}

START_TIME=$(date +%s)

# ════════════════════════════════════════
# TEST 1: ncu on CUTLASS FC2 baseline
# ════════════════════════════════════════
if [ "$RUN_CUTLASS" = "1" ]; then
    log ""
    log "==== TEST 1: NCU ON CUTLASS FC2 ===="

    # Build CUTLASS first
    if [ "$DRY_RUN" = "0" ]; then
        log "  Building cutlass-bench-fc2-max..."
        if make cutlass-bench-fc2-max > "$OUTDIR/cutlass_build.log" 2>&1; then
            log "  CUTLASS build OK"
            # Wall time reference
            cutlass_out=$(./cutlass-bench-fc2-max 2>&1)
            echo "$cutlass_out" > "$OUTDIR/cutlass_output.txt"
            cutlass_ms=$(echo "$cutlass_out" | grep -i 'best\|fastest\|min' | head -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)
            if [ -n "$cutlass_ms" ]; then
                log "  CUTLASS wall time: ${cutlass_ms}ms"
            else
                log "  CUTLASS output (first 5 lines):"
                echo "$cutlass_out" | head -5 | while read -r line; do log "    $line"; done
            fi
        else
            log "  CUTLASS BUILD FAILED"
            cat "$OUTDIR/cutlass_build.log" | tail -5 >> "$OUTDIR/session.log"
        fi
    fi

    # ncu on CUTLASS
    if [ -x "./cutlass-bench-fc2-max" ]; then
        run_ncu "ncu_cutlass" "./cutlass-bench-fc2-max"
    fi

    # ncu on our best config (for side-by-side comparison)
    run_ncu "ncu_ours_best" "./fc2" "$BASE"

    # ncu on strip (reference)
    run_ncu "ncu_strip" "./fc2" "$BASE -DSTRIP_EPILOGUE=1"
fi

# ════════════════════════════════════════
# TEST 2: W0_RES_PREFETCH × EPI_WARP_LIMIT
# ════════════════════════════════════════
log ""
log "==== TEST 2: PREFETCH × WARP LIMIT ===="

# Reference points
WALL_EXPERIMENTS=(
    # Baselines at best config (NS5 + TMAR1)
    "strip_best|$BASE -DSTRIP_EPILOGUE=1"
    "best_4w|$BASE"
    "best_wl1|$BASE -DEPI_WARP_LIMIT=1"
    "best_wl2|$BASE -DEPI_WARP_LIMIT=2"
    "best_wl3|$BASE -DEPI_WARP_LIMIT=3"

    # W0_RES_PREFETCH + warp limit (does prefetch reduce per-warp overhead?)
    "pf_4w|$BASE -DW0_RES_PREFETCH=1"
    "pf_wl1|$BASE -DW0_RES_PREFETCH=1 -DEPI_WARP_LIMIT=1"
    "pf_wl2|$BASE -DW0_RES_PREFETCH=1 -DEPI_WARP_LIMIT=2"
    "pf_wl3|$BASE -DW0_RES_PREFETCH=1 -DEPI_WARP_LIMIT=3"
)

for exp in "${WALL_EXPERIMENTS[@]}"; do
    IFS='|' read -r label dflags <<< "$exp"
    run_wall "$label" "$dflags"
done

# ════════════════════════════════════════
# TEST 3: NUM_EPI_WARPS=2 proper build
# ════════════════════════════════════════
log ""
log "==== TEST 3: NUM_EPI_WARPS=2 (128 threads) ===="

# NUM_EPI_WARPS=2 means only 2 epilogue warps (W2-W3), 4 total warps (128 threads).
# row_group = ew % 4 → ew=0 gets row_group=0, ew=1 gets row_group=1.
# rows 64-127 are NEVER written → output will be WRONG (but timing is valid).
# This tests whether 4-warp kernel (no idle warps) is faster than 6-warp + EPI_WARP_LIMIT=2.
NEPI_EXPERIMENTS=(
    "nepi2|$BASE -DNUM_EPI_WARPS=2"
    "nepi2_pf|$BASE -DNUM_EPI_WARPS=2 -DW0_RES_PREFETCH=1"
    "nepi3|$BASE -DNUM_EPI_WARPS=3"
    "nepi3_pf|$BASE -DNUM_EPI_WARPS=3 -DW0_RES_PREFETCH=1"
)

for exp in "${NEPI_EXPERIMENTS[@]}"; do
    IFS='|' read -r label dflags <<< "$exp"
    run_wall "$label" "$dflags"
done

# ════════════════════════════════════════
# Thermal drift check
# ════════════════════════════════════════
log ""
log "==== THERMAL DRIFT CHECK ===="
run_wall "strip_end" "$BASE -DSTRIP_EPILOGUE=1"

# ════════════════════════════════════════
# Summary
# ════════════════════════════════════════
if [ -f "$OUTDIR/wall_results.txt" ] && [ "$DRY_RUN" = "0" ]; then
    log ""
    log "==== WALL TIME SUMMARY ===="

    strip_ms=$(grep 'label=strip_best ' "$OUTDIR/wall_results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)

    {
        printf "%-20s %8s %8s %5s  %s\n" "LABEL" "MS" "DELTA" "REGS" "DFLAGS"
        printf "%-20s %8s %8s %5s  %s\n" "--------------------" "--------" "--------" "-----" "------"
        while IFS= read -r line; do
            ms=$(echo "$line" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
            label=$(echo "$line" | grep -o 'label=[^ ]*' | cut -d= -f2)
            regs=$(echo "$line" | grep -o 'regs=[0-9]*' | cut -d= -f2)
            dflags=$(echo "$line" | sed -n 's/.*dflags="\([^"]*\)".*/\1/p')
            valid=$(echo "$line" | grep -o 'valid=[01]' | cut -d= -f2)
            mark=""
            [ "$valid" = "0" ] && mark=" INV"

            delta="—"
            if [ -n "$strip_ms" ] && [ -n "$ms" ]; then
                delta=$(echo "$ms - $strip_ms" | bc -l 2>/dev/null | sed 's/^\./0./' || echo "?")
            fi

            printf "%-20s %8s %8s %5s  %s%s\n" "$label" "$ms" "$delta" "$regs" "$dflags" "$mark"
        done < "$OUTDIR/wall_results.txt"
    } | tee "$OUTDIR/wall_summary.txt"

    # Side-by-side: prefetch vs no-prefetch
    echo "" | tee -a "$OUTDIR/wall_summary.txt"
    echo "=== PREFETCH COMPARISON ===" | tee -a "$OUTDIR/wall_summary.txt"
    for wl in 1 2 3 4w; do
        base_label="best_wl${wl}"
        pf_label="pf_wl${wl}"
        if [ "$wl" = "4w" ]; then
            base_label="best_4w"
            pf_label="pf_4w"
        fi
        base_ms=$(grep "label=${base_label} " "$OUTDIR/wall_results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
        pf_ms=$(grep "label=${pf_label} " "$OUTDIR/wall_results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
        if [ -n "$base_ms" ] && [ -n "$pf_ms" ]; then
            diff_us=$(echo "($pf_ms - $base_ms) * 1000" | bc -l 2>/dev/null | sed 's/^\./0./' | cut -c1-6 || echo "?")
            printf "  %-10s  base=%sms  prefetch=%sms  Δ=%sμs\n" "$wl" "$base_ms" "$pf_ms" "$diff_us" | tee -a "$OUTDIR/wall_summary.txt"
        fi
    done

    # NUM_EPI_WARPS comparison
    echo "" | tee -a "$OUTDIR/wall_summary.txt"
    echo "=== NUM_EPI_WARPS COMPARISON ===" | tee -a "$OUTDIR/wall_summary.txt"
    for n in 2 3; do
        nepi_ms=$(grep "label=nepi${n} " "$OUTDIR/wall_results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
        wl_ms=$(grep "label=best_wl${n} " "$OUTDIR/wall_results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
        nepi_pf_ms=$(grep "label=nepi${n}_pf " "$OUTDIR/wall_results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
        printf "  NEpi=%d:  proper=%sms  warp_limit=%sms  proper+pf=%sms\n" "$n" "$nepi_ms" "$wl_ms" "$nepi_pf_ms" | tee -a "$OUTDIR/wall_summary.txt"
    done
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
log ""
log "Done in ${ELAPSED}s. Results: $OUTDIR/"
log "  wall_summary.txt  — wall time + prefetch comparison"
log "  ncu/              — CUTLASS + ours ncu CSV"
log "  session.log       — full log"
