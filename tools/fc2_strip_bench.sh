#!/bin/bash
# FC2 epilogue contention decomposition — full persistent-kernel scale benchmarks.
# Requires B200. Runs ~30 experiments in ~10 minutes.
#
# Usage:
#   ./tools/fc2_strip_bench.sh              # run everything
#   ./tools/fc2_strip_bench.sh --batch 1    # run only batch 1
#   ./tools/fc2_strip_bench.sh --dry-run    # print commands without running
#
# Output:  data/strip_bench_YYYYMMDD_HHMMSS/
#   results.txt     — parseable @@RESULT lines with labels
#   timing/         — full timing output per variant
#   summary.txt     — final table

set -uo pipefail
cd "$(dirname "$0")/.."

BATCH_ONLY=0
DRY_RUN=0
RUNS=10   # cudaEvent launches per variant (fc2.cu does 10 internally)
while [ $# -gt 0 ]; do
    case "$1" in
        --batch) BATCH_ONLY="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="data/strip_bench_${TIMESTAMP}"
mkdir -p "$OUTDIR/timing"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/session.log"; }
should_run() { [ "$BATCH_ONLY" = "0" ] || [ "$BATCH_ONLY" = "$1" ]; }

log "========================================"
log "  FC2 STRIP BENCH  $TIMESTAMP"
log "  Output: $OUTDIR"
log "========================================"

# Preflight
if [ "$DRY_RUN" = "0" ]; then
    nvidia-smi > /dev/null 2>&1 || { log "FATAL: no GPU"; exit 1; }
    nvidia-smi --query-gpu=gpu_name,clocks.sm --format=csv,noheader | tee -a "$OUTDIR/session.log"
fi

# ── Experiment definitions ──
# Format: "label|dflags|timing"
#   timing=0 -> wall time only (fc2)
#   timing=1 -> timing build (fc2-timing)

BATCH1_EXPERIMENTS=(
    "baseline||0"
    "strip_all|-DSTRIP_EPILOGUE=1|0"
    "strip_tma_load|-DSTRIP_EPI_TMA_LOAD=1|0"
    "strip_tma_store|-DSTRIP_EPI_TMA_STORE=1|0"
    "strip_compute|-DSTRIP_EPI_COMPUTE=1|0"
)

BATCH2_EXPERIMENTS=(
    "epi_sync|-DEPI_SYNC=1|0"
    "bar_pass|-DEPI_BAR_PASS=1|0"
    "bar_chunk|-DEPI_BAR_CHUNK=1|0"
    "sync_bar_pass|-DEPI_SYNC=1 -DEPI_BAR_PASS=1|0"
    "sync_bar_chunk|-DEPI_SYNC=1 -DEPI_BAR_CHUNK=1|0"
)

BATCH3_EXPERIMENTS=(
    "fp32_epi|-DFP32_EPILOGUE=1|0"
    "stagger_500|-DSTAGGER_CYCLES=500|0"
    "stagger_2000|-DSTAGGER_CYCLES=2000|0"
    "passes_4|-DNUM_PASSES_PARAM=4|0"
    "store_deferred|-DSTORE_TIMING=1|0"
)

BATCH4_EXPERIMENTS=(
    "T_baseline||1"
    "T_strip_all|-DSTRIP_EPILOGUE=1|1"
    "T_strip_tma_load|-DSTRIP_EPI_TMA_LOAD=1|1"
    "T_strip_tma_store|-DSTRIP_EPI_TMA_STORE=1|1"
    "T_strip_compute|-DSTRIP_EPI_COMPUTE=1|1"
    "T_bar_pass|-DEPI_BAR_PASS=1|1"
    "T_bar_chunk|-DEPI_BAR_CHUNK=1|1"
    "T_fp32_epi|-DFP32_EPILOGUE=1|1"
)

# Batch 5: Combinatorial strips — isolate SINGLE operations
# (pairs of strips = keep only the third operation)
BATCH5_EXPERIMENTS=(
    "only_tma_load|-DSTRIP_EPI_TMA_STORE=1 -DSTRIP_EPI_COMPUTE=1|0"
    "only_tma_store|-DSTRIP_EPI_TMA_LOAD=1 -DSTRIP_EPI_COMPUTE=1|0"
    "only_compute|-DSTRIP_EPI_TMA_LOAD=1 -DSTRIP_EPI_TMA_STORE=1|0"
)

# Batch 6: BAR.SYNC + strip — which operation does BAR.SYNC help?
BATCH6_EXPERIMENTS=(
    "bar_chunk_no_load|-DEPI_BAR_CHUNK=1 -DSTRIP_EPI_TMA_LOAD=1|0"
    "bar_chunk_no_store|-DEPI_BAR_CHUNK=1 -DSTRIP_EPI_TMA_STORE=1|0"
)

# Batch 8: Hypothesis isolation — dispatch pressure vs TMEM timing
# NOP_EPILOGUE=N: arrive first (free TMEM), then busy-wait N cycles (dispatch pressure only)
# EPI_DELAY=N: busy-wait N cycles, then arrive (dispatch + TMEM timing)
BATCH8_EXPERIMENTS=(
    "nop_3000|-DNOP_EPILOGUE=3000|0"
    "nop_5000|-DNOP_EPILOGUE=5000|0"
    "nop_10000|-DNOP_EPILOGUE=10000|0"
    "delay_3000|-DEPI_DELAY=3000|0"
    "delay_5000|-DEPI_DELAY=5000|0"
    "delay_10000|-DEPI_DELAY=10000|0"
)

run_experiment() {
    local label="$1" dflags="$2" use_timing="$3"
    local target binary

    if [ "$use_timing" = "1" ]; then
        target="fc2-timing"
        binary="./fc2-timing"
    else
        target="fc2"
        binary="./fc2"
    fi

    log "  [$label] DFLAGS='$dflags' ($([ "$use_timing" = 1 ] && echo timing || echo wall))"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  make $target DFLAGS=\"$dflags\" && $binary"
        return 0
    fi

    # Build
    if ! make "$target" DFLAGS="$dflags" > "$OUTDIR/${label}_build.log" 2>&1; then
        log "  [$label] BUILD FAILED"
        cat "$OUTDIR/${label}_build.log" >> "$OUTDIR/session.log"
        return 1
    fi

    # Extract register count from build log
    local regs
    regs=$(grep -o '[0-9]* registers' "$OUTDIR/${label}_build.log" | head -1 | grep -o '[0-9]*')

    # Run
    local output
    if ! output=$($binary 2>&1); then
        log "  [$label] RUN FAILED"
        echo "$output" >> "$OUTDIR/session.log"
        return 1
    fi

    # Save full output
    echo "$output" > "$OUTDIR/timing/${label}.txt"

    # Extract @@RESULT line and append label + regs
    local result_line
    result_line=$(echo "$output" | grep '@@RESULT' | head -1)
    if [ -n "$result_line" ]; then
        echo "${result_line} label=${label} regs=${regs} dflags=\"${dflags}\"" >> "$OUTDIR/results.txt"
        local ms
        ms=$(echo "$result_line" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
        local valid
        valid=$(echo "$result_line" | grep -o 'valid=[01]' | cut -d= -f2)
        log "  [$label] ${ms}ms  valid=${valid}  regs=${regs}"
    else
        log "  [$label] NO @@RESULT LINE"
        echo "@@RESULT ms=ERR label=${label}" >> "$OUTDIR/results.txt"
    fi
}

run_batch() {
    local batch_num="$1" batch_name="$2"
    shift 2
    local experiments=("$@")

    if ! should_run "$batch_num"; then
        return 0
    fi

    log ""
    log "==== BATCH $batch_num: $batch_name (${#experiments[@]} experiments) ===="

    for exp in "${experiments[@]}"; do
        IFS='|' read -r label dflags use_timing <<< "$exp"
        run_experiment "$label" "$dflags" "$use_timing"
    done
}

START_TIME=$(date +%s)

run_batch 1 "STRIP VARIANTS (decompose 475us)"   "${BATCH1_EXPERIMENTS[@]}"
run_batch 2 "BAR.SYNC SERIALIZATION"              "${BATCH2_EXPERIMENTS[@]}"
run_batch 3 "STRUCTURAL VARIANTS"                 "${BATCH3_EXPERIMENTS[@]}"
run_batch 4 "TIMING BUILDS (cycle distributions)" "${BATCH4_EXPERIMENTS[@]}"
run_batch 5 "COMBINATORIAL STRIP (isolate single ops)" "${BATCH5_EXPERIMENTS[@]}"
run_batch 6 "BAR.SYNC + STRIP (which op does BAR help?)" "${BATCH6_EXPERIMENTS[@]}"
run_batch 8 "HYPOTHESIS ISOLATION (dispatch vs TMEM)" "${BATCH8_EXPERIMENTS[@]}"

# ── CUTLASS reference (same thermal session) ──
if should_run 7; then
    log ""
    log "==== BATCH 7: CUTLASS REFERENCE + THERMAL DRIFT ===="

    if [ "$DRY_RUN" = "1" ]; then
        echo "  make cutlass-bench-fc2 && ./cutlass-bench-fc2"
        echo "  make fc2 && ./fc2  (baseline repeat)"
    else
        # CUTLASS FC2
        log "  [cutlass_fc2] Building cutlass-bench-fc2..."
        if make cutlass-bench-fc2 > "$OUTDIR/cutlass_fc2_build.log" 2>&1; then
            cutlass_out=$(./cutlass-bench-fc2 2>&1) || true
            echo "$cutlass_out" > "$OUTDIR/timing/cutlass_fc2.txt"
            cutlass_ms=$(echo "$cutlass_out" | grep -o 'Best:.*ms' | grep -o '[0-9.]*' | head -1)
            log "  [cutlass_fc2] Best: ${cutlass_ms:-ERR}ms"
            echo "@@RESULT ms=${cutlass_ms:-ERR} label=cutlass_fc2" >> "$OUTDIR/results.txt"
        else
            log "  [cutlass_fc2] BUILD FAILED"
        fi

        # End-of-session baseline repeat (thermal drift check)
        run_experiment "baseline_end" "" "0"
    fi
fi

# ── Summary table ──
log ""
log "==== SUMMARY ===="

if [ -f "$OUTDIR/results.txt" ] && [ "$DRY_RUN" = "0" ]; then
    {
        printf "%-25s %8s %6s %5s  %s\n" "LABEL" "MS" "TFLOPS" "REGS" "DFLAGS"
        printf "%-25s %8s %6s %5s  %s\n" "-------------------------" "--------" "------" "-----" "------"
        while IFS= read -r line; do
            ms=$(echo "$line" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
            tflops=$(echo "$line" | grep -o 'tflops=[0-9.]*' | cut -d= -f2)
            label=$(echo "$line" | grep -o 'label=[^ ]*' | cut -d= -f2)
            regs=$(echo "$line" | grep -o 'regs=[0-9]*' | cut -d= -f2)
            dflags=$(echo "$line" | sed -n 's/.*dflags="\([^"]*\)".*/\1/p')
            valid=$(echo "$line" | grep -o 'valid=[01]' | cut -d= -f2)
            mark=""
            [ "$valid" = "0" ] && mark=" INVALID"
            printf "%-25s %8s %6s %5s  %s%s\n" "$label" "$ms" "$tflops" "$regs" "$dflags" "$mark"
        done < "$OUTDIR/results.txt"
    } | tee "$OUTDIR/summary.txt"

    # Compute deltas from baseline
    baseline_ms=$(grep 'label=baseline ' "$OUTDIR/results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
    strip_all_ms=$(grep 'label=strip_all ' "$OUTDIR/results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)

    if [ -n "$baseline_ms" ] && [ -n "$strip_all_ms" ]; then
        echo "" | tee -a "$OUTDIR/summary.txt"
        echo "=== DECOMPOSITION ===" | tee -a "$OUTDIR/summary.txt"
        echo "Baseline:      ${baseline_ms}ms" | tee -a "$OUTDIR/summary.txt"
        echo "Strip all:     ${strip_all_ms}ms" | tee -a "$OUTDIR/summary.txt"
        overhead=$(echo "$baseline_ms - $strip_all_ms" | bc -l 2>/dev/null || echo "?")
        echo "Epi overhead:  ${overhead}ms" | tee -a "$OUTDIR/summary.txt"
        echo "" | tee -a "$OUTDIR/summary.txt"

        printf "%-25s %8s %8s %8s\n" "VARIANT" "MS" "DELTA" "% OF OVERHEAD" | tee -a "$OUTDIR/summary.txt"
        printf "%-25s %8s %8s %8s\n" "-------------------------" "--------" "--------" "-------------" | tee -a "$OUTDIR/summary.txt"
        for exp in "${BATCH1_EXPERIMENTS[@]}" "${BATCH2_EXPERIMENTS[@]}" "${BATCH3_EXPERIMENTS[@]}" "${BATCH5_EXPERIMENTS[@]}" "${BATCH6_EXPERIMENTS[@]}" "${BATCH8_EXPERIMENTS[@]}"; do
            IFS='|' read -r label dflags use_timing <<< "$exp"
            [ "$use_timing" = "1" ] && continue
            vms=$(grep "label=${label} " "$OUTDIR/results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
            if [ -n "$vms" ] && [ "$overhead" != "?" ]; then
                delta=$(echo "$vms - $strip_all_ms" | bc -l 2>/dev/null || echo "?")
                if [ "$delta" != "?" ] && [ "$overhead" != "0" ]; then
                    pct=$(echo "scale=1; $delta / $overhead * 100" | bc -l 2>/dev/null || echo "?")
                else
                    pct="?"
                fi
                printf "%-25s %8s %8.3f %7s%%\n" "$label" "$vms" "$delta" "$pct" | tee -a "$OUTDIR/summary.txt"
            fi
        done
    fi
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
log ""
log "Done in ${ELAPSED}s. Results: $OUTDIR/"
log "  summary.txt    — comparison table"
log "  results.txt    — raw @@RESULT lines"
log "  timing/        — full per-variant output"
