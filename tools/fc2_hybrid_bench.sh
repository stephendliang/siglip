#!/bin/bash
# FC2 hybrid sweep — test CUTLASS with explicit stage counts + our kernel.
# Goal: find CUTLASS mainloop config that matches our 1.092ms GEMM-only.
#
# Usage:
#   ./tools/fc2_hybrid_bench.sh              # run everything
#   ./tools/fc2_hybrid_bench.sh --dry-run    # print commands without running
#
# Output: data/hybrid_YYYYMMDD_HHMMSS/

set -uo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="data/hybrid_${TIMESTAMP}"
mkdir -p "$OUTDIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/session.log"; }

log "========================================"
log "  FC2 HYBRID SWEEP  $TIMESTAMP"
log "  Output: $OUTDIR"
log "========================================"

if [ "$DRY_RUN" = "0" ]; then
    nvidia-smi > /dev/null 2>&1 || { log "FATAL: no GPU"; exit 1; }
    nvidia-smi --query-gpu=gpu_name,clocks.sm --format=csv,noheader | tee -a "$OUTDIR/session.log"
fi

# label|build_cmd|run_cmd
EXPERIMENTS=()

# Our reference
EXPERIMENTS+=("w3_fused|make fc2-w3|./fc2-w3")
EXPERIMENTS+=("w3_strip|make fc2-w3 DFLAGS=-DSTRIP_EPILOGUE|./fc2-w3")

# CUTLASS auto (baseline)
EXPERIMENTS+=("cutlass_auto|make fc2-cutlass|./fc2-cutlass")
EXPERIMENTS+=("cutlass_auto_strip|make fc2-cutlass-strip|./fc2-cutlass-strip")

# CUTLASS explicit stage counts (fused + strip)
for S in 3 4 5 6 7; do
    EXPERIMENTS+=("cutlass_s${S}|make fc2-cutlass DFLAGS=-DMAINLOOP_STAGES=$S|./fc2-cutlass")
    EXPERIMENTS+=("cutlass_s${S}_strip|make fc2-cutlass-strip DFLAGS=-DMAINLOOP_STAGES=$S|./fc2-cutlass-strip")
done

log "Total experiments: ${#EXPERIMENTS[@]}"

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r label build_cmd run_cmd <<< "$exp"

    log "  [$label] $build_cmd"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  $build_cmd && $run_cmd"
        continue
    fi

    if ! eval "$build_cmd" > "$OUTDIR/${label}_build.log" 2>&1; then
        log "  [$label] BUILD FAILED"
        echo "@@RESULT ms=ERR label=$label" >> "$OUTDIR/results.txt"
        continue
    fi

    regs=$(grep -oP '(?<=Used )\d+(?= registers)' "$OUTDIR/${label}_build.log" | head -1)
    bars=$(grep -oP 'used \K\d+(?= barriers)' "$OUTDIR/${label}_build.log" | head -1)

    if ! output=$(timeout 30 $run_cmd 2>&1); then
        log "  [$label] RUN FAILED/TIMEOUT"
        echo "@@RESULT ms=HANG label=$label regs=$regs bars=$bars" >> "$OUTDIR/results.txt"
        continue
    fi

    echo "$output" > "$OUTDIR/${label}.txt"

    result_line=$(echo "$output" | grep '@@RESULT' | head -1)
    if [ -n "$result_line" ]; then
        echo "${result_line} label=${label} regs=${regs} bars=${bars}" >> "$OUTDIR/results.txt"
        ms=$(echo "$result_line" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
        valid=$(echo "$result_line" | grep -o 'valid=[01]' | cut -d= -f2)
        tflops=$(echo "$result_line" | grep -o 'tflops=[0-9.]*' | cut -d= -f2)
        log "  [$label] ${ms}ms  ${tflops} TFLOPS  valid=${valid}  regs=${regs}  bars=${bars}"
    else
        log "  [$label] NO @@RESULT LINE"
        echo "@@RESULT ms=ERR label=${label}" >> "$OUTDIR/results.txt"
    fi
done

# Summary
log ""
log "════════════════════════════════════════"
log "  SUMMARY"
log "════════════════════════════════════════"

if [ "$DRY_RUN" = "0" ] && [ -f "$OUTDIR/results.txt" ]; then
    while IFS= read -r line; do
        ms=$(echo "$line" | grep -o 'ms=[^ ]*' | cut -d= -f2)
        valid=$(echo "$line" | grep -o 'valid=[^ ]*' | cut -d= -f2)
        regs=$(echo "$line" | grep -o 'regs=[^ ]*' | cut -d= -f2)
        bars=$(echo "$line" | grep -o 'bars=[^ ]*' | cut -d= -f2)
        tflops=$(echo "$line" | grep -o 'tflops=[^ ]*' | cut -d= -f2)
        label=$(echo "$line" | grep -o 'label=[^ ]*' | cut -d= -f2)
        printf "%-25s  %8s  %7s TFLOPS  valid=%-1s  regs=%-3s  bars=%-1s\n" \
            "$label" "${ms}ms" "$tflops" "$valid" "$regs" "$bars"
    done < "$OUTDIR/results.txt" | tee -a "$OUTDIR/summary.txt"
    log ""
    log "Results: $OUTDIR/results.txt"
fi

log "Done."
