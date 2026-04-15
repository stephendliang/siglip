#!/bin/bash
# FC2 head-to-head: CUTLASS GemmUniversal vs our fc2_w3 hand-tuned kernel.
# Runs both fused and GEMM-only variants for decomposition.
#
# Usage:
#   ./tools/fc2_cutlass_vs_w3.sh              # run everything
#   ./tools/fc2_cutlass_vs_w3.sh --dry-run    # print commands without running
#
# Output: data/cutlass_vs_w3_YYYYMMDD_HHMMSS/

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
OUTDIR="data/cutlass_vs_w3_${TIMESTAMP}"
mkdir -p "$OUTDIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/session.log"; }

log "========================================"
log "  FC2: CUTLASS vs W3 HEAD-TO-HEAD"
log "  Output: $OUTDIR"
log "========================================"

if [ "$DRY_RUN" = "0" ]; then
    nvidia-smi > /dev/null 2>&1 || { log "FATAL: no GPU"; exit 1; }
    nvidia-smi --query-gpu=gpu_name,clocks.sm --format=csv,noheader | tee -a "$OUTDIR/session.log"
fi

# Variants: CUTLASS (reference), w3 dispatch variants, decomposition
EXPERIMENTS=(
    "cutlass_fused|make fc2-cutlass|./fc2-cutlass"
    "cutlass_strip|make fc2-cutlass-strip|./fc2-cutlass-strip"
    "w3_fused|make -B fc2-w3|./fc2-w3"
    "w3_lean|make -B fc2-w3-lean|./fc2-w3-lean"
    "w3_sched|make -B fc2-w3-sched|./fc2-w3-sched"
    "w3_dgswizzle|make -B fc2-w3-dgswizzle DFLAGS=-DNO_PREFILL|./fc2-w3-dgswizzle"
    "w3_zorder|make -B fc2-w3-zorder DFLAGS=-DNO_PREFILL|./fc2-w3-zorder"
    "w3_hilbert|make -B fc2-w3-hilbert DFLAGS=-DNO_PREFILL|./fc2-w3-hilbert"
    "w3_zigzag|make -B fc2-w3-zigzag DFLAGS=-DNO_PREFILL|./fc2-w3-zigzag"
    "w3_colfirst|make -B fc2-w3-colfirst DFLAGS=-DNO_PREFILL|./fc2-w3-colfirst"
    "w3_gemm|make -B fc2-w3-gemm|./fc2-w3-gemm"
    "w3_strip|make -B fc2-w3 DFLAGS=-DSTRIP_EPILOGUE|./fc2-w3"
)

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r label build_cmd run_cmd <<< "$exp"

    log "  [$label] Building: $build_cmd"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  $build_cmd && $run_cmd"
        continue
    fi

    if ! eval "$build_cmd" > "$OUTDIR/${label}_build.log" 2>&1; then
        log "  [$label] BUILD FAILED"
        echo "@@RESULT ms=ERR label=$label" >> "$OUTDIR/results.txt"
        continue
    fi

    regs=$(grep -o '[0-9]* registers' "$OUTDIR/${label}_build.log" | tail -1 | grep -o '[0-9]*')
    bars=$(grep -o 'used [0-9]* barriers' "$OUTDIR/${label}_build.log" | tail -1 | grep -o '[0-9]*')

    log "  [$label] Running: $run_cmd (regs=$regs, bars=$bars)"

    local_output=""
    if ! local_output=$(timeout 30 $run_cmd 2>&1); then
        log "  [$label] RUN FAILED/TIMEOUT"
        echo "@@RESULT ms=HANG label=$label regs=$regs bars=$bars" >> "$OUTDIR/results.txt"
        continue
    fi

    echo "$local_output" > "$OUTDIR/${label}.txt"

    result_line=$(echo "$local_output" | grep '@@RESULT' | head -1)
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
        printf "%-20s  %8s  %7s TFLOPS  valid=%-1s  regs=%-3s  bars=%-1s\n" \
            "$label" "${ms}ms" "$tflops" "$valid" "$regs" "$bars"
    done < "$OUTDIR/results.txt" | tee -a "$OUTDIR/summary.txt"

    log ""

    # Extract decomposition
    cutlass_fused=$(grep 'label=cutlass_fused' "$OUTDIR/results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
    cutlass_strip=$(grep 'label=cutlass_strip' "$OUTDIR/results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
    w3_fused=$(grep 'label=w3_fused' "$OUTDIR/results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
    w3_lean=$(grep 'label=w3_lean' "$OUTDIR/results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
    w3_strip=$(grep 'label=w3_strip' "$OUTDIR/results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)

    if [ -n "$cutlass_fused" ] && [ -n "$cutlass_strip" ] && [ -n "$w3_lean" ] && [ -n "$w3_strip" ]; then
        log "Decomposition:"
        log "  CUTLASS:  fused=${cutlass_fused}ms  strip=${cutlass_strip}ms  epi_overhead=$(echo "$cutlass_fused - $cutlass_strip" | bc)ms"
        log "  W3 lean:  fused=${w3_lean}ms  strip=${w3_strip}ms  epi_overhead=$(echo "$w3_lean - $w3_strip" | bc)ms"
        [ -n "$w3_fused" ] && \
        log "  W3 stride: fused=${w3_fused}ms  epi_overhead=$(echo "$w3_fused - $w3_strip" | bc)ms"
        log "  Gap (lean vs CUTLASS): $(echo "$w3_lean - $cutlass_fused" | bc)ms"
    fi

    log ""
    log "Results: $OUTDIR/results.txt"
fi

log "Done."
