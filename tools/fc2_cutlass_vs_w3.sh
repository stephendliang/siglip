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

# Variants: CUTLASS + cuBLAS baselines, w3 dispatch variants, decomposition.
# All w3 builds use -DPACKED_TILES for parity with tools/fc2_ncu_bench.sh.
EXPERIMENTS=(
    "cutlass_fused|make fc2-cutlass|./fc2-cutlass"
    "cutlass_strip|make fc2-cutlass-strip|./fc2-cutlass-strip"
    "cublas_gemm|make -B cublas-bench-fc2|./cublas-bench-fc2"
    "w3_fused|make -B fc2-w3 DFLAGS=-DPACKED_TILES|./fc2-w3"
    "w3_lean|make -B fc2-w3-lean DFLAGS=-DPACKED_TILES|./fc2-w3-lean"
    "w3_sched|make -B fc2-w3-sched DFLAGS=-DPACKED_TILES|./fc2-w3-sched"
    "w3_rowsteal|make -B fc2-w3-rowsteal DFLAGS=-DPACKED_TILES|./fc2-w3-rowsteal"
    "w3_tail|make -B fc2-w3-tail DFLAGS=-DPACKED_TILES|./fc2-w3-tail"
    "w3_tail_lean|make -B fc2-w3-tail-lean DFLAGS=-DPACKED_TILES|./fc2-w3-tail-lean"
    "w3_dgswizzle|make -B fc2-w3-dgswizzle DFLAGS='-DPACKED_TILES -DNO_PREFILL'|./fc2-w3-dgswizzle"
    "w3_zorder|make -B fc2-w3-zorder DFLAGS='-DPACKED_TILES -DNO_PREFILL'|./fc2-w3-zorder"
    "w3_hilbert|make -B fc2-w3-hilbert DFLAGS='-DPACKED_TILES -DNO_PREFILL'|./fc2-w3-hilbert"
    "w3_zigzag|make -B fc2-w3-zigzag DFLAGS='-DPACKED_TILES -DNO_PREFILL'|./fc2-w3-zigzag"
    "w3_rowmajor|make -B fc2-w3-rowmajor DFLAGS='-DPACKED_TILES -DNO_PREFILL'|./fc2-w3-rowmajor"
    "w3_ncycle|make -B fc2-w3-ncycle DFLAGS='-DPACKED_TILES -DNO_PREFILL'|./fc2-w3-ncycle"
    "w3_nflat|make -B fc2-w3-nflat DFLAGS='-DPACKED_TILES -DNO_PREFILL'|./fc2-w3-nflat"
    "w3_nsnake|make -B fc2-w3-nsnake DFLAGS='-DPACKED_TILES -DNO_PREFILL'|./fc2-w3-nsnake"
    "w3_dg4|make -B fc2-w3-dg4  DFLAGS='-DPACKED_TILES -DNO_PREFILL'|./fc2-w3-dg4"
    "w3_dg6|make -B fc2-w3-dg6  DFLAGS='-DPACKED_TILES -DNO_PREFILL'|./fc2-w3-dg6"
    "w3_dg12|make -B fc2-w3-dg12 DFLAGS='-DPACKED_TILES -DNO_PREFILL'|./fc2-w3-dg12"
    "w3_dg16|make -B fc2-w3-dg16 DFLAGS='-DPACKED_TILES -DNO_PREFILL'|./fc2-w3-dg16"
    "w3_dg24|make -B fc2-w3-dg24 DFLAGS='-DPACKED_TILES -DNO_PREFILL'|./fc2-w3-dg24"
    "w3_gemm|make -B fc2-w3-gemm DFLAGS=-DPACKED_TILES|./fc2-w3-gemm"
    "w3_lean_gemm|make -B fc2-w3-lean DFLAGS='-DPACKED_TILES -DGEMM_ONLY'|./fc2-w3-lean"
    "w3_sched_gemm|make -B fc2-w3-sched DFLAGS='-DPACKED_TILES -DGEMM_ONLY'|./fc2-w3-sched"
    "w3_rowsteal_gemm|make -B fc2-w3-rowsteal DFLAGS='-DPACKED_TILES -DGEMM_ONLY'|./fc2-w3-rowsteal"
    "w3_tail_gemm|make -B fc2-w3-tail DFLAGS='-DPACKED_TILES -DGEMM_ONLY'|./fc2-w3-tail"
    "w3_tail_lean_gemm|make -B fc2-w3-tail-lean DFLAGS='-DPACKED_TILES -DGEMM_ONLY'|./fc2-w3-tail-lean"
    "w3_dgswizzle_gemm|make -B fc2-w3-dgswizzle DFLAGS='-DPACKED_TILES -DNO_PREFILL -DGEMM_ONLY'|./fc2-w3-dgswizzle"
    "w3_zorder_gemm|make -B fc2-w3-zorder DFLAGS='-DPACKED_TILES -DNO_PREFILL -DGEMM_ONLY'|./fc2-w3-zorder"
    "w3_hilbert_gemm|make -B fc2-w3-hilbert DFLAGS='-DPACKED_TILES -DNO_PREFILL -DGEMM_ONLY'|./fc2-w3-hilbert"
    "w3_zigzag_gemm|make -B fc2-w3-zigzag DFLAGS='-DPACKED_TILES -DNO_PREFILL -DGEMM_ONLY'|./fc2-w3-zigzag"
    "w3_rowmajor_gemm|make -B fc2-w3-rowmajor DFLAGS='-DPACKED_TILES -DNO_PREFILL -DGEMM_ONLY'|./fc2-w3-rowmajor"
    "w3_ncycle_gemm|make -B fc2-w3-ncycle DFLAGS='-DPACKED_TILES -DNO_PREFILL -DGEMM_ONLY'|./fc2-w3-ncycle"
    "w3_nflat_gemm|make -B fc2-w3-nflat DFLAGS='-DPACKED_TILES -DNO_PREFILL -DGEMM_ONLY'|./fc2-w3-nflat"
    "w3_nsnake_gemm|make -B fc2-w3-nsnake DFLAGS='-DPACKED_TILES -DNO_PREFILL -DGEMM_ONLY'|./fc2-w3-nsnake"
    "w3_strip|make -B fc2-w3 DFLAGS='-DPACKED_TILES -DSTRIP_EPILOGUE'|./fc2-w3"
    "w3_lean_strip|make -B fc2-w3-lean DFLAGS='-DPACKED_TILES -DSTRIP_EPILOGUE'|./fc2-w3-lean"
    "w3_sched_strip|make -B fc2-w3-sched DFLAGS='-DPACKED_TILES -DSTRIP_EPILOGUE'|./fc2-w3-sched"
    "w3_rowsteal_strip|make -B fc2-w3-rowsteal DFLAGS='-DPACKED_TILES -DSTRIP_EPILOGUE'|./fc2-w3-rowsteal"
    "w3_tail_strip|make -B fc2-w3-tail DFLAGS='-DPACKED_TILES -DSTRIP_EPILOGUE'|./fc2-w3-tail"
    "w3_tail_lean_strip|make -B fc2-w3-tail-lean DFLAGS='-DPACKED_TILES -DSTRIP_EPILOGUE'|./fc2-w3-tail-lean"
    "w3_dgswizzle_strip|make -B fc2-w3-dgswizzle DFLAGS='-DPACKED_TILES -DNO_PREFILL -DSTRIP_EPILOGUE'|./fc2-w3-dgswizzle"
    "w3_zorder_strip|make -B fc2-w3-zorder DFLAGS='-DPACKED_TILES -DNO_PREFILL -DSTRIP_EPILOGUE'|./fc2-w3-zorder"
    "w3_hilbert_strip|make -B fc2-w3-hilbert DFLAGS='-DPACKED_TILES -DNO_PREFILL -DSTRIP_EPILOGUE'|./fc2-w3-hilbert"
    "w3_zigzag_strip|make -B fc2-w3-zigzag DFLAGS='-DPACKED_TILES -DNO_PREFILL -DSTRIP_EPILOGUE'|./fc2-w3-zigzag"
    "w3_rowmajor_strip|make -B fc2-w3-rowmajor DFLAGS='-DPACKED_TILES -DNO_PREFILL -DSTRIP_EPILOGUE'|./fc2-w3-rowmajor"
    "w3_ncycle_strip|make -B fc2-w3-ncycle DFLAGS='-DPACKED_TILES -DNO_PREFILL -DSTRIP_EPILOGUE'|./fc2-w3-ncycle"
    "w3_nflat_strip|make -B fc2-w3-nflat DFLAGS='-DPACKED_TILES -DNO_PREFILL -DSTRIP_EPILOGUE'|./fc2-w3-nflat"
    "w3_nsnake_strip|make -B fc2-w3-nsnake DFLAGS='-DPACKED_TILES -DNO_PREFILL -DSTRIP_EPILOGUE'|./fc2-w3-nsnake"
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

    # Extract decomposition: fused - gemm = epilogue overhead per dispatch method
    get_ms() { grep -E "label=$1( |$)" "$OUTDIR/results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2; }
    cutlass_fused=$(get_ms cutlass_fused)
    cutlass_strip=$(get_ms cutlass_strip)

    log "Decomposition (fused / gemm / strip):"
    printf "  %-16s %8s %8s %8s %10s %10s\n" "Variant" "fused" "gemm" "strip" "fused-gemm" "gemm-strip" | tee -a "$OUTDIR/session.log"
    printf "  %-16s %8s %8s %8s %10s %10s\n" "-------" "-----" "----" "-----" "----------" "----------" | tee -a "$OUTDIR/session.log"
    for variant in fused lean sched dgswizzle zorder hilbert zigzag rowmajor ncycle nflat nsnake; do
        fused_ms=$(get_ms "w3_${variant}")
        gemm_ms=$(get_ms "w3_${variant}_gemm")
        strip_ms=$(get_ms "w3_${variant}_strip")
        # w3_fused uses w3_gemm / w3_strip (default dispatch)
        [ "$variant" = "fused" ] && gemm_ms=$(get_ms "w3_gemm") && strip_ms=$(get_ms "w3_strip")
        fg=""; gs=""
        [ -n "$fused_ms" ] && [ -n "$gemm_ms" ] && fg=$(echo "$fused_ms - $gemm_ms" | bc)
        [ -n "$gemm_ms" ] && [ -n "$strip_ms" ] && gs=$(echo "$gemm_ms - $strip_ms" | bc)
        printf "  %-16s %7sms %7sms %7sms %9sms %9sms\n" \
            "$variant" "${fused_ms:--}" "${gemm_ms:--}" "${strip_ms:--}" "${fg:--}" "${gs:--}" | tee -a "$OUTDIR/session.log"
    done
    if [ -n "$cutlass_fused" ] && [ -n "$cutlass_strip" ]; then
        fg=$(echo "$cutlass_fused - $cutlass_strip" | bc)
        printf "  %-16s %7sms %8s %7sms %9sms %10s\n" \
            "cutlass" "$cutlass_fused" "-" "$cutlass_strip" "$fg" "-" | tee -a "$OUTDIR/session.log"
    fi

    log ""
    w3_lean=$(get_ms w3_lean)
    [ -n "$w3_lean" ] && [ -n "$cutlass_fused" ] && \
        log "Gap (lean vs CUTLASS fused): $(echo "$w3_lean - $cutlass_fused" | bc)ms"

    log ""
    log "Results: $OUTDIR/results.txt"
fi

log "Done."
