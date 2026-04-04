#!/bin/bash
# FC2 W3 architecture search — combinatorial sweep of epilogue structural variants.
# Requires B200. Tests all combinations of SINGLE_WARP_STORE, DELAY_TMA_STORE,
# NUM_EPI_STAGES, FP32_EPILOGUE plus STRIP baseline.
#
# Usage:
#   ./tools/fc2_w3_arch_bench.sh              # run everything (~5 min)
#   ./tools/fc2_w3_arch_bench.sh --dry-run    # print commands without running
#   ./tools/fc2_w3_arch_bench.sh --quick      # skip FP32 variants (halves configs)
#
# Output: data/w3_arch_YYYYMMDD_HHMMSS/
#   results.txt  — parseable @@RESULT lines with labels
#   summary.txt  — sorted table

set -uo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=0
SKIP_FP32=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --quick)   SKIP_FP32=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="data/w3_arch_${TIMESTAMP}"
mkdir -p "$OUTDIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/session.log"; }

log "========================================"
log "  FC2 W3 ARCHITECTURE SEARCH  $TIMESTAMP"
log "  Output: $OUTDIR"
log "========================================"

# Preflight
if [ "$DRY_RUN" = "0" ]; then
    nvidia-smi > /dev/null 2>&1 || { log "FATAL: no GPU"; exit 1; }
    nvidia-smi --query-gpu=gpu_name,clocks.sm --format=csv,noheader | tee -a "$OUTDIR/session.log"
fi

# Build experiments: label|dflags
EXPERIMENTS=()

# Baseline + strip
EXPERIMENTS+=("baseline|")
EXPERIMENTS+=("strip|-DSTRIP_EPILOGUE")

# CUTLASS_LOOP: nounroll + FP32 — STS interleaving breakthrough
EXPERIMENTS+=("cl1|-DCUTLASS_LOOP=1")
EXPERIMENTS+=("cl1_fp32|-DCUTLASS_LOOP=1 -DFP32_EPILOGUE")
EXPERIMENTS+=("cl2|-DCUTLASS_LOOP=2")
EXPERIMENTS+=("cl3|-DCUTLASS_LOOP=3")
EXPERIMENTS+=("cl1_fp32_sws1|-DCUTLASS_LOOP=1 -DFP32_EPILOGUE -DSINGLE_WARP_STORE=1")
EXPERIMENTS+=("cl1_fp32_dts1|-DCUTLASS_LOOP=1 -DFP32_EPILOGUE -DDELAY_TMA_STORE=1 -DNUM_EPI_STAGES=3")
EXPERIMENTS+=("cl3_sws1|-DCUTLASS_LOOP=3 -DSINGLE_WARP_STORE=1")
EXPERIMENTS+=("cl3_dts1|-DCUTLASS_LOOP=3 -DDELAY_TMA_STORE=1 -DNUM_EPI_STAGES=3")

# CPP_EPILOGUE: break asm volatile blocks into individual schedulable instructions.
# Test with BF16 baseline and best combo from previous run.
EXPERIMENTS+=("cpp|-DCPP_EPILOGUE=1")
EXPERIMENTS+=("cpp_fp32|-DCPP_EPILOGUE=1 -DFP32_EPILOGUE")
EXPERIMENTS+=("cpp_sws1_es3|-DCPP_EPILOGUE=1 -DSINGLE_WARP_STORE=1 -DNUM_EPI_STAGES=3")

# Combinatorial: SWS × DTS × ES × FP32
for SWS in 0 1; do
    for DTS in 0 1; do
        for ES in 2 3; do
            for FP32 in 0 1; do
                # Skip base config (already in baseline)
                [ "$SWS" = "0" ] && [ "$DTS" = "0" ] && [ "$ES" = "2" ] && [ "$FP32" = "0" ] && continue
                # Skip FP32 in quick mode
                [ "$SKIP_FP32" = "1" ] && [ "$FP32" = "1" ] && continue

                label="sws${SWS}_dts${DTS}_es${ES}"
                dflags="-DSINGLE_WARP_STORE=$SWS -DDELAY_TMA_STORE=$DTS -DNUM_EPI_STAGES=$ES"
                if [ "$FP32" = "1" ]; then
                    label="${label}_fp32"
                    dflags="$dflags -DFP32_EPILOGUE"
                fi
                EXPERIMENTS+=("$label|$dflags")
            done
        done
    done
done

log "Total experiments: ${#EXPERIMENTS[@]}"

run_one() {
    local label="$1" dflags="$2"

    log "  [$label] DFLAGS='$dflags'"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  make fc2-w3 DFLAGS=\"$dflags\" && ./fc2-w3"
        return 0
    fi

    # Build
    if ! make fc2-w3 DFLAGS="$dflags" > "$OUTDIR/${label}_build.log" 2>&1; then
        log "  [$label] BUILD FAILED"
        echo "@@RESULT ms=ERR label=$label dflags=\"$dflags\"" >> "$OUTDIR/results.txt"
        return 1
    fi

    local regs
    regs=$(grep -o '[0-9]* registers' "$OUTDIR/${label}_build.log" | tail -1 | grep -o '[0-9]*')

    # Run with 5s timeout (deadlock protection)
    local output
    if ! output=$(timeout 15 ./fc2-w3 2>&1); then
        log "  [$label] RUN FAILED/TIMEOUT"
        echo "@@RESULT ms=HANG label=$label regs=$regs dflags=\"$dflags\"" >> "$OUTDIR/results.txt"
        return 1
    fi

    echo "$output" > "$OUTDIR/${label}.txt"

    local result_line
    result_line=$(echo "$output" | grep '@@RESULT' | head -1)
    if [ -n "$result_line" ]; then
        echo "${result_line} label=${label} regs=${regs} dflags=\"${dflags}\"" >> "$OUTDIR/results.txt"
        local ms valid
        ms=$(echo "$result_line" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
        valid=$(echo "$result_line" | grep -o 'valid=[01]' | cut -d= -f2)
        log "  [$label] ${ms}ms  valid=${valid}  regs=${regs}"
    else
        log "  [$label] NO @@RESULT LINE"
        echo "@@RESULT ms=ERR label=${label}" >> "$OUTDIR/results.txt"
    fi
}

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r label dflags <<< "$exp"
    run_one "$label" "$dflags"
done

# Summary
log ""
log "════════════════════════════════════════"
log "  SUMMARY"
log "════════════════════════════════════════"

if [ "$DRY_RUN" = "0" ] && [ -f "$OUTDIR/results.txt" ]; then
    # Sort by ms, show table
    echo "label | ms | valid | regs | tflops" > "$OUTDIR/summary.txt"
    echo "------|-----|-------|------|-------" >> "$OUTDIR/summary.txt"
    while IFS= read -r line; do
        ms=$(echo "$line" | grep -o 'ms=[^ ]*' | cut -d= -f2)
        valid=$(echo "$line" | grep -o 'valid=[^ ]*' | cut -d= -f2)
        regs=$(echo "$line" | grep -o 'regs=[^ ]*' | cut -d= -f2)
        tflops=$(echo "$line" | grep -o 'tflops=[^ ]*' | cut -d= -f2)
        label=$(echo "$line" | grep -o 'label=[^ ]*' | cut -d= -f2)
        printf "%-30s  %8s  valid=%-1s  regs=%-3s  tflops=%s\n" "$label" "${ms}ms" "$valid" "$regs" "$tflops"
    done < <(sort -t= -k2 -n "$OUTDIR/results.txt") | tee -a "$OUTDIR/summary.txt"
    log ""
    log "Results: $OUTDIR/results.txt"
    log "Summary: $OUTDIR/summary.txt"
fi

log "Done."
