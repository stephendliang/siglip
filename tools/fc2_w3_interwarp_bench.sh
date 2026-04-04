#!/bin/bash
# FC2 W3 inter-warp experiments — barrier removal, staging depth, decoupled warps.
# Requires B200. Tests NO_PRE_STORE_BAR, NO_POST_STORE_BAR, NUM_EPI_STAGES=3/4,
# and combinations with CUTLASS_LOOP + FP32.
#
# Usage:
#   ./tools/fc2_w3_interwarp_bench.sh              # run everything
#   ./tools/fc2_w3_interwarp_bench.sh --dry-run    # print commands without running
#
# Output: data/w3_interwarp_YYYYMMDD_HHMMSS/

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
OUTDIR="data/w3_interwarp_${TIMESTAMP}"
mkdir -p "$OUTDIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/session.log"; }

log "========================================"
log "  FC2 W3 INTER-WARP BENCH  $TIMESTAMP"
log "  Output: $OUTDIR"
log "========================================"

if [ "$DRY_RUN" = "0" ]; then
    nvidia-smi > /dev/null 2>&1 || { log "FATAL: no GPU"; exit 1; }
    nvidia-smi --query-gpu=gpu_name,clocks.sm --format=csv,noheader | tee -a "$OUTDIR/session.log"
fi

EXPERIMENTS=()

# Baseline + strip reference
EXPERIMENTS+=("baseline|")
EXPERIMENTS+=("strip|-DSTRIP_EPILOGUE")

# === Barrier removal (inter-warp decoupling) ===
# NO_PRE_STORE_BAR: remove bar.sync between STS and TMA store
# (each warp stores its own 32-row region, doesn't need cross-warp sync)
EXPERIMENTS+=("npsb|-DNO_PRE_STORE_BAR=1")

# NO_POST_STORE_BAR: remove bar.sync after TMA store wait
# (warps proceed independently, consumed_mbar still enforces W2 ordering)
EXPERIMENTS+=("nwb|-DNO_POST_STORE_BAR=1")

# Both: fully decoupled warps (no epilogue bar.sync at all)
EXPERIMENTS+=("nb|-DNO_PRE_STORE_BAR=1 -DNO_POST_STORE_BAR=1")

# === Staging depth (W2 pipeline lookahead) ===
EXPERIMENTS+=("nes3|-DNUM_EPI_STAGES=3")
EXPERIMENTS+=("nes4|-DNUM_EPI_STAGES=4")

# === Combined: decoupled warps + deeper staging ===
EXPERIMENTS+=("nes3_nb|-DNUM_EPI_STAGES=3 -DNO_PRE_STORE_BAR=1 -DNO_POST_STORE_BAR=1")
EXPERIMENTS+=("nes4_nb|-DNUM_EPI_STAGES=4 -DNO_PRE_STORE_BAR=1 -DNO_POST_STORE_BAR=1")

# === With FP32/CUTLASS_LOOP SASS interleaving ===
EXPERIMENTS+=("nb_cl1fp32|-DNO_PRE_STORE_BAR=1 -DNO_POST_STORE_BAR=1 -DCUTLASS_LOOP=1 -DFP32_EPILOGUE")
EXPERIMENTS+=("nes4_nb_cl1fp32|-DNUM_EPI_STAGES=4 -DNO_PRE_STORE_BAR=1 -DNO_POST_STORE_BAR=1 -DCUTLASS_LOOP=1 -DFP32_EPILOGUE")
EXPERIMENTS+=("nes3_npsb|-DNUM_EPI_STAGES=3 -DNO_PRE_STORE_BAR=1")
EXPERIMENTS+=("nes4_npsb|-DNUM_EPI_STAGES=4 -DNO_PRE_STORE_BAR=1")

# === Individual bars with staging depth ===
EXPERIMENTS+=("nes3_nwb|-DNUM_EPI_STAGES=3 -DNO_POST_STORE_BAR=1")
EXPERIMENTS+=("nes4_nwb|-DNUM_EPI_STAGES=4 -DNO_POST_STORE_BAR=1")

log "Total experiments: ${#EXPERIMENTS[@]}"

run_one() {
    local label="$1" dflags="$2"

    log "  [$label] DFLAGS='$dflags'"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  make fc2-w3 DFLAGS=\"$dflags\" && ./fc2-w3"
        return 0
    fi

    if ! make fc2-w3 DFLAGS="$dflags" > "$OUTDIR/${label}_build.log" 2>&1; then
        log "  [$label] BUILD FAILED"
        echo "@@RESULT ms=ERR label=$label dflags=\"$dflags\"" >> "$OUTDIR/results.txt"
        return 1
    fi

    local regs
    regs=$(grep -o '[0-9]* registers' "$OUTDIR/${label}_build.log" | tail -1 | grep -o '[0-9]*')
    local bars
    bars=$(grep -o 'used [0-9]* barriers' "$OUTDIR/${label}_build.log" | tail -1 | grep -o '[0-9]*')

    local output
    if ! output=$(timeout 15 ./fc2-w3 2>&1); then
        log "  [$label] RUN FAILED/TIMEOUT"
        echo "@@RESULT ms=HANG label=$label regs=$regs bars=$bars dflags=\"$dflags\"" >> "$OUTDIR/results.txt"
        return 1
    fi

    echo "$output" > "$OUTDIR/${label}.txt"

    local result_line
    result_line=$(echo "$output" | grep '@@RESULT' | head -1)
    if [ -n "$result_line" ]; then
        echo "${result_line} label=${label} regs=${regs} bars=${bars} dflags=\"${dflags}\"" >> "$OUTDIR/results.txt"
        local ms valid
        ms=$(echo "$result_line" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
        valid=$(echo "$result_line" | grep -o 'valid=[01]' | cut -d= -f2)
        log "  [$label] ${ms}ms  valid=${valid}  regs=${regs}  bars=${bars}"
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
    echo "label | ms | valid | regs | bars | tflops" > "$OUTDIR/summary.txt"
    echo "------|-----|-------|------|------|-------" >> "$OUTDIR/summary.txt"
    while IFS= read -r line; do
        ms=$(echo "$line" | grep -o 'ms=[^ ]*' | cut -d= -f2)
        valid=$(echo "$line" | grep -o 'valid=[^ ]*' | cut -d= -f2)
        regs=$(echo "$line" | grep -o 'regs=[^ ]*' | cut -d= -f2)
        bars=$(echo "$line" | grep -o 'bars=[^ ]*' | cut -d= -f2)
        tflops=$(echo "$line" | grep -o 'tflops=[^ ]*' | cut -d= -f2)
        label=$(echo "$line" | grep -o 'label=[^ ]*' | cut -d= -f2)
        printf "%-25s  %8s  valid=%-1s  regs=%-3s  bars=%-1s  tflops=%s\n" "$label" "${ms}ms" "$valid" "$regs" "$bars" "$tflops"
    done < <(sort -t= -k2 -n "$OUTDIR/results.txt") | tee -a "$OUTDIR/summary.txt"
    log ""
    log "Results: $OUTDIR/results.txt"
    log "Summary: $OUTDIR/summary.txt"
fi

log "Done."
