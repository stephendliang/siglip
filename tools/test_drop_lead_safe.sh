#!/usr/bin/env bash
#
# test_drop_lead_safe.sh — paired-pass cycle comparison of fc2_w3x base
# vs SAFE_DROP_LEAD using the in-binary SWEEP harness.
#
# Both variants live in one binary as templated cells (gflip_blkswap with
# SAFE_DROP_LEAD=false, gflip_blkswap_safe with SAFE_DROP_LEAD=true).  The
# binary's own pass-major loop interleaves them every pass — same flow as
# tools/sweep_fc2_w3x_swizzle.sh, but tighter pairing (2 cells, no shell
# overhead, no CUDA-init-per-pass, no fork-per-launch).
#
# Output: data/drop_lead_safe_<ts>/
#   run.log         binary stdout + per-launch @@SAMPLE lines
#   wall_data.csv   variant,swizzle,rep,ms,cyc            (anova input)
#   compare.txt     anova_1way --paired rep --trim 0.33   (the verdict)
#
# Usage:
#   tools/test_drop_lead_safe.sh                     # REPS=512
#   REPS=2048  tools/test_drop_lead_safe.sh          # MODERATE-band
#   REPS=10978 tools/test_drop_lead_safe.sh          # TIE-band
#   tools/test_drop_lead_safe.sh data/my_outdir      # custom outdir

set -u
cd "$(dirname "$0")/.."

REPS=${REPS:-512}
OUT=${1:-data/drop_lead_safe_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
LOG="$OUT/run.log"
CSV="$OUT/wall_data.csv"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

log "=== drop_lead_safe in-binary paired test → $OUT (REPS=$REPS) ==="

log "--- build ---"
make fc2-w3x 2>&1 | tee -a "$LOG" | tail -3
[ -x ./fc2-w3x ] || { log "ERROR: build failed"; exit 1; }

log "--- run SWEEP=gflip_blkswap,gflip_blkswap_safe REPS=$REPS ---"
SWEEP=gflip_blkswap,gflip_blkswap_safe REPS=$REPS ./fc2-w3x 2>&1 | tee -a "$LOG"

log "--- extract @@SAMPLE → wall_data.csv ---"
{
    echo "variant,swizzle,rep,ms,cyc"
    grep -E '^@@SAMPLE pass=' "$LOG" | \
        sed -E 's/^@@SAMPLE pass=([0-9]+) variant=([^ ]+) ms=([0-9.]+) cyc=([0-9]+).*/\2,\2,\1,\3,\4/'
} > "$CSV"
n=$(( $(wc -l < "$CSV") - 1 ))
log "  $n samples"

log "--- anova_1way (paired rep, trim 0.33, metric cyc) ---"
python3 tools/anova_1way.py "$CSV" \
    --factor variant --metric cyc --paired rep --trim 0.33 \
    --out "$OUT/compare.txt" 2>&1 | tee -a "$LOG"

log ""
log "results:  $OUT/compare.txt"
log "raw csv:  $CSV"
