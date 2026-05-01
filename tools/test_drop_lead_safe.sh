#!/usr/bin/env bash
#
# test_drop_lead_safe.sh — paired-pass cycle comparison of fc2_w3x baseline
# vs fc2_w3x with -DDROP_LEAD_BARSYNC_SAFE.
#
# Two binaries built once, then invoked alternately with REPS=1 per call so
# each pass interleaves them tightly — pass-major paired comparison the same
# way tools/sweep_fc2_w3x_swizzle.sh handles dispatch cells.  Order swaps
# every other pass to neutralize within-pass thermal/cache bias.
#
# Output: data/drop_lead_safe_<ts>/
#   wall_data.csv     variant,swizzle,rep,ms,cyc          (anova input)
#   compare.txt       anova_1way --paired rep --trim 0.33 (the verdict)
#   run.log           build + per-pass progress
#
# Usage:
#   tools/test_drop_lead_safe.sh                      # REPS=512 (~3 min)
#   REPS=2048 tools/test_drop_lead_safe.sh            # MODERATE-band (~12 min)
#   REPS=10978 tools/test_drop_lead_safe.sh           # TIE-band (~55 min)
#   tools/test_drop_lead_safe.sh data/my_outdir       # custom outdir
#
# Effect-size context: prior DROP_LEAD_BARSYNC opt-in (race-having) measured
# −0.27 µs at z=−3.49 in EPI_2WARP mode (n=128, 4-cell combo sweep).  The
# safe variant carries identical structural change plus mbar spin overhead
# in tid==0 — so this is the upper bound on the win.

set -u
cd "$(dirname "$0")/.."

REPS=${REPS:-512}
OUT=${1:-data/drop_lead_safe_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
LOG="$OUT/run.log"
CSV="$OUT/wall_data.csv"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

log "=== drop_lead_safe paired-pass test → $OUT (REPS=$REPS) ==="

log "--- build baseline ---"
make -B fc2-w3x 2>&1 | tee -a "$LOG" | tail -3
[ -x ./fc2-w3x ] || { log "ERROR: baseline build failed"; exit 1; }
mv ./fc2-w3x ./fc2-w3x-base

log "--- build DROP_LEAD_BARSYNC_SAFE ---"
make -B fc2-w3x DFLAGS='-DDROP_LEAD_BARSYNC_SAFE' 2>&1 | tee -a "$LOG" | tail -3
[ -x ./fc2-w3x ] || { log "ERROR: SAFE build failed"; exit 1; }
mv ./fc2-w3x ./fc2-w3x-safe

log "--- correctness check ---"
for v in base safe; do
    line=$(./fc2-w3x-"$v" 2>>"$LOG" | grep -E 'valid|FAIL' | head -1)
    log "  $v: $line"
    if ! echo "$line" | grep -q 'valid=1'; then
        log "ERROR: $v failed correctness"
        exit 1
    fi
done

echo "variant,swizzle,rep,ms,cyc" > "$CSV"

log "--- pass-major interleaved run ---"
for p in $(seq 1 "$REPS"); do
    if [ $((p % 2)) -eq 1 ]; then
        order=("base" "safe")
    else
        order=("safe" "base")
    fi
    for v in "${order[@]}"; do
        line=$(REPS=1 ./fc2-w3x-"$v" 2>>"$LOG" | grep -E "^@@SAMPLE" | head -1)
        ms=$(echo "$line"  | grep -oE 'ms=[0-9.]+'  | head -1 | cut -d= -f2)
        cyc=$(echo "$line" | grep -oE 'cyc=[0-9]+'  | head -1 | cut -d= -f2)
        if [ -z "${ms:-}" ] || [ -z "${cyc:-}" ]; then
            log "WARN pass=$p variant=$v: no @@SAMPLE captured"
            continue
        fi
        echo "$v,$v,$p,$ms,$cyc" >> "$CSV"
    done
    if [ $((p % 64)) -eq 0 ] || [ "$p" -eq "$REPS" ]; then
        log "  pass $p/$REPS"
    fi
done

log "--- anova_1way (paired rep, trim 0.33, metric cyc) ---"
python3 tools/anova_1way.py "$CSV" \
    --factor variant --metric cyc --paired rep --trim 0.33 \
    --out "$OUT/compare.txt" 2>&1 | tee -a "$LOG"

log ""
log "results:  $OUT/compare.txt"
log "raw csv:  $CSV"
log ""
log "interpretation:"
log "  AUC < 0.55 TIE         → spin overhead matched bar.sync; don't ship"
log "  AUC < 0.65 WEAK        → small win; not worth race-fix maintenance"
log "  AUC < 0.75 MODERATE    → real win; consider promoting to default"
log "  AUC ≥ 0.75 STRONG/DEC  → ship it"
