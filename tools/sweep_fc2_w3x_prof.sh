#!/usr/bin/env bash
#
# Per-phase profile sweep across all fc2_w3x dispatch variants.
#
# Every variant is built with -DPROFILE_CYCLES (multi-launch atomicAdd
# accumulation; clock64 wall bracket).  Cycle numbers are clock-invariant
# by construction, so clock throttling between reps does not corrupt the
# per-phase comparison.
#
# Reps are *interleaved* across variants (pass-major ordering), so thermal
# drift is shared across variants rather than concentrated on one.
#
# Output: data/fc2_w3x_prof_<ts>/
#   build-<variant>.log
#   run-<variant>-<rep>.log      (raw per-run output, with @@RESULT + @@PROF)
#   prof_report.txt              (SINGLE-FILE aggregated report — the point)
#
# Usage:
#   tools/sweep_fc2_w3x_prof.sh            # REPS=5, default outdir
#   REPS=10 tools/sweep_fc2_w3x_prof.sh
#   tools/sweep_fc2_w3x_prof.sh my_outdir
#

set -u
cd "$(dirname "$0")/.."

OUT=${1:-"data/fc2_w3x_prof_$(date +%Y%m%d_%H%M%S)"}
REPS=${REPS:-5}
NVCC=${NVCC:-nvcc}
CFLAGS='-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -lineinfo --ptxas-options=-v --cudart=static'
LDFLAGS='-lcurand_static -lculibos -lcuda'

mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/run.log"; }

# name : extra build flags
# Order: dgswizzle family first, then structurally-different swizzles.
VARIANTS=(
    "baseline:"
    "dg4:-DDG_GROUP_SIZE=4"
    "dg16:-DDG_GROUP_SIZE=16"
    "dg32:-DDG_GROUP_SIZE=32"
    "zorder:-DTILE_DISPATCH=9"
    "hilbert:-DTILE_DISPATCH=10"
    "zigzag:-DTILE_DISPATCH=11"
    "rowmajor:-DTILE_DISPATCH=13"
    "ncycle:-DTILE_DISPATCH=14"
    "nflat:-DTILE_DISPATCH=15"
    "nsnake:-DTILE_DISPATCH=16"
    "nlock:-DTILE_DISPATCH=17"
    "checkered:-DTILE_DISPATCH=18"
    "dgsnake:-DTILE_DISPATCH=19"
    "ncyrot:-DTILE_DISPATCH=21"
    "gflip:-DTILE_DISPATCH=33"
    "tn2br:-DTILE_DISPATCH=34"
)

# ── Phase 1: build all variants ─────────────────────────────────────────
log "=== Phase 1: building ${#VARIANTS[@]} variants with -DPROFILE_CYCLES ==="
for entry in "${VARIANTS[@]}"; do
    IFS=: read -r name extra <<< "$entry"
    bin="$OUT/fc2-w3x-prof-$name"
    log "build $name ${extra:-(dgswizzle G=8)}"
    if ! $NVCC $CFLAGS -DPROFILE_CYCLES $extra fc2_w3x.cu -o "$bin" $LDFLAGS \
            > "$OUT/build-$name.log" 2>&1; then
        log "  FAIL build $name  (see $OUT/build-$name.log)"
    fi
done

# ── Phase 2: interleaved reps ───────────────────────────────────────────
log ""
log "=== Phase 2: $REPS reps/variant, pass-major interleaving ==="
for rep in $(seq 1 "$REPS"); do
    log "-- pass $rep/$REPS --"
    for entry in "${VARIANTS[@]}"; do
        IFS=: read -r name _ <<< "$entry"
        bin="$OUT/fc2-w3x-prof-$name"
        if [[ ! -x "$bin" ]]; then
            continue
        fi
        if ! "$bin" > "$OUT/run-$name-$rep.log" 2>&1; then
            log "  pass $rep $name FAIL"
        fi
    done
done

# ── Phase 3: aggregate ──────────────────────────────────────────────────
log ""
log "=== Phase 3: aggregating → $OUT/prof_report.txt ==="

# Preserve the VARIANTS order in the aggregator so columns line up sensibly.
VARIANT_NAMES=()
for entry in "${VARIANTS[@]}"; do
    IFS=: read -r name _ <<< "$entry"
    VARIANT_NAMES+=("$name")
done

python3 tools/aggregate_prof.py "$OUT" \
    --out "$OUT/prof_report.txt" \
    --variants "${VARIANT_NAMES[@]}"

log ""
log "single-file report: $OUT/prof_report.txt"
log "raw per-run logs:   $OUT/run-<variant>-<rep>.log"
