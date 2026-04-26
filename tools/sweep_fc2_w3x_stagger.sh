#!/usr/bin/env bash
#
# STAGGER × dispatch cross sweep for fc2_w3x.
#
# Tests STAGGER=0 (baseline shared-mbar A+B in W4) vs STAGGER=2 (split-mbar:
# separate mbar for A and B, W5 dual-waits). Wall-equivalent in expectation
# (W5's max(A_arrival, B_arrival) is unchanged), but TMA-issue ordering /
# cluster-multicast paths / ptxas scheduling differ — cross-tested with the
# 11 non-pathological dispatches to surface any STAGGER × dispatch
# interaction.
#
# Built with -DPROFILE_CYCLES so per-warp/per-phase brackets are recorded;
# aggregator produces a single prof_report.txt.
#
# Output: data/fc2_w3x_stagger_<ts>/
#   build-<variant>.log
#   run-<variant>-<rep>.log
#   prof_report.txt
#
# Usage:
#   tools/sweep_fc2_w3x_stagger.sh           # REPS=6, default outdir
#   REPS=10 tools/sweep_fc2_w3x_stagger.sh
#   tools/sweep_fc2_w3x_stagger.sh my_outdir
#

set -u
cd "$(dirname "$0")/.."

OUT=${1:-"data/fc2_w3x_stagger_$(date +%Y%m%d_%H%M%S)"}
REPS=${REPS:-6}
NVCC=${NVCC:-nvcc}
CFLAGS='-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -lineinfo --ptxas-options=-v --cudart=static'
LDFLAGS='-lcurand_static -lculibos -lcuda'

mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/run.log"; }

# 11 non-pathological dispatch flags (skip ncycle/nflat/nsnake/nlock/ncyrot;
# already +47-186 µs regressions, no point crossing).
DISPATCHES=(
    "dgsw:"
    "dg4:-DDG_GROUP_SIZE=4"
    "dg16:-DDG_GROUP_SIZE=16"
    "dg32:-DDG_GROUP_SIZE=32"
    "hilbert:-DTILE_DISPATCH=10"
    "zigzag:-DTILE_DISPATCH=11"
    "rowmajor:-DTILE_DISPATCH=13"
    "checkered:-DTILE_DISPATCH=18"
    "dgsnake:-DTILE_DISPATCH=19"
    "gflip:-DTILE_DISPATCH=33"
    "tn2br:-DTILE_DISPATCH=34"
)

STAGGERS=(0 2)

# Build full variant list as the cross product.
VARIANTS=()
for s in "${STAGGERS[@]}"; do
    for entry in "${DISPATCHES[@]}"; do
        IFS=: read -r dname dflags <<< "$entry"
        VARIANTS+=("${dname}_s${s}:-DSTAGGER=${s} ${dflags}")
    done
done

log "=== Phase 1: building ${#VARIANTS[@]} variants with -DPROFILE_CYCLES ==="
for entry in "${VARIANTS[@]}"; do
    IFS=: read -r name extra <<< "$entry"
    bin="$OUT/fc2-w3x-stagger-$name"
    log "build $name  flags='${extra}'"
    if ! $NVCC $CFLAGS -DPROFILE_CYCLES $extra fc2_w3x.cu -o "$bin" $LDFLAGS \
            > "$OUT/build-$name.log" 2>&1; then
        log "  FAIL build $name  (see $OUT/build-$name.log)"
    fi
done

log ""
log "=== Phase 2: $REPS reps/variant, pass-major interleaving ==="
for rep in $(seq 1 "$REPS"); do
    log "-- pass $rep/$REPS --"
    for entry in "${VARIANTS[@]}"; do
        IFS=: read -r name _ <<< "$entry"
        bin="$OUT/fc2-w3x-stagger-$name"
        if [[ ! -x "$bin" ]]; then
            continue
        fi
        if ! "$bin" > "$OUT/run-$name-$rep.log" 2>&1; then
            log "  pass $rep $name FAIL"
        fi
    done
done

log ""
log "=== Phase 3: aggregating → $OUT/prof_report.txt ==="
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
