#!/usr/bin/env bash
#
# Structurally-different tile-swizzle sweep on fc2_w3x (bias-only, B200).
#
# Tests non-dgswizzle dispatches (TILE_DISPATCH=9..21 from tile_dispatch.cuh):
#   zorder / hilbert / zigzag / rowmajor / ncycle / nflat / nsnake /
#   nlock / checkered / dgsnake / ncyrot.
#
# Phase 1: build + run each variant REPS times, capture @@RESULT ms.
# Phase 2: print summary (min/mean/max ms, PASS count, Δ vs baseline).
#
# Outputs land under data/fc2_w3x_tile_sweep_<timestamp>/ so re-runs are safe.
# All variants pre-verified bijective on FC2 shape (TM=3626, TN=3, NC=74)
# via /tmp/tile_dispatch_check.py.
#
# Usage:
#   tools/sweep_fc2_w3x_tiles.sh            # REPS=5, default outdir
#   REPS=10 tools/sweep_fc2_w3x_tiles.sh
#   tools/sweep_fc2_w3x_tiles.sh my_outdir
#

set -u
cd "$(dirname "$0")/.."

OUT=${1:-"data/fc2_w3x_tile_sweep_$(date +%Y%m%d_%H%M%S)"}
REPS=${REPS:-5}
NVCC=${NVCC:-nvcc}
CFLAGS='-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -lineinfo --ptxas-options=-v --cudart=static'
LDFLAGS='-lcurand_static -lculibos -lcuda'

mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/run.log"; }

# name : TILE_DISPATCH value
VARIANTS=(
    "baseline:"
    "zorder:9"
    "hilbert:10"
    "zigzag:11"
    "rowmajor:13"
    "ncycle:14"
    "nflat:15"
    "nsnake:16"
    "nlock:17"
    "checkered:18"
    "dgsnake:19"
    "ncyrot:21"
)

build_one() {
    local name="$1" td="$2" bin="$3"
    local dflag=""
    [[ -n "$td" ]] && dflag="-DTILE_DISPATCH=$td"
    log "build $name ${dflag:-(dgswizzle)}"
    $NVCC $CFLAGS $dflag fc2_w3x.cu -o "$bin" $LDFLAGS \
        > "$OUT/build-$name.log" 2>&1
    local rc=$?
    if (( rc != 0 )); then
        log "  FAIL build $name rc=$rc  (see $OUT/build-$name.log)"
        return 1
    fi
    return 0
}

run_reps() {
    local name="$1" bin="$2" n="$3"
    log "run $name × $n"
    for i in $(seq 1 "$n"); do
        if ! "$bin" > "$OUT/run-$name-$i.log" 2>&1; then
            log "  rep $i FAIL"
        fi
    done
}

log "=== fc2_w3x tile-swizzle sweep: $REPS reps × ${#VARIANTS[@]} variants → $OUT ==="
for entry in "${VARIANTS[@]}"; do
    IFS=: read -r name td <<< "$entry"
    bin="$OUT/fc2-w3x-tile-$name"
    build_one "$name" "$td" "$bin" || continue
    run_reps "$name" "$bin" "$REPS"
done

log ""
log "=== Summary ==="

summary_file="$OUT/summary.txt"
{
    printf "=== fc2_w3x structurally-different tile-swizzle sweep ===\n"
    printf "outdir: %s\n" "$OUT"
    printf "reps:   %d\n" "$REPS"
    printf "\n"
    printf "Wall-time (REPS=%d)\n" "$REPS"
    printf "  %-12s  %8s  %8s  %8s  %12s  %5s  %5s\n" \
        variant min_ms mean_ms max_ms dms_vs_base pass_n tot_n
    base_mean=
    for entry in "${VARIANTS[@]}"; do
        IFS=: read -r name _ <<< "$entry"
        ms_vals=$(grep -h '@@RESULT' "$OUT/run-$name-"*.log 2>/dev/null \
                  | awk -F'ms=' '{print $2}' | awk '{print $1}')
        if [[ -z "$ms_vals" ]]; then
            printf "  %-12s  %-s\n" "$name" "NO DATA"
            continue
        fi
        mn=$(echo "$ms_vals" | awk 'BEGIN{m=1e9}{if($1<m)m=$1}END{printf "%.4f",m}')
        mx=$(echo "$ms_vals" | awk 'BEGIN{m=0}   {if($1>m)m=$1}END{printf "%.4f",m}')
        mu=$(echo "$ms_vals" | awk '{s+=$1;n++}END{if(n)printf "%.4f",s/n; else print "nan"}')
        np=$(grep -l 'PASS' "$OUT/run-$name-"*.log 2>/dev/null | wc -l)
        nt=$(ls "$OUT/run-$name-"*.log 2>/dev/null | wc -l)
        if [[ -z "$base_mean" ]]; then
            base_mean="$mu"
            delta="0.0000"
        else
            delta=$(awk -v a="$mu" -v b="$base_mean" 'BEGIN{printf "%+.4f", a-b}')
        fi
        printf "  %-12s  %8s  %8s  %8s  %12s  %5d  %5d\n" \
            "$name" "$mn" "$mu" "$mx" "$delta" "$np" "$nt"
    done

    printf "\nNotes:\n"
    printf "  • Bijection pre-verified: /tmp/tile_dispatch_check.py returns\n"
    printf "    10878/10878 unique for every TD on FC2 shape.\n"
    printf "  • baseline = no -DTILE_DISPATCH (uses dgswizzle G=8).\n"
    printf "  • Raw logs: $OUT/run-<variant>-<rep>.log\n"
} | tee "$summary_file"

log ""
log "summary written to $summary_file"
