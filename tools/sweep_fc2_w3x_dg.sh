#!/usr/bin/env bash
#
# Dispatch-variant sweep on fc2_w3x (bias-only, B200-measured).
#
# Phase 1: build + run each dispatch variant REPS times, capture @@RESULT ms.
# Phase 2: build with -DPROFILE_W4 (where compatible) and capture one diagnostic
#          dump per variant.
# Phase 3: print summary table (min/mean/max ms, PASS count) + quick-reference
#          PROFILE_W4 per-variant overall lines.
#
# Outputs land under data/fc2_w3x_dg_sweep_<timestamp>/ so re-runs are safe.
#
# Usage:
#   tools/sweep_fc2_w3x_dg.sh            # REPS=5, default outdir
#   REPS=10 tools/sweep_fc2_w3x_dg.sh    # more reps
#   tools/sweep_fc2_w3x_dg.sh my_outdir  # custom outdir
#
# Skips: DG_GROUP_SIZE=32 under -DPROFILE_W4 because max in_g = 32*3 = 96
#        overflows the 6-bit in_g field in the per-tile pack (0..63).
#

set -u
cd "$(dirname "$0")/.."

OUT=${1:-"data/fc2_w3x_dg_sweep_$(date +%Y%m%d_%H%M%S)"}
REPS=${REPS:-5}
NVCC=${NVCC:-nvcc}
CFLAGS='-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -lineinfo --ptxas-options=-v --cudart=static'
LDFLAGS='-lcurand_static -lculibos -lcuda'

mkdir -p "$OUT"
log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/run.log"; }

# name : extra_dflags : supports_profile_w4 (1/0)
VARIANTS=(
    "baseline::1"
    "dg4:-DDG_GROUP_SIZE=4:1"
    "dg16:-DDG_GROUP_SIZE=16:1"
    "dg32:-DDG_GROUP_SIZE=32:0"
)

build_one() {
    local name="$1" flags="$2" extra="$3" out_bin="$4"
    log "build $name ${extra:+[$extra]}"
    $NVCC $CFLAGS $flags $extra fc2_w3x.cu -o "$out_bin" $LDFLAGS \
        > "$OUT/build-$name${extra:+-w4}.log" 2>&1
    local rc=$?
    if (( rc != 0 )); then
        log "  FAIL build $name rc=$rc  (see $OUT/build-$name${extra:+-w4}.log)"
        return 1
    fi
    return 0
}

run_reps() {
    local name="$1" bin="$2" n="$3"
    log "run $name × $n"
    for i in $(seq 1 "$n"); do
        if ! "$bin" > "$OUT/run-$name-$i.log" 2>&1; then
            log "  rep $i FAIL (kernel launch or validation)"
        fi
    done
}

# ── Phase 1: bare build + N-rep bench ────────────────────────────────────
log "=== Phase 1: bare build + $REPS reps/variant → $OUT ==="
for entry in "${VARIANTS[@]}"; do
    IFS=: read -r name flags _ <<< "$entry"
    bin="$OUT/fc2-w3x-$name"
    build_one "$name" "$flags" "" "$bin" || continue
    run_reps "$name" "$bin" "$REPS"
done

# ── Phase 2: PROFILE_W4 probe (1 rep per variant) ────────────────────────
log ""
log "=== Phase 2: PROFILE_W4 probe (1 rep, W4-compatible variants only) ==="
for entry in "${VARIANTS[@]}"; do
    IFS=: read -r name flags w4_ok <<< "$entry"
    if [[ "$w4_ok" != "1" ]]; then
        log "skip $name (PROFILE_W4 incompatible: in_g field too narrow for G*TN > 63)"
        continue
    fi
    bin="$OUT/fc2-w3x-$name-w4"
    build_one "$name" "$flags -DPROFILE_W4" "-w4" "$bin" || continue
    if ! "$bin" > "$OUT/run-$name-w4.log" 2>&1; then
        log "  w4 rep FAIL for $name"
    fi
done

# ── Phase 3: summary ──────────────────────────────────────────────────────
log ""
log "=== Phase 3: summary ==="

summary_file="$OUT/summary.txt"
{
    printf "=== fc2_w3x dispatch-variant sweep ===\n"
    printf "outdir: %s\n" "$OUT"
    printf "reps:   %d\n" "$REPS"
    printf "\n"
    printf "Phase 1: bare wall-time (REPS=%d)\n" "$REPS"
    printf "  %-10s  %8s  %8s  %8s  %8s  %5s  %5s\n" \
        variant min_ms mean_ms max_ms dms_vs_base pass_n tot_n
    base_mean=
    for entry in "${VARIANTS[@]}"; do
        IFS=: read -r name _ _ <<< "$entry"
        ms_vals=$(grep -h '@@RESULT' "$OUT/run-$name-"*.log 2>/dev/null \
                  | awk -F'ms=' '{print $2}' | awk '{print $1}')
        if [[ -z "$ms_vals" ]]; then
            printf "  %-10s  %-s\n" "$name" "NO DATA"
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
        printf "  %-10s  %8s  %8s  %8s  %8s  %5d  %5d\n" \
            "$name" "$mn" "$mu" "$mx" "$delta" "$np" "$nt"
    done

    printf "\nPhase 2: PROFILE_W4 overall (single rep each; W4-compatible only)\n"
    printf "  %-10s  %-s\n" "variant" "overall line"
    for entry in "${VARIANTS[@]}"; do
        IFS=: read -r name _ w4_ok <<< "$entry"
        [[ "$w4_ok" != "1" ]] && continue
        line=$(grep -h '^  overall' "$OUT/run-$name-w4.log" 2>/dev/null | head -1)
        [[ -z "$line" ]] && line="(no PROFILE_W4 output)"
        printf "  %-10s  %s\n" "$name" "$line"
    done

    printf "\nPhase 2b: PROFILE_W4 by-tn (W4-compatible only)\n"
    for entry in "${VARIANTS[@]}"; do
        IFS=: read -r name _ w4_ok <<< "$entry"
        [[ "$w4_ok" != "1" ]] && continue
        printf "  -- %s --\n" "$name"
        awk '/by N-col \(aggregated/,/^$/' "$OUT/run-$name-w4.log" 2>/dev/null \
            | sed 's/^/    /'
    done

    printf "\nNotes:\n"
    printf "  • DG_GROUP_SIZE=32 skipped under PROFILE_W4 (in_g overflow).\n"
    printf "  • Raw logs: $OUT/run-<variant>-<rep>.log (Phase 1),\n"
    printf "             $OUT/run-<variant>-w4.log (Phase 2).\n"
} | tee "$summary_file"

log ""
log "summary written to $summary_file"
