#!/usr/bin/env bash
#
# Cluster-axis sweep on fc2_w3x — runs the three CLUSTER_AXIS={0,1,2} variants
# interleaved on B200 to amortize session-correlated jitter (clock governor,
# L2 warm state, thermals).
#
# CLUSTER_AXIS labels which gridDim axis carries the 2-CTA cluster:
#   0 → __cluster_dims__(2,1,1), grid (SM_COUNT,1,1)   — current default
#   1 → __cluster_dims__(1,2,1), grid (1,SM_COUNT,1)
#   2 → __cluster_dims__(1,1,2), grid (1,1,SM_COUNT)
#
# SASS is byte-identical except for `S2R Rx, SR_CTAID.{X|Y|Z}`. Any wall
# difference is the GPU scheduler treating cluster axes asymmetrically.
#
# Usage:
#   tools/sweep_cluster_axis.sh                # REPS=10, interleaved
#   REPS=20 tools/sweep_cluster_axis.sh
#
# Output:
#   data/cluster_axis_<ts>/{build_caxN.log, fc2-w3x-caxN, run-caxN-<i>.log,
#                           summary.txt}

set -u
cd "$(dirname "$0")/.."

OUT=${OUT:-"data/cluster_axis_$(date +%Y%m%d_%H%M%S)"}
REPS=${REPS:-10}
AXES=${AXES:-"0 1 2"}
mkdir -p "$OUT"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/run.log"; }

log "=== cluster_axis sweep → $OUT  axes='$AXES'  reps=$REPS (interleaved) ==="

declare -A REGS_OF LINES_OF

for AX in $AXES; do
    LOG="$OUT/build_cax${AX}.log"
    BIN="$OUT/fc2-w3x-cax${AX}"
    SASS="$OUT/cax${AX}.sass"

    log "build CLUSTER_AXIS=$AX"
    if ! make -B fc2-w3x DFLAGS="-DCLUSTER_AXIS=$AX" >"$LOG" 2>&1; then
        log "  BUILD FAIL → $LOG"
        exit 1
    fi
    cp fc2-w3x "$BIN"
    cuobjdump --dump-sass "$BIN" > "$SASS" 2>&1
    REGS_OF[$AX]=$(grep -oP 'Used \K[0-9]+(?= registers)' "$LOG" | head -1)
    LINES_OF[$AX]=$(wc -l <"$SASS")
done

log "running interleaved: rep1{ax0,ax1,ax2}, rep2{...}, ..."

declare -A VALID_N TOT_N
for AX in $AXES; do VALID_N[$AX]=0; TOT_N[$AX]=0; done

for i in $(seq 1 "$REPS"); do
    for AX in $AXES; do
        BIN="$OUT/fc2-w3x-cax${AX}"
        RUNLOG="$OUT/run-cax${AX}-${i}.log"
        TOT_N[$AX]=$((TOT_N[$AX] + 1))
        if ! "$BIN" >"$RUNLOG" 2>&1; then
            log "  rep $i ax $AX: launch FAIL"
            continue
        fi
        result=$(grep '^@@RESULT' "$RUNLOG" | tail -1)
        valid=$(echo "$result" | grep -oP 'valid=\K[0-9]+')
        ms=$(echo "$result" | grep -oP 'ms=\K[0-9.]+')
        if [ "${valid:-0}" = "1" ]; then
            VALID_N[$AX]=$((VALID_N[$AX] + 1))
        fi
        log "  rep $i ax $AX: ms=${ms:-?} valid=${valid:-?}"
    done
done

# ── summary ─────────────────────────────────────────────────────────────
SUM="$OUT/summary.txt"
{
    printf '=== fc2_w3x CLUSTER_AXIS sweep ===\n'
    printf 'outdir: %s\n' "$OUT"
    printf 'axes: %s   reps: %d (interleaved)\n\n' "$AXES" "$REPS"

    printf '%-4s %-7s %-9s %-9s %-9s %-9s %-9s %-7s\n' \
        "ax" "regs" "sass_ln" "min_ms" "mean_ms" "max_ms" "stdev_ms" "valid"

    base_mean=""
    for AX in $AXES; do
        ms_vals=$(grep -h '^@@RESULT' "$OUT"/run-cax${AX}-*.log 2>/dev/null \
                  | awk -F'ms=' '{print $2}' | awk '{print $1}')
        if [ -z "$ms_vals" ]; then
            printf '%-4s %-7s %-9s NO_DATA\n' "$AX" "${REGS_OF[$AX]:-?}" "${LINES_OF[$AX]:-?}"
            continue
        fi
        stats=$(echo "$ms_vals" | awk '
            BEGIN{n=0; mn=1e9; mx=0; s=0; ss=0}
            {n++; if($1<mn)mn=$1; if($1>mx)mx=$1; s+=$1; ss+=$1*$1}
            END{ if(n){ mean=s/n; var=(ss-n*mean*mean)/(n>1?n-1:1); sd=(var<0?0:sqrt(var)); printf "%.4f %.4f %.4f %.4f", mn, mean, mx, sd } }')
        v="${VALID_N[$AX]}/${TOT_N[$AX]}"
        printf '%-4s %-7s %-9s ' "$AX" "${REGS_OF[$AX]:-?}" "${LINES_OF[$AX]:-?}"
        printf '%s ' $stats
        printf '%-7s\n' "$v"
    done

    printf '\nNotes:\n'
    printf '  • Interleaved: rep i runs ax0, then ax1, then ax2, then loops.\n'
    printf '  • SASS diff is one line (SR_CTAID.X|Y|Z). Any wall delta is HW scheduler.\n'
    printf '  • Per-rep logs: %s/run-cax<N>-<i>.log\n' "$OUT"
} | tee "$SUM"

log ""
log "summary written to $SUM"
