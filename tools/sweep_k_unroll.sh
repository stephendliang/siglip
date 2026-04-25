#!/usr/bin/env bash
#
# K-loop unroll-factor sweep on fc2_w3x (B200 wall + SASS).
#
# Builds with -DK_UNROLL=N for each factor in FACTORS (0 = no pragma, compiler
# default). For each build:
#   • dumps SASS via cuobjdump, counts MMA-warp scaffold ops
#   • records ptxas register usage and SASS line count
#   • runs the binary REPS times on B200, captures @@RESULT ms + valid=
#
# Background: K_UNROLL=1 (no unroll) shrinks SASS 9805→1645 lines and R2UR
# 495→32 but regresses B200 wall +200µs (lost cross-ki scheduling freedom).
# K_UNROLL=24 (full) is current default. Sweep tests intermediate factors
# 2/3/4/6/8/12 to look for a sweet spot.
#
# Usage:
#   tools/sweep_k_unroll.sh                       # default factors, REPS=3
#   FACTORS="1 6 24" REPS=5 tools/sweep_k_unroll.sh
#   NO_RUN=1 tools/sweep_k_unroll.sh              # SASS-only (e.g. cross-compile)
#
# Output:
#   data/k_unroll_<ts>/build_u<N>.log             # nvcc log (registers etc.)
#   data/k_unroll_<ts>/fc2-w3x-u<N>               # binary
#   data/k_unroll_<ts>/fc2-w3x-u<N>.sass          # SASS dump
#   data/k_unroll_<ts>/run-u<N>-<rep>.log         # per-run stdout (with @@RESULT)
#   data/k_unroll_<ts>/summary.txt                # combined SASS+wall table
#
# Exits non-zero only on usage errors. Per-build/per-run failures are logged
# and reflected in the table (BUILD FAIL / RUN FAIL / valid=0).

set -u
cd "$(dirname "$0")/.."

OUT=${OUT:-"data/k_unroll_$(date +%Y%m%d_%H%M%S)"}
FACTORS=${FACTORS:-"0 1 2 3 4 6 8 12 24"}
REPS=${REPS:-3}
NO_RUN=${NO_RUN:-0}
mkdir -p "$OUT"

declare -a OPS=(R2UR ELECT UTCQMMA UIADD3 UMOV LDS ULDS STS IMAD UTMASTG UTMALDG)

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/run.log"; }

log "=== k_unroll sweep → $OUT  factors='$FACTORS'  reps=$REPS  no_run=$NO_RUN ==="

declare -A WALL_MIN WALL_MEAN WALL_MAX VALID_N TOT_N STATUS REGS_OF LINES_OF
declare -A OP_COUNTS

for U in $FACTORS; do
    LOG="$OUT/build_u${U}.log"
    SASS="$OUT/fc2-w3x-u${U}.sass"
    BIN="$OUT/fc2-w3x-u${U}"

    if [ "$U" = "0" ]; then
        DFLAGS=""
        TAG="def"
    else
        DFLAGS="-DK_UNROLL=$U"
        TAG="u$U"
    fi

    log "build $TAG (DFLAGS='$DFLAGS')"
    if ! make -B fc2-w3x DFLAGS="$DFLAGS" >"$LOG" 2>&1; then
        log "  BUILD FAIL → $LOG"
        STATUS[$TAG]="BUILD_FAIL"
        continue
    fi

    cp fc2-w3x "$BIN"
    cuobjdump --dump-sass "$BIN" > "$SASS" 2>&1

    REGS_OF[$TAG]=$(grep -oP 'Used \K[0-9]+(?= registers)' "$LOG" | head -1)
    LINES_OF[$TAG]=$(wc -l <"$SASS")
    for op in "${OPS[@]}"; do
        OP_COUNTS["${TAG}__${op}"]=$(grep -c "$op" "$SASS")
    done

    if [ "$NO_RUN" = "1" ]; then
        STATUS[$TAG]="SASS_ONLY"
        continue
    fi

    log "run $TAG × $REPS"
    pass=0
    total=0
    for i in $(seq 1 "$REPS"); do
        RUNLOG="$OUT/run-${TAG}-${i}.log"
        total=$((total + 1))
        if ! "$BIN" >"$RUNLOG" 2>&1; then
            log "  rep $i: launch FAIL → $RUNLOG"
            continue
        fi
        result=$(grep '^@@RESULT' "$RUNLOG" | tail -1)
        if [ -z "$result" ]; then
            log "  rep $i: no @@RESULT line"
            continue
        fi
        valid=$(echo "$result" | grep -oP 'valid=\K[0-9]+')
        if [ "${valid:-0}" = "1" ]; then
            pass=$((pass + 1))
        else
            log "  rep $i: valid=$valid"
        fi
    done
    VALID_N[$TAG]="$pass"
    TOT_N[$TAG]="$total"

    ms_vals=$(grep -h '^@@RESULT' "$OUT"/run-${TAG}-*.log 2>/dev/null \
              | awk -F'ms=' '{print $2}' | awk '{print $1}')
    if [ -n "$ms_vals" ]; then
        WALL_MIN[$TAG]=$(echo "$ms_vals" | awk 'BEGIN{m=1e9}{if($1<m)m=$1}END{printf "%.4f",m}')
        WALL_MAX[$TAG]=$(echo "$ms_vals" | awk 'BEGIN{m=0}   {if($1>m)m=$1}END{printf "%.4f",m}')
        WALL_MEAN[$TAG]=$(echo "$ms_vals" | awk '{s+=$1;n++}END{if(n)printf "%.4f",s/n; else print "nan"}')
        STATUS[$TAG]="OK"
    else
        STATUS[$TAG]="RUN_FAIL"
    fi
done

# ── summary ─────────────────────────────────────────────────────────────
SUM="$OUT/summary.txt"
{
    printf '=== fc2_w3x K_UNROLL sweep ===\n'
    printf 'outdir: %s\n' "$OUT"
    printf 'factors: %s\n' "$FACTORS"
    printf 'reps: %d  no_run: %s\n\n' "$REPS" "$NO_RUN"

    printf '%-6s %-7s %-9s %-8s %-8s %-8s %-9s %-7s %-10s' \
        "tag" "regs" "sass_ln" "min_ms" "mean_ms" "max_ms" "Δvs_def" "valid" "status"
    for op in "${OPS[@]}"; do printf ' %-7s' "$op"; done
    echo

    base_mean=""
    for U in $FACTORS; do
        if [ "$U" = "0" ]; then TAG="def"; else TAG="u$U"; fi
        st=${STATUS[$TAG]:-UNKNOWN}
        regs=${REGS_OF[$TAG]:-?}
        ln=${LINES_OF[$TAG]:-?}
        mn=${WALL_MIN[$TAG]:-—}
        mu=${WALL_MEAN[$TAG]:-—}
        mx=${WALL_MAX[$TAG]:-—}
        valid="${VALID_N[$TAG]:-0}/${TOT_N[$TAG]:-0}"

        if [ "$TAG" = "def" ] && [ "$st" = "OK" ]; then
            base_mean="$mu"
            delta="0.0000"
        elif [ -n "$base_mean" ] && [ "$st" = "OK" ]; then
            delta=$(awk -v a="$mu" -v b="$base_mean" 'BEGIN{printf "%+.4f", a-b}')
        else
            delta="—"
        fi

        printf '%-6s %-7s %-9s %-8s %-8s %-8s %-9s %-7s %-10s' \
            "$TAG" "$regs" "$ln" "$mn" "$mu" "$mx" "$delta" "$valid" "$st"
        for op in "${OPS[@]}"; do
            printf ' %-7s' "${OP_COUNTS[${TAG}__${op}]:-?}"
        done
        echo
    done

    printf '\nNotes:\n'
    printf '  • def = no -DK_UNROLL (compiler default; full unroll of K_ITERS=%s)\n' '24 at FC2 K=3072'
    printf '  • Δvs_def is mean_ms − def mean_ms; positive = slower than default\n'
    printf '  • valid is N_pass/N_total of @@RESULT lines with valid=1\n'
    printf '  • Per-rep logs: %s/run-<tag>-<i>.log\n' "$OUT"
    printf '  • SASS dumps:   %s/fc2-w3x-u<N>.sass\n' "$OUT"
} | tee "$SUM"

log ""
log "summary written to $SUM"
