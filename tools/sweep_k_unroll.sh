#!/usr/bin/env bash
#
# K-loop unroll-factor sweep on fc2_w3x (SASS-only, no B200 needed).
#
# Builds with -DK_UNROLL=N for each divisor of K_ITERS (=24 at FC2 K=3072),
# dumps SASS via cuobjdump, counts MMA-warp scaffold ops + register usage +
# SASS line count.
#
# Background: K_UNROLL=1 (no unroll) shrinks SASS 9805→1645 lines and R2UR
# 495→32 but regresses B200 wall +200µs (lost cross-ki scheduling freedom).
# K_UNROLL=24 (full) is current default. Sweep tests intermediate factors
# 2/3/4/6/8/12 to look for a sweet spot.
#
# Usage:
#   tools/sweep_k_unroll.sh                # default factors, prints table
#   FACTORS="1 4 24" tools/sweep_k_unroll.sh
#
# Output: stdout table + per-build SASS dumps in data/k_unroll_<ts>/.

set -u
cd "$(dirname "$0")/.."

OUT=${OUT:-"data/k_unroll_$(date +%Y%m%d_%H%M%S)"}
FACTORS=${FACTORS:-"0 1 2 3 4 6 8 12 24"}
mkdir -p "$OUT"

declare -a OPS=(R2UR ELECT UTCQMMA UIADD3 UMOV LDS ULDS STS IMAD UTMASTG UTMALDG)

printf '%-8s %-7s %-9s' "unroll" "regs" "sass_ln"
for op in "${OPS[@]}"; do printf ' %-7s' "$op"; done
echo

for U in $FACTORS; do
    LOG="$OUT/build_u${U}.log"
    SASS="$OUT/fc2-w3x-u${U}.sass"
    BIN="$OUT/fc2-w3x-u${U}"

    if [ "$U" = "0" ]; then
        DFLAGS=""
        TAG="def"
    else
        DFLAGS="-DK_UNROLL=$U"
        TAG="$U"
    fi

    if ! make -B fc2-w3x DFLAGS="$DFLAGS" >"$LOG" 2>&1; then
        printf '%-8s BUILD FAILED (see %s)\n' "$TAG" "$LOG"
        continue
    fi

    cp fc2-w3x "$BIN"
    cuobjdump --dump-sass "$BIN" > "$SASS" 2>&1

    REGS=$(grep -oP 'Used \K[0-9]+(?= registers)' "$LOG" | head -1)
    LINES=$(wc -l <"$SASS")

    printf '%-8s %-7s %-9s' "$TAG" "${REGS:-?}" "$LINES"
    for op in "${OPS[@]}"; do
        printf ' %-7d' "$(grep -c "$op" "$SASS")"
    done
    echo
done

echo
echo "SASS dumps: $OUT/"
