#!/usr/bin/env bash
# sweep_epi.sh — 4-combo sweep of fc2_w3x.cu epilogue knobs.
#
# Axes:
#   swizzle ∈ {off (SWIZZLE_NONE + XOR=0), on (SWIZZLE_64B + XOR)}
#   ns_epi  ∈ {2, 4}
#
# Each of the 4 variants runs REPS times. Reports mean/min/max/std wall-time,
# pass rate, and per-variant log. All 4 builds use `make -B` so stale objects
# never contaminate results. Run this after bench/tma_swizzle_probe.cu PASSes
# both baseline and candidate modes — otherwise swizzle variants will FAIL
# correctness and the timing numbers will be meaningless.
#
# Usage:
#   ./tools/sweep_epi.sh                       # default 5 reps
#   REPS=10 ./tools/sweep_epi.sh               # 10 reps each (more stable)
#   ./tools/sweep_epi.sh --probe-first         # runs the probe binaries before
#                                              # the sweep; aborts on PROBE FAIL
#
# Output: data/sweep_epi_<timestamp>/
#   summary.txt   — one line per variant, human-readable
#   results.csv   — raw per-run ms, for downstream analysis
#   <variant>.log — full stdout from each variant

set -u
cd "$(dirname "$0")/.."

REPS="${REPS:-5}"
RUN_PROBE=0
if [ "${1:-}" = "--probe-first" ]; then
    RUN_PROBE=1
fi

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="data/sweep_epi_${STAMP}"
mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/results.csv"
SUMMARY="$OUT_DIR/summary.txt"
echo "variant,swizzle,ns_epi,rep,ms,tflops,valid" > "$CSV"

variants=(
    "fc2-w3x         off 2"
    "fc2-w3x-ns4     off 4"
    "fc2-w3x-sw      on  2"
    "fc2-w3x-sw-ns4  on  4"
)

if [ "$RUN_PROBE" = "1" ]; then
    echo "=== probe phase ==="
    for pt in tma-swizzle-probe tma-swizzle-probe-sw; do
        make -B "$pt" > "$OUT_DIR/probe_${pt}.build.log" 2>&1 || {
            echo "  BUILD FAIL: $pt (see $OUT_DIR/probe_${pt}.build.log)"
            exit 2
        }
        "./$pt" > "$OUT_DIR/probe_${pt}.log" 2>&1
        rc=$?
        if [ $rc -eq 0 ]; then
            echo "  PROBE PASS: $pt"
        else
            echo "  PROBE FAIL: $pt  (rc=$rc; see $OUT_DIR/probe_${pt}.log)"
            echo "  Aborting sweep — fix the XOR formula before timing."
            exit 3
        fi
    done
    echo
fi

echo "=== sweep phase ($REPS reps/variant) ==="
printf "%-20s %-10s %-8s %-10s %-10s %-10s %-10s %-8s\n" \
       variant swizzle ns_epi mean_ms min_ms max_ms std_ms pass%
printf '%.0s─' {1..94}; echo

: > "$SUMMARY"

for row in "${variants[@]}"; do
    read -r target swiz nsepi <<<"$row"
    log="$OUT_DIR/${target}.log"

    make -B "$target" > "$OUT_DIR/${target}.build.log" 2>&1 || {
        printf "%-20s BUILD FAIL (see %s)\n" "$target" "$OUT_DIR/${target}.build.log"
        continue
    }

    : > "$log"
    mss=()
    passes=0
    for ((r = 1; r <= REPS; r++)); do
        out=$("./$target" 2>&1)
        echo "--- rep $r ---"  >> "$log"
        echo "$out"             >> "$log"
        line=$(echo "$out" | grep -m1 '^@@RESULT')
        ms=$(   echo "$line" | sed -E 's/.*ms=([0-9.]+).*/\1/')
        tf=$(   echo "$line" | sed -E 's/.*tflops=([0-9.]+).*/\1/')
        valid=$(echo "$line" | sed -E 's/.*valid=([0-9]+).*/\1/')
        [ -z "$ms" ]    && ms=0
        [ -z "$valid" ] && valid=0
        echo "${target},${swiz},${nsepi},${r},${ms},${tf},${valid}" >> "$CSV"
        mss+=("$ms")
        [ "$valid" = "1" ] && passes=$((passes + 1))
    done

    # Mean / min / max / std across REPS
    stats=$(python3 -c "
import sys, math
xs = [float(x) for x in sys.argv[1:] if x and float(x) > 0]
if not xs:
    print('nan nan nan nan'); sys.exit(0)
m = sum(xs)/len(xs)
v = sum((x-m)**2 for x in xs) / max(1, len(xs)-1)
s = math.sqrt(v)
print(f'{m:.4f} {min(xs):.4f} {max(xs):.4f} {s:.4f}')
" "${mss[@]}")
    read -r mean_ms min_ms max_ms std_ms <<<"$stats"
    pass_pct=$(( passes * 100 / REPS ))

    line=$(printf "%-20s %-10s %-8s %-10s %-10s %-10s %-10s %-3d%%" \
                  "$target" "$swiz" "$nsepi" "$mean_ms" "$min_ms" "$max_ms" "$std_ms" "$pass_pct")
    echo "$line"
    echo "$line" >> "$SUMMARY"
done

echo
echo "Results: $OUT_DIR"
echo "Summary: $SUMMARY"
echo "CSV:     $CSV"
