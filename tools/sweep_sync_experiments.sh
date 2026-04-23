#!/usr/bin/env bash
# Time the 4 sync-experiment variants in pass-major interleaved order.
# Each variant is assumed pre-built at ./fc2-w3x-<name>.
set -u
cd "$(dirname "$0")/.."
OUT="data/sync_exp_$(date +%Y%m%d_%H%M%S)"
REPS=${REPS:-10}
mkdir -p "$OUT"

VARIANTS=(base trail wgread wgread-trail)

echo "=== sync-experiment sweep: REPS=$REPS × ${#VARIANTS[@]} variants → $OUT ===" | tee "$OUT/run.log"

for rep in $(seq 1 "$REPS"); do
    for v in "${VARIANTS[@]}"; do
        bin="./fc2-w3x-$v"
        log="$OUT/run-$v-$rep.log"
        if [[ ! -x "$bin" ]]; then
            echo "  [pass $rep] $v: binary missing" | tee -a "$OUT/run.log"
            continue
        fi
        "$bin" > "$log" 2>&1 || echo "  [pass $rep] $v FAIL" | tee -a "$OUT/run.log"
    done
done

echo "" | tee -a "$OUT/run.log"
echo "=== Summary ===" | tee -a "$OUT/run.log"
{
    printf "%-10s  %3s  %8s  %8s  %8s  %8s\n" variant n ms_med ms_p5 ms_p95 ms_sd
    for v in "${VARIANTS[@]}"; do
        # Digit-anchored glob so run-wgread-* doesn't also match run-wgread-trail-*
        ms_vals=$(grep -h '@@RESULT' "$OUT/run-$v-"[0-9]*.log 2>/dev/null \
                  | awk -F'ms=' '{print $2}' | awk '{print $1}')
        if [[ -z "$ms_vals" ]]; then
            printf "%-10s  NO DATA\n" "$v"; continue
        fi
        read n med p5 p95 sd < <(python3 -c "
import sys, math
xs = sorted(float(l) for l in sys.stdin if l.strip())
n = len(xs)
med = xs[n//2] if n % 2 else 0.5*(xs[n//2-1]+xs[n//2])
mu = sum(xs)/n
sd = math.sqrt(sum((x-mu)**2 for x in xs)/n) if n>1 else 0.0
p5 = xs[max(0, n//20)]
p95 = xs[min(n-1, (n*19)//20)]
print(f'{n} {med:.4f} {p5:.4f} {p95:.4f} {sd:.4f}')
" <<<"$ms_vals")
        printf "%-10s  %3d  %8s  %8s  %8s  %8s\n" "$v" "$n" "$med" "$p5" "$p95" "$sd"
    done

    # Delta vs base
    printf "\n"
    printf "%-10s  %+9s  %+7s\n" variant Δms_med Δ%
    base_med=$(grep -h '@@RESULT' "$OUT/run-base-"[0-9]*.log 2>/dev/null \
               | awk -F'ms=' '{print $2}' | awk '{print $1}' \
               | python3 -c "import sys; xs=sorted(float(l) for l in sys.stdin if l.strip()); n=len(xs); print(xs[n//2] if n%2 else 0.5*(xs[n//2-1]+xs[n//2]))")
    for v in "${VARIANTS[@]}"; do
        ms_vals=$(grep -h '@@RESULT' "$OUT/run-$v-"[0-9]*.log 2>/dev/null \
                  | awk -F'ms=' '{print $2}' | awk '{print $1}')
        [[ -z "$ms_vals" ]] && continue
        med=$(python3 -c "
import sys
xs=sorted(float(l) for l in sys.stdin if l.strip())
n=len(xs); print(xs[n//2] if n%2 else 0.5*(xs[n//2-1]+xs[n//2]))
" <<<"$ms_vals")
        awk -v v="$v" -v m="$med" -v b="$base_med" 'BEGIN{
            d=m-b; pct=100*d/b;
            printf "%-10s  %+9.4f  %+6.2f%%\n", v, d, pct
        }'
    done
} | tee "$OUT/summary.txt"

echo "summary → $OUT/summary.txt"
