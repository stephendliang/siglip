#!/bin/bash
# Loop-test fc2_w3x with the trailing bar.sync removed. Compares correctness
# (valid=1 across all runs) and mean ms to the baseline with the bar.sync in.
#
# Usage: tools/test_drop_barsync.sh [N]   (default N=20)

set -eu
cd "$(dirname "$0")/.."

N=${1:-20}

parse_one() {
    local line ms valid
    line=$(./fc2-w3x 2>&1 | grep '@@RESULT' || true)
    if [ -z "$line" ]; then echo "0.0 ERR"; return; fi
    ms=$(echo "$line"    | sed -n 's/.*ms=\([0-9.][0-9.]*\).*/\1/p')
    valid=$(echo "$line" | sed -n 's/.*valid=\([0-9][0-9]*\).*/\1/p')
    echo "$ms $valid"
}

run_n() {
    local label=$1 n=$2
    local pass=0 fail=0 err=0
    local sum=0 mn=99999 mx=0
    echo "[$label] N=$n"
    for i in $(seq 1 $n); do
        read ms valid <<< "$(parse_one)"
        if   [ "$valid" = "1" ]; then pass=$((pass+1))
        elif [ "$valid" = "0" ]; then
            fail=$((fail+1))
            echo "  FAIL iter $i  ms=$ms"
        else
            err=$((err+1))
            echo "  ERR iter $i (no @@RESULT line)"
        fi
        sum=$(awk -v s="$sum" -v m="$ms" 'BEGIN{printf "%.6f", s+m}')
        mn=$(awk  -v a="$mn"  -v b="$ms" 'BEGIN{printf "%.6f", (b<a)?b:a}')
        mx=$(awk  -v a="$mx"  -v b="$ms" 'BEGIN{printf "%.6f", (b>a)?b:a}')
        printf "."
    done
    echo ""
    local mean=$(awk -v s="$sum" -v n="$n" 'BEGIN{printf "%.4f", s/n}')
    echo "  pass=$pass/$n  fail=$fail  err=$err  mean=${mean}ms  min=${mn}ms  max=${mx}ms"
    echo "$mean" > "/tmp/fc2_w3x_${label}_mean.txt"
}

echo "=== Build BASELINE (trailing bar.sync kept) ==="
make -B fc2-w3x 2>&1 | tail -2

run_n baseline "$N"

echo ""
echo "=== Build DROP_TRAIL_BARSYNC (trailing bar.sync removed) ==="
make -B fc2-w3x DFLAGS='-DDROP_TRAIL_BARSYNC' 2>&1 | tail -2

run_n dropped "$N"

echo ""
b=$(cat /tmp/fc2_w3x_baseline_mean.txt)
d=$(cat /tmp/fc2_w3x_dropped_mean.txt)
echo "=== Delta ==="
awk -v b="$b" -v d="$d" 'BEGIN{
    printf "baseline mean: %.4f ms\n", b
    printf "dropped  mean: %.4f ms\n", d
    printf "delta:         %+.4f ms (%+.2f %%)\n", d-b, (d/b-1)*100
}'
