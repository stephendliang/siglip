#!/bin/bash
# Loop-test fc2_w3x across the 2x2 (EPI_2WARP x DROP_TRAIL_BARSYNC) grid.
# Each option rebuilt fresh and run N times; reports pass/fail (correctness)
# and mean/min/max ms per option, then a delta-vs-baseline summary.
#
# Option 0 (neither macro) is the 1.007 ms reference.
#
# Usage: tools/test_epi_grid.sh [N]   (default N=20)

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
    echo "$mean"      > "/tmp/fc2_w3x_${label}_mean.txt"
    echo "$pass $fail $err" > "/tmp/fc2_w3x_${label}_counts.txt"
}

build_run() {
    local label=$1 flags=$2
    echo ""
    echo "=== Build [$label]: DFLAGS='$flags' ==="
    make -B fc2-w3x DFLAGS="$flags" 2>&1 | grep -E 'Used|error|ptxas info' | head -5
    run_n "$label" "$N"
}

build_run opt0_base        ''
build_run opt1_drop        '-DDROP_TRAIL_BARSYNC'
build_run opt2_2warp       '-DEPI_2WARP'
build_run opt3_2warp_drop  '-DEPI_2WARP -DDROP_TRAIL_BARSYNC'

echo ""
echo "=== Summary ==="
m0=$(cat /tmp/fc2_w3x_opt0_base_mean.txt)
m1=$(cat /tmp/fc2_w3x_opt1_drop_mean.txt)
m2=$(cat /tmp/fc2_w3x_opt2_2warp_mean.txt)
m3=$(cat /tmp/fc2_w3x_opt3_2warp_drop_mean.txt)
c0=$(cat /tmp/fc2_w3x_opt0_base_counts.txt)
c1=$(cat /tmp/fc2_w3x_opt1_drop_counts.txt)
c2=$(cat /tmp/fc2_w3x_opt2_2warp_counts.txt)
c3=$(cat /tmp/fc2_w3x_opt3_2warp_drop_counts.txt)

awk -v n="$N" \
    -v m0="$m0" -v m1="$m1" -v m2="$m2" -v m3="$m3" \
    -v c0="$c0" -v c1="$c1" -v c2="$c2" -v c3="$c3" 'BEGIN{
    split(c0, a0); split(c1, a1); split(c2, a2); split(c3, a3)
    printf "%-34s %8s %12s %11s   %s\n", "option", "ms", "delta ms", "delta %", "pass/N"
    printf "%-34s %8.4f %12s %11s   %d/%d\n",     "opt0  4-epi + bar (baseline)", m0, "--", "--", a0[1], n
    printf "%-34s %8.4f %+12.4f %+10.2f%%   %d/%d\n", "opt1  4-epi + drop_bar",   m1, m1-m0, (m1/m0-1)*100, a1[1], n
    printf "%-34s %8.4f %+12.4f %+10.2f%%   %d/%d\n", "opt2  2-epi 4-warp + bar", m2, m2-m0, (m2/m0-1)*100, a2[1], n
    printf "%-34s %8.4f %+12.4f %+10.2f%%   %d/%d\n", "opt3  2-epi + drop_bar",   m3, m3-m0, (m3/m0-1)*100, a3[1], n
}'
