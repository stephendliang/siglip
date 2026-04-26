#!/usr/bin/env bash
# Interleave fc2-w3x (default 6-warp) vs fc2-w3x-3warp (2 epi + 1 combined TMA+MMA).
# Run on B200.
set -euo pipefail
cd "$(dirname "$0")/.."

REPS=${REPS:-10}

if [[ ! -x ./fc2-w3x      ]]; then make fc2-w3x;       fi
if [[ ! -x ./fc2-w3x-3warp ]]; then make fc2-w3x-3warp; fi

echo "rep,variant,ms,c0,valid"
for i in $(seq 1 "$REPS"); do
    for v in w3x 3warp; do
        bin="./fc2-w3x"; [[ "$v" == "3warp" ]] && bin="./fc2-w3x-3warp"
        line=$("$bin" 2>/dev/null | grep '^@@RESULT' || echo '@@RESULT ms=NaN tflops=NaN checksum=NaN valid=0 c0=NaN')
        ms=$(echo "$line" | sed -E 's/.*ms=([0-9.]+).*/\1/')
        c0=$(echo "$line" | sed -E 's/.*c0=([-0-9.NaN]+).*/\1/')
        valid=$(echo "$line" | sed -E 's/.*valid=([0-9]+).*/\1/')
        echo "$i,$v,$ms,$c0,$valid"
    done
done
