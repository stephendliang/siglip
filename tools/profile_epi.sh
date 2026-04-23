#!/bin/bash
# Profile fc2_w3x baseline vs EPI_2WARP, both with -DPROFILE_CYCLES.
# Dumps the [PROFILE] per-warp phase breakdown + @@RESULT line for each
# config and run, in an easy-to-paste-back format.
#
# Paste the output block back for analysis of where each warp's tile
# cycles go and how the 2-warp vs 4-warp epi layout shifts them.
#
# Usage: tools/profile_epi.sh [N]   (default N=1)

set -eu
cd "$(dirname "$0")/.."

N=${1:-1}
OUT=/tmp/fc2_w3x_profile_$(date +%Y%m%d_%H%M%S).log
: > "$OUT"

BAR="================================================================"

run_config() {
    local label=$1 flags=$2
    {
        echo ""
        echo "$BAR"
        echo "BUILD: $label   DFLAGS='-DPROFILE_CYCLES $flags'"
        echo "$BAR"
    } | tee -a "$OUT"
    make -B fc2-w3x DFLAGS="-DPROFILE_CYCLES $flags" 2>&1 \
        | grep -E 'Used [0-9]+ registers|spill|error:' | tee -a "$OUT"

    for i in $(seq 1 "$N"); do
        {
            echo ""
            echo "----- [$label] run $i/$N -----"
        } | tee -a "$OUT"
        ./fc2-w3x 2>&1 | awk '/^FC2-W3X kernel:/,/^@@RESULT/' | tee -a "$OUT"
    done
}

run_config baseline   ''
run_config epi_2warp  '-DEPI_2WARP'

{
    echo ""
    echo "$BAR"
    echo "full log: $OUT"
    echo "$BAR"
} | tee -a "$OUT"
