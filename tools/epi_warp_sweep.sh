#!/bin/bash
# Epilogue warp count sweep: test 1/2/4 warps × contiguous/strided group mapping
# Run on B200: bash tools/epi_warp_sweep.sh [runs]

set -e
RUNS=${1:-5}
STAMP=$(date +%Y%m%d_%H%M%S)
DIR="data/epi_sweep_${STAMP}"
mkdir -p "$DIR"

echo "=== Epilogue warp sweep (${RUNS} runs each) ==="
echo "Results: ${DIR}/"
echo ""

# Build all variants first, with unique binary names
declare -a TAGS BINS

build_one() {
    local tag=$1 bin=$2 warps=$3 dflags=$4
    echo "Building: ${tag} -> ${bin}"
    if [ "$warps" = "4" ]; then
        make -s fc2-w3 DFLAGS="$dflags" 2>&1 | tail -1
        cp fc2-w3 "$bin"
    else
        make -s "fc2-w3-epi${warps}" DFLAGS="$dflags" 2>&1 | tail -1
        cp "fc2-w3-epi${warps}" "$bin"
    fi
    TAGS+=("$tag")
    BINS+=("$bin")
}

# epi1: only contiguous (strided identical with 1 warp)
build_one "epi1" "/tmp/fc2-epi1" 1 ""

# epi2: both mappings
build_one "epi2_contiguous" "/tmp/fc2-epi2-contig" 2 ""
build_one "epi2_strided"    "/tmp/fc2-epi2-stride" 2 "-DEPI_STRIDED"

# epi4: only contiguous (GROUPS_PER_WARP=1, strided identical)
build_one "epi4" "/tmp/fc2-epi4" 4 ""

echo ""
echo "All built. Running benchmarks..."
echo ""

# Run each
for idx in "${!TAGS[@]}"; do
    TAG="${TAGS[$idx]}"
    BIN="${BINS[$idx]}"

    echo "--- ${TAG} (${RUNS} runs) ---"
    BEST_MS=999
    VALID="?"
    for i in $(seq 1 $RUNS); do
        LINE=$("$BIN" 2>&1 | grep '@@RESULT')
        MS=$(echo "$LINE" | sed 's/.*ms=\([^ ]*\).*/\1/')
        V=$(echo "$LINE" | sed 's/.*valid=\([^ ]*\).*/\1/')
        echo "  run $i: ${MS}ms valid=${V}"
        if [ "$(echo "$MS < $BEST_MS" | bc -l)" = "1" ]; then
            BEST_MS=$MS
        fi
        VALID=$V
    done

    echo "${TAG}: best=${BEST_MS}ms valid=${VALID}" >> "${DIR}/results.txt"
    echo ">>> ${TAG}: best=${BEST_MS}ms valid=${VALID}"
    echo ""
done

echo "============================================"
echo "SUMMARY"
echo "============================================"
cat "${DIR}/results.txt"
echo ""
echo "Saved to ${DIR}/results.txt"
