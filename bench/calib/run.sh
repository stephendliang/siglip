#!/bin/bash
# Generate, build, and run SASS calibration benchmarks.
# Usage:
#   ./bench/calib/run.sh              # all three (tput, lat, conflict)
#   ./bench/calib/run.sh tput         # throughput ILP sweep only
#   ./bench/calib/run.sh tput lat     # multiple targets

set -euo pipefail
cd "$(dirname "$0")/../.."

NVCC="nvcc"
CFLAGS="-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17"
OUTDIR="bench/calib"
GEN="$OUTDIR/gen_kernels.py"
DB="$OUTDIR/instruction_db.py"

if [ $# -eq 0 ]; then
    TARGETS=(tput lat conflict)
else
    TARGETS=("$@")
fi

log() { echo "[$(date +%H:%M:%S)] $*"; }

for t in "${TARGETS[@]}"; do
    src="$OUTDIR/gen_${t}.cu"

    # ── Regenerate if stale ──
    if [ ! -f "$src" ] || [ "$GEN" -nt "$src" ] || [ "$DB" -nt "$src" ]; then
        log "Regenerating $src..."
        python3 "$GEN" "--${t}"
    fi

    # ── Build ──
    bin="$OUTDIR/gen_${t}"
    log "Building $bin..."
    $NVCC $CFLAGS "$src" -o "$bin"

    # ── Run ──
    log "Running $bin..."
    echo ""
    "$bin"
    echo ""
done

log "Done."
