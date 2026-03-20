#!/bin/bash
# Generate, build, and run all SASS calibration benchmarks.
# Usage:
#   ./bench/calib/run.sh              # all three (tput, lat, conflict)
#   ./bench/calib/run.sh tput         # throughput ILP sweep only
#   ./bench/calib/run.sh lat          # latency only
#   ./bench/calib/run.sh conflict     # conflict matrix only
#   ./bench/calib/run.sh tput lat     # multiple targets

set -euo pipefail
cd "$(dirname "$0")/../.."

NVCC="nvcc"
CFLAGS="-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17"
OUTDIR="bench/calib"
BINDIR="$OUTDIR"

TARGETS=("${@:-tput lat conflict}")
if [ $# -eq 0 ]; then TARGETS=(tput lat conflict); fi

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ── Generate ──
log "Generating .cu files..."
python3 "$OUTDIR/gen_kernels.py"

for t in "${TARGETS[@]}"; do
    src="$OUTDIR/gen_${t}.cu"
    bin="$BINDIR/gen_${t}"

    if [ ! -f "$src" ]; then
        log "ERROR: $src not found"
        continue
    fi

    # ── Build ──
    log "Building $bin..."
    $NVCC $CFLAGS "$src" -o "$bin"
    log "Built $bin"

    # ── Run ──
    log "Running $bin..."
    echo ""
    "$bin"
    echo ""
done

log "Done."
