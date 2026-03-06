#!/bin/bash
# Build and run all CUTLASS bench variants with maximal tile/cluster/policy sweep.
#
# Each binary sweeps tile shapes × cluster configs × scheduling policies:
#   - Tile shapes: MxNxK (e.g. 256x256x128, 128x128x64, ...)
#   - Cluster configs: 2x1, 2x2, 1x1
#   - Policies: auto, 1SM (Sauto/S3/S4), 2SM (Sauto/S3/S4/S5)
#   - Stage counts guarded by compile-time SMEM constraint programming
#
# Extended sweep (-DCUTLASS_EXTENDED_SWEEP=1) adds:
#   - More tile shapes (cluster_n=2, K=64 variants, extra 1SM tiles)
#   - 2SM S5 policy for tiles that fit in SMEM
#
# Usage:
#   ./tools/cutlass_sweep.sh              # full sweep, default batch (32 imgs/SM)
#   ./tools/cutlass_sweep.sh 1            # quick test (1 img/SM = 148*196 rows)
#   ./tools/cutlass_sweep.sh 32 standard  # standard tile list only (no extended)
#
# Output goes to data/cutlass_sweep_<timestamp>.log

set -euo pipefail
cd "$(dirname "$0")/.."

IMGS_PER_SM="${1:-32}"
MODE="${2:-max}"  # "max" (default) or "standard"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="data"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/cutlass_sweep_${TIMESTAMP}.log"

echo "CUTLASS sweep: imgs_per_sm=$IMGS_PER_SM mode=$MODE"
echo "Log: $LOGFILE"
echo ""

# ── Build phase ──
build_targets=()
if [ "$MODE" = "max" ]; then
    build_targets=(cutlass-bench-max cutlass-bench-fc1-max cutlass-bench-fc2-max)
else
    build_targets=(cutlass-bench cutlass-bench-fc1 cutlass-bench-fc2)
fi

echo "Building ${#build_targets[@]} targets..."
for target in "${build_targets[@]}"; do
    echo -n "  $target ... "
    if make "$target" 2>&1 | tail -1 | grep -q "Error"; then
        echo "FAILED"
        echo "Build failed for $target. Run 'make $target' to see errors."
        exit 1
    fi
    echo "ok"
done
echo ""

# ── Run phase ──
{
    echo "═══════════════════════════════════════════════════════════════"
    echo "CUTLASS Sweep — $(date)"
    echo "  Mode: $MODE | imgs_per_sm: $IMGS_PER_SM"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""

    for target in "${build_targets[@]}"; do
        case "$target" in
            cutlass-bench|cutlass-bench-max)
                label="Patch Embed (N=768, K=768, PERIODIC_ADD)" ;;
            cutlass-bench-fc1|cutlass-bench-fc1-max)
                label="FC1 (N=3072, K=768, GELU_BIAS)" ;;
            cutlass-bench-fc2|cutlass-bench-fc2-max)
                label="FC2 (N=768, K=3072, BIAS_ONLY)" ;;
        esac
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  $label"
        echo "  Binary: ./$target $IMGS_PER_SM"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        ./"$target" "$IMGS_PER_SM"
        echo ""
    done

    echo "═══════════════════════════════════════════════════════════════"
    echo "Sweep complete — $(date)"
    echo "═══════════════════════════════════════════════════════════════"
} 2>&1 | tee "$LOGFILE"

echo ""
echo "Results saved to: $LOGFILE"
