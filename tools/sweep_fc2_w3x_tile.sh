#!/usr/bin/env bash
#
# 1-way tile-shape sweep on fc2_w3x:
#
#   tn256_ns6  — current default (256x256 cluster output, NS=6)
#   tn128_ns6  — TN halved, same NS
#   tn128_ns8  — TN halved, deeper pipeline (NS=8 fits since per-stage shrinks)
#
# Single-process per-cell timing: REPS=N env var loops the existing
# 3-launch cudaEvent window N times inside one process and emits a
# `@@SAMPLE rep=K ms=X` line per rep (added 2026-04-27 to fc2_w3x.cu).
# This drops ~200 ms × (REPS-1) of per-rep process churn per cell.
#
# 3 cells × REPS reps. Tradeoff: cell-major rather than pass-major
# interleaving — but the whole sweep wallclock is short enough that all
# cells stay inside one B200 drift window (drift is ~4 µs at the
# minute scale, far above the µs-scale lever).
#
# Each binary built with -DCOMBO_QUICK (fast cudaMemset init, N_WARMUP=1,
# N_TIMED_LAUNCHES=3, skip verify). Run the FULL build once first to
# confirm verify=1 passes per cell.
#
# Output: data/fc2_w3x_tile_<ts>/
#   build-<cell>.log
#   run-<cell>.log     (one log per cell; contains all @@SAMPLE lines)
#   wall_data.csv      (variant,tile_shape,rep,ms)
#   compare.txt        (per-cell stats + 1-way ANOVA + pairwise Welch)
#
# Usage:
#   tools/sweep_fc2_w3x_tile.sh                # REPS=20
#   REPS=64 tools/sweep_fc2_w3x_tile.sh        # tighter

set -u
cd "$(dirname "$0")/.."

OUT=${1:-"data/fc2_w3x_tile_$(date +%Y%m%d_%H%M%S)"}
REPS=${REPS:-20}
NVCC=${NVCC:-nvcc}
CFLAGS='-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -lineinfo --ptxas-options=-v --cudart=static'
LDFLAGS='-lcurand_static -lculibos -lcuda'

mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/run.log"; }

# Format: "<name>:<tile_shape>:<-D flag list>"
VARIANTS=(
    "tn256_ns6:tn256_ns6:"
    "tn128_ns6:tn128_ns6:-DTN=128"
    "tn128_ns8:tn128_ns8:-DTN=128 -DN_STAGES=8"
)

# ── Phase 1: build ──────────────────────────────────────────────────────
log "=== Phase 1: building ${#VARIANTS[@]} variants with -DCOMBO_QUICK ==="
BUILD_OK=()
for entry in "${VARIANTS[@]}"; do
    IFS=: read -r name tile_shape flags <<< "$entry"
    bin="$OUT/fc2-w3x-tile-$name"
    if $NVCC $CFLAGS -DCOMBO_QUICK $flags fc2_w3x.cu -o "$bin" $LDFLAGS \
            > "$OUT/build-$name.log" 2>&1; then
        BUILD_OK+=("$entry")
        regs=$(grep -oE 'Used [0-9]+ registers' "$OUT/build-$name.log" | head -1 | awk '{print $2}')
        log "  build OK   $name  flags='$flags'  regs=$regs"
    else
        log "  build FAIL $name  flags='$flags'"
    fi
done
log "build summary: ${#BUILD_OK[@]} / ${#VARIANTS[@]} succeeded"

if [[ ${#BUILD_OK[@]} -lt ${#VARIANTS[@]} ]]; then
    log "ERROR: need all ${#VARIANTS[@]} variants for clean comparison; aborting"
    exit 1
fi

# ── Phase 2: single-process per cell (REPS env var) ────────────────────
log ""
log "=== Phase 2: $REPS reps/variant, single-process per cell (REPS env) ==="
for entry in "${BUILD_OK[@]}"; do
    IFS=: read -r name tile_shape flags <<< "$entry"
    bin="$OUT/fc2-w3x-tile-$name"
    if ! REPS="$REPS" "$bin" > "$OUT/run-$name.log" 2>&1; then
        log "  $name run-FAIL"
        continue
    fi
    n_seen=$(grep -cE '^@@SAMPLE rep=' "$OUT/run-$name.log" || true)
    log "  $name  reps=$n_seen"
done

# ── Phase 3: extract @@SAMPLE lines into CSV ───────────────────────────
log ""
log "=== Phase 3: extracting @@SAMPLE → $OUT/wall_data.csv ==="
{
    echo "variant,tile_shape,rep,ms"
    for entry in "${BUILD_OK[@]}"; do
        IFS=: read -r name tile_shape flags <<< "$entry"
        f="$OUT/run-$name.log"
        if [[ -f "$f" ]]; then
            grep -E '^@@SAMPLE rep=' "$f" | \
                sed -E 's/^@@SAMPLE rep=([0-9]+) ms=([0-9.]+).*/\1,\2/' | \
                awk -F, -v n="$name" -v s="$tile_shape" '{print n","s","$1","$2}'
        fi
    done
} > "$OUT/wall_data.csv"
n_rows=$(( $(wc -l < "$OUT/wall_data.csv") - 1 ))
log "extracted $n_rows wall measurements"

# ── Phase 4: 1-way ANOVA + per-cell stats + Welch ──────────────────────
log ""
log "=== Phase 4: 1-way ANOVA → $OUT/compare.txt ==="
python3 tools/anova_1way.py "$OUT/wall_data.csv" \
    --factor tile_shape \
    --out "$OUT/compare.txt"

log ""
log "report:           $OUT/compare.txt"
log "raw wall data:    $OUT/wall_data.csv"
log "raw per-run logs: $OUT/run-<cell>-<rep>.log"
