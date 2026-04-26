#!/usr/bin/env bash
#
# 1-way B-load tile-width sweep on fc2_w3x:
#
#   bbox128  — 1 B TMA op/stage    (default; box[1]=128 N-rows)
#   bbox64   — 2 B TMA ops/stage   (box[1]=64)
#   bbox32   — 4 B TMA ops/stage   (box[1]=32)
#   bbox16   — 8 B TMA ops/stage   (box[1]=16)
#   bbox8    — 16 B TMA ops/stage  (box[1]=8)
#   bbox4    — 32 B TMA ops/stage  (box[1]=4)
#   bbox2    — 64 B TMA ops/stage  (box[1]=2)
#   bbox1    — 128 B TMA ops/stage (box[1]=1; degenerate floor)
#
# Total B bytes per stage stays 16 KB per CTA across all cells; only the
# fragmentation across cp.async.bulk issues changes. A and the cluster mbar
# accounting are untouched.
#
# 8 cells × REPS reps, pass-major interleaved so all cells share clock/
# thermal/queue state. Cross-session B200 baseline drift is ~4 µs, larger
# than any single lever effect we expect to detect.
#
# Each binary built with -DCOMBO_QUICK (fast cudaMemset init, N_WARMUP=1,
# N_TIMED_LAUNCHES=3, skip verify). Run the FULL build once first to
# confirm verify=1 on at least bbox128 + one fragmented cell.
#
# Output: data/fc2_w3x_bbox_<ts>/
#   build-<cell>.log
#   run-<cell>-<rep>.log
#   wall_data.csv      (variant,bbox,rep,ms)
#   compare.txt        (per-cell stats + 1-way ANOVA + pairwise Welch)
#
# Usage:
#   tools/sweep_fc2_w3x_bbox.sh                # REPS=20  (~3 min B200)
#   REPS=64 tools/sweep_fc2_w3x_bbox.sh        # tighter (~10 min)

set -u
cd "$(dirname "$0")/.."

OUT=${1:-"data/fc2_w3x_bbox_$(date +%Y%m%d_%H%M%S)"}
REPS=${REPS:-20}
NVCC=${NVCC:-nvcc}
CFLAGS='-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -lineinfo --ptxas-options=-v --cudart=static'
LDFLAGS='-lcurand_static -lculibos -lcuda'

mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/run.log"; }

# Format: "<name>:<bbox>:<-D flag list>"
VARIANTS=(
    "bbox128:128:"
    "bbox64:64:-DB_BOX_N=64"
    "bbox32:32:-DB_BOX_N=32"
    "bbox16:16:-DB_BOX_N=16"
    "bbox8:8:-DB_BOX_N=8"
    "bbox4:4:-DB_BOX_N=4"
    "bbox2:2:-DB_BOX_N=2"
    "bbox1:1:-DB_BOX_N=1"
)

# ── Phase 1: build ──────────────────────────────────────────────────────
log "=== Phase 1: building ${#VARIANTS[@]} variants with -DCOMBO_QUICK ==="
BUILD_OK=()
for entry in "${VARIANTS[@]}"; do
    IFS=: read -r name bbox flags <<< "$entry"
    bin="$OUT/fc2-w3x-bbox-$name"
    if $NVCC $CFLAGS -DCOMBO_QUICK $flags fc2_w3x.cu -o "$bin" $LDFLAGS \
            > "$OUT/build-$name.log" 2>&1; then
        BUILD_OK+=("$entry")
        regs=$(grep -oE 'Used [0-9]+ registers' "$OUT/build-$name.log" | head -1 | awk '{print $2}')
        stack=$(grep -oE '[0-9]+ bytes stack frame' "$OUT/build-$name.log" | head -1 | awk '{print $1}')
        log "  build OK   $name  bbox=$bbox  regs=$regs  stack=${stack}B"
    else
        log "  build FAIL $name  bbox=$bbox"
    fi
done
log "build summary: ${#BUILD_OK[@]} / ${#VARIANTS[@]} succeeded"

if [[ ${#BUILD_OK[@]} -lt ${#VARIANTS[@]} ]]; then
    log "ERROR: need all ${#VARIANTS[@]} variants for clean comparison; aborting"
    exit 1
fi

# ── Phase 2: interleaved reps ──────────────────────────────────────────
log ""
log "=== Phase 2: $REPS reps/variant, pass-major interleaving ==="
for rep in $(seq 1 "$REPS"); do
    log "-- pass $rep/$REPS --"
    for entry in "${BUILD_OK[@]}"; do
        IFS=: read -r name bbox flags <<< "$entry"
        bin="$OUT/fc2-w3x-bbox-$name"
        if ! "$bin" > "$OUT/run-$name-$rep.log" 2>&1; then
            log "  pass $rep $name run-FAIL"
        fi
    done
done

# ── Phase 3: extract wall ms into CSV ──────────────────────────────────
log ""
log "=== Phase 3: extracting wall ms → $OUT/wall_data.csv ==="
{
    echo "variant,bbox,rep,ms"
    for entry in "${BUILD_OK[@]}"; do
        IFS=: read -r name bbox flags <<< "$entry"
        for rep in $(seq 1 "$REPS"); do
            f="$OUT/run-$name-$rep.log"
            if [[ -f "$f" ]]; then
                ms=$(grep -oE 'FC2-W3X kernel: [0-9.]+ ms' "$f" | head -1 | awk '{print $3}')
                if [[ -n "$ms" ]]; then
                    echo "$name,$bbox,$rep,$ms"
                fi
            fi
        done
    done
} > "$OUT/wall_data.csv"
n_rows=$(( $(wc -l < "$OUT/wall_data.csv") - 1 ))
log "extracted $n_rows wall measurements"

# ── Phase 4: 1-way ANOVA + per-cell stats + Welch ──────────────────────
log ""
log "=== Phase 4: 1-way ANOVA → $OUT/compare.txt ==="
python3 tools/anova_1way.py "$OUT/wall_data.csv" \
    --factor bbox \
    --out "$OUT/compare.txt"

log ""
log "report:           $OUT/compare.txt"
log "raw wall data:    $OUT/wall_data.csv"
log "raw per-run logs: $OUT/run-<cell>-<rep>.log"
