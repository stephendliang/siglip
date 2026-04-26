#!/usr/bin/env bash
#
# 2-cell head-to-head: baseline vs BIAS_PRELOAD.
#
#   vbase     = (no flags)        — 4 LDS-per-rh bias path
#   vpreload  = -DBIAS_PRELOAD    — 12 LDS once + 4 shfls/subpass
#
# Pass-major interleave so both share clock / thermal / queue state —
# cross-session B200 baseline drift is ~4 µs, larger than the lever effect
# we're trying to detect.
#
# Each binary built with -DCOMBO_QUICK (fast cudaMemset init, N_WARMUP=1,
# N_TIMED_LAUNCHES=3, skip verify).  Run the FULL build (no COMBO_QUICK)
# separately first to confirm verify=1 passes.
#
# Output: data/fc2_w3x_bias_preload_<ts>/
#   build-v<name>.log
#   run-v<name>-<rep>.log
#   wall_data.csv
#   compare.txt    (vbase vs vpreload Welch t)
#
# Usage:
#   tools/sweep_fc2_w3x_bias_preload.sh           # REPS=20 (~40 s B200)
#   REPS=128 tools/sweep_fc2_w3x_bias_preload.sh  # tighter (~5 min)

set -u
cd "$(dirname "$0")/.."

OUT=${1:-"data/fc2_w3x_bias_preload_$(date +%Y%m%d_%H%M%S)"}
REPS=${REPS:-20}
NVCC=${NVCC:-nvcc}
CFLAGS='-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -lineinfo --ptxas-options=-v --cudart=static'
LDFLAGS='-lcurand_static -lculibos -lcuda'

mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/run.log"; }

VARIANTS=(
    "vbase:0:"
    "vpreload:1:-DBIAS_PRELOAD"
)

log "=== Phase 1: building ${#VARIANTS[@]} variants with -DCOMBO_QUICK ==="
BUILD_OK=()
for entry in "${VARIANTS[@]}"; do
    IFS=: read -r name bits flags <<< "$entry"
    bin="$OUT/fc2-w3x-bp-$name"
    if $NVCC $CFLAGS -DCOMBO_QUICK $flags fc2_w3x.cu -o "$bin" $LDFLAGS \
            > "$OUT/build-$name.log" 2>&1; then
        BUILD_OK+=("$entry")
        log "  build OK   $name  flags='$flags'"
    else
        log "  build FAIL $name  flags='$flags'"
    fi
done
log "build summary: ${#BUILD_OK[@]} / ${#VARIANTS[@]} succeeded"

if [[ ${#BUILD_OK[@]} -lt ${#VARIANTS[@]} ]]; then
    log "ERROR: need all ${#VARIANTS[@]} variants to compare; aborting"
    exit 1
fi

log ""
log "=== Phase 2: $REPS reps/variant, pass-major interleaving ==="
for rep in $(seq 1 "$REPS"); do
    log "-- pass $rep/$REPS --"
    for entry in "${BUILD_OK[@]}"; do
        IFS=: read -r name bits flags <<< "$entry"
        bin="$OUT/fc2-w3x-bp-$name"
        if ! "$bin" > "$OUT/run-$name-$rep.log" 2>&1; then
            log "  pass $rep $name run-FAIL"
        fi
    done
done

log ""
log "=== Phase 3: extracting wall ms → $OUT/wall_data.csv ==="
{
    echo "variant,bits,rep,BIAS_PRELOAD,ms"
    for entry in "${BUILD_OK[@]}"; do
        IFS=: read -r name bits flags <<< "$entry"
        b0=$(( bits & 1 ))
        for rep in $(seq 1 "$REPS"); do
            f="$OUT/run-$name-$rep.log"
            if [[ -f "$f" ]]; then
                ms=$(grep -oE 'FC2-W3X kernel: [0-9.]+ ms' "$f" | head -1 | awk '{print $3}')
                if [[ -n "$ms" ]]; then
                    echo "$name,$bits,$rep,$b0,$ms"
                fi
            fi
        done
    done
} > "$OUT/wall_data.csv"
n_rows=$(( $(wc -l < "$OUT/wall_data.csv") - 1 ))
log "extracted $n_rows wall measurements"

log ""
log "=== Phase 4: Welch t → $OUT/compare.txt ==="
python3 tools/two_cell_compare.py "$OUT/wall_data.csv" --out "$OUT/compare.txt"

log ""
log "report:           $OUT/compare.txt"
log "raw wall data:    $OUT/wall_data.csv"
log "raw per-run logs: $OUT/run-v<name>-<rep>.log"
