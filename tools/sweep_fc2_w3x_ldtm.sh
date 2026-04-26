#!/usr/bin/env bash
#
# 1-way A/B sweep on fc2_w3x epilogue lane-mapping:
#
#   stsm     — default: 2× tcgen05.ld.16x256b.x4 + 4× STSM (rank-1 shape)
#   ldtmx32  — wide:    1× tcgen05.ld.32x32b.x32 + 4× st.shared.v4.b32
#
# Same MMA, same TMA load, same dispatch, same bias-preload — only the
# epilogue's TMEM-load width and SMEM-store opcode differ.  ldtmx32
# replaces 2 LDTMs/rh with 1 wider LDTM (lane t = row t cols 0..31)
# and broadcasts 16 bias bf16x2 packs via shfl (vs 4 in STSM path).
#
# 2 cells × REPS reps, pass-major interleaved so both share clock/thermal/
# queue state.  Cross-session B200 baseline drift is ~4 µs, larger than
# the LDTM-shape effect we expect (LDTM is in MMA shadow on W0-W3, so
# expected wall delta ≤ a few µs).
#
# Each binary built with -DCOMBO_QUICK (fast cudaMemset init, N_WARMUP=1,
# N_TIMED_LAUNCHES=3, skip verify).  Run the FULL build (./fc2-w3x +
# ./fc2-w3x with -DLDTM_X32) once first to confirm verify=1 on both.
#
# Output: data/fc2_w3x_ldtm_<ts>/
#   build-<cell>.log
#   run-<cell>-<rep>.log
#   wall_data.csv      (variant,rep,ms)
#   compare.txt        (per-cell stats + 1-way ANOVA + Welch)
#
# Usage:
#   tools/sweep_fc2_w3x_ldtm.sh                # REPS=64  (~3 min B200)
#   REPS=128 tools/sweep_fc2_w3x_ldtm.sh       # tighter (~6 min)

set -u
cd "$(dirname "$0")/.."

OUT=${1:-"data/fc2_w3x_ldtm_$(date +%Y%m%d_%H%M%S)"}
REPS=${REPS:-64}
NVCC=${NVCC:-nvcc}
CFLAGS='-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -lineinfo --ptxas-options=-v --cudart=static'
LDFLAGS='-lcurand_static -lculibos -lcuda'

mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/run.log"; }

VARIANTS=(
    "stsm:"
    "ldtmx32:-DLDTM_X32"
)

log "=== Phase 1: building ${#VARIANTS[@]} variants with -DCOMBO_QUICK ==="
BUILD_OK=()
for entry in "${VARIANTS[@]}"; do
    IFS=: read -r name flags <<< "$entry"
    bin="$OUT/fc2-w3x-ldtm-$name"
    if $NVCC $CFLAGS -DCOMBO_QUICK $flags fc2_w3x.cu -o "$bin" $LDFLAGS \
            > "$OUT/build-$name.log" 2>&1; then
        BUILD_OK+=("$entry")
        regs=$(grep -oE 'Used [0-9]+ registers' "$OUT/build-$name.log" | head -1 | awk '{print $2}')
        stack=$(grep -oE '[0-9]+ bytes stack frame' "$OUT/build-$name.log" | head -1 | awk '{print $1}')
        log "  build OK   $name  regs=$regs  stack=${stack}B"
    else
        log "  build FAIL $name"
    fi
done
log "build summary: ${#BUILD_OK[@]} / ${#VARIANTS[@]} succeeded"

if [[ ${#BUILD_OK[@]} -lt ${#VARIANTS[@]} ]]; then
    log "ERROR: need both variants for clean comparison; aborting"
    exit 1
fi

log ""
log "=== Phase 2: $REPS reps/variant, pass-major interleaving ==="
for rep in $(seq 1 "$REPS"); do
    log "-- pass $rep/$REPS --"
    for entry in "${BUILD_OK[@]}"; do
        IFS=: read -r name flags <<< "$entry"
        bin="$OUT/fc2-w3x-ldtm-$name"
        if ! "$bin" > "$OUT/run-$name-$rep.log" 2>&1; then
            log "  pass $rep $name run-FAIL"
        fi
    done
done

log ""
log "=== Phase 3: extracting wall ms → $OUT/wall_data.csv ==="
{
    echo "variant,rep,ms"
    for entry in "${BUILD_OK[@]}"; do
        IFS=: read -r name flags <<< "$entry"
        for rep in $(seq 1 "$REPS"); do
            f="$OUT/run-$name-$rep.log"
            if [[ -f "$f" ]]; then
                ms=$(grep -oE 'FC2-W3X kernel: [0-9.]+ ms' "$f" | head -1 | awk '{print $3}')
                if [[ -n "$ms" ]]; then
                    echo "$name,$rep,$ms"
                fi
            fi
        done
    done
} > "$OUT/wall_data.csv"
n_rows=$(( $(wc -l < "$OUT/wall_data.csv") - 1 ))
log "extracted $n_rows wall measurements"

log ""
log "=== Phase 4: 1-way ANOVA → $OUT/compare.txt ==="
python3 tools/anova_1way.py "$OUT/wall_data.csv" \
    --factor variant \
    --out "$OUT/compare.txt"

log ""
log "report:           $OUT/compare.txt"
log "raw wall data:    $OUT/wall_data.csv"
log "raw per-run logs: $OUT/run-<cell>-<rep>.log"
