#!/usr/bin/env bash
#
# Combinatorial sweep of fc2_w3x's surviving "wash" levers.
#
# Post-purge state (2026-04-26): XPF_A and XPF_B were Bonferroni-confirmed
# regressions in the prior 128-cell sweep and have been deleted from the
# kernel. The 5 remaining levers are:
#
#   bit 0:  EPI_2WARP
#   bit 1:  DROP_LEAD_BARSYNC
#   bit 2:  DROP_TRAIL_BARSYNC
#   bit 3:  WAIT_GROUP_READ
#   bit 4:  NO_BULK_MEMCLBR
#
# Variant name = "v" + 5-bit binary (MSB=bit4=NO_BULK, LSB=bit0=EPI_2WARP).
# 2^5 = 32 variants, default REPS=10 for tight cell-mean stats. Total
# B200 wall ≈ 2 min with COMBO_QUICK fast init.
#
# Each binary is built with -DCOMBO_QUICK (fast cudaMemset init, N_WARMUP=1,
# N_TIMED_LAUNCHES=3, skip verify).
#
# Output: data/fc2_w3x_combo_<ts>/
#   build-v<bits>.log
#   run-v<bits>-<rep>.log
#   wall_data.csv      (variant,bits,rep,EPI_2WARP,...,ms)
#   anova.txt          (OLS report on mains + 2-way + 3-way)
#
# Usage:
#   tools/sweep_fc2_w3x_combo.sh           # MODE=full, REPS=10 (≈2 min)
#   REPS=20 tools/sweep_fc2_w3x_combo.sh   # tighter
#   tools/sweep_fc2_w3x_combo.sh my_outdir
#

set -u
cd "$(dirname "$0")/.."

OUT=${1:-"data/fc2_w3x_combo_$(date +%Y%m%d_%H%M%S)"}
REPS=${REPS:-10}
NVCC=${NVCC:-nvcc}
CFLAGS='-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -lineinfo --ptxas-options=-v --cudart=static'
LDFLAGS='-lcurand_static -lculibos -lcuda'

mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/run.log"; }

LEVERS=(EPI_2WARP DROP_LEAD_BARSYNC DROP_TRAIL_BARSYNC WAIT_GROUP_READ NO_BULK_MEMCLBR)
N_LEVERS=${#LEVERS[@]}
N_VARIANTS=$(( 1 << N_LEVERS ))

bits_to_flags() {
    local bits=$1
    local flags=""
    for i in $(seq 0 $((N_LEVERS - 1))); do
        if (( (bits >> i) & 1 )); then
            flags+=" -D${LEVERS[$i]}"
        fi
    done
    echo "$flags"
}

bits_to_name() {
    local bits=$1
    local name="v"
    for i in $(seq $((N_LEVERS - 1)) -1 0); do
        name+=$(( (bits >> i) & 1 ))
    done
    echo "$name"
}

# ── Phase 1: build $N_VARIANTS variants ─────────────────────────────────
log "=== Phase 1: building $N_VARIANTS variants with -DCOMBO_QUICK ==="
log "lever bits (MSB→LSB): ${LEVERS[4]} ${LEVERS[3]} ${LEVERS[2]} ${LEVERS[1]} ${LEVERS[0]}"

BUILD_OK=()
for bits in $(seq 0 $((N_VARIANTS - 1))); do
    name=$(bits_to_name "$bits")
    flags=$(bits_to_flags "$bits")
    bin="$OUT/fc2-w3x-combo-$name"
    if $NVCC $CFLAGS -DCOMBO_QUICK $flags fc2_w3x.cu -o "$bin" $LDFLAGS \
            > "$OUT/build-$name.log" 2>&1; then
        BUILD_OK+=("$bits:$name")
    else
        log "  build FAIL $name  flags='$flags'"
    fi
done
log "build summary: ${#BUILD_OK[@]} / $N_VARIANTS succeeded"

# ── Phase 2: interleaved reps across surviving variants ─────────────────
log ""
log "=== Phase 2: $REPS reps/variant, pass-major interleaving ==="
for rep in $(seq 1 "$REPS"); do
    log "-- pass $rep/$REPS --"
    for entry in "${BUILD_OK[@]}"; do
        IFS=: read -r bits name <<< "$entry"
        bin="$OUT/fc2-w3x-combo-$name"
        if ! "$bin" > "$OUT/run-$name-$rep.log" 2>&1; then
            log "  pass $rep $name run-FAIL"
        fi
    done
done

# ── Phase 3: extract wall ms into CSV ───────────────────────────────────
log ""
log "=== Phase 3: extracting wall ms → $OUT/wall_data.csv ==="
{
    echo "variant,bits,rep,EPI_2WARP,DROP_LEAD,DROP_TRAIL,WAIT_GROUP,NO_BULK_MEMCLBR,ms"
    for entry in "${BUILD_OK[@]}"; do
        IFS=: read -r bits name <<< "$entry"
        b0=$(( bits & 1 ))
        b1=$(( (bits >> 1) & 1 ))
        b2=$(( (bits >> 2) & 1 ))
        b3=$(( (bits >> 3) & 1 ))
        b4=$(( (bits >> 4) & 1 ))
        for rep in $(seq 1 "$REPS"); do
            f="$OUT/run-$name-$rep.log"
            if [[ -f "$f" ]]; then
                ms=$(grep -oE 'FC2-W3X kernel: [0-9.]+ ms' "$f" | head -1 | awk '{print $3}')
                if [[ -n "$ms" ]]; then
                    echo "$name,$bits,$rep,$b0,$b1,$b2,$b3,$b4,$ms"
                fi
            fi
        done
    done
} > "$OUT/wall_data.csv"
n_rows=$(( $(wc -l < "$OUT/wall_data.csv") - 1 ))
log "extracted $n_rows wall measurements"

# ── Phase 4: regression ─────────────────────────────────────────────────
log ""
log "=== Phase 4: OLS regression → $OUT/anova.txt ==="
python3 tools/combo_anova.py "$OUT/wall_data.csv" --out "$OUT/anova.txt"

log ""
log "report:           $OUT/anova.txt"
log "raw wall data:    $OUT/wall_data.csv"
log "raw per-run logs: $OUT/run-v<bits>-<rep>.log"
