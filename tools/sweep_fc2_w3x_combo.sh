#!/usr/bin/env bash
#
# Fractional factorial sweep of fc2_w3x "wash" levers.
#
# Each binary invocation costs ~1.5s (CUDA init + tensor map setup) regardless
# of kernel work. So variant count, not kernel time, dominates wall time.
# Default MODE=res7 (Resolution-VII half-fraction, generator I=ABCDEFG):
#   - 64 variants instead of 128
#   - all 7 main effects clean (alias only with 6-way, negligible)
#   - all 21 two-way interactions clean (alias only with 5-way, negligible)
#   - 3-way and higher are aliased; analyzer is told to skip them via --no-3way
#
# Lever bit assignment (variant name = "v" + 7-bit binary, MSB=bit6):
#   bit 0:  XPF_A
#   bit 1:  XPF_B
#   bit 2:  EPI_2WARP
#   bit 3:  DROP_LEAD_BARSYNC
#   bit 4:  DROP_TRAIL_BARSYNC
#   bit 5:  WAIT_GROUP_READ
#   bit 6:  NO_BULK_MEMCLBR
#
# Output: data/fc2_w3x_combo_<ts>/
#   build-v<bits>.log
#   run-v<bits>-<rep>.log
#   wall_data.csv      (variant,bits,rep,ms — fed to combo_anova.py)
#   anova.txt          (OLS report on main effects + 2-way interactions)
#
# Usage:
#   tools/sweep_fc2_w3x_combo.sh           # MODE=res7, REPS=3 (≈5 min B200)
#   MODE=full tools/sweep_fc2_w3x_combo.sh # 128 variants (≈30 min)
#   MODE=res4 tools/sweep_fc2_w3x_combo.sh # 32 variants (≈2.5 min)
#   REPS=10 tools/sweep_fc2_w3x_combo.sh
#   tools/sweep_fc2_w3x_combo.sh my_outdir
#

set -u
cd "$(dirname "$0")/.."

OUT=${1:-"data/fc2_w3x_combo_$(date +%Y%m%d_%H%M%S)"}
REPS=${REPS:-3}
MODE=${MODE:-res7}
NVCC=${NVCC:-nvcc}
CFLAGS='-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -lineinfo --ptxas-options=-v --cudart=static'
LDFLAGS='-lcurand_static -lculibos -lcuda'

mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/run.log"; }

LEVERS=(XPF_A XPF_B EPI_2WARP DROP_LEAD_BARSYNC DROP_TRAIL_BARSYNC WAIT_GROUP_READ NO_BULK_MEMCLBR)

bits_to_flags() {
    local bits=$1
    local flags=""
    for i in 0 1 2 3 4 5 6; do
        if (( (bits >> i) & 1 )); then
            flags+=" -D${LEVERS[$i]}"
        fi
    done
    echo "$flags"
}

bits_to_name() {
    local bits=$1
    local name="v"
    for i in 6 5 4 3 2 1 0; do
        name+=$(( (bits >> i) & 1 ))
    done
    echo "$name"
}

popcount() {
    local n=$1 c=0
    while (( n )); do c=$((c + (n & 1))); n=$((n >> 1)); done
    echo $c
}

case "$MODE" in
    full)
        SELECTED=($(seq 0 127))
        ANOVA_FLAGS=""
        ;;
    res7|half)
        # Generator I = ABCDEFG: keep variants with even popcount.
        # Mains alias with 6-ways, 2-ways alias with 5-ways — both clean.
        # 3-ways alias with 4-ways → analyzer must skip 3-way terms.
        SELECTED=()
        for b in $(seq 0 127); do
            (( $(popcount "$b") % 2 == 0 )) && SELECTED+=("$b")
        done
        ANOVA_FLAGS="--no-3way"
        ;;
    res4|quarter)
        # 2^(7-2) quarter-fraction. Generators I = ABCD, I = CDEFG.
        # Keep variants where (b0^b1^b2^b3)==0 AND (b2^b3^b4^b5^b6)==0.
        # All 7 mains clean; some 2-ways aliased — analyzer drops 3+way.
        SELECTED=()
        for b in $(seq 0 127); do
            local0=$(( ((b>>0)&1) ^ ((b>>1)&1) ^ ((b>>2)&1) ^ ((b>>3)&1) ))
            local1=$(( ((b>>2)&1) ^ ((b>>3)&1) ^ ((b>>4)&1) ^ ((b>>5)&1) ^ ((b>>6)&1) ))
            (( local0 == 0 && local1 == 0 )) && SELECTED+=("$b")
        done
        ANOVA_FLAGS="--no-3way"
        ;;
    *)
        echo "MODE must be one of: full, res7 (default), res4" >&2
        exit 1
        ;;
esac

N_VARIANTS=${#SELECTED[@]}

# ── Phase 1: build $N_VARIANTS variants ─────────────────────────────────
log "=== Phase 1: building $N_VARIANTS variants (MODE=$MODE) with -DPROFILE_CYCLES ==="
log "lever bits (MSB→LSB): ${LEVERS[6]} ${LEVERS[5]} ${LEVERS[4]} ${LEVERS[3]} ${LEVERS[2]} ${LEVERS[1]} ${LEVERS[0]}"

BUILD_OK=()
for bits in "${SELECTED[@]}"; do
    name=$(bits_to_name "$bits")
    flags=$(bits_to_flags "$bits")
    bin="$OUT/fc2-w3x-combo-$name"
    if $NVCC $CFLAGS -DPROFILE_CYCLES $flags fc2_w3x.cu -o "$bin" $LDFLAGS \
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
    echo "variant,bits,rep,XPF_A,XPF_B,EPI_2WARP,DROP_LEAD,DROP_TRAIL,WAIT_GROUP,NO_BULK_MEMCLBR,ms"
    for entry in "${BUILD_OK[@]}"; do
        IFS=: read -r bits name <<< "$entry"
        b0=$(( bits & 1 ))
        b1=$(( (bits >> 1) & 1 ))
        b2=$(( (bits >> 2) & 1 ))
        b3=$(( (bits >> 3) & 1 ))
        b4=$(( (bits >> 4) & 1 ))
        b5=$(( (bits >> 5) & 1 ))
        b6=$(( (bits >> 6) & 1 ))
        for rep in $(seq 1 "$REPS"); do
            f="$OUT/run-$name-$rep.log"
            if [[ -f "$f" ]]; then
                ms=$(grep -oE 'FC2-W3X kernel: [0-9.]+ ms' "$f" | head -1 | awk '{print $3}')
                if [[ -n "$ms" ]]; then
                    echo "$name,$bits,$rep,$b0,$b1,$b2,$b3,$b4,$b5,$b6,$ms"
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
python3 tools/combo_anova.py "$OUT/wall_data.csv" --out "$OUT/anova.txt" $ANOVA_FLAGS

log ""
log "report:           $OUT/anova.txt"
log "raw wall data:    $OUT/wall_data.csv"
log "raw per-run logs: $OUT/run-v<bits>-<rep>.log"
