#!/usr/bin/env bash
#
# Two-cell head-to-head: the only two fc2_w3x lever combos we still care
# about after the n=5 + n=10 sweeps showed nothing else is reproducibly
# different from baseline.
#
#   A = v10011 = NO_BULK_MEMCLBR + DROP_LEAD_BARSYNC + EPI_2WARP
#   B = v01101 = WAIT_GROUP_READ + DROP_TRAIL_BARSYNC + EPI_2WARP
#
# These were the strongest cells across the prior runs (top-2 in the
# n=5 follow-up: v10011 −1.00 µs, v01101 −0.80 µs vs baseline). Both
# share EPI_2WARP, so the head-to-head isolates which DROP / WAIT / BULK
# triplet wins under the 2-warp epi structure.
#
# No baseline cell here — the previous n=10 sweep already pinned baseline
# to within ±0.4 µs. This script answers a single question: A vs B.
#
# Each binary is built with -DCOMBO_QUICK (fast cudaMemset init,
# N_WARMUP=1, N_TIMED_LAUNCHES=3, skip verify).
#
# Output: data/fc2_w3x_combo_<ts>/
#   build-v<bits>.log
#   run-v<bits>-<rep>.log
#   wall_data.csv      (variant,bits,rep,EPI_2WARP,...,ms)
#   compare.txt        (mean ± SE per cell + Welch t between them)
#
# Usage:
#   tools/sweep_fc2_w3x_combo.sh           # REPS=20 (≈40 s B200)
#   REPS=40 tools/sweep_fc2_w3x_combo.sh   # tighter
#

set -u
cd "$(dirname "$0")/.."

OUT=${1:-"data/fc2_w3x_combo_$(date +%Y%m%d_%H%M%S)"}
REPS=${REPS:-20}
NVCC=${NVCC:-nvcc}
CFLAGS='-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -lineinfo --ptxas-options=-v --cudart=static'
LDFLAGS='-lcurand_static -lculibos -lcuda'

mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/run.log"; }

# 5-bit MSB→LSB layout (matches prior CSVs):
#   bit 4: NO_BULK_MEMCLBR
#   bit 3: WAIT_GROUP_READ
#   bit 2: DROP_TRAIL_BARSYNC
#   bit 1: DROP_LEAD_BARSYNC
#   bit 0: EPI_2WARP
#
# Format: "<name>:<bits>:<-D flag list>"
VARIANTS=(
    "v10011:19:-DNO_BULK_MEMCLBR -DDROP_LEAD_BARSYNC -DEPI_2WARP"
    "v01101:13:-DWAIT_GROUP_READ -DDROP_TRAIL_BARSYNC -DEPI_2WARP"
)

# ── Phase 1: build the two variants ─────────────────────────────────────
log "=== Phase 1: building ${#VARIANTS[@]} variants with -DCOMBO_QUICK ==="

BUILD_OK=()
for entry in "${VARIANTS[@]}"; do
    IFS=: read -r name bits flags <<< "$entry"
    bin="$OUT/fc2-w3x-combo-$name"
    if $NVCC $CFLAGS -DCOMBO_QUICK $flags fc2_w3x.cu -o "$bin" $LDFLAGS \
            > "$OUT/build-$name.log" 2>&1; then
        BUILD_OK+=("$entry")
        log "  build OK   $name  flags='$flags'"
    else
        log "  build FAIL $name  flags='$flags'"
    fi
done
log "build summary: ${#BUILD_OK[@]} / ${#VARIANTS[@]} succeeded"

if [[ ${#BUILD_OK[@]} -lt 2 ]]; then
    log "ERROR: need both variants to compare; aborting"
    exit 1
fi

# ── Phase 2: interleaved reps ───────────────────────────────────────────
log ""
log "=== Phase 2: $REPS reps/variant, pass-major interleaving ==="
for rep in $(seq 1 "$REPS"); do
    log "-- pass $rep/$REPS --"
    for entry in "${BUILD_OK[@]}"; do
        IFS=: read -r name bits flags <<< "$entry"
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
        IFS=: read -r name bits flags <<< "$entry"
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

# ── Phase 4: head-to-head comparison ────────────────────────────────────
log ""
log "=== Phase 4: Welch t-test → $OUT/compare.txt ==="
python3 tools/two_cell_compare.py "$OUT/wall_data.csv" --out "$OUT/compare.txt"

log ""
log "report:           $OUT/compare.txt"
log "raw wall data:    $OUT/wall_data.csv"
log "raw per-run logs: $OUT/run-v<bits>-<rep>.log"
