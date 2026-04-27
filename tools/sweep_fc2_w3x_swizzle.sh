#!/usr/bin/env bash
#
# 1-way swizzle sweep on fc2_w3x: pits the 2 newest TD>30 survivors
# (gflip=33, tn2br=34) against the established 6-way-tie set (dgsw
# baseline, checkered=18, dgsnake=19, dg4 via DG_GROUP_SIZE=4) plus
# zigzag=11 / rowmajor=13 as off-tie reference outliers so the ANOVA
# has at least one cell that is *expected* to break ties — keeps the
# F-statistic interpretable when the new TD>30 set ends up
# indistinguishable.
#
# INGH=32 dropped: 2026-04-26 n=20 sweep landed it at +3.35 µs vs
# leader (STRONG, p≪1e-9) — worse than the zigzag/rowmajor reference
# outliers. Re-probing it costs build+run time and adds nothing.
#
# Pass-major interleaving (all variants in pass i, then pass i+1) shares
# clock / thermal / queue state across cells, beating cross-session
# B200 drift (~4 µs, larger than the µs-scale lever effects we care
# about). REPS=20 default catches ~0.5 µs cell separations at α≈0.05;
# REPS=64 tightens to ~0.2 µs.
#
# Output: data/fc2_w3x_swizzle_<ts>/
#   build-<cell>.log
#   run-<cell>-<rep>.log
#   wall_data.csv      (variant,swizzle,rep,ms)
#   compare.txt        (per-cell stats + 1-way ANOVA + pairwise Welch)
#
# Usage:
#   tools/sweep_fc2_w3x_swizzle.sh                # REPS=20
#   REPS=64 tools/sweep_fc2_w3x_swizzle.sh
#   tools/sweep_fc2_w3x_swizzle.sh my_outdir

set -u
cd "$(dirname "$0")/.."

OUT=${1:-"data/fc2_w3x_swizzle_$(date +%Y%m%d_%H%M%S)"}
REPS=${REPS:-20}
NVCC=${NVCC:-nvcc}
CFLAGS='-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -lineinfo --ptxas-options=-v --cudart=static'
LDFLAGS='-lcurand_static -lculibos -lcuda'

mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/run.log"; }

# Format: "<name>:<swizzle>:<-D flag list>"
# swizzle column groups variants for ANOVA — one cell per name here.
VARIANTS=(
    "dgsw:dgsw:"
    "checkered:checkered:-DTILE_DISPATCH=18"
    "dgsnake:dgsnake:-DTILE_DISPATCH=19"
    "dg4:dg4:-DDG_GROUP_SIZE=4"
    "gflip:gflip:-DTILE_DISPATCH=33"
    "tn2br:tn2br:-DTILE_DISPATCH=34"
    "zigzag:zigzag:-DTILE_DISPATCH=11"
    "rowmajor:rowmajor:-DTILE_DISPATCH=13"
)

log "=== Phase 1: building ${#VARIANTS[@]} variants with -DCOMBO_QUICK ==="
BUILD_OK=()
for entry in "${VARIANTS[@]}"; do
    IFS=: read -r name swizzle flags <<< "$entry"
    bin="$OUT/fc2-w3x-swizzle-$name"
    if $NVCC $CFLAGS -DCOMBO_QUICK $flags fc2_w3x.cu -o "$bin" $LDFLAGS \
            > "$OUT/build-$name.log" 2>&1; then
        BUILD_OK+=("$entry")
        regs=$(grep -oE 'Used [0-9]+ registers' "$OUT/build-$name.log" | head -1 | awk '{print $2}')
        log "  build OK   $name  flags='$flags'  regs=$regs"
    else
        log "  build FAIL $name  flags='$flags'  (see $OUT/build-$name.log)"
    fi
done
log "build summary: ${#BUILD_OK[@]} / ${#VARIANTS[@]} succeeded"

if [[ ${#BUILD_OK[@]} -lt 2 ]]; then
    log "ERROR: need ≥2 cells for ANOVA; aborting"
    exit 1
fi

log ""
log "=== Phase 2: $REPS reps/variant, pass-major interleaving ==="
for rep in $(seq 1 "$REPS"); do
    log "-- pass $rep/$REPS --"
    for entry in "${BUILD_OK[@]}"; do
        IFS=: read -r name swizzle flags <<< "$entry"
        bin="$OUT/fc2-w3x-swizzle-$name"
        if ! "$bin" > "$OUT/run-$name-$rep.log" 2>&1; then
            log "  pass $rep $name run-FAIL"
        fi
    done
done

log ""
log "=== Phase 3: extracting wall ms → $OUT/wall_data.csv ==="
{
    echo "variant,swizzle,rep,ms"
    for entry in "${BUILD_OK[@]}"; do
        IFS=: read -r name swizzle flags <<< "$entry"
        for rep in $(seq 1 "$REPS"); do
            f="$OUT/run-$name-$rep.log"
            if [[ -f "$f" ]]; then
                ms=$(grep -oE 'FC2-W3X kernel: [0-9.]+ ms' "$f" | head -1 | awk '{print $3}')
                if [[ -n "$ms" ]]; then
                    echo "$name,$swizzle,$rep,$ms"
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
    --factor swizzle \
    --out "$OUT/compare.txt"

log ""
log "report:           $OUT/compare.txt"
log "raw wall data:    $OUT/wall_data.csv"
log "raw per-run logs: $OUT/run-<cell>-<rep>.log"
