#!/usr/bin/env bash
#
# Swizzle basin sweep for the FUSED fc2_w3 kernel (residual path).
#
# fc2_w3 is now templated on <TD, DGG> exactly like fc2_w3x, so the whole basin
# compiles into one -DCOMBO_QUICK binary (make fc2-w3-swizzle-sweep).  The host
# loops pass-major (outer pass p, inner variant v) and emits one
#   @@SAMPLE pass=p variant=v ms=X cyc=Y
# per sample.  cyc comes from the CLOCK_TOTAL ticker (atomicMax of the per-CTA
# clock64 span) — frequency-invariant and immune to fc2_w3's 1.4 GB residual
# HBM-contention ms noise, which is exactly why the ms-only timing was useless
# on shared Modal B200s.
#
# Variants (W3_VARIANTS in fc2_w3.cu): stride(TD=0) dgsw(8) dg4(8,DGG=4)
# zigzag(11) dgsnake(19) gflip(33) lmrev(39) snrot2(44) gflip_lmrev(52)
# gflip_snrot(53) gflip_blkswap(54) gflip_blklmrev(56) gflip_blkmul3(57)
# gflip_quartswap(58).  SWEEP=front drops dg4 for the curated basin set.
#
# Two modes:
#   1. Local GPU box: builds + runs + analyzes.
#        tools/sweep_fc2_w3_swizzle.sh                       # SWEEP=all REPS=20
#        SWEEP=front REPS=200 tools/sweep_fc2_w3_swizzle.sh
#   2. No local GPU (this VPS): run on Modal, then analyze the saved log:
#        modal run gpu_interface/modal.py --target fc2-w3-swizzle-sweep \
#            --run-args "SWEEP=front REPS=200" > /tmp/w3sweep.log 2>&1
#        ANALYZE=/tmp/w3sweep.log tools/sweep_fc2_w3_swizzle.sh

set -u
cd "$(dirname "$0")/.."

OUT=${1:-"data/fc2_w3_swizzle_$(date +%Y%m%d_%H%M%S)"}
REPS=${REPS:-20}
SWEEP=${SWEEP:-all}
ANALYZE=${ANALYZE:-}
mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

RUNLOG="$OUT/run.log"
if [ -n "$ANALYZE" ]; then
    log "=== analyze-only: $ANALYZE ==="
    cp "$ANALYZE" "$RUNLOG"
else
    NVCC=${NVCC:-nvcc}
    CFLAGS='-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -lineinfo --ptxas-options=-v --cudart=static'
    LDFLAGS='-lcurand_static -lculibos -lcuda'
    BIN="$OUT/fc2-w3-swizzle-sweep"
    log "=== build single COMBO_QUICK binary ==="
    if ! $NVCC $CFLAGS -DCOMBO_QUICK fc2_w3.cu -o "$BIN" $LDFLAGS > "$OUT/build.log" 2>&1; then
        log "ERROR: build FAIL — see $OUT/build.log"; exit 1
    fi
    log "=== run SWEEP=$SWEEP REPS=$REPS ==="
    if ! "$BIN" "SWEEP=$SWEEP" "REPS=$REPS" > "$RUNLOG" 2>&1; then
        log "ERROR: run FAIL — see $RUNLOG"; exit 1
    fi
fi

log "=== extract @@SAMPLE → $OUT/wall_data.csv ==="
{
    echo "variant,swizzle,rep,ms,cyc"
    grep -E '^@@SAMPLE pass=' "$RUNLOG" | \
        sed -E 's/^@@SAMPLE pass=([0-9]+) variant=([^ ]+) ms=([0-9.]+) cyc=([0-9]+).*/\2,\2,\1,\3,\4/'
} > "$OUT/wall_data.csv"
n=$(( $(wc -l < "$OUT/wall_data.csv") - 1 ))
log "extracted $n samples"

if [ "$n" -gt 0 ]; then
    log "=== cyc ANOVA (paired by pass, trim 33%) → $OUT/compare.txt ==="
    python3 tools/anova_1way.py "$OUT/wall_data.csv" \
        --factor swizzle --metric cyc --paired rep --trim 0.33 \
        --anchor "${ANCHOR:-dgsw}" --out "$OUT/compare.txt"
    log "compare:  $OUT/compare.txt"
fi
log "raw log:  $RUNLOG"
log "csv:      $OUT/wall_data.csv"
