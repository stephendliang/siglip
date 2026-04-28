#!/usr/bin/env bash
#
# 1-way swizzle sweep on fc2_w3x via the templated single-binary flow
# (refactor 2026-04-27): one build, one binary, SWEEP env drives the
# active variant list, REPS env drives pass-major outer iterations.
#
# All 20 cells exist as explicit template instantiations of
# fc2_w3x_kernel<TD, DGG> in the binary — each compiles to its own SASS,
# byte-identical to the per-build flow that preceded the refactor.
# The host loops pass-major: pass p ∈ 1..REPS × variant v ∈ ACTIVE,
# emitting one `@@SAMPLE pass=p variant=v ms=X` line per launch.
# Cross-cell drift cancels at the granularity of one pass instead of
# one whole-cell run.
#
# Variants in the sweep (matches main()'s VARIANT_TABLE — see fc2_w3x.cu):
#   dgsw / dg2 / dg4 / dg16 / dg32 — DG_GROUP_SIZE curve
#   zigzag / rowmajor              — reference outliers (TD=11/13)
#   checkered / dgsnake            — structural alternatives (TD=18/19)
#   gflip / tn2br                  — front-tier members (TD=33/34)
#   dg4diag / dg4pp                — dg4 + within-group probes (TD=35/36, DGG=4)
#   g4swap / lmrev                 — analyzer-driven (TD=38/39)
#   comboAB / comboAC              — tnRun-collapse falsifiers (TD=40/41)
#   snrot1 / snrot2 / lmsn         — tnRun-preserving (TD=43/44/45)
#
# History: TD=37 rowmaj and TD=42 tnblk are still in the binary as
# documented dead-ends but are not in this sweep (n=4 confirmed losers
# at +3.0 µs / +444 µs).  INGH=32 dropped earlier (n=20 at +3.35 µs).
#
# Output: data/fc2_w3x_swizzle_<ts>/
#   build.log
#   run.log            (single binary run, all @@SAMPLE lines)
#   wall_data.csv      (variant,swizzle,rep,ms)
#   compare.txt        (per-cell stats + 1-way ANOVA + pairwise Welch)
#
# Usage:
#   tools/sweep_fc2_w3x_swizzle.sh                         # REPS=20
#   REPS=64 tools/sweep_fc2_w3x_swizzle.sh
#   SWEEP=dgsw,dg4,lmsn REPS=128 tools/sweep_fc2_w3x_swizzle.sh
#   tools/sweep_fc2_w3x_swizzle.sh my_outdir

set -u
cd "$(dirname "$0")/.."

OUT=${1:-"data/fc2_w3x_swizzle_$(date +%Y%m%d_%H%M%S)"}
REPS=${REPS:-20}
SWEEP=${SWEEP:-all}
NVCC=${NVCC:-nvcc}
CFLAGS='-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -lineinfo --ptxas-options=-v --cudart=static'
LDFLAGS='-lcurand_static -lculibos -lcuda'

mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/run.log"; }

BIN="$OUT/fc2-w3x-sweep"

log "=== Phase 1: building single binary with -DCOMBO_QUICK ==="
if ! $NVCC $CFLAGS -DCOMBO_QUICK fc2_w3x.cu -o "$BIN" $LDFLAGS \
        > "$OUT/build.log" 2>&1; then
    log "ERROR: build FAIL — see $OUT/build.log"
    exit 1
fi
n_kerns=$(grep -cE 'Compiling entry function' "$OUT/build.log" || true)
log "  build OK  ($n_kerns kernel instantiations)"

log ""
log "=== Phase 2: SWEEP=$SWEEP, REPS=$REPS, pass-major in one process ==="
if ! REPS="$REPS" SWEEP="$SWEEP" "$BIN" > "$OUT/run.log" 2>&1; then
    log "ERROR: run FAIL — see $OUT/run.log"
    exit 1
fi
n_samples=$(grep -cE '^@@SAMPLE pass=' "$OUT/run.log" || true)
n_variants=$(grep -oE 'variant=[^ ]+' "$OUT/run.log" | sort -u | wc -l)
log "  $n_samples samples across $n_variants variants"

log ""
log "=== Phase 3: extracting @@SAMPLE → $OUT/wall_data.csv ==="
{
    echo "variant,swizzle,rep,ms"
    grep -E '^@@SAMPLE pass=' "$OUT/run.log" | \
        sed -E 's/^@@SAMPLE pass=([0-9]+) variant=([^ ]+) ms=([0-9.]+).*/\2,\2,\1,\3/'
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
log "single-run log:   $OUT/run.log"
