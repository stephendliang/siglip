#!/usr/bin/env bash
#
# NANOSLEEP_CYC grid search on fc2_w3x — single-binary in-process flow.
#
# fc2_w3x.cu's mbar_wait spin-wait body is now templated on NS_CYC (PTX
# immediate, baked per-template-instantiation, zero runtime cost — same
# scheme as TD/DGG).  VARIANT_TABLE has ten ns_* cells pinned to the
# basin-floor `gflip_blkswap` (TD=54, DGG=8): ns0 (busy-spin), ns4, ns8,
# ns12, ns16, ns20 (current default), ns24, ns32, ns48, ns64.
#
# This harness mirrors sweep_fc2_w3x_swizzle.sh: build one COMBO_QUICK
# binary, run with SWEEP=ns0,ns4,...,ns64 — the binary does the pass-major
# launch (1 warmup + 3 timed per pass) and emits @@SAMPLE lines.  No
# process-spawn overhead between cells.
#
# Default value set probes around current 20:
#   0   busy-spin control (no nap)
#   4   aggressive — fast wakeup, more polling
#   8, 12, 16   below current
#   20   current default (anchor)
#   24, 32   above current
#   48, 64   long nap, fewer polls
#
# Output: data/fc2_w3x_nanosleep_<ts>/
#   build.log         single COMBO_QUICK build
#   run.log           single binary run, all @@SAMPLE lines
#   wall_data.csv     variant,nanosleep,rep,ms,cyc
#   compare.txt       paired anova_1way.py (cyc, paired by rep, trim 0.33,
#                     anchor=ns20)
#
# Usage:
#   tools/sweep_fc2_w3x_nanosleep.sh                            # REPS=2048
#   REPS=10978 tools/sweep_fc2_w3x_nanosleep.sh                 # MODERATE-band n
#   SWEEP=ns0,ns8,ns20,ns48 REPS=512 tools/sweep_fc2_w3x_nanosleep.sh
#   tools/sweep_fc2_w3x_nanosleep.sh my_outdir
#
# Note: every cell in this sweep is the same dispatch (TD=54, DGG=8) and
# differs ONLY in the mbar_wait spin-wait nap duration.  σ_residual is
# expected to be ~1400 cyc (same as gflip_blkswap paired-pass on its own),
# so n<5000 won't resolve sub-µs differences.

set -u
cd "$(dirname "$0")/.."

OUT=${1:-"data/fc2_w3x_nanosleep_$(date +%Y%m%d_%H%M%S)"}
REPS=${REPS:-2048}
SWEEP=${SWEEP:-ns0,ns4,ns8,ns12,ns16,ns20,ns24,ns32,ns48,ns64}
ANCHOR=${ANCHOR:-ns20}
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
    echo "variant,nanosleep,rep,ms,cyc"
    grep -E '^@@SAMPLE pass=' "$OUT/run.log" | \
        sed -E 's/^@@SAMPLE pass=([0-9]+) variant=ns([0-9]+) ms=([0-9.]+) cyc=([0-9]+).*/ns\2,\2,\1,\3,\4/'
} > "$OUT/wall_data.csv"
n_rows=$(( $(wc -l < "$OUT/wall_data.csv") - 1 ))
log "  extracted $n_rows wall measurements"

log ""
log "=== Phase 4: paired anova (cyc, paired by rep, trim 0.33, anchor=$ANCHOR) ==="
python3 tools/anova_1way.py "$OUT/wall_data.csv" \
    --factor variant \
    --metric cyc \
    --paired rep \
    --trim 0.33 \
    --anchor "$ANCHOR" \
    --out "$OUT/compare.txt"

log ""
log "wall ANOVA:    $OUT/compare.txt"
log "raw csv:       $OUT/wall_data.csv"
log "single-run log:$OUT/run.log"
