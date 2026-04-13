#!/bin/bash
# Quick test: SASS inter-warp YIELD stagger on fc2_w3 epilogue.
#
# Replaces epilogue NOPs with YIELD instructions to break cross-warp
# STS clustering. Tests both default dispatch and TD=4 scheduler.
#
# Run on B200:
#   bash tools/test_yield_stagger.sh
#   bash tools/test_yield_stagger.sh --dry-run   # see patch plan only

set -uo pipefail
cd "$(dirname "$0")/.."

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

OUTDIR="data/yield_stagger_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/session.log"; }

log "=== YIELD STAGGER TEST ==="
log "Output: $OUTDIR"

if [ "$DRY" = "0" ]; then
    nvidia-smi --query-gpu=gpu_name,clocks.sm --format=csv,noheader 2>/dev/null | tee -a "$OUTDIR/session.log"
fi

# --- Build baselines ---
log "Building baselines..."
make fc2-w3 2>&1 | tail -1
make fc2-w3-sched 2>&1 | tail -1
make fc2-w3-strip 2>&1 | tail -1
make -B fc2-w3-sched DFLAGS='-DTILE_DISPATCH=4 -DSTRIP_EPILOGUE' 2>&1 | tail -1
cp fc2-w3-sched fc2-w3-sched-strip  # save strip variant before rebuild
make fc2-w3-sched 2>&1 | tail -1    # rebuild fused

# --- Dump SASS ---
log "Dumping SASS..."
cuobjdump --dump-sass fc2-w3 > "$OUTDIR/sass_default.txt" 2>/dev/null
cuobjdump --dump-sass fc2-w3-sched > "$OUTDIR/sass_sched.txt" 2>/dev/null

# --- Patch: yield-only (all warps, @PT) ---
log "Patching yield-only (all warps)..."
python3 tools/interwarp_stagger.py fc2-w3 \
    --sass "$OUTDIR/sass_default.txt" \
    --mode yield-only \
    -o fc2-w3-yield 2>&1 | tee "$OUTDIR/patch_default_yield.log"

python3 tools/interwarp_stagger.py fc2-w3-strip \
    --sass "$OUTDIR/sass_default.txt" \
    --mode yield-only \
    -o fc2-w3-yield-strip 2>&1 | tee "$OUTDIR/patch_default_yield_strip.log"

python3 tools/interwarp_stagger.py fc2-w3-sched \
    --sass "$OUTDIR/sass_sched.txt" \
    --mode yield-only \
    -o fc2-w3-sched-yield 2>&1 | tee "$OUTDIR/patch_sched_yield.log"

python3 tools/interwarp_stagger.py fc2-w3-sched-strip \
    --sass "$OUTDIR/sass_sched.txt" \
    --mode yield-only \
    -o fc2-w3-sched-yield-strip 2>&1 | tee "$OUTDIR/patch_sched_yield_strip.log"

if [ "$DRY" = "1" ]; then
    log "Dry run — not executing binaries"
    exit 0
fi

# --- Run experiments ---
run_exp() {
    local name="$1" bin="$2"
    if [ ! -x "$bin" ]; then
        log "SKIP $name — $bin not found"
        return
    fi
    log "Running $name..."
    local out
    out=$(./"$bin" 2>&1)
    echo "$out" > "$OUTDIR/${name}.txt"
    # Extract timing line
    local ms
    ms=$(echo "$out" | grep -oP '[\d.]+\s*ms' | head -1)
    local valid
    valid=$(echo "$out" | grep -oP 'valid=\d' | head -1)
    log "  $name: $ms  $valid"
}

log ""
log "=== RUNNING EXPERIMENTS ==="

# Baselines
run_exp "default_fused"       "fc2-w3"
run_exp "default_strip"       "fc2-w3-strip"
run_exp "sched_fused"         "fc2-w3-sched"
run_exp "sched_strip"         "fc2-w3-sched-strip"

# Yield-patched
run_exp "default_yield_fused" "fc2-w3-yield"
run_exp "default_yield_strip" "fc2-w3-yield-strip"
run_exp "sched_yield_fused"   "fc2-w3-sched-yield"
run_exp "sched_yield_strip"   "fc2-w3-sched-yield-strip"

# --- Summary ---
log ""
log "=== SUMMARY ==="
log "Variant                  | ms       | valid"
log "-------------------------|----------|------"
for f in "$OUTDIR"/*.txt; do
    name=$(basename "$f" .txt)
    ms=$(grep -oP '[\d.]+\s*ms' "$f" | head -1)
    valid=$(grep -oP 'valid=\d' "$f" | head -1)
    printf "%-25s| %-9s| %s\n" "$name" "$ms" "$valid" | tee -a "$OUTDIR/session.log"
done

log ""
log "Done. Full output in $OUTDIR/"
