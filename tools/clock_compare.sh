#!/bin/bash
# Compare clock timing between default dispatch and TD=4 scheduler.
# Builds both variants with CLOCK_TIMING, runs them, saves output.
#
# Usage:
#   bash tools/clock_compare.sh           # default dims
#   bash tools/clock_compare.sh K=4096    # custom K (tests crossover)
#   bash tools/clock_compare.sh K=6144    # where TD=4 wins

set -uo pipefail
cd "$(dirname "$0")/.."

DFLAGS=""
K_LABEL=""
for arg in "$@"; do
    case "$arg" in
        K=*) val="${arg#K=}"; DFLAGS="$DFLAGS -DK_DIM=$val"; K_LABEL="_K${val}";;
        M=*) val="${arg#M=}"; DFLAGS="$DFLAGS -DM_TOTAL=$val";;
        N=*) val="${arg#N=}"; DFLAGS="$DFLAGS -DN_DIM=$val";;
    esac
done

OUTDIR="data/clock${K_LABEL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/session.log"; }

log "=== CLOCK TIMING COMPARISON ==="
log "Output: $OUTDIR"
log "DFLAGS: ${DFLAGS:-none}"

nvidia-smi --query-gpu=gpu_name,clocks.sm --format=csv,noheader 2>/dev/null | tee -a "$OUTDIR/session.log"

# Build
log "Building default fused (clock)..."
make -B fc2-w3-clock ${DFLAGS:+DFLAGS="$DFLAGS"} 2>&1 | tail -1
log "Building default strip (clock)..."
make -B fc2-w3-strip-clock ${DFLAGS:+DFLAGS="$DFLAGS"} 2>&1 | tail -1
log "Building sched fused (clock)..."
make -B fc2-w3-sched-clock ${DFLAGS:+DFLAGS="$DFLAGS"} 2>&1 | tail -1

# Sched strip needs manual build (strip + TD=4 + clock)
log "Building sched strip (clock)..."
make -B fc2-w3-sched-clock DFLAGS="-DTILE_DISPATCH=4 -DSTRIP_EPILOGUE -DCLOCK_TIMING ${DFLAGS}" 2>&1 | tail -1
cp fc2-w3-sched-clock fc2-w3-sched-strip-clock
make -B fc2-w3-sched-clock ${DFLAGS:+DFLAGS="$DFLAGS"} 2>&1 | tail -1

# Run
run_exp() {
    local name="$1" bin="$2"
    if [ ! -x "$bin" ]; then
        log "SKIP $name — $bin not found"
        return
    fi
    log "Running $name..."
    ./"$bin" > "$OUTDIR/${name}.txt" 2>&1
    local ms valid
    ms=$(grep -oP '[\d.]+(?=\s*ms)' "$OUTDIR/${name}.txt" | head -1)
    valid=$(grep -oP 'valid=\d' "$OUTDIR/${name}.txt" | head -1)
    log "  $name: ${ms}ms  ${valid}"
}

log ""
log "=== RUNNING ==="
run_exp "default_fused"       "fc2-w3-clock"
run_exp "default_strip"       "fc2-w3-strip-clock"
run_exp "sched_fused"         "fc2-w3-sched-clock"
run_exp "sched_strip"         "fc2-w3-sched-strip-clock"

log ""
log "=== TIMING DATA ==="
for name in default_fused default_strip sched_fused sched_strip; do
    f="$OUTDIR/${name}.txt"
    [ -f "$f" ] || continue
    log ""
    log "--- $name ---"
    # Print the CLOCK TIMING section
    sed -n '/=== CLOCK TIMING/,/^$/p' "$f" | tee -a "$OUTDIR/session.log"
done

log ""
log "=== SUMMARY ==="
for name in default_fused default_strip sched_fused sched_strip; do
    f="$OUTDIR/${name}.txt"
    [ -f "$f" ] || continue
    ms=$(grep -oP '[\d.]+(?=\s*ms)' "$f" | head -1)
    valid=$(grep -oP 'valid=\d' "$f" | head -1)
    printf "%-20s %s ms  %s\n" "$name" "${ms:-N/A}" "${valid:-N/A}" | tee -a "$OUTDIR/session.log"
done

log ""
log "Done. Full output in $OUTDIR/"
