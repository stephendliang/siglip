#!/bin/bash
# Compare clock timing between dispatch modes: default, TD=4, COL_LOCK.
# Builds all variants with CLOCK_TIMING, runs them, saves output.
#
# Usage:
#   bash tools/clock_compare.sh                # default dims (K=3072)
#   bash tools/clock_compare.sh K=4096         # custom K (tests crossover)
#   bash tools/clock_compare.sh K=6144         # where TD=4 wins
#   bash tools/clock_compare.sh --strip-only   # skip fused builds

set -uo pipefail
cd "$(dirname "$0")/.."

DFLAGS=""
K_LABEL=""
K_VAL=3072
STRIP_ONLY=0
for arg in "$@"; do
    case "$arg" in
        K=*) K_VAL="${arg#K=}"; DFLAGS="$DFLAGS -DK_DIM=$K_VAL"; K_LABEL="_K${K_VAL}";;
        M=*) val="${arg#M=}"; DFLAGS="$DFLAGS -DM_TOTAL=$val";;
        N=*) val="${arg#N=}"; DFLAGS="$DFLAGS -DN_DIM=$val";;
        --strip-only) STRIP_ONLY=1;;
    esac
done

# Auto NO_PREFILL for short K-loops (PREFILL deadlocks at K_ITERS<20)
K_ITERS=$((K_VAL / 128))
if [ "$K_ITERS" -lt 20 ]; then
    DFLAGS="$DFLAGS -DNO_PREFILL"
    echo "Auto-adding -DNO_PREFILL (K_ITERS=$K_ITERS < 20)"
fi

OUTDIR="data/clock${K_LABEL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/session.log"; }

log "=== CLOCK TIMING COMPARISON ==="
log "Output: $OUTDIR"
log "DFLAGS: ${DFLAGS:-none}"

nvidia-smi --query-gpu=gpu_name,clocks.sm --format=csv,noheader 2>/dev/null | tee -a "$OUTDIR/session.log"

# Build all variants
build() {
    local name="$1"; shift
    log "Building $name..."
    "$@" 2>&1 | tail -1
}

build "default fused (clock)"   make -B fc2-w3-clock ${DFLAGS:+DFLAGS="$DFLAGS"}
build "default strip (clock)"   make -B fc2-w3-strip-clock ${DFLAGS:+DFLAGS="$DFLAGS"}
build "sched fused (clock)"     make -B fc2-w3-sched-clock ${DFLAGS:+DFLAGS="$DFLAGS"}

# Sched strip: TD=4 + STRIP + CLOCK
build "sched strip (clock)"     make -B fc2-w3-sched-clock DFLAGS="-DTILE_DISPATCH=4 -DSTRIP_EPILOGUE -DCLOCK_TIMING ${DFLAGS}"
cp fc2-w3-sched-clock fc2-w3-sched-strip-clock
build "sched fused (clock, rebuild)" make -B fc2-w3-sched-clock ${DFLAGS:+DFLAGS="$DFLAGS"}

# TD=7 inline atomic variants
build "inline7 fused (clock)"   make -B fc2-w3-inline7-clock ${DFLAGS:+DFLAGS="$DFLAGS"}

# Inline7 strip: TD=7 + STRIP + CLOCK
build "inline7 strip (clock)"   make -B fc2-w3-inline7-clock DFLAGS="-DTILE_DISPATCH=7 -DSTRIP_EPILOGUE -DCLOCK_TIMING ${DFLAGS}"
cp fc2-w3-inline7-clock fc2-w3-inline7-strip-clock
build "inline7 fused (clock, rebuild)" make -B fc2-w3-inline7-clock ${DFLAGS:+DFLAGS="$DFLAGS"}

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
run_exp "inline7_fused"       "fc2-w3-inline7-clock"
run_exp "inline7_strip"       "fc2-w3-inline7-strip-clock"

log ""
log "=== TIMING DATA ==="
for name in default_fused default_strip sched_fused sched_strip inline7_fused inline7_strip; do
    f="$OUTDIR/${name}.txt"
    [ -f "$f" ] || continue
    log ""
    log "--- $name ---"
    # Capture from CLOCK TIMING to end of output
    sed -n '/=== CLOCK TIMING/,$p' "$f" | tee -a "$OUTDIR/session.log"
done

log ""
log "=== SUMMARY ==="
printf "%-20s %10s %10s\n" "Variant" "ms" "valid" | tee -a "$OUTDIR/session.log"
printf "%-20s %10s %10s\n" "-------" "--" "-----" | tee -a "$OUTDIR/session.log"
for name in default_fused default_strip sched_fused sched_strip inline7_fused inline7_strip; do
    f="$OUTDIR/${name}.txt"
    [ -f "$f" ] || continue
    ms=$(grep -oP '[\d.]+(?=\s*ms)' "$f" | head -1)
    valid=$(grep -oP 'valid=\d' "$f" | head -1)
    printf "%-20s %10s %10s\n" "$name" "${ms:-N/A}" "${valid:-N/A}" | tee -a "$OUTDIR/session.log"
done

log ""
log "Done. Full output in $OUTDIR/"
