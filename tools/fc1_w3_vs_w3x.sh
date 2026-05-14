#!/bin/bash
#
# tools/fc1_w3_vs_w3x.sh
#
# Head-to-head perf compare of fc1_w3 (legacy) vs fc1_w3x (clean-sheet port).
# Plan verification rule: at default tune, fc1_w3x should match fc1_w3
# within ~5 µs. The first B200 run of fc1_w3x landed at 3.117 ms vs
# fc1_w3 baseline ~2.093 ms — this script reproduces and quantifies that
# gap, and isolates fused vs strip vs gemm-only to localize the regression.
#
# Usage:
#   bash tools/fc1_w3_vs_w3x.sh                 # 3 reps, fused only
#   bash tools/fc1_w3_vs_w3x.sh --reps 5        # 5 reps
#   bash tools/fc1_w3_vs_w3x.sh --diag          # + strip + gemm variants
#   bash tools/fc1_w3_vs_w3x.sh --spill-check   # cuobjdump | grep STL on each
#   bash tools/fc1_w3_vs_w3x.sh DFLAGS=...      # forward extra compile flags

set -uo pipefail
cd "$(dirname "$0")/.."

REPS=3
DIAG=0
SPILL=0
DFLAGS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --reps) REPS="$2"; shift 2;;
        --diag) DIAG=1; shift;;
        --spill-check) SPILL=1; shift;;
        DFLAGS=*) DFLAGS="${1#DFLAGS=}"; shift;;
        -h|--help)
            sed -n '3,20p' "$0"; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

OUTDIR="data/fc1_w3_vs_w3x_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/session.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== FC1 W3 vs W3X PERF COMPARE ==="
log "reps=$REPS  diag=$DIAG  spill=$SPILL  DFLAGS='${DFLAGS:-}'  out=$OUTDIR"
nvidia-smi --query-gpu=gpu_name,clocks.sm,temperature.gpu --format=csv,noheader 2>/dev/null | tee -a "$LOG" || true

build_one() {
    local target="$1"
    log "--- build $target ---"
    rm -f "$target"
    if [ -n "$DFLAGS" ]; then
        make -B "$target" DFLAGS="$DFLAGS" 2>&1 | tail -3 | tee -a "$LOG"
    else
        make -B "$target" 2>&1 | tail -3 | tee -a "$LOG"
    fi
    if [ ! -x "$target" ]; then
        log "BUILD FAILED: $target"
        return 1
    fi
    if [ "$SPILL" = 1 ]; then
        local stl
        stl=$(cuobjdump "$target" --dump-sass 2>/dev/null | grep -cE '\bSTL\b' || true)
        local ldl
        ldl=$(cuobjdump "$target" --dump-sass 2>/dev/null | grep -cE '\bLDL\b' || true)
        log "  SASS  STL=$stl  LDL=$ldl  (local-mem spill; want 0)"
    fi
}

run_one() {
    local target="$1"
    local label="$2"
    local outfile="$OUTDIR/$label.out"
    log "--- run $target (reps=$REPS) ---"
    : > "$outfile"
    for r in $(seq 1 "$REPS"); do
        echo "@@REP $r" >> "$outfile"
        ./"$target" 2>&1 | tee -a "$outfile" | grep -E '^(@@SAMPLE|@@RESULT)' | sed "s/^/  r$r /"
    done
}

build_and_run() {
    local target="$1"; local label="$2"
    build_one "$target" || return
    run_one "$target" "$label"
}

build_and_run fc1-w3   fc1_w3
build_and_run fc1-w3x  fc1_w3x

if [ "$DIAG" = 1 ]; then
    build_and_run fc1-w3-strip   fc1_w3_strip
    build_and_run fc1-w3x-strip  fc1_w3x_strip
    build_and_run fc1-w3-gemm    fc1_w3_gemm
    build_and_run fc1-w3x-gemm   fc1_w3x_gemm
fi

log ""
log "=== SUMMARY (ms, sorted by min) ==="
python3 - "$OUTDIR" <<'PY' | tee -a "$LOG"
import sys, re, os, glob
out = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(out, '*.out'))):
    label = os.path.basename(f)[:-4]
    samples = []
    for line in open(f):
        m = re.search(r'@@SAMPLE.* ms=([\d.]+)', line)
        if m: samples.append(float(m.group(1)))
    if not samples:
        rows.append((label, None, None, None, None, None, 0)); continue
    samples.sort()
    mn = samples[0]; mx = samples[-1]
    md = samples[len(samples)//2]
    mean = sum(samples)/len(samples)
    rows.append((label, mn, md, mean, mx, mx-mn, len(samples)))

print(f"  {'variant':<20}  {'n':>3}  {'min':>8}  {'med':>8}  {'mean':>8}  {'max':>8}  {'spread':>7}")
for label, mn, md, mean, mx, sp, n in rows:
    if mn is None:
        print(f"  {label:<20}  n=0  (no samples)")
    else:
        print(f"  {label:<20}  {n:>3}  {mn:8.4f}  {md:8.4f}  {mean:8.4f}  {mx:8.4f}  {sp:7.4f}")
PY

log ""
log "=== DELTA (w3x - w3, min over reps) ==="
python3 - "$OUTDIR" <<'PY' | tee -a "$LOG"
import sys, re, os
out = sys.argv[1]
def best(fname):
    f = os.path.join(out, fname)
    if not os.path.exists(f): return None
    vs = [float(m.group(1)) for line in open(f)
          for m in [re.search(r'@@SAMPLE.* ms=([\d.]+)', line)] if m]
    return min(vs) if vs else None
for tag, name in [('', 'fused'), ('_strip', 'strip'), ('_gemm', 'gemm')]:
    w3 = best(f'fc1_w3{tag}.out')
    w3x = best(f'fc1_w3x{tag}.out')
    if w3 is None or w3x is None: continue
    delta_ms = w3x - w3
    delta_us = delta_ms * 1000.0
    pct = 100.0 * delta_ms / w3 if w3 else 0
    if abs(delta_us) < 5:
        flag = 'TIE (plan verification: PASS)'
    elif delta_ms < 0:
        flag = 'WIN'
    elif delta_us < 100:
        flag = 'MINOR REGRESSION'
    else:
        flag = 'MAJOR REGRESSION'
    print(f"  {name:<6}  w3={w3:.4f}  w3x={w3x:.4f}  delta={delta_ms:+.4f} ms ({delta_us:+7.1f} µs, {pct:+5.2f}%)  [{flag}]")

print()
print("  Plan rule: fc1_w3x default should match fc1_w3 default within ~5 µs.")
print("  Anything beyond that is a bug, not tuning.")
PY

log "done. log: $LOG"
