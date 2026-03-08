#!/bin/bash
# Automated B200 rental session. Run in tmux. All output to data/session_*/
#
# Phases:
#   1. Machine snapshot                          (~5 sec)
#   2. Build + grid search + benchmark + ANOVA   (~30-90 min with grid search)
#   3. ncu profiling (source counters + full)     (~5 min)
#   4. cuBLAS SASS capture (per-layer JIT dump)   (~2 min)
#
# Usage:
#   tmux new -s b200
#   ./tools/b200_session.sh                    # full session
#   ./tools/b200_session.sh --phase 2          # single phase
#
# Monitor from another pane:
#   tail -f data/session_*/compare.txt
#   tail -f data/session_*/grid_search_*.log

set -uo pipefail
cd "$(dirname "$0")/.."

PHASE_ONLY=0
COMPARE_ARGS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --phase) PHASE_ONLY="$2"; shift 2 ;;
        *) COMPARE_ARGS="$COMPARE_ARGS $1"; shift ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="data/session_${TIMESTAMP}"
mkdir -p "$OUTDIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/session.log"; }
should_run() { [ "$PHASE_ONLY" = "0" ] || [ "$PHASE_ONLY" = "$1" ]; }

SESSION_START=$(date +%s)
log "========================================"
log "  B200 SESSION $TIMESTAMP"
log "  Output: $OUTDIR"
log "  Phase:  ${PHASE_ONLY:-all}"
log "========================================"

# ── Preflight ──
python3 -c "import numpy; import scipy" 2>/dev/null || \
    { log "FATAL: pip install numpy scipy"; exit 1; }
nvidia-smi > /dev/null 2>&1 || \
    { log "FATAL: no GPU detected"; exit 1; }
log "Preflight OK"

# ── Phase 1/4: Machine snapshot (~5 sec) ──
if should_run 1; then
    log ""
    log "==== PHASE 1/4: MACHINE SNAPSHOT ===="
    {
        echo "--- nvidia-smi ---"
        nvidia-smi
        echo ""
        echo "--- clocks ---"
        nvidia-smi -q -d CLOCK 2>/dev/null || true
        echo ""
        echo "--- nvcc --version ---"
        nvcc --version
        echo ""
        echo "--- git ---"
        git rev-parse HEAD
        git diff --stat
    } > "$OUTDIR/machine_info.txt" 2>&1
    log "Machine info saved to machine_info.txt"
fi

# ── Phase 2/4: Build + grid search + benchmark (~30-90 min) ──
if should_run 2; then
    log ""
    log "==== PHASE 2/4: BUILD + GRID SEARCH + BENCHMARK ===="
    log "This is the long phase. Grid search: ~673 configs/layer."
    log "Monitor: tail -f $OUTDIR/compare.txt"
    log "Grid search logs: tail -f $OUTDIR/grid_search_*.log"
    python3 tools/compare_all.py \
        --runs 20 \
        --cutlass-mode max \
        --grid-search \
        --csv "$OUTDIR/compare.csv" \
        $COMPARE_ARGS \
        2>&1 | tee "$OUTDIR/compare.txt" || log "WARN Phase 2 exited non-zero"
    log "Phase 2 complete"
fi

# ── Phase 3/4: ncu profiling (~5 min) ──
if should_run 3; then
    log ""
    log "==== PHASE 3/4: NCU PROFILING ===="
    for bin in ./patch_embed ./fc1-gelu ./fc2; do
        name=$(basename "$bin")
        if [ ! -x "$bin" ]; then
            log "SKIP ncu $name: binary not found (phase 2 builds these)"
            continue
        fi
        bin_mtime=$(stat -c %Y "$bin" 2>/dev/null || stat -f %m "$bin" 2>/dev/null)
        if [ -n "$bin_mtime" ] && [ "$bin_mtime" -lt "$SESSION_START" ]; then
            log "WARN ncu $name: binary predates this session — may be stale"
        fi
        log "ncu source counters: $name"
        ncu --set source --csv "$bin" > "$OUTDIR/source_counters_${name}.csv" 2>&1 || \
            log "FAIL ncu $name (may need root or --mode=launch)"
    done
    log "ncu full profile: patch_embed"
    ncu --set full -o "$OUTDIR/siglip_full" ./patch_embed 2>&1 || \
        log "FAIL ncu full (may need root or --mode=launch)"
    log "Phase 3 complete"
fi

# ── Phase 4/4: cuBLAS SASS capture (~2 min) ──
if should_run 4; then
    log ""
    log "==== PHASE 4/4: CUBLAS SASS CAPTURE ===="
    for t in cublas-bench cublas-bench-fc1 cublas-bench-fc2; do
        if ! make "$t" >> "$OUTDIR/session.log" 2>&1; then
            log "FAIL build $t"
            continue
        fi
        if [ -d ~/.nv/ComputeCache ]; then
            rm -rf ~/.nv/ComputeCache/*
        fi
        "./$t" > "$OUTDIR/${t}.txt" 2>&1 || { log "FAIL run $t"; continue; }
        cubins_found=0
        while IFS= read -r cubin; do
            base=$(basename "$cubin" .cubin)
            if cuobjdump --dump-sass "$cubin" > "$OUTDIR/cublas_sass_${t}_${base}.txt" 2>&1; then
                log "SASS captured: $t / $base"
            else
                log "FAIL cuobjdump on $t / $base"
            fi
            cubins_found=$((cubins_found + 1))
        done < <(find ~/.nv/ComputeCache/ -name '*.cubin' 2>/dev/null)
        if [ "$cubins_found" = 0 ]; then
            log "WARN no cubins found for $t"
        fi
    done
    log "Phase 4 complete"
fi

ELAPSED=$(( $(date +%s) - SESSION_START ))
log ""
log "========================================"
log "  SESSION COMPLETE: ${ELAPSED}s ($(( ELAPSED / 60 )) min)"
log "  Output: $OUTDIR"
log "========================================"
ls -lh "$OUTDIR/" | tee -a "$OUTDIR/session.log"
echo ""
echo "Analyze: python3 tools/analyze_session.py $OUTDIR"
