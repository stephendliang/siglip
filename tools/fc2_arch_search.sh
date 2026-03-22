#!/bin/bash
# FC2 Architecture Search: compile + benchmark structural kernel variants on B200.
# Two-pass per config: timing build (cycle breakdown) then clean build (perf numbers).
#
# Base config locked from 2026-03-19 sweep (48 configs):
#   N_STAGES=5 (mandatory, 10% gap vs NS4)
#   INTERLEAVE_STRATEGY=1 (IS1≈IS2, pick one)
#   BATCH_MMA=0, PREFETCH_MBAR=0, OVERLAP_EPI_WAIT=0 (all noise)
#   W0_RES_FULL=0 (catastrophic), W0_RES_PREFETCH=0 (neutral)
#
# Usage:
#   ./tools/fc2_arch_search.sh              # run all configs
#   ./tools/fc2_arch_search.sh --only NAME  # run single config by name

set -uo pipefail
cd "$(dirname "$0")/.."

ONLY=""
if [ "${1:-}" = "--only" ]; then ONLY="$2"; shift 2; fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="data/fc2_arch_${TIMESTAMP}"
mkdir -p "$OUTDIR"

NVCC="nvcc"
CFLAGS="-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -lineinfo --ptxas-options=-v"
LDFLAGS="-lcurand -lcuda"
SRC="fc2.cu"
BIN="$OUTDIR/fc2_test"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/session.log"; }

log "========================================="
log "  FC2 ARCHITECTURE SEARCH $TIMESTAMP"
log "========================================="
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem --format=csv,noheader 2>/dev/null | head -1 | tee -a "$OUTDIR/session.log" || true

# ── Locked base (all axes confirmed from 48-config sweep 2026-03-19) ──
BASE="-DN_STAGES=5 -DINTERLEAVE_STRATEGY=1 -DBIAS_SMEM=1 -DPHASE1_UNROLL=1 -DPRELOAD_MODE=0 -DSTAGGER_CYCLES=0 -DTMA_RESIDUAL=1"

# ── Configs: NAME FLAGS ──
declare -a NAMES FLAGS
add() { NAMES+=("$1"); FLAGS+=("$2"); }

# Baseline (current best: 1.452ms / 3016 TFLOPS)
add "baseline" "$BASE"

# ── FP32 epilogue experiments ──
add "fp32_res"       "$BASE -DFP32_EPILOGUE=1"
add "fp32_full"      "$BASE -DFP32_EPILOGUE=2"
add "fp32_res_is2"   "$BASE -DFP32_EPILOGUE=1 -DINTERLEAVE_STRATEGY=2"
add "fp32_full_is2"  "$BASE -DFP32_EPILOGUE=2 -DINTERLEAVE_STRATEGY=2"
add "fp32_res_x64"   "$BASE -DFP32_EPILOGUE=1 -DTMEM_LOAD_WIDTH=64"
add "fp32_full_x64"  "$BASE -DFP32_EPILOGUE=2 -DTMEM_LOAD_WIDTH=64"

# ── Pre-combine bias+residual (9 ops/STS instead of 13, -15 regs) ──
add "precombine"     "$BASE -DPRE_COMBINE=1"
add "precombine_is2" "$BASE -DPRE_COMBINE=1 -DINTERLEAVE_STRATEGY=2"
add "precombine_x64" "$BASE -DPRE_COMBINE=1 -DTMEM_LOAD_WIDTH=64"

# ── EPI_NOINLINE (prevents double-inlining of epilogue_store, -480 insns) ──
add "noinline"             "$BASE -DEPI_NOINLINE=1"
add "noinline_precombine"  "$BASE -DEPI_NOINLINE=1 -DPRE_COMBINE=1"
add "noinline_x64"         "$BASE -DEPI_NOINLINE=1 -DTMEM_LOAD_WIDTH=64"
add "noinline_x64_pc"      "$BASE -DEPI_NOINLINE=1 -DTMEM_LOAD_WIDTH=64 -DPRE_COMBINE=1"

# ── StagesC=2 (W0 pre-loads residual, CUTLASS-style pipeline, requires NS4 for SMEM) ──
SC_BASE="-DN_STAGES=4 -DINTERLEAVE_STRATEGY=1 -DBIAS_SMEM=1 -DPHASE1_UNROLL=1 -DPRELOAD_MODE=0 -DSTAGGER_CYCLES=0 -DTMA_RESIDUAL=1 -DSTAGES_C=2 -DEPI_NOINLINE=1"
add "stages_c"             "$SC_BASE"
add "stages_c_pc"          "$SC_BASE -DPRE_COMBINE=1"
add "stages_c_x64"         "$SC_BASE -DTMEM_LOAD_WIDTH=64"
add "stages_c_x64_pc"      "$SC_BASE -DTMEM_LOAD_WIDTH=64 -DPRE_COMBINE=1"

# ── Filter ──
if [ -n "$ONLY" ]; then
    for i in $(seq 0 $((${#NAMES[@]} - 1))); do
        if [ "${NAMES[$i]}" = "$ONLY" ]; then
            NAMES=("${NAMES[$i]}")
            FLAGS=("${FLAGS[$i]}")
            break
        fi
    done
fi

INDICES=($(seq 0 $((${#NAMES[@]} - 1))))
log "${#INDICES[@]} configs"

# ── Results table ──
SUMMARY="$OUTDIR/summary.txt"
CSV="$OUTDIR/results.csv"
printf "%-32s %7s %8s %5s  %7s %7s %7s %7s %7s %7s %7s  %s\n" \
    "CONFIG" "ms" "TFLOPS" "REGS" \
    "epi_w" "tma0_w" "kloop" "total" \
    "e_ml_w" "e_ph1" "e_ph2" \
    "VERDICT" > "$SUMMARY"
printf "%s\n" "$(printf '%.0s-' {1..160})" >> "$SUMMARY"
echo "config,ms,tflops,regs,w1_epi_wait,w1_tma0_wait,w1_kloop,w1_total,epi_ml_wait,epi_phase1,epi_phase2,verdict" > "$CSV"

for idx in "${INDICES[@]}"; do
    name="${NAMES[$idx]}"
    flags="${FLAGS[$idx]}"
    log ""
    log "── $name ──"
    log "  $flags"

    # ── Timing build ──
    TOUT="$OUTDIR/timing_${name}.txt"
    COUT="$OUTDIR/compile_${name}.txt"
    cmd="$NVCC $CFLAGS -DTIMING $flags $SRC -o $BIN $LDFLAGS"
    if ! eval "$cmd" > "$COUT" 2>&1; then
        log "  COMPILE FAIL (timing)"
        grep -i error "$COUT" | head -3 | tee -a "$OUTDIR/session.log"
        continue
    fi
    regs=$(grep -oP 'Used \K\d+(?= registers)' "$COUT" | head -1)
    spills=$(grep -oP '\d+(?= bytes spill)' "$COUT" | head -1)
    log "  Regs: ${regs:-?}, Spills: ${spills:-0}"
    [ "${spills:-0}" -gt 0 ] && { log "  SKIP: spills"; continue; }

    if timeout 20 "$BIN" > "$TOUT" 2>&1; then
        log "  Timing OK"
    else
        log "  RUNTIME ERROR/TIMEOUT (timing)"
        tail -5 "$TOUT" 2>/dev/null | tee -a "$OUTDIR/session.log"
        continue
    fi

    # ── Clean perf build ──
    POUT="$OUTDIR/perf_${name}.txt"
    cmd="$NVCC $CFLAGS $flags $SRC -o $BIN $LDFLAGS"
    if eval "$cmd" > /dev/null 2>&1; then
        if timeout 20 "$BIN" > "$POUT" 2>&1; then
            log "  Perf OK"
        else
            log "  RUNTIME ERROR (perf)"
            cp "$TOUT" "$POUT"
        fi
    else
        cp "$TOUT" "$POUT"
    fi

    # ── Parse ──
    perf_ms=$(grep -oP 'FC2 kernel:\s+\K[\d.]+(?=\s+ms)' "$POUT" | head -1)
    perf_tflops=$(grep -oP '[\d.]+(?=\s+TFLOPS)' "$POUT" | head -1)
    w1_epi=$(grep -oP 'Epilogue mbar wait:\s+\K\d+(?=\s+cycles)' "$TOUT" | head -1)
    w1_tma0=$(grep -oP 'TMA stage-0 wait:\s+\K\d+(?=\s+cycles)' "$TOUT" | head -1)
    w1_kloop=$(grep -oP 'K-loop[^:]*:\s+\K\d+(?=\s+cycles)' "$TOUT" | head -1)
    w1_total=$(grep -oP 'Total tile:\s+\K\d+(?=\s+cycles)' "$TOUT" | head -1)
    epi_ml=$(grep -oP 'Mainloop mbar wait:\s+\K\d+(?=\s+cycles)' "$TOUT" | head -1)
    epi_p1=$(grep -oP 'Phase 1[^:]*:\s+\K\d+(?=\s+cycles)' "$TOUT" | head -1)
    epi_p2=$(grep -oP 'Phase 2[^:]*:\s+\K\d+(?=\s+cycles)' "$TOUT" | head -1)

    # Verdict
    verdict="?"
    if [ -n "$w1_epi" ] && [ -n "$w1_kloop" ]; then
        if [ "$w1_epi" -lt 100 ]; then verdict="COMPUTE-BOUND"
        elif [ "$w1_epi" -gt "$w1_kloop" ]; then verdict="EPILOGUE-BOUND"
        else verdict="BALANCED"
        fi
    fi

    log "  ms=${perf_ms:-?} TFLOPS=${perf_tflops:-?} regs=${regs:-?} verdict=${verdict}"

    printf "%-32s %7s %8s %5s  %7s %7s %7s %7s %7s %7s %7s  %s\n" \
        "$name" "${perf_ms:-?}" "${perf_tflops:-?}" "${regs:-?}" \
        "${w1_epi:-?}" "${w1_tma0:-?}" "${w1_kloop:-?}" "${w1_total:-?}" \
        "${epi_ml:-?}" "${epi_p1:-?}" "${epi_p2:-?}" "$verdict" >> "$SUMMARY"
    echo "$name,${perf_ms:-},${perf_tflops:-},${regs:-},${w1_epi:-},${w1_tma0:-},${w1_kloop:-},${w1_total:-},${epi_ml:-},${epi_p1:-},${epi_p2:-},$verdict" >> "$CSV"

    # Full timing analysis
    [ -n "$w1_epi" ] && python3 tools/analyze_timing.py "$TOUT" --ref-tflops 3564 \
        > "$OUTDIR/analysis_${name}.txt" 2>&1 || true
done

# ── Summary ──
log ""
log "========================================="
log "  RESULTS"
log "========================================="
cat "$SUMMARY" | tee -a "$OUTDIR/session.log"
log ""
log "Per-config analysis: $OUTDIR/analysis_*.txt"
log "CSV: $CSV"

rm -f "$BIN"
log "Done. Output: $OUTDIR"
