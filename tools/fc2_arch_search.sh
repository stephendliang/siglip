#!/bin/bash
# FC2 Architecture Search: test structural kernel variants on B200.
# Each config: compile with -DTIMING (cycle breakdown), then compile clean (perf numbers).
#
# Architectural variants tested (from tests.txt):
#   1. NON_OVERLAPPED — barrier-separated K-loop + epilogue (no TMEM double-buffer)
#   2. SIX_WARP_EPI   — all 6 warps do epilogue (requires NON_OVERLAPPED)
#   3. DIRECT_STG     — st.global from registers, no SMEM staging or TMA stores
#   4. SINGLE_PRODUCER_RES — [TODO: not yet implemented]
#   5. FOLDED_RESIDUAL     — [TODO: not yet implemented]
#
# Key metrics:
#   - epi_wait: W1 waiting for epilogue to free TMEM (overlapped only)
#   - kloop:    W1 K-loop time
#   - ms/TFLOPS: wall-clock performance (clean build, no timing overhead)
#
# Usage:
#   ./tools/fc2_arch_search.sh              # full search
#   ./tools/fc2_arch_search.sh --quick      # 4 key configs only

set -uo pipefail
cd "$(dirname "$0")/.."

QUICK=0
if [ "${1:-}" = "--quick" ]; then QUICK=1; shift; fi

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

# ── Shared base flags (FC2 best known params, excluding arch flags) ──
BASE="-DBIAS_SMEM=1 -DN_STAGES=5 -DPHASE1_UNROLL=1 -DPRELOAD_MODE=0 -DSTAGGER_CYCLES=0 -DTMA_RESIDUAL=1"

# ── Configs: NAME FLAGS ──
declare -a NAMES FLAGS
add() { NAMES+=("$1"); FLAGS+=("$2"); }

# --- Baseline: current best overlapped architecture ---
add "overlapped_base"       "$BASE -DINTERLEAVE_STRATEGY=1"
add "overlapped_is0"        "$BASE -DINTERLEAVE_STRATEGY=0"

# --- Variant 1: NON_OVERLAPPED (single-buffered TMEM, barrier sync) ---
add "nonoverlap"            "$BASE -DNON_OVERLAPPED=1 -DINTERLEAVE_STRATEGY=1"
add "nonoverlap_is0"        "$BASE -DNON_OVERLAPPED=1 -DINTERLEAVE_STRATEGY=0"
add "nonoverlap_nores"      "$BASE -DNON_OVERLAPPED=1 -DINTERLEAVE_STRATEGY=1 -DTMA_RESIDUAL=0"

# --- Variant 2: SIX_WARP_EPI (all 6 warps, requires NON_OVERLAPPED) ---
# N_STAGES=4 mandatory — 6 staging regions won't fit with NS=5
add "6warp"                 "$BASE -DNON_OVERLAPPED=1 -DSIX_WARP_EPI=1 -DN_STAGES=4 -DINTERLEAVE_STRATEGY=1"
add "6warp_is0"             "$BASE -DNON_OVERLAPPED=1 -DSIX_WARP_EPI=1 -DN_STAGES=4 -DINTERLEAVE_STRATEGY=0"

# --- Variant 3: DIRECT_STG (no SMEM staging, st.global from regs) ---
add "direct_stg"            "$BASE -DDIRECT_STG=1 -DINTERLEAVE_STRATEGY=0 -DTMA_RESIDUAL=0"
add "direct_stg_nonoverlap" "$BASE -DDIRECT_STG=1 -DNON_OVERLAPPED=1 -DINTERLEAVE_STRATEGY=0 -DTMA_RESIDUAL=0"
add "direct_stg_6warp"      "$BASE -DDIRECT_STG=1 -DNON_OVERLAPPED=1 -DSIX_WARP_EPI=1 -DN_STAGES=4 -DINTERLEAVE_STRATEGY=0 -DTMA_RESIDUAL=0"

# --- Variant 4: SINGLE_PRODUCER_RES (one warp loads all residual) ---
add "spr"                   "$BASE -DNON_OVERLAPPED=1 -DSINGLE_PRODUCER_RES=1 -DINTERLEAVE_STRATEGY=1"
add "spr_is0"               "$BASE -DNON_OVERLAPPED=1 -DSINGLE_PRODUCER_RES=1 -DINTERLEAVE_STRATEGY=0"

# --- Variant 5: FOLDED_RESIDUAL (W0 preloads residual during K-loop) ---
add "fold_res"              "$BASE -DNON_OVERLAPPED=1 -DFOLDED_RESIDUAL=1 -DN_STAGES=3 -DINTERLEAVE_STRATEGY=1"
add "fold_res_direct"       "$BASE -DNON_OVERLAPPED=1 -DFOLDED_RESIDUAL=1 -DDIRECT_STG=1 -DINTERLEAVE_STRATEGY=0"
add "fold_res_ns3"          "$BASE -DNON_OVERLAPPED=1 -DFOLDED_RESIDUAL=1 -DN_STAGES=3 -DINTERLEAVE_STRATEGY=1"

# --- Parameter variations on best arch (for comparison) ---
add "overlapped_ph1u2"      "$BASE -DINTERLEAVE_STRATEGY=1 -DPHASE1_UNROLL=2"
add "overlapped_st1"        "$BASE -DINTERLEAVE_STRATEGY=1 -DSTORE_TIMING=1"
add "nonoverlap_ph1u2"      "$BASE -DNON_OVERLAPPED=1 -DINTERLEAVE_STRATEGY=1 -DPHASE1_UNROLL=2"

if [ "$QUICK" = "1" ]; then
    # Quick mode: baseline + nonoverlap + direct_stg + spr + fold_res
    INDICES=(0 2 8 11 13)
    log "Quick mode: ${#INDICES[@]} configs"
else
    INDICES=($(seq 0 $((${#NAMES[@]} - 1))))
    log "Full mode: ${#NAMES[@]} configs"
fi

# ── Results table ──
SUMMARY="$OUTDIR/summary.txt"
CSV="$OUTDIR/results.csv"
printf "%-24s %7s %8s %5s  %7s %7s %7s %7s %7s %7s %7s  %s\n" \
    "CONFIG" "ms" "TFLOPS" "REGS" \
    "epi_w" "tma0_w" "kloop" "total" \
    "e_ml_w" "e_ph1" "e_ph2" \
    "VERDICT" > "$SUMMARY"
printf "%s\n" "$(printf '%.0s-' {1..150})" >> "$SUMMARY"
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

    if timeout 120 "$BIN" > "$TOUT" 2>&1; then
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
        if timeout 120 "$BIN" > "$POUT" 2>&1; then
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

    # Verdict (overlapped configs only — NON_OVERLAPPED won't have epi_wait)
    verdict="?"
    if echo "$name" | grep -q "nonoverlap\|6warp\|direct_stg\|spr\|fold_res"; then
        verdict="NON-OVERLAP"
    elif [ -n "$w1_epi" ] && [ -n "$w1_kloop" ]; then
        if [ "$w1_epi" -lt 100 ]; then verdict="COMPUTE-BOUND"
        elif [ "$w1_epi" -gt "$w1_kloop" ]; then verdict="EPILOGUE-BOUND"
        else verdict="BALANCED"
        fi
    fi

    log "  ms=${perf_ms:-?} TFLOPS=${perf_tflops:-?} regs=${regs:-?} verdict=${verdict}"

    printf "%-24s %7s %8s %5s  %7s %7s %7s %7s %7s %7s %7s  %s\n" \
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
log "Key: epi_w = W1 waiting for epilogue to free TMEM (0 in NON-OVERLAP)"
log "     kloop = W1 K-loop time (should be similar across all)"
log "     ms/TFLOPS = wall-clock perf (clean build, no timing noise)"
log ""
log "Per-config analysis: $OUTDIR/analysis_*.txt"
log "CSV: $CSV"

rm -f "$BIN"
log "Done. Output: $OUTDIR"
