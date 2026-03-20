#!/bin/bash
# FC2 Architecture Search: combinatorial sweep of structural kernel variants on B200.
# Each config: compile with -DTIMING (cycle breakdown), then compile clean (perf numbers).
#
# Swept axes (full combinatorial):
#   BATCH_MMA          {0,1}  — all 4 sub-MMAs in single asm block
#   PREFETCH_MBAR      {0,1}  — W1 K-loop: try_wait prefetch for next TMA stage barrier
#   OVERLAP_EPI_WAIT   {0,1}  — prefetch epilogue mbar at end of K-loop
#   N_STAGES           {4,5}  — pipeline depth (may flip with shorter BATCH_MMA K-loop)
#   INTERLEAVE_STRATEGY {1,2} — TMA store interleaving (IS=2 may win with faster K-loop)
#
# EPI_LOAD_WARP excluded — hangs at runtime (mbarrier bug, never validated on B200)
#
# Total: 2^3 × 2 × 2 = 32 configs (--quick), +16 W0_RES variants (full) = 48
#
# Key metrics:
#   - epi_wait: W1 waiting for epilogue to free TMEM
#   - kloop:    W1 K-loop time
#   - ms/TFLOPS: wall-clock performance (clean build, no timing overhead)
#
# Usage:
#   ./tools/fc2_arch_search.sh              # full search (32 combos + 16 W0_RES = 48)
#   ./tools/fc2_arch_search.sh --quick      # 32 combinatorial configs only

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

# ── Shared base flags (FC2 best known params, N_STAGES and IS swept separately) ──
BASE_CORE="-DBIAS_SMEM=1 -DPHASE1_UNROLL=1 -DPRELOAD_MODE=0 -DSTAGGER_CYCLES=0 -DTMA_RESIDUAL=1"

# ── Configs: NAME FLAGS ──
declare -a NAMES FLAGS
add() { NAMES+=("$1"); FLAGS+=("$2"); }

# --- Full combinatorial: 3 binary flags × N_STAGES{4,5} × IS{1,2} = 32 configs ---
# EPI_LOAD_WARP excluded — hangs at runtime (mbarrier bug, never validated on B200)
BIN_NAMES=("BM" "PF" "OEW")
BIN_DEFS=("-DBATCH_MMA=1" "-DPREFETCH_MBAR=1" "-DOVERLAP_EPI_WAIT=1")

for ns in 4 5; do
    for is in 1 2; do
        for bits in $(seq 0 7); do
            name="NS${ns}.IS${is}"
            flags="$BASE_CORE -DN_STAGES=${ns} -DINTERLEAVE_STRATEGY=${is}"
            for i in 0 1 2; do
                if (( bits & (1 << i) )); then
                    name="${name}.${BIN_NAMES[$i]}"
                    flags="$flags ${BIN_DEFS[$i]}"
                fi
            done
            [ "$name" = "NS${ns}.IS${is}" ] && name="${name}.base"
            add "$name" "$flags"
        done
    done
done

if [ "$QUICK" = "0" ]; then
    # --- W0_RES variants: cross {W0_RES_PREFETCH, W0_RES_FULL} × {BM, no-BM} × {NS4, NS5} ---
    for ns in 4 5; do
        for bm in 0 1; do
            bm_flag=""; bm_tag=""
            [ "$bm" = "1" ] && { bm_flag=" -DBATCH_MMA=1"; bm_tag=".BM"; }
            add "NS${ns}.IS1${bm_tag}.W0RP" "$BASE_CORE -DN_STAGES=${ns} -DINTERLEAVE_STRATEGY=1${bm_flag} -DW0_RES_PREFETCH=1"
            add "NS${ns}.IS1${bm_tag}.W0RF" "$BASE_CORE -DN_STAGES=${ns} -DINTERLEAVE_STRATEGY=1${bm_flag} -DW0_RES_FULL=1"
            add "NS${ns}.IS1${bm_tag}.PF.W0RF" "$BASE_CORE -DN_STAGES=${ns} -DINTERLEAVE_STRATEGY=1${bm_flag} -DPREFETCH_MBAR=1 -DW0_RES_FULL=1"
            add "NS${ns}.IS1${bm_tag}.PF.OEW.W0RF" "$BASE_CORE -DN_STAGES=${ns} -DINTERLEAVE_STRATEGY=1${bm_flag} -DPREFETCH_MBAR=1 -DOVERLAP_EPI_WAIT=1 -DW0_RES_FULL=1"
        done
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
echo "config,ms,tflops,regs,N_STAGES,INTERLEAVE_STRATEGY,BATCH_MMA,PREFETCH_MBAR,OVERLAP_EPI_WAIT,W0_RES_PREFETCH,W0_RES_FULL,w1_epi_wait,w1_tma0_wait,w1_kloop,w1_total,epi_ml_wait,epi_phase1,epi_phase2,verdict" > "$CSV"

for idx in "${INDICES[@]}"; do
    name="${NAMES[$idx]}"
    flags="${FLAGS[$idx]}"
    log ""
    log "── $name ──"
    log "  $flags"

    # Extract flag values for CSV
    ns=4; is=1; bm=0; pf=0; oew=0; w0rp=0; w0rf=0
    [[ "$flags" =~ N_STAGES=([0-9]+) ]] && ns="${BASH_REMATCH[1]}"
    [[ "$flags" =~ INTERLEAVE_STRATEGY=([0-9]+) ]] && is="${BASH_REMATCH[1]}"
    [[ "$flags" == *"BATCH_MMA=1"* ]] && bm=1
    [[ "$flags" == *"PREFETCH_MBAR=1"* ]] && pf=1
    [[ "$flags" == *"OVERLAP_EPI_WAIT=1"* ]] && oew=1
    [[ "$flags" == *"W0_RES_PREFETCH=1"* ]] && w0rp=1
    [[ "$flags" == *"W0_RES_FULL=1"* ]] && w0rf=1

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

    # Verdict based on epilogue/K-loop balance
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
    echo "$name,${perf_ms:-},${perf_tflops:-},${regs:-},$ns,$is,$bm,$pf,$oew,$w0rp,$w0rf,${w1_epi:-},${w1_tma0:-},${w1_kloop:-},${w1_total:-},${epi_ml:-},${epi_p1:-},${epi_p2:-},$verdict" >> "$CSV"

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
log "Key: epi_w = W1 waiting for epilogue to free TMEM"
log "     kloop = W1 K-loop time"
log "     ms/TFLOPS = wall-clock perf (clean build, no timing noise)"
log ""
log "Swept axes (full combinatorial, ${#INDICES[@]} configs):"
log "  BATCH_MMA          {0,1}  = all 4 sub-MMAs in single asm block"
log "  PREFETCH_MBAR      {0,1}  = try_wait prefetch in W1 K-loop"
log "  OVERLAP_EPI_WAIT   {0,1}  = prefetch epilogue mbar at end of K-loop"
log "  N_STAGES           {4,5}  = pipeline depth"
log "  INTERLEAVE_STRATEGY {1,2} = TMA store interleaving"
log ""
log "Per-config analysis: $OUTDIR/analysis_*.txt"
log "CSV: $CSV"

rm -f "$BIN"
log "Done. Output: $OUTDIR"
