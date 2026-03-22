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
#   ./tools/fc2_arch_search.sh                       # run all configs, no clock lock
#   ./tools/fc2_arch_search.sh --only NAME           # run single config by name
#   ./tools/fc2_arch_search.sh --clocks 1770,2100    # run all configs at each clock rate
#   ./tools/fc2_arch_search.sh --clocks max          # lock to max supported SM clock
#   ./tools/fc2_arch_search.sh --clocks 2100 --only baseline

set -uo pipefail
cd "$(dirname "$0")/.."

ONLY=""
CLOCK_LIST=""
while [ $# -gt 0 ]; do
    case "$1" in
        --only)    ONLY="$2"; shift 2 ;;
        --clocks)  CLOCK_LIST="$2"; shift 2 ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="data/fc2_arch_${TIMESTAMP}"
mkdir -p "$OUTDIR"

NVCC="nvcc"
CFLAGS="-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -lineinfo --ptxas-options=-v --cudart=static"
LDFLAGS="-lcurand_static -lculibos -lcuda"
SRC="fc2.cu"
BIN="$OUTDIR/fc2_test"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/session.log"; }

# ── Clock management ──
CLOCKS_LOCKED=0
gpu_clock_reset() {
    if [ "$CLOCKS_LOCKED" -eq 1 ]; then
        nvidia-smi -rgc > /dev/null 2>&1 || true
        log "GPU clocks reset to default"
        CLOCKS_LOCKED=0
    fi
}
trap gpu_clock_reset EXIT INT TERM

gpu_clock_lock() {
    local freq="$1"
    if [ "$freq" = "max" ]; then
        freq=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        log "Max supported SM clock: ${freq} MHz"
    fi
    nvidia-smi -lgc "$freq","$freq" > /dev/null 2>&1
    CLOCKS_LOCKED=1
    # Verify
    local actual
    actual=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    log "GPU SM clock locked: requested=${freq} MHz, actual=${actual} MHz"
}

gpu_query_clocks() {
    nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.max.sm,power.draw,temperature.gpu \
        --format=csv,noheader 2>/dev/null | head -1
}

log "========================================="
log "  FC2 ARCHITECTURE SEARCH $TIMESTAMP"
log "========================================="
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,clocks.mem --format=csv,noheader 2>/dev/null | head -1 | tee -a "$OUTDIR/session.log" || true

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

# ── StagesC + SMEM reuse (NS5 + StagesC, staging overlaps pipeline stages) ──
REUSE_BASE="-DN_STAGES=5 -DINTERLEAVE_STRATEGY=1 -DBIAS_SMEM=1 -DPHASE1_UNROLL=1 -DPRELOAD_MODE=0 -DSTAGGER_CYCLES=0 -DTMA_RESIDUAL=1 -DSTAGES_C=2 -DEPI_NOINLINE=1 -DEPI_REUSE_SMEM=1"
add "reuse_sc"             "$REUSE_BASE"
add "reuse_sc_pc"          "$REUSE_BASE -DPRE_COMBINE=1"

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

# ── Resolve clock rates to test ──
declare -a CLOCK_RATES
if [ -n "$CLOCK_LIST" ]; then
    IFS=',' read -ra CLOCK_RATES <<< "$CLOCK_LIST"
else
    CLOCK_RATES=("")  # empty = no lock (default behavior)
fi

log "${#INDICES[@]} configs × ${#CLOCK_RATES[@]} clock rate(s)"
if [ -n "$CLOCK_LIST" ]; then
    log "Clock rates: ${CLOCK_RATES[*]}"
fi

# ── Run configs at each clock rate ──
for clk in "${CLOCK_RATES[@]}"; do
    CLK_LABEL=""
    if [ -n "$clk" ]; then
        CLK_LABEL="_${clk}mhz"
        gpu_clock_lock "$clk"
        # Warmup at new clock (let thermals settle)
        log "Warmup at ${clk} MHz..."
        sleep 2
    fi

    # ── Results table for this clock rate ──
    SUMMARY="$OUTDIR/summary${CLK_LABEL}.txt"
    CSV="$OUTDIR/results${CLK_LABEL}.csv"
    printf "%-32s %7s %8s %5s %6s  %7s %7s %7s %7s %7s %7s %7s  %s\n" \
        "CONFIG" "ms" "TFLOPS" "REGS" "GHz" \
        "epi_w" "tma0_w" "kloop" "total" \
        "e_ml_w" "e_ph1" "e_ph2" \
        "VERDICT" > "$SUMMARY"
    printf "%s\n" "$(printf '%.0s-' {1..168})" >> "$SUMMARY"
    echo "config,ms,tflops,regs,sm_ghz,w1_epi_wait,w1_tma0_wait,w1_kloop,w1_total,epi_ml_wait,epi_phase1,epi_phase2,verdict" > "$CSV"

    for idx in "${INDICES[@]}"; do
        name="${NAMES[$idx]}"
        flags="${FLAGS[$idx]}"
        log ""
        log "── ${name}${CLK_LABEL} ──"
        log "  $flags"
        [ -n "$clk" ] && log "  GPU: $(gpu_query_clocks)"

        # ── Timing build ──
        TOUT="$OUTDIR/timing_${name}${CLK_LABEL}.txt"
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
        POUT="$OUTDIR/perf_${name}${CLK_LABEL}.txt"
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
        sm_ghz=$(grep -oP 'SM clock \(measured\): \K[\d.]+(?= GHz)' "$TOUT" | head -1)
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

        log "  ms=${perf_ms:-?} TFLOPS=${perf_tflops:-?} regs=${regs:-?} sm_ghz=${sm_ghz:-?} verdict=${verdict}"

        printf "%-32s %7s %8s %5s %6s  %7s %7s %7s %7s %7s %7s %7s  %s\n" \
            "$name" "${perf_ms:-?}" "${perf_tflops:-?}" "${regs:-?}" "${sm_ghz:-?}" \
            "${w1_epi:-?}" "${w1_tma0:-?}" "${w1_kloop:-?}" "${w1_total:-?}" \
            "${epi_ml:-?}" "${epi_p1:-?}" "${epi_p2:-?}" "$verdict" >> "$SUMMARY"
        echo "$name,${perf_ms:-},${perf_tflops:-},${regs:-},${sm_ghz:-},${w1_epi:-},${w1_tma0:-},${w1_kloop:-},${w1_total:-},${epi_ml:-},${epi_p1:-},${epi_p2:-},$verdict" >> "$CSV"

        # Full timing analysis
        [ -n "$w1_epi" ] && python3 tools/analyze_timing.py "$TOUT" --ref-tflops 3564 \
            > "$OUTDIR/analysis_${name}${CLK_LABEL}.txt" 2>&1 || true
    done

    # ── Summary for this clock rate ──
    log ""
    log "========================================="
    [ -n "$clk" ] && log "  RESULTS @ ${clk} MHz" || log "  RESULTS (unlocked)"
    log "========================================="
    cat "$SUMMARY" | tee -a "$OUTDIR/session.log"

    # Reset before next clock rate
    [ -n "$clk" ] && gpu_clock_reset
done

# ── Multi-clock comparison ──
if [ "${#CLOCK_RATES[@]}" -gt 1 ]; then
    log ""
    log "========================================="
    log "  CROSS-CLOCK COMPARISON"
    log "========================================="
    # Merge all CSVs into one with clock column
    MERGED="$OUTDIR/results_merged.csv"
    echo "clock_mhz,config,ms,tflops,regs,sm_ghz,w1_epi_wait,w1_tma0_wait,w1_kloop,w1_total,epi_ml_wait,epi_phase1,epi_phase2,verdict" > "$MERGED"
    for clk in "${CLOCK_RATES[@]}"; do
        CLK_LABEL="_${clk}mhz"
        CSV_FILE="$OUTDIR/results${CLK_LABEL}.csv"
        [ -f "$CSV_FILE" ] && tail -n +2 "$CSV_FILE" | while IFS= read -r line; do
            echo "${clk},${line}"
        done >> "$MERGED"
    done
    log "Merged CSV: $MERGED"

    # Quick side-by-side for configs that appear in all clock rates
    python3 -c "
import csv, sys
rows = []
with open('$MERGED') as f:
    for r in csv.DictReader(f): rows.append(r)
configs = sorted(set(r['config'] for r in rows))
clocks = sorted(set(r['clock_mhz'] for r in rows))
print(f'  {\"CONFIG\":<24}', '  '.join(f'{c:>12} MHz' for c in clocks))
print('  ' + '-' * (24 + 15 * len(clocks)))
for cfg in configs:
    parts = []
    for clk in clocks:
        match = [r for r in rows if r['config'] == cfg and r['clock_mhz'] == clk]
        if match and match[0]['ms']:
            parts.append(f'{float(match[0][\"ms\"]):>8.3f}ms {match[0].get(\"sm_ghz\",\"?\"):>5}')
        else:
            parts.append(f'{\"---\":>14}')
    print(f'  {cfg:<24}', '  '.join(parts))
" 2>&1 | tee -a "$OUTDIR/session.log"
fi

rm -f "$BIN"
log ""
log "Per-config analysis: $OUTDIR/analysis_*.txt"
log "CSV(s): $OUTDIR/results*.csv"
log "Done. Output: $OUTDIR"
