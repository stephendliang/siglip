#!/bin/bash
# Dimension sweep: benchmark fc2_w3 (fused + gemm + sched) and CUTLASS (fused + strip)
# across a grid of M × N × K dimensions.
#
# Usage:
#   ./tools/dim_sweep.sh              # full grid (default)
#   ./tools/dim_sweep.sh --fast       # reduced grid (fewer M values)
#   ./tools/dim_sweep.sh --custom M1,M2 N1,N2 K1,K2   # explicit lists
#
# Constraints: N % 256 == 0, K % 128 == 0, M % 256 == 0
# Runs on B200. Each binary gets 30s timeout (deadlock protection).
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="data/dim_sweep_${TIMESTAMP}"
mkdir -p "$OUTDIR"
RESULTS="$OUTDIR/results.txt"
TIMEOUT=30

# ── Dimension grids ──
# Full grid: 6M × 6N × 7K = 252 configs × 5 variants = 1260 runs
M_FULL=(116032 232064 464128 928256 1856512 3713024)
N_FULL=(256 512 768 1024 1536 3072)
K_FULL=(512 1024 1536 2048 3072 4096 6144)

# Fast grid: 4M × 4N × 5K = 80 configs × 5 variants = 400 runs
M_FAST=(232064 464128 928256 1856512)
N_FAST=(256 768 1024 1536)
K_FAST=(1024 1536 2048 3072 4096)

# Parse args
MODE="full"
if [ "${1:-}" = "--fast" ]; then
    MODE="fast"
elif [ "${1:-}" = "--custom" ]; then
    MODE="custom"
    IFS=',' read -ra M_CUSTOM <<< "${2:?usage: --custom M1,M2 N1,N2 K1,K2}"
    IFS=',' read -ra N_CUSTOM <<< "${3:?}"
    IFS=',' read -ra K_CUSTOM <<< "${4:?}"
fi

case "$MODE" in
    full)   M_VALS=("${M_FULL[@]}"); N_VALS=("${N_FULL[@]}"); K_VALS=("${K_FULL[@]}") ;;
    fast)   M_VALS=("${M_FAST[@]}"); N_VALS=("${N_FAST[@]}"); K_VALS=("${K_FAST[@]}") ;;
    custom) M_VALS=("${M_CUSTOM[@]}"); N_VALS=("${N_CUSTOM[@]}"); K_VALS=("${K_CUSTOM[@]}") ;;
esac

TOTAL_CONFIGS=$(( ${#M_VALS[@]} * ${#N_VALS[@]} * ${#K_VALS[@]} ))

# NS6 fits N_DIM <= 1536 in 228KB SMEM. Larger N needs NS5.
ns_for_n() {
    if [ "$1" -le 1536 ]; then echo 6; else echo 5; fi
}

# PREFILL relies on K-loop being long enough for epilogue to drain.
# K_ITERS=K/128; safe threshold ~20 iters (empirical: 24 works, 12 deadlocks).
extra_flags_for_k() {
    local k_iters=$(($1 / 128))
    if [ "$k_iters" -lt 20 ]; then echo "-DNO_PREFILL"; else echo ""; fi
}

run_with_timeout() {
    local binary=$1 label=$2
    local result
    result=$(timeout "$TIMEOUT" ./"$binary" 2>&1 | grep '@@RESULT' || true)
    if [ -n "$result" ]; then
        echo "  $result label=$label" | tee -a "$RESULTS"
    else
        echo "  $label: NO OUTPUT (crash/deadlock/timeout)" | tee -a "$RESULTS"
    fi
}

build_w3() {
    local dims=$1 ns=$2 extra=${3:-}
    make -s -B fc2-w3 DFLAGS="$dims -DN_STAGES=$ns $extra" 2>"$OUTDIR/build.tmp" && return 0
    cat "$OUTDIR/build.tmp" >> "$OUTDIR/build_errors.txt"; return 1
}
build_w3_gemm() {
    local dims=$1 ns=$2
    make -s -B fc2-w3-gemm DFLAGS="$dims -DN_STAGES=$ns" 2>"$OUTDIR/build.tmp" && return 0
    cat "$OUTDIR/build.tmp" >> "$OUTDIR/build_errors.txt"; return 1
}
build_w3_sched() {
    local dims=$1 ns=$2 extra=${3:-}
    make -s -B fc2-w3-sched DFLAGS="$dims -DN_STAGES=$ns $extra" 2>"$OUTDIR/build.tmp" && return 0
    cat "$OUTDIR/build.tmp" >> "$OUTDIR/build_errors.txt"; return 1
}
build_cutlass() {
    local dims=$1 target=$2
    make -s -B "$target" DFLAGS="$dims" 2>"$OUTDIR/build.tmp" && return 0
    cat "$OUTDIR/build.tmp" >> "$OUTDIR/build_errors.txt"; return 1
}

# ── Header ──
echo "Dimension sweep ($MODE) — $(date)" | tee "$RESULTS"
echo "Grid: ${#M_VALS[@]}M × ${#N_VALS[@]}N × ${#K_VALS[@]}K = $TOTAL_CONFIGS configs" | tee -a "$RESULTS"
echo "M: ${M_VALS[*]}" | tee -a "$RESULTS"
echo "N: ${N_VALS[*]}" | tee -a "$RESULTS"
echo "K: ${K_VALS[*]}" | tee -a "$RESULTS"
echo "Timeout: ${TIMEOUT}s per run" | tee -a "$RESULTS"
echo "========================================" | tee -a "$RESULTS"

# ── OOM check (75GB limit, leave headroom on 80GB B200) ──
GPU_MEM_LIMIT=$((75 * 1024 * 1024 * 1024))
check_oom() {
    local m=$1 n=$2 k=$3
    # A(FP8) + B(FP8) + bias(BF16) + residual(BF16) + C(BF16)
    local total=$(( m*k + n*k + n*2 + m*n*2 + m*n*2 ))
    [ "$total" -lt "$GPU_MEM_LIMIT" ]
}

# ── CSV header ──
CSV="$OUTDIR/results.csv"
echo "M,N,K,NS,variant,ms,tflops,checksum,valid" > "$CSV"

parse_result() {
    # Extract fields from @@RESULT line
    local line=$1 m=$2 n=$3 k=$4 ns=$5 variant=$6
    local ms tflops checksum valid
    ms=$(echo "$line" | sed -n 's/.*ms=\([^ ]*\).*/\1/p')
    tflops=$(echo "$line" | sed -n 's/.*tflops=\([^ ]*\).*/\1/p')
    checksum=$(echo "$line" | sed -n 's/.*checksum=\([^ ]*\).*/\1/p')
    valid=$(echo "$line" | sed -n 's/.*valid=\([^ ]*\).*/\1/p')
    echo "$m,$n,$k,$ns,$variant,$ms,$tflops,$checksum,$valid" >> "$CSV"
}

# ── Main grid loop ──
IDX=0
for M in "${M_VALS[@]}"; do
for N in "${N_VALS[@]}"; do
for K in "${K_VALS[@]}"; do
    IDX=$((IDX + 1))

    if ! check_oom "$M" "$N" "$K"; then
        echo "" | tee -a "$RESULTS"
        echo "[$IDX/$TOTAL_CONFIGS] M=$M N=$N K=$K — SKIPPED (OOM)" | tee -a "$RESULTS"
        continue
    fi

    NS=$(ns_for_n "$N")
    EXTRA=$(extra_flags_for_k "$K")
    DIMS="-DM_TOTAL=$M -DN_DIM=$N -DK_DIM=$K"
    TAG="M${M}_N${N}_K${K}"

    echo "" | tee -a "$RESULTS"
    echo "[$IDX/$TOTAL_CONFIGS] M=$M N=$N K=$K (NS=$NS${EXTRA:+ $EXTRA})" | tee -a "$RESULTS"

    # Build
    W3F=1; W3G=1; W3S=1; CF=1; CS=1
    build_w3      "$DIMS" "$NS" "$EXTRA" && W3F=0 || echo "  ${TAG}_w3_fused: BUILD FAILED" | tee -a "$RESULTS"
    build_w3_gemm "$DIMS" "$NS"          && W3G=0 || echo "  ${TAG}_w3_gemm: BUILD FAILED" | tee -a "$RESULTS"
    build_w3_sched "$DIMS" "$NS" "$EXTRA" && W3S=0 || echo "  ${TAG}_w3_sched: BUILD FAILED" | tee -a "$RESULTS"
    build_cutlass "$DIMS" fc2-cutlass       && CF=0 || echo "  ${TAG}_cutlass_fused: BUILD FAILED" | tee -a "$RESULTS"
    build_cutlass "$DIMS" fc2-cutlass-strip && CS=0 || echo "  ${TAG}_cutlass_strip: BUILD FAILED" | tee -a "$RESULTS"

    # Run + parse
    for variant_info in "W3F:fc2-w3:w3_fused" "W3G:fc2-w3-gemm:w3_gemm" "W3S:fc2-w3-sched:w3_sched" "CF:fc2-cutlass:cutlass_fused" "CS:fc2-cutlass-strip:cutlass_strip"; do
        IFS=':' read -r flag_var binary variant <<< "$variant_info"
        eval "flag_val=\$$flag_var"
        if [ "$flag_val" -eq 0 ]; then
            result=$(timeout "$TIMEOUT" ./"$binary" 2>&1 | grep '@@RESULT' || true)
            if [ -n "$result" ]; then
                echo "  $result label=${TAG}_${variant}" | tee -a "$RESULTS"
                parse_result "$result" "$M" "$N" "$K" "$NS" "$variant"
            else
                echo "  ${TAG}_${variant}: NO OUTPUT (crash/deadlock/timeout)" | tee -a "$RESULTS"
                echo "$M,$N,$K,$NS,$variant,,,," >> "$CSV"
            fi
        fi
    done
done
done
done

echo "" | tee -a "$RESULTS"
echo "========================================" | tee -a "$RESULTS"
echo "Done. $IDX configs." | tee -a "$RESULTS"
echo "Results: $RESULTS" | tee -a "$RESULTS"
echo "CSV:     $CSV" | tee -a "$RESULTS"
