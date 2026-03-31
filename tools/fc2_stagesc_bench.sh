#!/bin/bash
# FC2 StagesC architecture bench — the CUTLASS-equivalent pre-load approach.
#
# StagesC=2: W0 pre-loads residual into staged SMEM, epilogue warps LDS from SMEM.
# Eliminates per-warp global memory traffic (the root cause of K-loop inflation).
#
# BIAS_SMEM=1: also loads bias from SMEM → zero global loads in epilogue path.
# EPI_REUSE_SMEM=1: staging overlaps pipeline stages 2-4 → fits in 161 KB.
#
# Tests:
#   1. Reference baselines (strip, 4w, WL1, WL2, NEPI2, CUTLASS)
#   2. SC2 + EPI_REUSE_SMEM at 4/3/2 warps
#   3. SC2 + EPI_REUSE_SMEM + BIAS_SMEM at 4/3/2 warps
#   4. SC2 without EPI_REUSE_SMEM at 2 warps (fits in SMEM)
#   5. Thermal drift check
#
# Usage:
#   ./tools/fc2_stagesc_bench.sh              # full run (~5 min)
#   ./tools/fc2_stagesc_bench.sh --dry-run    # print commands
#   ./tools/fc2_stagesc_bench.sh --quick      # skip CUTLASS + 3-warp
#
# Output:  data/stagesc_YYYYMMDD_HHMMSS/

set -uo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=0
QUICK=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --quick)   QUICK=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="data/stagesc_${TIMESTAMP}"
mkdir -p "$OUTDIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/session.log"; }

log "========================================"
log "  FC2 STAGES_C BENCH  $TIMESTAMP"
log "  Output: $OUTDIR"
log "========================================"

if [ "$DRY_RUN" = "0" ]; then
    nvidia-smi > /dev/null 2>&1 || { log "FATAL: no GPU"; exit 1; }
    nvidia-smi --query-gpu=gpu_name,clocks.sm --format=csv,noheader | tee -a "$OUTDIR/session.log"
fi

# Auto-init CUTLASS submodule if not already present
if [ ! -f "third_party/cutlass/include/cutlass/cutlass.h" ]; then
    log "Initializing CUTLASS submodule..."
    git submodule update --init third_party/cutlass 2>&1 | tail -1 | tee -a "$OUTDIR/session.log"
fi

BASE="-DN_STAGES=5 -DTMA_RESIDUAL=1"
SC2="$BASE -DSTAGES_C=2 -DEPI_REUSE_SMEM=1"

run_wall() {
    local label="$1" dflags="$2"
    log "  [wall] $label  DFLAGS='$dflags'"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  make fc2 DFLAGS=\"$dflags\" && ./fc2"
        return 0
    fi

    if ! make fc2 DFLAGS="$dflags" > "$OUTDIR/${label}_build.log" 2>&1; then
        log "  [$label] BUILD FAILED"
        cat "$OUTDIR/${label}_build.log" >> "$OUTDIR/session.log"
        return 1
    fi

    local regs
    regs=$(grep -o '[0-9]* registers' "$OUTDIR/${label}_build.log" | head -1 | grep -o '[0-9]*')

    local output
    if ! output=$(./fc2 2>&1); then
        log "  [$label] RUN FAILED"
        echo "$output" >> "$OUTDIR/session.log"
        return 1
    fi

    echo "$output" > "$OUTDIR/${label}_output.txt"

    local result_line
    result_line=$(echo "$output" | grep '@@RESULT' | head -1)
    if [ -n "$result_line" ]; then
        echo "${result_line} label=${label} regs=${regs} dflags=\"${dflags}\"" >> "$OUTDIR/wall_results.txt"
        local ms valid
        ms=$(echo "$result_line" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
        valid=$(echo "$result_line" | grep -o 'valid=[01]' | cut -d= -f2)
        log "  [$label] ${ms}ms  valid=${valid}  regs=${regs}"
    else
        log "  [$label] NO @@RESULT LINE"
        echo "@@RESULT ms=ERR label=${label}" >> "$OUTDIR/wall_results.txt"
    fi
}

run_wall_timing() {
    local label="$1" dflags="$2"
    log "  [timing] $label  DFLAGS='$dflags'"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  make fc2-timing DFLAGS=\"$dflags\" && ./fc2-timing"
        return 0
    fi

    if ! make fc2-timing DFLAGS="$dflags" > "$OUTDIR/${label}_timing_build.log" 2>&1; then
        log "  [$label] TIMING BUILD FAILED"
        return 1
    fi

    local output
    if ! output=$(./fc2-timing 2>&1); then
        log "  [$label] TIMING RUN FAILED"
        return 1
    fi

    echo "$output" > "$OUTDIR/${label}_timing.txt"
    log "  [$label] timing collected"
}

START_TIME=$(date +%s)

# ════════════════════════════════════════
# TEST 1: Reference baselines
# ════════════════════════════════════════
log ""
log "==== TEST 1: REFERENCE BASELINES ===="

# NOTE: WL<4 and NEPI<4 produce valid=0 — only 32*N rows per CTA rank are written
# (row_group = ew % 4, fewer warps = incomplete row coverage). Timing-only configs.
BASELINES=(
    "strip|$BASE -DSTRIP_EPILOGUE=1"
    "base_4w|$BASE"
    "base_wl1|$BASE -DEPI_WARP_LIMIT=1"
    "base_wl2|$BASE -DEPI_WARP_LIMIT=2"
    "base_nepi2|$BASE -DNUM_EPI_WARPS=2"
)

for exp in "${BASELINES[@]}"; do
    IFS='|' read -r label dflags <<< "$exp"
    run_wall "$label" "$dflags"
done

# CUTLASS reference (build if needed, run if available)
if [ "$QUICK" = "0" ]; then
    if [ ! -x "./cutlass-bench-fc2-max" ]; then
        log "  [cutlass] building cutlass-bench-fc2-max..."
        if [ "$DRY_RUN" = "0" ]; then
            if make cutlass-bench-fc2-max > "$OUTDIR/cutlass_build.log" 2>&1; then
                log "  [cutlass] build OK"
            else
                log "  [cutlass] BUILD FAILED (see cutlass_build.log)"
            fi
        fi
    fi
    if [ -x "./cutlass-bench-fc2-max" ]; then
        log "  [cutlass] running..."
        if [ "$DRY_RUN" = "0" ]; then
            cutlass_out=$(./cutlass-bench-fc2-max 2>&1)
            echo "$cutlass_out" > "$OUTDIR/cutlass_output.txt"
            cutlass_ms=$(echo "$cutlass_out" | grep 'Best Fused:' | grep -oP '\d+\.\d+(?=\s+ms)')
            log "  [cutlass] ${cutlass_ms:-ERR}ms"
            [ -n "$cutlass_ms" ] && echo "@@RESULT ms=${cutlass_ms} valid=1 label=cutlass" >> "$OUTDIR/wall_results.txt"
        fi
    fi
fi

# ════════════════════════════════════════
# TEST 2: SKIPPED — SC2+EPI_REUSE_SMEM is fundamentally broken
# (pre-loaded residual in borrowed stages overwritten by A/B at ki=2)
# ════════════════════════════════════════
log ""
log "==== TEST 2: SKIPPED (SC2+EPI_REUSE_SMEM broken — see kernel_common.cuh #error) ===="

# ════════════════════════════════════════
# TEST 3: SKIPPED — same root cause as TEST 2
# ════════════════════════════════════════
log ""
log "==== TEST 3: SKIPPED (SC2+EPI_REUSE_SMEM broken) ===="

# ════════════════════════════════════════
# TEST 4: SC2 without EPI_REUSE_SMEM (2 warps, fits naturally)
# ════════════════════════════════════════
log ""
log "==== TEST 4: SC2 no-reuse (2 warps) ===="

run_wall "sc2nr_nepi2" "$BASE -DSTAGES_C=2 -DNUM_EPI_WARPS=2"
run_wall "sc2nr_nepi2_bs" "$BASE -DSTAGES_C=2 -DNUM_EPI_WARPS=2 -DBIAS_SMEM=1"

# ════════════════════════════════════════
# TEST 5: Timing comparison (best SC2 vs baseline strip)
# ════════════════════════════════════════
log ""
log "==== TEST 5: TIMING (kloop/ph1/epi_wait breakdown) ===="

run_wall_timing "t_strip" "$BASE -DSTRIP_EPILOGUE=1"
run_wall_timing "t_base" "$BASE"
run_wall_timing "t_base_nepi2" "$BASE -DNUM_EPI_WARPS=2"
run_wall_timing "t_sc2nr_nepi2" "$BASE -DSTAGES_C=2 -DNUM_EPI_WARPS=2"

# ════════════════════════════════════════
# TEST 6: SASS-patched binaries (build + CP-SAT schedule + patch on target)
# ════════════════════════════════════════
log ""
log "==== TEST 6: SASS-PATCHED BINARIES ===="

# Pre-check: ortools required for CP-SAT scheduler
if ! python3 -c "from ortools.sat.python import cp_model" 2>/dev/null; then
    log "  [sass] ortools not found, installing..."
    pip install ortools 2>&1 | tail -1 | tee -a "$OUTDIR/session.log"
fi
if ! python3 -c "from ortools.sat.python import cp_model" 2>/dev/null; then
    log "  [sass] SKIPPED: ortools install failed"
else

CPSAT_TIME=120  # seconds per chunk

build_and_patch() {
    local label="$1" dflags="$2"
    local patched="${label}_patched"
    log "  [sass] $label: building..."

    if [ "$DRY_RUN" = "1" ]; then
        echo "  build + cuobjdump + sass_edit schedule + fatbin-patch → $patched"
        return 0
    fi

    # 1. Build fc2 with make (exact same binary that will be patched)
    if ! make fc2 DFLAGS="$dflags" > "$OUTDIR/${label}_build.log" 2>&1; then
        log "  [$label] BUILD FAILED"; return 1
    fi
    cp fc2 "$OUTDIR/${label}_unpatched"

    # 2. Extract cubin + SASS from the BUILT BINARY (not a separate nvcc --cubin)
    #    This guarantees recipe addresses match the embedded cubin exactly.
    local sass="$OUTDIR/${label}_sass.txt"
    cuobjdump -sass fc2 > "$sass" 2>/dev/null

    local tmpdir="$OUTDIR/${label}_cubin_tmp"
    mkdir -p "$tmpdir"
    local fc2_abs
    fc2_abs="$(pwd)/fc2"
    (cd "$tmpdir" && cuobjdump -xelf all "$fc2_abs" > /dev/null 2>&1)
    local cubin="$tmpdir/fc2.sm_100a.cubin"
    if [ ! -f "$cubin" ]; then
        log "  [$label] cubin extraction failed"
        return 1
    fi
    log "  [$label] cubin extracted ($(wc -c < "$cubin") bytes)"

    # 3. Find epilogue LDTM boundaries (grep cuobjdump SASS directly)
    local ldtm_addrs
    ldtm_addrs=($(grep -oP '/\*\K[0-9a-f]+(?=\*/\s+LDTM)' "$sass"))

    if [ ${#ldtm_addrs[@]} -lt 6 ]; then
        log "  [$label] only ${#ldtm_addrs[@]} LDTM found, skipping patch"
        return 1
    fi

    # Main-loop epilogue = first 6 LDTM
    # Schedule chunks between consecutive LDTM pairs (parallelized)
    local nchunks=5
    [ ${#ldtm_addrs[@]} -lt 6 ] && nchunks=$((${#ldtm_addrs[@]} - 1))
    log "  [$label] scheduling $nchunks epilogue chunks (${CPSAT_TIME}s limit each)..."
    local pids=() recipes=()
    for ((i=0; i<nchunks; i++)); do
        local s="0x${ldtm_addrs[$i]}"
        local e="0x${ldtm_addrs[$((i+1))]}"
        local recipe="$OUTDIR/${label}_chunk${i}.recipe"
        recipes+=("$recipe")
        python3 tools/sass_edit.py schedule "$cubin" --sass "$sass" \
            -s "$s" -e "$e" --recipe "$recipe" --time-limit "$CPSAT_TIME" --quiet \
            > "$OUTDIR/${label}_chunk${i}.log" 2>&1 &
        pids+=($!)
    done

    # Wait for all schedulers
    local ok=0 fail=0
    for pid in "${pids[@]}"; do
        if wait "$pid"; then ((ok++)); else ((fail++)); fi
    done
    log "  [$label] CP-SAT: ${ok}/${nchunks} chunks scheduled"

    # 4. Merge successful recipes
    local merged="$OUTDIR/${label}_merged.recipe"
    > "$merged"
    for recipe in "${recipes[@]}"; do
        [ -f "$recipe" ] && cat "$recipe" >> "$merged"
    done

    local nops
    nops=$(grep -c -E '^(reorder|stall)' "$merged" 2>/dev/null) || nops=0
    if [ "$nops" -lt 5 ]; then
        log "  [$label] only $nops recipe ops, skipping patch"
        return 1
    fi

    # 5. fatbin-patch the ORIGINAL fc2 binary
    log "  [$label] applying $nops edits via fatbin-patch..."
    if python3 tools/sass_edit.py fatbin-patch "$OUTDIR/${label}_unpatched" \
        --sass "$sass" --script "$merged" -o "$patched" \
        > "$OUTDIR/${label}_patch.log" 2>&1; then
        chmod +x "$patched"
        log "  [$label] patched → $patched"
    else
        log "  [$label] PATCH FAILED"
        tail -5 "$OUTDIR/${label}_patch.log" >> "$OUTDIR/session.log"
        return 1
    fi

    # Cleanup
    rm -rf "$tmpdir"
}

run_binary() {
    local label="$1" binary="$2" tag="${3:-}"
    log "  [run] $label  binary=$binary"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  ./$binary"
        return 0
    fi

    if [ ! -x "$binary" ]; then
        log "  [$label] binary not found: $binary"
        return 1
    fi

    local output
    if ! output=$(./$binary 2>&1); then
        log "  [$label] RUN FAILED"
        echo "$output" >> "$OUTDIR/session.log"
        return 1
    fi

    echo "$output" > "$OUTDIR/${label}_output.txt"

    local result_line
    result_line=$(echo "$output" | grep '@@RESULT' | head -1)
    if [ -n "$result_line" ]; then
        echo "${result_line} label=${label} regs=${tag:-bin} dflags=\"${binary}\"" >> "$OUTDIR/wall_results.txt"
        local ms valid
        ms=$(echo "$result_line" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
        valid=$(echo "$result_line" | grep -o 'valid=[01]' | cut -d= -f2)
        log "  [$label] ${ms}ms  valid=${valid}  ${tag:+($tag)}"
    else
        log "  [$label] NO @@RESULT LINE"
    fi
}

# Build + schedule + patch: baseline 4w (the config that matters)
build_and_patch "base_4w" "$BASE"
build_and_patch "base_bs" "$BASE -DBIAS_SMEM=1"

# Tight A/B: unpatched → patched → CUTLASS (minimal thermal drift between comparisons)
log "  --- A/B comparison (tight) ---"
run_binary "base_4w_unp" "$OUTDIR/base_4w_unpatched" "unpatched"
run_binary "base_4w_pat" "base_4w_patched" "patched"
run_binary "base_bs_unp" "$OUTDIR/base_bs_unpatched" "unpatched"
run_binary "base_bs_pat" "base_bs_patched" "patched"
if [ -x "./cutlass-bench-fc2-max" ] && [ "$DRY_RUN" = "0" ]; then
    log "  [cutlass_ab] running..."
    cutlass_out=$(./cutlass-bench-fc2-max 2>&1)
    echo "$cutlass_out" > "$OUTDIR/cutlass_ab_output.txt"
    cutlass_ms=$(echo "$cutlass_out" | grep 'Best Fused:' | grep -oP '\d+\.\d+(?=\s+ms)')
    log "  [cutlass_ab] ${cutlass_ms:-ERR}ms"
    [ -n "$cutlass_ms" ] && echo "@@RESULT ms=${cutlass_ms} valid=1 label=cutlass_ab" >> "$OUTDIR/wall_results.txt"
fi

fi  # ortools check

# ════════════════════════════════════════
# Thermal drift check
# ════════════════════════════════════════
log ""
log "==== THERMAL DRIFT CHECK ===="
run_wall "strip_end" "$BASE -DSTRIP_EPILOGUE=1"

# ════════════════════════════════════════
# Summary
# ════════════════════════════════════════
if [ -f "$OUTDIR/wall_results.txt" ] && [ "$DRY_RUN" = "0" ]; then
    log ""
    log "==== WALL TIME SUMMARY ===="

    strip_ms=$(grep 'label=strip ' "$OUTDIR/wall_results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)

    {
        printf "%-20s %8s %8s %5s  %s\n" "LABEL" "MS" "DELTA" "REGS" "DFLAGS"
        printf "%-20s %8s %8s %5s  %s\n" "--------------------" "--------" "--------" "-----" "------"
        while IFS= read -r line; do
            ms=$(echo "$line" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
            label=$(echo "$line" | grep -o 'label=[^ ]*' | cut -d= -f2)
            regs=$(echo "$line" | grep -o 'regs=[0-9]*' | cut -d= -f2)
            dflags=$(echo "$line" | sed -n 's/.*dflags="\([^"]*\)".*/\1/p')
            valid=$(echo "$line" | grep -o 'valid=[01]' | cut -d= -f2)
            mark=""
            [ "$valid" = "0" ] && mark=" INV"

            delta="—"
            if [ -n "$strip_ms" ] && [ -n "$ms" ]; then
                delta=$(echo "$ms - $strip_ms" | bc -l 2>/dev/null | sed 's/^\./0./' || echo "?")
            fi

            printf "%-20s %8s %8s %5s  %s%s\n" "$label" "$ms" "$delta" "$regs" "$dflags" "$mark"
        done < "$OUTDIR/wall_results.txt"
    } | tee "$OUTDIR/wall_summary.txt"

    # SC2 no-reuse vs baseline (2 warps only — the only valid SC2 config with NS5)
    echo "" | tee -a "$OUTDIR/wall_summary.txt"
    echo "=== SC2 no-reuse vs BASELINE (2 warps) ===" | tee -a "$OUTDIR/wall_summary.txt"
    for pair in "nepi2:base_nepi2:sc2nr_nepi2" "nepi2_bs:base_nepi2:sc2nr_nepi2_bs"; do
        IFS=: read -r plabel blab slab <<< "$pair"
        bms=$(grep "label=${blab} " "$OUTDIR/wall_results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
        sms=$(grep "label=${slab} " "$OUTDIR/wall_results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
        if [ -n "$bms" ] && [ -n "$sms" ]; then
            diff_us=$(echo "($sms - $bms) * 1000" | bc -l 2>/dev/null | sed 's/^\./0./' | cut -c1-7 || echo "?")
            printf "  %-12s base=%sms  sc2nr=%sms  Δ=%sμs\n" "$plabel" "$bms" "$sms" "$diff_us" | tee -a "$OUTDIR/wall_summary.txt"
        fi
    done

    # SASS-patched A/B comparison (tight — same thermal window)
    echo "" | tee -a "$OUTDIR/wall_summary.txt"
    echo "=== SASS-PATCHED vs UNPATCHED (tight A/B) ===" | tee -a "$OUTDIR/wall_summary.txt"
    for pair in "base_4w:base_4w_unp:base_4w_pat" "base_bs:base_bs_unp:base_bs_pat"; do
        IFS=: read -r plabel blab slab <<< "$pair"
        bms=$(grep "label=${blab} " "$OUTDIR/wall_results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
        sms=$(grep "label=${slab} " "$OUTDIR/wall_results.txt" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
        if [ -n "$bms" ] && [ -n "$sms" ]; then
            diff_us=$(echo "($sms - $bms) * 1000" | bc -l 2>/dev/null | sed 's/^\./0./' | cut -c1-7 || echo "?")
            printf "  %-15s unpatched=%sms  patched=%sms  Δ=%sμs\n" "$plabel" "$bms" "$sms" "$diff_us" | tee -a "$OUTDIR/wall_summary.txt"
        fi
    done
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
log ""
log "Done in ${ELAPSED}s. Results: $OUTDIR/"
log "  wall_summary.txt  — wall time comparison"
log "  *_timing.txt      — cycle breakdowns"
log "  session.log       — full log"
