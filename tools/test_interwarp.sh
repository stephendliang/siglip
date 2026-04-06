#!/usr/bin/env bash
set -uo pipefail

#
# Inter-warp STS stagger experiments.
#
# Tests whether YIELD-based warp staggering reduces epilogue barrier stalls.
# Follows fc2_cutlass_vs_w3.sh conventions: timestamped output, N runs, @@RESULT parsing.
#
# Experiments:
#   1. baseline:     Unmodified fc2-w3 (fused + strip)
#   2. yield-all:    All epilogue NOPs → YIELD (fused + strip)
#   3. yield-pred:   Source-level P6 setup + SASS-patch @P6 YIELD (fused + strip)
#
# Usage:
#   ./tools/test_interwarp.sh              # run everything
#   ./tools/test_interwarp.sh --dry-run    # print commands without running
#   ./tools/test_interwarp.sh -N 10        # 10 runs per variant (default 5)
#

cd "$(dirname "$0")/.."

DRY_RUN=0
N_RUNS=5
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        -N) N_RUNS=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="data/interwarp_${TIMESTAMP}"
mkdir -p "$OUTDIR"

STAGGER="tools/interwarp_stagger.py"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/session.log"; }

log "========================================"
log "  INTER-WARP STAGGER EXPERIMENTS"
log "  N=$N_RUNS runs per variant"
log "  Output: $OUTDIR"
log "========================================"

if [ "$DRY_RUN" = "0" ]; then
    nvidia-smi > /dev/null 2>&1 || { log "FATAL: no GPU"; exit 1; }
    nvidia-smi --query-gpu=gpu_name,clocks.sm --format=csv,noheader | tee -a "$OUTDIR/session.log"
fi

# ── helpers ──────────────────────────────────────────────────────────

parse_result() {
    # Extract fields from @@RESULT line
    local line=$1 field=$2
    echo "$line" | grep -oP "${field}=\K[^ ]+" || echo "n/a"
}

run_n_times() {
    # Run binary N times, collect ms values, report min/median/max
    local bin=$1 label=$2 n=$3
    local ms_vals=()

    # Warmup (discarded)
    timeout 30 "$bin" > /dev/null 2>&1 || true

    for i in $(seq 1 "$n"); do
        local out
        out=$(timeout 30 "$bin" 2>&1) || { echo "RUN_FAIL"; return 1; }
        echo "$out" > "$OUTDIR/${label}_run${i}.txt"

        local result_line
        result_line=$(echo "$out" | grep '@@RESULT' | head -1)
        if [ -z "$result_line" ]; then
            echo "NO_RESULT"; return 1
        fi

        local ms valid checksum
        ms=$(parse_result "$result_line" "ms")
        valid=$(parse_result "$result_line" "valid")
        checksum=$(parse_result "$result_line" "checksum")

        if [ "$valid" != "1" ]; then
            log "  [$label] run $i: valid=$valid (known issue)"
        fi

        ms_vals+=("$ms")
        log "  [$label] run $i/$n: ${ms}ms checksum=$checksum"
    done

    # Sort and extract min/median/max
    local sorted
    sorted=$(printf '%s\n' "${ms_vals[@]}" | sort -n)
    local min_ms median_ms max_ms
    min_ms=$(echo "$sorted" | head -1)
    max_ms=$(echo "$sorted" | tail -1)
    local mid=$(( (n + 1) / 2 ))
    median_ms=$(echo "$sorted" | sed -n "${mid}p")

    # Save aggregate
    echo "min=$min_ms median=$median_ms max=$max_ms n=$n" > "$OUTDIR/${label}_stats.txt"
    printf '%s\n' "${ms_vals[@]}" >> "$OUTDIR/${label}_stats.txt"

    # Return the last @@RESULT line augmented with stats
    local last_result
    last_result=$(grep '@@RESULT' "$OUTDIR/${label}_run${n}.txt" | head -1)
    echo "${last_result} label=${label} min=${min_ms} median=${median_ms} max=${max_ms} n=${n}"
}

verify_sass() {
    # Dump SASS of a binary, optionally diff against baseline
    local bin=$1 label=$2 baseline_sass=${3:-}
    local sass="$OUTDIR/${label}_sass.txt"
    cuobjdump --dump-sass "$bin" > "$sass" 2>/dev/null || { log "  cuobjdump failed for $label"; return 1; }

    local n_yield n_nop n_ldtm
    n_yield=$(grep -c 'YIELD' "$sass" || true)
    n_nop=$(grep -c '  NOP ' "$sass" || true)
    n_ldtm=$(grep -c 'LDTM' "$sass" || true)
    log "  [$label] SASS: ${n_ldtm} LDTM, ${n_nop} NOP, ${n_yield} YIELD"

    if [ -n "$baseline_sass" ] && [ -f "$baseline_sass" ]; then
        diff "$baseline_sass" "$sass" > "$OUTDIR/${label}_sass_diff.txt" 2>&1 || true
        local n_diff
        n_diff=$(grep -c '^[<>]' "$OUTDIR/${label}_sass_diff.txt" || true)
        log "  [$label] SASS diff: $n_diff changed lines vs baseline"
    fi
}

# ── Experiment 1: Baseline ─────────────────────────────────────────

log ""
log "════════════════════════════════════════"
log "  EXP 1: BASELINE (unmodified fc2-w3)"
log "════════════════════════════════════════"

# Build fused
log "Building fc2-w3 (fused)..."
if [ "$DRY_RUN" = "1" ]; then
    echo "  make fc2-w3 && run N=$N_RUNS"
else
    if ! make fc2-w3 > "$OUTDIR/baseline_fused_build.log" 2>&1; then
        log "FATAL: fc2-w3 fused build failed"
        cat "$OUTDIR/baseline_fused_build.log"
        exit 1
    fi
    cp fc2-w3 "$OUTDIR/fc2-w3-baseline"

    # SASS dump (reference for diffs)
    verify_sass "./fc2-w3" "baseline_fused"
    BASELINE_SASS="$OUTDIR/baseline_fused_sass.txt"

    FUSED_RESULT=$(run_n_times "./fc2-w3" "baseline_fused" "$N_RUNS")
    if echo "$FUSED_RESULT" | grep -qE 'RUN_FAIL|NO_RESULT'; then
        log "FATAL: Baseline fused failed: $FUSED_RESULT"
        exit 1
    fi
    echo "$FUSED_RESULT" >> "$OUTDIR/results.txt"
    BASE_FUSED_MS=$(parse_result "$FUSED_RESULT" "median")
    BASE_CHECKSUM=$(parse_result "$FUSED_RESULT" "checksum")
    log "  baseline_fused: median=${BASE_FUSED_MS}ms"
fi

# Build strip
log "Building fc2-w3 (strip)..."
if [ "$DRY_RUN" = "1" ]; then
    echo "  make fc2-w3 DFLAGS=-DSTRIP_EPILOGUE && run N=$N_RUNS"
else
    if ! make fc2-w3 DFLAGS=-DSTRIP_EPILOGUE > "$OUTDIR/baseline_strip_build.log" 2>&1; then
        log "FATAL: fc2-w3 strip build failed"
        exit 1
    fi
    cp fc2-w3 "$OUTDIR/fc2-w3-strip"

    STRIP_RESULT=$(run_n_times "./fc2-w3" "baseline_strip" "$N_RUNS")
    if echo "$STRIP_RESULT" | grep -qE 'RUN_FAIL|NO_RESULT'; then
        log "FATAL: Baseline strip failed: $STRIP_RESULT"
        exit 1
    fi
    echo "$STRIP_RESULT" >> "$OUTDIR/results.txt"
    BASE_STRIP_MS=$(parse_result "$STRIP_RESULT" "median")
    log "  baseline_strip: median=${BASE_STRIP_MS}ms"
fi

# Rebuild fused for patching (strip overwrote the binary)
if [ "$DRY_RUN" = "0" ]; then
    make fc2-w3 > /dev/null 2>&1
fi

# ── Experiment 2: YIELD-ALL ────────────────────────────────────────

log ""
log "════════════════════════════════════════"
log "  EXP 2: YIELD-ALL (every epilogue NOP → YIELD)"
log "════════════════════════════════════════"

if [ "$DRY_RUN" = "1" ]; then
    echo "  python3 $STAGGER fc2-w3 --mode yield-only -o ... && run N=$N_RUNS"
else
    YIELD_BIN="$OUTDIR/fc2-w3-yield"
    python3 "$STAGGER" fc2-w3 --mode yield-only -o "$YIELD_BIN" 2>&1 | tee -a "$OUTDIR/session.log"

    if [ ! -f "$YIELD_BIN" ]; then
        log "ERROR: yield-only binary not generated"
    else
        verify_sass "$YIELD_BIN" "yield_fused" "$BASELINE_SASS"

        # Check YIELD count in SASS
        n_yield=$(grep -c 'YIELD' "$OUTDIR/yield_fused_sass.txt" || true)
        if [ "$n_yield" -eq 0 ]; then
            log "ERROR: No YIELD instructions in patched binary — patches didn't land"
        else
            YIELD_RESULT=$(run_n_times "$YIELD_BIN" "yield_fused" "$N_RUNS")
            if echo "$YIELD_RESULT" | grep -qE 'RUN_FAIL|NO_RESULT|INVALID'; then
                log "ERROR: yield_fused failed: $YIELD_RESULT"
            else
                echo "$YIELD_RESULT" >> "$OUTDIR/results.txt"
                YIELD_MS=$(parse_result "$YIELD_RESULT" "median")
                YIELD_CS=$(parse_result "$YIELD_RESULT" "checksum")
                log "  yield_fused: median=${YIELD_MS}ms"

                if [ "$YIELD_CS" != "$BASE_CHECKSUM" ]; then
                    log "  WARNING: checksum mismatch! yield=$YIELD_CS baseline=$BASE_CHECKSUM"
                fi
            fi
        fi
    fi

    # Strip variant: patch the strip binary
    log "Building fc2-w3 strip for yield patching..."
    make fc2-w3 DFLAGS=-DSTRIP_EPILOGUE > /dev/null 2>&1
    YIELD_STRIP_BIN="$OUTDIR/fc2-w3-yield-strip"
    python3 "$STAGGER" fc2-w3 --mode yield-only -o "$YIELD_STRIP_BIN" 2>&1 | tee -a "$OUTDIR/session.log"

    if [ -f "$YIELD_STRIP_BIN" ]; then
        YIELD_STRIP_RESULT=$(run_n_times "$YIELD_STRIP_BIN" "yield_strip" "$N_RUNS")
        if ! echo "$YIELD_STRIP_RESULT" | grep -qE 'RUN_FAIL|NO_RESULT|INVALID'; then
            echo "$YIELD_STRIP_RESULT" >> "$OUTDIR/results.txt"
            YIELD_STRIP_MS=$(parse_result "$YIELD_STRIP_RESULT" "median")
            log "  yield_strip: median=${YIELD_STRIP_MS}ms"
        fi
    fi

    # Rebuild fused for next experiment
    make fc2-w3 > /dev/null 2>&1
fi

# ── Experiment 3: Predicated @P6 YIELD (source-level P6 setup) ────

log ""
log "════════════════════════════════════════"
log "  EXP 3: PREDICATED @P6 YIELD"
log "  (source: WARP_STAGGER sets P6, SASS-patch: NOP→@P6 YIELD)"
log "════════════════════════════════════════"

if [ "$DRY_RUN" = "1" ]; then
    echo "  make fc2-w3 DFLAGS=-DWARP_STAGGER && python3 $STAGGER ... --mode yield-only -o ..."
    echo "  (requires WARP_STAGGER support in fc2_w3.cu)"
else
    # Check if WARP_STAGGER is implemented
    if grep -q 'WARP_STAGGER' fc2_w3.cu 2>/dev/null; then
        log "Building fc2-w3 with WARP_STAGGER..."
        if make fc2-w3 DFLAGS=-DWARP_STAGGER > "$OUTDIR/pred_build.log" 2>&1; then
            cp fc2-w3 "$OUTDIR/fc2-w3-stagger-base"

            # Patch: NOP → @P6 YIELD (the stagger.py in yield-only mode but with pred)
            PRED_BIN="$OUTDIR/fc2-w3-pred"
            python3 "$STAGGER" fc2-w3 --mode yield-only --pred-reg 6 -o "$PRED_BIN" 2>&1 | tee -a "$OUTDIR/session.log"

            if [ -f "$PRED_BIN" ]; then
                verify_sass "$PRED_BIN" "pred_fused" "$BASELINE_SASS"

                PRED_RESULT=$(run_n_times "$PRED_BIN" "pred_fused" "$N_RUNS")
                if ! echo "$PRED_RESULT" | grep -qE 'RUN_FAIL|NO_RESULT|INVALID'; then
                    echo "$PRED_RESULT" >> "$OUTDIR/results.txt"
                    PRED_MS=$(parse_result "$PRED_RESULT" "median")
                    PRED_CS=$(parse_result "$PRED_RESULT" "checksum")
                    log "  pred_fused: median=${PRED_MS}ms"

                    if [ "$PRED_CS" != "$BASE_CHECKSUM" ]; then
                        log "  WARNING: checksum mismatch! pred=$PRED_CS baseline=$BASE_CHECKSUM"
                    fi
                else
                    log "  pred_fused: FAILED ($PRED_RESULT)"
                fi
            fi

            # Strip variant
            log "Building fc2-w3 strip+WARP_STAGGER..."
            if make fc2-w3 DFLAGS="-DSTRIP_EPILOGUE -DWARP_STAGGER" > /dev/null 2>&1; then
                PRED_STRIP_BIN="$OUTDIR/fc2-w3-pred-strip"
                python3 "$STAGGER" fc2-w3 --mode yield-only --pred-reg 6 -o "$PRED_STRIP_BIN" 2>&1 | tee -a "$OUTDIR/session.log"

                if [ -f "$PRED_STRIP_BIN" ]; then
                    PRED_STRIP_RESULT=$(run_n_times "$PRED_STRIP_BIN" "pred_strip" "$N_RUNS")
                    if ! echo "$PRED_STRIP_RESULT" | grep -qE 'RUN_FAIL|NO_RESULT|INVALID'; then
                        echo "$PRED_STRIP_RESULT" >> "$OUTDIR/results.txt"
                        PRED_STRIP_MS=$(parse_result "$PRED_STRIP_RESULT" "median")
                        log "  pred_strip: median=${PRED_STRIP_MS}ms"
                    fi
                fi
            fi
        else
            log "  Build failed (WARP_STAGGER not implemented yet?)"
            cat "$OUTDIR/pred_build.log" | tail -5
        fi
    else
        log "  SKIPPED: WARP_STAGGER not in fc2_w3.cu yet"
        log "  To enable: add #ifdef WARP_STAGGER block in epilogue that sets P6 = (warp_id & 1)"
    fi
fi

# ── Summary ────────────────────────────────────────────────────────

log ""
log "════════════════════════════════════════"
log "  SUMMARY"
log "════════════════════════════════════════"
log ""

if [ "$DRY_RUN" = "0" ] && [ -f "$OUTDIR/results.txt" ]; then
    printf "  %-20s %9s %9s %9s  %s\n" "variant" "min" "median" "max" "checksum" | tee -a "$OUTDIR/summary.txt"
    printf "  %-20s %9s %9s %9s  %s\n" "────────────" "──────" "──────" "──────" "────────" | tee -a "$OUTDIR/summary.txt"

    while IFS= read -r line; do
        label=$(parse_result "$line" "label")
        min_ms=$(parse_result "$line" "min")
        median_ms=$(parse_result "$line" "median")
        max_ms=$(parse_result "$line" "max")
        checksum=$(parse_result "$line" "checksum")
        valid=$(parse_result "$line" "valid")

        printf "  %-20s %8sms %8sms %8sms  %s\n" \
            "$label" "$min_ms" "$median_ms" "$max_ms" "$checksum"
    done < "$OUTDIR/results.txt" | tee -a "$OUTDIR/summary.txt"

    log ""

    # Decomposition
    if [ -n "${BASE_FUSED_MS:-}" ] && [ -n "${BASE_STRIP_MS:-}" ]; then
        epi_base=$(echo "$BASE_FUSED_MS - $BASE_STRIP_MS" | bc 2>/dev/null || echo "n/a")
        log "  Baseline epilogue overhead: ${epi_base}ms (${BASE_FUSED_MS} - ${BASE_STRIP_MS})"
    fi
    if [ -n "${YIELD_MS:-}" ] && [ -n "${YIELD_STRIP_MS:-}" ]; then
        epi_yield=$(echo "$YIELD_MS - $YIELD_STRIP_MS" | bc 2>/dev/null || echo "n/a")
        log "  Yield-all epilogue overhead: ${epi_yield}ms (${YIELD_MS} - ${YIELD_STRIP_MS})"
    fi
    if [ -n "${PRED_MS:-}" ] && [ -n "${PRED_STRIP_MS:-}" ]; then
        epi_pred=$(echo "$PRED_MS - $PRED_STRIP_MS" | bc 2>/dev/null || echo "n/a")
        log "  Predicated epilogue overhead: ${epi_pred}ms (${PRED_MS} - ${PRED_STRIP_MS})"
    fi

    log ""
    log "Results: $OUTDIR/results.txt"
    log "Summary: $OUTDIR/summary.txt"
    log "SASS diffs: $OUTDIR/*_sass_diff.txt"
fi

log "Done."
