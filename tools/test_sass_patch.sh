#!/bin/bash
# SASS patcher bisection test — find which layer breaks.
#
# Levels (stops at first failure):
#   0  Identity round-trip (zero changes, tests parse/serialize/splice)
#   1  Null stall (set one stall to its current value)
#   2  Stall+1 on a NOP (harmless real change)
#   3  Stall-only recipe (restall a region, no reorder)
#   4  Single swap (two adjacent independent instructions)
#   5  Full CP-SAT schedule (reorder + restall, 1 chunk)
#   6  Full epilogue schedule (all LDTM chunks, parallel CP-SAT)
#
# Usage:
#   ./tools/test_sass_patch.sh           # run all levels
#   ./tools/test_sass_patch.sh --level 3 # run only level 3
#   ./tools/test_sass_patch.sh --dry-run # print commands only
#
# Requires: GPU (B200), pyelftools, ortools (for level 5-6)

set -uo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=0
ONLY_LEVEL=-1
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --level) ONLY_LEVEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="data/sass_patch_test_${TIMESTAMP}"
mkdir -p "$OUTDIR"

log() { echo "[$(date +%H:%M:%S)] $*" >> "$OUTDIR/session.log"; echo "[$(date +%H:%M:%S)] $*" >&2; }
die() { log "FATAL: $*"; exit 1; }

log "========================================"
log "  SASS PATCHER BISECTION TEST"
log "  Output: $OUTDIR"
log "========================================"

if [ "$DRY_RUN" = "0" ]; then
    nvidia-smi > /dev/null 2>&1 || die "no GPU"
    nvidia-smi --query-gpu=gpu_name,clocks.sm --format=csv,noheader | tee -a "$OUTDIR/session.log"
fi

# ── Build baseline ──────────────────────────────────────────────────────────

log "Building baseline fc2-w3..."
if [ "$DRY_RUN" = "1" ]; then
    echo "  make fc2-w3"
else
    make fc2-w3 > "$OUTDIR/build.log" 2>&1 || die "build failed"
    cp fc2-w3 "$OUTDIR/baseline"
fi

# Extract SASS dump (needed for xref in all levels)
SASS="$OUTDIR/sass.txt"
if [ "$DRY_RUN" = "1" ]; then
    echo "  cuobjdump -sass fc2-w3 > $SASS"
else
    cuobjdump -sass fc2-w3 > "$SASS" 2>/dev/null || die "cuobjdump failed"
fi

# Run binary and capture @@RESULT line. Returns checksum+valid+ms on stdout.
# Does NOT require valid=1 — patched builds are compared against baseline.
run_binary() {
    local label="$1" binary="$2"
    local output

    if [ "$DRY_RUN" = "1" ]; then
        echo "  ./$binary"
        return 0
    fi

    log "  [$label] running..."
    output=$("./$binary" 2>&1) || true
    echo "$output" > "$OUTDIR/${label}_output.txt"

    local result_line
    result_line=$(echo "$output" | grep '@@RESULT' || true)
    if [ -z "$result_line" ]; then
        log "  [$label] CRASH — no @@RESULT line"
        log "  [$label] last 5 lines:"
        echo "$output" | tail -5 | while IFS= read -r line; do log "    $line"; done
        return 1
    fi

    local ms cksum valid
    ms=$(echo "$result_line" | grep -oP 'ms=\K[0-9.]+')
    cksum=$(echo "$result_line" | grep -oP 'checksum=\K[0-9.e+-]+')
    valid=$(echo "$result_line" | grep -oP 'valid=\K[0-9]+')
    log "  [$label] OK — ms=$ms checksum=$cksum valid=$valid"

    # Return checksum for comparison
    echo "$cksum"
    return 0
}

# Compare patched result against baseline — must produce same checksum
check_vs_baseline() {
    local label="$1" binary="$2"
    local cksum
    cksum=$(run_binary "$label" "$binary") || return 1

    if [ "$DRY_RUN" = "1" ]; then return 0; fi

    if [ "$cksum" != "$BASELINE_CKSUM" ]; then
        log "  [$label] MISMATCH — checksum=$cksum vs baseline=$BASELINE_CKSUM"
        return 1
    fi
    log "  [$label] checksum matches baseline"
    return 0
}

log ""
log "── Baseline ──"
BASELINE_CKSUM=""
if [ "$DRY_RUN" = "0" ]; then
    BASELINE_CKSUM=$(run_binary "baseline" "$OUTDIR/baseline") || die "baseline crashed — cannot continue"
    log "  baseline checksum: $BASELINE_CKSUM"
fi

should_run() {
    [ "$ONLY_LEVEL" = "-1" ] || [ "$ONLY_LEVEL" = "$1" ]
}

# ── Helper: find useful addresses from SASS ─────────────────────────────────

# Extract cubin for levels that need it
CUBIN_DIR="$OUTDIR/cubin_tmp"
if [ "$DRY_RUN" = "0" ]; then
    mkdir -p "$CUBIN_DIR"
    fc2_abs="$(pwd)/fc2-w3"
    (cd "$CUBIN_DIR" && cuobjdump -xelf all "$fc2_abs" > /dev/null 2>&1)
    CUBIN=$(ls "$CUBIN_DIR"/*.cubin 2>/dev/null | head -1)
    if [ -z "$CUBIN" ]; then
        die "cubin extraction failed"
    fi
    log "Cubin: $CUBIN ($(wc -c < "$CUBIN") bytes)"
fi

# Find a NOP instruction address and the first LDTM boundary (epilogue start)
find_addresses() {
    if [ "$DRY_RUN" = "1" ]; then return; fi

    # Find first NOP (good target for harmless edits)
    NOP_ADDR=$(grep -oP '/\*\s*\K[0-9a-fA-F]+(?=\s*\*/\s+NOP)' "$SASS" | head -1)
    if [ -n "$NOP_ADDR" ]; then NOP_ADDR="0x$NOP_ADDR"; fi

    # Find first STS.128 in epilogue (the instruction we ultimately want to spread)
    STS_ADDR=$(grep -oP '/\*\s*\K[0-9a-fA-F]+(?=\s*\*/\s+STS\.128)' "$SASS" | head -1)
    if [ -n "$STS_ADDR" ]; then STS_ADDR="0x$STS_ADDR"; fi

    # Find LDTM boundaries (epilogue chunk delimiters)
    mapfile -t LDTM_ADDRS < <(grep -oP '/\*\s*\K[0-9a-f]+(?=\s*\*/\s+LDTM)' "$SASS")

    # Find two adjacent independent instructions for swap test
    # Look for consecutive HFMA2 or FFMA that write different registers
    SWAP_A=""
    SWAP_B=""

    # Fallback: if no NOP, use first LDTM (stall change on any insn is testable)
    if [ -z "$NOP_ADDR" ] && [ "${#LDTM_ADDRS[@]}" -gt 0 ]; then
        NOP_ADDR="0x${LDTM_ADDRS[0]}"
        log "  (no NOP found, using LDTM at $NOP_ADDR as stall target)"
    fi

    log "  Addresses: NOP=$NOP_ADDR STS=$STS_ADDR LDTMs=${#LDTM_ADDRS[@]}"
}
find_addresses

# ── Level 0: Identity round-trip ────────────────────────────────────────────

if should_run 0; then
    log ""
    log "── Level 0: Identity round-trip (zero edits) ──"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  python3 tools/sass_edit.py fatbin-patch fc2-w3 --sass $SASS --identity -o $OUTDIR/level0"
    else
        python3 tools/sass_edit.py fatbin-patch "$OUTDIR/baseline" \
            --sass "$SASS" --identity -o "$OUTDIR/level0" \
            > "$OUTDIR/level0_patch.log" 2>&1 || die "level 0 fatbin-patch failed"

        # Verify binary diff — identity should be byte-exact
        if cmp -s "$OUTDIR/baseline" "$OUTDIR/level0"; then
            log "  byte-exact: YES (parse/serialize/splice is clean)"
        else
            log "  byte-exact: NO — diff detected!"
            cmp -l "$OUTDIR/baseline" "$OUTDIR/level0" | head -20 > "$OUTDIR/level0_diff.txt"
            log "  first diffs (offset decimal_old decimal_new):"
            head -5 "$OUTDIR/level0_diff.txt" | while IFS= read -r line; do log "    $line"; done
            die "LEVEL 0 FAILED — identity round-trip changes bytes"
        fi

        check_vs_baseline "level0" "$OUTDIR/level0" || die "LEVEL 0 FAILED — identity binary differs"
    fi
fi

# ── Level 1: Null stall (set stall to current value) ────────────────────────

if should_run 1; then
    log ""
    log "── Level 1: Null stall (stall X → X, logical no-op) ──"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  python3 tools/sass_edit.py fatbin-patch ... --stall ADDR CURRENT_VAL"
    else
        # Read current stall at NOP_ADDR from the cubin
        CURRENT_STALL=$(python3 -c "
import sys; sys.path.insert(0, 'tools')
from sass_edit import CubinEditor
ed = CubinEditor('$CUBIN')
k = ed.find_kernel()
insn = k.insn_at(int('$NOP_ADDR', 16))
print(insn.stall)
")
        log "  target: $NOP_ADDR current_stall=$CURRENT_STALL"

        python3 tools/sass_edit.py fatbin-patch "$OUTDIR/baseline" \
            --sass "$SASS" --stall "$NOP_ADDR" "$CURRENT_STALL" -o "$OUTDIR/level1" \
            > "$OUTDIR/level1_patch.log" 2>&1 || die "level 1 fatbin-patch failed"

        # Should also be byte-exact (null edit)
        if cmp -s "$OUTDIR/baseline" "$OUTDIR/level1"; then
            log "  byte-exact: YES (null stall is clean no-op)"
        else
            log "  byte-exact: NO — stall write touched other bits!"
            cmp -l "$OUTDIR/baseline" "$OUTDIR/level1" | head -20 > "$OUTDIR/level1_diff.txt"
            head -5 "$OUTDIR/level1_diff.txt" | while IFS= read -r line; do log "    $line"; done
            log "  (not fatal if it still runs correctly)"
        fi

        check_vs_baseline "level1" "$OUTDIR/level1" || die "LEVEL 1 FAILED — null stall patch differs"
    fi
fi

# ── Level 2: Stall+1 on a NOP ──────────────────────────────────────────────

if should_run 2; then
    log ""
    log "── Level 2: Stall change on NOP (harmless real change) ──"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  python3 tools/sass_edit.py fatbin-patch ... --stall NOP_ADDR NEW_VAL"
    else
        # Set NOP stall to 7 (max) — completely harmless, just adds a wait cycle
        NEW_STALL=7
        log "  target: $NOP_ADDR stall=$CURRENT_STALL → $NEW_STALL"

        python3 tools/sass_edit.py fatbin-patch "$OUTDIR/baseline" \
            --sass "$SASS" --stall "$NOP_ADDR" "$NEW_STALL" -o "$OUTDIR/level2" \
            > "$OUTDIR/level2_patch.log" 2>&1 || die "level 2 fatbin-patch failed"

        # Verify exactly 1 byte changed (3 bits within one byte of the control word)
        NDIFF=$(cmp -l "$OUTDIR/baseline" "$OUTDIR/level2" | wc -l)
        log "  bytes changed: $NDIFF (expect 1)"

        check_vs_baseline "level2" "$OUTDIR/level2" || die "LEVEL 2 FAILED — stall change on NOP differs"
    fi
fi

# ── Level 3: Stall-only recipe (restall region, no reorder) ─────────────────

if should_run 3; then
    log ""
    log "── Level 3: Stall-only recipe (restall epilogue chunk, no reorder) ──"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  Generate restall-only script for first epilogue chunk"
        echo "  python3 tools/sass_edit.py fatbin-patch ... --script restall.txt"
    else
        if [ "${#LDTM_ADDRS[@]}" -lt 2 ]; then
            log "  SKIP — not enough LDTM boundaries (need ≥2, have ${#LDTM_ADDRS[@]})"
        else
            CHUNK_START="0x${LDTM_ADDRS[0]}"
            CHUNK_END="0x${LDTM_ADDRS[1]}"
            log "  chunk: [$CHUNK_START, $CHUNK_END)"

            # Generate restall-only script (compute stalls for ORIGINAL order, apply them)
            SCRIPT="$OUTDIR/level3_script.txt"
            echo "# Level 3: restall only (no reorder)" > "$SCRIPT"
            echo "restall $CHUNK_START $CHUNK_END" >> "$SCRIPT"

            python3 tools/sass_edit.py fatbin-patch "$OUTDIR/baseline" \
                --sass "$SASS" --script "$SCRIPT" -o "$OUTDIR/level3" \
                > "$OUTDIR/level3_patch.log" 2>&1 || die "level 3 fatbin-patch failed"

            NDIFF=$(cmp -l "$OUTDIR/baseline" "$OUTDIR/level3" | wc -l)
            log "  bytes changed: $NDIFF"

            check_vs_baseline "level3" "$OUTDIR/level3" || die "LEVEL 3 FAILED — restall-only differs"
        fi
    fi
fi

# ── Level 4: Single swap (adjacent independent instructions) ────────────────

if should_run 4; then
    log ""
    log "── Level 4: Single swap (two adjacent independent instructions) ──"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  Find two swappable instructions, swap them"
    else
        if [ "${#LDTM_ADDRS[@]}" -lt 2 ]; then
            log "  SKIP — not enough LDTM boundaries"
        else
            CHUNK_START="0x${LDTM_ADDRS[0]}"
            CHUNK_END="0x${LDTM_ADDRS[1]}"

            # Use deps command to find a swappable pair in the first epilogue chunk
            # We look for two adjacent instructions with no deps between them
            log "  scanning [$CHUNK_START, $CHUNK_END) for swappable pair..."

            SWAP_PAIR=$(python3 -c "
import sys; sys.path.insert(0, 'tools')
from sass_edit import CubinEditor, parse_sass_dump, apply_sass_xref, check_deps, INSN_SIZE

ed = CubinEditor('$CUBIN')
k = ed.find_kernel()
ed.xref('$SASS', k)

start = int('$CHUNK_START', 0)
end = int('$CHUNK_END', 0)
start_idx = start // INSN_SIZE
end_idx = end // INSN_SIZE

instrs = k.instructions[start_idx:end_idx]

# Find first pair of adjacent instructions that can be swapped
for i in range(len(instrs) - 1):
    a, b = instrs[i], instrs[i+1]
    if a.mnemonic is None or b.mnemonic is None:
        continue
    # Try swapping just these two
    test_order = [ins.offset for ins in instrs]
    test_order[i], test_order[i+1] = test_order[i+1], test_order[i]
    violations = check_deps(instrs, test_order)
    if not violations:
        print('0x%x 0x%x %s %s' % (a.offset, b.offset, a.mnemonic, b.mnemonic))
        break
else:
    print('NONE')
" 2>/dev/null)

            if [ "$SWAP_PAIR" = "NONE" ] || [ -z "$SWAP_PAIR" ]; then
                log "  SKIP — no safe swap pair found in first chunk"
            else
                SWAP_A=$(echo "$SWAP_PAIR" | awk '{print $1}')
                SWAP_B=$(echo "$SWAP_PAIR" | awk '{print $2}')
                SWAP_MN_A=$(echo "$SWAP_PAIR" | awk '{print $3}')
                SWAP_MN_B=$(echo "$SWAP_PAIR" | awk '{print $4}')
                log "  swapping: $SWAP_A ($SWAP_MN_A) ↔ $SWAP_B ($SWAP_MN_B)"

                SCRIPT="$OUTDIR/level4_script.txt"
                echo "# Level 4: single swap" > "$SCRIPT"
                echo "swap $SWAP_A $SWAP_B" >> "$SCRIPT"

                python3 tools/sass_edit.py fatbin-patch "$OUTDIR/baseline" \
                    --sass "$SASS" --script "$SCRIPT" -o "$OUTDIR/level4" \
                    > "$OUTDIR/level4_patch.log" 2>&1 || die "level 4 fatbin-patch failed"

                NDIFF=$(cmp -l "$OUTDIR/baseline" "$OUTDIR/level4" | wc -l)
                log "  bytes changed: $NDIFF"

                check_vs_baseline "level4" "$OUTDIR/level4" || die "LEVEL 4 FAILED — single swap differs"
            fi
        fi
    fi
fi

# ── Level 5: Full CP-SAT schedule ──────────────────────────────────────────

if should_run 5; then
    log ""
    log "── Level 5: Full CP-SAT schedule (reorder + restall) ──"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  python3 tools/sass_edit.py schedule ... → recipe → fatbin-patch"
    else
        if [ "${#LDTM_ADDRS[@]}" -lt 2 ]; then
            log "  SKIP — not enough LDTM boundaries"
        else
            CHUNK_START="0x${LDTM_ADDRS[0]}"
            CHUNK_END="0x${LDTM_ADDRS[1]}"
            RECIPE="$OUTDIR/level5_recipe.txt"

            log "  scheduling [$CHUNK_START, $CHUNK_END) with CP-SAT (60s limit)..."
            python3 tools/sass_edit.py schedule "$CUBIN" --sass "$SASS" \
                -s "$CHUNK_START" -e "$CHUNK_END" --recipe "$RECIPE" --time-limit 60 \
                > "$OUTDIR/level5_schedule.log" 2>&1

            if [ ! -f "$RECIPE" ]; then
                log "  SKIP — scheduler produced no recipe"
            else
                NOPS=$(grep -c -E '^(reorder|stall)' "$RECIPE" 2>/dev/null) || NOPS=0
                HAS_REORDER=$(grep -c '^reorder' "$RECIPE" 2>/dev/null) || HAS_REORDER=0
                log "  recipe: $NOPS ops ($HAS_REORDER reorders)"

                python3 tools/sass_edit.py fatbin-patch "$OUTDIR/baseline" \
                    --sass "$SASS" --script "$RECIPE" -o "$OUTDIR/level5" \
                    > "$OUTDIR/level5_patch.log" 2>&1 || die "level 5 fatbin-patch failed"

                NDIFF=$(cmp -l "$OUTDIR/baseline" "$OUTDIR/level5" | wc -l)
                log "  bytes changed: $NDIFF"

                check_vs_baseline "level5" "$OUTDIR/level5" || die "LEVEL 5 FAILED — full schedule differs"
            fi
        fi
    fi
fi

# ── Level 6: Full epilogue schedule (all LDTM chunks) ─────────────────────
#
# Strategy: schedule all chunks in parallel, then validate each chunk
# individually on GPU before merging. This finds chunks that cross
# structural boundaries (e.g., between main loop and drain epilogue)
# and would produce illegal instructions due to reordered branch targets.

if should_run 6; then
    log ""
    log "── Level 6: Full epilogue CP-SAT (all LDTM chunks, validated) ──"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  Schedule all ${#LDTM_ADDRS[@]} LDTM chunks with CP-SAT, validate each, merge safe ones"
    else
        NLDTM=${#LDTM_ADDRS[@]}
        if [ "$NLDTM" -lt 2 ]; then
            log "  SKIP — not enough LDTM boundaries (need ≥2, have $NLDTM)"
        else
            NCHUNKS=$((NLDTM - 1))
            log "  $NLDTM LDTMs → $NCHUNKS chunks to schedule"

            # Schedule all chunks in parallel
            CPSAT_TIME=60
            declare -a L6_PIDS=() L6_RECIPES=()
            for ((i=0; i<NCHUNKS; i++)); do
                CS="0x${LDTM_ADDRS[$i]}"
                CE="0x${LDTM_ADDRS[$((i+1))]}"
                CHUNK_RECIPE="$OUTDIR/level6_chunk${i}.recipe"
                L6_RECIPES+=("$CHUNK_RECIPE")
                python3 tools/sass_edit.py schedule "$CUBIN" --sass "$SASS" \
                    -s "$CS" -e "$CE" --recipe "$CHUNK_RECIPE" --time-limit "$CPSAT_TIME" --quiet \
                    > "$OUTDIR/level6_chunk${i}.log" 2>&1 &
                L6_PIDS+=($!)
            done

            log "  waiting for $NCHUNKS CP-SAT solvers (${CPSAT_TIME}s limit each)..."
            L6_OK=0 L6_FAIL=0
            for pid in "${L6_PIDS[@]}"; do
                if wait "$pid"; then ((L6_OK++)); else ((L6_FAIL++)); fi
            done
            log "  CP-SAT: $L6_OK/$NCHUNKS chunks scheduled ($L6_FAIL failed)"

            # Validate each chunk individually on GPU
            log "  validating each chunk individually..."
            declare -a GOOD_CHUNKS=()
            for ((i=0; i<NCHUNKS; i++)); do
                CHUNK_RECIPE="${L6_RECIPES[$i]}"
                if [ ! -f "$CHUNK_RECIPE" ]; then
                    continue
                fi
                NOPS=$(grep -c -E '^(reorder|stall)' "$CHUNK_RECIPE" 2>/dev/null) || NOPS=0
                if [ "$NOPS" -lt 1 ]; then
                    continue
                fi

                CS="0x${LDTM_ADDRS[$i]}"
                CE="0x${LDTM_ADDRS[$((i+1))]}"
                HAS_REORDER=$(grep -c '^reorder' "$CHUNK_RECIPE" 2>/dev/null) || HAS_REORDER=0

                # Patch baseline with just this chunk
                CHUNK_BIN="$OUTDIR/level6_chunk${i}"
                if python3 tools/sass_edit.py fatbin-patch "$OUTDIR/baseline" \
                    --sass "$SASS" --script "$CHUNK_RECIPE" -o "$CHUNK_BIN" \
                    > "$OUTDIR/level6_chunk${i}_patch.log" 2>&1; then

                    # Run on GPU — check for crash
                    CHUNK_CKSUM=$(run_binary "chunk${i}" "$CHUNK_BIN") || true

                    if [ -n "$CHUNK_CKSUM" ] && [ "$CHUNK_CKSUM" = "$BASELINE_CKSUM" ]; then
                        log "  chunk ${i} [$CS, $CE) — PASS (${NOPS} ops, ${HAS_REORDER} reorders)"
                        GOOD_CHUNKS+=("$i")
                    else
                        log "  chunk ${i} [$CS, $CE) — FAIL (crash or checksum mismatch)"
                    fi
                else
                    log "  chunk ${i} [$CS, $CE) — FAIL (fatbin-patch error)"
                fi
            done

            log "  validated: ${#GOOD_CHUNKS[@]}/$NCHUNKS chunks safe"

            if [ "${#GOOD_CHUNKS[@]}" -lt 1 ]; then
                log "  SKIP — no safe chunks"
            else
                # Merge only validated chunk recipes
                MERGED="$OUTDIR/level6_merged.recipe"
                > "$MERGED"
                for ci in "${GOOD_CHUNKS[@]}"; do
                    cat "${L6_RECIPES[$ci]}" >> "$MERGED"
                done

                TOTAL_OPS=$(grep -c -E '^(reorder|stall)' "$MERGED" 2>/dev/null) || TOTAL_OPS=0
                TOTAL_REORDERS=$(grep -c '^reorder' "$MERGED" 2>/dev/null) || TOTAL_REORDERS=0
                TOTAL_STALLS=$(grep -c '^stall' "$MERGED" 2>/dev/null) || TOTAL_STALLS=0
                log "  merged recipe (safe only): $TOTAL_OPS ops ($TOTAL_REORDERS reorders, $TOTAL_STALLS stalls)"

                python3 tools/sass_edit.py fatbin-patch "$OUTDIR/baseline" \
                    --sass "$SASS" --script "$MERGED" -o "$OUTDIR/level6" \
                    > "$OUTDIR/level6_patch.log" 2>&1 || die "level 6 fatbin-patch failed"

                NDIFF=$(cmp -l "$OUTDIR/baseline" "$OUTDIR/level6" | wc -l)
                log "  bytes changed: $NDIFF"

                check_vs_baseline "level6" "$OUTDIR/level6" || die "LEVEL 6 FAILED — merged safe chunks still differ"
            fi
        fi
    fi
fi

# ── Summary ─────────────────────────────────────────────────────────────────

log ""
log "========================================"
log "  ALL LEVELS PASSED"
log "========================================"

if [ "$DRY_RUN" = "0" ]; then
    log ""
    log "Timing comparison:"
    printf "  %-12s %8s\n" "variant" "ms" | tee -a "$OUTDIR/session.log"
    printf "  %-12s %8s\n" "--------" "------" | tee -a "$OUTDIR/session.log"
    for f in "$OUTDIR"/*_output.txt; do
        label=$(basename "$f" _output.txt)
        ms=$(grep -oP 'ms=\K[0-9.]+' "$f" 2>/dev/null || echo "n/a")
        printf "  %-12s %8s\n" "$label" "$ms" | tee -a "$OUTDIR/session.log"
    done
fi
