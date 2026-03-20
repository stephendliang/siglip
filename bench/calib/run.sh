#!/bin/bash
# Generate, build, run, and SASS-verify calibration benchmarks.
# Usage:
#   ./bench/calib/run.sh              # all three (tput, lat, conflict)
#   ./bench/calib/run.sh tput         # throughput ILP sweep only
#   ./bench/calib/run.sh tput lat     # multiple targets
#   ./bench/calib/run.sh --sass-only  # just dump+verify SASS for already-built binaries

set -euo pipefail
cd "$(dirname "$0")/../.."

NVCC="nvcc"
CFLAGS="-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17"
OUTDIR="bench/calib"
SASSDIR="$OUTDIR/sass"
GEN="$OUTDIR/gen_kernels.py"
DB="$OUTDIR/instruction_db.py"

SASS_ONLY=0
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--sass-only" ]; then SASS_ONLY=1
    else ARGS+=("$arg"); fi
done

if [ ${#ARGS[@]} -eq 0 ]; then
    TARGETS=(tput lat conflict)
else
    TARGETS=("${ARGS[@]}")
fi

log() { echo "[$(date +%H:%M:%S)] $*"; }

mkdir -p "$SASSDIR"

sass_verify() {
    local bin="$1" name="$2" sassfile="$3"

    log "Dumping SASS: $bin -> $sassfile"
    cuobjdump --dump-sass "$bin" > "$sassfile"

    local n_funcs=$(grep -c 'Function :' "$sassfile" || true)
    log "  $n_funcs kernel functions in SASS"

    if [ "$name" = "conflict" ]; then
        log "  Conflict interleaving check (first 5 kernels):"
        grep 'Function :' "$sassfile" | head -5 | while read -r line; do
            local func=$(echo "$line" | sed 's/.*Function : //')
            local sass_block=$(sed -n "/Function : ${func}/,/Function :/p" "$sassfile" | head -80)

            local insn_types=$(echo "$sass_block" | grep -oP '^\s+/\*[^*]+\*/\s+\K[A-Z0-9_.]+' | head -30)
            local unique=$(echo "$insn_types" | sort -u | tr '\n' ' ')
            local transitions=0
            local prev=""
            while read -r op; do
                [ -z "$op" ] && continue
                local base=${op%%.*}
                if [ -n "$prev" ] && [ "$base" != "$prev" ]; then
                    transitions=$((transitions + 1))
                fi
                prev="$base"
            done <<< "$insn_types"

            local n_insns=$(echo "$insn_types" | wc -l)
            local ratio=0
            [ "$n_insns" -gt 1 ] && ratio=$((transitions * 100 / (n_insns - 1)))
            local verdict="INTERLEAVED"
            [ "$ratio" -lt 30 ] && verdict="CLUSTERED"
            echo "    $func: $ratio% transitions ($transitions/$((n_insns-1))) -> $verdict  [$unique]"
        done
    fi
}

for t in "${TARGETS[@]}"; do
    src="$OUTDIR/gen_${t}.cu"
    bin="$OUTDIR/gen_${t}"
    sassfile="$SASSDIR/${t}.sass"

    if [ "$SASS_ONLY" -eq 1 ]; then
        if [ -f "$bin" ]; then
            sass_verify "$bin" "$t" "$sassfile"
        else
            log "SKIP: $bin not found (build first)"
        fi
        continue
    fi

    # ── Regenerate if stale ──
    if [ ! -f "$src" ] || [ "$GEN" -nt "$src" ] || [ "$DB" -nt "$src" ]; then
        log "Regenerating $src..."
        python3 "$GEN" "--${t}"
    fi

    # ── Build ──
    log "Building $bin..."
    $NVCC $CFLAGS "$src" -o "$bin"

    # ── SASS dump + verify ──
    sass_verify "$bin" "$t" "$sassfile"

    # ── Run ──
    log "Running $bin..."
    echo ""
    "$bin"
    echo ""
done

log "Done. SASS dumps: $SASSDIR/"
