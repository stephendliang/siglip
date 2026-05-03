#!/usr/bin/env bash
# probe_cublaslt_picker.sh — capture cuBLASLt's rank-1 algo (tile/stages/
# cluster/splitk/swizzle) for a list of (M,N,K,EPI) cells and write per-cell
# + summary output to data/cublaslt_picker_<ts>/.
#
# Default cells target the open questions from CLAUDE.md "Three loss patterns":
#   * N=2048 K=1024 (+27% catastrophe, eff=0.54)  → BIAS+noBIAS
#   * N=512/1024/2048 × K=4096 (+1.7..3.5% systematic) → BIAS
#   * N=768 K=3072 production sanity check (we beat rank-1 by 38..45 µs) → BIAS
# Plus a small-N (256) K=4096 reference to see if cuBLASLt picks 256x96
# (TILE_ID=495).
#
# Override:
#   CELLS="M:N:K:E,M:N:K:E,..." ./tools/probe_cublaslt_picker.sh
#   M_FC=464128 ./tools/probe_cublaslt_picker.sh
#   OUT_DIR=data/my_picker ./tools/probe_cublaslt_picker.sh
#   TIMEOUT=120 ./tools/probe_cublaslt_picker.sh
#
# Each cell writes:
#   <OUT>/m<M>_n<N>_k<K>_epi<E>.txt   full introspect dump (TSV + caps + winner)
#   <OUT>/summary.tsv                  rank-1 row per cell
#   <OUT>/summary.txt                  human table
#   <OUT>/session.log

set -uo pipefail
cd "$(dirname "$0")/.."

trap 'echo; echo "[INT] caught Ctrl+C"; exit 130' INT TERM

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${OUT_DIR:-data/cublaslt_picker_${STAMP}}"
TIMEOUT="${TIMEOUT:-60}"
M_FC="${M_FC:-928256}"
mkdir -p "$OUT_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT_DIR/session.log"; }

# Default cells: the three loss patterns + production sanity + small-N K=4096 reference.
# Format: M:N:K:EPI  (EPI 0=none, 2=gelu_bias, 3=bias_only)
DEFAULT_CELLS=(
    "${M_FC}:2048:1024:3"   # +27% catastrophe (BIAS)
    "${M_FC}:2048:1024:0"   # +27% catastrophe (plain GEMM)
    "${M_FC}:256:4096:3"    # K=4096 small-N reference
    "${M_FC}:512:4096:3"    # K=4096 systematic loss
    "${M_FC}:1024:4096:3"   # K=4096 systematic loss
    "${M_FC}:2048:4096:3"   # K=4096 systematic loss
    "${M_FC}:768:3072:3"    # production sanity check (we win)
)

if [ -n "${CELLS:-}" ]; then
    IFS=',' read -r -a CELLS_ARR <<< "$CELLS"
else
    CELLS_ARR=("${DEFAULT_CELLS[@]}")
fi

log "============================================================"
log "  cuBLASLt picker probe  ${STAMP}"
log "  out=$OUT_DIR  timeout=${TIMEOUT}s  cells=${#CELLS_ARR[@]}"
log "============================================================"
nvidia-smi --query-gpu=gpu_name,clocks.sm --format=csv,noheader 2>&1 \
    | tee -a "$OUT_DIR/session.log" || true

if [ ! -x ./cublaslt-introspect ]; then
    log "building cublaslt-introspect ..."
    if ! make -B cublaslt-introspect > "$OUT_DIR/build.log" 2>&1; then
        log "BUILD FAILED — see $OUT_DIR/build.log"
        tail -20 "$OUT_DIR/build.log" | tee -a "$OUT_DIR/session.log"
        exit 1
    fi
fi

TSV="$OUT_DIR/summary.tsv"
echo -e "M\tN\tK\tEPI\tstatus\tms\tcyc\ttile\tstages\tcluster\tsplitk\tswizzle\ttile_name" > "$TSV"

epi_name() {
    case "$1" in 0) echo "none";; 2) echo "gelu_bias";; 3) echo "bias_only";; *) echo "?";; esac
}

# Tile ID → geometry. Source: rank1.sass + cublasLt.h enum, mirror of TILE_ID_NAME
# in tools/dim_sweep_w3x.py (kept in sync — update both).
tile_name() {
    case "$1" in
        23)  echo "128x256";;
        24)  echo "256x128";;
        32)  echo "128x192";;
        197) echo "168x128";;
        201) echo "176x128";;
        495) echo "256x96";;
        535) echo "320x192";;
        *)   echo "id=$1";;
    esac
}

for cell in "${CELLS_ARR[@]}"; do
    IFS=':' read -r M N K E <<< "$cell"
    EN=$(epi_name "$E")
    OUT_FILE="$OUT_DIR/m${M}_n${N}_k${K}_epi${E}.txt"
    log ""
    log "── M=$M N=$N K=$K epi=$E ($EN) ──"
    log "   → $(basename "$OUT_FILE")"
    if ! timeout "$TIMEOUT" ./cublaslt-introspect "$M" "$N" "$K" "$E" \
            > "$OUT_FILE" 2>&1; then
        rc=$?
        log "   FAIL ec=$rc (see $(basename "$OUT_FILE"))"
        echo -e "$M\t$N\t$K\t$E\tFAIL($rc)\t-\t-\t-\t-\t-\t-\t-\t-" >> "$TSV"
        continue
    fi
    # Parse the "# Winner:" line — single line, well-defined format from
    # bench/cublaslt_introspect.cu:241.
    winner=$(grep -E '^# Winner:' "$OUT_FILE" | head -1)
    if [ -z "$winner" ]; then
        log "   no Winner line (heuristic returned 0 valid algos?)"
        echo -e "$M\t$N\t$K\t$E\tNO_WINNER\t-\t-\t-\t-\t-\t-\t-\t-" >> "$TSV"
        continue
    fi
    tile=$(echo    "$winner" | grep -oE 'tile=[0-9]+'    | head -1 | cut -d= -f2)
    stages=$(echo  "$winner" | grep -oE 'stages=[0-9]+'  | head -1 | cut -d= -f2)
    cluster=$(echo "$winner" | grep -oE 'cluster=[0-9]+' | head -1 | cut -d= -f2)
    splitk=$(echo  "$winner" | grep -oE 'splitk=[0-9]+'  | head -1 | cut -d= -f2)
    swizzle=$(echo "$winner" | grep -oE 'swizzle=[0-9]+' | head -1 | cut -d= -f2)
    ms=$(echo      "$winner" | grep -oE 'ms=[0-9.]+'     | head -1 | cut -d= -f2)
    cyc=$(echo     "$winner" | grep -oE 'cyc=[0-9]+'     | head -1 | cut -d= -f2)
    tn=$(tile_name "$tile")
    log "   rank-1: tile=$tile ($tn) stages=$stages cluster=$cluster splitk=$splitk swizzle=$swizzle ms=$ms cyc=$cyc"
    echo -e "$M\t$N\t$K\t$E\tOK\t$ms\t$cyc\t$tile\t$stages\t$cluster\t$splitk\t$swizzle\t$tn" >> "$TSV"
done

log ""
log "── summary ──"
{
    echo
    echo "rank-1 per cell (cuBLASLt heuristic top hit, full enumeration timed):"
    column -t -s $'\t' "$TSV"
    echo
} | tee "$OUT_DIR/summary.txt" | tee -a "$OUT_DIR/session.log"

log ""
log "TSV  : $TSV"
log "table: $OUT_DIR/summary.txt"
log "raw  : $OUT_DIR/m*_n*_k*_epi*.txt  (full algo TSV + top-3 caps per cell)"
