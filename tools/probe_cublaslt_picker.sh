#!/usr/bin/env bash
# probe_cublaslt_picker.sh — comprehensive cuBLASLt rank-1/2/3 sweep.
#
# For each (M,N,K,EPI) cell, runs cublaslt-introspect and extracts the
# top-3 algos (timed by introspect, sorted by ms ascending). Per-cell raw
# output goes to <OUT>/m<M>_n<N>_k<K>_epi<E>.txt; summary.tsv has one row
# per (cell, rank ∈ {1,2,3}); summary.txt is the grouped human table.
#
# Default sweep groups:
#   * production:        FC2 K=3072 N=768 (we win), FC1 K=768 N=3072
#   * fc2_ksweep:        K ∈ {1024,2048,4096,6144,8192} @ N=768
#   * dim_sweep_bias:    pow2 N×K grid × EPI=3 (BIAS_ONLY)
#   * dim_sweep_none:    pow2 N×K grid × EPI=0 (plain GEMM)
#   * loss_focus:        N=2048 K=1024 (+27% catastrophe), both EPI modes
#
# Override:
#   CELLS="M:N:K:E,M:N:K:E,..." ./tools/probe_cublaslt_picker.sh
#   PROBE_GROUPS="production,loss_focus" ./tools/probe_cublaslt_picker.sh
#   M_FC=464128 ./tools/probe_cublaslt_picker.sh
#   OUT_DIR=data/my_picker ./tools/probe_cublaslt_picker.sh
#   TIMEOUT=120 ./tools/probe_cublaslt_picker.sh
#   TOP_N=5 ./tools/probe_cublaslt_picker.sh        (capture top-N instead of 3)
#
# Each cell writes:
#   <OUT>/m<M>_n<N>_k<K>_epi<E>.txt   full introspect dump (TSV + caps + winner)
#   <OUT>/summary.tsv                  one row per (cell, rank)
#   <OUT>/summary.txt                  grouped human table
#   <OUT>/session.log

set -uo pipefail
cd "$(dirname "$0")/.."

trap 'echo; echo "[INT] caught Ctrl+C"; exit 130' INT TERM

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${OUT_DIR:-data/cublaslt_picker_${STAMP}}"
TIMEOUT="${TIMEOUT:-90}"
M_FC="${M_FC:-928256}"
TOP_N="${TOP_N:-3}"
PROBE_GROUPS="${PROBE_GROUPS:-production,loss_focus,fc2_ksweep,dim_sweep_bias,dim_sweep_none}"
mkdir -p "$OUT_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT_DIR/session.log"; }

# --- Cell groups -----------------------------------------------------------
# Format: M:N:K:EPI  (EPI 0=none, 2=gelu_bias, 3=bias_only)

PRODUCTION=(
    "${M_FC}:768:3072:3"      # FC2 production (we beat rank-1)
    "${M_FC}:3072:768:2"      # FC1 production (gelu_bias)
)

LOSS_FOCUS=(
    "${M_FC}:2048:1024:3"     # +27% catastrophe (BIAS)
    "${M_FC}:2048:1024:0"     # +27% catastrophe (plain GEMM)
)

FC2_KSWEEP=(
    "${M_FC}:768:1024:3"
    "${M_FC}:768:2048:3"
    "${M_FC}:768:4096:3"
    "${M_FC}:768:6144:3"
    "${M_FC}:768:8192:3"
)

# Pow2 N×K grid mirrors tools/dim_sweep_w3x.py default.
DIM_SWEEP_NS=(256 512 1024 2048)
DIM_SWEEP_KS=(1024 2048 4096 8192)

build_dim_sweep() {
    local epi="$1"
    local out=()
    local n k
    for n in "${DIM_SWEEP_NS[@]}"; do
        for k in "${DIM_SWEEP_KS[@]}"; do
            out+=("${M_FC}:${n}:${k}:${epi}")
        done
    done
    printf '%s\n' "${out[@]}"
}

CELLS_ARR=()
if [ -n "${CELLS:-}" ]; then
    IFS=',' read -r -a CELLS_ARR <<< "$CELLS"
else
    IFS=',' read -r -a GROUP_LIST <<< "$PROBE_GROUPS"
    declare -A SEEN
    for grp in "${GROUP_LIST[@]}"; do
        case "$grp" in
            production)      G=("${PRODUCTION[@]}");;
            loss_focus)      G=("${LOSS_FOCUS[@]}");;
            fc2_ksweep)      G=("${FC2_KSWEEP[@]}");;
            dim_sweep_bias)  mapfile -t G < <(build_dim_sweep 3);;
            dim_sweep_none)  mapfile -t G < <(build_dim_sweep 0);;
            *)               echo "unknown group: $grp" >&2; exit 2;;
        esac
        for c in "${G[@]}"; do
            if [ -z "${SEEN[$c]:-}" ]; then
                CELLS_ARR+=("$c")
                SEEN[$c]=1
            fi
        done
    done
fi

log "============================================================"
log "  cuBLASLt picker probe  ${STAMP}"
log "  out=$OUT_DIR  timeout=${TIMEOUT}s  top_n=${TOP_N}"
log "  groups=${PROBE_GROUPS}  cells=${#CELLS_ARR[@]}"
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

epi_name() {
    case "$1" in 0) echo "none";; 2) echo "gelu_bias";; 3) echo "bias_only";; *) echo "?";; esac
}

# Tile ID → geometry. Mirrors TILE_ID_NAME in tools/dim_sweep_w3x.py
# (kept in sync — update both). Source: cublasLt.h enum + rank1.sass.
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

TSV="$OUT_DIR/summary.tsv"
echo -e "M\tN\tK\tEPI\trank\tstatus\tms\tcyc\talgoId\ttile\tstages\tcluster\tsplitk\tswizzle\twaves\tws_MB\ttile_name" > "$TSV"

# Extract top-N data rows from the "# TSV" block in introspect output.
# Block layout (bench/cublaslt_introspect.cu:217-226):
#   # TSV
#   rank<TAB>heur<TAB>ms<TAB>cyc<TAB>algoId<TAB>tile<TAB>stages<TAB>cluster<TAB>splitk<TAB>redux<TAB>swizzle<TAB>custom<TAB>waves<TAB>ws_MB
#   1<TAB>...
#   2<TAB>...
#   ...
#   <blank>
extract_top() {
    local file="$1" n="$2"
    awk -v n="$n" '
        /^# TSV$/      { in_tsv = 1; next }
        in_tsv && /^rank\t/ { next }
        in_tsv && /^[0-9]+\t/ { print; if (++c >= n) exit }
        in_tsv && /^$/  { exit }
    ' "$file"
}

idx=0
for cell in "${CELLS_ARR[@]}"; do
    idx=$((idx+1))
    IFS=':' read -r M N K E <<< "$cell"
    EN=$(epi_name "$E")
    OUT_FILE="$OUT_DIR/m${M}_n${N}_k${K}_epi${E}.txt"
    log ""
    log "[$idx/${#CELLS_ARR[@]}] M=$M N=$N K=$K epi=$E ($EN) → $(basename "$OUT_FILE")"
    if ! timeout "$TIMEOUT" ./cublaslt-introspect "$M" "$N" "$K" "$E" \
            > "$OUT_FILE" 2>&1; then
        rc=$?
        log "   FAIL ec=$rc (see $(basename "$OUT_FILE"))"
        echo -e "$M\t$N\t$K\t$E\t-\tFAIL($rc)\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-" >> "$TSV"
        continue
    fi
    # Pull top-N rows from the TSV block.
    rows=$(extract_top "$OUT_FILE" "$TOP_N")
    if [ -z "$rows" ]; then
        log "   no TSV rows (heuristic returned 0 valid algos?)"
        echo -e "$M\t$N\t$K\t$E\t-\tNO_RESULT\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-" >> "$TSV"
        continue
    fi
    # Echo top-3 to log; emit one TSV row per rank.
    while IFS=$'\t' read -r rank heur ms cyc algoId tile stages cluster splitk redux swizzle custom waves ws_MB; do
        tn=$(tile_name "$tile")
        if [ "$rank" = "1" ]; then
            log "   r1 ms=$ms tile=$tile($tn) st=$stages cl=$cluster sk=$splitk sw=$swizzle cyc=$cyc"
        else
            log "   r$rank ms=$ms tile=$tile($tn) st=$stages cl=$cluster sk=$splitk"
        fi
        echo -e "$M\t$N\t$K\t$E\t$rank\tOK\t$ms\t$cyc\t$algoId\t$tile\t$stages\t$cluster\t$splitk\t$swizzle\t$waves\t$ws_MB\t$tn" >> "$TSV"
    done <<< "$rows"
done

log ""
log "── summary ──"
{
    echo
    echo "Top-${TOP_N} per cell (cuBLASLt heuristic ranked by measured ms; cluster/stages/splitk/swizzle = raw IDs):"
    column -t -s $'\t' "$TSV"
    echo
    echo "Tile IDs: 23=128x256  24=256x128  32=128x192  197=168x128  201=176x128  495=256x96  535=320x192"
    echo
} | tee "$OUT_DIR/summary.txt" | tee -a "$OUT_DIR/session.log" >/dev/null

log ""
log "TSV   : $TSV"
log "table : $OUT_DIR/summary.txt"
log "raw   : $OUT_DIR/m*_n*_k*_epi*.txt  (full algo TSV + top-3 caps per cell)"
