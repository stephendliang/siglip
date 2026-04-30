#!/bin/bash
# FC2 epi-store-flags combinatorial sweep.
#
# Axes (3-bit ABC):
#   A = TMA_STORE_WIDE  (0=narrow 64-col box, 1=wide 128-col)
#   B = EPI_SINGLE_PASS (0=multi-subiter,    1=single-pass-over-TN)
#   C = USE_STMATRIX    (0=STS.128,          1=STSM.16.MT88.4)
#
# Modes:
#   res        — full residual + bias epilogue (fc2-w3-epi-ABC)
#   gemm       — no residual, no bias         (fc2-w3-epi-ABC-gemm)
#   gemm-reuse — gemm + EPI_REUSE_SMEM (NS=6 single-pass path via borrow;
#                only legal for single-pass variants B=1)
#
# Timeouts guard against variants that deadlock on B200 (EPI_SINGLE_PASS / REUSE
# paths are new and untested). Per-binary reps are averaged via @@RESULT.
#
# Usage:
#   ./tools/epi_flags_sweep.sh                   # build + run all
#   ./tools/epi_flags_sweep.sh --build-only      # build only, no run
#   ./tools/epi_flags_sweep.sh --run-only        # skip build
#   ./tools/epi_flags_sweep.sh --reps=3          # 3 runs per variant, take min
#   TIMEOUT=20 ./tools/epi_flags_sweep.sh        # override 15s default
#   ./tools/epi_flags_sweep.sh --only=gemm       # only GEMM_ONLY group
#   ./tools/epi_flags_sweep.sh --out=data/run1   # custom outdir

set -uo pipefail
cd "$(dirname "$0")/.."

TIMEOUT=${TIMEOUT:-15}
REPS=1
BUILD_ONLY=0
RUN_ONLY=0
ONLY_SPEC="all"
OUT="data/epi_sweep_$(date +%Y%m%d_%H%M%S)"

for arg in "$@"; do
    case "$arg" in
        --build-only) BUILD_ONLY=1 ;;
        --run-only)   RUN_ONLY=1 ;;
        --reps=*)     REPS="${arg#*=}" ;;
        --only=*)     ONLY_SPEC="${arg#*=}" ;;
        --out=*)      OUT="${arg#*=}" ;;
        -h|--help)    sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "unknown: $arg"; exit 1 ;;
    esac
done

mkdir -p "$OUT"
LOG="$OUT/sweep.log"
TSV="$OUT/results.tsv"

# ─── Variant definitions ──────────────────────────────────────────────
# ABC / mode / NS / subiters / UTMASTG-per-path / STSM?
# NS and sub-iters auto-derived; recorded here for the output table.
RES_VARIANTS=(
    "000 res 6 4 STS"
    "001 res 6 4 STSM"
    "100 res 5 2 STS"
    "101 res 5 2 STSM"
    "010 res 5 1 STS"
    "011 res 5 1 STSM"
    "110 res 5 1 STS"
    "111 res 5 1 STSM"
)
GEMM_VARIANTS=(
    "000 gemm 6 4 STS"
    "001 gemm 6 4 STSM"
    "100 gemm 6 2 STS"
    "101 gemm 6 2 STSM"
    "010 gemm 5 1 STS"
    "011 gemm 5 1 STSM"
    "110 gemm 5 1 STS"
    "111 gemm 5 1 STSM"
)
REUSE_VARIANTS=(
    "010 gemm-reuse 6 1 STS"
    "011 gemm-reuse 6 1 STSM"
    "110 gemm-reuse 6 1 STS"
    "111 gemm-reuse 6 1 STSM"
)

# ─── Selection ────────────────────────────────────────────────────────
want_res=0; want_gemm=0; want_reuse=0
case "$ONLY_SPEC" in
    all)            want_res=1; want_gemm=1; want_reuse=1 ;;
    res|residual)   want_res=1 ;;
    gemm)           want_gemm=1; want_reuse=1 ;;
    gemm-basic)     want_gemm=1 ;;
    reuse)          want_reuse=1 ;;
    *) echo "unknown --only: $ONLY_SPEC"; exit 1 ;;
esac

# ─── Build phase ──────────────────────────────────────────────────────
if (( ! RUN_ONLY )); then
    targets=()
    (( want_res ))   && targets+=(fc2-w3-epi-all)
    (( want_gemm ))  && targets+=(fc2-w3-epi-all-gemm)  # covers reuse variants too
    echo "[build] make -j12 ${targets[*]}" | tee -a "$LOG"
    if ! make -j12 "${targets[@]}" >>"$LOG" 2>&1; then
        echo "[build] FAILED — see $LOG"
        exit 1
    fi
fi

(( BUILD_ONLY )) && { echo "[build-only] done"; exit 0; }

# ─── Run phase ────────────────────────────────────────────────────────
run_one() {
    local bin=$1 label=$2
    local ms="-" tflops="-" valid="-" status="MISSING"
    if [[ -x "$bin" ]]; then
        local best_ms=""
        local ok_valid=1
        for r in $(seq 1 "$REPS"); do
            local out ec
            out=$(timeout "$TIMEOUT" "./$bin" 2>&1); ec=$?
            if (( ec == 124 )); then status="TIMEOUT"; break; fi
            local result
            result=$(grep -m1 "^@@RESULT" <<<"$out" || true)
            if [[ -z "$result" ]]; then
                status="FAIL($ec)"; echo "[fail] $label ec=$ec" >>"$LOG"
                echo "$out" >>"$LOG"
                break
            fi
            local rms rv rt
            rms=$(sed -n 's/.*ms=\([0-9.]*\).*/\1/p' <<<"$result")
            rv=$(sed  -n 's/.*valid=\([0-9]*\).*/\1/p'  <<<"$result")
            rt=$(sed  -n 's/.*tflops=\([0-9.]*\).*/\1/p' <<<"$result")
            (( rv == 0 )) && ok_valid=0
            if [[ -z "$best_ms" ]] || awk "BEGIN{exit !($rms < $best_ms)}"; then
                best_ms=$rms; tflops=$rt
            fi
            status="ok"
        done
        [[ -n "$best_ms" ]] && ms=$best_ms
        if [[ "$status" == "ok" ]]; then
            (( ok_valid )) || status="INVALID"
        fi
    fi
    printf "%-30s %-7s %-7s %s\n" "$label" "$ms" "$tflops" "$status"
    printf "%s\t%s\t%s\t%s\n"     "$label" "$ms" "$tflops" "$status" >>"$TSV"
}

sweep_group() {
    local -n arr=$1
    local suffix=$2
    for row in "${arr[@]}"; do
        read -r abc mode ns sub stype <<<"$row"
        local bin label
        case "$mode" in
            res)        bin="fc2-w3-epi-$abc"                 ;;
            gemm)       bin="fc2-w3-epi-$abc-gemm"            ;;
            gemm-reuse) bin="fc2-w3-epi-$abc-gemm-reuse"      ;;
        esac
        label="epi-$abc-$mode[NS$ns,si$sub,$stype]"
        run_one "$bin" "$label"
    done
}

: >"$TSV"
printf "%-30s %-7s %-7s %s\n" "variant[NS,sub,sts]" "ms" "TFLOPS" "status" | tee -a "$LOG"
printf '%s\n' "------------------------------------------------------" | tee -a "$LOG"

if (( want_res )); then
    echo "-- residual path --"           | tee -a "$LOG"
    sweep_group RES_VARIANTS "" 2>&1     | tee -a "$LOG"
fi
if (( want_gemm )); then
    echo "-- gemm-only --"               | tee -a "$LOG"
    sweep_group GEMM_VARIANTS "" 2>&1    | tee -a "$LOG"
fi
if (( want_reuse )); then
    echo "-- gemm-only + REUSE (NS=6) --" | tee -a "$LOG"
    sweep_group REUSE_VARIANTS "" 2>&1    | tee -a "$LOG"
fi

echo                                     | tee -a "$LOG"
echo "[out]  $TSV"                       | tee -a "$LOG"
echo "[log]  $LOG"                       | tee -a "$LOG"
echo "[hint] rank-2 twin = epi-101-gemm[NS6,si2,STSM]; target 1.046 ms" | tee -a "$LOG"
