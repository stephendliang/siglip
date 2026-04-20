#!/usr/bin/env bash
# dump_cublaslt_sass.sh — capture SASS of cuBLASLt's nvjet_sm100_qqtst kernels.
#
# WHY: cuBLASLt's Blackwell (SM100) kernels do NOT appear as ELF symbols in
# any .so on disk — they are JIT-compiled from PTX or stored in obfuscated
# fatbins that cuobjdump --extract-elf cannot decode. ncu sees what the
# driver actually executed and can dump SASS for it directly, sidestepping
# the entire .so-search problem.
#
# Default shape is FC2 (M=928256 N=768 K=3072, bias_only). Override via env:
#   M=464128 N=1024 K=4096 EPI=2 ./tools/dump_cublaslt_sass.sh
#
# Two known top kernels (FC2 K=3072 rank-1/rank-2 from nsys 2026-04-20) get
# their own SASS files automatically. Edit TARGETS below to dump others.

set -uo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="${OUT_DIR:-data/cublaslt_sass_${STAMP}}"
mkdir -p "$OUT"

M=${M:-928256}
N=${N:-768}
K=${K:-3072}
EPI=${EPI:-3}
LAUNCH_COUNT=${LAUNCH_COUNT:-32}

TARGETS=(
    "128x256_128x6_2x1_2cta_v"   # FC2 K=3072 rank-2 — our exact geometry
    "176x128_128x8_1x2_2cta_h"   # FC2 K=3072 rank-1 — NS=8 winner
)

log() { echo "[$(date +%H:%M:%S)] $*"; }

if [[ ! -x ./cublaslt-introspect ]]; then
    log "building cublaslt-introspect ..."
    make cublaslt-introspect >/dev/null
fi

if ! command -v ncu >/dev/null; then
    echo "ERROR: ncu not in PATH. Install nsight-compute or load the cuda module."
    exit 1
fi

NCU_REP="$OUT/run.ncu-rep"
log "ncu capture (FC2 M=$M N=$N K=$K epi=$EPI, launch_count=$LAUNCH_COUNT) → $NCU_REP"
ncu --kernel-name 'regex:nvjet_sm100_qqtst' \
    --launch-skip 0 --launch-count "$LAUNCH_COUNT" \
    --cache-control none \
    --set full \
    --export "${NCU_REP%.ncu-rep}" -f \
    ./cublaslt-introspect "$M" "$N" "$K" "$EPI" \
    > "$OUT/introspect.out" 2>&1 || true

if [[ ! -f "$NCU_REP" ]]; then
    echo "ERROR: $NCU_REP not produced"
    tail -40 "$OUT/introspect.out"
    exit 1
fi

log "kernel summary"
ncu --import "$NCU_REP" --csv --page raw \
    --metrics gpu__time_duration.sum 2>/dev/null \
    | tee "$OUT/kernels.csv" \
    | awk -F, 'NR>1 && NF>1 {print "    " $0}' | head -16

for tag in "${TARGETS[@]}"; do
    safe=${tag//_/-}
    file="$OUT/sass_${safe}.txt"
    log "dumping SASS for nvjet_sm100_qqtst_${tag}_..."
    ncu --import "$NCU_REP" \
        --kernel-name "regex:nvjet_sm100_qqtst_${tag}" \
        --print-sass > "$file" 2>&1 || true
    lines=$(wc -l < "$file")
    log "  $file ($lines lines)"
done

cat <<EOF

==> Output: $OUT/
    To dump SASS for any captured kernel:
      ncu --import $NCU_REP --kernel-name 'regex:<pattern>' --print-sass
    To list every captured kernel:
      ncu --import $NCU_REP --csv --page raw --metrics gpu__time_duration.sum
EOF
