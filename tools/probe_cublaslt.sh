#!/usr/bin/env bash
# probe_cublaslt.sh — one-shot reconnaissance against cuBLASLt's optimal
# per-tensor FP8 GEMM.  Designed for a short VPS session with no ncu.
#
# Runs 5 probes into data/cublaslt_probe_<ts>/:
#   1. algo introspection  (every CUBLASLT_ALGO_CONFIG_* for each heuristic,
#                           timed; ranked by measured ms)
#   2. verbose log         (CUBLASLT_LOG_LEVEL=5 → selected kernel name,
#                           tile/cluster/stages as seen by the library)
#   3. SASS dump           (cuobjdump --dump-sass on the cubin containing the
#                           selected kernel → grep for CLC / UTMA / MULTICAST /
#                           UCGABAR; counts register banks, smem ops, mbar ops)
#   4. K-sweep             (K ∈ {1024, 2048, 3072, 4096, 6144, 8192} —
#                           cuBLASLt vs fc2-w3-zigzag/dgswizzle/lean/sched;
#                           looks for split-K signature)
#   5. env-sweep           (CUBLASLT_* env vars to disable reduction
#                           schemes / split-K / force workspace — sees which
#                           optimization they depend on)
#
# Usage:
#   ./tools/probe_cublaslt.sh
#   OUT_DIR=data/my_probe ./tools/probe_cublaslt.sh
#   SKIP=4,5 ./tools/probe_cublaslt.sh          # skip K-sweep + env-sweep
#
# Total wall time on B200: ~8 min with all probes.

set -uo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${OUT_DIR:-data/cublaslt_probe_${STAMP}}"
SKIP="${SKIP:-}"
mkdir -p "$OUT_DIR"

skip_probe() { [[ ",${SKIP}," == *",${1},"* ]]; }

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT_DIR/session.log"; }

log "========================================"
log "  cuBLASLt probe  ${STAMP}"
log "  out=$OUT_DIR  skip=$SKIP"
log "========================================"

nvidia-smi --query-gpu=gpu_name,clocks.sm --format=csv,noheader | tee -a "$OUT_DIR/session.log"

# ── Shapes ─────────────────────────────────────────────────────────────
M_FC=928256
FC1_N=3072; FC1_K=768;  FC1_EPI=2     # GELU_BIAS
FC2_N=768;  FC2_K=3072; FC2_EPI=3     # BIAS_ONLY

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║ Probe 1: algo introspection                                           ║
# ╚═══════════════════════════════════════════════════════════════════════╝
if ! skip_probe 1; then
    log ""
    log "── Probe 1: algo introspection ──"
    log "  building cublaslt-introspect ..."
    if ! make -B cublaslt-introspect > "$OUT_DIR/introspect_build.log" 2>&1; then
        log "  BUILD FAILED — see introspect_build.log"
        tail -20 "$OUT_DIR/introspect_build.log" | tee -a "$OUT_DIR/session.log"
    else
        for shape in fc1 fc2; do
            if [ "$shape" = "fc1" ]; then N=$FC1_N; K=$FC1_K; E=$FC1_EPI; else N=$FC2_N; K=$FC2_K; E=$FC2_EPI; fi
            log "  ${shape}: M=$M_FC N=$N K=$K epi=$E ..."
            ./cublaslt-introspect "$M_FC" "$N" "$K" "$E" \
                > "$OUT_DIR/introspect_${shape}.txt" 2>&1
            # Bubble the winner + TSV row up to the session log
            grep -E '^# Winner|^### |^rank\t|^1\t' "$OUT_DIR/introspect_${shape}.txt" \
                | sed "s/^/    [${shape}] /" | tee -a "$OUT_DIR/session.log"
        done
    fi
fi

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║ Probe 2: verbose log — capture selected kernel name                    ║
# ╚═══════════════════════════════════════════════════════════════════════╝
if ! skip_probe 2; then
    log ""
    log "── Probe 2: verbose log ──"
    for shape in fc1 fc2; do
        binary="./cublas-bench-${shape}"
        if [ ! -x "$binary" ]; then
            log "  building ${binary} ..."
            make -B "cublas-bench-${shape}" > "$OUT_DIR/${shape}_build.log" 2>&1 || {
                log "  BUILD FAIL ${shape}"; continue; }
        fi
        log "  ${shape}: CUBLASLT_LOG_LEVEL=5 ${binary} ..."
        CUBLASLT_LOG_LEVEL=5 \
        CUBLASLT_LOG_FILE="$OUT_DIR/${shape}_cublaslt.log" \
        CUBLAS_LOG_LEVEL=5 \
        CUBLAS_LOGDEST_DBG="$OUT_DIR/${shape}_cublas.log" \
        CUBLAS_LOGINFO_DBG=1 \
            timeout 60 "$binary" > "$OUT_DIR/${shape}_stdout.txt" 2>&1 || true
        # Extract every kernel name hit during the verbose log
        grep -hoE '[A-Za-z_][A-Za-z0-9_]*sm100[A-Za-z0-9_]*' \
            "$OUT_DIR/${shape}_cublaslt.log" "$OUT_DIR/${shape}_cublas.log" 2>/dev/null \
            | sort -u > "$OUT_DIR/${shape}_kernels.txt"
        nk=$(wc -l < "$OUT_DIR/${shape}_kernels.txt")
        log "  [${shape}] captured $nk distinct sm100* symbol strings"
        head -6 "$OUT_DIR/${shape}_kernels.txt" | sed "s/^/    /" | tee -a "$OUT_DIR/session.log"
    done
fi

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║ Probe 3: SASS dump of selected kernel                                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝
if ! skip_probe 3; then
    log ""
    log "── Probe 3: SASS dump ──"
    LIB=$(ldconfig -p 2>/dev/null | awk '/libcublasLt\.so\./ && !/stubs/ {print $NF; exit}')
    if [ -z "$LIB" ] || [ ! -f "$LIB" ]; then
        LIB=$(find /usr/local/cuda*/lib64 /usr/lib/x86_64-linux-gnu -maxdepth 3 \
              -name 'libcublasLt.so*' 2>/dev/null | head -1)
    fi
    log "  libcublasLt: ${LIB:-not found}"

    if [ -n "$LIB" ] && command -v cuobjdump >/dev/null 2>&1; then
        log "  enumerating sm_100 kernels (this is slow) ..."
        cuobjdump --dump-elf-symbols "$LIB" 2>/dev/null \
            | grep -E '^[0-9a-f]+\s+[^\s]*\s+(FUNC|OBJECT)' \
            | awk '{print $NF}' | sort -u > "$OUT_DIR/libcublaslt_symbols.txt"
        nsym=$(wc -l < "$OUT_DIR/libcublaslt_symbols.txt")
        log "    $nsym symbols total"

        # For each kernel name seen in the verbose log, try to find a matching
        # symbol in libcublasLt and dump its SASS.
        for shape in fc1 fc2; do
            [ -f "$OUT_DIR/${shape}_kernels.txt" ] || continue
            # Pick the longest sm100* string — that's typically the kernel name.
            target=$(awk '{print length, $0}' "$OUT_DIR/${shape}_kernels.txt" \
                     | sort -rn | head -1 | cut -d' ' -f2-)
            [ -z "$target" ] && { log "  [${shape}] no sm100 kernel name to probe"; continue; }
            log "  [${shape}] target symbol: $target"

            # cuobjdump matches by function name (demangled substring)
            log "  [${shape}] cuobjdump --dump-sass --function=$target ..."
            timeout 180 cuobjdump --dump-sass --function "$target" "$LIB" \
                > "$OUT_DIR/${shape}_sass.txt" 2>"$OUT_DIR/${shape}_sass.err" || {
                    log "  [${shape}] SASS dump failed/timed out"; continue; }

            sass="$OUT_DIR/${shape}_sass.txt"
            nins=$(grep -cE '^\s*/\*' "$sass" 2>/dev/null || echo 0)

            cat > "$OUT_DIR/${shape}_sass_summary.txt" <<SASS_EOF
### SASS summary for ${shape} (target=$target)
### libcublasLt.so  size=$(stat -c%s "$LIB" 2>/dev/null)
### instruction lines: $nins

## CLC / cluster dispatch
    CLC                 $(grep -cE '\bCLC\b' "$sass")  — clusterlaunchcontrol (if >0: uses HW dispatch)
    UCLC                $(grep -cE '\bUCLC\b' "$sass")
    CLUSTERLAUNCH       $(grep -cEi 'clusterlaunch' "$sass")

## TMA
    UTMALDG             $(grep -cE '\bUTMALDG\b' "$sass")  — TMA load (global)
    UTMASTG             $(grep -cE '\bUTMASTG\b' "$sass")  — TMA store
    UTMALDG.MULTICAST   $(grep -cE 'UTMALDG.*MULTICAST' "$sass")
    UTMAPREFLD          $(grep -cE 'UTMAPREFLD' "$sass")  — prefetch

## MMA family
    tcgen05 / QMMA      $(grep -cEi 'tcgen05|QMMA' "$sass")
    UMMA                $(grep -cE '\bUMMA\b' "$sass")
    WGMMA               $(grep -cE '\bWGMMA\b' "$sass")

## Cluster + async
    UCGABAR             $(grep -cE '\bUCGABAR\b' "$sass")  — cluster barrier
    MBARRIER            $(grep -cEi 'mbarrier|MBAR' "$sass")
    FENCE.PROXY         $(grep -cEi 'fence.proxy|PROXY_FENCE' "$sass")

## Stores / epilogue
    STG                 $(grep -cE '\bSTG\b' "$sass")
    STS                 $(grep -cE '\bSTS\b' "$sass")
    LDS                 $(grep -cE '\bLDS\b' "$sass")
    STMATRIX            $(grep -cEi 'STMATRIX|stmatrix' "$sass")

## Branches (warp-specialization count proxy)
    distinct BRA tgts   $(grep -oE 'BRA 0x[0-9a-f]+' "$sass" | sort -u | wc -l)
    S2R.TID.Y           $(grep -cE 'S2R.*TID\.Y' "$sass")
    S2R.CTAID           $(grep -cE 'S2R.*CTAID' "$sass")

## Resource usage (from cuobjdump --function output if present)
$(grep -E 'REG:|SMEM|STACK|shared' "$sass" | head -8 | sed 's/^/    /')
SASS_EOF
            log "  [${shape}] SASS summary written"
            cat "$OUT_DIR/${shape}_sass_summary.txt" | sed "s/^/    /" \
                | tee -a "$OUT_DIR/session.log"
        done
    else
        log "  cuobjdump or libcublasLt missing — skip SASS"
    fi
fi

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║ Probe 4: K-sweep — looks for split-K signature                         ║
# ╚═══════════════════════════════════════════════════════════════════════╝
if ! skip_probe 4; then
    log ""
    log "── Probe 4: K-sweep ──"
    csv="$OUT_DIR/k_sweep.csv"
    echo "layer,variant,K,ms" > "$csv"

    for K in 1024 2048 3072 4096 6144 8192; do
        # FC2 has K≠constant; need to rebuild cublas-bench-fc2 with -DBENCH_K=$K
        log "  K=$K: rebuilding cublas-bench-fc2 ..."
        make -B cublas-bench-fc2 \
             NVCC=nvcc "CFLAGS=-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -lineinfo --cudart=static -DBENCH_K=$K" \
             > "$OUT_DIR/k${K}_cublas_build.log" 2>&1 || {
                 log "  [K=$K] cublas build FAIL"; continue; }
        cp cublas-bench-fc2 "$OUT_DIR/bin_cublas_fc2_K${K}"

        log "  K=$K: timing cublas-bench-fc2 ..."
        ms=$(timeout 60 "$OUT_DIR/bin_cublas_fc2_K${K}" 2>&1 \
             | grep -oP 'Per-tensor FP8.*?\K[0-9.]+(?= ms)' | head -1)
        [ -z "$ms" ] && ms=$(timeout 60 "$OUT_DIR/bin_cublas_fc2_K${K}" 2>&1 \
             | grep -oP 'best.*?ms=\K[0-9.]+' | head -1)
        echo "fc2,cublaslt,${K},${ms:-ERR}" >> "$csv"
        log "    cublaslt fc2 K=$K  ms=${ms:-ERR}"

        # Our kernel at the same K (zigzag default, since it's FC2-champion-ish)
        for variant in zigzag dgswizzle lean; do
            target="fc2-w3-$variant"
            log "  K=$K: rebuilding $target ..."
            make -B "$target" DFLAGS="-DPACKED_TILES -DK_DIM=$K" \
                 > "$OUT_DIR/k${K}_${variant}_build.log" 2>&1 || {
                     log "  [$target K=$K] build FAIL"; continue; }
            cp "$target" "$OUT_DIR/bin_${variant}_K${K}"
            ms=$(timeout 60 "$OUT_DIR/bin_${variant}_K${K}" 2>&1 \
                | grep -oP '@@RESULT.*ms=\K[0-9.]+' | head -1)
            echo "fc2,${variant},${K},${ms:-ERR}" >> "$csv"
            log "    $variant fc2 K=$K  ms=${ms:-ERR}"
        done
    done
    log "  k-sweep CSV: $csv"
    log ""
    log "  K     cublaslt   zigzag     dgsw       lean"
    log "  ----  ---------  ---------  ---------  ---------"
    python3 - "$csv" <<'PY' | tee -a "$OUT_DIR/session.log"
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
bag = collections.defaultdict(dict)
for r in rows:
    try: bag[int(r["K"])][r["variant"]] = float(r["ms"])
    except: pass
for K in sorted(bag):
    m = bag[K]
    def f(k): return f"{m[k]:>9.3f}" if k in m else "      ERR"
    print(f"  {K:>4}  {f('cublaslt')}  {f('zigzag')}  {f('dgswizzle')}  {f('lean')}")
PY
fi

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║ Probe 5: env-var sweep                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════╝
if ! skip_probe 5; then
    log ""
    log "── Probe 5: env-sweep ──"
    csv="$OUT_DIR/env_sweep.csv"
    echo "config,fc,ms" > "$csv"
    for shape in fc1 fc2; do
        binary="./cublas-bench-${shape}"
        [ -x "$binary" ] || { log "  skip: no $binary"; continue; }

        # Baseline
        ms=$(timeout 60 "$binary" 2>&1 | grep -oP 'Per-tensor FP8.*?\K[0-9.]+(?= ms)' | head -1)
        echo "baseline,${shape},${ms:-ERR}" >> "$csv"
        log "  [${shape}] baseline                       ms=${ms:-ERR}"

        # Force disable the algorithm-id the heuristic first picked
        # (checks whether there's a materially slower-but-close second choice).

        # CUBLASLT_MATMUL_DISABLE_* env vars known on CUDA 12.x / 13:
        declare -a env_cfgs=(
            "CUBLASLT_DISABLE_REQUIRED_WORKSPACE=1"
            "CUBLASLT_MATMUL_WORKSPACE_SIZE=0"
            "CUBLAS_WORKSPACE_CONFIG=:0:0"
            "CUDA_LAUNCH_BLOCKING=1"
        )
        for ec in "${env_cfgs[@]}"; do
            k="${ec%%=*}"
            ms=$(env "$ec" timeout 60 "$binary" 2>&1 | grep -oP 'Per-tensor FP8.*?\K[0-9.]+(?= ms)' | head -1)
            echo "${k},${shape},${ms:-ERR}" >> "$csv"
            log "  [${shape}] $ec   ms=${ms:-ERR}"
        done
    done
    log "  env-sweep CSV: $csv"
fi

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║ Final summary                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════╝
log ""
log "── Summary ──"
log "  Output:  $OUT_DIR"
log "  Key files to download back:"
log "    introspect_fc{1,2}.txt            # top heuristics, tile/cluster/splitk"
log "    {fc1,fc2}_cublaslt.log            # verbose log (grep for 'Solution' / 'Chose')"
log "    {fc1,fc2}_sass_summary.txt        # CLC/UTMA/MULTICAST instruction counts"
log "    {fc1,fc2}_kernels.txt             # candidate kernel symbols"
log "    k_sweep.csv + env_sweep.csv       # shape/env time curves"
log ""
log "Done."
