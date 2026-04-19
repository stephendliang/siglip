#!/bin/bash
# Unified FC1/FC2 bench + ncu harness.
#
# Combinatorial axes:
#   layer     : fc1 | fc2
#   dispatch  : default, sched, lean, tail*, tail-lean*, rowsteal*,
#               dgswizzle, zorder, hilbert, zigzag, rowmajor,
#               ncycle, ncyrot, nflat, nsnake    (* = fc2-only)
#   packed    : yes (DPACKED_TILES) | no
#   mode      : fused | gemm (DGEMM_ONLY) | strip (DSTRIP_EPILOGUE)
#   epi-warps : 4 (default) | 2 | 1  — rebuilds with -DNUM_EPI_WARPS=N (e2 is dead: FC1 regresses 1.6x, FC2 often deadlocks/mismatches; flag kept for explicit one-offs)
#
# Each enabled (layer, dispatch, packed, mode, epi_warps) tuple is one config.
# Configs are built to unique binaries via COPY so they don't clobber each
# other when re-running.  --ncu collects metrics on top of wall-time.
#
# Usage:
#   ./tools/bench.sh                                # default smoke (both layers, core dispatches, both packings, fused, 4 epi-warps)
#   ./tools/bench.sh --comprehensive                # packed-only, all dispatches, f/g/s, baselines, epi-warps={4,2}
#   ./tools/bench.sh --comprehensive --ncu          # same + ncu metrics on every config
#   ./tools/bench.sh --dispatch=all --decomp=all    # full combinatorial (packed+unpacked both)
#   ./tools/bench.sh --fc2 --ncu                    # FC2 wall-time + ncu
#   ./tools/bench.sh --baselines --ncu              # include cutlass/cublas
#   ./tools/bench.sh --dispatch=lean,dgswizzle --packed=yes --ncu --full
#   ./tools/bench.sh --epi-warps=2,4                # add 2-warp epilogue sweep
#   ./tools/bench.sh --dry-run                      # print configs, build nothing
#
# Output: data/bench_YYYYMMDD_HHMMSS/

set -uo pipefail
cd "$(dirname "$0")/.."

# ─── Arg parsing ───────────────────────────────────────────────────────

LAYERS="fc1 fc2"
DISPATCH_SPEC="core"
PACKED_SPEC="both"
DECOMP_SPEC="fused"
EPI_WARPS_SPEC="4"
BASELINES=0
DO_NCU=0
DO_FULL=0
DRY_RUN=0
REPS=1
OUT_OVERRIDE=""

for arg in "$@"; do
    case "$arg" in
        --fc1)              LAYERS="fc1" ;;
        --fc2)              LAYERS="fc2" ;;
        --both)             LAYERS="fc1 fc2" ;;
        --dispatch=*)       DISPATCH_SPEC="${arg#*=}" ;;
        --packed=*)         PACKED_SPEC="${arg#*=}" ;;
        --decomp=*)         DECOMP_SPEC="${arg#*=}" ;;
        --epi-warps=*)      EPI_WARPS_SPEC="${arg#*=}" ;;
        --baselines)        BASELINES=1 ;;
        --ncu)              DO_NCU=1 ;;
        --full)             DO_NCU=1; DO_FULL=1 ;;
        --dry-run)          DRY_RUN=1 ;;
        --reps=*)           REPS="${arg#*=}" ;;
        --out=*)            OUT_OVERRIDE="${arg#*=}" ;;
        --comprehensive)
            # force-packed + every dispatch + fused/gemm/strip + baselines.
            # Decomposition subtracts (f-g, g-s) populate automatically.
            # e2 axis intentionally excluded: FC1 regresses 1.6x, FC2 often deadlocks/mismatches.
            DISPATCH_SPEC="all"
            PACKED_SPEC="yes"
            DECOMP_SPEC="all"
            EPI_WARPS_SPEC="4"
            BASELINES=1
            ;;
        -h|--help)          sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# ─── Dispatch sets ─────────────────────────────────────────────────────

DISP_CORE="default dgswizzle zigzag"
DISP_FC1_ALL="default dgswizzle zorder hilbert zigzag rowmajor ncycle ncyrot nflat nsnake nlock checkered dgsnake kstagger dg4 dg6 dg10 dg12 dg16 dg24 dg32 ck2 ck3 ck4 ck5 ck6 ck7 ck8 ck11"
DISP_FC2_ALL="default dgswizzle zorder hilbert zigzag rowmajor ncycle ncyrot nflat nsnake nlock checkered dgsnake kstagger kstagger2 kstagger3 dg2 dg3 dg4 dg6 dg10 dg12 dg16 dg20 dg24 dg32 ck2"

resolve_dispatches() {
    local layer="$1" spec="$2"
    case "$spec" in
        core) echo "$DISP_CORE" ;;
        all)  [ "$layer" = "fc1" ] && echo "$DISP_FC1_ALL" || echo "$DISP_FC2_ALL" ;;
        *)    echo "${spec//,/ }" ;;
    esac
}

resolve_packings() {
    case "$PACKED_SPEC" in
        both) echo "yes no" ;;
        yes)  echo "yes" ;;
        no)   echo "no" ;;
        *) echo "Bad --packed: $PACKED_SPEC" >&2; exit 1 ;;
    esac
}

resolve_modes() {
    case "$DECOMP_SPEC" in
        fused) echo "fused" ;;
        all)   echo "fused gemm strip" ;;
        *) echo "Bad --decomp: $DECOMP_SPEC" >&2; exit 1 ;;
    esac
}

# Epi-warp sweep. Valid values: 1, 2, 4 (kernel_common.cuh enforces 4 % N == 0).
resolve_epi_warps() {
    local ews="${EPI_WARPS_SPEC//,/ }"
    for n in $ews; do
        case "$n" in
            1|2|4) ;;
            *) echo "Bad --epi-warps value: $n (valid: 1, 2, 4)" >&2; exit 1 ;;
        esac
    done
    echo "$ews"
}

# Build the make-target suffix for a dispatch; also returns required flags.
# Echoes "<suffix>|<extra_flags>".  Empty suffix means base target.
dispatch_spec() {
    local layer="$1" disp="$2"
    local suffix="" extra=""
    case "$disp" in
        default)    suffix="" ;;
        sched)      suffix="-sched" ;;
        lean)       suffix="-lean" ;;
        tail)       suffix="-tail" ;;
        tail-lean)  suffix="-tail-lean" ;;
        rowsteal)   suffix="-rowsteal" ;;
        dgswizzle|zorder|hilbert|zigzag|rowmajor|ncycle|ncyrot|nflat|nsnake| \
        nlock|checkered|dgsnake| \
        kstagger|kstagger2|kstagger3| \
        dg2|dg3|dg4|dg6|dg10|dg12|dg16|dg20|dg24|dg32| \
        ck2|ck3|ck4|ck5|ck6|ck7|ck8|ck11)
            suffix="-${disp}"
            ;;
        *) echo "Bad dispatch: $disp" >&2; return 1 ;;
    esac
    echo "${suffix}|${extra}"
}

# Emit a label + build_cmd + binary for one (layer,disp,packed,mode,epi_warps) tuple.
# Format: "label|build_cmd|binary|kfilter"
# epi_warps=4 (default) keeps legacy labels unchanged.  For 1/2 an "-eN" suffix
# is injected into the disp slot so parse_label() still splits into 4 fields.
make_cfg() {
    local layer="$1" disp="$2" packed="$3" mode="$4" epi_warps="$5"
    local spec suffix extra
    spec=$(dispatch_spec "$layer" "$disp") || return 1
    suffix="${spec%|*}"
    extra="${spec##*|}"

    local target="${layer}-w3${suffix}"
    local flags="$extra"
    [ "$packed" = "yes" ]  && flags="$flags -DPACKED_TILES"
    [ "$mode"   = "gemm"  ] && flags="$flags -DGEMM_ONLY"
    [ "$mode"   = "strip" ] && flags="$flags -DSTRIP_EPILOGUE"
    [ "$epi_warps" != "4" ] && flags="$flags -DNUM_EPI_WARPS=${epi_warps}"
    flags="$(echo "$flags" | xargs)"  # collapse whitespace

    local pktag="np"; [ "$packed" = "yes" ] && pktag="p"
    local disp_tag="$disp"
    [ "$epi_warps" != "4" ] && disp_tag="${disp}-e${epi_warps}"
    local label="${layer}_${disp_tag//-/_}_${pktag}_${mode}"
    local out_bin="bench-${layer}-${disp_tag}-${pktag}-${mode}"

    local build_cmd
    if [ -z "$flags" ]; then
        build_cmd="make -B $target"
    else
        build_cmd="make -B $target DFLAGS='$flags'"
    fi
    build_cmd="$build_cmd && cp $target $out_bin"

    local kfilter="${layer}_w3_kernel"
    echo "${label}|${build_cmd}|./${out_bin}|${kfilter}"
}

# ─── Build config list ─────────────────────────────────────────────────

CFGS=()
for layer in $LAYERS; do
    for disp in $(resolve_dispatches "$layer" "$DISPATCH_SPEC"); do
        for packed in $(resolve_packings); do
            for mode in $(resolve_modes); do
                for ew in $(resolve_epi_warps); do
                    cfg=$(make_cfg "$layer" "$disp" "$packed" "$mode" "$ew") || exit 1
                    CFGS+=("$cfg")
                done
            done
        done
    done
done

# Baselines: cutlass (fc2 only) + cublas (per layer if requested).
if [ "$BASELINES" = "1" ]; then
    if [[ "$LAYERS" == *fc2* ]]; then
        CFGS+=("cutlass_fc2_fused|make fc2-cutlass && cp fc2-cutlass bench-cutlass-fc2-fused|./bench-cutlass-fc2-fused|regex:^(?!init)")
        CFGS+=("cutlass_fc2_strip|make fc2-cutlass-strip && cp fc2-cutlass-strip bench-cutlass-fc2-strip|./bench-cutlass-fc2-strip|regex:^(?!init)")
        if [ "$DO_NCU" = "1" ]; then
            CFGS+=("cublas_fc2_gemm|make -B cublas-bench-fc2-ncu && cp cublas-bench-fc2-ncu bench-cublas-fc2|./bench-cublas-fc2|regex:.")
        else
            CFGS+=("cublas_fc2_gemm|make -B cublas-bench-fc2 && cp cublas-bench-fc2 bench-cublas-fc2|./bench-cublas-fc2|regex:.")
        fi
    fi
    if [[ "$LAYERS" == *fc1* ]]; then
        if [ "$DO_NCU" = "1" ]; then
            CFGS+=("cublas_fc1_gemm|make -B cublas-bench-fc1-ncu && cp cublas-bench-fc1-ncu bench-cublas-fc1|./bench-cublas-fc1|regex:.")
        else
            CFGS+=("cublas_fc1_gemm|make -B cublas-bench-fc1 && cp cublas-bench-fc1 bench-cublas-fc1|./bench-cublas-fc1|regex:.")
        fi
    fi
fi

# ─── Output dir ────────────────────────────────────────────────────────

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="${OUT_OVERRIDE:-data/bench_${TIMESTAMP}}"
mkdir -p "$OUTDIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/session.log"; }

log "========================================"
log "  unified bench  $TIMESTAMP"
log "  layers=$LAYERS dispatch=$DISPATCH_SPEC packed=$PACKED_SPEC decomp=$DECOMP_SPEC epi-warps=$EPI_WARPS_SPEC"
log "  baselines=$BASELINES ncu=$DO_NCU full=$DO_FULL reps=$REPS"
log "  configs: ${#CFGS[@]}  out=$OUTDIR"
log "========================================"

if [ "$DRY_RUN" = "1" ]; then
    for cfg in "${CFGS[@]}"; do
        IFS='|' read -r label build_cmd binary kfilter <<< "$cfg"
        printf "  %-44s  %s  ->  %s\n" "$label" "$build_cmd" "$binary"
    done
    exit 0
fi

nvidia-smi > /dev/null 2>&1 || { log "FATAL: no GPU"; exit 1; }
nvidia-smi --query-gpu=gpu_name,clocks.sm --format=csv,noheader | tee -a "$OUTDIR/session.log"

# ─── Phase 1: build ────────────────────────────────────────────────────

log ""
log "── Phase 1: build (${#CFGS[@]} configs) ──"

for cfg in "${CFGS[@]}"; do
    IFS='|' read -r label build_cmd binary kfilter <<< "$cfg"
    log "  [$label] $build_cmd"
    if ! eval "$build_cmd" > "$OUTDIR/${label}_build.log" 2>&1; then
        log "  [$label] BUILD FAILED"
        cat "$OUTDIR/${label}_build.log" >> "$OUTDIR/session.log"
        continue
    fi
    regs=$(grep -o '[0-9]* registers' "$OUTDIR/${label}_build.log" | tail -1 | grep -o '[0-9]*')
    bars=$(grep -o 'used [0-9]* barriers' "$OUTDIR/${label}_build.log" | tail -1 | grep -o '[0-9]*')
    smem=$(grep -o '[0-9]* bytes of shared memory' "$OUTDIR/${label}_build.log" | tail -1 | grep -o '^[0-9]*')
    log "  [$label] built (regs=${regs:-?} bars=${bars:-?} smem=${smem:-?})"
done

# ─── Phase 2: wall time ────────────────────────────────────────────────

log ""
log "── Phase 2: wall time ──"

for cfg in "${CFGS[@]}"; do
    IFS='|' read -r label build_cmd binary kfilter <<< "$cfg"
    if [ ! -x "$binary" ]; then
        log "  [$label] MISSING: $binary"
        echo "@@RESULT ms=ERR label=$label" >> "$OUTDIR/results.txt"
        continue
    fi

    best_ms=""; last_line=""
    for ((r=1; r<=REPS; r++)); do
        output=$(timeout 60 $binary 2>&1) || true
        echo "$output" > "$OUTDIR/${label}_wall_r${r}.txt"
        line=$(echo "$output" | grep '@@RESULT' | head -1)
        [ -z "$line" ] && continue
        ms=$(echo "$line" | grep -o 'ms=[0-9.]*' | cut -d= -f2)
        last_line="$line"
        if [ -n "$ms" ]; then
            if [ -z "$best_ms" ] || awk -v a="$ms" -v b="$best_ms" 'BEGIN{exit !(a<b)}'; then
                best_ms="$ms"
            fi
        fi
    done

    regs=$(grep -o '[0-9]* registers' "$OUTDIR/${label}_build.log" 2>/dev/null | tail -1 | grep -o '[0-9]*')
    bars=$(grep -o 'used [0-9]* barriers' "$OUTDIR/${label}_build.log" 2>/dev/null | tail -1 | grep -o '[0-9]*')

    if [ -n "$last_line" ]; then
        # Patch the @@RESULT line's ms= to the best across reps, then append label/regs/bars.
        if [ -n "$best_ms" ]; then
            last_line=$(echo "$last_line" | sed "s/ms=[0-9.]*/ms=$best_ms/")
        fi
        echo "${last_line} label=${label} regs=${regs:-?} bars=${bars:-?}" >> "$OUTDIR/results.txt"
        valid=$(echo "$last_line" | grep -o 'valid=[01]' | cut -d= -f2)
        tflops=$(echo "$last_line" | grep -o 'tflops=[0-9.]*' | cut -d= -f2)
        log "  [$label] ${best_ms:-?}ms  ${tflops:-?} TFLOPS  valid=${valid:-?}  (reps=$REPS)"
    else
        log "  [$label] NO @@RESULT (hang/crash?)"
        echo "@@RESULT ms=ERR label=${label}" >> "$OUTDIR/results.txt"
    fi
done

# ─── Phase 3: ncu metrics (optional) ───────────────────────────────────

if [ "$DO_NCU" = "1" ]; then
    log ""
    log "── Phase 3: ncu metrics ──"

    METRICS_STALL="smsp__warps_issue_stalled_long_scoreboard.avg,smsp__warps_issue_stalled_short_scoreboard.avg,smsp__warps_issue_stalled_wait.avg,smsp__warps_issue_stalled_barrier.avg,smsp__warps_issue_stalled_sleeping.avg,smsp__warps_issue_stalled_not_selected.avg,smsp__warps_issue_stalled_mio_throttle.avg,smsp__warps_issue_stalled_math_pipe_throttle.avg"
    METRICS_MEM="dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,dram__sectors_read.sum,dram__sectors_write.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sector_hit_rate.pct,l1tex__throughput.avg.pct_of_peak_sustained_elapsed"
    METRICS_PIPE="sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed,sm__mio_pq_read_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__mio_pq_write_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.per_cycle_active,smsp__cycles_active.avg,smsp__inst_executed.sum"
    METRICS_SCHED="smsp__warps_eligible.avg.per_cycle_active,launch__registers_per_thread,launch__occupancy"
    METRICS_SMEM="l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_global_op_ld.sum"
    METRICS_TMA="l1tex__t_requests_pipe_tma_opc_read.sum,l1tex__t_requests_pipe_tma_opc_write.sum,l1tex__t_bytes_pipe_tma_opc_read.sum,l1tex__t_bytes_pipe_tma_opc_write.sum"
    METRICS_ALL="${METRICS_STALL},${METRICS_MEM},${METRICS_PIPE},${METRICS_SCHED},${METRICS_SMEM},${METRICS_TMA}"

    for cfg in "${CFGS[@]}"; do
        IFS='|' read -r label build_cmd binary kfilter <<< "$cfg"
        [ ! -x "$binary" ] && { log "  [$label] skip (no binary)"; continue; }

        log "  [$label] ncu ..."
        extra_ncu=""
        [[ "$label" == cublas_* ]] && extra_ncu="--profile-from-start off --launch-count 1 --launch-skip 3"
        if timeout 240 ncu --metrics "$METRICS_ALL" \
               --kernel-name "$kfilter" $extra_ncu \
               -o "$OUTDIR/${label}" "$binary" \
               > "$OUTDIR/${label}_ncu_stdout.txt" 2>"$OUTDIR/${label}_ncu_stderr.txt"; then
            ncu --import "$OUTDIR/${label}.ncu-rep" --csv > "$OUTDIR/${label}.csv" 2>/dev/null
            log "  [$label] ncu done"
        else
            rc=$?
            [ $rc -eq 124 ] && log "  [$label] NCU TIMEOUT" || log "  [$label] NCU FAIL rc=$rc"
            cat "$OUTDIR/${label}_ncu_stderr.txt" >> "$OUTDIR/session.log"
        fi

        if [ "$DO_FULL" = "1" ]; then
            log "  [$label] ncu --set full ..."
            if timeout 360 ncu --set full --kernel-name "$kfilter" $extra_ncu \
                   -o "$OUTDIR/full_${label}" "$binary" \
                   > "$OUTDIR/full_${label}_stdout.txt" 2>"$OUTDIR/full_${label}_stderr.txt"; then
                ncu --import "$OUTDIR/full_${label}.ncu-rep" --csv --page source \
                    > "$OUTDIR/full_${label}_source.csv" 2>/dev/null
                ncu --import "$OUTDIR/full_${label}.ncu-rep" --csv \
                    > "$OUTDIR/full_${label}.csv" 2>/dev/null
                log "  [$label] full done"
            else
                rc=$?
                [ $rc -eq 124 ] && log "  [$label] FULL TIMEOUT" || log "  [$label] FULL FAIL rc=$rc"
            fi
        fi
    done
fi

# ─── Phase 4: pairwise diffs (when ncu csvs exist) ─────────────────────

if [ "$DO_NCU" = "1" ]; then
    log ""
    log "── Phase 4: pairwise diffs ──"
    mkdir -p "$OUTDIR/diffs"

    # Collect the set of (layer, disp, pk) triples that successfully emitted
    # a CSV for at least one mode.  Avoid bash's lack of set datatypes by
    # accumulating into a uniq'd list.
    triples=$(ls "$OUTDIR"/*.csv 2>/dev/null \
        | sed -E 's#.*/([^/]+)\.csv$#\1#' \
        | awk -F'_' 'NF>=4 && ($1=="fc1"||$1=="fc2"){
              pk=$(NF-1); mode=$NF; disp=$2;
              for(i=3;i<=NF-2;i++) disp=disp"_"$(i);
              print $1"_"disp"_"pk
          }' | sort -u)

    # Per (layer, disp, pk): fused-vs-gemm, gemm-vs-strip.
    while read -r t; do
        [ -z "$t" ] && continue
        for pair in fused_vs_gemm:fused:gemm gemm_vs_strip:gemm:strip fused_vs_strip:fused:strip; do
            IFS=':' read -r pname ma mb <<< "$pair"
            a="$OUTDIR/${t}_${ma}.csv"; b="$OUTDIR/${t}_${mb}.csv"
            [ -f "$a" ] && [ -f "$b" ] && \
                python3 tools/ncu_diff.py "$b" "$a" > "$OUTDIR/diffs/${t}_${pname}.txt" 2>&1 || true
        done
    done <<< "$triples"

    # Per (layer, pk, mode): each dispatch vs default (baseline = default).
    # Needs default to be present; otherwise pick first sorted dispatch as base.
    modes=$(echo "$triples" | awk -F'_' '{print $1}' | sort -u)
    for layer in $modes; do
        for pk in p np; do
            for mode in fused gemm strip; do
                base="$OUTDIR/${layer}_default_${pk}_${mode}.csv"
                if [ ! -f "$base" ]; then
                    base=$(ls "$OUTDIR/${layer}_"*"_${pk}_${mode}.csv" 2>/dev/null | head -1)
                fi
                [ -z "$base" ] || [ ! -f "$base" ] && continue
                bname=$(basename "$base" .csv)
                for peer in "$OUTDIR/${layer}_"*"_${pk}_${mode}.csv"; do
                    [ -f "$peer" ] || continue
                    pname=$(basename "$peer" .csv)
                    [ "$pname" = "$bname" ] && continue
                    python3 tools/ncu_diff.py "$base" "$peer" \
                        > "$OUTDIR/diffs/${pname}_vs_$(echo "$bname" | sed -E "s/${layer}_//; s/_${pk}_${mode}//").txt" 2>&1 || true
                done
            done
        done
    done

    # CUTLASS vs best w3 (if both exist) for FC2.
    for mode in fused strip; do
        cl="$OUTDIR/cutlass_fc2_${mode}.csv"
        [ -f "$cl" ] || continue
        for w3 in "$OUTDIR/fc2_"*"_p_${mode}.csv"; do
            [ -f "$w3" ] || continue
            wname=$(basename "$w3" .csv)
            python3 tools/ncu_diff.py "$cl" "$w3" \
                > "$OUTDIR/diffs/${wname}_vs_cutlass_${mode}.txt" 2>&1 || true
        done
    done

    log "  wrote $(ls "$OUTDIR/diffs" 2>/dev/null | wc -l) diff files to $OUTDIR/diffs/"
fi

# ─── Phase 5: summary ──────────────────────────────────────────────────

log ""
log "── Phase 5: summary ──"

# Per-layer/per-packing decomposition table from results.txt.
python3 - "$OUTDIR" <<'PYEOF' 2>&1 | tee "$OUTDIR/summary.txt" | tee -a "$OUTDIR/session.log"
import sys, os, csv, re

outdir = sys.argv[1]
res_path = os.path.join(outdir, "results.txt")
if not os.path.exists(res_path):
    print("no results.txt"); sys.exit(0)

rows = []
with open(res_path) as f:
    for line in f:
        m = dict(re.findall(r"(\w+)=(\S+)", line))
        if "label" not in m: continue
        rows.append(m)

# ── Wall-time table ──
print("\n=== WALL TIME ===")
print(f"{'label':<42}  {'ms':>8}  {'tflops':>8}  {'valid':>5}  {'regs':>4}  {'bars':>4}")
print("-" * 82)
for r in sorted(rows, key=lambda x: x.get("label", "")):
    print(f"{r['label']:<42}  {r.get('ms','?'):>8}  {r.get('tflops','?'):>8}  "
          f"{r.get('valid','?'):>5}  {r.get('regs','?'):>4}  {r.get('bars','?'):>4}")

# ── Decomposition: per (layer, dispatch, packed) show fused / gemm / strip ──
def parse_label(lbl):
    if lbl.startswith("cutlass_") or lbl.startswith("cublas_"):
        return None
    parts = lbl.split("_")
    if len(parts) < 4: return None
    layer = parts[0]
    mode = parts[-1]
    pk = parts[-2]
    disp = "_".join(parts[1:-2])
    return layer, disp, pk, mode

# Build lookup: (layer, disp, pk) -> {mode: ms}
table = {}
for r in rows:
    lbl = r.get("label", "")
    parsed = parse_label(lbl)
    if parsed is None: continue
    layer, disp, pk, mode = parsed
    ms = r.get("ms", "")
    try: ms_f = float(ms)
    except: ms_f = None
    table.setdefault((layer, disp, pk), {})[mode] = ms_f

if table:
    print("\n=== DECOMPOSITION (ms) ===")
    hdr = f"{'layer':<4}  {'dispatch':<12}  {'pk':<2}  {'fused':>8}  {'gemm':>8}  {'strip':>8}  {'f-g':>8}  {'g-s':>8}"
    print(hdr); print("-" * len(hdr))
    for (layer, disp, pk), modes in sorted(table.items()):
        def fmt(v): return "—" if v is None else f"{v:.3f}"
        f = modes.get("fused"); g = modes.get("gemm"); s = modes.get("strip")
        fg = f - g if (f is not None and g is not None) else None
        gs = g - s if (g is not None and s is not None) else None
        print(f"{layer:<4}  {disp:<12}  {pk:<2}  "
              f"{fmt(f):>8}  {fmt(g):>8}  {fmt(s):>8}  {fmt(fg):>8}  {fmt(gs):>8}")

# ── DRAM amplification from ncu CSVs (if any) ──
UNIT_SCALE = {"Gbyte": 1e9, "Mbyte": 1e6, "Kbyte": 1e3, "byte": 1,
              "Gsector": 1e9, "Msector": 1e6, "Ksector": 1e3, "sector": 1}
def to_base(s, u):
    try: v = float(s.replace(",", ""))
    except: return None
    return v * UNIT_SCALE.get(u, 1.0)

# FC1: M=928256 K=768 N=3072; FC2: M=928256 K=3072 N=768.
THEORY = {
    "fc1": {"M": 928256, "K": 768, "N": 3072},
    "fc2": {"M": 928256, "K": 3072, "N": 768},
}
def theoretical(layer, mode):
    d = THEORY[layer]
    M, K, N = d["M"], d["K"], d["N"]
    # A (FP8) + B (FP8) + optional residual (BF16, FC2 fused/gemm only) + bias (BF16)
    if layer == "fc2" and mode in ("fused", "gemm"):
        rd = M*K + K*N + M*N*2 + N*2
    else:
        rd = M*K + K*N + N*2
    wr = M*N*2 if mode != "strip" else 0
    return rd, wr

ncu_rows = []
for r in rows:
    lbl = r.get("label", "")
    csv_path = os.path.join(outdir, f"{lbl}.csv")
    if not os.path.exists(csv_path): continue
    metrics = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            metrics[row.get("Metric Name", "")] = (row.get("Metric Value", ""), row.get("Metric Unit", ""))
    dr = metrics.get("dram__bytes_read.sum", ("", ""))
    dw = metrics.get("dram__bytes_write.sum", ("", ""))
    dr_v = to_base(*dr); dw_v = to_base(*dw)
    ncu_rows.append((lbl, dr_v, dw_v, metrics))

if ncu_rows:
    print("\n=== DRAM AMPLIFICATION ===")
    hdr = f"{'label':<42}  {'read_GB':>9}  {'read_amp':>9}  {'write_GB':>9}  {'write_amp':>9}"
    print(hdr); print("-" * len(hdr))
    for lbl, dr_v, dw_v, _m in ncu_rows:
        parsed = parse_label(lbl)
        if parsed is None:
            print(f"{lbl:<42}  "
                  f"{'?' if dr_v is None else f'{dr_v/1e9:.3f}':>9}  {'—':>9}  "
                  f"{'?' if dw_v is None else f'{dw_v/1e9:.3f}':>9}  {'—':>9}")
            continue
        layer, disp, pk, mode = parsed
        th_r, th_w = theoretical(layer, mode)
        ra = dr_v/th_r if (dr_v is not None and th_r) else None
        wa = dw_v/th_w if (dw_v is not None and th_w) else None
        def f(x, fmt): return "?" if x is None else (fmt % x)
        print(f"{lbl:<42}  {f(dr_v/1e9 if dr_v is not None else None,'%.3f'):>9}  "
              f"{f(ra,'%.2fx'):>9}  "
              f"{f(dw_v/1e9 if dw_v is not None else None,'%.3f'):>9}  "
              f"{f(wa,'%.2fx'):>9}")

# ── Full 35-metric side-by-side, grouped by (layer, pk, mode) ──
if ncu_rows:
    FULL_METRICS = [
        # Stall reasons
        ("long_sb",           "smsp__warps_issue_stalled_long_scoreboard.avg"),
        ("short_sb",          "smsp__warps_issue_stalled_short_scoreboard.avg"),
        ("wait",              "smsp__warps_issue_stalled_wait.avg"),
        ("barrier",           "smsp__warps_issue_stalled_barrier.avg"),
        ("sleeping",          "smsp__warps_issue_stalled_sleeping.avg"),
        ("not_selected",      "smsp__warps_issue_stalled_not_selected.avg"),
        ("mio_thr",           "smsp__warps_issue_stalled_mio_throttle.avg"),
        ("math_thr",          "smsp__warps_issue_stalled_math_pipe_throttle.avg"),
        # DRAM
        ("dram_rd_B",         "dram__bytes_read.sum"),
        ("dram_wr_B",         "dram__bytes_write.sum"),
        ("dram_rd_sec",       "dram__sectors_read.sum"),
        ("dram_wr_sec",       "dram__sectors_write.sum"),
        ("dram_pct",          "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        # L2
        ("lts_pct",           "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
        ("lts_sec",           "lts__t_sectors.sum"),
        ("lts_rd_sec",        "lts__t_sectors_op_read.sum"),
        ("lts_wr_sec",        "lts__t_sectors_op_write.sum"),
        ("lts_hit",           "lts__t_sector_hit_rate.pct"),
        ("l1tex_pct",         "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"),
        # Pipe util
        ("sm_pct",            "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        ("shared_pct",        "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed"),
        ("lsu_pct",           "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed"),
        ("miorq_rd_pct",      "sm__mio_pq_read_cycles_active.avg.pct_of_peak_sustained_elapsed"),
        ("miorq_wr_pct",      "sm__mio_pq_write_cycles_active.avg.pct_of_peak_sustained_elapsed"),
        ("warps_act",         "sm__warps_active.avg.per_cycle_active"),
        ("cyc_act",           "smsp__cycles_active.avg"),
        ("inst_exec",         "smsp__inst_executed.sum"),
        # Scheduler
        ("warps_elig",        "smsp__warps_eligible.avg.per_cycle_active"),
        ("regs",              "launch__registers_per_thread"),
        ("occ",               "launch__occupancy"),
        # SMEM wavefronts
        ("smem_wf",           "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"),
        ("smem_ld_wf",        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"),
        ("smem_st_wf",        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"),
        ("gbl_ld_wf",         "l1tex__data_pipe_lsu_wavefronts_mem_global_op_ld.sum"),
        # TMA
        ("tma_rd_reqs",       "l1tex__t_requests_pipe_tma_opc_read.sum"),
        ("tma_wr_reqs",       "l1tex__t_requests_pipe_tma_opc_write.sum"),
        ("tma_rd_B",          "l1tex__t_bytes_pipe_tma_opc_read.sum"),
        ("tma_wr_B",          "l1tex__t_bytes_pipe_tma_opc_write.sum"),
    ]

    def fmt_metric(vu):
        v, u = vu if vu else ("","")
        bv = to_base(v, u)
        if bv is None: return v[:12] if v else "—"
        if abs(bv) >= 1e9: return f"{bv/1e9:.2f}G"
        if abs(bv) >= 1e6: return f"{bv/1e6:.2f}M"
        if abs(bv) >= 1e3: return f"{bv/1e3:.1f}K"
        return f"{bv:.2f}"

    # Group rows by (layer, pk, mode) so each table is apples-to-apples.
    groups = {}
    baselines = []
    for lbl, _dr, _dw, m in ncu_rows:
        parsed = parse_label(lbl)
        if parsed is None:
            baselines.append((lbl, m))
            continue
        layer, disp, pk, mode = parsed
        groups.setdefault((layer, pk, mode), []).append((disp, lbl, m))

    for (layer, pk, mode), entries in sorted(groups.items()):
        print(f"\n=== FULL METRICS: {layer} pk={pk} mode={mode} ===")
        labels = sorted(entries, key=lambda x: x[0])
        col_w = 14
        hdr = f"{'metric':<16}"
        for disp, _lbl, _m in labels:
            hdr += f"  {disp[:col_w]:>{col_w}}"
        print(hdr); print("-" * len(hdr))
        for short, full in FULL_METRICS:
            row = f"{short:<16}"
            for _disp, _lbl, m in labels:
                row += f"  {fmt_metric(m.get(full)):>{col_w}}"
            print(row)

    if baselines:
        print("\n=== FULL METRICS: baselines (cutlass / cublas) ===")
        col_w = 16
        hdr = f"{'metric':<16}"
        for lbl, _m in baselines:
            hdr += f"  {lbl[:col_w]:>{col_w}}"
        print(hdr); print("-" * len(hdr))
        for short, full in FULL_METRICS:
            row = f"{short:<16}"
            for _lbl, m in baselines:
                row += f"  {fmt_metric(m.get(full)):>{col_w}}"
            print(row)

PYEOF

log ""
log "Outputs: $OUTDIR/{results.txt,summary.txt,*_build.log,*.csv,*.ncu-rep,diffs/}"
log "Analysis: python3 tools/ncu_anova.py $OUTDIR"
log "Done."
