#!/usr/bin/env bash
# ncu_fc2_w3x_strip_diff.sh — strip-vs-full localization for the 16 µs gap.
#
# Drop-in companion to ncu_fc2_w3x.sh that only profiles fc2_w3x (full)
# and fc2_w3x-strip (NS=6 + PREFILL + W4-TMA structural floor). The delta
# between them is the exposed epilogue work (15.9 µs / 31460 cyc /
# 214 cyc/tile on production geometry, B200 2026-04-30 measurement) — this
# script tells us WHICH stall category that work lives in.
#
# No rank-1, no gemm — just w3x and w3x-strip with --set full on both by
# default (since dropping rank-1 leaves both as the primary diagnostic
# targets).
#
# Non-docker requirement: ncu needs CAP_SYS_ADMIN or sudo for perf
# counter access; if you get "ERR_NVGPUCTRPERM" run with sudo.
#
# Budget: ~12-18 min default, ~20-30 min with --max.
#
# Flags:
#   (none)              Phase 1 + Phase 2.    ~12-18 min.
#   --focused-only      Phase 1 only.         ~1-2 min.
#   --full-only         Phase 2 only.         ~10-15 min.
#   --with-source       Embed source-line attribution in --set full export
#                       (requires -lineinfo in CFLAGS, already on).
#   --reps N            Run Phase 1 N times for variance.
#   --max               Shortcut: --with-source --reps 3.   ~20-30 min.
#
# Output: data/ncu_w3x_strip_<timestamp>/
#   {w3x,w3x-strip}[_r<N>].csv         focused metrics
#   full_{w3x,w3x-strip}.ncu-rep       --set full reports
#   summary.txt                        decoded diff (full vs strip + delta)
#
set -u
cd "$(dirname "$0")/.."

OUT="data/ncu_w3x_strip_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
LOG="$OUT/run.log"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

DO_FULL=1
DO_FOCUSED=1
WITH_SOURCE=0
REPS=1
while [ $# -gt 0 ]; do
    case "$1" in
        --focused-only)  DO_FULL=0;     shift ;;
        --full-only)     DO_FOCUSED=0;  shift ;;
        --with-source)   WITH_SOURCE=1; shift ;;
        --reps)          REPS="$2";     shift 2 ;;
        --max)
            DO_FOCUSED=1
            DO_FULL=1
            WITH_SOURCE=1
            REPS=3
            shift ;;
        -h|--help)
            sed -n '1,32p' "$0"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SOURCE_FLAGS=""
if [ "$WITH_SOURCE" = 1 ]; then
    SOURCE_FLAGS="--import-source yes"
fi

command -v ncu >/dev/null || { echo "ncu not in PATH" >&2; exit 1; }

log "=== ncu fc2_w3x strip-vs-full diff → $OUT ==="

log "--- build ---"
make fc2-w3x-ncu fc2-w3x-ncu-strip 2>&1 | tee -a "$LOG" | tail -8

TARGETS=(
    "w3x       ./fc2-w3x-ncu        fc2_w3x_kernel  1"
    "w3x-strip ./fc2-w3x-ncu-strip  fc2_w3x_kernel  1"
)

METRICS="$(cat <<'EOF' | tr -d ' \n'
sm__warps_active.avg.pct_of_peak_sustained_active,
smsp__thread_inst_executed_per_inst_executed.ratio,
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,
smsp__average_warps_issue_stalled_membar_per_issue_active.ratio,
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,
smsp__average_warps_issue_stalled_drain_per_issue_active.ratio,
smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio,
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,
smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio,
smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio,
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,
smsp__average_warps_issue_stalled_selected_per_issue_active.ratio,
lts__t_sector_hit_rate.pct,
lts__t_sectors_op_read.sum,
lts__t_sectors_op_write.sum,
dram__bytes_read.sum,
dram__bytes_write.sum,
l1tex__t_sector_hit_rate.pct,
smsp__inst_executed.sum,
sm__cycles_elapsed.avg,
smsp__issue_active.avg.pct_of_peak_sustained_active,
smsp__sass_thread_inst_executed_op_memory_shared_st.sum,
smsp__sass_thread_inst_executed_op_memory_shared_ld.sum
EOF
)"

if [ "$DO_FOCUSED" = 1 ]; then
    log ""
    log "--- phase 1: focused metric pack (reps=$REPS) ---"
    for rep in $(seq 1 "$REPS"); do
        [ "$REPS" -gt 1 ] && log "  -- rep $rep/$REPS --"
        for entry in "${TARGETS[@]}"; do
            read -r name bin filt lskip <<<"$entry"
            if [ ! -x "$bin" ]; then
                log "  SKIP $name: $bin missing"
                continue
            fi
            suffix=""
            [ "$REPS" -gt 1 ] && suffix="_r$rep"
            log "  ncu $name$suffix"
            ncu --target-processes all \
                -k "regex:$filt" \
                --launch-skip "$lskip" --launch-count 1 \
                --metrics "$METRICS" \
                --csv \
                "$bin" > "$OUT/$name$suffix.csv" 2> "$OUT/$name$suffix.stderr" \
                || log "    WARN ncu returned nonzero for $name$suffix (see $OUT/$name$suffix.stderr)"
        done
    done
fi

if [ "$DO_FULL" = 1 ]; then
    log ""
    log "--- phase 2: --set full (both variants) ($SOURCE_FLAGS) ---"
    for entry in "${TARGETS[@]}"; do
        read -r name bin filt lskip <<<"$entry"
        if [ ! -x "$bin" ]; then
            log "  SKIP full/$name: $bin missing"
            continue
        fi
        log "  full/$name (3-5 min)"
        ncu --target-processes all \
            -k "regex:$filt" \
            --launch-skip "$lskip" --launch-count 1 \
            --set full \
            $SOURCE_FLAGS \
            --export "$OUT/full_$name" \
            --force-overwrite \
            "$bin" > "$OUT/full_$name.stdout" 2> "$OUT/full_$name.stderr" \
            || log "    WARN --set full returned nonzero for $name"
    done
fi

log ""
log "--- phase 3: decoded summary ---"
python3 - "$OUT" <<'PY' | tee "$OUT/summary.txt"
import csv, os, sys

out = sys.argv[1]
variants = ["w3x", "w3x-strip"]
rows_by_variant = {}
for v in variants:
    f = os.path.join(out, f"{v}.csv")
    if not os.path.exists(f) or os.path.getsize(f) == 0:
        continue
    with open(f) as fh:
        reader = csv.reader(fh)
        header = None
        data = {}
        for row in reader:
            if not row or row[0].startswith('='):
                continue
            if row[0] == 'ID':
                header = row
                continue
            if header and len(row) == len(header):
                mname = row[header.index('Metric Name')] if 'Metric Name' in header else None
                mval  = row[header.index('Metric Value')] if 'Metric Value' in header else None
                if mname and mval is not None:
                    data[mname] = mval
        rows_by_variant[v] = data

if not rows_by_variant:
    print("(no csv data — check phase 1 logs)")
    sys.exit(0)

key_metrics = [
    ("sm__warps_active.avg.pct_of_peak_sustained_active",          "warps_act %"),
    ("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",  "long_sb"),
    ("smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",     "mio_thr"),
    ("smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",          "barrier"),
    ("smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",           "membar"),
    ("smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio", "short_sb"),
    ("smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",             "wait"),
    ("smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",            "drain"),
    ("smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",   "dispatch"),
    ("lts__t_sector_hit_rate.pct",                                                "L2 hit %"),
    ("l1tex__t_sector_hit_rate.pct",                                              "L1 hit %"),
    ("dram__bytes_read.sum",                                                      "DRAM rd"),
    ("dram__bytes_write.sum",                                                     "DRAM wr"),
    ("sm__cycles_elapsed.avg",                                                    "cyc avg"),
]

def to_float(s):
    if s is None or s == "-":
        return None
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None

print("=== focused metric diff (w3x = full kernel, w3x-strip = NS=6 staging floor) ===")
print()
print("    delta column = w3x - w3x-strip = exposed epilogue work signature")
print()

col_w = 16
hdr = f"{'metric':<{col_w}}{'w3x':>14}{'w3x-strip':>14}{'delta':>14}"
print(hdr)
print("-" * len(hdr))

for metric, short in key_metrics:
    vfull   = rows_by_variant.get("w3x", {}).get(metric, "-")
    vstrip  = rows_by_variant.get("w3x-strip", {}).get(metric, "-")
    nfull   = to_float(vfull)
    nstrip  = to_float(vstrip)
    if nfull is not None and nstrip is not None:
        delta = nfull - nstrip
        delta_str = f"{delta:+,.4g}" if abs(delta) >= 0.01 else f"{delta:+.6f}"
    else:
        delta_str = "-"
    print(f"{short:<{col_w}}{vfull:>14}{vstrip:>14}{delta_str:>14}")

print()
print("=== reading this table ===")
print("  Each row's DELTA is the strip→full increase. Largest positive deltas")
print("  point at WHERE the 16 µs of exposed epi lives.")
print()
print("    long_sb   ↑  : exposed LDTM/LDS serialization in epi subpasses")
print("    mio_thr   ↑  : TMEM/SMEM port contention between W0-W3 epi warps")
print("    barrier   ↑  : mbarrier wait — epi-done handoff or PREFILL boundary")
print("    membar    ↑  : proxy fence / cluster bar.sync at tile boundary")
print("    drain     ↑  : pipeline draining on final tile (no shadow available)")
print("    short_sb  ↑  : SMEM bank conflicts in STSM or CVT register file")
print("    wait      ↑  : DEPBAR — generic dependency wait (often subpass chain)")
print()
print("  cycles delta should track 31460 cyc (≈ 16 µs at B200 2.0 GHz).")
print("  warps_act delta = how much extra W0-W3 activity full kernel surfaces.")
print()
print("=== where to dig from here ===")
print("  ncu --import {0}/full_w3x.ncu-rep       --page details | less -R".format(out))
print("  ncu --import {0}/full_w3x-strip.ncu-rep --page details | less -R".format(out))
print("  python3 tools/ncu_diff.py {0}/full_w3x.ncu-rep {0}/full_w3x-strip.ncu-rep".format(out))
PY

log ""
log "--- artifacts ---"
log "  focused csvs:  $OUT/{w3x,w3x-strip}.csv"
log "  full reports:  $OUT/full_{w3x,w3x-strip}.ncu-rep"
log "  decoded diff:  $OUT/summary.txt"
log ""
log "  paste summary.txt to claude for breakdown of the 16 µs / 214 cyc/tile."
