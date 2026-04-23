#!/usr/bin/env bash
# ncu comprehensive W5-bottleneck diagnosis for fc2_w3x vs rank-1.
#
# Runs in two phases:
#   1. Focused metric pack per variant (fast: ~30 s each)
#        stall-reason breakdown (long_scoreboard, mio_throttle, barrier,
#        membar, drain, wait, etc.), L2 hit rate, DRAM bytes, warps_active
#        across: fc2_w3x (baseline), fc2_w3x-strip (MMA-only floor),
#                fc2_w3x-gemm   (no epi), rank-1 (cuBLASLt L2 listing)
#   2. --set full capture on fc2_w3x + rank-1 only (~3-5 min each).
#      Exports ncu-rep so you can open in Nsight GUI or `ncu --import`.
#
# Non-docker requirement: ncu needs CAP_SYS_ADMIN or sudo for perf
# counter access; if you get "ERR_NVGPUCTRPERM" run with sudo.
#
# Budget: ~25-35 min total (well under the 1-hour window).
#
# Output: data/ncu_w3x_<timestamp>/
#   {w3x,w3x-strip,w3x-gemm,rank1}.csv           focused metrics
#   full_{w3x,rank1}.ncu-rep                     exhaustive reports
#   summary.txt                                  decoded diff table
#
set -u
cd "$(dirname "$0")/.."

OUT="data/ncu_w3x_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
LOG="$OUT/run.log"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

DO_FULL=1
DO_FOCUSED=1
while [ $# -gt 0 ]; do
    case "$1" in
        --focused-only) DO_FULL=0;    shift ;;
        --full-only)    DO_FOCUSED=0; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

command -v ncu >/dev/null || { echo "ncu not in PATH" >&2; exit 1; }

log "=== ncu fc2_w3x W5-diagnosis sweep → $OUT ==="

log "--- build ---"
make fc2-w3x fc2-w3x-strip fc2-w3x-gemm cublaslt-fc2 2>&1 \
    | tee -a "$LOG" | tail -8

TARGETS=(
    "w3x      ./fc2-w3x        fc2_w3x_kernel"
    "w3x-strip ./fc2-w3x-strip fc2_w3x_kernel"
    "w3x-gemm  ./fc2-w3x-gemm  fc2_w3x_kernel"
    "rank1    ./cublaslt-fc2   nvjet_sm100_qqtst"
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
    log "--- phase 1: focused metric pack ---"
    for entry in "${TARGETS[@]}"; do
        read -r name bin filt <<<"$entry"
        if [ ! -x "$bin" ]; then
            log "  SKIP $name: $bin missing"
            continue
        fi
        log "  ncu $name"
        ncu --target-processes all \
            -k "regex:$filt" \
            --launch-skip 2 --launch-count 1 \
            --metrics "$METRICS" \
            --csv \
            "$bin" > "$OUT/$name.csv" 2> "$OUT/$name.stderr" \
            || log "    WARN ncu returned nonzero for $name (see $OUT/$name.stderr)"
    done
fi

if [ "$DO_FULL" = 1 ]; then
    log ""
    log "--- phase 2: --set full (w3x + rank1 only) ---"
    for entry in "w3x ./fc2-w3x fc2_w3x_kernel" \
                 "rank1 ./cublaslt-fc2 nvjet_sm100_qqtst"; do
        read -r name bin filt <<<"$entry"
        if [ ! -x "$bin" ]; then
            log "  SKIP full/$name: $bin missing"
            continue
        fi
        log "  full/$name (3-5 min)"
        ncu --target-processes all \
            -k "regex:$filt" \
            --launch-skip 2 --launch-count 1 \
            --set full \
            --export "$OUT/full_$name" \
            --force-overwrite \
            "$bin" > "$OUT/full_$name.stdout" 2> "$OUT/full_$name.stderr" \
            || log "    WARN --set full returned nonzero for $name"
    done
fi

log ""
log "--- phase 3: decoded summary ---"
python3 - "$OUT" <<'PY' | tee "$OUT/summary.txt"
import csv, os, sys, glob

out = sys.argv[1]
variants = ["w3x", "w3x-strip", "w3x-gemm", "rank1"]
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

print("=== focused metric diff (rank1 = apples-to-apples baseline) ===\n")
col_w = 16
hdr = f"{'metric':<{col_w}}"
for v in variants:
    if v in rows_by_variant:
        hdr += f"{v:>14}"
print(hdr)
print("-" * len(hdr))

for metric, short in key_metrics:
    line = f"{short:<{col_w}}"
    for v in variants:
        if v not in rows_by_variant:
            continue
        val = rows_by_variant[v].get(metric, "-")
        line += f"{val:>14}"
    print(line)

print()
print("=== reading this table ===")
print("  long_sb  : global-memory stall. High on w3x but low on rank1 → we wait on B-tile arrival.")
print("  mio_thr  : TMEM/SMEM handoff stall. Finding A (epi-TMEM backpressure) signature.")
print("  barrier  : mbarrier wait. High on w3x → sync-side stall (but we tried all those → dead).")
print("  membar   : memory fence wait. Store-side serialization.")
print("  short_sb : SMEM/reg file stall. Usually small.")
print("  wait     : DEPBAR/other generic wait.")
print("  drain    : warp retiring, pipeline draining.")
print("  dispatch : issue slot contention.")
print()
print("=== interpretation guide ===")
print("  Compare w3x vs rank1 column-by-column. The w3x column that is")
print("  SIGNIFICANTLY HIGHER than rank1 is the stall category that must be")
print("  reduced to close the gap (or open it further if we're ahead).")
print()
print("  Cross-reference with PROFILE_W5 per-in_g output:")
print("    ncu long_sb high  +  PROFILE_W5 residual high at in_g=0..3")
print("      → finding A: epi-TMEM backpressure, cross-group B-prefetch is the lever")
print("    ncu mio_thr high  +  PROFILE_W5 residual high at in_g=0..3")
print("      → finding A (same conclusion)")
print("    ncu uniform       +  PROFILE_W5 residual uniform")
print("      → finding D: we're done at 1.006 ms")
PY

log ""
log "--- artifacts ---"
log "  focused csvs:  $OUT/{w3x,w3x-strip,w3x-gemm,rank1}.csv"
log "  full reports:  $OUT/full_{w3x,rank1}.ncu-rep   (open with Nsight or ncu --import)"
log "  decoded diff:  $OUT/summary.txt"
log ""
log "  quick dig:"
log "    ncu --import $OUT/full_w3x.ncu-rep   --page details | less -R"
log "    ncu --import $OUT/full_rank1.ncu-rep --page details | less -R"
log "    # side-by-side (existing tool):"
log "    python3 tools/ncu_diff.py $OUT/full_w3x.ncu-rep $OUT/full_rank1.ncu-rep"
