#!/usr/bin/env bash
# ncu_fc2_w3x_tile_sweep.sh — per-variant ncu metric pack across tile-dispatch cells.
#
# Drives the in-binary VARIANT= selector with one ncu invocation per cell, so
# each variant's hardware counters are isolated.  Default cell list is the
# 7-member basin floor (blkswap, blkx5/6/7, blk_qrt0/2/3) + dgsw baseline +
# dg_snlmrev (TIE-with-dgsw reference) + gflip_cidperm (DECISIVE loser
# contrast).  Override with --variants name1,name2,...
#
# Phase 1 (default): focused metric pack — long_sb / mio / barrier / membar /
#   drain / L2 / DRAM / cyc_avg etc.  ~30-45 s per variant, ~5-8 min for 9.
# Phase 2 (--full):  --set full ncu-rep export per variant.  3-5 min each,
#   ~30-45 min for 9.  Skipped by default.
#
# Non-docker requirement: ncu needs CAP_SYS_ADMIN or sudo for perf counter
# access; if you get "ERR_NVGPUCTRPERM" run with sudo.
#
# Flags:
#   (none)                   Phase 1 only.                     ~5-8 min.
#   --full                   Phase 1 + Phase 2 (--set full).   ~35-50 min.
#   --full-only              Phase 2 only.                     ~30-45 min.
#   --variants name1,name2   Custom cell list (comma-separated, must be in VARIANTS[]).
#   --reps N                 Phase 1 reps for variance.        each rep adds ~5-8 min.
#   --with-source            Embed source attribution in --set full export.
#
# Output: data/ncu_w3x_tiles_<timestamp>/
#   <name>[_r<R>].csv        focused metrics per variant
#   full_<name>.ncu-rep      --set full report per variant (if --full)
#   summary.txt              per-row table + delta-vs-dgsw
set -u
cd "$(dirname "$0")/.."

OUT="data/ncu_w3x_tiles_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
LOG="$OUT/run.log"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

DO_FOCUSED=1
DO_FULL=0
WITH_SOURCE=0
REPS=1
VARIANTS_CSV="dgsw,gflip_blkswap,gflip_blkx5,gflip_blkx6,gflip_blkx7,gflip_blk_qrt0,gflip_blk_qrt2,gflip_blk_qrt3,dg_snlmrev,gflip_cidperm"
BASELINE="dgsw"

while [ $# -gt 0 ]; do
    case "$1" in
        --full)         DO_FULL=1;            shift ;;
        --full-only)    DO_FULL=1; DO_FOCUSED=0; shift ;;
        --with-source)  WITH_SOURCE=1;        shift ;;
        --reps)         REPS="$2";            shift 2 ;;
        --variants)     VARIANTS_CSV="$2";    shift 2 ;;
        --baseline)     BASELINE="$2";        shift 2 ;;
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

IFS=',' read -r -a VARIANT_LIST <<<"$VARIANTS_CSV"

log "=== ncu fc2_w3x tile-dispatch sweep → $OUT ==="
log "    cells: ${VARIANT_LIST[*]}"
log "    baseline for delta column: $BASELINE"

log "--- build ---"
make fc2-w3x-ncu 2>&1 | tee -a "$LOG" | tail -8

if [ ! -x ./fc2-w3x-ncu ]; then
    echo "build failed: fc2-w3x-ncu missing" >&2
    exit 1
fi

KERN_REGEX="fc2_w3x_kernel"

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
        for name in "${VARIANT_LIST[@]}"; do
            suffix=""
            [ "$REPS" -gt 1 ] && suffix="_r$rep"
            log "  ncu $name$suffix (~30-45s)"
            VARIANT="$name" ncu --target-processes all \
                -k "regex:$KERN_REGEX" \
                --launch-skip 1 --launch-count 1 \
                --metrics "$METRICS" \
                --csv \
                ./fc2-w3x-ncu > "$OUT/$name$suffix.csv" 2> "$OUT/$name$suffix.stderr" \
                || log "    WARN ncu returned nonzero for $name$suffix (see $OUT/$name$suffix.stderr)"
        done
    done
fi

if [ "$DO_FULL" = 1 ]; then
    log ""
    log "--- phase 2: --set full per variant ($SOURCE_FLAGS) ---"
    for name in "${VARIANT_LIST[@]}"; do
        log "  full/$name (3-5 min)"
        VARIANT="$name" ncu --target-processes all \
            -k "regex:$KERN_REGEX" \
            --launch-skip 1 --launch-count 1 \
            --set full \
            $SOURCE_FLAGS \
            --export "$OUT/full_$name" \
            --force-overwrite \
            ./fc2-w3x-ncu > "$OUT/full_$name.stdout" 2> "$OUT/full_$name.stderr" \
            || log "    WARN --set full returned nonzero for $name"
    done
fi

log ""
log "--- phase 3: decoded summary ---"
python3 - "$OUT" "$BASELINE" "${VARIANT_LIST[@]}" <<'PY' | tee "$OUT/summary.txt"
import csv, os, sys

out, baseline, *variants = sys.argv[1:]

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
    ("sm__cycles_elapsed.avg",                                                    "cyc avg"),
    ("sm__warps_active.avg.pct_of_peak_sustained_active",                         "warps_act %"),
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
]

def to_float(s):
    if s is None or s == "-":
        return None
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None

print(f"=== focused metric sweep across {len(rows_by_variant)} cells (baseline = {baseline}) ===")
print()
print(f"    delta column = <variant> - {baseline}.  Per-row meaning:")
print("      cyc avg ↓  : faster on this cell")
print("      long_sb ↓  : less synchronous-A-wavefront stall (cluster→L2 contiguity)")
print("      mio_thr ↓  : less TMEM/SMEM port contention")
print("      barrier ↓  : less mbarrier wait / less epi handoff coupling")
print("      L2 hit% ↑  : better cluster→L2 partition affinity")
print("      DRAM rd ↓  : less amplification (NB: not the bottleneck — see CLAUDE.md)")
print()

ordered = [v for v in variants if v in rows_by_variant]
if baseline not in ordered:
    print(f"WARN: baseline {baseline!r} not present, deltas omitted")
    base_data = None
else:
    base_data = rows_by_variant[baseline]

col_w = 12
hdr = f"{'metric':<{col_w}}"
for v in ordered:
    label = v[:13] if len(v) > 13 else v
    hdr += f"{label:>14}"
print(hdr)
print("-" * len(hdr))

for metric, short in key_metrics:
    line = f"{short:<{col_w}}"
    base_val = to_float(base_data.get(metric)) if base_data else None
    for v in ordered:
        val = rows_by_variant[v].get(metric, "-")
        n = to_float(val)
        if v == baseline or base_val is None or n is None:
            cell = val if val != "-" else "-"
        else:
            d = n - base_val
            if abs(base_val) > 1e-9 and abs(d / base_val) > 0.0001:
                pct = d / base_val * 100.0
                cell = f"{d:+.3g}({pct:+.1f}%)"
            else:
                cell = f"{d:+.3g}"
        cell = cell[:13]
        line += f"{cell:>14}"
    print(line)

print()
print("=== reading this table ===")
print(f"  Compare each non-{baseline} column to {baseline}.  Within the basin floor")
print("  (gflip_blkswap, blkx5/6/7, blk_qrt0/2/3), expect TIE-band metrics:")
print("  the gflip XOR=1 lever already saturates the cluster_tm_corr axis at")
print("  0.65 (vs 0.94 for dgsw), and the m-axis perturbation that distinguishes")
print("  basin members lives below ncu single-launch resolution.")
print()
print("  High-information rows:")
print(f"    {baseline} vs basin: ~600-770 cyc gap, expect long_sb ↓ + L2 hit% ↑ on basin")
print(f"    gflip_cidperm vs all: cluster permutation breaks SM→L2 contiguity,")
print(f"      expect L2 hit% ↓ + DRAM rd ↑ + cyc ↑ DECISIVE")
print()
print("  Cross-reference with project_w3x_n29420_basin.md:")
print("    if focused metrics ALSO TIE on the basin → the within-basin signal is")
print("    below counter-resolution, consistent with η²=0.0075 NEGLIGIBLE on wall.")
PY

log ""
log "--- artifacts ---"
log "  focused csvs:  $OUT/<variant>.csv"
[ "$DO_FULL" = 1 ] && log "  full reports:  $OUT/full_<variant>.ncu-rep"
log "  decoded diff:  $OUT/summary.txt"
log ""
log "  to dig into a specific variant's full report (if --full was set):"
log "    ncu --import $OUT/full_<name>.ncu-rep --page details | less -R"
log "  pairwise diff between two full reports:"
log "    python3 tools/ncu_diff.py $OUT/full_A.ncu-rep $OUT/full_B.ncu-rep"
