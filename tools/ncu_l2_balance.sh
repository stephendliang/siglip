#!/usr/bin/env bash
# ncu_l2_balance.sh — test the L2-bandwidth-balancing hypothesis for gflip.
#
# Hypothesis: gflip basin floor (default fc2_w3x dispatch, TD=54
# gflip_blkswap) compresses the per-launch cross-CTA cycle tail by
# ~4-5 µs vs dgsw (TD=8) because it decorrelates paired-CTA tm-traversal
# (cluster_tm_corr 0.99 -> 0.65), which spreads L2/DRAM traffic uniformly
# across slices/channels instead of bunching paired clusters onto the
# same cache slice at the same time.
#
# Distribution metrics where the hypothesis predicts gflip < dgsw:
#   - Per-SM cycle .max / .avg ratio   (tail measurement; closer to 1.0 is tighter)
#   - L2 sectors .max / .avg ratio     (per-slice bunching)
#   - DRAM bytes .max / .avg ratio     (per-channel bunching)
#   - L2 sectors-per-request           (queue depth proxy)
#   - long_scoreboard stall pct        (waiting on global mem)
#
# Metrics where gflip == dgsw (sanity check):
#   - Total L2 sectors .sum            (same kernel = same I/O)
#   - Total DRAM bytes .sum
#   - sm__pipe_tensor_cycles_active    (both MMA-bound)
#
# Verdict logic in the inline Python:
#   - All 5 distribution metrics tighter for gflip -> CONFIRMED
#   - 3-4 of 5 tighter -> SUPPORTED (mechanism likely correct)
#   - <=2 tighter      -> NOT SUPPORTED (hypothesis wrong, look elsewhere)
#
# Output: data/ncu_l2_balance_<ts>/
#   {gflip,dgsw}.csv          raw ncu output
#   {gflip,dgsw}.stderr       ncu errors
#   verdict.txt               head-to-head + hypothesis call
#   run.log                   build + invocation log
#
# Budget: ~3-5 min on B200.  Two ncu runs, ~30-60 s each (10 metric replays).
#
# Requires: ncu in PATH, perf-counter access (run with sudo if you get
# ERR_NVGPUCTRPERM).
#
# Usage:
#   bash tools/ncu_l2_balance.sh           # default 1 rep
#   bash tools/ncu_l2_balance.sh --reps 3  # variance check
set -uo pipefail
cd "$(dirname "$0")/.."

REPS=1
while [ $# -gt 0 ]; do
    case "$1" in
        --reps)    REPS="$2"; shift 2 ;;
        -h|--help) sed -n '1,42p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

command -v ncu >/dev/null || { echo "ncu not in PATH" >&2; exit 1; }

OUT="data/ncu_l2_balance_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
LOG="$OUT/run.log"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

build_variant() {
    local label="$1"; shift
    log "--- build $label ---"
    rm -f fc2-w3x-ncu
    make -B fc2-w3x-ncu "$@" 2>&1 | tee -a "$LOG" | tail -3
    local rc=${PIPESTATUS[0]}
    if [ "$rc" != 0 ] || [ ! -x fc2-w3x-ncu ]; then
        log "ERROR: $label build failed (make rc=$rc)"
        exit 1
    fi
    cp fc2-w3x-ncu "$OUT/fc2-w3x-$label"
}

log "=== fc2_w3x L2 bandwidth balance test -> $OUT ==="

# Build gflip variant (default = TD=54 gflip_blkswap per CLAUDE.md).
build_variant gflip

# Build dgsw variant (TD=8).
build_variant dgsw DFLAGS='-DTILE_DISPATCH=8'

# ── Distribution metrics: where gflip should differ from dgsw ──
# .avg / .max / .min families on SM and L2/DRAM partitions are the
# natural distribution probe — ncu aggregates across all instances of
# the unit and exposes summary statistics.
DIST_METRICS="$(cat <<'EOF' | tr -d ' \n'
sm__cycles_active.avg,
sm__cycles_active.max,
sm__cycles_active.min,
lts__t_sectors.avg,
lts__t_sectors.max,
lts__t_sectors.sum,
lts__t_sectors_op_read.avg,
lts__t_sectors_op_read.max,
lts__t_sectors_op_read.sum,
lts__t_sectors_op_write.avg,
lts__t_sectors_op_write.max,
lts__t_sectors_op_write.sum,
dram__bytes.avg,
dram__bytes.max,
dram__bytes.sum,
dram__bytes_read.avg,
dram__bytes_read.max,
dram__bytes_read.sum,
dram__bytes_write.avg,
dram__bytes_write.max,
dram__bytes_write.sum
EOF
)"

# ── Throughput / queue-depth metrics ──
# L2 throughput pct: uniform load -> sustained peak; bunched -> peak-then-idle
# -> lower avg pct.  sectors-per-request is a direct queue-depth proxy on
# the LSU port. lts__t_sector_hit_rate sanity-checks the L2 hit rate is
# unchanged (it's an aggregate, shouldn't shift on dispatch alone).
THRU_METRICS="$(cat <<'EOF' | tr -d ' \n'
lts__throughput.avg.pct_of_peak_sustained_elapsed,
dram__throughput.avg.pct_of_peak_sustained_elapsed,
lts__average_t_sectors_per_request_pipe_lsu_op_ld.ratio,
lts__average_t_sectors_per_request_pipe_lsu_op_st.ratio,
lts__t_sector_hit_rate.pct,
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,
sm__warps_active.avg.pct_of_peak_sustained_active
EOF
)"

# ── Stall reasons ──
# long_scoreboard = waiting on global memory (A/B-tensor read).
# If gflip reduces L2 contention, this should drop.
# membar / barrier are sanity (should not differ much across dispatch).
# Naming follows ncu_fc2_w3x.sh: smsp__average_warps_issue_stalled_<reason>
# _per_issue_active.ratio is the canonical per-issue stall fraction on
# sm_100a; the bare smsp__pcsamp_* form is incomplete and won't enumerate.
STALL_METRICS="$(cat <<'EOF' | tr -d ' \n'
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,
smsp__average_warps_issue_stalled_membar_per_issue_active.ratio,
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,
smsp__average_warps_issue_stalled_drain_per_issue_active.ratio
EOF
)"

ALL_METRICS="${DIST_METRICS},${THRU_METRICS},${STALL_METRICS}"

run_ncu() {
    local name="$1" suffix="$2"
    local bin="$OUT/fc2-w3x-$name"
    if [ ! -x "$bin" ]; then
        log "  SKIP $name$suffix: $bin missing"
        return 1
    fi
    log "  ncu $name$suffix"
    # NCU_PROFILE binary launches N_WARMUP=1 + N_TIMED_LAUNCHES=1 (see
    # fc2_w3x.cu:1666-1690). --launch-skip 1 targets the timed launch,
    # matching ncu_fc2_pipes.sh's convention.
    ncu -k "regex:fc2_w3x_kernel" \
        --launch-skip 1 --launch-count 1 \
        --metrics "$ALL_METRICS" \
        --csv \
        "$bin" > "$OUT/$name$suffix.csv" 2> "$OUT/$name$suffix.stderr" \
        || log "    WARN ncu nonzero for $name$suffix (see .stderr)"
}

log ""
log "--- ncu (2 variants × $REPS reps) ---"
for rep in $(seq 1 "$REPS"); do
    [ "$REPS" -gt 1 ] && log "  -- rep $rep/$REPS --"
    suffix=""
    [ "$REPS" -gt 1 ] && suffix="_r$rep"
    for name in gflip dgsw; do
        run_ncu "$name" "$suffix"
    done
done

log ""
log "--- decode + verdict ---"
python3 - "$OUT" "$REPS" <<'PY' | tee "$OUT/verdict.txt"
import csv, glob, os, statistics, sys

out, reps = sys.argv[1], int(sys.argv[2])

def load_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    data = {}
    with open(path) as fh:
        header = None
        for row in csv.reader(fh):
            if not row or row[0].startswith('=') or row[0].startswith('WARNING'):
                continue
            if row[0] == 'ID':
                header = row
                continue
            if header and len(row) == len(header):
                try:
                    mname = row[header.index('Metric Name')]
                    mval  = row[header.index('Metric Value')]
                except (ValueError, IndexError):
                    continue
                if mname:
                    try:
                        data[mname] = float(mval.replace(',', ''))
                    except ValueError:
                        pass
    return data

def load_variant(name):
    if reps == 1:
        return load_csv(os.path.join(out, f"{name}.csv"))
    # Median over reps for each metric.
    accum = {}
    for f in sorted(glob.glob(os.path.join(out, f"{name}_r*.csv"))):
        for k, v in load_csv(f).items():
            accum.setdefault(k, []).append(v)
    return {k: statistics.median(v) for k, v in accum.items() if v}

g = load_variant("gflip")
d = load_variant("dgsw")
if not g or not d:
    print("FAIL: no csv data — see .stderr files in the output dir")
    sys.exit(2)

def fmt(v):
    if v is None: return "?"
    a = abs(v)
    if a >= 1e9: return f"{v/1e9:.3f}G"
    if a >= 1e6: return f"{v/1e6:.3f}M"
    if a >= 1e3: return f"{v/1e3:.3f}k"
    return f"{v:.4f}"

def row(label, g_v, d_v, lower_better=True, pred=None):
    if g_v is None or d_v is None:
        return f"  {label:<60} {'(missing)':>12}"
    delta = (g_v - d_v) / d_v * 100 if d_v else 0
    win = "gflip" if (g_v < d_v) == lower_better else "dgsw"
    if abs(delta) < 0.1:
        win = "tie"
    pred_mark = ""
    if pred is not None:
        pred_mark = " ✓" if win == pred else (" ✗" if win != "tie" else " ~")
    return (f"  {label:<60} {fmt(g_v):>12} {fmt(d_v):>12} "
            f"{delta:>+7.2f}%  {win:>5}{pred_mark}")

print(f"{'='*100}")
print(f"  fc2_w3x L2 bandwidth balance test — gflip (default) vs dgsw (TD=8)")
print(f"  Results: {out}")
print(f"{'='*100}\n")

print(f"{'metric':<62} {'gflip':>12} {'dgsw':>12} {'Δ':>9}  {'win':>5}")
print(f"{'-'*100}")

# Distribution: max/avg ratios.  Closer to 1.0 = tighter = uniform
# distribution = hypothesis SUPPORTED if gflip's ratio is lower.
print("\n[Distribution — max/avg ratios; closer to 1.0 = tighter; lower wins]")
results = {}
n_dist_attempted = 0
def _ratio(d, mavg, mmax):
    a, x = d.get(mavg), d.get(mmax)
    if a in (None, 0) or x is None:
        return None
    return x / a
for label, mavg, mmax, key in [
    ("SM cycle tail (max/avg)",      "sm__cycles_active.avg",
                                      "sm__cycles_active.max",          "sm_cyc"),
    ("L2 sectors per slice (max/avg)", "lts__t_sectors.avg",
                                       "lts__t_sectors.max",            "l2_all"),
    ("L2 read sectors per slice (max/avg)", "lts__t_sectors_op_read.avg",
                                            "lts__t_sectors_op_read.max", "l2_rd"),
    ("L2 write sectors per slice (max/avg)", "lts__t_sectors_op_write.avg",
                                             "lts__t_sectors_op_write.max", "l2_wr"),
    ("DRAM bytes per channel (max/avg)", "dram__bytes.avg",
                                          "dram__bytes.max",            "dram"),
]:
    n_dist_attempted += 1
    g_r = _ratio(g, mavg, mmax)
    d_r = _ratio(d, mavg, mmax)
    if g_r is None or d_r is None:
        print(f"  {label:<60} {'(missing — metric unavailable on this arch)':>42}")
        continue
    win = "gflip" if g_r < d_r else ("tie" if abs(g_r-d_r) < 0.005 else "dgsw")
    mark = " ✓" if win == "gflip" else (" ~" if win == "tie" else " ✗")
    print(f"  {label:<60} {g_r:>12.4f} {d_r:>12.4f} "
          f"{(g_r-d_r):>+9.4f}  {win:>5}{mark}")
    results[key] = win

print("\n[Throughput / queue depth — gflip-favored:]")
for label, key, lower_better, pred in [
    ("L2 throughput (% of peak)",     "lts__throughput.avg.pct_of_peak_sustained_elapsed",  False, "gflip"),
    ("DRAM throughput (% of peak)",   "dram__throughput.avg.pct_of_peak_sustained_elapsed", False, "gflip"),
    ("L2 sectors/req (LSU loads)",    "lts__average_t_sectors_per_request_pipe_lsu_op_ld.ratio", True, "gflip"),
    ("L2 sectors/req (LSU stores)",   "lts__average_t_sectors_per_request_pipe_lsu_op_st.ratio", True, "gflip"),
    ("L2 hit rate (pct, sanity)",     "lts__t_sector_hit_rate.pct",                          False, None),
    ("Tensor pipe active (pct, sanity)", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", False, None),
    ("Warps active (pct)",            "sm__warps_active.avg.pct_of_peak_sustained_active",   False, None),
]:
    print(row(label, g.get(key), d.get(key), lower_better, pred))

STALL_KEY_LSB = "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
print("\n[Stall reasons — long_scoreboard predicted lower for gflip:]")
for label, key, pred in [
    ("Stall: long_scoreboard (global mem wait)",  STALL_KEY_LSB, "gflip"),
    ("Stall: short_scoreboard (shared/local wait)", "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio", None),
    ("Stall: membar",                              "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",           None),
    ("Stall: barrier (bar.sync)",                  "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",          None),
    ("Stall: lg_throttle",                         "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",      None),
    ("Stall: mio_throttle",                        "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",     None),
    ("Stall: drain",                               "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",            None),
]:
    print(row(label, g.get(key), d.get(key), True, pred))

print("\n[Sanity: total traffic should match within 0.5%]")
for label, key in [
    ("Total L2 sectors",        "lts__t_sectors.sum"),
    ("Total L2 read sectors",   "lts__t_sectors_op_read.sum"),
    ("Total L2 write sectors",  "lts__t_sectors_op_write.sum"),
    ("Total DRAM bytes",        "dram__bytes.sum"),
]:
    print(row(label, g.get(key), d.get(key), True, None))

# ── Final verdict ───────────────────────────────────────────────────────
print(f"\n{'='*100}")
n_dist_wins = sum(1 for v in results.values() if v == "gflip")
n_dist_resolved = len(results)
gflip_lsr = g.get(STALL_KEY_LSB)
dgsw_lsr  = d.get(STALL_KEY_LSB)
lsr_available = gflip_lsr is not None and dgsw_lsr is not None
lsr_drop = lsr_available and gflip_lsr < dgsw_lsr * 0.97

print(f"Hypothesis verdict:")
print(f"  Distribution wins (gflip tighter): {n_dist_wins}/{n_dist_resolved}"
      f" resolved (of {n_dist_attempted} attempted)")
if lsr_available:
    print(f"  long_scoreboard stall lower for gflip: {lsr_drop} "
          f"(g={gflip_lsr:.4f} d={dgsw_lsr:.4f})")
else:
    print(f"  long_scoreboard stall: METRIC UNAVAILABLE — verdict cannot use it")
total_traffic_match = True
for k in ["lts__t_sectors.sum", "dram__bytes.sum"]:
    if g.get(k) and d.get(k):
        rel = abs(g[k] - d[k]) / d[k]
        if rel > 0.005:
            total_traffic_match = False
print(f"  Total-traffic sanity (kernels do same I/O within 0.5%): {total_traffic_match}")

if n_dist_resolved < 3:
    print(f"\n  ⚠  Only {n_dist_resolved}/{n_dist_attempted} distribution metrics")
    print(f"     resolved. Most rows missing — likely an ncu metric-naming")
    print(f"     issue on this arch. Check .stderr files and run")
    print(f"     'ncu --query-metrics ./fc2-w3x-ncu | grep -E \"lts__|dram__|sm__cycles\"'")
    print(f"     to find the correct paths, then patch the script.")
elif not total_traffic_match:
    print(f"\n  ⚠  TOTAL TRAFFIC DIFFERS by >0.5% — kernels aren't doing the")
    print(f"     same work. Verdict invalid; check that both binaries built")
    print(f"     against the same source state.")
elif n_dist_wins >= 4 and lsr_drop:
    print(f"\n  ★  L2-bandwidth-balancing hypothesis CONFIRMED.")
    print(f"     gflip reduces per-instance traffic spread AND reduces")
    print(f"     long_scoreboard stalls. Mechanism: paired CTAs decorrelate")
    print(f"     temporal cache-slice access, queue depth drops.")
    print(f"     Next: design L2-set-aware swizzle directly targeting bank")
    print(f"     distribution — see project_w3x_n29420_basin.md for the")
    print(f"     m-axis basin floor; the L2-aware design space hasn't been")
    print(f"     searched yet.  Realistic recoverable: 1-2 µs.")
elif n_dist_wins >= 4 and not lsr_drop:
    print(f"\n  ◆  STRUCTURAL win without stall confirmation.")
    print(f"     gflip wins {n_dist_wins}/{n_dist_resolved} distribution metrics, but")
    print(f"     long_scoreboard does NOT drop (or wasn't measurable). Either:")
    print(f"     - The bunching IS in cache lines, but the wait is hidden")
    print(f"       in the MMA shadow (W5 100% pipe-active masks W0/W4 stalls).")
    print(f"     - Stall metric unavailable — interpret structural wins alone.")
    print(f"     Carefully proceed with L2-aware design, but re-confirm the")
    print(f"     mechanism with a per-warp PROFILE_W5 dispatch comparison.")
elif n_dist_wins == 3:
    print(f"\n  ◇  Hypothesis PARTIALLY SUPPORTED.  gflip differentiates on")
    print(f"     3 of {n_dist_resolved} distribution metrics. The mechanism")
    print(f"     is L2-bandwidth-related but the strongest signal isn't in")
    print(f"     every dimension.  Read the per-row marks above to see")
    print(f"     WHICH dimension (SM cycle / L2 slice / DRAM channel)")
    print(f"     carries the bunching signature.")
elif n_dist_wins <= 1:
    print(f"\n  ✗  Hypothesis NOT SUPPORTED.  gflip's tail compression is")
    print(f"     NOT showing up as L2/DRAM/SM distribution differences in")
    print(f"     ncu's aggregate metrics.  Possible explanations:")
    print(f"     1. Mechanism is sub-window — ncu averages over the whole")
    print(f"        kernel run; the bunching is per-tile and washes out.")
    print(f"     2. Mechanism is at a level ncu doesn't expose well")
    print(f"        (e.g., crossbar arbitration, mbarrier wait queues).")
    print(f"     3. clusters.log spread differential isn't actually L2-")
    print(f"        bandwidth-related — could be cluster_bar.sync wait or")
    print(f"        something else. Need a different test.")
    print(f"     Action: don't pursue L2-aware swizzle redesign; revisit")
    print(f"     the mechanism question with a finer-grained probe.")
else:
    print(f"\n  ?  AMBIGUOUS.  {n_dist_wins}/{n_dist_resolved} distribution")
    print(f"     metrics differentiate, long_scoreboard "
          f"{'drops' if lsr_drop else ('unchanged' if lsr_available else 'unmeasured')}.")
    print(f"     Re-run with --reps 3 for variance, then re-read.")
print(f"{'='*100}")

if not g or not d or n_dist_resolved < 3:
    sys.exit(2)
PY

PY_RC=${PIPESTATUS[0]}

log ""
if [ "$PY_RC" = 0 ]; then
    log "Done. Verdict: $OUT/verdict.txt"
else
    log "Decoder exit=$PY_RC — see $OUT/verdict.txt and .stderr files"
    exit "$PY_RC"
fi
