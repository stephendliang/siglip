#!/usr/bin/env bash
# ncu_l2_basin_sweep.sh — sweep the n=29420 basin floor + a catastrophic
# control through the same ncu distribution metrics as ncu_l2_balance.sh
# to answer two questions the 2-variant probe couldn't:
#
#   1. Sensitivity: does gflip_cidperm (TD=55, +1718 cyc DECISIVE regression
#      from `c'=(c*15)%74` cluster perm that breaks SM->L2 contiguity) show
#      pathological L2 distribution? If yes, the metric IS sensitive to L2
#      contiguity at all — and the basin's near-tie with dgsw means basin
#      really doesn't use L2 as its lever. If no, the metric is blind at
#      this granularity and the original verdict was unfalsifiable.
#
#   2. Within-basin spread: do the 7 tied basin cells (TD=54, 87, 88, 89,
#      91, 92, 93) cluster together on the L2 metrics, or does TD=54
#      (blkswap, which we tested in ncu_l2_balance) look like an outlier?
#
# Variants (9 total):
#   - TD=8   dgsw            (control, off-basin)
#   - TD=54  gflip_blkswap   (basin floor, current default)
#   - TD=87  gflip_blkx5     (basin floor, alt-xor variant)
#   - TD=88  gflip_blkx6     (basin floor, alt-xor variant)
#   - TD=89  gflip_blkx7     (basin floor, alt-xor variant)
#   - TD=91  gflip_blk_qrt0  (basin floor, qrt-density variant)
#   - TD=92  gflip_blk_qrt2  (basin floor, qrt-density variant)
#   - TD=93  gflip_blk_qrt3  (basin floor, qrt-density variant)
#   - TD=55  gflip_cidperm   (catastrophic, +1718 cyc DECISIVE, sensitivity)
#
# Budget: ~8-10 min on B200. 9 builds (~30s each) + 9 ncu runs (~30s each
# at --launch-skip 1 --launch-count 1, no replay).
#
# Output: data/ncu_l2_basin_sweep_<ts>/
#   {label}.csv          per-variant raw ncu output
#   {label}.stderr       per-variant ncu errors
#   fc2-w3x-ncu-{label}  saved binary per variant (so re-decode can rerun)
#   verdict.txt          cross-variant table + sensitivity / spread call
#   run.log              build + invocation log
#
# Requires: ncu in PATH, perf-counter access.
#
# Usage:
#   bash tools/ncu_l2_basin_sweep.sh                # 1 rep per variant
#   bash tools/ncu_l2_basin_sweep.sh --reps 3       # variance across reps

set -uo pipefail
cd "$(dirname "$0")/.."

REPS=1
while [ $# -gt 0 ]; do
    case "$1" in
        --reps)    REPS="$2"; shift 2 ;;
        -h|--help) sed -n '1,49p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

command -v ncu >/dev/null || { echo "ncu not in PATH" >&2; exit 1; }

OUT="data/ncu_l2_basin_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
LOG="$OUT/run.log"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

# Variant definitions: label  TD-id
#                      ─────  ─────
VARIANTS=(
    "blkswap:54"
    "blkx5:87"
    "blkx6:88"
    "blkx7:89"
    "blk_qrt0:91"
    "blk_qrt2:92"
    "blk_qrt3:93"
    "cidperm:55"
    "dgsw:8"
)

log "=== fc2_w3x L2 basin sweep -> $OUT ==="
log "9 variants × $REPS rep(s); ~$(( (9 * (30 + 30)) / 60 )) min estimated."

build_variant() {
    local label="$1" td="$2"
    log "--- build $label (TD=$td) ---"
    rm -f fc2-w3x-ncu
    make -B fc2-w3x-ncu DFLAGS="-DTILE_DISPATCH=$td" 2>&1 | tee -a "$LOG" | tail -3
    local rc=${PIPESTATUS[0]}
    if [ "$rc" != 0 ] || [ ! -x fc2-w3x-ncu ]; then
        log "ERROR: $label build failed (make rc=$rc)"
        exit 1
    fi
    cp fc2-w3x-ncu "$OUT/fc2-w3x-ncu-$label"
}

for entry in "${VARIANTS[@]}"; do
    label="${entry%%:*}"
    td="${entry##*:}"
    build_variant "$label" "$td"
done

# Metrics — same definitions as ncu_l2_balance.sh so results are directly
# comparable. Three groups: distribution (per-instance .max/.avg), throughput
# / queue depth, stall reasons.
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

THRU_METRICS="$(cat <<'EOF' | tr -d ' \n'
lts__throughput.avg.pct_of_peak_sustained_elapsed,
dram__throughput.avg.pct_of_peak_sustained_elapsed,
lts__t_sector_hit_rate.pct,
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,
sm__warps_active.avg.pct_of_peak_sustained_active
EOF
)"

STALL_METRICS="$(cat <<'EOF' | tr -d ' \n'
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
EOF
)"

ALL_METRICS="${DIST_METRICS},${THRU_METRICS},${STALL_METRICS}"

run_ncu() {
    local name="$1" suffix="$2"
    local bin="$OUT/fc2-w3x-ncu-$name"
    if [ ! -x "$bin" ]; then
        log "  SKIP $name$suffix: $bin missing"
        return 1
    fi
    log "  ncu $name$suffix"
    ncu -k "regex:fc2_w3x_kernel" \
        --launch-skip 1 --launch-count 1 \
        --metrics "$ALL_METRICS" \
        --csv \
        "$bin" > "$OUT/$name$suffix.csv" 2> "$OUT/$name$suffix.stderr" \
        || log "    WARN ncu nonzero for $name$suffix (see .stderr)"
}

log ""
log "--- ncu (9 variants × $REPS reps) ---"
for rep in $(seq 1 "$REPS"); do
    [ "$REPS" -gt 1 ] && log "  -- rep $rep/$REPS --"
    suffix=""
    [ "$REPS" -gt 1 ] && suffix="_r$rep"
    for entry in "${VARIANTS[@]}"; do
        label="${entry%%:*}"
        run_ncu "$label" "$suffix"
    done
done

log ""
log "--- decode + cross-variant verdict ---"
python3 - "$OUT" "$REPS" <<'PY' | tee "$OUT/verdict.txt"
import csv, glob, os, statistics, sys

out, reps = sys.argv[1], int(sys.argv[2])

# Order matters for the printed table — basin cells together, then
# cidperm (the sensitivity probe), then dgsw (control).
VARIANT_ORDER = [
    ("blkswap",   54, "basin"),
    ("blkx5",     87, "basin"),
    ("blkx6",     88, "basin"),
    ("blkx7",     89, "basin"),
    ("blk_qrt0",  91, "basin"),
    ("blk_qrt2",  92, "basin"),
    ("blk_qrt3",  93, "basin"),
    ("cidperm",   55, "CATASTROPHIC"),
    ("dgsw",       8, "control"),
]

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
    accum = {}
    for f in sorted(glob.glob(os.path.join(out, f"{name}_r*.csv"))):
        for k, v in load_csv(f).items():
            accum.setdefault(k, []).append(v)
    return {k: statistics.median(v) for k, v in accum.items() if v}

data = {}
for name, td, tier in VARIANT_ORDER:
    data[name] = load_variant(name)
    if not data[name]:
        print(f"WARN: no csv data for {name} — see {name}.stderr", file=sys.stderr)

# any totally empty variant kills the verdict
any_empty = [n for n, _, _ in VARIANT_ORDER if not data.get(n)]
if any_empty:
    print(f"FAIL: missing csv data for: {', '.join(any_empty)}")
    sys.exit(2)

def fmt(v):
    if v is None: return "?"
    a = abs(v)
    if a >= 1e9: return f"{v/1e9:.2f}G"
    if a >= 1e6: return f"{v/1e6:.2f}M"
    if a >= 1e3: return f"{v/1e3:.1f}k"
    return f"{v:.3f}"

def ratio(d, mavg, mmax):
    a, x = d.get(mavg), d.get(mmax)
    if a in (None, 0) or x is None:
        return None
    return x / a

# Column widths
HEAD = ["blkswap", "blkx5", "blkx6", "blkx7", "blk_qrt0", "blk_qrt2", "blk_qrt3", "cidperm", "dgsw"]
COL = 10

print("=" * (40 + COL * 9))
print(f"  fc2_w3x L2 basin sweep — 7 basin cells + cidperm (sensitivity) + dgsw (control)")
print(f"  Results: {out}")
print("=" * (40 + COL * 9))

# Header rows
title = f"  {'metric':<38}"
for h in HEAD:
    title += f"{h:>{COL}}"
print(title)
tier_row = f"  {'tier →':<38}"
for h in HEAD:
    tier = next(t for n, _, t in VARIANT_ORDER if n == h)
    tier_label = {"basin": "BASIN", "CATASTROPHIC": "CATAST", "control": "ctrl"}[tier]
    tier_row += f"{tier_label:>{COL}}"
print(tier_row)
print("-" * (40 + COL * 9))

def cells_for(get_value):
    """get_value(name)->float|None; emit 9 right-aligned cells."""
    cells = []
    for h in HEAD:
        v = get_value(h)
        cells.append((f"{fmt(v):>{COL}}") if v is not None else f"{'?':>{COL}}")
    return "".join(cells)

# DISTRIBUTION: max/avg ratios (closer to 1.0 = tighter)
print("\n[Distribution — max/avg ratio; closer to 1.0 = tighter. Sensitivity check: cidperm should be much higher if metric works.]")
DIST_ROWS = [
    ("SM cycle tail (max/avg)",          "sm__cycles_active.avg",        "sm__cycles_active.max"),
    ("L2 sectors/slice (max/avg)",       "lts__t_sectors.avg",            "lts__t_sectors.max"),
    ("L2 read sectors/slice (max/avg)",  "lts__t_sectors_op_read.avg",    "lts__t_sectors_op_read.max"),
    ("L2 write sectors/slice (max/avg)", "lts__t_sectors_op_write.avg",   "lts__t_sectors_op_write.max"),
    ("DRAM bytes/channel (max/avg)",     "dram__bytes.avg",                "dram__bytes.max"),
    ("DRAM read bytes/channel (max/avg)","dram__bytes_read.avg",           "dram__bytes_read.max"),
]
dist_values = {}  # name -> {row_label: ratio}
for label, mavg, mmax in DIST_ROWS:
    row_vals = {h: ratio(data[h], mavg, mmax) for h in HEAD}
    dist_values[label] = row_vals
    line = f"  {label:<38}"
    for h in HEAD:
        v = row_vals[h]
        line += f"{(f'{v:.4f}' if v is not None else '?'):>{COL}}"
    print(line)

# THROUGHPUT
print("\n[Throughput / hit / pipe — context for the distribution rows.]")
for label, key in [
    ("L2 throughput (% peak)",     "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("DRAM throughput (% peak)",   "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("L2 hit rate (%)",            "lts__t_sector_hit_rate.pct"),
    ("Tensor pipe active (%)",     "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"),
    ("Warps active (%)",           "sm__warps_active.avg.pct_of_peak_sustained_active"),
]:
    print(f"  {label:<38}" + cells_for(lambda h, k=key: data[h].get(k)))

# STALLS
print("\n[Stall reasons — per-issue-active ratios.]")
for label, key in [
    ("Stall: long_scoreboard",     "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"),
    ("Stall: short_scoreboard",    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"),
    ("Stall: barrier (bar.sync)",  "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"),
]:
    print(f"  {label:<38}" + cells_for(lambda h, k=key: data[h].get(k)))

# SANITY — totals across variants should match within 0.5% (same problem,
# only schedule changes). If they don't, the kernel is doing different
# total work and the comparison is invalid.
print("\n[Sanity — totals should match across all variants within 0.5%.]")
for label, key in [
    ("Total L2 sectors",       "lts__t_sectors.sum"),
    ("Total L2 read sectors",  "lts__t_sectors_op_read.sum"),
    ("Total L2 write sectors", "lts__t_sectors_op_write.sum"),
    ("Total DRAM bytes",       "dram__bytes.sum"),
]:
    print(f"  {label:<38}" + cells_for(lambda h, k=key: data[h].get(k)))

# ── Cross-variant verdict ─────────────────────────────────────────────────
print("\n" + "=" * (40 + COL * 9))
print("Cross-variant verdict:\n")

# Sanity: do all 9 do the same total work?
def total_match():
    vals = [data[h].get("lts__t_sectors.sum") for h in HEAD]
    if any(v is None for v in vals):
        return None
    mn, mx = min(vals), max(vals)
    return (mx - mn) / mn < 0.005

sane = total_match()
if sane is False:
    print("  ⚠  TOTAL TRAFFIC DIFFERS across variants by >0.5%.")
    print("     The 9 binaries aren't computing the same thing. Halt — check")
    print("     that all builds went through with the same source state.")
elif sane is None:
    print("  ⚠  Total-traffic sanity unknown (some variants missing metric).")
else:
    print("  ✓ Total-traffic sanity OK (all variants within 0.5%).")

# Sensitivity: is cidperm's L2-distribution pathologically worse than the
# basin cells'?  If yes → metric is sensitive; basin's near-tie with dgsw
# is therefore evidence that basin doesn't use L2 as its lever.
print()
sens_signals = 0
sens_total = 0
for label, _, _ in DIST_ROWS:
    basin_vals = [dist_values[label][n] for n, _, t in VARIANT_ORDER
                  if t == "basin" and dist_values[label][n] is not None]
    cid = dist_values[label]["cidperm"]
    dgs = dist_values[label]["dgsw"]
    if not basin_vals or cid is None:
        continue
    basin_max = max(basin_vals)
    basin_med = statistics.median(basin_vals)
    sens_total += 1
    cid_excess_vs_basin = (cid - basin_max) / basin_med if basin_med else 0
    # >5% excess over the worst basin cell on this metric = pathological
    pathological = cid_excess_vs_basin > 0.05
    if pathological:
        sens_signals += 1
        tag = "PATHO"
    else:
        tag = "in-band"
    dgs_str = f"{dgs:.4f}" if dgs is not None else "?"
    print(f"    {label:<38} cid={cid:.4f}  basin_max={basin_max:.4f}  dgsw={dgs_str}  excess={cid_excess_vs_basin:+.2%} [{tag}]")

print()
sens_call = sens_signals >= 3  # 3 of 6 distribution metrics flag cidperm

# Within-basin spread: do basin cells cluster together on L2 distribution?
print("  Within-basin spread (basin_max - basin_min, ideally tight):")
for label, _, _ in DIST_ROWS:
    bv = [dist_values[label][n] for n, _, t in VARIANT_ORDER if t == "basin"
          and dist_values[label][n] is not None]
    if not bv:
        continue
    spread = max(bv) - min(bv)
    med = statistics.median(bv)
    rel = spread / med * 100 if med else 0
    flag = "tight" if rel < 1.0 else ("loose" if rel < 5.0 else "WIDE")
    print(f"    {label:<38} spread={spread:.4f} ({rel:+.2f}%)  [{flag}]")

print()
print("=" * (40 + COL * 9))
print("Verdict:")
print()
if sens_call:
    print(f"  ★  L2 distribution metric IS SENSITIVE: cidperm is pathological")
    print(f"     on {sens_signals}/{sens_total} distribution rows, well outside the")
    print(f"     basin cells' spread. Therefore the basin's near-tie with dgsw")
    print(f"     in ncu_l2_balance is REAL evidence that basin doesn't use L2.")
    print(f"     gflip's basin lever is something the L2/DRAM distribution")
    print(f"     metrics don't see — most likely the cluster-pair tm-traversal")
    print(f"     correlation documented in project_w3x_n29420_basin.md")
    print(f"     (cluster_tm_corr 0.94→0.65), not L2-bandwidth balance.")
    print(f"     ACTION: kill the L2-aware-swizzle direction; the 1-2 µs")
    print(f"     recoverable in CLAUDE.md's compute-floor section must come")
    print(f"     from somewhere else.")
else:
    print(f"  ?  L2 distribution metric INSENSITIVE: cidperm shows pathology")
    print(f"     on only {sens_signals}/{sens_total} distribution rows. Either")
    print(f"     cidperm's +1718 cyc damage doesn't show up in ncu's spatial")
    print(f"     distribution aggregates, or the metric averages out the")
    print(f"     temporal bunching. The basin's near-tie with dgsw in the")
    print(f"     original L2-balance probe is therefore UNFALSIFIABLE on this")
    print(f"     instrument. Need a different probe — per-CTA L2 hit rate")
    print(f"     traces, or a clock64-based paired-CTA tm coord correlation")
    print(f"     dump. ACTION: don't kill the hypothesis yet; design a")
    print(f"     sharper probe.")

print("=" * (40 + COL * 9))

if any_empty:
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
