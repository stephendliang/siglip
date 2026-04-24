#!/usr/bin/env bash
# ncu_fc2_pipes.sh — pipe-breakdown diagnosis for fc2_w3x vs rank-1.
#
# Purpose: of the ~140M (w3x) vs ~109M (rank-1) total-instruction gap,
# figure out WHICH PIPE the extra work lives in. That tells us whether
# the instruction-count axis is worth pursuing as a perf lever.
#
# Why a new script (vs ncu_fc2_w3x.sh --max):
# --set full deadlocks on replay ~36/38 on our kernel (tcgen05.mma +
# cta_group::2 + mbarrier interacts badly with certain sm_100a metric
# groups; replay state corruption). This script uses a curated 24-metric
# pack sized to avoid the deadlock while still answering the pipe
# question. No --set full, no --import-source.
#
# Metric groups (all focused, ~24 replays total):
#   G1  Pipe-active cycles     — where is each pipe busy      ( 6 metrics)
#   G2  Per-pipe inst counts   — where are insts issued       ( 8 metrics)
#   G3  SASS op distribution   — LDSM/STSM/FFMA/IMAD/etc.     ( 8 metrics)
#   G4  Sanity (warps_act, tensor%, total inst, cycles)       ( 4 metrics)
#
# Flags:
#   (none)     single-pack run (24 metrics in one ncu invocation).  ~90 s.
#   --bisect   run G1, G2, G3, G4 in separate ncu invocations.
#              Use if the single pack deadlocks — one of the four
#              groups will complete cleanly and partially answer.
#   --reps N   run N times for variance (as _r1, _r2, ...).
#
# Output: data/ncu_fc2_pipes_<ts>/
#   {w3x,rank1}[_gN][_rN].csv   raw ncu csv
#   pipe_diff.txt               decoded table (pipe deltas sorted)
#   run.log                     invocation trace
#
# Requires: sudo (CAP_SYS_ADMIN for perf counters on B200).
#
# Reading the output:
#   If LSU_sum(w3x) - LSU_sum(rank1) is the largest positive delta →
#     the extra work is STS/LDS — register-packed epilogue is the lever.
#   If FMA + ALU are largest →
#     the extra work is epi compute / address math — UR-based desc
#     advance or fewer CVT passes.
#   If Uniform is largest →
#     rank-1 is doing more work on the uniform datapath — lift
#     per-CTA scalars to UR.
#   If pipes match within ~5% →
#     the 31M gap is ncu counting W5-CTA1 early-exit differently;
#     no real lever exists on this axis. Ship 1.007 and move on.

set -u
cd "$(dirname "$0")/.."

OUT="data/ncu_fc2_pipes_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
LOG="$OUT/run.log"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

BISECT=0
REPS=1
while [ $# -gt 0 ]; do
    case "$1" in
        --bisect) BISECT=1; shift ;;
        --reps)   REPS="$2"; shift 2 ;;
        -h|--help) sed -n '1,50p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

command -v ncu >/dev/null || { echo "ncu not in PATH" >&2; exit 1; }

log "=== fc2_w3x pipe breakdown → $OUT ==="

log "--- build ---"
make fc2-w3x-ncu cublaslt-fc2 2>&1 | tee -a "$LOG" | tail -4

TARGETS=(
    "w3x    ./fc2-w3x-ncu    fc2_w3x_kernel     1"
    "rank1  ./cublaslt-fc2   nvjet_sm100_qqtst  2"
)

# ── G1: pipe cycle utilization (which pipe is active how often) ───────────
# These are "pct of peak sustained active" so they're directly comparable
# across kernels regardless of absolute inst count. tensor_pct is our "are
# we at the ceiling?" gauge; fma/alu/lsu tell us where non-tensor time goes.
G1_METRICS="$(cat <<'EOF' | tr -d ' \n'
sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active,
sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active,
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,
sm__inst_executed_pipe_tex.avg.pct_of_peak_sustained_active,
sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active
EOF
)"

# ── G2: per-pipe instruction counts (absolute, for delta computation) ─────
# The .sum form gives us raw emitted-instruction counts per pipe. Absolute
# w3x - rank1 deltas here are the load-bearing numbers. NOTE: not all
# pipe-specific .sum metrics exist on every arch — if any of these IMA
# (deadlock), drop to --bisect and G2 will skip but G1/G3/G4 still run.
G2_METRICS="$(cat <<'EOF' | tr -d ' \n'
smsp__inst_executed_pipe_alu.sum,
smsp__inst_executed_pipe_fma.sum,
smsp__inst_executed_pipe_fp64.sum,
smsp__inst_executed_pipe_tensor.sum,
smsp__inst_executed_pipe_lsu.sum,
smsp__inst_executed_pipe_tex.sum,
smsp__inst_executed_pipe_xu.sum,
smsp__inst_executed_pipe_uniform.sum
EOF
)"

# ── G3: SASS op distribution (what kind of inst, not just which pipe) ─────
# If LSU pipe differs, is it LDSM (ldmatrix), STSM (stmatrix), generic
# STS/LDS, or global LDG/STG?  Op-class counts answer that.
G3_METRICS="$(cat <<'EOF' | tr -d ' \n'
smsp__sass_inst_executed_op_ldsm.sum,
smsp__sass_inst_executed_op_stsm.sum,
smsp__sass_thread_inst_executed_op_memory_shared_ld.sum,
smsp__sass_thread_inst_executed_op_memory_shared_st.sum,
smsp__sass_thread_inst_executed_op_memory_global_ld.sum,
smsp__sass_thread_inst_executed_op_memory_global_st.sum,
smsp__sass_inst_executed_op_branch.sum,
smsp__sass_inst_executed_op_uniform.sum
EOF
)"

# ── G4: sanity checks (total inst, cycles, warps_active, tensor_pct) ──────
G4_METRICS="$(cat <<'EOF' | tr -d ' \n'
smsp__inst_executed.sum,
sm__cycles_elapsed.avg,
sm__warps_active.avg.pct_of_peak_sustained_active,
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
EOF
)"

ALL_METRICS="${G1_METRICS},${G2_METRICS},${G3_METRICS},${G4_METRICS}"

run_ncu() {
    local name="$1" bin="$2" filt="$3" lskip="$4" metrics="$5" suffix="$6"
    if [ ! -x "$bin" ]; then
        log "  SKIP $name$suffix: $bin missing"
        return 1
    fi
    log "  ncu $name$suffix"
    ncu --target-processes all \
        -k "regex:$filt" \
        --launch-skip "$lskip" --launch-count 1 \
        --metrics "$metrics" \
        --csv \
        "$bin" > "$OUT/$name$suffix.csv" 2> "$OUT/$name$suffix.stderr" \
        || log "    WARN ncu nonzero for $name$suffix (see .stderr)"
}

if [ "$BISECT" = 0 ]; then
    log ""
    log "--- single-pack run (24 metrics × 2 variants × $REPS reps) ---"
    for rep in $(seq 1 "$REPS"); do
        [ "$REPS" -gt 1 ] && log "  -- rep $rep/$REPS --"
        for entry in "${TARGETS[@]}"; do
            read -r name bin filt lskip <<<"$entry"
            suffix=""
            [ "$REPS" -gt 1 ] && suffix="_r$rep"
            run_ncu "$name" "$bin" "$filt" "$lskip" "$ALL_METRICS" "$suffix"
        done
    done
else
    log ""
    log "--- bisect mode: G1, G2, G3, G4 in separate runs ---"
    for rep in $(seq 1 "$REPS"); do
        [ "$REPS" -gt 1 ] && log "  -- rep $rep/$REPS --"
        for gidx in 1 2 3 4; do
            case "$gidx" in
                1) M="$G1_METRICS" ;;
                2) M="$G2_METRICS" ;;
                3) M="$G3_METRICS" ;;
                4) M="$G4_METRICS" ;;
            esac
            for entry in "${TARGETS[@]}"; do
                read -r name bin filt lskip <<<"$entry"
                suffix="_g$gidx"
                [ "$REPS" -gt 1 ] && suffix="${suffix}_r$rep"
                run_ncu "$name" "$bin" "$filt" "$lskip" "$M" "$suffix"
            done
        done
    done
fi

log ""
log "--- decode: pipe delta table ---"
python3 - "$OUT" "$BISECT" <<'PY' | tee "$OUT/pipe_diff.txt"
import csv, os, sys, glob, re

out = sys.argv[1]
bisect = int(sys.argv[2])

def load_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    data = {}
    with open(path) as fh:
        reader = csv.reader(fh)
        header = None
        for row in reader:
            if not row or row[0].startswith('=') or row[0].startswith('WARNING'):
                continue
            if row[0] == 'ID':
                header = row
                continue
            if header and len(row) == len(header):
                try:
                    mname = row[header.index('Metric Name')]
                    mval  = row[header.index('Metric Value')]
                    munit = row[header.index('Metric Unit')] if 'Metric Unit' in header else ''
                except (ValueError, IndexError):
                    continue
                if mname:
                    data[mname] = (mval, munit)
    return data

def merge_bisect(name):
    merged = {}
    for g in (1, 2, 3, 4):
        for f in sorted(glob.glob(os.path.join(out, f"{name}_g{g}*.csv"))):
            merged.update(load_csv(f))
    return merged

def load_variant(name):
    if bisect:
        return merge_bisect(name)
    # single-pack: may have _r1, _r2, ... — first rep wins for simplicity
    for f in sorted(glob.glob(os.path.join(out, f"{name}.csv")) +
                    sorted(glob.glob(os.path.join(out, f"{name}_r*.csv")))):
        d = load_csv(f)
        if d:
            return d
    return {}

w3x   = load_variant("w3x")
rank1 = load_variant("rank1")

if not w3x and not rank1:
    print("(no csv data — check run.log / .stderr files)")
    sys.exit(0)

def to_num(entry):
    if not entry:
        return None
    val = entry[0] if isinstance(entry, tuple) else entry
    try:
        return float(val.replace(",", ""))
    except (ValueError, AttributeError):
        return None

def scale(entry):
    if not isinstance(entry, tuple):
        return None
    val, unit = entry
    try:
        n = float(val.replace(",", ""))
    except (ValueError, AttributeError):
        return None
    mult = {"Ginst": 1e9, "Minst": 1e6, "Kinst": 1e3, "inst": 1,
            "Gcycle": 1e9, "Mcycle": 1e6, "Kcycle": 1e3, "cycle": 1}.get(unit, 1.0)
    return n * mult

# ── table 1: pipe cycle utilization ────────────────────────────────────────
print("=" * 72)
print("G1: pipe CYCLE utilization (% of peak) — where each pipe is busy")
print("=" * 72)
pipes_pct = [
    ("tensor", "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active"),
    ("fma",    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active"),
    ("alu",    "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active"),
    ("lsu",    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active"),
    ("tex",    "sm__inst_executed_pipe_tex.avg.pct_of_peak_sustained_active"),
    ("xu",     "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active"),
]
print(f"  {'pipe':<10} {'w3x':>10} {'rank1':>10} {'delta':>10}")
print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for short, full in pipes_pct:
    a = to_num(w3x.get(full))
    b = to_num(rank1.get(full))
    if a is None and b is None:
        continue
    da = f"{a:.2f}" if a is not None else "-"
    db = f"{b:.2f}" if b is not None else "-"
    dd = f"{a-b:+.2f}" if (a is not None and b is not None) else "-"
    print(f"  {short:<10} {da:>10} {db:>10} {dd:>10}")
print()

# ── table 2: per-pipe absolute inst count (the load-bearing table) ─────────
print("=" * 72)
print("G2: per-pipe INSTRUCTION COUNTS (absolute, millions)")
print("    This is the key table — largest positive delta = the axis to chase")
print("=" * 72)
pipes_sum = [
    ("alu",     "smsp__inst_executed_pipe_alu.sum"),
    ("fma",     "smsp__inst_executed_pipe_fma.sum"),
    ("fp64",    "smsp__inst_executed_pipe_fp64.sum"),
    ("tensor",  "smsp__inst_executed_pipe_tensor.sum"),
    ("lsu",     "smsp__inst_executed_pipe_lsu.sum"),
    ("tex",     "smsp__inst_executed_pipe_tex.sum"),
    ("xu",     "smsp__inst_executed_pipe_xu.sum"),
    ("uniform", "smsp__inst_executed_pipe_uniform.sum"),
]
rows = []
for short, full in pipes_sum:
    a = scale(w3x.get(full))
    b = scale(rank1.get(full))
    if a is None and b is None:
        continue
    da_M = (a / 1e6) if a is not None else None
    db_M = (b / 1e6) if b is not None else None
    dd_M = (da_M - db_M) if (da_M is not None and db_M is not None) else None
    rows.append((short, da_M, db_M, dd_M))

rows.sort(key=lambda r: -abs(r[3]) if r[3] is not None else 0)
print(f"  {'pipe':<10} {'w3x (M)':>12} {'rank1 (M)':>12} {'delta (M)':>12}  (sorted by |delta|)")
print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
for short, a, b, d in rows:
    sa = f"{a:,.2f}" if a is not None else "-"
    sb = f"{b:,.2f}" if b is not None else "-"
    sd = f"{d:+,.2f}" if d is not None else "-"
    print(f"  {short:<10} {sa:>12} {sb:>12} {sd:>12}")
print()

# ── table 3: SASS op distribution ─────────────────────────────────────────
print("=" * 72)
print("G3: SASS OP-CLASS counts (absolute, millions)")
print("=" * 72)
ops = [
    ("ldsm (ldmatrix)",   "smsp__sass_inst_executed_op_ldsm.sum"),
    ("stsm (stmatrix)",   "smsp__sass_inst_executed_op_stsm.sum"),
    ("shared_ld (LDS)",   "smsp__sass_thread_inst_executed_op_memory_shared_ld.sum"),
    ("shared_st (STS)",   "smsp__sass_thread_inst_executed_op_memory_shared_st.sum"),
    ("global_ld (LDG)",   "smsp__sass_thread_inst_executed_op_memory_global_ld.sum"),
    ("global_st (STG)",   "smsp__sass_thread_inst_executed_op_memory_global_st.sum"),
    ("branch",            "smsp__sass_inst_executed_op_branch.sum"),
    ("uniform",           "smsp__sass_inst_executed_op_uniform.sum"),
]
rows = []
for short, full in ops:
    a = scale(w3x.get(full))
    b = scale(rank1.get(full))
    if a is None and b is None:
        continue
    da_M = (a / 1e6) if a is not None else None
    db_M = (b / 1e6) if b is not None else None
    dd_M = (da_M - db_M) if (da_M is not None and db_M is not None) else None
    rows.append((short, da_M, db_M, dd_M))

rows.sort(key=lambda r: -abs(r[3]) if r[3] is not None else 0)
print(f"  {'op':<22} {'w3x (M)':>12} {'rank1 (M)':>12} {'delta (M)':>12}")
print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12}")
for short, a, b, d in rows:
    sa = f"{a:,.2f}" if a is not None else "-"
    sb = f"{b:,.2f}" if b is not None else "-"
    sd = f"{d:+,.2f}" if d is not None else "-"
    print(f"  {short:<22} {sa:>12} {sb:>12} {sd:>12}")
print()

# ── table 4: sanity ─────────────────────────────────────────────────────
print("=" * 72)
print("G4: SANITY (total inst, cycles, warps_active, tensor%)")
print("=" * 72)
sanity = [
    ("total_inst (M)",  "smsp__inst_executed.sum",                        1e6),
    ("cycles (M)",      "sm__cycles_elapsed.avg",                         1e6),
    ("warps_act %",     "sm__warps_active.avg.pct_of_peak_sustained_active",  1),
    ("tensor_cyc %",    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", 1),
]
print(f"  {'metric':<18} {'w3x':>12} {'rank1':>12} {'delta':>12}")
print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*12}")
for short, full, div in sanity:
    a = scale(w3x.get(full))
    b = scale(rank1.get(full))
    if a is None and b is None:
        continue
    sa = f"{a/div:,.2f}" if a is not None else "-"
    sb = f"{b/div:,.2f}" if b is not None else "-"
    sd = f"{(a-b)/div:+,.2f}" if (a is not None and b is not None) else "-"
    print(f"  {short:<18} {sa:>12} {sb:>12} {sd:>12}")
print()

# ── verdict ─────────────────────────────────────────────────────────────
print("=" * 72)
print("VERDICT")
print("=" * 72)
if rows:
    # G3 sorted rows; look at top delta for action
    top = rows[0] if rows[0][3] is not None else None
    # look up LSU / FMA+ALU / Uniform deltas from G2
    d_by_pipe = {}
    for short, _, _, d in [(r[0], r[1], r[2], r[3]) for r in sorted(
            [(s, a, b, d2) for s, a, b, d2 in [(sn, scale(w3x.get(f)), scale(rank1.get(f)), None)
                                                for sn, f in pipes_sum]
             if a is not None and b is not None
             for d2 in [((a or 0) - (b or 0)) / 1e6]], key=lambda x: x[0])]:
        d_by_pipe[short] = d
    # recompute cleanly
    d_by_pipe = {}
    for short, full in pipes_sum:
        a = scale(w3x.get(full))
        b = scale(rank1.get(full))
        if a is not None and b is not None:
            d_by_pipe[short] = (a - b) / 1e6
    if d_by_pipe:
        biggest = max(d_by_pipe.items(), key=lambda kv: abs(kv[1]))
        name, delta = biggest
        print(f"  Largest pipe delta: {name} at {delta:+.2f} M instructions (w3x - rank1)")
        if abs(delta) < 5:
            print()
            print("  All pipes match within ~5 M instructions.")
            print("  → Instruction-count axis has no leverage. Ship 1.007 ms.")
        elif name == "lsu" and delta > 0:
            print()
            print("  LSU is the axis. w3x emits extra STS/LDS work in epi.")
            print("  → Register-packed epilogue is the lever.")
            print("  → Check G3 'shared_st (STS)' delta to confirm it's store-side.")
        elif name in ("fma", "alu") and delta > 0:
            print()
            print("  Compute pipe is the axis. w3x does extra epi math / address calc.")
            print("  → UR-based descriptor advance or consolidated CVT is the lever.")
        elif name == "uniform" and delta < 0:
            print()
            print("  Rank-1 does more on the UNIFORM datapath than we do.")
            print("  → We're missing UR scalarization. Lift per-cta constants to UR.")
        else:
            print()
            print(f"  {name} differs by {delta:+.2f} M — consult G3 for op-class breakdown.")
    else:
        print("  (G2 per-pipe sums unavailable — rerun with --bisect)")
else:
    print("  (not enough data — check run.log)")
PY

log ""
log "--- artifacts ---"
log "  csvs:        $OUT/{w3x,rank1}*.csv"
log "  decoded:     $OUT/pipe_diff.txt"
log "  trace:       $OUT/run.log"
log ""
log "  next: paste pipe_diff.txt to claude"
