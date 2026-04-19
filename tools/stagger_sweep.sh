#!/usr/bin/env bash
# stagger_sweep.sh — sweep K_STAGGER × N_STAGGER × {dispatch} × {layer} × {mode}.
#
# K_STAGGER shifts per-cluster K-iteration start.  N_STAGGER rotates the
# per-cluster tile-visit order (bijective permutation of each cluster's
# tile set).  The goal: find a (ks, ns) pair that captures K_STAGGER's g-s
# decorrelation win (on FC1, ks=1 saves ~270us of g-s) without the strip
# and f-g regressions that K-phase staggering induces on the mainloop.
#
# Output: data/stagger_<ts>/{results.csv, <label>.log ...} plus a 2D
# surface table on stdout.
#
# Usage:
#   ./tools/stagger_sweep.sh                                           # default grid
#   LAYERS=fc1 DISPATCHES=zigzag KSTAGGERS="0 1" NSTAGGERS="0 1 2 3 5" ./tools/stagger_sweep.sh

set -u

cd "$(dirname "$0")/.."

LAYERS="${LAYERS:-fc1}"
DISPATCHES="${DISPATCHES:-zigzag}"
KSTAGGERS="${KSTAGGERS:-0 1 2 3 5}"
NSTAGGERS="${NSTAGGERS:-0 1 2 3 5}"
MODES="${MODES:-fused gemm strip}"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="data/stagger_${STAMP}"
mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/results.csv"
echo "layer,dispatch,kstagger,nstagger,mode,ms,tflops,valid" > "$CSV"

suffix_for() {
    case "$1" in
        default) echo "" ;;
        *)       echo "-$1" ;;
    esac
}

run_one() {
    local layer="$1" disp="$2" ks="$3" ns="$4" mode="$5"
    local suf target flags label log
    suf=$(suffix_for "$disp")
    target="${layer}-w3${suf}"
    flags="-DPACKED_TILES"
    [ "$ks" != "0" ]        && flags="$flags -DK_STAGGER=$ks"
    [ "$ns" != "0" ]        && flags="$flags -DN_STAGGER=$ns"
    [ "$mode" = "gemm"  ]   && flags="$flags -DGEMM_ONLY"
    [ "$mode" = "strip" ]   && flags="$flags -DSTRIP_EPILOGUE"
    label="${layer}_${disp}_ks${ks}_ns${ns}_${mode}"
    log="$OUT_DIR/${label}.log"

    printf "[%-44s] build ... " "$label"
    if ! make -B "$target" DFLAGS="$flags" >"$log" 2>&1; then
        echo "BUILD FAIL"
        echo "$layer,$disp,$ks,$ns,$mode,BUILD_FAIL,," >> "$CSV"
        return
    fi
    printf "run ... "
    if ! "./$target" >>"$log" 2>&1; then
        echo "RUN FAIL"
        echo "$layer,$disp,$ks,$ns,$mode,RUN_FAIL,," >> "$CSV"
        return
    fi
    local line ms tflops valid
    line=$(grep '^@@RESULT' "$log" | tail -1)
    ms=$(echo     "$line" | grep -oP 'ms=\K[0-9.]+')
    tflops=$(echo "$line" | grep -oP 'tflops=\K[0-9.]+')
    valid=$(echo  "$line" | grep -oP 'valid=\K[0-9]+')
    echo "ms=$ms valid=$valid"
    echo "$layer,$disp,$ks,$ns,$mode,$ms,$tflops,$valid" >> "$CSV"
}

TOTAL=0
for l in $LAYERS; do for d in $DISPATCHES; do for k in $KSTAGGERS; do for n in $NSTAGGERS; do for m in $MODES; do
    TOTAL=$((TOTAL + 1))
done; done; done; done; done
echo "Sweep: $TOTAL configs → $OUT_DIR"
echo

i=0
T0=$(date +%s)
for layer in $LAYERS; do
    for disp in $DISPATCHES; do
        for ks in $KSTAGGERS; do
            for ns in $NSTAGGERS; do
                for mode in $MODES; do
                    i=$((i + 1))
                    printf "(%3d/%3d) " "$i" "$TOTAL"
                    run_one "$layer" "$disp" "$ks" "$ns" "$mode"
                done
            done
        done
    done
done
T1=$(date +%s)
echo
echo "Done in $((T1 - T0))s."
echo

python3 - "$CSV" <<'PY'
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
bag = collections.defaultdict(dict)
for r in rows:
    try: ms = float(r["ms"])
    except Exception: continue
    bag[(r["layer"], r["dispatch"], int(r["kstagger"]), int(r["nstagger"]))][r["mode"]] = ms

# Per-mode 2D surface: rows=ks, cols=ns
def surface(mode):
    per_ld = collections.defaultdict(dict)
    for (l, d, k, n), m in bag.items():
        v = m.get(mode)
        if v is not None:
            per_ld[(l, d)][(k, n)] = v
    for (l, d), grid in sorted(per_ld.items()):
        kss = sorted({k for (k, _) in grid})
        nss = sorted({n for (_, n) in grid})
        print(f"[{l} {d}  {mode}]  rows=ks  cols=ns")
        hdr = "  ks\\ns  " + "  ".join(f"{n:>7d}" for n in nss)
        print(hdr)
        print("-" * len(hdr))
        for k in kss:
            row = f"  ks={k:<3d} " + "  ".join(
                f"{grid[(k,n)]:7.3f}" if (k,n) in grid else f"{'-':>7}"
                for n in nss
            )
            print(row)
        print()

for mode in ["fused", "gemm", "strip"]:
    if any(mode in m for m in bag.values()):
        surface(mode)

# Decomposition for each (ks, ns) cell
print("== Decomp (f, g, s, f-g, g-s) ==")
hdr = f"{'layer':<4}  {'dispatch':<12}  {'ks':>2} {'ns':>2}  {'fused':>7}  {'gemm':>7}  {'strip':>7}  {'f-g':>7}  {'g-s':>7}"
print(hdr)
print("-" * len(hdr))
for (layer, disp, ks, ns), m in sorted(bag.items()):
    f, g, s = m.get("fused"), m.get("gemm"), m.get("strip")
    fg = (f - g) if (f is not None and g is not None) else None
    gs = (g - s) if (g is not None and s is not None) else None
    def fmt(x):
        return f"{x:7.3f}" if x is not None else f"{'-':>7}"
    print(f"{layer:<4}  {disp:<12}  {ks:>2} {ns:>2}  {fmt(f)}  {fmt(g)}  {fmt(s)}  {fmt(fg)}  {fmt(gs)}")

print()
print("== Deltas vs (ks=0, ns=0) fused ==")
base = {}
for (layer, disp, ks, ns), m in bag.items():
    if ks == 0 and ns == 0 and "fused" in m:
        base[(layer, disp)] = m["fused"]
for (layer, disp, ks, ns), m in sorted(bag.items()):
    if ks == 0 and ns == 0: continue
    f = m.get("fused"); b = base.get((layer, disp))
    if f is None or b is None: continue
    delta = f - b
    marker = "  "
    if delta < -0.010: marker = "++"
    elif delta < -0.003: marker = " +"
    elif delta > 0.010: marker = "--"
    elif delta > 0.003: marker = " -"
    print(f"  {layer}  {disp:<12}  ks={ks} ns={ns}  {f:7.3f}  Δ={delta:+.3f} {marker}")
PY

echo
echo "CSV: $CSV"
