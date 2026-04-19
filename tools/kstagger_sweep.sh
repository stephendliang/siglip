#!/usr/bin/env bash
# kstagger_sweep.sh — sweep K_STAGGER × {dispatch} × {layer} × {fused, gemm, strip}.
#
# Tests whether K-phase decorrelation composes with good static dispatches
# (dgswizzle, zigzag, checkered, dg16) under PACKED_TILES parity.  K_STAGGER
# was previously tried only on `default`; if it generalizes, we expect strip
# to drop (TMA arrival pattern) without breaking the good mainloop L2
# behavior of the host dispatch.
#
# Output: data/kstagger_<ts>/{results.csv, <label>.log ...} plus a
# decomposition table on stdout.
#
# Usage:
#   ./tools/kstagger_sweep.sh                       # defaults (fc2+fc1)
#   LAYERS="fc2" ./tools/kstagger_sweep.sh          # fc2 only
#   DISPATCHES="dgswizzle zigzag" KSTAGGERS="0 1" ./tools/kstagger_sweep.sh
#   MODES="fused" ./tools/kstagger_sweep.sh         # fused-only (3× faster)

set -u

cd "$(dirname "$0")/.."

LAYERS="${LAYERS:-fc2 fc1}"
DISPATCHES="${DISPATCHES:-default dgswizzle zigzag checkered dg16}"
KSTAGGERS="${KSTAGGERS:-0 1 2 3}"
MODES="${MODES:-fused gemm strip}"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="data/kstagger_${STAMP}"
mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/results.csv"
echo "layer,dispatch,kstagger,mode,ms,tflops,valid" > "$CSV"

suffix_for() {
    case "$1" in
        default) echo "" ;;
        *)       echo "-$1" ;;
    esac
}

run_one() {
    local layer="$1" disp="$2" ks="$3" mode="$4"
    local suf target flags label log
    suf=$(suffix_for "$disp")
    target="${layer}-w3${suf}"
    flags="-DPACKED_TILES"
    [ "$ks" != "0" ]        && flags="$flags -DK_STAGGER=$ks"
    [ "$mode" = "gemm"  ]   && flags="$flags -DGEMM_ONLY"
    [ "$mode" = "strip" ]   && flags="$flags -DSTRIP_EPILOGUE"
    label="${layer}_${disp}_ks${ks}_${mode}"
    log="$OUT_DIR/${label}.log"

    printf "[%-38s] build ... " "$label"
    if ! make -B "$target" DFLAGS="$flags" >"$log" 2>&1; then
        echo "BUILD FAIL"
        echo "$layer,$disp,$ks,$mode,BUILD_FAIL,," >> "$CSV"
        return
    fi
    printf "run ... "
    if ! "./$target" >>"$log" 2>&1; then
        echo "RUN FAIL"
        echo "$layer,$disp,$ks,$mode,RUN_FAIL,," >> "$CSV"
        return
    fi
    local line ms tflops valid
    line=$(grep '^@@RESULT' "$log" | tail -1)
    ms=$(echo     "$line" | grep -oP 'ms=\K[0-9.]+')
    tflops=$(echo "$line" | grep -oP 'tflops=\K[0-9.]+')
    valid=$(echo  "$line" | grep -oP 'valid=\K[0-9]+')
    echo "ms=$ms valid=$valid"
    echo "$layer,$disp,$ks,$mode,$ms,$tflops,$valid" >> "$CSV"
}

TOTAL=0
for l in $LAYERS; do for d in $DISPATCHES; do for k in $KSTAGGERS; do for m in $MODES; do
    TOTAL=$((TOTAL + 1))
done; done; done; done
echo "Sweep: $TOTAL configs → $OUT_DIR"
echo

i=0
T0=$(date +%s)
for layer in $LAYERS; do
    for disp in $DISPATCHES; do
        for ks in $KSTAGGERS; do
            for mode in $MODES; do
                i=$((i + 1))
                printf "(%3d/%3d) " "$i" "$TOTAL"
                run_one "$layer" "$disp" "$ks" "$mode"
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
    bag[(r["layer"], r["dispatch"], int(r["kstagger"]))][r["mode"]] = ms

hdr = f"{'layer':<4}  {'dispatch':<12}  {'ks':>2}  {'fused':>7}  {'gemm':>7}  {'strip':>7}  {'f-g':>7}  {'g-s':>7}"
print(hdr)
print("-" * len(hdr))
for (layer, disp, ks), m in sorted(bag.items()):
    f, g, s = m.get("fused"), m.get("gemm"), m.get("strip")
    fg = (f - g) if (f is not None and g is not None) else None
    gs = (g - s) if (g is not None and s is not None) else None
    def fmt(x):
        return f"{x:7.3f}" if x is not None else f"{'-':>7}"
    print(f"{layer:<4}  {disp:<12}  {ks:>2}  {fmt(f)}  {fmt(g)}  {fmt(s)}  {fmt(fg)}  {fmt(gs)}")

print()
print("== Deltas vs ks=0 (fused only) ==")
base = {}
for (layer, disp, ks), m in sorted(bag.items()):
    if ks == 0 and "fused" in m:
        base[(layer, disp)] = m["fused"]
for (layer, disp, ks), m in sorted(bag.items()):
    if ks == 0: continue
    f = m.get("fused"); b = base.get((layer, disp))
    if f is None or b is None: continue
    delta = f - b
    marker = "  "
    if delta < -0.005: marker = "++"
    elif delta < -0.001: marker = " +"
    elif delta > 0.005: marker = "--"
    elif delta > 0.001: marker = " -"
    print(f"  {layer}  {disp:<12}  ks={ks}  {f:7.3f}  Δ={delta:+.3f} {marker}")
PY

echo
echo "CSV: $CSV"
