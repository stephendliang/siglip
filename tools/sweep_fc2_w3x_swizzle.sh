#!/usr/bin/env bash
#
# 1-way swizzle sweep on fc2_w3x: maps the DG_GROUP_SIZE curve
# (dg2 / dg4 / dg8(=dgsw default) / dg16 / dg32) alongside the 2 TD>30
# survivors (gflip=33, tn2br=34), 2 structural alternatives
# (checkered=18, dgsnake=19), and 8 analyzer-driven new probes
# (dg4_diag=35, dg4_pingpong=36 from 2026-04-27 round 1; rowmaj=37,
# g4swap=38, lmrev=39, comboAB=40, comboAC=41, tnblk=42 from round 2,
# 2026-04-27 from tools/cluster_swizzle.py + Kendall-τ analysis on the
# n=200 wall sweep).  zigzag=11 / rowmajor=13 stay in as off-tie
# reference outliers so the F-statistic stays interpretable.
#
# Round-2 hypothesis matrix (cluster_swizzle.py):
#   front-tier {dgsnake, gflip, tn2br} centroid is 1.32 std-units from
#   dgsw_G8 along three orthogonal feature axes.  TD=37..42 isolate
#   them:
#     38 g4swap  — gflip-style group XOR=3 (vs gflip's XOR=1)
#     39 lmrev   — push adj_tm_diff via 3-bit lm reverse (tn2br lever
#                  generalized across all (tm, tn))
#     40 comboAB — TD=37 + TD=38 (composes A+B)
#     41 comboAC — TD=37 + TD=39 (composes A+C)
#
# 2026-04-27 round-2 cull: TD=37 rowmaj and TD=42 tnblk dropped from the
# rotation after the n=4 sweep confirmed structural-loser predictions:
#   rowmaj: +3.0 µs STRONG slower (tnRun=1.0 = rowmajor disease)
#   tnblk:  +444 µs catastrophic (global tn-block is architecturally
#           hostile — kernel still functions but is far off the floor)
# Source still in fc2_w3x.cu as documented dead-ends.  combo_AB / combo_AC
# kept for now (also lost +3.5/+2.75 µs at n=4) so the sweep retains a
# falsifier showing the tnRun-collapse mechanism vs the tnRun-preserving
# rot1/rot2/lmsn cells.
#
# At n=100 (2026-04-26) dg4 won by ~1.85 µs STRONG vs the dg8/checkered/
# dgsnake/gflip cluster, with tn2br alone at +0.47 µs MODERATE.
# Analyzer (tools/analyze_swizzle.py) predicts:
#   - dg2: densest within-wavefront A multicast (tmMult=2.85 vs dg4=2.71,
#     dg8=2.49) and zero tick-to-tick irregularity (tIrM=0.0). Should tie
#     dg4 or beat by ≤0.5 µs if multicast-density is the lever.
#   - dg4_diag (TD=35): dg4 + within-group diagonal tn rotation.  Same
#     wavefront tm-shape as dg4 but breaks same-tn cluster lockstep across
#     the 4 group-mates — every cluster slot in a group stamps a different
#     tn at any tick.  Tests whether the residual dg4 cost is
#     within-group same-tn cluster lockstep.
#   - dg4_pingpong (TD=36): dg4 + tn-shift on odd groups.  Different lever
#     from dg_diag — varies tn run length per cluster (tnRun=2.38 vs
#     dg4=1.99, dg4_diag=1.00) instead of shifting per cluster slot.
#   - dg16 / dg32: high cross-tick L2 warmth (0.216, 0.430) but low
#     wavefront density (uTm=35.3, 45.9).  Confirm L2-warmth is NOT the
#     lever (i.e. dg32 still loses despite best L2w8).
#
# INGH=32 dropped: n=20 sweep landed it at +3.35 µs vs leader (STRONG,
# p≪1e-9) — worse than the zigzag/rowmajor reference outliers.  Analyzer
# diagnosis: tIrM=1.9, tIr95=4 (highest among dg-family) + frMax=30
# vs dg4's 26 → wavefront shape oscillates between even-group dgsw layout
# and odd-group tn-major layout, creating bursty/regular DRAM demand
# alternation that doesn't smooth out.  Don't re-probe.
#
# Pass-major interleaving (all variants in pass i, then pass i+1) shares
# clock / thermal / queue state across cells, beating cross-session
# B200 drift (~4 µs, larger than the µs-scale lever effects we care
# about). REPS=20 default catches ~0.5 µs cell separations at α≈0.05;
# REPS=64 tightens to ~0.2 µs.
#
# Output: data/fc2_w3x_swizzle_<ts>/
#   build-<cell>.log
#   run-<cell>-<rep>.log
#   wall_data.csv      (variant,swizzle,rep,ms)
#   compare.txt        (per-cell stats + 1-way ANOVA + pairwise Welch)
#
# Usage:
#   tools/sweep_fc2_w3x_swizzle.sh                # REPS=20
#   REPS=64 tools/sweep_fc2_w3x_swizzle.sh
#   tools/sweep_fc2_w3x_swizzle.sh my_outdir

set -u
cd "$(dirname "$0")/.."

OUT=${1:-"data/fc2_w3x_swizzle_$(date +%Y%m%d_%H%M%S)"}
REPS=${REPS:-20}
NVCC=${NVCC:-nvcc}
CFLAGS='-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -lineinfo --ptxas-options=-v --cudart=static'
LDFLAGS='-lcurand_static -lculibos -lcuda'

mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/run.log"; }

# Format: "<name>:<swizzle>:<-D flag list>"
# swizzle column groups variants for ANOVA — one cell per name here.
VARIANTS=(
    "dgsw:dgsw:"
    "checkered:checkered:-DTILE_DISPATCH=18"
    "dgsnake:dgsnake:-DTILE_DISPATCH=19"
    "dg2:dg2:-DDG_GROUP_SIZE=2"
    "dg4:dg4:-DDG_GROUP_SIZE=4"
    "dg16:dg16:-DDG_GROUP_SIZE=16"
    "dg32:dg32:-DDG_GROUP_SIZE=32"
    "dg4diag:dg4diag:-DTILE_DISPATCH=35 -DDG_GROUP_SIZE=4"
    "dg4pp:dg4pp:-DTILE_DISPATCH=36 -DDG_GROUP_SIZE=4"
    "gflip:gflip:-DTILE_DISPATCH=33"
    "tn2br:tn2br:-DTILE_DISPATCH=34"
    "g4swap:g4swap:-DTILE_DISPATCH=38"
    "lmrev:lmrev:-DTILE_DISPATCH=39"
    "comboAB:comboAB:-DTILE_DISPATCH=40"
    "comboAC:comboAC:-DTILE_DISPATCH=41"
    "snrot1:snrot1:-DTILE_DISPATCH=43"
    "snrot2:snrot2:-DTILE_DISPATCH=44"
    "lmsn:lmsn:-DTILE_DISPATCH=45"
    "zigzag:zigzag:-DTILE_DISPATCH=11"
    "rowmajor:rowmajor:-DTILE_DISPATCH=13"
)

log "=== Phase 1: building ${#VARIANTS[@]} variants with -DCOMBO_QUICK ==="
BUILD_OK=()
for entry in "${VARIANTS[@]}"; do
    IFS=: read -r name swizzle flags <<< "$entry"
    bin="$OUT/fc2-w3x-swizzle-$name"
    if $NVCC $CFLAGS -DCOMBO_QUICK $flags fc2_w3x.cu -o "$bin" $LDFLAGS \
            > "$OUT/build-$name.log" 2>&1; then
        BUILD_OK+=("$entry")
        regs=$(grep -oE 'Used [0-9]+ registers' "$OUT/build-$name.log" | head -1 | awk '{print $2}')
        log "  build OK   $name  flags='$flags'  regs=$regs"
    else
        log "  build FAIL $name  flags='$flags'  (see $OUT/build-$name.log)"
    fi
done
log "build summary: ${#BUILD_OK[@]} / ${#VARIANTS[@]} succeeded"

if [[ ${#BUILD_OK[@]} -lt 2 ]]; then
    log "ERROR: need ≥2 cells for ANOVA; aborting"
    exit 1
fi

log ""
log "=== Phase 2: $REPS reps/variant, pass-major interleaving ==="
for rep in $(seq 1 "$REPS"); do
    log "-- pass $rep/$REPS --"
    for entry in "${BUILD_OK[@]}"; do
        IFS=: read -r name swizzle flags <<< "$entry"
        bin="$OUT/fc2-w3x-swizzle-$name"
        if ! "$bin" > "$OUT/run-$name-$rep.log" 2>&1; then
            log "  pass $rep $name run-FAIL"
        fi
    done
done

log ""
log "=== Phase 3: extracting wall ms → $OUT/wall_data.csv ==="
{
    echo "variant,swizzle,rep,ms"
    for entry in "${BUILD_OK[@]}"; do
        IFS=: read -r name swizzle flags <<< "$entry"
        for rep in $(seq 1 "$REPS"); do
            f="$OUT/run-$name-$rep.log"
            if [[ -f "$f" ]]; then
                ms=$(grep -oE 'FC2-W3X kernel: [0-9.]+ ms' "$f" | head -1 | awk '{print $3}')
                if [[ -n "$ms" ]]; then
                    echo "$name,$swizzle,$rep,$ms"
                fi
            fi
        done
    done
} > "$OUT/wall_data.csv"
n_rows=$(( $(wc -l < "$OUT/wall_data.csv") - 1 ))
log "extracted $n_rows wall measurements"

log ""
log "=== Phase 4: 1-way ANOVA → $OUT/compare.txt ==="
python3 tools/anova_1way.py "$OUT/wall_data.csv" \
    --factor swizzle \
    --out "$OUT/compare.txt"

log ""
log "report:           $OUT/compare.txt"
log "raw wall data:    $OUT/wall_data.csv"
log "raw per-run logs: $OUT/run-<cell>-<rep>.log"
