#!/usr/bin/env bash
#
# 1-way swizzle sweep on fc2_w3x via the templated single-binary flow
# (refactor 2026-04-27): one build, one binary, SWEEP env drives the
# active variant list, REPS env drives pass-major outer iterations.
#
# All 20 cells exist as explicit template instantiations of
# fc2_w3x_kernel<TD, DGG> in the binary — each compiles to its own SASS,
# byte-identical to the per-build flow that preceded the refactor.
# The host loops pass-major: pass p ∈ 1..REPS × variant v ∈ ACTIVE,
# emitting one `@@SAMPLE pass=p variant=v ms=X` line per launch.
# Cross-cell drift cancels at the granularity of one pass instead of
# one whole-cell run.
#
# Variants in the sweep (matches main()'s VARIANT_TABLE in fc2_w3x.cu;
# 31 templated cells, all built into one binary, SWEEP=all enumerates):
#   dgsw / dg2 / dg4 / dg16 / dg32 — DG_GROUP_SIZE curve
#   zigzag / rowmajor              — reference outliers (TD=11/13)
#   checkered / dgsnake            — structural alternatives (TD=18/19)
#   gflip / tn2br                  — front-tier members (TD=33/34)
#   dg4diag / dg4pp                — dg4 + within-group probes (TD=35/36, DGG=4)
#   g4swap / lmrev                 — analyzer-driven (TD=38/39)
#   comboAB / comboAC              — tnRun-collapse falsifiers (TD=40/41)
#   snrot1 / snrot2 / lmsn         — tnRun-preserving (TD=43/44/45)
#   dg_psh3 / dg_sn_psh3 /         — round-3 probes (TD=46/47/48,
#     dg_snlmrev                     adj_tm_diff hull-extension)
#   dgsnake_G4 / dgsnake_G16       — dgsnake at non-G8 (TD=19, DGG=4/16)
#   dg_antisnake / dg_tt_phase /   — round-4 probes (TD=49/50/51,
#     wfd_latin                      lm-half / cluster-band / per-cluster
#                                    tt-rotation)
#   gflip_lmrev / gflip_snrot /    — round-5 (TD=52/53), round-6 (TD=54/55)
#     gflip_blkswap / gflip_cidperm  bloom-filter survivors;
#                                    blkswap+lmrev TIE @ n=43910 (front).
#   gflip_blklmrev / gflip_blkmul3 — round-7 probes (TD=56/57/58, 2026-04-28
#     / gflip_quartswap              bloom WORTHY, asking saturation/
#                                    perturbation/density questions).
#
# History: TD=37 rowmaj and TD=42 tnblk are still in the binary as
# documented dead-ends but are not in this sweep (n=4 confirmed losers
# at +3.0 µs / +444 µs).  INGH=32 dropped earlier (n=20 at +3.35 µs).
#
# Output: data/fc2_w3x_swizzle_<ts>/
#   build.log          (28-template kernel build)
#   run.log            (single binary run, all @@SAMPLE lines)
#   wall_data.csv      (variant,swizzle,rep,ms,cyc)
#   compare.txt        (per-cell stats + 1-way ANOVA + paired pairwise
#                       AUC/d/mean_rank with bootstrap CIs if BOOT>0)
#   metrics.csv        (structural metrics from analyze_swizzle.py)
#   study.txt          (Kendall τ + tier-conditional + collinearity from
#                       study_summary.py — joins wall × structure)
#   clusters.log       (per-cluster cyc, only when CLUSTERS=1)
#
# Usage:
#   tools/sweep_fc2_w3x_swizzle.sh                         # REPS=20, BOOT=0
#   REPS=5489 BOOT=1000 tools/sweep_fc2_w3x_swizzle.sh     # full B200 run
#   SWEEP=dgsw,dg4,lmsn REPS=128 tools/sweep_fc2_w3x_swizzle.sh
#   SWEEP=front REPS=16384 tools/sweep_fc2_w3x_swizzle.sh  # front-tier only,
#                                                          # higher N to crack tie
#   CLUSTERS=1 REPS=128 tools/sweep_fc2_w3x_swizzle.sh     # +per-cluster
#   tools/sweep_fc2_w3x_swizzle.sh my_outdir
#
# SWEEP=front expands to the top 10 wall-vetted variants + 3 bloom-filter
# survivors awaiting wall verification:
#   front pair (TIE @ AUC=0.541): gflip_blkswap (TD=54, 3.29) + gflip_lmrev
#     (TD=52, 3.48)
#   MODERATE band: dgsnake (4.80) + gflip (4.98) + gflip_snrot (5.07) +
#     lmrev (5.17) + snrot2 (5.34) + dgsw_G8 (5.40)
#   STRONG anchors (full-sweep): dg_antisnake (10.04, +1140 cyc) +
#     dg4 (11.99, +1809 cyc) — different-mechanism references.
#   round-2 candidates (bloom WORTHY, 2026-04-28):
#     gflip_blklmrev  (TD=56, s=+1.53) — lmrev × blkswap stack, asks "is
#                     Lever C saturated?"
#     gflip_blkmul3   (TD=57, s=+0.67) — alt-group lm = (lm*3)%8 instead of
#                     blkswap's lm^4, asks "is the perturbation magnitude
#                     the lever or any decorrelation?"
#     gflip_quartswap (TD=58, s=+0.15) — lm^4 only every 4th group (vs
#                     blkswap's every other), density-calibration probe.
# Dropped: gflip_cidperm (DEAD, +1568 DECISIVE).
# 13 variants vs all=31 → ~2.4× more passes per wall-clock minute.

set -u
cd "$(dirname "$0")/.."

OUT=${1:-"data/fc2_w3x_swizzle_$(date +%Y%m%d_%H%M%S)"}
REPS=${REPS:-20}
SWEEP=${SWEEP:-all}
BOOT=${BOOT:-0}
CLUSTERS=${CLUSTERS:-0}

if [ "$SWEEP" = "front" ]; then
    SWEEP="gflip_blkswap,gflip_lmrev,dgsnake,gflip,gflip_snrot,lmrev,snrot2,dgsw,dg_antisnake,dg4,gflip_blklmrev,gflip_blkmul3,gflip_quartswap"
fi
NVCC=${NVCC:-nvcc}
CFLAGS='-gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -lineinfo --ptxas-options=-v --cudart=static'
LDFLAGS='-lcurand_static -lculibos -lcuda'

mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/run.log"; }

BIN="$OUT/fc2-w3x-sweep"

log "=== Phase 1: building single binary with -DCOMBO_QUICK ==="
if ! $NVCC $CFLAGS -DCOMBO_QUICK fc2_w3x.cu -o "$BIN" $LDFLAGS \
        > "$OUT/build.log" 2>&1; then
    log "ERROR: build FAIL — see $OUT/build.log"
    exit 1
fi
n_kerns=$(grep -cE 'Compiling entry function' "$OUT/build.log" || true)
log "  build OK  ($n_kerns kernel instantiations)"

log ""
log "=== Phase 2: SWEEP=$SWEEP, REPS=$REPS, BOOT=$BOOT, CLUSTERS=$CLUSTERS, pass-major in one process ==="
EMIT_CLUSTERS_VAL=0
[ "$CLUSTERS" -gt 0 ] && EMIT_CLUSTERS_VAL=1
if ! REPS="$REPS" SWEEP="$SWEEP" EMIT_CLUSTERS="$EMIT_CLUSTERS_VAL" \
        "$BIN" > "$OUT/run.log" 2>&1; then
    log "ERROR: run FAIL — see $OUT/run.log"
    exit 1
fi
n_samples=$(grep -cE '^@@SAMPLE pass=' "$OUT/run.log" || true)
n_variants=$(grep -oE 'variant=[^ ]+' "$OUT/run.log" | sort -u | wc -l)
log "  $n_samples samples across $n_variants variants"

log ""
log "=== Phase 3: extracting @@SAMPLE → $OUT/wall_data.csv ==="
{
    echo "variant,swizzle,rep,ms,cyc"
    grep -E '^@@SAMPLE pass=' "$OUT/run.log" | \
        sed -E 's/^@@SAMPLE pass=([0-9]+) variant=([^ ]+) ms=([0-9.]+) cyc=([0-9]+).*/\2,\2,\1,\3,\4/'
} > "$OUT/wall_data.csv"
n_rows=$(( $(wc -l < "$OUT/wall_data.csv") - 1 ))
log "extracted $n_rows wall measurements"

log ""
log "=== Phase 4: 1-way ANOVA (cyc, paired by pass, trim 33%) → $OUT/compare.txt ==="
ANOVA_BOOT_ARG=""
[ "$BOOT" -gt 0 ] && ANOVA_BOOT_ARG="--boot $BOOT"
python3 tools/anova_1way.py "$OUT/wall_data.csv" \
    --factor swizzle \
    --metric cyc \
    --paired rep \
    --trim 0.33 \
    $ANOVA_BOOT_ARG \
    --out "$OUT/compare.txt"

log ""
log "=== Phase 5: structural metrics → $OUT/metrics.csv ==="
python3 tools/analyze_swizzle.py --csv "$OUT/metrics.csv" --summary \
    > "$OUT/metrics.txt" 2>&1 || log "  WARN: analyze_swizzle failed"

log ""
log "=== Phase 6: study (wall × structure) → $OUT/study.txt ==="
STUDY_BOOT_ARG=""
[ "$BOOT" -gt 0 ] && STUDY_BOOT_ARG="--boot $BOOT"
python3 tools/study_summary.py "$OUT/compare.txt" "$OUT/metrics.csv" \
    $STUDY_BOOT_ARG > "$OUT/study.txt" 2>&1 || log "  WARN: study_summary failed"

if [ "$CLUSTERS" -gt 0 ]; then
    log ""
    log "=== Phase 7: cluster affinity → $OUT/clusters.txt ==="
    grep -E '^@@CLUSTER ' "$OUT/run.log" > "$OUT/clusters.log" || true
    n_cl=$(wc -l < "$OUT/clusters.log")
    if [ "$n_cl" -gt 0 ]; then
        python3 tools/cluster_affinity.py "$OUT/clusters.log" \
            > "$OUT/clusters.txt" 2>&1 || log "  WARN: cluster_affinity failed"
    else
        log "  no @@CLUSTER lines emitted (CLUSTERS=$CLUSTERS but binary didn't dump)"
    fi
fi

log ""
log "wall ANOVA:       $OUT/compare.txt"
log "structural:       $OUT/metrics.txt"
log "study (joined):   $OUT/study.txt"
[ "$CLUSTERS" -gt 0 ] && log "cluster affinity: $OUT/clusters.txt"
log "raw wall data:    $OUT/wall_data.csv"
log "single-run log:   $OUT/run.log"
