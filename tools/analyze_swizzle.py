#!/usr/bin/env python3
"""
Tile-swizzle reuse analyzer for fc2_w3x.

Models the persistent dispatch lin_tile = cluster_id + tt * NC, with NC=74
clusters, TILES_M=3626, TILES_N=3 (FC2 K=3072 BIAS_ONLY shape). At each tick
tt, the 74 concurrent clusters span lin values [tt*74, tt*74+73] — one
contiguous run of 74 lins per wavefront.

Per-tile bytes (FP8 K=3072 → BF16 epilogue):
  A_strip per tm : TM=128, K=3072 →  384 KB (1 CTA)  / cluster covers 256 M → 768 KB
  B_strip per tn : TN=256, K=3072 →  768 KB

A-footprint = 3626 * 768 KB ≈ 2.7 GB (DRAM-bound).
B-footprint = 3   * 768 KB =   2.3 MB (always L2-resident — one-shot stream).

So the lever is A. We measure each variant on:

  1. per-cluster intra-tn run length    (B-strip reuse, but B is L2-resident
                                          so this only matters for SMEM-prefetch
                                          + W4 same-tile retirement patterns)
  2. per-cluster intra-tm run length    (A-strip reuse — same A bytes back-to-
                                          back from L2 if still warm)
  3. wavefront unique-tm count          (how many distinct A strips the
                                          74-cluster wavefront pulls from L2)
  4. wavefront tm-multiplicity          (mean # clusters per shared tm — dense
                                          multiplicity = good multicast / L2)
  5. cross-cluster L2-window A reuse    (at tick T, what fraction of the 74
                                          required A-strips were ALSO
                                          required in [T-w, T-1] for L2-warm)
  6. fresh-A-strip influx per tick      (DRAM-side: new tm seen across all
                                          ticks ≤ T, additional unique tm at T)
  7. peak concurrent unique tm          (max over wavefronts — DRAM peak load)
  8. group-locality score               (how clustered each tm's tick set is —
                                          tight cluster = bursty DRAM-friendly,
                                          spread-out = trickle-load)

Variants modeled match the C/CUDA implementations:
  dgsw (G ∈ {2,4,8,16,32}), dgsnake (TD=19), checkered (TD=18, CK_GROUP_M=8,
  CK_GROUP_N=2), zigzag (TD=11), rowmajor (TD=13), HYB_INGH (TD=32),
  HYB_CHET (TD=30), HYB_PMIX (TD=31), gflip (TD=33), tn2br (TD=34),
  ncycle (TD=14), ncyrot (TD=21), nflat (TD=15), nsnake (TD=16), nlock (TD=17).

Plus several proposals tested only here (see propose_*).

Usage:
    python3 tools/analyze_swizzle.py
    python3 tools/analyze_swizzle.py --csv data/swizzle_metrics.csv
    python3 tools/analyze_swizzle.py --only dgsw,ingh,gflip
    python3 tools/analyze_swizzle.py --plot   (writes PNGs if matplotlib avail)
"""
import argparse
import csv
import math
import statistics
import sys
from collections import Counter, defaultdict


SM_COUNT  = 148
NC        = SM_COUNT // 2          # 74
TILES_M   = 3626
TILES_N   = 3
TOTAL     = TILES_M * TILES_N      # 10878
TICKS     = TOTAL // NC            # 147

L2_BYTES_BUDGET = 125 * 1024 * 1024
A_STRIP_BYTES   = 256 * 3072       # cluster A: TM*2 × K, FP8 = 768 KB
B_STRIP_BYTES   = 256 * 3072       # B per tn: TN × K, FP8 = 768 KB


# ---------------------------------------------------------------- swizzles


def sw_dgsw(lin, G):
    group_tiles = TILES_N * G
    group_idx = lin // group_tiles
    first_m = group_idx * G
    in_g = lin - group_idx * group_tiles
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    tm_local = in_g % nig
    tn = in_g // nig
    return first_m + tm_local, tn


def sw_dgsnake(lin, G=8):
    group_tiles = TILES_N * G
    group_idx = lin // group_tiles
    first_m = group_idx * G
    in_g = lin - group_idx * group_tiles
    nig = min(G, TILES_M - first_m)
    local_m = in_g % nig
    local_n = in_g // nig
    if local_m & 1:
        local_n = TILES_N - 1 - local_n
    return first_m + local_m, local_n


def sw_checkered(lin, G_M=8, G_N=2):
    stripes_per_row = (TILES_N + G_N - 1) // G_N
    row_tiles = G_M * TILES_N
    row_group = lin // row_tiles
    in_row = lin - row_group * row_tiles
    first_m = row_group * G_M
    num_m = G_M if first_m + G_M <= TILES_M else TILES_M - first_m
    full_ss = num_m * G_N
    interior_total = (stripes_per_row - 1) * full_ss
    if stripes_per_row == 1 or in_row < interior_total:
        stripe = 0 if stripes_per_row == 1 else in_row // full_ss
        in_stripe = in_row - stripe * full_ss
    else:
        stripe = stripes_per_row - 1
        in_stripe = in_row - interior_total
    tm = first_m + in_stripe % num_m
    tn = stripe * G_N + in_stripe // num_m
    return tm, tn


def sw_zigzag(lin):
    tm = lin // TILES_N
    tn = lin % TILES_N
    if tm & 1:
        tn = TILES_N - 1 - tn
    return tm, tn


def sw_rowmajor(lin):
    return lin // TILES_N, lin % TILES_N


def sw_ingh(lin, G=8):
    group_tiles = TILES_N * G
    group_idx = lin // group_tiles
    first_m = group_idx * G
    in_g = lin - group_idx * group_tiles
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    if (group_idx & 1) == 0:
        tm_local = in_g % nig
        tn = in_g // nig
    else:
        tm_local = in_g // TILES_N
        tn = in_g % TILES_N
        if tm_local >= nig:
            tm_local = nig - 1
    return first_m + tm_local, tn


def sw_chet(lin, G=8):
    NC_HALF = NC // 2
    TM_HALF = TILES_M // 2
    c = lin % NC
    tt = lin // NC
    if c < NC_HALF:
        sub_lin = c + tt * NC_HALF
        gt = TILES_N * G
        group_idx = sub_lin // gt
        first_m = group_idx * G
        in_g = sub_lin - group_idx * gt
        nig = G if first_m + G <= TM_HALF else TM_HALF - first_m
        tm_local = in_g % nig
        tn = in_g // nig
        return first_m + tm_local, tn
    else:
        sub_lin = (c - NC_HALF) + tt * NC_HALF
        tm_local = sub_lin // TILES_N
        tn = sub_lin - tm_local * TILES_N
        return TM_HALF + tm_local, tn


def sw_pmix(lin, G=8):
    PHASE1 = TILES_M * (TILES_N - 1)
    if lin < PHASE1:
        TN_P1 = TILES_N - 1
        gt = TN_P1 * G
        group_idx = lin // gt
        first_m = group_idx * G
        in_g = lin - group_idx * gt
        nig = G if first_m + G <= TILES_M else TILES_M - first_m
        tm_local = in_g % nig
        tn = in_g // nig
        return first_m + tm_local, tn
    sub_lin = lin - PHASE1
    return sub_lin, TILES_N - 1


def sw_gflip(lin, G=8):
    gt = TILES_N * G
    num_groups = (TILES_M + G - 1) // G
    group_idx = lin // gt
    in_g = lin - group_idx * gt
    paired = group_idx ^ 1
    if paired < num_groups:
        group_idx = paired
    first_m = group_idx * G
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    tm_local = in_g % nig
    tn = in_g // nig
    return first_m + tm_local, tn


def sw_tn2br(lin, G=8):
    gt = TILES_N * G
    group_idx = lin // gt
    first_m = group_idx * G
    in_g = lin - group_idx * gt
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    tn = in_g // nig
    tm_local = in_g - tn * nig
    if tn == TILES_N - 1 and nig == 8:
        tm_local = ((tm_local & 1) << 2) | (tm_local & 2) | ((tm_local >> 2) & 1)
    return first_m + tm_local, tn


def sw_ncycle(lin):
    c = lin % NC
    _ti = lin // NC
    super_ = _ti // TILES_N
    tn = _ti % TILES_N
    tm = super_ * NC + c
    return min(tm, TILES_M - 1), tn


def sw_ncyrot(lin):
    c = lin % NC
    _ti = lin // NC
    super_ = _ti // TILES_N
    sub = _ti % TILES_N
    tn_shift = (1 if c >= 25 else 0) + (1 if c >= 50 else 0)
    tn = sub + tn_shift
    if tn >= TILES_N:
        tn -= TILES_N
    tm = super_ * NC + c
    return min(tm, TILES_M - 1), tn


def sw_nflat(lin):
    BAND = (NC + TILES_N - 1) // TILES_N
    c = lin % NC
    _ti = lin // NC
    band = c // BAND
    band_lane = c % BAND
    super_ = _ti // TILES_N
    sub = _ti % TILES_N
    tn = (sub + band) % TILES_N
    tm = super_ * NC + band * BAND + band_lane
    return min(tm, TILES_M - 1), tn


def sw_nsnake(lin):
    c = lin % NC
    _ti = lin // NC
    super_ = _ti // TILES_N
    sub = _ti % TILES_N
    tn = (TILES_N - 1 - sub) if (super_ & 1) else sub
    tm = super_ * NC + c
    return min(tm, TILES_M - 1), tn


# ---------------------------------------------------------------- proposals
#
# Below: design experiments only — not built into the kernel. Each one tries
# a different lever that the metrics (below) suggest could help. These are
# the rows you'd promote into tile_dispatch.cuh after the sim picks winners.


def sw_dgsw_offset(lin, G=8, OFF=1):
    """dgsw with per-tick lin offset applied. Each tick rotates the
    cluster→group assignment by OFF positions. Bijection: cyclic
    permutation of lin space — preserved.
    Hypothesis: fixed cluster→group binding may pin a cluster's tm
    history into a narrow strip; an OFF != 0 rotates the binding
    cyclically so each cluster eventually visits all M-row positions
    within its assigned group sequence."""
    rotated = (lin + (lin // NC) * OFF) % TOTAL
    return sw_dgsw(rotated, G)


def sw_dg6(lin):
    """Maps the dg-curve at G=6 (between dg4 winner and dg8 baseline).
    74 = 4*(6*3) + 2 → 4 full groups + 2 spillover per wavefront.
    dgcurve was: G=4 wins by ~1.85 µs vs G=8/checkered/dgsnake/gflip cluster
    at n=100. G=6 should land in between."""
    return sw_dgsw(lin, 6)


def sw_dg3(lin):
    """G=3: 74 = 8*(3*3) + 2 → 8 full groups per wavefront. Smaller groups
    than dg4 → tighter cluster-multicast (lower uTm) at the cost of weaker
    cross-tick L2 reuse. Tests whether dg4 bottoms the curve or dg3/dg2 is
    smaller-still better."""
    return sw_dgsw(lin, 3)


def sw_xband(lin, G=4):
    """xband: TRUE bijective version — partition cluster IDs into TILES_N
    sub-bands {25, 25, 24}. Each sub-band b owns its own M-row slice,
    runs dgsw on a rectangular subgrid (TILES_M_b × TILES_N), but all
    sub-bands stamp DIFFERENT tn at any single tick by phase-locking.

    With 3 sub-bands and TILES_N=3, at every tick all three tn columns
    are simultaneously active across the wavefront, eliminating the
    'all 74 clusters wanting tn=k' queue.

    M-partition: rows [0, M1) for b=0, [M1, M2) for b=1, [M2, TILES_M)
    for b=2, where the band sizes are scaled to match band cluster
    counts (25, 25, 24).
    """
    c = lin % NC
    tt = lin // NC
    if c < 25:
        b, c_in_band, band_size = 0, c, 25
    elif c < 50:
        b, c_in_band, band_size = 1, c - 25, 25
    else:
        b, c_in_band, band_size = 2, c - 50, 24
    band_starts = [0, 25 * 147, 50 * 147]
    tiles_per_band = band_size * 147
    sub_lin = c_in_band + tt * band_size
    band_TM = TILES_M // 3
    if b == 2:
        band_TM = TILES_M - 2 * band_TM
    sub_TILES = band_TM * TILES_N
    sub_lin = sub_lin % sub_TILES
    sub_TILES_N = TILES_N
    G_eff = min(G, band_TM)
    gt = sub_TILES_N * G_eff
    group_idx = sub_lin // gt
    first_m_local = group_idx * G_eff
    in_g = sub_lin - group_idx * gt
    nig = G_eff if first_m_local + G_eff <= band_TM else band_TM - first_m_local
    tm_local = in_g % nig
    tn = in_g // nig
    tm_global = (TILES_M // 3) * b + first_m_local + tm_local
    if tm_global >= TILES_M:
        tm_global = TILES_M - 1
    return tm_global, tn


def sw_dgsw_tnrot_g4(lin):
    """dg4 with within-group diagonal tn rotation (= dgnrot for G=4).
    Within each full M-group, the 12 (lm, ln) pairs map to
    tn = (ln + lm) mod TILES_N so the 4 cluster-instances at different
    lm values visit 4 different (lm, tn) combos that distribute evenly
    across all TILES_N columns simultaneously. Bijective by the diagonal
    argument in TD=24."""
    G = 4
    gt = TILES_N * G
    group_idx = lin // gt
    first_m = group_idx * G
    in_g = lin - group_idx * gt
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    lm = in_g % nig
    ln = in_g // nig
    tn = (ln + lm) % TILES_N
    return first_m + lm, tn


def sw_pingpong(lin, G=4):
    """pingpong: dg4 with a tn-stride mod-3 in odd groups.
    Half-and-half between vanilla dg4 (good) and dg-shifted-tn (different
    cluster→tn assignment) so consecutive groups don't lockstep on tn.
    Bijection: trivial — within each group it's a tile-set-preserving
    permutation."""
    gt = TILES_N * G
    group_idx = lin // gt
    first_m = group_idx * G
    in_g = lin - group_idx * gt
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    lm = in_g % nig
    ln = in_g // nig
    if group_idx & 1:
        tn = (ln + 1) % TILES_N
    else:
        tn = ln
    return first_m + lm, tn


# ---------------------------------------------------------------- metrics


def collect_traces(swizzle_fn):
    """Returns per_cluster_tm[c][tt], per_cluster_tn[c][tt], wavefront_tms[tt],
    wavefront_tns[tt]."""
    per_cluster_tm = [[0] * TICKS for _ in range(NC)]
    per_cluster_tn = [[0] * TICKS for _ in range(NC)]
    wavefront_tms = [[] for _ in range(TICKS)]
    wavefront_tns = [[] for _ in range(TICKS)]
    for tt in range(TICKS):
        for c in range(NC):
            lin = c + tt * NC
            tm, tn = swizzle_fn(lin)
            per_cluster_tm[c][tt] = tm
            per_cluster_tn[c][tt] = tn
            wavefront_tms[tt].append(tm)
            wavefront_tns[tt].append(tn)
    return per_cluster_tm, per_cluster_tn, wavefront_tms, wavefront_tns


def avg_intra_run_length(seq):
    """Mean run length of equal-valued consecutive elements."""
    if not seq:
        return 0.0
    runs = []
    cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return statistics.mean(runs)


def total_intra_runs(seqs):
    """List of all run lengths across all sequences (for histogram)."""
    runs = []
    for seq in seqs:
        if not seq:
            continue
        cur = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i - 1]:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)
    return runs


def shannon_entropy(seq):
    cnts = Counter(seq)
    n = len(seq)
    ent = 0.0
    for v in cnts.values():
        p = v / n
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def pearson_corr_abs(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sx2 = sum((x - mx) ** 2 for x in xs)
    sy2 = sum((y - my) ** 2 for y in ys)
    if sx2 == 0 or sy2 == 0:
        return 0.0
    return abs(sxy / math.sqrt(sx2 * sy2))


def adjacent_diff(seq):
    if len(seq) < 2:
        return 0.0
    return sum(abs(seq[i + 1] - seq[i]) for i in range(len(seq) - 1)) / (len(seq) - 1)


def cross_cluster_l2_reuse(per_cluster_tm, window):
    """For each tick tt, how many of the 74 (cluster, tm) pairs at tt have
    their tm appear in any cluster's required-tm set across ticks
    [tt-window, tt-1]? Fraction = warm-from-L2-window."""
    warm_counts = []
    for tt in range(window, TICKS):
        recent = set()
        for w in range(tt - window, tt):
            for c in range(NC):
                recent.add(per_cluster_tm[c][w])
        cur_warm = sum(1 for c in range(NC) if per_cluster_tm[c][tt] in recent)
        warm_counts.append(cur_warm / NC)
    return statistics.mean(warm_counts) if warm_counts else 0.0


def metrics(swizzle_fn, name):
    pcm, pcn, wfm, wfn = collect_traces(swizzle_fn)

    intra_tm_runs = [avg_intra_run_length(pcm[c]) for c in range(NC)]
    intra_tn_runs = [avg_intra_run_length(pcn[c]) for c in range(NC)]

    wf_unique_tm = [len(set(wfm[tt])) for tt in range(TICKS)]
    wf_unique_tn = [len(set(wfn[tt])) for tt in range(TICKS)]

    wf_tm_mult = [
        sum(v for v in Counter(wfm[tt]).values()) / len(set(wfm[tt]))
        for tt in range(TICKS)
    ]
    wf_tn_mult = [
        sum(v for v in Counter(wfn[tt]).values()) / len(set(wfn[tt]))
        for tt in range(TICKS)
    ]

    fresh_per_tick = []
    seen_tm = set()
    for tt in range(TICKS):
        new = sum(1 for tm in set(wfm[tt]) if tm not in seen_tm)
        fresh_per_tick.append(new)
        seen_tm |= set(wfm[tt])

    l2_warm_w8 = cross_cluster_l2_reuse(pcm, window=8)
    l2_warm_w24 = cross_cluster_l2_reuse(pcm, window=24)

    tm_extent_per_tick = []
    for tt in range(TICKS):
        s = wfm[tt]
        tm_extent_per_tick.append(max(s) - min(s) + 1)
    tm_extent_mean = statistics.mean(tm_extent_per_tick)

    tm_density_per_tick = [
        len(set(wfm[tt])) / (max(wfm[tt]) - min(wfm[tt]) + 1)
        for tt in range(TICKS)
    ]
    tm_density_mean = statistics.mean(tm_density_per_tick)

    same_tn_max_mass = []
    for tt in range(TICKS):
        cnts = Counter(wfn[tt])
        same_tn_max_mass.append(max(cnts.values()) / NC)
    same_tn_mass_mean = statistics.mean(same_tn_max_mass)
    same_tn_mass_max = max(same_tn_max_mass)

    tick_irregularity = []
    for tt in range(1, TICKS):
        delta_unique_tm = abs(len(set(wfm[tt])) - len(set(wfm[tt - 1])))
        tick_irregularity.append(delta_unique_tm)
    tick_irregularity_mean = (
        statistics.mean(tick_irregularity) if tick_irregularity else 0.0
    )
    tick_irregularity_p95 = (
        sorted(tick_irregularity)[int(0.95 * len(tick_irregularity))]
        if tick_irregularity else 0
    )

    cluster_ids = list(range(NC))
    wf_tn_entropy = [shannon_entropy(wfn[tt]) for tt in range(TICKS)]
    wf_tm_entropy = [shannon_entropy(wfm[tt]) for tt in range(TICKS)]
    cluster_tn_corr = [pearson_corr_abs(cluster_ids, wfn[tt]) for tt in range(TICKS)]
    cluster_tm_corr = [pearson_corr_abs(cluster_ids, wfm[tt]) for tt in range(TICKS)]
    adj_tn_diff = [adjacent_diff(wfn[tt]) for tt in range(TICKS)]
    adj_tm_diff = [adjacent_diff(wfm[tt]) for tt in range(TICKS)]

    pair_same_tn = []
    for tt in range(TICKS):
        s = wfn[tt]
        n = len(s)
        same = 0
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += 1
                if s[i] == s[j]:
                    same += 1
        pair_same_tn.append(same / total if total else 0.0)

    cluster_tm_var = []
    for c in range(NC):
        seq = pcm[c]
        m = sum(seq) / len(seq)
        cluster_tm_var.append(sum((x - m) ** 2 for x in seq) / len(seq))
    cluster_tm_var_mean = statistics.mean(cluster_tm_var)

    per_cluster_tn_traj_dist = []
    for i in range(NC):
        for j in range(i + 1, min(i + 8, NC)):
            d = sum(1 for tt in range(TICKS) if pcn[i][tt] != pcn[j][tt]) / TICKS
            per_cluster_tn_traj_dist.append(d)
    pc_tn_traj_dist_mean = (
        statistics.mean(per_cluster_tn_traj_dist) if per_cluster_tn_traj_dist else 0.0
    )

    bijection_tiles = set()
    for c in range(NC):
        for tt in range(TICKS):
            bijection_tiles.add((pcm[c][tt], pcn[c][tt]))
    bijective = len(bijection_tiles) == TOTAL

    tm_tick_set = defaultdict(list)
    for c in range(NC):
        for tt in range(TICKS):
            tm_tick_set[pcm[c][tt]].append(tt)
    spreads = []
    for tm, ticks in tm_tick_set.items():
        if len(ticks) < 2:
            continue
        ticks_sorted = sorted(ticks)
        span = ticks_sorted[-1] - ticks_sorted[0] + 1
        spreads.append(span / len(ticks))

    return {
        "name":             name,
        "bijective":        bijective,
        "intra_tm_run":     statistics.mean(intra_tm_runs),
        "intra_tn_run":     statistics.mean(intra_tn_runs),
        "wf_uniq_tm_mean":  statistics.mean(wf_unique_tm),
        "wf_uniq_tn_mean":  statistics.mean(wf_unique_tn),
        "wf_tm_mult_mean":  statistics.mean(wf_tm_mult),
        "wf_tn_mult_mean":  statistics.mean(wf_tn_mult),
        "wf_uniq_tm_max":   max(wf_unique_tm),
        "fresh_tm_p50":     statistics.median(fresh_per_tick),
        "fresh_tm_max":     max(fresh_per_tick),
        "fresh_tm_total":   sum(fresh_per_tick),
        "l2_warm_w8":       l2_warm_w8,
        "l2_warm_w24":      l2_warm_w24,
        "tm_tick_spread":   statistics.mean(spreads) if spreads else 0.0,
        "tm_extent_mean":   tm_extent_mean,
        "tm_density_mean":  tm_density_mean,
        "same_tn_mass_mean": same_tn_mass_mean,
        "same_tn_mass_max": same_tn_mass_max,
        "tick_irreg_mean":  tick_irregularity_mean,
        "tick_irreg_p95":   tick_irregularity_p95,
        "wf_tn_entropy":    statistics.mean(wf_tn_entropy),
        "wf_tm_entropy":    statistics.mean(wf_tm_entropy),
        "cluster_tn_corr":  statistics.mean(cluster_tn_corr),
        "cluster_tm_corr":  statistics.mean(cluster_tm_corr),
        "adj_tn_diff":      statistics.mean(adj_tn_diff),
        "adj_tm_diff":      statistics.mean(adj_tm_diff),
        "pair_same_tn":     statistics.mean(pair_same_tn),
        "cluster_tm_var":   cluster_tm_var_mean,
        "pc_tn_traj_dist":  pc_tn_traj_dist_mean,
    }


VARIANTS = [
    ("dgsw_G2",  lambda l: sw_dgsw(l, 2)),
    ("dgsw_G4",  lambda l: sw_dgsw(l, 4)),
    ("dgsw_G8",  lambda l: sw_dgsw(l, 8)),
    ("dgsw_G16", lambda l: sw_dgsw(l, 16)),
    ("dgsw_G32", lambda l: sw_dgsw(l, 32)),
    ("dgsnake",  sw_dgsnake),
    ("checkered", sw_checkered),
    ("zigzag",   sw_zigzag),
    ("rowmajor", sw_rowmajor),
    ("ingh",     sw_ingh),
    ("chet",     sw_chet),
    ("pmix",     sw_pmix),
    ("gflip",    sw_gflip),
    ("tn2br",    sw_tn2br),
    ("ncycle",   sw_ncycle),
    ("ncyrot",   sw_ncyrot),
    ("nflat",    sw_nflat),
    ("nsnake",   sw_nsnake),

    ("dgsw_G3",          sw_dg3),
    ("dgsw_G6",          sw_dg6),
    ("dgsw_off1",        lambda l: sw_dgsw_offset(l, 4, 1)),
    ("dgsw_off2",        lambda l: sw_dgsw_offset(l, 4, 2)),
    ("xband_G4",         lambda l: sw_xband(l, 4)),
    ("dg4_diag",         sw_dgsw_tnrot_g4),
    ("dg4_pingpong",     sw_pingpong),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="dump full metrics table to CSV")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of variant names")
    ap.add_argument("--summary", action="store_true",
                    help="emit one-line summary per variant only")
    args = ap.parse_args()

    selected = VARIANTS
    if args.only:
        wanted = set(args.only.split(","))
        selected = [v for v in VARIANTS if v[0] in wanted]
        if not selected:
            print(f"no variants matched: {args.only}", file=sys.stderr)
            sys.exit(1)

    rows = []
    for name, fn in selected:
        try:
            m = metrics(fn, name)
            rows.append(m)
        except Exception as e:
            print(f"  {name:14s} FAILED: {e}", file=sys.stderr)
            continue

    print(f"FC2 K=3072 BIAS_ONLY shape: TILES_M={TILES_M}, TILES_N={TILES_N}, "
          f"NC={NC}, TICKS={TICKS}")
    print(f"A_strip_per_cluster = {A_STRIP_BYTES/1024:.0f} KB, "
          f"B_strip_per_tn = {B_STRIP_BYTES/1024:.0f} KB")
    print()

    hdr1 = (
        f"{'name':14s}  {'bij':3s}  "
        f"{'tmRun':>5s}  {'tnRun':>5s}  "
        f"{'uTm':>5s}  {'uTn':>4s}  "
        f"{'tmMult':>6s}  "
        f"{'frP50':>5s}  {'frMax':>5s}  "
        f"{'L2w8':>5s}  "
        f"{'tmExt':>6s}  {'tmDen':>5s}  "
        f"{'tnQu':>5s}  "
        f"{'tIrM':>4s}  {'tIr95':>5s}"
    )
    print(hdr1)
    print("-" * len(hdr1))
    for m in rows:
        bij = "ok" if m["bijective"] else "X"
        print(
            f"{m['name']:14s}  {bij:3s}  "
            f"{m['intra_tm_run']:5.2f}  {m['intra_tn_run']:5.2f}  "
            f"{m['wf_uniq_tm_mean']:5.1f}  {m['wf_uniq_tn_mean']:4.2f}  "
            f"{m['wf_tm_mult_mean']:6.2f}  "
            f"{m['fresh_tm_p50']:5.1f}  {m['fresh_tm_max']:5.0f}  "
            f"{m['l2_warm_w8']:5.3f}  "
            f"{m['tm_extent_mean']:6.0f}  {m['tm_density_mean']:5.3f}  "
            f"{m['same_tn_mass_mean']:5.3f}  "
            f"{m['tick_irreg_mean']:4.1f}  {m['tick_irreg_p95']:5d}"
        )

    print()
    print("Legend:")
    print("  bij    = bijective tile coverage (X = duplicate or missing tiles)")
    print("  tmRun  = mean intra-cluster run length of equal tm (consecutive same-A; matters only for SMEM-prefetch)")
    print("  tnRun  = mean intra-cluster run length of equal tn (consecutive same-B; B is L2-resident, ≥1 fine)")
    print("  uTm    = mean unique tm count per 74-wavefront (lower = denser within-wavefront A multicast)")
    print("  uTn    = mean unique tn count per wavefront (3.0 = ALL tn columns active per tick — best B parallelism)")
    print("  tmMult = mean clusters-per-shared-tm at one tick (high = each A strip multicasts to many clusters)")
    print("  frP50  = median fresh (never-seen) tm strips per wavefront — DRAM influx rate")
    print("  frMax  = peak fresh tm strips per wavefront — peak DRAM burst, large = stalls")
    print("  L2w8   = fraction of (cluster, tm) at tick T whose tm was issued in [T-8, T-1] — short-window L2 warmth")
    print("  tmExt  = mean (max_tm - min_tm + 1) per wavefront — small = tight DRAM page locality")
    print("  tmDen  = mean uniq_tm / tm_extent per wavefront — how 'dense' the wavefront's M-coverage is in M-space")
    print("  tnQu   = mean (peak same-tn mass)/74 per wavefront — high = many clusters queueing on same B-strip")
    print("  tIrM   = mean |Δuniq_tm| between consecutive ticks — high = bursty/irregular wavefront shape")
    print("  tIr95  = 95th percentile of |Δuniq_tm| — peak irregularity")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
