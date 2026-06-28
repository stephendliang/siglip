/*
swizzle_w3x.cuh — shared tile-swizzle templates for fc1_w3x / fc2_w3x

Extracted from fc2_w3x.cu / fc1_w3x.cu where these helpers were duplicated
verbatim (~1240 lines each). Swizzle math is dim-only; depends on
TILES_M / TILES_N / NUM_CLUSTERS / TILES_PER_CLUSTER from the includer's
geometry header. Include AFTER the dim defines.

Provides:
  - 48 per-(TD,DGG) swizzle templates (TD = 11, 13, 18, 19, 33..58, 80..99)
  - tile_swizzle_t<TD,DGG>(lin)  — single dispatch point used by the kernel
  - tile_in_group_t<TD,DGG>(lin) — group-membership helper for BIAS_PER_TILE

Pure header — every function is __device__ __forceinline__, fully expanded
at every call site. Including this header costs zero SASS vs the prior
in-line copy (verified via cuobjdump diff).
*/
#pragma once

/*
  Dispatch variants for fc2_w3x.  All swizzle helpers and the kernel are
  templated on (TD, DGG), so every variant compiles to its own SASS via
  explicit instantiation in main()'s VARIANT_TABLE.  A single binary now
  holds 20 fully-specialized kernels with byte-identical codegen to the
  per-build flow that preceded this refactor.

  Base: dgswizzle(DGG=G). Within each group of G*TILES_N tiles, iterate
  tm fastest then tn (tn-run length = G).  TD=0 (default) → dgswizzle.
  DG_ROT (legacy probe, 0-delta perf): rotate tn by group_idx — kept for
  the tn=0-surplus structural diagnosis (memory/project_w3x_tn0_in_g_structural.md).
*/
template<int DGG>
static __device__ __forceinline__
int dgswizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int group_idx = lin / group_tiles;
    const int first_m = group_idx * DGG;
    const int in_group = lin - group_idx * group_tiles;
    const int nig = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int tm_local = in_group % nig;
    const int tn_raw   = in_group / nig;
#ifdef DG_ROT
    int tn = tn_raw + group_idx;
    while (tn >= TILES_N) tn -= TILES_N;
#else
    const int tn = tn_raw;
#endif
    return (first_m + tm_local) * TILES_N + tn;
}

template<int DGG>
static __device__ __forceinline__
int dg_in_group_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    return lin - (lin / group_tiles) * group_tiles;
}

/*
  Variants TD=11 zigzag, TD=13 rowmajor, TD=18 checkered, TD=19 dgsnake
  lifted from tile_dispatch.cuh (which stays macro-based for fc1_w3 /
  fc2_w3) so all coexist as templates within fc2_w3x's TU.
*/
static __device__ __forceinline__
int zigzag_swizzle(int lin) {
    int tm = lin / TILES_N;
    int tn = lin - tm * TILES_N;
    if (tm >= TILES_M) tm = TILES_M - 1;
    if (tm & 1) tn = TILES_N - 1 - tn;
    return tm * TILES_N + tn;
}

static __device__ __forceinline__
int rowmajor_swizzle(int lin) {
    int tm = lin / TILES_N;
    int tn = lin - tm * TILES_N;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + tn;
}

#ifndef CK_GROUP_M
#define CK_GROUP_M 8
#endif
#ifndef CK_GROUP_N
#if TILES_N > 4
#define CK_GROUP_N 5
#else
#define CK_GROUP_N 2
#endif
#endif
static __device__ __forceinline__
int checkered_swizzle(int lin) {
    const int G_M = CK_GROUP_M;
    const int G_N = CK_GROUP_N;
    const int stripes_per_row = (TILES_N + G_N - 1) / G_N;
    const int row_tiles = G_M * TILES_N;
    const int row_group = lin / row_tiles;
    const int in_row = lin - row_group * row_tiles;
    const int first_m = row_group * G_M;
    int tm, tn;
    if (first_m + G_M <= TILES_M) {
        const int full_ss = G_M * G_N;
        const int interior_total = (stripes_per_row - 1) * full_ss;
        int stripe, in_stripe;
        if (stripes_per_row == 1 || in_row < interior_total) {
            stripe = (stripes_per_row == 1) ? 0 : (in_row / full_ss);
            in_stripe = in_row - stripe * full_ss;
        } else {
            stripe = stripes_per_row - 1;
            in_stripe = in_row - interior_total;
        }
        tm = first_m + in_stripe % G_M;
        tn = stripe * G_N + in_stripe / G_M;
    } else {
        const int num_m = TILES_M - first_m;
        const int denom_m = num_m > 0 ? num_m : 1;
        const int full_ss = denom_m * G_N;
        const int interior_total = (stripes_per_row - 1) * full_ss;
        int stripe, in_stripe;
        if (stripes_per_row == 1 || in_row < interior_total) {
            stripe = (stripes_per_row == 1) ? 0 : (in_row / full_ss);
            in_stripe = in_row - stripe * full_ss;
        } else {
            stripe = stripes_per_row - 1;
            in_stripe = in_row - interior_total;
        }
        tm = first_m + in_stripe % denom_m;
        tn = stripe * G_N + in_stripe / denom_m;
    }
    if (tm >= TILES_M) tm = TILES_M - 1;
    if (tn >= TILES_N) tn = TILES_N - 1;
    return tm * TILES_N + tn;
}

template<int DGG>
static __device__ __forceinline__
int dgsnake_swizzle(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int group_idx = lin / group_tiles;
    const int first_m = group_idx * DGG;
    const int in_group = lin - group_idx * group_tiles;
    const int num_in_group = min(DGG, TILES_M - first_m);
    const int local_m = in_group % num_in_group;
    int local_n = in_group / num_in_group;
    if (local_m & 1) local_n = TILES_N - 1 - local_n;
    int tm = first_m + local_m;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + local_n;
}

/*
  fc2_w3x-local probes (see git log for the design notes):
    33 gflip, 34 tn2br, 35 dg_diag, 36 dg_pingpong, 37 dg_rowmaj,
    38 dg_g4swap, 39 dg_lmrev, 40 dg_combo_ab, 41 dg_combo_ac,
    42 dg_tn_blk, 43 dg_sn_rot1, 44 dg_sn_rot2, 45 dg_lmsn.
  All but TD=42 are dgsw-shaped (tile_in_group_t = dg_in_group_t<DGG>);
  TD=42 has no group concept and returns 0.
*/
template<int DGG>
static __device__ __forceinline__
int gflip_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m  = group_idx * DGG;
    const int nig      = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int tm_local = in_group - (in_group / nig) * nig;
    const int tn       = in_group / nig;
    int tm = first_m + tm_local;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + tn;
}

template<int DGG>
static __device__ __forceinline__
int tn2br_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int group_idx   = lin / group_tiles;
    const int first_m     = group_idx * DGG;
    const int in_group    = lin - group_idx * group_tiles;
    const int nig         = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int tn          = in_group / nig;
    int tm_local          = in_group - tn * nig;
    if (tn == TILES_N - 1 && nig == 8) {
        tm_local = ((tm_local & 1) << 2) | (tm_local & 2) | ((tm_local >> 2) & 1);
    }
    int tm = first_m + tm_local;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + tn;
}

template<int DGG>
static __device__ __forceinline__
int dg_diag_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int group_idx   = lin / group_tiles;
    const int first_m     = group_idx * DGG;
    const int in_group    = lin - group_idx * group_tiles;
    const int nig         = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm          = in_group - (in_group / nig) * nig;
    const int ln          = in_group / nig;
    int tn = ln + lm;
    while (tn >= TILES_N) tn -= TILES_N;
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + tn;
}

template<int DGG>
static __device__ __forceinline__
int dg_pingpong_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int group_idx   = lin / group_tiles;
    const int first_m     = group_idx * DGG;
    const int in_group    = lin - group_idx * group_tiles;
    const int nig         = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm          = in_group - (in_group / nig) * nig;
    const int ln          = in_group / nig;
    int tn = ln + (group_idx & 1);
    while (tn >= TILES_N) tn -= TILES_N;
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + tn;
}

template<int DGG>
static __device__ __forceinline__
int dg_rowmaj_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int group_idx   = lin / group_tiles;
    const int first_m     = group_idx * DGG;
    const int in_group    = lin - group_idx * group_tiles;
    const int nig         = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int slots       = nig * TILES_N;
    const int ig_clamped  = (in_group < slots) ? in_group : slots - 1;
    const int tn          = ig_clamped - (ig_clamped / TILES_N) * TILES_N;
    const int lm          = ig_clamped / TILES_N;
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + tn;
}

template<int DGG>
static __device__ __forceinline__
int dg_g4swap_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - (lin / group_tiles) * group_tiles;
    const int paired      = group_idx ^ 3;
    if (paired < num_groups) group_idx = paired;
    const int first_m     = group_idx * DGG;
    const int nig         = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm          = in_group - (in_group / nig) * nig;
    const int ln          = in_group / nig;
    const int tn = ln;
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + tn;
}

template<int DGG>
static __device__ __forceinline__
int dg_lmrev_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int group_idx   = lin / group_tiles;
    const int first_m     = group_idx * DGG;
    const int in_group    = lin - group_idx * group_tiles;
    const int nig         = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw      = in_group - (in_group / nig) * nig;
    const int ln          = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        lm = ((lm_raw & 1) << 2) | (lm_raw & 2) | ((lm_raw >> 2) & 1);
    }
    const int tn = ln;
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + tn;
}

template<int DGG>
static __device__ __forceinline__
int dg_combo_ab_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - (lin / group_tiles) * group_tiles;
    const int paired      = group_idx ^ 3;
    if (paired < num_groups) group_idx = paired;
    const int first_m     = group_idx * DGG;
    const int nig         = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int slots       = nig * TILES_N;
    const int ig_clamped  = (in_group < slots) ? in_group : slots - 1;
    const int tn          = ig_clamped - (ig_clamped / TILES_N) * TILES_N;
    const int lm          = ig_clamped / TILES_N;
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + tn;
}

template<int DGG>
static __device__ __forceinline__
int dg_combo_ac_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int group_idx   = lin / group_tiles;
    const int first_m     = group_idx * DGG;
    const int in_group    = lin - group_idx * group_tiles;
    const int nig         = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int slots       = nig * TILES_N;
    const int ig_clamped  = (in_group < slots) ? in_group : slots - 1;
    const int tn          = ig_clamped - (ig_clamped / TILES_N) * TILES_N;
    const int lm_raw      = ig_clamped / TILES_N;
    int lm = lm_raw;
    if (nig == 8) {
        lm = ((lm_raw & 1) << 2) | (lm_raw & 2) | ((lm_raw >> 2) & 1);
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + tn;
}

static __device__ __forceinline__
int dg_tn_blk_swizzle(int lin) {
    const int tn = lin / TILES_M;
    const int tm = lin - tn * TILES_M;
    return tm * TILES_N + tn;
}

template<int DGG>
static __device__ __forceinline__
int dg_sn_rot1_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int group_idx   = lin / group_tiles;
    const int first_m     = group_idx * DGG;
    const int in_group    = lin - group_idx * group_tiles;
    const int nig         = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm          = in_group - (in_group / nig) * nig;
    const int ln          = in_group / nig;
    int tn = (lm & 1) ? (ln + 1) : ln;
    if (tn >= TILES_N) tn -= TILES_N;
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + tn;
}

template<int DGG>
static __device__ __forceinline__
int dg_sn_rot2_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int group_idx   = lin / group_tiles;
    const int first_m     = group_idx * DGG;
    const int in_group    = lin - group_idx * group_tiles;
    const int nig         = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm          = in_group - (in_group / nig) * nig;
    const int ln          = in_group / nig;
    int tn = (lm & 1) ? (ln + 2) : ln;
    if (tn >= TILES_N) tn -= TILES_N;
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + tn;
}

template<int DGG>
static __device__ __forceinline__
int dg_lmsn_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int group_idx   = lin / group_tiles;
    const int first_m     = group_idx * DGG;
    const int in_group    = lin - group_idx * group_tiles;
    const int nig         = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw      = in_group - (in_group / nig) * nig;
    const int ln          = in_group / nig;
    const int lm = (ln & 1) ? (nig - 1 - lm_raw) : lm_raw;
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/*
  TD=46..48 added 2026-04-27 round 3 to beat dgsnake.  Mechanism: dgsnake's
  lm-parity phase is the unique cluster-trajectory-invariant phasing on the
  lm axis (under in_g advancing by 2 per tick), capping it at 2 phases.
  The orthogonal axis we exploit here is the per-cluster TT (visit-order)
  permutation: cluster c's set of visited tiles is unchanged but the ORDER
  is shifted, so bijection holds automatically and intra_tn_run is
  preserved (the tn-stream shape per cluster is unchanged, just rotated).

    46 dg_phasesh3 — cluster c with c%3 phase shifts its tt-sequence by
                phase × 4 ticks (= one ln-block).  Cluster c=1 starts at
                ln=1, c=2 starts at ln=2.  At any wall tick, clusters
                split 25/25/24 across the 3 ln values — finer cross-
                cluster ln-spread than dgsnake's 2-way lm-parity split.
                intra_tn_run = 3.92 preserved.  Bijective by construction.
    47 dg_sn_phasesh3 — dgsnake + c%3 tt-phaseshift.  Six effective
                cluster phases across (lm%2, c%3).  Hypothesis: dgsnake's
                L2-adjacency win on ln-block transitions (2→1→0) plus
                phaseshift's cluster spread.  Bijective.
    48 dg_sn_lmrev — dgsnake × lm-bitrev (G=8).  3-bit reverse of lm
                gates dgsnake's tn-reversal.  Pulls lmrev's adj_tm_diff
                lever plus dgsnake's tn-reversal.  intra_tn_run drops to
                2.96 (lm-parity after bitrev = bit-2 of original lm,
                breaking dgsnake's lm-parity invariance).  Likely a
                LOSS but informative: tests whether adj_tm_diff
                compensates for tnRun degradation.  Bijective.
*/
template<int DGG>
static __device__ __forceinline__
int dg_phasesh3_swizzle_t(int lin) {
    const int c = lin - (lin / NUM_CLUSTERS) * NUM_CLUSTERS;  // lin % NC
    const int tt = lin / NUM_CLUSTERS;
    const int phase = c - (c / 3) * 3;  // c % 3
    int eff_tt = tt + phase * 4;
    if (eff_tt >= TILES_PER_CLUSTER) eff_tt -= TILES_PER_CLUSTER;
    const int eff_lin = c + eff_tt * NUM_CLUSTERS;
    return dgswizzle_t<DGG>(eff_lin);
}

template<int DGG>
static __device__ __forceinline__
int dg_sn_phasesh3_swizzle_t(int lin) {
    const int c = lin - (lin / NUM_CLUSTERS) * NUM_CLUSTERS;
    const int tt = lin / NUM_CLUSTERS;
    const int phase = c - (c / 3) * 3;
    int eff_tt = tt + phase * 4;
    if (eff_tt >= TILES_PER_CLUSTER) eff_tt -= TILES_PER_CLUSTER;
    const int eff_lin = c + eff_tt * NUM_CLUSTERS;
    return dgsnake_swizzle<DGG>(eff_lin);
}

template<int DGG>
static __device__ __forceinline__
int dg_sn_lmrev_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int group_idx   = lin / group_tiles;
    const int first_m     = group_idx * DGG;
    const int in_group    = lin - group_idx * group_tiles;
    const int nig         = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw      = in_group - (in_group / nig) * nig;
    const int ln          = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        lm = ((lm_raw & 1) << 2) | (lm_raw & 2) | ((lm_raw >> 2) & 1);
    }
    int tn = (lm & 1) ? (TILES_N - 1 - ln) : ln;
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + tn;
}

/*
  TD=49..51 added 2026-04-27 round 4 to disentangle the dgsnake +443-cyc
  in-front-tier lever vs dgsw_G8.  All bijective.  See bijection proof in
  /tmp/bij_test.py and structural metrics in tools/analyze_swizzle.py.

    49 dg_antisnake — flip ln direction on second half of lm range
                  within each group instead of dgsnake's lm-parity.  Tests
                  whether snake's lm-parity choice or any flip is the
                  lever.  Predicted in-hull near checkered/dgsw (FRONT
                  cluster).  intra_tn_run = 2.96 (vs dgsnake's 3.92).
    50 dg_tt_phase — like dg_phasesh3 but phase keyed on cluster-band
                  (c // 25) % 3.  Three contiguous bands of ~25 clusters
                  share a phase.  Lands at adj_tm_diff = 6.72 (in the gap
                  between labeled hull max 3.75 and dg_phasesh3's 251.7),
                  exploring the moderate-extrapolation regime.
    51 wfd_latin — per-cluster tt-rotation by c.  Each cluster's coverage
                  is unchanged (cluster-trajectory invariant) but rotation
                  amount varies per cluster.  Targets cluster_tn_corr
                  below dgsnake's 0.0005.  uTm=74 (every cluster unique
                  tm per tick) — likely regression (control).
*/
template<int DGG>
static __device__ __forceinline__
int dg_antisnake_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int group_idx   = lin / group_tiles;
    const int first_m     = group_idx * DGG;
    const int in_group    = lin - group_idx * group_tiles;
    const int nig         = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm          = in_group - (in_group / nig) * nig;
    int ln                = in_group / nig;
    if (lm >= nig / 2) ln = TILES_N - 1 - ln;
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

template<int DGG>
static __device__ __forceinline__
int dg_tt_phase_swizzle_t(int lin) {
    const int c = lin - (lin / NUM_CLUSTERS) * NUM_CLUSTERS;
    const int tt = lin / NUM_CLUSTERS;
    const int band = c / 25;
    const int phase = band - (band / 3) * 3;
    int eff_tt = tt + phase * 4;
    if (eff_tt >= TILES_PER_CLUSTER) eff_tt -= TILES_PER_CLUSTER;
    const int eff_lin = c + eff_tt * NUM_CLUSTERS;
    return dgswizzle_t<DGG>(eff_lin);
}

template<int DGG>
static __device__ __forceinline__
int wfd_latin_swizzle_t(int lin) {
    const int c = lin - (lin / NUM_CLUSTERS) * NUM_CLUSTERS;
    const int tt = lin / NUM_CLUSTERS;
    int eff_tt = tt + c;
    while (eff_tt >= TILES_PER_CLUSTER) eff_tt -= TILES_PER_CLUSTER;
    const int eff_lin = c + eff_tt * NUM_CLUSTERS;
    return dgswizzle_t<DGG>(eff_lin);
}

/*
  TD=52..53 added 2026-04-28 from bloom_filter score-vs-gflip + analyzer
  feature signatures.  Both compose gflip's group_idx XOR 1 swap (pair-axis:
  cluster_tm_corr 0.94→0.65) with a within-group transform.

    52 gflip_lmrev — gflip + 3-bit reverse on lm at G=8.  Pushes adj_tm_diff
                    2.11 → 4.36 (sign-stable τ=-0.50, predicts faster).
                    Bloom score = +1.46 vs gflip → WORTHY.
    53 gflip_snrot — gflip + snrot2's tn rotation on odd lm.  Pushes
                    adj_tn_diff 0.17 → 1.33.  τ not sign-stable globally,
                    but matches snrot2's empirical 2nd-place signature at
                    n=32768.  Bloom score ≈ 0 → BUILD-ANYWAY.
                    Boundary guard: snrot only applied for full-DGG groups
                    AND in-bounds ln, falls back to plain gflip otherwise
                    (avoids (ln+2)%3 collisions when in_g overflows nig*TN).
*/
template<int DGG>
static __device__ __forceinline__
int gflip_lmrev_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        lm = ((lm_raw & 1) << 2) | (lm_raw & 2) | ((lm_raw >> 2) & 1);
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

template<int DGG>
static __device__ __forceinline__
int gflip_snrot_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm      = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int tn = ln;
    if (nig == DGG && ln < TILES_N && (lm & 1)) {
        tn = ln + 2;
        if (tn >= TILES_N) tn -= TILES_N;
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + tn;
}

/* TD=54: gflip_blkswap.  gflip XOR=1 + lm^4 on every other group (block-swap
                of upper/lower halves of a DGG=8 group: {0..7}→{4..7,0..3}).
                Bloom filter WORTHY (+0.31): adj_tm_diff +0.12, no overshoot.
                Per-lm bijection trivial (lm^4 is a permutation on [0,8)).
*/
template<int DGG>
static __device__ __forceinline__
int gflip_blkswap_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8 && (group_idx & 1)) lm = lm_raw ^ 4;
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/* TD=55: gflip_cidperm.  Permute cluster_id via c' = (c*15) % 74 (gcd(15,74)=1
                → bijection on [0,74)) before plugging into gflip's body.
                Wavefront SET unchanged at every tick — only the cluster→tile
                slot mapping shifts.  Probes SM→L2-partition affinity (the
                BLIND axis cluster_swizzle.py flagged) without disturbing
                wavefront geometry.  Bloom filter overshoot-flagged because
                cluster_tm_corr drops to 0.16 (well past g4swap's 0.63), but
                tm_extent_mean is identical to gflip's 38.90 — the metric
                set genuinely cannot rank this axis.  Either a big win or
                a cautionary case.
*/
template<int DGG>
static __device__ __forceinline__
int gflip_cidperm_swizzle_t(int lin) {
    const int c   = lin - (lin / NUM_CLUSTERS) * NUM_CLUSTERS;  // lin % NC
    const int tt  = lin / NUM_CLUSTERS;
    const int cp_full = c * 15;
    const int cp  = cp_full - (cp_full / NUM_CLUSTERS) * NUM_CLUSTERS;
    const int eff_lin = cp + tt * NUM_CLUSTERS;
    return gflip_swizzle_t<DGG>(eff_lin);
}

/* TD=56: gflip_blklmrev.  Stack of lmrev (uniform lm bit-reverse on every
                group) + blkswap (^4 on alt groups).  Asks whether m-axis
                (adj_tm_diff) composes or saturates: blkswap and lmrev each
                tied for first at n=43910, this stacks both mechanisms.
                Bloom WORTHY (+1.53, edges lmrev's +1.46 by 0.07).  Bijection:
                bitrev × XOR_alt = permutation × permutation = permutation.
*/
template<int DGG>
static __device__ __forceinline__
int gflip_blklmrev_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        lm = ((lm_raw & 1) << 2) | (lm_raw & 2) | ((lm_raw >> 2) & 1);
        if (group_idx & 1) lm ^= 4;
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/* TD=57: gflip_blkmul3.  Same alt-group structure as blkswap, but uses a
                multiplicative permutation lm = (lm * 3) % 8 instead of XOR ^4.
                gcd(3,8)=1 → bijection {0..7} → {0,3,6,1,4,7,2,5}.  More
                disruptive than ^4: ^4 swaps two halves; *3 mod 8 scrambles
                the order within.  Bloom WORTHY (+0.67).  Tests "is the
                SPECIFIC lm^4 in blkswap optimal or any decorrelation works."
*/
template<int DGG>
static __device__ __forceinline__
int gflip_blkmul3_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8 && (group_idx & 1)) lm = (lm_raw * 3) & 7;  // mod 8
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/* TD=58: gflip_quartswap.  Lighter blkswap — apply lm^4 only on group_idx %
                4 == 1 (every 4th group), not every 2nd.  Halves the
                perturbation density: 1/4 of paired groups have the ^4 twist
                vs blkswap's 1/2.  Bloom WORTHY (+0.15, just above +0.10
                threshold).  Calibration probe: if quartswap > blkswap, the
                density optimum is finer; if quartswap < blkswap, blkswap is
                already at the right density.
*/
template<int DGG>
static __device__ __forceinline__
int gflip_quartswap_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8 && (group_idx & 3) == 1) lm = lm_raw ^ 4;
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

// BEGIN COORD_DESCEND

/* TD=80: gflip_xk2_blkswap.  XK=2 pairing × blkswap-^4 alt1 (vs gflip's XK=1)
            XK=2, p_u=id, alt=xor4, dens=alt1.  Bijection-checked. */
template<int DGG>
static __device__ __forceinline__
int gflip_xk2_blkswap_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 2;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        if ((group_idx & 1)) { lm = lm ^ 4; }
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/* TD=81: gflip_xk3_blkswap.  XK=3 pairing × blkswap-^4 alt1 (vs gflip's XK=1)
            XK=3, p_u=id, alt=xor4, dens=alt1.  Bijection-checked. */
template<int DGG>
static __device__ __forceinline__
int gflip_xk3_blkswap_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 3;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        if ((group_idx & 1)) { lm = lm ^ 4; }
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/* TD=82: gflip_xk5_blkswap.  XK=5 pairing × blkswap-^4 alt1 (vs gflip's XK=1)
            XK=5, p_u=id, alt=xor4, dens=alt1.  Bijection-checked. */
template<int DGG>
static __device__ __forceinline__
int gflip_xk5_blkswap_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 5;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        if ((group_idx & 1)) { lm = lm ^ 4; }
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/* TD=83: gflip_xk7_blkswap.  XK=7 pairing × blkswap-^4 alt1 (vs gflip's XK=1)
            XK=7, p_u=id, alt=xor4, dens=alt1.  Bijection-checked. */
template<int DGG>
static __device__ __forceinline__
int gflip_xk7_blkswap_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 7;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        if ((group_idx & 1)) { lm = lm ^ 4; }
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/* TD=84: gflip_blkx1.  alt1 × xor1 (vs blkswap's xor4) — alt-mask scan
            XK=1, p_u=id, alt=xor1, dens=alt1.  Bijection-checked. */
template<int DGG>
static __device__ __forceinline__
int gflip_blkx1_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        if ((group_idx & 1)) { lm = lm ^ 1; }
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/* TD=85: gflip_blkx2.  alt1 × xor2 (vs blkswap's xor4) — alt-mask scan
            XK=1, p_u=id, alt=xor2, dens=alt1.  Bijection-checked. */
template<int DGG>
static __device__ __forceinline__
int gflip_blkx2_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        if ((group_idx & 1)) { lm = lm ^ 2; }
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/* TD=86: gflip_blkx3.  alt1 × xor3 (vs blkswap's xor4) — alt-mask scan
            XK=1, p_u=id, alt=xor3, dens=alt1.  Bijection-checked. */
template<int DGG>
static __device__ __forceinline__
int gflip_blkx3_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        if ((group_idx & 1)) { lm = lm ^ 3; }
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/* TD=87: gflip_blkx5.  alt1 × xor5 (vs blkswap's xor4) — alt-mask scan
            XK=1, p_u=id, alt=xor5, dens=alt1.  Bijection-checked. */
template<int DGG>
static __device__ __forceinline__
int gflip_blkx5_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        if ((group_idx & 1)) { lm = lm ^ 5; }
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/* TD=88: gflip_blkx6.  alt1 × xor6 (vs blkswap's xor4) — alt-mask scan
            XK=1, p_u=id, alt=xor6, dens=alt1.  Bijection-checked. */
template<int DGG>
static __device__ __forceinline__
int gflip_blkx6_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        if ((group_idx & 1)) { lm = lm ^ 6; }
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/* TD=89: gflip_blkx7.  alt1 × xor7 (vs blkswap's xor4) — alt-mask scan
            XK=1, p_u=id, alt=xor7, dens=alt1.  Bijection-checked. */
template<int DGG>
static __device__ __forceinline__
int gflip_blkx7_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        if ((group_idx & 1)) { lm = lm ^ 7; }
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/* TD=90: gflip_blk_alt0.  xor4 × density=alt0 (vs blkswap's alt1, quartswap's qrt1)
            XK=1, p_u=id, alt=xor4, dens=alt0.  Bijection-checked. */
template<int DGG>
static __device__ __forceinline__
int gflip_blk_alt0_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        if (((group_idx & 1) == 0)) { lm = lm ^ 4; }
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/* TD=91: gflip_blk_qrt0.  xor4 × density=qrt0 (vs blkswap's alt1, quartswap's qrt1)
            XK=1, p_u=id, alt=xor4, dens=qrt0.  Bijection-checked. */
template<int DGG>
static __device__ __forceinline__
int gflip_blk_qrt0_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        if (((group_idx & 3) == 0)) { lm = lm ^ 4; }
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/* TD=92: gflip_blk_qrt2.  xor4 × density=qrt2 (vs blkswap's alt1, quartswap's qrt1)
            XK=1, p_u=id, alt=xor4, dens=qrt2.  Bijection-checked. */
template<int DGG>
static __device__ __forceinline__
int gflip_blk_qrt2_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        if (((group_idx & 3) == 2)) { lm = lm ^ 4; }
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/* TD=93: gflip_blk_qrt3.  xor4 × density=qrt3 (vs blkswap's alt1, quartswap's qrt1)
            XK=1, p_u=id, alt=xor4, dens=qrt3.  Bijection-checked. */
template<int DGG>
static __device__ __forceinline__
int gflip_blk_qrt3_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        if (((group_idx & 3) == 3)) { lm = lm ^ 4; }
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/* TD=94: gflip_blk_n3k1.  xor4 × density=n3k1 (vs blkswap's alt1, quartswap's qrt1)
            XK=1, p_u=id, alt=xor4, dens=n3k1.  Bijection-checked. */
template<int DGG>
static __device__ __forceinline__
int gflip_blk_n3k1_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        if (((group_idx - (group_idx / 3) * 3) == 1)) { lm = lm ^ 4; }
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/* TD=95: gflip_mul3.  uniform p_u=mul3 only (vs lmrev's bitrev)
            XK=1, p_u=mul3, alt=none, dens=all.  Bijection-checked. */
template<int DGG>
static __device__ __forceinline__
int gflip_mul3_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        lm = (lm_raw * 3) & 7;
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/* TD=96: gflip_mul5.  uniform p_u=mul5 only (vs lmrev's bitrev)
            XK=1, p_u=mul5, alt=none, dens=all.  Bijection-checked. */
template<int DGG>
static __device__ __forceinline__
int gflip_mul5_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        lm = (lm_raw * 5) & 7;
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/* TD=97: gflip_bitrev_xor1_alt1.  composition: uniform=bitrev + alt=xor1 dens=alt1
            XK=1, p_u=bitrev, alt=xor1, dens=alt1.  Bijection-checked. */
template<int DGG>
static __device__ __forceinline__
int gflip_bitrev_xor1_alt1_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        lm = ((lm_raw & 1) << 2) | (lm_raw & 2) | ((lm_raw >> 2) & 1);
        if ((group_idx & 1)) { lm = lm ^ 1; }
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/* TD=98: gflip_bitrev_xor2_alt1.  composition: uniform=bitrev + alt=xor2 dens=alt1
            XK=1, p_u=bitrev, alt=xor2, dens=alt1.  Bijection-checked. */
template<int DGG>
static __device__ __forceinline__
int gflip_bitrev_xor2_alt1_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        lm = ((lm_raw & 1) << 2) | (lm_raw & 2) | ((lm_raw >> 2) & 1);
        if ((group_idx & 1)) { lm = lm ^ 2; }
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

/* TD=99: gflip_mul3_xor4_alt1.  composition: uniform=mul3 + alt=xor4 dens=alt1
            XK=1, p_u=mul3, alt=xor4, dens=alt1.  Bijection-checked. */
template<int DGG>
static __device__ __forceinline__
int gflip_mul3_xor4_alt1_swizzle_t(int lin) {
    const int group_tiles = TILES_N * DGG;
    const int num_groups  = (TILES_M + DGG - 1) / DGG;
    int group_idx         = lin / group_tiles;
    const int in_group    = lin - group_idx * group_tiles;
    const int paired      = group_idx ^ 1;
    if (paired < num_groups) group_idx = paired;
    const int first_m = group_idx * DGG;
    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
    const int lm_raw  = in_group - (in_group / nig) * nig;
    const int ln      = in_group / nig;
    int lm = lm_raw;
    if (nig == 8) {
        lm = (lm_raw * 3) & 7;
        if ((group_idx & 1)) { lm = lm ^ 4; }
    }
    int tm = first_m + lm;
    if (tm >= TILES_M) tm = TILES_M - 1;
    return tm * TILES_N + ln;
}

// END COORD_DESCEND

// Single entry point — dispatched at compile time per kernel instantiation.
template<int TD, int DGG>
static __device__ __forceinline__
int tile_swizzle_t(int lin) {
    if constexpr (TD == 11) return zigzag_swizzle(lin);
    else if constexpr (TD == 13) return rowmajor_swizzle(lin);
    else if constexpr (TD == 18) return checkered_swizzle(lin);
    else if constexpr (TD == 19) return dgsnake_swizzle<DGG>(lin);
    else if constexpr (TD == 33) return gflip_swizzle_t<DGG>(lin);
    else if constexpr (TD == 34) return tn2br_swizzle_t<DGG>(lin);
    else if constexpr (TD == 35) return dg_diag_swizzle_t<DGG>(lin);
    else if constexpr (TD == 36) return dg_pingpong_swizzle_t<DGG>(lin);
    else if constexpr (TD == 37) return dg_rowmaj_swizzle_t<DGG>(lin);
    else if constexpr (TD == 38) return dg_g4swap_swizzle_t<DGG>(lin);
    else if constexpr (TD == 39) return dg_lmrev_swizzle_t<DGG>(lin);
    else if constexpr (TD == 40) return dg_combo_ab_swizzle_t<DGG>(lin);
    else if constexpr (TD == 41) return dg_combo_ac_swizzle_t<DGG>(lin);
    else if constexpr (TD == 42) return dg_tn_blk_swizzle(lin);
    else if constexpr (TD == 43) return dg_sn_rot1_swizzle_t<DGG>(lin);
    else if constexpr (TD == 44) return dg_sn_rot2_swizzle_t<DGG>(lin);
    else if constexpr (TD == 45) return dg_lmsn_swizzle_t<DGG>(lin);
    else if constexpr (TD == 46) return dg_phasesh3_swizzle_t<DGG>(lin);
    else if constexpr (TD == 47) return dg_sn_phasesh3_swizzle_t<DGG>(lin);
    else if constexpr (TD == 48) return dg_sn_lmrev_swizzle_t<DGG>(lin);
    else if constexpr (TD == 49) return dg_antisnake_swizzle_t<DGG>(lin);
    else if constexpr (TD == 50) return dg_tt_phase_swizzle_t<DGG>(lin);
    else if constexpr (TD == 51) return wfd_latin_swizzle_t<DGG>(lin);
    else if constexpr (TD == 52) return gflip_lmrev_swizzle_t<DGG>(lin);
    else if constexpr (TD == 53) return gflip_snrot_swizzle_t<DGG>(lin);
    else if constexpr (TD == 54) return gflip_blkswap_swizzle_t<DGG>(lin);
    else if constexpr (TD == 55) return gflip_cidperm_swizzle_t<DGG>(lin);
    else if constexpr (TD == 56) return gflip_blklmrev_swizzle_t<DGG>(lin);
    else if constexpr (TD == 57) return gflip_blkmul3_swizzle_t<DGG>(lin);
    else if constexpr (TD == 58) return gflip_quartswap_swizzle_t<DGG>(lin);
    // BEGIN COORD_DESCEND dispatch
    else if constexpr (TD == 80) return gflip_xk2_blkswap_swizzle_t<DGG>(lin);
    else if constexpr (TD == 81) return gflip_xk3_blkswap_swizzle_t<DGG>(lin);
    else if constexpr (TD == 82) return gflip_xk5_blkswap_swizzle_t<DGG>(lin);
    else if constexpr (TD == 83) return gflip_xk7_blkswap_swizzle_t<DGG>(lin);
    else if constexpr (TD == 84) return gflip_blkx1_swizzle_t<DGG>(lin);
    else if constexpr (TD == 85) return gflip_blkx2_swizzle_t<DGG>(lin);
    else if constexpr (TD == 86) return gflip_blkx3_swizzle_t<DGG>(lin);
    else if constexpr (TD == 87) return gflip_blkx5_swizzle_t<DGG>(lin);
    else if constexpr (TD == 88) return gflip_blkx6_swizzle_t<DGG>(lin);
    else if constexpr (TD == 89) return gflip_blkx7_swizzle_t<DGG>(lin);
    else if constexpr (TD == 90) return gflip_blk_alt0_swizzle_t<DGG>(lin);
    else if constexpr (TD == 91) return gflip_blk_qrt0_swizzle_t<DGG>(lin);
    else if constexpr (TD == 92) return gflip_blk_qrt2_swizzle_t<DGG>(lin);
    else if constexpr (TD == 93) return gflip_blk_qrt3_swizzle_t<DGG>(lin);
    else if constexpr (TD == 94) return gflip_blk_n3k1_swizzle_t<DGG>(lin);
    else if constexpr (TD == 95) return gflip_mul3_swizzle_t<DGG>(lin);
    else if constexpr (TD == 96) return gflip_mul5_swizzle_t<DGG>(lin);
    else if constexpr (TD == 97) return gflip_bitrev_xor1_alt1_swizzle_t<DGG>(lin);
    else if constexpr (TD == 98) return gflip_bitrev_xor2_alt1_swizzle_t<DGG>(lin);
    else if constexpr (TD == 99) return gflip_mul3_xor4_alt1_swizzle_t<DGG>(lin);
    // END COORD_DESCEND dispatch
    else return dgswizzle_t<DGG>(lin);
}

template<int TD, int DGG>
static __device__ __forceinline__
int tile_in_group_t(int lin) {
    if constexpr (TD == 11 || TD == 13 || TD == 18 || TD == 42) {
        (void)lin; return 0;
    } else {
        return dg_in_group_t<DGG>(lin);
    }
}
