/*
   Static tile swizzle for TD=8..16.

   Shared between fc1_w3.cu and fc2_w3.cu.  Both kernels stride through a
   linear block_idx = _ti * num_clusters + cluster_id; static_swizzle()
   remaps that to a flat tile_idx = tm * TILES_N + tn.

   Requires TILES_M, TILES_N, SM_COUNT to be #defined before inclusion.
*/

#pragma once

/* Legacy macro-selected swizzle for fc1_w3 / fc2_w3 TD=8..32. The gflip basin
   (TD=33..99) lives only in the templated static_swizzle_t below, so this
   ladder is bounded to <=32; an out-of-range TD (e.g. the new fc2_w3 default
   TD=54) skips it entirely and dispatches through static_swizzle_t instead. */
#if TILE_DISPATCH >= 8 && TILE_DISPATCH <= 32

#ifndef DG_GROUP_SIZE
#define DG_GROUP_SIZE 8
#endif

static __device__ __forceinline__ int static_swizzle(int block_idx) {
#if TILE_DISPATCH == 8
    /* DeepGEMM 2D swizzle: group DG_GROUP_SIZE M-blocks, sweep all N within
       group.  (m0,n0)(m1,n0)...(m7,n0)(m0,n1)...
       Hot path uses compile-time DG_GROUP_SIZE divisor so ptxas emits a
       shift, not MUFU.RCP.  Tail group (TILES_M % DG_GROUP_SIZE != 0) is
       the only case with a variable divisor. */
    const int group_tiles = TILES_N * DG_GROUP_SIZE;
    const int group_idx = block_idx / group_tiles;
    const int first_m = group_idx * DG_GROUP_SIZE;
    const int in_group = block_idx % group_tiles;
    if (first_m + DG_GROUP_SIZE <= TILES_M) {
        return (first_m + in_group % DG_GROUP_SIZE) * TILES_N
             + in_group / DG_GROUP_SIZE;
    }
    const int tail = TILES_M - first_m;
    return (first_m + in_group % tail) * TILES_N + in_group / tail;

#elif TILE_DISPATCH == 9
    /* Z-order (Morton): 4x4 super-blocks tiled over (TILES_M, TILES_N) in
       raster, traversed internally in Morton order.  TD=8-style tail
       handling keeps the map bijective for arbitrary TILES_M, TILES_N. */
    {
        const int ZB = 4;
        const int num_bn = (TILES_N + ZB - 1) / ZB;
        const int num_m_full = TILES_M / ZB;
        const int tail_m = TILES_M - num_m_full * ZB;
        const int last_bn_n = TILES_N - (num_bn - 1) * ZB;
        const int full_mgroup_tiles = ZB * TILES_N;
        const int full_m_total = num_m_full * full_mgroup_tiles;

        int bm, within_mg;
        if (block_idx < full_m_total) {
            bm = block_idx / full_mgroup_tiles;
            within_mg = block_idx % full_mgroup_tiles;
        } else {
            bm = num_m_full;
            within_mg = block_idx - full_m_total;
        }
        const int num_m = (bm < num_m_full) ? ZB : tail_m;

        int bn = 0;
        for (int b = 0; b < num_bn; b++) {
            int n_in_b = (b == num_bn - 1) ? last_bn_n : ZB;
            int tiles_in_b = num_m * n_in_b;
            if (within_mg < tiles_in_b) { bn = b; break; }
            within_mg -= tiles_in_b;
        }
        const int num_n = (bn == num_bn - 1) ? last_bn_n : ZB;

        int zr = 0, zc = 0, count = 0;
        for (int z = 0; z < 16; z++) {
            int r = ((z >> 1) & 1) | ((z >> 2) & 2);
            int col = (z & 1) | ((z >> 1) & 2);
            if (r >= num_m || col >= num_n) continue;
            if (count == within_mg) { zr = r; zc = col; break; }
            count++;
        }
        int tm = bm * ZB + zr;
        int tn = bn * ZB + zc;
        if (tm >= TILES_M) tm = TILES_M - 1;
        if (tn >= TILES_N) tn = TILES_N - 1;
        return tm * TILES_N + tn;
    }

#elif TILE_DISPATCH == 10
    /* Hilbert: 4x4 super-blocks tiled over (TILES_M, TILES_N) in raster,
       traversed internally along a 4x4 Hilbert curve.  Same TD=8-style
       tail handling as TD=9 to stay bijective for arbitrary dims. */
    {
        const int HB = 4;
        const int num_bn = (TILES_N + HB - 1) / HB;
        const int num_m_full = TILES_M / HB;
        const int tail_m = TILES_M - num_m_full * HB;
        const int last_bn_n = TILES_N - (num_bn - 1) * HB;
        const int full_mgroup_tiles = HB * TILES_N;
        const int full_m_total = num_m_full * full_mgroup_tiles;
        const int h_x[16] = {0,1,1,0,0,0,1,1,2,2,3,3,3,2,2,3};
        const int h_y[16] = {0,0,1,1,2,3,3,2,2,3,3,2,1,1,0,0};

        int bm, within_mg;
        if (block_idx < full_m_total) {
            bm = block_idx / full_mgroup_tiles;
            within_mg = block_idx % full_mgroup_tiles;
        } else {
            bm = num_m_full;
            within_mg = block_idx - full_m_total;
        }
        const int num_m = (bm < num_m_full) ? HB : tail_m;

        int bn = 0;
        for (int b = 0; b < num_bn; b++) {
            int n_in_b = (b == num_bn - 1) ? last_bn_n : HB;
            int tiles_in_b = num_m * n_in_b;
            if (within_mg < tiles_in_b) { bn = b; break; }
            within_mg -= tiles_in_b;
        }
        const int num_n = (bn == num_bn - 1) ? last_bn_n : HB;

        int hr = 0, hc = 0, count = 0;
        for (int d = 0; d < 16; d++) {
            int r = h_x[d], col = h_y[d];
            if (r >= num_m || col >= num_n) continue;
            if (count == within_mg) { hr = r; hc = col; break; }
            count++;
        }
        int tm = bm * HB + hr;
        int tn = bn * HB + hc;
        if (tm >= TILES_M) tm = TILES_M - 1;
        if (tn >= TILES_N) tn = TILES_N - 1;
        return tm * TILES_N + tn;
    }

#elif TILE_DISPATCH == 11
    /* Zigzag-N: row-major, reverse N direction on odd M-rows. */
    {
        int tm = block_idx / TILES_N;
        int tn = block_idx % TILES_N;
        if (tm >= TILES_M) tm = TILES_M - 1;
        if (tm & 1) tn = TILES_N - 1 - tn;
        return tm * TILES_N + tn;
    }

#elif TILE_DISPATCH == 12
    /* Column-first: all M-rows for n=0, then n=1, ... */
    {
        int tn = block_idx / TILES_M;
        int tm = block_idx % TILES_M;
        if (tn >= TILES_N) { tn = TILES_N - 1; tm = TILES_M - 1; }
        return tm * TILES_N + tn;
    }

#elif TILE_DISPATCH == 13
    /* Pure row-major. */
    {
        int tm = block_idx / TILES_N;
        int tn = block_idx % TILES_N;
        if (tm >= TILES_M) tm = TILES_M - 1;
        return tm * TILES_N + tn;
    }

#elif TILE_DISPATCH == 14
    /* Cluster-N-cycle: all NC clusters synchronized on same sub-tick, one B-
       column live at a time, 3x intra-cluster A reuse. */
    {
        const int NC = SM_COUNT / 2;
        const int c = block_idx % NC;
        const int _ti_local = block_idx / NC;
        const int super = _ti_local / TILES_N;
        const int tn = _ti_local % TILES_N;
        int tm = super * NC + c;
        if (tm >= TILES_M) tm = TILES_M - 1;
        return tm * TILES_N + tn;
    }

#elif TILE_DISPATCH == 15
    /* Banded N-flat: TD=14 split into TILES_N bands, each band holds its own
       tn offset so all tn active simultaneously. */
    {
        const int NC = SM_COUNT / 2;
        const int BAND_SIZE = (NC + TILES_N - 1) / TILES_N;
        const int c = block_idx % NC;
        const int _ti_local = block_idx / NC;
        const int band = c / BAND_SIZE;
        const int band_lane = c % BAND_SIZE;
        const int super = _ti_local / TILES_N;
        const int sub = _ti_local % TILES_N;
        const int tn = (sub + band) % TILES_N;
        int tm = super * NC + band * BAND_SIZE + band_lane;
        if (tm >= TILES_M) tm = TILES_M - 1;
        return tm * TILES_N + tn;
    }

#elif TILE_DISPATCH == 16
    /* nsnake: TD=14 with alternating tn direction per super-tick for
       cross-super page coherence. */
    {
        const int NC = SM_COUNT / 2;
        const int c = block_idx % NC;
        const int _ti_local = block_idx / NC;
        const int super = _ti_local / TILES_N;
        const int sub = _ti_local % TILES_N;
        const int tn = (super & 1) ? (TILES_N - 1 - sub) : sub;
        int tm = super * NC + c;
        if (tm >= TILES_M) tm = TILES_M - 1;
        return tm * TILES_N + tn;
    }

#elif TILE_DISPATCH == 17
    /*
    nlock (static-N cluster bind): each cluster is mostly-permanently bound
    to one N-column and sweeps M.  Bands are balanced: the first `extra`
    bands have ceil(NC/N) clusters, the rest have floor(NC/N).  When
    NC % N != 0, big bands' spill iterations cover the short (small) bands'
    tails so the map stays bijective.  Assumes tile_count * NC ==
    TILES_M * TILES_N.
    */
    {
        const int NC = SM_COUNT / 2;
        const int c = block_idx % NC;
        const int _ti = block_idx / NC;
        const int TC = (TILES_M * TILES_N) / NC;

        const int fcpb = NC / TILES_N;
        const int ex = NC - fcpb * TILES_N;
        const int bb = fcpb + 1;
        const int bcap = bb * ex;

        int tn_p, idx, bs;
        if (c < bcap) {
            tn_p = c / bb;
            idx = c - tn_p * bb;
            bs = bb;
        } else {
            tn_p = ex + (c - bcap) / fcpb;
            idx = (c - bcap) - (tn_p - ex) * fcpb;
            bs = fcpb;
        }

        const int pa_ideal = TILES_M / bs;
        const int pr = pa_ideal < TC ? pa_ideal : TC;
        const int own_uncov = TILES_M - pr * bs;

        int tm, tn;
        if (_ti < pr) {
            tm = _ti * bs + idx;
            tn = tn_p;
        } else {
            const int spill = (_ti - pr) * bs + idx;
            if (spill < own_uncov) {
                tm = pr * bs + spill;
                tn = tn_p;
            } else {
                const int help_local = spill - own_uncov;
                const int big_sto = (TC - pr) * bs - own_uncov;
                const int gh = tn_p * big_sto + help_local;
                const int spi = TILES_M / fcpb;
                const int spr = spi < TC ? spi : TC;
                const int su = TILES_M - spr * fcpb;
                const int so = gh / su;
                const int within = gh - so * su;
                tn = ex + so;
                tm = spr * fcpb + within;
            }
        }
        if (tm >= TILES_M) tm = TILES_M - 1;
        if (tn >= TILES_N) tn = TILES_N - 1;
        return tm * TILES_N + tn;
    }

#elif TILE_DISPATCH == 18
    /*
    checkered (2D M × N group): generalization of dgswizzle to a G_M × G_N
    block of tiles per group.  Within a row_group of G_M M-rows, tiles are
    laid out stripe-by-stripe where each stripe is a range of G_N N-columns
    (last stripe may be narrower when G_N does not divide TILES_N).  Within
    a stripe, traverse column-first so a cluster stays on one N-tile for
    G_M ticks, then advances N.

    Bijection across the full block_idx range requires:
      - row_tiles = G_M * TILES_N  (constant per row_group stride)
      - stripe stride scales with the current row_group's num_m so the
        M-tail row_group (num_m < G_M) still packs real tiles densely
        at the start of its slot range
      - last stripe width = TILES_N - (spr-1)*G_N when G_N does not divide
    */
    {
#ifndef CK_GROUP_M
#define CK_GROUP_M 8
#endif
#ifndef CK_GROUP_N
/* Default picks a G_N that does NOT divide typical TILES_N (3, 12). For
   TILES_N >= 5 use G_N=5 (stripes 5,5,2,... on TILES_N=12). For small
   TILES_N use G_N=2 (stripes 2,1 on TILES_N=3). Override with -DCK_GROUP_N=N
   to sweep. */
#if TILES_N > 4
#define CK_GROUP_N 5
#else
#define CK_GROUP_N 2
#endif
#endif
        const int G_M = CK_GROUP_M;
        const int G_N = CK_GROUP_N;
        const int stripes_per_row = (TILES_N + G_N - 1) / G_N;
        const int row_tiles = G_M * TILES_N;
        const int row_group = block_idx / row_tiles;
        const int in_row = block_idx % row_tiles;
        const int first_m = row_group * G_M;

        int tm, tn;
        if (first_m + G_M <= TILES_M) {
            /* Fast path: full row_group, all divisors compile-time. */
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
            /* Tail row_group (num_m < G_M): runtime divisors; rare. */
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

#elif TILE_DISPATCH == 19
    /*
    dg-snake (zigzag-within-dgswizzle-band): dgswizzle M-band layout, but
    traverse N in zigzag within each M-row of the band.  At M-row boundary
    N stays put → A changes but B persists across the boundary.  Gives
    dgswizzle's B-column reuse AND zigzag's row-transition continuity.
    */
    {
#ifndef DG_GROUP_SIZE
#define DG_GROUP_SIZE 8
#endif
        const int G = DG_GROUP_SIZE;
        const int group_tiles = TILES_N * G;
        const int group_idx = block_idx / group_tiles;
        const int first_m = group_idx * G;
        const int in_group = block_idx % group_tiles;
        const int num_in_group = min(G, TILES_M - first_m);
        const int local_m = in_group % num_in_group;
        int local_n = in_group / num_in_group;
        /* zigzag N within group: reverse N direction on odd local_m */
        if (local_m & 1) local_n = TILES_N - 1 - local_n;
        int tm = first_m + local_m;
        if (tm >= TILES_M) tm = TILES_M - 1;
        return tm * TILES_N + local_n;
    }

#elif TILE_DISPATCH == 21
    /*
    ncyrot: ncycle + per-cluster tn rotation.

    Preserves ncycle's within-cluster A-reuse (same tm for TILES_N ticks)
    while rotating tn per cluster-band so the 74 clusters hold DIFFERENT tn
    at each tick — breaks ncycle's simultaneous-same-N-column store
    contention (FC1 g-s = 0.681 ms on vanilla ncycle).

    Hypothesis: within-cluster A-reuse is the source of ncycle's fast strip.
    If across-cluster B-reuse via L2 is actually what matters, strip will
    regress toward rowmajor-class (~1.38 ms FC1).  Experiment.

    Avoids runtime '%' operators.  '/' appears only on compile-time const
    divisors (NC, TILES_N) — nvcc magic-multiplies these.  Band thresholds
    are hand-tabulated for SM_COUNT=148 × TILES_N ∈ {3, 12}; extend the
    #if ladder for other configs.
    */
    {
        const int NC = SM_COUNT / 2;
        const int _ti_local = block_idx / NC;
        const int c = block_idx - _ti_local * NC;
        const int super = _ti_local / TILES_N;
        const int sub = _ti_local - super * TILES_N;

        int tn_shift;
#if (SM_COUNT == 148) && (TILES_N == 3)
        tn_shift = (c >= 25) + (c >= 50);
#elif (SM_COUNT == 148) && (TILES_N == 12)
        tn_shift = (c >= 7) + (c >= 13) + (c >= 19) + (c >= 25)
                 + (c >= 31) + (c >= 37) + (c >= 44) + (c >= 50)
                 + (c >= 56) + (c >= 62) + (c >= 68);
#else
#error "TILE_DISPATCH=21 (ncyrot) needs per-(SM_COUNT, TILES_N) threshold table"
#endif
        int tn = sub + tn_shift;
        if (tn >= TILES_N) tn -= TILES_N;

        int tm = super * NC + c;
        if (tm >= TILES_M) tm = TILES_M - 1;
        return tm * TILES_N + tn;
    }

#elif TILE_DISPATCH == 20
    /* Cluster-M-cycle: pair 0 and pair 1 within a 4-CTA cluster share tm and
       take adjacent tn.  Symmetric twin of TD=14 (N-cycle), designed for
       C4_A_MULTICAST (A shared across pairs).

       Requires C4_DUAL_PAIR and TILES_N % 2 == 0.

       Mapping: block_idx = _ti * NC + pair_cluster_id, where consecutive
       pair_cluster_ids (2k, 2k+1) live in the same 4-CTA cluster.  Group by
       2:  q = block_idx >> 1 indexes the 4-CTA-cluster step, pair = block_idx
       & 1 selects tn offset.  tn = 2*tn_base + pair so the two pairs land
       at adjacent tn with matching tm → mcast coalesces. */
    {
        static_assert(TILES_N % 2 == 0,
            "TILE_DISPATCH=20 (mcycle / C4_A_MULTICAST) requires TILES_N % 2 == 0");
        const int q       = block_idx >> 1;
        const int pair    = block_idx & 1;
        const int tn_pairs = TILES_N / 2;
        int tm      = q / tn_pairs;
        int tn_base = q % tn_pairs;
        int tn = 2 * tn_base + pair;
        if (tm >= TILES_M) { tm = TILES_M - 1; tn = TILES_N - 1; }
        return tm * TILES_N + tn;
    }

#elif TILE_DISPATCH == 23
    /*
    dgphase: dgswizzle with cluster-dependent time-shift.

    Rationale: in TD=8, all NC clusters at tick t visit the same dgswizzle
    group_idx range.  Cluster c here is time-shifted by c ticks (block_idx
    offset of c * NC), so at any given tick the 74 clusters are spread across
    different M-row groups — decorrelating DRAM arrival patterns without
    changing per-cluster locality (within-cluster sequence is a cyclic
    rotation, so A/B reuse within a cluster is unchanged).

    Bijection: (block_idx + c*NC) mod TOTAL preserves c = block_idx % NC
    (because c*NC % NC = 0), and within a cluster is a cyclic permutation
    of the tick range.  TD=8 is bijective on the shifted block_idx, so the
    overall map is bijective.

    Open question: is decorrelation helpful at K=3072 where L2 is the
    bottleneck (arrival-pattern stalls)?  Empirical test. */
    {
#ifndef DG_GROUP_SIZE
#define DG_GROUP_SIZE 8
#endif
        const int NC = SM_COUNT / 2;
        const int TOTAL = TILES_M * TILES_N;
        const int c = block_idx - (block_idx / NC) * NC;   /* block_idx % NC */
        int bi = block_idx + c * NC;
        if (bi >= TOTAL) bi -= TOTAL;

        const int group_tiles = TILES_N * DG_GROUP_SIZE;
        const int group_idx = bi / group_tiles;
        const int first_m = group_idx * DG_GROUP_SIZE;
        const int in_group = bi - group_idx * group_tiles;
        int tm, tn;
        if (first_m + DG_GROUP_SIZE <= TILES_M) {
            tm = first_m + in_group % DG_GROUP_SIZE;
            tn = in_group / DG_GROUP_SIZE;
        } else {
            const int tail = TILES_M - first_m;
            tm = first_m + in_group % tail;
            tn = in_group / tail;
        }
        if (tm >= TILES_M) tm = TILES_M - 1;
        if (tn >= TILES_N) tn = TILES_N - 1;
        return tm * TILES_N + tn;
    }

#elif TILE_DISPATCH == 24
    /*
    dgnrot: dgswizzle with within-group diagonal tn rotation.

    Within each full M-row group, the 24 (lm, ln) pairs map to
      tn = (ln + lm) mod TILES_N
    so the 8 clusters at different lm values visit 8 different (lm, tn)
    combinations that distribute evenly across all TILES_N columns
    simultaneously — rather than 8 clusters hitting the same tn in lockstep
    (as vanilla TD=8 does) before all advancing to the next tn together.

    For FC2 (TILES_N=3) this cuts peak same-tn cluster count within a group
    from 8 → 3, which should reduce store-phase HBM-page contention.

    Bijection (within a group): fix target (tm, tn).  tm → lm determined.
    Given lm, solve ln = (tn - lm + TILES_N) mod TILES_N — unique
    ln ∈ [0, TILES_N).  Across groups, first_m differs, so no cross-group
    collision.  Tail group (num_m < DG_GROUP_SIZE) uses num_m in place of 8
    — same diagonal logic remains bijective.  */
    {
#ifndef DG_GROUP_SIZE
#define DG_GROUP_SIZE 8
#endif
        const int group_tiles = TILES_N * DG_GROUP_SIZE;
        const int group_idx = block_idx / group_tiles;
        const int first_m = group_idx * DG_GROUP_SIZE;
        const int in_group = block_idx - group_idx * group_tiles;
        int tm, tn;
        if (first_m + DG_GROUP_SIZE <= TILES_M) {
            const int lm = in_group % DG_GROUP_SIZE;
            const int ln = in_group / DG_GROUP_SIZE;
            int tn_rot = ln + lm;
            while (tn_rot >= TILES_N) tn_rot -= TILES_N;
            tm = first_m + lm;
            tn = tn_rot;
        } else {
            const int num_m = TILES_M - first_m;
            const int lm = in_group % num_m;
            const int ln = in_group / num_m;
            int tn_rot = ln + lm;
            while (tn_rot >= TILES_N) tn_rot -= TILES_N;
            tm = first_m + lm;
            tn = tn_rot;
        }
        if (tm >= TILES_M) tm = TILES_M - 1;
        if (tn >= TILES_N) tn = TILES_N - 1;
        return tm * TILES_N + tn;
    }

#elif TILE_DISPATCH == 22
    /*
       cuBLASLt nvjet_sm100_qqtst rank-1/2 tile swizzle (3-level hierarchical
       divmod with main/tail split and boustrophedon).  Faithful port of the
       SASS at rank1.sass:+2272..+2880.

       Flow:
         r1 = block_idx mod RK_D1              (outer wrap; q1 discarded)
         (q2, r2) = divmod(r1, RK_D2)          (raster-row stride)
         if q2 >= RK_THRESHOLD:
             (q3, r3) = divmod(r2, RK_D3A)     (main body)
         else:
             (q3, r3) = divmod(r2, RK_D3B)     (tail region)
         r3 += q2 * RK_D3B                      (SASS adds q2*d3b; d3b=0 disables)
         if q2 even: q3 = RK_FLIP_BASE - q3     (boustrophedon on M)
         tm = q3; tn = r3

       RK_D3A=0 or RK_D3B=0 in the picked branch disables inner divmod
       (q3 = r2, r3 = 0) to mirror the @UP2 UMOV path in SASS.

       Defaults reproduce the geometry interpretation fit from rank-2's SASS:
       column-major with full-M boustrophedon.  With these values:
         q2 = cluster_tn (N-column index)
         r2 = cluster_tm within N-column
         q3 = r2, r3 = 0
         tn = r3 + q2*1 = q2
         tm = boustrophedon(q3) = (q2 even) ? (TM-1)-r2 : r2
       i.e. all clusters in a wave share the same tn (maximum B-tile L2 reuse),
       M-direction flips per N-column to kill the "jump back to tm=0" cache
       cliff.  Override RK_D2/RK_D3A/RK_D3B/RK_FLIP_BASE via -D to explore
       other decompositions (e.g. dgswizzle-style blocks).
    */
    {
#ifndef RK_D1
#define RK_D1 (TILES_M * TILES_N)
#endif
#ifndef RK_D2
#define RK_D2 TILES_M
#endif
#ifndef RK_D3A
#define RK_D3A 1
#endif
#ifndef RK_D3B
#define RK_D3B 1
#endif
#ifndef RK_THRESHOLD
#define RK_THRESHOLD 0
#endif
#ifndef RK_FLIP_BASE
#define RK_FLIP_BASE (TILES_M - 1)
#endif

        const int d1        = RK_D1;
        const int d2        = RK_D2;
        const int d3a       = RK_D3A;
        const int d3b       = RK_D3B;
        const int threshold = RK_THRESHOLD;
        const int flip_base = RK_FLIP_BASE;

        int r1 = (d1 > 0) ? (block_idx % d1) : block_idx;
        int q2 = (d2 > 0) ? (r1 / d2) : 0;
        int r2 = (d2 > 0) ? (r1 % d2) : r1;

        int q3, r3;
        if (q2 >= threshold) {
            q3 = (d3a > 0) ? (r2 / d3a) : r2;
            r3 = (d3a > 0) ? (r2 % d3a) : 0;
        } else {
            q3 = (d3b > 0) ? (r2 / d3b) : r2;
            r3 = (d3b > 0) ? (r2 % d3b) : 0;
        }

        r3 += q2 * d3b;

        if ((q2 & 1) == 0) q3 = flip_base - q3;

        int tm = q3;
        int tn = r3;
        if (tm < 0) tm = 0;
        if (tn < 0) tn = 0;
        if (tm >= TILES_M) tm = TILES_M - 1;
        if (tn >= TILES_N) tn = TILES_N - 1;
        return tm * TILES_N + tn;
    }
#elif TILE_DISPATCH == 30
    /*
    HYB_CHET (cluster-heterogeneous): partition cluster space and tile space
    by M-halves. First NC/2 clusters run dgswizzle on M-blocks [0, TM/2);
    last NC/2 clusters run rowmajor on M-blocks [TM/2, TM).

    At any given tt, half the wavefront has dgswizzle's 8-cluster cooperative
    A-loading pattern; the other half has rowmajor's tighter per-M-block
    coverage.  Each half's tile space is fully-covered by its rule alone,
    so bijection is trivial.

    Constraints: TILES_M must be even, NC must be even.
    */
    {
#ifndef DG_GROUP_SIZE
#define DG_GROUP_SIZE 8
#endif
        const int NC      = SM_COUNT / 2;
        const int NC_HALF = NC / 2;
        const int TM_HALF = TILES_M / 2;
        const int c       = block_idx - (block_idx / NC) * NC;
        const int tt      = block_idx / NC;
        int tm, tn;
        if (c < NC_HALF) {
            const int sub_lin     = c + tt * NC_HALF;
            const int G           = DG_GROUP_SIZE;
            const int group_tiles = TILES_N * G;
            const int group_idx   = sub_lin / group_tiles;
            const int first_m     = group_idx * G;
            const int in_group    = sub_lin - group_idx * group_tiles;
            const int nig         = (first_m + G <= TM_HALF) ? G : TM_HALF - first_m;
            const int tm_local    = in_group - (in_group / nig) * nig;
            tn                    = in_group / nig;
            tm                    = first_m + tm_local;
        } else {
            const int sub_lin     = (c - NC_HALF) + tt * NC_HALF;
            const int tm_local    = sub_lin / TILES_N;
            tn                    = sub_lin - tm_local * TILES_N;
            tm                    = TM_HALF + tm_local;
        }
        if (tm >= TILES_M) tm = TILES_M - 1;
        if (tn >= TILES_N) tn = TILES_N - 1;
        return tm * TILES_N + tn;
    }

#elif TILE_DISPATCH == 31
    /*
    HYB_PMIX (phase-mixed): every cluster runs dgswizzle on tn in [0, TILES_N-1)
    for its first (TILES_N-1)*TILES_M/NC tt's, then rowmajor on tn=TILES_N-1
    for its remaining TILES_M/NC tt's.  Tile partition is by N-column, NOT
    by M-range.

    At FC2 (TILES_N=3, TILES_M=3626, NC=74): phase 1 = 98 tt × 74 clusters =
    7252 tiles (covering tn=0,1); phase 2 = 49 tt × 74 clusters = 3626 tiles
    (covering tn=2).  Within phase 1, dgswizzle groups are of size 8*2=16
    (restricted to tn=0,1).

    Constraints: TILES_M*(TILES_N-1) % NC == 0 and TILES_M % NC == 0
    (both hold at FC2: 3626*2 % 74 = 0 and 3626 % 74 = 0).
    */
    {
#ifndef DG_GROUP_SIZE
#define DG_GROUP_SIZE 8
#endif
        const int NC             = SM_COUNT / 2;
        const int PHASE1_TILES   = TILES_M * (TILES_N - 1);
        int tm, tn;
        if (block_idx < PHASE1_TILES) {
            const int G           = DG_GROUP_SIZE;
            const int TN_PHASE1   = TILES_N - 1;
            const int group_tiles = TN_PHASE1 * G;
            const int group_idx   = block_idx / group_tiles;
            const int first_m     = group_idx * G;
            const int in_group    = block_idx - group_idx * group_tiles;
            const int nig         = (first_m + G <= TILES_M) ? G : TILES_M - first_m;
            const int tm_local    = in_group - (in_group / nig) * nig;
            tn                    = in_group / nig;
            tm                    = first_m + tm_local;
        } else {
            const int sub_lin     = block_idx - PHASE1_TILES;
            tm                    = sub_lin;
            tn                    = TILES_N - 1;
        }
        if (tm >= TILES_M) tm = TILES_M - 1;
        if (tn >= TILES_N) tn = TILES_N - 1;
        return tm * TILES_N + tn;
    }

#elif TILE_DISPATCH == 32
    /*
    HYB_INGH (within-group hybrid): same group structure as dgswizzle, but
    alternate the in-group traversal per group.  Even group_idx uses
    M-major (tm_local = in_group % nig, tn = in_group / nig — dgswizzle's
    default).  Odd group_idx uses N-major (tm_local = in_group / TILES_N,
    tn = in_group % TILES_N — DG_INNER_T's default).

    Preserves dgswizzle's cross-cluster A-sharing at group level (same 8
    M-blocks in play across 24 cluster slots per group), but alternates the
    cluster → (tm, tn) mapping within the group to break synchronous
    arrival patterns. Each group independently bijective; disjoint groups
    compose to full bijection.
    */
    {
#ifndef DG_GROUP_SIZE
#define DG_GROUP_SIZE 8
#endif
        const int G           = DG_GROUP_SIZE;
        const int group_tiles = TILES_N * G;
        const int group_idx   = block_idx / group_tiles;
        const int first_m     = group_idx * G;
        const int in_group    = block_idx - group_idx * group_tiles;
        const int nig         = (first_m + G <= TILES_M) ? G : TILES_M - first_m;
        int tm_local, tn;
        if ((group_idx & 1) == 0) {
            tm_local = in_group - (in_group / nig) * nig;
            tn       = in_group / nig;
        } else {
            tm_local = in_group / TILES_N;
            tn       = in_group - tm_local * TILES_N;
            if (tm_local >= nig) tm_local = nig - 1;
        }
        int tm = first_m + tm_local;
        if (tm >= TILES_M) tm = TILES_M - 1;
        if (tn >= TILES_N) tn = TILES_N - 1;
        return tm * TILES_N + tn;
    }

#endif
}

#endif /* TILE_DISPATCH 8..32 */

/*
   Templated tile swizzle for the single-binary basin sweep (COMBO_QUICK).

   Always compiled (independent of the TILE_DISPATCH macro), only instantiated
   for the (TD, DGG) pairs the templated fc2_w3_kernel actually references.
   Bodies for TD=8/11/13/19 are copied verbatim from the legacy static_swizzle
   ladder above (DG_GROUP_SIZE -> DGG); for DGG==DG_GROUP_SIZE the expressions
   are identical, so static_swizzle_t<8,8> compiles byte-identical SASS to the
   production static_swizzle at TILE_DISPATCH=8.  The gflip basin (TD=33..58)
   is ported from swizzle_w3x.cuh's tile_swizzle_t — pure (TILES_M, TILES_N,
   DGG) index math, transparent to the fused residual path (residual is keyed
   on the output (tm,tn), so any bijective remap stays correct).
*/
template<int TD, int DGG>
static __device__ __forceinline__ int static_swizzle_t(int block_idx) {
    if constexpr (TD == 8) {
        const int group_tiles = TILES_N * DGG;
        const int group_idx = block_idx / group_tiles;
        const int first_m = group_idx * DGG;
        const int in_group = block_idx % group_tiles;
        if (first_m + DGG <= TILES_M) {
            return (first_m + in_group % DGG) * TILES_N
                 + in_group / DGG;
        }
        const int tail = TILES_M - first_m;
        return (first_m + in_group % tail) * TILES_N + in_group / tail;

    } else if constexpr (TD == 11) {
        int tm = block_idx / TILES_N;
        int tn = block_idx % TILES_N;
        if (tm >= TILES_M) tm = TILES_M - 1;
        if (tm & 1) tn = TILES_N - 1 - tn;
        return tm * TILES_N + tn;

    } else if constexpr (TD == 13) {
        int tm = block_idx / TILES_N;
        int tn = block_idx % TILES_N;
        if (tm >= TILES_M) tm = TILES_M - 1;
        return tm * TILES_N + tn;

    } else if constexpr (TD == 19) {
        const int group_tiles = TILES_N * DGG;
        const int group_idx = block_idx / group_tiles;
        const int first_m = group_idx * DGG;
        const int in_group = block_idx - group_idx * group_tiles;
        const int num_in_group = min(DGG, TILES_M - first_m);
        const int local_m = in_group % num_in_group;
        int local_n = in_group / num_in_group;
        if (local_m & 1) local_n = TILES_N - 1 - local_n;
        int tm = first_m + local_m;
        if (tm >= TILES_M) tm = TILES_M - 1;
        return tm * TILES_N + local_n;

    } else if constexpr (TD == 33) {
        const int group_tiles = TILES_N * DGG;
        const int num_groups  = (TILES_M + DGG - 1) / DGG;
        int group_idx         = block_idx / group_tiles;
        const int in_group    = block_idx - group_idx * group_tiles;
        const int paired      = group_idx ^ 1;
        if (paired < num_groups) group_idx = paired;
        const int first_m  = group_idx * DGG;
        const int nig      = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
        const int tm_local = in_group - (in_group / nig) * nig;
        const int tn       = in_group / nig;
        int tm = first_m + tm_local;
        if (tm >= TILES_M) tm = TILES_M - 1;
        return tm * TILES_N + tn;

    } else if constexpr (TD == 39) {
        const int group_tiles = TILES_N * DGG;
        const int group_idx   = block_idx / group_tiles;
        const int first_m     = group_idx * DGG;
        const int in_group    = block_idx - group_idx * group_tiles;
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

    } else if constexpr (TD == 44) {
        const int group_tiles = TILES_N * DGG;
        const int group_idx   = block_idx / group_tiles;
        const int first_m     = group_idx * DGG;
        const int in_group    = block_idx - group_idx * group_tiles;
        const int nig         = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
        const int lm          = in_group - (in_group / nig) * nig;
        const int ln          = in_group / nig;
        int tn = (lm & 1) ? (ln + 2) : ln;
        if (tn >= TILES_N) tn -= TILES_N;
        int tm = first_m + lm;
        if (tm >= TILES_M) tm = TILES_M - 1;
        return tm * TILES_N + tn;

    } else if constexpr (TD == 52) {
        const int group_tiles = TILES_N * DGG;
        const int num_groups  = (TILES_M + DGG - 1) / DGG;
        int group_idx         = block_idx / group_tiles;
        const int in_group    = block_idx - group_idx * group_tiles;
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

    } else if constexpr (TD == 53) {
        const int group_tiles = TILES_N * DGG;
        const int num_groups  = (TILES_M + DGG - 1) / DGG;
        int group_idx         = block_idx / group_tiles;
        const int in_group    = block_idx - group_idx * group_tiles;
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

    } else if constexpr (TD == 54) {
        const int group_tiles = TILES_N * DGG;
        const int num_groups  = (TILES_M + DGG - 1) / DGG;
        int group_idx         = block_idx / group_tiles;
        const int in_group    = block_idx - group_idx * group_tiles;
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

    } else if constexpr (TD == 56) {
        const int group_tiles = TILES_N * DGG;
        const int num_groups  = (TILES_M + DGG - 1) / DGG;
        int group_idx         = block_idx / group_tiles;
        const int in_group    = block_idx - group_idx * group_tiles;
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

    } else if constexpr (TD == 57) {
        const int group_tiles = TILES_N * DGG;
        const int num_groups  = (TILES_M + DGG - 1) / DGG;
        int group_idx         = block_idx / group_tiles;
        const int in_group    = block_idx - group_idx * group_tiles;
        const int paired      = group_idx ^ 1;
        if (paired < num_groups) group_idx = paired;
        const int first_m = group_idx * DGG;
        const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;
        const int lm_raw  = in_group - (in_group / nig) * nig;
        const int ln      = in_group / nig;
        int lm = lm_raw;
        if (nig == 8 && (group_idx & 1)) lm = (lm_raw * 3) & 7;
        int tm = first_m + lm;
        if (tm >= TILES_M) tm = TILES_M - 1;
        return tm * TILES_N + ln;

    } else if constexpr (TD == 58) {
        const int group_tiles = TILES_N * DGG;
        const int num_groups  = (TILES_M + DGG - 1) / DGG;
        int group_idx         = block_idx / group_tiles;
        const int in_group    = block_idx - group_idx * group_tiles;
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

    } else {
        return block_idx;
    }
}
