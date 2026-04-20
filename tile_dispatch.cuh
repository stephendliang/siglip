/*
   Static tile swizzle for TD=8..16.

   Shared between fc1_w3.cu and fc2_w3.cu.  Both kernels stride through a
   linear block_idx = _ti * num_clusters + cluster_id; static_swizzle()
   remaps that to a flat tile_idx = tm * TILES_N + tn.

   Requires TILES_M, TILES_N, SM_COUNT to be #defined before inclusion.
*/

#pragma once

#if TILE_DISPATCH >= 8

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
#endif
}

#endif /* TILE_DISPATCH >= 8 */
