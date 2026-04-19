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
       group.  (m0,n0)(m1,n0)...(m7,n0)(m0,n1)... */
    const int group_tiles = TILES_N * DG_GROUP_SIZE;
    const int group_idx = block_idx / group_tiles;
    const int first_m = group_idx * DG_GROUP_SIZE;
    const int in_group = block_idx % group_tiles;
    const int num_in_group = min(DG_GROUP_SIZE, TILES_M - first_m);
    return (first_m + in_group % num_in_group) * TILES_N + in_group / num_in_group;

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
#endif
}

#endif /* TILE_DISPATCH >= 8 */
