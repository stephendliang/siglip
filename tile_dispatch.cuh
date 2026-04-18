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
    /* Z-order (Morton) on 4xTILES_N blocks. */
    {
        const int ZB = 4;
        const int ztiles = ZB * TILES_N;
        const int zgroup = block_idx / ztiles;
        const int zlocal = block_idx % ztiles;
        int tm_local = 0, tn_local = 0;
        int count = 0;
        for (int z = 0; z < ZB * 4; z++) {
            int zr = ((z >> 1) & 1) | ((z >> 2) & 2);
            int zc = (z & 1) | ((z >> 1) & 2);
            if (zc >= TILES_N) continue;
            if (count == zlocal) { tm_local = zr; tn_local = zc; break; }
            count++;
        }
        int tm = zgroup * ZB + tm_local;
        if (tm >= TILES_M) tm = TILES_M - 1;
        return tm * TILES_N + tn_local;
    }

#elif TILE_DISPATCH == 10
    /* Hilbert curve on 4x4 blocks, filtered to 4xTILES_N. */
    {
        const int HB = 4;
        const int htiles = HB * TILES_N;
        const int hgroup = block_idx / htiles;
        const int hlocal = block_idx % htiles;

        const int h_x[16] = {0,1,1,0,0,0,1,1,2,2,3,3,3,2,2,3};
        const int h_y[16] = {0,0,1,1,2,3,3,2,2,3,3,2,1,1,0,0};

        int tm_local = 0, tn_local = 0;
        int count = 0;
        for (int d = 0; d < 16; d++) {
            int hx = h_x[d], hy = h_y[d];
            if (hy >= TILES_N) continue;
            if (count == hlocal) { tm_local = hx; tn_local = hy; break; }
            count++;
        }
        int tm = hgroup * HB + tm_local;
        if (tm >= TILES_M) tm = TILES_M - 1;
        return tm * TILES_N + tn_local;
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
    nlock (static-N cluster bind): each cluster is permanently bound to one
    N-column and sweeps M.  Tests the pure "B-column persistence" hypothesis
    (dgswizzle holds B for ~4 ticks; nlock holds B for the full run).
    Trade-off: slight load imbalance when NC % TILES_N != 0 (FC2 NC=74, N=3
    → 25/25/24 clusters per band).  Each cluster only ever touches one B-tile
    stream → maximal L2 B-hit, worst A-locality (A changes every tick).
    */
    {
        const int NC = SM_COUNT / 2;
        const int c = block_idx % NC;
        const int _ti_local = block_idx / NC;
        const int cpb = (NC + TILES_N - 1) / TILES_N;  /* ceil(NC/N) */
        int tn = c / cpb;
        if (tn >= TILES_N) tn = TILES_N - 1;
        const int idx_in_band = c - tn * cpb;
        /* Last band may be smaller; use full cpb for stride to keep coverage. */
        int tm = _ti_local * cpb + idx_in_band;
        if (tm >= TILES_M) tm = TILES_M - 1;
        return tm * TILES_N + tn;
    }

#elif TILE_DISPATCH == 18
    /*
    checkered (2D M×N group): generalization of dgswizzle to a G_M × G_N
    block of tiles per group.  Within group, traverse column-first so a
    cluster stays on one N-tile for G_M ticks, then advances N.
    CK_GROUP_M × CK_GROUP_N defines the tile; CK_GROUP_N must divide TILES_N
    evenly (or the remainder is handled via min()).
    */
    {
#ifndef CK_GROUP_M
#define CK_GROUP_M 8
#endif
#ifndef CK_GROUP_N
#if TILES_N >= 4
#define CK_GROUP_N 4
#else
#define CK_GROUP_N TILES_N
#endif
#endif
        const int G_M = CK_GROUP_M;
        const int G_N = CK_GROUP_N;
        const int stripes_per_row = (TILES_N + G_N - 1) / G_N;
        const int group_tiles = G_M * G_N;
        const int row_tiles = stripes_per_row * group_tiles;
        const int row_group = block_idx / row_tiles;
        const int row_off = block_idx % row_tiles;
        const int stripe = row_off / group_tiles;
        const int in_group = row_off % group_tiles;
        /* Column-first within group: fill G_M M-rows for each G_N column. */
        const int local_m = in_group % G_M;
        const int local_n = in_group / G_M;
        int tm = row_group * G_M + local_m;
        int tn = stripe * G_N + local_n;
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
#endif
}

#endif /* TILE_DISPATCH >= 8 */
