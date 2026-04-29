/* coord_descend_gen.py output — paste between BEGIN/END markers in fc2_w3x.cu */
/* BEGIN COORD_DESCEND */

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

/* END COORD_DESCEND */
