"""sw_<name>() defs — splice into tools/analyze_swizzle.py near sw_gflip_*."""
# BEGIN COORD_DESCEND

def sw_gflip_xk2_blkswap(lin, G=8):
    """XK=2 pairing × blkswap-^4 alt1 (vs gflip's XK=1)  XK=2, p_u=id, alt=xor4, dens=alt1."""
    gt = TILES_N * G
    num_groups = (TILES_M + G - 1) // G
    group_idx = lin // gt
    in_g = lin - group_idx * gt
    paired = group_idx ^ 2
    if paired < num_groups: group_idx = paired
    first_m = group_idx * G
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    lmr = in_g % nig
    ln = in_g // nig
    lm = lmr
    if nig == 8:
        if (group_idx & 1) == 1: lm = lm ^ 4
    return first_m + lm, ln


def sw_gflip_xk3_blkswap(lin, G=8):
    """XK=3 pairing × blkswap-^4 alt1 (vs gflip's XK=1)  XK=3, p_u=id, alt=xor4, dens=alt1."""
    gt = TILES_N * G
    num_groups = (TILES_M + G - 1) // G
    group_idx = lin // gt
    in_g = lin - group_idx * gt
    paired = group_idx ^ 3
    if paired < num_groups: group_idx = paired
    first_m = group_idx * G
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    lmr = in_g % nig
    ln = in_g // nig
    lm = lmr
    if nig == 8:
        if (group_idx & 1) == 1: lm = lm ^ 4
    return first_m + lm, ln


def sw_gflip_xk5_blkswap(lin, G=8):
    """XK=5 pairing × blkswap-^4 alt1 (vs gflip's XK=1)  XK=5, p_u=id, alt=xor4, dens=alt1."""
    gt = TILES_N * G
    num_groups = (TILES_M + G - 1) // G
    group_idx = lin // gt
    in_g = lin - group_idx * gt
    paired = group_idx ^ 5
    if paired < num_groups: group_idx = paired
    first_m = group_idx * G
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    lmr = in_g % nig
    ln = in_g // nig
    lm = lmr
    if nig == 8:
        if (group_idx & 1) == 1: lm = lm ^ 4
    return first_m + lm, ln


def sw_gflip_xk7_blkswap(lin, G=8):
    """XK=7 pairing × blkswap-^4 alt1 (vs gflip's XK=1)  XK=7, p_u=id, alt=xor4, dens=alt1."""
    gt = TILES_N * G
    num_groups = (TILES_M + G - 1) // G
    group_idx = lin // gt
    in_g = lin - group_idx * gt
    paired = group_idx ^ 7
    if paired < num_groups: group_idx = paired
    first_m = group_idx * G
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    lmr = in_g % nig
    ln = in_g // nig
    lm = lmr
    if nig == 8:
        if (group_idx & 1) == 1: lm = lm ^ 4
    return first_m + lm, ln


def sw_gflip_blkx1(lin, G=8):
    """alt1 × xor1 (vs blkswap's xor4) — alt-mask scan  XK=1, p_u=id, alt=xor1, dens=alt1."""
    gt = TILES_N * G
    num_groups = (TILES_M + G - 1) // G
    group_idx = lin // gt
    in_g = lin - group_idx * gt
    paired = group_idx ^ 1
    if paired < num_groups: group_idx = paired
    first_m = group_idx * G
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    lmr = in_g % nig
    ln = in_g // nig
    lm = lmr
    if nig == 8:
        if (group_idx & 1) == 1: lm = lm ^ 1
    return first_m + lm, ln


def sw_gflip_blkx2(lin, G=8):
    """alt1 × xor2 (vs blkswap's xor4) — alt-mask scan  XK=1, p_u=id, alt=xor2, dens=alt1."""
    gt = TILES_N * G
    num_groups = (TILES_M + G - 1) // G
    group_idx = lin // gt
    in_g = lin - group_idx * gt
    paired = group_idx ^ 1
    if paired < num_groups: group_idx = paired
    first_m = group_idx * G
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    lmr = in_g % nig
    ln = in_g // nig
    lm = lmr
    if nig == 8:
        if (group_idx & 1) == 1: lm = lm ^ 2
    return first_m + lm, ln


def sw_gflip_blkx3(lin, G=8):
    """alt1 × xor3 (vs blkswap's xor4) — alt-mask scan  XK=1, p_u=id, alt=xor3, dens=alt1."""
    gt = TILES_N * G
    num_groups = (TILES_M + G - 1) // G
    group_idx = lin // gt
    in_g = lin - group_idx * gt
    paired = group_idx ^ 1
    if paired < num_groups: group_idx = paired
    first_m = group_idx * G
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    lmr = in_g % nig
    ln = in_g // nig
    lm = lmr
    if nig == 8:
        if (group_idx & 1) == 1: lm = lm ^ 3
    return first_m + lm, ln


def sw_gflip_blkx5(lin, G=8):
    """alt1 × xor5 (vs blkswap's xor4) — alt-mask scan  XK=1, p_u=id, alt=xor5, dens=alt1."""
    gt = TILES_N * G
    num_groups = (TILES_M + G - 1) // G
    group_idx = lin // gt
    in_g = lin - group_idx * gt
    paired = group_idx ^ 1
    if paired < num_groups: group_idx = paired
    first_m = group_idx * G
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    lmr = in_g % nig
    ln = in_g // nig
    lm = lmr
    if nig == 8:
        if (group_idx & 1) == 1: lm = lm ^ 5
    return first_m + lm, ln


def sw_gflip_blkx6(lin, G=8):
    """alt1 × xor6 (vs blkswap's xor4) — alt-mask scan  XK=1, p_u=id, alt=xor6, dens=alt1."""
    gt = TILES_N * G
    num_groups = (TILES_M + G - 1) // G
    group_idx = lin // gt
    in_g = lin - group_idx * gt
    paired = group_idx ^ 1
    if paired < num_groups: group_idx = paired
    first_m = group_idx * G
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    lmr = in_g % nig
    ln = in_g // nig
    lm = lmr
    if nig == 8:
        if (group_idx & 1) == 1: lm = lm ^ 6
    return first_m + lm, ln


def sw_gflip_blkx7(lin, G=8):
    """alt1 × xor7 (vs blkswap's xor4) — alt-mask scan  XK=1, p_u=id, alt=xor7, dens=alt1."""
    gt = TILES_N * G
    num_groups = (TILES_M + G - 1) // G
    group_idx = lin // gt
    in_g = lin - group_idx * gt
    paired = group_idx ^ 1
    if paired < num_groups: group_idx = paired
    first_m = group_idx * G
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    lmr = in_g % nig
    ln = in_g // nig
    lm = lmr
    if nig == 8:
        if (group_idx & 1) == 1: lm = lm ^ 7
    return first_m + lm, ln


def sw_gflip_blk_alt0(lin, G=8):
    """xor4 × density=alt0 (vs blkswap's alt1, quartswap's qrt1)  XK=1, p_u=id, alt=xor4, dens=alt0."""
    gt = TILES_N * G
    num_groups = (TILES_M + G - 1) // G
    group_idx = lin // gt
    in_g = lin - group_idx * gt
    paired = group_idx ^ 1
    if paired < num_groups: group_idx = paired
    first_m = group_idx * G
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    lmr = in_g % nig
    ln = in_g // nig
    lm = lmr
    if nig == 8:
        if (group_idx & 1) == 0: lm = lm ^ 4
    return first_m + lm, ln


def sw_gflip_blk_qrt0(lin, G=8):
    """xor4 × density=qrt0 (vs blkswap's alt1, quartswap's qrt1)  XK=1, p_u=id, alt=xor4, dens=qrt0."""
    gt = TILES_N * G
    num_groups = (TILES_M + G - 1) // G
    group_idx = lin // gt
    in_g = lin - group_idx * gt
    paired = group_idx ^ 1
    if paired < num_groups: group_idx = paired
    first_m = group_idx * G
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    lmr = in_g % nig
    ln = in_g // nig
    lm = lmr
    if nig == 8:
        if (group_idx & 3) == 0: lm = lm ^ 4
    return first_m + lm, ln


def sw_gflip_blk_qrt2(lin, G=8):
    """xor4 × density=qrt2 (vs blkswap's alt1, quartswap's qrt1)  XK=1, p_u=id, alt=xor4, dens=qrt2."""
    gt = TILES_N * G
    num_groups = (TILES_M + G - 1) // G
    group_idx = lin // gt
    in_g = lin - group_idx * gt
    paired = group_idx ^ 1
    if paired < num_groups: group_idx = paired
    first_m = group_idx * G
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    lmr = in_g % nig
    ln = in_g // nig
    lm = lmr
    if nig == 8:
        if (group_idx & 3) == 2: lm = lm ^ 4
    return first_m + lm, ln


def sw_gflip_blk_qrt3(lin, G=8):
    """xor4 × density=qrt3 (vs blkswap's alt1, quartswap's qrt1)  XK=1, p_u=id, alt=xor4, dens=qrt3."""
    gt = TILES_N * G
    num_groups = (TILES_M + G - 1) // G
    group_idx = lin // gt
    in_g = lin - group_idx * gt
    paired = group_idx ^ 1
    if paired < num_groups: group_idx = paired
    first_m = group_idx * G
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    lmr = in_g % nig
    ln = in_g // nig
    lm = lmr
    if nig == 8:
        if (group_idx & 3) == 3: lm = lm ^ 4
    return first_m + lm, ln


def sw_gflip_blk_n3k1(lin, G=8):
    """xor4 × density=n3k1 (vs blkswap's alt1, quartswap's qrt1)  XK=1, p_u=id, alt=xor4, dens=n3k1."""
    gt = TILES_N * G
    num_groups = (TILES_M + G - 1) // G
    group_idx = lin // gt
    in_g = lin - group_idx * gt
    paired = group_idx ^ 1
    if paired < num_groups: group_idx = paired
    first_m = group_idx * G
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    lmr = in_g % nig
    ln = in_g // nig
    lm = lmr
    if nig == 8:
        if (group_idx % 3) == 1: lm = lm ^ 4
    return first_m + lm, ln


def sw_gflip_mul3(lin, G=8):
    """uniform p_u=mul3 only (vs lmrev's bitrev)  XK=1, p_u=mul3, alt=none, dens=all."""
    gt = TILES_N * G
    num_groups = (TILES_M + G - 1) // G
    group_idx = lin // gt
    in_g = lin - group_idx * gt
    paired = group_idx ^ 1
    if paired < num_groups: group_idx = paired
    first_m = group_idx * G
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    lmr = in_g % nig
    ln = in_g // nig
    lm = lmr
    if nig == 8:
        lm = (lmr * 3) % 8
    return first_m + lm, ln


def sw_gflip_mul5(lin, G=8):
    """uniform p_u=mul5 only (vs lmrev's bitrev)  XK=1, p_u=mul5, alt=none, dens=all."""
    gt = TILES_N * G
    num_groups = (TILES_M + G - 1) // G
    group_idx = lin // gt
    in_g = lin - group_idx * gt
    paired = group_idx ^ 1
    if paired < num_groups: group_idx = paired
    first_m = group_idx * G
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    lmr = in_g % nig
    ln = in_g // nig
    lm = lmr
    if nig == 8:
        lm = (lmr * 5) % 8
    return first_m + lm, ln


def sw_gflip_bitrev_xor1_alt1(lin, G=8):
    """composition: uniform=bitrev + alt=xor1 dens=alt1  XK=1, p_u=bitrev, alt=xor1, dens=alt1."""
    gt = TILES_N * G
    num_groups = (TILES_M + G - 1) // G
    group_idx = lin // gt
    in_g = lin - group_idx * gt
    paired = group_idx ^ 1
    if paired < num_groups: group_idx = paired
    first_m = group_idx * G
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    lmr = in_g % nig
    ln = in_g // nig
    lm = lmr
    if nig == 8:
        lm = ((lmr & 1) << 2) | (lmr & 2) | ((lmr >> 2) & 1)
        if (group_idx & 1) == 1: lm = lm ^ 1
    return first_m + lm, ln


def sw_gflip_bitrev_xor2_alt1(lin, G=8):
    """composition: uniform=bitrev + alt=xor2 dens=alt1  XK=1, p_u=bitrev, alt=xor2, dens=alt1."""
    gt = TILES_N * G
    num_groups = (TILES_M + G - 1) // G
    group_idx = lin // gt
    in_g = lin - group_idx * gt
    paired = group_idx ^ 1
    if paired < num_groups: group_idx = paired
    first_m = group_idx * G
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    lmr = in_g % nig
    ln = in_g // nig
    lm = lmr
    if nig == 8:
        lm = ((lmr & 1) << 2) | (lmr & 2) | ((lmr >> 2) & 1)
        if (group_idx & 1) == 1: lm = lm ^ 2
    return first_m + lm, ln


def sw_gflip_mul3_xor4_alt1(lin, G=8):
    """composition: uniform=mul3 + alt=xor4 dens=alt1  XK=1, p_u=mul3, alt=xor4, dens=alt1."""
    gt = TILES_N * G
    num_groups = (TILES_M + G - 1) // G
    group_idx = lin // gt
    in_g = lin - group_idx * gt
    paired = group_idx ^ 1
    if paired < num_groups: group_idx = paired
    first_m = group_idx * G
    nig = G if first_m + G <= TILES_M else TILES_M - first_m
    lmr = in_g % nig
    ln = in_g // nig
    lm = lmr
    if nig == 8:
        lm = (lmr * 3) % 8
        if (group_idx & 1) == 1: lm = lm ^ 4
    return first_m + lm, ln


# END COORD_DESCEND
