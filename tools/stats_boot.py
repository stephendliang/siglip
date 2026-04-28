"""Bootstrap CI helpers — stdlib only, paired-resample by default.

Used by anova_1way.py / study_summary.py / predict_swizzle.py to attach
95% percentile-bootstrap confidence intervals to point estimates.

Resample model: we resample BLOCKS (passes/reps), not individual rows.
This preserves the paired structure that the rest of the analysis relies
on — within a resampled pass, all variants' observations stay together.
For per-cell statistics that don't have a block dimension (e.g. mean of
n cycles), the same function falls back to iid bootstrap if `blocks` is
None.

Default n_resamples=1000 → ±0.4% accuracy on tail percentiles, runs in
<1s for 5489 passes × 20 cells.
"""
import bisect
import math
import random


def percentile(sorted_vals, q):
    """Linear-interpolated percentile q ∈ [0,100] over a pre-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (q / 100.0) * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def bootstrap_ci(stat_fn, samples, n_resamples=1000, ci=95.0, seed=0xC0FFEE):
    """iid bootstrap of `stat_fn(resample)` over `samples`.

    Returns (point_estimate, ci_lo, ci_hi) where the CI is the symmetric
    percentile bootstrap interval.  Use bootstrap_paired_ci for paired
    designs.
    """
    rng = random.Random(seed)
    n = len(samples)
    if n < 2:
        v = stat_fn(samples) if n else 0.0
        return v, v, v
    point = stat_fn(samples)
    boots = []
    for _ in range(n_resamples):
        rs = [samples[rng.randrange(n)] for _ in range(n)]
        try:
            boots.append(stat_fn(rs))
        except Exception:
            continue
    if not boots:
        return point, point, point
    boots.sort()
    lo_q = (100.0 - ci) / 2.0
    hi_q = 100.0 - lo_q
    return point, percentile(boots, lo_q), percentile(boots, hi_q)


def bootstrap_paired_ci(stat_fn, blocks_to_rows, n_resamples=1000,
                        ci=95.0, seed=0xC0FFEE):
    """Block-bootstrap: resample BLOCK keys with replacement; rebuild rows
    by concatenating the rows of each chosen block.  Pass to `stat_fn`.

    blocks_to_rows : dict block_key -> list of rows (or list of values)
    stat_fn        : function that takes the flattened resample and
                     returns a scalar.

    Returns (point_estimate, ci_lo, ci_hi).
    """
    rng = random.Random(seed)
    keys = list(blocks_to_rows.keys())
    n = len(keys)
    if n < 2:
        flat = [v for k in keys for v in blocks_to_rows[k]]
        v = stat_fn(flat) if flat else 0.0
        return v, v, v
    flat_full = [v for k in keys for v in blocks_to_rows[k]]
    point = stat_fn(flat_full)
    boots = []
    for _ in range(n_resamples):
        chosen = [keys[rng.randrange(n)] for _ in range(n)]
        flat = []
        for k in chosen:
            flat.extend(blocks_to_rows[k])
        try:
            boots.append(stat_fn(flat))
        except Exception:
            continue
    if not boots:
        return point, point, point
    boots.sort()
    lo_q = (100.0 - ci) / 2.0
    hi_q = 100.0 - lo_q
    return point, percentile(boots, lo_q), percentile(boots, hi_q)


def bootstrap_paired_dual(stat_fn, blocks_to_xy, n_resamples=1000,
                          ci=95.0, seed=0xC0FFEE):
    """Like bootstrap_paired_ci but each block contributes a (xs, ys)
    tuple — for paired pairwise stats (AUC, Cohen's d) where you need
    two cells' observations from the SAME block kept together.

    blocks_to_xy : dict block_key -> (xs_block, ys_block)
                   each xs_block / ys_block is a list (usually 1 element
                   per block in paired-pass design).
    stat_fn      : function(flat_xs, flat_ys) -> scalar.
    """
    rng = random.Random(seed)
    keys = list(blocks_to_xy.keys())
    n = len(keys)
    if n < 2:
        xs = [v for k in keys for v in blocks_to_xy[k][0]]
        ys = [v for k in keys for v in blocks_to_xy[k][1]]
        v = stat_fn(xs, ys) if xs and ys else 0.0
        return v, v, v
    xs_full = [v for k in keys for v in blocks_to_xy[k][0]]
    ys_full = [v for k in keys for v in blocks_to_xy[k][1]]
    point = stat_fn(xs_full, ys_full)
    boots = []
    for _ in range(n_resamples):
        chosen = [keys[rng.randrange(n)] for _ in range(n)]
        xs = []
        ys = []
        for k in chosen:
            xb, yb = blocks_to_xy[k]
            xs.extend(xb)
            ys.extend(yb)
        try:
            boots.append(stat_fn(xs, ys))
        except Exception:
            continue
    if not boots:
        return point, point, point
    boots.sort()
    lo_q = (100.0 - ci) / 2.0
    hi_q = 100.0 - lo_q
    return point, percentile(boots, lo_q), percentile(boots, hi_q)


def empirical_auc(xs, ys):
    n_x, n_y = len(xs), len(ys)
    if n_x == 0 or n_y == 0:
        return 0.5
    ys_sorted = sorted(ys)
    total = 0.0
    for x in xs:
        lo = bisect.bisect_left(ys_sorted, x)
        hi = bisect.bisect_right(ys_sorted, x)
        total += (n_y - hi) + 0.5 * (hi - lo)
    return total / (n_x * n_y)


def cohens_d(xs, ys):
    n_x, n_y = len(xs), len(ys)
    if n_x < 2 or n_y < 2:
        return 0.0
    mx = sum(xs) / n_x
    my = sum(ys) / n_y
    vx = sum((x - mx) ** 2 for x in xs) / (n_x - 1)
    vy = sum((y - my) ** 2 for y in ys) / (n_y - 1)
    pooled_var = ((n_x - 1) * vx + (n_y - 1) * vy) / (n_x + n_y - 2)
    if pooled_var <= 0:
        return 0.0
    return (my - mx) / math.sqrt(pooled_var)


def kendall_tau(xs, ys):
    """Kendall-τ-b on paired observations.  Concordant minus discordant
    pair count, normalized by sqrt((C+D+Tx)(C+D+Ty)) to handle ties."""
    n = len(xs)
    if n != len(ys) or n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    tied_x = 0
    tied_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tied_x += 1
                continue
            if dy == 0:
                tied_y += 1
                continue
            if (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    denom_x = (concordant + discordant + tied_x)
    denom_y = (concordant + discordant + tied_y)
    if denom_x <= 0 or denom_y <= 0:
        return 0.0
    return (concordant - discordant) / math.sqrt(denom_x * denom_y)


def fmt_ci(point, lo, hi, prec=3, sign=False):
    """Format a `point [lo..hi]` triple."""
    fmt = f"%+.{prec}f" if sign else f"%.{prec}f"
    return f"{fmt % point} [{fmt % lo}..{fmt % hi}]"
