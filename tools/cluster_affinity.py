#!/usr/bin/env python3
"""Per-cluster slot affinity analysis.

Reads @@CLUSTER lines from a sweep run with EMIT_CLUSTERS=1:
  @@CLUSTER pass=P variant=V c=C cyc=Y

Produces:
  - Per-(variant, cluster_slot) summary: mean cyc, σ, p95, slow-pass count
  - "Always-slow" diagnosis: which cluster slots land in top-K cyc most often,
    surfacing persistent SM/L2-partition affinity (B200: 2 L2 partitions per
    GPC, 74 clusters / 2 partitions = 37 each — uneven cluster→partition
    mapping is a candidate lever the dispatch sim can't see).

Stdlib only.

Usage:
  EMIT_CLUSTERS=1 REPS=128 SWEEP=dgsnake,dgsw_G8,gflip ./fc2-w3x > clusters.log
  python3 tools/cluster_affinity.py clusters.log [--top 5] [--variant V]
"""
import argparse
import math
import re
import sys
from collections import defaultdict


def parse(path):
    """Return rows: [(pass, variant, c, cyc), ...] from @@CLUSTER lines."""
    rows = []
    pat = re.compile(r"^@@CLUSTER pass=(\d+) variant=(\S+) c=(\d+) cyc=(\d+)")
    with open(path) as f:
        for line in f:
            m = pat.match(line)
            if m:
                rows.append((int(m.group(1)), m.group(2),
                             int(m.group(3)), int(m.group(4))))
    return rows


def slot_summary(rows, variant=None):
    """Return dict[c] = (mean_cyc, sigma, p95, n) for one variant
    (or all variants pooled if variant is None)."""
    by_slot = defaultdict(list)
    for p, v, c, cyc in rows:
        if variant is not None and v != variant:
            continue
        by_slot[c].append(cyc)
    out = {}
    for c, vals in by_slot.items():
        n = len(vals)
        m = sum(vals) / n if n else 0.0
        var = sum((x - m) ** 2 for x in vals) / (n - 1) if n > 1 else 0.0
        s = math.sqrt(var)
        sv = sorted(vals)
        idx95 = min(n - 1, max(0, int(0.95 * (n - 1))))
        p95 = sv[idx95] if sv else 0.0
        out[c] = (m, s, p95, n)
    return out


def slow_count(rows, top_k=5, variant=None):
    """For each pass, find the top-K slowest cluster slots; return
    dict[c] = #passes where c was in the top-K slowest."""
    by_pass = defaultdict(list)
    for p, v, c, cyc in rows:
        if variant is not None and v != variant:
            continue
        by_pass[(p, v)].append((cyc, c))
    counts = defaultdict(int)
    n_passes = len(by_pass)
    for key, items in by_pass.items():
        items.sort(reverse=True)
        for cyc, c in items[:top_k]:
            counts[c] += 1
    return counts, n_passes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="output of EMIT_CLUSTERS=1 ./fc2-w3x")
    ap.add_argument("--variant", default=None,
                    help="restrict to one variant (default: per-variant + all)")
    ap.add_argument("--top", type=int, default=5,
                    help="K for slow-slot count (default 5)")
    args = ap.parse_args()

    rows = parse(args.log)
    if not rows:
        print(f"ERROR: no @@CLUSTER lines in {args.log}", file=sys.stderr)
        sys.exit(1)
    variants = sorted({v for _, v, _, _ in rows})
    n_passes = len({(p, v) for p, v, _, _ in rows})
    print(f"Parsed {len(rows)} cluster samples, {len(variants)} variant(s), "
          f"{n_passes} (pass, variant) launches\n")

    targets = [args.variant] if args.variant else variants + [None]

    for tgt in targets:
        label = tgt if tgt else "ALL"
        sm = slot_summary(rows, variant=tgt)
        counts, np_v = slow_count(rows, top_k=args.top, variant=tgt)

        if not sm:
            continue

        items = sorted(sm.items(), key=lambda kv: -kv[1][0])
        all_means = [m for _, (m, _, _, _) in items]
        global_mean = sum(all_means) / len(all_means)
        global_max = max(all_means)
        global_min = min(all_means)
        spread = global_max - global_min

        print(f"=== {label} (n_passes={np_v}, mean cyc across slots = "
              f"{global_mean:.0f}, slot-spread = {spread:.0f} cyc) ===")
        print(f"  Top-{args.top} slowest slots by mean cyc "
              f"(deltas vs slot-grand-mean):")
        for c, (m, s, p95, n) in items[:args.top]:
            slow_pct = 100.0 * counts[c] / max(np_v, 1)
            print(f"    c={c:3d}  mean={m:>10.0f} cyc  σ={s:>6.0f}  "
                  f"p95={p95:>10.0f}  Δ={m - global_mean:>+7.0f}  "
                  f"top{args.top}-pct={slow_pct:5.2f}%  n={n}")
        print(f"  Top-{args.top} fastest slots by mean cyc:")
        for c, (m, s, p95, n) in items[-args.top:][::-1]:
            print(f"    c={c:3d}  mean={m:>10.0f} cyc  σ={s:>6.0f}  "
                  f"p95={p95:>10.0f}  Δ={m - global_mean:>+7.0f}  n={n}")

        # Persistence diagnosis: under the null "slot is random", count
        # of top-K appearances per slot is binomial(np_v, K/74).
        # μ = np_v · p, σ = √(np_v · p · (1−p)). Flag a slot as
        # PERSISTENT if observed count is ≥ 4σ above null mean (≈99.99%).
        if np_v > 0:
            p = args.top / 74.0
            mu = np_v * p
            sigma = math.sqrt(np_v * p * (1.0 - p)) if np_v > 1 else 0.0
            top_slot, top_count = max(counts.items(), key=lambda kv: kv[1]) \
                if counts else (None, 0)
            if top_slot is not None and sigma > 0:
                z = (top_count - mu) / sigma
                if z >= 4.0:
                    verdict = "PERSISTENT"
                elif z >= 2.5:
                    verdict = "MILD"
                else:
                    verdict = "RANDOM (noise)"
                print(f"  Slow-slot persistence: c={top_slot} in top-{args.top} "
                      f"{top_count}/{np_v} passes "
                      f"(null μ={mu:.1f}, σ={sigma:.1f}; z={z:+.2f}) "
                      f"→ {verdict}")
        print()


if __name__ == "__main__":
    main()
