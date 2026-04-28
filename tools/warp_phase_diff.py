#!/usr/bin/env python3
"""Per-warp phase diff: localize "where did the −443 cyc come from?".

PROFILE_CYCLES emits @@PROF lines at end of run, atomically accumulated over
all launches:

  @@PROF warp=W tag=TAG phase=P label=L cyc_per_tile=Y wall_pct=Z

One binary run = one @@PROF block (covers ALL passes/variants summed).  To
diff variants, run each in its own binary:

  make fc2-w3x DFLAGS='-DPROFILE_CYCLES'
  for v in dgsnake dgsw lmrev gflip; do
    VARIANT=$v REPS=10 ./fc2-w3x > prof_$v.log
  done
  python3 tools/warp_phase_diff.py --ref dgsnake prof_*.log

For each (warp, phase) cell, prints:
  - cyc_per_tile per variant
  - Δ vs --ref variant (positive = slower than ref on that warp/phase)
  - rank by |Δ| so top "where the time is spent differently" rows surface

Usage:
  python3 tools/warp_phase_diff.py [--ref REF] [--top N] [--phase-only]
                                    [--warp W] LOG [LOG ...]

Filename → variant inferred from `prof_VARIANT.log` pattern, or override
via `@@SAMPLE variant=...` if present in the file.

Stdlib only.
"""
import argparse
import os
import re
import sys
from collections import defaultdict


PROF_RE = re.compile(
    r"^@@PROF warp=(\d+) tag=(\S+) phase=(\S+) label=(\S+) "
    r"cyc_per_tile=([\d.]+) wall_pct=([\d.]+)")
META_RE = re.compile(
    r"^@@PROFMETA wall_cyc_per_tile=([\d.]+) eff_clock_ghz=([\d.]+) "
    r"tiles_per_cluster=(\d+) launches=(\d+)")
SAMPLE_VAR_RE = re.compile(r"^@@SAMPLE\s+\S+\s+variant=(\S+)\s")


def variant_from_path(path):
    base = os.path.basename(path)
    m = re.match(r"prof_(.+)\.log$", base)
    if m:
        return m.group(1)
    m = re.match(r"(.+)\.log$", base)
    if m:
        return m.group(1)
    return base


def parse_log(path):
    """Return (variant_name, dict[(warp, phase, label, tag)] = cyc_per_tile,
    wall_cyc_per_tile)."""
    rows = {}
    wall = None
    inferred_variant = None
    with open(path) as f:
        for line in f:
            m = PROF_RE.match(line)
            if m:
                w = int(m.group(1))
                tag = m.group(2)
                phase = m.group(3)
                label = m.group(4)
                cyc = float(m.group(5))
                rows[(w, phase, label, tag)] = cyc
                continue
            mm = META_RE.match(line)
            if mm:
                wall = float(mm.group(1))
                continue
            sm = SAMPLE_VAR_RE.match(line)
            if sm and inferred_variant is None:
                inferred_variant = sm.group(1)
    name = inferred_variant if inferred_variant else variant_from_path(path)
    return name, rows, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="profile logs (one per variant)")
    ap.add_argument("--ref", default=None,
                    help="reference variant for Δ (default: fastest by wall)")
    ap.add_argument("--top", type=int, default=20,
                    help="show top N (warp, phase) deltas (default 20)")
    ap.add_argument("--warp", type=int, default=None,
                    help="filter to one warp")
    ap.add_argument("--phase-only", action="store_true",
                    help="hide SUM/WALL rows, show numbered phases only")
    args = ap.parse_args()

    parsed = {}
    walls = {}
    for path in args.logs:
        name, rows, wall = parse_log(path)
        if not rows:
            print(f"# skip {path}: no @@PROF rows", file=sys.stderr)
            continue
        parsed[name] = rows
        if wall is not None:
            walls[name] = wall

    if len(parsed) < 2:
        print(f"ERROR: need ≥2 variants, got {len(parsed)} from "
              f"{len(args.logs)} logs", file=sys.stderr)
        sys.exit(1)

    if args.ref is None and walls:
        ref = min(walls, key=lambda k: walls[k])
    elif args.ref is None:
        ref = sorted(parsed.keys())[0]
    else:
        ref = args.ref
    if ref not in parsed:
        print(f"ERROR: --ref {ref} not in parsed variants {list(parsed)}",
              file=sys.stderr)
        sys.exit(1)

    print(f"=== {len(parsed)} variant(s); ref = {ref}"
          f"{f' (wall {walls[ref]:.1f} cyc/tile)' if ref in walls else ''} ===\n")

    keys = sorted({k for v in parsed.values() for k in v.keys()})
    if args.warp is not None:
        keys = [k for k in keys if k[0] == args.warp]
    if args.phase_only:
        keys = [k for k in keys if k[1] not in ("SUM", "WALL")]

    variants = sorted(parsed.keys())
    others = [v for v in variants if v != ref]

    print("Per-(warp, phase) cyc_per_tile and Δ vs ref:")
    name_w = max(8, max((len(v) for v in variants), default=8))
    hdr = f"  {'warp':>4}  {'tag':>4}  {'ph':>3}  {'label':<22}  {ref[:name_w]:>{name_w}}"
    for v in others:
        hdr += f"  {v[:name_w]:>{name_w}}  {'Δ':>{name_w}}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows_sorted = []
    for k in keys:
        ref_v = parsed[ref].get(k, 0.0)
        max_abs_delta = 0.0
        for v in others:
            d = abs(parsed[v].get(k, 0.0) - ref_v)
            if d > max_abs_delta:
                max_abs_delta = d
        rows_sorted.append((max_abs_delta, k, ref_v))
    rows_sorted.sort(reverse=True)

    shown = 0
    for max_d, k, ref_v in rows_sorted:
        if shown >= args.top:
            break
        w, ph, lbl, tag = k
        line = f"  W{w:<3} {tag:>4}  {ph:>3}  {lbl:<22}  {ref_v:>{name_w}.0f}"
        for v in others:
            x = parsed[v].get(k, 0.0)
            d = x - ref_v
            line += f"  {x:>{name_w}.0f}  {d:>+{name_w-1}.0f}"
        print(line)
        shown += 1

    print()
    print("Wall summary:")
    for v in variants:
        if v in walls:
            d = walls[v] - walls.get(ref, 0.0)
            tag = "(ref)" if v == ref else f"Δ {d:+.1f} cyc/tile"
            print(f"  {v:>{name_w}}  wall = {walls[v]:8.1f} cyc/tile  {tag}")


if __name__ == "__main__":
    main()
