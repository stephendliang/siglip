#!/usr/bin/env python3
"""
Aggregate fc2_w3x PROFILE_CYCLES output across variants × reps into a single
consolidated report.

Parses:
  @@RESULT ms=... tflops=... valid=...
  @@PROFMETA wall_cyc_per_tile=... eff_clock_ghz=... tiles_per_cluster=... launches=...
  @@PROF warp=N tag=TAG phase=P label=LABEL cyc_per_tile=... wall_pct=...

For each variant, computes median / p5 / p95 across reps of each metric.

Emits one human-readable table to stdout (or -o path) plus a machine CSV
alongside it.

Input: <outdir>/run-<variant>-<rep>.log

Usage:
  tools/aggregate_prof.py <outdir> [--out summary.txt]
"""
from __future__ import annotations
import argparse
import collections
import math
import pathlib
import re
import sys


RE_RESULT = re.compile(
    r"@@RESULT\s+ms=(?P<ms>[\d.]+)\s+tflops=(?P<tflops>[\d.]+)\s+checksum=\S+\s+valid=(?P<valid>\d)")
RE_PROFMETA = re.compile(
    r"@@PROFMETA\s+wall_cyc_per_tile=(?P<wall>[\d.]+)\s+eff_clock_ghz=(?P<ghz>[\d.]+)\s+tiles_per_cluster=(?P<tpc>\d+)\s+launches=(?P<n>\d+)")
RE_PROF = re.compile(
    r"@@PROF\s+warp=(?P<warp>\S+)\s+tag=(?P<tag>\S+)\s+phase=(?P<phase>\S+)\s+label=(?P<label>\S+)\s+cyc_per_tile=(?P<cyc>[\d.]+)\s+wall_pct=(?P<pct>[\d.-]+)")


def stats(xs):
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    med = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    mn = xs[0]
    mx = xs[-1]
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / n if n > 1 else 0.0
    sd = math.sqrt(var)
    return {"n": n, "min": mn, "max": mx, "med": med, "mu": mu, "sd": sd}


def parse_log(path):
    result = {"ms": None, "tflops": None, "valid": None, "wall": None,
              "ghz": None, "phases": {}}
    try:
        txt = path.read_text()
    except FileNotFoundError:
        return None
    for line in txt.splitlines():
        if m := RE_RESULT.search(line):
            result["ms"] = float(m["ms"])
            result["tflops"] = float(m["tflops"])
            result["valid"] = int(m["valid"])
        elif m := RE_PROFMETA.search(line):
            result["wall"] = float(m["wall"])
            result["ghz"] = float(m["ghz"])
        elif m := RE_PROF.search(line):
            key = (m["warp"], m["tag"], m["phase"], m["label"])
            result["phases"][key] = {
                "cyc": float(m["cyc"]),
                "pct": float(m["pct"]),
            }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--variants", nargs="*", default=None,
                    help="Override variant order (default: glob)")
    args = ap.parse_args()

    if args.variants:
        variants = list(args.variants)
    else:
        # Glob all run-<variant>-<rep>.log patterns, collect unique <variant>
        seen = {}
        for p in sorted(args.outdir.glob("run-*.log")):
            stem = p.stem  # run-<variant>-<rep>
            parts = stem.split("-")
            if len(parts) < 3 or parts[0] != "run":
                continue
            var = "-".join(parts[1:-1])
            seen.setdefault(var, True)
        variants = list(seen.keys())
        # Put baseline first if present
        if "baseline" in variants:
            variants.remove("baseline")
            variants.insert(0, "baseline")

    data = {}
    for var in variants:
        reps = []
        for p in sorted(args.outdir.glob(f"run-{var}-*.log")):
            r = parse_log(p)
            if r and r["ms"] is not None:
                reps.append(r)
        data[var] = reps

    out_lines = []
    def W(s=""):
        out_lines.append(s)

    W("=" * 96)
    W("fc2_w3x per-phase profile sweep — PROFILE_CYCLES on all variants")
    W(f"outdir: {args.outdir}")
    W("=" * 96)
    W()

    # ── Table 1: wall ms + effective clock ──────────────────────────────
    W("## 1. Wall time + effective clock")
    W()
    W(f"{'variant':<14} {'n':>3}  {'ms_med':>8} {'ms_p5':>8} {'ms_p95':>8} {'ms_sd':>8}  "
      f"{'wall_cyc':>10} {'clock_GHz':>10} {'pass':>5}")
    W("-" * 96)
    base_ms = None
    for var in variants:
        reps = data[var]
        if not reps:
            W(f"{var:<14} --- NO DATA")
            continue
        ms_stats = stats([r["ms"] for r in reps])
        wall_stats = stats([r["wall"] for r in reps if r["wall"]])
        ghz_stats = stats([r["ghz"] for r in reps if r["ghz"]])
        p5 = sorted(r["ms"] for r in reps)[max(0, len(reps)//20)]
        p95 = sorted(r["ms"] for r in reps)[min(len(reps)-1, (len(reps)*19)//20)]
        n_pass = sum(1 for r in reps if r["valid"] == 1)
        if base_ms is None:
            base_ms = ms_stats["med"]
        W(f"{var:<14} {ms_stats['n']:>3}  "
          f"{ms_stats['med']:>8.4f} {p5:>8.4f} {p95:>8.4f} {ms_stats['sd']:>8.4f}  "
          f"{wall_stats['med'] if wall_stats else float('nan'):>10.0f} "
          f"{ghz_stats['med'] if ghz_stats else float('nan'):>10.4f} "
          f"{n_pass:>3}/{ms_stats['n']}")

    # Compute delta vs baseline in ms
    W()
    if base_ms is not None:
        W(f"{'variant':<14}  {'Δms_med':>9}  {'Δ%':>7}")
        W("-" * 40)
        for var in variants:
            reps = data[var]
            if not reps:
                continue
            ms = stats([r["ms"] for r in reps])
            d = ms["med"] - base_ms
            pct = 100.0 * d / base_ms if base_ms else 0.0
            W(f"{var:<14}  {d:>+9.4f}  {pct:>+6.2f}%")
    W()

    # ── Table 2: Per-warp WALL cyc/tile, side-by-side variants ──────────
    # Find all (warp, tag) pairs across variants
    W("## 2. Per-warp wall cyc/tile (loop-bracket), median across reps")
    W()
    warp_tags = set()
    for reps in data.values():
        for r in reps:
            for (w, t, p, _lab) in r["phases"]:
                if p == "WALL":
                    warp_tags.add((w, t))
    warp_tags = sorted(warp_tags, key=lambda x: (int(x[0]) if x[0].isdigit() else 99, x[1]))
    hdr = f"{'warp':<6} {'tag':<4}"
    for var in variants:
        hdr += f" {var:>12}"
    W(hdr)
    W("-" * len(hdr))
    for (w, t) in warp_tags:
        row = f"W{w:<5} {t:<4}"
        key = (w, t, "WALL", "wall_bracket")
        for var in variants:
            reps = data[var]
            vals = [r["phases"].get(key, {}).get("cyc") for r in reps]
            vals = [v for v in vals if v is not None]
            if vals:
                row += f" {stats(vals)['med']:>12.0f}"
            else:
                row += f" {'--':>12}"
        W(row)
    W()

    # ── Table 3: Per-phase breakdown per warp ───────────────────────────
    W("## 3. Per-phase cyc/tile breakdown (median across reps)")
    W("    — phase 0..N are instrumented brackets; SUM = sum of brackets;")
    W("      WALL = full-loop bracket (critical path length)")
    W()

    # Collect phase keys by (warp, tag), preserving in-kernel order
    phase_order = collections.OrderedDict()
    for var in variants:
        for r in data[var]:
            for key in r["phases"]:
                phase_order.setdefault(key, None)

    # Group by (warp, tag)
    by_warp = collections.OrderedDict()
    for key in phase_order:
        (w, t, p, lab) = key
        by_warp.setdefault((w, t), []).append(key)

    for (w, t), keys in by_warp.items():
        W(f"### W{w} {t}")
        # Header: phase/label, then one column per variant
        hdr = f"  {'ph':>3} {'label':<18}"
        for var in variants:
            hdr += f" {var:>12}"
        W(hdr)
        W("  " + "-" * (len(hdr) - 2))
        for key in keys:
            (_w, _t, p, lab) = key
            row = f"  {p:>3} {lab[:18]:<18}"
            for var in variants:
                reps = data[var]
                vals = [r["phases"].get(key, {}).get("cyc") for r in reps]
                vals = [v for v in vals if v is not None]
                if vals:
                    s = stats(vals)
                    row += f" {s['med']:>12.0f}"
                else:
                    row += f" {'--':>12}"
            W(row)

        # Per-phase delta vs baseline
        if "baseline" in data and data["baseline"]:
            W(f"  Δ vs baseline (median cyc/tile):")
            base_reps = data["baseline"]
            hdr2 = f"  {'ph':>3} {'label':<18}"
            for var in variants:
                if var == "baseline":
                    continue
                hdr2 += f" {var:>12}"
            W(hdr2)
            for key in keys:
                (_w, _t, p, lab) = key
                base_vals = [r["phases"].get(key, {}).get("cyc") for r in base_reps]
                base_vals = [v for v in base_vals if v is not None]
                if not base_vals:
                    continue
                b = stats(base_vals)["med"]
                row = f"  {p:>3} {lab[:18]:<18}"
                for var in variants:
                    if var == "baseline":
                        continue
                    reps = data[var]
                    vals = [r["phases"].get(key, {}).get("cyc") for r in reps]
                    vals = [v for v in vals if v is not None]
                    if vals:
                        d = stats(vals)["med"] - b
                        row += f" {d:>+12.0f}"
                    else:
                        row += f" {'--':>12}"
                W(row)
        W()

    # ── Table 4: Top movers (biggest per-phase delta across variants) ───
    W("## 4. Top movers — phases with largest spread across variants")
    W("    (max cyc/tile across variants − min cyc/tile across variants)")
    W()
    movers = []
    for key in phase_order:
        (w, t, p, lab) = key
        per_var = {}
        for var in variants:
            reps = data[var]
            vals = [r["phases"].get(key, {}).get("cyc") for r in reps]
            vals = [v for v in vals if v is not None]
            if vals:
                per_var[var] = stats(vals)["med"]
        if len(per_var) < 2:
            continue
        spread = max(per_var.values()) - min(per_var.values())
        movers.append((spread, (w, t, p, lab), per_var))
    movers.sort(reverse=True)
    W(f"  {'spread':>8} {'warp':<4} {'tag':<4} {'phase':<5} {'label':<18}  best_var / worst_var")
    W("  " + "-" * 88)
    for spread, (w, t, p, lab), per_var in movers[:20]:
        best_var = min(per_var, key=per_var.get)
        worst_var = max(per_var, key=per_var.get)
        W(f"  {spread:>8.0f} W{w:<3} {t:<4} {p:<5} {lab[:18]:<18}  "
          f"{best_var}({per_var[best_var]:.0f}) / {worst_var}({per_var[worst_var]:.0f})")
    W()

    # ── Write ───────────────────────────────────────────────────────────
    text = "\n".join(out_lines) + "\n"
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
