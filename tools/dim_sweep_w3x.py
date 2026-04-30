#!/usr/bin/env python3
"""
fc2_w3x N×K dimension sweep.

Sweeps fc2_w3x across a grid of (N, K) at fixed M to expose how K:N ratio
and raw size affect wall time. N is capped at 1536 (NS=6 SMEM ceiling).
M kept fixed because it only sets tile count, not per-tile shape.

Constraints (from fc2_w3x.cu):
    - N must be multiple of TN=256
    - K must be multiple of TK=128
    - M must be multiple of TM*2=256 (cluster stride)
    - K_iters < 20 → must build with -DNO_PREFILL (kernel doesn't auto-guard)
    - N > 1536 → must use NS5 (we cap at 1536 to stay on NS6)

Output:
    data/dim_sweep_w3x_<ts>/results.tsv
    Pretty tables sorted by (N, K), by K/N ratio, by ms.

Usage:
    ./tools/dim_sweep_w3x.py                       # default 5N × 4K grid
    ./tools/dim_sweep_w3x.py --full                # 6N × 8K = 48 cells
    ./tools/dim_sweep_w3x.py --quick               # 3N × 3K = 9 cells
    ./tools/dim_sweep_w3x.py --n 256,768,1536 --k 1024,3072,8192
    ./tools/dim_sweep_w3x.py --reps 5              # 5 launches/cell, take min
    ./tools/dim_sweep_w3x.py --build-only          # CPU-VPS-friendly
    ./tools/dim_sweep_w3x.py --run-only            # skip build phase
    ./tools/dim_sweep_w3x.py --m 464128            # override fixed M
"""

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TM, TN, TK = 128, 256, 128
TM_PACK = TM * 2
M_DEFAULT = 928256
N_CLUSTERS = 74
CYC_PER_ITER_FLOOR = 460
GHZ_BASE = 1.813

GRIDS = {
    "default": {
        "n": [256, 512, 768, 1024, 1536],
        "k": [768, 1536, 3072, 6144],
    },
    "full": {
        "n": [256, 512, 768, 1024, 1280, 1536],
        "k": [768, 1024, 1536, 2048, 3072, 4096, 6144, 8192],
    },
    "quick": {
        "n": [256, 768, 1536],
        "k": [1024, 3072, 6144],
    },
}

RESULT_RE = re.compile(
    r"^@@RESULT\s+ms=(?P<ms>[\d.]+)\s+tflops=(?P<tf>[\d.]+)"
    r".*?valid=(?P<v>\d+)"
)


def validate_dim(n, k, m):
    errs = []
    if n % TN: errs.append(f"N={n} not %{TN}=0")
    if n > 1536: errs.append(f"N={n} > 1536 (NS=6 SMEM ceiling)")
    if k % TK: errs.append(f"K={k} not %{TK}=0")
    if m % TM_PACK: errs.append(f"M={m} not %{TM_PACK}=0")
    return errs


def cell_binary(n, k, save_per_cell):
    """Path to binary for this cell. If save_per_cell, give a unique name
    so build-only sweeps don't overwrite each other (useful for build-on-CPU,
    scp, run-on-B200 workflows)."""
    if save_per_cell:
        return REPO / f"fc2-w3x-n{n}-k{k}"
    return REPO / "fc2-w3x"


def build_cell(n, k, m, log_fp, save_per_cell=False):
    k_iters = k // TK
    dflags = [f"-DM_TOTAL={m}", f"-DN_DIM={n}", f"-DK_DIM={k}"]
    if k_iters < 20:
        dflags.append("-DNO_PREFILL")
    binary = cell_binary(n, k, save_per_cell)
    cmd = ["make", "-B", "fc2-w3x", f"DFLAGS={' '.join(dflags)}"]
    log_fp.write(f"\n[build] N={n} K={k} M={m} flags={dflags}\n$ {' '.join(cmd)}\n")
    log_fp.flush()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=180)
    log_fp.write(proc.stdout)
    log_fp.write(proc.stderr)
    log_fp.flush()
    if proc.returncode != 0:
        return False
    if save_per_cell:
        src = REPO / "fc2-w3x"
        if src.exists():
            src.replace(binary)
    return True


def run_cell(reps, timeout, log_fp, binary=None):
    if binary is None:
        binary = REPO / "fc2-w3x"
    if not binary.exists():
        return {"status": "MISSING", "ms": None, "tflops": None, "valid": None}
    best = None
    last_valid = None
    last_tf = None
    status = "ok"
    for r in range(reps):
        try:
            proc = subprocess.run(
                [str(binary)], cwd=REPO, capture_output=True, text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
            break
        log_fp.write(f"--- rep {r+1}/{reps} ec={proc.returncode}\n")
        log_fp.write(proc.stdout)
        if proc.returncode != 0:
            log_fp.write(proc.stderr)
            status = f"FAIL({proc.returncode})"
            break
        m = None
        for line in proc.stdout.splitlines():
            m = RESULT_RE.match(line)
            if m: break
        if not m:
            status = "NO_RESULT"
            break
        ms = float(m["ms"]); tf = float(m["tf"]); v = int(m["v"])
        last_valid = v if last_valid is None else (last_valid and v)
        last_tf = tf
        if best is None or ms < best:
            best = ms
        if v == 0:
            status = "INVALID"
    log_fp.flush()
    return {"status": status, "ms": best, "tflops": last_tf, "valid": last_valid}


def floor_ms(n, k, m):
    """Theoretical MMA-retirement floor for this dim, base clock."""
    tiles_m = m // TM_PACK
    tiles_n = n // TN
    total_tiles = tiles_m * tiles_n
    tiles_per_cluster = (total_tiles + N_CLUSTERS - 1) // N_CLUSTERS
    k_iters = k // TK
    cyc = k_iters * tiles_per_cluster * CYC_PER_ITER_FLOOR
    return cyc / (GHZ_BASE * 1e6)


def cell_meta(n, k, m):
    tiles_m = m // TM_PACK
    tiles_n = n // TN
    total_tiles = tiles_m * tiles_n
    tiles_per_cluster = (total_tiles + N_CLUSTERS - 1) // N_CLUSTERS
    k_iters = k // TK
    return dict(
        n=n, k=k, k_iters=k_iters, n_tiles=tiles_n,
        tiles_per_cluster=tiles_per_cluster,
        kn_ratio=k / n, floor_ms=floor_ms(n, k, m),
        prefill="off" if k_iters < 20 else "on",
    )


def fmt_row(meta, res, fields):
    out = []
    for f in fields:
        v = meta.get(f, res.get(f))
        if v is None: out.append("-"); continue
        if f == "ms" and v is not None: out.append(f"{v:.4f}")
        elif f in ("kn_ratio",): out.append(f"{v:.2f}")
        elif f in ("floor_ms",): out.append(f"{v:.3f}")
        elif f in ("tflops",) and v is not None: out.append(f"{v:.1f}")
        elif f == "eff" and v is not None: out.append(f"{v:.2f}")
        else: out.append(str(v))
    return out


def print_table(rows, title, sort_key, fields, headers, fp=None):
    out = fp or sys.stdout
    rows_sorted = sorted(rows, key=sort_key)
    widths = [len(h) for h in headers]
    cells = []
    for meta, res in rows_sorted:
        merged = {**meta, **res}
        if res.get("ms") and meta.get("floor_ms"):
            merged["eff"] = meta["floor_ms"] / res["ms"]
        cells.append(fmt_row(merged, merged, fields))
    for row in cells:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))
    out.write(f"\n{title}\n")
    out.write(" | ".join(h.ljust(w) for h, w in zip(headers, widths)) + "\n")
    out.write("-+-".join("-" * w for w in widths) + "\n")
    for row in cells:
        out.write(" | ".join(c.ljust(w) for c, w in zip(row, widths)) + "\n")
    out.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n", help="comma-sep N list (overrides grid)")
    ap.add_argument("--k", help="comma-sep K list (overrides grid)")
    ap.add_argument("--m", type=int, default=M_DEFAULT)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--out", help="output dir (default data/dim_sweep_w3x_<ts>)")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--run-only", action="store_true")
    args = ap.parse_args()

    grid_name = "full" if args.full else "quick" if args.quick else "default"
    grid = GRIDS[grid_name]
    n_list = [int(x) for x in args.n.split(",")] if args.n else grid["n"]
    k_list = [int(x) for x in args.k.split(",")] if args.k else grid["k"]

    bad = []
    for n in n_list:
        for k in k_list:
            errs = validate_dim(n, k, args.m)
            if errs: bad.append(f"N={n} K={k}: {'; '.join(errs)}")
    if bad:
        print("invalid dims:\n  " + "\n  ".join(bad), file=sys.stderr)
        sys.exit(2)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.out) if args.out else REPO / "data" / f"dim_sweep_w3x_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "sweep.log"
    tsv_path = outdir / "results.tsv"

    cells = [(n, k) for n in n_list for k in k_list]
    print(f"[grid] {grid_name}: {len(n_list)}N × {len(k_list)}K = {len(cells)} cells")
    print(f"[m]    {args.m}")
    print(f"[out]  {outdir}")

    rows = []
    log_fp = open(log_path, "w")
    tsv_fp = open(tsv_path, "w")
    tsv_fp.write("\t".join([
        "n", "k", "k_iters", "n_tiles", "tiles_per_cluster",
        "kn_ratio", "prefill", "floor_ms", "ms", "tflops", "eff",
        "valid", "status",
    ]) + "\n")

    t0 = time.time()
    for i, (n, k) in enumerate(cells, 1):
        meta = cell_meta(n, k, args.m)
        ts_cell = time.time() - t0
        print(f"[{i:2d}/{len(cells)}] N={n} K={k} (K/N={meta['kn_ratio']:.2f} "
              f"K_it={meta['k_iters']} prefill={meta['prefill']} "
              f"floor={meta['floor_ms']:.3f} ms) "
              f"[+{ts_cell:.1f}s]", flush=True)
        save_per_cell = args.build_only or args.run_only
        binary = cell_binary(n, k, save_per_cell)
        if args.run_only:
            if not binary.exists():
                rows.append((meta, {"status": "MISSING_BINARY", "ms": None,
                                     "tflops": None, "valid": None}))
                print(f"    MISSING_BINARY ({binary.name})")
                continue
        else:
            built = build_cell(n, k, args.m, log_fp, save_per_cell=save_per_cell)
            if not built:
                rows.append((meta, {"status": "BUILD_FAIL", "ms": None,
                                     "tflops": None, "valid": None}))
                print("    BUILD_FAIL")
                continue
        if args.build_only:
            rows.append((meta, {"status": "BUILT", "ms": None,
                                 "tflops": None, "valid": None}))
            print(f"    BUILT → {binary.name}")
            continue
        res = run_cell(args.reps, args.timeout, log_fp, binary=binary)
        eff = (meta["floor_ms"] / res["ms"]) if res["ms"] else None
        rows.append((meta, res))
        ms = f"{res['ms']:.4f}" if res["ms"] else "-"
        tf = f"{res['tflops']:.1f}" if res["tflops"] else "-"
        ef = f"{eff:.2f}" if eff else "-"
        print(f"    ms={ms} tflops={tf} eff={ef} v={res.get('valid','-')} "
              f"{res['status']}")
        tsv_fp.write("\t".join(str(x) for x in [
            meta["n"], meta["k"], meta["k_iters"], meta["n_tiles"],
            meta["tiles_per_cluster"], f"{meta['kn_ratio']:.4f}",
            meta["prefill"], f"{meta['floor_ms']:.4f}",
            res["ms"] if res["ms"] else "",
            res["tflops"] if res["tflops"] else "",
            f"{eff:.4f}" if eff else "",
            res["valid"] if res["valid"] is not None else "",
            res["status"],
        ]) + "\n")
        tsv_fp.flush()

    log_fp.close()
    tsv_fp.close()

    if args.build_only:
        print(f"\n[build-only] {len(rows)} variants built. Run with --run-only on B200.")
        return

    fields = ["n", "k", "k_iters", "kn_ratio", "tiles_per_cluster",
              "floor_ms", "ms", "eff", "tflops", "valid", "status"]
    headers = ["N", "K", "K_it", "K/N", "tile/cl",
               "floor", "ms", "eff", "TFLOPS", "v", "status"]

    print_table(rows, "── by raw size (N then K) ──",
                lambda r: (r[0]["n"], r[0]["k"]), fields, headers)
    print_table(rows, "── by K/N ratio ──",
                lambda r: r[0]["kn_ratio"], fields, headers)
    print_table(rows, "── by ms (fastest first) ──",
                lambda r: r[1].get("ms") or 1e9, fields, headers)

    summary = outdir / "summary.txt"
    with open(summary, "w") as fp:
        for title, key in [
            ("by raw size", lambda r: (r[0]["n"], r[0]["k"])),
            ("by K/N ratio", lambda r: r[0]["kn_ratio"]),
            ("by ms", lambda r: r[1].get("ms") or 1e9),
        ]:
            print_table(rows, f"── {title} ──", key, fields, headers, fp=fp)

    print(f"\n[tsv] {tsv_path}")
    print(f"[log] {log_path}")
    print(f"[summary] {summary}")


if __name__ == "__main__":
    main()
