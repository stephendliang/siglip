#!/usr/bin/env python3
"""
fc2_w3x N×K dimension sweep, head-to-head vs cuBLASLt rank-1 (BIAS and noBIAS).

Sweeps fc2_w3x across a grid of (N, K) at fixed M, and at each cell runs
cublaslt-introspect twice (EPI=3 BIAS_ONLY and EPI=0 plain GEMM, both
full heuristic search → rank-1) to get cuBLASLt's best for the same
shape with and without epilogue cost. Comparison is in **cycles**, not
ms: fc2_w3x emits per-CTA wall_cyc via clock64, cublaslt-introspect
brackets its N_TIME loop with clock64-sentinel kernels — both
measurements are SM-clock cycles, invariant to GPU frequency throttling
(essential on unlocked-clock vast.ai boxes).

Default grid is power-of-2 (1024/2048/4096/8192) to dodge cuBLASLt's
sparse heuristic table for non-pow2 K (per-tensor FP8 BIAS_ONLY misses
~1/3 of cells at K=1536 etc.) and our kernel's K_iters==NS=6 boundary
bug at K=768. N >1536 uses adaptive N_STAGES (NS5/NS4/NS3) to fit SMEM.

Constraints (from fc2_w3x.cu):
    - N must be multiple of TN=256, and of 64 (BIAS_REG_COUNT static)
    - K must be multiple of TK=128
    - M must be multiple of TM*2=256 (cluster stride)
    - K_iters < 20 → must build with -DNO_PREFILL (kernel doesn't auto-guard)
    - N adaptive NS: 6 if N≤1536, 5 if N≤2048, 4 if N≤4096, 3 if N≤8192

Output:
    data/dim_sweep_w3x_<ts>/results.tsv
    Tables sorted by (N, K), by K/N ratio, by cyc (us, fastest first),
    and by Δ% vs cuBLASLt rank-1 BIAS (most negative = biggest fc2_w3x win).

Usage:
    ./tools/dim_sweep_w3x.py                       # default 4N × 4K pow2 grid + cuBLASLt bias+nobias
    ./tools/dim_sweep_w3x.py --full                # 5N × 8K = 40 cells
    ./tools/dim_sweep_w3x.py --quick               # 2N × 2K = 4 cells
    ./tools/dim_sweep_w3x.py --legacy              # old non-pow2 grid (5N × 4K)
    ./tools/dim_sweep_w3x.py --n 1024,2048 --k 2048,4096
    ./tools/dim_sweep_w3x.py --reps 5              # 5 launches/cell, take min
    ./tools/dim_sweep_w3x.py --no-cublaslt         # skip both head-to-heads
    ./tools/dim_sweep_w3x.py --no-cublaslt-nobias  # skip the EPI=0 second run
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
        "n": [256, 512, 1024, 2048],
        "k": [1024, 2048, 4096, 8192],
    },
    "full": {
        "n": [256, 512, 1024, 1536, 2048],
        "k": [1024, 2048, 4096, 8192],
    },
    "quick": {
        "n": [1024, 2048],
        "k": [2048, 4096],
    },
    "legacy": {
        "n": [256, 512, 768, 1024, 1536],
        "k": [768, 1536, 3072, 6144],
    },
}


def pick_n_stages(n):
    """Pick N_STAGES so MAIN_SMEM (NS * 32KB) + epi/bias fits in 228KB.
    NS=6 for N<=1536 is the in-tree default; larger N forces shallower
    pipeline depth to make room for the bigger BIAS_BYTES (=2*N).
    """
    if n <= 1536: return 6
    if n <= 2048: return 5
    if n <= 4096: return 4
    return 3

RESULT_RE = re.compile(
    r"^@@RESULT\s+ms=(?P<ms>[\d.]+)\s+tflops=(?P<tf>[\d.]+)"
    r".*?valid=(?P<v>\d+)"
)
SAMPLE_CYC_RE = re.compile(r"^@@SAMPLE\s+\S+.*?\bcyc=(?P<cyc>\d+)")
CUBLAS_RESULT_RE = re.compile(
    r"^@@RESULT\s+ms=(?P<ms>[\d.]+)\s+cyc=(?P<cyc>\d+)"
)


def validate_dim(n, k, m):
    errs = []
    if n % TN: errs.append(f"N={n} not %{TN}=0")
    if n % 64: errs.append(f"N={n} not %64=0 (BIAS_REG_COUNT static_assert)")
    if n > 8192: errs.append(f"N={n} > 8192 (no NS that fits)")
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
    ns = pick_n_stages(n)
    dflags = [f"-DM_TOTAL={m}", f"-DN_DIM={n}", f"-DK_DIM={k}",
              f"-DN_STAGES={ns}"]
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
        return {"status": "MISSING", "ms": None, "cyc": None,
                "tflops": None, "valid": None}
    best_ms = None
    best_cyc = None
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
        cyc_run = None
        for line in proc.stdout.splitlines():
            cm = SAMPLE_CYC_RE.match(line)
            if cm:
                cyc = int(cm["cyc"])
                if cyc_run is None or cyc < cyc_run:
                    cyc_run = cyc
            if m is None:
                rm = RESULT_RE.match(line)
                if rm: m = rm
        if not m:
            status = "NO_RESULT"
            break
        ms = float(m["ms"]); tf = float(m["tf"]); v = int(m["v"])
        last_valid = v if last_valid is None else (last_valid and v)
        last_tf = tf
        if best_ms is None or ms < best_ms:
            best_ms = ms
        if cyc_run is not None and (best_cyc is None or cyc_run < best_cyc):
            best_cyc = cyc_run
        if v == 0:
            status = "INVALID"
    log_fp.flush()
    return {"status": status, "ms": best_ms, "cyc": best_cyc,
            "tflops": last_tf, "valid": last_valid}


def run_cublaslt(n, k, m, reps, timeout, log_fp, epi):
    """Run cublaslt-introspect at the given EPI and return best (ms, cyc).
    epi: 0 = no epilogue, 3 = BIAS_ONLY."""
    binary = REPO / "cublaslt-introspect"
    if not binary.exists():
        return {"status": "MISSING_CUBLASLT", "ms": None, "cyc": None}
    args = [str(binary), str(m), str(n), str(k), str(epi)]
    best_ms = None
    best_cyc = None
    status = "ok"
    for r in range(reps):
        try:
            proc = subprocess.run(args, cwd=REPO, capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
            break
        log_fp.write(f"--- cublas epi={epi} rep {r+1}/{reps} ec={proc.returncode}\n")
        log_fp.write(proc.stdout)
        if proc.returncode != 0:
            log_fp.write(proc.stderr)
            status = f"FAIL({proc.returncode})"
            break
        rm = None
        for line in proc.stdout.splitlines():
            cm = CUBLAS_RESULT_RE.match(line)
            if cm: rm = cm; break
        if not rm:
            status = "NO_RESULT"
            break
        ms = float(rm["ms"]); cyc = int(rm["cyc"])
        if best_ms is None or ms < best_ms: best_ms = ms
        if best_cyc is None or cyc < best_cyc: best_cyc = cyc
    log_fp.flush()
    return {"status": status, "ms": best_ms, "cyc": best_cyc}


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
        n_stages=pick_n_stages(n),
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
        elif f in ("cyc", "cublas_cyc", "cublas_none_cyc") and v:
            out.append(f"{int(v)/1e3:.1f}k")
        elif f in ("cublas_ms", "cublas_none_ms") and v: out.append(f"{v:.4f}")
        elif f in ("dcyc", "dn_cyc") and v is not None:
            out.append(f"{int(v)/1e3:+.1f}k")
        elif f in ("dpct", "dn_pct") and v is not None:
            out.append(f"{v:+.2f}%")
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
    ap.add_argument("--legacy", action="store_true",
                    help="use the old non-pow2 grid (5N × 4K)")
    ap.add_argument("--n", help="comma-sep N list (overrides grid)")
    ap.add_argument("--k", help="comma-sep K list (overrides grid)")
    ap.add_argument("--m", type=int, default=M_DEFAULT)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--out", help="output dir (default data/dim_sweep_w3x_<ts>)")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--run-only", action="store_true")
    ap.add_argument("--no-cublaslt", action="store_true",
                    help="skip both cuBLASLt rank-1 head-to-heads")
    ap.add_argument("--no-cublaslt-nobias", action="store_true",
                    help="skip the EPI=0 (no-epilogue) cuBLASLt run; keep BIAS only")
    ap.add_argument("--cublaslt-reps", type=int, default=1,
                    help="reps per cell per EPI mode (each rep does its own "
                         "full heuristic search)")
    ap.add_argument("--cublaslt-timeout", type=int, default=240)
    args = ap.parse_args()

    if args.legacy:
        grid_name = "legacy"
    else:
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

    do_cublaslt = (not args.no_cublaslt) and (not args.build_only)
    if do_cublaslt and not (REPO / "cublaslt-introspect").exists():
        print("[cublaslt] cublaslt-introspect missing — building ...", flush=True)
        rc = subprocess.run(["make", "cublaslt-introspect"], cwd=REPO).returncode
        if rc != 0:
            print("[cublaslt] build FAILED — disabling --vs-cublaslt", file=sys.stderr)
            do_cublaslt = False

    rows = []
    log_fp = open(log_path, "w")
    tsv_fp = open(tsv_path, "w")
    tsv_fp.write("\t".join([
        "n", "k", "n_stages", "k_iters", "n_tiles", "tiles_per_cluster",
        "kn_ratio", "prefill", "floor_ms", "ms", "cyc", "tflops", "eff",
        "cublas_ms", "cublas_cyc", "dcyc", "dpct",
        "cublas_none_ms", "cublas_none_cyc", "dn_cyc", "dn_pct",
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
        cub = {"status": "skip", "ms": None, "cyc": None}
        cubn = {"status": "skip", "ms": None, "cyc": None}
        if do_cublaslt:
            cub = run_cublaslt(n, k, args.m, args.cublaslt_reps,
                                args.cublaslt_timeout, log_fp, epi=3)
            if not args.no_cublaslt_nobias:
                cubn = run_cublaslt(n, k, args.m, args.cublaslt_reps,
                                    args.cublaslt_timeout, log_fp, epi=0)
        dcyc = None; dpct = None
        if res["cyc"] and cub["cyc"]:
            dcyc = res["cyc"] - cub["cyc"]
            dpct = 100.0 * dcyc / cub["cyc"]
        dn_cyc = None; dn_pct = None
        if res["cyc"] and cubn["cyc"]:
            dn_cyc = res["cyc"] - cubn["cyc"]
            dn_pct = 100.0 * dn_cyc / cubn["cyc"]
        merged = dict(res)
        merged["cublas_ms"] = cub["ms"]
        merged["cublas_cyc"] = cub["cyc"]
        merged["cublas_status"] = cub["status"]
        merged["dcyc"] = dcyc
        merged["dpct"] = dpct
        merged["cublas_none_ms"] = cubn["ms"]
        merged["cublas_none_cyc"] = cubn["cyc"]
        merged["cublas_none_status"] = cubn["status"]
        merged["dn_cyc"] = dn_cyc
        merged["dn_pct"] = dn_pct
        rows.append((meta, merged))
        ms = f"{res['ms']:.4f}" if res["ms"] else "-"
        tf = f"{res['tflops']:.1f}" if res["tflops"] else "-"
        ef = f"{eff:.2f}" if eff else "-"
        cy_us = f"{res['cyc']/1e3:.1f}" if res["cyc"] else "-"
        def _fmt_cb(c, status, dc, dp):
            if c:
                sign = "" if dc is None or dc >= 0 else "−"
                mag = abs(dc) if dc is not None else 0
                d = f"{sign}{mag/1e3:.1f}k ({dp:+.2f}%)" if dc is not None else "-"
                return f"{c/1e3:.1f}k", d
            return status[:8], "-"
        cb_cy, dlabel = _fmt_cb(cub["cyc"], cub["status"], dcyc, dpct)
        cbn_cy, dnlabel = _fmt_cb(cubn["cyc"], cubn["status"], dn_cyc, dn_pct)
        print(f"    ms={ms} cyc={cy_us}k tflops={tf} eff={ef} "
              f"v={res.get('valid','-')} NS={meta['n_stages']}  "
              f"cb_bias={cb_cy} Δb={dlabel}  cb_none={cbn_cy} Δn={dnlabel}  "
              f"{res['status']}")
        tsv_fp.write("\t".join(str(x) for x in [
            meta["n"], meta["k"], meta["n_stages"],
            meta["k_iters"], meta["n_tiles"],
            meta["tiles_per_cluster"], f"{meta['kn_ratio']:.4f}",
            meta["prefill"], f"{meta['floor_ms']:.4f}",
            res["ms"] if res["ms"] else "",
            res["cyc"] if res["cyc"] else "",
            res["tflops"] if res["tflops"] else "",
            f"{eff:.4f}" if eff else "",
            cub["ms"] if cub["ms"] else "",
            cub["cyc"] if cub["cyc"] else "",
            dcyc if dcyc is not None else "",
            f"{dpct:.4f}" if dpct is not None else "",
            cubn["ms"] if cubn["ms"] else "",
            cubn["cyc"] if cubn["cyc"] else "",
            dn_cyc if dn_cyc is not None else "",
            f"{dn_pct:.4f}" if dn_pct is not None else "",
            res["valid"] if res["valid"] is not None else "",
            res["status"],
        ]) + "\n")
        tsv_fp.flush()

    log_fp.close()
    tsv_fp.close()

    if args.build_only:
        print(f"\n[build-only] {len(rows)} variants built. Run with --run-only on B200.")
        return

    fields = ["n", "k", "n_stages", "k_iters", "kn_ratio", "tiles_per_cluster",
              "floor_ms", "ms", "cyc", "eff",
              "cublas_cyc", "dcyc", "dpct",
              "cublas_none_cyc", "dn_cyc", "dn_pct",
              "valid", "status"]
    headers = ["N", "K", "NS", "K_it", "K/N", "tile/cl",
               "floor", "ms", "cyc", "eff",
               "cb_bias", "Δb", "Δb%",
               "cb_none", "Δn", "Δn%",
               "v", "status"]

    print_table(rows, "── by raw size (N then K) ──",
                lambda r: (r[0]["n"], r[0]["k"]), fields, headers)
    print_table(rows, "── by K/N ratio ──",
                lambda r: r[0]["kn_ratio"], fields, headers)
    print_table(rows, "── by cyc (fastest first) ──",
                lambda r: r[1].get("cyc") or 1e18, fields, headers)
    print_table(rows, "── by Δ% vs cuBLASLt rank-1 BIAS (most negative = biggest win) ──",
                lambda r: (r[1].get("dpct") if r[1].get("dpct") is not None else 1e9),
                fields, headers)
    print_table(rows, "── by Δ% vs cuBLASLt rank-1 noBIAS (apples-to-oranges; for awareness) ──",
                lambda r: (r[1].get("dn_pct") if r[1].get("dn_pct") is not None else 1e9),
                fields, headers)

    summary = outdir / "summary.txt"
    with open(summary, "w") as fp:
        for title, key in [
            ("by raw size", lambda r: (r[0]["n"], r[0]["k"])),
            ("by K/N ratio", lambda r: r[0]["kn_ratio"]),
            ("by cyc", lambda r: r[1].get("cyc") or 1e18),
            ("by Δ% vs cuBLASLt rank-1 BIAS",
             lambda r: (r[1].get("dpct") if r[1].get("dpct") is not None else 1e9)),
            ("by Δ% vs cuBLASLt rank-1 noBIAS",
             lambda r: (r[1].get("dn_pct") if r[1].get("dn_pct") is not None else 1e9)),
        ]:
            print_table(rows, f"── {title} ──", key, fields, headers, fp=fp)

    print(f"\n[tsv] {tsv_path}")
    print(f"[log] {log_path}")
    print(f"[summary] {summary}")


if __name__ == "__main__":
    main()
