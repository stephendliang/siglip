#!/usr/bin/env python3
"""
fc1_w3 N×K dimension sweep, head-to-head vs cuBLASLt rank-1 (GELU+BIAS and GEMM-only).

Sweeps fc1_w3 across a grid of (N, K) at fixed M, and at each cell runs
cublaslt-introspect twice (EPI=2 GELU_BIAS and EPI=0 plain GEMM, both
full heuristic search → rank-1) to get cuBLASLt's best for the same shape
with and without epilogue cost.

Comparison is in **ms**, not cycles: fc1_w3 currently emits only @@RESULT
ms (no per-CTA wall_cyc clock64 bracket like fc2_w3x). cuBLASLt
introspect emits both ms and cyc; we consume ms for symmetry. On
locked-clock B200 boxes the ms reading is fine for the dim-sweep
granularity (~50-200 µs). For tighter measurements, add wall_cyc emission
to fc1_w3 first.

Production tuning: zigzag TILE_DISPATCH=11 + K_STAGGER=1 — same flags as
the published 1.998 ms FC1 reference. Override via --no-prod-tune to
sweep against the bare fc1-w3 default scheduler.

Default grid is centered on FC1 production point (N=3072, K=768). Includes
non-pow2 K=768 / N=3072 which cuBLASLt enumerates at sm_100a / CUDA 13
with BF16 bias.

Constraints (from fc1_w3.cu):
    - N must be multiple of TN=256 (and of 64 for BIAS_REG_COUNT static_assert)
    - K must be multiple of TK=128
    - M must be multiple of TM*2=256 (cluster stride)
    - K_iters < 20 → kernel auto-defines NO_PREFILL (no flag needed from us)
    - N_STAGES default 5; K_iters < 5 → we pass -DN_STAGES=K_iters-1
    - N_STAGES >= 6 enables BIAS_PER_TILE — we keep NS=5 default to avoid

Output:
    data/dim_sweep_fc1_<ts>/results.tsv
    Tables sorted by (N, K), by K/N ratio, by ms (fastest first),
    and by Δ% vs cuBLASLt rank-1 GELU+BIAS (most negative = biggest fc1_w3 win).

Usage:
    ./tools/dim_sweep_fc1.py                       # default 4N × 4K grid + cuBLASLt bias+nobias
    ./tools/dim_sweep_fc1.py --full                # 5N × 5K = 25 cells
    ./tools/dim_sweep_fc1.py --quick               # 2N × 2K = 4 cells
    ./tools/dim_sweep_fc1.py --pow2                # pure pow2 grid
    ./tools/dim_sweep_fc1.py --n 2048,3072 --k 512,768
    ./tools/dim_sweep_fc1.py --reps 5              # 5 launches/cell, take min
    ./tools/dim_sweep_fc1.py --no-cublaslt         # skip cuBLASLt comparison
    ./tools/dim_sweep_fc1.py --no-cublaslt-nobias  # skip the EPI=0 second run
    ./tools/dim_sweep_fc1.py --no-prod-tune        # bare fc1-w3 default scheduler
    ./tools/dim_sweep_fc1.py --m 464128            # override fixed M
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
        "n": [1024, 2048, 3072, 4096],
        "k": [512, 768, 1024, 1536],
    },
    "full": {
        "n": [1024, 2048, 3072, 4096, 6144],
        "k": [512, 768, 1024, 1536, 2048],
    },
    "quick": {
        "n": [2048, 3072],
        "k": [768, 1024],
    },
    "pow2": {
        "n": [1024, 2048, 4096, 8192],
        "k": [512, 1024, 2048],
    },
}


def pick_n_stages(k):
    """fc1_w3 picks N_STAGES=5 by default. We override only when K_iters
    is too small for NS=5 (need NS <= K_iters). Keep NS<=5 to avoid
    BIAS_PER_TILE complexity at NS>=6.
    """
    k_iters = k // TK
    return max(2, min(5, k_iters - 1))


RESULT_RE = re.compile(
    r"^@@RESULT\s+ms=(?P<ms>[\d.]+)\s+tflops=(?P<tf>[\d.]+)"
    r".*?valid=(?P<v>\d+)"
)
CUBLAS_RESULT_RE = re.compile(
    r"^@@RESULT\s+ms=(?P<ms>[\d.]+)\s+cyc=(?P<cyc>\d+)"
)
CUBLAS_WINNER_RE = re.compile(
    r"^# Winner:\s+rank=1\s+tile=(?P<tile>\d+)\s+stages=(?P<stages>\d+)"
    r"\s+cluster=(?P<cluster>\d+)(?:\s+inner=-?\d+)?"
    r"\s+splitk=(?P<splitk>\d+)\s+swizzle=(?P<swizzle>\d+)"
)
TILE_ID_NAME = {
    23: "128x256", 24: "256x128", 32: "128x192",
    197: "168x128", 201: "176x128", 495: "256x96", 535: "320x192",
}
"""Auto-extend TILE_ID_NAME from cublasLt.h — see dim_sweep_w3x.py."""
def _load_tile_names():
    paths = ("/usr/local/cuda/include/cublasLt.h", "/opt/cuda/include/cublasLt.h")
    rx = re.compile(r"CUBLASLT_MATMUL_TILE_(\d+x\d+)\s*=\s*(\d+)")
    for p in paths:
        try:
            with open(p) as fp:
                for line in fp:
                    m = rx.search(line)
                    if m:
                        tid = int(m.group(2))
                        TILE_ID_NAME.setdefault(tid, m.group(1))
            break
        except FileNotFoundError:
            continue
_load_tile_names()
CLUSTER_SHAPE_NAME = {0: "AUTO"}
"""Auto-load cluster shape enum — see dim_sweep_w3x.py."""
def _load_cluster_names():
    paths = ("/usr/local/cuda/include/cublasLt.h", "/opt/cuda/include/cublasLt.h")
    rx = re.compile(r"CUBLASLT_CLUSTER_SHAPE_(\d+x\d+x\d+)\s*=\s*(\d+)")
    for p in paths:
        try:
            with open(p) as fp:
                for line in fp:
                    m = rx.search(line)
                    if m:
                        CLUSTER_SHAPE_NAME.setdefault(int(m.group(2)), m.group(1))
            break
        except FileNotFoundError:
            continue
_load_cluster_names()


def validate_dim(n, k, m):
    errs = []
    if n % TN: errs.append(f"N={n} not %{TN}=0")
    if n % 64: errs.append(f"N={n} not %64=0 (BIAS_REG_COUNT static_assert)")
    if k % TK: errs.append(f"K={k} not %{TK}=0")
    if m % TM_PACK: errs.append(f"M={m} not %{TM_PACK}=0")
    return errs


def cell_binary(n, k, save_per_cell):
    if save_per_cell:
        return REPO / f"fc1-w3-n{n}-k{k}"
    return REPO / "fc1-w3"


def build_cell(n, k, m, log_fp, prod_tune=True, save_per_cell=False):
    dflags = [f"-DM_TOTAL={m}", f"-DN_DIM={n}", f"-DK_DIM={k}"]
    ns = pick_n_stages(k)
    if ns < 5:
        dflags.append(f"-DN_STAGES={ns}")
    if prod_tune:
        dflags.append("-DTILE_DISPATCH=11")
        dflags.append("-DK_STAGGER=1")
    binary = cell_binary(n, k, save_per_cell)
    cmd = ["make", "-B", "fc1-w3", f"DFLAGS={' '.join(dflags)}"]
    log_fp.write(f"\n[build] N={n} K={k} M={m} flags={dflags}\n$ {' '.join(cmd)}\n")
    log_fp.flush()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=180)
    log_fp.write(proc.stdout)
    log_fp.write(proc.stderr)
    log_fp.flush()
    if proc.returncode != 0:
        return False
    if save_per_cell:
        src = REPO / "fc1-w3"
        if src.exists():
            src.replace(binary)
    return True


def run_cell(reps, timeout, log_fp, binary=None):
    if binary is None:
        binary = REPO / "fc1-w3"
    if not binary.exists():
        return {"status": "MISSING", "ms": None,
                "tflops": None, "valid": None}
    best_ms = None
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
        if v == 0:
            status = "INVALID"
    log_fp.flush()
    return {"status": status, "ms": best_ms,
            "tflops": last_tf, "valid": last_valid}


def run_cublaslt(n, k, m, reps, timeout, log_fp, epi):
    """epi: 0 = no epilogue, 2 = GELU_BIAS, 3 = BIAS_ONLY."""
    binary = REPO / "cublaslt-introspect"
    if not binary.exists():
        return {"status": "MISSING_CUBLASLT", "ms": None, "cyc": None,
                "tile": None, "stages": None, "cluster": None,
                "splitk": None, "swizzle": None}
    args = [str(binary), str(m), str(n), str(k), str(epi)]
    best_ms = None
    best_cyc = None
    best_algo = {"tile": None, "stages": None, "cluster": None,
                 "splitk": None, "swizzle": None}
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
        wm = None
        for line in proc.stdout.splitlines():
            if rm is None:
                m_ = CUBLAS_RESULT_RE.match(line)
                if m_: rm = m_
            if wm is None:
                m_ = CUBLAS_WINNER_RE.match(line)
                if m_: wm = m_
        if not rm:
            status = "NO_RESULT"
            break
        ms = float(rm["ms"]); cyc = int(rm["cyc"])
        is_better = (best_ms is None or ms < best_ms)
        if best_ms is None or ms < best_ms: best_ms = ms
        if best_cyc is None or cyc < best_cyc: best_cyc = cyc
        if is_better and wm is not None:
            best_algo = {"tile":    int(wm["tile"]),
                         "stages":  int(wm["stages"]),
                         "cluster": int(wm["cluster"]),
                         "splitk":  int(wm["splitk"]),
                         "swizzle": int(wm["swizzle"])}
    log_fp.flush()
    return {"status": status, "ms": best_ms, "cyc": best_cyc, **best_algo}


def floor_ms(n, k, m):
    """MMA-retirement floor for this shape. tcgen05 cta_group::2 retires one
    K-iter per ~460 cyc per cluster. Total cyc = K_iters * tiles_per_cluster
    * 460. Useful as a denominator for eff."""
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
        n_stages=pick_n_stages(k),
    )


def _algo_brief(merged, prefix):
    tile = merged.get(f"cb_{prefix}_tile")
    if tile is None: return None
    tname = TILE_ID_NAME.get(tile, f"id={tile}")
    cl = merged.get(f"cb_{prefix}_cluster")
    cl_str = CLUSTER_SHAPE_NAME.get(cl, f"cl={cl}") if cl is not None else "?"
    return (f"{tname}/st{merged.get(f'cb_{prefix}_stages','?')}"
            f"/{cl_str}"
            f"/sk{merged.get(f'cb_{prefix}_splitk','?')}")


def fmt_row(meta, res, fields):
    out = []
    for f in fields:
        if f == "cb_g_algo":
            v = _algo_brief(res, "g")
            out.append(v if v else "-"); continue
        if f == "cb_n_algo":
            v = _algo_brief(res, "n")
            out.append(v if v else "-"); continue
        v = meta.get(f, res.get(f))
        if v is None: out.append("-"); continue
        if f == "ms" and v is not None: out.append(f"{v:.4f}")
        elif f in ("kn_ratio",): out.append(f"{v:.2f}")
        elif f in ("floor_ms",): out.append(f"{v:.3f}")
        elif f in ("tflops",) and v is not None: out.append(f"{v:.1f}")
        elif f == "eff" and v is not None: out.append(f"{v:.2f}")
        elif f in ("cublas_ms", "cublas_none_ms") and v: out.append(f"{v:.4f}")
        elif f in ("dms", "dn_ms") and v is not None:
            out.append(f"{v*1000:+.1f}us")
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
    ap.add_argument("--pow2", action="store_true")
    ap.add_argument("--n", help="comma-sep N list (overrides grid)")
    ap.add_argument("--k", help="comma-sep K list (overrides grid)")
    ap.add_argument("--m", type=int, default=M_DEFAULT)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--out", help="output dir (default data/dim_sweep_fc1_<ts>)")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--run-only", action="store_true")
    ap.add_argument("--no-prod-tune", action="store_true",
                    help="skip -DTILE_DISPATCH=11 -DK_STAGGER=1 production flags")
    ap.add_argument("--no-cublaslt", action="store_true")
    ap.add_argument("--no-cublaslt-nobias", action="store_true")
    ap.add_argument("--cublaslt-reps", type=int, default=1)
    ap.add_argument("--cublaslt-timeout", type=int, default=240)
    args = ap.parse_args()

    if args.pow2:
        grid_name = "pow2"
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
    outdir = Path(args.out) if args.out else REPO / "data" / f"dim_sweep_fc1_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "sweep.log"
    tsv_path = outdir / "results.tsv"

    cells = [(n, k) for n in n_list for k in k_list]
    print(f"[grid] {grid_name}: {len(n_list)}N × {len(k_list)}K = {len(cells)} cells")
    print(f"[m]    {args.m}")
    print(f"[tune] {'production (zigzag TD=11 + K_STAGGER=1)' if not args.no_prod_tune else 'bare default scheduler'}")
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
        "kn_ratio", "prefill", "floor_ms", "ms", "tflops", "eff",
        "cublas_ms", "cublas_cyc", "dms", "dpct",
        "cb_g_tile", "cb_g_stages", "cb_g_cluster", "cb_g_splitk",
        "cublas_none_ms", "cublas_none_cyc", "dn_ms", "dn_pct",
        "cb_n_tile", "cb_n_stages", "cb_n_cluster", "cb_n_splitk",
        "valid", "status",
    ]) + "\n")

    t0 = time.time()
    for i, (n, k) in enumerate(cells, 1):
        meta = cell_meta(n, k, args.m)
        ts_cell = time.time() - t0
        print(f"[{i:2d}/{len(cells)}] N={n} K={k} (K/N={meta['kn_ratio']:.2f} "
              f"K_it={meta['k_iters']} prefill={meta['prefill']} "
              f"floor={meta['floor_ms']:.3f} ms NS={meta['n_stages']}) "
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
            built = build_cell(n, k, args.m, log_fp,
                                prod_tune=not args.no_prod_tune,
                                save_per_cell=save_per_cell)
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
        cubg = {"status": "skip", "ms": None, "cyc": None}
        cubn = {"status": "skip", "ms": None, "cyc": None}
        if do_cublaslt:
            cubg = run_cublaslt(n, k, args.m, args.cublaslt_reps,
                                 args.cublaslt_timeout, log_fp, epi=2)
            if not args.no_cublaslt_nobias:
                cubn = run_cublaslt(n, k, args.m, args.cublaslt_reps,
                                     args.cublaslt_timeout, log_fp, epi=0)
        dms = None; dpct = None
        if res["ms"] and cubg["ms"]:
            dms = res["ms"] - cubg["ms"]
            dpct = 100.0 * dms / cubg["ms"]
        dn_ms = None; dn_pct = None
        if res["ms"] and cubn["ms"]:
            dn_ms = res["ms"] - cubn["ms"]
            dn_pct = 100.0 * dn_ms / cubn["ms"]
        merged = dict(res)
        merged["cublas_ms"] = cubg["ms"]
        merged["cublas_cyc"] = cubg["cyc"]
        merged["cublas_status"] = cubg["status"]
        merged["dms"] = dms
        merged["dpct"] = dpct
        merged["cb_g_tile"]    = cubg.get("tile")
        merged["cb_g_stages"]  = cubg.get("stages")
        merged["cb_g_cluster"] = cubg.get("cluster")
        merged["cb_g_splitk"]  = cubg.get("splitk")
        merged["cublas_none_ms"] = cubn["ms"]
        merged["cublas_none_cyc"] = cubn["cyc"]
        merged["cublas_none_status"] = cubn["status"]
        merged["dn_ms"] = dn_ms
        merged["dn_pct"] = dn_pct
        merged["cb_n_tile"]    = cubn.get("tile")
        merged["cb_n_stages"]  = cubn.get("stages")
        merged["cb_n_cluster"] = cubn.get("cluster")
        merged["cb_n_splitk"]  = cubn.get("splitk")
        rows.append((meta, merged))
        ms = f"{res['ms']:.4f}" if res["ms"] else "-"
        tf = f"{res['tflops']:.1f}" if res["tflops"] else "-"
        ef = f"{eff:.2f}" if eff else "-"
        def _fmt_cb(c, status, dval, dp):
            if c:
                d = f"{dval*1000:+.1f}us ({dp:+.2f}%)" if dval is not None else "-"
                return f"{c:.4f}", d
            return status[:8], "-"
        cb_ms, dlabel = _fmt_cb(cubg["ms"], cubg["status"], dms, dpct)
        cbn_ms, dnlabel = _fmt_cb(cubn["ms"], cubn["status"], dn_ms, dn_pct)
        def _algo_str(a):
            if a.get("tile") is None: return ""
            tname = TILE_ID_NAME.get(a["tile"], f"id={a['tile']}")
            cl_str = CLUSTER_SHAPE_NAME.get(a["cluster"], f"cl={a['cluster']}")
            return (f"  algo[{tname} st={a['stages']} {cl_str}"
                    f" sk={a['splitk']}]")
        algo_g = _algo_str(cubg)
        algo_n = _algo_str(cubn)
        print(f"    ms={ms} tflops={tf} eff={ef} v={res.get('valid','-')}  "
              f"cb_gelu={cb_ms} Δg={dlabel}{algo_g}  "
              f"cb_none={cbn_ms} Δn={dnlabel}{algo_n}  "
              f"{res['status']}")
        def _opt(x): return "" if x is None else x
        tsv_fp.write("\t".join(str(x) for x in [
            meta["n"], meta["k"], meta["n_stages"],
            meta["k_iters"], meta["n_tiles"],
            meta["tiles_per_cluster"], f"{meta['kn_ratio']:.4f}",
            meta["prefill"], f"{meta['floor_ms']:.4f}",
            res["ms"] if res["ms"] else "",
            res["tflops"] if res["tflops"] else "",
            f"{eff:.4f}" if eff else "",
            cubg["ms"] if cubg["ms"] else "",
            cubg["cyc"] if cubg["cyc"] else "",
            f"{dms:.4f}" if dms is not None else "",
            f"{dpct:.4f}" if dpct is not None else "",
            _opt(cubg.get("tile")), _opt(cubg.get("stages")),
            _opt(cubg.get("cluster")), _opt(cubg.get("splitk")),
            cubn["ms"] if cubn["ms"] else "",
            cubn["cyc"] if cubn["cyc"] else "",
            f"{dn_ms:.4f}" if dn_ms is not None else "",
            f"{dn_pct:.4f}" if dn_pct is not None else "",
            _opt(cubn.get("tile")), _opt(cubn.get("stages")),
            _opt(cubn.get("cluster")), _opt(cubn.get("splitk")),
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
              "floor_ms", "ms", "eff",
              "cublas_ms", "dms", "dpct", "cb_g_algo",
              "cublas_none_ms", "dn_ms", "dn_pct", "cb_n_algo",
              "valid", "status"]
    headers = ["N", "K", "NS", "K_it", "K/N", "tile/cl",
               "floor", "ms", "eff",
               "cb_gelu", "Δg", "Δg%", "cb_g algo",
               "cb_none", "Δn", "Δn%", "cb_n algo",
               "v", "status"]

    print_table(rows, "── by raw size (N then K) ──",
                lambda r: (r[0]["n"], r[0]["k"]), fields, headers)
    print_table(rows, "── by K/N ratio ──",
                lambda r: r[0]["kn_ratio"], fields, headers)
    print_table(rows, "── by ms (fastest first) ──",
                lambda r: r[1].get("ms") or 1e18, fields, headers)
    print_table(rows, "── by Δ% vs cuBLASLt rank-1 GELU+BIAS (most negative = biggest win) ──",
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
            ("by ms", lambda r: r[1].get("ms") or 1e18),
            ("by Δ% vs cuBLASLt rank-1 GELU+BIAS",
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
