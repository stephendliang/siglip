"""Dump the SASS of the cuBLASLt rank-1 kernel at a given geometry on a
Modal B200, via CUPTI module capture (bench/cublaslt_cubindump.cu).

nvjet kernels have no ELF symbols in any .so on disk and Modal has no ncu;
CUPTI's RESOURCE_MODULE_LOADED callback is the counter-free path. The
in-container step disassembles only the launched nvjet function(s) with
cuobjdump and ships gzipped text back as the function return value —
nothing large goes through the log stream.

    modal run gpu_interface/cubin_dump.py > cubin_dump.log 2>&1
    modal run gpu_interface/cubin_dump.py --m 928256 --n 768 --k 3072 --epi 3
"""

import gzip
import shlex
import subprocess
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.2.0-devel-ubuntu24.04", add_python="3.14"
    )
    .apt_install("make")
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/root/src",
        ignore=[
            "data/**",
            "*.log",
            ".git/**",
            ".claude/**",
            "third_party/**",
            "scratchpad/**",
            "*.ncu-rep",
            "*.csv",
        ],
    )
)

app = modal.App("sm100-cubin-dump", image=image)

SRC = "/root/src"


def _run(cmd, cwd=SRC, env=None, echo=True):
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    p = subprocess.run(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if echo:
        sys.stdout.write(p.stdout)
        sys.stdout.flush()
    else:
        print(f"  [{len(p.stdout)} bytes captured, not echoed]", flush=True)
    return p.returncode, p.stdout


@app.function(gpu="B200", timeout=1800)
def dump(m: int = 928256, n: int = 3072, k: int = 768, epi: int = 2):
    import os

    _run(["bash", "-c",
          "ls /usr/local/cuda/lib64/libcublasLt.so.* "
          "/usr/local/cuda/extras/CUPTI/lib64/libcupti.so* 2>&1; "
          "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"])

    rc, out = _run(["make", "-B", "cublaslt-cubindump"])
    if rc != 0:
        return {"error": f"build failed rc={rc}", "run_log": out}

    workdir = "/root/dumpwork"
    os.makedirs(workdir, exist_ok=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (
        "/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    rc, out = _run(
        ["timeout", "-k", "5", "600", f"{SRC}/cublaslt-cubindump",
         str(m), str(n), str(k), str(epi)],
        cwd=workdir, env=env,
    )
    result = {"run_log": out, "files": {}, "symbols": {}}
    if rc != 0:
        result["error"] = f"cubindump rc={rc}"
        return result

    syms, seen = [], set()
    for line in out.splitlines():
        if line.startswith("@@LAUNCH sym="):
            s = line.split("sym=", 1)[1].strip()
            if s not in seen:
                seen.add(s)
                syms.append(s)
    nvjets = [s for s in syms if "nvjet" in s]
    cubins = sorted(f for f in os.listdir(workdir) if f.startswith("cubin_"))
    print(f"launched syms: {syms}", flush=True)
    print(f"nvjet targets: {nvjets}", flush=True)
    print(f"captured modules: {cubins}", flush=True)

    for cb in cubins:
        rc2, symout = _run(["cuobjdump", "-symbols", cb], cwd=workdir, echo=False)
        result["symbols"][cb] = symout if rc2 == 0 else f"rc={rc2}\n{symout[:2000]}"

    for target in nvjets:
        for cb in cubins:
            rc2, sass = _run(
                ["cuobjdump", "--dump-sass", "--function", target, cb],
                cwd=workdir, echo=False,
            )
            if rc2 == 0 and sass.count("\n") > 50:
                result["files"][f"{target}.sass"] = gzip.compress(sass.encode())
                print(f"SASS for {target}: {sass.count(chr(10))} lines from {cb}",
                      flush=True)
                break
        else:
            print(f"!!! no cubin yielded SASS for {target}", flush=True)

    if not result["files"] and cubins:
        # fallback: full-module dumps so nothing is lost if name-matching failed
        for cb in cubins:
            rc2, sass = _run(["cuobjdump", "--dump-sass", cb], cwd=workdir, echo=False)
            if rc2 == 0 and sass.count("\n") > 50:
                result["files"][f"{cb}.full.sass"] = gzip.compress(sass.encode())
    return result


@app.local_entrypoint()
def main(m: int = 928256, n: int = 3072, k: int = 768, epi: int = 2,
         out: str = ""):
    res = dump.remote(m=m, n=n, k=k, epi=epi)
    outdir = Path(out) if out else REPO_ROOT / "data" / f"cubin_dump_N{n}_K{k}_epi{epi}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "run.log").write_text(res.get("run_log", ""))
    for cb, txt in res.get("symbols", {}).items():
        (outdir / f"{cb}.symbols.txt").write_text(txt)
    for name, gz in res.get("files", {}).items():
        text = gzip.decompress(gz).decode()
        p = outdir / name.replace("/", "_")
        p.write_text(text)
        print(f"saved {p} ({len(text.splitlines())} lines)")
    if "error" in res:
        print(f"ERROR: {res['error']}")
    print(f"artifacts in {outdir}")
