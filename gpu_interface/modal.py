import shlex
import subprocess
import sys
from pathlib import Path

import modal

# Invoke with `modal run gpu_interface/modal.py ...` (NOT `python modal.py` —
# this file is named modal.py, so running it as a script would shadow the modal
# package on sys.path[0]; under `modal run` the real package is already imported
# so `import modal` resolves fine). The repo root is derived from this file's
# location, so the mount is correct regardless of the CWD `modal run` is called
# from. Sibling verda/vast backends will live alongside this module.
REPO_ROOT = Path(__file__).resolve().parent.parent

# CUDA 13.2 devel image — has nvcc for sm_100a, plus make and python3 (for the
# gen/bias_switch_inc_*.cuh codegen step the Makefile runs). The whole repo is
# folded into the image at /root/src, minus the heavy artifacts we never need
# on the GPU (benchmark data, logs, git, and stray prebuilt binaries).
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

app = modal.App("sm100-gemm-bench", image=image)

SRC = "/root/src"


def _stream(cmd: list[str]) -> int:
    """Run cmd, streaming stdout+stderr live so @@SAMPLE / @@RESULT lines show
    up in the Modal logs as they happen instead of after the run finishes."""
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=SRC,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    return proc.wait()


@app.function(gpu="B200", timeout=600)
def build_and_run(
    target: str = "fc1-w3x",
    dflags: str = "",
    rebuild: bool = True,
    run_args: str = "",
):
    """Build one Makefile target with optional -DPROFILE_* / custom-dim DFLAGS,
    then run the binary it produces (binary name == target name, in /root/src).

    rebuild=True forces `make -B` — required whenever DFLAGS change, since the
    .cu source itself is unchanged and make would otherwise skip the recompile.
    """
    make_cmd = ["make"]
    if rebuild:
        make_cmd.append("-B")
    make_cmd.append(target)
    if dflags:
        make_cmd.append(f"DFLAGS={dflags}")

    print("=== BUILD ===", flush=True)
    if _stream(make_cmd) != 0:
        print(f"!!! build failed for target '{target}'", file=sys.stderr)
        return

    print("\n=== RUN ===", flush=True)
    binary = f"{SRC}/{target}"
    # stdbuf -oL: force the C binary's stdout to line-buffer so @@SAMPLE /
    # @@RESULT (and "Alloc + init + pack done") stream live instead of sitting
    # in libc's 4 KB block buffer until exit — otherwise a kernel hang looks
    # identical to a slow run.
    # timeout -k: a deadlocked kernel (the residual-load handshake bug) must not
    # keep this B200 container alive billing — SIGTERM at 120 s, SIGKILL 5 s
    # later, then _stream returns nonzero, the function returns, container frees.
    rc = _stream(
        ["timeout", "-k", "5", "120", "stdbuf", "-oL", "-eL", binary,
         *shlex.split(run_args)]
    )
    print(f"\n=== exit {rc} ===", flush=True)


@app.local_entrypoint()
def main(
    target: str = "fc1-w3x",
    dflags: str = "",
    rebuild: bool = True,
    run_args: str = "",
):
    """
    Examples:
      modal run gpu_interface/modal.py
      modal run gpu_interface/modal.py --target fc2-w3x
      modal run gpu_interface/modal.py --target fc1-w3x --dflags "-DPER_WARP_STORE"
      modal run gpu_interface/modal.py --target fc1-w3x --dflags "-DPROFILE_CYCLES"
      modal run gpu_interface/modal.py --target fc2-w3 \\
          --dflags "-DM_TOTAL=464128 -DN_DIM=1024 -DK_DIM=2048"
    """
    build_and_run.remote(
        target=target, dflags=dflags, rebuild=rebuild, run_args=run_args
    )
