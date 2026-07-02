"""One-off paired FC2 fused-residual bench: fc2_w3 vs CUTLASS vs cuBLASLt
rank-1 (beta=1), interleaved rounds in ONE B200 container so all cyc numbers
share a clock domain. All three binaries emit avg SM-clock cycles per launch
via the same stream-serialized clock64 sentinel bracket.

Run: modal run gpu_interface/paired_residual.py [--rounds 3]
"""

import shlex
import subprocess
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

# Same image as gpu_interface/modal.py, but third_party/cutlass's include
# trees are re-included (dockerignore-style "!" negation) for fc2-cutlass.
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
            "!third_party/cutlass/include/**",
            "!third_party/cutlass/tools/util/include/**",
            "scratchpad/**",
            "*.ncu-rep",
            "*.csv",
        ],
    )
)

app = modal.App("sm100-residual-paired", image=image)

SRC = "/root/src"

BUILDS = [
    ["make", "-B", "fc2-w3", "DFLAGS=-DCLOCK_TOTAL"],
    ["make", "-B", "fc2-cutlass"],
    ["make", "-B", "cublaslt-fc2"],
]

# (binary, args) — cublaslt-fc2 args: M N K epi=3(bias) scale=0(PT) beta=1
RUNS = [
    ("fc2-w3", []),
    ("fc2-cutlass", []),
    ("cublaslt-fc2", ["928256", "768", "3072", "3", "0", "1"]),
]


def _stream(cmd: list[str]) -> int:
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


@app.function(gpu="B200", timeout=2400)
def bench(rounds: int = 3):
    print("=== BUILD ===", flush=True)
    for cmd in BUILDS:
        if _stream(cmd) != 0:
            print(f"!!! build failed: {cmd}", file=sys.stderr)
            return
    for r in range(rounds):
        for name, args in RUNS:
            print(f"\n=== ROUND {r} RUN {name} ===", flush=True)
            rc = _stream(
                ["timeout", "-k", "5", "180", "stdbuf", "-oL", "-eL",
                 f"{SRC}/{name}", *args]
            )
            print(f"=== ROUND {r} {name} exit {rc} ===", flush=True)


@app.local_entrypoint()
def main(rounds: int = 3):
    bench.remote(rounds=rounds)
