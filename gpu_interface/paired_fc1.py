"""One-off paired FC1 GELU+BIAS bench: fc1_w3 vs CUTLASS (GELU_taylor) vs
cuBLASLt rank-1 PerTensor vs MXFP8, interleaved rounds in ONE B200 container
so all cyc numbers share a clock domain. Every binary emits avg SM-clock
cycles per launch via the same stream-serialized clock64 sentinel bracket.
After the rounds, one fc1-w3 -DSELF_DIFF=100 pass gates validity.

Run: modal run gpu_interface/paired_fc1.py [--rounds 3]
"""

import shlex
import subprocess
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

# Same image as gpu_interface/runner.py, but third_party/cutlass's include
# trees are re-included (dockerignore-style "!" negation) for fc1-cutlass.
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

app = modal.App("sm100-fc1-paired", image=image)

SRC = "/root/src"

# fc1-w3 built twice (plain + SELF_DIFF) — same binary name, so copy aside.
BUILDS = [
    (["make", "-B", "fc1-w3"], "fc1-w3-plain"),
    (["make", "-B", "fc1-w3", "DFLAGS=-DSELF_DIFF=100"], "fc1-w3-sd"),
    (["make", "-B", "fc1-cutlass"], None),
    (["make", "-B", "cublaslt-fc1"], None),
]

# (binary, args) — cublaslt-fc1 args: M N K epi=2(GELU+bias) scale beta=0
RUNS = [
    ("fc1-w3-plain", []),
    ("fc1-cutlass", []),
    ("cublaslt-fc1", ["928256", "3072", "768", "2", "0", "0"]),
    ("cublaslt-fc1", ["928256", "3072", "768", "2", "1", "0"]),
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
    for cmd, rename in BUILDS:
        if _stream(cmd) != 0:
            print(f"!!! build failed: {cmd}", file=sys.stderr)
            return
        if rename:
            _stream(["cp", f"{SRC}/fc1-w3", f"{SRC}/{rename}"])
    for r in range(rounds):
        for name, args in RUNS:
            tag = f"{name} {' '.join(args)}".strip()
            print(f"\n=== ROUND {r} RUN {tag} ===", flush=True)
            rc = _stream(
                ["timeout", "-k", "5", "180", "stdbuf", "-oL", "-eL",
                 f"{SRC}/{name}", *args]
            )
            print(f"=== ROUND {r} {tag} exit {rc} ===", flush=True)
    print("\n=== SELF_DIFF GATE fc1-w3 (100 pairs) ===", flush=True)
    rc = _stream(
        ["timeout", "-k", "5", "600", "stdbuf", "-oL", "-eL",
         f"{SRC}/fc1-w3-sd"]
    )
    print(f"=== SELF_DIFF exit {rc} ===", flush=True)


@app.local_entrypoint()
def main(rounds: int = 3):
    bench.remote(rounds=rounds)
