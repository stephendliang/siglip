"""One-off paired FC1 pacing gate: fc1_w3 base (ring prod) vs -DWGREAD vs
-DWGREAD=2 vs cuBLASLt rank-1 PerTensor, interleaved rounds in ONE B200
container so all cyc numbers share a clock domain. WGREAD swaps the epi
bulk waits to wait_group.read (drops per-wait CCTL.IVALL L1 invalidates);
=2 keeps the tile-last drain a full wait_group 0. Two SELF_DIFF=100 passes
gate both WGREAD variants after the rounds.

Run: modal run gpu_interface/paired_wgread.py [--rounds 5]
"""

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

app = modal.App("sm100-fc1-paired-wgread", image=image)

SRC = "/root/src"

BUILDS = [
    (["make", "-B", "fc1-w3"], "fc1-w3-base"),
    (["make", "-B", "fc1-w3", "DFLAGS=-DWGREAD"], "fc1-w3-wgread"),
    (["make", "-B", "fc1-w3", "DFLAGS=-DWGREAD=2"], "fc1-w3-wgread2"),
    (["make", "-B", "fc1-w3", "DFLAGS=-DWGREAD -DSELF_DIFF=100"], "fc1-w3-sd-wgread"),
    (["make", "-B", "fc1-w3", "DFLAGS=-DWGREAD=2 -DSELF_DIFF=100"], "fc1-w3-sd-wgread2"),
    (["make", "-B", "cublaslt-fc1"], None),
]

RUNS = [
    ("fc1-w3-base", []),
    ("fc1-w3-wgread", []),
    ("fc1-w3-wgread2", []),
    ("cublaslt-fc1", ["928256", "3072", "768", "2", "0", "0"]),
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
def bench(rounds: int = 5):
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
    for sd in ("fc1-w3-sd-wgread", "fc1-w3-sd-wgread2"):
        print(f"\n=== SELF_DIFF GATE {sd} (100 pairs) ===", flush=True)
        rc = _stream(
            ["timeout", "-k", "5", "600", "stdbuf", "-oL", "-eL",
             f"{SRC}/{sd}"]
        )
        print(f"=== SELF_DIFF {sd} exit {rc} ===", flush=True)


@app.local_entrypoint()
def main(rounds: int = 5):
    bench.remote(rounds=rounds)
