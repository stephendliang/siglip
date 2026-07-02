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

app = modal.App("sm100-fc1-ns-matrix", image=image)

SRC = "/root/src"

# Feasible knob grid under the 227 KB SMEM ceiling (fc1_w3.cu static_assert
# rejects the rest): NS=6 and ES>=4 both need LDG_BIAS headroom, NS=6 also
# forces ES=2. es2/es2_ldg isolate the bias-source cost at matched ES so the
# ns6 delta is attributable to N_STAGES alone.
CELLS = [
    ("es3", ""),
    ("es3_ldg", "-DLDG_BIAS"),
    ("es4_ldg", "-DNUM_EPI_STAGES=4 -DLDG_BIAS"),
    ("es2", "-DNUM_EPI_STAGES=2"),
    ("es2_ldg", "-DNUM_EPI_STAGES=2 -DLDG_BIAS"),
    ("ns6_es2_ldg", "-DN_STAGES=6 -DNUM_EPI_STAGES=2 -DLDG_BIAS"),
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


@app.function(gpu="B200", timeout=1800)
def run_matrix(common: str = "-DSELF_DIFF=100", cells: str = ""):
    """NS x ES x bias-source matrix for fc1_w3, all cells built and run in one
    container so @@CYC values share clock/thermal conditions. Every cell must
    report valid=1 and @@SELFDIFF_SUMMARY dirty=0 to count (timing-only cells
    like STRIP_EPILOGUE pass common="" to skip the gate).
    cells overrides the default grid: "name:flags,name:flags"."""
    cell_list = CELLS
    if cells:
        cell_list = [c.split(":", 1) for c in cells.split(",")]
    for name, flags in cell_list:
        dflags = f"{common} {flags}".strip()
        print(f"\n@@CELL name={name} flags={dflags}", flush=True)
        if _stream(["make", "-B", "fc1-w3", f"DFLAGS={dflags}"]) != 0:
            print(f"@@CELL_FAIL name={name} phase=build", flush=True)
            continue
        rc = _stream(
            ["timeout", "-k", "5", "120", "stdbuf", "-oL", "-eL",
             f"{SRC}/fc1-w3"]
        )
        print(f"@@CELL_EXIT name={name} rc={rc}", flush=True)


@app.local_entrypoint()
def main(common: str = "-DSELF_DIFF=100", cells: str = ""):
    run_matrix.remote(common=common, cells=cells)
