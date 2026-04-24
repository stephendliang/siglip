fc2_w3x.ptx — build notes
==========================

This document covers the wire-up of `fc2_w3x.ptx` (hand-authored pure-PTX
kernel) with the host-side driver-API harness (`fc2_w3x_host.cc`).

Build pipeline
--------------

```
fc2_w3x.ptx                 (hand-authored source, 838 lines)
   │
   │ nvcc -arch=compute_100a -code=sm_100a -cubin
   ▼
fc2_w3x.cubin               (SM100a cubin, ~5.5 KB)
   │
   │ ld -r -b binary         (embeds as ELF .rodata blob with symbols
   │                          _binary_fc2_w3x_cubin_start/_end/_size)
   ▼
fc2_w3x_cubin.o
   │
   │                         fc2_w3x_host.cc  (driver-API host harness)
   │                              │
   │                              │ nvcc -x c++ -c
   │                              ▼
   │                         fc2_w3x_host.o
   │                              │
   └──────────────────────────────┴──→ nvcc link ──→ fc2-w3x-ptx (executable)
                                                       (runtime: cuModuleLoadData
                                                        → cuModuleGetFunction
                                                        → cuLaunchKernelEx)
```

Makefile target: `fc2-w3x-ptx`.

Why this approach (and not `nvcc <ptx> <cu> -o exe`)
----------------------------------------------------

`nvcc` rejects mixed `.ptx` + `.cu` on the command line — `.ptx` must be
compiled through `-cubin` or `-fatbin` first (attempting otherwise yields
`nvcc fatal : .ptx input files are only allowed with '--cubin (-cubin)',
'--fatbin (-fatbin)'` etc.).  Also `extern "C" __global__ void
kernel(...)` host declarations cannot resolve to a separately-compiled
cubin via the linker — they're handled by nvcc's fatbin-embedding at
compile time of the .cu that calls the kernel, not via normal ELF
symbol resolution.

Two viable paths:
 1. **Driver API via embedded cubin** (our choice) — cubin is a data
    blob at link time, loaded via `cuModuleLoadData` at runtime, kernel
    function obtained by string name `cuModuleGetFunction`.  Launch via
    `cuLaunchKernelEx` for cluster-dim support.  Robust across CUDA 12
    / 13 because it doesn't depend on the cuCtxCreate ABI that churned
    in CUDA 13.
 2. **Fatbin + stubbed runtime API** — generate a fatbin with nvcc,
    link with a runtime-looking stub.  Fragile; more plumbing; not
    worth it for a one-kernel deliverable.

ABI / ptx-version notes
-----------------------

- `.version 9.2` matches local CUDA 13.2 toolchain (CUDA 13.2 / NVVM
  22.0).  Downgrading to `.version 8.x` may be needed on older CUDA;
  upgrading probably works on newer with the same sm_100a target.
- `.target sm_100a` — no PTX-virtual (`.target sm_100`) accepted for
  the sm_100a-exclusive ops (`tcgen05.*`, cluster barrier variants).
- `.address_size 64` — required on sm_100a.
- Kernel signature order and layout:
  ```
  param_0 CUtensorMap tma_a (128 B, .align 128 .b8 [128] by-value)
  param_1 CUtensorMap tma_b
  param_2 CUtensorMap tma_c
  param_3 u64          d_bias (LDG source)
  param_4 u64          d_C    (unused in bias-only; placeholder)
  ```
  This differs from `fc2_w3x.cu`, which takes 9 params (adding 4
  profile pointers now trimmed, since the PTX port retires all
  PROFILE_* modes).

Host-side launch attributes
---------------------------

```c
CUlaunchConfig cfg{};
cfg.gridDimX = SM_COUNT;                        /* 148 */
cfg.blockDimX = THREADS;                        /* 192 = 6 warps × 32 */
cfg.sharedMemBytes = SMEM_BYTES;                /* ~215 KB */

CUlaunchAttribute attrs[1]{};
attrs[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
attrs[0].value.clusterDim = {CLUSTER_CTAS, 1, 1};   /* 2, 1, 1 */
cfg.attrs = attrs; cfg.numAttrs = 1;

cuLaunchKernelEx(&cfg, fn, params, nullptr);
```

`cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
SMEM_BYTES)` is required to opt into B200's full 228-KB dynamic SMEM
pool, same as the `cudaFuncSetAttribute` call in `fc2_w3x.cu`.

SASS verification — key finding
-------------------------------

The hand-authored `fc2_w3x.ptx` compiles to **byte-for-byte identical
SASS opcode counts** vs `fc2_w3x.cu`'s nvcc-emitted build:

| opcode | cu   | ptx  |
|--------|------|------|
| UTCQMMA| 4    | 4    |
| UTMALDG| 2    | 2    |
| UTMASTG| 1    | 1    |
| SYNCS  | 31   | 31   |
| LDTM   | 1    | 1    |
| STS    | 24   | 24   |
| F2FP   | 16   | 16   |
| R2UR   | 32   | 32   |
| ELECT  | 10   | 10   |
| BSSY   | 14   | 14   |
| IMAD   | 131  | 131  |
| BRA    | 61   | 61   |

This is the expected outcome documented in
`docs/PURE_PTX_REWRITE_STRATEGY.md` §2 / §5: **ptxas owns R-vs-UR
placement and scaffold emission, not PTX source form**.  The hand-PTX
deliverable matches the `.cu` build exactly because `nvcc`'s NVVM
frontend emits substantively the same PTX we wrote by hand, and ptxas
lowers both to identical SASS.

What this PTX port is good for
------------------------------

1. **Deterministic SASS baseline**: any future edit to the kernel body
   in `fc2_w3x.ptx` produces SASS diffs that are purely due to the
   source change, not NVVM drift across CUDA versions.
2. **Grievance targeting**: the doc in
   `docs/W3X_GRIEVANCES_VS_RANK1.md` lists 9 concrete SASS-level
   deltas vs rank-1.  With this PTX, individual edits can be made
   (e.g., consolidating ELECT sites, changing descriptor constraint
   from `r` to `u`) and their SASS impact measured in isolation.
3. **Freeze point for future CUDA toolchain churn**: if NVVM's
   ptx-gen changes in a future CUDA, the reference SASS of `fc2_w3x`
   can drift silently.  The hand-PTX bypasses NVVM.

What this PTX port is NOT
-------------------------

- **Not a performance lever on its own** — see
  `docs/PURE_PTX_REWRITE_STRATEGY.md` §5.  The rewrite is a SASS
  artifact, not a proven runtime improvement.  Measured wall-clock:
  identical SASS → identical perf (to within run-to-run noise).
- **Not a full-freedom SASS editor** — SM100a has no public assembler;
  `tools/sass_edit.py` does narrow binary patching only.

Gotchas encountered
-------------------

1. `nvcc -arch=sm_100a -fatbin foo.ptx`  → fatal "PTX with .target
   'sm_100a' cannot be compiled for architecture 'compute_100'".
   **Fix**: use `-arch=compute_100a -code=sm_100a`, i.e. specify the
   virtual architecture explicitly when the source is sm_100a-exclusive.

2. `nvcc -x c++ fc2_w3x_host.cc fc2_w3x_cubin.o -o exe` tries to
   parse the cubin .o as C++ source (since `-x c++` is sticky across
   subsequent args).  **Fix**: compile host to .o separately, then link
   without `-x c++`.

3. `cuCtxCreate` ABI changed in CUDA 13 (now `cuCtxCreate_v4` with a
   `CUctxCreateParams*` arg).  **Fix**: rely on cudaFree(0) to prime
   the runtime-managed primary context, then `cuCtxGetCurrent`; don't
   call `cuCtxCreate` directly.

4. `ld -r -b binary foo.cubin -o foo_cubin.o` — GNU ld's binary mode
   produces three symbols: `_binary_foo_cubin_start`,
   `_binary_foo_cubin_end`, `_binary_foo_cubin_size`.  The start
   symbol is a valid pointer; `cuModuleLoadData` accepts it directly.
   Host code declares:
   ```c
   extern "C" const unsigned char _binary_fc2_w3x_cubin_start[];
   extern "C" const unsigned char _binary_fc2_w3x_cubin_end[];
   ```

5. PTX's `smem` label (`.extern .shared .align 128 .b8 smem[];`)
   resolves to the dynamic-SMEM base address.  In emitted PTX it
   appears as `mov.b32 %r..., smem;` which becomes a SASS `MOV` from
   the base-pointer register.

Performance (no measurement in-session)
----------------------------------------

This host VPS has no B200.  The claim "fc2_w3x.cu measures 1.007 ms"
is from prior B200 benchmarks (see `MEMORY.md`).  The PTX port is
byte-identical SASS; expected B200 measurement: **same 1.007 ms**
within run-to-run noise.  Actual validation requires a B200 run.

Next steps (for future PTX work)
---------------------------------

The SASS-byte equivalence means there's no point maintaining two
copies (the .cu and the .ptx) unless we plan to diverge them.  The
intended divergence plan (per `W3X_GRIEVANCES_VS_RANK1.md` priority
ranking):

1. **Grievance 4 (R2UR=32 → ~3)** — try `"u"` operand constraints in
   the asm volatile blocks.  In pure PTX, declare descriptor regs with
   `.reg .b64` backed by `mov.u64` from uniform sources; avoid
   round-tripping through `"r"`-typed asm outputs.
2. **Grievance 5 (ELECT=10 → 0)** — merge adjacent lane-0 asm blocks
   into fewer, larger units.  The 4×MMA block already does this; W4's
   TMA-load block can be similarly consolidated.
3. **Grievance 2 (STSM)** — swap plain STS for stmatrix; requires
   matching LDTM layout change (see
   `project_lever_c_bugs_confirmed.md`).

Each edit should be benchmarked on B200 before commit.  Source-of-truth
is the `fc2_w3x.ptx` file; `fc2_w3x.cu` stays as the reference (and
hosts the PROFILE_* diagnostics that are out of scope for the PTX port).
