# fc2_w3y — fused-residual port, Round 3 (W4-does-both) B200 result

Date: 2026-06-27. Hardware: B200 via Modal (`modal run dummy_modal.py --target fc2-w3y`).
Dims: FC2 production M=928256, K=3072, N=768.

`fc2_w3y.cu` = `fc2_w3x.cu` + `-DHAS_RESIDUAL`. All residual code lives in the
shared `gemm_w3x_body.cuh` / `epilogue_ops.cuh` behind `#ifdef HAS_RESIDUAL`,
so the OFF path (production fc2_w3x / fc1_w3x) stays byte-identical SASS.

## Outcome: CORRECT, but a perf regression. W4-does-both is disproven.

```
@@SAMPLE rep=1 ms=2.55876 cyc=4699216 cyc_min=4649527 cyc_p50=4681829 cyc_p95=4693415 cyc_max=4699216
FC2-W3X kernel: 2.559 ms  1711.79 TFLOPS
PASS  errors=0/32
@@RESULT ms=2.5588 tflops=1711.79 checksum=0.000000 valid=1 c0=4608.0
```

| kernel | ms | regs | note |
|---|---|---|---|
| fc2_w3x (bias-only) | 1.001 | 71 | production best |
| fc2_w3 (legacy fused-residual) | 1.063 | — | production residual path |
| fc2_w3y `-DNO_RESIDUAL` | 1.003 | 71 | scaffold, residual compiled out |
| **fc2_w3y (Round 3, W4-does-both)** | **2.559** | 87 | this run — correct, ~2.4× legacy |
| target | ~1.027 | — | not reached |

The `-DNO_RESIDUAL` row (shim has `#ifndef NO_RESIDUAL / #define HAS_RESIDUAL`)
matches fc2_w3x within run noise (same 71 regs) — the scaffold itself costs
nothing, so the whole 2.559 ms is the W4-does-both residual work (+16 regs +
serialized TMA), not port overhead. Empirically confirms the SASS-byte-identical
gate on real hardware.

## What this proves

The whole Round-3 machinery is **correct and structurally sound**:

- `valid=1`, `errors=0/32` against a row-dependent golden residual
  (`res(r,c) = 0.25*((r+c)&3)`). This is the only test that catches a residual
  addressing or handshake bug — bias-only / strip cannot. It passed.
- The 16dp256b4x residual gather (16 `ld.shared.b32`, each CVT pair = one
  aligned bf16x2), the fp32 `CVT_ADD_RES_BF16X2`, the packed-4D `tma_res`
  descriptor + store-coord addressing, and the arrive-before-wait handshake
  are all right.
- **No Xid 13** (Round 2's "CTA Not Present" cluster crash from `ld.global` in
  the compute warps) and **no deadlock** (Round 1's TMA-load mbar wedge). This
  is the first fused-residual attempt that runs to completion.

## Why it is slow — localized via PROFILE_CYCLES (ncu-free)

Modal has no Nsight Compute, so the stall was localized with the in-kernel
`clock64()` per-warp/phase profiler (`-DPROFILE_CYCLES`, `@@PROF` lines).
Full residual vs `-DNO_RESIDUAL` control, cyc/tile (147 tiles/cluster):

| metric | NO_RESIDUAL (1.009 ms) | full residual (2.601 ms) | Δ |
|---|---|---|---|
| wall (critical path) | 12,532 | 32,349 | **+19,817** |
| W5 MMA — full-slot **wait** | 4,143 (33%) | **24,164 (74.7%)** | +20,021 |
| W5 MMA — 4× UTCQMMA (compute) | 6,352 | 6,160 | ~0 |
| W4 TMA — A+B work (instrumented) | 9,946 | 9,466 | ~0 |
| W4 TMA — wall | 12,465 | 32,336 | +19,871 |
| W4 — unbracketed = residual issue + RES_CONSUMED wait | ~2,519 | **~22,870** | +20,351 |

The MMA *compute* is unchanged (~6,200 cyc/tile both ways) — nothing executes
slower. The whole +19,817 cyc/tile regression is **W5 starving on the full-slot
mbar** (+20,021), which maps one-to-one onto **W4's residual block growing by
+20,351**. Almost none of that is residual TMA bandwidth (~197 cyc × subpasses);
it is W4 **blocking on `RES_CONSUMED`** — waiting for the epilogue to drain
`out_smem` — *before it can issue the next tile's A+B*. The mainloop load is thus
serialized behind the epilogue drain, and the MMA sits idle ~75% of the time.

This is the KERN_3WARP "an idle warp isn't free issue bandwidth" risk (and the
W0 TMA-sensitivity) materializing exactly as flagged. The "cheap W4-does-both"
hypothesis is **disproven on perf** — but the profiler shows the cause is the
*handshake serialization on the load warp*, not the residual math or BW, so the
fix is purely structural (move the issue off the A+B warp). Everything else
validated, so the rest of the port carries forward unchanged.

## Next

Part 2 of the two-part plan: `RESIDUAL_DEDICATED_WARP` — move the residual TMA
off W4 to its own 7th load warp (w3x dropped to 6 warps only because bias-only
has nothing to load; residual needs a load path). Handshake, gather, descriptor,
and SMEM-reuse all carry over; only the issuing warp moves. Legacy fc2_w3's W2
EpilogueLoad (valid=1 at 1.063) is the proven reference for a separate
residual-load warp coexisting with the mainloop. Once it reaches ~1.063, chase
the ~43 µs epi-structure prize toward the 1.027 target.

Full history: `memory/project-fc2-resadd-port.md` (Rounds 1–3).
