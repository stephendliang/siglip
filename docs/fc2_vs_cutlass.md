# FC2 vs CUTLASS: Why We're 20% Slower

## Background

FC2 is a fused GEMM kernel for the SigLIP2 vision encoder MLP: `[928256, 3072] x [3072, 768]^T` with a bias + residual epilogue. Both our kernel and CUTLASS use the same hardware primitives on B200 (SM100a):

- `cta_group::2` with `__cluster_dims__(2,1,1)` (74 clusters of 2 CTAs)
- 6 warps per CTA: W0 (TMA loads), W1 (tcgen05 MMA), W2-W5 (epilogue)
- 256x256x128 tiles, FP8 inputs, BF16 output
- TMA async bulk loads for A, B, and residual (C) matrices
- TMEM for accumulator storage, SWIZZLE_128B SMEM staging for output

| Metric | Ours | CUTLASS | Gap |
|--------|------|---------|-----|
| Time (ms) | 1.466 | 1.225 | **+20%** |
| TFLOPS | 2977 | 3564 | -16% |
| Registers | 227 | 253 | -26 |
| Tile shape | 256x256x128 | 128x256x128 (2x1) | different M |

Both kernels are correct, validated, and use identical tile/cluster configurations at the hardware level.

## SASS Instruction Profile

Static instruction counts from `cuobjdump -sass`, compiled with each kernel's best known parameters.

| Instruction | CUTLASS | Ours | Ratio | What it does |
|-------------|:-------:|:----:|:-----:|--------------|
| **R2UR** | **22** | **480** | **21.8x** | Register → Uniform Register transfer |
| **UTMALDG** | **8** | **60** | **7.5x** | TMA async load |
| **ELECT** | **10** | **93** | **9.3x** | Elect one thread in warp |
| BSSY/BSYNC | 36 | 134 | 3.7x | Branch convergence bookkeeping |
| FADD | 0 | 256 | n/a | FP32 scalar add |
| F2FP | 128 | 65 | 0.51x | FP32 → BF16 conversion |
| LDS | 142 | 54 | 0.38x | Shared memory load |
| STS | 38 | 27 | 0.71x | Shared memory store |
| UTMASTG | 4 | 4 | 1.0x | TMA async store |
| FENCE | 15 | 5 | 0.33x | Memory fence |
| NANOSLEEP | 34 | 41 | 1.2x | Mbarrier polling |
| Total insts | 3,080 | 3,400 | 1.10x | |

The three rows in bold are the problem. The next sections explain why each is inflated.

## Problem 1: R2UR — Coordinate Transfer Overhead

This is the dominant bottleneck. Every TMA load (`UTMALDG`) requires coordinates (smem destination, tensor coordinates, mbarrier address, TMA descriptor pointer) in Uniform Register (UR) space. How those coordinates get into UR space is where the architectures diverge.

### CUTLASS: UR-native coordinates (0 R2UR per K-loop iteration)

CUTLASS keeps all TMA operands permanently in UR registers. The K-loop body:

```
UIADD3 UR28, ..., UR24, 0x180, URZ     ; update smem ptr (UR → UR)
UIADD3 UR32, ..., UR32, 0x10800, URZ   ; advance coordinate (UR → UR)
UIADD3 UR8, ..., UR8, 0x1c800, URZ     ; next mbar stage (UR → UR)
UTMALDG.3D [UR32], [UR30], desc[UR18]  ; TMA load — all operands already in UR
UTMALDG.3D [UR8], [UR28], desc[UR18]   ; second tile load — reuse UR18 descriptor
```

Per iteration: **2 UTMALDG, 0 R2UR, 8 UIADD3**. The `desc[UR18]` operand is the TMA descriptor, loaded once at kernel start and never touched again. Coordinates advance via UR-to-UR arithmetic (`UIADD3`).

### Ours: R-register coordinates (5 R2UR per UTMALDG)

Our inline PTX computes coordinates in R registers and transfers them:

```
R2UR UR4, R18           ; x-coordinate (R → UR)
R2UR UR5, R19           ; y-coordinate (R → UR)
R2UR UR9, R79           ; mbar address (R → UR)
R2UR UR8, R61           ; smem destination (R → UR)
PLOP3.LUT P0, ...       ; predicate setup
ELECT P4, URZ, PT       ; elect lane 0
R2UR.BROADCAST UR11, R7 ; TMA descriptor (R → UR)
PLOP3.LUT ...           ; convergence predicate
UTMALDG.2D.2CTA [UR8], [UR4]  ; TMA load
BRA.U.ANY ...            ; re-elect loop
```

Per UTMALDG: **5 R2UR + 1 ELECT + 2 PLOP3 + 1 BRA = 9 instructions of overhead**. With KLU=4 unrolling, 8 UTMALDG per loop body × 6 iterations = 48 dynamic loads, each paying this cost.

### Dynamic R2UR comparison

| Region | CUTLASS dynamic R2UR | Our dynamic R2UR |
|--------|:--------------------:|:----------------:|
| K-loop (24 iters × 2 loads) | 0 | 240 |
| Epilogue source loads | ~4 | ~189 |
| **Total per tile** | **~4** | **~429** |

At ~8 cycles effective cost per R2UR: **~3,400 extra cycles per tile** from R2UR alone.

### Why our code uses R2UR

Our TMA loads use inline PTX:

```c
asm volatile(
    "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
    :: "r"(smem_addr), "r"(coord_x), "r"(coord_y), "r"(tma_desc) : "memory");
```

The `"r"` constraint forces all operands through R registers. The compiler must emit R2UR to move them into UR space for the UTMALDG instruction. CUTLASS's CuTe abstraction generates UR-native code where coordinates never touch R registers.

## Problem 2: FP32 vs BF16 Epilogue Arithmetic

| | CUTLASS | Ours |
|---|:---:|:---:|
| Accumulator readback | FP32 from TMEM | FP32 from TMEM |
| Bias precision | BF16 (from SMEM) | FP32 (from SMEM or LDG) |
| Residual precision | BF16 (from SMEM) | BF16→FP32 (widened via LDS unpack) |
| Arithmetic | BF16 packed ops | FP32 FADD |
| Output conversion | F2FP (128 per tile) | F2FP (65 per tile) |
| FADD instructions | 0 | 256 |

CUTLASS converts the FP32 accumulator to BF16 early (128 F2FP), then does bias+residual in BF16 using packed operations — zero FADD. It uses more LDS (142 vs 54) because both bias and residual come through SMEM in BF16.

We keep everything in FP32: add bias (128 FADD), add residual (128 FADD), then convert to BF16 (65 F2FP). Higher precision, but 256 extra FADD at ~4 cycles each = **~1,024 extra cycles per tile**.

## Problem 3: K-Loop Unrolling Creates Code Bloat

CUTLASS software-pipelines 2 K-iterations with alternating TMA descriptor pairs (`desc[UR18]` / `desc[UR28]`):

| | CUTLASS K-loop | Our K-loop |
|---|:---:|:---:|
| Static UTMALDG | 4 (2 per iter × 2 pipelined) | 48 (8 per iter × KLU=4 × unrolled) |
| Static R2UR | 0 | 240 |
| Static ELECT | 0 | 48 |
| Static BSSY/BSYNC | ~4 | ~48 |
| Loop body size | 45 instructions | ~400 instructions |
| Dynamic TMA loads | 48 (same) | 48 (same) |

Both execute the same 48 dynamic TMA loads (24 K-iterations × 2 tiles). But our KLU=4 unrolling replicates the per-load overhead 4x within the loop body, while CUTLASS's tight 45-instruction body reuses UR state across iterations.

Each of our 48 static ELECT calls generates a BSSY/BSYNC convergence pair and a BRA.U.ANY re-elect branch — 3 extra instructions per UTMALDG that CUTLASS doesn't have at all.

## Problem 4: Epilogue Source Loading Architecture

### CUTLASS: single elected producer

One thread (`@!UP0`, via `elect_one_sync()`) issues ALL residual TMA loads for the tile. The 4 epilogue `@!UP0 UTMALDG` instructions are the complete source loading code — one thread, 4 R2UR total, pipelined across StagesC=3 SMEM buffers.

```
@!UP0 UTMALDG.3D [UR16], [UR14], desc[UR10]   ; source load (elected thread only)
@!UP0 UTMALDG.3D [UR8], [UR14], desc[UR20]    ; next region
```

Consumer threads wait on pipeline stages, read from SMEM, compute, and store. Producer and consumers operate on the **same tile simultaneously** — no cross-tile synchronization needed.

### Ours: per-warp independent loading

Each epilogue warp (W2-W5) independently issues its own residual TMA loads for its 32 rows. With W0_RES_PREFETCH, W0 handles pass 0 but each warp still self-loads pass 1. Every warp's lane 0 pays the full ELECT + R2UR + UTMALDG cost independently.

The W0_RES_FULL approach (centralizing all loads in W0) was tested and found to be **net negative** (+65 R2UR, +505 total stalls in SASS) because it introduced cross-tile synchronization (res_pass_mbar, res_consumed_mbar) that added more overhead than it saved. CUTLASS avoids this problem entirely because its producer operates within the epilogue on the same tile.

## Cost Model

Per-tile overhead relative to CUTLASS, estimated from SASS instruction counts and calibrated latencies:

| Source | Extra instructions | Est. cycles/tile | Est. ms (×147 tiles/cluster, 2.1 GHz) |
|--------|:-:|:-:|:-:|
| K-loop R2UR (240 × 8 cyc) | 240 R2UR | 1,920 | 0.134 |
| Epilogue R2UR (~185 × 8 cyc) | 185 R2UR | 1,480 | 0.104 |
| FP32 FADD (256 × 4 cyc) | 256 FADD | 1,024 | 0.072 |
| BSSY/BSYNC (98 × 5 cyc) | 98 pairs | 490 | 0.034 |
| ELECT overhead (83 × 5 cyc) | 83 ELECT | 415 | 0.029 |
| **Total estimated** | | **5,329** | **0.373** |
| **Measured gap** | | **~3,400** | **0.241** |

The estimate over-predicts by ~1.5x because many of these costs overlap: W0's R2UR hides behind W1's MMA stalls, some FADD hides behind STS latency (32 cycles), and the epilogue runs concurrently with the next tile's K-loop. The ranking of contributors is reliable even if the absolute magnitudes aren't precise.

## Summary

The 20% gap to CUTLASS in FC2 is not a single bottleneck but a compounding of architectural differences:

1. **R2UR dominance (~60% of gap)**: Our inline PTX forces R→UR transfers on every TMA load. CUTLASS keeps coordinates in UR space permanently, paying zero R2UR in the hot loop. This is the single largest contributor and affects both the K-loop and epilogue.

2. **FP32 arithmetic (~20% of gap)**: We do bias+residual in FP32 (256 FADD). CUTLASS does it in BF16 (0 FADD). We get ~0.1% better precision; CUTLASS gets ~1,024 fewer cycles per tile.

3. **Code bloat (~20% of gap)**: KLU=4 unrolling replicates per-load overhead (ELECT, R2UR, BSSY/BSYNC) that CUTLASS's tight software-pipelined loop avoids. Same dynamic work, much larger instruction footprint.

None of these are fixable by parameter tuning — they're structural properties of how we emit TMA operations. The grid search correctly identified that parameter space is exhausted; the remaining gap is architectural.
