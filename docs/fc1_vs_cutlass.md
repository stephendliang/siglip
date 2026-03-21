# FC1 vs CUTLASS: Why We're 3% Faster (and Were Once 82% Slower)

## Background

FC1 is a fused GEMM kernel for the SigLIP2 vision encoder MLP: `[928256, 768] x [768, 3072]^T` with a bias + GELU epilogue. Both our kernel and CUTLASS use the same hardware primitives on B200 (SM100a):

- `cta_group::2` with `__cluster_dims__(2,1,1)` (74 clusters of 2 CTAs)
- 6 warps per CTA: W0 (TMA loads), W1 (tcgen05 MMA), W2-W5 (epilogue)
- 256x256x128 tiles, FP8 inputs, BF16 output
- TMA async bulk loads for A and B matrices
- TMEM for accumulator storage, SWIZZLE_128B SMEM staging for output

FC1 differs from FC2 in two critical ways: (1) the GELU epilogue is compute-intensive (MUFU.TANH + polynomial evaluation per element), and (2) the short K-loop (6 iterations for K=768 vs 24 for FC2) makes the epilogue the bottleneck.

| Metric | Ours (best) | CUTLASS | Gap |
|--------|-------------|---------|-----|
| Time (ms) | 2.247 | 2.320 | **-3.1%** |
| TFLOPS | 1949 | 1888 | +3.2% |
| Registers | 244 | 253 | -9 |
| Tile shape | 256x256x128 | 128x256x128 (2x1) | different M |

## Performance Timeline

The path from 82% slower to 3% faster spanned 7 days and 4 discrete breakthroughs:

| Session | Our ms | CUTLASS ms | Gap | What changed |
|---------|--------|-----------|-----|--------------|
| 20260308 | 4.220 | 2.317 | **+82%** | N_STAGES=4, no GELU optimization |
| 20260313 | 2.471 | 2.320 | **+6.5%** | N_STAGES=5, IS=1, PH1U=2 (default GELU) |
| 20260315 | 2.247 | 2.320 | **-3.1%** | GV=4, IS=0, PH1U=4, ST=1 (breakthrough) |

Three of the four changes that closed the gap are epilogue-specific: GELU variant, phase-1 unroll depth, and store timing. The fourth (N_STAGES 4→5) is pipeline depth, worth ~38% alone (4.22→3.04ms).

## SASS Instruction Profile

Static instruction counts from `cuobjdump -sass`, compiled with each kernel's best known parameters.

| Instruction | CUTLASS | Ours OLD (2.47ms) | Ours BEST (2.25ms) | What it does |
|-------------|:-------:|:-----------------:|:------------------:|----|
| **MUFU** | **32** | **384** | **256** | MUFU.TANH (GELU core) |
| **FADD** | **0** | **384** | **256** | FP32 scalar add (bias + post-tanh) |
| **FFMA** | **128** | **768** | **512** | FP32 fused multiply-add |
| **FMUL** | **98** | **1,152** | **768** | FP32 multiply (GELU polynomial) |
| R2UR | 36 | 279 | 387 | Register → Uniform Register transfer |
| F2FP | 16 | 192 | 128 | FP32 → BF16 conversion |
| LDG | 7 | 96 | 64 | Global memory load (bias) |
| LDS | 49 | 6 | 6 | Shared memory load |
| STS | 11 | 51 | 35 | Shared memory store |
| UTMALDG | 12 | 2 | 12 | TMA async load |
| UTMASTG | 2 | 6 | 8 | TMA async store |
| ELECT | 12 | 11 | 45 | Elect one thread in warp |
| BSSY/BSYNC | 24 | 66 | 62 | Branch convergence |
| UIADD3 | 158 | 33 | 38 | Uniform register arithmetic |
| NANOSLEEP | 38 | 9 | 18 | Mbarrier polling |
| Total insts | **2,392** | **4,600** | **3,736** | |

Our best kernel has 56% more total instructions and 8x more GELU compute instructions (MUFU+FADD+FFMA+FMUL) than CUTLASS — yet runs 3% faster. The next sections explain this paradox.

## Advantage 1: Full Epilogue Unrolling (PH1U=4)

This is the single biggest contributor to our advantage. The interaction sweep showed PH1U has η²=1.000 (complete explanatory power):

| PH1U | ms | MUFU static | Epilogue structure |
|:----:|:----:|:-----:|----|
| 1 | 3.482 | 64 | 64 cols/body, loop 4× |
| 2 | 3.128 | 128 | 128 cols/body, loop 2× |
| 4 | **2.250** | 256 | All 256 cols unrolled, no loop |

PH1U=4 eliminates the epilogue loop entirely. Each epilogue warp processes all 4 SMEM regions (64 cols × 4 = 256 cols) in a single straight-line sequence. This gives the compiler maximum ILP for the MUFU pipeline.

### Why full unrolling helps GELU specifically

GELU has a high-latency critical path per element:

```
FADD (acc+bias)  →  FMUL,FMUL,FFMA (polynomial)  →  MUFU.TANH (~6 cyc)  →  FMUL,FFMA (post-tanh)
```

Each element's chain is ~20 cycles deep (pre-tanh arithmetic + 6-cycle MUFU + post-tanh arithmetic). With 256 independent elements fully unrolled, the hardware scheduler has 256 independent chains to interleave, filling every functional unit pipe every cycle.

With PH1U=2 (128 elements, loop 2×), only 128 chains are available per iteration, and the loop branch between iterations creates a scheduling barrier. With PH1U=1 (64 elements), the scheduler can't fill the MUFU pipeline — only 64 MUFU.TANH spread across 32 threads = 2 MUFU per thread, not enough to hide the 6-cycle latency.

### CUTLASS: 32-element loop body

CUTLASS processes 32 elements per loop body (32 MUFU static), looping ~8 times:

```
LDS.128 R36-R64, [R0+0x8200..0x8270]  ; load 32 accumulators from SMEM staging
LDS.128 bias ...                        ; load bias from SMEM
; 32 × GELU polynomial (FMUL, FFMA, MUFU.TANH, FMUL, FFMA)
@P0 BRA loop_top                        ; branch back for next 32 elements
```

Per iteration: 32 MUFU, 8 loop overhead instructions (BRA, BSSY/BSYNC, address update), plus LDS loads for both accumulator and bias from SMEM. With 32 MUFU per loop body, each thread computes 1 GELU per iteration — insufficient to hide the MUFU pipeline latency.

## Advantage 2: Batched MUFU Scheduling (GV=4)

GELU variant 4 explicitly structures the computation into three phases per 8-element batch:

```c
/* Phase 1: all pre-tanh polynomial computation (8 independent chains) */
float i0 = x0 * (0.7978f + 0.03568f * x0 * x0);
... (i1 through i7)

/* Phase 2: all 8 MUFU.TANH back-to-back */
asm("tanh.approx.f32 %0, %1;" : "=f"(t0) : "f"(i0));
... (t1 through t7)

/* Phase 3: all post-tanh computation + cvt_sts_v4 */
cvt_sts_v4(0.5f*(x0+x0*t0), ..., 0.5f*(x7+x7*t7), SADDR);
```

The `asm` constraint forces 8 consecutive MUFU.TANH instructions in the SASS. With PH1U=4, the full tile body contains 256 MUFU.TANH in 32 batches of 8 — creating long runs of MUFU instructions that saturate the special function unit throughput.

The grid search confirmed GV=4 beats alternatives:

| Variant | ms | Description |
|:-------:|:---:|------------|
| GV=0 (scalar asm) | 3.13 | Per-element asm tanh, interleaved with other ops |
| GV=4 (batch8 asm) | **2.25** | 8 MUFU back-to-back, 3-phase structure |
| GV=5 (batch4+4 asm) | 2.31 | Split batches, lower register pressure |
| BATCH_EPILOGUE=1 | 2.96 | Separate compute/store phases (no GV batching) |

GV=4's 3-phase structure differs from GV=0's per-element approach: GV=0 computes one GELU then immediately stores, creating a long serial dependency. GV=4 computes all 8 GELUs, then stores all 8 — the compiler can issue the 8 MUFU instructions with no intervening STS stalls.

## Advantage 3: Store Timing and Interleave Strategy

| Parameter | Ours (best) | CUTLASS |
|-----------|:-----------:|:-------:|
| STORE_TIMING | 1 (deferred) | interleaved |
| INTERLEAVE_STRATEGY | 0 (no interleave) | N/A |

For FC1, `ST=1` defers all TMA stores to Phase 2, after all GELU computation is complete. Combined with `IS=0` (no interleaved TMA stores during Phase 1), this maximizes the contiguous compute window for GELU.

This is the opposite of what works for PE and FC2, where IS=2 wins by hiding TMA stores in STS stall windows. The difference: FC1's GELU compute is so heavy that there are no stall windows to exploit — the FP32 pipeline is fully utilized. Interleaving TMA stores during GELU compute (IS=1 or IS=2) would compete for instruction issue bandwidth.

The grid search confirmed: IS=0+ST=1 → 2.25ms, IS=1+ST=0 → 2.74ms (21% worse).

## What Made Us Inferior Before

### Era 1: Session 20260308 (4.22ms, +82%)

The earliest FC1 kernel used N_STAGES=4 (4-stage pipeline). The critical limitation: fewer pipeline stages meant the K-loop couldn't fully overlap with the epilogue — W0 had to wait for epilogue warps to free SMEM before issuing next-tile TMA loads. This serialized the K-loop and epilogue, turning the total time into approximately K-loop + epilogue rather than max(K-loop, epilogue).

N_STAGES=5 (the fix in session 20260313) added one more pipeline stage, allowing W0 to issue TMA loads one stage ahead. This alone dropped FC1 from 4.22ms to 3.04ms — a 28% improvement — by enabling full overlap of K-loop with epilogue.

### Era 2: Session 20260313 (2.47ms, +6.5%)

With N_STAGES=5, the kernel had correct pipeline depth but suboptimal epilogue parameters:

| What | Old value | New value | Effect |
|------|:---------:|:---------:|--------|
| PHASE1_UNROLL | 2 | 4 | Full unroll → 28% faster (3.13→2.25ms) |
| GELU_VARIANT | 0 | 4 | Batched MUFU → ~5% faster |
| INTERLEAVE_STRATEGY | 1 | 0 | No interleave → ~3% faster for GELU |
| STORE_TIMING | 0 | 1 | Deferred stores → ~3% faster |

PH1U=4 was the dominant change. But it was only discovered through the grid search interaction sweep, not the tier search — because PH1U=4 is catastrophic without GV=4 (register pressure causes spills with the default GV=0 GELU at 4× unroll). The interaction sweep tested PH1U=4 + GV=4 together, revealing the synergy.

### SASS comparison: old vs new epilogue

The old kernel (session 20260313, PH1U=2, GV=0) had:
- 384 MUFU: more elements in the static binary (old code had different region structure, possibly 6 regions)
- 2 UTMALDG: bias loaded via 96 LDG instead of TMA
- 1152 FMUL + 768 FFMA: per-element GELU interleaved with STS, preventing batched MUFU scheduling
- 279 R2UR: fewer than current (387) because UTMALDG was 2 vs 12

The current kernel trades more UTMALDG (2→12, adding R2UR overhead) for eliminating 96 LDG bias loads and enabling TMA-based bias preloading. But the real win isn't the TMA change — it's the GELU restructuring (GV=4 + PH1U=4) that the UTMALDG change enabled by freeing register pressure from LDG address computation.

## R2UR Analysis: Not a Bottleneck (Yet)

Our FC1 has 387 R2UR — 10.7× more than CUTLASS's 36. This is the same architectural problem documented in the FC2 analysis: inline PTX forces R→UR transfers for every UTMALDG/UTMASTG, while CUTLASS keeps TMA coordinates in UR space permanently.

### Why R2UR doesn't hurt FC1

Unlike FC2 (where R2UR accounts for ~60% of the gap), FC1's R2UR overhead is fully hidden:

| R2UR source | Count | Where it runs | Hidden behind |
|-------------|:-----:|:-------------:|---------------|
| K-loop UTMALDG (12 × ~5) | ~60 | W0 (TMA warp) | W1's MMA compute |
| Epilogue UTMASTG (8 × ~5) | ~40 | W2-W5 | GELU compute in adjacent warps |
| Mbarrier + ELECT ops | ~190 | W0 + W2-W5 | K-loop MMA / GELU compute |
| Other (setup, drain) | ~97 | Various | N/A |
| **Total** | **387** | | |

The critical difference from FC2: FC1's GELU compute (256 MUFU + 512 FFMA + 768 FMUL + 256 FADD = 1,792 FP32 instructions on epilogue warps) creates a massive compute window. W0's R2UR for K-loop TMA loads runs concurrently with this GELU compute on W2-W5. The GELU computation takes thousands of cycles — far longer than the R2UR overhead.

In FC2, the epilogue is a simple bias + residual add (256 FADD + 128 F2FP) — much lighter. With less epilogue compute to hide behind, R2UR overhead becomes exposed.

### R2UR breakdown vs CUTLASS

| Region | CUTLASS R2UR | Our R2UR |
|--------|:------------:|:--------:|
| K-loop (TMA A+B loads) | 0 | ~60 |
| Epilogue source loads | ~4 | 0 (no source load for FC1) |
| Epilogue TMA stores | ~4 | ~40 |
| Mbarrier + coordination | ~28 | ~190 |
| **Total** | **36** | **387** |

CUTLASS's 36 R2UR are almost entirely from mbarrier operations and initial setup — zero in the K-loop hot path. Our 387 include ~60 from K-loop UTMALDG and ~40 from epilogue UTMASTG, but these are overlapped with compute.

## CUTLASS's Architecture: Where It Loses

### Looped epilogue limits GELU throughput

CUTLASS's epilogue uses a 32-element loop body for GELU computation. The SASS shows the pattern:

```
/* Load 32 accumulator values from SMEM staging */
LDS.128 R36, [R0+0x8200]    ; 4 values
LDS.128 R40, [R0+0x8210]    ; 4 values
... (8 total LDS.128 = 32 values)

/* GELU polynomial on 32 values */
FMUL R3, R36, 0.035677...   ; c * x
FFMA Rx, Ry, Rz, 0.7979...  ; k + c*x²
FMUL Rx, Ry, Rz             ; x * inner
MUFU.TANH Rx, Ry             ; tanh(inner)
FFMA Rx, Ry, Rz, Rw          ; 0.5*(x + x*tanh)
... (repeat for 32 elements, interleaved)

/* F2FP conversion + STS */
F2FP ...                      ; FP32 → BF16
STS ...                       ; store to SMEM

@P1 BRA loop_top              ; back for next 32 elements
```

Key weakness: 32 MUFU.TANH per iteration, spread across 32 threads = 1 MUFU per thread per iteration. MUFU has 6-cycle latency — with only 1 in-flight MUFU per thread, the pipeline can't be saturated. The loop branch adds ~5 cycles overhead per iteration.

With 8 iterations (256/32 = 8), CUTLASS pays 8 × ~5 = ~40 cycles of loop overhead. More importantly, each iteration's 32 MUFU instructions must all complete before the next iteration's LDS loads can begin (loop dependency), preventing cross-iteration MUFU pipelining.

### SMEM intermediation adds latency

CUTLASS reads both accumulator values and bias from SMEM:

| Data | CUTLASS path | Our path |
|------|:------------:|:--------:|
| Accumulator | TMEM → SMEM → LDS.128 → FP32 | TMEM → FP32 (tcgen05.ld direct) |
| Bias | TMA → SMEM → LDS.128 → FP32 | LDG → L1 cache → FP32 |

CUTLASS's 49 LDS instructions (vs our 6) reflect the SMEM intermediation cost. Each LDS.128 has ~20-cycle latency. In the epilogue loop body, 8 LDS.128 loads (32 accumulator values) + 4 LDS.128 loads (bias from SMEM) = 12 LDS.128 per iteration, creating a latency floor even before GELU computation begins.

Our kernel reads accumulators directly from TMEM via `tcgen05.ld` (not counted as LDS — these are TMEM loads, much lower latency) and bias from global memory via `LDG.E.128.CONSTANT` with L1 cache hits (the [3072] bias vector fits in L1; all threads in a warp access the same cache lines).

### UR-native advantage doesn't help CUTLASS here

CUTLASS's signature advantage — UR-native TMA coordinates with 0 R2UR in the K-loop — provides its usual cycle savings:

```
UIADD3 UR28, ..., UR24, 0x180, URZ     ; advance smem ptr (UR → UR, no R2UR)
UIADD3 UR32, ..., UR32, 0x10800, URZ   ; advance coordinate (UR → UR)
UTMALDG.3D [UR32], [UR30], desc[UR18]  ; all operands in UR
```

But for FC1, this advantage is worth ~480 cycles per tile (60 R2UR × 8 cycles) while the GELU compute takes ~7,000+ cycles per tile. The R2UR savings is noise-level relative to the epilogue compute time. CUTLASS's UR-native design provides its real advantage in FC2 (short epilogue) and PE (tiny epilogue), not FC1 (heavy epilogue).

## Opportunities for Further Improvement

### 1. BF16 GELU (potential: ~5-10%)

Both kernels currently compute GELU in FP32. CUTLASS's `GELU_taylor<T>` template is instantiated with float. A BF16 GELU using packed HFMA2 instructions could theoretically halve the GELU compute instructions:

| | FP32 GELU (current) | BF16 GELU (hypothetical) |
|---|:---:|:---:|
| MUFU per tile | 256 | 128 (packed) |
| FP arithmetic | 1,536 (FFMA+FMUL+FADD) | ~384 (HFMA2+HMUL2) |
| Precision | Full FP32 | BF16 (~0.1% less accurate) |

Risk: MUFU.TANH may not support packed BF16 on SM100a. If not, the BF16 approach would need to convert to FP32 for tanh, eliminating most of the benefit. Also, GELU's polynomial coefficients (0.7979, 0.0357) require FP32 precision for the intermediate computation — BF16's limited mantissa could introduce visible accuracy loss.

### 2. Reduce epilogue R2UR (~40 R2UR, ~320 cycles)

The 8 UTMASTG in the epilogue each pay ~5 R2UR. These run on epilogue warps after GELU computation — they're on the critical path between GELU completion and tile done. Eliminating these would save ~320 cycles per tile.

This would require the same UR-native coordinate approach as CUTLASS (UIADD3 for address advancement, TMA descriptors permanently in UR). The challenge: our inline PTX uses `"r"` constraints that force R→UR transfers. Fixing this would need either CuTe-style code generation or manual UR register management in inline PTX.

Estimated impact: ~320 cycles / ~7,000 cycle epilogue ≈ 4.6% of epilogue time. With epilogue overlap, the actual wall-clock impact would be smaller.

### 3. SMEM bias loading (uncertain)

Switching from LDG to TMA-based bias loading (like CUTLASS) would:
- Replace 32-64 LDG with ~4 UTMALDG (but adding ~20 R2UR)
- Require SMEM space for the bias buffer (~512 bytes for 256 floats)
- Allow bias preloading before GELU computation begins

The tradeoff is unclear: our current LDG.E.128.CONSTANT loads hit L1 cache reliably (the [3072] bias vector is only 12 KB — fits entirely in L1). The LDG path also avoids SMEM pressure. CUTLASS uses SMEM for bias because it already has the TMA infrastructure, not because SMEM is intrinsically better.

### 4. Instruction footprint reduction (marginal)

Our 3,736 total instructions vs CUTLASS's 2,392 means a 56% larger instruction cache footprint. On SM100a with 256KB L1 instruction cache, both kernels fit comfortably. But in multi-kernel scenarios or with high occupancy, the larger footprint could cause icache thrashing.

The main sources of our larger footprint:
- Full unrolling: 256 MUFU + 768 FMUL + 512 FFMA vs 32 MUFU + 98 FMUL + 128 FFMA (~1,200 extra)
- R2UR overhead: 387 vs 36 (~350 extra)
- ELECT/BSSY/BSYNC: 45+62 vs 12+24 (~70 extra)

The first item (full unrolling) is the source of our speed advantage and should not be reduced. The second and third (R2UR and ELECT overhead) are architectural — fixable only by changing how TMA coordinates are managed.

## Cost Model

Why we're faster despite more instructions. Per-tile cycle estimates:

| Component | CUTLASS cycles | Our cycles | Notes |
|-----------|:-:|:-:|---|
| K-loop (6 iters × 2 TMA loads) | ~4,000 | ~4,000 | Same MMA work, same K=768 |
| K-loop R2UR overhead | 0 | ~480 | Hidden behind MMA on separate warp |
| GELU compute (per warp) | ~5,600 | ~4,200 | **Our advantage**: full unroll + batched MUFU |
| GELU loop overhead | ~320 | 0 | 8 iterations × ~40 cycle/iter |
| LDS bias loading | ~400 | 0 | 12 LDS.128 × ~20 cyc/iter × 8 iters ÷ pipelining |
| LDG bias loading | 0 | ~150 | 8 LDG.128 × ~20 cyc (L1 hit), batched |
| Epilogue STS + TMA store | ~800 | ~1,100 | More STS from full unroll, but deferred |
| Epilogue R2UR | ~30 | ~320 | 4 vs 40 R2UR, on critical path |
| **Total epilogue** | **~7,150** | **~5,770** | |
| **Overlap: max(K-loop, epilogue)** | **~7,150** | **~5,770** | Epilogue-bound for both |
| **Gap** | | **-19%** | Over-predicts (real gap ~3%) |

The cost model over-predicts our advantage (~19%) vs the measured gap (~3%). This is because:
1. CUTLASS's loop overhead is partially hidden by instruction pipelining within iterations
2. CUTLASS's LDS loads pipeline well with compute (not fully serial)
3. Our 1,792 FP32 instructions create register pressure (244 regs) that may cause scheduling inefficiencies the model doesn't capture
4. Our 387 R2UR, while hidden, consume instruction issue bandwidth

The ranking of contributors is reliable: **full GELU unrolling is the primary advantage**, batched MUFU is secondary, and eliminated loop overhead + LDS latency explain the remaining gap.

## Summary

Our 3% advantage over CUTLASS in FC1 stems from a compute-bound epilogue where full unrolling + batched MUFU scheduling outweighs CUTLASS's leaner instruction footprint:

1. **Full epilogue unrolling (PH1U=4)**: 256 independent GELU chains fully exposed to the scheduler vs CUTLASS's 32-element loop with branch overhead and limited cross-iteration ILP. This is the dominant factor — PH1U alone explains a 28% improvement (3.13→2.25ms).

2. **Batched MUFU scheduling (GV=4)**: 8 MUFU.TANH back-to-back per batch saturates the special function unit. CUTLASS interleaves MUFU with polynomial and LDS ops, reducing MUFU pipe utilization.

3. **Store timing (IS=0, ST=1)**: Deferred stores maximize the contiguous compute window. FC1's compute-heavy epilogue has no STS stall windows for TMA stores to hide in — unlike PE/FC2 where interleaved stores are beneficial.

4. **R2UR is irrelevant for FC1**: Our 387 R2UR (10.7× CUTLASS) is entirely hidden behind the ~4,200-cycle GELU compute window. R2UR becomes a problem only when the epilogue is lightweight (FC2's bias+residual).

The path forward is limited. FC1's epilogue is already well-optimized: parameter search found the global optimum (confirmed by grid search η²=1.000 for both PH1U and BATCH_EPILOGUE). Remaining opportunities (BF16 GELU, epilogue R2UR reduction) offer single-digit percentage improvements at most, and carry accuracy or complexity risks.
