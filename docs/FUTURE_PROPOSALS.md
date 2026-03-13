# Future Proposals — FC1 GELU Epilogue Rewrite

**Kernel state (2026-03-13):**
- **Patch embed:** 0.524 ms / 2090 TFLOPS — **beats CUTLASS by 11 μs** (0.535 ms)
- **FC1+GELU:** 2.472 ms / 1772 TFLOPS — **loses to CUTLASS by 151 μs** (2.320 ms)
- **FC2:** 1.514 ms / 1232 TFLOPS — loses to CUTLASS by 290 μs (may not be apples-to-apples)

Session: `data/session_20260313_031324/` (B200, 148 SMs, 20 runs, ANOVA).

## Why patch embed wins and FC1 loses

The mainloop is identical — 4 UTCQMMA sites, ~1.61 ms, matched to CUTLASS. The entire
151 μs gap is in the epilogue. The two kernels have fundamentally different epilogue
compute weight:

```
Patch embed epilogue: acc → BF16 CVT (fused bias+pos add) → STS → TMA store
  1 FP32 ALU, 0 MUFU, 129 F2FP. Compute vanishes inside TMEM stall windows.

FC1 epilogue: acc + bias → x² → FMA polynomial → MUFU.TANH → FMA → mul 0.5 → CVT → STS → TMA store
  2304 FP32 ALU, 384 MUFU.TANH, 192 F2FP. Compute IS the bottleneck.
```

Our architecture bets that epilogue compute is cheap — it reads TMEM in 32-col chunks,
interleaving computation and TMA stores to fill stall windows. This bet pays when the
compute fits inside the stall window (patch embed). When the compute dwarfs the stall
window (GELU), the interleaving machinery adds overhead with no benefit.

## Static SASS comparison (per-kernel, not per-warp)

| Metric | Our FC1 | CUTLASS FC1 | Ratio |
|---|---|---|---|
| LDTM.x32 (TMEM load sites) | 18 | 1 | 18x |
| MUFU.TANH | 384 | 32 | 12x |
| FFMA/FMUL/FADD | 2304 | 226 | 10.2x |
| F2FP (BF16 pack) | 192 | 16 | 12x |
| STS | 51 | 11 | 4.6x |
| UTMASTG (TMA stores) | 6 | 2 | 3x |
| WARPSYNC | 37 | 4 | 9.3x |
| MEMBAR + FENCE pairs | 6 | 1 | 6x |
| Total SASS lines | 9227 | 4789 | 1.9x |
| UTCQMMA (mainloop MMA) | 4 | 4 | 1x |

These are static SASS site counts (code shape), not dynamic per-tile execution counts.
Both CUTLASS and our kernel use hardware MUFU.TANH (not software polynomial). Both use
F2FP.BF16.F32.PACK_AB (not free TMA conversion). The difference is code organization:
CUTLASS has 1 LDTM site inside an 8-iteration loop; we have 18 sites from unrolled
paths across 5 epi warps including split-warp variants.

Dynamic per-warp per-tile: both execute 8 TMEM loads (256 cols / 32). But our code
shape causes more instruction cache pressure, more control flow, and prevents the
compiler from optimizing the GELU computation across asm block boundaries.

## What the existing analysis gets wrong

FC1_LOSS.md is directionally right but three claims are too strong:

1. "18 vs 1 TMEM loads" is a static site count, not dynamic traffic. Both kernels read
   256 cols per warp per tile. It still matters (code bloat, icache pressure) but is not
   18x runtime load count.

2. "CUTLASS uses software Taylor GELU." Wrong — the winner SASS contains 32 MUFU.TANH
   sites. Both kernels use hardware tanh approximation.

3. "CUTLASS gets BF16 conversion for free in the store path." Wrong — the winner SASS
   has 16 F2FP.BF16.F32.PACK_AB instructions. CUTLASS just has fewer of them (16 vs 192)
   because its loop body processes 32 values once, while our unrolled epilogue has 12x
   more sites.

## What the grid search missed

The grid search found the best point inside the current architecture. The top 6 FC1
configs are within 14 μs of each other (2.472-2.486 ms). The 151 μs gap to CUTLASS is
structural, not parametric. But the search itself had significant waste:

**Tier 2 wasted half its budget.** 64 of 128 tier-2 configs were COMPILE_FAIL because
every `TMEM_LOAD_WIDTH=64` FC1 variant died. The one axis closest to CUTLASS's wider
strip-read style never got data. Fix or remove from sweep.

**Tier 3 wasted most of its budget.** 540 configs expanded from 135 because
SNAKE_ORDER (η²=0.000, FC1) and CVT_ADD_FUSED (η²=0.000, FC1) multiplied the space by
4x. ~405 evaluations went to dead knobs.

**Tier 4 is partly fake for FC1.** PRELOAD_MODE=2 only differs from mode 1 for
BIAS_ADD (full preload of all 8 uint4). For BIAS_GELU, modes 1 and 2 take the same
code path (preload first 8 bias floats, reload remaining 56 in the hot loop). The
"6-config" tier only had ~4 meaningful behaviors. The 1 μs top-two difference is noise.

**Balanced ANOVA parameter importance (FC1):**

| Parameter | balanced η² | Effect | Status |
|---|---|---|---|
| PHASE1_UNROLL | 0.986 | =4 catastrophic (+1.4 ms) | Understood, =2 locked |
| PRELOAD_MODE | 0.909 | 41 μs spread (but modes 1≈2 for GELU) | Mostly fake |
| INTERLEAVE_STRATEGY | 0.827 | =1 best by 63-134 μs | Understood |
| NUM_EPI_WARPS | 0.813 | =5 best by 359 μs | Locked |
| STAGGER_CYCLES | 0.050 | ~37 μs spread, =0 best | Noise-level |
| K_LOOP_UNROLL | 0.003 | ~87 μs spread but not significant | Noise for FC1 |
| Everything else | <0.003 | <5 μs | Dead |

## The structural gap: what needs to change

### Problem 1: Inline PTX GELU prevents compiler optimization

`GELU_CVT_STS_V4` (fc1_gelu.cu:38) is a monolithic asm block processing 8 elements.
It declares 32 internal `.reg` variables. The compiler cannot:
- Eliminate common subexpressions across the 4 calls per 32-col chunk
- Reorder MUFU.TANH with FFMA to hide SFU latency
- Pipeline loads from the next chunk while computing the current one
- Share intermediate registers across invocations

CUTLASS writes GELU in C++ (`GELU_taylor` class with `operator()` on scalar float).
The nvcc backend sees all 32 values in registers simultaneously and schedules the
full GELU polynomial across all of them. Result: 226 FP ALU + 32 MUFU vs our
2304 FP ALU + 384 MUFU for the same math.

**Fix:** Replace the inline PTX GELU macro with a C++ device function that operates
on float values already in named C++ variables. Let nvcc handle scheduling and register
allocation for the compute. Keep inline PTX only for operations the compiler cannot
emit: TMEM loads, TMA stores, mbarrier ops, fence.proxy.async.

### Problem 2: 32-col interleaved loop is wrong granularity for heavy epilogues

The epilogue walks 256 cols in 32-col chunks (TMEM_LOAD_WIDTH=32), with each chunk
doing: TMEM_LOAD → TMEM_WAIT → GELU on 32 values → STS → conditionally TMA store +
sync. This means:

- 8 TMEM_WAIT barriers per warp per tile (each a pipeline drain)
- Up to 4 syncwarp + fence.proxy.async + TMA store points per tile (INTERLEAVE_STRATEGY=1)
- GELU compute fragmented into 4 batches of 8 elements per asm block, 4 blocks per chunk,
  8 chunks per tile

CUTLASS does: 1 LDTM.x32 → GELU on all 32 values in registers → F2FP → STS → loop.
One TMEM_WAIT per 32-col chunk (same), but GELU compute is one contiguous block of
32 MUFU + ~100 FP ALU that the compiler schedules holistically. TMA stores happen
after the full 256-col loop (batch), not inline.

**Fix options (new structural knobs to add):**

`GELU_VECTOR_WIDTH={8,16,32}` — How many elements to process per GELU call. Current
is 8 (the asm block width). Increasing to 16 or 32 gives the compiler more values to
schedule across, but needs more live registers. With FC1 at 188 regs (67 free), there
is headroom. 32-wide would match CUTLASS's per-iteration granularity.

`BIAS_PRELOAD_DEPTH={8,32,64}` — How many bias floats to preload before TMEM_WAIT.
Currently 8 (one float4x2 load). CUTLASS preloads bias for all 32 cols. Loading all
32 bias values before TMEM_WAIT fills the latency window with useful LDG instead of
idle cycles.

`STORE_TIMING={inline,after128,after256}` — When to issue TMA stores. `inline`=current
(per-region or half-batch). `after128`=batch 2 regions. `after256`=batch all 4 regions
(CUTLASS style). For GELU, the compute dominates anyway — inline stores add sync
overhead with no benefit. CUTLASS does one MEMBAR + FENCE + BAR.SYNC after the entire
tile; we do up to 4 syncwarp + fence pairs.

`EPILOGUE_LAYOUT={staged,register_strip}` — Whether to interleave TMEM loads with
compute (current "staged") or do bulk TMEM read → compute entirely in registers →
bulk store ("register_strip", CUTLASS style). The register_strip approach reads
more TMEM cols before computing (filling register file), then processes them without
interleaving.

### Problem 3: TMEM_LOAD_WIDTH=64 compile failure on FC1

Every `TMEM_LOAD_WIDTH=64` config in tier 2 was COMPILE_FAIL (64 of 128 configs — half
the tier 2 budget burned). The x64 path uses `tcgen05.ld.sync.aligned.32x32b.x64.b32`
which loads 64 FP32 values (256 bytes of register state). With FC1's GELU adding ~30
registers of intermediate state per 8-element block, the 64-wide load + GELU live
set may exceed 255 registers.

**Fix:** Either:
- Debug the compile failure (likely register overflow) and add register pressure
  relief (e.g., process first 32 of the 64 cols before loading the second 32)
- Or remove TMEM_LOAD_WIDTH=64 from FC1 sweeps and stop burning budget on it

### Problem 4: Wasted sweep dimensions for FC1

The grid search treats all 14 parameters as potentially important for all kernels.
For FC1, several are dead:

| Parameter | FC1 balanced η² | Action |
|---|---|---|
| SNAKE_ORDER | 0.000 | Remove from FC1 sweep |
| CVT_ADD_FUSED | 0.000 | Remove from FC1 sweep (CVT_ADD is for BIAS_ADD only) |
| PRELOAD_MODE | 0.909 but fake | Modes 1≈2 for GELU; collapse or remove |
| TMEM_LOAD_WIDTH | not tested | Fix compile or remove |

Removing these 4 from the cross-product saves 4-8x tier budget, which can be
reallocated to new structural knobs.

## Implementation plan

### Phase 1: C++ GELU ✓ IMPLEMENTED

Replaced `GELU_CVT_STS_V4` monolithic asm block with C++ `gelu_approx()` + `cvt_sts_v4()`.
Only `tanh.approx.f32` and `st.shared.v4.b32` remain in inline asm; all FP ALU is C++.
Combined with `PRELOAD_MODE=2` which now preloads all 32 bias before TMEM_WAIT for GELU.

Original proposal for reference — replace `GELU_CVT_STS_V4` with a C++ device function:

```cpp
__device__ __forceinline__ void gelu_cvt_sts_32(
    float acc[32], const float bias[32], uint32_t saddr, uint32_t xor_val
) {
    __nv_bfloat162 out[8];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        float x = acc[i] + bias[i];
        float inner = 0.7978845608f * x * (1.0f + 0.044715f * x * x);
        float t;
        asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(inner));
        float g = 0.5f * x * (1.0f + t);
        /* pack pairs into bf16x2 */
        if (i % 2 == 1) {
            asm("cvt.rn.bf16x2.f32 %0, %1, %2;"
                : "=r"(reinterpret_cast<uint32_t&>(out[i/2]))
                : "f"(g), "f"(prev_g));
        }
        float prev_g = g;  /* compiler will optimize */
    }
    /* STS */
    for (int i = 0; i < 8; i++) {
        uint32_t addr = saddr + ((i * 4) ^ xor_val);
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
            :: "r"(addr), ...);
    }
}
```

The key insight: only `tanh.approx.f32` and `st.shared` need inline PTX. Everything
else (add, mul, fma, cvt) can be C++ and the compiler will optimize it. This should
dramatically reduce the FP ALU instruction count because nvcc can:
- Schedule MUFU.TANH early and overlap with subsequent FMAs
- Use FFMA (fused multiply-add) instead of separate MUL + ADD
- Reuse registers across the 32-element unrolled loop

**Risk:** Low. Same math, same hardware tanh, just different codegen. Validation
unchanged.

### Phase 2: Batch TMA stores for FC1

Add `STORE_TIMING=after256` option: no inline TMA stores during the GELU loop, all
4 regions stored after the full 256-col pass. One syncwarp + fence + 4 TMA stores +
commit. This eliminates 3 of 4 sync points in the current INTERLEAVE_STRATEGY=1 path.

The ANOVA data already hints at this: INTERLEAVE_STRATEGY=0 (all-at-end) is only
134 μs worse than strategy 1 in the current code, but much of that gap may be from
PHASE1_UNROLL/K_LOOP_UNROLL confounds in the tiered search. With the C++ GELU reducing
compute time, the sync overhead becomes a larger fraction and batching may win.

### Phase 3: Wider GELU vectorization

Add `GELU_VECTOR_WIDTH` knob. Process 16 or 32 elements per C++ function call
instead of 8. With C++ codegen (Phase 1), this is just changing the loop bound.
The compiler sees more values simultaneously and can better hide MUFU latency
(MUFU throughput is 1 per 4 cycles; with 32 in-flight, later MUFUs overlap with
earlier FMAs).

Register pressure concern: 32 FP32 accumulators (from TMEM) + 32 bias floats + ~16
intermediates = ~80 registers just for the GELU hot path. FC1 winner is at 188 regs
with 67 free — tight but feasible.

### Phase 4: Wider bias preload ✓ IMPLEMENTED (with Phase 1)

`PRELOAD_MODE=2` now preloads all 32 bias floats (8x float4) before TMEM_WAIT for
BIAS_GELU, matching BIAS_ADD behavior. The GELU transform path uses preloaded values
with no interleaved LDG between GELU calls.

### Phase 5: Fix or prune FC1 grid search (zero risk, better signal)

1. Remove SNAKE_ORDER, CVT_ADD_FUSED, PRELOAD_MODE from FC1 sweeps (η²≈0)
2. Fix TMEM_LOAD_WIDTH=64 compilation or remove from sweep (50% of tier 2 wasted)
3. Add new structural knobs from Phases 1-3 as sweep parameters
4. Re-run top-10 configs with 5x replication (single-run noise is ~5 μs, top cluster
   spread is ~14 μs — not enough signal-to-noise)

## What NOT to do

| Idea | Why dead (for FC1) |
|---|---|
| More epilogue warps (6+) | FC1 at 5 warps is 188 regs. 6 warps would need fewer regs per warp, but GELU compute is the bottleneck, not warp count |
| TMEM_LOAD_WIDTH=64 without fixing compile | Already wasted 64 tier-2 configs |
| STAGGER_CYCLES tuning | η²=0.05, best=0, 37 μs spread. Noise for FC1 (matters for PE) |
| Shorter K-loop | Mainloop already matches CUTLASS (~1.61 ms). Not the bottleneck |
| Different tile shape | 256x256x128 2x1 is best for both us and CUTLASS |
| N_STAGES tuning | η²=0.31 but only 6 balanced samples. N=4 is already best |
| PHASE1_UNROLL >2 | =4 is catastrophic (+1.4 ms). =1 is 115 μs worse than =2. Locked at 2 |

## Cross-layer status

| Layer | Shape | Our ms | CUTLASS ms | Gap | Status |
|---|---|---|---|---|---|
| Patch embed | 768×768 | **0.524** | 0.535 | **-11 μs** | Won. Search exhausted |
| FC1+GELU | 768×3072 | 2.472 | **2.320** | **+151 μs** | This document |
| FC2+bias+res | 3072×768 | 1.514 | **1.224** | **+290 μs** | Needs investigation (may not be apples-to-apples — our FC2 fuses residual, CUTLASS does bias-only) |

## Files

- `FC1_LOSS.md` — original loss analysis (directionally right, some overclaims corrected above)
- `data/session_20260313_031324/` — session data (SASS, sweeps, ncu, compare)
- `data/session_20260313_031324/sweep_fc1_gelu.csv` — 615 configs, run through `tools/analyze_sweep.py`
- `data/session_20260313_031324/sass_cutlass_fc1_winner.txt` — CUTLASS winner SASS (32dp32b32x variant)
- `data/session_20260313_031324/sass_fc1-gelu.txt` — our FC1 SASS
