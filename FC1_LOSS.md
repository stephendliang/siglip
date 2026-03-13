# FC1+GELU: Why CUTLASS Wins by 150 μs

Session: `data/session_20260313_031324/` (2026-03-13, B200, 148 SMs)

## The Numbers

```
GEMM: [928256, 768] × [768, 3072]^T + bias + GELU → BF16
FP8 E4M3 inputs, FP32 accumulation, BF16 output
```

| Approach | Fused ms | GEMM-only ms | Epilogue ms | TFLOPS |
|---|---|---|---|---|
| CUTLASS (256x256x128 2x1 2sm/Sauto) | **2.320** | 1.611 | 0.709 | 1885 |
| Our kernel (256x256x128 2x1 cta_group::2) | **2.471** | ~1.61 | ~0.86 | 1772 |
| Gap | +151 μs | ~0 | **~150 μs** | -113 |

Same tile, same cluster, same GEMM shape. The entire gap is in the epilogue.

Our winning config: `INTERLEAVE_STRATEGY=1 K_LOOP_UNROLL=1 MBAR_EARLY=1
NUM_EPI_WARPS=5 PREFETCH_BEFORE_STORE=1 PRELOAD_MODE=2 STAGGER_CYCLES=0
SUB_MMA_UNROLL=1 W0_LOOP_UNROLL=1` → 188 regs, 0 spills.

## SASS Instruction Mix

| Category | Our kernel | CUTLASS | Ratio |
|---|---|---|---|
| Total instructions | 4,600 | ~2,392 | **1.9x** |
| alu_fp | 2,688 | 258 | **10.4x** |
| special (tanh.approx) | 290 | 61 | 4.8x |
| cvt (format conversion) | 192 | 18 | **10.7x** |
| tmem_load | 18 | 1 | **18x** |
| sts (shared stores) | 51 | 11 | 4.6x |
| Total stall cycles | 15,064 | 13,719 | 1.10x |

Stall cycles differ by only 10%, but we issue **1.9x more instructions** and
**10x more FP ALU** for the same mathematical result. The extra instructions
mostly fill pipeline bubbles, but the sheer volume creates execution port
pressure and a ~150 μs wall-clock gap.

## Root Causes

### 1. GELU instruction bloat (10x alu_fp)

Our GELU macro (`GELU_CVT_STS_V4` in fc1_gelu.cu) processes 8 elements at a
time with explicit inline PTX: per-element x² → mul → FMA → tanh.approx →
FMA → mul → cvt → sts. That's ~20 FP ops + 8 tanh.approx per 8 elements.
Repeated across 256 columns × 32 rows per warp = 8,192 elements per warp,
divided into uint4 batches.

CUTLASS achieves the same GELU with 258 total FP instructions. They likely:
- Use the nvcc compiler's GELU_taylor implementation which may optimize
  the polynomial differently than our hand-written PTX
- Vectorize across more elements per instruction
- Let the compiler schedule and coalesce redundant intermediate values
- May use a shorter polynomial approximation that's "close enough"

### 2. TMEM read amplification (18x tmem_load)

We issue 18 `tcgen05.ld` instructions per tile epilogue (Phase 1 loop across
4 SWIZZLE_128B regions × multiple row groups). CUTLASS issues 1 TMEM load —
they read the full tile in a single coalesced operation using
`SM100_TMEM_LOAD_32dp32b32x`, then process entirely in registers.

This is a fundamental architectural difference: we do many small TMEM reads
interleaved with computation (to fill TMEM stall windows), while CUTLASS does
one bulk read and then computes. Their approach is better for a heavy epilogue
like GELU because the compute dominates anyway — there are no stall windows
to fill.

### 3. CVT bloat (10.7x)

Our epilogue does explicit FP32→BF16 conversion per element via inline PTX
`cvt.rn.bf16x2.f32`. CUTLASS's 18 CVT instructions suggest they fuse
conversion into the store path or use the TMA hardware's implicit conversion.

## What We Got Right

- **Mainloop parity**: our K-loop is ~1.61 ms, matching CUTLASS's GEMM-only.
  The tcgen05 MMA scheduling is not the problem.
- **NUM_EPI_WARPS=5**: the tanh.approx improvement (F37, commit 10fd006) made
  5 epi warps viable for FC1, taking us from 4.220 ms → 2.471 ms (41% faster).
- **Epilogue scheduling params**: tier 4 params (PRELOAD_MODE=2,
  PREFETCH_BEFORE_STORE=1) squeezed another 14 μs.
- **Low register pressure**: 188 regs at K_LOOP_UNROLL=1, no spills.

## What We Got Wrong

Our epilogue philosophy is "fill TMEM stall windows with interleaved TMA
stores." This works when the epilogue is light (patch_embed: bias+add, where
we beat CUTLASS by 11 μs). But for GELU, the computation IS the bottleneck —
there are no stall windows to fill because the FP ALU is saturated. The
interleaved approach adds complexity (more TMEM loads, more control flow)
without benefit.

CUTLASS's approach: bulk TMEM read → compute everything in registers → bulk
TMA store. Simpler, fewer instructions, less control overhead. Better for
heavy epilogues.

## The Comparison is Fair

Both compute the same GELU: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x³)))`.
CUTLASS uses `cutlass::epilogue::thread::GELU_taylor` (software polynomial).
We use `tanh.approx.f32` (hardware SFU). Same formula, different codegen.
Same tile (256x256x128), same cluster (2x1), same precision (FP8→BF16).

## Cross-Layer Context

| Layer | Our kernel | CUTLASS | Winner | Gap |
|---|---|---|---|---|
| Patch Embed (768×768) | **0.524 ms** | 0.535 ms | **Us** | -11 μs |
| FC1+GELU (768×3072) | 2.471 ms | **2.320 ms** | **CUTLASS** | +151 μs |
| FC2+Bias (3072×768) | 1.514 ms | **1.224 ms** | **CUTLASS** | +290 μs |

FC2 gap (290 μs) may not be apples-to-apples — CUTLASS FC2 does bias-only
(no residual read), our kernel fuses bias+residual. Needs investigation.

## Files

- `data/session_20260313_031324/compare.txt` — full benchmark with ANOVA
- `data/session_20260313_031324/compare.csv` — raw timing samples
- `data/session_20260313_031324/cutlass-bench-fc1.txt` — CUTLASS per-tile results
- `data/session_20260313_031324/sass_fc1-gelu.txt` — our kernel SASS
- `data/session_20260313_031324/sass_cutlass_fc1.txt` — CUTLASS winner SASS
- `data/session_20260313_031324/sweep_fc1_gelu.csv` — grid search raw data
- `data/session_20260313_031324/grid_search_fc1_gelu.log` — grid search log
