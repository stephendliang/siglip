# stmatrix migration plan

## Goal

Replace STS.128 (st.shared.v4.b32) with STSM (stmatrix.sync.aligned.x4.m8n8.shared.b16)
in fc2_w3's epilogue. This is the last known source-level approach to fix STS clustering.

## Why stmatrix might help

1. Warp-cooperative instruction — all 32 threads participate atomically, hardware handles distribution
2. Different instruction class (STSM vs STS) — may have different ptxas scheduling behavior
3. Internal bank-conflict-free addressing — our manual XOR swizzle scheme becomes unnecessary
4. CUTLASS uses it — it's one of 3 SASS differences vs our epilogue (STSM.16.M88.4, @!PT LDS drains, STS interleaving)

## Critical constraint discovered

**stmatrix requires all 8 threads in each m8n8 tile to address within ONE 128-byte aligned segment.**

- Our SMEM: 128 bytes per row (one per lane). 8 consecutive lanes span 8 × 128 = 1024 bytes.
- stmatrix m8n8.b16: 8 rows × 16B = 128B per tile. Must fit in one 128B block.
- **These are incompatible.** stmatrix forces 16B row stride. Our SMEM has 128B row stride.
- stmatrix CANNOT produce our SWIZZLE_128B layout.

**Consequence**: stmatrix writes to a DIFFERENT SMEM layout (compact 16B-stride tiles).
TMA store descriptors must change. Multiple smaller TMA stores may be needed.

## CUTLASS's actual instruction choice (corrected)

For ACC=FP32, D=BF16, row-major output:
- **Non-transposed**: SM90_U32x4_STSM_N → `stmatrix.sync.aligned.x4.m8n8.shared.b16` (4 regs)
- **Transposed**: SM90_U16x8_STSM_T → `stmatrix.sync.aligned.x4.trans.m8n8.shared.b16` (4 regs)

CUTLASS selects STSM_N (non-transposed) for N-major (row-major) output.
Non-transposed: thread's register data stored directly as its row. No cross-thread shuffle.

CUTLASS's epilogue SMEM uses Swizzle<3,4,3> (= SW128 = SWIZZLE_128B equivalent).
The swizzle is applied to per-thread addresses BEFORE passing to stmatrix.
But per-tile 128B constraint means the "rows" must be 16B-stride, not 128B-stride.
CUTLASS solves this with cute layout algebra — compact tiles with swizzle, TMA descriptors match.

Copy_Traits for STSM_N (from copy_traits_sm75.hpp):
- DstLayout (SMEM): Shape<_32, _128>, Stride<_128, _1> in bits
  → thread t → bytes [t*16 .. t*16+15] = contiguous 512B block
- SrcLayout (Registers): Shape<_32, Shape<_32, _4>>, Stride<_32, Stride<_1, _1024>>

## Step 1: Microbenchmark (GO/NO-GO gate) — DONE, NO-GO

**File**: bench/stmatrix_bench.cu, target: `make stmatrix-bench`
Compiles. 6 kernels (12 regs for bench, 32 regs for layout).

Tests:
- A: stmatrix x4 NON-transposed layout characterization (contiguous 128B-aligned tiles)
- B: stmatrix x4 TRANSPOSED layout characterization
- C: STS.128 with SWIZZLE_128B reference
- D: Throughput at 1/2/4/8 warps: STS.128 vs STSM (non-trans) vs STSM (trans)

### B200 results (2026-04-05)

**Raw throughput: IDENTICAL at every warp count.**

| Warps | STS.128 | STSM (norm) | STSM (trans) |
|-------|---------|-------------|--------------|
| 1 | 10.0 cyc/store | 10.0 | 10.0 |
| 2 | 10.4 | 10.4 | 10.4 |
| 4 | 18.4 | 18.4 | 18.4 |
| 8 | 36.6 | 36.6 | 36.6 |

**Epilogue-realistic at 4 warps (the decision row):**

| Chain | STS.128 | STSM | Δ |
|-------|---------|------|---|
| BF16 (our kernel) | 73.3 cyc/store | 71.3 | -2.7% (noise) |
| FP32 (CUTLASS-like) | 72.8 cyc/store | 73.6 | +1.1% (noise) |

### Verdict: NO-GO

stmatrix hits the exact same SMEM ports as STS.128. The contention is architectural —
the shared memory subsystem, not the instruction encoding. Swapping instruction class
while keeping the same store volume produces the same bottleneck. The SMEM layout change,
TMA descriptor rework, and potential 8× TMA store cost would be pain for zero gain.

Also: FP32 vs BF16 compute chain makes no throughput difference in this microbench
(291 vs 291 cyc/iter at 4W). The real kernel's STS clustering problem comes from ptxas
scheduling, not instruction choice — and stmatrix doesn't change how ptxas schedules.

**stmatrix migration is dead. This was the last source-level approach.**

## Step 2: SMEM layout redesign — CANCELLED (step 1 NO-GO)

Current: 32 rows × 128B/row = 4096B per row_group, SWIZZLE_128B, one TMA store per sub-iter
New: 32 rows × compact tiles (16B/row per m8n8 tile), multiple tiles per sub-iter

Options for TMA store:
- **A**: 4 TMA stores per chunk (one per 8-col strip): box=[8,32], SWIZZLE_NONE, 16B stride
  → 8× more TMA per sub-iter (197 cyc each, +~1400 cyc per sub-iter)
- **B**: Rearrange tiles to form larger TMA-compatible blocks
- **C**: Use TMA with matching swizzle mode for the compact tile layout

## Step 3: Integration in fc2_w3.cu

- #ifdef STMATRIX_EPILOGUE
- Replace STS.128 with stmatrix.x4.m8n8.b16 (4 calls per chunk, same as STS count)
- Change SMEM layout: OFF_STAGING → compact tile format
- Change TMA store descriptor: different box, stride, swizzle mode
- Makefile: fc2-w3-stm

## Step 4: SASS verification + B200 benchmark

## Risks (updated)

- **HIGH**: ptxas may cluster STSM identically to STS (BF16 compute still too short)
- **HIGH**: SMEM layout change + TMA descriptor changes = significant refactor
- **MEDIUM**: 8× more TMA stores could negate any STSM throughput benefit
- **LOW**: Register rearrangement (non-transposed matches our register layout)
- **LOW**: stmatrix .sync semantics overhead

## Key CUTLASS references

- Store op selection: sm100_builder.inl:618-798
- PTX wrappers: copy_sm90.hpp:40-155 (STSM_N and STSM_T variants)
- Copy_Traits: copy_traits_sm90.hpp:43-85 (STSM_N), copy_traits_sm75.hpp:76-90 (LDSM_N base)
- SMEM swizzle: sm100_common.inl:82-97 (sm100_smem_selector → SW128 for BF16)
- SMEM layout atom: mma_traits_sm90_gmma.hpp:84 (Layout_K_SW128_Atom_Bits = Swizzle<3,4,3>)
