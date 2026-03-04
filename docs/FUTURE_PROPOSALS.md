# Future Proposals

**Kernel state (2026-03-04):** 0.519 ms / 2108 TFLOPS (MBAR_EARLY=1 STAGGER_CYCLES=160). Defaults: 0.524 ms / 2090 TFLOPS. 209 regs, 0 spills.
**Reference:** cuBLAS pure GEMM = 0.365 ms / 3001 TFLOPS | cuBLAS + unfused pos_embed = 0.835 ms
**CUTLASS best fused BF16 (256x256x128 2x1):** 0.536 ms / 2044 TFLOPS

## Current architecture (post-F38/F40)

F38 unified the epilogue: all 256 cols through 4 SWIZZLE_128B regions, all stores via TMA. No more dual-staging, no Phase 2A manual LDS+STG. F40 recovered stall-window utilization by interleaving TMA stores during TMEM readback (strategy 2, half-batch: 2 TMA stores after every 2nd region).

```
Phase 1:  TMEM → CVT+epilogue_op → STS staging (swizzle, 4 regions of 32x64 BF16)  [256 cols]
          ↕ interleaved TMA stores every 2 regions (hide in TMEM stall windows)
Phase 2:  cp.async.bulk.commit_group only — all TMA stores already issued inline
```

The epilogue op is kernel-specific: bias+pos_embed add (megakernel.cu) or bias+GELU (fc1_gelu.cu).

Grid search (4 runs x 145 configs) confirms parameter space is exhausted — all top configs within 0.001 ms.

## Next frontier: multi-kernel architecture

The codebase now has `kernel_common.cuh` (shared pipeline/TMEM/TMA infrastructure) with two kernel files:
- `megakernel.cu` — patch embed: [928256,768]x[768,768]^T + bias + pos_embed (N_DIM=768)
- `fc1_gelu.cu` — FC1+GELU: [928256,768]x[768,3072]^T + bias + GELU (N_DIM=3072)

### FC1+GELU tuning (next)

fc1_gelu.cu compiles but has not been run on B200 yet. Key differences from patch embed:
- N_DIM=3072 → TILES_N=12 (vs 3), TOTAL_TILES=43,512 (vs 10,878). 4x more tiles.
- Same K_DIM=768, same tile shape 256x256x128.
- GELU activation in epilogue instead of pos_embed table add. GELU is compute-heavier (tanhf) — may shift the epilogue/K-loop balance.
- No pos_embed table — no `combined` precomputation, simpler memory footprint.
- Needs its own grid search (STAGGER_CYCLES, MBAR_EARLY, etc. may have different optima for 4x more tiles).

### FC2 (after FC1)

FC2: [928256,3072]x[3072,768]^T + bias + residual add. Tall-K (K_DIM=3072 → K_ITERS=24 vs 6). Residual add requires loading pre-MLP activation from global — similar to combined load pattern. Will need a third kernel file.

### Attention (after MLP)

Q/K/V/Out projections are same shape as patch embed (768x768). Can reuse megakernel.cu's epilogue structure with different fusion (no pos_embed). QK^T and softmax are entirely different — small per-head matmuls, not persistent megakernel territory.

### Long-term: single persistent kernel

See `docs/architecture.md` for the full vision: single kernel launch for the entire SigLIP2 encoder. Requires:
- Weight streaming across layers (TMA double-buffered)
- Warp parking between layers (different GEMM shapes need different tile counts)
- Cluster barriers for intra-block fusion (LN → GEMM → activation without global roundtrip)
- Grid sync between transformer blocks

The current multi-kernel approach builds the primitives needed for fusion.

---

## Patch embed GEMM — remaining optimizations

The patch embed GEMM parameter search is exhausted. Remaining ideas are architectural changes with high risk:

### Shorter K-loop

K-loop = 4,059 cycles is the floor for W1 productive time. Any reduction here directly speeds every tile. Options:
- SASS post-compile scheduling optimizer (see `docs/sass_optimizer.md`)
- Reduced MMA latency through better descriptor caching
- Both are speculative with uncertain payoff.

### Phase transition (equilibrium flip)

If epilogue reliably outruns W1, epi_wait drops to ~0 and per-tile time drops from ~6,100 to ~4,721 cycles (22.6% reduction → ~0.43 ms / ~2,646 TFLOPS). The post-F40 architecture is close to this boundary but hasn't crossed it. Further Phase 1 reduction would require fundamentally faster TMEM readback, which appears hardware-limited.

---

## Completed experiments (F38-F40)

| Exp | Result | Key number |
|-----|--------|-----------|
| F38 ✅ | Unified epilogue (all swizzle+TMA) | Clean architecture, 0.554 ms (4.7% regression — scheduling) |
| F39 ✅ | x32 TMEM default | Code cleanup, perf-neutral |
| F40 ✅ | Interleaved TMA stores | **0.524 ms / 2090 TFLOPS**, recovered F38 regression + 1.1% faster |
| Grid search ✅ | 4 runs x 145 configs | 145/145 valid, 0.519 ms best, defaults 0.524 ms |

### F38 outcome

F38 unified the epilogue by deleting staging_a (linear layout) and processing all 256 cols through 4 SWIZZLE_128B regions with TMA stores. Eliminated 64 STG + 32 LDS + L1 contention. But batching all 4 TMA stores at the end left TMEM stall windows empty, causing a 4.7% regression (0.554 ms vs 0.530 ms pre-F38). F40 fixed this.

### F40 outcome

Interleaved TMA stores during Phase 1 (strategy 2: 2 stores after every 2nd region). Fills TMEM stall windows with useful TMA work instead of Phase 2A's old LDS+STG. Result: 0.524 ms — 1.1% faster than pre-F38, with cleaner architecture. Half-batch strategy is optimal; per-region too aggressive (extra syncwarp overhead), all-at-end too conservative.

---

## Earlier completed experiments

| Exp | Result | Key number |
|-----|--------|-----------|
| F25 ✅ | Per-warp timing diagnostic | 172-cycle spread |
| F21 ✗ | B already L2-resident | 7-cycle difference (noise) |
| F22 ✅ | BF16 epilogue arithmetic | +1.3%, 229 regs |
| F23C ✗ | 2-warp epilogue contention | TMEM contention <10%, 42% regression |
| F28 ✅ | K-loop restructuring | perf-neutral, -76 cyc K-loop |
| F24 ✅ | Swizzled staging_b + TMA stores | +0.7%, Phase 2B 899→273 cycles, Phase 1 +440 |
| F29 ✗ | PACK_16b_IN_32b TMEM load | Produces zeros with 32x32b layout |
| F30 — | Swizzle address precompute | No-op, compiler hoists already |
| F31 ✅ | Per-warp stagger | +0.4%, STAGGER=80, spread 330→269 |
| F32 ✗ | x16 TMEM granularity | 10.6% regression, 174 regs |
| F33 ✗ | tcgen05.cp readback | SMEM→TMEM only, ruled out |
| F34 — | Parallel TMEM load diagnostic | Loads pipeline (2×x16 ≈ 1×x32) |
| F35 ✗ | Software-pipelined readback | WAIT is global fence, batch-drain |
| F36 — | SASS static analysis (CUTLASS vs custom) | See above |
| F37 — | TMEM load width x16/x32/x64 | Identical wall clock (p=0.617) |

---

## Ruled out

| Proposal | Why dead |
|----------|----------|
| Next-tile TMA prefetch (F20) | DRAM bandwidth, not scheduling |
| cp.async.bulk Phase 2B stores (F19b) | 3x slower. Superseded by F24 TMA tensor stores |
| 5-stage pipeline | SMEM-limited (163 KB pipeline → only 65 KB for staging) |
| TMA multicast for B | B is N-split across CTAs. Zero redundancy |
| L2 promotion for B (F21, F26) | B already L2-resident. TMA0_wait is A-matrix DRAM |
| Epilogue warp count (F23) | TMEM contention <10%. 2 warps: 42% regression |
| TN=128 | 2x tiles, same overhead, 11.4% regression |
| Combined load L1 bypass | Blocked layout gives near-perfect L1 locality |
| Direct CVT_STG (F17) | st.global.v8 contends with TMEM on L1. +19% Phase 1 |
| 6 epilogue warps (F3) | TMEM bandwidth saturation |
| x16 TMEM loads (F32) | Fixed ~200-cycle latency per load. 2x loads = 10.6% regression |
| Software-pipelined readback (F35) | WAIT is global fence, batch-drain. No per-load hiding |
| Register-staged transpose | 32+ extra regs (exceeds 255). 160 shuffles ≈ SMEM cost |
| SMEM staging elimination (F17) | L1 contention. SMEM staging mandatory |
| tcgen05.cp readback (F33) | SMEM→TMEM only. No TMEM→SMEM path in hardware |
| Tile shape changes | TMEM caps TN, SMEM caps TM and TK |
| Split-K | TMEM double-buffering precludes second accumulator |
| W1 helps epilogue | TMEM per-SM, register pressure, late arrival |
| BF16 accumulation | Hardware mandates FP32 for FP8 inputs |
| Triple-buffered TMEM | 512 cols = TN*2 buffers (maxed) |
| Dual-staging epilogue | Replaced by F38 unified. Historical debt, L1 contention |
| Parameter search | Exhausted. 4 runs x 145 configs, all within 0.001 ms |
