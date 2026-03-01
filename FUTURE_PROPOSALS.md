# Future Proposals

**Kernel state (2026-03-01):** F31 best fused path = 0.530 ms / 2067 TFLOPS. 236 regs, 0 spills. Per-warp Phase 1 stagger (STAGGER_CYCLES=80) reduces TMEM scheduling contention.
**Reference:** cuBLAS pure GEMM = 0.365 ms / 3001 TFLOPS | cuBLAS + unfused pos_embed = 0.835 ms

## The problem: Phase 1 TMEM readback dominates the epilogue

Phase 1 is 70.6% of epilogue time (4,345 cycles). The epilogue is the binding constraint. The kernel is stuck.

**Phase 1 is ~90% TMEM_WAIT stall, ~10% compute.** Each of the 8 TMEM_LOAD_X32 instructions has ~200+ cycle latency. The `tcgen05.wait::ld` global fence blocks until ALL outstanding loads complete — no selective waiting, no multi-load pipelining within a warp. The compute between loads (~50 cycles of CVT + HADD2 + STS) fills only ~10% of the stall window.

**Equilibrium model (post-F31):**
```
W1:       epi_wait(1,344) + TMA0(662) + K-loop(4,059) = 6,066 cycles/tile
Epilogue: ml_wait(1,538) + Phase1(4,345) + Phase2B(273) = 6,156 cycles/tile
Deficit:  1,162 cycles (epilogue slower) → amplified to ~1,344-cycle epi_wait
```

**Ceilings:**
- Eliminate epi_wait (Phase 1 ≤ K-loop+TMA0 = 4,721): ~0.43 ms / ~2,646 TFLOPS (+28%)
- Also eliminate TMA0 (impossible — A-matrix DRAM latency): ~0.36 ms / ~3,077 TFLOPS (cuBLAS territory)

Phase 1 must drop ~376 cycles to reach equilibrium. Every 100-cycle Phase 1 reduction yields ~120-cycle wall clock improvement (due to deficit amplification).

**Per-warp asymmetry (post-F31, STAGGER=80):**
```
W2 (rg=0):  avg=4,284  p95=5,581
W3 (rg=1):  avg=4,345  p95=5,567
W4 (rg=2):  avg=4,553  p95=5,740  ← 269 cycles slower than W2 (was 330)
W5 (rg=3):  avg=4,553  p95=5,617
```
Stagger reduced spread from 330 → 269 cycles. All warps improved. Residual asymmetry likely scheduling arbitration that stagger doesn't fully eliminate.

**What is known:**
- Phase 1 is TMEM bandwidth-limited, not instruction-bound or contention-limited (F23C: <10% contention)
- The `tcgen05.wait::ld` global fence prevents multi-load pipelining within a warp
- SMEM staging is mandatory (no TMEM→GMEM direct path exists in hardware)
- Each warp independently reads its own TMEM partition (32 lanes × own columns) — no cross-warp TMEM conflicts
- Register pressure (235/255) constrains all approaches — only 20 regs headroom
- Phase 2B is solved (273 cycles via TMA tensor stores)
- K-loop improvements alone cannot improve wall clock (equilibrium absorption)

### Execution order

```
Tier 1 (attack Phase 1 directly):
  F29 (PACK mode) ──→ REJECTED — produces zeros with 32x32b layout
  F31 (dephasing) ──→ DONE — +0.4%, spread 330→269 cycles

Tier 2 (secondary targets):
  F32 (x16 TMEM granularity) ──→ re-test at new operating point
  F33 (tcgen05.cp diagnostic) ──→ speculative but potentially transformative

Recommended next: F32 → F33
```

---

## Go/no-go template

Every proposal below uses this format:

```
**Go/no-go:**
- Success: [metric] improves by ≥[threshold]
- Fail-fast: if [quick check] shows [condition], abort
- Max effort: [hours]
- Rollback: revert to [commit/state]
```

---

## F29: PACK_16b_IN_32b TMEM load mode — DIAGNOSTIC

**Effort:** Low (change TMEM_LOAD_X32 macro, adjust downstream consumer code)
**Expected impact:** Eliminates 128 CVT instructions per Phase 1, halves TMEM readback register pressure (32 → 16 regs). Could unlock unroll 4 or other register-gated optimizations.
**Risk:** The PACK mode may not perform FP32→BF16 conversion on read — it may only work with natively 16-bit TMEM data (which we don't have, since FP8 MMA always accumulates to FP32).

**Rationale:** The `tcgen05.ld` instruction supports a `PACK_16b_IN_32b` option that packs two 16-bit values into each 32-bit register during the load. If this performs FP32→BF16 conversion during the TMEM read, each `TMEM_LOAD_X32` would produce 16 BF16x2 registers instead of 32 FP32 registers. The downstream code becomes:

```
Current:  TMEM_LOAD_X32 → 32 FP32 regs → cvt.rn.bf16x2.f32 (×16) → hadd2 (×16) → STS_V4 (×4)
With PACK: TMEM_LOAD_X32.PACK → 16 BF16x2 regs → hadd2 (×16) → STS_V4 (×4)
```

This eliminates 16 CVT instructions per iteration × 8 iterations = 128 CVT instructions per Phase 1. More importantly, it halves the TMEM readback register footprint (32 → 16 registers live after each load). This could allow `#pragma unroll 4` (full unroll) at ~229 regs instead of the current 235, or enable other register-constrained optimizations.

**The uncertainty:** The PTX ISA documents `PACK_16b_IN_32b` on `tcgen05.ld` shapes, but it's unclear whether this works with FP32 accumulators. If PACK only works with natively 16-bit TMEM data (e.g., from an FP16 MMA accumulator), it's inapplicable. The diagnostic must determine:
1. Does `tcgen05.ld.sync.aligned.32x32b.x32.pack` compile for our FP32 accumulator TMEM?
2. If so, does it produce correct BF16x2 values (not garbage from raw bit reinterpretation)?
3. What is the register count impact?
4. What is the cycle impact on Phase 1?

**Implementation:**
1. Modify `TMEM_LOAD_X32` macro to add `.pack` suffix (or use the PTX `PACK_16b_IN_32b` qualifier)
2. Change output registers from `a0..a31` (32 × FP32) to `p0..p15` (16 × BF16x2)
3. Remove the 16 × `cvt_add_bf16x2` calls, replace with 16 × `__hadd2(p_i, comb_i)`
4. Adjust STS_V4 to write the BF16x2 values directly

**If PACK doesn't exist for FP32 accumulators:** Try the `tcgen05.ld.sync.aligned.32x32b.x16` variant (half the columns per load, 16 iterations instead of 8). This doesn't eliminate CVT but creates finer-grained TMEM_WAIT windows, which is the thesis of F32.

**Go/no-go:**
- Success: PACK mode compiles, produces correct BF16x2 output, register count drops ≥10
- Fail-fast: if PACK doesn't compile or produces incorrect values, abort immediately
- Max effort: 2 hours
- Rollback: revert TMEM_LOAD_X32 macro

---

## F30: Staging_b swizzle address precomputation — NO-OP (compiler already hoisted)

**Result:** Identical SASS. nvcc `-O3` already hoists all loop-invariant swizzle computations. Source change kept as cleanup (5→3 lines per iteration). See EXPERIMENTS.md for SASS diff details.

**Key insight:** The +440 cycle Phase 1 regression from F24 is NOT from redundant address computation. The cost is intrinsic to the swizzle XOR operations and/or register pressure effects.

---

## F31: Dephasing revisited (per-warp stagger) — DONE (+0.4%)

**Result: 0.530 ms / 2067 TFLOPS**, 236 regs, 0 spills. STAGGER_CYCLES=80.

**Diagnostic (rg-swap):** Swapped W2↔W4 row_group assignments. Timing followed warp ID (W2 stayed fast, W4 stayed slow regardless of which row_group they processed). **Root cause = scheduling contention** (hypothesis 3), not structural TMEM column or SMEM address effects.

**Stagger sweep:** `clock64()` spin of `ew * STAGGER_CYCLES` after mbar_wait, before Phase 1.

| STAGGER | Wall clock | TFLOPS |
|---|---|---|
| 0 | 0.532 ms | 2059 |
| 50 | 0.531 ms | 2063 |
| **80** | **0.530 ms** | **2067** |
| 100 | 0.530 ms | 2065 |
| 200 | 0.532 ms | 2060 |

Phase 1 avg: 4,569 → 4,345 (-224 cycles, -4.9%). Per-warp spread: 330 → 269 (-18%). All warps improved. ml_wait increased +169 cycles (stagger shifts epilogue start later), partially offsetting the Phase 1 gain.

---

## F32: TMEM x16 granularity re-evaluation

**Effort:** Low (change TMEM_LOAD_X32 to TMEM_LOAD_X16, double loop iterations)
**Expected impact:** Uncertain. May improve Phase 1 scheduling by creating finer-grained TMEM_WAIT windows for interleaving work.
**Risk:** Could regress from doubled instruction overhead (16 TMEM loads instead of 8).

**Rationale:** The x16-vs-x32 comparison (commit `abf04a5`) was done at 1433 TFLOPS with no double-buffered staging, no Phase 2A interleaving, no BF16 math. The kernel has changed fundamentally since then. At the current operating point:

- Phase 1B interleaves Phase 2A stores inside TMEM_WAIT stalls
- `#pragma unroll 2` controls register pressure
- The equilibrium has shifted from K-loop-bound to epilogue-bound

With x16 TMEM loads:
- 16 loads × ~100 cycle latency each (half the data per load → potentially half the latency)
- Twice as many TMEM_WAIT windows, each half as long
- Each iteration processes 16 cols instead of 32 → half the compute per iteration
- Phase 2A stores could be spread across 8 Phase 1B iterations instead of 4

**The key question:** Is TMEM load latency proportional to data volume? If `tcgen05.ld.x16` has ~100-cycle latency (vs ~200 for x32), the total TMEM stall is unchanged (16 × 100 = 8 × 200 = 1,600 cycles). But the finer granularity allows better interleaving: 16 × ~25 cycles compute per window vs 8 × ~50 cycles. If TMEM latency is fixed (not proportional to volume), x16 would be 2× worse (16 × 200 = 3,200 cycles).

**Register impact:** x16 loads use 16 FP32 registers instead of 32. This frees 16 registers, which could enable `#pragma unroll 4` or other register-intensive optimizations. Combined with F29 (PACK mode), this could reduce TMEM readback registers to 8, opening significant headroom.

**Go/no-go:**
- Success: Phase 1 drops ≥100 cycles OR wall clock improves ≥0.5%
- Fail-fast: if Phase 1 regresses ≥200 cycles, abort (TMEM latency is fixed, not proportional)
- Max effort: 2 hours
- Rollback: revert to x32 loads

---

## F33: tcgen05.cp TMEM→SMEM async copy — SPECULATIVE DIAGNOSTIC

**Effort:** Medium (unfamiliar instruction, ISA validation needed)
**Expected impact:** Potentially transformative — could bypass the register file entirely for TMEM→SMEM data movement. If it works, Phase 1 becomes: async TMEM→SMEM copy, then SMEM→register load + BF16 math + global store.
**Risk:** High. `tcgen05.cp` is primarily documented for loading *inputs* INTO TMEM for MMA consumption, not for reading accumulators out. Its applicability to epilogue readback is undocumented. No public kernel uses it this way.

**Rationale:** The PTX ISA includes `tcgen05.cp.cta_group::N.SZ` instructions that copy data between TMEM and shared memory. If this instruction can copy FROM TMEM TO SMEM (reading accumulators out), it would use a DMA path that bypasses the register file entirely. The epilogue flow would become:

```
Current:     TMEM → registers (tcgen05.ld) → BF16 math → SMEM (st.shared) → global
With cp:     TMEM → SMEM (tcgen05.cp, async) → registers (ld.shared) → BF16 math → global
```

The async copy would run concurrently with other work. The register pressure for TMEM readback drops to zero (data goes directly to SMEM). The subsequent `ld.shared` + BF16 math can be software-pipelined more easily than `tcgen05.ld` + wait (SMEM loads are fast, ~20 cycles, vs TMEM loads at ~200 cycles).

**Key unknowns:**
1. Does `tcgen05.cp` support reading FROM TMEM? (direction may be SMEM→TMEM only)
2. What is the SMEM layout of the copied data? (must match our staging buffer organization)
3. What is the throughput? (could be slower than `tcgen05.ld` if the DMA path is optimized for input loading)
4. Is the FP32→BF16 conversion included, or must it happen after the copy?

**Implementation plan:**
1. Write a minimal test: `tcgen05.cp` from the TMEM accumulator region to a scratch SMEM buffer
2. Verify the copy produces correct FP32 data in SMEM
3. If correct: measure latency, compare with `tcgen05.ld` path
4. If faster: redesign Phase 1 around the async copy path

**Go/no-go:**
- Success: `tcgen05.cp` produces correct data from TMEM to SMEM AND total Phase 1 improves ≥10%
- Fail-fast: if `tcgen05.cp` doesn't support TMEM→SMEM direction, or produces incorrect data, abort immediately
- Max effort: 4 hours (mostly ISA validation)
- Rollback: no production code change (diagnostic only)

---

## Completed experiments (F21-F28, see EXPERIMENTS.md for full details)

| Experiment | Result | Key number |
|-----------|--------|-----------|
| F25 ✅ | Diagnostic: symmetric + mild fat tail | 172-cycle spread (pre-F24) |
| F21 ✗ | B already L2-resident | 7-cycle difference (noise) |
| F22 ✅ | BF16 epilogue arithmetic | +1.3%, 229 regs, -1480 SASS instructions |
| F23C ✗ | 2-warp epilogue contention test | TMEM contention <10%, 42% regression |
| F28 ✅ | K-loop restructuring | perf-neutral, -76 cyc K-loop |
| F24 ✅ | Swizzled staging + TMA tensor stores | +0.7%, Phase 2B -626 cycles, Phase 1 +440 cycles |
| F30 — | Swizzle address precomputation | no-op — compiler already hoisted; source cleanup only |
| F31 ✅ | Per-warp Phase 1 stagger | +0.4%, STAGGER=80, contention confirmed via rg-swap diagnostic |

---

## Ruled out (see EXPERIMENTS.md for full details)

| Proposal | Why dead |
|----------|----------|
| Next-tile TMA prefetch (F20) | ~0% — DRAM bandwidth, not scheduling. Prefetch shifts TMA issue by ~20 cycles vs 818-cycle shortfall. |
| cp.async.bulk Phase 2B stores (F19b) | 3× slower than manual stores. **Superseded by F24** (TMA tensor stores). |
| 5-stage pipeline | SMEM-limited: 5-stage = 163 KB pipeline → only 65 KB for staging (needs 68 KB). |
| TMA multicast for B | B is N-split across CTAs (each loads different half). Zero redundancy. |
| L2 promotion/persistence for B (F21, F26) | B already fully L2-resident. TMA0_wait is pure A-matrix DRAM latency. |
| Epilogue warp count (F23) | TMEM contention <10% of Phase 1. 2 warps: 1.85x per-warp Phase 1. All variants ruled out. |
| TN=128 | 2× tiles, same per-tile overhead, 11.4% regression. Definitively ruled out. |
| Combined load L1 bypass | Blocked layout already gives near-perfect L1 locality. All bypass variants regressed. |
| Direct CVT_STG stores (F17) | Uncoalesced st.global.v8 contends with TMEM reads on L1. Phase 1 +19%. |
| 6 epilogue warps (F3) | TMEM bandwidth saturation. |
| Split TMEM loads x32→2×x16 (F1) | tcgen05.wait::ld is a global fence — waits for ALL loads. Split adds overhead without reducing wait. |
| Register-staged transpose (warp shuffle) | Would need 32+ extra registers (exceeds 255 limit). 160 shuffles per 32-col chunk ≈ SMEM staging cost. |
| SMEM staging elimination | F17 proved L1 contention. SMEM transpose is mandatory for coalesced global stores. |
| Tile shape changes (TM, TN, TK) | All directions hardware-limited: TMEM (512 col) caps TN, SMEM (228 KB) caps TM and TK. |
| Split-K | TMEM double-buffering precludes second accumulator. GPU already fully occupied (10,878 tiles on 74 clusters). |
| Warp role restructuring (W1 helps epilogue) | TMEM is per-SM (CTA0 can't read CTA1's TMEM). Register pressure. Late arrival timing. |
| BF16 accumulation in TMEM | Hardware mandates FP32 for FP8 MMA inputs. Even if possible, precision loss unacceptable for K=768. |
| Triple-buffered TMEM | Only 512 TMEM columns available. TN=256 × 2 buffers = 512 (maxed). |
