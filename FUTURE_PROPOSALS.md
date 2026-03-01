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
- The `tcgen05.wait::ld` global fence prevents per-load latency hiding — WAIT drains the entire TMEM read pipeline as a batch (F35 confirmed)
- SMEM staging is mandatory (no TMEM→GMEM direct path exists in hardware)
- Each warp independently reads its own TMEM partition (32 lanes × own columns) — no cross-warp TMEM conflicts
- Register pressure (236/255) constrains all approaches — only 19 regs headroom (F32 showed x16 drops to 174 but the latency penalty outweighs register savings)
- Phase 2B is solved (273 cycles via TMA tensor stores)
- K-loop improvements alone cannot improve wall clock (equilibrium absorption)

### Execution order

```
Tier 1 (attack Phase 1 directly):
  F29 (PACK mode) ──→ REJECTED — produces zeros with 32x32b layout
  F31 (dephasing) ──→ DONE — +0.4%, spread 330→269 cycles

Tier 2 (secondary targets):
  F32 (x16 TMEM granularity) ──→ REJECTED — 10.6% regression, fixed latency per load
  F33 (tcgen05.cp diagnostic) ──→ RULED OUT (SMEM→TMEM only)
  F34 (parallel TMEM load)   ──→ DIAGNOSTIC — loads pipeline, but...
  F35 (software-pipelined readback) ──→ REJECTED — WAIT is global fence, batch-drain port
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

## F32: TMEM x16 granularity re-evaluation — REJECTED (10.6% regression)

**Result: 0.586 ms / 1868 TFLOPS** — 10.6% regression vs 0.530 ms baseline. 174 regs (-62), 0 spills. Checksum correct.

**Answer to the key question:** TMEM load latency is **fixed per instruction** (~200 cycles), not proportional to data volume. `tcgen05.ld.x16` takes roughly the same time per load as x32 but moves half the data. Doubling the load count (8→16) adds ~56 µs wall clock.

**Confirmed at two operating points:** F8 (1433 TFLOPS, no staging) and F32 (2067 TFLOPS, full staging + interleaving). x32 is definitively the optimal TMEM load granularity.

**Key insight:** The 62-register savings (236→174) are real but irrelevant — no register-gated optimization can overcome the 2× fixed-latency penalty. Phase 1 optimization must reduce the number of `tcgen05.ld` instructions, not their granularity.

---

## F33: tcgen05.cp TMEM→SMEM async copy — RULED OUT (SMEM→TMEM only)

**Result:** Killed by ISA research. `tcgen05.cp` is architecturally SMEM→TMEM only — no hardware path exists for TMEM→SMEM. No code changes.

**Evidence (3 independent sources):**
1. **CUTLASS**: Has `make_s2t_copy()` (SMEM-to-TMEM) but NO `make_t2s_copy()`. No `SM100_TMEM_LOAD` copy traits use `tcgen05.cp`. All epilogue code uses `tcgen05.ld`.
2. **Colfax tutorial**: "data gets _into_ TMEM via UMMA operations, and is explicitly moved _out_ to registers using `tcgen05.ld`." Explicitly skips `tcgen05.cp` as irrelevant to epilogue.
3. **JAX/Pallas docs**: "only way to move data out from tensor memory is through `tcgen05.ld`"

**Conclusion:** The only way to read accumulators out of TMEM is `tcgen05.ld` (TMEM→registers). Phase 1 optimization must work within this constraint.

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
| F32 ✗ | x16 TMEM loads — fixed latency per load | 0.586 ms, 10.6% regression, 174 regs |
| F33 ✗ | tcgen05.cp is SMEM→TMEM only | No code — ruled out by ISA research |
| F34 — | Parallel TMEM load diagnostic | 0.531 ms, loads pipeline (2×x16 ≈ 1×x32) |
| F35 ✗ | Software-pipelined TMEM readback | 0.540–0.542 ms, WAIT is batch-drain, not per-load |

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
| x16 TMEM load granularity (F32) | Fixed ~200-cycle latency per `tcgen05.ld` regardless of data volume. x16 = 2× loads = 10.6% regression. Confirmed at two operating points (F8 + F32). |
| Software-pipelined TMEM readback (F35) | `tcgen05.wait::ld` is a global fence with batch-drain behavior. Separating loads in time does not create independent completion. Two variants tested (doubled WAITs: 220 regs, 0.540 ms; same WAITs split LOADs: 244 regs, 0.542 ms). Both regressed vs 0.536 ms baseline. TMEM port processes outstanding loads as a batch — no per-load latency hiding possible. |
| Register-staged transpose (warp shuffle) | Would need 32+ extra registers (exceeds 255 limit). 160 shuffles per 32-col chunk ≈ SMEM staging cost. |
| SMEM staging elimination | F17 proved L1 contention. SMEM transpose is mandatory for coalesced global stores. |
| tcgen05.cp epilogue readback (F33) | tcgen05.cp is SMEM→TMEM only. No hardware path for TMEM→SMEM. Only tcgen05.ld (TMEM→registers) can read accumulators out. |
| Tile shape changes (TM, TN, TK) | All directions hardware-limited: TMEM (512 col) caps TN, SMEM (228 KB) caps TM and TK. |
| Split-K | TMEM double-buffering precludes second accumulator. GPU already fully occupied (10,878 tiles on 74 clusters). |
| Warp role restructuring (W1 helps epilogue) | TMEM is per-SM (CTA0 can't read CTA1's TMEM). Register pressure. Late arrival timing. |
| BF16 accumulation in TMEM | Hardware mandates FP32 for FP8 MMA inputs. Even if possible, precision loss unacceptable for K=768. |
| Triple-buffered TMEM | Only 512 TMEM columns available. TN=256 × 2 buffers = 512 (maxed). |
