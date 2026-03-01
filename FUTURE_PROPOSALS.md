# Future Proposals

**Kernel state (2026-02-28):** F18 best fused path = 0.543 ms / 2018 TFLOPS. F19a early-mbar variant = 0.542 ms / 2020 TFLOPS (performance-neutral, structurally kept). 223 regs, 0 spills.
**Reference:** cuBLAS pure GEMM = 0.365 ms / 3001 TFLOPS | cuBLAS + unfused pos_embed = 0.835 ms
**Bottleneck:** Epilogue-bound producer-consumer equilibrium. 403-cycle deficit (epilogue slower), amplified to ~1,162 epi_wait. Phase 1 TMEM readback (4,288 cycles, 68% of epilogue) is the binding constraint. F25 confirmed all 4 epilogue warps are equally bound (symmetric, 172-cycle spread).

### Execution order

```
Dependencies:
  F25 ✅ (diagnostic) ──→ F23 (warp count sweep, low priority)
                      └──→ F27 ✗ (skipped: symmetric)
  F21 (L2 promotion) ─→ F26 (L2 persistence)
  F22 (BF16 math) ────→ F24 (swizzled staging)
  F28 (K-loop) ────────→ best combined with F22 (attacks both sides of equilibrium)

Recommended serial order:
  F25 ✅ → F21 → F22 → F28 → F23C → (conditional: F26, F24)
  │         │      │      │      │
  │         │      │      │      └─ low priority (F25: symmetric, bandwidth-limited)
  │         │      │      └──────── pair with F22: K-loop + epilogue = both sides
  │         │      └──────────────── highest expected value, attacks 403-cycle deficit
  │         └─────────────────────── 30 min, confirms/eliminates L2 hypothesis
  └────────────────────────────────── DONE: symmetric + mild fat tail → F27 skipped
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

## F25: Full epilogue-warp timing (W2–W5) — DIAGNOSTIC ✅ DONE

**Effort:** Low (extend existing clock64 infrastructure)
**Expected impact:** No performance change. Informs F23 and F27.
**Risk:** Register pressure in timing build (currently 254 regs).

**Result: SYMMETRIC (Result A) + mild fat tail (Result C)**

```
W2 (ew=0, rg=0):  avg=4,252  p95=5,589
W3 (ew=1, rg=1):  avg=4,288  p95=5,599
W4 (ew=2, rg=2):  avg=4,424  p95=5,618
W5 (ew=3, rg=3):  avg=4,392  p95=5,673
Spread of averages: 172 cycles (<200 threshold)
Per-tile spread:    avg=615  p95=1,606 (2.6x fat tail)
```

All 4 warps are essentially equal (172-cycle spread < 200-cycle threshold). No systematic TMEM port bias or bank conflict asymmetry. rg=2/3 are ~140-170 cycles slower on average but p95 values are nearly identical (84-cycle spread), confirming this is noise not hardware bias. Per-tile spread shows mild fat tail (p95/avg = 2.6x) — burst contention creates outlier tiles but no consistent victim.

**Implications:**
- **F27 (dephasing): Skip** — no queueing to exploit; stagger adds idle time without benefit
- **F23C (2 warps): Low priority** — contention is bandwidth-limited; reduced warps may not compensate for doubled per-warp work
- **F22 (BF16 arithmetic): Confirmed highest EV** — Phase 1 is the binding constraint across all warps equally; reducing it attacks the deficit directly
- Timing build: 252 regs, 0 spills (3 regs headroom vs 255 ceiling)

**Rationale:** Current clock64 timing measures W3 only (ew=1). But `epilogue_mbar` fires when the **last** warp arrives — W3 could be average while W2 or W5 is consistently 200–300 cycles slower due to TMEM port ordering or SMEM bank conflict asymmetry. Without per-warp data, F23's contention hypothesis and F27's dephasing idea are flying blind.

**What to instrument:**
1. Extend W3's timestamp pattern to W2, W4, W5. To control register pressure, instrument only Phase 1 start/end per warp (2 timestamps instead of 4) — Phase 2B is already known to be small (899 cycles) and overlapped. This costs 4 warps × 2 timestamps × 8 bytes = 64 bytes. If 64 bytes still causes spills, fall back to instrumenting only W2 and W5 (first and last row_groups) alongside existing W3.
2. Report per-warp: avg / min / max / p95 for Phase 1.
3. Report inter-warp spread per tile: `max(Phase1) - min(Phase1)` across instrumented warps, averaged over all tiles.

**What to look for:**
- **Symmetric Phase 1 times across warps:** TMEM contention is bandwidth-limited, not port-queued. Dephasing (F27) won't help. F23C's 2-warp test is sufficient.
- **One warp consistently slower:** TMEM port queueing or bank conflict asymmetry. Identifies which row_group is disadvantaged. Dephasing (F27) has a target.
- **Whether row_group 0 or 3 is systematically slower:** Row group 0 hits TMEM cols 0–31, row group 3 hits 96–127. Hardware port bias would show here.
- **Tail distribution (p95/max):** If the max-warp Phase 1 time has a fat tail, the mbar is gated by outliers, not averages.

**Outcomes (diagnostic — all results are informative):**
- **Result A — symmetric Phase 1 across warps (spread <200 cycles):** TMEM contention is bandwidth-limited, not port-queued. Rules out F27 (dephasing). F23C is still worth testing as a pure bandwidth reduction experiment, but expectations should be low.
- **Result B — one warp consistently slower (spread ≥200 cycles):** TMEM port queueing or bank conflict asymmetry exists. Identifies which row_group is disadvantaged. F27 has a target. F23C has higher expected value.
- **Result C — fat tail on max-warp Phase 1 (p95/max spread ≫ avg spread):** Burst contention causes outlier tiles that gate the mbar. Dephasing (F27) would shave the tail even if averages look symmetric.

**Go/no-go:**
- Success: data collected (always — diagnostic can't fail)
- Fail-fast: n/a
- Max effort: 1–2 hours
- Rollback: `#ifdef TIMING` — no production code change

---

## F21: L2 promotion for B matrix

**Effort:** Low (single line change)
**Expected impact:** Small-to-medium reduction in TMA0_wait (currently 856 cycles, 14% of tile)
**Risk:** Near zero

**Rationale:** B = 576 KB (768×768 FP8). Only 3 N-tiles, so all 74 clusters reuse the same B data. B should already be mostly L2-hot, but A streaming (680 MB, zero reuse) may evict B entries. TMA0_wait = 856 cycles reflects ~1,940-cycle DRAM latency minus the ~1,100-cycle epilogue overlap window. If B consistently hits L2, DRAM bandwidth is freed for A loads.

**Change:** In the B TMA descriptor:
```c
CU_TENSOR_MAP_L2_PROMOTION_NONE  →  CU_TENSOR_MAP_L2_PROMOTION_L2_128B
```

Also try `_L2_256B` on B. Do **not** try L2 promotion on A — A is 680 MB streaming with zero reuse; promoting A cache lines actively evicts B and other L2-resident data.

**Why it might not help:** B may already be fully L2-resident given its small size and high reuse. The 856-cycle TMA0_wait could be entirely from A loads (16 KB per tile from 680 MB, pure streaming). In that case, L2 promotion changes nothing.

**Measurement:** Compare TMA stage-0 wait in clock64 timing. Wall clock change may be small since TMA0_wait is only 14% of tile.

**Go/no-go:**
- Success: TMA0_wait drops ≥50 cycles (currently 856)
- Fail-fast: if TMA0_wait is unchanged with `_L2_128B` on B, B is already L2-resident — skip `_L2_256B` and skip F26
- Max effort: 30 minutes
- Rollback: revert descriptor constant

---

## F22: BF16 Epilogue Arithmetic

**Effort:** Medium (rewrite Phase 1 math path in `epilogue_store`)
**Expected impact:** Uncertain but potentially meaningful. If Phase 1 drops by ~100-300 cycles, the current ~272-cycle producer/consumer imbalance could shrink materially.
**Risk:** Numeric behavior changes (different rounding point), possible register/scheduling side effects.

**Rationale:** Current Phase 1 math converts packed BF16 combined values to FP32, does FP32 adds, then converts back to BF16 for SMEM staging. A BF16-native variant may reduce instruction pressure in the post-`TMEM_WAIT` region.

**Proposed direction:**
1. Keep combined values in packed BF16x2 form instead of converting to FP32.
2. Convert FP32 accumulator pairs to packed BF16x2 via `cvt.rn.bf16x2.f32` (TMEM gives FP32 accumulators — this conversion cost doesn't disappear, it shifts from per-element scalar `cvt.rn.bf16.f32` to packed pair conversion).
3. Perform packed BF16 adds via `hadd2` (or equivalent `add.bf16x2`).
4. Store packed BF16x2 directly to SMEM via `st.shared.b32`.
5. Verify exact PTX op names/availability with `ptxas` (mnemonics can differ by ISA version).

The savings come from eliminating the separate per-element `BF16X2_TO_F32` unpack of combined data + per-element `cvt.rn.bf16.f32` repack, replacing scalar add chains with packed ops. The accumulator conversion cost shifts from many scalar CVTs to fewer packed CVTs — the question is whether packed ops are faster than the scalar equivalent.

**Precision note:** This changes arithmetic order from `round(acc + combined)` to approximately `round(round(acc) + combined)`. Error can increase versus the current path, especially at larger magnitudes.

**Verification plan:**
1. Compare against the current FP32-add epilogue path on representative inputs.
2. Report max absolute error, max relative error, and BF16 ULP-distance stats.
3. Use magnitude-aware ULP checks (BF16 ULP is value-dependent; around 1728, 1 ULP is about 8.0).
4. Accept only if quality is acceptable for downstream model metrics, not checksum equality.
5. **Critical-path diagnostic (from F25):** Before implementing F22, F25 should instrument the gap between the last `tcgen05.ld` and the `__syncwarp()` before early mbar arrive. If this gap is near-zero, Phase 1 is TMEM-gated and arithmetic is fully in shadow — F22 saves instructions but zero wall-clock cycles. If the gap shows arithmetic-dependent latency (>50 cycles after last TMEM read), F22 has a target. This diagnostic should be added to F25's instrumentation list.

**Go/no-go:**
- Success: Phase 1 drops ≥200 cycles AND wall clock improves ≥1%
- Fail-fast (perf): if Phase 1 time is unchanged or increases (instruction count reduction eaten by register pressure / scheduling), abort
- Fail-fast (precision): if max absolute error exceeds 16.0 (2 ULPs of BF16 at magnitude ~1728) on the validation set, abort or flag for downstream accuracy review before committing
- Max effort: 3–4 hours
- Rollback: revert `epilogue_store` math path to FP32

---

## F23: 3-warp epilogue contention sweep

**Effort:** Medium (warp structure change + load balance solution)
**Expected impact:** Uncertain. Directly targets Phase 1 TMEM contention (4 warps reading simultaneously).
**Risk:** Load imbalance may negate contention reduction.

**Rationale:** F16 showed 5→4 warps helped (TMEM contention reduction + eliminating unbalanced split warps). F3 showed 6 warps hurt (TMEM bandwidth saturation). The optimal warp count hasn't been swept — it could be 3 if TMEM contention is non-linear (queueing effects between 4 concurrent readers).

Phase 1 TMEM readback = 4,140 cycles with 4 warps. If TMEM port contention follows a queueing model (latency grows super-linearly with concurrent readers), 3 warps could see per-warp Phase 1 time increase by less than 33%, making the slowest warp faster than the current 4-warp slowest.

**Challenge:** 4 row_groups (128 rows / 32 rows each) don't divide into 3 warps evenly.

Options:
- **A) 2+1+1 split:** W2 gets 2 row_groups (64 rows), W3-W4 get 1 each. W2 takes ~2x longer → mbar bottleneck. Same structural problem as old 5-warp split. Likely worse.
- **B) 3 row_groups of 42/43 rows:** Breaks 32-row TMEM alignment. TMEM_LOAD_X32 loads 32 elements per lane, tied to 32-row structure. Would need variable-length TMEM reads. Major restructuring.
- **C) 2 epilogue warps, 2 row_groups each:** Equal split, minimal contention. Each warp does 2× work. If TMEM contention drops enough (2 readers vs 4), per-warp time could be < 2× current. Test this first as a simpler version.
- **D) 3 warps with NC-split instead of row-split:** Each warp handles all 4 row_groups but only 85-86 columns. Keeps 32-row TMEM structure. But cols must be multiples of 32 (TMEM_LOAD_X32 granularity): 96+96+64 or 128+64+64. Unequal → mbar bottleneck again.

**Recommended approach:** Test option C first (2 epilogue warps). `NUM_EPI_WARPS=2`, each warp processes row_groups 0+1 or 2+3 (sequential, not parallel). Measures pure contention effect without load-balance confounds. If wall clock improves, explore 3-warp variants.

**Changes for option C:**
```c
#define NUM_EPI_WARPS 2
// Each warp processes 2 row_groups sequentially
const int rg_start = ew * 2;  // ew=0: row_groups 0,1. ew=1: row_groups 2,3
for (int rg = rg_start; rg < rg_start + 2; rg++) {
    epilogue_store<0, TN>(..., rg, ...);
}
```

**Drain epilogue fix required:** The drain epilogue (after tile loop) uses `ew % 4` for row_group assignment. With `NUM_EPI_WARPS=2`, `ew = {0, 1}` gives row_groups {0, 1} only — row_groups 2 and 3 are never drained. The drain epilogue must use the same `rg_start + loop` pattern as the main loop above.

Epilogue mbar expected arrivals: 2 × 2 × 32 = 128 (was 256).

**Thread count:** With `NUM_EPI_WARPS=2`, `THREADS = 32*(2+2) = 128` (was 192). Drop THREADS to 128 — don't keep 192 and park 2 warps at a barrier, because parked warps still consume register file slots (223 regs × 32 threads × 2 warps = 14,336 registers wasted). Reducing THREADS to 128 frees those register file slots, potentially allowing higher occupancy or reducing register pressure. Verify `ptxas` output: register count may change with fewer warps competing for the register file.

**Measurement:** Compare Phase 1 per-warp timing, epi_wait, and wall clock vs 4-warp baseline. Key metric: does reduced contention compensate for doubled per-warp work?

**Go/no-go:**
- Success: wall clock improves with 2 warps (option C)
- Fail-fast: if 2-warp per-warp Phase 1 time is ≥1.8× the 4-warp per-warp time, contention at 4 warps is ≤10% of Phase 1 — abort all warp-count variants
- Max effort: 2–3 hours (option C only)
- Rollback: revert `NUM_EPI_WARPS` to 4

---

## F24: Swizzled SMEM staging layout (long-term)

**Effort:** High (staging buffer architecture redesign)
**Expected impact:** Eliminates Phase 2B L1/LSU contention with K-loop (confirmed 170 cycles by F19a). Enables single-shot TMA tensor store for Phase 2B.
**Risk:** Complexity. Swizzled addressing affects Phase 1B writes and Phase 2B reads.

**Rationale:** F19a proved Phase 2B's `ld.shared.v2 + st.global.v2` stores contend with the K-loop on L1/LSU (+170 cycles to K-loop, perfect cancellation with early mbar savings). F19b proved TMA bulk stores eliminate this contention (K-loop improved 12.6%) but fail due to per-instruction overhead (32 × 256B stores at ~70 cycles each = 3× slower).

The solution is a single `cp.async.bulk.tensor.2d` TMA tensor store covering the entire 128-col × 32-row staging half (8 KB), amortizing the per-instruction overhead across one large transfer. This requires SMEM layout compatible with the TMA descriptor's swizzle mode — currently impossible because of the 272-byte padded rows (256 useful + 16 pad).

**Current staging layout:**
```
STAGING_HALF_ROW_BYTES = 272 (256 data + 16 pad)
Purpose of pad: bank-conflict-free Phase 1B writes
  lane 0 → offset 0 → bank 0
  lane 1 → offset 272 → bank 4  (272/16 = 17, 17 % 32 = 17)
  lane 2 → offset 544 → bank 2  (544/16 = 34, 34 % 32 = 2)
  Without pad (256B stride): all lanes map to bank 0 → 32-way conflict
```

**Required layout for TMA tensor store:**
```
CU_TENSOR_MAP_SWIZZLE_NONE: contiguous 256-byte rows (no pad)
  → 32-way bank conflict in Phase 1B writes

CU_TENSOR_MAP_SWIZZLE_128B: hardware swizzles 128B chunks
  → Phase 1B writes must use swizzle-aware addressing
  → Phase 2B reads (currently ld.shared.v2) replaced by TMA store (reads from SMEM internally)
```

**Changes needed:**
1. Remove STAGING_ROW_PAD. Staging rows become exactly 256 bytes (128 BF16 cols).
2. Phase 1B `st.shared` addressing must use 128B swizzle pattern to avoid bank conflicts. The `CVT_STS` macro's store address computation changes from `lane * 272 + col_offset` to a swizzled address that XORs row and column bits per the 128B swizzle formula.
3. Create TMA descriptor for staging buffer (2D, BF16, 128 cols × 32 rows, SWIZZLE_128B).
4. Replace Phase 2B's 32-iteration `COALESCED_STORE_V2` loop with single `cp.async.bulk.tensor.2d` store from staging_b to global C.
5. Add `cp.async.bulk.wait_group 0` before next Phase 1B (protect staging_b from overwrite).
6. Phase 2A stores from staging_a need same treatment (or keep manual if Phase 2A is in TMEM overlap window — contention with K-loop is less relevant there since it's hiding behind TMEM stalls).

**Swizzle addressing for Phase 1B writes:**
Do not hardcode a guessed XOR formula. Derive the exact addressing from the Blackwell TMA swizzle semantics and validate against `cuTensorMapEncodeTiled` behavior + microbench conflict checks.

**SMEM budget impact:**
- Current: 4 warps × 17,408 = 69,632 bytes (272B per half-row × 32 rows × 2 halves)
- New: 4 warps × 16,384 = 65,536 bytes (256B per half-row × 32 rows × 2 halves)
- Saves 4,096 bytes of SMEM. Total drops from ~201 KB to ~197 KB.

**Prerequisite:** F22 (BF16 arithmetic) should be done first. If F22 flips the equilibrium and epi_wait collapses, the 170-cycle Phase 2B contention becomes a larger fraction of the remaining gap and F24 becomes higher priority.

**Go/no-go:**
- Success: Phase 2B L1/LSU contention eliminated (K-loop improves ≥5% with TMA tensor store) AND total epilogue doesn't regress
- Fail-fast: if swizzled Phase 1B writes introduce bank conflicts worse than the current baseline (re-measure with `analyze_source_counters.py` before attempting), abort
- Max effort: 8–10 hours
- Rollback: revert staging layout to padded 272-byte rows

---

## F26: L2 persistence window for B (stream access-policy)

**Effort:** Low (host-side `cudaStreamAttrValue` setup)
**Expected impact:** Small — only relevant if F21 shows partial improvement
**Risk:** Near zero
**Prerequisite:** F21 must show partial TMA0_wait improvement (B is sometimes evicted). If F21 shows zero change, skip F26 entirely.

**Rationale:** `CU_TENSOR_MAP_L2_PROMOTION_*` (F21) is a TMA hint — it promotes fetched lines to L2 but doesn't prevent eviction. `cudaAccessPropertyPersisting` with `cudaStreamAttrValue.accessPolicyWindow` reserves actual L2 capacity for a specific address range. B = 576 KB (1.6% of B200's 36 MB L2). If A's 680 MB streaming traffic intermittently evicts B despite L2 promotion hints, hard persistence guarantees B stays resident.

**Change:**
```c
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr = d_B;
attr.accessPolicyWindow.num_bytes = 768 * 768;  // 768*768 FP8 = 576 KB
attr.accessPolicyWindow.hitRatio = 1.0f;
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

**Why it might not help:** If F21 shows zero TMA0_wait change, B is already fully L2-resident and the 856 cycles are pure A DRAM latency. No amount of L2 policy changes will help — the bottleneck is A's 680 MB streaming through DRAM, not B eviction.

**Go/no-go:**
- Success: TMA0_wait drops ≥50 cycles beyond F21's improvement
- Fail-fast: if F21 shows zero TMA0_wait change, skip F26 entirely — B is already L2-hot
- Max effort: 30 minutes
- Rollback: remove `cudaStreamSetAttribute` call

---

## F27: TMEM contention dephasing test — SKIPPED (F25 ruled out)

**Effort:** Low–medium (add staggered delays to epilogue warp Phase 1 start)
**Expected impact:** Uncertain. Tests whether burst TMEM read contention is worse than sustained.
**Risk:** Artificial delays waste cycles; net could be worse.
**Prerequisite:** F25 must show clear per-warp asymmetry (≥200-cycle spread in avg Phase 1). If all 4 warps have symmetric Phase 1 times, contention is bandwidth-limited (not port-queued) and dephasing won't help.
**Status:** F25 showed 172-cycle spread (< 200 threshold). Contention is bandwidth-limited. Skipped.

**Rationale:** After `mainloop_mbar` fires, all 4 epilogue warps wake near-simultaneously and issue `TMEM_LOAD_X32` within a few cycles of each other. If the TMEM read port has a shallow issue queue, 4 simultaneous requests create burst contention — queueing delays that grow super-linearly with concurrent requesters. Staggering Phase 1 start times would spread TMEM read pressure over time.

**This tests the same hypothesis as F23 option C but from the opposite direction.** F23C reduces warps (2 instead of 4) to reduce contention. F27 keeps 4 warps but spreads their TMEM reads in time. F25's per-warp data determines which approach (if either) has a target.

**Proposed stagger:** After `mainloop_mbar` wait, each epilogue warp delays by `ew * STAGGER_CYCLES` before starting Phase 1. Sweep `STAGGER_CYCLES` = {50, 100, 200, 400}. The stagger trades intentional idle cycles for reduced contention — it only wins if contention reduction exceeds the stagger cost.

**Implementation:** For staggers ≤200 cycles (~95 ns), use `clock64()` spin — `__nanosleep` minimum granularity is likely ~64 ns (hardware timer), making it unable to achieve 50-cycle (24 ns) or 100-cycle (48 ns) staggers accurately. For staggers ≥400 cycles (~190 ns), prefer `__nanosleep(ew * stagger_ns)` which parks the warp via hardware NANOSLEEP, freeing issue slots for W0/W1. The `clock64()` spin wastes issue slots but gives sub-ns control.

**What F25 tells us first:**
- **Symmetric Phase 1 across warps:** Contention is bandwidth, not queueing. F27 won't help (stagger adds idle time without reducing per-read latency). Skip.
- **One warp consistently faster (it "won" the port):** Port queueing exists. F27 can target the queue depth. Proceed.
- **Fat tail on max-warp Phase 1:** Burst contention causes outlier tiles. Dephasing would shave the tail. Proceed.

**Go/no-go:**
- Success: wall clock improves ≥1% at any stagger value
- Fail-fast: if F25 shows symmetric per-warp Phase 1 (spread <200 cycles), skip entirely
- Max effort: 2 hours
- Rollback: remove stagger spin-wait

---

## F28: K-loop restructuring — tighter MMA dispatch

**Effort:** Medium (rewrite W1 K-loop body, touch `make_smem_desc` and MMA inline asm)
**Expected impact:** ~200–350 cycle K-loop reduction (~5–8%), worth ~0.5–1.5% wall clock **only if paired with F22** (epilogue absorbs gains otherwise)
**Risk:** Register pressure shift, compiler scheduling differences. Inline PTX changes can surprise `ptxas`.
**Prerequisite:** Must be done together with or after F22. Alone, the epilogue equilibrium swallows most of the savings (see analysis below).

### The problem: 20 overhead instructions per MMA

`analyze_source_counters.py` on current SourceCounters data shows W1's tile budget:

```
Total W1 instructions per tile:  411
  UTCQMMA.2CTA (MMA):            24    (5.8%)
  SYNCS (mbar_wait):               7    (1.7%)
  UTCBAR (commit):                 7    (1.7%)
  Everything else:               373   (90.8%)  ← this is the problem
```

24 MMA instructions do all the FLOPs. 373 instructions do bookkeeping. That's **15.5 overhead instructions per MMA issue.**

To understand why this matters, consider what the tensor core sees. The TC doesn't know about our tile loop or our pipeline — it just sees a stream of MMA instructions arriving at its input, with gaps between them. Every gap is a cycle the TC sits idle. We can count the gap:

```
K-loop:          4,154 cycles
MMA instructions:   24
Cycles per MMA slot: 4,154 / 24 = 173 cycles

If MMA execution takes ~140 cycles (estimated from TC pipe active = 50%
  at 24 MMAs × 148 SMs over 0.543 ms), then:
  Gap between MMA issues: 173 - 140 = ~33 cycles
  TC idle fraction from gaps: 33 / 173 = 19%
```

cuBLAS achieves ~79% TC utilization. We're at ~50%. Stripping out the epilogue wait (17%) and TMA0 wait (14%), our K-loop-only TC utilization is roughly:

```
K-loop fraction of tile: 4,154 / 6,067 = 68.5%
TC active while K-loop runs: 50% / 68.5% = ~73%
TC idle within K-loop: ~27%
  → inter-MMA gaps:   ~19%
  → pipeline bubbles:  ~8% (MMA internal, not actionable)
```

cuBLAS's K-loop-only TC utilization is ~79% / ~80% ≈ ~99% — essentially gap-free. They've squeezed the inter-MMA overhead to near zero. Our 19% gap costs us roughly:

```
19% of 4,154 cycles = ~789 cycles of TC idle time within the K-loop
Even halving this = ~395 cycles saved
```

### Where the 437 instructions come from

Here is the current K-loop body (lines 582–630), one iteration, annotated with what the compiler actually emits:

```
① mbar_wait(tma_mbar[s], tma_phase[s])          SASS: SYNCS (try_wait) + SELP + BRA
                                                  ~3–5 instructions, usually passes first try
② tma_phase[s] ^= 1                              SASS: LOP3.LUT (XOR) + STL (phase to local)
                                                  ~2 instructions
③ tcgen05.fence::after_thread_sync               SASS: MEMBAR variant
                                                  ~1 instruction, but may have pipeline drain cost
④ desc_a = make_smem_desc(smem_a[s])             SASS: see breakdown below
⑤ desc_b = make_smem_desc(smem_b[s])                   ~10–16 instructions total for ④+⑤
⑥ setp + MMA (first sub-tile, accumulate=0|1)    SASS: SETP + R2UR×2 + ELECT + PLOP3 + UTCQMMA
                                                  ~5–7 instructions around each MMA
⑦⑧⑨ desc += 2; setp + MMA (×3 sub-tiles)         SASS: IADD3 + R2UR×2 + SETP + UTCQMMA (×3)
                                                  ~5 instructions each × 3 = ~15
⑩ tcgen05_commit_mcast(mma_mbar[s])              SASS: UTCBAR
                                                  ~1 instruction
```

Per iteration: ~37–47 instructions for 4 MMAs. Over 6 iterations: ~222–282. Add tile-level overhead (epilogue mbar wait, buf/tile index computation, mainloop commit, phase tracking) for the remaining ~155–215.

Now zoom into the two costliest items:

**`make_smem_desc` — the descriptor tax (steps ④⑤):**

```c
uint64_t make_smem_desc(uint32_t addr) {
    uint64_t d = 0;
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);           // extract base addr bits [17:4]
    d |= (uint64_t)((SBO  & 0x3FFFF) >> 4) << 32;     // encode stride byte offset
    d |= (1ULL << 46);                                 // LBO bit
    d |= (2ULL << 61);                                 // SWIZZLE_128B mode
    return d;
}
```

This looks small in C. In SASS it expands to: AND (mask 0x3FFFF), SHF.R (shift right 4), IMAD.MOV (zero-extend to 64-bit), AND+SHF for SBO (compile-time, but the 64-bit OR chain forces IMAD.WIDE or LOP3 pairs), two more OR/IMAD.MOV for the constant bits. The compiler emits **~5–8 instructions per call**, doing 64-bit register arithmetic on a 32-bit machine. Called twice per K-iteration = **10–16 instructions** that produce the same values every single tile.

The key observation: `smem_a[s]` is computed once before the tile loop (line 505: `smem_a[s] = smem_to_uint(smem + s * STAGE_BYTES)`). It never changes. SBO is `#define 1024`. So `make_smem_desc(smem_a[0])` returns the *exact same 64-bit value* on tile 1 as on tile 10,878. We recompute it 10,878 × 6 = 65,268 times.

**R2UR moves — the uniform register tax (step ⑥):**

`tcgen05.mma.cta_group::2` requires its SMEM descriptor operands in **uniform registers** (URs). The descriptors are computed in regular registers (the `asm volatile` takes `"l"(desc_a)` — a 64-bit regular register). Before the MMA fires, `ptxas` must insert `R2UR` (Register-to-Uniform-Register) moves to shuttle the descriptors across. Each 64-bit descriptor = 2 × R2UR (one per 32-bit half). Two descriptors × 4 sub-MMAs × 6 iterations = up to **96 R2UR instructions per tile**. In practice the compiler reuses some across sub-MMAs (desc only changes by +2), but the base transfer for each iteration is unavoidable.

### The four changes

**A) Descriptor precomputation:**

`smem_a[s]` and `smem_b[s]` are constant across all tiles. Compute all descriptors once, before the tile loop:

```c
uint64_t desc_a_base[N_STAGES], desc_b_base[N_STAGES];
for (int s = 0; s < N_STAGES; s++) {
    desc_a_base[s] = make_smem_desc(smem_a[s]);
    desc_b_base[s] = make_smem_desc(smem_b[s]);
}
```

Inside the K-loop, replace:
```c
uint64_t desc_a = make_smem_desc(smem_a[s]);  // 5-8 instructions
uint64_t desc_b = make_smem_desc(smem_b[s]);  // 5-8 instructions
```
with:
```c
uint64_t desc_a = desc_a_base[s];  // 1 register move (or zero, if compiler keeps in reg)
uint64_t desc_b = desc_b_base[s];  // 1 register move
```

**Saves:** ~10–16 instructions per K-iteration × 6 = **60–96 instructions per tile**.

**Register cost:** 8 registers (4 stages × 2 matrices × 1 64-bit value = 4 × 2 × 2 = 16 × 32-bit regs). Current: 223. After: ~231. Still below 255. But watch for `ptxas` deciding to spill — the 8-reg increase is meaningful at 223.

**Why it wasn't done before:** It was listed as "secondary residual" in the ruled-out table. That was correct when the epilogue was the binding constraint — K-loop savings got swallowed by epi_wait. With F22 potentially shrinking the epilogue, K-loop overhead becomes a larger fraction of the remaining gap.

**B) Persistent uniform registers:**

The deeper fix. Instead of computing descriptors in regular registers and letting `ptxas` insert R2UR moves, compute them directly in URs using uniform register inline PTX:

```c
// Before tile loop — compute once, store in URs
uint64_t ur_desc_a[N_STAGES], ur_desc_b[N_STAGES];
for (int s = 0; s < N_STAGES; s++) {
    uint32_t lo_a = (smem_a[s] & 0x3FFFF) >> 4;
    uint32_t hi_a = ((SBO & 0x3FFFF) >> 4) | (1u << 14) | (2u << 29);
    asm volatile("mov.u32 %0, %1;" : "=r"(ur_desc_a_lo[s]) : "r"(lo_a));
    // ... pack into UR pair
}
```

The challenge: CUDA doesn't expose uniform registers directly. The programmer writes regular C/PTX and `ptxas` decides what goes in URs. You can't force a variable into a UR from PTX — it's a `ptxas` register allocation decision. What you *can* do is structure the code so that `ptxas` recognizes the descriptors as uniform (same value across all lanes in the warp) and promotes them to URs itself. Since W1 is single-lane (`lane == 0`), everything is trivially uniform — but `ptxas` may not exploit this because the MMA asm block takes regular register constraints (`"l"(desc_a)`).

**Practical approach:** Change the MMA inline asm to use the descriptor as an *immediate-like* input. Instead of passing `desc_a` as a register operand, construct the descriptor inline within the asm block using UR arithmetic. This is fragile and ISA-version-dependent. A safer alternative: precompute descriptors (change A above) and trust that `ptxas` can hoist the R2UR out of the loop when the source register is loop-invariant. Verify by inspecting SASS output (`cuobjdump --dump-sass`).

**Saves (if R2UR elimination works):** Up to ~48–96 R2UR instructions per tile. Realistically, `ptxas` already eliminates some. Estimate: ~24–48 instructions saved.

**Risk:** High. Fighting the compiler's register allocator rarely ends well. The SASS output must be inspected to verify actual R2UR elimination. If `ptxas` stubbornly inserts R2UR inside the loop regardless of hoisting, this change does nothing.

**Recommendation:** Implement A (precomputation) first, inspect SASS. If R2UR instructions persist in the inner loop, attempt B. If R2UR is already hoisted, B is unnecessary.

**C) Fully unrolled with hardcoded constants:**

The K-loop is `for (int ki = 0; ki < K_ITERS; ki++)` with `K_ITERS = 6` (compile-time constant). `nvcc` should auto-unroll this. But there are obstacles:

1. **`ki % N_STAGES` inside `asm volatile`:** The stage index `s = ki % 4` is constant per unrolled iteration (0, 1, 2, 3, 0, 1). But the `asm volatile` blocks are opaque — the compiler can't propagate `s` through them. It must compute `tma_mbar[s]`, `tma_phase[s]`, `smem_a[s]` at runtime even when `s` is a known constant.

2. **The accumulate predicate is inside the asm block:**
    ```c
    uint32_t accumulate = (ki == 0) ? 0 : 1;
    asm volatile(
        "setp.ne.b32 p, %4, 0;\n\t"   // THIS runs every MMA
        "tcgen05.mma ... p;\n\t"
        : : ... "r"(accumulate) ...);
    ```
    After unrolling, `accumulate` is 0 for `ki=0` and 1 for `ki=1..5`. The compiler knows this. But the `setp` is inside `asm volatile` — it can't be eliminated. Every MMA gets a redundant `setp` even though the predicate is compile-time known.

**Fix for (1):** Manual unrolling with explicit stage indices. Replace the `for` loop with 6 explicit blocks:
```c
// ki=0, s=0
mbar_wait(tma_mbar[0], tma_phase[0]);
tma_phase[0] ^= 1;
asm volatile("tcgen05.fence::after_thread_sync;");
// Use desc_a_base[0], desc_b_base[0] from precomputation
MMA_BLOCK(buf, desc_a_base[0], desc_b_base[0], /*accumulate=*/0);
tcgen05_commit_mcast(mma_mbar[0], 0x3);

// ki=1, s=1
mbar_wait(tma_mbar[1], tma_phase[1]);
tma_phase[1] ^= 1;
asm volatile("tcgen05.fence::after_thread_sync;");
MMA_BLOCK(buf, desc_a_base[1], desc_b_base[1], /*accumulate=*/1);
tcgen05_commit_mcast(mma_mbar[1], 0x3);

// ... ki=2..5
```

**Fix for (2):** Two separate MMA macros — one without the predicate (ki=0, clear accumulator), one with `p = true` hardcoded (ki=1..5, accumulate). Eliminates 24 `setp` instructions per tile:

```c
#define MMA_FIRST(tmem_off, da, db, idesc) \
    asm volatile( \
        "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, " \
        "{0,0,0,0, 0,0,0,0};\n\t" \
        :: "r"(tmem_off), "l"(da), "l"(db), "r"(idesc));

#define MMA_ACCUM(tmem_off, da, db, idesc) \
    asm volatile( \
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, 1, 0;\n\t" \
        "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, " \
        "{0,0,0,0, 0,0,0,0}, p;\n\t}" \
        :: "r"(tmem_off), "l"(da), "l"(db), "r"(idesc));
```

Wait — `setp.ne.b32 p, 1, 0` still emits a `setp`. The real question is whether `tcgen05.mma` even needs the predicate when accumulate is always true. Check the ISA: if the predicate operand is mandatory, we can't eliminate the `setp`. But we can make it a compile-time constant (`setp.ne.b32 p, 1, 0` → p is always true → `ptxas` might optimize). If the ISA supports an unpredicated accumulate form, use that instead.

**Saves:** ~24 setp + associated PLOP3 = ~30–40 instructions per tile from (2). Array indexing elimination from (1) saves another ~20–40 instructions (IMAD for tma_mbar[s] address, tma_phase[s] load/store with runtime index). **Total: ~50–80 instructions.**

**Risk:** Code size explosion. 6 manually unrolled iterations × ~20 instructions each = ~120 instructions of MMA body. Plus the sub-MMA loop (×3 per iteration) must also be unrolled: 6 × 3 = 18 additional MMA blocks. Total: 24 MMA issue sites in the source. This is large but bounded, and icache is at 0.22% — no pressure concern.

**D) Software-pipelined MMA dispatch:**

The current iteration order within the K-loop is:

```
wait_tma[s] → fence → build_desc → MMA×4 → commit
```

The descriptor building sits on the **critical path** between the TMA data arriving and the first MMA issue. In a trace:

```
time →
TMA data arrives ──┐
                   ├─ fence (1+ cycle)
                   ├─ desc build (10-16 cycles)    ← TC idle, waiting for desc
                   ├─ R2UR moves (4-8 cycles)      ← TC still idle
                   └─ MMA[0] issues ──── TC starts ─────────────→
                      MMA[1] issues (desc_a += 2) ──────────────→
                      MMA[2] issues ────────────────────────────→
                      MMA[3] issues ────────────────────────────→
                      commit
```

Those 15–25 cycles of desc build + R2UR are dead time between "data is ready" and "TC starts." Precomputation (change A) already eliminates the desc build, leaving only R2UR. But software pipelining eliminates the remaining gap entirely:

```
Pipelined order (within each iteration):
  MMA×4 using descs prepared in PREVIOUS iteration
  commit
  prep descs for NEXT iteration      ← overlaps with MMA async execution
  wait_tma[s+1]
  fence
```

The descriptor preparation (even without precomputation — building from scratch) now runs while the current iteration's MMAs execute on the TC. By the time the fence completes for the next stage, descriptors are ready and the MMA fires immediately.

**With precomputation (A) already done**, software pipelining is less impactful — the desc fetch is just a register load (~1 cycle). The remaining value is moving the `mbar_wait` + `fence` after the MMA issue, so TMA wait time overlaps with MMA execution rather than preceding it. In steady state the TMA data is usually ready (mbar passes immediately), so the reorder saves ~1–5 cycles per iteration at best. **Only worth attempting if A alone doesn't close enough gap.**

**Implementation sketch:**
```c
// Prologue: wait for ki=0, prep descs
mbar_wait(tma_mbar[0], tma_phase[0]);
tma_phase[0] ^= 1;
asm volatile("tcgen05.fence::after_thread_sync;");
uint64_t next_desc_a = desc_a_base[0];
uint64_t next_desc_b = desc_b_base[0];

for (int ki = 0; ki < K_ITERS; ki++) {
    // Issue MMAs with already-prepared descs
    uint64_t da = next_desc_a, db = next_desc_b;
    // ... 4 sub-MMAs using da, db ...
    tcgen05_commit_mcast(mma_mbar[ki % N_STAGES], 0x3);

    // Prep next iteration (overlaps with MMA execution)
    if (ki + 1 < K_ITERS) {
        int ns = (ki + 1) % N_STAGES;
        next_desc_a = desc_a_base[ns];  // or recompute
        next_desc_b = desc_b_base[ns];
        mbar_wait(tma_mbar[ns], tma_phase[ns]);
        tma_phase[ns] ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;");
    }
}
```

**Saves:** With A already applied, marginal (~5–30 cycles from fence/wait reorder). Without A, significant (~60–100 cycles from overlapping desc build with MMA execution).

**Risk:** The conditional `if (ki + 1 < K_ITERS)` must be resolved by unrolling or it blocks compiler optimization (this is exactly what killed F20's first attempt — runtime loop bounds → 44% regression). With manual unrolling (change C), this is a non-issue.

### Why this must be paired with F22

The kernel is in producer-consumer equilibrium with a ~257-cycle deficit (epilogue slower):

```
W1 productive:  TMA0(857) + K-loop(4,154) = 5,011 cycles
Epilogue eff:   ml_wait(1,139) + Phase1(4,129) = 5,268 (what W1 blocks on)
Deficit:        5,268 - 5,011 = 257 cycles → amplified to ~1,056 epi_wait
```

**K-loop savings alone widen the deficit.** Saving 300 cycles makes W1 productive = 4,711, deficit = 557, epi_wait grows to ~1,600+ → tile time *increases*. This is confirmed by `analyze_timing.py` which projects K-loop -300 alone as a regression.

**Pairing with F22 attacks both sides.** If F22 reduces Phase 1, the deficit shrinks (or holds) even as K-loop shrinks W1 productive time. The exact outcome depends on the amplification model, which uses a linear approximation (epi_wait ≈ deficit × 4.1) that is **unreliable for large perturbations** (>200 cycles on either side). At the F22(-200)+F28(-300) operating point, the model gives answers near 1.0x — the scenario is right at the boundary between slight improvement and slight regression.

**Bottom line:** F28 alone is guaranteed to regress. F28+F22 is uncertain but plausible. Run `python3 analyze_timing.py` for current projections, but treat combined scenarios as "needs measurement" rather than "predicted improvement." F28 should be gated on measured post-F22 equilibrium, not on pre-implementation projections.

### Implementation plan

Implement A and C together (precomputed descriptors + manual unroll). These are synergistic — manual unrolling makes precomputed descriptor indexing trivial (hardcoded stage constants), and eliminates the setp/PLOP3 overhead.

Skip B initially (fighting `ptxas` UR allocation) — inspect SASS after A+C to see if R2UR is still a problem. Skip D initially (software pipelining) — with precomputed descriptors the critical-path desc tax is already eliminated.

**Changes:**
1. Before tile loop: compute `desc_a_base[4]`, `desc_b_base[4]` from `smem_a[s]`, `smem_b[s]`
2. Replace `for (int ki = 0; ki < K_ITERS; ki++)` with 6 explicit blocks
3. Each block uses `desc_a_base[s]` directly (s is a literal constant: 0, 1, 2, 3, 0, 1)
4. ki=0 uses `MMA_FIRST` (no accumulate predicate), ki=1..5 use `MMA_ACCUM` (accumulate=1 hardcoded)
5. Sub-MMA loop `for (int sub = 1; sub < MMA_PER_KI; sub++)` also manually unrolled (3 iterations, all `MMA_ACCUM`)
6. `tma_phase[s]` tracking: each unrolled block directly names its phase variable (no array indexing)

**Expected SASS reduction:** ~110–160 instructions eliminated per tile (from 411 → ~251–301). Primarily from descriptor recomputation (50–80), setp/PLOP3 (25–35), array index arithmetic (15–30). Estimates scaled from stale 475-instruction baseline; re-verify category counts with `analyze_source_counters.py --mma-detail` before implementation.

**Expected cycle reduction:** ~150–350 cycles from the 4,154-cycle K-loop. Not all instruction eliminations translate 1:1 to cycle savings — the compiler may fill some slots with other work, and some instructions were hiding behind MMA latency. The honest range accounts for this absorption.

**Register impact:** +16 32-bit registers for descriptor precomputation (4 stages × 2 matrices × 2 regs per 64-bit desc = 16). Current: 223. After: ~239 — just below the 240 fail-fast threshold. Manual unrolling may increase or decrease register pressure depending on `ptxas` live range analysis. Monitor: if regs exceed 240 or spills appear, abort.

**Verification:**
1. Compare SASS instruction count in W1's K-loop region (before vs after)
2. `cuobjdump --dump-sass` and count IMAD/SHF/LOP3/R2UR between UTCQMMA instructions
3. clock64 timing: K-loop cycles should drop
4. Checksum must be unchanged (this is a pure scheduling change, no arithmetic modification)

**Go/no-go:**
- Success: K-loop drops ≥150 cycles AND wall clock improves ≥0.5% (when combined with F22)
- Fail-fast: if `ptxas` register count exceeds 240 or spills appear, abort — register pressure is destroying scheduling. If K-loop cycles are unchanged despite instruction reduction, the eliminated instructions were fully hidden behind MMA latency — the overhead was never on the critical path, and no restructuring will help.
- Max effort: 3–4 hours (A+C only, manual unroll is mechanical)
- Rollback: revert K-loop to `for` loop with inline `make_smem_desc`

---

## Ruled out (see EXPERIMENTS.md for full details)

| Proposal | Why dead |
|----------|----------|
| Next-tile TMA prefetch (F20) | ~0% — DRAM bandwidth, not scheduling. Prefetch shifts TMA issue by ~20 cycles vs 818-cycle shortfall. |
| cp.async.bulk Phase 2B stores (F19b) | 3× slower than manual stores. 32 × 256B at ~70 cycles/instruction overhead. |
| 5-stage pipeline | Would need SMEM freed and register impact validated. Current 4-stage at 131 KB leaves 27 KB headroom. 5-stage = 163 KB → only 65 KB for staging (needs 70 KB). |
| TMA multicast for B | B is N-split across CTAs (each loads different half). Zero redundancy. dest_multicast=0 is correct. |
| TN=128 | 2× tiles, same per-tile overhead, 11.4% regression. Definitively ruled out at both 1190 and 1560 TFLOPS. |
| Combined load L1 bypass | F13 relayout already solved combined loads. L1 at ~67% is equilibrium — remaining L1 is mostly K-loop traffic. |
| Direct CVT_STG stores (F17) | Uncoalesced st.global.v8 contends with TMEM reads on L1. Phase 1 +19%, net regression. |
| 6 epilogue warps (F3) | TMEM bandwidth saturation. |
| ≥5 pipeline stages + TN=128 combos (F12) | Best was 1560 TFLOPS (6-stage). Tile transition overhead dominance. |
