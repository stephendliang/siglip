# Future Proposals

**Kernel state (2026-03-01):** 0.530 ms / 2067 TFLOPS. 236 regs, 0 spills.
**Reference:** cuBLAS pure GEMM = 0.365 ms / 3001 TFLOPS | cuBLAS + unfused pos_embed = 0.835 ms
**CUTLASS best fused BF16 (256x256x128 2x1):** 0.536 ms / 2044 TFLOPS

## The problem: our epilogue architecture is needlessly complex

The kernel's value is fusion — the overlapped epilogue hides epilogue time behind the K-loop. This is the right idea. But the *implementation* of the epilogue is a mess of historical accidents that creates unnecessary L1 contention.

### What we actually compute

The epilogue computation is trivial: `D[r,c] = bf16(acc[r,c]) + combined[r,c]`

That's it. One F2FP conversion, one BF16 add, per element. CUTLASS does the same thing. The computation is not the problem.

### What we actually do (the problem)

Each epilogue warp processes 32 rows x 256 cols. The 256 cols are split into two halves, each with a *different* staging layout and a *different* store mechanism:

```
Phase 1A:  TMEM → regs → F2FP+HADD2 → STS to staging_a (LINEAR, 272B rows, 16B pad)
Phase 1B:  TMEM → regs → F2FP+HADD2 → STS to staging_b (SWIZZLE_128B, 128B rows)
           ↑ interleaved with Phase 2A ↓
Phase 2A:  LDS from staging_a → STG to global (64 st.global.v2 through L1)
Phase 2B:  TMA tensor store from staging_b → global (2 cp.async.bulk, async, off L1)
```

Count the redundancy:
- **Two staging buffer layouts** (linear + swizzle) for the same data type doing the same job
- **Two store mechanisms** (64 manual STG + 2 TMA stores) writing to the same output matrix
- **64 LDS instructions** to read back data we just wrote to SMEM, solely because staging_a can't use TMA
- **64 LDG instructions** loading combined from global through L1, competing with everything else
- **16B padding per row** in staging_a (272B vs 256B) to work around bank conflicts from the linear layout — a problem swizzle solves for free

The Phase 2A STG instructions go through L1 and contend with TMEM reads during Phase 1B. F19a measured **~170 cycles of L1/LSU contention** from this. We're paying a performance penalty because half our output path uses the wrong SMEM layout.

### Why it's this way (historical debt)

Before F24, *both* halves used manual LDS+STG (Phase 2B was 899 cycles). F24 converted staging_b to swizzle + TMA stores, dropping Phase 2B to 273 cycles. But staging_a was left as-is because converting it to swizzle caused a +440 cycle Phase 1 regression (swizzle address overhead and/or register pressure). The net was still positive (+0.7%), but only because Phase 2B improved so dramatically.

Result: a half-migrated architecture where one half is clean (swizzle + TMA) and the other half is legacy (linear + manual STG).

### What CUTLASS does instead

CUTLASS 256x256x128 BF16 epilogue (from SASS comparison):
```
TMA load C → SMEM (async, off L1)         0 LDG
LDTM.x64 → 64 FP32 regs                  4 loads (vs our 40 LDTM.x16)
LDS C from SMEM                           81 LDS
F2FP + HFMA2                              128 F2FP + 143 HFMA2
STS → swizzle SMEM                        36 STS
TMA store → global                        0 STG
```

One staging layout (swizzle). One store mechanism (TMA). Zero STG. Zero LDG in the hot loop.
The computation is the same. The memory path is cleaner.

CUTLASS runs this epilogue *sequentially* after the K-loop. We overlap ours with the next K-loop. Our architecture is better in principle — we just execute it worse.

### The SASS evidence

```
                              CUTLASS    Custom    Delta
TMEM loads (LDTM)                   4        40      -36
R2UR (descriptor setup)            26       428     -402
F2FP (BF16 pack)                  128       256     -128
STG (global stores)                 0        64      -64
LDG (global loads)                  5        64      -59
Total instructions              3032      2832     +200
```

We have fewer total instructions (2832 vs 3032) but more of the *wrong kind*. The 64 STG + 64 LDG create L1 traffic that contends with TMEM reads. CUTLASS's 0 STG + 5 LDG keeps the L1 clean.

---

## Equilibrium model

```
W1:       epi_wait(1,344) + TMA0(662) + K-loop(4,059) = 6,066 cycles/tile
Epilogue: ml_wait(1,538) + Phase1(4,345) + Phase2B(273) = 6,156 cycles/tile
Deficit:  90 cycles (epilogue slower) → amplified to 1,344-cycle epi_wait
```

W1 productive = TMA0 + K-loop = 4,721 cycles.
Epilogue productive = Phase1 + Phase2B = 4,618 cycles.

Epilogue productive is **already 103 cycles faster** than W1 productive. Yet epi_wait is 1,344 cycles because of double-buffer amplification — startup deficit compounds over tiles.

**Phase transition:** if we can eliminate enough Phase 1 overhead to reliably outrun W1, epi_wait drops to ~0 and per-tile throughput drops from ~6,100 to ~4,721 cycles. That's a 22.6% reduction.

**Ceilings:**
- Eliminate epi_wait: ~0.43 ms / ~2,646 TFLOPS (+28%)
- Also eliminate TMA0 (impossible — A-matrix DRAM latency): ~0.36 ms / ~3,077 TFLOPS

---

## F38: Unified epilogue — eliminate Phase 2A entirely

**Effort:** 8-12 hours
**Expected impact:** Phase 1 saves ~170 cycles (L1 contention elimination). Possible equilibrium phase transition → 2,300-2,600 TFLOPS. If transition doesn't occur: ~2,100-2,200 TFLOPS.

### Thesis

Don't fix staging_a. Delete it. Process all 256 cols through swizzle staging, store everything via TMA. No Phase 2A, no dual-staging, no LDS+STG.

### Current vs proposed

**Current (asymmetric, two store paths):**
```
Phase 1A: TMEM → CVT+ADD → STS staging_a (linear)     [128 cols]
Phase 1B: TMEM → CVT+ADD → STS staging_b (swizzle)    [128 cols]
          ↕ interleaved with Phase 2A: LDS staging_a → STG global (64 STG, L1 traffic)
Phase 2B: TMA store staging_b → global                 [128 cols, during next K-loop]
```

**Proposed (symmetric, one store path):**
```
Phase 1:  TMEM → CVT+ADD → STS staging (swizzle, 4 regions of 32x64 BF16)  [256 cols]
Phase 2:  TMA store staging → global (4 cp.async.bulk, during next K-loop)  [256 cols]
```

### What this eliminates

| Removed | Count | Why it existed |
|---------|-------|----------------|
| STG (st.global) | 64 | Phase 2A manual stores (staging_a not TMA-compatible) |
| LDS (ld.shared) | ~32 | Phase 2A reads from staging_a |
| staging_a buffer | 8,704 B/warp | Linear layout, bank-conflict pad, can't use TMA |
| L1 contention | ~170 cycles | STG during Phase 1B competes with TMEM reads |
| Phase 2A interleaving | all | No longer needed — nothing to interleave |

### What this adds

| Added | Count | Cost |
|-------|-------|------|
| TMA stores | +2 (2→4) | ~130 extra cycles in Phase 2, fully overlapped with K-loop |
| Swizzle XOR per STS | same | Already done for staging_b; extending to all 256 cols |

### SMEM layout

4 swizzle regions per warp, each 32 rows x 64 cols x 2 bytes = 4,096 bytes:
- Per warp: 4 x 4,096 = 16,384 bytes
- 4 warps: 65,536 bytes

Current staging: 4 x 16,896 = 67,584 bytes. **New staging is 2,048 bytes smaller** — the 16B/row linear padding in staging_a is gone.

### Why F24's +440 cycle regression doesn't apply

F24 converted staging_a to swizzle but *kept Phase 2A* (LDS + STG from the now-swizzled staging_a). The regression was from the combination of swizzle STS + swizzle LDS in the same Phase 1B loop, plus possible register pressure from holding swizzle constants for both buffers simultaneously.

The proposed redesign has no Phase 2A at all. During the TMEM stall windows in Phase 1, the only work is combined LDG (~50 cycles, L2-cached). The stall windows are "wasted" but **uncontended** — no L1 traffic from STG means TMEM reads face no contention.

The trade-off: we lose ~100 cycles of useful Phase 2A work that filled TMEM stall windows, but we save ~170 cycles of L1 contention that Phase 2A caused. Net: ~70 cycles Phase 1 reduction, plus whatever the swizzle STS overhead is for the staging_a half (should be small — one XOR per store, F30 confirmed compiler hoists loop-invariant swizzle math).

### The phase transition math

Current deficit: epilogue 90 cycles slower than W1 per tile.
Phase 1 reduction needed to flip: >90 cycles.
Expected Phase 1 reduction: 70-170 cycles (conservative to optimistic).

If it flips:
- epi_wait drops from 1,344 to ~0
- Per-tile time drops from ~6,100 to ~4,721 cycles
- Wall clock: ~0.43 ms / ~2,646 TFLOPS

If it doesn't quite flip (70-cycle reduction, deficit shrinks but doesn't invert):
- Per the amplification model: 70 cycles Phase 1 → ~84 cycles wall clock/tile
- Wall clock: ~0.52 ms / ~2,106 TFLOPS

**Range: 2,100-2,600 TFLOPS. Binary outcome depending on equilibrium flip.**

### Phase 2 budget

Current Phase 2B: 2 TMA stores in 273 cycles (during 4,059-cycle K-loop — 93% margin).
Proposed Phase 2: 4 TMA stores in ~400 cycles (during 4,059-cycle K-loop — 90% margin).

TMA stores are fully async and overlap with the K-loop via early mbar_arrive. The extra 2 stores are free.

### Implementation

1. Replace `staging_a` + `staging_b` with 4 identical swizzle regions per warp (32x64 BF16, SWIZZLE_128B)
2. Phase 1: loop over 8 x32 TMEM chunks (256 cols / 32), each chunk → CVT+ADD → STS to the appropriate swizzle region
3. Remove Phase 2A interleaving from the Phase 1B loop
4. syncwarp after Phase 1 complete
5. Early mbar_arrive (same as current — signals TMEM buffer free)
6. Phase 2: 4 TMA tensor stores from 4 staging regions (lane==0)
7. Create TMA tensor map for output C with SWIZZLE_128B matching staging layout

### Interaction with x32 TMEM loads (F39)

This proposal assumes x32 TMEM loads (available via LOAD_32_COLS when TMEM_LOAD_WIDTH=32; not the current default). Combining with F39 (switch default to x32, remove x16 codepath) simplifies the implementation — one TMEM load width, one staging layout, one store path.

### Go/no-go

- Success: wall clock ≤ 0.510 ms (≥ 2,148 TFLOPS, +4% over baseline)
- Stretch goal: wall clock ≤ 0.476 ms (≥ 2,300 TFLOPS, equilibrium flip)
- Fail-fast: if Phase 1 cycle count increases (swizzle overhead > contention savings), abort
- Fail-fast: if register count exceeds 250 (too close to ceiling), abort
- Max effort: 12 hours
- Rollback: revert to current dual-staging architecture

---

## F39: Default to x32 TMEM loads — code cleanup

**Effort:** 1 hour
**Expected impact:** Zero wall clock change. Cleaner code, fewer SASS instructions.

### Rationale

F37 proved x16/x32/x64 TMEM loads are statistically identical in wall clock (p=0.617). The LOAD_32_COLS macro already uses x32 when TMEM_LOAD_WIDTH=32, but the default is x16 (two TMEM_LOAD x16 instructions per 32-col chunk).

x32 halves the static instruction count in the epilogue:
- LDTM: 40 → 20 (half the load instructions)
- R2UR: ~428 → ~214 (half the descriptor setup)
- ELECT: ~45 → ~23
- PLOP3: ~171 → ~86

Register count may drop modestly (fewer R2UR temporaries). No wall clock change expected — R2UR and LDTM overhead is hidden in TMEM stall windows.

### Implementation

1. Change `#define TMEM_LOAD_WIDTH 16` → `#define TMEM_LOAD_WIDTH 32` (line 20)
2. Optionally: remove the x16 codepath from LOAD_32_COLS (dead code elimination)
3. Keep TMEM_LOAD_X64 and the x64 codepath for reference/future use

### Interaction with F38

Do F39 first. It simplifies the codebase before the larger F38 restructuring and establishes x32 as the baseline.

### Go/no-go

- Success: wall clock unchanged (within measurement noise ±0.003 ms), checksum correct
- Bonus: register count drops (any amount)
- Fail-fast: wall clock regresses by >0.005 ms → revert
- Max effort: 1 hour
- Rollback: change one `#define` back

---

## Execution order

```
F39 (x32 default)  ──→  F38 (unified epilogue)
     1 hour                  8-12 hours
     code cleanup            the real bet
```

F39 is a no-risk cleanup. F38 is the only remaining proposal with a realistic path to 2,300+ TFLOPS.

---

## Completed experiments

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
