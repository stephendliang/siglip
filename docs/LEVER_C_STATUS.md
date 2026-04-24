# Lever C (USE_STMATRIX=1) on fc2_w3x — implementation status

Status: compiles, emits target SASS shape, correctness **unverified** on B200.
Implemented in worktree `worktree-agent-a4cce994` on top of master.

## What changed

### `fc2_w3x.cu`

Two new PTX macros, one default-off ifdef guard, and a rewritten epilogue
subpass body under `#if USE_STMATRIX`:

- `TMEM_LOAD_16X256_X4(a0..a15, TADDR)` — `tcgen05.ld.sync.aligned.16x256b
  .x4.b32` into 16 fp32 regs per lane.  Emits `LDTM.16dp256bit.x4` in SASS.
- `STSM_X4_TRANS(SADDR, r0..r3)` — `stmatrix.sync.aligned.x4.trans.m8n8
  .shared.b16`.  Emits `STSM.16.MT88.4` in SASS.
- `#ifndef USE_STMATRIX / #define USE_STMATRIX 0 / #endif` — default-off
  guard; enabled via `-DUSE_STMATRIX=1` make target.
- Under `#if USE_STMATRIX`, the 1×LDTM + 4×STS-per-rh subpass body is
  replaced with 2×LDTM + 4×STSM (32 rows of a 32-col subpass-rh block
  covered as two 16-row bands via TMEM taddr+0x100000).  Bias is added
  per-lane via a conservative lane-c / lane-r index guess, then F2FP
  .PACK_AB pairs adjacent fp32 → bf16x2 and STSM writes to SMEM.

### `Makefile`

Added target `fc2-w3x-stsm` that builds `fc2_w3x.cu` with `-DUSE_STMATRIX=1`.
Also added to the `.PHONY` list entries near top.

## SASS opcode diff

Per-kernel counts from `cuobjdump --dump-sass`.  Baseline = `fc2-w3x`
without any flag.  New = `fc2-w3x-stsm`.  `rank1.sass` included as the
shape target (different kernel with different unroll/scheduling patterns;
absolute counts aren't directly comparable).

| opcode         | baseline fc2-w3x | fc2-w3x-stsm | rank1 (ref) |
|----------------|-----------------:|-------------:|------------:|
| STS (all variants) | 24           | 20           | 1           |
|   STS.128          | 4            | 0            | —           |
|   STS.U16          | 17           | 17           | —           |
|   STS (raw)        | 3            | 3            | —           |
| STSM.16.MT88.4 | 0                | **4**        | 32          |
| LDTM.16dp256bit.x4 | 0            | **2**        | 16          |
| LDTM.32x32b.x32 | 1               | 0            | 0           |
| LDTM (total)   | 1                | 2            | 16          |
| F2FP (total)   | 16               | 16           | 128         |
| R2UR           | 32               | 33           | 3           |
| ELECT          | 10               | 10           | 0           |
| BSSY / BSYNC   | 14 / 14          | 14 / 14      | 79 / 79     |
| UTCQMMA        | 4                | 4            | 8           |
| FFMA2          | 0                | 0            | 256         |
| FADD           | 0                | 0            | 256         |

The STS / STSM flip in the epilogue path is clean: 4 STS.128 → 0, and
4 STSM.16.MT88.4 emitted.  `STS.U16` count is unchanged (bias LDG+STS on
W4 runs once pre-tile-loop and has nothing to do with Lever C).  The 3 raw
`STS` instructions are `mbarrier.init` scaffolds, also unrelated.

Rank-1 emits far more LDTM/STSM because its K-loop is tight-looped and all
24 K-iters worth of tile-wide mainloop+epi show up in-inst.  Ours runs a
tight persistent loop — we emit one iter's worth of each op, which unrolls
to the same dynamic inst count.

## Register + stack

`ptxas -v`:

```
baseline:   Used 64 registers, 224 bytes stack, 0 spill stores/loads
stsm:       Used 54 registers, 224 bytes stack, 0 spill stores/loads
```

Register count **dropped by 10** (fewer live fp32 per subpass-rh under
the new 16+16 split).  Stack frame unchanged at 224 bytes.  No spills.

First-draft (with `bp_lane[8]` array indexed by lane_c) grew stack by 48
bytes and introduced STL.128/LDL.128 spill traffic.  Rewrote bias load
as per-lane `ld.shared.v4.u32` directly into 4 scalars — eliminates the
stack spill and keeps everything in registers.

## What is NOT verified on this CPU VPS

1. **Per-lane register layout after LDTM.16dp256bit.x4**.  The memory note
   (`project_lever_c_bugs_confirmed.md`) claims this variant produces a
   "stmatrix-native" layout — lane t=8c+r supplies row r, cols 8c..8c+7
   for matrix c.  **This is not independently confirmed here**.  The bias
   lookup `bp_lane[]` and the output SMEM addresses both assume this
   mapping.

2. **Bias-per-lane routing**.  We load directly via
     ```
     asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
         : "=r"(bl0), "=r"(bl1), "=r"(bl2), "=r"(bl3)
         : "r"(smem_bias + (prev_n + nc + lane_c * 8) * 2));
     ```
   This assumes lane t=8c+r's fp32 regs cover 8 contiguous cols
   `nc + 8*lane_c .. nc + 8*lane_c + 7`.  Rank-1's SASS hints this is
   ROUGHLY right (FADD pairs R16,R17 both use bias R12 — same col for
   adjacent fp32), but the exact mapping could vary.  All 4 epilogue
   warps run the same lane_c path, so bias SMEM port gets 4-way
   replicated lane-c accesses.

   If first B200 run shows wrong output, expected mode is "rows correct,
   cols swapped" (lane-c decode wrong) OR "cols correct, rows swapped"
   (LDTM actually gives col-contiguous, not row-contiguous, per-lane).
   Diagnose by running w3x with row-varying A as in the
   `project_lever_c_bugs_confirmed.md` back-solve.

3. **STSM destination address**.  We pass
     `out_base + (row_start + lane_r)*64 + lane_c*16`
   to each STSM call.  With `.trans`, the instruction will transpose
   within-matrix before write; the addressing still places matrix `c`
   (cols 8c..8c+7) at SMEM col-offset `16*c` bytes, which is where a
   row-major 32-col bf16 tile expects those cols.  Row ordering within
   the 8-row tile: we assume the `.trans` produces row-major output
   (lane r writes what becomes SMEM row r).  Needs confirmation.

4. **TMA store descriptor**.  Kept as `SWIZZLE_NONE` — no host-side
   change.  This is an intentional deviation from rank-1 (which uses
   SWIZZLE_32B with the XOR-toggle `R11 = R10 ^ 0x20` pattern visible in
   `rank1.sass`).  A linear SMEM layout is valid for stmatrix as long as
   the per-lane STSM address we pass matches the target (which we
   verified mathematically; see code comments).

## What's still to do (on B200)

1. `make fc2-w3x-stsm && ./fc2-w3x-stsm` — does it PASS correctness?
2. If FAIL, run the row-varying A diagnostic (see
   `project_lever_c_bugs_confirmed.md`) to determine which of the
   three layout assumptions is wrong.
3. If PASS, time vs `fc2-w3x` baseline and rank-1 at K=3072.  Expected
   upside 5–10 µs per Grievance 2.

## Explicit failure modes NOT hit (yet)

The four stop conditions from the task spec:

- [OK]  ptxas errors — none (clean compile, 0 spills).
- [OK]  "Wrong SASS emitted" — LDTM.16dp256bit.x4 + STSM.16.MT88.4
        are both present.  PTX → SASS path is good.
- [OK]  TMA store descriptor change required — no, we kept
        `SWIZZLE_NONE` and chose STSM target addresses to land in the
        existing linear layout.  No host-side change.
- [OK]  Deeper existing USE_STMATRIX scaffolding bug — there was no
        existing scaffolding in fc2_w3x.cu (only in fc2_w3.cu, which is
        a different kernel).  Built fresh.

## Commits

One commit, one file (`fc2_w3x.cu`), one Makefile target added.  Commit
lives on branch `worktree-agent-a4cce994`.  NOT merged to master.

## Open risks

The biggest unknown is whether `LDTM.16dp256bit.x4` actually produces a
stmatrix-native register layout in the specific subset we use (subpass-rh
nc-offset = multiples of 32 × 4B).  If it does, the kernel PASSes on first
run and we've landed 5–10 µs.  If it doesn't, the back-solve diagnostic in
the memory doc says we'll see a specific permutation pattern in the output
that identifies exactly which assumption is wrong.

Lever C was tagged as obsolete in the 2026-04-21 update (fc2_w3x already
beats rank-1 by 39 µs without it).  Running it is still worthwhile as a
SASS-shape validation — if correct it gives us the emission pattern rank-1
uses, which helps for future ports (fused-residual, FC1, different K).
