# R2UR: The Uniform Register Problem

## What This Document Covers

Analysis of the R2UR (Register to Uniform Register) overhead in our SM100a persistent GEMM kernels. Explains what uniform registers are, why TMA requires them, how CUTLASS achieves ~21 R2UR per kernel vs our ~388, what we've done to reduce R2UR, what worked, what didn't, and what the remaining gap means.

## Current State (post-commit a70b4e2)

Three changes reduced R2UR by 21-34% in the W0 (load warp) scope:

| Change | Mechanism |
|--------|-----------|
| `__cvta_generic_to_shared` intrinsic | Replaced `asm volatile` in `smem_to_uint()` — compiler can track uniformity through the intrinsic |
| `warp_uniform()` on W0 `smem_base` | `__shfl_sync(0xFFFFFFFF, x, 0)` tells the compiler the value is warp-uniform |
| Batched TMA asm blocks in W0 | Multiple TMA instructions in one `asm volatile` — compiler reuses UR regs for shared operands |

### Measured results

**FC2** (N_STAGES=5, TMAR=1, W0_RES_PREFETCH=1, BIAS_SMEM=1 — apples-to-apples config):

| Metric | Before (6ebc5cd) | After (a70b4e2) | Delta |
|--------|:-----------------:|:----------------:|:-----:|
| R2UR | 492 | 323 | -34% |
| ELECT | 93 | 93 | 0 |
| SHFL | 0 | 1 | +1 |
| Regs | 230 | 184 | -46 |

**FC2** production (N_STAGES=5, TMAR=1, W0_RES_FULL=1, BIAS_SMEM=1 — new config, no pre-commit equivalent):
388 R2UR, 97 ELECT, 1 SHFL, 186 regs.

**FC1** (N_STAGES=5, GV=4, PH1U=4, IS=1, KLU=5, ST=1, SMU=3):

| Metric | Before | After | Delta |
|--------|:------:|:-----:|:-----:|
| R2UR | 387 | 316 | -18% |
| ELECT | 45 | 45 | 0 |
| SHFL | 0 | 1 | +1 |
| Regs | 248 | 238 | -10 |

**CUTLASS FC2** (per-kernel, all variants): 21-36 R2UR, 10 ELECT, 7 SHFL.

### What this means

We cut R2UR by ~170 in FC2 (492→323), but CUTLASS is at 21. The remaining **323 R2UR is ~15x CUTLASS**. The low-hanging fruit is gone. The remaining R2UR comes from structural issues that can't be fixed incrementally.

## Part 1: The Hardware — R Registers vs UR Registers

### The two register files

Blackwell (SM100a) has two distinct 32-bit register files per warp:

| Property | R (Regular) | UR (Uniform) |
|----------|:-----------:|:------------:|
| Count per warp | 256 | ~64 |
| Scope | Per-thread (32 copies per warp) | Per-warp (1 copy, shared by all 32 lanes) |
| Arithmetic | IADD3, IMAD, FMUL, FFMA, ... | UIADD3, UIMAD, UMOV, USHF, ULOP3 |
| Constant memory load | LDC (→ R) | LDCU (→ UR) |
| Special register read | S2R (→ R) | S2UR (→ UR) |
| Cross-file transfer | — | R2UR (R → UR), UR2R (UR → R) |

**R registers** hold per-thread data: thread indices, per-element accumulators, addresses that differ across lanes. Each R register is really 32 independent values (one per thread in the warp).

**UR registers** hold warp-uniform data: values that are identical across all 32 lanes. Block indices, tile coordinates, SMEM base addresses, TMA descriptors, mbarrier addresses — anything derived solely from `blockIdx`, kernel parameters, or compile-time constants.

### R2UR: the cross-file transfer

`R2UR URx, Ry` copies the value from thread 0's R register into the UR register. It exists because sometimes a value that is logically uniform (same across all lanes) has been computed through per-thread arithmetic and ended up in an R register. The hardware must transfer it to UR space for instructions that require UR operands.

R2UR is not free:
- **~8 cycles effective cost** (calibrated on B200 via microbenchmarks)
- **Serializes the warp**: all 32 threads must converge before the transfer (only lane 0's value is meaningful)
- **Generates ELECT + PLOP3 + BRA overhead**: the compiler must elect lane 0 and guard the transfer
- **Cannot be pipelined**: unlike FMUL or FFMA, R2UR has a hard data dependency — the UR value isn't available until the transfer completes

There is a broadcast variant `R2UR.BROADCAST URx, Ry` that copies from a specific lane (used for TMA descriptors), but it has the same cost.

### Why TMA requires UR operands

The TMA (Tensor Memory Accelerator) unit on SM100a is a hardware DMA engine that moves tiles between global memory and shared memory. The key TMA instruction is:

```
UTMALDG.3D [smem_dst], [coords], desc[tma_desc]
```

Every operand of UTMALDG is a UR register:
- `smem_dst` (UR): SMEM destination address
- `coords` (UR): tensor coordinates (x, y, z)
- `tma_desc` (UR): TMA descriptor pointer
- The mbarrier address (UR) is embedded in the smem_dst encoding

This makes architectural sense: TMA is a warp-level operation. Only one copy of the coordinates is needed — all 32 lanes issue the same TMA load. The hardware enforces UR operands to guarantee this warp-level uniformity.

Similarly, `UTMASTG` (TMA store), mbarrier operations (`mbarrier.arrive`, `mbarrier.init`), and `ELECT` all operate in UR space. Any value that feeds these instructions must be in a UR register before use.

## Part 2: Why Our Code Generates R2UR

### The root cause: `if (lane == 0)` divergent branching

```c
if (lane == 0) {   // lane = threadIdx.x % 32
    tma_load_2d(smem_addr, &tma_a, k_start, m_start, tma_mbar_s);
}
```

`lane == 0` tests `threadIdx.x`, which is inherently non-uniform. The `if` creates a **divergent branch** in the compiler's control flow graph. Inside the divergent branch, the compiler treats all values as potentially non-uniform — even though `smem_addr`, `k_start`, `m_start`, and `tma_mbar_s` are all actually uniform.

This is the dominant remaining uniformity break. Every `if (lane == 0)` in our code (40+ of them) creates a scope where the compiler cannot prove uniformity, forcing R allocation for enclosed values.

CUTLASS avoids this. It uses `@!UP0` — a **uniform predicate** from `elect_one_sync()` — to guard TMA operations. A uniform predicate does not create a divergent branch. Values used under `@!UP0` retain their uniformity status.

**We tested the ELECT approach** (replacing `if (lane == 0)` with `elect.sync` + `@P0` predication) and it **increased** R2UR. The reason: `if (lane == 0)` keeps TMA operands in R space — no UR transfer needed because the compiler just uses R. ELECT predication forces the compiler to allocate UR for every operand (since the predicated instruction requires UR), triggering R2UR. Each separate elect-predicated helper creates its own asm boundary, preventing UR reuse.

The ELECT approach only works if the operands are ALREADY in UR — which requires all the other uniformity fixes. CUTLASS gets this because CuTe's code structure keeps values in UR from the start. For us, ELECT is the last step, not the first.

### Secondary breaks (partially fixed)

**`smem_to_uint()` inline asm** — originally used `asm volatile` with `"=r"` output, which broke uniformity tracking. **Fixed**: replaced with `__cvta_generic_to_shared()` intrinsic that the compiler can analyze.

**Per-iteration array lookups** — `smem_a[s]` and `tma_mbar[s]` indexed by loop variable inside divergent `if (lane == 0)`. **Partially fixed**: W0 K-loop now uses integer offsets from `smem_base` (`smem_base + s * STAGE_BYTES`) instead of array lookups.

### Per UTMALDG overhead pattern

Each UTMALDG in our kernel generates this preamble:

```
R2UR UR4, R154           ; coord_x:  R → UR (~8 cycles)
R2UR UR5, R155           ; coord_y:  R → UR (~8 cycles)
R2UR UR9, R111           ; mbar:     R → UR (~8 cycles)
R2UR UR8, R62            ; smem_dst: R → UR (~8 cycles)
PLOP3.LUT P0, ...
@P0 ELECT P4, URZ, PT
@P4 R2UR.BROADCAST UR11, R8  ; tma_desc: R → UR (~8 cycles)
UTMALDG.2D.2CTA [UR8], [UR4]
```

5 R2UR + ELECT overhead per UTMALDG. At ~8 cycles per R2UR: ~40-50 cycles before each TMA load starts.

CUTLASS:

```
UIADD3 UR8, ..., UR4, 0x180, URZ       ; advance smem ptr (2 cycles)
UIADD3 UR32, ..., UR32, 0x10800, URZ   ; advance coord    (2 cycles)
@!UP0 UTMALDG.3D [UR32], [UR30], desc[UR18]  ; all UR, 0 R2UR
```

2 UIADD3 + 0 R2UR = 2 instructions overhead per UTMALDG.

## Part 3: What We Did — The Three Fixes

### Fix 1: `__cvta_generic_to_shared` intrinsic (committed dfa3e78, refined a70b4e2)

**Before:**
```c
uint32_t smem_to_uint(const void* p) {
    uint32_t r;
    asm volatile("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }"
        : "=r"(r) : "l"(p));
    return r;
}
```

**After:**
```c
uint32_t smem_to_uint(const void* p) {
    return static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(__cvta_generic_to_shared(p)));
}
```

The `asm volatile` with `"=r"` output killed uniformity tracking. The compiler intrinsic `__cvta_generic_to_shared()` preserves it — the compiler knows the output has the same uniformity as the input.

### Fix 2: `warp_uniform()` on W0 smem_base (a70b4e2)

```c
template<typename T>
static __device__ __forceinline__ T warp_uniform(T x) {
    return __shfl_sync(0xFFFFFFFF, x, 0);
}

const uint32_t smem_base = warp_uniform(smem_to_uint(smem));
```

`__shfl_sync(0xFFFFFFFF, x, 0)` tells the compiler the result is warp-uniform. All addresses derived from `smem_base` (TMA load destinations, mbarrier addresses) inherit this uniformity.

Applied only to W0 (load warp) scope. Applying to epilogue warps **increased** R2UR (+38) and register count (+4) — the compiler compensated for the extra SHFL instructions by generating more R2UR elsewhere.

### Fix 3: Batched TMA asm blocks in W0 (a70b4e2)

**Before** — 3 separate `asm volatile` blocks:
```c
if (lane == 0) {
    tma_load_2d(smem_base + s * STAGE_BYTES, &tma_a,
                k_start, m_start, tma_mbar_s);
    tma_load_2d(smem_base + s * STAGE_BYTES + 16384, &tma_b,
                k_start, n_start + cta_rank * (TN/2), tma_mbar_s);
    mbar_arrive_expect_tx(tma_mbar_s, TMA_BYTES);
}
```

**After** — single asm block:
```c
if (lane == 0) {
    const uint32_t a_dst = smem_base + s * STAGE_BYTES;
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3}], [%4];\n\t"
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%5], [%6, {%2, %7}], [%4];\n\t"
        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%4], %8;"
        :: "r"(a_dst), "l"(&tma_a), "r"(k_start), "r"(m_start),
           "r"(tma_mbar_s), "r"(a_dst + 16384), "l"(&tma_b),
           "r"(n_start + cta_rank * (TN/2)), "r"(TMA_BYTES)
        : "memory");
}
```

Shared operands `%2` (k_start) and `%4` (tma_mbar_s) are used by all 3 instructions. The compiler reuses UR registers for them instead of doing separate R2UR transfers at each asm boundary. Applied to W0 K-loop, W0_RES_FULL, and W0_RES_PREFETCH sections.

## Part 4: What Didn't Work

### ELECT predication (tested, reverted)

Replaced `if (lane == 0)` with `elect.sync @P0` predication. **Increased R2UR** because:
- `if (lane == 0)` keeps operands in R — no UR transfer needed (the compiler just uses R for everything)
- `elect.sync` `@P0` forces the compiler to allocate UR for every TMA operand → triggers R2UR for each one
- Each separate elect-predicated helper creates its own asm boundary, preventing UR reuse across calls

ELECT only wins if operands are already in UR. It's the last step of a full fix, not a standalone improvement.

### `warp_uniform()` on epilogue warps (tested, reverted)

Added `epi_smem_base = warp_uniform(smem_to_uint(smem))` in the epilogue warp section and replaced `smem_to_uint` calls with `epi_smem_base + offset`.

**Result:** R2UR increased from 388 → 426 (+10%), regs from 186 → 190 (+4). The extra SHFL instructions and the additional live `epi_smem_base` register increased register pressure, causing the compiler to spill more values through R2UR.

The epilogue warps are already at high register pressure (the GELU/residual compute fills the register file). Adding another long-lived uniform value tipped the balance.

### Batched TMA stores in epilogue (tested, reverted alongside epilogue warp_uniform)

Batched pairs of IS=2 TMA stores into single asm blocks. Combined with the epilogue `warp_uniform` change, the net effect was negative. Not tested in isolation.

## Part 5: Why CUTLASS Is at 21 R2UR

CUTLASS achieves ~21 R2UR per kernel (vs our 323-388) through structural differences, not compiler hints:

**1. No `if (lane == 0)` anywhere.** CUTLASS uses uniform warp-group role checks (`warp_group_role == Producer`) which the compiler recognizes as uniform branches. TMA issuance uses `elect_one_sync()` — a uniform predicate that doesn't break uniformity for enclosed values.

**2. CuTe coordinate iterators keep values in UR from the start.** Coordinates are `ArithmeticTuple<int32_t>` objects initialized from `Int<0>{}` (compile-time constant) and advanced via `operator++` (constant stride addition). The compiler proves uniformity at every step → UIADD3 stays in UR.

**3. SMEM addresses as integer arithmetic from a UR base.** CuTe computes `base_smem + stage * stage_stride` where `base_smem` enters UR at kernel start and `stage_stride` is constexpr. Emits UIADD3 (UR→UR). Our code goes through `smem_to_uint()` which, even with the intrinsic fix, still passes through a divergent `if (lane == 0)` scope.

**4. TMA descriptors loaded via LDCU.** CUTLASS gets `LDCU.64 UR8, c[0x0][0x890]` — direct constant memory → UR. Our `&tma_a` goes through a pointer cast in R space, requiring `R2UR.BROADCAST`.

**5. Fewer total TMA instructions.** CUTLASS doesn't have our overlapped epilogue with its heavy TMA store interleaving. Simpler epilogue = fewer TMA callsites = fewer R2UR amplification points.

The 21 remaining CUTLASS R2UR likely come from epilogue setup (bias pointer addresses, output tensor coordinates) where CuTe's abstraction doesn't fully preserve uniformity.

## Part 6: The Remaining Gap and What It Would Take

### R2UR distribution (FC2 current, 388 total)

The R2UR in our kernel falls in three regions:
- **W0 (load warp):** ~130 — reduced from ~190 by our fixes. Still high because `if (lane == 0)` blocks all uniformity.
- **Epilogue (W2-W5):** ~170 — untouched. The `warp_uniform` approach failed here due to register pressure.
- **Drain:** ~88 — untouched.

### What would close the gap

To reach CUTLASS-level R2UR (~21), ALL of these would need to happen together:

1. **Replace all `if (lane == 0)` with uniform ELECT** — but this only works if operands are already in UR (chicken-and-egg with fix 2-4)
2. **Make all SMEM address derivations stay in UR** — requires computing addresses outside divergent scope, which conflicts with the `if (lane == 0)` warp-specialization pattern
3. **In-place coordinate advancement via UIADD3** — replace per-iteration `ki * TK` recomputation with `cur_k += TK`. Only useful if `cur_k` is in UR.
4. **TMA descriptors via LDCU** — requires the compiler to recognize `__grid_constant__` parameters as constant-memory-resident and use LDCU instead of LDC + R2UR.BROADCAST

These are all interdependent. Each alone does nothing (or makes things worse, as we demonstrated with ELECT). They only work as a package, which is essentially a rewrite of the TMA coordination layer.

### The fundamental architectural mismatch

Our warp-specialization pattern (`if (warp == 0)` for loads, `if (warp == 1)` for MMA, `if (warp >= 2)` for epilogue) is structurally incompatible with the compiler's uniformity analysis. The `if (lane == 0)` inside each warp's scope is a hard wall for uniformity tracking.

CUTLASS avoids this by using warp-group-level role assignment (uniform) and `elect_one_sync()` (uniform predicate) rather than lane-level branching. Their code structure was designed from the start to preserve compiler uniformity tracking.

We would need to restructure our entire TMA coordination layer — replacing lane-level guards with uniform predicates, pre-computing all addresses outside divergent scope, and batching all TMA operations into single asm blocks where the compiler can share UR registers. This is roughly equivalent to building our own CuTe-style TMA abstraction.

### Practical significance

The 323-388 R2UR costs ~2600-3100 cycles/tile at ~8 cycles each. For FC2 at 1.47ms across 3626×3=10878 tiles on 74 clusters, that's ~0.26ms or ~18% of kernel time.

However, R2UR is NOT proven to be the dominant source of FC2's 20% gap to CUTLASS. The gap could equally be explained by:
- CUTLASS's simpler epilogue (no overlapped interleaving) having better memory access patterns
- Different pipeline depth choices
- Register pressure differences enabling different occupancy

R2UR reduction is worth pursuing for register savings (-46 regs in FC2, -10 in FC1) and instruction count reduction, but it should not be assumed to be the silver bullet for the CUTLASS gap.

## Summary

| Question | Answer |
|----------|--------|
| What is R2UR? | Transfer from per-thread R register to warp-uniform UR register. ~8 cycles, serializes warp. |
| Why does TMA need UR? | TMA is a warp-level operation — coordinates must be uniform. Hardware enforces UR operands. |
| How much R2UR do we have? | FC2: 388 (production), FC1: 316. Down from 492/387 pre-fix. |
| How much does CUTLASS have? | 21-36 per kernel. ~15x less than us. |
| What reduced R2UR? | `__cvta_generic_to_shared` intrinsic, `warp_uniform()` on W0 smem_base, batched TMA asm blocks. All in W0 scope only. |
| What didn't work? | ELECT predication (increased R2UR), epilogue `warp_uniform` (increased R2UR and regs), batched epilogue TMA stores (net negative with epilogue changes). |
| Why is CUTLASS so much lower? | Structural: no divergent branches for TMA, CuTe preserves uniformity through coordinate iterators, LDCU for descriptors, simpler epilogue. |
| Can we match CUTLASS? | Would require rewriting the TMA coordination layer to eliminate all `if (lane == 0)` divergent branches and keep all addresses in UR from the start. Interdependent changes that only work as a package. |
| Is R2UR the FC2 performance gap? | Unproven. The ~0.26ms R2UR overhead is plausible but the gap could also be memory access patterns or pipeline choices. |
