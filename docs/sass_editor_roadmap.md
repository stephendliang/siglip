# SASS Editor Roadmap — Checklist

Improvements needed before the FC2 epilogue STS reorder experiment.
Items ordered by priority: 1-3 are prerequisites for safe reordering,
4-6 unlock further capabilities.

## Status key

- `[ ]` not started
- `[~]` in progress
- `[x]` done

---

## 1. Register def/use parser + dependency checker

`[x]` **Parse operands from SASS into structured register sets (defs, uses)**
`[x]` **Build RAW/WAR/WAW dependency graph for an instruction range**
`[x]` **Block reorder/swap if it violates a data dependency**

### Why this matters

The reorder and swap commands (`sass_edit.py:280-330`) operate purely on
addresses — zero awareness of register dependencies. A reorder that puts a
consumer before its producer silently corrupts results. For the FC2 epilogue
STS interleave (~30 instructions), manual register tracing is doable but
error-prone and must be redone every time the binary changes.

### What exists today

`Instruction` (line 86) stores `mnemonic` and `operands` as raw strings
from cuobjdump. `parse_sass_dump()` (line 152) already extracts per-instruction
`(opcode, operands)` tuples. `apply_sass_xref()` (line 189) maps them onto
binary instructions. The operand strings are never parsed further.

### What to build

1. **Operand parser**: regex-based extraction of register references from
   cuobjdump operand strings. Must handle:
   - GPRs: `R0`-`R255`, pairs (`R92`-`R95` for `.v4`), ranges for `.128`
   - Uniform regs: `UR0`-`UR63`
   - Predicates: `P0`-`P6`, `!P3` (guard predicates)
   - Special: `RZ` (zero reg, ignore), `SRxx` (system regs, read-only)
   - Addressing: `[R107+-0x2000]` — R107 is a use, not a def
   - Immediates/constants: `c[0x0][0x348]`, hex literals — skip

2. **Def/use classification per mnemonic**: most instructions define
   the first operand and use the rest. Exceptions:
   - `STS`/`STG`/`STL`: no register def (store to memory), all operands are uses
   - `LDG`/`LDS`/`LDC`/`LDTM`: first operand (+ vector width) is def, address reg is use
   - `IMAD`/`ISETP`/predicated: predicate output may be a separate def

3. **Dependency graph**: for instruction range [A, B), build edges:
   - RAW: insn X defines Rn, insn Y (Y > X) uses Rn → Y depends on X
   - WAW: insn X defines Rn, insn Y (Y > X) defines Rn → Y must follow X
   - WAR: insn X uses Rn, insn Y (Y > X) defines Rn → Y must follow X
   - Expose as `check_reorder(kernel, start, end, new_order) -> list[violations]`

4. **Integration**: call `check_reorder()` from `CubinEditor.reorder()` and
   `CubinEditor.swap()`. Print violations and refuse unless `--force`.

### Scope control

Don't try to handle every SASS mnemonic perfectly. Start with the ~15
instruction types that appear in the FC2 epilogue (F2FP, HADD2, STS, IMAD,
LDC, LOP3, PRMT, ISETP, etc.). Unknown mnemonics get conservative
treatment: assume they def first operand, use the rest.

---

## 2. Fence/barrier boundary detection

`[x]` **Tag synchronization instructions as reorder barriers**
`[x]` **Refuse reorder across barrier boundaries unless --force**

### Why this matters

`fence.proxy.async.shared::cta` must precede TMA stores that read SMEM
written by `st.shared`. `BAR.SYNC`, `BSYNC`, mbarrier arrive/wait are
synchronization points. Moving instructions across these boundaries
changes program semantics even if register dependencies are satisfied.

### What exists today

Nothing. The tool treats all instructions as freely reorderable within
an address range.

### What to build

1. **Barrier instruction set**: hardcoded list of mnemonics that act as
   reorder barriers:
   - Fences: `MEMBAR`, `FENCE`
   - Barriers: `BAR.SYNC`, `BSYNC`, `WARPSYNC`
   - Mbarrier: `MBAR.*` patterns
   - Control flow: `BRA`, `EXIT`, `RET`, `CALL`, `BREAK`, `CONT`
   - Async: `UTCBAR`

2. **Boundary check in reorder()**: scan the instruction range for barrier
   instructions. If any instruction crosses a barrier boundary in the
   proposed new ordering, warn and refuse.

### Scope

Simple set membership check on mnemonics. No need to understand barrier
semantics (which barriers protect which stores) — just refuse to cross them.
Correct-by-construction: the FC2 epilogue STS region is straight-line code
with no barriers inside it, so this check should be a no-op for the planned
experiment, but catches mistakes in other regions.

---

## 3. Latency table integration + stall validation

`[ ]` **Import calibrated latency values into sass_edit.py**
`[ ]` **After reorder/stall-patch, warn if stall count is below minimum for data dependency**

### Why this matters

`patch_stall()` (line 350) blindly sets stall to whatever you ask. Setting
stall=0 between a producer and its dependent consumer creates a register
hazard — the read happens before the write completes. The calibrated
latency table (STS=32, IADD3=2, PRMT≈0, HADD2=5, F2FP=4, LDG=39) gives
the minimum stall needed between a producer and its first consumer.

### What exists today

Latency data lives in `bench/calib/instruction_db.py` (18 families) and
`tools/sass/sass.cpp` (hardcoded table). Neither is imported by sass_edit.py.

### What to build

1. **Latency dict in sass_edit.py**: mnemonic base → minimum latency cycles.
   Hardcode the known values from calibration. Unknown mnemonics get a
   conservative default (e.g., 4 cycles).

2. **Post-edit stall audit**: after any reorder or stall patch, walk the
   instruction sequence. For each instruction that has a RAW dependency on
   a prior instruction, sum the stall counts in between. If the sum is
   below the producer's latency, print a warning:
   `WARNING: 0x5120 (HADD2) needs 5 cycles after producer 0x5100 (F2FP), but only 2 stall cycles available`

3. **`--audit` flag on reorder/swap/patch/script**: run the stall audit
   and print all violations. Default on; `--no-audit` to suppress.

### Depends on

Item 1 (register def/use parser) — needs the dependency graph to know
which instructions are producer/consumer pairs.

---

## 4. SM100a control word field verification (requires B200)

`[ ]` **Verify barrier write/read/wait_mask bit positions on SM100a**
`[ ]` **Document any divergence from SM89 layout**

### Why this matters

Control word fields beyond stall[3:0] are assumed from SM89. Values like
wr_bar=8,16,19 extracted from fc2.cubin are out of range for SM75's
6-barrier system. If the bit positions have shifted on SM100a, barrier
patches will silently produce wrong behavior.

### What exists today

`decode_ctrl()` / `encode_ctrl()` (lines 44-65) use SM89 positions.
The doc (`docs/sass_binary_editing.md:34-44`) marks all fields except
stall as "unverified on SM100a".

### What to build

1. **Probing kernels**: simple known barrier patterns (LDGSTS sequence,
   mbarrier wait/arrive pair). Compile, extract control words, map
   barrier set/wait instructions to bit positions.

2. **Binary patching probe**: on B200, take a working cubin, patch one
   barrier field at a time (e.g., flip wr_bar bit 5), observe if kernel
   hangs/produces wrong results. This isolates each field.

3. **Updated decode_ctrl()**: once positions are known, update the field
   extraction masks and document the SM100a layout.

### Blocked by

B200 access. Can prepare probing kernels locally, run on B200.

---

## 5. gen-loader completion (requires B200)

`[ ]` **Auto-discover kernel param layout from .nv.info ELF section**
`[ ]` **Generate argument setup matching fc2.cu's main()**

### Why this matters

`cmd_gen_loader()` (line 744) produces a skeleton with explicit TODOs
for kernel arguments and launch config. Can't run a patched cubin without
manually porting fc2.cu's ~50 lines of argument setup (pointers, dimensions,
descriptors). This makes the edit→test cycle slow and fragile.

### What exists today

The loader skeleton handles cuModuleLoad, cuModuleGetFunction, and grid/block
dims. Everything else is a comment saying "TODO: port from fc2.cu".

### What to build

1. **Parse .nv.info**: EIATTR_PARAM_CBANK entries specify param offset and
   size for each kernel argument. pyelftools can read these sections.
   Extract `{param_index: (offset, size, name_hint)}`.

2. **Generate cuParamSetv calls**: one per kernel argument, with correct
   offset and size. For pointer args, generate `CUdeviceptr` declarations.
   For scalar args, generate typed locals.

3. **Mirror fc2.cu's allocation**: this part can't be fully auto-generated
   (needs to know which pointers are inputs vs outputs, tensor shapes, etc.).
   But we can generate a commented template that matches the param layout,
   so the user only needs to fill in cudaMalloc calls, not guess offsets.

### Alternative approach

Instead of gen-loader, patch the cubin inline in the compiled fc2 binary's
`.nv_fatbin` section. This avoids the loader entirely — the existing main()
just works. Harder (must find/replace the embedded cubin within the fatbin
container) but more powerful. Could be a separate subcommand.

---

## 6. Register pressure estimation

`[ ]` **Compute live register intervals from def/use graph**
`[ ]` **Warn if a proposed reorder increases max live registers**

### Why this matters

FC2 uses 207 registers (max 256). A reorder that extends a live range
(e.g., moving a producer earlier, pushing its last consumer later)
increases peak register pressure. If it crosses 256, the kernel can't
launch.

### What exists today

Nothing. Register pressure is only visible post-compile via
`cuobjdump --dump-resource-usage`.

### What to build

1. **Liveness intervals**: from the def/use graph (item 1), compute
   [first_def, last_use] for each physical register in the instruction range.

2. **Peak pressure at each program point**: count overlapping live intervals.

3. **Delta report**: before and after proposed reorder, show max live regs
   and which registers grew. This is approximate (doesn't account for
   ptxas's register allocation outside the patched region) but catches
   obvious blowups.

### Depends on

Item 1 (register def/use parser).

---

## Order of attack

```
1. Register def/use parser     ← prerequisite for safe reordering
2. Fence/barrier boundaries    ← quick safety net, low effort
3. Latency table integration   ← depends on 1, validates stall patches
   ─── at this point, FC2 epilogue STS reorder is safe to attempt ───
4. Control word verification   ← B200 required, unlocks barrier edits
5. gen-loader / fatbin patcher ← B200 required, enables running patched cubins
6. Register pressure           ← depends on 1, nice-to-have
```

Items 1-3 are local work (no GPU). Items 4-5 need B200 access.
Item 6 is a nice-to-have that depends on 1.
