# SASS Post-Compile Optimizer — Design Proposal

Post-compile SASS optimizer for SM100a (Blackwell). Operates on cubin ELF binaries:
patches instruction control words (stall counts, barrier masks) and optionally
reorders instructions within basic blocks to achieve globally-optimal scheduling.

## Why this exists

ptxas is a good-enough scheduler but not optimal. On the patch embed K-loop:
- 24 UTCQMMA instructions, each with stall=10, min=7 (slack=3)
- 8 SYNCS.PHASECHK instructions with stall=15, min=0 (slack=15)
- Total: 238 cycles of static scheduling slack in 417 instructions
- The slack is probably absorbed by the producer-consumer equilibrium today,
  but on a different kernel or problem shape it may not be

The tool is general-purpose: any SM100a cubin, any kernel, not just this one.

## Architecture

```
                 sass_analysis.py (existing)
                         |
                   parsed SASS + deps
                         |
                   +-----------+
                   | Optimizer |  ← new: sass_optimizer.py
                   +-----------+
                   /           \
           Phase 1              Phase 2
         (restall)            (reorder)
              |                    |
        patch stalls         CP solver
        in-place               output
              |                    |
              +--------+-----------+
                       |
                 cubin patcher
                       |
                 patched cubin
                       |
                 cuModuleLoad → benchmark
```

---

## Phase 1: Restaller

Patches stall counts only. No instruction reordering. Safe, minimal, validates
the approach.

### 1.1 Cubin ELF parser

- [ ] Extract cubin from fatbin (`cuobjdump --dump-cubin` or parse fatbin header)
- [ ] Parse ELF headers to find `.text.<kernel>` section (offset + size)
  - patch_embed: section index 0xe, offset 0x1700, size 0x8a80 (2216 instructions)
  - each instruction = 16 bytes (128 bits)
- [ ] Verify: instruction count matches sass_analysis.py output
- [ ] Map SASS addresses to ELF byte offsets: `elf_offset = section_offset + addr`

### 1.2 Control word bit layout

Known from calibration (K1-K12, verified on B200):

```
Instruction encoding (128 bits = 16 bytes):
  Bytes [7:0]   = 64-bit instruction word (opcode + operands)
  Bytes [15:8]  = 64-bit control word

Control word fields (from CTRL_LAYOUT in sass_analysis.py):
  bits [3:0]    stall count (0-15 cycles before issue)
  bit  [4]      yield hint (suggest warp switch)
  bits [9:5]    write barrier index (0-5, 31=none)
  bits [14:10]  read barrier index (0-5, 31=none)
  bits [20:15]  wait barrier mask (6 barriers, bitmask)
  bits [22:21]  register reuse flags
```

**CRITICAL**: Verify bit layout on each new toolkit version. The layout is NOT
documented by NVIDIA and may change between CUDA toolkit releases. The
calibration pipeline (calibration.cu + `--calibrate-compare`) exists for this.

- [ ] Read control word from bytes [15:8] of each instruction (little-endian u64)
- [ ] Extract stall = ctrl & 0xF
- [ ] Verify extracted stalls match cuobjdump output (cross-check all 2216 instructions)

### 1.3 Stall patching

- [ ] For each instruction in the target section:
  - Read current stall from control word
  - Look up min stall from sass_analysis.py dependency analysis
  - If current > min: patch to min (or to solver-specified value)
  - Compute new control word: `(ctrl & ~0xF) | new_stall`
  - Write back bytes [15:8]
- [ ] Preserve all other control word bits (yield, barriers, reuse) unchanged
- [ ] Checksum the patched section (detect accidental corruption)

### 1.4 Cubin writer

- [ ] Write patched cubin back to file
- [ ] ELF section headers, symbol tables, relocations: do NOT modify (they're
  position-independent for in-section patches)
- [ ] If fatbin wrapper present: either patch in-place within the fatbin, or
  write standalone cubin + load with cuModuleLoad

### 1.5 Loader + benchmarker

- [ ] Load patched cubin via CUDA driver API:
  ```c
  cuModuleLoad(&module, "patched.cubin");
  cuModuleGetFunction(&func, module, "patch_embed_gemm");
  cuLaunchKernel(func, ...);
  ```
- [ ] OR: replace the cubin inside the fatbin and run the original executable
- [ ] Compare wall clock: original vs patched (median of 100 runs, same methodology
  as grid_search.py)
- [ ] Validate correctness: run non-uniform validation, compare checksum

### 1.6 CLI interface

```bash
# Analyze slack (existing)
python3 sass_analysis.py --cubin siglip_best --deps

# Generate patch (new)
python3 sass_optimizer.py siglip_best --restall --section 0x4300 0x5d00 -o patched.cubin

# Full kernel restall
python3 sass_optimizer.py siglip_best --restall -o patched.cubin

# Dry run (show what would change)
python3 sass_optimizer.py siglip_best --restall --dry-run
```

### 1.7 Validation checklist

- [ ] Patched cubin loads without CUDA errors
- [ ] Kernel launches without illegal instruction / misaligned access
- [ ] Non-uniform validation passes (checksum match)
- [ ] Wall clock measured and compared to baseline
- [ ] Register the delta: if < 0.1% improvement, equilibrium hypothesis confirmed
  for the K-loop (but tool is still valid for other kernels/sections)

---

## Phase 2: Reorderer

Reorders instructions within basic blocks for globally-optimal scheduling.
Requires Phase 1 infrastructure (cubin parser, patcher, loader).

### 2.1 Basic block extraction

- [ ] Identify basic block boundaries in SASS:
  - **Leaders**: first instruction, branch targets, instruction after branch/exit
  - **Terminators**: BRA, EXIT, RET, BSSY, BSYNC
  - Control flow graph edges from branch targets
- [ ] Extract basic blocks as instruction lists
- [ ] For the K-loop: 6 unrolled iterations, each containing:
  - Mbarrier wait (SYNCS.PHASECHK + BRA retry loop) — block boundary
  - Descriptor setup (R2UR x6 + PLOP3 + ELECT) — reorderable
  - MMA issue (UTCQMMA + PLOP3 + BRA.U.ANY retry) — block boundary
  - TMA barrier (UTCBAR + ELECT loop) — block boundary
- [ ] The reorderable blocks are small (6-13 instructions). Most gain comes from
  cross-block motion (hoisting descriptor setup into previous block's stall window).
  This requires special handling.

### 2.2 Dependency graph builder

- [ ] Build per-basic-block dependency DAG:
  - **RAW** (read-after-write): instruction B reads register that A wrote
    - Latency = A's issue latency from calibrated LATENCY table
  - **WAR** (write-after-read): instruction B writes register that A read
    - Latency = 0 (B can issue same cycle A reads, but must not issue before A)
  - **WAW** (write-after-write): instruction B writes same register as A
    - Latency = 0 (ordering constraint only)
  - **Barrier deps**: if A sets barrier K and B waits on barrier K, B depends on A
  - **Memory deps**: STS→STS to same address = WAW, LDS→STS = WAR, STS→LDS = RAW
    - Conservative: treat all STS/LDS to unknown addresses as dependent
    - Precise: parse address expressions to identify independent accesses
  - **Control deps**: instructions guarded by predicates depend on the predicate producer
- [ ] Export as JSON for solver input:
  ```json
  {
    "instructions": [
      {"id": 0, "addr": "0x44f0", "opcode": "R2UR", "latency": 8,
       "reads": ["R178"], "writes": ["UR4"], "barrier_wait": [], "barrier_set": []},
      ...
    ],
    "edges": [
      {"from": 0, "to": 5, "type": "RAW", "latency": 8, "reg": "UR4"},
      ...
    ],
    "functional_units": {
      "alu_int": {"throughput": 1},
      "special": {"throughput": 1},
      "tma": {"throughput": 1},
      "sts": {"throughput": 1, "issue_latency": 32}
    }
  }
  ```

### 2.3 Constraint solver integration

- [ ] Define solver interface (input: dependency DAG + resource constraints, output:
  schedule = instruction order + stall assignments)
- [ ] Resource constraints:
  - **Issue width**: 1 instruction/cycle per warp (SM100a is single-issue per warp)
  - **Functional unit throughput**: most ALU ops = 1/cycle, STS = 1/32 cycles (but
    this is handled by stall counts, not issue restrictions)
  - **Barrier budget**: 6 scoreboard barriers max (wr_bar IDs 0-5)
  - **Register liveness**: if reordering changes register lifetime, verify no spills
    (207/256 regs used — 49 free, tight but workable)
- [ ] Objective: minimize total cycles = sum of (stall + 1) for all instructions,
  subject to all dependency edges being satisfied
- [ ] Solver options:
  - **Your existing CP solver** (adapted from AVX-512): port the constraint model,
    replace x86 latency table with SASS latency table, replace functional units
  - **Simple list scheduler** (baseline): topological sort by critical path length,
    greedy assignment. Fast, gets ~90% of optimal for in-order issue.
  - **ILP solver** (if blocks are small enough): exact optimum via integer linear
    programming. Blocks are 6-13 instructions — trivially solvable.

### 2.4 Schedule application

- [ ] Given solver output (new instruction order + stall assignments):
  - Reorder 16-byte instruction words in the ELF section
  - Assign new stall counts (from solver)
  - Update wait barrier masks if barrier assignments changed
  - **Branch offset fixup**: for any BRA within or targeting the reordered block,
    recompute PC-relative offset
    - Branch offset = (target_addr - branch_addr) encoded in instruction word
    - Must decode the BRA encoding to find the offset field, then re-encode
  - **Barrier reassignment**: if the solver changes which barrier ID an instruction
    uses (to avoid conflicts), update both the wr_bar field on the producer and
    the wait_mask bit on the consumer(s)
- [ ] Cross-block motion (advanced):
  - Move instructions from block B into the stall window of block A's terminator
  - Example: hoist R2UR descriptor setup from MMA block N+1 into the tail of
    MMA block N (after the UTCQMMA, during the 10-cycle stall before BRA.U.ANY)
  - This requires: no dependency on block N's MMA result, no WAR conflict with
    block N's live registers, destination regs not clobbered between blocks
  - Very high reward: the 3-cycle slack per MMA comes from exactly this gap

### 2.5 Validation

- [ ] Round-trip test: reorder then un-reorder, verify identical cubin
- [ ] Equivalence check: dependency graph of reordered schedule must be a valid
  topological sort of the original dependency graph
- [ ] Functional test: run patched kernel, verify non-uniform checksum
- [ ] Performance test: median wall clock over 100 runs

---

## Phase 3: Cross-block motion + global optimizer (future)

### 3.1 Global instruction motion

- [ ] Extend basic block analysis to superblocks (trace-based) or entire loop bodies
- [ ] Software pipelining: overlap iteration N's epilogue with iteration N+1's prologue
  - In the K-loop: overlap MMA group N's descriptor setup with MMA group N-1's
    completion stall
  - Requires renaming: if MMA N-1 uses UR4-UR9 and MMA N also needs UR4-UR9,
    you need alternate register sets or careful lifetime analysis
- [ ] Speculative motion: move instructions above branches when safe (predicate
  analysis, no side effects)

### 3.2 Epilogue optimizer

- [ ] Apply same methodology to epilogue Phase 1 (LDTM + CVT + ADD + STS)
- [ ] STS = 32 cycles/issue. Between STS issues, there are ~30 cycles of dead time.
  The current code fills this with TMEM loads (which stall on scoreboard) and
  CVT/ADD (which chain to the next STS). Reordering could interleave more useful
  work into STS stall windows.
- [ ] Profile epilogue section separately: `--section 0x6000 0x8000` (approximate —
  find exact LDTM-to-STS region from sass_analysis.py output)

### 3.3 Whole-kernel profile-guided optimization

- [ ] Integrate ncu source counters: use runtime stall data (not just static analysis)
  to identify which stalls are actually hit at runtime
- [ ] Feed runtime stall distribution into the solver as weights: prioritize reducing
  stalls that are actually observed, not just statically possible
- [ ] Iterative: patch → profile → re-analyze → patch again

---

## Implementation order

```
Phase 1.1-1.2  Cubin parser + control word extraction     ██░░░░  ~150 lines
Phase 1.3-1.4  Stall patcher + writer                     ██░░░░  ~100 lines
Phase 1.5      Loader (cuModuleLoad wrapper)               █░░░░░  ~80 lines (C)
Phase 1.6-1.7  CLI + validation                            █░░░░░  ~50 lines
------- validate Phase 1 on patch embed K-loop, measure delta -------
Phase 2.1      Basic block extraction                      ██░░░░  ~200 lines
Phase 2.2      Dependency graph builder (extend existing)  ███░░░  ~300 lines
Phase 2.3      Solver integration (adapt CP solver)        ████░░  variable
Phase 2.4      Schedule application + branch fixup         ███░░░  ~250 lines
Phase 2.5      Validation suite                            ██░░░░  ~150 lines
------- validate Phase 2, measure on K-loop + epilogue -------
Phase 3        Cross-block + profile-guided                ██████  research
```

Phase 1 is the critical path. If restalling shows zero wall-clock improvement on
the patch embed kernel (confirming the equilibrium hypothesis), the tool is still valuable:
apply it to the epilogue section, or to other kernels where the equilibrium is
different.

---

## Key risks and mitigations

**Risk: Control word bit layout changes between CUDA versions.**
Mitigation: calibration.cu + `--calibrate-compare` mode. Run on every toolkit update.
The calibration kernels are self-checking: if bits move, the runtime measurements
will disagree with the decoder predictions.

**Risk: Patched cubin fails to load (ELF integrity check).**
Mitigation: NVIDIA's cubin loader does minimal validation (no code signing as of
CUDA 13.1). The `.text` section is raw instruction bytes. As long as ELF headers
are intact and section sizes match, it loads. Test: patch a NOP's stall count and
verify it loads.

**Risk: cubin has a CAPMERC (capability mercury) section that depends on instruction layout.**
Mitigation: The `.nv.capmerc.text.<kernel>` section exists in the ELF (size 0x2c8e
for the patch embed kernel). This is an NVIDIA-internal performance hint table. If
reordering corrupts it, the kernel may still run but lose hardware-level optimizations.
Phase 1 (restall only) does not reorder instructions, so CAPMERC is safe. Phase 2
may need to zero out or regenerate this section. Test: zero the section and measure
if performance changes.

**Risk: Barrier wait mask bits are wrong in the decoder.**
Mitigation: The `[bar 0bXXXXXX]` annotation in cuobjdump output provides ground
truth. Cross-check decoded wait_mask against cuobjdump annotations for all 2216
instructions. Any mismatch means the bit layout is wrong for that field. Current
calibration confirms bits [3:0] (stall) are correct; barrier fields are assumed
from Ampere but not independently verified.

**Risk: Reordering around SYNCS/UTCQMMA/UTCBAR has hidden ordering requirements.**
Mitigation: Mark all synchronization-related instructions (SYNCS, ELECT, PLOP3,
UTCBAR, UTCQMMA, R2UR.BROADCAST) as having implicit ordering constraints. The
solver must not reorder within the MMA issue sequence (ELECT → BROADCAST → UTCQMMA)
or the barrier sequence (ELECT → UTCBAR). Only the descriptor setup (R2UR chain)
is safe to reorder relative to other independent code.

---

## Files

| File | What |
|------|------|
| `sass_optimizer.py` | Main optimizer: CLI, cubin I/O, restaller, reorderer |
| `sass_analysis.py` | Existing: parser, deps, slack analysis (extended for solver export) |
| `cubin_loader.cu` | Minimal C program: loads cubin via cuModuleLoad, launches kernel, validates |
| `calibration.cu` | Existing: control word verification microbenchmarks |
| `proposals/sass_optimizer.md` | This document |

---

## Cubin byte layout reference

```
Fatbin container (may contain multiple cubins)
  └─ Cubin (ELF format, ET_EXEC)
       ├─ .shstrtab (section name strings)
       ├─ .strtab (symbol name strings)
       ├─ .symtab (symbols)
       ├─ .note.nv.tkinfo (toolkit version)
       ├─ .note.nv.cuinfo (CUDA version)
       ├─ .nv.info.<kernel> (kernel metadata: regs, smem, barriers)
       ├─ .text.<kernel> ← THIS IS WHAT WE PATCH
       │    ├─ Instruction 0:  bytes [0:15]  (encoding[7:0] + control[15:8])
       │    ├─ Instruction 1:  bytes [16:31]
       │    └─ ...
       ├─ .rela.text.<kernel> (relocations — do NOT modify)
       └─ .nv.capmerc.text.<kernel> (performance hints — leave alone for Phase 1)

Megakernel .text section:
  Offset in ELF:  0x1700
  Size:           0x8a80 (35,456 bytes)
  Instructions:   2216 (35,456 / 16)

Per-instruction byte layout (16 bytes, little-endian):
  Offset  Size  Content
  0       8     Instruction word (opcode encoding, operand fields)
  8       8     Control word:
                  [3:0]    stall (0-15)
                  [4]      yield
                  [9:5]    wr_bar
                  [14:10]  rd_bar
                  [20:15]  wait_mask
                  [22:21]  reuse

To patch stall count at instruction address A:
  file_byte_offset = fatbin_offset + elf_text_section_offset + A + 8
  read byte at file_byte_offset
  new_byte = (old_byte & 0xF0) | new_stall
  write byte at file_byte_offset
```
