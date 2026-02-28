# SASS Analysis & Editing Playbook

Sister doc to `profiling.md` (ncu/diagnostics). This doc covers reading, analyzing, and (eventually) editing SASS — the actual machine code that runs on the GPU.

**Scope:** SM100a (Blackwell / B200). SASS analysis is useful now; SASS editing is future work (blocked on tooling).

---

## Tools

| Tool | Purpose | When to use |
|------|---------|-------------|
| `cuobjdump --dump-sass` | Full SASS disassembly with raw control words | Every change — primary SASS inspection tool |
| `nvdisasm` | Disassembly with CFG, source mapping, decoded scheduling | Structural analysis, branch verification |
| `CuAssembler` | Disassemble → edit → reassemble SASS | **Not yet available for SM100a** (supports up to SM86) |
| `ncu --set source` | Per-instruction stall cycles (SASS-level hotspots) | Pinpointing exact bottleneck instructions |
| `cuobjdump --dump-resource-usage` | Register count, SMEM, stack per kernel | Quick build verification |

### SM100a tooling gap

CuAssembler (https://github.com/cloudcores/CuAssembler) is the only viable SASS assembler. It supports SM60-SM86 (Pascal through Ampere). **SM100a is not supported.** Until it is, SASS editing means:
1. Analyze SASS to identify what the compiler did wrong
2. Modify PTX or C to steer the compiler toward the desired output
3. Verify via `cuobjdump --dump-sass` that the compiler obeyed

This is the current workflow. Direct SASS patching is a future option.

### Historical/alternative tools

| Tool | Status |
|------|--------|
| MaxAs | Maxwell only (SM52). Deprecated. Predecessor to CuAssembler. |
| KeplerAs | Kepler only (SM30/SM35). Dead. |
| SASSI (NVlabs) | Instrumentation framework, not an assembler. |
| DocumentSASS (0xD0GF00D) | Community-maintained ISA reference. Useful for opcode details. |

---

## Reading SM100a SASS

### Dump command

```bash
cuobjdump --dump-sass siglip_vision > docs/reference/sass_dump.txt
```

For a specific kernel only:

```bash
cuobjdump --dump-sass --function patch_embed_gemm siglip_vision
```

### Instruction format

Each SM100a instruction is two 64-bit words:

```
        /*0050*/                   R2UR UR17, R24 ;                   /* 0x00000000181172ca */
                                                                      /* 0x000fdc00000e0000 */
```

- **Line 1:** Address, optional predicate (`@P0`, `@!P0`), mnemonic, operands, instruction encoding (hex)
- **Line 2:** Control word (hex) — contains scheduling info (stall counts, dependency barriers, yield hints, reuse flags)

### SM100a mnemonic mapping

SM100a uses different instruction names than older arches and PTX mnemonics. Key translations:

| PTX / expected name | SM100a SASS mnemonic | Notes |
|---------------------|---------------------|-------|
| `tcgen05.mma` | `UTCQMMA` | WGMMA via TMEM. Operands: `gdesc[UR], tmem[UR], idesc[UR]` |
| `tcgen05.ld` | `LDTM.x8` / `LDTM.x16` / `LDTM.x32` | TMEM readback. Width suffix = elements per thread |
| `tcgen05.wait::ld` | `DEPBAR.LE SB0, ...` | Global fence on all outstanding TMEM loads |
| `tcgen05.alloc` | (register setup sequence) | No single instruction; compiler emits UR setup |
| `cp.async.bulk.tensor.2d` | `UTMALDG` / `UTMASTG` | TMA load/store. Uses `desc[UR]` addressing |
| `mbar.arrive.expect_tx` | `MBARRIER.ARRIVE.EXPECT_TX` | mbarrier TX count |
| `mbar.try_wait.parity` | `MBARRIER.TRY_WAIT.PARITY` | mbarrier polling |
| `st.shared` | `STS` / `STS.U8` | Shared memory store |
| `ld.shared` | `LDS` / `LDS.U8` | Shared memory load |
| `st.global` | `STG.E` | Global store via descriptor: `desc[UR][R.64]` |
| `ld.global` | `LDG.E.CONSTANT` | Global load via descriptor. `.CONSTANT` = uniform addressing hint |
| `nanosleep` | `NANOSLEEP` / `NANOSLEEP.SYNCS` | `.SYNCS` variant syncs convergence before sleep |
| `bar.sync` | `BAR.SYNC.DEFER_BLOCKING` | `.DEFER_BLOCKING` = SM100a default barrier mode |

### Descriptor-based addressing

SM100a uses TMA descriptors (`desc[UR]`) for global memory access instead of raw pointers:

```
LDG.E.CONSTANT R33, desc[UR14][R38.64] ;       // load from global via descriptor
STG.E desc[UR14][R32.64+0x4], R5 ;              // store to global via descriptor
UTCQMMA gdesc[UR4], gdesc[UR6], tmem[UR8], tmem[UR20], idesc[UR21], UP0 ;  // WGMMA
```

- `desc[URn]` — TMA descriptor in uniform register
- `[Rn.64]` — 64-bit offset in general register
- `gdesc` — global descriptor (for WGMMA operand tiles)
- `tmem[URn]` — TMEM base in uniform register
- `idesc[URn]` — instruction descriptor (encoding MMA shape/type)

---

## Control word decoding

The second 64-bit hex word encodes scheduling information. SM100a uses a different bit layout than SM89, but the concepts are the same.

### What the control word contains

| Field | Purpose |
|-------|---------|
| **Stall count** | Minimum cycles before next instruction can issue (0-15) |
| **Yield flag** | Hint to scheduler: yield to another warp during stall |
| **Dependency barriers** (wait) | Which scoreboard barriers to wait on before issuing |
| **Dependency barriers** (set) | Which scoreboard barrier to set when this instruction completes |
| **Reuse flags** | Operand register reuse hints (saves register file reads) |

### Scoreboard barriers

SM100a has 6 dependency barriers (SB0-SB5) per warp, used for variable-latency instructions:

```
LDG.E.CONSTANT R33, desc[UR14][R38.64] ;    // issues load, sets barrier SBn
... (other instructions) ...
DEPBAR.LE SBn, 0x0 ;                         // wait until barrier SBn clears (load done)
... (use R33 safely) ...
```

Key patterns in our kernel:
- **TMEM loads** (`LDTM.x8/x16/x32`): Set a barrier, cleared by `DEPBAR.LE SB0`
- **Global loads** (`LDG.E.CONSTANT`): Set a barrier, compiler inserts wait before use
- **TMA copies**: Tracked by mbarrier, not scoreboard barriers

### Stall count analysis

The stall count (0-15) is the minimum issue latency to the next instruction. ptxas sets these conservatively. Common patterns:

| Stall count | Typical meaning |
|-------------|----------------|
| 0 | Dual-issue possible (back-to-back with previous) |
| 1-2 | Register dependency, minimal pipeline latency |
| 4-6 | Shared memory dependency chain |
| 12-15 | Memory load latency (must set yield flag for >11) |

**What to look for:** If ptxas inserts stall=13 between two independent instructions (no data dependency), it's being conservative. With SASS editing, you could reduce this to 1-2. Without SASS editing, reorder your PTX to give the compiler better scheduling freedom.

### Reuse flags

The `.reuse` suffix on operands (visible in cuobjdump) hints that the operand register value will be reused by the next instruction, so the register file read can be cached:

```
ISETP.GT.U32.AND P0, PT, R23.reuse, 0x1f, PT ;   // R23 reused in next instruction
```

ptxas generally handles this well. Not a manual optimization target.

---

## Practical SASS analysis workflows

### 1. Register pressure analysis

```bash
cuobjdump --dump-resource-usage siglip_vision
```

Output includes per-kernel: registers/thread, SMEM, stack, spills. Quick check after every build.

For detailed register liveness, dump SASS and count unique register names in a loop body:

```bash
cuobjdump --dump-sass --function patch_embed_gemm siglip_vision | \
    grep -oP 'R\d+' | sort -t'R' -k2 -n -u | tail -5
```

This shows the highest-numbered register used. If you see R221 with 222 regs allocated, the compiler is using the full budget.

### 2. Instruction mix in a loop body

Identify the epilogue Phase 1 loop by finding `LDTM` (TMEM load) and the surrounding compute:

```bash
cuobjdump --dump-sass --function patch_embed_gemm siglip_vision | \
    grep -A 100 'LDTM'
```

Count instruction types in the loop body:

```bash
# From the SASS dump, extract a loop body and tally mnemonics
cuobjdump --dump-sass --function patch_embed_gemm siglip_vision | \
    sed -n '/LDTM/,/BRA/p' | \
    grep -oP '^\s+/\*[^*]*\*/\s+\K\S+' | sort | uniq -c | sort -rn
```

Expected for epilogue Phase 1:
```
 16  FADD        (bias + pos_embed add, FP32)
  8  F2FP        (FP32 → BF16 conversion, if CVT compiles to this)
  4  STS         (st.shared to staging buffer)
  4  LDG.E       (combined tensor loads)
  1  LDTM        (TMEM readback)
  1  DEPBAR      (TMEM wait)
  1  BRA         (loop branch)
```

If you see unexpected instructions (extra MOVs, spill LDL/STL, IMAD for address calc), the compiler is being suboptimal.

### 3. Spill detection

Spills show up as local memory loads/stores (`LDL`/`STL`) in the SASS:

```bash
cuobjdump --dump-sass --function patch_embed_gemm siglip_vision | grep -c 'LDL\|STL'
```

Should be 0. If nonzero, the compiler is spilling registers to local memory (L1 cache). Each spill costs ~20-30 cycles. Find which loop they're in and reduce register pressure in that section.

### 4. Identify compiler-inserted synchronization

The compiler sometimes inserts barriers or fences you didn't ask for:

```bash
cuobjdump --dump-sass --function patch_embed_gemm siglip_vision | \
    grep -n 'BAR\.\|MEMBAR\|FENCE\|DEPBAR\|WARPSYNC'
```

Compare against your source. Every barrier in the SASS should correspond to one in your PTX/C. Unexpected barriers mean the compiler is being defensive — usually fixable by restructuring the code.

### 5. Branch structure verification

Verify warp specialization creates clean, disjoint code paths:

```bash
cuobjdump --dump-sass --function patch_embed_gemm siglip_vision | \
    grep -n 'BRA\|@P.\|@!P.\|EXIT\|RET'
```

You should see the early `@P0 BRA` that separates W0/W1 from W2-W6, and within each path, minimal branching. Excessive conditional branches inside loops indicate the compiler didn't fully unroll or is generating predicated fallback paths.

### 6. Loop unroll verification

Check that `#pragma unroll` actually unrolled:

```bash
# Count backward branches (BRA to lower address = loop back-edge)
cuobjdump --dump-sass --function patch_embed_gemm siglip_vision | \
    grep 'BRA' | head -20
```

A fully unrolled loop has no backward branch at the end — just a straight sequence of repeated instructions. If you see `BRA 0xNNN` where NNN is a lower address, the loop wasn't unrolled.

### 7. WGMMA instruction analysis

Find the MMA instruction and verify its configuration:

```bash
cuobjdump --dump-sass --function patch_embed_gemm siglip_vision | grep 'UTCQMMA'
```

Should see exactly one UTCQMMA per K-loop iteration (or one per unrolled iteration). The operands encode:
- `gdesc[UR4], gdesc[UR6]` — A and B tile descriptors
- `tmem[UR8], tmem[UR20]` — accumulator TMEM bases (double-buffered)
- `idesc[UR21]` — instruction descriptor (shape, type, cta_group)
- `UP0` — completion predicate

Multiple UTCQMMAs per loop body means the compiler unrolled the MMA (expected with K-unroll).

---

## nvdisasm workflows

nvdisasm provides higher-level analysis than cuobjdump.

### Extract cubin first

```bash
cuobjdump --dump-cubin siglip_vision > /tmp/kernel.cubin
```

### Control flow graph

```bash
nvdisasm -cfg /tmp/kernel.cubin > /tmp/cfg.dot          # full CFG
nvdisasm -bbcfg /tmp/kernel.cubin > /tmp/bbcfg.dot      # basic-block level CFG
dot -Tpng /tmp/bbcfg.dot -o /tmp/bbcfg.png              # render (requires graphviz)
```

What to look for:
- 3 disjoint subgraphs (W0 path, W1 path, W2-6 path) connected only at barriers
- No unexpected merge points inside loop bodies
- Clean loop structure: single back-edge per loop

### Physical layout

```bash
nvdisasm -playout /tmp/kernel.cubin
```

Shows instruction addresses and sizes. Useful for icache analysis — if the kernel exceeds 16 KB of SASS, icache misses become a factor (ncu metric: `inst_executed_op_icache_hit_rate`).

### Source-correlated SASS

Requires `-lineinfo` build:

```bash
nvdisasm -g /tmp/kernel.cubin
```

Maps each SASS instruction back to a source line. Combined with `ncu --set source`, this pinpoints exactly which source line generates the bottleneck instruction.

---

## Per-instruction hotspot analysis

The most powerful SASS analysis combines `ncu` source-level profiling with SASS disassembly.

### Collect source-level data

```bash
# Build with line info
nvcc -gencode arch=compute_100a,code=sm_100a -O3 -lineinfo megakernel.cu -o siglip_vision -lcurand -lcuda

# Profile with source-level detail
ncu --set source --source-level all \
    -k patch_embed_gemm -o source.ncu-rep ./siglip_vision
```

### Analyze (GUI or CLI)

Open `source.ncu-rep` in Nsight Compute GUI → Source tab. Each SASS instruction shows:
- **Stall cycles** — how many cycles this instruction was the bottleneck
- **Issue count** — how many times it was issued
- **Stall reasons** — which scoreboard/barrier caused the stall

**What to look for:**
- A single `DEPBAR.LE SB0` (TMEM wait) with 60%+ of all stall cycles → TMEM latency is the bottleneck
- `BAR.SYNC` with high stall cycles → barrier is over-synchronizing
- `STG.E` with high `mio_throttle` → store path is congested
- `LDG.E` with high `long_scoreboard` → global load latency not hidden

---

## SASS editing (future — when CuAssembler adds SM100a)

### Workflow

```
nvcc → siglip_vision (ELF with embedded cubin)
    ↓
cuobjdump --dump-cubin → kernel.cubin
    ↓
cuasm disasm kernel.cubin → kernel.cuasm (editable text)
    ↓
[edit kernel.cuasm — change stall counts, reorder instructions, remap registers]
    ↓
cuasm assemble kernel.cuasm → kernel_patched.cubin
    ↓
[repack cubin into ELF — replace cubin section]
    ↓
./siglip_vision (runs patched SASS)
```

### What's worth editing

From most to least impactful:

**1. Stall count reduction** — ptxas inserts conservative stalls between independent instructions. Reducing stall counts in tight loops directly reduces latency. Typical win: 5-15%.

**2. Instruction reordering** — move independent instructions between a long-latency instruction and its consumer to fill the stall window. Example: issue a `LDG` early, interleave FADDs, consume the load result later. ptxas does this but is conservative with inline PTX.

**3. Register remapping** — force different phases of the kernel to reuse the same physical registers. This is the register pressure solution: manually guarantee that GEMM epilogue registers and flash attention registers map to the same physical registers (they're never live simultaneously). ptxas can't prove this across inline PTX scopes.

**4. Barrier/scoreboard tuning** — adjust which instructions set/wait on which scoreboard barriers. ptxas sometimes assigns barriers suboptimally, causing unnecessary stalls.

### What's NOT worth editing

- **Adding/removing instructions** — changing the instruction sequence is extremely fragile (offset tables, branch targets, all break)
- **Modifying WGMMA/TMA instructions** — these have complex encoding with many implicit constraints
- **Anything in straight-line non-loop code** — setup/teardown runs once, not worth optimizing

### Risks

- Every recompilation invalidates all edits (the cubin changes completely)
- SASS encoding is undocumented by NVIDIA — community-reverse-engineered only
- SM100a encoding is not yet decoded by any open-source tool
- Binary patches can silently produce wrong results if scoreboard barriers are incorrect

**Rule: only edit SASS after the kernel structure is frozen.** During active development, steer the compiler via PTX instead.

---

## Steering the compiler without SASS editing

Since direct SASS editing isn't available for SM100a, these techniques achieve similar goals through PTX/C:

### 1. Control instruction ordering via inline PTX

The compiler respects instruction order within a single `asm volatile` block:

```c
asm volatile(
    "ld.global.v4.b32 {%0,%1,%2,%3}, [%4];\n"  // issue load early
    "add.f32 %5, %6, %7;\n"                      // independent work fills latency
    "add.f32 %8, %9, %10;\n"
    : "=r"(a), "=r"(b), "=r"(c), "=r"(d), "=r"(x), "=r"(y)
    : "r"(ptr), "f"(p), "f"(q), "f"(r), "f"(s)
);
```

But `asm volatile` also prevents the compiler from reordering across blocks. Use it judiciously — too many small `asm volatile` blocks fragment the scheduler's view.

### 2. Separate `asm` blocks for independent work

Use `asm` (non-volatile) for pure computation that the compiler can freely reorder:

```c
asm("add.f32 %0, %1, %2;" : "=f"(x) : "f"(a), "f"(b));  // compiler can move this
```

Reserve `asm volatile` for instructions with side effects (memory ops, barriers, TMEM).

### 3. `__noinline__` for register isolation

```c
__device__ __noinline__ void epilogue_phase(...) {
    // compiler allocates registers independently for this function
    // spills/fills at call boundary, not within the function body
}
```

This is the practical solution to the register union problem for the persistent megakernel. Each major phase (GEMM epilogue, flash attention, LayerNorm) gets independent register allocation. The call overhead is negligible at phase-transition frequency.

### 4. Register pressure hints

```c
__launch_bounds__(224, 1)  // 224 threads, 1 block/SM → compiler gets full 255 regs
```

Already in use. Verify with `cuobjdump --dump-resource-usage`.

### 5. Verify after every change

The feedback loop — this is the whole point of SASS analysis:

```bash
make && cuobjdump --dump-sass --function patch_embed_gemm siglip_vision | \
    grep -c 'LDL\|STL'           # spill check

make && cuobjdump --dump-sass --function patch_embed_gemm siglip_vision | \
    sed -n '/LDTM/,/BRA/p'       # inspect epilogue loop body
```

If the SASS doesn't match intent, restructure the PTX until it does. This is slower than SASS editing but works today on SM100a.

---

## Quick reference: the minimum set

For the iterative optimize → verify loop on this kernel:

1. **`cuobjdump --dump-sass`** — read SASS, verify compiler output matches intent
2. **`cuobjdump --dump-resource-usage`** — register count, spill check
3. **`ncu --set source`** — per-instruction stall cycles (where exactly is time spent)
4. **`nvdisasm -bbcfg`** — verify branch structure and loop shapes
5. **Steering via PTX** — reorder instructions, separate `asm volatile` blocks, `__noinline__` functions

Items 1-4 are analysis. Item 5 is the current "editing" workflow until CuAssembler supports SM100a.
