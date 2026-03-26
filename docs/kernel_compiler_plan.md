# SM100a Kernel Compiler — Long-Term Plan

From s-expression GEMM+epilogue descriptions to optimal scheduled SASS,
patched into runnable binaries. Targeting Blackwell B200 persistent GEMMs.

---

## Honest situation assessment

**Where you stand:**

You have three hand-tuned persistent GEMM kernels for SigLIP2. Two beat CUTLASS.
FC2 is 17% behind, and months of work proved the gap is purely structural —
the wrong warp architecture for a memory-heavy epilogue, not bad instructions.

You also have a nearly-complete SASS assembler (encode, decode, schedule, patch)
and the deepest SM100a hardware characterization I've seen outside NVIDIA.
CP-SAT scheduling alone predicts 686 vs 1417 cycle epilogues. You have the
raw ingredients.

What you don't have is a systematic way to go from "GEMM + bias + GELU" to
the right structural decisions — warp count, who loads what, pipeline depth,
staging layout, overlap strategy. These are the decisions that determine 90%
of performance. CUTLASS solves this with hand-written C++ template policies.
You want to solve it with a compiler.

**What makes this tractable:**

The structural decision space is small. For SM100a persistent GEMMs with
cta_group::2, the architecture is:

- 1 MMA warp (always W1, always CTA0-lane0, non-negotiable)
- 1 TMA load warp (always W0)
- N epilogue warps (2-6, typically 4)
- 0-2 auxiliary warps (dedicated epilogue loader, tile scheduler)
- 1 pipeline depth (3-5 SMEM stages)
- 1 epilogue staging layout (per-warp vs per-pipeline-stage)
- 1 overlap strategy (epilogue overlaps with next tile's K-loop)

For a given GEMM shape + epilogue, maybe 20-50 viable configurations exist.
You already sweep most of them with grid_search.py. The compiler formalizes
this into deterministic code generation instead of compile-flag combinatorics.

**What makes this hard:**

The structural decisions interact. Warp count affects register pressure which
affects spills which affects performance. Pipeline depth affects SMEM budget
which constrains staging layout. Epilogue compute intensity determines whether
a dedicated loader helps or hurts (FC1 GELU = compute-bound, loader useless;
FC2 residual = memory-bound, loader essential). These interactions are why
grid search found that 14 of 18 "promising" FC2 configs were within 14us of
each other — most knobs are noise, but the RIGHT structural change is 17%.

---

## Architecture

```
┌──────────────────────────────────────────────┐
│  S-expression input                          │
│  (gemm [M N K] :epilogue (add bias residual))│
└──────────────────┬───────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │  Shape analyzer     │  ← K-iterations, N-tiles, compute/memory ratio
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │  Config selector    │  ← warp count, pipeline depth, staging layout
         │  (rule-based +      │     load strategy, epilogue sub-iterations
         │   constraint solver)│
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │  Template emitter   │  ← mainloop template + epilogue template
         │  (parameterized     │     register slots, mbar protocol, SMEM offsets
         │   SASS fragments)   │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │  Register allocator │  ← assign physical regs to template slots
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │  CP-SAT scheduler   │  ← instruction ordering within basic blocks
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │  Assembler + patcher│  ← Phases 1-5 (DONE)
         └─────────┬──────────┘
                   │
              final binary
```

---

## Phase A: Formalize the decision space

Before generating code, codify what you already know about which configurations
work for which shapes. This is the distillation of months of FC2 work into rules.

### A.1 Shape analysis

Given (M, N, K, epilogue_type), compute derived properties:

```python
tiles_m = ceildiv(M, TM * 2)          # 2 from cta_group::2
tiles_n = ceildiv(N, TN)
tiles_total = tiles_m * tiles_n
k_iters = K // TK                      # TK=128 always
tiles_per_sm = ceildiv(tiles_total, 74) # 74 clusters, 148 SMs

# Epilogue characterization
epi_compute_ops = ...   # FLOPs per output element in epilogue
epi_load_bytes = ...    # bytes loaded per output element (residual, bias, etc.)
epi_compute_bound = epi_compute_ops / epi_load_bytes > threshold
```

**Epilogue taxonomy:**

| Type | compute ops/elem | load bytes/elem | Bound | Example |
|------|-----------------|-----------------|-------|---------|
| BIAS_ADD | 1 add | 2B (BF16 bias, amortized over rows) | neither | PE |
| BIAS_GELU | 1 add + ~10 GELU ops | 4B (FP32 bias, amortized) | compute | FC1 |
| BIAS_RESIDUAL | 1-2 adds | 2-4B (bias) + 2B (BF16 residual per elem) | memory | FC2 |
| BIAS_RES_GELU | ~12 ops | 2-4B + 2B residual | mixed | future |
| IDENTITY | 0 | 0 | trivial | raw GEMM |

### A.2 Configuration constraints

Hard constraints (violating these = hang or wrong results):

```
SMEM_total <= 228 KB
TMEM = 512 cols (single alloc, double-buffered)
warps * 32 * regs_per_thread <= 65536  (register file)
regs_per_thread <= 256
N_STAGES >= 3 (pipeline correctness)
OFF_STAGING % 1024 == 0 (SWIZZLE_128B)
TMA stores need fence.proxy.async before reading SMEM written by st.shared
```

Soft constraints (learned from benchmarking):

```
N_STAGES = 5 mandatory for K >= 24 iters (FC2): 10% gap vs NS4
N_STAGES = 4 acceptable for K = 6 iters (PE, FC1)
EPI_LOAD_WARP = 1 only useful if epilogue is memory-bound
   AND staging is double-buffered (our serial impl regressed +13%)
STAGES_C >= 2 is dead (cross-tile SMEM pipelining, always regresses K-loop)
W0_RES_FULL is catastrophic (always)
FP32 vs BF16 epilogue math: noise when epilogue is overlapped
```

### A.3 Decision rules (codified from FC2 experience)

```python
def select_config(shape, epilogue):
    cfg = Config()

    # Pipeline depth
    if shape.k_iters >= 16:
        cfg.n_stages = 5
    elif shape.k_iters >= 8:
        cfg.n_stages = 5  # still wins marginally
    else:
        cfg.n_stages = 4  # K=6: NS4 is fine, saves SMEM

    # Warp structure
    cfg.num_epi_warps = 4  # always 4 epilogue warps

    if epilogue.is_memory_bound and epilogue.load_bytes_per_elem >= 2:
        # FC2-like: dedicated loader with pipeline staging
        cfg.epi_load_warp = True
        cfg.epi_pipeline_stages = 4  # 4 sub-iters of 64 cols
        cfg.total_warps = 7  # W0=TMA, W1=MMA, W2=loader, W3-W6=epilogue
    else:
        # PE/FC1-like: epilogue warps self-load (or no load needed)
        cfg.epi_load_warp = False
        cfg.epi_pipeline_stages = 0
        cfg.total_warps = 6  # W0=TMA, W1=MMA, W2-W5=epilogue

    # Staging layout
    if cfg.epi_pipeline_stages > 0:
        # Per-stage: 4 stages x NUM_EPI_WARPS x REGION_BYTES
        cfg.staging = 'pipeline'
    else:
        # Per-warp: NUM_EPI_WARPS x REGIONS_PER_WARP x REGION_BYTES
        cfg.staging = 'per_warp'

    # Interleave strategy: IS1 ≈ IS2 for FC2, IS1 wins for PE
    cfg.interleave = 1

    # Epilogue sub-iteration structure
    if cfg.epi_pipeline_stages > 0:
        # CUTLASS-style: 4 sub-iters x 64 cols, BAR.SYNC pipelined
        cfg.sub_iters = 4
        cfg.cols_per_sub_iter = 64
        cfg.intra_epi_barrier = True
    else:
        # Current: 2 passes x 128 cols (or 4 x 64, doesn't matter when overlapped)
        cfg.sub_iters = 2
        cfg.cols_per_sub_iter = 128
        cfg.intra_epi_barrier = False

    return cfg
```

### A.4 Deliverable

`tools/kernel_config.py` — given (M, N, K, epilogue_type), outputs a complete
configuration dict. Includes all SMEM offsets, mbarrier offsets, warp roles,
register budget, staging layout. This replaces the maze of `#if` in
kernel_common.cuh with a single source of truth.

---

## Phase B: SASS template library — ✅ PARTIAL (B.1-B.3 DONE)

The core code generation. Each template is a parameterized SASS fragment
with register slots, label placeholders, and compile-time constants.

**Implementation:** `tools/template_emitter.py` (530 lines) + `tools/test_template_emitter.py` (30 tests).
Emits two FC2 epilogue variants as valid SASS assembly:
- **Template A** (`templates/fc2_epi_current.s`): Current IS1 architecture, 294 instructions (128 cols)
- **Template B** (`templates/fc2_epi_barsync.s`): BAR.SYNC pipelined, 587 instructions (256 cols)
  - Interleaved STS (1 STS.128 per 12 compute ops, matching CUTLASS pattern)
  - BAR.SYNC.DEFER_BLOCKING 0x1, 0x80 (barrier 1, 128 epi threads)
  - Protocol: MEMBAR → FENCE → BAR.SYNC → UTMASTG → DEPBAR → BAR.SYNC per sub-iter
  - Single staging buffer reused via BAR.SYNC ownership (vs 4 separate regions)
Both templates assemble to valid binary via `sass_edit.py assemble`.

### B.1 Template language

Templates are annotated SASS text with substitution markers:

```asm
# Template: bias_residual_sub_iter
# Params: $STAGE, $EW_IDX, $STG_BASE, $BIAS_SADDR, $RES_MBAR
#
# Registers (allocated by register allocator):
#   %acc0-%acc7     FP32 accumulator pairs (from TMEM)
#   %bias0-%bias3   BF16 bias values
#   %res0-%res3     BF16 residual values
#   %cvt0-%cvt3     BF16 converted output
#   %tmp0-%tmp1     temporaries

.ctrl 0x...
    LDTM.x32 %acc0, [TMEM+$TMEM_OFF]        # load 32 cols from TMEM
.ctrl 0x...
    LDTM.x32 %acc4, [TMEM+$TMEM_OFF+0x20]   # next 32 cols

    # Wait for W2's TMA load of residual into staging SMEM
.ctrl 0x...
    DEPBAR.LE SB0, $DEP_COUNT

    # Bias from SMEM (preloaded once per tile)
.ctrl 0x...
    LDS.128 %bias0, [$BIAS_SADDR+$BIAS_OFF]

    # Residual from staging SMEM (loaded by W2 via TMA)
.ctrl 0x...
    LDS.128 %res0, [$STG_BASE+$RES_OFF]

    # FP32 -> BF16 convert, interleaved with adds
.ctrl 0x...
    F2FP.BF16.F32.PACK_AB %cvt0, %acc0, %acc1
.ctrl 0x...
    HADD2.BF16_V2 %cvt0, %cvt0, %bias0
.ctrl 0x...
    HADD2.BF16_V2 %cvt0, %cvt0, %res0
    # ... (8 F2FP + 8 HADD2, interleaved)

    # Store to staging SMEM (ReuseSmemC — same buffer residual was in)
.ctrl 0x...
    STS.128 [%stg_addr], %cvt0
.ctrl 0x...
    STS.128 [%stg_addr+0x80], %cvt2
```

### B.2 Template inventory

**Mainloop (largely common, parameterized):**

| Template | Description | Params |
|----------|-------------|--------|
| `preamble` | Tile scheduler, TMEM alloc, mbar init, CTA rank | warps, mbar_count |
| `tma_load_ab` | W0 TMA bulk copy A+B tiles | n_stages, tile_shape |
| `mma_loop` | W1 UTCQMMA accumulation | k_iters, batch_mma |
| `mbar_protocol` | Inter-warp synchronization | n_stages, warp_roles |
| `drain` | Final tile cleanup | warp_roles |

**Epilogue (per epilogue type):**

| Template | Description | Params |
|----------|-------------|--------|
| `epi_bias_add` | TMEM → F2FP → HADD2(bias) → STS | sub_iters, interleave |
| `epi_bias_gelu` | TMEM → F2FP → FADD(bias) → GELU → F2FP → STS | gelu_variant |
| `epi_bias_residual` | TMEM → F2FP → HADD2(bias,res) → STS | staging, barrier |
| `epi_bias_res_gelu` | TMEM → F2FP → FADD(bias,res) → GELU → STS | staging, barrier |
| `epi_identity` | TMEM → F2FP → STS | (minimal) |

**Structural variants (orthogonal to epilogue type):**

| Template | Description | When |
|----------|-------------|------|
| `epi_self_load` | Epilogue warps issue LDG/TMA for residual | no dedicated loader |
| `epi_pipeline_load` | W2 fires TMA loads into pipeline stages | dedicated loader |
| `epi_bar_sync` | Intra-sub-iter BAR.SYNC pipelining | CUTLASS-style |
| `epi_no_barrier` | No intra-epilogue barriers | overlapped, no staging reuse |

### B.3 Template composition

The emitter stitches templates:

```python
def emit_kernel(config, epilogue):
    parts = []
    parts.append(templates['preamble'].render(config))
    parts.append(templates['tma_load_ab'].render(config))
    parts.append(templates['mma_loop'].render(config))

    # Epilogue: combine type + structural variant
    epi_template = templates['epi_' + epilogue.name]
    if config.epi_load_warp:
        load_template = templates['epi_pipeline_load']
    else:
        load_template = templates['epi_self_load']
    if config.intra_epi_barrier:
        barrier_template = templates['epi_bar_sync']
    else:
        barrier_template = templates['epi_no_barrier']

    parts.append(compose_epilogue(epi_template, load_template, barrier_template, config))
    parts.append(templates['drain'].render(config))

    return '\n'.join(parts)
```

### B.4 Effort and approach

The hard part isn't writing templates — it's getting the mbarrier protocol right
for each structural variant. The protocol has subtle ordering requirements
(arrive_expect_tx before TMA issue, phase toggle timing, first-tile skip).

Approach: write the first template by literally disassembling the existing
FC2 binary, parameterizing register names and constants. Then derive the other
epilogue templates by substituting the compute portion (GELU ops, different
adds, etc.) while keeping the structural skeleton.

---

## Phase C: Register allocator

Templates use virtual register names (%acc0, %bias0, etc.). The allocator
assigns physical registers (R0-R255) respecting:

### C.1 Constraints

- **Consecutive groups**: LDTM loads N consecutive registers. If %acc0 = R80,
  then %acc1 = R81, ..., %acc7 = R87. Same for LDS.128 (4 consecutive).
- **Alignment**: Some TMA/TMEM ops need aligned register numbers.
- **Cross-warp isolation**: Each warp has independent register state, but the
  kernel binary is shared. Register "assignment" is per-template-slot, and
  different warps execute different code paths (branched by `warp_id`).
- **MMA implicit registers**: UTCQMMA writes to TMEM, not GPR — no conflict.
  But TMEM reads (LDTM) write to GPR and need consecutive blocks.
- **Avoid spills**: With 207 regs used by ptxas and 256 max (at 6 warps),
  there's headroom. The allocator should pack tightly to leave room.

### C.2 Approach

Don't do graph coloring. The templates have ~20-30 virtual register groups
with known lifetimes (defined by the template structure). Use a simple
linear scan:

1. List all virtual register groups with their live ranges (instruction indices)
2. Sort by start of live range
3. Assign physical registers greedily, respecting consecutiveness
4. If assignment fails, report — don't spill (templates are designed to fit)

### C.3 Integration with ptxas

For the mainloop (non-epilogue portions), we can keep using ptxas-compiled
code. Only the epilogue is SASS-generated. The register allocator only needs
to assign registers for the epilogue region, and must avoid registers that
ptxas uses across the epilogue boundary (i.e., registers live-in and live-out
of the epilogue).

These boundary registers are discoverable: disassemble the existing binary,
identify registers read in the epilogue preamble (inputs) and written before
the epilogue exit (outputs). Typically ~15-20 registers.

---

## Phase D: Scheduling

CP-SAT scheduling is already implemented in sass_edit.py. The integration:

### D.1 Per-basic-block scheduling

Templates emit SASS with `.ctrl 0x...` directives that carry the control word
skeleton (wait barriers, write barriers, yield). CP-SAT optimizes instruction
ORDER within basic blocks (between barriers/branches) while respecting:

- Data dependencies (read-after-write, write-after-read)
- Resource conflicts (STS+ALU contention from calibration data)
- Latency constraints (LDTM→use, LDS→use, STS throughput)

### D.2 Barrier placement

The templates define barrier structure (BAR.SYNC placement for intra-epilogue
pipelining). CP-SAT schedules within each barrier-delimited region but doesn't
move instructions across barriers.

### D.3 Expected impact

CP-SAT on the current FC2 epilogue predicts 686 cycles vs ptxas's 1417.
Even if this doesn't help wall time today (epilogue is overlapped), it
WILL matter once the structural changes (dedicated loader, BAR.SYNC
pipelining) change the overlap dynamics.

---

## Phase E: End-to-end pipeline

### E.1 Input format

```lisp
(kernel siglip-fc2
  (gemm
    (shape 928256 768 3072)
    (precision :input fp8e4m3 :output bf16 :accumulator fp32)
    (tile 256 256 128)
    (cluster 2 1 1))
  (epilogue
    (add (bias :dtype bf16 :broadcast :row))
    (add (residual :dtype bf16 :shape (M N)))))

(kernel siglip-fc1
  (gemm
    (shape 928256 3072 768)
    (precision :input fp8e4m3 :output bf16 :accumulator fp32)
    (tile 256 256 128)
    (cluster 2 1 1))
  (epilogue
    (add (bias :dtype fp32 :broadcast :row))
    (gelu :variant fast)))
```

### E.2 Compilation pipeline

```bash
# Full pipeline
python3 tools/kernel_compiler.py kernel_spec.sexp -o fc2_compiled

# Steps (can run individually):
python3 tools/kernel_compiler.py kernel_spec.sexp --analyze     # shape analysis + config selection
python3 tools/kernel_compiler.py kernel_spec.sexp --emit        # template rendering
python3 tools/kernel_compiler.py kernel_spec.sexp --schedule    # CP-SAT scheduling
python3 tools/kernel_compiler.py kernel_spec.sexp --assemble    # SASS assembly
python3 tools/kernel_compiler.py kernel_spec.sexp --patch       # cubin/fatbin patching
```

### E.3 Hybrid compilation

The most practical near-term approach: compile the mainloop with nvcc/ptxas
(it's good at this), then replace ONLY the epilogue with SASS-generated code:

```bash
# 1. Compile kernel normally
nvcc --cubin -arch=sm_100a fc2.cu -o fc2.cubin

# 2. Generate optimal epilogue SASS
python3 tools/kernel_compiler.py --epilogue-only fc2_spec.sexp -o epilogue.s

# 3. Patch epilogue into compiled cubin
python3 tools/sass_edit.py fatbin-patch fc2 --asm epilogue.s \
    --start 0x51c0 --end 0x63b0 -o fc2_patched
```

This is where the assembler infrastructure pays off — you don't need to
compile the entire kernel from SASS. Just the part where ptxas makes
suboptimal decisions.

---

## Phase F: Auto-tuning loop

For configurations where the analytical model is uncertain, generate
multiple variants and benchmark:

### F.1 Variant generation

```python
configs = config_selector.enumerate_viable(shape, epilogue)
# Typically 5-20 configs varying:
#   - warp count (6 vs 7)
#   - staging layout (per-warp vs pipeline)
#   - sub-iteration count (2 vs 4)
#   - intra-epilogue barriers (yes/no)
#   - interleave strategy (0-3)
```

### F.2 Batch benchmark

```bash
# Generate all variants
for cfg in configs:
    emit + schedule + assemble + patch → fc2_variant_N

# Benchmark on B200
for variant in variants:
    ./variant --warmup 100 --runs 200 → timing.csv

# Select best
analyze_sweep.py timing.csv
```

### F.3 Feedback

Benchmark results feed back into the rule system (Phase A.3):
- If a rule's prediction was wrong, update the threshold
- If a new configuration wins, add it to the viable set
- Over time, the analytical model improves and fewer variants need benchmarking

---

## Phasing and dependencies

```
Phase A (formalize decisions)     ─── no deps, start now
    ↓
Phase B (SASS templates)          ─── needs A for config params
    ↓                                  needs Phase 7 (SASS assembler) for first template
Phase C (register allocator)      ─── needs B for virtual reg specs
    ↓
Phase D (scheduling integration)  ─── needs B+C, already mostly done (CP-SAT exists)
    ↓
Phase E (end-to-end pipeline)     ─── needs A+B+C+D
    ↓
Phase F (auto-tuning)             ─── needs E + B200 access
```

**What can happen now (no B200):**
- Phase A: entirely analytical, codify existing knowledge
- ~~Phase B: write template skeletons from disassembled FC2 SASS~~ ✅ DONE (2026-03-25)
- Phase C: design allocator, test on FC2 epilogue register usage
- Phase D: wire CP-SAT into template pipeline (schedule Template B)

**What needs B200:**
- Phase B validation: do patched templates produce correct results?
- Phase F: benchmarking
- Any execution testing

---

## Scope reality check

**This is NOT a general-purpose GPU compiler.** It generates code for one
specific kernel pattern: SM100a persistent GEMM with cta_group::2,
TMA+UTCQMMA, warp-specialized. That pattern covers all of SigLIP and most
transformer MLP layers, but it's not cuBLAS.

**What it replaces:** The manual process of editing kernel_body.cuh C++ macros,
recompiling with nvcc, running grid search, analyzing SASS, finding ptxas made
bad scheduling decisions, trying to trick ptxas with source-level changes,
failing because ptxas is immutable, then binary-patching the SASS.

**What it doesn't replace:** Algorithmic decisions (tile shape, cluster config,
pipeline protocol design). Those are baked into the template library and
require human insight to change. The compiler automates the MECHANICAL parts
(code generation, register allocation, scheduling) while the STRUCTURAL
decisions remain in the templates and config selector.

**Estimated complexity by phase:**

| Phase | New code | Key challenge |
|-------|----------|---------------|
| A. Config selector | ~500 lines | Encoding soft constraints correctly |
| B. Template library | ~2000 lines | Mbarrier protocol correctness |
| C. Register allocator | ~400 lines | Consecutive-group packing |
| D. Scheduling integration | ~200 lines | Already exists, just wiring |
| E. End-to-end pipeline | ~600 lines | S-expr parser, glue code |
| F. Auto-tuning | ~300 lines | Extends existing grid_search.py |

Total: ~4000 lines of new Python. The SASS templates (Phase B) are the bulk
and the hardest to get right — one wrong mbarrier phase toggle = deadlock.

---

## Relationship to existing work

**SASS assembler (Phases 1-7):** Foundation. The compiler emits SASS text
that the assembler encodes and patches. All of Phases 1-6 are prerequisites.
Phase 7 (FC2 epilogue via assembly) is the FIRST template — the proof that
hand-written SASS can match or beat ptxas for the epilogue.

**EPI_PIPELINE plan:** The structural variant that the compiler's config
selector would choose for FC2-like epilogues (memory-bound, dedicated loader,
4-stage pipeline). Currently a C++ implementation plan; becomes a SASS template.

**grid_search.py:** The auto-tuning loop (Phase F) is a generalization of
grid search. Instead of sweeping compile flags and recompiling with nvcc,
it sweeps structural configs and regenerates SASS templates.

**CP-SAT scheduler:** Already exists in sass_edit.py. Phase D just wires it
into the compilation pipeline.

---

## First milestone: FC2 epilogue parity

Before building the full compiler, prove the approach on FC2:

1. Disassemble FC2 epilogue → SASS text (Phase 7.1, existing plan)
2. Parameterize into a template with virtual registers
3. Implement register allocator, verify same physical assignments
4. CP-SAT schedule → patch → verify correctness on B200
5. Implement CUTLASS-style BAR.SYNC pipelining as alternate template
6. Benchmark both variants

If the BAR.SYNC variant closes the 17% FC2 gap, the approach is validated
and generalizing to other epilogues is mechanical template work.
