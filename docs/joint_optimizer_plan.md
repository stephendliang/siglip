# Joint Instruction Selection + Scheduling Optimizer — Plan

CP-SAT model that jointly optimizes WHAT instructions implement the FC2 epilogue
AND how they're ordered. Current tools do these separately (template_emitter picks
instructions, schedule_cpsat orders them). Separating them is suboptimal — the
optimal mix depends on interleaving, and optimal interleaving depends on mix.

**Why**: FC2 epilogue is 24:1 BF16:ALU imbalanced. 384 BF16 ops vs 21 ALU ops.
BF16 pipe is the bottleneck. Moving some ops to ALU (HFMA2→2×FFMA, HADD2→2×FFMA)
rebalances pipes but increases instruction count. Only CP-SAT can find the
optimal tradeoff between pipe balance and instruction count, jointly with
scheduling to exploit STS shadows and cross-pipe parallelism.

**Scope**: One epilogue warp's complete 256-col pass. 128 semantic operations
(bias-add + residual-add per lane pair), each with 2-3 lowering options,
producing 500-1000 concrete instructions depending on choices.

---

## Status key

- `[ ]` not started
- `[~]` in progress
- `[x]` done

---

## Phase 1: Lowering Rules Database

Define all valid concrete instruction sequences for each semantic epilogue operation.
This is the foundation — every other module consumes it.

`[x]` **1.1 Semantic operation types**
Define the 7 semantic ops: LOAD_TMEM, TMEM_WAIT, LOAD_BIAS, LOAD_RES,
ADD_BIAS, ADD_RES, CONVERT_BF16, STORE_SMEM. Each with typed inputs/outputs
(FP32_pair, BF16_packed, SMEM_addr).

`[x]` **1.2 Lowering rules for ADD_BIAS**
- HFMA2 path (BF16 context): 1× HFMA2.BF16_V2. Pipe: BF16, 3 cyc @ILP≤2.
- FFMA path (FP32 context, FP32 bias): 2× FFMA. Pipe: ALU, 2×2 cyc.
- FFMA path (FP32 context, BF16 bias): 2× LOP3+SHF(unpack) + 2× FFMA. Pipe: ALU.
Each lowering: instruction list, pipe assignments, register pattern, latency chain.

`[x]` **1.3 Lowering rules for ADD_RES**
- HADD2 path (BF16 context): 1× HADD2.BF16_V2. Pipe: BF16, 3 cyc @ILP≤2.
- FFMA path (FP32 context): 2× LOP3+SHF(unpack res) + 2× FFMA.
  Residual is always BF16 from TMA — unpack mandatory for FP32 path.

`[x]` **1.4 Lowering rules for CONVERT (F2FP)**
- EARLY: F2FP.BF16.F32.PACK_AB immediately after TMEM. 1× F2FP, BF16 pipe.
  Everything downstream is BF16.
- LATE: F2FP.BF16.F32.PACK_AB after bias+residual adds (CUTLASS style).
  Everything upstream is FP32. 1× F2FP, BF16 pipe.
- F2FP is LOCKED to BF16 pipe — no ALU substitute exists.

`[x]` **1.5 Lowering rules for fixed ops**
- LOAD_TMEM: 1× LDTM.x32 (TMEM pipe, ~20 cyc). No alternatives.
- TMEM_WAIT: 1× NOP with DEPBAR (control word). No alternatives.
- LOAD_BIAS: 4× LDS.128 per chunk (LOAD pipe, 4 cyc each). BF16 format: 4 loads.
  FP32 format: 8 loads (2× bandwidth).
- LOAD_RES: 4× LDS.128 per chunk (LOAD pipe). Always BF16, always 4 loads.
- STORE_SMEM: 4× STS.128 per chunk (STORE pipe, 32 cyc each). No alternatives.

`[x]` **1.6 Compatibility matrix**
3 strategies (early_bf16bias, late_fp32bias, late_bf16bias).
`compatible_lowerings()` filters by F2FP position and bias format.

`[x]` **1.7 Cost model per lowering**
Per-lowering `pipe_cycles` dict with BF16 slow family penalty (3 cyc).
`ChunkCost` aggregates per-chunk. `analyze_hybrid()` models per-pair path
selection with correct BF16 vs ALU path costs.

`[x]` **1.8 Unit tests**
28 tests in `tools/test_lowering_rules.py`. Covers: lowering existence,
pipe assignments, compatibility filtering, cost computation, hybrid sweep
monotonicity/crossover, store cycle invariance.

**Key finding from hybrid sweep (2026-03-25):**
Optimal hybrid: k=6 of 16 pairs on ALU path → compute bottleneck 96 cyc (25%
reduction vs 128 cyc pure BF16). But STORE pipe = 128 cyc regardless of k.
At k=6 compute (96) < STORE (128), so STS interleaving (hiding compute in
32-cyc STS shadow) is critical — only CP-SAT joint scheduling can evaluate this.

**Deliverable:** `tools/lowering_rules.py` — importable module with `LoweringRule`
dataclass, enumeration functions, compatibility checker, cost model.
`tools/test_lowering_rules.py` — 28 unit tests.

---

## Phase 2: Joint CP-SAT Model — Staged Build

The core optimizer. Jointly decides instruction selection + ordering + stalls.
Built in 4 stages, each producing runnable output.

Phase 2 subsumes the original Phase 2 (DAG builder — inlined into model
construction) and Phase 3 (CP-SAT model).

### Stage A: Per-chunk joint model (32 cols = 16 pairs)

Smallest meaningful scope. Proof of concept. ~60-160 instructions depending
on lowering choices. Should solve in seconds.

`[x]` **A.1 Chunk instruction generator**
`generate_chunk_instructions(k_alu)`: generates both paths (k_alu=None) or
fixed assignment. Per-pair: BF16 path (3 insns) or ALU path (9 insns).

`[x]` **A.2 Dependency graph from read/write sets**
`build_deps()`: RAW/WAW/WAR from symbolic register names. Correctly handles
LDS→compute chains, LDTM→DEPBAR→compute, shared TMEM registers.

`[x]` **A.3 Lowering selection variables**
Per-pair `BoolVar(alu_p)`. `active[i]` linked to pair's path choice via
`OnlyEnforceIf`. Inactive insns get `time=0`, excluded from no_overlap.

`[x]` **A.4 Scheduling variables + constraints**
`time_var[i]` per instruction. **IntervalVar + no_overlap** per pipe
(O(n) vs O(n²) pairwise). All OPTIMAL in <0.1s up to 158 insns.

`[x]` **A.5 STS shadow model**
STS.128 intervals on STORE pipe (32 cyc each). Other pipes issue freely in
parallel. Solver naturally interleaves: BF16+ALU+STORE on same cycle.

`[x]` **A.6 Objective + solve**
`minimize(makespan * 1000 + total_active)`. Solves to OPTIMAL in 44s for
joint model (206 candidate insns, 92 active). Fixed-k: <0.1s each.

`[x]` **A.7 Solution printer**
Detailed schedule (pos, cycle, stall, pipe, mnemonic), pipe utilization bars,
makespan vs throughput lower bound, scheduling overhead.

`[x]` **A.8 Validation**
k=0: 150 cyc OPTIMAL (matches BF16 bottleneck + 22 cyc dep overhead).
k=16: 282 cyc OPTIMAL (ALU-dominated). k=1-7: 144-150 plateau (STORE-bound).
Joint: 144 cyc OPTIMAL, k=1 (pair 8 on ALU), 68 insns.

**Key finding (2026-03-25, corrected):**
Joint model: **144 cyc OPTIMAL**, k=1, 68 active insns (7.5s).
- vs pure BF16 (k=0): 150 cyc → **4% reduction**
- STORE-bound at 128 cyc. Overhead = 16 cyc (12%).
- Previous claim of 116 cyc was WRONG — caused by a dependency bug where
  `build_deps` last_writer lost BF16 path's HADD2→STS RAW dependency when
  both paths wrote the same `cvt_{p}` register. Fixed in Stage B by using
  path-specific output register names (`cvt_bf16_{p}`, `cvt_alu_{p}`) so
  STS reads from both, and the active path's dep is always enforced.
- Model uses IntervalVar + no_overlap: solves to OPTIMAL in <0.1s (fixed-k)
  or 7.5s (joint). Previous O(n²) pairwise formulation was 1000× slower.

**Deliverable:** `tools/joint_optimizer.py` — per-chunk joint optimize + print.

### Stage B: Multi-chunk model (64 cols = 2 chunks = 1 sub-iteration)

`[x]` **B.1 Cross-chunk overlap**
`_generate_chunk(chunk_idx, start_idx)` with prefixed register names
(`c{chunk}_tmem_*`, `c{chunk}_bias_*`, etc.). No cross-chunk register deps.
Cross-chunk pipe contention handled by shared no_overlap constraints.
LDTM_c1 naturally fires during STS_c0 shadow (TMEM vs STORE = independent).

`[x]` **B.2 Sub-iteration boundary ops**
IS1 pattern after both chunks: WARPSYNC(lat=2) → FENCE(lat=10) →
UTMASTG(TMA pipe) → COMMIT(CONTROL pipe). Synthetic `sts_done` registers
link all STS → WARPSYNC. Boundary tail = 14 cyc (verified).

`[x]` **B.3 Per-pair lowering independence across chunks**
Joint model: 32 independent BoolVars (16 per chunk). Each chunk can have
different k_alu and pair assignments. `k_alu` accepts None (joint), int
(same for all chunks), or list (per-chunk).

`[x]` **B.4 Validation**
- k=0 (pure BF16): **292 cyc OPTIMAL** (256 LB + 36 overhead = 14%).
  vs 2×150 = 300 → **8 cyc cross-chunk overlap gain (2.7%)**.
- k=1: **286 cyc OPTIMAL** (256 LB + 30 overhead = 12%). Best fixed-k.
- k=5: **286 cyc OPTIMAL** (tied). k=1-7 plateau at 286-292 (STORE-bound).
- Joint: **286 cyc FEASIBLE** (300s, matches fixed-k optimum). k=1 total,
  asymmetric: chunk 0 pure BF16, chunk 1 pair 5 on ALU. 134 active of 416.
- All 17 fixed-k solutions OPTIMAL in <1s. Joint found 286 at 9.3s.
- Boundary tail exactly 14 cyc as predicted.
- **STORE pipe (256 cyc) dominates** — instruction selection barely matters
  at this scope. Only 6 cyc (2.1%) improvement from k=0→k=1.

**Key finding (2026-03-25):**
At 64-col (2 chunk) scope, STORE pipe is the absolute bottleneck at 256 cyc.
Cross-chunk overlap saves only ~8 cyc. Instruction mix selection gives only
~6 cyc. Total overhead: 30 cyc (12%). Stage C may show more benefit from
inter-sub-iteration pipelining where TMA stores overlap with next sub-iter.

**Deliverable:** Extended `joint_optimizer.py` — `--chunks 2` mode.

### Stage C: Full tile model (256 cols = 8 chunks = 4 sub-iterations)

`[ ]` **C.1 Full 256-col scope**
All 8 chunks, 4 sub-iterations, 128 lowering BoolVars.
500-1000+ instructions. May need hours to solve optimally.

`[ ]` **C.2 Structural config sweep**
Enumerate IS1 vs BAR_SYNC, BF16 vs FP32 bias. Solve each, report Pareto.

`[ ]` **C.3 Assembly output**
CP-SAT solution → `.s` file compatible with `sass_edit.py` assembler.
Include `.ctrl` directives with computed stalls, yield, barrier fields.

`[ ]` **C.4 Comparison table**
Side-by-side: current Template A, Template B, each optimized config.

**Deliverable:** `--full-tile` mode, `.s` output, sweep driver.

### Stage D: Register allocation co-optimization (optional)

`[ ]` **D.1 Physical register assignment as CP-SAT variables**
Replace symbolic register names with `IntVar(0, REG_BUDGET)` per value.
Add all_different for simultaneously-live values.

`[ ]` **D.2 SMEM bank conflict model**
LDS/STS addresses depend on register values. Model bank conflicts as
additional throughput penalties.

`[ ]` **D.3 Encoding constraints**
Some register combinations produce shorter encodings or avoid extra
mode bits. Model as soft constraints or secondary objective.

**Deliverable:** Extended model with register allocation. Only pursue if
Stages A-C leave measurable gap vs CUTLASS.

---

## Phase 3: Integration + Validation

`[ ]` **3.1 End-to-end: optimize → assemble → patch**
Full pipeline: joint optimizer → `.s` file → `sass_edit.py` assemble →
patch into cubin/fatbin.

`[ ]` **3.2 Round-trip correctness**
Assemble optimized output, disassemble, verify matches solution.

`[ ]` **3.3 B200 benchmark** (requires hardware)
Patch optimized epilogue into FC2, run on B200, compare timing vs baseline.
Final validation: does the optimized instruction mix close the CUTLASS gap?

**Deliverable:** Working end-to-end pipeline, benchmark results.
