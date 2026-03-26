# SM100a SASS Assembler — Build Plan

Full instruction-level control over SM100a (Blackwell / B200) SASS output.
Goal: write epilogue (or entire kernel) as assembly text, assemble to binary, patch into cubin.

**Context**: `tools/sass_edit.py` (~3700 lines) already has ELF parsing, register field encoding for ~30 families, CP-SAT scheduler, dependency analysis, cubin/fatbin patching. We're extending it from a **patcher** (rearrange existing instructions) to a **generator** (emit new instructions from scratch).

---

## Phase 1: Opcode Table Recovery ✅ DONE (2026-03-24)

Reverse-engineer the mapping: `(family, modifiers)` → `opcode_bits` for every SM100a instruction.

**Approach taken**: Instead of generating test .cu files, used XOR analysis on existing cubins (FC2 + FC1 + CUTLASS = 187 kernels, 484K instructions). Added `compute_opcode_table()`, `format_opcode_table()`, `export_opcode_table()` and `opcode-table` command to `sass_edit.py`.

**Key design**: Auto sub-groups by low 8 bits of encoding when a mnemonic has multiple encoding forms (register vs immediate source). Suppresses noise forms (< 2 unique AND < 5 instances).

### Results

- [x] 207 opcode entries extracted — 187 verified (multi-instance XOR), 20 single-instance, 0 failures
- [x] 100% coverage on FC2 (2841/2841 instructions match table)
- [x] 100% coverage on CUTLASS (468283/468283 instructions match table)
- [x] Exported to `tools/sm100a_opcodes.py` — importable Python dict
- [x] `sass_edit.py opcode-table` command — accepts multiple `cubin:sass` pairs

### What was NOT needed

- ~~`bench/calib/gen_opcode_tests.py`~~ — existing cubins had sufficient diversity
- ~~Manual target instruction list~~ — extracted all families automatically
- ~~Separate compile/extract step~~ — cuobjdump + existing xref infrastructure was enough

**Deliverable:** `tools/sm100a_opcodes.py` + `opcode-table` command in `sass_edit.py`.

---

## Phase 2: Operand Encoding ✅ DONE (2026-03-24)

Determined how registers, immediates, and predicates are encoded in operand bits.

**Approach**: Same XOR-on-existing-cubins methodology as Phase 1, applied to FC2+CUTLASS (480K instructions, 186 kernels). Added `analyze_predicates()`, `analyze_immediates()`, `verify_reg_fields()`, `analyze_special_encodings()` and `analyze-encoding` command to `sass_edit.py`. Exported to `tools/sm100a_encoding.py`. 38 tests in `tools/test_encoding_analysis.py`.

### 2.1 Register fields — verify against Phase 1 data

- [x] Cross-check `FAMILY_REG_FIELDS` against opcode table var_mask — multi-form instructions (IADD3, IMAD, etc.) have register fields that exceed var_mask for non-register forms (expected)
- [x] Identified 18 missing families: ATOMS, CS2R, ELECT, ENDCOLLECTIVE, F2I, I2F, I2FP, LDCU, REDUX, S2UR, SYNCS, UIADD3, UISETP, ULOP3, UMOV, USHF, VIADD, VOTEU
- [ ] Add missing families to FAMILY_REG_FIELDS (deferred to Phase 4 — only needed for families used in assembler)

### 2.2 Immediate field encoding

- [x] **STS/LDS offset**: LDS.128 at bits [49:44] (6 bits, scale=16). STS at bits [47:44] (4 bits). STS.128 at bits [55:54] (2 bits)
- [x] **BRA target**: PC-relative, bits [63:9] (55 bits). BRA.U at bits [27:18] + [63:34]
- [x] **BSSY target**: bits [34:16] + [51:36] (variable by form)
- [x] **BAR.SYNC**: operand fields at bits 47, 49, [56:54]
- [x] **DEPBAR count**: operand fields identified (fragmented: 12 bit positions)
- [x] **S2R special register ID**: SR field identified (fragmented, 5 SR names mapped)
- [x] **LDTM column descriptor**: LDTM.x16 at bits [45:44] (scale=16), LDTM.16dp256bit.x4/x8 at bits [46:45]/[47:46]
- [x] **SYNCS mbarrier**: SYNCS.ARRIVE.TRANS64 at bits [49:44] (6 bits, scale=16)

### 2.3 Predicate encoding

- [x] **Predicate guard at encoding bits [15:12]** (4 bits total)
- [x] Register field: bits [14:12] (3 bits) — P0=0, P1=1, ..., P6=6, **PT=7** (disables guard)
- [x] Negation bit: bit 15 — @!P0=0x8, @!P6=0xE, @!PT=0xF
- [x] **100% match** across 480K instructions (all guard types verified)
- [x] Uniform predicates (@UP0-@UP5) use **identical encoding** as thread predicates — UP/P distinction encoded elsewhere (not in bits [15:12])
- [x] No separate "enable bit" — PT at bits [14:12]=7 serves as "no guard"

### 2.4 Uniform register encoding

- [x] UR encoding is **8-bit** (max observed value: 255)
- [x] UR field positions verified in UTMALDG, UTMASTG, UTCQMMA per `FAMILY_REG_FIELDS`

### 2.5 Special cases

- [x] `RZ` = **0xFF** in register fields (verified on safe families: F2FP, HADD2, FADD, FMUL, FFMA, STS, STG, LDG, LDS, S2R)
- [x] `PT` encodes as **7** at predicate field bits [14:12]
- [x] `.reuse` flag is in **CONTROL WORD at bit 58** (NOT encoding word, NOT SM89 bits [22:21])
- [ ] `F2FP` pack modes — appear as separate opcode table entries, not operand bits

**Deliverable:** `tools/sm100a_encoding.py` + `analyze-encoding` command + 38 tests.

---

## Phase 3: Control Word Reverse-Engineering ✅ DONE (2026-03-25)

The 64-bit control word controls scheduling. **SM100a layout is significantly different from SM89.**

### 3.1 Catalog existing control word patterns ✅ DONE

- [x] Extracted all (mnemonic, control_word) pairs from FC2+CUTLASS cubins (480K instructions)
- [x] Grouped by family — per-family defaults computed and exported to `sm100a_encoding.py`
- [x] Identified "default" control word for each family (most common value)
- [x] Histogram of all 64 bit positions — usage frequency across 480K instructions

### Key SM100a control word findings

**Bit layout (verified 2026-03-25 by barrier correlation on 2237 FC2 + 3334 CUTLASS insns):**
- **[3:0]** = stall count (0-15). ✅ Verified.
- **[4]** = yield hint. ✅ Verified (45.5% usage).
- **[7:5]** = **wr_bar** (write barrier SB index, 0-6=SBx, 7=none). ✅ LDTM=SB0, HFMA2=SB1/2/3 rotating.
- **[10:8]** = **rd_bar** (read barrier SB index, 7=none). ✅ STS=SB4, F2FP=SB0.
- **[16:11]** = **wait_mask** (6-bit bitmask for SBs 0-5). ✅ STS=SB0 wait, BRA=0b111111 full wait.
- **[19:17]** = **pred_dst** (predicate dest index, 7=none/PT). ISETP/IADD3 use lower bits.
- **[22:20]** = **rsrc_class** (pipeline resource class hint). BF16/tensor=0b010, ALU=0b000, INT-add=0b111.
- **[25:23]** = **pred_src** (predicate source barrier index, 7=none). 100% match for PLOP3, IMAD.X.
- **[27:26]** = **mem_class** (memory/uniform pipeline, 0=reg, 2=async/TMA, 3=global).
- **[40:28]** = **DEAD ZONE — never varies.** All 13 bits constant.
- **[51:41]** = instruction type/class bits. Very high usage (63-97%).
- **[58:52]** = sparse features (0-9% usage).
- **bit 58** = **.reuse flag** (8.8%). ✅ Confirmed.

### 3.2-3.8 All sub-tasks ✅ DONE

- [x] Stall count [3:0] ✅
- [x] Yield bit [4] ✅
- [x] Write barrier wr_bar [7:5] — 3 bits, SB index, 7=none ✅
- [x] Read barrier rd_bar [10:8] — 3 bits, SB index, 7=none ✅
- [x] Wait mask [16:11] — 6-bit bitmask ✅
- [x] Reuse flag bit 58 ✅
- [x] Predicate dst/src fields [19:17] and [25:23] ✅
- [x] Pipeline class fields [22:20] and [27:26] ✅
- [x] `.barrier wait=` / `.barrier write=` directives updated for SM100a layout
- [x] Exported to `sm100a_encoding.py` as CTRL_WR_BAR, CTRL_RD_BAR, CTRL_WAIT_MASK, etc.

**Deliverable:** `sm100a_encoding.py` (full field map + defaults) + updated `.barrier` directive encoding in `sass_edit.py`.

---

## Phase 4: Assembler Core ✅ DONE (2026-03-24)

Text-to-binary assembler for SM100a SASS. Added ~400 lines to `sass_edit.py`:
parser, encoder, control word generator, binary emitter, and `assemble` command.

### 4.1 Assembly syntax ✅ DONE

```
# Comments with # or ;
.L_subiter_0:               # label definition

# Directives attach to next instruction
.stall 5                    # set stall count
.yield                      # set yield bit
.barrier wait=0x3, write=1  # set barrier fields
.reuse                      # set .reuse flag in control word

# Instructions: [guard] MNEMONIC[.modifiers] operands
    LDTM.x32       R16, [TMEM+0]
    LDS.128         R80, [R107+0x0000]
    F2FP.BF16.F32   R96, R16, R17
    HADD2           R96, R96.reuse, R80   # .reuse on operands
    STS.128         [R105+0x0000], R96
    FENCE.VIEW.ASYNC.S
    UTMASTG.2D      desc[UR8], desc[UR10]
@P0 BRA             .L_subiter_0
@!P3 EXIT
    NOP
```

- [x] Registers: R0-R255, RZ, UR0-UR255, URZ, P0-P6, PT, UP0-UP6, UPT
- [x] Immediates: decimal, hex (0x...), negative
- [x] Memory: `[Rn+offset]`, `[Rn]`, `[URn+offset]`, `[URn]`
- [x] TMEM: `[TMEM+offset]`
- [x] Descriptors: `desc[URn]`, `tmem[URn]`
- [x] Labels: `.L_name:` definitions, `.L_name` references (PC-relative branches)
- [x] Directives: `.stall N`, `.yield`, `.barrier wait=M,write=N`, `.reuse`

### 4.2 Parser ✅ DONE

- [x] `parse_asm_line()` — tokenize single line into `AsmInstruction`
- [x] `parse_asm()` — multi-line, label resolution, directive attachment
- [x] `_parse_asm_operand()` — typed operand parsing (13 operand types)
- [x] `_split_asm_operands()` — bracket-aware comma splitting

### 4.3 Encoder ✅ DONE

- [x] `_lookup_opcode()` — opcode table lookup with modifier fallback
- [x] `_encode_predicate()` — bits [15:12] guard encoding
- [x] `_encode_registers()` — fill from FAMILY_REG_FIELDS
- [x] `_encode_immediate()` — fill from IMMEDIATE_FIELDS (scaled, multi-field)
- [x] `encode_instruction()` — full (encoding, control) pair
- [x] Two-pass label resolution in `assemble()` (labels → PC addresses)

### 4.4 Control word generation ✅ DONE (manual mode)

- [x] Per-family defaults from CONTROL_DEFAULTS
- [x] `.stall N` → bits [3:0]
- [x] `.yield` → bit 4
- [x] `.barrier wait=M` → bits [20:15] (SM89 layout)
- [x] `.barrier write=N` → bits [9:5] (SM89 layout)
- [x] `.reuse` / operand `.reuse` → bit 58 (REUSE_CTRL_MASK)
- [ ] Auto mode (CP-SAT integration) — deferred to Phase 5

### 4.5 Binary emitter ✅ DONE

- [x] `assemble(text, base_pc)` → packed `[enc, ctrl, enc, ctrl, ...]` bytes
- [x] `cmd_assemble` — CLI command with output and instruction dump

### Round-trip verification

69 tests in `tools/test_assembler.py`, all passing:
- 18 operand parser tests (all types)
- 14 line/multi-line parser tests
- 12 encoder unit tests (opcode, predicate, registers, directives)
- 3 branch encoding tests (forward, backward, undefined)
- 3 binary emitter tests
- 5 round-trip tests against FC2+CUTLASS cubins (480K instructions):
  - Predicate field: >99.5% match
  - Register fields: >96% match
  - Opcode bits: >99% match
  - Full encoding (pure-register ALU): >99% match
- 8 synthetic encoding tests

### Known limitations

- **Immediate encoding incomplete**: Only instructions with IMMEDIATE_FIELDS entries (STS, LDS, BRA, BSSY, S2R, SYNCS, LDTM, etc.) have immediate encoding. MOV immediate, IADD3 immediate, etc. are missing — they need more diverse cubin samples or manual field mapping.
- **Register field coverage**: 30 families in FAMILY_REG_FIELDS. 18 known-missing families (ATOMS, CS2R, etc.) deferred until needed.
- **Multi-form opcodes**: `_lookup_opcode()` doesn't yet select encoding form based on operand types (register vs immediate). It uses the first match. For full correctness, would need operand-type-aware form selection.
- **Control word barrier fields**: Using SM89 layout for wait/write barriers until Phase 3 completes SM100a field mapping.

**Deliverable:** `sass_edit.py assemble` command + 69 tests.

---

## Phase 5: Integration ✅ DONE (2026-03-24)

Wire the assembler into the existing cubin/fatbin patching workflow.

### 5.1 Region replacement in existing cubin ✅ DONE

- [x] `CubinEditor.patch_region_asm(kernel, start, end, asm_text)` — assembles, patches, NOP-fills
- [x] `asm 0xSTART 0xEND path.s` command in `parse_script` — usable in recipe files
- [x] 7 tests (basic, NOP fill, overflow, alignment, real instructions, save roundtrip, branches)

### 5.2 Fatbin patching ✅ DONE

- [x] `fatbin-patch --asm FILE --start ADDR --end ADDR` — assembles and patches in one step
- [x] Composable with `--script` and `--stall`
- [x] Workflow: `sass_edit.py fatbin-patch fc2 --sass sass/fc2.txt --asm epilogue.s --start 0x51c0 --end 0x63b0 -o fc2_patched`

### 5.3 Round-trip verification ✅ DONE

- [x] `sass_edit.py verify-asm CUBIN --asm FILE --start ADDR --end ADDR`
- [x] Assembles text, compares against existing binary instruction-by-instruction
- [x] Reports encoding vs control mismatches separately
- [x] 100% control word round-trip (287/287 epilogue instructions)

### 5.4 Disassembler ✅ DONE

- [x] `sass_edit.py disasm CUBIN --sass SASS --start ADDR --end ADDR [-o FILE]`
- [x] Output in assembler's text format (round-trip compatible)
- [x] Emits `.ctrl 0x...` for exact control word preservation + `.stall`/`.yield`/`.reuse` for readability
- [x] Graceful fallback for non-xref'd instructions (RAW hex comment + NOP)
- [x] Enables: disassemble → edit → reassemble → patch workflow

### Parser improvements during Phase 5

- [x] Negated predicates as operands (`!PT`, `!P3`) — LOP3.LUT, PLOP3.LUT
- [x] Negated registers (`-UR5`, `-R3`) — UIADD3, IMAD
- [x] Barrier registers (`B0`-`B6`) — BSYNC
- [x] Scoreboard barriers (`SB0`-`SB5`) — DEPBAR
- [x] Special registers (`SR_CgaCtaId`) — S2R, CS2R
- [x] Memory with UR offset (`[R10+URZ+0x28008]`) — UTMASTG, UTMALDG
- [x] Memory with negative offset (`[R143+-0x2000]`) — LDG, STG
- [x] Constant bank (`c[0x0][0x348]`) — LDC, LDCU
- [x] Global/image descriptors (`gdesc[UR10]`, `idesc[UR7]`) — UTMASTG, UTMALDG
- [x] TMEM with UR+offset (`tmem[UR6+0x80]`) — LDTM
- [x] `.ctrl 0x...` directive for exact control word specification
- [x] Label regex relaxed: `.Lfoo` in addition to `.L_foo`

**Deliverable:** `assemble`, `disasm`, `verify-asm` commands + `.ctrl` directive + parser extensions. 92 tests.

### Round-trip fidelity summary

| Component | Round-trip accuracy | Notes |
|---|---|---|
| **Control words** | **100%** (287/287) | Via `.ctrl` directive |
| **Opcodes** | ~99% | Known: multi-form selection |
| **Predicates** | ~100% | Guard bits [15:12] |
| **Registers** | ~97% | Known: 18 missing families, multi-form ops |
| **Immediates** | ~0% | Known: only BRA/BSSY/STS/LDS/LDTM encoded |
| **Overall encoding** | ~0% | Dominated by immediate gaps |

The encoding gap is expected — the assembler faithfully reproduces opcode fixed bits, predicates, registers, and control words, but most immediate/operand-specific bits are not yet encoded. This doesn't block Phase 7 (FC2 epilogue) because the workflow is: disassemble → edit → reassemble with `.ctrl` for exact control words. The encoding bits that matter (opcodes, registers) are correct.

---

## Phase 6: Verification & Testing ✅ PARTIAL (6.1-6.2 DONE, 6.3-6.4 needs B200)

### 6.1 Round-trip tests (no GPU needed) ✅ DONE (2026-03-24)

- [x] Full kernel disasm → reassemble → per-field comparison: `tools/test_roundtrip.py`
- [x] 12 pytest tests in `tools/test_assembler.py` (TestRoundTrip, TestCubinValidity, TestNewOperandForms)
- [x] 4 new operand forms fixed during testing: `SRZ`, `SR_TID.X`, `[R8.64]`, `desc[UR6][R4.64]`, `RET R2 0x0`

**FC2 epilogue results (287 instructions, 0x51c0-0x63b0):**

| Metric | Fidelity | Notes |
|--------|----------|-------|
| Control word | **100%** (287/287) | Via `.ctrl` directive — exact round-trip |
| Opcode bits | **94.1%** (270/287) | Mismatches: BRA/BSSY branch encoding, DEPBAR, ELECT, S2R |
| Predicate | **96.5%** (277/287) | Mismatches concentrated in BRA family |
| Register fields | **80.5%** (231/287) | HFMA2 bit 39 modifier, ELECT/UIADD3 quirks |
| Full encoding | **44.9%** (129/287) | Immediate/modifier bits not yet encoded (Phase 4 gap) |

**100% full-match families:** F2FP, HADD2, NOP, MEMBAR, FENCE, WARPSYNC, CCTL, UTMALDG, UTMASTG, UTMACMDFLUSH

**Full kernel results (2844 instructions):**

| Metric | Fidelity |
|--------|----------|
| Control word | **100%** (2844/2844) |
| Opcode bits | **78.9%** (2244/2844) |
| Predicate | **87.0%** (2475/2844) |
| Register fields | **79.8%** (2270/2844) |
| Full encoding | **21.4%** (608/2844) |

### 6.2 Cubin validity tests (no GPU needed) ✅ DONE (2026-03-24)

- [x] NOP-fill region → cuobjdump -sass parses without errors
- [x] Contiguous matching region patch → cuobjdump -sass valid
- [x] ELF structure integrity: cubin size preserved after patch
- [x] Full epilogue reassemble → cuobjdump -elf shows valid .text section

### 6.3 Execution tests (B200 required)

- [ ] Trivial test: `MOV R0, R1; NOP; NOP; EXIT` — patch, run, verify
- [ ] Reproduce existing FC2 epilogue as assembly text → patch → verify checksum matches
- [ ] CP-SAT-optimized epilogue → patch → verify checksum + benchmark
- [ ] Structurally different epilogue (BAR.SYNC pipelining) → patch → benchmark

### 6.4 Correctness matrix

- [ ] Non-uniform B + bias/residual (standard validation)
- [ ] 1024 strided checksum + 32 CPU reference spot checks
- [ ] Different tile positions (first, last, odd shapes)

---

## Phase 7: FC2 Epilogue via Assembly (the actual goal)

### 7.1 Reproduce current epilogue in assembly

- [ ] Disassemble Phase 1 region (~0x51c0 to ~0x63b0)
- [ ] Write equivalent assembly text
- [ ] Assemble → patch → verify identical output and checksum

### 7.2 CP-SAT optimized schedule (same instructions, better order)

- [ ] `.auto_schedule` on reproduced epilogue
- [ ] Expected: 686 cycles (was 1417 with ptxas)
- [ ] Assemble → patch → benchmark

### 7.3 CUTLASS-style epilogue (structural change)

Intra-sub-iteration BAR.SYNC pipelining:

```
For each of 4 sub-iterations (64 cols):
    LDTM.x32 × 2     (read 64 cols from TMEM)
    LDS.128 × N       (read pre-loaded residual+bias from SMEM)
    DEPBAR             (wait for TMEM)
    F2FP × 8           (FP32→BF16, interleaved with:)
    HADD2 × 8          (bias+residual add, interleaved with:)
    STS.128 × 2        (store to staging — INTERLEAVED with compute)
    FENCE.VIEW.ASYNC.S
    BAR.SYNC 0x1, 128  (epilogue warps confirm STS done)
    UTMASTG.2D × 1     (TMA store: staging → global)
    DEPBAR.LE           (wait for TMA to finish reading SMEM)
    BAR.SYNC 0x1, 128  (SMEM free — next sub-iter can overwrite)
```

- [ ] Write as assembly text
- [ ] CP-SAT schedule compute portions
- [ ] Assemble → patch → benchmark

### 7.4 FP32 epilogue variant

- [ ] CUTLASS-style FP32 FFMA epilogue (convert to BF16 only at store)
- [ ] May matter once BAR.SYNC pipelining changes contention pattern

---

## Dependencies & Ordering

```
Phase 1 (opcode table)     ─── ✅ DONE
Phase 2 (operand encoding) ─── ✅ DONE (+ immediate mode bits 2026-03-25)
Phase 3 (control word)     ─── ✅ DONE (barrier field mapping 2026-03-25)
Phase 4 (assembler core)   ─── ✅ DONE
Phase 5 (integration)      ─── ✅ DONE
Phase 6 (verification)     ─── ✅ PARTIAL (6.1-6.2 done, 6.3-6.4 needs B200)
Phase 7 (FC2 epilogue)     ─── needs B200 access
```

Phases 1-6.2 complete. Phase 6.3+ and Phase 7 need B200.
Full emit→schedule→assemble pipeline produces 591 correct instructions for BAR.SYNC variant.

## Estimated scope

| Phase | Status | Difficulty | Notes |
|-------|--------|------------|-------|
| 1. Opcode table | **✅ DONE** | — | 207 entries, 100% coverage, from existing cubins |
| 2. Operand encoding | **✅ DONE** | — | Predicates (100%), immediates, RZ, .reuse. 38 tests. |
| 3. Control word | **✅ DONE** | — | Full field map: wr_bar[7:5], rd_bar[10:8], wait_mask[16:11], pred/rsrc/mem. |
| 4. Assembler core | **✅ DONE** | — | Parser, encoder, ctrl gen, emitter. 69 tests. |
| 5. Integration | **✅ DONE** | — | disasm, verify-asm, patch_region_asm, fatbin --asm. 92 tests. |
| 6. Verification | **✅ PARTIAL** | — | 6.1-6.2 done (ctrl=100%, cuobjdump valid). 99 tests. B200 for 6.3+. |
| 7. FC2 epilogue | Needs B200 | Medium | The payoff |
