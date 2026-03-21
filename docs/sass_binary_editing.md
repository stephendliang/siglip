# SASS Binary Editing — SM100a cubin patching

## Why this exists

ptxas STS scheduling is **immutable from source level** (5 approaches tested, all produce identical SASS). CUTLASS gets 8-12 instructions between each STS.128; ours gets 0-1. This gap accounts for most of the 19% FC2 deficit (1.452ms vs 1.225ms).

The only way to test whether STS interleaving is truly the bottleneck — without reverse-engineering CUTLASS's dependency graph or porting to EVT — is to patch the compiled binary directly.

## What we can do

`tools/sass_edit.py` is a working SM100a cubin editor. Verified capabilities:

| Operation | Status | Verified by |
|-----------|--------|-------------|
| Parse cubin ELF, extract kernel .text | Working | pyelftools + fc2.cubin |
| Read 128-bit instructions (encoding + control word) | Working | Byte-exact match with cuobjdump (2856/2856) |
| Cross-reference with cuobjdump SASS for mnemonics | Working | Automatic addr→mnemonic mapping |
| Swap two instructions | Working | cuobjdump confirms swapped binary |
| Reorder instruction range (arbitrary permutation) | Working | cuobjdump confirms reordered binary |
| Patch control word fields (stall counts, etc.) | Working | Diff confirms field changes |
| Batch edits via script file | Working | Multi-op scripts tested |
| Produce valid ELF that cuobjdump can disassemble | Working | All patched cubins parse clean |

## Binary layout (SM100a)

Each instruction = 16 bytes (128 bits), stored little-endian in ELF `.text` sections:

```
Offset  Content
[0:8]   Encoding (opcode + operands) — 64-bit LE
[8:16]  Control word — 64-bit LE
```

Control word bit layout (stall field **verified**, others from SM89 — may differ on SM100a):

```
[3:0]    Stall count (0-15)     ← VERIFIED (distribution: 1625 at 0, 774 at 15, rest 1-14)
[4]      Yield hint             ← plausible
[9:5]    Write barrier index    ← SM89 layout, unverified on SM100a
[14:10]  Read barrier index     ← SM89 layout, unverified on SM100a
[20:15]  Barrier wait mask      ← SM89 layout, unverified on SM100a
[22:21]  Register reuse         ← SM89 layout, unverified on SM100a
[63:23]  Upper bits (preserved on edits, purpose partially unknown)
```

## Tool usage

```bash
# View
python3 tools/sass_edit.py info fc2.cubin
python3 tools/sass_edit.py dump fc2.cubin --sass sass/fc2.txt --start 0x50f0 --end 0x5160
python3 tools/sass_edit.py sass fc2.cubin --start 0x50f0 --end 0x5160    # cuobjdump wrapper

# Edit
python3 tools/sass_edit.py swap fc2.cubin 0x50f0 0x5110 -o fc2_patched.cubin
python3 tools/sass_edit.py reorder fc2.cubin 0x5120 0x5160 0x5120,0x5140,0x5130,0x5150 -o fc2_patched.cubin
python3 tools/sass_edit.py patch fc2.cubin 0x5120 --stall 8 -o fc2_patched.cubin
python3 tools/sass_edit.py script fc2.cubin recipe.txt -o fc2_patched.cubin

# Verify & compare
python3 tools/sass_edit.py verify fc2.cubin --sass sass/fc2.txt
python3 tools/sass_edit.py diff fc2.cubin fc2_patched.cubin --sass sass/fc2.txt

# Generate loader stub for B200
python3 tools/sass_edit.py gen-loader fc2.cubin -o loader.cu
```

### Script format

```
# Comments start with #
swap 0xADDR_A 0xADDR_B            # swap two 128-bit instructions
stall 0xADDR VALUE                 # set stall count (0-15)
ctrl 0xADDR 0xRAW_CTRL_HEX        # replace entire control word
reorder 0xSTART 0xEND A,B,C,...    # permute instructions in range
```

## FC2 epilogue STS cluster

The target region. In the current binary, ptxas clusters STS.128 stores:

```
[50f0] STS.128 [R107+-0x2000], R92          ← STS #1
[5100] @!P3 LDC.64 R104, c[0x0][0x348]     ← 1 insn gap
[5110] LOP3.LUT R149, R4, ...                ← 1 insn gap
[5120] STS.128 [R108+-0x2000], R88          ← STS #2  ← 0 insn gap
[5130] STS.128 [R109+-0x2000], R96          ← STS #3  ← 0 insn gap
[5140] IMAD.IADD R110, R130, ...             ← 1 insn gap
[5150] STS.128 [R106+-0x2000], R100         ← STS #4
```

4 STS in ~7 instructions = terrible throughput utilization (need ~32 cycles between each STS.128 for full throughput, only getting 0-4).

CUTLASS has 8-12 PRMT/HFMA2/HADD2 instructions between each STS.128 pair.

## What patching can prove

**Hypothesis to test:** If we reorder the epilogue STS instructions to have 8+ compute instructions between each pair (by moving compute into the STS gaps), and adjust stall counts to maintain correctness, the kernel should approach CUTLASS's epilogue timing.

**If it works** (epilogue Phase 1 drops from 8430 to ~6000 cycles, wall time drops by ~100-200μs): proves the gap is purely STS scheduling. Next step: figure out how to get ptxas to produce this schedule from source (match CUTLASS dependency structure).

**If it doesn't work** (same wall time despite better STS interleaving): the gap is elsewhere — memory bandwidth, occupancy, or something else. Saves us from chasing STS scheduling further.

## Constraints and risks

**Safe operations** (within a straight-line basic block):
- Swapping instructions that don't cross branch boundaries
- Reordering within the epilogue compute→STS region
- Adjusting stall counts

**Unsafe operations** (require deep analysis):
- Moving instructions across branch/barrier boundaries
- Changing barrier wait/set indices (control word fields unverified)
- Moving instructions that have RAW dependencies (dest reg of one = src reg of next)

**Must preserve:**
- Total instruction count (section size unchanged)
- Barrier semantics (don't move instructions past their barrier waits/sets)
- Data dependencies (a consuming instruction must still execute after its producer)

**Not required for FC2 epilogue reorder:**
- Branch target fixup (epilogue is straight-line)
- Relocation updates (fc2.cubin has zero relocations)

## Running patched cubins on B200

1. Generate standalone cubin: `nvcc --cubin -arch=sm_100a -O3 fc2.cu -o fc2.cubin`
2. Patch it: `python3 tools/sass_edit.py script fc2.cubin recipe.txt -o fc2_patched.cubin`
3. Load via driver API: Use `cuModuleLoad` + `cuModuleGetFunction` to load the patched cubin
4. The gen-loader subcommand produces a skeleton — fc2.cu's `main()` argument setup needs to be ported

Alternative: replace the embedded cubin inside the fc2 executable's .nv_fatbin section. More complex but avoids writing a loader.

## Calibration benchmarks (bench/calib/)

Complementary to the editor — empirically measures SM100a instruction costs:

```bash
./bench/calib/run.sh              # generate, build, SASS-verify, run all 3 suites
./bench/calib/run.sh tput         # ILP sweep: 18 instructions × 5 ILP levels = 90 kernels
./bench/calib/run.sh lat          # latency chains: 16 instructions
./bench/calib/run.sh conflict     # NxN conflict matrix: C(18,2) = 153 pairwise tests
```

- `bench/calib/instruction_db.py` — 18 SM100a instruction families (FP32, BF16, INT, special, memory)
- `bench/calib/gen_kernels.py` — generates .cu files from the database
- Results feed into the SASS analysis tool's latency table and a future constraint-programming scheduler

## Control word reverse engineering (TODO)

Stall count bits [3:0] are verified. The barrier fields appear to have shifted from the SM89 layout. To fully crack SM100a control words:

1. Use the calibration throughput benchmarks (simple known instruction sequences)
2. Vary stall counts in the binary, measure cycle impact on B200
3. Identify barrier fields by looking at known synchronization patterns (mbarrier wait/arrive, scoreboard barriers around LDTM/UTCBAR)
4. Build a SM100a control word decoder that matches cuobjdump's implicit decode
