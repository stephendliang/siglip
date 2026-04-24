fc2_w3x pure-PTX hand-rewrite — strategy document
===================================================

Goal: deterministic control over SASS emission for `fc2_w3x`, hand-authored
as `.ptx` fed to `nvcc -arch=sm_100a`. Target architecture: match rank-1
(`nvjet_sm100_qqtst_128x256_128x6_2x1_2cta_v_bz_bias_TNT`, 1.046 ms baseline,
we currently beat it at 1.007 ms) in uniform-datapath usage while preserving
our +39 µs lead.

TL;DR of the strategy research
------------------------------

- `nvcc -arch=sm_100a` accepts `.ptx` natively (verified). Single `.ptx`
  file is a viable deliverable; we just need a small host `.cu`/`.cpp`
  for CUtensorMap setup, kernel launch, and timing.
- **PTX has no uniform-register type.** `.reg .b32` / `.reg .u32` /
  `.reg .u64` are all ptxas-lowering hints at best. The R vs UR
  partition is ptxas's call, period. Verified by hand-authoring PTX that
  mixes a warp-uniform value and seeing ptxas still emit `IADD3 R0, PT,
  PT, R0, UR4, RZ` — even though our handwritten PTX referenced the
  uniform special register directly.
- **SASS hand-write is NOT an option on sm_100a.** There is no public
  assembler. `tools/sass_edit.py` does binary patching for narrow
  intra-warp reorder experiments; it does not author SASS from scratch.
  "SASS or PTX" → PTX.
- **The grievances doc overcounts the runtime problem.** Current
  fc2_w3x (with `#pragma unroll 1` on W4+W5 K-loops) has R2UR=32,
  ELECT=10, BSSY=14, UIADD3=7 — *already close* to rank-1's 3/0/0/117.
  Full grievances-doc numbers (R2UR=495, ELECT=292) are from the
  fully-unrolled build; rolling the K-loop collapsed them. PTX rewrite
  upside is a further 5–10 µs *at most* in this regime.

The strategy below assumes the user wants deterministic SASS
regardless of whether the final delta beats noise — i.e., the exercise
has engineering-artifact value independent of perf.

1. What rank-1 actually does at SASS level
------------------------------------------

Read via `rank1.sass` (dumped from `libcublasLt.so.13`). 1650 lines; one
kernel function covering setup, one of four warp-role paths, and the
persistent loop for each. Key addresses called out below.

### 1a. Single-lane dispatch

Rank-1 does NOT use `elect.sync` to pick the one lane. The pattern is
**uniform-predicate gating via UR**. Two distinct mechanisms:

**UR-address-indexed `SYNCS.PHASECHK.TRANS64.TRYWAIT`** (sass +1888):

```sass
SYNCS.PHASECHK.TRANS64.TRYWAIT P1[UR9],R5       ; poll mbarrier, UR9 = mbar addr
@!P1 NANOSLEEP.SYNCS  0xc350                    ; sleep if not ready
@!P1 SYNCS.PHASECHK.TRANS64 P1[UR9],R5
```

This is a broadcast — all lanes of the warp execute, with the mbarrier
index in a uniform register. No ELECT/BSSY wrapper.

**UGETNEXTWORKID.BROADCAST** (sass +22592):

```sass
@!UP0 UGETNEXTWORKID.BROADCAST [UR8][UR9]       ; one lane pulls next tile,
                                                ; result broadcast to all lanes as UR
```

Works exactly because the whole warp can execute together when the
operation is warp-collective — NVIDIA's dispatcher broadcasts the
result as a UR.

Neither uses ELECT/BSSY. Rank-1's `ELECT=0` count is achieved by
*choosing warp-collective ops whose hardware semantic is already
"execute once, broadcast"*.

### 1b. Descriptor advance

TMA descriptors are held in UR pairs (UR54/UR55, UR42/UR44, ...)
computed once at startup (sass +19136):

```sass
UIADD3.64 UR42,UPT,UPT,UR54,0x480,URZ    ; descriptor for A-tile base
UIADD3.64 UR44,UPT,UPT,UR54,0x580,URZ
UIADD3.64 UR46,UPT,UPT,UR54,0x8c0,URZ
...
```

Per-K-iter advances are also UR→UR (sass +19312):

```sass
UIADD3 UR8,UPT,UPT,UR8,0x8000,URZ        ; advance A desc by one K-chunk
UIADD3 UR16,UPT,UPT,UR16,0x8000,URZ      ; advance B desc
UIADD3 UR30,UPT,UPT,UR30,0x8,URZ         ; advance mbar phase counter
```

Zero R-side touches. Contrast with our w3x, where descriptor advance
lives inside the `asm volatile` MMA block:

```
"add.s64 da, da, 2;\n\t"
"add.s64 db, db, 2;\n\t"
```

with `da`/`db` declared as `.reg .u64` *inside* the PTX block but fed
from `"l"(desc_a)` / `"l"(desc_b)` R-type inputs. ptxas sees this as an
R operand and emits `IADD3 R..., R..., UR..., RZ` for the add. That's
the mechanism behind our residual R2UR count.

### 1c. MMA issue

K-loop body (sass +24432) — this is the "inner" loop, one pass through
the 6-stage pipeline window:

```sass
@P2     LOP3.LUT    R5,R5,RZ,RZ,0xf,!PT           ; flip phase bit (scalar)
@!P0    SYNCS.PHASECHK.TRANS64.TRYWAIT P1[UR8],R5 ; wait next stage
        UTCQMMA.2CTA  gdesc[UR16],gdesc[UR18],tmem[UR20],tmem[UR14],idesc[UR15],!UP1
        UIADD3        UR16,UPT,UPT,UR16,0x2,URZ   ; advance desc
        UIADD3        UR18,UPT,UPT,UR18,0x2,URZ
        UTCQMMA.2CTA  gdesc[UR16],gdesc[UR18],tmem[UR20],tmem[UR14],idesc[UR15],UPT
        UIADD3        UR16,UPT,UPT,UR16,0x2,URZ
        UIADD3        UR18,UPT,UPT,UR18,0x2,URZ
        UTCQMMA.2CTA  gdesc[UR16],gdesc[UR18],tmem[UR20],tmem[UR14],idesc[UR15],UPT
        ...  ; 4 UTCQMMAs total per ki
        UMOV          UR9,UR8
        UIADD3        UR10,UPT,UPT,UR10,0x8,URZ
        UTCBAR.2CTA.MULTICAST [UR10],URZ,UR22       ; tcgen05.commit equivalent
        UISETP.NE.AND UP1,UPT,URZ,URZ,UPT
        BRA.U         !P0,!UP2,<top>               ; back edge, uniform branch
```

**Zero ELECT/BSSY scaffold around UTCQMMA.** The instruction is
inherently warp-collective; the whole warp issues and the hardware
retires once per MMA. UR-indexed descriptors get lifted to UIADD3
because the PTX (of rank-1) used a UR operand slot.

### 1d. Epilogue

SASS +3408 onward. One 16-row store pass per iteration of an outer
loop. Pattern per row group:

```sass
LDTM.16dp256bit.x4 R16,tmem[UR8]          ; 16 regs of fp32 acc
LDTM.16dp256bit.x4 R32,tmem[UR8+0x100000] ; 16 more
FFMA2 R16,R6.F32,R16.F32x2.HI_LO,RZ.F32   ; packed fp32x2 scale
FFMA2 R18,R6.F32,R18.F32x2.HI_LO,RZ.F32
... 8 FFMA2 ...
FADD R16,R16,R12                          ; add bias
FADD R17,R17,R12
... 16 FADD ...
F2FP.BF16.F32.PACK_AB R16,R17,R16         ; packed fp32→bf16
... 16 F2FP ...
STSM.16.MT88.4 [R10+UR32],R16             ; stmatrix 16-lane cooperative
...
FENCE.VIEW.ASYNC.S
BAR.SYNC.DEFER_BLOCKING 0x0,0x80           ; warpgroup sync
@UP2 UTMASTG.3D [UR32][UR26]              ; TMA store
UTMACMDFLUSH
```

Key observations:
- **Packed math throughout**: FFMA2 + F2FP.PACK_AB — 2 fp32 lanes per op.
- **STSM**: 16-lane cooperative store. Our w3x does 24 plain STS per tile.
- **UR-indexed TMA store** on UP2 — UP2 is the warp-ID gate (one of
  4 warps wins). No explicit ELECT.

### 1e. Scheduler / persistent loop

Rank-1 has a persistent scheduler. Sass +22208 branches on R2 (tid)
into one of three warp roles:
- `R2 < 0xa0` (tid < 160) → "main/epi" role (5 warps)
- `R2 < 0xc0` (tid < 192) → "scheduler/producer" role (1 warp)
- `R2 < 0xe0 && crank==1`  → other CTA-specific paths
- Otherwise → early EXIT (warps 7+ on CTA0, W6+W7 on CTA1 in w3x-speak)

The producer warp (sass +22208) uses `UGETNEXTWORKID.BROADCAST` against
a system-wide atomic counter to pull the next tile index, broadcast as
a UR to the whole warp. All four UTCQMMAs in a K-iter then consume
that UR-held tile offset via their UR descriptor fields.

Tile boundary includes `UCGABAR_WAIT` (cluster barrier wait) +
`ACQBULK` fence, then the producer arrives on consumer's mbarrier
via `SYNCS.ARRIVE.TRANS64.RED` (UR-indexed).

### 1f. The "how rank-1 gets zero ELECT" answer in one sentence

**Rank-1 never needs `elect.sync` because every place w3x has
`if (lane == 0) { asm ... }` is replaced by an inherently-uniform op
(`UGETNEXTWORKID`, `SYNCS.PHASECHK`, `UTCATOMSWS.FIND_AND_SET`,
`UTCBAR.MULTICAST`, `UTMASTG`) whose PTX form is not
lane-gated — it's warp-collective.**

We have the opposite pattern: our TMA load PTX is gated
`if (elect)` because `cp.async.bulk.tensor` in PTX *looks* per-lane.
Ptxas correctly forbids all 32 lanes from issuing the same TMA and
emits the ELECT/BSSY/BSYNC to pick one. Rank-1's SASS-level
`UTMALDG.2CTA [UR8][UR54]` is issued by *all 32 lanes* with a
uniform-register address, and the hardware collapses it into one
transaction.

**This is the single most important finding.** PTX `cp.async.bulk.tensor`
emits `UTMALDG` but has an implicit per-lane semantic; ptxas wraps it
in an ELECT scaffold. Whatever rank-1's source is, it must be using a
different PTX construct (or a newer ISA variant) that is warp-collective.

2. PTX ISA capabilities for uniform-register control
----------------------------------------------------

Research findings from the local CUDA 13.2 install (`/opt/cuda/`,
`/opt/cuda/targets/x86_64-linux/include/`):

### 2a. Does PTX have `.reg .ur`?

**No.** PTX has `.reg .b32`, `.reg .u32`, `.reg .s32`, `.reg .b64`,
etc. — all are virtual scalar registers. ptxas decides R-vs-UR at
lowering. Verified by hand-authoring `test_ur_ptx.ptx`:

```ptx
mov.u32 %r2, %cluster_ctarank;      ; warp-uniform value
add.s32 %r3, %r2, %r1;
mul.lo.s32 %r4, %r3, 128;
mov.u32 %r5, %tid.x;
mad.lo.s32 %r6, %r5, 4, %r4;        ; per-thread, uses uniform %r4
```

Output SASS:
```
S2UR UR4, SR_CgaCtaId          ; crank → UR (automatic)
LDC R0, c[0x0][0x388]           ; kernel arg n loaded to R (not UR)
IADD3 R0, PT, PT, R0, UR4, RZ   ; R0 = n + crank — but stored in R,
                                ; not UR, because add needed R operand.
```

So even if we *start* with UR-dominant source, ptxas may still pull
values into R for computation convenience. The decision ptxas makes is
based on (a) how the special register arrived (SR_CgaCtaId → S2UR
automatically), (b) whether the consumer is a UR-capable instruction,
(c) whether any R-side value needs to join the compute.

### 2b. Instructions that force UR placement (PTX source)

These PTX operations are the dependable "promote to UR" hints:
- `mov.u32 %reg, %cluster_ctarank;` → S2UR (always)
- `mov.u32 %reg, %ctaid.x;` → S2UR when consumed uniformly
- `mov.u32 %reg, %smid;` → S2UR
- `mov.u32 %reg, %nctaid.x;` → LDC → UR (usually)
- `redux.sync.{min,max,and,or,xor,add}.u32` → UR result via URF writeback
- `vote.sync.ballot.b32` → UR result

### 2c. Do special registers get UR automatically?

Yes for `%cluster_ctarank`, `%ctaid.*`, `%smid`, `%nctaid.*`. Verified:
all of these lower to `S2UR` directly when the value is read once.

### 2d. u32 vs b32 typing trick

No measurable effect. Ptxas treats `.u32` and `.b32` identically in
register-class selection for sm_100. Tested with a small variant — SASS
output byte-identical.

### 2e. How CUTLASS encourages UR placement

Key technique: **use `cute::elect_one_sync()` for "one lane does
something"**. Implemented as:

```cpp
asm volatile(
  "{\n"
  ".reg .b32 %%rx;\n"
  ".reg .pred %%px;\n"
  "     elect.sync %%rx|%%px, %2;\n"
  "@%%px mov.s32 %1, 1;\n"
  "     mov.s32 %0, %%rx;\n"
  "}\n"
  : "+r"(laneid), "+r"(pred)
  : "r"(0xFFFFFFFF));
```

Then the following block is gated by the predicate. Ptxas emits this
as `ELECT` / `BSSY` / `BSYNC` — the same scaffold we see in w3x.
So **CUTLASS hits exactly the same scaffold** — it doesn't avoid it.

The place CUTLASS *does* stay UR-clean is the UMMA dispatcher in
`cute/arch/mma_sm100_umma.hpp`, which passes descriptor values through
`uint64_t` locals initialized once outside any per-thread branching.

**Bottom line: PTX has no dependable source-level knob for UR
placement beyond special registers and collective ops.** PTX rewrite
will not beat ptxas at R-vs-UR partition. Rank-1's UR dominance comes
from *not having single-lane gates in the first place* — its PTX source
(NVIDIA internal) uses warp-collective constructs that w3x can't easily
emit via the public PTX ISA.

3. Realistic deliverable format
-------------------------------

### 3a. `.ptx` files and nvcc

**Verified**: `nvcc -arch=sm_100a -cubin foo.ptx -o foo.cubin` works.
Also `nvcc -arch=sm_100a foo.ptx host.cu -o exe` for mixed host+device.

The `.ptx` needs:
```
.version 9.2              ; or whatever local CUDA provides
.target  sm_100a
.address_size 64

.visible .entry fc2_w3x_kernel(
    .param .align 64 .b8 tma_a[128],      ; CUtensorMap is 128 bytes
    .param .align 64 .b8 tma_b[128],
    .param .align 64 .b8 tma_c[128],
    .param .u64 .ptr .align 2 d_bias,
    .param .u64 .ptr .align 2 d_C,
    ...
)
.maxntid 192, 1, 1
.reqnctapercluster 2, 1, 1
.explicitcluster
{
    ; ... body ...
}
```

### 3b. Host-side code unchanged

`cuTensorMapEncodeTiled` (driver API) for the 3 `CUtensorMap` objects,
`cudaLaunchKernelEx` with cluster dims, kernel-arg packing via
`CUlaunchConfig.attrs` — all stays in the existing `fc2_w3x.cu` main()
or gets split off into a small `host.cu`. No issue here.

### 3c. SASS hand-write is NOT realistic

- sm_100a has no public assembler.
- `cuobjdump --dump-sass` dumps, but there is no round-trip assembler
  in the toolchain.
- `tools/sass_edit.py` (~5500 LoC) does binary patching for **narrow,
  constrained experiments** (intra-warp instruction reorder with
  CP-SAT scheduling; it tracks stall bits 53-55, barrier fields, etc.).
  It is not an assembler — it cannot create new instructions, only
  reorder and tweak existing ones within careful bounds. See
  `docs/sass_binary_editing.md`.

Deliverable: **single `fc2_w3x.ptx` file, plus keep the existing
`fc2_w3x.cu` main() as-is renamed to `fc2_w3x_host.cc` (or similar)**.
New Makefile rule compiles the `.ptx` to `.cubin` and links.

4. Proposed PTX structure
-------------------------

### 4a. Top-level skeleton

```ptx
.version 9.2
.target  sm_100a
.address_size 64

; External globals for tensor maps in param space handled by CUTensorMap
; encoding on host; no declarations needed here.

.visible .entry fc2_w3x_kernel(
    .param .align 64 .b8 fc2_w3x_kernel_param_0[128],   ; CUtensorMap tma_a
    .param .align 64 .b8 fc2_w3x_kernel_param_1[128],   ; CUtensorMap tma_b
    .param .align 64 .b8 fc2_w3x_kernel_param_2[128],   ; CUtensorMap tma_c
    .param .u64        fc2_w3x_kernel_param_3,          ; d_bias
    .param .u64        fc2_w3x_kernel_param_4,          ; d_C  (unused)
    .param .u64        fc2_w3x_kernel_param_5,          ; d_dbg_prof
    .param .u64        fc2_w3x_kernel_param_6,          ; d_dbg_prof_ki
    .param .u64        fc2_w3x_kernel_param_7,          ; d_dbg_prof_tile
    .param .u64        fc2_w3x_kernel_param_8           ; d_dbg_prof_w5
)
.maxntid 192, 1, 1
.reqnctapercluster 2, 1, 1
.explicitcluster
{
    ; ================ REGISTER DECLARATIONS ================
    ; (Virtual; ptxas assigns physical R/UR. Generous counts;
    ;  ptxas will prune unused.)
    .reg .pred %p<64>;          ; predicates, include %pu* style UR hints
    .reg .b32  %r<256>;         ; per-thread 32-bit
    .reg .b64  %rd<64>;         ; per-thread 64-bit (descriptors, ptrs)
    .reg .b16  %rh<64>;         ; bf16 values
    .reg .f32  %f<128>;         ; fp32 accumulators

    ; ================ SHARED MEMORY LAYOUT (constants) ================
    ; These are byte offsets; see fc2_w3x.cu OFF_* macros.
    ; OFF_AB = 0; OFF_OUT = N_STAGES * STAGE_BYTES = 6*32768 = 196608;
    ; OFF_BIAS = OFF_OUT + NUM_EPI_STAGES*SUBPASS_BYTES = 196608+16384 = 212992;
    ; etc. — hardcode as immediates.

    .shared .align 128 .b8 smem[228352];  ; full SMEM budget

    ; ================ SETUP ================
    ld.param.u64 %rd0, [fc2_w3x_kernel_param_3];     ; d_bias
    mov.u32      %r0, %cluster_ctarank;               ; → UR (auto via S2UR)
    mov.u32      %r1, %tid.x;                         ; → R
    shr.u32      %r2, %r1, 5;                         ; warp_id
    and.b32      %r3, %r1, 31;                        ; lane

    ; ...
    ; persistent loop, warp dispatch, etc.
    ; ...

    exit;
}
```

### 4b. Register budget

Approximate worst-case R usage in current w3x (from SASS):
- Epilogue warps (W0-W3): ~120 R per thread (acc + bias + STSM bufs)
- TMA warp (W4): ~16 R (tile counters, K-loop state)
- MMA warp (W5): ~24 R (phase counters, accum_flag, mbar addrs)

Total virtual regs: plenty of headroom with `%r<256>`.

UR estimate (ptxas decides): ~40 UR per warp depending on descriptor
layout. w3x currently uses UR4, UR5, UR8-UR34 etc. (roughly 30 UR
simultaneously live in W5's K-loop).

### 4c. W5 K-loop sample in PTX (focus: explicit UR-friendly forms)

This is the most important section — showing how we'd write the
MMA-issuing K-loop to maximize UR carryover across the loop boundary.

```ptx
; --- W5 K-loop, pre-loop setup ---
; Load descriptor base addresses (uniform across warp).
; Host has set up constant memory c[0] with descriptor pointers;
; we load once per warp.

ld.param.u64 %rd10, [fc2_w3x_kernel_param_0];   ; &tma_a (CUtensorMap ptr)
cvta.global.u64 %rd10, %rd10;

; Pre-materialize 6 (N_STAGES) SMEM descriptors. All uniform within warp.
mov.u64 %rd20, OFF_AB_S0_DESC;                  ; hardcoded immediate
mov.u64 %rd21, OFF_AB_S1_DESC;
; ... etc

; Phase-tracking scalars (uniform):
mov.b32 %r40, 0;                                ; tma_full_phase[0]
mov.b32 %r41, 0;                                ; tma_full_phase[1]
; ... six phases

; buf = tt & 1:
mov.b32 %r50, 0;                                ; tt counter

$L_w5_tile_loop:
    ; Compute lin_tile = cluster_id + tt*num_clusters
    ; (cluster_id is uniform; tt is uniform)
    ; Skip dispatch logic for brevity — assume tile_m/tile_n in %r60/%r61
    
    and.b32 %r55, %r50, 1;                      ; buf = tt & 1
    
    ; --- K-loop (rolled, #pragma unroll 1 equivalent) ---
    mov.b32 %r70, 0;                            ; ki counter
    $L_w5_k_loop:
        ; Wait on tma_full mbarrier for stage s = ki % N_STAGES
        ; Uniform-address mbar wait — compiles to SYNCS.PHASECHK
        rem.s32 %r71, %r70, 6;                  ; s
        mul.lo.s32 %r72, %r71, 8;               ; offset in mbar array
        add.s32 %r73, %r72, MBAR_TMA_FULL_BASE;
        ; <mbar_wait PTX> — emits SYNCS.PHASECHK, no ELECT because
        ; mbarrier.try_wait.parity PTX is warp-collective.
        
        mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 %p10, [%r73], 0;
        @!%p10 bra $L_w5_k_loop;                ; spin on failure
        
        ; 4x UTCQMMA fold — single PTX asm body so ptxas can't split
        ; descriptor advance out.
        ; Note: MMA descriptors already in UR (%rd20-25); the
        ; %rd80 holds the current pair, +=2 advances per fold.
        
        ; Select descriptors based on s:
        ; (in practice, index into desc_a_arr[s] with a UR-lift)
        
        ; MMA block — 4 back-to-back tcgen05.mma.cta_group::2
        ; with p_init for first, p_acc for rest, then commit:
        {
            .reg .b32 %accum_flag;
            .reg .b64 %da, %db;
            mov.b32 %accum_flag, <acc_flag_expr>;
            mov.b64 %da, <desc_a>;
            mov.b64 %db, <desc_b>;
            
            tcgen05.mma.cta_group::2.kind::f8f6f4 
                [tmem_addr], %da, %db, IDESC, {0,0,...,0}, p_init;
            add.s64 %da, %da, 2;
            add.s64 %db, %db, 2;
            tcgen05.mma.cta_group::2.kind::f8f6f4 
                [tmem_addr], %da, %db, IDESC, {0,0,...,0}, p_acc;
            add.s64 %da, %da, 2;
            add.s64 %db, %db, 2;
            tcgen05.mma.cta_group::2.kind::f8f6f4 
                [tmem_addr], %da, %db, IDESC, {0,0,...,0}, p_acc;
            add.s64 %da, %da, 2;
            add.s64 %db, %db, 2;
            tcgen05.mma.cta_group::2.kind::f8f6f4 
                [tmem_addr], %da, %db, IDESC, {0,0,...,0}, p_acc;
            tcgen05.commit.cta_group::2.mbarrier::arrive::one
                .shared::cluster.multicast::cluster.b64 [%tma_empty], 0x3;
        }
        
        add.s32 %r70, %r70, 1;
        setp.lt.s32 %p20, %r70, 24;             ; K_ITERS
        @%p20 bra $L_w5_k_loop;
    
    ; --- End of K-loop, commit MMA → TMEM ready mbar ---
    ; (also emits tcgen05.commit for the ready signal)
    
    add.s32 %r50, %r50, 1;
    setp.lt.s32 %p30, %r50, <tiles_per_cluster>;
    @%p30 bra $L_w5_tile_loop;

exit;
```

**Key difference from C++ version**: the `tcgen05.mma` operands
`%da`/`%db` are declared with `.reg .b64` inside the PTX block, but
*fed from outer-scope `%rd20-25` values that are materialized via
`mov.u64 %rd, immediate`*. Ptxas is MORE likely to keep these in UR
because it sees the full lifetime inside a single asm section.

The current w3x code feeds them as `"l"(desc_a)` from a C++ variable
— ptxas lowers the C++ local to R unless it can prove it's uniform
across the whole warp, and the proof fails because the variable is
declared outside any `#pragma uniform` context.

**This is the realistic upside of the rewrite.** Not "zero R2UR" —
rank-1's 3 R2URs aren't actually zero either — but "25-40% fewer
R2UR" by keeping descriptor arithmetic entirely within ptx-visible
UR-friendly lifetimes.

### 4d. W4 TMA dispatch — zero-ELECT attempt

```ptx
; --- W4 per-ki TMA issue, no elect ---
; Key: use cp.async.bulk.tensor.{g2s,s2g} with .cta_group::2
; and ensure the PTX uses UR-compatible forms.

; The PTX for TMA load:
cp.async.bulk.tensor.2d.shared::cluster.global.tile
    .mbarrier::complete_tx::bytes.cta_group::2
    [%r_a_dst], [%rd_a_desc, {%r_c0, %r_c1}], [%r_mbar];
```

This PTX form currently emits ELECT/BSSY in our codegen. The open
question: does PTX have a **warp-broadcast** variant? PTX ISA 8.7
introduces `cp.async.bulk.tensor.2d.shared::cluster.global.tile.bulk_group`
which may be collective. If yes, use it. If not, the ELECT scaffold is
unavoidable on the current ISA.

**Recommendation**: empirically test both forms during the rewrite. If
no collective form exists, accept the scaffold — rank-1's PTX is
NVIDIA-internal and may use opcodes (`UTMALDG.2CTA`-emitting forms)
not exposed to public ISA.

5. Risks and open questions
---------------------------

1. **R2UR floor is not zero in PTX.** Ptxas is the compiler on both
   paths; PTX source only changes what it sees, not how it lowers.
   Best-case realistic: match rank-1's ~3–10 R2UR (already
   close: w3x has 32 with unroll=1). Worst-case: the asm-block
   boundary elision we're targeting is ptxas-independent of source
   form. **Payoff could be 0 µs.**

2. **`cp.async.bulk.tensor` scaffold may be unavoidable.** If no
   warp-collective PTX form exists on sm_100a public ISA, ELECT=10
   per tile × 147 tiles = 1470 extra dynamic insts — same as today.
   Rank-1's zero ELECT may depend on internal-only PTX.

3. **CUtensorMap setup still needs driver API.** `cuTensorMapEncodeTiled`
   can only be called from host C/C++. Our `fc2_w3x.cu`'s `main()`
   stays. Possible deliverable form: `fc2_w3x.ptx` (device) +
   `fc2_w3x_host.cc` (host), linked. Makefile change required.

4. **PTX-level mbarrier/TMEM quirks.** Our `tcgen05.alloc`,
   `tcgen05.commit`, `mbarrier.*`, and cluster-barrier asm blocks use
   the PTX 8.7 opcodes correctly in the `.cu` form. The open question
   is whether concatenating them into one giant `.ptx` file (no
   intervening C++ scopes) breaks anything in ptxas's lowering of
   `fence.proxy.async` ordering. Validation required.

5. **Debugging is strictly harder.** No `printf` from pure PTX (unless
   we emit the `vprintf` ABI call ourselves). No line-number
   attribution from `nvcc -lineinfo`. No `compute-sanitizer` source
   mapping. Any bug → `cuobjdump --dump-sass` + manual reasoning.

6. **Maintenance burden: every PTX change requires SASS verification.**
   We'd commit a regression-detection rule: `cuobjdump --dump-sass
   fc2-w3x | grep -c <key opcodes>` must match a locked manifest, or
   CI fails. Without that, a silent PTX-vs-ptxas version drift could
   regress us to pre-rewrite SASS.

7. **PTX version lock.** ISA version (`.version 9.2` today on CUDA
   13.2) advances. On CUDA 14+, the same source might lower
   differently. Perf-sensitive PTX deserves a version guard and a
   "golden SASS" artifact in the repo.

### What I'd do first (to de-risk)

Before writing the full `.ptx`, do a **targeted experiment** on the
current `fc2_w3x.cu` that *proves* PTX rewrite can move the needle:

1. Extract just the W5 K-loop body. Rewrite as pure PTX via
   `__device__ __forceinline__` with a single giant `asm volatile`
   block and no C++ scalars crossing the boundary.
2. Compile, compare SASS: R2UR / ELECT / BSSY counts for that
   function's range.
3. Benchmark fc2-w3x with the replaced W5. Any µs delta?

If step 3 shows 0 µs delta, the full rewrite is a SASS-artifact
exercise, not a perf project. Proceed or abandon with that data.

If step 3 shows ≥5 µs, do the full rewrite; the budget is defensible.

Summary
-------

**PTX rewrite is technically viable** (nvcc accepts `.ptx`, host-side
glue stays C++). **It is not a proven perf lever**: ptxas owns R vs UR
placement, not source form. The three SASS patterns that would
genuinely close the rank-1 gap (UR-indexed TMA loads, scaffold-free
single-lane dispatch, UR-native descriptor arithmetic) depend more on
the availability of warp-collective public PTX opcodes than on
authoring form. Recommend the de-risk experiment before committing to
the full rewrite.

Files to produce, if committed:
- `fc2_w3x.ptx`   (~1000–1500 lines)
- `fc2_w3x_host.cc`  (current main() renamed, trimmed)
- `Makefile` rule  (nvcc of .ptx + .cc)
- `tools/verify_sass.sh`  (regression harness)
