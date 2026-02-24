# SigLIP2 Vision Encoder — Persistent Megakernel Plan

Target: B200 (148 SMs, cluster-of-4) / B300 (160 SMs, cluster-of-16)
Model: google/siglip2-base-patch16-224 (86M vision encoder)
Precision: MXFP8 (E4M3 values, E8M0 block scales, block size 32)
ISA: tcgen05 WGMMA, TMA, cooperative grid launch
Build: cross-compile on CPU-only VPS, run on target GPU

## Architecture Quick Reference

- Input: [B, 3, 224, 224] → Conv2d(3, 768, k=16, s=16) → [B, 196, 768]
- Positional embedding: Embedding(196, 768), added after patch embed
- 12x transformer blocks:
  - LN(768) → Attn(Q/K/V/Out all 768→768, 12 heads, head_dim=64) → residual
  - LN(768) → MLP(768→3072, GELU-tanh, 3072→768) → residual
- Post-LayerNorm(768)
- Attention pooling head: learned query, cross-attn → LN → MLP(768→3072→768)
- Output: [B, 768]

## Key Numbers

| Param               | Value                          |
|----------------------|--------------------------------|
| B (batch, B200)      | 592 (148 × 4 imgs/SM)         |
| B (batch, B300)      | 640 (160 × 4 imgs/SM)         |
| seq_len              | 196                            |
| d_model              | 768                            |
| d_ff                 | 3072                           |
| n_heads              | 12                             |
| head_dim             | 64                             |
| n_layers             | 12                             |
| Total vision params  | ~93M (~93 MB in FP8)           |
| SMEM/SM              | 228 KB                         |
| DSMEM cluster-of-4   | 912 KB                         |
| DSMEM cluster-of-16  | 3.6 MB                         |
| HBM BW (B200)        | ~8 TB/s                        |

## Codegen Approach

The kernel is NOT written by hand. A Python3 codegen script (`gen.py`) emits the
complete `.cu` file from parameterized templates. This ensures:

1. **Consistency**: SMEM offsets, barrier IDs, register names, TMA descriptors are
   all computed from root constants. Change one param, regenerate, rebuild.
2. **Auditability**: The generator is ~400-600 lines of readable Python. The output
   is thousands of lines of PTX inline asm that no human should read directly.
3. **Tweakability**: Tile sizes, cluster size, double-buffer depth, snake ordering,
   batch size — all configurable knobs at the top of gen.py.
4. **Correctness**: Offset arithmetic, barrier numbering, and register allocation
   are computed programmatically, not typed by hand.

### gen.py structure

```
@dataclass
class HWConfig:
    sm_count: int           # 148 or 160
    smem_per_sm: int        # 228 KB
    cluster_size: int       # 4 or 16
    max_registers: int      # 255
    ...

@dataclass
class ModelConfig:
    d_model: int            # 768
    d_ff: int               # 3072
    n_heads: int            # 12
    head_dim: int           # 64
    seq_len: int            # 196
    n_layers: int           # 12
    ...

@dataclass
class TileConfig:           # derived from HW + Model
    tile_m: int             # rows per tile
    tile_n: int             # cols per tile
    tile_k: int             # reduction dim per tile
    ...

def compute_smem_layout(hw, model, tile) -> SMEMLayout:
    # computes byte offsets for weight buffer A, weight buffer B,
    # activation scratch, attn matrix, barriers, etc.

def emit_tma_descriptor_setup(...) -> str:     # host-side TMA init
def emit_wgmma_tile_loop(...) -> str:          # the core MXFP8 matmul
def emit_layernorm(...) -> str:                # warp-shuffle reduction
def emit_softmax(...) -> str:                  # online softmax in regs
def emit_gelu_tanh(...) -> str:                # fused activation
def emit_residual_add(...) -> str:             # epilogue fusion
def emit_grid_sync(...) -> str:                # cooperative groups sync
def emit_cluster_barrier(...) -> str:          # cluster-level sync
def emit_transformer_block(layer_idx) -> str:  # full block, calls above
def emit_kernel() -> str:                      # top-level __global__
def emit_host_main() -> str:                   # launch code
def main():                                    # write .cu to disk
```

Workflow: edit gen.py → `python3 gen.py > megakernel.cu` → `make` → ship binary.

## Milestones

### 1st Base — Patch Embed GEMM (tcgen05 proof of life)

Prove: persistent cooperative kernel launches on all SMs, executes one MXFP8
GEMM via tcgen05 WGMMA, grid syncs, produces correct output.

- [ ] gen.py scaffold: HWConfig, ModelConfig, TileConfig dataclasses. Derived
      quantity computation (tiles per SM, SMEM layout, batch size).
- [ ] emit_kernel(): cooperative grid launch, SM ID → row range mapping,
      grid sync at end. Compiles with `nvcc -arch=sm_100a`.
- [ ] emit_wgmma_tile_loop(): the MXFP8 matmul inner loop in inline PTX.
      Parameterized by M_tile, N_tile, K_tile. TMA async copy for weights.
      Patch embed GEMM: [B*196, 768] × [768, 768] + bias.
      On cluster-of-4, weight (576 KB) fits in DSMEM — load once.
- [ ] Position embedding add: lookup pos_embed[tok_idx % 196], elementwise add.
      Fused into GEMM epilogue.
- [ ] Input stub: curandGenerateUniform fills [B, 3, 224, 224].
      Reshape to [B*196, 768] (gather 16×16×3 patches into rows).
- [ ] Validation: host-side dump, compare vs PyTorch F.linear + pos_embed.
      Establish MXFP8 error budget.

### Rounding 1st — MLP (FC1 + GELU-tanh + FC2)

Prove: can do the large non-square GEMMs with fused activation and tiled
weight streaming for matrices that exceed cluster DSMEM.

- [ ] FC1 GEMM: [B*196, 768] × [768, 3072] + bias.
      Weight is 2.25 MB — does NOT fit cluster-of-4 DSMEM.
      Tile along N: 3072/tile_n tiles, stream each [768, tile_n] slab.
      (On B300 cluster-of-16, whole weight fits — codegen emits different
      path based on hw.cluster_size.)
- [ ] Fused GELU-tanh epilogue: applied in registers after each output tile,
      before writeback. emit_gelu_tanh() generates the PTX
      (tanh.approx.f32 or FP16 intrinsic).
- [ ] FC2 GEMM: [B*196, 3072] × [3072, 768] + bias.
      Tall-K matmul. Tile along K, accumulate partial sums.
- [ ] Residual add fused into FC2 epilogue: load pre-MLP activation from
      global, add in registers, write final output.
- [ ] Grid sync after MLP.
- [ ] Validate: random [B, 196, 768] → MLP path → compare vs PyTorch.

### 2nd Base — Attention (QKV + Softmax + Out Projection)

Prove: batched small-matmul attention with in-SMEM softmax works. Different
character from the large GEMMs — tiny tiles, many independent heads.

- [ ] Q, K, V projections: three separate [B*196, 768] × [768, 768] GEMMs.
      Each weight (576 KB) fits in cluster-of-4 DSMEM. Reuse the WGMMA
      tile loop from 1st base.
- [ ] Reshape for multi-head: [B, 196, 768] → [B*12, 196, 64].
      Pure index arithmetic in the codegen, no data movement.
- [ ] QK^T batched matmul: [196, 64] × [64, 196] → [196, 196] per head.
      B*12 heads, ~48 heads per SM. Small tiles — emit HMMA (mma.sync)
      rather than tcgen05 WGMMA (codegen picks based on tile size).
      Output stays in SMEM (196*196*2B = 75 KB per head, fits).
- [ ] Scale + softmax: divide by sqrt(64)=8, online softmax over 196
      elements in warp shuffles. emit_softmax() in PTX. No global traffic.
- [ ] Attn × V: [196, 196] × [196, 64] → [196, 64] per head.
      Read softmax result from SMEM, V from global/SMEM.
- [ ] Out projection: [B*196, 768] × [768, 768] + bias + residual add.
      Same as patch embed GEMM shape.
- [ ] Grid sync.
- [ ] Validate: full attention path vs PyTorch.

### Rounding 2nd — LayerNorm + Cluster-Level DSMEM Fusion

Prove: cluster barriers + DSMEM eliminate global memory roundtrips between
sub-operations. This is WHERE THE TRT ADVANTAGE COMES FROM.

- [ ] emit_layernorm(): reduce over dim=768 for mean/var using
      __shfl_xor_sync. Apply gamma*(x-mean)/sqrt(var+eps)+beta.
      768 = 24 warps of 32, or 1 warp doing 24 serial steps.
- [ ] Cluster fusion pattern — the key optimization:
      FC2 epilogue → residual add → LN1 of next sub-block, all
      within DSMEM. Activation never touches global memory.
      - FC2 output written to producer SM's SMEM.
      - Cluster barrier (NOT grid sync).
      - LN reads from DSMEM (local or remote SM's bank).
      - Compute LN in-place, feed directly to next GEMM.
- [ ] emit_cluster_barrier(): lighter than grid sync, only 4 (or 16)
      SMs participate. Codegen assigns barrier IDs, avoids collisions.
- [ ] Sync retardation strategy: grid sync ONLY between transformer
      blocks. Within a block, cluster barriers suffice because each
      cluster owns disjoint images (no cross-cluster activation deps).
- [ ] B300 cluster-of-16 enhanced path: with 3.6 MB DSMEM, can cache
      weight matrices (Q/K/V/Out at 576 KB each = 2.3 MB for all 4,
      fits). Codegen emits weight-caching path when cluster_size >= 16.
- [ ] Global memory traffic audit (computed by gen.py):
      - Without fusion: ~6 global R/W of [B, 196, 768] per block.
      - With cluster fusion: ~2 global R/W per block.
      gen.py prints this analysis when run.
- [ ] Validate: LN→Attn→res→LN→MLP→res fused sequence vs PyTorch.
      Verify no deadlocks or races from cluster barriers.

### 3rd Base — Full Transformer Block, Assembled

Prove: all sub-components wire together into one correct, fused block.
Integration milestone, no new primitives.

- [ ] emit_transformer_block(layer_idx): calls emit_layernorm,
      emit_wgmma_tile_loop (for Q/K/V/Out), emit_softmax,
      emit_wgmma_tile_loop (for FC1/FC2), emit_gelu_tanh,
      emit_residual_add, emit_cluster_barrier, emit_grid_sync.
- [ ] SM ↔ image assignment formalized in codegen:
      B200: 148 SMs, B=592, 4 imgs/SM, 784 tokens/SM.
      Cluster of 4 → 16 imgs/cluster. Attention is intra-image (196
      tokens), so NO cross-SM dependency during attention.
- [ ] SMEM budget audit (computed by gen.py at generation time):
      - Weight tile: ~64-128 KB
      - Activation working tile: 128 toks × 128 dims = 16 KB
      - Attention matrix: 75 KB per head (sequential over 48 heads/SM)
      - Double-buffer overhead
      gen.py asserts total < 228 KB or errors at generation time.
- [ ] Validate: full block output vs PyTorch layer 0.

### Rounding 3rd — 12-Layer Loop + Weight Double-Buffering

Prove: the block loop with pipelined weight streaming sustains throughput
across full encoder depth.

- [ ] Block loop: for (layer = 0; layer < 12; layer++) with weight
      pointer arithmetic. Per-layer weight offset computed by gen.py
      and baked into the generated code as constants.
- [ ] Weight double-buffering:
      - SMEM split: buffer A (compute) + buffer B (prefetch).
      - While computing with layer i's weights from buffer A,
        TMA async prefetch loads layer i+1's first tile into buffer B.
      - Grid sync at layer boundary, flip buffers.
      - Within a layer: pipeline W_j compute with W_{j+1} prefetch.
- [ ] Total weight traffic: 12 × ~6.8 MB = ~82 MB streamed once.
      At 8 TB/s = ~10 µs. Kernel is compute-bound, not memory-bound.
- [ ] Post-LayerNorm after the 12-block loop.
- [ ] gen.py emits a weight offset table as a __constant__ array.
- [ ] Validate: full 12-layer output (before pooling head) vs PyTorch.

### Home Run — Pooling Head + End-to-End

Prove: pixels in → embedding out, single kernel launch, correct, fast.

- [ ] Attention pooling head:
      - Learned query [1, 768] broadcast across batch.
      - K/V proj: [B*196, 768] × [768, 768] (standard).
      - QK^T: [B, 12, 1, 196] — trivially small, 196 scores per head.
      - Softmax over 196.
      - Attn×V → [B, 12, 1, 64] → concat → [B, 768].
      - Out proj: [B, 768] × [768, 768]. M=B only (592 or 640).
- [ ] Pooling LN + MLP: [B, 768] → LN → FC1(3072) → GELU → FC2(768).
      Tiny GEMMs (M=B). Possibly fits entirely in SMEM.
- [ ] Output: [B, 768] written to global memory. Kernel returns.
- [ ] Weight packing: gen.py emits a weight_layout.h with byte offsets
      for every weight tensor in the contiguous buffer. Also emits a
      Python script (pack_weights.py) that loads the HuggingFace
      checkpoint, quantizes to MXFP8, and writes the binary blob.
- [ ] Build system: Makefile.
      nvcc -arch=sm_100a -O3 megakernel.cu -o siglip_vision -lcurand
- [ ] Benchmark harness: 1000 iters (skip 100 warmup), report img/s,
      FLOP/s, HBM BW utilization. Compare vs TensorRT FP8 baseline.
- [ ] Full numerical validation vs PyTorch SiglipVisionModel.forward().

## File Layout

```
siglip/
├── CLAUDE.md              # LLM entry point — read first
├── docs/
│   ├── architecture.md    # this file — model specs, HW, codegen, milestones
│   ├── tasks.md           # active optimization action items
│   ├── profiling.md       # ncu/cuobjdump/sanitizer profiling playbook
│   └── reference/
│       ├── model.txt      # PyTorch model architecture dump
│       └── sass_dump.txt  # SASS disassembly (large, load on demand)
├── gen.py                 # THE codegen script — edit this, not .cu
├── Makefile               # build rules
├── megakernel.cu          # GENERATED — do not edit
├── bench_sweep.sh         # M-size benchmark sweep
├── pack_weights.py        # weight quantization + packing (future)
└── validate.py            # PyTorch reference + comparison (future)
```
