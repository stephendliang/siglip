# SigLIP2 Vision Encoder — Persistent Megakernel Plan

Target: B200 (148 SMs, cluster-of-2) / B300 (160 SMs, cluster-of-16)
Model: google/siglip2-base-patch16-224 (86M vision encoder)
Precision: FP8 E4M3 (per-tensor scaling)
ISA: tcgen05 WGMMA, TMA, persistent grid
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
| B (batch, B200)      | 4736 (current bench) / 592 (148 × 4 imgs/SM target) |
| seq_len              | 196                            |
| d_model              | 768                            |
| d_ff                 | 3072                           |
| n_heads              | 12                             |
| head_dim             | 64                             |
| n_layers             | 12                             |
| Total vision params  | ~93M (~93 MB in FP8)           |
| SMEM/SM              | 228 KB                         |
| HBM BW (B200)        | ~8 TB/s                        |

## Development Approach

The kernels are **hand-written inline PTX**, not codegen output. All shared infrastructure (pipeline, TMEM loads, TMA helpers, mbarrier ops, tile math) lives in `kernel_common.cuh`. Each kernel file `#define`s `N_DIM` before including the header — tile counts, K-iterations, and SMEM layout are derived automatically.

```
edit kernel_common.cuh / megakernel.cu / fc1_gelu.cu → make → run on B200
```

Tuning parameters (STAGGER_CYCLES, MBAR_EARLY, INTERLEAVE_STRATEGY, etc.) are compile-time `#define`s overridable via `-D` flags. Grid search (`tools/grid_search.py`) sweeps parameter space automatically.

## Milestones

### 1st Base — Patch Embed GEMM [DONE]

Persistent warp-specialized kernel: [928256,768]×[768,768]^T + bias + pos_embed.
**0.519 ms / 2108 TFLOPS** fused — 38% faster than cuBLAS end-to-end (0.835 ms).

- [x] Persistent grid on all 148 SMs, cta_group::2, cluster_dims(2,1,1)
- [x] tcgen05.mma FP8 E4M3 WGMMA with TMA async loads
- [x] Warp-specialized: W0 (TMA loads), W1 (MMA), W2-W5 (overlapped epilogue)
- [x] Fused bias + pos_embed in epilogue (TMEM→CVT→add→STS→TMA store)
- [x] TMEM double-buffered (single 512-col alloc), 4-stage SMEM pipeline
- [x] Unified SWIZZLE_128B epilogue with interleaved TMA stores (F38+F40)
- [x] Non-uniform validation, 1024-strided checksum + 32 CPU spot checks
- [x] Grid search: 4 runs × 145 configs, 145/145 valid, parameter space exhausted

### Rounding 1st — MLP (FC1 + GELU-tanh + FC2) [IN PROGRESS]

FC1 kernel written (`fc1_gelu.cu`), not yet run on B200.

- [x] FC1+GELU kernel: [928256,768]×[768,3072]^T + bias + GELU
      N_DIM=3072, TILES_N=12, TOTAL_TILES=43,512. Same tile shape 256x256x128.
      GELU fused in epilogue (tanhf approximation).
      Shares all infrastructure via kernel_common.cuh.
- [ ] Run FC1+GELU on B200, validate, tune (grid search with new tile count)
- [ ] FC2 GEMM: [928256,3072]×[3072,768]^T + bias + residual add
      K_DIM=3072 → K_ITERS=24 (vs 6). Tall-K matmul.
      Residual add in epilogue (load pre-MLP activation from global).
- [ ] Grid sync after MLP
- [ ] Validate: random [B, 196, 768] → MLP path → compare vs PyTorch

### 2nd Base — Attention (QKV + Softmax + Out Projection)

Different character from the large GEMMs — tiny tiles, many independent heads.

- [ ] Q, K, V projections: three [B*196, 768] × [768, 768] GEMMs
      Same shape as patch embed — reuse megakernel's GEMM structure.
- [ ] Reshape for multi-head: [B, 196, 768] → [B*12, 196, 64]
      Pure index arithmetic, no data movement.
- [ ] QK^T batched matmul: [196, 64] × [64, 196] → [196, 196] per head
      B*12 heads. Small tiles — may need mma.sync instead of tcgen05 WGMMA.
      Output stays in SMEM (196*196*2B = 75 KB per head).
- [ ] Scale + softmax: divide by sqrt(64)=8, online softmax via warp shuffles
- [ ] Attn × V: [196, 196] × [196, 64] → [196, 64] per head
- [ ] Out projection: [B*196, 768] × [768, 768] + bias + residual add
- [ ] Validate: full attention path vs PyTorch

### Rounding 2nd — LayerNorm + Cluster Fusion

Cluster barriers + DSMEM eliminate global memory roundtrips between sub-operations.

- [ ] LayerNorm: reduce over dim=768 for mean/var using __shfl_xor_sync
- [ ] Cluster fusion pattern:
      FC2 epilogue → residual add → LN of next sub-block, all within DSMEM.
      Activation never touches global memory.
      Cluster barrier (not grid sync) — only 2 SMs participate (cta_group::2).
- [ ] Grid sync only between transformer blocks (disjoint image assignment per cluster)

### 3rd Base — Full Transformer Block

Integration milestone — wire sub-components into one correct, fused block.

- [ ] SM ↔ image assignment: 148 SMs, B images, images/SM, tokens/SM
      Cluster of 2 → images/cluster. Attention is intra-image (196 tokens),
      so NO cross-SM dependency during attention.
- [ ] SMEM budget audit: weight tile + activation tile + attention matrix + double-buffer
- [ ] Validate: full block output vs PyTorch layer 0

### Rounding 3rd — 12-Layer Loop + Weight Streaming

- [ ] Block loop with weight pointer arithmetic. Per-layer weight offsets as constants.
- [ ] Weight double-buffering: prefetch layer i+1's first tile during layer i compute
- [ ] Total weight traffic: 12 × ~6.8 MB = ~82 MB. At 8 TB/s = ~10 µs (compute-bound).
- [ ] Warp parking: different GEMM shapes (768×768 vs 768×3072) may need different
      tile counts. Park excess warps/CTAs between phases.

### Home Run — Pooling Head + End-to-End

Single kernel launch: pixels in → embedding out.

- [ ] Attention pooling head: learned query, cross-attn over 196 tokens
- [ ] Pooling LN + MLP: [B, 768] → LN → FC1(3072) → GELU → FC2(768). Tiny GEMMs.
- [ ] Weight packing: Python script loads HuggingFace checkpoint, quantizes to FP8,
      writes contiguous binary blob with offset table.
- [ ] Benchmark vs TensorRT FP8 baseline

## File Layout

```
siglip/
├── CLAUDE.md              # Project overview — read first
├── kernel_common.cuh      # Shared infrastructure (pipeline, TMEM, TMA, mbarriers)
├── megakernel.cu          # Patch embed GEMM (N_DIM=768) — hand-written
├── fc1_gelu.cu            # FC1+GELU GEMM (N_DIM=3072) — hand-written
├── Makefile               # Build rules (sm_100a)
├── tools/                 # Analysis & sweep scripts
│   ├── grid_search.py     # Compile-time parameter sweep
│   ├── sass_analysis.py   # SASS scheduling analyzer
│   ├── analyze_timing.py  # clock64 timing → equilibrium analysis
│   └── ...
├── bench/                 # Baseline benchmarks (CUTLASS, cuBLAS, calibration)
├── data/                  # Profiling data & sweep results
└── docs/                  # Documentation
    ├── architecture.md    # This file — model specs, milestones
    ├── EXPERIMENTS.md     # Experiment log (F1-F40)
    ├── FUTURE_PROPOSALS.md # Optimization roadmap
    └── reference/         # Model dump, SASS disassembly
```
