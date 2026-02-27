# SigLIP2 Vision Encoder — Persistent Megakernel

Hand-tuned Blackwell (SM100a) persistent megakernel for `google/siglip2-base-patch16-224` patch embed GEMM.
MXFP8 (E4M3) precision, tcgen05 WGMMA, TMA, `cta_group::2` with 2-CTA clusters. Cross-compiled on CPU VPS, runs on B200.

## Current state

**0.917 ms / 1190 TFLOPS** — 92% of cuBLAS (0.845 ms / 1295 TFLOPS).

GEMM: `[928256, 768] x [768, 768]^T` with fused bias + positional embedding add, BF16 output.
Batch = 4736 images x 196 patches = 928256 rows. Square weight matrix (768x768).

The kernel is correct (checksum validated) and stable. The remaining 8% gap to cuBLAS is dominated by TMEM load latency (`long_scoreboard 5.1%`). See `LATEST_AUDIT.md` for full ncu breakdown.

## Kernel structure

Warp-specialized, 6 warps (192 threads), `cta_group::2`, `__cluster_dims__(2,1,1)`:

- **W0**: TMA async bulk loads (A + B tiles, both CTAs load independently)
- **W1**: `tcgen05.mma.cta_group::2` accumulation into TMEM (CTA0 lane-0 only, multicast commit to both CTAs)
- **W2-W5**: Overlapped epilogue — `tcgen05.ld` from TMEM, inline BF16 bias+pos_embed add, FP32->BF16 convert, `st.global.v8`

The overlapped epilogue for tile N-1 runs concurrently with the K-loop for tile N (double-buffered TMEM, mbarrier-protected). After the tile loop, W2-5 run a drain epilogue for the last tile.

### Tile config
- Tile: 256x128x128 (M=2x128 from cta_group::2, N=128, K=128)
- TMEM: 128 cols x 2 buffers = 256 cols (at SM100 capacity limit)
- SMEM: 6-stage pipeline, 144 KB data + mbarriers = 147,712 bytes total
- Tiles: 3626 M-tiles x 6 N-tiles = 21,756 total, snake ordering
- 168 registers/thread, 0 spills

## Development workflow

**Edit `megakernel.cu` directly.** It is the hand-tuned source of truth.

```
edit megakernel.cu -> make -> ./siglip_vision
```

`gen.py` is outdated (old 4-warp structure). Will be updated after kernel is finalized.

## File map

| File | What |
|------|------|
| `megakernel.cu` | **Source of truth** — hand-tuned CUDA kernel |
| `LATEST_AUDIT.md` | ncu profiling data and performance analysis |
| `docs/profiling.md` | Profiling commands (ncu, cuobjdump, compute-sanitizer) |
| `docs/architecture.md` | Model specs, HW config, milestones (partially stale) |
| `gen.py` | Codegen script — **outdated, do not use** |
| `Makefile` | Build rules (sm_100a, nvcc flags) |
| `docs/reference/model.txt` | PyTorch model architecture dump |
| `docs/reference/sass_dump.txt` | SASS disassembly (load on demand only) |

## Build and run

```bash
make                    # compile megakernel.cu -> siglip_vision
./siglip_vision         # run on B200, prints timing + TFLOPS + checksum
```

## Key constraints

- Target: `sm_100a` (B200, 148 SMs)
- `cta_group::2` with `__cluster_dims__(2,1,1)` — 74 clusters of 2 CTAs
- TMEM: 256 cols/SM max — limits double-buffered tile to N=128 (2x128=256)
- SMEM: 228 KB/SM — current usage 144 KB (room for more stages if needed)
- All inline PTX — no CUTLASS dependency for the kernel itself
- `megakernel.cu` is hand-edited directly
- Expected checksum: 1769472.0, C[0,0..3] = 1728.0
