# SigLIP2 Vision Encoder — Persistent Megakernel

Hand-tuned Blackwell (SM100a) persistent megakernel for `google/siglip2-base-patch16-224` patch embed GEMM.
FP8 (E4M3) precision, tcgen05 WGMMA, TMA, `cta_group::2` with 2-CTA clusters. Cross-compiled on CPU VPS, runs on B200.

## Current state

**0.743 ms** fused (GEMM + bias + pos_embed) — **11% faster** than cuBLAS end-to-end (0.835 ms = best GEMM + unfused pos_embed).

The kernel's value is **fusion**: the overlapped epilogue eliminates the 0.470 ms unfused pos_embed overhead entirely.

cuBLAS pure GEMM is faster: 0.365 ms / 3001 TFLOPS (per-tensor FP8, best-of-8 algos, 256MB workspace).
Our effective TFLOPS (1475) counts fused epilogue time in the denominator — not a fair GEMM-only comparison.

GEMM: `[928256, 768] x [768, 768]^T` with fused bias + positional embedding add, BF16 output.
Batch = 4736 images x 196 patches = 928256 rows. Square weight matrix (768x768).

The kernel is correct (checksum validated) and stable.

## Current bottlenecks (from ncu profiling)

The kernel is **epilogue-bound**. The MMA pipe is idle most of the time because the TMEM readback → add → convert → store chain takes longer than the K-loop.

1. **TMEM load latency (`long_scoreboard`)** — dominant stall. Each `tcgen05.ld` is ~200 cycles. 4 epilogue warps can't fully hide it. Switching from x16 to x32 loads (halving instruction count) was perf-neutral — confirming the bottleneck is TMEM bandwidth, not instruction overhead.

2. **Uncoalesced stores (49% excess L2 sectors)** — each warp writes 32 rows, so adjacent threads store to addresses 1536B apart (768×2). Every lane hits a different 128B sector. Inherent to row-major output with per-row TMEM readback. Same pattern exists in cuBLAS.

3. **Register pressure (248 regs/thread, 0 spills)** — limits occupancy to 1 CTA/SM. Any approach requiring more live state will spill.

## Ideas for further improvement

**Most promising:**

- **More epilogue warps (8 instead of 4)**: Would better hide TMEM latency (~6-8 warps needed for full coverage). Requires going from 6→8 total warps (256 threads). Risk: register pressure — 248 regs × 256 threads = 63,488 > 64K reg file limit. Would need `--maxrregcount=240` or epilogue code shrinkage. Alternatively, split epilogue work differently so more warps share fewer columns each.

- **Epilogue SMEM staging**: Instead of each warp independently loading from TMEM and writing to global, have warps cooperatively load TMEM into SMEM, then write from SMEM with coalesced access patterns. Could fix the 49% uncoalesced store problem. Cost: extra SMEM (have 36 KB headroom) + sync overhead.

- **Smaller output tile (TN=128, revisit)**: With x32 TMEM loads and other epilogue improvements applied, the shorter epilogue loop might tip the balance differently than when we last tried TN=128 (which was 1190 TFLOPS with x16 loads).

**Speculative / higher-effort:**

- **Two-pass output**: Write coalesced to a scratch buffer, then transpose. Eliminates uncoalesced stores but adds a second kernel pass. Only wins if store inefficiency > transpose cost.

- **Warp-specialized epilogue roles**: Instead of 4 identical epilogue warps, have some warps do TMEM→SMEM and others do SMEM→global. Decouples TMEM latency from store latency. Complex to implement.

- **Reduce TMEM traffic**: If accumulator precision could be BF16 instead of FP32, TMEM readback halves. But tcgen05.mma only accumulates to FP32 — would need explicit FP32→BF16 before TMEM store (not available in current ISA).

**Ruled out:**

- **x32 TMEM loads** (done, perf-neutral) — bandwidth-bound, not instruction-bound.
- **SMEM prefetch of combined** (tried commit 4ff9644→892766c) — inline BF16 loads from global were faster, likely because L1 cache hits on the small combined tensor.
- **Centralized bar.sync for mbar broadcast** (done, 0.764→0.743 ms) — warp 2 polling + bar.sync to W3-W5 cost 10.3% stall. All epilogue warps polling independently is faster.
- **`cta_group::4`** — would need 1024 TMEM cols (exceeds 512/SM HW limit).

## Kernel structure

Warp-specialized, 6 warps (192 threads), `cta_group::2`, `__cluster_dims__(2,1,1)`:

- **W0**: TMA async bulk loads (A + B tiles, both CTAs load independently)
- **W1**: TMEM alloc (512 cols, single alloc for double buffering) + `tcgen05.mma.cta_group::2` accumulation into TMEM (CTA0 lane-0 only, multicast commit to both CTAs)
- **W2-W5**: Overlapped epilogue — each warp independently polls mainloop mbarrier, `tcgen05.ld` from TMEM, inline BF16 bias+pos_embed add, FP32->BF16 convert, `st.global.v8`

The overlapped epilogue for tile N-1 runs concurrently with the K-loop for tile N (double-buffered TMEM, mbarrier-protected). After the tile loop, W2-5 run a drain epilogue for the last tile.

### Tile config
- Tile: 256x256x128 (M=2x128 from cta_group::2, N=256, K=128)
- TMEM: single alloc of 512 cols (TN*2), double-buffered via column offset (buf*TN)
- SMEM: 6-stage pipeline, 192 KB data + mbarriers = 196,864 bytes total
- Tiles: 3626 M-tiles x 3 N-tiles = 10,878 total, snake ordering
- 248 registers/thread, 0 spills

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
- TMEM: 512 cols/SM total, single alloc of TN*2 for double buffering (learned from matmul_v7: two separate allocs deadlock, single alloc works)
- SMEM: 228 KB/SM — current usage 192 KB
- All inline PTX — no CUTLASS dependency for the kernel itself
- `megakernel.cu` is hand-edited directly
- Expected checksum: 1769472.0, C[0,0..3] = 1728.0
- ml_phase init must account for odd tile_start (start_buf-dependent)
