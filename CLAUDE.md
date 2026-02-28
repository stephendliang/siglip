# SigLIP2 Vision Encoder — Persistent Megakernel

Hand-tuned Blackwell (SM100a) persistent megakernel for `google/siglip2-base-patch16-224` patch embed GEMM.
FP8 (E4M3) precision, tcgen05 WGMMA, TMA, `cta_group::2` with 2-CTA clusters. Cross-compiled on CPU VPS, runs on B200.

## Current state

**0.579 ms / 1892 TFLOPS** fused (GEMM + bias + pos_embed) — **31% faster** than cuBLAS end-to-end (0.835 ms = best GEMM + unfused pos_embed).

The kernel's value is **fusion**: the overlapped epilogue eliminates the 0.470 ms unfused pos_embed overhead entirely.

cuBLAS pure GEMM is faster: 0.365 ms / 3001 TFLOPS (per-tensor FP8, best-of-8 algos, 256MB workspace).
Our effective TFLOPS (1892) counts fused epilogue time in the denominator — not a fair GEMM-only comparison.

GEMM: `[928256, 768] x [768, 768]^T` with fused bias + positional embedding add, BF16 output.
Batch = 4736 images x 196 patches = 928256 rows. Square weight matrix (768x768).

The kernel is correct (checksum validated) and stable.

## Current bottlenecks (from ncu profiling)

The kernel is **MMA/K-loop bound**. `selected` (productive issue) at 22.9% is 2.5× the sum of all stall categories (~8.8%). No single stall dominates. The epilogue completes within the K-loop's shadow — epilogue optimizations are irrelevant unless they also speed up the K-loop.

**Warp stall breakdown:**

| Stall | % of peak | Notes |
|---|---|---|
| **selected (issuing)** | **22.9%** | Productive execution — dominant |
| long_scoreboard (TMEM) | 4.7% | TMEM readback latency |
| wait (TMA) | 1.5% | TMA pipeline stalls |
| sleeping | 0.95% | Warps parked between tiles |
| barrier | 0.8% | Cluster sync |
| short_scoreboard (SMEM) | 0.57% | SMEM staging ld/st chains |
| not_selected | 0.2% | Eligible but not picked |
| mio_throttle | 0.04% | Memory I/O backpressure — negligible |

**Memory throughput:**

| Subsystem | Utilization |
|---|---|
| L1 tex | 67% — no longer the ceiling |
| L2 | 49% |
| DRAM | 32% |

**Instruction mix (per kernel invocation):**

| Type | Count |
|---|---|
| Total | 122.6M |
| Global loads | 2.78M |
| Global stores | 3.48M |
| Shared loads | 3.48M |
| Shared stores | 2.79M |

**Key findings:**

1. **TC pipe active only 50%** — tensor cores idle half the time. cuBLAS achieves ~79% on this shape. The 29pp gap is the primary optimization target. See `docs/make_better.md` for full analysis.

2. **K-loop bound, not epilogue-bound** (confirmed F10/F11/F13/F14). The epilogue runs in the K-loop's shadow. All epilogue-only optimizations tested (software pipelining, L1 bypass, cache hints, SMEM staging of combined) either neutral or regressed.

3. **Zero TMA multicast** — B matrix loaded independently by each CTA. `dest_multicast = 0`, `dest_self = 133M sectors`. Enabling multicast for B loads would halve B bandwidth.

4. **SMEM bank conflicts 32%/22%** (ld/st) — significant but hidden in K-loop shadow. STAGING_ROW_PAD doesn't fully prevent Phase 2 transposed read conflicts.

5. **Warp latency dominated by TMEM** — `long_scoreboard` at 390K cycles/warp is 3.3× productive `selected` (119K). The scheduler hides this (only 4.7% in pct view) but it means epilogue warps spend most of their time waiting.

6. **Register pressure (223 regs/thread, 0 spills)** — limits occupancy to 1 CTA/SM. Not actionable without major restructuring.

**To go faster:** Reduce TC idle time — fewer tiles (larger TM), fewer K-iterations per tile, less per-tile overhead, or enable TMA multicast for B. See `docs/make_better.md`.

**Tested and ruled out:** See `EXPERIMENTS.md` for 14 experiments with hypotheses, results, and analysis.

## Kernel structure

Warp-specialized, 7 warps (224 threads), `cta_group::2`, `__cluster_dims__(2,1,1)`:

- **W0**: TMA async bulk loads (A + B tiles, both CTAs load independently)
- **W1**: TMEM alloc (512 cols, single alloc for double buffering) + `tcgen05.mma.cta_group::2` accumulation into TMEM (CTA0 lane-0 only, multicast commit to both CTAs)
- **W2-W6**: Overlapped epilogue (5 warps) — each warp independently polls mainloop mbarrier, then runs two-phase SMEM-staged store: Phase 1 (`tcgen05.ld` from TMEM, inline BF16 add, CVT, `st.shared` to per-warp staging buffer), `__syncwarp()`, Phase 2 (transposed `ld.shared` → coalesced `st.global.v4`/`st.global.v2`)

TM=128 rows / 32 rows per warp = 4 row groups. W2-W5 each own a row group (all 256 cols). W6 column-splits with W2: both handle row_group 0, W2 does cols 0..127, W6 does cols 128..255. `epilogue_store` is templated on `<NC_START, NC_END>` so the compiler sees constant loop bounds and fully unrolls.

The overlapped epilogue for tile N-1 runs concurrently with the K-loop for tile N (double-buffered TMEM, mbarrier-protected). After the tile loop, W2-6 run a drain epilogue for the last tile.

### Tile config
- Tile: 256x256x128 (M=2x128 from cta_group::2, N=256, K=128)
- TMEM: single alloc of 512 cols (TN*2), double-buffered via column offset (buf*TN)
- SMEM: 4-stage pipeline (131 KB) + epilogue staging (5 warps x 16,896 = 84 KB) = ~211 KB total of 228 KB
- Tiles: 3626 M-tiles x 3 N-tiles = 10,878 total, snake ordering
- 223 registers/thread, 0 spills
- `NUM_EPI_WARPS` controls epilogue warp count (currently 5); `THREADS` derived as `32*(2+NUM_EPI_WARPS)`

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
| `EXPERIMENTS.md` | Experiment log, profiling data, and optimization history |
| `docs/make_better.md` | Advanced profiling — TC utilization, per-pipe breakdown, bank conflicts, TMA multicast |
| `docs/whats_wrong.md` | Triage-level profiling — barrier stalls, TMA serialization, correctness |
| `docs/architecture.md` | Model specs, HW config, milestones (partially stale) |
| `gen.py` | Codegen script — **outdated, do not use** |
| `compare.py` | ncu CSV diff tool — usage: `python compare.py a.csv b.csv` |
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
- SMEM: 228 KB/SM — current usage ~211 KB (4-stage pipeline + epilogue staging, ~17 KB free)
- All inline PTX — no CUTLASS dependency for the kernel itself
- `megakernel.cu` is hand-edited directly
- Expected checksum: 1769472.0, C[0,0..3] = 1728.0
- ml_phase init must account for odd tile_start (start_buf-dependent)
