# SigLIP2 Vision Encoder — Persistent Megakernel

Hand-tuned Blackwell (SM100a) persistent megakernel for `google/siglip2-base-patch16-224` patch embed GEMM.
FP8 (E4M3) precision, tcgen05 WGMMA, TMA, `cta_group::2` with 2-CTA clusters. Cross-compiled on CPU VPS, runs on B200.

## Current state

**0.560 ms / 1955 TFLOPS** fused (GEMM + bias + pos_embed) — **33% faster** than cuBLAS end-to-end (0.835 ms = best GEMM + unfused pos_embed).

The kernel's value is **fusion**: the overlapped epilogue eliminates the 0.470 ms unfused pos_embed overhead entirely.

cuBLAS pure GEMM is faster: 0.365 ms / 3001 TFLOPS (per-tensor FP8, best-of-8 algos, 256MB workspace).
Our effective TFLOPS (1955) counts fused epilogue time in the denominator — not a fair GEMM-only comparison.

GEMM: `[928256, 768] x [768, 768]^T` with fused bias + positional embedding add, BF16 output.
Batch = 4736 images x 196 patches = 928256 rows. Square weight matrix (768x768).

The kernel is correct (checksum validated) and stable.

## Current bottlenecks (from ncu profiling)

The kernel is **epilogue-bound** (confirmed by clock64 F15 profiling). W1 stalls 2,381 cycles/tile (35% of tile time) waiting for the epilogue to release the TMEM buffer. The epilogue takes ~5,833 cycles (Phase 1: 4,762 TMEM readback 81.6%, Phase 2: 1,071 SMEM→global 18.4%), overrunning the K-loop's 4,073 cycles by ~1,760 cycles.

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

2. **Epilogue-bound** (confirmed by clock64 F15). Phase 1 (TMEM readback) at 81.6% is the hard bottleneck. `tcgen05.wait::ld` global fence prevents further TMEM overlap. Direct global stores (F17) made Phase 1 19% slower due to L1 contention — SMEM staging is essential.

3. **Zero TMA multicast** — B matrix loaded independently by each CTA. `dest_multicast = 0`, `dest_self = 133M sectors`. Enabling multicast for B loads would halve B bandwidth.

4. **SMEM bank conflicts 32%/22%** (ld/st) — significant but hidden in K-loop shadow. STAGING_ROW_PAD doesn't fully prevent Phase 2 transposed read conflicts.

5. **Warp latency dominated by TMEM** — `long_scoreboard` at 390K cycles/warp is 3.3× productive `selected` (119K). The scheduler hides this (only 4.7% in pct view) but it means epilogue warps spend most of their time waiting.

6. **Register pressure (219 regs/thread, 0 spills)** — limits occupancy to 1 CTA/SM. Not actionable without major restructuring.

**To go faster:** Reduce epilogue Phase 1 time — double-buffered SMEM staging (overlap Phase 2 of first half with Phase 1 of second half), TMA multicast for B (frees L2/DRAM bandwidth), or larger TM to amortize per-tile overhead. See `docs/make_better.md`.

**Tested and ruled out:** See `EXPERIMENTS.md` for 17 experiments with hypotheses, results, and analysis.

## Kernel structure

Warp-specialized, 6 warps (192 threads), `cta_group::2`, `__cluster_dims__(2,1,1)`:

- **W0**: TMA async bulk loads (A + B tiles, both CTAs load independently)
- **W1**: TMEM alloc (512 cols, single alloc for double buffering) + `tcgen05.mma.cta_group::2` accumulation into TMEM (CTA0 lane-0 only, multicast commit to both CTAs)
- **W2-W5**: Overlapped epilogue (4 warps) — each warp independently polls mainloop mbarrier, then runs two-phase SMEM-staged store: Phase 1 (`tcgen05.ld` from TMEM, inline BF16 add, CVT, `st.shared` to per-warp staging buffer), `__syncwarp()`, Phase 2 (transposed `ld.shared` → coalesced `st.global.v4`)

TM=128 rows / 32 rows per warp = 4 row groups. W2-W5 each own a row group (all 256 cols). No column splitting (is_split always 0). `epilogue_store` is templated on `<NC_START, NC_END>` so the compiler sees constant loop bounds and fully unrolls.

The overlapped epilogue for tile N-1 runs concurrently with the K-loop for tile N (double-buffered TMEM, mbarrier-protected). After the tile loop, W2-5 run a drain epilogue for the last tile.

### Tile config
- Tile: 256x256x128 (M=2x128 from cta_group::2, N=256, K=128)
- TMEM: single alloc of 512 cols (TN*2), double-buffered via column offset (buf*TN)
- SMEM: 4-stage pipeline (131 KB) + epilogue staging (4 warps x 16,896 = 68 KB) = ~199 KB total of 228 KB
- Tiles: 3626 M-tiles x 3 N-tiles = 10,878 total, snake ordering
- 219 registers/thread, 0 spills
- `NUM_EPI_WARPS` controls epilogue warp count (currently 4); `THREADS` derived as `32*(2+NUM_EPI_WARPS)`

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
| `clock64_timing.txt` | W1 clock64 timing output (5-warp baseline) |
| `clock64_timing_analysis.md` | Analysis of clock64 W1 timing — epilogue IS the bottleneck |
| `source_counters_analysis.md` | SourceCounters per-instruction analysis — W1 stall isolation |
| `source_counters_raw.csv` | Raw SourceCounters CSV data |
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
