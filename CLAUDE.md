# SigLIP2 Vision Encoder — Persistent Megakernel

Hand-tuned Blackwell (SM100a) persistent megakernel for `google/siglip2-base-patch16-224` patch embed GEMM.
FP8 (E4M3) precision, tcgen05 WGMMA, TMA, `cta_group::2` with 2-CTA clusters. Cross-compiled on CPU VPS, runs on B200.

## Current state

**0.543 ms / 2018 TFLOPS** fused (GEMM + bias + pos_embed) — **35% faster** than cuBLAS end-to-end (0.835 ms = best GEMM + unfused pos_embed).

The kernel's value is **fusion**: the overlapped epilogue eliminates the 0.470 ms unfused pos_embed overhead entirely.

cuBLAS pure GEMM is faster: 0.365 ms / 3001 TFLOPS (per-tensor FP8, best-of-8 algos, 256MB workspace).
Our effective TFLOPS (2018) counts fused epilogue time in the denominator — not a fair GEMM-only comparison.

GEMM: `[928256, 768] x [768, 768]^T` with fused bias + positional embedding add, BF16 output.
Batch = 4736 images x 196 patches = 928256 rows. Square weight matrix (768x768).

The kernel is correct (checksum validated) and stable.

## Current bottlenecks (clock64 timing build, post-F19)

The kernel is **epilogue-bound** in a balanced producer-consumer equilibrium. Per-tile cycle budget:

```
W1:       epi_wait(1,056) + TMA0(857) + K-loop(4,154) = 6,067
Epilogue: ml_wait(1,139) + Phase1(4,129) + Phase2B(899) = 6,167
```

Equilibrium deficit: ~257 cycles (epilogue slightly slower). Double-buffer lag amplifies this to 1,056-cycle epi_wait (4.1x amplification).

**Key facts:**
- TC pipe active ~50% (cuBLAS: ~79%). Gap = epilogue wait (17pp) + TMA0 DRAM latency (7pp) + K-loop overhead (5pp).
- Phase 1 TMEM readback = 67% of epilogue cycle. Primary target.
- K-loop: 411 instructions/tile, only 24 MMA (5.8%). R2UR (42%), PLOP3 (23%), ELECT (8%) dominate overhead.
- TMA multicast not applicable (B is N-split across CTAs).
- 223 regs/thread, 0 spills. Limits occupancy to 1 CTA/SM.

**Ceiling:** Eliminate epi_wait → 2,486 TFLOPS. Also TMA0 → 2,998 TFLOPS (≈ cuBLAS 3,001).

Run `python3 analyze_timing.py clock64_timing.txt` for full equilibrium analysis and what-if projections.
Run `python3 analyze_source_counters.py source_counters_raw.csv` for per-instruction stall breakdown.
See `EXPERIMENTS.md` for 20 experiments (F1-F20) with hypotheses, results, and analysis. See `FUTURE_PROPOSALS.md` for active optimization roadmap.

## Kernel structure

Warp-specialized, 6 warps (192 threads), `cta_group::2`, `__cluster_dims__(2,1,1)`:

- **W0**: TMA async bulk loads (A + B tiles, both CTAs load independently)
- **W1**: TMEM alloc (512 cols, single alloc for double buffering) + `tcgen05.mma.cta_group::2` accumulation into TMEM (CTA0 lane-0 only, multicast commit to both CTAs)
- **W2-W5**: Overlapped epilogue (4 warps) — each warp independently polls mainloop mbarrier, then runs double-buffered SMEM-staged store: Phase 1A (first 128 cols: `tcgen05.ld` → BF16 add → CVT → `st.shared` to staging_a), `__syncwarp()`, Phase 1B+2A interleaved (second 128 cols → staging_b, with Phase 2A coalesced `ld.shared`+`st.global.v2` from staging_a hiding in TMEM stalls), `__syncwarp()` + **early mbar_arrive** (signals TMEM free — W1 can start next K-loop while Phase 2B runs), Phase 2B (`ld.shared` → `st.global.v2` from staging_b, overlapped with K-loop)

TM=128 rows / 32 rows per warp = 4 row groups. W2-W5 each own a row group (all 256 cols). No column splitting (is_split always 0). `epilogue_store` is templated on `<NC_START, NC_END>` so the compiler sees constant loop bounds and fully unrolls.

The overlapped epilogue for tile N-1 runs concurrently with the K-loop for tile N (double-buffered TMEM, mbarrier-protected). After the tile loop, W2-5 run a drain epilogue for the last tile.

### Tile config
- Tile: 256x256x128 (M=2x128 from cta_group::2, N=256, K=128)
- TMEM: single alloc of 512 cols (TN*2), double-buffered via column offset (buf*TN)
- SMEM: 4-stage pipeline (131 KB) + epilogue staging (4 warps x 17,408 = 70 KB, double-buffered halves) = ~201 KB total of 228 KB
- Tiles: 3626 M-tiles x 3 N-tiles = 10,878 total, snake ordering
- 223 registers/thread, 0 spills
- `NUM_EPI_WARPS` controls epilogue warp count (currently 4); `THREADS` derived as `32*(2+NUM_EPI_WARPS)`

## Development workflow

**Edit `megakernel.cu` directly.** It is the hand-tuned source of truth.

```
edit megakernel.cu -> make -> ./siglip_vision
```

`gen.py` is outdated — do not use.

## File map

| File | What |
|------|------|
| `megakernel.cu` | **Source of truth** — hand-tuned CUDA kernel |
| `EXPERIMENTS.md` | Experiment log (F1-F20), profiling data, optimization history |
| `FUTURE_PROPOSALS.md` | Active optimization roadmap (F21-F28) with execution order |
| `Makefile` | Build rules (sm_100a, nvcc flags). `make timing` for clock64 build |
| **Analysis scripts** | |
| `analyze_timing.py` | Parses `clock64_timing.txt` → equilibrium analysis, ceiling projections, what-if scenarios |
| `analyze_source_counters.py` | Parses ncu SourceCounters CSV → stall breakdown, MMA stats, W1 budget, instruction mix |
| `compare.py` | ncu CSV diff tool — usage: `python compare.py a.csv b.csv` |
| **Raw profiling data** | |
| `clock64_timing.txt` | Kernel printf from timing build (34 lines). Analyze with `analyze_timing.py` |
| `source_counters_raw.csv` | Raw ncu `--set source --csv` data (4K lines). Analyze with `analyze_source_counters.py` |
| `clock64_timing_analysis.md` | Historical prose analysis — **superseded by `analyze_timing.py`** |
| `source_counters_analysis.md` | Historical prose analysis — **superseded by `analyze_source_counters.py`** |
| **Reference docs** | |
| `docs/make_better.md` | Deep ncu profiling — TC utilization, per-pipe breakdown, bank conflicts |
| `docs/reference/model.txt` | PyTorch model architecture dump |
| `docs/reference/sass_dump.txt` | SASS disassembly (load on demand only) |

## Build and run

```bash
make                    # compile megakernel.cu -> siglip_vision
./siglip_vision         # run on B200, prints timing + TFLOPS + checksum
make timing && ./siglip_timing | tee clock64_timing.txt | python3 analyze_timing.py
ncu --set source --csv ./siglip_vision > source_counters_raw.csv && python3 analyze_source_counters.py source_counters_raw.csv
```

## Key constraints

- Target: `sm_100a` (B200, 148 SMs)
- `cta_group::2` with `__cluster_dims__(2,1,1)` — 74 clusters of 2 CTAs
- TMEM: 512 cols/SM total, single alloc of TN*2 for double buffering (learned from matmul_v7: two separate allocs deadlock, single alloc works)
- SMEM: 228 KB/SM — current usage ~201 KB (4-stage pipeline + double-buffered epilogue staging, ~27 KB free)
- All inline PTX — no CUTLASS dependency for the kernel itself
- `megakernel.cu` is hand-edited directly
- Expected checksum: 1769472.0, C[0,0..3] = 1728.0
- ml_phase init must account for odd tile_start (start_buf-dependent)
