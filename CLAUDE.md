# SigLIP2 Vision Encoder — Persistent Megakernel

Hand-tuned Blackwell (SM100a) persistent megakernel for `google/siglip2-base-patch16-224` patch embed GEMM.
FP8 (E4M3) precision, tcgen05 WGMMA, TMA, `cta_group::2` with 2-CTA clusters. Cross-compiled on CPU VPS, runs on B200.

## Current state

**0.519 ms / 2108 TFLOPS** fused (GEMM + bias + pos_embed) with `MBAR_EARLY=1 STAGGER_CYCLES=160` — **38% faster** than cuBLAS end-to-end (0.835 ms = best GEMM + unfused pos_embed). Defaults produce 0.524 ms / 2090 TFLOPS.

The kernel's value is **fusion**: the overlapped epilogue eliminates the 0.470 ms unfused pos_embed overhead entirely.

cuBLAS pure GEMM is faster: 0.365 ms / 3001 TFLOPS (per-tensor FP8, best-of-8 algos, 256MB workspace).
Our effective TFLOPS (2090-2108) counts fused epilogue time in the denominator — not a fair GEMM-only comparison.

GEMM: `[928256, 768] x [768, 768]^T` with fused bias + positional embedding add, BF16 output.
Batch = 4736 images x 196 patches = 928256 rows. Square weight matrix (768x768).

The kernel is correct (non-uniform validation, checksum validated) and stable.

## Current bottlenecks (clock64 timing build, post-F31)

The kernel is **epilogue-bound** in a balanced producer-consumer equilibrium. Per-tile cycle budget (pre-F40):

```
W1:       epi_wait(1,344) + TMA0(662) + K-loop(4,059) = 6,066
Epilogue: ml_wait(1,538) + Phase1(4,345) + Phase2B(273) = 6,156
```

F40 interleaved TMA stores hide Phase 2 latency inside Phase 1 TMEM stall windows, reducing effective epilogue cost.

**Key facts:**
- Phase 1 TMEM readback = binding constraint. F40 overlaps TMA stores with TMEM stalls.
- K-loop: 4,059 cycles. Precomputed descriptors + manual unroll from F28.
- TMA multicast not applicable (B is N-split across CTAs).
- ~205 regs/thread (varies with `CVT_ADD_FUSED` and `PHASE1_UNROLL`), 0 spills. Limits occupancy to 1 CTA/SM.
- Per-warp Phase 1 spread: 269 cycles (reduced from 330 by F31 stagger). Contention-based (confirmed by rg-swap diagnostic).
- Timing build uses ~245 regs (distorts cycles vs production at ~205 regs). Wall clock is ground truth.

Run `python3 analyze_timing.py clock64_timing.txt` for full equilibrium analysis and what-if projections.
Run `python3 analyze_source_counters.py source_counters_raw.csv` for per-instruction stall breakdown.
See `EXPERIMENTS.md` for experiments (F1-F40) with hypotheses, results, and analysis. See `FUTURE_PROPOSALS.md` for optimization roadmap.

## Kernel structure

Warp-specialized, 6 warps (192 threads), `cta_group::2`, `__cluster_dims__(2,1,1)`:

- **W0**: TMA async bulk loads (A + B tiles, both CTAs load independently)
- **W1**: TMEM alloc (512 cols, single alloc for double buffering) + `tcgen05.mma.cta_group::2` accumulation into TMEM (CTA0 lane-0 only, multicast commit to both CTAs)
- **W2-W5**: Overlapped epilogue (4 warps) — each warp independently polls mainloop mbarrier, then runs unified SMEM-staged store: Phase 1 (all 256 cols: x32 `tcgen05.ld` → BF16 add → CVT → `st.shared` to 4 SWIZZLE_128B staging regions, with **interleaved TMA stores** every 2 regions hiding in TMEM stall windows), Phase 2 (`cp.async.bulk.commit_group` only — all TMA stores already issued inline), **mbar_arrive** signals TMEM free for W1

TM=128 rows / 32 rows per warp = 4 row groups. W2-W5 each own a row group (all 256 cols). No column splitting (is_split always 0). `epilogue_store` is templated on `<NC_START, NC_END>` so the compiler sees constant loop bounds; unroll depth controlled by `PHASE1_UNROLL` (default 2).

The overlapped epilogue for tile N-1 runs concurrently with the K-loop for tile N (double-buffered TMEM, mbarrier-protected). After the tile loop, W2-5 run a drain epilogue for the last tile.

### Tile config
- Tile: 256x256x128 (M=2x128 from cta_group::2, N=256, K=128)
- TMEM: single alloc of 512 cols (TN*2), double-buffered via column offset (buf*TN)
- SMEM: 4-stage pipeline (131 KB) + epilogue staging (4 warps x 16,384 = 64 KB, SWIZZLE_128B) = ~197 KB total of 228 KB
- Tiles: 3626 M-tiles x 3 N-tiles = 10,878 total, snake ordering
- ~205-211 registers/thread, 0 spills (varies with `CVT_ADD_FUSED`, `PHASE1_UNROLL`, `MBAR_EARLY`)
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
| `cutlass_bench.cu` | CUTLASS grid search — sweeps 13 tile/cluster configs × FP32/BF16 epilogue. Single binary, no `-D` flags |
| `EXPERIMENTS.md` | Experiment log (F1-F40), profiling data, optimization history |
| `FUTURE_PROPOSALS.md` | Optimization roadmap (F21-F28, all executed) with results |
| `grid_search.py` | Compile-time parameter sweep — tiered search, cross-product, CSV output. Replaces manual Makefile targets |
| `Makefile` | Build rules (sm_100a, nvcc flags). `make timing` for clock64 build |
| **Analysis scripts** | |
| `analyze_timing.py` | Parses `clock64_timing.txt` → equilibrium analysis, ceiling projections, what-if scenarios |
| `analyze_source_counters.py` | Parses ncu SourceCounters CSV → stall breakdown, MMA stats, W1 budget, instruction mix |
| `compare.py` | ncu CSV diff tool — usage: `python compare.py a.csv b.csv` |
| `sass_analysis.py` | SASS scheduling analyzer — decodes control words, builds dependency graphs, identifies slack. Has `--calibrate-compare` mode for closed-loop decoder verification against `calibration.cu` runtime |
| `calibration.cu` | 10 microbenchmark kernels (K1-K9) measuring instruction throughput/latency on B200. Used to verify SASS control word bit layout and populate `sass_analysis.py` latency table |
| **Grid search data** | |
| `sweep_results.csv` | Run 1 grid search (145 configs, Tier 1→4 epi warps). Best: 0.519 ms |
| `sweep_results_run2.csv` | Run 2 grid search (145 configs, Tier 1→5 epi warps). Best: 0.522 ms |
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

# CUTLASS grid search (single binary, sweeps all tile/cluster configs)
make cutlass-bench      # ~1-2 min compile (26 CUTLASS template instantiations)
./cutlass-bench         # full sweep (4736 imgs, ~5 min runtime)
./cutlass-bench 1       # quick test (148 imgs, ~30s)

# CUTLASS extended search (stronger baseline pass)
make cutlass-bench-max  # broader tile/cluster sweep, slower compile/runtime
./cutlass-bench-max     # use when locking baseline
./cutlass-bench-max 1   # quick sanity pass

# Megakernel parameter grid search (replaces manual -D flag Makefile targets)
python3 grid_search.py --tier 1          # structure: N_STAGES × NUM_EPI_WARPS
python3 grid_search.py --tier 2          # epilogue: INTERLEAVE × MBAR × STAGGER × TMEM
python3 grid_search.py --tier 3          # tuning: PHASE1_UNROLL × SNAKE_ORDER × CVT_ADD_FUSED
python3 grid_search.py --tier all        # sequential 1→2→3, pinning winners
python3 grid_search.py --full-cross      # all parameters crossed (~3000 configs)

# SASS scheduling analysis
cuobjdump --dump-sass siglip_vision > sass_dump.txt
python3 sass_analysis.py sass_dump.txt                          # annotated listing
python3 sass_analysis.py sass_dump.txt --section 0x1300 0x1a70  # address range (e.g., epilogue)
python3 sass_analysis.py sass_dump.txt --deps                   # dependency + slack analysis
python3 sass_analysis.py --cubin siglip_vision                  # runs cuobjdump internally

# Calibration: verify SASS control word decoder on B200
make calibration          # compile calibration.cu
./calibration > cal_output.txt
cuobjdump --dump-sass calibration > cal_sass.txt
python3 sass_analysis.py cal_sass.txt --calibrate-compare                          # SASS-only
python3 sass_analysis.py cal_sass.txt --calibrate-compare --runtime cal_output.txt # compare vs runtime
```

### CUTLASS bench details

`cutlass_bench.cu` is a self-contained grid search over 13 tile/cluster configs (standard build), or 20 configs in extended mode (`-DCUTLASS_EXTENDED_SWEEP=1`). For each config, it measures:
1. **GEMM-only** (beta=0, FP32 epilogue) — pure compute baseline
2. **Fused FP32** (beta=1, FP32 epilogue) — `D = float(acc) + float(C)`
3. **Fused BF16** (beta=1, BF16 epilogue) — `D = bf16(acc) + C` (matches custom kernel's cvt_add_bf16x2)

Core template: `GemmInstance<TM, TN, TK, CM, CN, EpiCompute>` — full CUTLASS type chain parameterized on tile shape, cluster shape, and epilogue compute type (`float` or `cutlass::bfloat16_t`). Invalid configs return -1.0f via `can_implement` check. Results sorted by fused BF16 ms.

## Key constraints

- Target: `sm_100a` (B200, 148 SMs)
- `cta_group::2` with `__cluster_dims__(2,1,1)` — 74 clusters of 2 CTAs
- TMEM: 512 cols/SM total, single alloc of TN*2 for double buffering (learned from matmul_v7: two separate allocs deadlock, single alloc works)
- SMEM: 228 KB/SM — current usage ~192 KB (4-stage pipeline + unified SWIZZLE_128B epilogue staging, ~36 KB free)
- All inline PTX — no CUTLASS dependency for the kernel itself
- `megakernel.cu` is hand-edited directly
- Validation: non-uniform B (alternating FP8 rows: 1.5/1.0 → distinct even/odd col accumulators), non-uniform bias/pos_embed, 1024 strided checksum + 32 CPU reference spot checks (valid=1 in @@RESULT)
- **OFF_STAGING must be 1024-byte aligned** for SWIZZLE_128B correctness — TMA swizzle operates on absolute SMEM address bits `addr[6:4] ^= addr[9:7]`; misalignment causes systematic 8-col (16-byte) group swaps in output. Swizzle period = 8 rows x 128 bytes = 1024 bytes.
- ml_phase init must account for odd tile_start (start_buf-dependent)

## Grid search findings (2 runs × 145 configs)

Full parameter sweep results in `sweep_results.csv` (run 1) and `sweep_results_run2.csv` (run 2). Run 1 pinned 4 epi warps at Tier 1; Run 2 pinned 5 epi warps. Key findings:

### Performance (valid configs only, sorted by impact)

| Parameter | Best | Default | Effect |
|-----------|------|---------|--------|
| SNAKE_ORDER | 1 | 1 | **Critical**: 0=1 costs ~50-200 TFLOPS (10-40%). Snake ordering essential. |
| PHASE1_UNROLL | 2 | 2 | **Critical**: 1 is catastrophic (-30%, ~1500 TFLOPS). 4 slightly slower + 255 regs. |
| INTERLEAVE_STRATEGY | 2 | 2 | **Best for correctness + perf.** 0 = ~30 TFLOPS slower. 3 = ~50 TFLOPS slower. 1 = fast but has sporadic race conditions. |
| MBAR_EARLY | 1 | 0 | +5-18 TFLOPS consistently. 0.519 vs 0.524 ms at optimal stagger. |
| STAGGER_CYCLES | 120-160 | 80 | Sweet spot 80-160; 0 costs ~5 TFLOPS; 200 slightly worse. |
| N_STAGES | 4 | 4 | 3 too few (0.557 ms). 5 works but +20 regs for marginal gain. |
| TMEM_LOAD_WIDTH | 32 | 32 | 64 burns 255 regs, ~10-20 TFLOPS slower than 32 (209 regs). |
| CVT_ADD_FUSED | 0 or 1 | 1 | Equivalent within noise. |
| NUM_EPI_WARPS | 4 or 5 | 4 | 5 matches 4 in perf (~0.523 ms) but has more validation failures with some interleave strategies. |

### Top configs (cross-run)

```
Run 1 best: MBAR_EARLY=1 STAGGER_CYCLES=160              → 0.519 ms / 2108 TFLOPS (211 regs, 4 epi warps)
Run 1 #2:   MBAR_EARLY=1 STAGGER_CYCLES=120              → 0.520 ms / 2104 TFLOPS (211 regs)
Run 2 best: INTER=1 MBAR=1 EPI=5 PH1U=4 STAG=160 TLW=64 → 0.522 ms / 2096 TFLOPS (235 regs)
Defaults:                                                  → 0.524 ms / 2090 TFLOPS (209 regs)
```

### Validation failure patterns (remaining race conditions)

The OFF_STAGING 1024-alignment fix resolved the systematic column-swap bug. Remaining failures are interleave-strategy-dependent races:

| Config | 4 epi warps | 5 epi warps |
|--------|-------------|-------------|
| INTERLEAVE_STRATEGY=0 | All pass (32/32) | **ALL FAIL (32/32)** |
| INTERLEAVE_STRATEGY=1 | Sporadic (~17/32 pass) | Sporadic (~22/32 pass) |
| INTERLEAVE_STRATEGY=2 | All pass (32/32) | Almost all pass (31/32) |
| INTERLEAVE_STRATEGY=3 | All pass (32/32) | **ALL FAIL (32/32)** |

INTERLEAVE_STRATEGY=2 (default) is the only fully robust strategy across both warp counts. Strategies 0 and 3 have latent race conditions exposed by 5 epi warps. Strategy 1 has sporadic races at both counts.
