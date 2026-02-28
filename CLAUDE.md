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

## Current bottlenecks (from ncu profiling)

The kernel is **epilogue-bound** (confirmed by clock64 F15 profiling). W1 stalls 1,530 cycles/tile (25% of tile time) waiting for the epilogue to release the TMEM buffer. The epilogue takes ~4,926 cycles (Phase 1+2A interleaved: 4,102 cycles 83.3%, Phase 2B: 824 cycles 16.7%), overrunning the K-loop's 4,017 cycles by ~909 cycles. Double-buffered SMEM staging (F18) reduced the overrun from ~1,760 to ~909 cycles by hiding Phase 2A stores inside Phase 1B's TMEM_WAIT stalls.

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

2. **Epilogue-bound** (confirmed by clock64 F15, partially mitigated by F18). Phase 1 (TMEM readback) still dominates at 83.3%. Double-buffered SMEM staging (F18) hides Phase 2A stores in TMEM stalls, reducing epilogue from 5,833 to 4,926 cycles (-15.6%). Direct global stores (F17) made Phase 1 19% slower due to L1 contention — SMEM staging is essential.

3. **Zero TMA multicast** — not actionable. B is N-split across CTAs (CTA0 loads B[k, n:n+128], CTA1 loads B[k, n+128:n+256]). cta_group::2 MMA reads B from both CTAs (`scope_2cta`). Multicast requires identical data at the same SMEM offset — inapplicable to complementary B halves.

4. **SMEM bank conflicts 32%/22%** (ld/st) — significant but hidden in K-loop shadow. STAGING_ROW_PAD doesn't fully prevent Phase 2 transposed read conflicts.

5. **Warp latency dominated by TMEM** — `long_scoreboard` at 390K cycles/warp is 3.3× productive `selected` (119K). The scheduler hides this (only 4.7% in pct view) but it means epilogue warps spend most of their time waiting.

6. **Register pressure (223 regs/thread, 0 spills)** — limits occupancy to 1 CTA/SM. Not actionable without major restructuring.

**To go faster:** Larger TM to amortize per-tile overhead, further epilogue Phase 1 optimization, or next-tile TMA prefetch (F20 showed ~0% — DRAM latency bottleneck, not scheduling). TMA multicast for B is not applicable (B is N-split across CTAs). See `docs/make_better.md`.

**Tested and ruled out:** See `EXPERIMENTS.md` for 19 experiments with hypotheses, results, and analysis. F19 confirmed Phase 2B's LSU stores contend with K-loop (+170 cycles, +4.2%) when overlapped via early mbar signal — but TMA bulk stores (`cp.async.bulk`, 32×256B) are 3× slower than parallel manual stores due to per-instruction overhead. Padded SMEM (272-byte rows for bank-conflict-free Phase 1B) prevents single-shot TMA tensor stores.

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
- SMEM: 228 KB/SM — current usage ~201 KB (4-stage pipeline + double-buffered epilogue staging, ~27 KB free)
- All inline PTX — no CUTLASS dependency for the kernel itself
- `megakernel.cu` is hand-edited directly
- Expected checksum: 1769472.0, C[0,0..3] = 1728.0
- ml_phase init must account for odd tile_start (start_buf-dependent)
