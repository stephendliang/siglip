# SigLIP2 Vision Encoder — Persistent Megakernel

Hand-tuned Blackwell (SM100a) persistent megakernel for `google/siglip2-base-patch16-224` patch embed GEMM.
FP8 (E4M3) precision, tcgen05 WGMMA, TMA, `cta_group::2` with 2-CTA clusters. Cross-compiled on CPU VPS, runs on B200.

## Current state

**0.633 ms** fused (GEMM + bias + pos_embed) — **24% faster** than cuBLAS end-to-end (0.835 ms = best GEMM + unfused pos_embed).

The kernel's value is **fusion**: the overlapped epilogue eliminates the 0.470 ms unfused pos_embed overhead entirely.

cuBLAS pure GEMM is faster: 0.365 ms / 3001 TFLOPS (per-tensor FP8, best-of-8 algos, 256MB workspace).
Our effective TFLOPS (1729) counts fused epilogue time in the denominator — not a fair GEMM-only comparison.

GEMM: `[928256, 768] x [768, 768]^T` with fused bias + positional embedding add, BF16 output.
Batch = 4736 images x 196 patches = 928256 rows. Square weight matrix (768x768).

The kernel is correct (checksum validated) and stable.

## Current bottlenecks (from ncu profiling)

The kernel is **L1 throughput-bound** (85% of peak). SMEM-staged coalesced stores eliminated the uncoalesced store bottleneck (100% excess L2 sectors → 33%), boosting warp issue rate by 35%.

**Warp stall breakdown:**

| Stall | % of peak | Notes |
|---|---|---|
| long_scoreboard (TMEM) | 4.4% | Largest stall, reduced from 6.4% |
| short_scoreboard (SMEM) | 1.1% | New: SMEM staging ld/st dependency chains |
| wait (TMA) | 1.2% | Pipeline wraparound stalls |
| sleeping | 1.1% | Warps parked between tiles |
| barrier | 0.8% | Cluster sync |
| selected (issuing) | 19.1% | Productive execution (up from 14.1%) |

**Memory throughput:**

| Subsystem | Utilization |
|---|---|
| **L1 tex** | **85%** — primary bottleneck |
| L2 | 54% |
| DRAM | 27% |

**Key findings:**

1. **L1 throughput (85%)** — still the dominant bottleneck, but now with healthier instruction mix. SMEM staging eliminated L1 read-modify-write amplification from uncoalesced stores (L1 hit rate dropped 61.7% → 33.6%, confirming elimination of partial-line writes).

2. **TMEM load latency (`long_scoreboard`, 4.4%)** — reduced from 6.4%. Less L1 backpressure from coalesced stores means TMEM loads complete faster.

3. **SMEM staging cost (`short_scoreboard`, 1.1%)** — new stall from SMEM ld/st chains in Phase 2. Acceptable tradeoff for the 9.6% speedup.

4. **Register pressure (222 regs/thread, 0 spills)** — up from 216 due to Phase 2 loop variables. Still limits occupancy to 1 CTA/SM.

## Ideas for further improvement

**Most promising:**

- **Smaller output tile (TN=128, revisit)**: With SMEM staging and other epilogue improvements applied, the shorter epilogue loop might tip the balance differently than when we last tried TN=128 (which was 1190 TFLOPS with x16 loads).

- **Warp-specialized epilogue roles**: Instead of 5 identical epilogue warps, have some warps do TMEM→SMEM and others do SMEM→global. Decouples TMEM latency from store latency. Complex to implement.

- **Reduce TMEM traffic**: If accumulator precision could be BF16 instead of FP32, TMEM readback halves. But tcgen05.mma only accumulates to FP32 — would need explicit FP32→BF16 before TMEM store (not available in current ISA).

**Tested and ruled out:** See `EXPERIMENTS.md` for detailed experiment log with hypotheses, results, and analysis.

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
- 222 registers/thread, 0 spills
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
| `docs/profiling.md` | Profiling commands (ncu, cuobjdump, compute-sanitizer) |
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
