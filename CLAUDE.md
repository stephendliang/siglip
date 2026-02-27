# SigLIP2 Vision Encoder — Persistent Megakernel

Hand-tuned Blackwell (SM100a) persistent megakernel for `google/siglip2-base-patch16-224` patch embed GEMM.
FP8 (E4M3) precision, tcgen05 WGMMA, TMA, `cta_group::2` with 2-CTA clusters. Cross-compiled on CPU VPS, runs on B200.

## Current state

**0.700 ms** fused (GEMM + bias + pos_embed) — **16% faster** than cuBLAS end-to-end (0.835 ms = best GEMM + unfused pos_embed).

The kernel's value is **fusion**: the overlapped epilogue eliminates the 0.470 ms unfused pos_embed overhead entirely.

cuBLAS pure GEMM is faster: 0.365 ms / 3001 TFLOPS (per-tensor FP8, best-of-8 algos, 256MB workspace).
Our effective TFLOPS (1564) counts fused epilogue time in the denominator — not a fair GEMM-only comparison.

GEMM: `[928256, 768] x [768, 768]^T` with fused bias + positional embedding add, BF16 output.
Batch = 4736 images x 196 patches = 928256 rows. Square weight matrix (768x768).

The kernel is correct (checksum validated) and stable.

## Current bottlenecks (from ncu profiling)

The kernel is **L1 throughput-bound** (82% of peak). After reducing pipeline stages (6→4) and register pressure (247→216), warps issue instructions more aggressively and L1 bandwidth is now the ceiling. The previous dominant bottleneck (TMEM stalls) has been substantially reduced.

**Warp stall breakdown:**

| Stall | % of peak | Notes |
|---|---|---|
| long_scoreboard (TMEM) | 6.4% | Still largest stall (76% of stall cycles), but much reduced |
| sleeping | 1.3% | Warps parked between tiles |
| wait (TMA) | 0.9% | Pipeline wraparound stalls — minimal even with 4 stages |
| barrier | 0.8% | Cluster sync |
| selected (issuing) | 14.1% | Productive execution |

**Memory throughput:**

| Subsystem | Utilization |
|---|---|
| **L1 tex** | **82%** — primary bottleneck |
| L2 | 60% |
| DRAM write | 24% (1.38 GB written) |

**Key findings:**

1. **L1 throughput (82%)** — the dominant bottleneck. Driven by uncoalesced `st.global` stores: 32 sectors/request vs 16 ideal (100% excess). Each warp writes 32 rows with 1536B stride between lanes, scattering across 32 different 128B cache lines. Fixing this via SMEM staging would directly attack the ceiling.

2. **TMEM load latency (`long_scoreboard`, 6.4%)** — still the biggest single stall reason, but much lower than the previous 56% relative share. 5 epilogue warps + lower register pressure hide latency well.

3. **Register pressure (216 regs/thread, 0 spills)** — improved from 247 after 4-stage pipeline change. Still limits occupancy to 1 CTA/SM.

## Ideas for further improvement

**Most promising:**

- **Epilogue SMEM staging**: Instead of each warp independently loading from TMEM and writing to global, have warps cooperatively load TMEM into SMEM, then write from SMEM with coalesced access patterns. Would fix the 100% excess L1 store sectors (32→16 sectors/request) and directly attack the L1 throughput bottleneck (82% of peak). Cost: extra SMEM (have ~97 KB headroom with 4-stage pipeline) + sync overhead.

- **Smaller output tile (TN=128, revisit)**: With x32 TMEM loads and other epilogue improvements applied, the shorter epilogue loop might tip the balance differently than when we last tried TN=128 (which was 1190 TFLOPS with x16 loads).

**Speculative / higher-effort:**

- **Two-pass output**: Write coalesced to a scratch buffer, then transpose. Eliminates uncoalesced stores but adds a second kernel pass. Only wins if store inefficiency > transpose cost.

- **Warp-specialized epilogue roles**: Instead of 5 identical epilogue warps, have some warps do TMEM→SMEM and others do SMEM→global. Decouples TMEM latency from store latency. Complex to implement.

- **Reduce TMEM traffic**: If accumulator precision could be BF16 instead of FP32, TMEM readback halves. But tcgen05.mma only accumulates to FP32 — would need explicit FP32→BF16 before TMEM store (not available in current ISA).

**Ruled out:**

- **6 epilogue warps** (tested, 0.747 ms) — TMEM bandwidth contention. 5 is the sweet spot; 6 warps issuing concurrent `tcgen05.ld` saturates bandwidth and regresses vs 5.
- **5th epilogue warp with runtime loop bounds** (tested, 0.826 ms at 4 warps) — passing `nc_start`/`nc_end` as function args prevented loop unrolling. Fixed by templating `epilogue_store<NC_START, NC_END>`.
- **x32 TMEM loads** (done, perf-neutral) — bandwidth-bound, not instruction-bound.
- **SMEM prefetch of combined** (tried commit 4ff9644→892766c) — inline BF16 loads from global were faster, likely because L1 cache hits on the small combined tensor.
- **Centralized bar.sync for mbar broadcast** (done, 0.764→0.743 ms) — warp 2 polling + bar.sync to W3-W5 cost 10.3% stall. All epilogue warps polling independently is faster.
- **6 pipeline stages** (was default) — 192 KB SMEM, 247 regs. Reduced to 4 stages: freed 64 KB SMEM, dropped to 216 regs, +3.1% faster (0.722→0.700 ms). 5 stages also tested (0.715 ms, 236 regs).
- **`cta_group::4`** — would need 1024 TMEM cols (exceeds 512/SM HW limit).

## Kernel structure

Warp-specialized, 7 warps (224 threads), `cta_group::2`, `__cluster_dims__(2,1,1)`:

- **W0**: TMA async bulk loads (A + B tiles, both CTAs load independently)
- **W1**: TMEM alloc (512 cols, single alloc for double buffering) + `tcgen05.mma.cta_group::2` accumulation into TMEM (CTA0 lane-0 only, multicast commit to both CTAs)
- **W2-W6**: Overlapped epilogue (5 warps) — each warp independently polls mainloop mbarrier, `tcgen05.ld` from TMEM, inline BF16 bias+pos_embed add, FP32->BF16 convert, `st.global.v8`

TM=128 rows / 32 rows per warp = 4 row groups. W2-W5 each own a row group (all 256 cols). W6 column-splits with W2: both handle row_group 0, W2 does cols 0..127, W6 does cols 128..255. `epilogue_store` is templated on `<NC_START, NC_END>` so the compiler sees constant loop bounds and fully unrolls.

The overlapped epilogue for tile N-1 runs concurrently with the K-loop for tile N (double-buffered TMEM, mbarrier-protected). After the tile loop, W2-6 run a drain epilogue for the last tile.

### Tile config
- Tile: 256x256x128 (M=2x128 from cta_group::2, N=256, K=128)
- TMEM: single alloc of 512 cols (TN*2), double-buffered via column offset (buf*TN)
- SMEM: 4-stage pipeline, 128 KB data + mbarriers = ~131 KB total (N_STAGES parameterized)
- Tiles: 3626 M-tiles x 3 N-tiles = 10,878 total, snake ordering
- 216 registers/thread, 0 spills
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
| `LATEST_AUDIT.md` | ncu profiling data and performance analysis |
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
- SMEM: 228 KB/SM — current usage ~131 KB (4-stage pipeline, ~97 KB free)
- All inline PTX — no CUTLASS dependency for the kernel itself
- `megakernel.cu` is hand-edited directly
- Expected checksum: 1769472.0, C[0,0..3] = 1728.0
- ml_phase init must account for odd tile_start (start_buf-dependent)
