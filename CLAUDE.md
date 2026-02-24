# SigLIP2 Vision Encoder — Persistent Megakernel

Blackwell (SM100) persistent megakernel for `google/siglip2-base-patch16-224` (86M vision encoder). MXFP8 precision, tcgen05 WGMMA, TMA, cooperative grid launch. Cross-compiled on CPU VPS, runs on B200/B300.

## Current task

Optimizing the patch embed GEMM epilogue. The GEMM mainloop (FP8×FP8→FP32, tcgen05.mma) is near cuBLAS speed. The epilogue (TMEM readback → bias+pos add → FP32→BF16 convert → global store) is the bottleneck due to TMA store serialization, `__syncthreads` coupling all 6 warps, and insufficient TMEM load latency hiding.

**Read `docs/tasks.md` for the specific action items and dependency order.**

## Kernel structure

Warp-specialized, 6 warps (192 threads), `cta_group::1`:
- **W0**: TMA async bulk loads (weight tiles A, B)
- **W1**: `tcgen05.mma` accumulation into TMEM (lane-0 only)
- **W2-W5**: Overlapped epilogue — `tcgen05.ld` from TMEM, bias+pos_embed add, FP32→BF16 convert, store to global

The overlapped epilogue for tile N-1 runs concurrently with the K-loop for tile N. After the tile loop, W0-W3 run a drain epilogue for the last tile.

## Codegen workflow

**Never edit `megakernel.cu` directly.** It is generated.

```
edit gen.py → python3 gen.py → make → run on GPU
```

`gen.py` emits the complete `.cu` from parameterized templates. SMEM offsets, barrier IDs, TMA descriptors, tile arithmetic — all computed from root constants.

## File map

| File | What | Read when |
|------|------|-----------|
| `CLAUDE.md` | This file — project context and orientation | **Always read first** |
| `docs/tasks.md` | Active optimization action items with dependency order | Starting any implementation work |
| `docs/profiling.md` | ncu/cuobjdump/compute-sanitizer commands per task | Before and after every change |
| `docs/architecture.md` | Model specs, HW config, codegen structure, milestones | Understanding overall project scope |
| `gen.py` | Codegen script — **the source of truth** | Implementing any kernel change |
| `megakernel.cu` | Generated CUDA kernel — **do not edit** | Reading current kernel behavior |
| `Makefile` | Build rules (sm_100a, nvcc flags) | Building |
| `bench_sweep.sh` | M-size benchmark sweep | Performance testing |
| `docs/reference/model.txt` | PyTorch model architecture dump | Understanding model dimensions |
| `docs/reference/sass_dump.txt` | SASS disassembly (147KB) | **Only when analyzing specific instructions** |

## Build and run

```bash
make                    # compile megakernel.cu → siglip_vision
./siglip_vision         # run on B200, prints timing + TFLOPS
make dry-run            # print tile analysis without compiling
```

## Key constraints

- Target: `sm_100a` (B200 with 148 SMs, cluster-of-4)
- SMEM: 228 KB per SM
- All inline PTX — no CUTLASS dependency for the kernel itself
- `megakernel.cu` is generated, not hand-edited (except for the current hand-tuned version being optimized)
