# B200 Task List

Shopping list for B200 rental sessions ($5/hr — batch tasks efficiently).

## cuBLAS SASS Analysis

**Goal:** Dump and analyze the actual FP8 GEMM SASS that cuBLAS JIT-compiles on B200.

**Why:** cuBLAS FP8 GEMM kernels ship as zstd-compressed PTX in `libcublasLt.so` (150 numeric XMMA entries like `_ZN4xmma-28063.sm_100.elf.bin`). The CUDA driver JIT-compiles them to sm_100a SASS at runtime. Zero UTCQMMA instructions exist in the distributed library — static analysis from a non-B200 machine is impossible.

The only pre-compiled sm_100a kernels in the library are 8 DGEMM variants (`cublasLt_fused_imma_dgemm_*`). Everything else (FP8, BF16, FP16 GEMM) is PTX-only.

**Steps:**
1. Build and run cuBLAS FC1 bench: `make cublas-bench-fc1 && ./cublas-bench-fc1`
2. Capture the kernel with ncu: `ncu --set full -o cublas_fc1 ./cublas-bench-fc1`
3. Dump SASS from the JIT cache or ncu report:
   - `ncu --import cublas_fc1.ncu-rep --page source` for annotated SASS
   - Or find JIT cache: `find ~/.nv/ComputeCache/ -newer /tmp/marker -name '*.cubin'` then `cuobjdump --dump-sass`
4. Run through our SASS analyzer: `python3 tools/sass_analysis.py <sass_dump>`
5. Compare epilogue structure (UTCQMMA count, UTMASTG pattern, FENCE placement) against our kernel and CUTLASS

**Also capture:** cuBLAS patch_embed (`cublas-bench`) and FC2 (`cublas-bench-fc2`) for comparison.

## FC1/FC2 Grid Search

**Goal:** Run parameter grid search on FC1 and FC2 kernels (currently only patch_embed has been swept).

**Steps:**
1. `python3 tools/grid_search.py --kernel fc1_gelu --tier all`
2. `python3 tools/grid_search.py --kernel fc2 --tier all`
3. Save results to `data/sweep_results_fc1.csv` and `data/sweep_results_fc2.csv`

FC1 (N=3072, K=768) and FC2 (N=768, K=3072) have different tile counts and K-iteration counts — optimal parameters may differ from patch_embed.

## CUTLASS Sweep

**Goal:** Run full CUTLASS benchmark sweep for all three layers.

**Steps:**
1. `./tools/cutlass_sweep.sh 32 max` — full extended sweep, all layers
2. Save output for comparison against our kernels

## ncu Profiling

**Goal:** Fresh ncu profiles of production kernels after any code changes.

**Steps:**
1. `ncu --set source --csv ./siglip_vision > data/source_counters_raw.csv`
2. `ncu --set source --csv ./fc1-gelu > data/source_counters_fc1.csv`
3. `ncu --set source --csv ./fc2 > data/source_counters_fc2.csv`
4. `python3 tools/analyze_source_counters.py data/source_counters_raw.csv`

## Calibration Microbenchmarks

**Goal:** Run expanded calibration suite (K13-K26) for SASS decoder verification.

**Steps:**
1. `make calibration && ./calibration > data/cal_output.txt`
2. `cuobjdump --dump-sass calibration > data/cal_sass.txt`
3. `python3 tools/sass_analysis.py data/cal_sass.txt --calibrate-compare --runtime data/cal_output.txt`
