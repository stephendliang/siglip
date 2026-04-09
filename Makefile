NVCC     = nvcc
ARCH     = sm_100a
CFLAGS   = -gencode arch=compute_100a,code=$(ARCH) -O3 -std=c++17 -lineinfo --ptxas-options=-v --cudart=static
LDFLAGS  = -lcurand_static -lculibos -lcuda
TARGET   = patch_embed
CU       = patch_embed.cu

CUTLASS_DIR = third_party/cutlass
CUTLASS_INC = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include
CUTLASS_FLAGS = -std=c++17 --expt-relaxed-constexpr

.PHONY: all clean timing fc1-gelu fc2 fc2-timing fc2-w3 fc2-w3-8w fc2-w3-fp32 fc2-w3-strip fc2-w3-self fc2-w3-atomic fc2-w3-spin fc2-w3-grid fc2-w3-inline fc2-w3-noprefill fc2-w3-ns7 fc2-w3-clc fc2-ldg fc2-ldg-strip fc2-ldg-gemm fc2-cutlass fc2-cutlass-strip fc2-cutlass-static fc2-cutlass-static-strip fc2-hybrid fc2-hybrid-strip fc2-hybrid-mma fc2-hybrid-phase3 cutlass-bench cutlass-bench-fc1 cutlass-bench-fc2 cutlass-bench-max cutlass-bench-fc1-max cutlass-bench-fc2-max cutlass-sass calibration cublas-bench cublas-bench-fc1 cublas-bench-fc2 sweep sweep-fast sweep-full sass-tool compare calib-tput calib-lat calib-conflict calib-warp calib-all tma-bench mma-bench stmatrix-bench

all: $(TARGET)

$(TARGET): $(CU) kernel_common.cuh kernel_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) $< -o $@ $(LDFLAGS)

timing: $(CU) kernel_common.cuh kernel_body.cuh
	$(NVCC) $(CFLAGS) -DTIMING $< -o patch_embed_timing $(LDFLAGS)

# ── FC1+GELU kernel ──
fc1-gelu: fc1_gelu.cu kernel_common.cuh kernel_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) $< -o $@ $(LDFLAGS)

# ── FC2 kernel ──
fc2: fc2.cu kernel_common.cuh kernel_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) $< -o $@ $(LDFLAGS)

fc2-timing: fc2.cu kernel_common.cuh kernel_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTIMING $< -o $@ $(LDFLAGS)

# ── FC2 W3 kernel (standalone, CUTLASS-style shared-SMEM epilogue) ──
fc2-w3: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $< -o $@ $(LDFLAGS)

fc2-w3-8w: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DNUM_IDLE_WARPS=1 $< -o $@ $(LDFLAGS)

fc2-w3-epi1: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DNUM_EPI_WARPS=1 $< -o $@ $(LDFLAGS)

fc2-w3-epi2: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DNUM_EPI_WARPS=2 $< -o $@ $(LDFLAGS)


fc2-w3-fp32: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DFP32_EPILOGUE $< -o $@ $(LDFLAGS)

fc2-w3-strip: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DSTRIP_EPILOGUE $< -o $@ $(LDFLAGS)

fc2-w3-gemm: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DGEMM_ONLY $< -o $@ $(LDFLAGS)

fc2-w3-drain: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DLDS_DRAIN $< -o $@ $(LDFLAGS)

fc2-w3-reorder: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DEPI_REORDER $< -o $@ $(LDFLAGS)

fc2-w3-bidir: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DBIDIR_SNAKE $< -o $@ $(LDFLAGS)

fc2-w3-msnake: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DM_SNAKE $< -o $@ $(LDFLAGS)

fc2-w3-self: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DSELF_LOAD $< -o $@ $(LDFLAGS)

fc2-w3-atomic: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=1 $< -o $@ $(LDFLAGS)

fc2-w3-spin: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=2 $< -o $@ $(LDFLAGS)

fc2-w3-grid: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=3 $< -o $@ $(LDFLAGS)

fc2-w3-sched: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=4 $< -o $@ $(LDFLAGS)

fc2-w3-inline: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=6 $< -o $@ $(LDFLAGS)

fc2-w3-noprefill: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DNO_PREFILL $< -o $@ $(LDFLAGS)

fc2-w3-ns7: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DN_STAGES=7 $< -o $@ $(LDFLAGS)

fc2-w3-clc: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DL2_HINTS=1 $< -o $@ $(LDFLAGS)

# ── FC2 LDG kernel (LDG/STG epilogue, zero staging SMEM) ──
fc2-ldg: fc2_ldg.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $< -o $@ $(LDFLAGS)

fc2-ldg-strip: fc2_ldg.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DSTRIP_EPILOGUE $< -o $@ $(LDFLAGS)

fc2-ldg-gemm: fc2_ldg.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DGEMM_ONLY $< -o $@ $(LDFLAGS)

# ── FC2 CUTLASS kernel (CUTLASS GemmUniversal, reference epilogue) ──
fc2-cutlass: fc2_cutlass.cu
	$(NVCC) $(CFLAGS) $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

fc2-cutlass-strip: fc2_cutlass.cu
	$(NVCC) $(CFLAGS) -DSTRIP_EPILOGUE $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

fc2-cutlass-static: fc2_cutlass.cu
	$(NVCC) $(CFLAGS) -DSTATIC_SCHED $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

fc2-cutlass-static-strip: fc2_cutlass.cu
	$(NVCC) $(CFLAGS) -DSTATIC_SCHED -DSTRIP_EPILOGUE $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

# ── FC2 Hybrid kernel (our PTX mainloop + CUTLASS epilogue) ──
fc2-hybrid: fc2_hybrid.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

fc2-hybrid-strip: fc2_hybrid.cu
	$(NVCC) $(CFLAGS) -DSTRIP_EPILOGUE $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

fc2-hybrid-mma: fc2_hybrid.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DHYBRID_MMA $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

fc2-hybrid-phase3: fc2_hybrid.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DHYBRID_PHASE3 $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

fc2-hybrid-phase4: fc2_hybrid.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DHYBRID_PHASE4 $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

# ── CUTLASS benchmark (per-tensor FP8, grid search) ──
cutlass-bench: bench/cutlass_bench.cu bench/siglip_periodic_add.hpp
	$(NVCC) $(CFLAGS) $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

cutlass-bench-fc1: bench/cutlass_bench.cu
	$(NVCC) $(CFLAGS) -DBENCH_N=3072 -DBENCH_EPILOGUE=2 $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

cutlass-bench-fc2: bench/cutlass_bench.cu
	$(NVCC) $(CFLAGS) -DBENCH_N=768 -DBENCH_K=3072 -DBENCH_EPILOGUE=3 $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

# Extended CUTLASS sweep for stronger baseline search (more tile/cluster configs)
cutlass-bench-max: bench/cutlass_bench.cu bench/siglip_periodic_add.hpp
	$(NVCC) $(CFLAGS) -DCUTLASS_EXTENDED_SWEEP=1 $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

cutlass-bench-fc1-max: bench/cutlass_bench.cu
	$(NVCC) $(CFLAGS) -DBENCH_N=3072 -DBENCH_EPILOGUE=2 -DCUTLASS_EXTENDED_SWEEP=1 $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

cutlass-bench-fc2-max: bench/cutlass_bench.cu
	$(NVCC) $(CFLAGS) -DBENCH_N=768 -DBENCH_K=3072 -DBENCH_EPILOGUE=3 -DCUTLASS_EXTENDED_SWEEP=1 $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

# ── SASS dump ──
cutlass-sass: cutlass-bench
	@mkdir -p sass
	cuobjdump --dump-sass cutlass-bench > sass/cutlass.txt
	@echo "SASS dumped to sass/cutlass.txt"

# ── Calibration microbenchmarks (SASS decoder verification) ──
calibration: bench/calibration.cu
	$(NVCC) $(CFLAGS) $< -o $@

# ── Generated calibration benchmarks (instruction DB + generator) ──
CALIB_GEN = bench/calib/gen_kernels.py bench/calib/instruction_db.py

bench/calib/gen_tput.cu bench/calib/gen_lat.cu bench/calib/gen_conflict.cu: $(CALIB_GEN)
	python3 bench/calib/gen_kernels.py

calib-tput: bench/calib/gen_tput.cu
	$(NVCC) $(CFLAGS) $< -o $@

calib-lat: bench/calib/gen_lat.cu
	$(NVCC) $(CFLAGS) $< -o $@

calib-conflict: bench/calib/gen_conflict.cu
	$(NVCC) $(CFLAGS) $< -o $@

bench/calib/gen_warp_scaling.cu: bench/calib/gen_warp_scaling.py
	python3 bench/calib/gen_warp_scaling.py

calib-warp: bench/calib/gen_warp_scaling.cu
	$(NVCC) $(CFLAGS) $< -o $@

calib-all: calib-tput calib-lat calib-conflict calib-warp

# ── TMA microbenchmark (latency, throughput, SMEM contention) ──
tma-bench: bench/tma_bench.cu
	$(NVCC) $(CFLAGS) $< -o $@ -lcuda

# ── MMA microbenchmark (UTCQMMA throughput, latency, shadow budget) ──
mma-bench: bench/mma_bench.cu
	$(NVCC) $(CFLAGS) $< -o $@ -lcuda

# ── stmatrix microbenchmark (layout characterization + throughput vs STS.128) ──
stmatrix-bench: bench/stmatrix_bench.cu
	$(NVCC) $(CFLAGS) $< -o $@ -lcuda

cublas-bench: bench/cublas_bench.cu
	$(NVCC) $(CFLAGS) -std=c++17 $< -o $@ -lcublasLt -lcublas

cublas-bench-fc1: bench/cublas_bench.cu
	$(NVCC) $(CFLAGS) -std=c++17 -DBENCH_N=3072 -DBENCH_EPILOGUE=2 $< -o $@ -lcublasLt -lcublas

cublas-bench-fc2: bench/cublas_bench.cu
	$(NVCC) $(CFLAGS) -std=c++17 -DBENCH_N=768 -DBENCH_K=3072 -DBENCH_EPILOGUE=3 $< -o $@ -lcublasLt -lcublas

# ── SASS analysis C++ tool ──
sass-tool:
	$(MAKE) -C tools/sass

# ── Grid search (Python sweep) ──
sweep: tools/grid_search.py $(CU)
	python3 tools/grid_search.py --tier all

sweep-fast: tools/grid_search.py $(CU)
	python3 tools/grid_search.py --tier 2

sweep-full: tools/grid_search.py $(CU)
	python3 tools/grid_search.py --full-cross

# ── Unified comparison (cuBLAS vs CUTLASS vs ours, ANOVA) ──
compare:
	python3 tools/compare_all.py --csv data/compare.csv

compare-fast:
	python3 tools/compare_all.py --runs 5 --layer patch_embed --csv data/compare.csv

clean:
	rm -f $(TARGET) patch_embed_timing fc1-gelu fc2 fc2-w3 fc2-w3-8w fc2-w3-fp32 fc2-w3-strip fc2-w3-reorder fc2-w3-self fc2-w3-atomic fc2-w3-spin fc2-w3-grid fc2-w3-inline fc2-w3-noprefill fc2-w3-ns7 fc2-w3-clc fc2-ldg fc2-ldg-strip fc2-ldg-gemm fc2-cutlass fc2-cutlass-strip fc2-cutlass-static fc2-cutlass-static-strip fc2-hybrid fc2-hybrid-strip fc2-hybrid-mma fc2-hybrid-phase3 cutlass-bench cutlass-bench-fc1 cutlass-bench-fc2 cutlass-bench-max cutlass-bench-fc1-max cutlass-bench-fc2-max cublas-bench cublas-bench-fc1 cublas-bench-fc2 calibration calib-tput calib-lat calib-conflict calib-warp tma-bench mma-bench stmatrix-bench
	rm -rf sass/
