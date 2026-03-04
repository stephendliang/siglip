NVCC     = nvcc
ARCH     = sm_100a
CFLAGS   = -gencode arch=compute_100a,code=$(ARCH) -O3 -std=c++17 --ptxas-options=-v
LDFLAGS  = -lcurand -lcuda
TARGET   = siglip_vision
CU       = megakernel.cu

CUTLASS_DIR = third_party/cutlass
CUTLASS_INC = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include
CUTLASS_FLAGS = -std=c++17 --expt-relaxed-constexpr

.PHONY: all clean timing fc1-gelu cutlass-bench cutlass-bench-max cutlass-sass calibration cublas-bench sweep sweep-fast sweep-full sass-tool

all: $(TARGET)

$(TARGET): $(CU) kernel_common.cuh kernel_body.cuh
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

timing: $(CU) kernel_common.cuh kernel_body.cuh
	$(NVCC) $(CFLAGS) -DTIMING $< -o siglip_timing $(LDFLAGS)

# ── FC1+GELU kernel ──
fc1-gelu: fc1_gelu.cu kernel_common.cuh kernel_body.cuh
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# ── CUTLASS benchmark (per-tensor FP8, grid search) ──
cutlass-bench: bench/cutlass_bench.cu
	$(NVCC) $(CFLAGS) $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

# Extended CUTLASS sweep for stronger baseline search (more tile/cluster configs)
cutlass-bench-max: bench/cutlass_bench.cu
	$(NVCC) $(CFLAGS) -DCUTLASS_EXTENDED_SWEEP=1 $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

# ── SASS dump ──
cutlass-sass: cutlass-bench
	@mkdir -p sass
	cuobjdump --dump-sass cutlass-bench > sass/cutlass.txt
	@echo "SASS dumped to sass/cutlass.txt"

# ── Calibration microbenchmarks (SASS decoder verification) ──
calibration: bench/calibration.cu
	$(NVCC) $(CFLAGS) $< -o $@

cublas-bench: bench/cublas_bench.cu
	$(NVCC) $(CFLAGS) -std=c++17 $< -o $@ -lcublasLt -lcublas

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

clean:
	rm -f $(TARGET) siglip_timing fc1-gelu cutlass-bench cutlass-bench-max cublas-bench calibration
	rm -rf sass/
