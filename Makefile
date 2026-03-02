NVCC     = nvcc
ARCH     = sm_100a
CFLAGS   = -gencode arch=compute_100a,code=$(ARCH) -O3 --ptxas-options=-v
LDFLAGS  = -lcurand -lcuda
TARGET   = siglip_vision
GEN      = gen.py
CU       = megakernel.cu

CUTLASS_DIR = third_party/cutlass
CUTLASS_INC = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include
CUTLASS_FLAGS = -std=c++17 --expt-relaxed-constexpr

.PHONY: all gen clean dry-run timing cutlass-sass sweep sweep-fast sweep-full

all: $(TARGET)

$(TARGET): $(CU)
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

timing: $(CU)
	$(NVCC) $(CFLAGS) -DTIMING $< -o siglip_timing $(LDFLAGS)

# ── CUTLASS benchmark (per-tensor FP8, grid search) ──
cutlass-bench: cutlass_bench.cu
	$(NVCC) $(CFLAGS) $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

# Extended CUTLASS sweep for stronger baseline search (more tile/cluster configs)
cutlass-bench-max: cutlass_bench.cu
	$(NVCC) $(CFLAGS) -DCUTLASS_EXTENDED_SWEEP=1 $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

# ── SASS dump ──
cutlass-sass: cutlass-bench
	@mkdir -p sass
	cuobjdump --dump-sass cutlass-bench > sass/cutlass.txt
	@echo "SASS dumped to sass/cutlass.txt"

# ── Calibration microbenchmarks (SASS decoder verification) ──
calibration: calibration.cu
	$(NVCC) $(CFLAGS) $< -o $@

cublas-bench: cublas_bench.cu
	$(NVCC) $(CFLAGS) -std=c++17 $< -o $@ -lcublasLt -lcublas

# ── Grid search (Python sweep) ──
sweep: grid_search.py $(CU)
	python3 grid_search.py --tier all

sweep-fast: grid_search.py $(CU)
	python3 grid_search.py --tier 2

sweep-full: grid_search.py $(CU)
	python3 grid_search.py --full-cross

clean:
	rm -f $(TARGET) siglip_timing cutlass-bench cutlass-bench-max cublas-bench calibration
	rm -rf sass/
