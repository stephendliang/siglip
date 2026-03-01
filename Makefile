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
CUTLASS_TILE ?=

.PHONY: all gen clean dry-run timing tmem-x32 cutlass-sass cutlass-sweep

all: $(TARGET)

$(TARGET): $(CU)
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

timing: $(CU)
	$(NVCC) $(CFLAGS) -DTIMING $< -o siglip_timing $(LDFLAGS)

# F37: wider TMEM loads (SASS analysis — compile with -DTMEM_LOAD_WIDTH=32)
tmem-x32: $(CU)
	$(NVCC) $(CFLAGS) -DTMEM_LOAD_WIDTH=32 $< -o siglip_x32 $(LDFLAGS)

# ── CUTLASS benchmark (per-tensor FP8) ──
# Override tile: make cutlass-bench CUTLASS_TILE="-DTILE_N=256"
cutlass-bench: cutlass_bench.cu
	$(NVCC) $(CFLAGS) $(CUTLASS_INC) $(CUTLASS_FLAGS) $(CUTLASS_TILE) $< -o $@ $(LDFLAGS)

# ── Named tile configs for sweep ──
cutlass-256x128x64: cutlass_bench.cu
	$(NVCC) $(CFLAGS) $(CUTLASS_INC) $(CUTLASS_FLAGS) \
	  -DTILE_M=256 -DTILE_N=128 -DTILE_K=64 -DCLUSTER_M=2 -DCLUSTER_N=1 \
	  $< -o $@ $(LDFLAGS)

cutlass-256x256x64: cutlass_bench.cu
	$(NVCC) $(CFLAGS) $(CUTLASS_INC) $(CUTLASS_FLAGS) \
	  -DTILE_M=256 -DTILE_N=256 -DTILE_K=64 -DCLUSTER_M=2 -DCLUSTER_N=1 \
	  $< -o $@ $(LDFLAGS)

cutlass-256x128x128: cutlass_bench.cu
	$(NVCC) $(CFLAGS) $(CUTLASS_INC) $(CUTLASS_FLAGS) \
	  -DTILE_M=256 -DTILE_N=128 -DTILE_K=128 -DCLUSTER_M=2 -DCLUSTER_N=1 \
	  $< -o $@ $(LDFLAGS)

cutlass-128x128x64: cutlass_bench.cu
	$(NVCC) $(CFLAGS) $(CUTLASS_INC) $(CUTLASS_FLAGS) \
	  -DTILE_M=128 -DTILE_N=128 -DTILE_K=64 -DCLUSTER_M=1 -DCLUSTER_N=2 \
	  $< -o $@ $(LDFLAGS)

# Build all tile configs
cutlass-sweep: cutlass-256x128x64 cutlass-256x256x64 cutlass-256x128x128 cutlass-128x128x64
	@echo "All CUTLASS tile configs built."

# ── SASS dumps ──
# Dumps SASS for default config + custom kernel side-by-side
cutlass-sass: cutlass-bench $(TARGET)
	@mkdir -p sass
	cuobjdump --dump-sass cutlass-bench > sass/cutlass_default.txt
	cuobjdump --dump-sass $(TARGET) > sass/custom_kernel.txt
	@echo "SASS dumped to sass/"
	@echo "--- CUTLASS (cutlass-bench) ---"
	@grep -c 'UTCQMMA' sass/cutlass_default.txt 2>/dev/null && echo " MMA instructions" || echo "  MMA: 0 (check opcode name)"
	@grep -c 'tcgen05' sass/cutlass_default.txt 2>/dev/null && echo " tcgen05 instructions" || true
	@echo "--- Custom kernel ---"
	@grep -c 'UTCQMMA' sass/custom_kernel.txt 2>/dev/null && echo " MMA instructions" || true

# SASS for all tile configs
cutlass-sass-all: cutlass-sweep $(TARGET)
	@mkdir -p sass
	cuobjdump --dump-sass cutlass-256x128x64 > sass/cutlass_256x128x64.txt
	cuobjdump --dump-sass cutlass-256x256x64 > sass/cutlass_256x256x64.txt
	cuobjdump --dump-sass cutlass-256x128x128 > sass/cutlass_256x128x128.txt
	cuobjdump --dump-sass cutlass-128x128x64 > sass/cutlass_128x128x64.txt
	cuobjdump --dump-sass $(TARGET) > sass/custom_kernel.txt
	@echo "All SASS dumped to sass/"

cublas-bench: cublas_bench.cu
	$(NVCC) $(CFLAGS) -std=c++17 $< -o $@ -lcublasLt -lcublas

clean:
	rm -f $(TARGET) siglip_timing siglip_x32 cutlass-bench cublas-bench \
	      cutlass-256x128x64 cutlass-256x256x64 cutlass-256x128x128 cutlass-128x128x64
	rm -rf sass/
