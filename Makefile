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

.PHONY: all gen clean dry-run timing tmem-x32 tmem-x64 cutlass-sass

all: $(TARGET)

$(TARGET): $(CU)
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

timing: $(CU)
	$(NVCC) $(CFLAGS) -DTIMING $< -o siglip_timing $(LDFLAGS)

# F37: wider TMEM loads (SASS analysis — compile with -DTMEM_LOAD_WIDTH=32 or 64)
tmem-x32: $(CU)
	$(NVCC) $(CFLAGS) -DTMEM_LOAD_WIDTH=32 $< -o siglip_x32 $(LDFLAGS)

tmem-x64: $(CU)
	$(NVCC) $(CFLAGS) -DTMEM_LOAD_WIDTH=64 $< -o siglip_x64 $(LDFLAGS)

# ── CUTLASS benchmark (per-tensor FP8, grid search) ──
cutlass-bench: cutlass_bench.cu
	$(NVCC) $(CFLAGS) $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

# ── SASS dump ──
cutlass-sass: cutlass-bench
	@mkdir -p sass
	cuobjdump --dump-sass cutlass-bench > sass/cutlass.txt
	@echo "SASS dumped to sass/cutlass.txt"

cublas-bench: cublas_bench.cu
	$(NVCC) $(CFLAGS) -std=c++17 $< -o $@ -lcublasLt -lcublas

clean:
	rm -f $(TARGET) siglip_timing siglip_x32 siglip_x64 cutlass-bench cublas-bench
	rm -rf sass/
