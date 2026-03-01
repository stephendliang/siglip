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

.PHONY: all gen clean dry-run timing tmem-x16 tmem-x64 cutlass-sass \
       f40-v1 f40-v2 f40-v3 f40-v4 f40-v5 f40-v6 f40-v7 f40-v8 f40-v9 f40-v10

all: $(TARGET)

$(TARGET): $(CU)
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

timing: $(CU)
	$(NVCC) $(CFLAGS) -DTIMING $< -o siglip_timing $(LDFLAGS)

# F37: alternate TMEM load widths (default is x32; compile with -DTMEM_LOAD_WIDTH=16 or 64)
tmem-x16: $(CU)
	$(NVCC) $(CFLAGS) -DTMEM_LOAD_WIDTH=16 $< -o siglip_x16 $(LDFLAGS)

tmem-x64: $(CU)
	$(NVCC) $(CFLAGS) -DTMEM_LOAD_WIDTH=64 $< -o siglip_x64 $(LDFLAGS)

# ── F40: Interleaved TMA stores grid search ──
# V1: x32, per-region, mbar_early
f40-v1: $(CU)
	$(NVCC) $(CFLAGS) -DINTERLEAVE_STRATEGY=1 -DMBAR_EARLY=1 $< -o $@ $(LDFLAGS)

# V2: x32, per-region, mbar_late
f40-v2: $(CU)
	$(NVCC) $(CFLAGS) -DINTERLEAVE_STRATEGY=1 $< -o $@ $(LDFLAGS)

# V3: x64, per-region, mbar_early
f40-v3: $(CU)
	$(NVCC) $(CFLAGS) -DTMEM_LOAD_WIDTH=64 -DINTERLEAVE_STRATEGY=1 -DMBAR_EARLY=1 $< -o $@ $(LDFLAGS)

# V4: x64, per-region, mbar_late
f40-v4: $(CU)
	$(NVCC) $(CFLAGS) -DTMEM_LOAD_WIDTH=64 -DINTERLEAVE_STRATEGY=1 $< -o $@ $(LDFLAGS)

# V5: x32, half-batch, mbar_early
f40-v5: $(CU)
	$(NVCC) $(CFLAGS) -DINTERLEAVE_STRATEGY=2 -DMBAR_EARLY=1 $< -o $@ $(LDFLAGS)

# V6: x32, half-batch, mbar_late
f40-v6: $(CU)
	$(NVCC) $(CFLAGS) -DINTERLEAVE_STRATEGY=2 $< -o $@ $(LDFLAGS)

# V7: x64, half-batch, mbar_late
f40-v7: $(CU)
	$(NVCC) $(CFLAGS) -DTMEM_LOAD_WIDTH=64 -DINTERLEAVE_STRATEGY=2 $< -o $@ $(LDFLAGS)

# V8: x32, all-at-end (F38 baseline)
f40-v8: $(CU)
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# V9: x64, all-at-end (F38 baseline)
f40-v9: $(CU)
	$(NVCC) $(CFLAGS) -DTMEM_LOAD_WIDTH=64 $< -o $@ $(LDFLAGS)

# V10: x32, three-plus-one, mbar_late
f40-v10: $(CU)
	$(NVCC) $(CFLAGS) -DINTERLEAVE_STRATEGY=3 $< -o $@ $(LDFLAGS)

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

cublas-bench: cublas_bench.cu
	$(NVCC) $(CFLAGS) -std=c++17 $< -o $@ -lcublasLt -lcublas

clean:
	rm -f $(TARGET) siglip_timing siglip_x16 siglip_x64 cutlass-bench cutlass-bench-max cublas-bench
	rm -f f40-v1 f40-v2 f40-v3 f40-v4 f40-v5 f40-v6 f40-v7 f40-v8 f40-v9 f40-v10
	rm -rf sass/
