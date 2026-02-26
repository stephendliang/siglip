NVCC     = nvcc
ARCH     = sm_100a
CFLAGS   = -gencode arch=compute_100a,code=$(ARCH) -O3 --ptxas-options=-v
LDFLAGS  = -lcurand -lcuda
TARGET   = siglip_vision
GEN      = gen.py
CU       = megakernel.cu

CUTLASS_DIR = third_party/cutlass
CUTLASS_INC = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include

.PHONY: all gen clean dry-run

all: $(TARGET)

$(TARGET): $(CU)
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

cutlass-bench: cutlass_bench.cu
	$(NVCC) $(CFLAGS) $(CUTLASS_INC) -std=c++17 --expt-relaxed-constexpr $< -o $@ $(LDFLAGS)

cublas-bench: cublas_bench.cu
	$(NVCC) $(CFLAGS) -std=c++17 $< -o $@ -lcublasLt -lcublas

clean:
	rm -f $(TARGET) cutlass-bench cublas-bench
