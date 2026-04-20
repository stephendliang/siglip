NVCC     = nvcc
ARCH     = sm_100a
CFLAGS   = -gencode arch=compute_100a,code=$(ARCH) -O3 -std=c++17 -lineinfo --ptxas-options=-v --cudart=static
LDFLAGS  = -lcurand_static -lculibos -lcuda
TARGET   = patch_embed
CU       = patch_embed.cu

CUTLASS_DIR = third_party/cutlass
CUTLASS_INC = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include
CUTLASS_FLAGS = -std=c++17 --expt-relaxed-constexpr

.PHONY: all clean timing fc1-gelu fc1-w3 fc1-w3-gemm fc1-w3-strip fc1-w3-sched fc1-w3-lean fc1-w3-dgswizzle fc1-w3-zorder fc1-w3-hilbert fc1-w3-zigzag fc1-w3-rowmajor fc1-w3-ncycle fc1-w3-ncyrot fc1-w3-nflat fc1-w3-nsnake fc1-w3-packed fc1-w3-packed-sched fc1-w3-packed-lean fc1-w3-packed-dgswizzle fc1-w3-packed-zorder fc1-w3-packed-hilbert fc1-w3-packed-zigzag fc1-w3-packed-rowmajor fc1-w3-packed-ncycle fc1-w3-packed-ncyrot fc1-w3-packed-nflat fc1-w3-packed-nsnake fc1-w3-nlock fc1-w3-checkered fc1-w3-dgsnake fc1-w3-kstagger fc1-w3-dg4 fc1-w3-dg6 fc1-w3-dg10 fc1-w3-dg12 fc1-w3-dg16 fc1-w3-dg24 fc1-w3-dg32 fc2 fc2-timing fc2-w3 fc2-w3-c4 fc2-w3-c4-gemm fc2-w3-c4-strip fc2-w3-clock fc2-w3-sched-clock fc2-w3-collock fc2-w3-collock-clock fc2-w3-8w fc2-w3-fp32 fc2-w3-strip fc2-w3-self fc2-w3-atomic fc2-w3-spin fc2-w3-grid fc2-w3-inline fc2-w3-inline7 fc2-w3-inline7-clock fc2-w3-noprefill fc2-w3-ns7 fc2-w3-clc fc2-w3-rowsteal fc2-w3-tail fc2-w3-tail-lean fc2-w3-nlock fc2-w3-ncyrot fc2-w3-checkered fc2-w3-dgsnake fc2-w3-kstagger fc2-w3-kstagger2 fc2-w3-kstagger3 fc2-w3-dg2 fc2-w3-dg3 fc2-w3-dg10 fc2-w3-dg20 fc2-w3-dg32 fc2-ldg fc2-ldg-strip fc2-ldg-gemm fc2-cutlass fc2-cutlass-strip fc2-cutlass-static fc2-cutlass-static-strip fc2-hybrid fc2-hybrid-strip fc2-hybrid-mma fc2-hybrid-phase3 cutlass-bench cutlass-bench-fc1 cutlass-bench-fc2 cutlass-bench-max cutlass-bench-fc1-max cutlass-bench-fc2-max cutlass-sass calibration cublas-bench cublas-bench-fc1 cublas-bench-fc2 sweep sweep-fast sweep-full sass-tool compare calib-tput calib-lat calib-conflict calib-warp calib-all tma-bench mma-bench stmatrix-bench

all: $(TARGET)

$(TARGET): $(CU) kernel_common.cuh kernel_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) $< -o $@ $(LDFLAGS)

timing: $(CU) kernel_common.cuh kernel_body.cuh
	$(NVCC) $(CFLAGS) -DTIMING $< -o patch_embed_timing $(LDFLAGS)

# ── FC1+GELU kernel ──
fc1-gelu: fc1_gelu.cu kernel_common.cuh kernel_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) $< -o $@ $(LDFLAGS)

# ── FC1 W3 kernel (standalone, GELU epilogue) ──
fc1-w3: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $< -o $@ $(LDFLAGS)

fc1-w3-gemm: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DGEMM_ONLY $< -o $@ $(LDFLAGS)

fc1-w3-strip: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DSTRIP_EPILOGUE $< -o $@ $(LDFLAGS)

fc1-w3-sched: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=4 $< -o $@ $(LDFLAGS)

fc1-w3-lean: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=4 -DLEAN_DISPATCH $< -o $@ $(LDFLAGS)

fc1-w3-dgswizzle: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=8 $< -o $@ $(LDFLAGS)

fc1-w3-zorder: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=9 $< -o $@ $(LDFLAGS)

fc1-w3-hilbert: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=10 $< -o $@ $(LDFLAGS)

fc1-w3-zigzag: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=11 $< -o $@ $(LDFLAGS)

fc1-w3-rowmajor: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=13 $< -o $@ $(LDFLAGS)

fc1-w3-ncycle: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=14 $< -o $@ $(LDFLAGS)

fc1-w3-ncyrot: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=21 $< -o $@ $(LDFLAGS)

fc1-w3-nflat: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=15 $< -o $@ $(LDFLAGS)

fc1-w3-nsnake: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=16 $< -o $@ $(LDFLAGS)

fc1-w3-packed: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DPACKED_TILES $< -o $@ $(LDFLAGS)

fc1-w3-packed-sched: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DPACKED_TILES -DTILE_DISPATCH=4 $< -o $@ $(LDFLAGS)

fc1-w3-packed-lean: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DPACKED_TILES -DTILE_DISPATCH=4 -DLEAN_DISPATCH $< -o $@ $(LDFLAGS)

fc1-w3-packed-dgswizzle: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DPACKED_TILES -DTILE_DISPATCH=8 $< -o $@ $(LDFLAGS)

fc1-w3-packed-zorder: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DPACKED_TILES -DTILE_DISPATCH=9 $< -o $@ $(LDFLAGS)

fc1-w3-packed-hilbert: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DPACKED_TILES -DTILE_DISPATCH=10 $< -o $@ $(LDFLAGS)

fc1-w3-packed-zigzag: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DPACKED_TILES -DTILE_DISPATCH=11 $< -o $@ $(LDFLAGS)

fc1-w3-packed-rowmajor: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DPACKED_TILES -DTILE_DISPATCH=13 $< -o $@ $(LDFLAGS)

fc1-w3-packed-ncycle: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DPACKED_TILES -DTILE_DISPATCH=14 $< -o $@ $(LDFLAGS)

fc1-w3-packed-ncyrot: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DPACKED_TILES -DTILE_DISPATCH=21 $< -o $@ $(LDFLAGS)

fc1-w3-packed-nflat: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DPACKED_TILES -DTILE_DISPATCH=15 $< -o $@ $(LDFLAGS)

fc1-w3-packed-nsnake: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DPACKED_TILES -DTILE_DISPATCH=16 $< -o $@ $(LDFLAGS)

# ── FC2 kernel ──
fc2: fc2.cu kernel_common.cuh kernel_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) $< -o $@ $(LDFLAGS)

fc2-timing: fc2.cu kernel_common.cuh kernel_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTIMING $< -o $@ $(LDFLAGS)

# ── FC2 W3 kernel (standalone, CUTLASS-style shared-SMEM epilogue) ──
fc2-w3: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $< -o $@ $(LDFLAGS)

fc2-w3-clock: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DCLOCK_TIMING $< -o $@ $(LDFLAGS)

fc2-w3-strip-clock: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DSTRIP_EPILOGUE -DCLOCK_TIMING $< -o $@ $(LDFLAGS)

fc2-w3-sched-clock: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=4 -DCLOCK_TIMING $< -o $@ $(LDFLAGS)

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

fc2-w3-packed: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DPACKED_TILES $< -o $@ $(LDFLAGS)

fc2-w3-packed-lean: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DPACKED_TILES -DTILE_DISPATCH=4 -DLEAN_DISPATCH $< -o $@ $(LDFLAGS)

fc2-w3-preswizzle: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DPACKED_TILES -DPRESWIZZLE $< -o $@ $(LDFLAGS)

fc2-w3-preswizzle-lean: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DPACKED_TILES -DPRESWIZZLE -DTILE_DISPATCH=4 -DLEAN_DISPATCH $< -o $@ $(LDFLAGS)

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

fc2-w3-collock: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=4 -DCOL_LOCK $< -o $@ $(LDFLAGS)

fc2-w3-collock-clock: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=4 -DCOL_LOCK -DCLOCK_TIMING $< -o $@ $(LDFLAGS)

fc2-w3-rowsteal: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=4 -DROW_STEAL $< -o $@ $(LDFLAGS)

fc2-w3-inline: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=6 $< -o $@ $(LDFLAGS)

fc2-w3-lean: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=4 -DLEAN_DISPATCH $< -o $@ $(LDFLAGS)

fc2-w3-c4dual: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=4 -DLEAN_DISPATCH -DC4_DUAL_PAIR $< -o $@ $(LDFLAGS)

fc2-w3-c4dual-strip: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=4 -DLEAN_DISPATCH -DC4_DUAL_PAIR -DSTRIP_EPILOGUE $< -o $@ $(LDFLAGS)

fc2-w3-c4dual-gemm: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=4 -DLEAN_DISPATCH -DC4_DUAL_PAIR -DGEMM_ONLY $< -o $@ $(LDFLAGS)

fc2-w3-c4bmc: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=14 -DC4_DUAL_PAIR -DC4_B_MULTICAST $< -o $@ $(LDFLAGS)

fc2-w3-c4bmc-strip: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=14 -DC4_DUAL_PAIR -DC4_B_MULTICAST -DSTRIP_EPILOGUE $< -o $@ $(LDFLAGS)

fc2-w3-c4bmc-gemm: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=14 -DC4_DUAL_PAIR -DC4_B_MULTICAST -DGEMM_ONLY $< -o $@ $(LDFLAGS)

fc2-w3-c4amc: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=20 -DC4_DUAL_PAIR -DC4_A_MULTICAST $< -o $@ $(LDFLAGS)

fc2-w3-c4amc-strip: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=20 -DC4_DUAL_PAIR -DC4_A_MULTICAST -DSTRIP_EPILOGUE $< -o $@ $(LDFLAGS)

fc2-w3-c4amc-gemm: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=20 -DC4_DUAL_PAIR -DC4_A_MULTICAST -DGEMM_ONLY $< -o $@ $(LDFLAGS)

fc2-w3-tail: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=4 -DTAIL_STEAL $< -o $@ $(LDFLAGS)

fc2-w3-tail-lean: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=4 -DTAIL_STEAL -DLEAN_DISPATCH $< -o $@ $(LDFLAGS)

fc2-w3-c4dual2: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DPACKED_TILES -DTILE_DISPATCH=4 -DLEAN_DISPATCH -DC4_DUAL_PAIR $< -o $@ $(LDFLAGS)

fc2-w3-c4dual2-strip: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DPACKED_TILES -DTILE_DISPATCH=4 -DLEAN_DISPATCH -DC4_DUAL_PAIR -DSTRIP_EPILOGUE $< -o $@ $(LDFLAGS)

fc2-w3-lean-clock: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=4 -DLEAN_DISPATCH -DCLOCK_TIMING $< -o $@ $(LDFLAGS)

fc2-w3-inline7: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=7 $< -o $@ $(LDFLAGS)

fc2-w3-dgswizzle: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=8 $< -o $@ $(LDFLAGS)

fc2-w3-zorder: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=9 $< -o $@ $(LDFLAGS)

fc2-w3-hilbert: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=10 $< -o $@ $(LDFLAGS)

fc2-w3-zigzag: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=11 $< -o $@ $(LDFLAGS)

fc2-w3-colfirst: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=12 $< -o $@ $(LDFLAGS)

fc2-w3-rowmajor: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=13 $< -o $@ $(LDFLAGS)

fc2-w3-ncycle: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=14 $< -o $@ $(LDFLAGS)

fc2-w3-ncyrot: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=21 $< -o $@ $(LDFLAGS)

fc2-w3-nflat: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=15 $< -o $@ $(LDFLAGS)

fc2-w3-nsnake: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=16 $< -o $@ $(LDFLAGS)

fc2-w3-dg4: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=8 -DDG_GROUP_SIZE=4  $< -o $@ $(LDFLAGS)
fc2-w3-dg6: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=8 -DDG_GROUP_SIZE=6  $< -o $@ $(LDFLAGS)
fc2-w3-dg12: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=8 -DDG_GROUP_SIZE=12 $< -o $@ $(LDFLAGS)
fc2-w3-dg16: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=8 -DDG_GROUP_SIZE=16 $< -o $@ $(LDFLAGS)
fc2-w3-dg24: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=8 -DDG_GROUP_SIZE=24 $< -o $@ $(LDFLAGS)

# ── Tile-dispatch experiments (2026-04-18) ──
# TD=17 nlock: each cluster bound to one N-column for the whole run.
# TD=18 checkered: G_M × G_N 2D block per group (column-first within).
# TD=19 dg-snake: dgswizzle M-band + zigzag N within band.
# K_STAGGER: shift W0's K-block index by cluster_id*STAGGER (mod K_ITERS).
# All add -DPACKED_TILES via DFLAGS when invoked through bench.sh --packed=yes.
fc2-w3-nlock: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=17 $< -o $@ $(LDFLAGS)
fc2-w3-checkered: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=18 $< -o $@ $(LDFLAGS)
fc2-w3-dgsnake: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=19 $< -o $@ $(LDFLAGS)
fc2-w3-kstagger: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DK_STAGGER=1 $< -o $@ $(LDFLAGS)
fc2-w3-kstagger2: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DK_STAGGER=2 $< -o $@ $(LDFLAGS)
fc2-w3-kstagger3: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DK_STAGGER=3 $< -o $@ $(LDFLAGS)

# ── Extended DG_GROUP_SIZE sweep (TD=8 dgswizzle) ──
fc2-w3-dg2:  fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=8 -DDG_GROUP_SIZE=2  $< -o $@ $(LDFLAGS)
fc2-w3-dg3:  fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=8 -DDG_GROUP_SIZE=3  $< -o $@ $(LDFLAGS)
fc2-w3-dg10: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=8 -DDG_GROUP_SIZE=10 $< -o $@ $(LDFLAGS)
fc2-w3-dg20: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=8 -DDG_GROUP_SIZE=20 $< -o $@ $(LDFLAGS)
fc2-w3-dg32: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=8 -DDG_GROUP_SIZE=32 $< -o $@ $(LDFLAGS)

# ── CK_GROUP_N sweep (TD=18 checkered; G_M fixed at 8) ──
# FC2 TILES_N=3: only ck2 is meaningfully non-dividing; ck>=3 collapses to single stripe.
# FC1 TILES_N=12: divisors are 2,3,4,6,12; ck5/7/8/10/11 are non-dividing.
fc2-w3-ck2: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=18 -DCK_GROUP_N=2 $< -o $@ $(LDFLAGS)

# ── FC1 tile-dispatch experiments ──
fc1-w3-nlock: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=17 $< -o $@ $(LDFLAGS)
fc1-w3-checkered: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=18 $< -o $@ $(LDFLAGS)
fc1-w3-dgsnake: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=19 $< -o $@ $(LDFLAGS)
fc1-w3-kstagger: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DK_STAGGER=1 $< -o $@ $(LDFLAGS)

# ── FC1 DG_GROUP_SIZE sweep ──
fc1-w3-dg4:  fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=8 -DDG_GROUP_SIZE=4  $< -o $@ $(LDFLAGS)
fc1-w3-dg6:  fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=8 -DDG_GROUP_SIZE=6  $< -o $@ $(LDFLAGS)
fc1-w3-dg10: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=8 -DDG_GROUP_SIZE=10 $< -o $@ $(LDFLAGS)
fc1-w3-dg12: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=8 -DDG_GROUP_SIZE=12 $< -o $@ $(LDFLAGS)
fc1-w3-dg16: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=8 -DDG_GROUP_SIZE=16 $< -o $@ $(LDFLAGS)
fc1-w3-dg24: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=8 -DDG_GROUP_SIZE=24 $< -o $@ $(LDFLAGS)
fc1-w3-dg32: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=8 -DDG_GROUP_SIZE=32 $< -o $@ $(LDFLAGS)

# ── FC1 CK_GROUP_N sweep (TD=18 checkered; G_M fixed at 8) ──
# TILES_N=12 divisors: 2,3,4,6,12.  Non-dividing (exercises tail path): 5,7,8,10,11.
fc1-w3-ck2:  fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=18 -DCK_GROUP_N=2  $< -o $@ $(LDFLAGS)
fc1-w3-ck3:  fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=18 -DCK_GROUP_N=3  $< -o $@ $(LDFLAGS)
fc1-w3-ck4:  fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=18 -DCK_GROUP_N=4  $< -o $@ $(LDFLAGS)
fc1-w3-ck5:  fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=18 -DCK_GROUP_N=5  $< -o $@ $(LDFLAGS)
fc1-w3-ck6:  fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=18 -DCK_GROUP_N=6  $< -o $@ $(LDFLAGS)
fc1-w3-ck7:  fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=18 -DCK_GROUP_N=7  $< -o $@ $(LDFLAGS)
fc1-w3-ck8:  fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=18 -DCK_GROUP_N=8  $< -o $@ $(LDFLAGS)
fc1-w3-ck11: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=18 -DCK_GROUP_N=11 $< -o $@ $(LDFLAGS)

fc2-w3-inline7-clock: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=7 -DCLOCK_TIMING $< -o $@ $(LDFLAGS)

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
	$(NVCC) $(CFLAGS) $(DFLAGS) $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

fc2-cutlass-strip: fc2_cutlass.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DSTRIP_EPILOGUE $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

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

# ── cp.async.bulk (1D) vs cp.async.bulk.tensor.2d ──
bulk-vs-tensor: bench/bulk_vs_tensor.cu
	$(NVCC) $(CFLAGS) $< -o $@ -lcuda

# ── 1D multicast + manual mbarrier relay (approach 2) ──
relay-mbar: bench/relay_mbar.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $< -o $@ -lcuda

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

cublas-bench-fc1-ncu: bench/cublas_bench.cu
	$(NVCC) $(CFLAGS) -std=c++17 -DBENCH_N=3072 -DBENCH_EPILOGUE=2 -DNCU_MODE $< -o $@ -lcublasLt -lcublas

cublas-bench-fc2-ncu: bench/cublas_bench.cu
	$(NVCC) $(CFLAGS) -std=c++17 -DBENCH_N=768 -DBENCH_K=3072 -DBENCH_EPILOGUE=3 -DNCU_MODE $< -o $@ -lcublasLt -lcublas

cublaslt-introspect: bench/cublaslt_introspect.cu
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

# ── Unified comparison (cuBLAS vs CUTLASS vs ours, ANOVA) ──
compare:
	python3 tools/compare_all.py --csv data/compare.csv

compare-fast:
	python3 tools/compare_all.py --runs 5 --layer patch_embed --csv data/compare.csv

clean:
	rm -f $(TARGET) patch_embed_timing fc1-gelu fc1-w3 fc1-w3-gemm fc1-w3-strip fc1-w3-sched fc2 fc2-w3 fc2-w3-8w fc2-w3-fp32 fc2-w3-strip fc2-w3-reorder fc2-w3-self fc2-w3-atomic fc2-w3-spin fc2-w3-grid fc2-w3-inline fc2-w3-noprefill fc2-w3-ns7 fc2-w3-clc fc2-w3-rowsteal fc2-w3-tail fc2-w3-tail-lean fc2-ldg fc2-ldg-strip fc2-ldg-gemm fc2-cutlass fc2-cutlass-strip fc2-cutlass-static fc2-cutlass-static-strip fc2-hybrid fc2-hybrid-strip fc2-hybrid-mma fc2-hybrid-phase3 cutlass-bench cutlass-bench-fc1 cutlass-bench-fc2 cutlass-bench-max cutlass-bench-fc1-max cutlass-bench-fc2-max cublas-bench cublas-bench-fc1 cublas-bench-fc2 calibration calib-tput calib-lat calib-conflict calib-warp tma-bench mma-bench stmatrix-bench
	rm -rf sass/
