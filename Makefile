NVCC     = nvcc
ARCH     = sm_100a
CFLAGS   = -gencode arch=compute_100a,code=$(ARCH) -O3 -std=c++17 -lineinfo --ptxas-options=-v --cudart=static
LDFLAGS  = -lcurand_static -lculibos -lcuda
TARGET   = patch_embed
CU       = patch_embed.cu

CUTLASS_DIR = third_party/cutlass
CUTLASS_INC = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include
CUTLASS_FLAGS = -std=c++17 --expt-relaxed-constexpr

# Phony only — every other target produces a binary with its own name.
.PHONY: all clean compare compare-fast sweep sweep-fast sweep-full sass-tool calib-all \
        fc2-w3x-dg-sweep fc2-w3x-tile-sweep fc2-w3x-sync-sweep fc2-w3x-r-sweep \
        fc1-w3x-tile-sweep fc1-w3x-ks-sweep fc1-w3x-r-sweep

all: $(TARGET)

$(TARGET): $(CU) kernel_common.cuh kernel_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) $< -o $@ $(LDFLAGS)

# ── FC1 W3 kernel (legacy; superseded by fc1_w3x for non-residual) ──
fc1-w3: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $< -o $@ $(LDFLAGS)

fc1-w3-gemm: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DGEMM_ONLY $< -o $@ $(LDFLAGS)

fc1-w3-strip: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DSTRIP_EPILOGUE $< -o $@ $(LDFLAGS)

fc1-w3-zigzag: fc1_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=11 $< -o $@ $(LDFLAGS)

# ── Generated bias-load switch chain (avoids local-memory spill on
# lane_bias[reg_idx_sub]). fc2_w3x N=768 → BIAS_REG_COUNT=12;
# fc1_w3x N=3072 → BIAS_REG_COUNT=48. Both .cu files include the
# corresponding header inside the kernel's per-subpass bias-fetch site. ──
gen/bias_switch_inc_%.cuh: tools/gen_bias_switch.py
	@mkdir -p gen
	python3 $< $* -o $@

# ── FC2 W3X kernel (6-warp bias-only rank-1-shaped, persistent, PACKED+dgswizzle only) ──
fc2-w3x: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) $< -o $@ $(LDFLAGS)

fc2-w3x-noprefill: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DNO_PREFILL $< -o $@ $(LDFLAGS)

fc2-w3x-strip: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DSTRIP_EPILOGUE $< -o $@ $(LDFLAGS)

fc2-w3x-gemm: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DGEMM_ONLY $< -o $@ $(LDFLAGS)

fc2-w3x-ncu: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DNCU_PROFILE $< -o $@ $(LDFLAGS)

fc2-w3x-ncu-strip: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DNCU_PROFILE -DSTRIP_EPILOGUE $< -o $@ $(LDFLAGS)

fc2-w3x-ncu-gemm: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DNCU_PROFILE -DGEMM_ONLY $< -o $@ $(LDFLAGS)

# ── fc2_w3x PTX port — hand-authored .ptx + driver-API host harness ──
# fc2_w3x.ptx is the hand-authored PTX deliverable (see docs/PTX_BUILD_NOTES.md
# for design notes).  Build pipeline:
#   1. nvcc compiles .ptx → .cubin
#   2. ld -r -b binary embeds .cubin bytes into a .o (symbols _binary_*_start/end/size)
#   3. nvcc links .cc host + embedded-cubin .o + CUDA driver lib
fc2_w3x.cubin: fc2_w3x.ptx
	$(NVCC) -arch=compute_100a -code=sm_100a -cubin $< -o $@

fc2_w3x_cubin.o: fc2_w3x.cubin
	ld -r -b binary $< -o $@

fc2_w3x_host.o: fc2_w3x_host.cc
	$(NVCC) $(CFLAGS) $(DFLAGS) -x c++ -c $< -o $@

fc2-w3x-ptx: fc2_w3x_host.o fc2_w3x_cubin.o
	$(NVCC) $(CFLAGS) $(DFLAGS) fc2_w3x_host.o fc2_w3x_cubin.o -o $@ $(LDFLAGS)

# ── fc2_w3x dispatch-variant sweep (CPU-design, B200-measured) ────────────
# All bijective on FC2 shape (TM=3626, TN=3, NC=74); verified via
# /tmp/dg_variants_check.py.  Hypothesis-motivated by in_g-structural
# tn=0 surplus (project_w3x_tn0_in_g_structural.md) + W4 53%-idle
# (W5-bound, PROFILE_W4 retired).  Expected: all likely 0-delta given
# prior dgphase / DG_ROT / dgnrot 0-deltas — but cheap to measure and rule out.
fc2-w3x-dg4: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DDG_GROUP_SIZE=4 $< -o $@ $(LDFLAGS)

fc2-w3x-dg16: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DDG_GROUP_SIZE=16 $< -o $@ $(LDFLAGS)

fc2-w3x-dg32: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DDG_GROUP_SIZE=32 $< -o $@ $(LDFLAGS)

fc2-w3x-dg-sweep: fc2-w3x fc2-w3x-dg4 fc2-w3x-dg16 fc2-w3x-dg32

# ── fc2_w3x structurally-different tile swizzles (tile_dispatch.cuh TD=9..21) ──
# Not dgswizzle variants — genuinely different traversal structures.  TILES_N=3
# makes most 2D space-filling curves collapse or degrade; included for empirical
# measurement.  nlock (TD=17) exposes column-locked dispatch (cluster bound to
# one tn, sweeps M).  checkered (TD=18) is a 2D M×N block.  dg-snake (TD=19) is
# zigzag-within-dgswizzle-band.
fc2-w3x-tile-zorder:    fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=9  $< -o $@ $(LDFLAGS)
fc2-w3x-tile-hilbert:   fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=10 $< -o $@ $(LDFLAGS)
fc2-w3x-tile-zigzag:    fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=11 $< -o $@ $(LDFLAGS)
fc2-w3x-tile-rowmajor:  fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=13 $< -o $@ $(LDFLAGS)
fc2-w3x-tile-ncycle:    fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=14 $< -o $@ $(LDFLAGS)
fc2-w3x-tile-nflat:     fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=15 $< -o $@ $(LDFLAGS)
fc2-w3x-tile-nsnake:    fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=16 $< -o $@ $(LDFLAGS)
fc2-w3x-tile-nlock:     fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=17 $< -o $@ $(LDFLAGS)
fc2-w3x-tile-checkered: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=18 $< -o $@ $(LDFLAGS)
fc2-w3x-tile-dgsnake:   fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=19 $< -o $@ $(LDFLAGS)
fc2-w3x-tile-ncyrot:    fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=21 $< -o $@ $(LDFLAGS)
fc2-w3x-tile-chet:      fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=30 $< -o $@ $(LDFLAGS)
fc2-w3x-tile-pmix:      fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=31 $< -o $@ $(LDFLAGS)
fc2-w3x-tile-ingh:      fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=32 $< -o $@ $(LDFLAGS)
fc2-w3x-tile-gflip:     fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=33 $< -o $@ $(LDFLAGS)
fc2-w3x-tile-tn2br:     fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=34 $< -o $@ $(LDFLAGS)

fc2-w3x-tile-sweep: fc2-w3x \
    fc2-w3x-tile-zorder fc2-w3x-tile-hilbert fc2-w3x-tile-zigzag \
    fc2-w3x-tile-rowmajor fc2-w3x-tile-ncycle fc2-w3x-tile-nflat \
    fc2-w3x-tile-nsnake fc2-w3x-tile-nlock fc2-w3x-tile-checkered \
    fc2-w3x-tile-dgsnake fc2-w3x-tile-ncyrot \
    fc2-w3x-tile-chet fc2-w3x-tile-pmix fc2-w3x-tile-ingh \
    fc2-w3x-tile-gflip fc2-w3x-tile-tn2br

# Sync/fence experiments (DROP_TRAIL_BARSYNC × WAIT_GROUP_READ).  Driver:
# tools/sweep_sync_experiments.sh.  Each binary is the baseline dgswizzle
# kernel + the macro guards from fc2_w3x.cu's top-of-file comment.
fc2-w3x-base: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) $< -o $@ $(LDFLAGS)

fc2-w3x-trail: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DDROP_TRAIL_BARSYNC $< -o $@ $(LDFLAGS)

fc2-w3x-wgread: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DWAIT_GROUP_READ $< -o $@ $(LDFLAGS)

fc2-w3x-wgread-trail: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DWAIT_GROUP_READ -DDROP_TRAIL_BARSYNC $< -o $@ $(LDFLAGS)

fc2-w3x-sync-sweep: fc2-w3x-base fc2-w3x-trail fc2-w3x-wgread fc2-w3x-wgread-trail

# W5 critical-path profiler.  Combine with PROFILE_TILE if you want the
# tma_wait_sum breakdown alongside W5's tile_total / mma_asm / commit.
fc2-w3x-prof-w5: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DPROFILE_W5 $< -o $@ $(LDFLAGS)

fc2-w3x-prof-w5-full: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DPROFILE_W5 -DPROFILE_TILE $< -o $@ $(LDFLAGS)

# Warpgroup-asymmetric regs: default LO=48 in fc2_w3x.cu, HI set via target name.
# Pattern rule: fc2-w3x-r<HI> sweeps epilogue-warpgroup reg target (8-aligned,
# valid range 24..256).  LO override via DFLAGS='-DSETMAXNREG_LO=N'.
# Pool ceiling at 192 threads: HI*128 + LO*64 <= 65536 -> HI <= (65536-LO*64)/128
# (with LO=48 -> HI<=488; never binds at sane values).
fc2-w3x-regs: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DUSE_SETMAXNREG --maxrregcount=120 $< -o $@ $(LDFLAGS)

fc2-w3x-regs-strip: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DUSE_SETMAXNREG -DSTRIP_EPILOGUE --maxrregcount=120 $< -o $@ $(LDFLAGS)

fc2-w3x-regs-gemm: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DUSE_SETMAXNREG -DGEMM_ONLY --maxrregcount=120 $< -o $@ $(LDFLAGS)

# fc2-w3x-r<HI>: build w/ USE_SETMAXNREG and SETMAXNREG_HI=<HI>.
# Example: make -B fc2-w3x-r192  ->  HI=192, LO=default(48).
# Override LO:  make -B fc2-w3x-r192 DFLAGS='-DSETMAXNREG_LO=40'
fc2-w3x-r%: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DUSE_SETMAXNREG -DSETMAXNREG_HI=$* --maxrregcount=120 $< -o $@ $(LDFLAGS)

fc2-w3x-r%-strip: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DUSE_SETMAXNREG -DSETMAXNREG_HI=$* -DSTRIP_EPILOGUE --maxrregcount=120 $< -o $@ $(LDFLAGS)

fc2-w3x-r%-gemm: fc2_w3x.cu gen/bias_switch_inc_12.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DUSE_SETMAXNREG -DSETMAXNREG_HI=$* -DGEMM_ONLY --maxrregcount=120 $< -o $@ $(LDFLAGS)

# Full HI sweep (24..240 in 8-reg steps covers the interesting range).
# Known: HI=168 -> 88B spills, HI=192 -> 32B spills (sweet), HI=216 -> 88B spills.
fc2-w3x-r-sweep: fc2-w3x-r120 fc2-w3x-r136 fc2-w3x-r152 fc2-w3x-r160 fc2-w3x-r168 fc2-w3x-r176 fc2-w3x-r184 fc2-w3x-r192 fc2-w3x-r200 fc2-w3x-r208 fc2-w3x-r216 fc2-w3x-r224 fc2-w3x-r232 fc2-w3x-r240
# ── FC1 W3X kernel (clean-sheet port of fc2_w3x to FC1 GELU+BIAS) ──
# 6-warp rank-1-shaped persistent (W0-W3 epi / W4 TMA / W5 MMA-CTA0-only),
# register-cached bias preload (BIAS_REG_COUNT=48 for N=3072), 4D packed-tile
# output, K_STAGGER default=1, NO_PREFILL auto-fires at K_ITERS=6, N_STAGES=5.
fc1-w3x: fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) $< -o $@ $(LDFLAGS)

fc1-w3x-strip: fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DSTRIP_EPILOGUE $< -o $@ $(LDFLAGS)

fc1-w3x-gemm: fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DGEMM_ONLY $< -o $@ $(LDFLAGS)

fc1-w3x-noprefill: fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DNO_PREFILL $< -o $@ $(LDFLAGS)

fc1-w3x-ncu: fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DNCU_PROFILE $< -o $@ $(LDFLAGS)

fc1-w3x-prof-w5: fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DPROFILE_W5 $< -o $@ $(LDFLAGS)

fc1-w3x-prof-w5-full: fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DPROFILE_W5 -DPROFILE_TILE $< -o $@ $(LDFLAGS)

# K_STAGGER sweep (FC1 production tuning lever; odd values decorrelate).
fc1-w3x-ks0: fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DK_STAGGER=0 $< -o $@ $(LDFLAGS)
fc1-w3x-ks2: fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DK_STAGGER=2 $< -o $@ $(LDFLAGS)
fc1-w3x-ks3: fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DK_STAGGER=3 $< -o $@ $(LDFLAGS)
fc1-w3x-ks5: fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DK_STAGGER=5 $< -o $@ $(LDFLAGS)

fc1-w3x-ks-sweep: fc1-w3x-ks0 fc1-w3x fc1-w3x-ks2 fc1-w3x-ks3 fc1-w3x-ks5

# Tile-dispatch variants (mirror fc2_w3x-tile-*; same TD ids → same swizzles).
# Re-tune at FC1 dims; prior FC1 winner (fc1_w3) is zigzag (TD=11) + ks=1, but
# fc1_w3x's basin may differ. gflip_blkswap (TD=54) is fc2_w3x's default.
fc1-w3x-tile-zorder:    fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=9  $< -o $@ $(LDFLAGS)
fc1-w3x-tile-hilbert:   fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=10 $< -o $@ $(LDFLAGS)
fc1-w3x-tile-zigzag:    fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=11 $< -o $@ $(LDFLAGS)
fc1-w3x-tile-rowmajor:  fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=13 $< -o $@ $(LDFLAGS)
fc1-w3x-tile-ncycle:    fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=14 $< -o $@ $(LDFLAGS)
fc1-w3x-tile-nflat:     fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=15 $< -o $@ $(LDFLAGS)
fc1-w3x-tile-nsnake:    fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=16 $< -o $@ $(LDFLAGS)
fc1-w3x-tile-nlock:     fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=17 $< -o $@ $(LDFLAGS)
fc1-w3x-tile-checkered: fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=18 $< -o $@ $(LDFLAGS)
fc1-w3x-tile-dgsnake:   fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=19 $< -o $@ $(LDFLAGS)
fc1-w3x-tile-ncyrot:    fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=21 $< -o $@ $(LDFLAGS)
fc1-w3x-tile-chet:      fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=30 $< -o $@ $(LDFLAGS)
fc1-w3x-tile-pmix:      fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=31 $< -o $@ $(LDFLAGS)
fc1-w3x-tile-ingh:      fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=32 $< -o $@ $(LDFLAGS)
fc1-w3x-tile-gflip:     fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=33 $< -o $@ $(LDFLAGS)
fc1-w3x-tile-blkswap:   fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=54 $< -o $@ $(LDFLAGS)

fc1-w3x-tile-sweep: fc1-w3x \
    fc1-w3x-tile-zorder fc1-w3x-tile-hilbert fc1-w3x-tile-zigzag \
    fc1-w3x-tile-rowmajor fc1-w3x-tile-ncycle fc1-w3x-tile-nflat \
    fc1-w3x-tile-nsnake fc1-w3x-tile-nlock fc1-w3x-tile-checkered \
    fc1-w3x-tile-dgsnake fc1-w3x-tile-ncyrot \
    fc1-w3x-tile-chet fc1-w3x-tile-pmix fc1-w3x-tile-ingh \
    fc1-w3x-tile-gflip fc1-w3x-tile-blkswap

fc1-w3x-r%: fc1_w3x.cu gen/bias_switch_inc_48.cuh gemm_w3x_body.cuh
	$(NVCC) $(CFLAGS) $(DFLAGS) -DUSE_SETMAXNREG -DSETMAXNREG_HI=$* --maxrregcount=120 $< -o $@ $(LDFLAGS)

fc1-w3x-r-sweep: fc1-w3x-r120 fc1-w3x-r136 fc1-w3x-r152 fc1-w3x-r160 fc1-w3x-r168 fc1-w3x-r176 fc1-w3x-r184 fc1-w3x-r192 fc1-w3x-r200 fc1-w3x-r208 fc1-w3x-r216 fc1-w3x-r224 fc1-w3x-r232 fc1-w3x-r240

# ── FC2 W3 kernel (legacy, retained for residual path; reference dispatch family) ──
fc2-w3: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $< -o $@ $(LDFLAGS)

# Single-binary swizzle basin sweep (ported gflip basin + dgsw/zigzag/stride
# baselines). Run: ./fc2-w3-swizzle-sweep SWEEP=front REPS=200  (or env vars).
# Reports frequency-invariant cyc via CLOCK_TOTAL — fc2_w3's HBM-noise-immune
# metric. Drive with tools/sweep_fc2_w3_swizzle.sh, or:
# modal run dummy_modal.py --target fc2-w3-swizzle-sweep --run-args "SWEEP=front REPS=200"
fc2-w3-swizzle-sweep: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DCOMBO_QUICK $< -o $@ $(LDFLAGS)

fc2-w3-strip: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DSTRIP_EPILOGUE $< -o $@ $(LDFLAGS)

fc2-w3-gemm: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DGEMM_ONLY $< -o $@ $(LDFLAGS)

fc2-w3-bias: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DBIAS_ONLY $< -o $@ $(LDFLAGS)

fc2-w3-noprefill: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DNO_PREFILL $< -o $@ $(LDFLAGS)

fc2-w3-dgswizzle: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=8 $< -o $@ $(LDFLAGS)

fc2-w3-zigzag: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=11 $< -o $@ $(LDFLAGS)

fc2-w3-rowmajor: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=13 $< -o $@ $(LDFLAGS)

fc2-w3-ncycle: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=14 $< -o $@ $(LDFLAGS)

fc2-w3-nflat: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=15 $< -o $@ $(LDFLAGS)

fc2-w3-nsnake: fc2_w3.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DTILE_DISPATCH=16 $< -o $@ $(LDFLAGS)

# ── FC2 CUTLASS kernel (CUTLASS GemmUniversal, reference epilogue) ──
fc2-cutlass: fc2_cutlass.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

fc2-cutlass-strip: fc2_cutlass.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) -DSTRIP_EPILOGUE $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

fc2-cutlass-static: fc2_cutlass.cu
	$(NVCC) $(CFLAGS) -DSTATIC_SCHED $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)

fc2-cutlass-static-strip: fc2_cutlass.cu
	$(NVCC) $(CFLAGS) -DSTATIC_SCHED -DSTRIP_EPILOGUE $(CUTLASS_INC) $(CUTLASS_FLAGS) $< -o $@ $(LDFLAGS)
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

# ── TMA swizzle probe: validates CU_TENSOR_MAP_SWIZZLE_64B STS-XOR pattern ──
# Run tma-swizzle-probe first (must PASS — SWIZZLE_NONE sanity).
# Then tma-swizzle-probe-sw. If PASS, the (row & 3) << 4 XOR is correct; if
# FAIL, try -sw-x2, -sw-x3, -sw-x4 and read the chunk-permutation map to
# decide the right formula before porting to fc2_w3x.cu.
tma-swizzle-probe: bench/tma_swizzle_probe.cu
	$(NVCC) $(CFLAGS) -DXOR_MODE=0 $< -o $@ -lcuda

tma-swizzle-probe-sw: bench/tma_swizzle_probe.cu
	$(NVCC) $(CFLAGS) -DXOR_MODE=1 $< -o $@ -lcuda

tma-swizzle-probe-sw-x2: bench/tma_swizzle_probe.cu
	$(NVCC) $(CFLAGS) -DXOR_MODE=2 $< -o $@ -lcuda

tma-swizzle-probe-sw-x3: bench/tma_swizzle_probe.cu
	$(NVCC) $(CFLAGS) -DXOR_MODE=3 $< -o $@ -lcuda

tma-swizzle-probe-sw-x4: bench/tma_swizzle_probe.cu
	$(NVCC) $(CFLAGS) -DXOR_MODE=4 $< -o $@ -lcuda

cublas-bench: bench/cublas_bench.cu bench/fc_problem.cuh
	$(NVCC) $(CFLAGS) -std=c++17 $< -o $@ -lcublasLt -lcublas

cublas-bench-fc1: bench/cublas_bench.cu bench/fc_problem.cuh
	$(NVCC) $(CFLAGS) -std=c++17 -DBENCH_N=3072 -DBENCH_EPILOGUE=2 $< -o $@ -lcublasLt -lcublas

cublas-bench-fc2: bench/cublas_bench.cu bench/fc_problem.cuh
	$(NVCC) $(CFLAGS) -std=c++17 -DBENCH_N=768 -DBENCH_K=3072 -DBENCH_EPILOGUE=3 $< -o $@ -lcublasLt -lcublas

cublas-bench-fc1-ncu: bench/cublas_bench.cu bench/fc_problem.cuh
	$(NVCC) $(CFLAGS) -std=c++17 -DBENCH_N=3072 -DBENCH_EPILOGUE=2 -DNCU_MODE $< -o $@ -lcublasLt -lcublas

cublas-bench-fc2-ncu: bench/cublas_bench.cu bench/fc_problem.cuh
	$(NVCC) $(CFLAGS) -std=c++17 -DBENCH_N=768 -DBENCH_K=3072 -DBENCH_EPILOGUE=3 -DNCU_MODE $< -o $@ -lcublasLt -lcublas

cublaslt-introspect: bench/cublaslt_introspect.cu bench/fc_problem.cuh
	$(NVCC) $(CFLAGS) -std=c++17 $< -o $@ -lcublasLt -lcublas

cublaslt-fc1: bench/cublaslt_introspect.cu bench/fc_problem.cuh
	$(NVCC) $(CFLAGS) -std=c++17 -DDEFAULT_M=928256 -DDEFAULT_N=3072 -DDEFAULT_K=768 -DDEFAULT_EPI=2 $< -o $@ -lcublasLt -lcublas

cublaslt-fc2: bench/cublaslt_introspect.cu bench/fc_problem.cuh
	$(NVCC) $(CFLAGS) -std=c++17 -DDEFAULT_M=928256 -DDEFAULT_N=768 -DDEFAULT_K=3072 -DDEFAULT_EPI=3 $< -o $@ -lcublasLt -lcublas

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
	rm -f $(TARGET) fc1-w3 fc1-w3-* fc2-w3 fc2-w3-* fc2-w3x fc2-w3x-* fc1-w3x fc1-w3x-* \
	      fc2-cutlass fc2-cutlass-* cutlass-bench cutlass-bench-* cublas-bench cublas-bench-* \
	      cublaslt-* tma-bench mma-bench stmatrix-bench tma-swizzle-probe* \
	      bulk-vs-tensor relay-mbar calibration calib-tput calib-lat calib-conflict calib-warp \
	      fc2_w3x.cubin fc2_w3x_cubin.o fc2_w3x_host.o
	rm -rf gen/ sass/
