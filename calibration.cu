// SASS calibration microbenchmarks for SM 100a (Blackwell)
//
// Measures per-instruction throughput and latency for the instruction types
// that dominate the megakernel epilogue, plus pipe conflict tests and
// control word decoder verification.
//
// Build:  nvcc -arch=sm_100a -O3 calibration.cu -o calibration
// Run:    ./calibration
// SASS:   cuobjdump --dump-sass calibration > calibration_sass.txt
//         python3 sass_analysis.py calibration_sass.txt
//
// All measurements: 1 block, 1 warp (32 threads), clock64() timing.
// Each kernel repeats the measurement block REPS times for stability.

#include <cstdio>
#include <cstdint>
#include <cuda_bf16.h>

#define REPS 1024
#define WARMUP 64

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: F2FP throughput (independent streams)
// 16 independent cvt.rn.bf16x2.f32 — no RAW deps between them.
// Throughput = total_cycles / (REPS * 16)
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void k1_f2fp_throughput(long long* out) {
    float a0=1.f, a1=2.f, a2=3.f, a3=4.f, a4=5.f, a5=6.f, a6=7.f, a7=8.f;
    float b0=9.f, b1=10.f, b2=11.f, b3=12.f, b4=13.f, b5=14.f, b6=15.f, b7=16.f;
    float c0=17.f, c1=18.f, c2=19.f, c3=20.f, c4=21.f, c5=22.f, c6=23.f, c7=24.f;
    float d0=25.f, d1=26.f, d2=27.f, d3=28.f, d4=29.f, d5=30.f, d6=31.f, d7=32.f;
    unsigned r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;

    // warmup
    for (int i = 0; i < WARMUP; i++) {
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r0) : "f"(a0), "f"(b0));
    }

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "cvt.rn.bf16x2.f32 %0,  %16, %17;\n\t"
            "cvt.rn.bf16x2.f32 %1,  %18, %19;\n\t"
            "cvt.rn.bf16x2.f32 %2,  %20, %21;\n\t"
            "cvt.rn.bf16x2.f32 %3,  %22, %23;\n\t"
            "cvt.rn.bf16x2.f32 %4,  %24, %25;\n\t"
            "cvt.rn.bf16x2.f32 %5,  %26, %27;\n\t"
            "cvt.rn.bf16x2.f32 %6,  %28, %29;\n\t"
            "cvt.rn.bf16x2.f32 %7,  %30, %31;\n\t"
            "cvt.rn.bf16x2.f32 %8,  %32, %33;\n\t"
            "cvt.rn.bf16x2.f32 %9,  %34, %35;\n\t"
            "cvt.rn.bf16x2.f32 %10, %36, %37;\n\t"
            "cvt.rn.bf16x2.f32 %11, %38, %39;\n\t"
            "cvt.rn.bf16x2.f32 %12, %40, %41;\n\t"
            "cvt.rn.bf16x2.f32 %13, %42, %43;\n\t"
            "cvt.rn.bf16x2.f32 %14, %44, %45;\n\t"
            "cvt.rn.bf16x2.f32 %15, %46, %47;\n\t"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3),
              "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7),
              "=r"(r8), "=r"(r9), "=r"(r10), "=r"(r11),
              "=r"(r12), "=r"(r13), "=r"(r14), "=r"(r15)
            : "f"(a0), "f"(b0), "f"(a1), "f"(b1),
              "f"(a2), "f"(b2), "f"(a3), "f"(b3),
              "f"(a4), "f"(b4), "f"(a5), "f"(b5),
              "f"(a6), "f"(b6), "f"(a7), "f"(b7),
              "f"(c0), "f"(c1), "f"(c2), "f"(c3),
              "f"(c4), "f"(c5), "f"(c6), "f"(c7),
              "f"(d0), "f"(d1), "f"(d2), "f"(d3),
              "f"(d4), "f"(d5), "f"(d6), "f"(d7)
        );
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        // prevent DCE
        out[1] = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 +
                 r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: F2FP latency (dependent chain)
// Each cvt feeds its result back as input via mov.
// Latency = total_cycles / (REPS * 16)
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void k2_f2fp_latency(long long* out) {
    float a = 1.0f;
    unsigned r;

    for (int i = 0; i < WARMUP; i++) {
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(r) : "f"(a));
        asm volatile("mov.b32 %0, %1;" : "=f"(a) : "r"(r));  // bitcast u32→f32, creates RAW dep
    }

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        // 8 iterations of cvt→mov(bitcast) dependency chain
        // Each pair: cvt reads a (float), produces r (uint); mov feeds r back as a
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(r) : "f"(a));
        asm volatile("mov.b32 %0, %1;" : "=f"(a) : "r"(r));
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(r) : "f"(a));
        asm volatile("mov.b32 %0, %1;" : "=f"(a) : "r"(r));
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(r) : "f"(a));
        asm volatile("mov.b32 %0, %1;" : "=f"(a) : "r"(r));
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(r) : "f"(a));
        asm volatile("mov.b32 %0, %1;" : "=f"(a) : "r"(r));
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(r) : "f"(a));
        asm volatile("mov.b32 %0, %1;" : "=f"(a) : "r"(r));
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(r) : "f"(a));
        asm volatile("mov.b32 %0, %1;" : "=f"(a) : "r"(r));
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(r) : "f"(a));
        asm volatile("mov.b32 %0, %1;" : "=f"(a) : "r"(r));
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(r) : "f"(a));
        asm volatile("mov.b32 %0, %1;" : "=f"(a) : "r"(r));
        // 8 cvt+mov pairs = 16 dependent instructions per iteration
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = r;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: F2FP + STS interleaved (pipe conflict test)
// Alternates cvt and st.shared. If same pipe: serializes. If different: overlaps.
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void k3_f2fp_sts_conflict(long long* out) {
    __shared__ int smem[1024];
    float a0=1.f, b0=2.f, a1=3.f, b1=4.f;
    unsigned r0, r1, r2, r3;
    int tid = threadIdx.x;
    uint32_t saddr = (uint32_t)(uint64_t)(&smem[tid * 4]);

    for (int i = 0; i < WARMUP; i++) {
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r0) : "f"(a0), "f"(b0));
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(saddr), "r"(r0));
    }

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "cvt.rn.bf16x2.f32 %0, %4, %5;\n\t"
            "st.shared.b32 [%8], %0;\n\t"
            "cvt.rn.bf16x2.f32 %1, %6, %7;\n\t"
            "st.shared.b32 [%9], %1;\n\t"
            "cvt.rn.bf16x2.f32 %2, %4, %5;\n\t"
            "st.shared.b32 [%10], %2;\n\t"
            "cvt.rn.bf16x2.f32 %3, %6, %7;\n\t"
            "st.shared.b32 [%11], %3;\n\t"
            "cvt.rn.bf16x2.f32 %0, %4, %5;\n\t"
            "st.shared.b32 [%8], %0;\n\t"
            "cvt.rn.bf16x2.f32 %1, %6, %7;\n\t"
            "st.shared.b32 [%9], %1;\n\t"
            "cvt.rn.bf16x2.f32 %2, %4, %5;\n\t"
            "st.shared.b32 [%10], %2;\n\t"
            "cvt.rn.bf16x2.f32 %3, %6, %7;\n\t"
            "st.shared.b32 [%11], %3;\n\t"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "f"(a0), "f"(b0), "f"(a1), "f"(b1),
              "r"(saddr), "r"(saddr+4), "r"(saddr+8), "r"(saddr+12)
        );
        // 8 cvt + 8 sts = 16 instructions per iteration
    }
    long long t1 = clock64();

    // All threads write — prevents compiler from sinking loop into tid==0 branch
    out[0] = t1 - t0;
    out[1] = r0 + r1 + r2 + r3;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 4: HFMA2 throughput (independent streams)
// 16 independent bf16x2 FMAs — no RAW deps.
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void k4_hfma2_throughput(long long* out) {
    unsigned r0=0x3c003c00, r1=0x3c003c00, r2=0x3c003c00, r3=0x3c003c00;
    unsigned r4=0x3c003c00, r5=0x3c003c00, r6=0x3c003c00, r7=0x3c003c00;
    unsigned r8=0x3c003c00, r9=0x3c003c00, r10=0x3c003c00, r11=0x3c003c00;
    unsigned r12=0x3c003c00, r13=0x3c003c00, r14=0x3c003c00, r15=0x3c003c00;
    unsigned a = 0x3c003c00; // bf16 1.0

    for (int i = 0; i < WARMUP; i++) {
        asm volatile("fma.rn.bf16x2 %0, %1, %2, %0;" : "+r"(r0) : "r"(a), "r"(a));
    }

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "fma.rn.bf16x2 %0,  %16, %16, %0;\n\t"
            "fma.rn.bf16x2 %1,  %16, %16, %1;\n\t"
            "fma.rn.bf16x2 %2,  %16, %16, %2;\n\t"
            "fma.rn.bf16x2 %3,  %16, %16, %3;\n\t"
            "fma.rn.bf16x2 %4,  %16, %16, %4;\n\t"
            "fma.rn.bf16x2 %5,  %16, %16, %5;\n\t"
            "fma.rn.bf16x2 %6,  %16, %16, %6;\n\t"
            "fma.rn.bf16x2 %7,  %16, %16, %7;\n\t"
            "fma.rn.bf16x2 %8,  %16, %16, %8;\n\t"
            "fma.rn.bf16x2 %9,  %16, %16, %9;\n\t"
            "fma.rn.bf16x2 %10, %16, %16, %10;\n\t"
            "fma.rn.bf16x2 %11, %16, %16, %11;\n\t"
            "fma.rn.bf16x2 %12, %16, %16, %12;\n\t"
            "fma.rn.bf16x2 %13, %16, %16, %13;\n\t"
            "fma.rn.bf16x2 %14, %16, %16, %14;\n\t"
            "fma.rn.bf16x2 %15, %16, %16, %15;\n\t"
            : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3),
              "+r"(r4), "+r"(r5), "+r"(r6), "+r"(r7),
              "+r"(r8), "+r"(r9), "+r"(r10), "+r"(r11),
              "+r"(r12), "+r"(r13), "+r"(r14), "+r"(r15)
            : "r"(a)
        );
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 +
                 r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 5: HFMA2 + F2FP interleaved (pipe conflict test)
// Alternates bf16x2 fma and cvt. Same pipe → serializes. Different → overlaps.
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void k5_hfma2_f2fp_conflict(long long* out) {
    unsigned r0=0x3c003c00, r1=0x3c003c00, r2=0x3c003c00, r3=0x3c003c00;
    unsigned r4=0x3c003c00, r5=0x3c003c00, r6=0x3c003c00, r7=0x3c003c00;
    unsigned a = 0x3c003c00;
    float fa0=1.f, fb0=2.f, fa1=3.f, fb1=4.f;
    float fa2=5.f, fb2=6.f, fa3=7.f, fb3=8.f;
    unsigned c0, c1, c2, c3, c4, c5, c6, c7;

    for (int i = 0; i < WARMUP; i++) {
        asm volatile("fma.rn.bf16x2 %0, %1, %1, %0;" : "+r"(r0) : "r"(a));
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(c0) : "f"(fa0), "f"(fb0));
    }

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "fma.rn.bf16x2 %0, %16, %16, %0;\n\t"
            "cvt.rn.bf16x2.f32 %8,  %17, %18;\n\t"
            "fma.rn.bf16x2 %1, %16, %16, %1;\n\t"
            "cvt.rn.bf16x2.f32 %9,  %19, %20;\n\t"
            "fma.rn.bf16x2 %2, %16, %16, %2;\n\t"
            "cvt.rn.bf16x2.f32 %10, %21, %22;\n\t"
            "fma.rn.bf16x2 %3, %16, %16, %3;\n\t"
            "cvt.rn.bf16x2.f32 %11, %23, %24;\n\t"
            "fma.rn.bf16x2 %4, %16, %16, %4;\n\t"
            "cvt.rn.bf16x2.f32 %12, %17, %18;\n\t"
            "fma.rn.bf16x2 %5, %16, %16, %5;\n\t"
            "cvt.rn.bf16x2.f32 %13, %19, %20;\n\t"
            "fma.rn.bf16x2 %6, %16, %16, %6;\n\t"
            "cvt.rn.bf16x2.f32 %14, %21, %22;\n\t"
            "fma.rn.bf16x2 %7, %16, %16, %7;\n\t"
            "cvt.rn.bf16x2.f32 %15, %23, %24;\n\t"
            : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3),
              "+r"(r4), "+r"(r5), "+r"(r6), "+r"(r7),
              "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3),
              "=r"(c4), "=r"(c5), "=r"(c6), "=r"(c7)
            : "r"(a),
              "f"(fa0), "f"(fb0), "f"(fa1), "f"(fb1),
              "f"(fa2), "f"(fb2), "f"(fa3), "f"(fb3)
        );
        // 8 hfma2 + 8 f2fp = 16 instructions per iteration
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 +
                 c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 6: STS throughput (independent, no RAW deps)
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void k6_sts_throughput(long long* out) {
    __shared__ int smem[1024];
    int tid = threadIdx.x;
    uint32_t base = (uint32_t)(uint64_t)(&smem[tid * 16]);
    unsigned v0 = 0xdeadbeef, v1 = 0xcafebabe, v2 = 0x12345678, v3 = 0x9abcdef0;

    for (int i = 0; i < WARMUP; i++) {
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(base), "r"(v0));
    }

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "st.shared.v4.b32 [%0],     {%4,%5,%6,%7};\n\t"
            "st.shared.v4.b32 [%0+16],  {%4,%5,%6,%7};\n\t"
            "st.shared.v4.b32 [%1],     {%4,%5,%6,%7};\n\t"
            "st.shared.v4.b32 [%1+16],  {%4,%5,%6,%7};\n\t"
            "st.shared.v4.b32 [%2],     {%4,%5,%6,%7};\n\t"
            "st.shared.v4.b32 [%2+16],  {%4,%5,%6,%7};\n\t"
            "st.shared.v4.b32 [%3],     {%4,%5,%6,%7};\n\t"
            "st.shared.v4.b32 [%3+16],  {%4,%5,%6,%7};\n\t"
            "st.shared.v4.b32 [%0],     {%4,%5,%6,%7};\n\t"
            "st.shared.v4.b32 [%0+16],  {%4,%5,%6,%7};\n\t"
            "st.shared.v4.b32 [%1],     {%4,%5,%6,%7};\n\t"
            "st.shared.v4.b32 [%1+16],  {%4,%5,%6,%7};\n\t"
            "st.shared.v4.b32 [%2],     {%4,%5,%6,%7};\n\t"
            "st.shared.v4.b32 [%2+16],  {%4,%5,%6,%7};\n\t"
            "st.shared.v4.b32 [%3],     {%4,%5,%6,%7};\n\t"
            "st.shared.v4.b32 [%3+16],  {%4,%5,%6,%7};\n\t"
            :: "r"(base), "r"(base+32), "r"(base+128), "r"(base+160),
               "r"(v0), "r"(v1), "r"(v2), "r"(v3)
            : "memory"
        );
        // 16 st.shared.v4 per iteration
    }
    long long t1 = clock64();

    if (tid == 0) {
        out[0] = t1 - t0;
        out[1] = smem[0];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 7a: IADD3 independent (control word decoder — expect stall=0)
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void k7a_iadd_independent(long long* out) {
    // Load from global memory — opaque to ptxas, prevents constant folding
    volatile int* gin = (volatile int*)out;
    int inc = gin[threadIdx.x];
    int a=inc+1, b=inc+2, c=inc+3, d=inc+4, e=inc+5, f=inc+6, g=inc+7, h=inc+8;
    int i0=inc+9, j=inc+10, k=inc+11, l=inc+12, m=inc+13, n=inc+14, o=inc+15, p=inc+16;

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "add.s32 %0, %0, %16;\n\t"
            "add.s32 %1, %1, %16;\n\t"
            "add.s32 %2, %2, %16;\n\t"
            "add.s32 %3, %3, %16;\n\t"
            "add.s32 %4, %4, %16;\n\t"
            "add.s32 %5, %5, %16;\n\t"
            "add.s32 %6, %6, %16;\n\t"
            "add.s32 %7, %7, %16;\n\t"
            "add.s32 %8, %8, %16;\n\t"
            "add.s32 %9, %9, %16;\n\t"
            "add.s32 %10, %10, %16;\n\t"
            "add.s32 %11, %11, %16;\n\t"
            "add.s32 %12, %12, %16;\n\t"
            "add.s32 %13, %13, %16;\n\t"
            "add.s32 %14, %14, %16;\n\t"
            "add.s32 %15, %15, %16;\n\t"
            : "+r"(a), "+r"(b), "+r"(c), "+r"(d),
              "+r"(e), "+r"(f), "+r"(g), "+r"(h),
              "+r"(i0), "+r"(j), "+r"(k), "+r"(l),
              "+r"(m), "+r"(n), "+r"(o), "+r"(p)
            : "r"(inc)
        );
    }
    long long t1 = clock64();

    // All threads write (same values) — prevents compiler from sinking loop
    // into a thread-0-only branch, which would move it after both clock reads
    out[0] = t1 - t0;
    out[1] = a+b+c+d+e+f+g+h+i0+j+k+l+m+n+o+p;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 7b: IADD3 dependent chain (control word decoder — expect stall=latency)
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void k7b_iadd_dependent(long long* out) {
    volatile int* gin = (volatile int*)out;
    int a = gin[threadIdx.x];
    int inc = gin[threadIdx.x + 32];

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "add.s32 %0, %0, %1;\n\t"
            "add.s32 %0, %0, %1;\n\t"
            "add.s32 %0, %0, %1;\n\t"
            "add.s32 %0, %0, %1;\n\t"
            "add.s32 %0, %0, %1;\n\t"
            "add.s32 %0, %0, %1;\n\t"
            "add.s32 %0, %0, %1;\n\t"
            "add.s32 %0, %0, %1;\n\t"
            "add.s32 %0, %0, %1;\n\t"
            "add.s32 %0, %0, %1;\n\t"
            "add.s32 %0, %0, %1;\n\t"
            "add.s32 %0, %0, %1;\n\t"
            "add.s32 %0, %0, %1;\n\t"
            "add.s32 %0, %0, %1;\n\t"
            "add.s32 %0, %0, %1;\n\t"
            "add.s32 %0, %0, %1;\n\t"
            : "+r"(a)
            : "r"(inc)
        );
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = a;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 8: PRMT throughput (independent)
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void k8_prmt_throughput(long long* out) {
    unsigned a=0x12345678, b=0x9abcdef0;
    unsigned r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;

    for (int i = 0; i < WARMUP; i++) {
        asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(r0) : "r"(a), "r"(b));
    }

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "prmt.b32 %0,  %16, %17, 0x5410;\n\t"
            "prmt.b32 %1,  %17, %16, 0x5410;\n\t"
            "prmt.b32 %2,  %16, %17, 0x3210;\n\t"
            "prmt.b32 %3,  %17, %16, 0x3210;\n\t"
            "prmt.b32 %4,  %16, %17, 0x7654;\n\t"
            "prmt.b32 %5,  %17, %16, 0x7654;\n\t"
            "prmt.b32 %6,  %16, %17, 0x1032;\n\t"
            "prmt.b32 %7,  %17, %16, 0x1032;\n\t"
            "prmt.b32 %8,  %16, %17, 0x5410;\n\t"
            "prmt.b32 %9,  %17, %16, 0x5410;\n\t"
            "prmt.b32 %10, %16, %17, 0x3210;\n\t"
            "prmt.b32 %11, %17, %16, 0x3210;\n\t"
            "prmt.b32 %12, %16, %17, 0x7654;\n\t"
            "prmt.b32 %13, %17, %16, 0x7654;\n\t"
            "prmt.b32 %14, %16, %17, 0x1032;\n\t"
            "prmt.b32 %15, %17, %16, 0x1032;\n\t"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3),
              "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7),
              "=r"(r8), "=r"(r9), "=r"(r10), "=r"(r11),
              "=r"(r12), "=r"(r13), "=r"(r14), "=r"(r15)
            : "r"(a), "r"(b)
        );
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = r0+r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13+r14+r15;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 9: F2FP throughput at 32 wide (match epilogue pattern)
// 32 independent cvts — closer to the actual 32-per-warp epilogue pattern
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void k9_f2fp_wide(long long* out) {
    // 32 independent F2FPs — matches epilogue's 32 CVTs per warp per 64-col chunk
    // Split into two asm blocks of 16 (operand limit)
    float a0=1.f,a1=2.f,a2=3.f,a3=4.f,a4=5.f,a5=6.f,a6=7.f,a7=8.f;
    float a8=9.f,a9=10.f,a10=11.f,a11=12.f,a12=13.f,a13=14.f,a14=15.f,a15=16.f;
    float b0=17.f,b1=18.f,b2=19.f,b3=20.f,b4=21.f,b5=22.f,b6=23.f,b7=24.f;
    float b8=25.f,b9=26.f,b10=27.f,b11=28.f,b12=29.f,b13=30.f,b14=31.f,b15=32.f;
    // Second set of 16 — reuse same float values (independent results)
    float c0=33.f,c1=34.f,c2=35.f,c3=36.f,c4=37.f,c5=38.f,c6=39.f,c7=40.f;
    float c8=41.f,c9=42.f,c10=43.f,c11=44.f,c12=45.f,c13=46.f,c14=47.f,c15=48.f;
    float d0=49.f,d1=50.f,d2=51.f,d3=52.f,d4=53.f,d5=54.f,d6=55.f,d7=56.f;
    float d8=57.f,d9=58.f,d10=59.f,d11=60.f,d12=61.f,d13=62.f,d14=63.f,d15=64.f;
    unsigned r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;
    unsigned s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15;

    for (int i = 0; i < WARMUP; i++) {
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r0) : "f"(a0), "f"(b0));
    }

    long long t0 = clock64();
    for (int rep = 0; rep < REPS; rep++) {
        asm volatile(
            "cvt.rn.bf16x2.f32 %0,  %16, %17;\n\t"
            "cvt.rn.bf16x2.f32 %1,  %18, %19;\n\t"
            "cvt.rn.bf16x2.f32 %2,  %20, %21;\n\t"
            "cvt.rn.bf16x2.f32 %3,  %22, %23;\n\t"
            "cvt.rn.bf16x2.f32 %4,  %24, %25;\n\t"
            "cvt.rn.bf16x2.f32 %5,  %26, %27;\n\t"
            "cvt.rn.bf16x2.f32 %6,  %28, %29;\n\t"
            "cvt.rn.bf16x2.f32 %7,  %30, %31;\n\t"
            "cvt.rn.bf16x2.f32 %8,  %32, %33;\n\t"
            "cvt.rn.bf16x2.f32 %9,  %34, %35;\n\t"
            "cvt.rn.bf16x2.f32 %10, %36, %37;\n\t"
            "cvt.rn.bf16x2.f32 %11, %38, %39;\n\t"
            "cvt.rn.bf16x2.f32 %12, %40, %41;\n\t"
            "cvt.rn.bf16x2.f32 %13, %42, %43;\n\t"
            "cvt.rn.bf16x2.f32 %14, %44, %45;\n\t"
            "cvt.rn.bf16x2.f32 %15, %46, %47;\n\t"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3),
              "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7),
              "=r"(r8), "=r"(r9), "=r"(r10), "=r"(r11),
              "=r"(r12), "=r"(r13), "=r"(r14), "=r"(r15)
            : "f"(a0), "f"(b0), "f"(a1), "f"(b1),
              "f"(a2), "f"(b2), "f"(a3), "f"(b3),
              "f"(a4), "f"(b4), "f"(a5), "f"(b5),
              "f"(a6), "f"(b6), "f"(a7), "f"(b7),
              "f"(a8), "f"(b8), "f"(a9), "f"(b9),
              "f"(a10), "f"(b10), "f"(a11), "f"(b11),
              "f"(a12), "f"(b12), "f"(a13), "f"(b13),
              "f"(a14), "f"(b14), "f"(a15), "f"(b15)
        );
        asm volatile(
            "cvt.rn.bf16x2.f32 %0,  %16, %17;\n\t"
            "cvt.rn.bf16x2.f32 %1,  %18, %19;\n\t"
            "cvt.rn.bf16x2.f32 %2,  %20, %21;\n\t"
            "cvt.rn.bf16x2.f32 %3,  %22, %23;\n\t"
            "cvt.rn.bf16x2.f32 %4,  %24, %25;\n\t"
            "cvt.rn.bf16x2.f32 %5,  %26, %27;\n\t"
            "cvt.rn.bf16x2.f32 %6,  %28, %29;\n\t"
            "cvt.rn.bf16x2.f32 %7,  %30, %31;\n\t"
            "cvt.rn.bf16x2.f32 %8,  %32, %33;\n\t"
            "cvt.rn.bf16x2.f32 %9,  %34, %35;\n\t"
            "cvt.rn.bf16x2.f32 %10, %36, %37;\n\t"
            "cvt.rn.bf16x2.f32 %11, %38, %39;\n\t"
            "cvt.rn.bf16x2.f32 %12, %40, %41;\n\t"
            "cvt.rn.bf16x2.f32 %13, %42, %43;\n\t"
            "cvt.rn.bf16x2.f32 %14, %44, %45;\n\t"
            "cvt.rn.bf16x2.f32 %15, %46, %47;\n\t"
            : "=r"(s0), "=r"(s1), "=r"(s2), "=r"(s3),
              "=r"(s4), "=r"(s5), "=r"(s6), "=r"(s7),
              "=r"(s8), "=r"(s9), "=r"(s10), "=r"(s11),
              "=r"(s12), "=r"(s13), "=r"(s14), "=r"(s15)
            : "f"(c0), "f"(d0), "f"(c1), "f"(d1),
              "f"(c2), "f"(d2), "f"(c3), "f"(d3),
              "f"(c4), "f"(d4), "f"(c5), "f"(d5),
              "f"(c6), "f"(d6), "f"(c7), "f"(d7),
              "f"(c8), "f"(d8), "f"(c9), "f"(d9),
              "f"(c10), "f"(d10), "f"(c11), "f"(d11),
              "f"(c12), "f"(d12), "f"(c13), "f"(d13),
              "f"(c14), "f"(d14), "f"(c15), "f"(d15)
        );
        // 32 independent cvts per iteration
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = r0+r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13+r14+r15+
                 s0+s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host: run all kernels, print results
// ─────────────────────────────────────────────────────────────────────────────

struct Bench {
    const char* name;
    int instrs_per_iter;  // instructions measured per REPS iteration
    void (*fn)(long long*);
};

int main() {
    long long *d_out, h_out[2];
    cudaMalloc(&d_out, 1024);  // extra space for K7a/K7b global memory reads
    cudaMemset(d_out, 0, 1024);

    Bench benches[] = {
        {"K1: F2FP throughput (16 indep)",        16, k1_f2fp_throughput},
        {"K2: F2FP latency (dep chain)",           8, k2_f2fp_latency},
            // 8 cvt+mov pairs; latency = cycles / (REPS*8) for cvt only
            // (mov adds to chain, so real cvt latency ≈ measured - mov_latency)
        {"K3: F2FP+STS conflict (interleaved)",   16, k3_f2fp_sts_conflict},
        {"K4: HFMA2 throughput (16 indep)",        16, k4_hfma2_throughput},
        {"K5: HFMA2+F2FP conflict (interleaved)", 16, k5_hfma2_f2fp_conflict},
        {"K6: STS.v4 throughput (16 indep)",       16, k6_sts_throughput},
        {"K7a: IADD independent (decoder)",        16, k7a_iadd_independent},
        {"K7b: IADD dependent (decoder)",          16, k7b_iadd_dependent},
        {"K8: PRMT throughput (16 indep)",         16, k8_prmt_throughput},
        {"K9: F2FP throughput (32 indep)",         32, k9_f2fp_wide},
    };
    int n_benches = sizeof(benches) / sizeof(benches[0]);

    printf("SASS Calibration — SM 100a\n");
    printf("REPS=%d per kernel, 1 block, 1 warp (32 threads)\n\n", REPS);
    printf("%-45s %10s %10s %10s\n", "Kernel", "Cycles", "Cyc/instr", "Throughput");
    printf("%-45s %10s %10s %10s\n", "------", "------", "---------", "----------");

    for (int i = 0; i < n_benches; i++) {
        cudaMemset(d_out, 0, 2 * sizeof(long long));

        // warmup launch
        benches[i].fn<<<1, 32>>>(d_out);
        cudaDeviceSynchronize();

        // real launch
        benches[i].fn<<<1, 32>>>(d_out);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out, d_out, 2 * sizeof(long long), cudaMemcpyDeviceToHost);

        long long total = h_out[0];
        double per_instr = (double)total / (REPS * benches[i].instrs_per_iter);
        double throughput = 1.0 / per_instr;

        printf("%-45s %10lld %10.2f %10.4f\n",
               benches[i].name, total, per_instr, throughput);
    }

    printf("\n@@CALIBRATION\n");
    printf("Interpretation:\n");
    printf("  K1 cyc/instr = F2FP throughput (cycles between independent issues)\n");
    printf("  K2 cyc/instr = F2FP+MOV chain latency (subtract MOV latency for pure F2FP)\n");
    printf("  K3 vs K1+K6: if K3 ≈ max(K1,K6) → different pipes; if K3 ≈ K1+K6 → same pipe\n");
    printf("  K5 vs K1+K4: if K5 ≈ max(K1,K4) → different pipes; if K5 ≈ K1+K4 → same pipe\n");
    printf("  K7a vs K7b: stall count = K7b_cyc/instr - K7a_cyc/instr (verify bits [3:0])\n");
    printf("  K9 vs K1: check if throughput degrades at 32-wide (register file pressure)\n");

    cudaFree(d_out);
    return 0;
}
