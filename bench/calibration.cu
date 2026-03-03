#include <cstdio>
#include <cstdint>
#include <cuda_bf16.h>

#define REPS 1024
#define WARMUP 64

extern "C" __global__ void k1_f2fp_throughput(long long* out) {
    // 16 independent F2FP feedback chains: each accumulator feeds back as one
    // f32 input to cvt, cvt output replaces accumulator. Loop-carried dependency
    // through the target instruction itself prevents ptxas loop elimination.
    // No XOR needed — cvt IS the accumulation. Zero overhead instructions.
    volatile unsigned* gin = (volatile unsigned*)(out + 16);
    unsigned f0=gin[0], f1=gin[1], f2=gin[2], f3=gin[3];
    unsigned f4=gin[4], f5=gin[5], f6=gin[6], f7=gin[7];

    volatile unsigned* uin = (volatile unsigned*)out;
    unsigned r0=uin[0], r1=uin[1], r2=uin[2], r3=uin[3];
    unsigned r4=uin[4], r5=uin[5], r6=uin[6], r7=uin[7];
    unsigned r8=uin[8], r9=uin[9], r10=uin[10], r11=uin[11];
    unsigned r12=uin[12], r13=uin[13], r14=uin[14], r15=uin[15];

    for (int i = 0; i < WARMUP; i++)
        asm volatile("{ .reg .f32 fa, fb; mov.b32 fa, %0; mov.b32 fb, %1; cvt.rn.bf16x2.f32 %0, fa, fb; }"
                     : "+r"(r0) : "r"(f0));

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "{ .reg .f32 fa, fb;\n\t"
            "mov.b32 fa, %0;  mov.b32 fb, %16; cvt.rn.bf16x2.f32 %0,  fa, fb;\n\t"
            "mov.b32 fa, %1;  mov.b32 fb, %17; cvt.rn.bf16x2.f32 %1,  fa, fb;\n\t"
            "mov.b32 fa, %2;  mov.b32 fb, %18; cvt.rn.bf16x2.f32 %2,  fa, fb;\n\t"
            "mov.b32 fa, %3;  mov.b32 fb, %19; cvt.rn.bf16x2.f32 %3,  fa, fb;\n\t"
            "mov.b32 fa, %4;  mov.b32 fb, %20; cvt.rn.bf16x2.f32 %4,  fa, fb;\n\t"
            "mov.b32 fa, %5;  mov.b32 fb, %21; cvt.rn.bf16x2.f32 %5,  fa, fb;\n\t"
            "mov.b32 fa, %6;  mov.b32 fb, %22; cvt.rn.bf16x2.f32 %6,  fa, fb;\n\t"
            "mov.b32 fa, %7;  mov.b32 fb, %23; cvt.rn.bf16x2.f32 %7,  fa, fb;\n\t"
            "mov.b32 fa, %8;  mov.b32 fb, %16; cvt.rn.bf16x2.f32 %8,  fa, fb;\n\t"
            "mov.b32 fa, %9;  mov.b32 fb, %17; cvt.rn.bf16x2.f32 %9,  fa, fb;\n\t"
            "mov.b32 fa, %10; mov.b32 fb, %18; cvt.rn.bf16x2.f32 %10, fa, fb;\n\t"
            "mov.b32 fa, %11; mov.b32 fb, %19; cvt.rn.bf16x2.f32 %11, fa, fb;\n\t"
            "mov.b32 fa, %12; mov.b32 fb, %20; cvt.rn.bf16x2.f32 %12, fa, fb;\n\t"
            "mov.b32 fa, %13; mov.b32 fb, %21; cvt.rn.bf16x2.f32 %13, fa, fb;\n\t"
            "mov.b32 fa, %14; mov.b32 fb, %22; cvt.rn.bf16x2.f32 %14, fa, fb;\n\t"
            "mov.b32 fa, %15; mov.b32 fb, %23; cvt.rn.bf16x2.f32 %15, fa, fb;\n\t"
            "}\n\t"
            : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3),
              "+r"(r4), "+r"(r5), "+r"(r6), "+r"(r7),
              "+r"(r8), "+r"(r9), "+r"(r10), "+r"(r11),
              "+r"(r12), "+r"(r13), "+r"(r14), "+r"(r15)
            : "r"(f0), "r"(f1), "r"(f2), "r"(f3),
              "r"(f4), "r"(f5), "r"(f6), "r"(f7)
        );
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 +
                 r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15;
    }
}

extern "C" __global__ void k2_f2fp_latency(long long* out) {
    // Latency chain: cvt reads from "+r" a (bitcast to f32 inside asm), writes back to a.
    // mov.b32 inside asm is a type bitcast (u32→f32), likely SASS NOP.
    unsigned a = 0x3f800000; // 1.0f bit pattern

    for (int i = 0; i < WARMUP; i++)
        asm volatile("{ .reg .f32 fa; mov.b32 fa, %0; cvt.rn.bf16x2.f32 %0, fa, fa; }" : "+r"(a));

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "{ .reg .f32 fa;\n\t"
            "mov.b32 fa, %0; cvt.rn.bf16x2.f32 %0, fa, fa;\n\t"
            "mov.b32 fa, %0; cvt.rn.bf16x2.f32 %0, fa, fa;\n\t"
            "mov.b32 fa, %0; cvt.rn.bf16x2.f32 %0, fa, fa;\n\t"
            "mov.b32 fa, %0; cvt.rn.bf16x2.f32 %0, fa, fa;\n\t"
            "mov.b32 fa, %0; cvt.rn.bf16x2.f32 %0, fa, fa;\n\t"
            "mov.b32 fa, %0; cvt.rn.bf16x2.f32 %0, fa, fa;\n\t"
            "mov.b32 fa, %0; cvt.rn.bf16x2.f32 %0, fa, fa;\n\t"
            "mov.b32 fa, %0; cvt.rn.bf16x2.f32 %0, fa, fa;\n\t"
            "}\n\t"
            : "+r"(a));
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = a;
    }
}

extern "C" __global__ void k3_f2fp_sts_conflict(long long* out) {
    __shared__ int smem[1024];
    // Float bit patterns: 1.0f, 2.0f, 3.0f, 4.0f — passed as "r", bitcast inside asm.
    unsigned a0=0x3f800000, b0=0x40000000, a1=0x40400000, b1=0x40800000;
    unsigned r0, r1, r2, r3;
    int tid = threadIdx.x;
    uint32_t saddr = (uint32_t)(uint64_t)(&smem[tid * 4]);

    for (int i = 0; i < WARMUP; i++) {
        asm volatile("{ .reg .f32 fa, fb; mov.b32 fa, %1; mov.b32 fb, %2; cvt.rn.bf16x2.f32 %0, fa, fb; }"
                     : "=r"(r0) : "r"(a0), "r"(b0));
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(saddr), "r"(r0));
    }

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "{ .reg .f32 fa, fb;\n\t"
            "mov.b32 fa, %4; mov.b32 fb, %5; cvt.rn.bf16x2.f32 %0, fa, fb;\n\t"
            "st.shared.b32 [%8], %0;\n\t"
            "mov.b32 fa, %6; mov.b32 fb, %7; cvt.rn.bf16x2.f32 %1, fa, fb;\n\t"
            "st.shared.b32 [%9], %1;\n\t"
            "mov.b32 fa, %4; mov.b32 fb, %5; cvt.rn.bf16x2.f32 %2, fa, fb;\n\t"
            "st.shared.b32 [%10], %2;\n\t"
            "mov.b32 fa, %6; mov.b32 fb, %7; cvt.rn.bf16x2.f32 %3, fa, fb;\n\t"
            "st.shared.b32 [%11], %3;\n\t"
            "mov.b32 fa, %4; mov.b32 fb, %5; cvt.rn.bf16x2.f32 %0, fa, fb;\n\t"
            "st.shared.b32 [%8], %0;\n\t"
            "mov.b32 fa, %6; mov.b32 fb, %7; cvt.rn.bf16x2.f32 %1, fa, fb;\n\t"
            "st.shared.b32 [%9], %1;\n\t"
            "mov.b32 fa, %4; mov.b32 fb, %5; cvt.rn.bf16x2.f32 %2, fa, fb;\n\t"
            "st.shared.b32 [%10], %2;\n\t"
            "mov.b32 fa, %6; mov.b32 fb, %7; cvt.rn.bf16x2.f32 %3, fa, fb;\n\t"
            "st.shared.b32 [%11], %3;\n\t"
            "}\n\t"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(a0), "r"(b0), "r"(a1), "r"(b1),
              "r"(saddr), "r"(saddr+4), "r"(saddr+8), "r"(saddr+12)
        );
    }
    long long t1 = clock64();

    out[0] = t1 - t0;
    out[1] = r0 + r1 + r2 + r3;
}

extern "C" __global__ void k4_hfma2_throughput(long long* out) {
    // Volatile inputs prevent constant-folding. Distinct accumulator inits prevent CSE.
    volatile unsigned* gin = (volatile unsigned*)out;
    unsigned a = gin[threadIdx.x] | 0x3c003c00; // opaque multiplier, ≈ bf16 1.0
    unsigned r0=gin[0], r1=gin[1], r2=gin[2], r3=gin[3];
    unsigned r4=gin[4], r5=gin[5], r6=gin[6], r7=gin[7];
    unsigned r8=gin[8], r9=gin[9], r10=gin[10], r11=gin[11];
    unsigned r12=gin[12], r13=gin[13], r14=gin[14], r15=gin[15];

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

extern "C" __global__ void k5_hfma2_f2fp_conflict(long long* out) {
    // Volatile inputs prevent constant-fold/CSE. XOR accum for F2FP outputs.
    volatile unsigned* gin = (volatile unsigned*)out;
    unsigned a = gin[threadIdx.x] | 0x3c003c00;
    unsigned r0=gin[0], r1=gin[1], r2=gin[2], r3=gin[3];
    unsigned r4=gin[4], r5=gin[5], r6=gin[6], r7=gin[7];
    unsigned c0=gin[8], c1=gin[9], c2=gin[10], c3=gin[11];
    unsigned c4=gin[12], c5=gin[13], c6=gin[14], c7=gin[15];

    volatile unsigned* fin = (volatile unsigned*)(out + 16);
    unsigned fa0=fin[0], fb0=fin[1], fa1=fin[2], fb1=fin[3];
    unsigned fa2=fin[4], fb2=fin[5], fa3=fin[6], fb3=fin[7];

    for (int i = 0; i < WARMUP; i++) {
        asm volatile("fma.rn.bf16x2 %0, %1, %1, %0;" : "+r"(r0) : "r"(a));
        asm volatile("{ .reg .f32 fa, fb; mov.b32 fa, %0; mov.b32 fb, %1; cvt.rn.bf16x2.f32 %0, fa, fb; }"
                     : "+r"(c0) : "r"(fa0));
    }

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "{ .reg .f32 fa, fb;\n\t"
            "fma.rn.bf16x2 %0, %16, %16, %0;\n\t"
            "mov.b32 fa, %8;  mov.b32 fb, %17; cvt.rn.bf16x2.f32 %8,  fa, fb;\n\t"
            "fma.rn.bf16x2 %1, %16, %16, %1;\n\t"
            "mov.b32 fa, %9;  mov.b32 fb, %18; cvt.rn.bf16x2.f32 %9,  fa, fb;\n\t"
            "fma.rn.bf16x2 %2, %16, %16, %2;\n\t"
            "mov.b32 fa, %10; mov.b32 fb, %19; cvt.rn.bf16x2.f32 %10, fa, fb;\n\t"
            "fma.rn.bf16x2 %3, %16, %16, %3;\n\t"
            "mov.b32 fa, %11; mov.b32 fb, %20; cvt.rn.bf16x2.f32 %11, fa, fb;\n\t"
            "fma.rn.bf16x2 %4, %16, %16, %4;\n\t"
            "mov.b32 fa, %12; mov.b32 fb, %21; cvt.rn.bf16x2.f32 %12, fa, fb;\n\t"
            "fma.rn.bf16x2 %5, %16, %16, %5;\n\t"
            "mov.b32 fa, %13; mov.b32 fb, %22; cvt.rn.bf16x2.f32 %13, fa, fb;\n\t"
            "fma.rn.bf16x2 %6, %16, %16, %6;\n\t"
            "mov.b32 fa, %14; mov.b32 fb, %23; cvt.rn.bf16x2.f32 %14, fa, fb;\n\t"
            "fma.rn.bf16x2 %7, %16, %16, %7;\n\t"
            "mov.b32 fa, %15; mov.b32 fb, %24; cvt.rn.bf16x2.f32 %15, fa, fb;\n\t"
            "}\n\t"
            : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3),
              "+r"(r4), "+r"(r5), "+r"(r6), "+r"(r7),
              "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3),
              "+r"(c4), "+r"(c5), "+r"(c6), "+r"(c7)
            : "r"(a),
              "r"(fa0), "r"(fb0), "r"(fa1), "r"(fb1),
              "r"(fa2), "r"(fb2), "r"(fa3), "r"(fb3)
        );
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 +
                 c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
    }
}

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
    }
    long long t1 = clock64();

    if (tid == 0) {
        out[0] = t1 - t0;
        out[1] = smem[0];
    }
}

extern "C" __global__ void k7a_iadd_independent(long long* out) {
    volatile int* gin = (volatile int*)out; // opaque to ptxas
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

    out[0] = t1 - t0; // all threads write — prevents compiler sinking loop past clock
    out[1] = a+b+c+d+e+f+g+h+i0+j+k+l+m+n+o+p;
}

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

extern "C" __global__ void k8_prmt_throughput(long long* out) {
    // 16 independent PRMT feedback chains: accumulator feeds back as first
    // prmt source. Loop-carried dependency through prmt prevents elimination.
    // No XOR needed — prmt IS the accumulation. Zero overhead instructions.
    volatile unsigned* gin = (volatile unsigned*)out;
    unsigned a = gin[threadIdx.x];
    unsigned b = gin[threadIdx.x + 32];
    unsigned r0=gin[0], r1=gin[1], r2=gin[2], r3=gin[3];
    unsigned r4=gin[4], r5=gin[5], r6=gin[6], r7=gin[7];
    unsigned r8=gin[8], r9=gin[9], r10=gin[10], r11=gin[11];
    unsigned r12=gin[12], r13=gin[13], r14=gin[14], r15=gin[15];

    for (int i = 0; i < WARMUP; i++)
        asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(r0) : "r"(a));

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "prmt.b32 %0,  %0,  %16, 0x5410;\n\t"
            "prmt.b32 %1,  %1,  %17, 0x5410;\n\t"
            "prmt.b32 %2,  %2,  %16, 0x3210;\n\t"
            "prmt.b32 %3,  %3,  %17, 0x3210;\n\t"
            "prmt.b32 %4,  %4,  %16, 0x7654;\n\t"
            "prmt.b32 %5,  %5,  %17, 0x7654;\n\t"
            "prmt.b32 %6,  %6,  %16, 0x1032;\n\t"
            "prmt.b32 %7,  %7,  %17, 0x1032;\n\t"
            "prmt.b32 %8,  %8,  %16, 0x4567;\n\t"
            "prmt.b32 %9,  %9,  %17, 0x4567;\n\t"
            "prmt.b32 %10, %10, %16, 0x0123;\n\t"
            "prmt.b32 %11, %11, %17, 0x0123;\n\t"
            "prmt.b32 %12, %12, %16, 0x6745;\n\t"
            "prmt.b32 %13, %13, %17, 0x6745;\n\t"
            "prmt.b32 %14, %14, %16, 0x2301;\n\t"
            "prmt.b32 %15, %15, %17, 0x2301;\n\t"
            : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3),
              "+r"(r4), "+r"(r5), "+r"(r6), "+r"(r7),
              "+r"(r8), "+r"(r9), "+r"(r10), "+r"(r11),
              "+r"(r12), "+r"(r13), "+r"(r14), "+r"(r15)
            : "r"(a), "r"(b)
        );
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = r0+r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13+r14+r15;
    }
}

extern "C" __global__ void k9_f2fp_wide(long long* out) {
    // 32-wide F2FP feedback chains: same as K1 but 2×16 blocks.
    // Checks if throughput degrades at higher ILP (register file pressure).
    volatile unsigned* gin = (volatile unsigned*)(out + 16);
    unsigned f0=gin[0], f1=gin[1], f2=gin[2], f3=gin[3];
    unsigned f4=gin[4], f5=gin[5], f6=gin[6], f7=gin[7];
    unsigned g0=gin[8], g1=gin[9], g2=gin[10], g3=gin[11];
    unsigned g4=gin[12], g5=gin[13], g6=gin[14], g7=gin[15];

    volatile unsigned* uin = (volatile unsigned*)out;
    unsigned r0=uin[0], r1=uin[1], r2=uin[2], r3=uin[3];
    unsigned r4=uin[4], r5=uin[5], r6=uin[6], r7=uin[7];
    unsigned r8=uin[8], r9=uin[9], r10=uin[10], r11=uin[11];
    unsigned r12=uin[12], r13=uin[13], r14=uin[14], r15=uin[15];
    unsigned s0=uin[16], s1=uin[17], s2=uin[18], s3=uin[19];
    unsigned s4=uin[20], s5=uin[21], s6=uin[22], s7=uin[23];
    unsigned s8=uin[24], s9=uin[25], s10=uin[26], s11=uin[27];
    unsigned s12=uin[28], s13=uin[29], s14=uin[30], s15=uin[31];

    for (int i = 0; i < WARMUP; i++)
        asm volatile("{ .reg .f32 fa, fb; mov.b32 fa, %0; mov.b32 fb, %1; cvt.rn.bf16x2.f32 %0, fa, fb; }"
                     : "+r"(r0) : "r"(f0));

    long long t0 = clock64();
    for (int rep = 0; rep < REPS; rep++) {
        asm volatile(
            "{ .reg .f32 fa, fb;\n\t"
            "mov.b32 fa, %0;  mov.b32 fb, %16; cvt.rn.bf16x2.f32 %0,  fa, fb;\n\t"
            "mov.b32 fa, %1;  mov.b32 fb, %17; cvt.rn.bf16x2.f32 %1,  fa, fb;\n\t"
            "mov.b32 fa, %2;  mov.b32 fb, %18; cvt.rn.bf16x2.f32 %2,  fa, fb;\n\t"
            "mov.b32 fa, %3;  mov.b32 fb, %19; cvt.rn.bf16x2.f32 %3,  fa, fb;\n\t"
            "mov.b32 fa, %4;  mov.b32 fb, %20; cvt.rn.bf16x2.f32 %4,  fa, fb;\n\t"
            "mov.b32 fa, %5;  mov.b32 fb, %21; cvt.rn.bf16x2.f32 %5,  fa, fb;\n\t"
            "mov.b32 fa, %6;  mov.b32 fb, %22; cvt.rn.bf16x2.f32 %6,  fa, fb;\n\t"
            "mov.b32 fa, %7;  mov.b32 fb, %23; cvt.rn.bf16x2.f32 %7,  fa, fb;\n\t"
            "mov.b32 fa, %8;  mov.b32 fb, %16; cvt.rn.bf16x2.f32 %8,  fa, fb;\n\t"
            "mov.b32 fa, %9;  mov.b32 fb, %17; cvt.rn.bf16x2.f32 %9,  fa, fb;\n\t"
            "mov.b32 fa, %10; mov.b32 fb, %18; cvt.rn.bf16x2.f32 %10, fa, fb;\n\t"
            "mov.b32 fa, %11; mov.b32 fb, %19; cvt.rn.bf16x2.f32 %11, fa, fb;\n\t"
            "mov.b32 fa, %12; mov.b32 fb, %20; cvt.rn.bf16x2.f32 %12, fa, fb;\n\t"
            "mov.b32 fa, %13; mov.b32 fb, %21; cvt.rn.bf16x2.f32 %13, fa, fb;\n\t"
            "mov.b32 fa, %14; mov.b32 fb, %22; cvt.rn.bf16x2.f32 %14, fa, fb;\n\t"
            "mov.b32 fa, %15; mov.b32 fb, %23; cvt.rn.bf16x2.f32 %15, fa, fb;\n\t"
            "}\n\t"
            : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3),
              "+r"(r4), "+r"(r5), "+r"(r6), "+r"(r7),
              "+r"(r8), "+r"(r9), "+r"(r10), "+r"(r11),
              "+r"(r12), "+r"(r13), "+r"(r14), "+r"(r15)
            : "r"(f0), "r"(f1), "r"(f2), "r"(f3),
              "r"(f4), "r"(f5), "r"(f6), "r"(f7)
        );
        asm volatile(
            "{ .reg .f32 fa, fb;\n\t"
            "mov.b32 fa, %0;  mov.b32 fb, %16; cvt.rn.bf16x2.f32 %0,  fa, fb;\n\t"
            "mov.b32 fa, %1;  mov.b32 fb, %17; cvt.rn.bf16x2.f32 %1,  fa, fb;\n\t"
            "mov.b32 fa, %2;  mov.b32 fb, %18; cvt.rn.bf16x2.f32 %2,  fa, fb;\n\t"
            "mov.b32 fa, %3;  mov.b32 fb, %19; cvt.rn.bf16x2.f32 %3,  fa, fb;\n\t"
            "mov.b32 fa, %4;  mov.b32 fb, %20; cvt.rn.bf16x2.f32 %4,  fa, fb;\n\t"
            "mov.b32 fa, %5;  mov.b32 fb, %21; cvt.rn.bf16x2.f32 %5,  fa, fb;\n\t"
            "mov.b32 fa, %6;  mov.b32 fb, %22; cvt.rn.bf16x2.f32 %6,  fa, fb;\n\t"
            "mov.b32 fa, %7;  mov.b32 fb, %23; cvt.rn.bf16x2.f32 %7,  fa, fb;\n\t"
            "mov.b32 fa, %8;  mov.b32 fb, %16; cvt.rn.bf16x2.f32 %8,  fa, fb;\n\t"
            "mov.b32 fa, %9;  mov.b32 fb, %17; cvt.rn.bf16x2.f32 %9,  fa, fb;\n\t"
            "mov.b32 fa, %10; mov.b32 fb, %18; cvt.rn.bf16x2.f32 %10, fa, fb;\n\t"
            "mov.b32 fa, %11; mov.b32 fb, %19; cvt.rn.bf16x2.f32 %11, fa, fb;\n\t"
            "mov.b32 fa, %12; mov.b32 fb, %20; cvt.rn.bf16x2.f32 %12, fa, fb;\n\t"
            "mov.b32 fa, %13; mov.b32 fb, %21; cvt.rn.bf16x2.f32 %13, fa, fb;\n\t"
            "mov.b32 fa, %14; mov.b32 fb, %22; cvt.rn.bf16x2.f32 %14, fa, fb;\n\t"
            "mov.b32 fa, %15; mov.b32 fb, %23; cvt.rn.bf16x2.f32 %15, fa, fb;\n\t"
            "}\n\t"
            : "+r"(s0), "+r"(s1), "+r"(s2), "+r"(s3),
              "+r"(s4), "+r"(s5), "+r"(s6), "+r"(s7),
              "+r"(s8), "+r"(s9), "+r"(s10), "+r"(s11),
              "+r"(s12), "+r"(s13), "+r"(s14), "+r"(s15)
            : "r"(g0), "r"(g1), "r"(g2), "r"(g3),
              "r"(g4), "r"(g5), "r"(g6), "r"(g7)
        );
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = r0+r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13+r14+r15+
                 s0+s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15;
    }
}

extern "C" __global__ void k10_hadd2_throughput(long long* out) {
    // Direct accumulation: UNIQUE increment per lane + individual volatile stores.
    // 16 "+r" accumulators + 16 "r" increments = 32 operands (max).
    // Different increments prevent constant-offset merging. Individual stores prevent sum merging.
    volatile unsigned* gin = (volatile unsigned*)out;
    unsigned i0=gin[0]|0x00010001,  i1=gin[1]|0x00020002,  i2=gin[2]|0x00030003,  i3=gin[3]|0x00040004;
    unsigned i4=gin[4]|0x00050005,  i5=gin[5]|0x00060006,  i6=gin[6]|0x00070007,  i7=gin[7]|0x00080008;
    unsigned i8=gin[8]|0x00090009,  i9=gin[9]|0x000a000a,  i10=gin[10]|0x000b000b, i11=gin[11]|0x000c000c;
    unsigned i12=gin[12]|0x000d000d, i13=gin[13]|0x000e000e, i14=gin[14]|0x000f000f, i15=gin[15]|0x00100010;
    unsigned r0=gin[16], r1=gin[17], r2=gin[18], r3=gin[19];
    unsigned r4=gin[20], r5=gin[21], r6=gin[22], r7=gin[23];
    unsigned r8=gin[24], r9=gin[25], r10=gin[26], r11=gin[27];
    unsigned r12=gin[28], r13=gin[29], r14=gin[30], r15=gin[31];

    for (int i = 0; i < WARMUP; i++)
        asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(r0) : "r"(i0));

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "add.rn.bf16x2 %0,  %0,  %16;\n\t"
            "add.rn.bf16x2 %1,  %1,  %17;\n\t"
            "add.rn.bf16x2 %2,  %2,  %18;\n\t"
            "add.rn.bf16x2 %3,  %3,  %19;\n\t"
            "add.rn.bf16x2 %4,  %4,  %20;\n\t"
            "add.rn.bf16x2 %5,  %5,  %21;\n\t"
            "add.rn.bf16x2 %6,  %6,  %22;\n\t"
            "add.rn.bf16x2 %7,  %7,  %23;\n\t"
            "add.rn.bf16x2 %8,  %8,  %24;\n\t"
            "add.rn.bf16x2 %9,  %9,  %25;\n\t"
            "add.rn.bf16x2 %10, %10, %26;\n\t"
            "add.rn.bf16x2 %11, %11, %27;\n\t"
            "add.rn.bf16x2 %12, %12, %28;\n\t"
            "add.rn.bf16x2 %13, %13, %29;\n\t"
            "add.rn.bf16x2 %14, %14, %30;\n\t"
            "add.rn.bf16x2 %15, %15, %31;\n\t"
            : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3),
              "+r"(r4), "+r"(r5), "+r"(r6), "+r"(r7),
              "+r"(r8), "+r"(r9), "+r"(r10), "+r"(r11),
              "+r"(r12), "+r"(r13), "+r"(r14), "+r"(r15)
            : "r"(i0), "r"(i1), "r"(i2), "r"(i3),
              "r"(i4), "r"(i5), "r"(i6), "r"(i7),
              "r"(i8), "r"(i9), "r"(i10), "r"(i11),
              "r"(i12), "r"(i13), "r"(i14), "r"(i15)
        );
    }
    long long t1 = clock64();

    // Store each accumulator individually — prevents ptxas from merging.
    volatile unsigned* vout = (volatile unsigned*)out;
    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        vout[2] = r0; vout[3] = r1; vout[4] = r2; vout[5] = r3;
        vout[6] = r4; vout[7] = r5; vout[8] = r6; vout[9] = r7;
        vout[10] = r8; vout[11] = r9; vout[12] = r10; vout[13] = r11;
        vout[14] = r12; vout[15] = r13; vout[16] = r14; vout[17] = r15;
    }
}

extern "C" __global__ void k11_hadd2_latency(long long* out) {
    volatile int* gin = (volatile int*)out;
    unsigned a = (unsigned)gin[threadIdx.x] | 0x3c003c00; // opaque to compiler
    unsigned inc = 0x00010001;

    for (int i = 0; i < WARMUP; i++) {
        asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(a) : "r"(inc));
    }

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "add.rn.bf16x2 %0, %0, %1;\n\t"
            "add.rn.bf16x2 %0, %0, %1;\n\t"
            "add.rn.bf16x2 %0, %0, %1;\n\t"
            "add.rn.bf16x2 %0, %0, %1;\n\t"
            "add.rn.bf16x2 %0, %0, %1;\n\t"
            "add.rn.bf16x2 %0, %0, %1;\n\t"
            "add.rn.bf16x2 %0, %0, %1;\n\t"
            "add.rn.bf16x2 %0, %0, %1;\n\t"
            "add.rn.bf16x2 %0, %0, %1;\n\t"
            "add.rn.bf16x2 %0, %0, %1;\n\t"
            "add.rn.bf16x2 %0, %0, %1;\n\t"
            "add.rn.bf16x2 %0, %0, %1;\n\t"
            "add.rn.bf16x2 %0, %0, %1;\n\t"
            "add.rn.bf16x2 %0, %0, %1;\n\t"
            "add.rn.bf16x2 %0, %0, %1;\n\t"
            "add.rn.bf16x2 %0, %0, %1;\n\t"
            : "+r"(a)
            : "r"(inc)
        );
    }
    long long t1 = clock64();

    out[0] = t1 - t0;
    out[1] = a;
}

extern "C" __global__ void k12_ldg_latency(long long* out) {
    const int* data = (const int*)(out + 64); // chase buffer (pre-zeroed)
    int idx = 0;

    for (int i = 0; i < WARMUP; i++) {
        asm volatile("ld.global.b32 %0, [%1];" : "=r"(idx) : "l"(data + idx));
    }

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile("ld.global.b32 %0, [%1];" : "=r"(idx) : "l"(data + idx));
        asm volatile("ld.global.b32 %0, [%1];" : "=r"(idx) : "l"(data + idx));
        asm volatile("ld.global.b32 %0, [%1];" : "=r"(idx) : "l"(data + idx));
        asm volatile("ld.global.b32 %0, [%1];" : "=r"(idx) : "l"(data + idx));
        asm volatile("ld.global.b32 %0, [%1];" : "=r"(idx) : "l"(data + idx));
        asm volatile("ld.global.b32 %0, [%1];" : "=r"(idx) : "l"(data + idx));
        asm volatile("ld.global.b32 %0, [%1];" : "=r"(idx) : "l"(data + idx));
        asm volatile("ld.global.b32 %0, [%1];" : "=r"(idx) : "l"(data + idx));
        asm volatile("ld.global.b32 %0, [%1];" : "=r"(idx) : "l"(data + idx));
        asm volatile("ld.global.b32 %0, [%1];" : "=r"(idx) : "l"(data + idx));
        asm volatile("ld.global.b32 %0, [%1];" : "=r"(idx) : "l"(data + idx));
        asm volatile("ld.global.b32 %0, [%1];" : "=r"(idx) : "l"(data + idx));
        asm volatile("ld.global.b32 %0, [%1];" : "=r"(idx) : "l"(data + idx));
        asm volatile("ld.global.b32 %0, [%1];" : "=r"(idx) : "l"(data + idx));
        asm volatile("ld.global.b32 %0, [%1];" : "=r"(idx) : "l"(data + idx));
        asm volatile("ld.global.b32 %0, [%1];" : "=r"(idx) : "l"(data + idx));
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = idx;
    }
}

extern "C" __global__ void k13a_imad_throughput(long long* out) {
    volatile int* gin = (volatile int*)out;
    int inc = gin[threadIdx.x];
    int mul = gin[threadIdx.x + 32];
    int a=inc+1, b=inc+2, c=inc+3, d=inc+4, e=inc+5, f=inc+6, g=inc+7, h=inc+8;
    int i0=inc+9, j=inc+10, k=inc+11, l=inc+12, m=inc+13, n=inc+14, o=inc+15, p=inc+16;

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "mad.lo.s32 %0, %0, %16, %17;\n\t"
            "mad.lo.s32 %1, %1, %16, %17;\n\t"
            "mad.lo.s32 %2, %2, %16, %17;\n\t"
            "mad.lo.s32 %3, %3, %16, %17;\n\t"
            "mad.lo.s32 %4, %4, %16, %17;\n\t"
            "mad.lo.s32 %5, %5, %16, %17;\n\t"
            "mad.lo.s32 %6, %6, %16, %17;\n\t"
            "mad.lo.s32 %7, %7, %16, %17;\n\t"
            "mad.lo.s32 %8, %8, %16, %17;\n\t"
            "mad.lo.s32 %9, %9, %16, %17;\n\t"
            "mad.lo.s32 %10, %10, %16, %17;\n\t"
            "mad.lo.s32 %11, %11, %16, %17;\n\t"
            "mad.lo.s32 %12, %12, %16, %17;\n\t"
            "mad.lo.s32 %13, %13, %16, %17;\n\t"
            "mad.lo.s32 %14, %14, %16, %17;\n\t"
            "mad.lo.s32 %15, %15, %16, %17;\n\t"
            : "+r"(a), "+r"(b), "+r"(c), "+r"(d),
              "+r"(e), "+r"(f), "+r"(g), "+r"(h),
              "+r"(i0), "+r"(j), "+r"(k), "+r"(l),
              "+r"(m), "+r"(n), "+r"(o), "+r"(p)
            : "r"(mul), "r"(inc)
        );
    }
    long long t1 = clock64();

    out[0] = t1 - t0;
    out[1] = a+b+c+d+e+f+g+h+i0+j+k+l+m+n+o+p;
}

extern "C" __global__ void k13b_imad_latency(long long* out) {
    volatile int* gin = (volatile int*)out;
    int a = gin[threadIdx.x];
    int mul = gin[threadIdx.x + 32];
    int add = gin[threadIdx.x + 64];

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "mad.lo.s32 %0, %0, %1, %2;\n\t"
            "mad.lo.s32 %0, %0, %1, %2;\n\t"
            "mad.lo.s32 %0, %0, %1, %2;\n\t"
            "mad.lo.s32 %0, %0, %1, %2;\n\t"
            "mad.lo.s32 %0, %0, %1, %2;\n\t"
            "mad.lo.s32 %0, %0, %1, %2;\n\t"
            "mad.lo.s32 %0, %0, %1, %2;\n\t"
            "mad.lo.s32 %0, %0, %1, %2;\n\t"
            "mad.lo.s32 %0, %0, %1, %2;\n\t"
            "mad.lo.s32 %0, %0, %1, %2;\n\t"
            "mad.lo.s32 %0, %0, %1, %2;\n\t"
            "mad.lo.s32 %0, %0, %1, %2;\n\t"
            "mad.lo.s32 %0, %0, %1, %2;\n\t"
            "mad.lo.s32 %0, %0, %1, %2;\n\t"
            "mad.lo.s32 %0, %0, %1, %2;\n\t"
            "mad.lo.s32 %0, %0, %1, %2;\n\t"
            : "+r"(a)
            : "r"(mul), "r"(add)
        );
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = a;
    }
}

extern "C" __global__ void k15a_lop3_throughput(long long* out) {
    volatile int* gin = (volatile int*)out;
    int inc = gin[threadIdx.x];
    int x = gin[threadIdx.x + 32];
    int a=inc+1, b=inc+2, c=inc+3, d=inc+4, e=inc+5, f=inc+6, g=inc+7, h=inc+8;
    int i0=inc+9, j=inc+10, k=inc+11, l=inc+12, m=inc+13, n=inc+14, o=inc+15, p=inc+16;

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "lop3.b32 %0, %0, %16, %17, 0x96;\n\t"
            "lop3.b32 %1, %1, %16, %17, 0x96;\n\t"
            "lop3.b32 %2, %2, %16, %17, 0x96;\n\t"
            "lop3.b32 %3, %3, %16, %17, 0x96;\n\t"
            "lop3.b32 %4, %4, %16, %17, 0x96;\n\t"
            "lop3.b32 %5, %5, %16, %17, 0x96;\n\t"
            "lop3.b32 %6, %6, %16, %17, 0x96;\n\t"
            "lop3.b32 %7, %7, %16, %17, 0x96;\n\t"
            "lop3.b32 %8, %8, %16, %17, 0x96;\n\t"
            "lop3.b32 %9, %9, %16, %17, 0x96;\n\t"
            "lop3.b32 %10, %10, %16, %17, 0x96;\n\t"
            "lop3.b32 %11, %11, %16, %17, 0x96;\n\t"
            "lop3.b32 %12, %12, %16, %17, 0x96;\n\t"
            "lop3.b32 %13, %13, %16, %17, 0x96;\n\t"
            "lop3.b32 %14, %14, %16, %17, 0x96;\n\t"
            "lop3.b32 %15, %15, %16, %17, 0x96;\n\t"
            : "+r"(a), "+r"(b), "+r"(c), "+r"(d),
              "+r"(e), "+r"(f), "+r"(g), "+r"(h),
              "+r"(i0), "+r"(j), "+r"(k), "+r"(l),
              "+r"(m), "+r"(n), "+r"(o), "+r"(p)
            : "r"(inc), "r"(x)
        );
    }
    long long t1 = clock64();

    out[0] = t1 - t0;
    out[1] = a+b+c+d+e+f+g+h+i0+j+k+l+m+n+o+p;
}

extern "C" __global__ void k15b_lop3_latency(long long* out) {
    volatile int* gin = (volatile int*)out;
    int a = gin[threadIdx.x];
    int x = gin[threadIdx.x + 32];
    int y = gin[threadIdx.x + 64];

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "lop3.b32 %0, %0, %1, %2, 0x96;\n\t"
            "lop3.b32 %0, %0, %1, %2, 0x96;\n\t"
            "lop3.b32 %0, %0, %1, %2, 0x96;\n\t"
            "lop3.b32 %0, %0, %1, %2, 0x96;\n\t"
            "lop3.b32 %0, %0, %1, %2, 0x96;\n\t"
            "lop3.b32 %0, %0, %1, %2, 0x96;\n\t"
            "lop3.b32 %0, %0, %1, %2, 0x96;\n\t"
            "lop3.b32 %0, %0, %1, %2, 0x96;\n\t"
            "lop3.b32 %0, %0, %1, %2, 0x96;\n\t"
            "lop3.b32 %0, %0, %1, %2, 0x96;\n\t"
            "lop3.b32 %0, %0, %1, %2, 0x96;\n\t"
            "lop3.b32 %0, %0, %1, %2, 0x96;\n\t"
            "lop3.b32 %0, %0, %1, %2, 0x96;\n\t"
            "lop3.b32 %0, %0, %1, %2, 0x96;\n\t"
            "lop3.b32 %0, %0, %1, %2, 0x96;\n\t"
            "lop3.b32 %0, %0, %1, %2, 0x96;\n\t"
            : "+r"(a)
            : "r"(x), "r"(y)
        );
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = a;
    }
}

extern "C" __global__ void k16a_shf_throughput(long long* out) {
    volatile int* gin = (volatile int*)out;
    int inc = gin[threadIdx.x];
    int shamt = gin[threadIdx.x + 32] & 31;
    int a=inc+1, b=inc+2, c=inc+3, d=inc+4, e=inc+5, f=inc+6, g=inc+7, h=inc+8;
    int i0=inc+9, j=inc+10, k=inc+11, l=inc+12, m=inc+13, n=inc+14, o=inc+15, p=inc+16;

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "shf.r.clamp.b32 %0, %0, %16, %0;\n\t"
            "shf.r.clamp.b32 %1, %1, %16, %1;\n\t"
            "shf.r.clamp.b32 %2, %2, %16, %2;\n\t"
            "shf.r.clamp.b32 %3, %3, %16, %3;\n\t"
            "shf.r.clamp.b32 %4, %4, %16, %4;\n\t"
            "shf.r.clamp.b32 %5, %5, %16, %5;\n\t"
            "shf.r.clamp.b32 %6, %6, %16, %6;\n\t"
            "shf.r.clamp.b32 %7, %7, %16, %7;\n\t"
            "shf.r.clamp.b32 %8, %8, %16, %8;\n\t"
            "shf.r.clamp.b32 %9, %9, %16, %9;\n\t"
            "shf.r.clamp.b32 %10, %10, %16, %10;\n\t"
            "shf.r.clamp.b32 %11, %11, %16, %11;\n\t"
            "shf.r.clamp.b32 %12, %12, %16, %12;\n\t"
            "shf.r.clamp.b32 %13, %13, %16, %13;\n\t"
            "shf.r.clamp.b32 %14, %14, %16, %14;\n\t"
            "shf.r.clamp.b32 %15, %15, %16, %15;\n\t"
            : "+r"(a), "+r"(b), "+r"(c), "+r"(d),
              "+r"(e), "+r"(f), "+r"(g), "+r"(h),
              "+r"(i0), "+r"(j), "+r"(k), "+r"(l),
              "+r"(m), "+r"(n), "+r"(o), "+r"(p)
            : "r"(shamt)
        );
    }
    long long t1 = clock64();

    out[0] = t1 - t0;
    out[1] = a+b+c+d+e+f+g+h+i0+j+k+l+m+n+o+p;
}

extern "C" __global__ void k16b_shf_latency(long long* out) {
    volatile int* gin = (volatile int*)out;
    int a = gin[threadIdx.x];
    int shamt = gin[threadIdx.x + 32] & 31;

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "shf.r.clamp.b32 %0, %0, %1, %0;\n\t"
            "shf.r.clamp.b32 %0, %0, %1, %0;\n\t"
            "shf.r.clamp.b32 %0, %0, %1, %0;\n\t"
            "shf.r.clamp.b32 %0, %0, %1, %0;\n\t"
            "shf.r.clamp.b32 %0, %0, %1, %0;\n\t"
            "shf.r.clamp.b32 %0, %0, %1, %0;\n\t"
            "shf.r.clamp.b32 %0, %0, %1, %0;\n\t"
            "shf.r.clamp.b32 %0, %0, %1, %0;\n\t"
            "shf.r.clamp.b32 %0, %0, %1, %0;\n\t"
            "shf.r.clamp.b32 %0, %0, %1, %0;\n\t"
            "shf.r.clamp.b32 %0, %0, %1, %0;\n\t"
            "shf.r.clamp.b32 %0, %0, %1, %0;\n\t"
            "shf.r.clamp.b32 %0, %0, %1, %0;\n\t"
            "shf.r.clamp.b32 %0, %0, %1, %0;\n\t"
            "shf.r.clamp.b32 %0, %0, %1, %0;\n\t"
            "shf.r.clamp.b32 %0, %0, %1, %0;\n\t"
            : "+r"(a)
            : "r"(shamt)
        );
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = a;
    }
}

extern "C" __global__ void k17a_mov_throughput(long long* out) {
    volatile int* gin = (volatile int*)out;
    int src = gin[threadIdx.x];
    int a, b, c, d, e, f, g, h, i0, j, k, l, m, n, o, p;

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "mov.b32 %0, %16;\n\t"
            "mov.b32 %1, %16;\n\t"
            "mov.b32 %2, %16;\n\t"
            "mov.b32 %3, %16;\n\t"
            "mov.b32 %4, %16;\n\t"
            "mov.b32 %5, %16;\n\t"
            "mov.b32 %6, %16;\n\t"
            "mov.b32 %7, %16;\n\t"
            "mov.b32 %8, %16;\n\t"
            "mov.b32 %9, %16;\n\t"
            "mov.b32 %10, %16;\n\t"
            "mov.b32 %11, %16;\n\t"
            "mov.b32 %12, %16;\n\t"
            "mov.b32 %13, %16;\n\t"
            "mov.b32 %14, %16;\n\t"
            "mov.b32 %15, %16;\n\t"
            : "=r"(a), "=r"(b), "=r"(c), "=r"(d),
              "=r"(e), "=r"(f), "=r"(g), "=r"(h),
              "=r"(i0), "=r"(j), "=r"(k), "=r"(l),
              "=r"(m), "=r"(n), "=r"(o), "=r"(p)
            : "r"(src)
        );
    }
    long long t1 = clock64();

    out[0] = t1 - t0;
    out[1] = a+b+c+d+e+f+g+h+i0+j+k+l+m+n+o+p;
}

extern "C" __global__ void k17b_mov_latency(long long* out) {
    // MOV + ADD interleave: pure MOV ping-pong is idempotent (a,b swap back).
    // ADD breaks symmetry (monotonic). 8 MOV + 8 ADD = 16 instrs.
    // Subtract ADD latency (K7b = VIADD latency) for pure MOV.
    volatile int* gin = (volatile int*)out;
    int a = gin[threadIdx.x];

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "{ .reg .s32 b;\n\t"
            "mov.b32 b, %0;\n\t"
            "add.s32 b, b, 1;\n\t"
            "mov.b32 %0, b;\n\t"
            "add.s32 %0, %0, 1;\n\t"
            "mov.b32 b, %0;\n\t"
            "add.s32 b, b, 1;\n\t"
            "mov.b32 %0, b;\n\t"
            "add.s32 %0, %0, 1;\n\t"
            "mov.b32 b, %0;\n\t"
            "add.s32 b, b, 1;\n\t"
            "mov.b32 %0, b;\n\t"
            "add.s32 %0, %0, 1;\n\t"
            "mov.b32 b, %0;\n\t"
            "add.s32 b, b, 1;\n\t"
            "mov.b32 %0, b;\n\t"
            "add.s32 %0, %0, 1;\n\t"
            "}\n\t"
            : "+r"(a)
        );
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = a;
    }
}

extern "C" __global__ void k19a_viadd_throughput(long long* out) {
    volatile int* gin = (volatile int*)out;
    int inc = gin[threadIdx.x];
    int a=inc+1, b=inc+2, c=inc+3, d=inc+4, e=inc+5, f=inc+6, g=inc+7, h=inc+8;
    int i0=inc+9, j=inc+10, k=inc+11, l=inc+12, m=inc+13, n=inc+14, o=inc+15, p=inc+16;

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "vadd.s32.s32.s32 %0, %0, %16;\n\t"
            "vadd.s32.s32.s32 %1, %1, %16;\n\t"
            "vadd.s32.s32.s32 %2, %2, %16;\n\t"
            "vadd.s32.s32.s32 %3, %3, %16;\n\t"
            "vadd.s32.s32.s32 %4, %4, %16;\n\t"
            "vadd.s32.s32.s32 %5, %5, %16;\n\t"
            "vadd.s32.s32.s32 %6, %6, %16;\n\t"
            "vadd.s32.s32.s32 %7, %7, %16;\n\t"
            "vadd.s32.s32.s32 %8, %8, %16;\n\t"
            "vadd.s32.s32.s32 %9, %9, %16;\n\t"
            "vadd.s32.s32.s32 %10, %10, %16;\n\t"
            "vadd.s32.s32.s32 %11, %11, %16;\n\t"
            "vadd.s32.s32.s32 %12, %12, %16;\n\t"
            "vadd.s32.s32.s32 %13, %13, %16;\n\t"
            "vadd.s32.s32.s32 %14, %14, %16;\n\t"
            "vadd.s32.s32.s32 %15, %15, %16;\n\t"
            : "+r"(a), "+r"(b), "+r"(c), "+r"(d),
              "+r"(e), "+r"(f), "+r"(g), "+r"(h),
              "+r"(i0), "+r"(j), "+r"(k), "+r"(l),
              "+r"(m), "+r"(n), "+r"(o), "+r"(p)
            : "r"(inc)
        );
    }
    long long t1 = clock64();

    out[0] = t1 - t0;
    out[1] = a+b+c+d+e+f+g+h+i0+j+k+l+m+n+o+p;
}

extern "C" __global__ void k19b_viadd_latency(long long* out) {
    volatile int* gin = (volatile int*)out;
    int a = gin[threadIdx.x];
    int inc = gin[threadIdx.x + 32];

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "vadd.s32.s32.s32 %0, %0, %1;\n\t"
            "vadd.s32.s32.s32 %0, %0, %1;\n\t"
            "vadd.s32.s32.s32 %0, %0, %1;\n\t"
            "vadd.s32.s32.s32 %0, %0, %1;\n\t"
            "vadd.s32.s32.s32 %0, %0, %1;\n\t"
            "vadd.s32.s32.s32 %0, %0, %1;\n\t"
            "vadd.s32.s32.s32 %0, %0, %1;\n\t"
            "vadd.s32.s32.s32 %0, %0, %1;\n\t"
            "vadd.s32.s32.s32 %0, %0, %1;\n\t"
            "vadd.s32.s32.s32 %0, %0, %1;\n\t"
            "vadd.s32.s32.s32 %0, %0, %1;\n\t"
            "vadd.s32.s32.s32 %0, %0, %1;\n\t"
            "vadd.s32.s32.s32 %0, %0, %1;\n\t"
            "vadd.s32.s32.s32 %0, %0, %1;\n\t"
            "vadd.s32.s32.s32 %0, %0, %1;\n\t"
            "vadd.s32.s32.s32 %0, %0, %1;\n\t"
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

extern "C" __global__ void k21a_isetp_throughput(long long* out) {
    // 16 independent ISETP feedback chains: accumulator → setp → p → selp → t → add → accum.
    // setp input varies per iteration (accumulator changes via add). add is monotonic.
    // Overhead: 16 SEL + 16 IADD per iter. instrs_per_iter = 16 (count ISETP only).
    volatile int* gin = (volatile int*)out;
    int sel_a = gin[threadIdx.x] | 1;     // selp "true" source (non-zero)
    int sel_b = gin[threadIdx.x + 32];    // selp "false" source
    int r0=gin[0], r1=gin[1], r2=gin[2], r3=gin[3];
    int r4=gin[4], r5=gin[5], r6=gin[6], r7=gin[7];
    int r8=gin[8], r9=gin[9], r10=gin[10], r11=gin[11];
    int r12=gin[12], r13=gin[13], r14=gin[14], r15=gin[15];

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "{ .reg .pred p; .reg .b32 t;\n\t"
            "setp.ne.s32 p, %0,  0; selp.b32 t, %16, %17, p; add.s32 %0,  %0,  t;\n\t"
            "setp.ne.s32 p, %1,  0; selp.b32 t, %16, %17, p; add.s32 %1,  %1,  t;\n\t"
            "setp.ne.s32 p, %2,  0; selp.b32 t, %16, %17, p; add.s32 %2,  %2,  t;\n\t"
            "setp.ne.s32 p, %3,  0; selp.b32 t, %16, %17, p; add.s32 %3,  %3,  t;\n\t"
            "setp.ne.s32 p, %4,  0; selp.b32 t, %16, %17, p; add.s32 %4,  %4,  t;\n\t"
            "setp.ne.s32 p, %5,  0; selp.b32 t, %16, %17, p; add.s32 %5,  %5,  t;\n\t"
            "setp.ne.s32 p, %6,  0; selp.b32 t, %16, %17, p; add.s32 %6,  %6,  t;\n\t"
            "setp.ne.s32 p, %7,  0; selp.b32 t, %16, %17, p; add.s32 %7,  %7,  t;\n\t"
            "setp.ne.s32 p, %8,  0; selp.b32 t, %16, %17, p; add.s32 %8,  %8,  t;\n\t"
            "setp.ne.s32 p, %9,  0; selp.b32 t, %16, %17, p; add.s32 %9,  %9,  t;\n\t"
            "setp.ne.s32 p, %10, 0; selp.b32 t, %16, %17, p; add.s32 %10, %10, t;\n\t"
            "setp.ne.s32 p, %11, 0; selp.b32 t, %16, %17, p; add.s32 %11, %11, t;\n\t"
            "setp.ne.s32 p, %12, 0; selp.b32 t, %16, %17, p; add.s32 %12, %12, t;\n\t"
            "setp.ne.s32 p, %13, 0; selp.b32 t, %16, %17, p; add.s32 %13, %13, t;\n\t"
            "setp.ne.s32 p, %14, 0; selp.b32 t, %16, %17, p; add.s32 %14, %14, t;\n\t"
            "setp.ne.s32 p, %15, 0; selp.b32 t, %16, %17, p; add.s32 %15, %15, t;\n\t"
            "}\n\t"
            : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3),
              "+r"(r4), "+r"(r5), "+r"(r6), "+r"(r7),
              "+r"(r8), "+r"(r9), "+r"(r10), "+r"(r11),
              "+r"(r12), "+r"(r13), "+r"(r14), "+r"(r15)
            : "r"(sel_a), "r"(sel_b)
        );
    }
    long long t1 = clock64();

    out[0] = t1 - t0;
    out[1] = r0+r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13+r14+r15;
}

extern "C" __global__ void k21b_isetp_selp_latency(long long* out) {
    volatile int* gin = (volatile int*)out;
    int a = gin[threadIdx.x];
    int sel_true = gin[threadIdx.x + 32] | 1;
    int sel_false = 0;

    asm volatile(".reg .pred kp;");

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "setp.ne.s32 kp, %0, 0;\n\t"
            "selp.b32 %0, %1, %2, kp;\n\t"
            "setp.ne.s32 kp, %0, 0;\n\t"
            "selp.b32 %0, %1, %2, kp;\n\t"
            "setp.ne.s32 kp, %0, 0;\n\t"
            "selp.b32 %0, %1, %2, kp;\n\t"
            "setp.ne.s32 kp, %0, 0;\n\t"
            "selp.b32 %0, %1, %2, kp;\n\t"
            "setp.ne.s32 kp, %0, 0;\n\t"
            "selp.b32 %0, %1, %2, kp;\n\t"
            "setp.ne.s32 kp, %0, 0;\n\t"
            "selp.b32 %0, %1, %2, kp;\n\t"
            "setp.ne.s32 kp, %0, 0;\n\t"
            "selp.b32 %0, %1, %2, kp;\n\t"
            "setp.ne.s32 kp, %0, 0;\n\t"
            "selp.b32 %0, %1, %2, kp;\n\t"
            : "+r"(a)
            : "r"(sel_true), "r"(sel_false)
        );
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = a;
    }
}

extern "C" __global__ void k20a_r2ur_throughput(long long* out) {
    // R2UR has no PTX equivalent — it's SASS-internal (ptxas inserts it for R→UR
    // domain crossing). %%ur<N> is not valid PTX. We use redux.sync.add.s32 as
    // the closest proxy: SASS is REDUX.SUM.S32 URx, Ry (R input → UR output).
    // This measures REDUX throughput (warp reduction + R→UR), NOT pure R2UR.
    volatile int* gin = (volatile int*)out;
    int inc = gin[threadIdx.x];
    int a=inc+1, b=inc+2, c=inc+3, d=inc+4, e=inc+5, f=inc+6, g=inc+7, h=inc+8;
    int i0=inc+9, j=inc+10, k=inc+11, l=inc+12, m=inc+13, n=inc+14, o=inc+15, p=inc+16;

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "redux.sync.add.s32 %0,  %0,  0xffffffff;\n\t"
            "redux.sync.add.s32 %1,  %1,  0xffffffff;\n\t"
            "redux.sync.add.s32 %2,  %2,  0xffffffff;\n\t"
            "redux.sync.add.s32 %3,  %3,  0xffffffff;\n\t"
            "redux.sync.add.s32 %4,  %4,  0xffffffff;\n\t"
            "redux.sync.add.s32 %5,  %5,  0xffffffff;\n\t"
            "redux.sync.add.s32 %6,  %6,  0xffffffff;\n\t"
            "redux.sync.add.s32 %7,  %7,  0xffffffff;\n\t"
            "redux.sync.add.s32 %8,  %8,  0xffffffff;\n\t"
            "redux.sync.add.s32 %9,  %9,  0xffffffff;\n\t"
            "redux.sync.add.s32 %10, %10, 0xffffffff;\n\t"
            "redux.sync.add.s32 %11, %11, 0xffffffff;\n\t"
            "redux.sync.add.s32 %12, %12, 0xffffffff;\n\t"
            "redux.sync.add.s32 %13, %13, 0xffffffff;\n\t"
            "redux.sync.add.s32 %14, %14, 0xffffffff;\n\t"
            "redux.sync.add.s32 %15, %15, 0xffffffff;\n\t"
            : "+r"(a), "+r"(b), "+r"(c), "+r"(d),
              "+r"(e), "+r"(f), "+r"(g), "+r"(h),
              "+r"(i0), "+r"(j), "+r"(k), "+r"(l),
              "+r"(m), "+r"(n), "+r"(o), "+r"(p)
        );
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = a+b+c+d+e+f+g+h+i0+j+k+l+m+n+o+p;
    }
}

extern "C" __global__ void k20b_r2ur_latency(long long* out) {
    // Cross-domain ping-pong: redux.sync (R→UR via REDUX.SUM) then compiler
    // inserts IMAD.U32 Rd, RZ, RZ, URx to read back UR→R. Each iteration =
    // 16 redux + 16 readback = 32 SASS instructions. Measures REDUX+readback latency.
    volatile int* gin = (volatile int*)out;
    int a = gin[threadIdx.x];

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "redux.sync.add.s32 %0, %0, 0xffffffff;\n\t"
            "redux.sync.add.s32 %0, %0, 0xffffffff;\n\t"
            "redux.sync.add.s32 %0, %0, 0xffffffff;\n\t"
            "redux.sync.add.s32 %0, %0, 0xffffffff;\n\t"
            "redux.sync.add.s32 %0, %0, 0xffffffff;\n\t"
            "redux.sync.add.s32 %0, %0, 0xffffffff;\n\t"
            "redux.sync.add.s32 %0, %0, 0xffffffff;\n\t"
            "redux.sync.add.s32 %0, %0, 0xffffffff;\n\t"
            "redux.sync.add.s32 %0, %0, 0xffffffff;\n\t"
            "redux.sync.add.s32 %0, %0, 0xffffffff;\n\t"
            "redux.sync.add.s32 %0, %0, 0xffffffff;\n\t"
            "redux.sync.add.s32 %0, %0, 0xffffffff;\n\t"
            "redux.sync.add.s32 %0, %0, 0xffffffff;\n\t"
            "redux.sync.add.s32 %0, %0, 0xffffffff;\n\t"
            "redux.sync.add.s32 %0, %0, 0xffffffff;\n\t"
            "redux.sync.add.s32 %0, %0, 0xffffffff;\n\t"
            : "+r"(a)
        );
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = a;
    }
}

extern "C" __global__ void k24_s2r_throughput(long long* out) {
    // Only 5 special regs produce S2R on SM100a: tid.x/y/z, laneid, warpid.
    // nwarpid/ctaid.y/ctaid.z → S2UR (uniform pipe). Duplicates get CSE'd by ptxas.
    int r0;
    for (int i = 0; i < WARMUP; i++)
        asm volatile("mov.u32 %0, %tid.x;" : "=r"(r0));

    int r1,r2,r3,r4;
    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "mov.u32 %0, %tid.x;\n\t"
            "mov.u32 %1, %tid.y;\n\t"
            "mov.u32 %2, %tid.z;\n\t"
            "mov.u32 %3, %laneid;\n\t"
            "mov.u32 %4, %warpid;\n\t"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4)
        );
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = r0+r1+r2+r3+r4;
    }
}

extern "C" __global__ void k25a_sel_throughput(long long* out) {
    // 16 independent SEL feedback chains: predicate set once (loop-invariant),
    // selp uses accumulator as src_true, constant as src_false. If p=true:
    // selp returns accumulator → add doubles it (exponential, never converges).
    // If p=false: selp returns constant → add is linear. ptxas can't determine p
    // at compile time (volatile load), so loop survives either way.
    // Overhead: 16 IADD per iter. instrs_per_iter = 16 (count SEL only).
    volatile int* gin = (volatile int*)out;
    int cond = gin[threadIdx.x];
    int a = gin[threadIdx.x + 32];
    int b = gin[threadIdx.x + 64];
    int r0=gin[0], r1=gin[1], r2=gin[2], r3=gin[3];
    int r4=gin[4], r5=gin[5], r6=gin[6], r7=gin[7];
    int r8=gin[8], r9=gin[9], r10=gin[10], r11=gin[11];
    int r12=gin[12], r13=gin[13], r14=gin[14], r15=gin[15];

    asm volatile(".reg .pred kp25;");
    asm volatile("setp.ne.s32 kp25, %0, 0;" :: "r"(cond));

    for (int i = 0; i < WARMUP; i++)
        asm volatile("{ .reg .b32 t; selp.b32 t, %0, %1, kp25; add.s32 %0, %0, t; }"
                     : "+r"(r0) : "r"(a));

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "{ .reg .b32 t;\n\t"
            "selp.b32 t, %0,  %16, kp25; add.s32 %0,  %0,  t;\n\t"
            "selp.b32 t, %1,  %17, kp25; add.s32 %1,  %1,  t;\n\t"
            "selp.b32 t, %2,  %16, kp25; add.s32 %2,  %2,  t;\n\t"
            "selp.b32 t, %3,  %17, kp25; add.s32 %3,  %3,  t;\n\t"
            "selp.b32 t, %4,  %16, kp25; add.s32 %4,  %4,  t;\n\t"
            "selp.b32 t, %5,  %17, kp25; add.s32 %5,  %5,  t;\n\t"
            "selp.b32 t, %6,  %16, kp25; add.s32 %6,  %6,  t;\n\t"
            "selp.b32 t, %7,  %17, kp25; add.s32 %7,  %7,  t;\n\t"
            "selp.b32 t, %8,  %16, kp25; add.s32 %8,  %8,  t;\n\t"
            "selp.b32 t, %9,  %17, kp25; add.s32 %9,  %9,  t;\n\t"
            "selp.b32 t, %10, %16, kp25; add.s32 %10, %10, t;\n\t"
            "selp.b32 t, %11, %17, kp25; add.s32 %11, %11, t;\n\t"
            "selp.b32 t, %12, %16, kp25; add.s32 %12, %12, t;\n\t"
            "selp.b32 t, %13, %17, kp25; add.s32 %13, %13, t;\n\t"
            "selp.b32 t, %14, %16, kp25; add.s32 %14, %14, t;\n\t"
            "selp.b32 t, %15, %17, kp25; add.s32 %15, %15, t;\n\t"
            "}\n\t"
            : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3),
              "+r"(r4), "+r"(r5), "+r"(r6), "+r"(r7),
              "+r"(r8), "+r"(r9), "+r"(r10), "+r"(r11),
              "+r"(r12), "+r"(r13), "+r"(r14), "+r"(r15)
            : "r"(a), "r"(b)
        );
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = r0+r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13+r14+r15;
    }
}

extern "C" __global__ void k25b_sel_latency(long long* out) {
    // Pure SEL data-path latency: output feeds back as data source.
    // Predicate set once per iteration from independent reg (not in dep chain).
    // 16 chained selp.b32, alternating src order to prevent constant-fold.
    volatile int* gin = (volatile int*)out;
    int a = gin[threadIdx.x];
    int b = gin[threadIdx.x + 32];
    int cond = gin[threadIdx.x + 64];

    for (int i = 0; i < WARMUP; i++)
        asm volatile(
            "{ .reg .pred %%p; setp.ne.s32 %%p, %1, 0; selp.b32 %0, %0, %2, %%p; }"
            : "+r"(a) : "r"(cond), "r"(b));

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "{ .reg .pred %%p;\n\t"
            "setp.ne.s32 %%p, %1, 0;\n\t"
            "selp.b32 %0, %0, %2, %%p;\n\t"
            "selp.b32 %0, %2, %0, %%p;\n\t"
            "selp.b32 %0, %0, %2, %%p;\n\t"
            "selp.b32 %0, %2, %0, %%p;\n\t"
            "selp.b32 %0, %0, %2, %%p;\n\t"
            "selp.b32 %0, %2, %0, %%p;\n\t"
            "selp.b32 %0, %0, %2, %%p;\n\t"
            "selp.b32 %0, %2, %0, %%p;\n\t"
            "selp.b32 %0, %0, %2, %%p;\n\t"
            "selp.b32 %0, %2, %0, %%p;\n\t"
            "selp.b32 %0, %0, %2, %%p;\n\t"
            "selp.b32 %0, %2, %0, %%p;\n\t"
            "selp.b32 %0, %0, %2, %%p;\n\t"
            "selp.b32 %0, %2, %0, %%p;\n\t"
            "selp.b32 %0, %0, %2, %%p;\n\t"
            "selp.b32 %0, %2, %0, %%p;\n\t"
            "}\n\t"
            : "+r"(a) : "r"(cond), "r"(b)
        );
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = a;
    }
}

extern "C" __global__ void k26a_iabs_throughput(long long* out) {
    // IABS is idempotent (abs(abs(x))=abs(x)), so pure abs loops collapse.
    // Interleave abs+add on each of 16 independent streams.
    // 16 IABS + 16 VIADD per iter. instrs_per_iter = 16 (count IABS only).
    // Subtract VIADD throughput overhead (K19a) for pure IABS.
    volatile int* gin = (volatile int*)out;
    int inc = gin[threadIdx.x];
    int a=inc+1, b=inc+2, c=inc+3, d=inc+4, e=inc+5, f=inc+6, g=inc+7, h=inc+8;
    int i0=inc+9, j=inc+10, k=inc+11, l=inc+12, m=inc+13, n=inc+14, o=inc+15, p=inc+16;

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "abs.s32 %0, %0; add.s32 %0, %0, 1;\n\t"
            "abs.s32 %1, %1; add.s32 %1, %1, 1;\n\t"
            "abs.s32 %2, %2; add.s32 %2, %2, 1;\n\t"
            "abs.s32 %3, %3; add.s32 %3, %3, 1;\n\t"
            "abs.s32 %4, %4; add.s32 %4, %4, 1;\n\t"
            "abs.s32 %5, %5; add.s32 %5, %5, 1;\n\t"
            "abs.s32 %6, %6; add.s32 %6, %6, 1;\n\t"
            "abs.s32 %7, %7; add.s32 %7, %7, 1;\n\t"
            "abs.s32 %8, %8; add.s32 %8, %8, 1;\n\t"
            "abs.s32 %9, %9; add.s32 %9, %9, 1;\n\t"
            "abs.s32 %10, %10; add.s32 %10, %10, 1;\n\t"
            "abs.s32 %11, %11; add.s32 %11, %11, 1;\n\t"
            "abs.s32 %12, %12; add.s32 %12, %12, 1;\n\t"
            "abs.s32 %13, %13; add.s32 %13, %13, 1;\n\t"
            "abs.s32 %14, %14; add.s32 %14, %14, 1;\n\t"
            "abs.s32 %15, %15; add.s32 %15, %15, 1;\n\t"
            : "+r"(a), "+r"(b), "+r"(c), "+r"(d),
              "+r"(e), "+r"(f), "+r"(g), "+r"(h),
              "+r"(i0), "+r"(j), "+r"(k), "+r"(l),
              "+r"(m), "+r"(n), "+r"(o), "+r"(p)
        );
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = a+b+c+d+e+f+g+h+i0+j+k+l+m+n+o+p;
    }
}

extern "C" __global__ void k26b_iabs_latency(long long* out) {
    // IABS + VIADD chain: pure abs chains are idempotent (ptxas eliminates them).
    // Interleave abs with +1 (monotonic, non-reducible). add.s32 → VIADD in SASS.
    // 8 pairs = 16 instructions. Subtract VIADD latency (K19b) for pure IABS.
    volatile int* gin = (volatile int*)out;
    int a = gin[threadIdx.x];

    for (int i = 0; i < WARMUP; i++) {
        asm volatile("abs.s32 %0, %0;" : "+r"(a));
        asm volatile("add.s32 %0, %0, 1;" : "+r"(a));
    }

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "abs.s32 %0, %0;\n\t"
            "add.s32 %0, %0, 1;\n\t"
            "abs.s32 %0, %0;\n\t"
            "add.s32 %0, %0, 1;\n\t"
            "abs.s32 %0, %0;\n\t"
            "add.s32 %0, %0, 1;\n\t"
            "abs.s32 %0, %0;\n\t"
            "add.s32 %0, %0, 1;\n\t"
            "abs.s32 %0, %0;\n\t"
            "add.s32 %0, %0, 1;\n\t"
            "abs.s32 %0, %0;\n\t"
            "add.s32 %0, %0, 1;\n\t"
            "abs.s32 %0, %0;\n\t"
            "add.s32 %0, %0, 1;\n\t"
            "abs.s32 %0, %0;\n\t"
            "add.s32 %0, %0, 1;\n\t"
            : "+r"(a)
        );
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = a;
    }
}

extern "C" __global__ void k22a_lds_throughput(long long* out) {
    // ld.volatile.shared prevents LICM — ptxas can't prove loaded value is loop-invariant.
    // XOR accumulation safe: ptxas can't prove period-2 when load result is "unknown".
    __shared__ int smem[1024];
    int tid = threadIdx.x;

    // Pre-fill: each thread writes 32 entries (1024 total)
    for (int k = 0; k < 32; k++)
        smem[tid + k * 32] = tid + k;
    __syncthreads();

    // Base address in SMEM for this thread (bank = tid, zero conflicts)
    uint32_t base = (uint32_t)(uint64_t)(&smem[tid]);

    volatile int* gin = (volatile int*)out;
    int r0=gin[0], r1=gin[1], r2=gin[2], r3=gin[3];
    int r4=gin[4], r5=gin[5], r6=gin[6], r7=gin[7];
    int r8=gin[8], r9=gin[9], r10=gin[10], r11=gin[11];
    int r12=gin[12], r13=gin[13], r14=gin[14], r15=gin[15];

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "{ .reg .s32 t;\n\t"
            "ld.volatile.shared.b32 t, [%16 + 0];    xor.b32 %0,  %0,  t;\n\t"
            "ld.volatile.shared.b32 t, [%16 + 128];  xor.b32 %1,  %1,  t;\n\t"
            "ld.volatile.shared.b32 t, [%16 + 256];  xor.b32 %2,  %2,  t;\n\t"
            "ld.volatile.shared.b32 t, [%16 + 384];  xor.b32 %3,  %3,  t;\n\t"
            "ld.volatile.shared.b32 t, [%16 + 512];  xor.b32 %4,  %4,  t;\n\t"
            "ld.volatile.shared.b32 t, [%16 + 640];  xor.b32 %5,  %5,  t;\n\t"
            "ld.volatile.shared.b32 t, [%16 + 768];  xor.b32 %6,  %6,  t;\n\t"
            "ld.volatile.shared.b32 t, [%16 + 896];  xor.b32 %7,  %7,  t;\n\t"
            "ld.volatile.shared.b32 t, [%16 + 1024]; xor.b32 %8,  %8,  t;\n\t"
            "ld.volatile.shared.b32 t, [%16 + 1152]; xor.b32 %9,  %9,  t;\n\t"
            "ld.volatile.shared.b32 t, [%16 + 1280]; xor.b32 %10, %10, t;\n\t"
            "ld.volatile.shared.b32 t, [%16 + 1408]; xor.b32 %11, %11, t;\n\t"
            "ld.volatile.shared.b32 t, [%16 + 1536]; xor.b32 %12, %12, t;\n\t"
            "ld.volatile.shared.b32 t, [%16 + 1664]; xor.b32 %13, %13, t;\n\t"
            "ld.volatile.shared.b32 t, [%16 + 1792]; xor.b32 %14, %14, t;\n\t"
            "ld.volatile.shared.b32 t, [%16 + 1920]; xor.b32 %15, %15, t;\n\t"
            "}\n\t"
            : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3),
              "+r"(r4), "+r"(r5), "+r"(r6), "+r"(r7),
              "+r"(r8), "+r"(r9), "+r"(r10), "+r"(r11),
              "+r"(r12), "+r"(r13), "+r"(r14), "+r"(r15)
            : "r"(base)
        );
    }
    long long t1 = clock64();

    out[0] = t1 - t0;
    out[1] = r0+r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13+r14+r15;
}

extern "C" __global__ void k22b_lds_latency(long long* out) {
    __shared__ int smem[1024];
    int tid = threadIdx.x;

    // Zero-fill SMEM
    smem[tid] = 0;
    __syncthreads();

    // Pointer chase: loaded value becomes next index, smem[0]=0 → always hits addr 0
    uint32_t base = (uint32_t)(uint64_t)(&smem[0]);

    int idx = 0;

    for (int i = 0; i < WARMUP; i++) {
        unsigned addr = base + idx * 4;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(idx) : "r"(addr));
    }

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        unsigned addr;
        asm volatile("mad.lo.u32 %0, %1, 4, %2;" : "=r"(addr) : "r"(idx), "r"(base));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(idx) : "r"(addr));
        asm volatile("mad.lo.u32 %0, %1, 4, %2;" : "=r"(addr) : "r"(idx), "r"(base));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(idx) : "r"(addr));
        asm volatile("mad.lo.u32 %0, %1, 4, %2;" : "=r"(addr) : "r"(idx), "r"(base));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(idx) : "r"(addr));
        asm volatile("mad.lo.u32 %0, %1, 4, %2;" : "=r"(addr) : "r"(idx), "r"(base));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(idx) : "r"(addr));
        asm volatile("mad.lo.u32 %0, %1, 4, %2;" : "=r"(addr) : "r"(idx), "r"(base));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(idx) : "r"(addr));
        asm volatile("mad.lo.u32 %0, %1, 4, %2;" : "=r"(addr) : "r"(idx), "r"(base));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(idx) : "r"(addr));
        asm volatile("mad.lo.u32 %0, %1, 4, %2;" : "=r"(addr) : "r"(idx), "r"(base));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(idx) : "r"(addr));
        asm volatile("mad.lo.u32 %0, %1, 4, %2;" : "=r"(addr) : "r"(idx), "r"(base));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(idx) : "r"(addr));
        asm volatile("mad.lo.u32 %0, %1, 4, %2;" : "=r"(addr) : "r"(idx), "r"(base));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(idx) : "r"(addr));
        asm volatile("mad.lo.u32 %0, %1, 4, %2;" : "=r"(addr) : "r"(idx), "r"(base));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(idx) : "r"(addr));
        asm volatile("mad.lo.u32 %0, %1, 4, %2;" : "=r"(addr) : "r"(idx), "r"(base));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(idx) : "r"(addr));
        asm volatile("mad.lo.u32 %0, %1, 4, %2;" : "=r"(addr) : "r"(idx), "r"(base));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(idx) : "r"(addr));
        asm volatile("mad.lo.u32 %0, %1, 4, %2;" : "=r"(addr) : "r"(idx), "r"(base));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(idx) : "r"(addr));
        asm volatile("mad.lo.u32 %0, %1, 4, %2;" : "=r"(addr) : "r"(idx), "r"(base));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(idx) : "r"(addr));
        asm volatile("mad.lo.u32 %0, %1, 4, %2;" : "=r"(addr) : "r"(idx), "r"(base));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(idx) : "r"(addr));
        asm volatile("mad.lo.u32 %0, %1, 4, %2;" : "=r"(addr) : "r"(idx), "r"(base));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(idx) : "r"(addr));
    }
    long long t1 = clock64();

    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = idx;
    }
}

extern "C" __global__ void k23_stg_throughput(long long* out) {
    int tid = threadIdx.x;
    // Store target: out + 128 ints (byte offset 512), past all timing/chase data
    char* base = (char*)(out + 128) + tid * 64;
    int val = tid + 1;

    long long t0 = clock64();
    for (int i = 0; i < REPS; i++) {
        asm volatile(
            "st.global.b32 [%16 + 0],  %0;\n\t"
            "st.global.b32 [%16 + 4],  %1;\n\t"
            "st.global.b32 [%16 + 8],  %2;\n\t"
            "st.global.b32 [%16 + 12], %3;\n\t"
            "st.global.b32 [%16 + 16], %4;\n\t"
            "st.global.b32 [%16 + 20], %5;\n\t"
            "st.global.b32 [%16 + 24], %6;\n\t"
            "st.global.b32 [%16 + 28], %7;\n\t"
            "st.global.b32 [%16 + 32], %8;\n\t"
            "st.global.b32 [%16 + 36], %9;\n\t"
            "st.global.b32 [%16 + 40], %10;\n\t"
            "st.global.b32 [%16 + 44], %11;\n\t"
            "st.global.b32 [%16 + 48], %12;\n\t"
            "st.global.b32 [%16 + 52], %13;\n\t"
            "st.global.b32 [%16 + 56], %14;\n\t"
            "st.global.b32 [%16 + 60], %15;\n\t"
            :: "r"(val), "r"(val+1), "r"(val+2), "r"(val+3),
               "r"(val+4), "r"(val+5), "r"(val+6), "r"(val+7),
               "r"(val+8), "r"(val+9), "r"(val+10), "r"(val+11),
               "r"(val+12), "r"(val+13), "r"(val+14), "r"(val+15),
               "l"(base)
            : "memory"
        );
    }
    long long t1 = clock64();

    out[0] = t1 - t0;
    // Read back one value to prevent full DCE of stores
    int readback;
    asm volatile("ld.global.b32 %0, [%1];" : "=r"(readback) : "l"(base));
    out[1] = readback;
}

struct Bench {
    const char* name;
    int instrs_per_iter;
    void (*fn)(long long*);
};

int main() {
    long long *d_out, h_out[2];
    cudaMalloc(&d_out, 8192);  // needs room for volatile loads + K23 STG target
    cudaMemset(d_out, 0, 8192);

    Bench benches[] = {
        {"K1: F2FP throughput (16 indep)",        16, k1_f2fp_throughput},
        {"K2: F2FP latency (dep chain)",           8, k2_f2fp_latency},
        {"K3: F2FP+STS conflict (interleaved)",   16, k3_f2fp_sts_conflict},
        {"K4: HFMA2 throughput (16 indep)",        16, k4_hfma2_throughput},
        {"K5: HFMA2+F2FP conflict (interleaved)", 16, k5_hfma2_f2fp_conflict},
        {"K6: STS.v4 throughput (16 indep)",       16, k6_sts_throughput},
        {"K7a: IADD independent (decoder)",        16, k7a_iadd_independent},
        {"K7b: IADD dependent (decoder)",          16, k7b_iadd_dependent},
        {"K8: PRMT throughput (16 feedback)",       16, k8_prmt_throughput},
        {"K9: F2FP throughput (32 indep)",         32, k9_f2fp_wide},
        {"K10: HADD2 throughput (8 after ptxas)",   8, k10_hadd2_throughput},
        {"K11: HADD2 latency (dep chain)",         16, k11_hadd2_latency},
        {"K12: LDG latency (pointer chase)",       16, k12_ldg_latency},
        {"K13a: IMAD throughput (16 indep)",       16, k13a_imad_throughput},
        {"K13b: IMAD latency (dep chain)",         16, k13b_imad_latency},
        {"K15a: LOP3 throughput (16 indep)",       16, k15a_lop3_throughput},
        {"K15b: LOP3 latency (dep chain)",         16, k15b_lop3_latency},
        {"K16a: SHF throughput (16 indep)",        16, k16a_shf_throughput},
        {"K16b: SHF latency (dep chain)",          16, k16b_shf_latency},
        {"K17a: MOV throughput (16 indep)",        16, k17a_mov_throughput},
        {"K17b: MOV+ADD latency (UNFIXABLE:1 IADD)", 1, k17b_mov_latency},
        {"K19a: VIADD throughput (16 indep)",      16, k19a_viadd_throughput},
        {"K19b: VIADD latency (dep chain)",        16, k19b_viadd_latency},
        {"K21a: ISETP throughput (16 indep)",      16, k21a_isetp_throughput},
        {"K21b: ISETP+SEL latency (dep chain)",   16, k21b_isetp_selp_latency},
        {"K20a: REDUX throughput (16 indep)",      16, k20a_r2ur_throughput},
        {"K20b: REDUX latency (dep chain)",       16, k20b_r2ur_latency},
        {"K24:  S2R throughput (5 indep)",          5, k24_s2r_throughput},
        {"K25a: SEL throughput (16 indep)",        16, k25a_sel_throughput},
        {"K25b: SEL latency (dep chain)",          16, k25b_sel_latency},
        {"K26a: IABS throughput (16 indep)",       16, k26a_iabs_throughput},
        {"K26b: IABS latency (dep chain)",         16, k26b_iabs_latency},
        {"K22a: LDS throughput (16 indep)",        16, k22a_lds_throughput},
        {"K22b: LDS latency (pointer chase)",      16, k22b_lds_latency},
        {"K23:  STG throughput (16 indep)",         16, k23_stg_throughput},
    };
    int n_benches = sizeof(benches) / sizeof(benches[0]);

    printf("SASS Calibration — SM 100a\n");
    printf("REPS=%d per kernel, 1 block, 1 warp (32 threads)\n\n", REPS);
    printf("%-45s %10s %10s %10s\n", "Kernel", "Cycles", "Cyc/instr", "Throughput");
    printf("%-45s %10s %10s %10s\n", "------", "------", "---------", "----------");

    for (int i = 0; i < n_benches; i++) {
        cudaMemset(d_out, 0, 2 * sizeof(long long));
        benches[i].fn<<<1, 32>>>(d_out);
        cudaDeviceSynchronize();
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
    printf("  K1  cyc/instr = F2FP throughput (feedback chain, zero LOP3 overhead)\n");
    printf("  K2  cyc/instr = F2FP chain latency (mov.b32 bitcast likely SASS NOP)\n");
    printf("  K3  vs K1+K6: if K3 ≈ max(K1,K6) → different pipes; if K3 ≈ K1+K6 → same pipe\n");
    printf("  K4  cyc/instr = HFMA2 throughput (volatile accumulators, 16 independent)\n");
    printf("  K5  vs K1+K4: HFMA2 self-accum + F2FP feedback chains interleaved\n");
    printf("  K7a vs K7b: stall count = K7b_cyc/instr - K7a_cyc/instr (verify bits [3:0])\n");
    printf("  K8  cyc/instr = PRMT throughput (16 feedback chains, 8 control words)\n");
    printf("  K9  vs K1: check if throughput degrades at 32-wide (register file pressure)\n");
    printf("  K10 cyc/instr = HADD2 throughput (ptxas reduces 16→8, still valid)\n");
    printf("  K11 cyc/instr = HADD2 dep chain latency (pure, no MOV in chain)\n");
    printf("  K12 cyc/instr = LDG L1-hit latency (includes addr calc overhead)\n");
    printf("  K13-K16,K19: throughput = independent issue rate, latency = dep chain\n");
    printf("  K17a: MOV throughput — UNFIXABLE (copy propagation eliminates all MOV)\n");
    printf("  K17b cyc/instr = UNFIXABLE (ptxas collapses MOV+ADD to 1 IADD3)\n");
    printf("  K20a cyc/instr = REDUX throughput (redux.sync R→UR, proxy for R2UR)\n");
    printf("  K20b cyc/instr = REDUX dep chain (includes IMAD readback UR→R)\n");
    printf("    NOTE: R2UR has no PTX equivalent. REDUX is the closest R→UR proxy.\n");
    printf("    Pure R2UR latency must be calibrated from SASS control word stalls.\n");
    printf("  K21a cyc/instr = ISETP throughput (setp+selp+add feedback, count SETP only)\n");
    printf("  K21b cyc/instr = avg of ISETP+SEL (interleaved setp/selp chain)\n");
    printf("  K22a cyc/instr = LDS throughput (volatile loads + XOR, prevents LICM)\n");
    printf("  K22b cyc/instr = LDS+IMAD chain latency (subtract IMAD for pure LDS)\n");
    printf("  K23  cyc/instr = STG global store throughput\n");
    printf("  K24  cyc/instr = S2R throughput — UNFIXABLE (loop-invariant special reg reads)\n");
    printf("  K25a cyc/instr = SEL throughput (selp+add feedback, pred set once)\n");
    printf("  K25b cyc/instr = SEL data-path latency (output→input chain, pred independent)\n");
    printf("  K26a cyc/instr = IABS+VIADD throughput (16 indep streams, abs+add interleave)\n");
    printf("  K26b cyc/instr = IABS+VIADD chain latency (subtract VIADD latency from K19b)\n");

    cudaFree(d_out);
    return 0;
}
