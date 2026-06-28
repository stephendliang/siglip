/*
epilogue_ops.cuh — shared epilogue math + host helper for fc1_w3x / fc2_w3x

CVT_ADD_BF16X2:        bf16x2 pack + bias-add (fc2_w3x BIAS_ONLY default).
gelu_approx / gelu_fwd: device + host fp32 GELU (tanh.approx.f32 form).
CVT_GELU_ADD_BF16X2:   fused fp32-precision bias-add → GELU → bf16x2 pack
                       (fc1_w3x default fused-GELU path).
pack_idx_C:            host-side index into the 4D packed-tile C buffer
                       (depends on TM_PACK / TILES_N / TN from the includer).

Pure header — every device fn / macro is __device__ __forceinline__,
fully expanded at every call site. Including this header costs zero
SASS vs the prior in-line copy (verified via cuobjdump diff).

Include AFTER the geometry header (TM_PACK / TILES_N / TN must be defined).
*/
#pragma once

#include <cuda_bf16.h>
#include <cstddef>

/*
  CVT_ADD_BF16X2 — native bf16 epilogue math.
  Pack the two fp32 accumulator lanes into one bf16x2 register (F2FP.PACK),
  then add the packed bf16x2 bias in one HADD2. Output is bf16x2 ready
  for the STS.

  Collapses { 2x FADD fp32 + 1x cvt.bf16x2.f32 } per pair into
  { 1x F2FP.PACK + 1x HADD2 }. The acc rounds to bf16 before the add,
  so precision differs from rank-1's fp32-add-then-convert pattern.
  Output-valid to 2% rel for sensible acc scales; SigLIP2 FC2 accs
  stay well within bf16's representable add range.

  In fc1_w3x this macro is kept for STRIP_EPILOGUE / baseline reference;
  the production fused path uses CVT_GELU_ADD_BF16X2 below.
*/
#define CVT_ADD_BF16X2(p_out, a_lo, a_hi, b_in) \
    asm volatile( \
        "{\n\t" \
        ".reg .b32 ap;\n\t" \
        "cvt.rn.bf16x2.f32 ap, %2, %1;\n\t" \
        "add.rn.bf16x2 %0, ap, %3;\n\t" \
        "}" \
        : "=r"(p_out) \
        : "f"(a_lo), "f"(a_hi), "r"(b_in))

/*
  GELU approximation, fp32. tanh(0.7979 * (x + 0.04472 * x^3)) form,
  rearranged to GELU(x) = 0.5 * (x + x * tanh(x * (0.7979 + 0.03568 * x^2))).
  8 fp32 ops/element: FFMA, FMUL, MUFU.TANH, FMUL, FMUL, FADD, FMUL, FADD.
  MUFU.TANH is the hardware-pipelined tanh.approx.f32 instruction.
*/
static __device__ __forceinline__ float gelu_approx(float acc, float bias_f32) {
    float x = acc + bias_f32;
    float inner = x * (0.7978845608f + 0.035677408136f * x * x);
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(inner));
    return 0.5f * (x + x * t);
}

/* Host-side reference for golden validation. */
static __host__ __forceinline__ float gelu_fwd(float x) {
    const float k = 0.7978845608f;
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

/*
  CVT_GELU_ADD_BF16X2 — fused-GELU pack+CVT.
  Unpacks the bf16x2 bias to two fp32, adds bias to the fp32 accs,
  applies GELU in fp32 (preserves precision past the bf16 round of
  CVT_ADD_BF16X2), then packs the two GELU'd values to bf16x2 ready
  for STSM. (bf16 IS the upper 16 bits of an fp32 mantissa, so the
  shift+mask reinterpret is bit-exact.)
*/
#define CVT_GELU_ADD_BF16X2(p_out, a_lo, a_hi, b_in) do {                  \
    const float _b_lo = __uint_as_float((uint32_t)(b_in) << 16);           \
    const float _b_hi = __uint_as_float((uint32_t)(b_in) & 0xFFFF0000u);   \
    const float _g_lo = gelu_approx((a_lo), _b_lo);                        \
    const float _g_hi = gelu_approx((a_hi), _b_hi);                        \
    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;"                           \
        : "=r"(p_out) : "f"(_g_lo), "f"(_g_hi));                           \
} while (0)

/*
  Host-side index into the packed-tile C buffer for absolute (m, n).
  Mirrors the 4D TMA descriptor; used by verify and any tooling that
  reads C back to host as a flat blob. Requires TM_PACK / TILES_N / TN
  visible at include time (i.e. the geometry header was already included).
*/
static inline size_t pack_idx_C(long long m, int n) {
    long long tm_idx = m / TM_PACK;
    int       mc     = (int)(m % TM_PACK);
    int       tn_idx = n / TN;
    int       nc     = n % TN;
    return ((size_t)tm_idx * TILES_N + (size_t)tn_idx)
         * (size_t)(TM_PACK * TN)
         + (size_t)mc * TN + (size_t)nc;
}
