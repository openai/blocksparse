#include "gpu_types.h"

#define UNARY_VEC2(op, res, a) \
    res.x = op(a.x); \
    res.y = op(a.y);

#define BINARY_VEC2(op, res, a, b) \
    res.x = op(a.x, b.x); \
    res.y = op(a.y, b.y);

#define BINARY_VEC2_S(op, res, a, b) \
    res.x = op(a.x, b); \
    res.y = op(a.y, b);

#define TERNARY_VEC2(op, res, a, b, c) \
    res.x = op(a.x, b.x, c.x); \
    res.y = op(a.y, b.y, c.y);

#define TERNARY_VEC2_S(op, res, a, b, c) \
    res.x = op(a.x, b.x, c); \
    res.y = op(a.y, b.y, c);


#define UNARY_VEC4(op, res, a) \
    res.x = op(a.x); \
    res.y = op(a.y); \
    res.z = op(a.z); \
    res.w = op(a.w)

#define BINARY_VEC4(op, res, a, b) \
    res.x = op(a.x, b.x); \
    res.y = op(a.y, b.y); \
    res.z = op(a.z, b.z); \
    res.w = op(a.w, b.w)

#define BINARY_VEC4_S(op, res, a, b) \
    res.x = op(a.x, b); \
    res.y = op(a.y, b); \
    res.z = op(a.z, b); \
    res.w = op(a.w, b)

#define TERNARY_VEC4(op, res, a, b, c) \
    res.x = op(a.x, b.x, c.x); \
    res.y = op(a.y, b.y, c.y); \
    res.z = op(a.z, b.z, c.z); \
    res.w = op(a.w, b.w, c.w)

#define TERNARY_VEC4_S(op, res, a, b, c) \
    res.x = op(a.x, b.x, c); \
    res.y = op(a.y, b.y, c); \
    res.z = op(a.z, b.z, c); \
    res.w = op(a.w, b.w, c)

__device__ __forceinline__ float  to_float(float  v) { return v; }
__device__ __forceinline__ float2 to_float(float2 v) { return v; }
__device__ __forceinline__ float4 to_float(float4 v) { return v; }
__device__ __forceinline__ float8 to_float(float8 v) { return v; }

__device__ __forceinline__ float round8(float v)
{
    // extract sign/exponent, scale it with an ulp, add to val and truncate (mask)
    float r;
    asm("{                                       \n\t"
        ".reg .f32 exp, val;                     \n\t"
        "and.b32 exp, %1, 0xff800000;            \n\t"
        "fma.rz.ftz.f32 val, exp, 0F3b000000, %1;\n\t"  // ulp: "0F%08x" % ((127 - 8 - 1) << 23)
        "and.b32 %0, val, 0xffff8000;            \n\t"
        "}" : "=f"(r) : "f"(v));
    return r;
}
__device__ __forceinline__ float round7(float v)
{
    // extract sign/exponent, scale it with an ulp, add to val and truncate (mask)
    float r;
    asm("{                                       \n\t"
        ".reg .f32 exp, val;                     \n\t"
        "and.b32 exp, %1, 0xff800000;            \n\t"
        "fma.rz.ftz.f32 val, exp, 0F3b800000, %1;\n\t"  // ulp: "0F%08x" % ((127 - 7 - 1) << 23)
        "and.b32 %0, val, 0xffff0000;            \n\t"
        "}" : "=f"(r) : "f"(v));
    return r;
}
__device__ __forceinline__ float round6(float v)
{
    // extract sign/exponent, scale it with an ulp, add to val and truncate (mask)
    float r;
    asm("{                                       \n\t"
        ".reg .f32 exp, val;                     \n\t"
        "and.b32 exp, %1, 0xff800000;            \n\t"
        "fma.rz.ftz.f32 val, exp, 0F3c000000, %1;\n\t"  // ulp: "0F%08x" % ((127 - 6 - 1) << 23)
        "and.b32 %0, val, 0xfffe0000;            \n\t"
        "}" : "=f"(r) : "f"(v));
    return r;
}
__device__ __forceinline__ float round5(float v)
{
    // extract sign/exponent, scale it with an ulp, add to val and truncate (mask)
    float r;
    asm("{                                       \n\t"
        ".reg .f32 exp, val;                     \n\t"
        "and.b32 exp, %1, 0xff800000;            \n\t"
        "fma.rz.ftz.f32 val, exp, 0F3c800000, %1;\n\t"  // ulp: "0F%08x" % ((127 - 5 - 1) << 23)
        "and.b32 %0, val, 0xfffc0000;            \n\t"
        "}" : "=f"(r) : "f"(v));
    return r;
}
__device__ __forceinline__ float round4(float v)
{
    // extract sign/exponent, scale it with an ulp, add to val and truncate (mask)
    float r;
    asm("{                                       \n\t"
        ".reg .f32 exp, val;                     \n\t"
        "and.b32 exp, %1, 0xff800000;            \n\t"
        "fma.rz.ftz.f32 val, exp, 0F3d000000, %1;\n\t"  // ulp: "0F%08x" % ((127 - 4 - 1) << 23)
        "and.b32 %0, val, 0xfff80000;            \n\t"
        "}" : "=f"(r) : "f"(v));

    r = fmaxf(r, -powf(2.f,  4.f));
    r = fminf(r,  powf(2.f,  4.f));
    r = fabs(r) < powf(2.f, -4.f) ? 0.0f : r;
    return r;
}
__device__ __forceinline__ float round4w(float v)
{
    // extract sign/exponent, scale it with an ulp, add to val and truncate (mask)
    float r;
    asm("{                                       \n\t"
        ".reg .f32 exp, val;                     \n\t"
        "and.b32 exp, %1, 0xff800000;            \n\t"
        "fma.rz.ftz.f32 val, exp, 0F3d000000, %1;\n\t"  // ulp: "0F%08x" % ((127 - 4 - 1) << 23)
        "and.b32 %0, val, 0xfff80000;            \n\t"
        "}" : "=f"(r) : "f"(v));

    r = fmaxf(r, -powf(2.f,  1.f));
    r = fminf(r,  powf(2.f,  1.f));
    r = fabs(r) < powf(2.f, -7.f) ? 0.0f : r;
    return r;
}
__device__ __forceinline__ float round3w(float v)
{
    // extract sign/exponent, scale it with an ulp, add to val and truncate (mask)
    float r;
    asm("{                                       \n\t"
        ".reg .f32 exp, val;                     \n\t"
        "and.b32 exp, %1, 0xff800000;            \n\t"
        "fma.rz.ftz.f32 val, exp, 0F3d800000, %1;\n\t"  // ulp: "0F%08x" % ((127 - 3 - 1) << 23)
        "and.b32 %0, val, 0xfff00000;            \n\t"
        "}" : "=f"(r) : "f"(v));

    r = fmaxf(r, -powf(2.f,  1.f));
    r = fminf(r,  powf(2.f,  1.f));
    r = fabs(r) < powf(2.f, -15.f) ? 0.0f : r;
    return r;
}
__device__ __forceinline__ float round3f(float v)
{
    // extract sign/exponent, scale it with an ulp, add to val and truncate (mask)
    float r;
    asm("{                                       \n\t"
        ".reg .f32 exp, val;                     \n\t"
        "and.b32 exp, %1, 0xff800000;            \n\t"
        "fma.rz.ftz.f32 val, exp, 0F3d800000, %1;\n\t"  // ulp: "0F%08x" % ((127 - 3 - 1) << 23)
        "and.b32 %0, val, 0xfff00000;            \n\t"
        "}" : "=f"(r) : "f"(v));

    r = fmaxf(r, -powf(2.f,  10.f));
    r = fminf(r,  powf(2.f,  10.f));
    r = fabs(r) < powf(2.f, -6.f) ? 0.0f : r;
    return r;
}
__device__ __forceinline__ float round3(float v)
{
    // extract sign/exponent, scale it with an ulp, add to val and truncate (mask)
    float r;
    asm("{                                       \n\t"
        ".reg .f32 exp, val;                     \n\t"
        "and.b32 exp, %1, 0xff800000;            \n\t"
        "fma.rz.ftz.f32 val, exp, 0F3d800000, %1;\n\t"  // ulp: "0F%08x" % ((127 - 3 - 1) << 23)
        "and.b32 %0, val, 0xfff00000;            \n\t"
        "}" : "=f"(r) : "f"(v));
    return r;
}
__device__ __forceinline__ float round2g(float v)
{
    // extract sign/exponent, scale it with an ulp, add to val and truncate (mask)
    float r;
    asm("{                                       \n\t"
        ".reg .f32 exp, val;                     \n\t"
        "and.b32 exp, %1, 0xff800000;            \n\t"
        "fma.rz.ftz.f32 val, exp, 0F3e000000, %1;\n\t"  // ulp: "0F%08x" % ((127 - 2 - 1) << 23)
        "and.b32 %0, val, 0xffe00000;            \n\t"
        "}" : "=f"(r) : "f"(v));

    r = fmaxf(r, -powf(2.f,  1.f));
    r = fminf(r,  powf(2.f,  1.f));
    r = fabs(r) < powf(2.f, -31.f) ? 0.0f : r;
    return r;
}
__device__ __forceinline__ float round1(float v)
{
    // extract sign/exponent, scale it with an ulp, add to val and truncate (mask)
    float r;
    asm("{                                       \n\t"
        ".reg .f32 exp, val;                     \n\t"
        "and.b32 exp, %1, 0xff800000;            \n\t"
        "fma.rz.ftz.f32 val, exp, 0F3e800000, %1;\n\t"  // ulp: "0F%08x" % ((127 - 1 - 1) << 23)
        "and.b32 %0, val, 0xffc00000;            \n\t"
        "}" : "=f"(r) : "f"(v));

    r = fmaxf(r, -powf(2.f,  20.f));
    r = fminf(r,  powf(2.f,  20.f));
    r = fabs(r) < powf(2.f, -44.f) ? 0.0f : r;
    return r;
}

// __device__ __forceinline__ float quant(float x)
// {
//     x = fmaxf(x, -16777216.0f);
//     x = fminf(x,  16777216.0f);
//     x = fabs(x) < 3.637978807091713e-12f ? 0.0f : x;
//     return x;
// }

// bfe exp,     uint8, 1, 6;
// bfe sign,    uint8, 7, 1;
// shl float32, exp,     23;           // exponent
// add float32, float32, 0x29800000;   // exponent bias shift (add 83<<23)
// bfi float32, float32, uint8, 22, 1; // mantissa
// bfi float32, float32, sign,  31, 1; // sign


// and exp,   float32, 0xff800000;
// fma float32, exp, 0F3e800000, float32; // round
// bfe sign,  float32, 31, 1;
// bfe exp,   float32, 23, 8;
// bfe uint8, float32, 22, 1;    // mantissa
// sub exp, exp, 83;             // exponent bias shift
// bfi uint8, uint8, exp,  1, 6; // exponent
// bfi uint8, uint8, sign, 7, 1; // sign

// __device__ __forceinline__ float trunc8(float v) { asm("and.b32 %0, %0, 0xffff8000;" : "+f"(v) : ); return v; }
// __device__ __forceinline__ float trunc7(float v) { asm("and.b32 %0, %0, 0xffff0000;" : "+f"(v) : ); return v; }
// __device__ __forceinline__ float trunc6(float v) { asm("and.b32 %0, %0, 0xfffe0000;" : "+f"(v) : ); return v; }
// __device__ __forceinline__ float trunc5(float v) { asm("and.b32 %0, %0, 0xfffc0000;" : "+f"(v) : ); return v; }
// __device__ __forceinline__ float trunc4(float v) { asm("and.b32 %0, %0, 0xfff80000;" : "+f"(v) : ); return v; }
// __device__ __forceinline__ float trunc3(float v) { asm("and.b32 %0, %0, 0xfff00000;" : "+f"(v) : ); return v; }


// __device__ __forceinline__ float2 round8(float2 v) { UNARY_VEC2(round8, v, v); return v; }
// __device__ __forceinline__ float2 round7(float2 v) { UNARY_VEC2(round7, v, v); return v; }
// __device__ __forceinline__ float2 round6(float2 v) { UNARY_VEC2(round6, v, v); return v; }
// __device__ __forceinline__ float2 round5(float2 v) { UNARY_VEC2(round5, v, v); return v; }
// __device__ __forceinline__ float2 round4(float2 v) { UNARY_VEC2(round4, v, v); return v; }
// __device__ __forceinline__ float2 round3(float2 v) { UNARY_VEC2(round3, v, v); return v; }
// __device__ __forceinline__ float2 round2(float2 v) { UNARY_VEC2(round2, v, v); return v; }
// __device__ __forceinline__ float2 round1(float2 v) { UNARY_VEC2(round1, v, v); return v; }

__device__ __forceinline__ float2 round4w(float2 v) { UNARY_VEC2(round4w, v, v); return v; }
__device__ __forceinline__ float4 round4w(float4 v) { UNARY_VEC4(round4w, v, v); return v; }

__device__ __forceinline__ float2 round3w(float2 v) { UNARY_VEC2(round3w, v, v); return v; }
__device__ __forceinline__ float4 round3w(float4 v) { UNARY_VEC4(round3w, v, v); return v; }

__device__ __forceinline__ float2 round3f(float2 v) { UNARY_VEC2(round3f, v, v); return v; }
__device__ __forceinline__ float4 round3f(float4 v) { UNARY_VEC4(round3f, v, v); return v; }

__device__ __forceinline__ float2 round2g(float2 v) { UNARY_VEC2(round2g, v, v); return v; }
__device__ __forceinline__ float4 round2g(float4 v) { UNARY_VEC4(round2g, v, v); return v; }

__device__ __forceinline__ float2 round1(float2 v) { UNARY_VEC2(round1, v, v); return v; }
__device__ __forceinline__ float4 round1(float4 v) { UNARY_VEC4(round1, v, v); return v; }

__device__ __forceinline__ float2 round3(float2 v) { UNARY_VEC2(round3, v, v); return v; }
__device__ __forceinline__ float4 round3(float4 v) { UNARY_VEC4(round3, v, v); return v; }

// __device__ __forceinline__ float4 round8(float4 v) { UNARY_VEC4(round8, v, v); return v; }
// __device__ __forceinline__ float4 round7(float4 v) { UNARY_VEC4(round7, v, v); return v; }
// __device__ __forceinline__ float4 round6(float4 v) { UNARY_VEC4(round6, v, v); return v; }
// __device__ __forceinline__ float4 round5(float4 v) { UNARY_VEC4(round5, v, v); return v; }
// __device__ __forceinline__ float4 round4(float4 v) { UNARY_VEC4(round4, v, v); return v; }
// __device__ __forceinline__ float4 round3(float4 v) { UNARY_VEC4(round3, v, v); return v; }
// __device__ __forceinline__ float4 round2(float4 v) { UNARY_VEC4(round2, v, v); return v; }
// __device__ __forceinline__ float4 round1(float4 v) { UNARY_VEC4(round1, v, v); return v; }

// __device__ __forceinline__ float2 trunc8(float2 v) { UNARY_VEC2(trunc8, v, v); return v; }
// __device__ __forceinline__ float2 trunc7(float2 v) { UNARY_VEC2(trunc7, v, v); return v; }
// __device__ __forceinline__ float2 trunc6(float2 v) { UNARY_VEC2(trunc6, v, v); return v; }
// __device__ __forceinline__ float2 trunc5(float2 v) { UNARY_VEC2(trunc5, v, v); return v; }
// __device__ __forceinline__ float2 trunc4(float2 v) { UNARY_VEC2(trunc4, v, v); return v; }
// __device__ __forceinline__ float2 trunc3(float2 v) { UNARY_VEC2(trunc3, v, v); return v; }

// __device__ __forceinline__ float4 trunc8(float4 v) { UNARY_VEC4(trunc8, v, v); return v; }
// __device__ __forceinline__ float4 trunc7(float4 v) { UNARY_VEC4(trunc7, v, v); return v; }
// __device__ __forceinline__ float4 trunc6(float4 v) { UNARY_VEC4(trunc6, v, v); return v; }
// __device__ __forceinline__ float4 trunc5(float4 v) { UNARY_VEC4(trunc5, v, v); return v; }
// __device__ __forceinline__ float4 trunc4(float4 v) { UNARY_VEC4(trunc4, v, v); return v; }
// __device__ __forceinline__ float4 trunc3(float4 v) { UNARY_VEC4(trunc3, v, v); return v; }


__device__ __forceinline__ float  to_float(ehalf  v)
{
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(v.x));
    return r;
}
__device__ __forceinline__ float2 to_float(ehalf2 v)
{
    float2 r;
    asm("{\n\t"
        ".reg .f16 a, b;\n\t"
        "mov.b32 {a, b}, %2;\n\t"
        "cvt.f32.f16 %0, a;\n\t"
        "cvt.f32.f16 %1, b;\n\t"
        "}" : "=f"(r.x),"=f"(r.y) : "r"(v.x));
    return r;
}
__device__ __forceinline__ float4 to_float(ehalf4 v)
{
    float4 r;
    asm("{\n\t"
        ".reg .f16 a, b, c, d;\n\t"
        "mov.b32 {a, b}, %4;\n\t"
        "mov.b32 {c, d}, %5;\n\t"
        "cvt.f32.f16 %0, a;\n\t"
        "cvt.f32.f16 %1, b;\n\t"
        "cvt.f32.f16 %2, c;\n\t"
        "cvt.f32.f16 %3, d;\n\t"
        "}" : "=f"(r.x),"=f"(r.y),"=f"(r.z),"=f"(r.w) : "r"(v.x),"r"(v.y));
    return r;
}
__device__ __forceinline__ float8 to_float(ehalf8 v)
{
    float8 r;
    asm("{\n\t"
        ".reg .f16 v<8>;\n\t"
        "mov.b32 {v0, v1}, %8;\n\t"
        "mov.b32 {v2, v3}, %9;\n\t"
        "mov.b32 {v4, v5}, %10;\n\t"
        "mov.b32 {v6, v7}, %11;\n\t"
        "cvt.f32.f16 %0, v0;\n\t"
        "cvt.f32.f16 %1, v1;\n\t"
        "cvt.f32.f16 %2, v2;\n\t"
        "cvt.f32.f16 %3, v3;\n\t"
        "cvt.f32.f16 %4, v4;\n\t"
        "cvt.f32.f16 %5, v5;\n\t"
        "cvt.f32.f16 %6, v6;\n\t"
        "cvt.f32.f16 %7, v7;\n\t"
        "}" : "=f"(r.a.x),"=f"(r.a.y),"=f"(r.a.z),"=f"(r.a.w),
              "=f"(r.b.x),"=f"(r.b.y),"=f"(r.b.z),"=f"(r.b.w)
            :  "r"(v.x),   "r"(v.y),   "r"(v.z),   "r"(v.w));
    return r;
}

__device__ __forceinline__ ehalf  to_ehalf(float  v)
{
    ehalf r;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r.x) : "f"(v));
    return r;
}

__device__ __forceinline__ ehalf2 to_ehalf(float2 v)
{
    ehalf2 r;
    asm("{\n\t"
        ".reg .f16 a, b;\n\t"
        "cvt.rn.f16.f32 a, %1;\n\t"
        "cvt.rn.f16.f32 b, %2;\n\t"
        "mov.b32 %0, {a, b};\n\t"
        "}" : "=r"(r.x) : "f"(v.x),"f"(v.y));
    return r;
}
__device__ __forceinline__ ehalf4 to_ehalf(float4 v)
{
    ehalf4 r;
    asm("{\n\t"
        ".reg .f16 a, b, c, d;\n\t"
        "cvt.rn.f16.f32 a, %2;\n\t"
        "cvt.rn.f16.f32 b, %3;\n\t"
        "cvt.rn.f16.f32 c, %4;\n\t"
        "cvt.rn.f16.f32 d, %5;\n\t"
        "mov.b32 %0, {a, b};\n\t"
        "mov.b32 %1, {c, d};\n\t"
        "}" : "=r"(r.x),"=r"(r.y) : "f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w));
    return r;
}
__device__ __forceinline__ ehalf8 to_ehalf(float8 v)
{
    ehalf8 r;
    asm("{\n\t"
        ".reg .f16 v<8>;\n\t"
        "cvt.rn.f16.f32 v0, %4 ;\n\t"
        "cvt.rn.f16.f32 v1, %5 ;\n\t"
        "cvt.rn.f16.f32 v2, %6 ;\n\t"
        "cvt.rn.f16.f32 v3, %7 ;\n\t"
        "cvt.rn.f16.f32 v4, %8 ;\n\t"
        "cvt.rn.f16.f32 v5, %9 ;\n\t"
        "cvt.rn.f16.f32 v6, %10;\n\t"
        "cvt.rn.f16.f32 v7, %11;\n\t"
        "mov.b32 %0, {v0, v1};\n\t"
        "mov.b32 %1, {v2, v3};\n\t"
        "mov.b32 %2, {v4, v5};\n\t"
        "mov.b32 %3, {v6, v7};\n\t"
        "}" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
            :  "f"(v.a.x),"f"(v.a.y),"f"(v.a.z),"f"(v.a.w),
               "f"(v.b.x),"f"(v.b.y),"f"(v.b.z),"f"(v.b.w));
    return r;
}



__device__ __forceinline__ float  to_float(bhalf  v)
{
    float r;
    asm("mov.b32 %0, {0, %1};" : "=f"(r) : "h"(v.x));
    return r;
}
__device__ __forceinline__ float2 to_float(bhalf2 v)
{
    float2 r;
    asm("{\n\t"
        ".reg .u16 u0, u1;\n\t"
        "mov.b32 {u0, u1}, %2;\n\t" // force XMAD.PSL.CLO instead of SHL
        "mov.b32 %0, {0, u0};\n\t"
        "and.b32 %1, %2, 0xffff0000;\n\t"
        "}" : "=f"(r.x),"=f"(r.y) : "r"(v.x));
    return r;
}
__device__ __forceinline__ float4 to_float(bhalf4 v)
{
    float4 r;
    asm("{\n\t"
        ".reg .u16 u0, u1;\n\t"
        "mov.b32 {u0, u1}, %4;\n\t" // force XMAD.PSL.CLO instead of SHL
        "mov.b32 %0, {0, u0};\n\t"
        "and.b32 %1, %4, 0xffff0000;\n\t"
        "mov.b32 {u0, u1}, %5;\n\t"
        "mov.b32 %2, {0, u0};\n\t"
        "and.b32 %3, %5, 0xffff0000;\n\t"
        "}" : "=f"(r.x),"=f"(r.y),"=f"(r.z),"=f"(r.w) : "r"(v.x),"r"(v.y));
    return r;
}
__device__ __forceinline__ float8 to_float(bhalf8 v)
{
    float8 r;
    asm("{\n\t"
        ".reg .u16 v<8>;\n\t"
        "mov.b32 {v0, v1}, %8;\n\t"
        "mov.b32 {v2, v3}, %9;\n\t"
        "mov.b32 {v4, v5}, %10;\n\t"
        "mov.b32 {v6, v7}, %11;\n\t"
        "mov.b32 %0, {0, v0};\n\t"
        "mov.b32 %2, {0, v2};\n\t"
        "mov.b32 %4, {0, v4};\n\t"
        "mov.b32 %6, {0, v6};\n\t"
        "and.b32 %1, %8,  0xffff0000;\n\t"
        "and.b32 %3, %9,  0xffff0000;\n\t"
        "and.b32 %5, %10, 0xffff0000;\n\t"
        "and.b32 %7, %11, 0xffff0000;\n\t"
        "}" : "=f"(r.a.x),"=f"(r.a.y),"=f"(r.a.z),"=f"(r.a.w),
              "=f"(r.b.x),"=f"(r.b.y),"=f"(r.b.z),"=f"(r.b.w)
            :  "r"(v.x),   "r"(v.y),   "r"(v.z),   "r"(v.w));
    return r;
}

// __device__ __forceinline__ float quant(float x)
// {
//     x = fmaxf(x, -16777216.0f);
//     x = fminf(x,  16777216.0f);
//     x = fabs(x) < 3.637978807091713e-12f ? 0.0f : x;
//     return x;
// }


__device__ __forceinline__ bhalf  to_bhalf(float  v)
{
    //v = quant(v);
    bhalf r;
    asm("{\n\t"
        ".reg .f32 f32, exp;\n\t"
        ".reg .u32 u32;\n\t"
        ".reg .u16 u16;\n\t"
        "and.b32 exp, %1, 0xff800000;\n\t"
        "fma.rz.ftz.f32 f32, exp, 0F3b800000, %1;\n\t"  // "0F%08x" % ((127 - 7 - 1) << 23)
        "shr.b32 u32, f32, 16;\n\t"
        "mov.b32 { %0, u16 }, u32;\n\t"
        "}" : "=h"(r.x) : "f"(v));
    return r;
}
__device__ __forceinline__ bhalf2 to_bhalf(float2 v)
{
    //UNARY_VEC2(quant, v, v);
    bhalf2 r;
    asm("{\n\t"
        ".reg .f32 exp0, exp1, f0, f1;\n\t"
        ".reg .u32 u0, u1;\n\t"
        "and.b32 exp0, %1, 0xff800000;\n\t"
        "and.b32 exp1, %2, 0xff800000;\n\t"
        "fma.rz.ftz.f32 f0, exp0, 0F3b800000, %1;\n\t"  // "0F%08x" % ((127 - 7 - 1) << 23)
        "fma.rz.ftz.f32 f1, exp1, 0F3b800000, %2;\n\t"
        "mov.b32 u0, f0;\n\t"
        "mov.b32 u1, f1;\n\t"
        "vadd.u32.u32.u32 %0.h0, u0.h1, 0, u1;\n\t" // use 16 bit merge functionality of vadd
        "}" : "=r"(r.x) : "f"(v.x),"f"(v.y));
    return r;
}
__device__ __forceinline__ bhalf4 to_bhalf(float4 v)
{
    //UNARY_VEC4(quant, v, v);
    bhalf4 r;
    asm("{\n\t"
        ".reg .f32 exp<4>, f<4>;\n\t"
        ".reg .u32 u<4>;\n\t"
        "and.b32 exp0, %2, 0xff800000;\n\t"
        "and.b32 exp1, %3, 0xff800000;\n\t"
        "and.b32 exp2, %4, 0xff800000;\n\t"
        "and.b32 exp3, %5, 0xff800000;\n\t"
        "fma.rz.ftz.f32 f0, exp0, 0F3b800000, %2;\n\t"  // "0F%08x" % ((127 - 7 - 1) << 23)
        "fma.rz.ftz.f32 f1, exp1, 0F3b800000, %3;\n\t"
        "fma.rz.ftz.f32 f2, exp2, 0F3b800000, %4;\n\t"
        "fma.rz.ftz.f32 f3, exp3, 0F3b800000, %5;\n\t"
        "mov.b32 u0, f0;\n\t"
        "mov.b32 u1, f1;\n\t"
        "mov.b32 u2, f2;\n\t"
        "mov.b32 u3, f3;\n\t"
        "vadd.u32.u32.u32 %0.h0, u0.h1, 0, u1;\n\t" // use 16 bit merge functionality of vadd
        "vadd.u32.u32.u32 %1.h0, u2.h1, 0, u3;\n\t"
        "}" : "=r"(r.x),"=r"(r.y) : "f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w));
    return r;
}
__device__ __forceinline__ bhalf8 to_bhalf(float8 v)
{
    //UNARY_VEC4(quant, v.a, v.a);
    //UNARY_VEC4(quant, v.b, v.b);
    bhalf8 r;
    asm("{\n\t"
        ".reg .f32 exp<8>, f<8>;\n\t"
        ".reg .u32 u<8>;\n\t"
        "and.b32 exp0, %4 , 0xff800000;\n\t"
        "and.b32 exp1, %5 , 0xff800000;\n\t"
        "and.b32 exp2, %6 , 0xff800000;\n\t"
        "and.b32 exp3, %7 , 0xff800000;\n\t"
        "and.b32 exp4, %8 , 0xff800000;\n\t"
        "and.b32 exp5, %9 , 0xff800000;\n\t"
        "and.b32 exp6, %10, 0xff800000;\n\t"
        "and.b32 exp7, %11, 0xff800000;\n\t"
        "fma.rz.ftz.f32 f0, exp0, 0F3b800000, %4 ;\n\t"  // "0F%08x" % ((127 - 7 - 1) << 23)
        "fma.rz.ftz.f32 f1, exp1, 0F3b800000, %5 ;\n\t"
        "fma.rz.ftz.f32 f2, exp2, 0F3b800000, %6 ;\n\t"
        "fma.rz.ftz.f32 f3, exp3, 0F3b800000, %7 ;\n\t"
        "fma.rz.ftz.f32 f4, exp4, 0F3b800000, %8 ;\n\t"
        "fma.rz.ftz.f32 f5, exp5, 0F3b800000, %9 ;\n\t"
        "fma.rz.ftz.f32 f6, exp6, 0F3b800000, %10;\n\t"
        "fma.rz.ftz.f32 f7, exp7, 0F3b800000, %11;\n\t"
        "mov.b32 u0, f0;\n\t"
        "mov.b32 u1, f1;\n\t"
        "mov.b32 u2, f2;\n\t"
        "mov.b32 u3, f3;\n\t"
        "mov.b32 u4, f4;\n\t"
        "mov.b32 u5, f5;\n\t"
        "mov.b32 u6, f6;\n\t"
        "mov.b32 u7, f7;\n\t"
        "vadd.u32.u32.u32 %0.h0, u0.h1, 0, u1;\n\t" // use 16 bit merge functionality of vadd
        "vadd.u32.u32.u32 %1.h0, u2.h1, 0, u3;\n\t"
        "vadd.u32.u32.u32 %2.h0, u4.h1, 0, u5;\n\t"
        "vadd.u32.u32.u32 %3.h0, u6.h1, 0, u7;\n\t"
        "}" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
            :  "f"(v.a.x),"f"(v.a.y),"f"(v.a.z),"f"(v.a.w),
               "f"(v.b.x),"f"(v.b.y),"f"(v.b.z),"f"(v.b.w));
    return r;
}

__device__ __forceinline__ ehalf __ldg(const ehalf *ptr)
{
    ehalf ret;
    asm volatile ("ld.global.nc.u16 %0, [%1];" : "=h"(ret.x) : "l"(ptr));
    return ret;
}
__device__ __forceinline__ bhalf __ldg(const bhalf *ptr)
{
    bhalf ret;
    asm volatile ("ld.global.nc.u16 %0, [%1];" : "=h"(ret.x) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ ehalf2 __ldg(const ehalf2 *ptr)
{
    ehalf2 ret;
    asm volatile ("ld.global.nc.u32 %0, [%1];" : "=r"(ret.x) : "l"(ptr));
    return ret;
}
__device__ __forceinline__ bhalf2 __ldg(const bhalf2 *ptr)
{
    bhalf2 ret;
    asm volatile ("ld.global.nc.u32 %0, [%1];" : "=r"(ret.x) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ ehalf4 __ldg(const ehalf4 *ptr)
{
    ehalf4 ret;
    asm volatile ("ld.global.nc.v2.u32 {%0, %1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l"(ptr));
    return ret;
}
__device__ __forceinline__ bhalf4 __ldg(const bhalf4 *ptr)
{
    bhalf4 ret;
    asm volatile ("ld.global.nc.v2.u32 {%0, %1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ ehalf8 __ldg(const ehalf8 *ptr)
{
    ehalf8 ret;
    asm volatile ("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
    return ret;
}
__device__ __forceinline__ bhalf8 __ldg(const bhalf8 *ptr)
{
    bhalf8 ret;
    asm volatile ("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
    return ret;
}
__device__ __forceinline__ float8 __ldg(const float8 *ptr)
{
    float8 ret;  // not used.
    return ret;
}


__device__ __forceinline__ void __stg(const ehalf *ptr, ehalf val)
{
    asm volatile ("st.global.wb.u16 [%0], %1;" :: "l"(ptr), "h"(val.x)  );
}
__device__ __forceinline__ void __stg(const ehalf2 *ptr, ehalf2 val)
{
    asm volatile ("st.global.wb.u32 [%0], %1;" :: "l"(ptr), "r"(val.x)  );
}
__device__ __forceinline__ void __stg(const ehalf4 *ptr, ehalf4 val)
{
    asm volatile ("st.global.wb.v2.u32 [%0], {%1, %2};" :: "l"(ptr), "r"(val.x), "r"(val.y)  );
}
__device__ __forceinline__ void __stg(const ehalf8 *ptr, ehalf8 val)
{
    asm volatile ("st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)  );
}
__device__ __forceinline__ void __stg(const bhalf *ptr, bhalf val)
{
    asm volatile ("st.global.wb.u16 [%0], %1;" :: "l"(ptr), "h"(val.x)  );
}
__device__ __forceinline__ void __stg(const bhalf2 *ptr, bhalf2 val)
{
    asm volatile ("st.global.wb.u32 [%0], %1;" :: "l"(ptr), "r"(val.x)  );
}
__device__ __forceinline__ void __stg(const bhalf4 *ptr, bhalf4 val)
{
    asm volatile ("st.global.wb.v2.u32 [%0], {%1, %2};" :: "l"(ptr), "r"(val.x), "r"(val.y)  );
}
__device__ __forceinline__ void __stg(const bhalf8 *ptr, bhalf8 val)
{
    asm volatile ("st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)  );
}
__device__ __forceinline__ void __stg(const float *ptr, float val)
{
    asm volatile ("st.global.wb.f32 [%0], %1;" :: "l"(ptr), "f"(val)  );
}
__device__ __forceinline__ void __stg(const float2 *ptr, float2 val)
{
    asm volatile ("st.global.wb.v2.f32 [%0], {%1, %2};" :: "l"(ptr), "f"(val.x), "f"(val.y)  );
}
__device__ __forceinline__ void __stg(const float4 *ptr, float4 val)
{
    asm volatile ("st.global.wb.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(ptr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w)  );
}


__device__ __forceinline__ void ld_shared_v1(int a, float* v)
{
    asm volatile ("ld.shared.f32 %0, [%1];"  : "=f"(v[0]) : "r"(a));
}

__device__ __forceinline__  void st_shared_v1(int a, float v)
{
    asm volatile ("st.shared.f32 [%0], %1;" :: "r"(a), "f"(v) );
}

__device__ __forceinline__  void st_shared_v1(int a, float* v)
{
    asm volatile ("st.shared.f32 [%0], %1;" :: "r"(a), "f"(v[0]) );
}

__device__ __forceinline__ void ld_shared_v2(int a, float* v)
{
    asm volatile ("ld.shared.v2.f32 {%0, %1}, [%2];"  : "=f"(v[0]),"=f"(v[1]) : "r"(a));
}

__device__ __forceinline__  void st_shared_v2(int a, float2 v)
{
    asm volatile ("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(a), "f"(v.x),"f"(v.y) );
}

__device__ __forceinline__  void st_shared_v2(int a, float* v)
{
    asm volatile ("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(a), "f"(v[0]),"f"(v[1]) );
}

__device__ __forceinline__ void ld_shared_v4(int a, float* v)
{
    asm volatile ("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"  : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]) : "r"(a));
}

__device__ __forceinline__  void st_shared_v4(int a, float4 v)
{
    asm volatile ("st.shared.v4.f32 [%0], {%1, %2, %3, %4};" :: "r"(a), "f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w) );
}

__device__ __forceinline__  void st_shared_v4(int a, float* v)
{
    asm volatile ("st.shared.v4.f32 [%0], {%1, %2, %3, %4};" :: "r"(a), "f"(v[0]),"f"(v[1]),"f"(v[2]),"f"(v[3]) );
}

__device__ __forceinline__ void ld_shared(int a, float* v, int x)
{
         if (x == 4) ld_shared_v4(a, v);
    else if (x == 2) ld_shared_v2(a, v);
    else             ld_shared_v1(a, v);
}

__device__ __forceinline__ void st_shared(int a, float* v, int x)
{
         if (x == 4) st_shared_v4(a, v);
    else if (x == 2) st_shared_v2(a, v);
    else             st_shared_v1(a, v);
}

__device__ __forceinline__ void atomicRed(float *ptr, float val)
{
    asm volatile ("red.global.gpu.add.f32 [%0], %1;" :: "l"(ptr), "f"(val)  );
}

__device__ __forceinline__ float shfl_xor(float var, int laneMask)
{
    float ret;
    asm volatile ("shfl.bfly.b32 %0, %1, %2, %3;" : "=f"(ret) : "f"(var), "r"(laneMask), "r"(0x1f));
    return ret;
}

// Fast low precision integer division helper.
// Adds an ULP to avoid rounding errors
__device__ __forceinline__ int div_blkX(int bxy, float rcp_blkX)
{
    int by;
    asm("{\n\t"
        ".reg .f32 bxy, by;\n\t"
        "cvt.rn.f32.s32 bxy, %1;\n\t"
        "mul.f32 by, bxy, %2;\n\t"
        "fma.rn.f32 by, by, 0F33800000, by;\n\t"
        "cvt.rzi.s32.f32 %0, by;\n\t"
        "}" : "=r"(by) : "r"(bxy), "f"(rcp_blkX));
    return by;
}
__device__ __forceinline__ int mod_blkX(int bxy, int blkX, int by)
{
    int bx;
    asm("vmad.s32.u32.u32 %0, -%1.h0, %2.h0, %3;" : "=r"(bx) : "r"(by), "r"(blkX), "r"(bxy));
    return bx;
}

#define ADD_PTR(T) \
__device__ __forceinline__ const T* add_ptr(const T* src, int offset) \
{                                                                     \
    const T* dst;                                                     \
    asm("{\n\t"                                                       \
        ".reg .u64 offset;\n\t"                                       \
        "mov.b64 offset, {%2,0};\n\t"                                 \
        "add.u64 %0, %1, offset;\n\t"                                 \
        "}" : "=l"(dst) : "l"(src), "r"(offset));                     \
    return dst;                                                       \
}

#define ADD_PTR_S(T,V) \
__device__ __forceinline__ const T* add_ptr_s(const T* src, int offset)     \
{                                                                           \
    const T* dst;                                                           \
    asm("{                       \n\t"                                      \
        ".reg .u32 lo,hi,cc,of;  \n\t"                                      \
        "mul.lo.u32 of, %2, %3;  \n\t"                                      \
        "shr.s32    cc, %2, 30;  \n\t"                                      \
        "mov.b64    {lo,hi}, %1; \n\t"                                      \
        "add.cc.u32  lo,lo,  of; \n\t"                                      \
        "addc.u32    hi,hi,  cc; \n\t"                                      \
        "mov.b64 %0, {lo,hi};    \n\t"                                      \
        "}" : "=l"(dst) : "l"(src), "r"(offset), "r"((int)sizeof(*src)/V)); \
    return dst;                                                             \
}                                                                           \
__device__ __forceinline__ T* add_ptr_s(T* src, int offset)                 \
{                                                                           \
    T* dst;                                                                 \
    asm("{                       \n\t"                                      \
        ".reg .u32 lo,hi,cc,of;  \n\t"                                      \
        "mul.lo.u32 of, %2, %3;  \n\t"                                      \
        "shr.s32    cc, %2, 30;  \n\t"                                      \
        "mov.b64    {lo,hi}, %1; \n\t"                                      \
        "add.cc.u32  lo,lo,  of; \n\t"                                      \
        "addc.u32    hi,hi,  cc; \n\t"                                      \
        "mov.b64 %0, {lo,hi};    \n\t"                                      \
        "}" : "=l"(dst) : "l"(src), "r"(offset), "r"((int)sizeof(*src)/V)); \
    return dst;                                                             \
}

#define ADD_PTR_U(T,V) \
__device__ __forceinline__ const T* add_ptr_u(const T* src, int offset)      \
{                                                                            \
    const T* dst;                                                            \
    asm("{                       \n\t"                                       \
        ".reg .u32 lo,hi,of;     \n\t"                                       \
        "mul.lo.u32 of, %2, %3;  \n\t"                                       \
        "mov.b64    {lo,hi}, %1; \n\t"                                       \
        "add.cc.u32  lo,lo,  of; \n\t"                                       \
        "addc.u32    hi,hi,  0;  \n\t"                                       \
        "mov.b64 %0, {lo,hi};    \n\t"                                       \
        "}" : "=l"(dst) : "l"(src), "r"(offset), "r"((int)sizeof(*src)/V));  \
    return dst;                                                              \
}                                                                            \
__device__ __forceinline__ T* add_ptr_u(T* src, int offset)                  \
{                                                                            \
    T* dst;                                                                  \
    asm("{                       \n\t"                                       \
        ".reg .u32 lo,hi,of;     \n\t"                                       \
        "mul.lo.u32 of, %2, %3;  \n\t"                                       \
        "mov.b64    {lo,hi}, %1; \n\t"                                       \
        "add.cc.u32  lo,lo,  of; \n\t"                                       \
        "addc.u32    hi,hi,  0;  \n\t"                                       \
        "mov.b64 %0, {lo,hi};    \n\t"                                       \
        "}" : "=l"(dst) : "l"(src), "r"(offset), "r"((int)sizeof(*src)/V));  \
    return dst;                                                              \
}

ADD_PTR(ehalf )
ADD_PTR(ehalf2)
ADD_PTR(ehalf4)
ADD_PTR(ehalf8)

ADD_PTR(bhalf )
ADD_PTR(bhalf2)
ADD_PTR(bhalf4)
ADD_PTR(bhalf8)

ADD_PTR(float )
ADD_PTR(float2)
ADD_PTR(float4)
ADD_PTR(float8)

ADD_PTR_S(ehalf ,1)
ADD_PTR_S(ehalf2,2)
ADD_PTR_S(ehalf4,4)
ADD_PTR_S(ehalf8,8)

ADD_PTR_S(bhalf ,1)
ADD_PTR_S(bhalf2,2)
ADD_PTR_S(bhalf4,4)
ADD_PTR_S(bhalf8,8)

ADD_PTR_S(float ,1)
ADD_PTR_S(float2,2)
ADD_PTR_S(float4,4)
ADD_PTR_S(float8,8)


ADD_PTR_U(ehalf ,1)
ADD_PTR_U(ehalf2,2)
ADD_PTR_U(ehalf4,4)
ADD_PTR_U(ehalf8,8)

ADD_PTR_U(bhalf ,1)
ADD_PTR_U(bhalf2,2)
ADD_PTR_U(bhalf4,4)
ADD_PTR_U(bhalf8,8)

ADD_PTR_U(float ,1)
ADD_PTR_U(float2,2)
ADD_PTR_U(float4,4)
ADD_PTR_U(float8,8)

__device__ __forceinline__ void ew_zero(ehalf  &a) { a.x = 0; }
__device__ __forceinline__ void ew_zero(ehalf2 &a) { a.x = 0; }
__device__ __forceinline__ void ew_zero(ehalf4 &a) { a.x = a.y = 0; }
__device__ __forceinline__ void ew_zero(ehalf8 &a) { a.x = a.y = a.z = a.w = 0; }

__device__ __forceinline__ void ew_zero(bhalf  &a) { a.x = 0; }
__device__ __forceinline__ void ew_zero(bhalf2 &a) { a.x = 0; }
__device__ __forceinline__ void ew_zero(bhalf4 &a) { a.x = a.y = 0; }
__device__ __forceinline__ void ew_zero(bhalf8 &a) { a.x = a.y = a.z = a.w = 0; }

__device__ __forceinline__ void ew_zero(float  &a) { a = 0.0f; }
__device__ __forceinline__ void ew_zero(float2 &a) { a.x = a.y = 0.0f; }
__device__ __forceinline__ void ew_zero(float4 &a) { a.x = a.y = a.z = a.w = 0.0f; }
__device__ __forceinline__ void ew_zero(float8 &v) { ew_zero(v.a); ew_zero(v.b); }


__device__ __forceinline__ float  _add(float x, float y) { return x + y; }
__device__ __forceinline__ float  _sub(float x, float y) { return x - y; }
__device__ __forceinline__ float  _mul(float x, float y) { return x * y; }
__device__ __forceinline__ float  _div(float x, float y) { return x / y; }
__device__ __forceinline__ float  _neg(float x) { return -x; }
__device__ __forceinline__ float  _rcp(float x) { return 1.0f / x; }
__device__ __forceinline__ float  _sqr(float x) { return x*x; }
__device__ __forceinline__ float  _sig(float x) { return 1.0f/(1.0f + expf(-x)); }
__device__ __forceinline__ float _relu(float x) { return fmaxf(x, 0.0f); }
__device__ __forceinline__ float  _elu(float x, float a) { return x > 0.0f ? x : a * (expf(x) - 1.0f); }

__device__ __forceinline__ float  _div_grad(float dz, float x, float y) { return -dz * x / (y*y); }
__device__ __forceinline__ float  _max_grad(float dz, float x, float y) { return dz * (x >= y); }
__device__ __forceinline__ float  _min_grad(float dz, float x, float y) { return dz * (x <= y); }
__device__ __forceinline__ float _relu_grad(float dz, float z) { return z > 0.0f ? dz : 0.0f; }
__device__ __forceinline__ float  _sig_grad(float dz, float z) { return dz * (z - z*z); }
__device__ __forceinline__ float _tanh_grad(float dz, float z) { return dz * (1.0f - z*z); }
__device__ __forceinline__ float  _rcp_grad(float dz, float x) { return -dz / (x*x); }
__device__ __forceinline__ float  _sqr_grad(float dz, float x) { return dz * x * 2.0f; }
__device__ __forceinline__ float _sqrt_grad(float dz, float x) { return 0.5f * dz * rsqrtf(x); }
__device__ __forceinline__ float  _exp_grad(float dz, float x) { return dz * expf(x); }
__device__ __forceinline__ float  _log_grad(float dz, float x) { return dz / x; }
__device__ __forceinline__ float  _elu_grad(float dz, float x, float a) { return x > 0.0f ? dz : dz * (a * (expf(x) - 1.0f) + a); }




// default load non-coherent
__device__ __forceinline__ float  load(const float*  __restrict__ in, int i=0, bool b=true) { float  v; ew_zero(v); if (b) v = __ldg(in + i); return v; }
__device__ __forceinline__ float2 load(const float2* __restrict__ in, int i=0, bool b=true) { float2 v; ew_zero(v); if (b) v = __ldg(in + i); return v; }
__device__ __forceinline__ float4 load(const float4* __restrict__ in, int i=0, bool b=true) { float4 v; ew_zero(v); if (b) v = __ldg(in + i); return v; }
__device__ __forceinline__ float8 load(const float8* __restrict__ in, int i=0, bool b=true) { float8 v; ew_zero(v); return v; } // not used

__device__ __forceinline__ float  load(const ehalf*  __restrict__ in, int i=0, bool b=true) { ehalf  v; ew_zero(v); if (b) v = __ldg(in + i); return to_float(v); }
__device__ __forceinline__ float2 load(const ehalf2* __restrict__ in, int i=0, bool b=true) { ehalf2 v; ew_zero(v); if (b) v = __ldg(in + i); return to_float(v); }
__device__ __forceinline__ float4 load(const ehalf4* __restrict__ in, int i=0, bool b=true) { ehalf4 v; ew_zero(v); if (b) v = __ldg(in + i); return to_float(v); }
__device__ __forceinline__ float8 load(const ehalf8* __restrict__ in, int i=0, bool b=true) { ehalf8 v; ew_zero(v); if (b) v = __ldg(in + i); return to_float(v); }

__device__ __forceinline__ float  load(const bhalf*  __restrict__ in, int i=0, bool b=true) { bhalf  v; ew_zero(v); if (b) v = __ldg(in + i); return to_float(v); }
__device__ __forceinline__ float2 load(const bhalf2* __restrict__ in, int i=0, bool b=true) { bhalf2 v; ew_zero(v); if (b) v = __ldg(in + i); return to_float(v); }
__device__ __forceinline__ float4 load(const bhalf4* __restrict__ in, int i=0, bool b=true) { bhalf4 v; ew_zero(v); if (b) v = __ldg(in + i); return to_float(v); }
__device__ __forceinline__ float8 load(const bhalf8* __restrict__ in, int i=0, bool b=true) { bhalf8 v; ew_zero(v); if (b) v = __ldg(in + i); return to_float(v); }

__device__ __forceinline__ int    load(const int*    __restrict__ in, int i=0, bool b=true) { int v = 0;   if (b) v = __ldg(in + i); return v; }

// load into float array
__device__ __forceinline__ void load(float* ret, const float*  __restrict__ in, int i=0, bool b=true) { float  v; ew_zero(v); if (b) v = __ldg(in + i); *(float *)ret = v; }
__device__ __forceinline__ void load(float* ret, const float2* __restrict__ in, int i=0, bool b=true) { float2 v; ew_zero(v); if (b) v = __ldg(in + i); *(float2*)ret = v; }
__device__ __forceinline__ void load(float* ret, const float4* __restrict__ in, int i=0, bool b=true) { float4 v; ew_zero(v); if (b) v = __ldg(in + i); *(float4*)ret = v; }
__device__ __forceinline__ void load(float* ret, const float8* __restrict__ in, int i=0, bool b=true) { } // not used

__device__ __forceinline__ void load(float* ret, const ehalf*  __restrict__ in, int i=0, bool b=true) { ehalf  v; ew_zero(v); if (b) v = __ldg(in + i); *(float *)ret = to_float(v); }
__device__ __forceinline__ void load(float* ret, const ehalf2* __restrict__ in, int i=0, bool b=true) { ehalf2 v; ew_zero(v); if (b) v = __ldg(in + i); *(float2*)ret = to_float(v); }
__device__ __forceinline__ void load(float* ret, const ehalf4* __restrict__ in, int i=0, bool b=true) { ehalf4 v; ew_zero(v); if (b) v = __ldg(in + i); *(float4*)ret = to_float(v); }
__device__ __forceinline__ void load(float* ret, const ehalf8* __restrict__ in, int i=0, bool b=true) { ehalf8 v; ew_zero(v); if (b) v = __ldg(in + i); *(float8*)ret = to_float(v); }

__device__ __forceinline__ void load(float* ret, const bhalf*  __restrict__ in, int i=0, bool b=true) { bhalf  v; ew_zero(v); if (b) v = __ldg(in + i); *(float *)ret = to_float(v); }
__device__ __forceinline__ void load(float* ret, const bhalf2* __restrict__ in, int i=0, bool b=true) { bhalf2 v; ew_zero(v); if (b) v = __ldg(in + i); *(float2*)ret = to_float(v); }
__device__ __forceinline__ void load(float* ret, const bhalf4* __restrict__ in, int i=0, bool b=true) { bhalf4 v; ew_zero(v); if (b) v = __ldg(in + i); *(float4*)ret = to_float(v); }
__device__ __forceinline__ void load(float* ret, const bhalf8* __restrict__ in, int i=0, bool b=true) { bhalf8 v; ew_zero(v); if (b) v = __ldg(in + i); *(float8*)ret = to_float(v); }

// load coherent
__device__ __forceinline__ float  load_c(const float*  __restrict__ in, int i=0, bool b=true) { float  v; ew_zero(v); if (b) v = in[i]; return v; }
__device__ __forceinline__ float2 load_c(const float2* __restrict__ in, int i=0, bool b=true) { float2 v; ew_zero(v); if (b) v = in[i]; return v; }
__device__ __forceinline__ float4 load_c(const float4* __restrict__ in, int i=0, bool b=true) { float4 v; ew_zero(v); if (b) v = in[i]; return v; }
__device__ __forceinline__ float8 load_c(const float8* __restrict__ in, int i=0, bool b=true) { float8 v; ew_zero(v); return v; } // not used

__device__ __forceinline__ float  load_c(const ehalf*  __restrict__ in, int i=0, bool b=true) { ehalf  v; ew_zero(v); if (b) v = in[i]; return to_float(v); }
__device__ __forceinline__ float2 load_c(const ehalf2* __restrict__ in, int i=0, bool b=true) { ehalf2 v; ew_zero(v); if (b) v = in[i]; return to_float(v); }
__device__ __forceinline__ float4 load_c(const ehalf4* __restrict__ in, int i=0, bool b=true) { ehalf4 v; ew_zero(v); if (b) v = in[i]; return to_float(v); }
__device__ __forceinline__ float8 load_c(const ehalf8* __restrict__ in, int i=0, bool b=true) { ehalf8 v; ew_zero(v); if (b) v = in[i]; return to_float(v); }

__device__ __forceinline__ float  load_c(const bhalf*  __restrict__ in, int i=0, bool b=true) { bhalf  v; ew_zero(v); if (b) v = in[i]; return to_float(v); }
__device__ __forceinline__ float2 load_c(const bhalf2* __restrict__ in, int i=0, bool b=true) { bhalf2 v; ew_zero(v); if (b) v = in[i]; return to_float(v); }
__device__ __forceinline__ float4 load_c(const bhalf4* __restrict__ in, int i=0, bool b=true) { bhalf4 v; ew_zero(v); if (b) v = in[i]; return to_float(v); }
__device__ __forceinline__ float8 load_c(const bhalf8* __restrict__ in, int i=0, bool b=true) { bhalf8 v; ew_zero(v); if (b) v = in[i]; return to_float(v); }

__device__ __forceinline__ void store(float*  out, float  v, int i=0, bool b=true) { if (b) __stg(out + i, v); }
__device__ __forceinline__ void store(float2* out, float2 v, int i=0, bool b=true) { if (b) __stg(out + i, v); }
__device__ __forceinline__ void store(float4* out, float4 v, int i=0, bool b=true) { if (b) __stg(out + i, v); }
__device__ __forceinline__ void store(float8* out, float8 v, int i=0, bool b=true) { return; } // not used

__device__ __forceinline__ void store_f(float*  out, float  v, int i=0, bool b=true) { if (b) __stg(out + i, v); }
__device__ __forceinline__ void store_f(float2* out, float2 v, int i=0, bool b=true) { if (b) __stg(out + i, v); }
__device__ __forceinline__ void store_f(float4* out, float4 v, int i=0, bool b=true) { if (b) __stg(out + i, v); }
__device__ __forceinline__ void store_f(float8* out, float8 v, int i=0, bool b=true) { return; } // not used

__device__ __forceinline__ void store_g(float*  out, float  v, int i=0, bool b=true) { if (b) __stg(out + i, v); }
__device__ __forceinline__ void store_g(float2* out, float2 v, int i=0, bool b=true) { if (b) __stg(out + i, v); }
__device__ __forceinline__ void store_g(float4* out, float4 v, int i=0, bool b=true) { if (b) __stg(out + i, v); }
__device__ __forceinline__ void store_g(float8* out, float8 v, int i=0, bool b=true) { return; } // not used

// __device__ __forceinline__ void store_f(float*  out, float  v, int i=0, bool b=true) { float  r = round3f(v); if (b) __stg(out + i, r); }
// __device__ __forceinline__ void store_f(float2* out, float2 v, int i=0, bool b=true) { float2 r = round3f(v); if (b) __stg(out + i, r); }
// __device__ __forceinline__ void store_f(float4* out, float4 v, int i=0, bool b=true) { float4 r = round3f(v); if (b) __stg(out + i, r); }
// __device__ __forceinline__ void store_f(float8* out, float8 v, int i=0, bool b=true) { return; } // not used

// __device__ __forceinline__ void store_g(float*  out, float  v, int i=0, bool b=true) { float  r = round2g(v); if (b) __stg(out + i, r); }
// __device__ __forceinline__ void store_g(float2* out, float2 v, int i=0, bool b=true) { float2 r = round2g(v); if (b) __stg(out + i, r); }
// __device__ __forceinline__ void store_g(float4* out, float4 v, int i=0, bool b=true) { float4 r = round2g(v); if (b) __stg(out + i, r); }
// __device__ __forceinline__ void store_g(float8* out, float8 v, int i=0, bool b=true) { return; } // not used



__device__ __forceinline__ void store(ehalf*  out, float  v, int i=0, bool b=true) { ehalf  r = to_ehalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store(ehalf2* out, float2 v, int i=0, bool b=true) { ehalf2 r = to_ehalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store(ehalf4* out, float4 v, int i=0, bool b=true) { ehalf4 r = to_ehalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store(ehalf8* out, float8 v, int i=0, bool b=true) { ehalf8 r = to_ehalf(v); if (b) __stg(out + i, r); }

__device__ __forceinline__ void store(bhalf*  out, float  v, int i=0, bool b=true) { bhalf  r = to_bhalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store(bhalf2* out, float2 v, int i=0, bool b=true) { bhalf2 r = to_bhalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store(bhalf4* out, float4 v, int i=0, bool b=true) { bhalf4 r = to_bhalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store(bhalf8* out, float8 v, int i=0, bool b=true) { bhalf8 r = to_bhalf(v); if (b) __stg(out + i, r); }

__device__ __forceinline__ void store_f(ehalf*  out, float  v, int i=0, bool b=true) { ehalf  r = to_ehalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store_f(ehalf2* out, float2 v, int i=0, bool b=true) { ehalf2 r = to_ehalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store_f(ehalf4* out, float4 v, int i=0, bool b=true) { ehalf4 r = to_ehalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store_f(ehalf8* out, float8 v, int i=0, bool b=true) { ehalf8 r = to_ehalf(v); if (b) __stg(out + i, r); }

__device__ __forceinline__ void store_f(bhalf*  out, float  v, int i=0, bool b=true) { bhalf  r = to_bhalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store_f(bhalf2* out, float2 v, int i=0, bool b=true) { bhalf2 r = to_bhalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store_f(bhalf4* out, float4 v, int i=0, bool b=true) { bhalf4 r = to_bhalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store_f(bhalf8* out, float8 v, int i=0, bool b=true) { bhalf8 r = to_bhalf(v); if (b) __stg(out + i, r); }

__device__ __forceinline__ void store_g(ehalf*  out, float  v, int i=0, bool b=true) { ehalf  r = to_ehalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store_g(ehalf2* out, float2 v, int i=0, bool b=true) { ehalf2 r = to_ehalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store_g(ehalf4* out, float4 v, int i=0, bool b=true) { ehalf4 r = to_ehalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store_g(ehalf8* out, float8 v, int i=0, bool b=true) { ehalf8 r = to_ehalf(v); if (b) __stg(out + i, r); }

__device__ __forceinline__ void store_g(bhalf*  out, float  v, int i=0, bool b=true) { bhalf  r = to_bhalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store_g(bhalf2* out, float2 v, int i=0, bool b=true) { bhalf2 r = to_bhalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store_g(bhalf4* out, float4 v, int i=0, bool b=true) { bhalf4 r = to_bhalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store_g(bhalf8* out, float8 v, int i=0, bool b=true) { bhalf8 r = to_bhalf(v); if (b) __stg(out + i, r); }



// For unused code paths but the compiler still needs to process.
__device__ __forceinline__ void store(float* out, float2 v, int i=0, bool b=true) {}
__device__ __forceinline__ void store(ehalf* out, float2 v, int i=0, bool b=true) {}
__device__ __forceinline__ void store(bhalf* out, float2 v, int i=0, bool b=true) {}
__device__ __forceinline__ void store(float2* out, float v, int i=0, bool b=true) {}
__device__ __forceinline__ void store(ehalf2* out, float v, int i=0, bool b=true) {}
__device__ __forceinline__ void store(bhalf2* out, float v, int i=0, bool b=true) {}


__device__ __forceinline__ float ew_sum(float  a) { return a; }
__device__ __forceinline__ float ew_sum(float2 a) { return a.x + a.y; }
__device__ __forceinline__ float ew_sum(float4 a) { return a.x + a.y + a.z + a.w; }
__device__ __forceinline__ float ew_sum(float8 v) { return ew_sum(v.a) + ew_sum(v.b); }

__device__ __forceinline__ float  ew_warp_sum(float  a, int i) { a   += shfl_xor(a, i); return a; }
__device__ __forceinline__ float2 ew_warp_sum(float2 a, int i) { a.x += shfl_xor(a.x, i); a.y += shfl_xor(a.y, i); return a; }
__device__ __forceinline__ float4 ew_warp_sum(float4 a, int i) { a.x += shfl_xor(a.x, i); a.y += shfl_xor(a.y, i); a.z += shfl_xor(a.z, i); a.w += shfl_xor(a.w, i); return a; }
__device__ __forceinline__ float8 ew_warp_sum(float8 v, int i) { ew_warp_sum(v.a, i); ew_warp_sum(v.b, i); return v; }

__device__ __forceinline__ float  ew_elu(float  x, float a) { return _elu(x, a); }
__device__ __forceinline__ float2 ew_elu(float2 x, float a) { float2 r; BINARY_VEC2_S(_elu,r,x,a); return r; }
__device__ __forceinline__ float4 ew_elu(float4 x, float a) { float4 r; BINARY_VEC4_S(_elu,r,x,a); return r; }
__device__ __forceinline__ float8 ew_elu(float8 x, float a) { float8 r; BINARY_VEC4_S(_elu,r.a,x.a,a); BINARY_VEC4_S(_elu,r.b,x.b,a); return r; }

__device__ __forceinline__ float  ew_elu_grad(float  dz, float  x, float a) { return _elu_grad(dz,x,a); }
__device__ __forceinline__ float2 ew_elu_grad(float2 dz, float2 x, float a) { float2 r; TERNARY_VEC2_S(_elu_grad,r,dz,x,a); return r; }
__device__ __forceinline__ float4 ew_elu_grad(float4 dz, float4 x, float a) { float4 r; TERNARY_VEC4_S(_elu_grad,r,dz,x,a); return r; }
__device__ __forceinline__ float8 ew_elu_grad(float8 dz, float8 x, float a) { float8 r; TERNARY_VEC4_S(_elu_grad,r.a,dz.a,x.a,a); TERNARY_VEC4_S(_elu_grad,r.b,dz.b,x.b,a); return r; }

#define MATH_Z_XY(name, impl) \
__device__ __forceinline__ float  name(float  x, float  y) { return impl(x,y); } \
__device__ __forceinline__ float2 name(float2 x, float2 y) { float2 r; BINARY_VEC2(impl,r,x,y);  return r; } \
__device__ __forceinline__ float2 name(float2 x, float  y) { float2 r; BINARY_VEC2_S(impl,r,x,y); return r; } \
__device__ __forceinline__ float4 name(float4 x, float4 y) { float4 r; BINARY_VEC4(impl,r,x,y);  return r; } \
__device__ __forceinline__ float4 name(float4 x, float  y) { float4 r; BINARY_VEC4_S(impl,r,x,y); return r; } \
__device__ __forceinline__ float8 name(float8 x, float8 y) { float8 r; BINARY_VEC4(impl,r.a,x.a,y.a); BINARY_VEC4(impl,r.b,x.b,y.b);  return r; } \
__device__ __forceinline__ float8 name(float8 x, float  y) { float8 r; BINARY_VEC4_S(impl,r.a,x.a,y);  BINARY_VEC4_S(impl,r.b,x.b,y); return r; }

#define MATH_Z_X(name, impl) \
__device__ __forceinline__ float  name(float  x) { return impl(x); } \
__device__ __forceinline__ float2 name(float2 x) { float2 r; UNARY_VEC2(impl,r,x); return r; } \
__device__ __forceinline__ float4 name(float4 x) { float4 r; UNARY_VEC4(impl,r,x); return r; } \
__device__ __forceinline__ float8 name(float8 x) { float8 r; UNARY_VEC4(impl,r.a,x.b); UNARY_VEC4(impl,r.b,x.b); return r; }

#define MATH_DZ_XY(name, impl) \
__device__ __forceinline__ float  name(float  dz, float  x, float  y) { return impl(dz,x,y); } \
__device__ __forceinline__ float2 name(float2 dz, float2 x, float2 y) { float2 r; TERNARY_VEC2(impl,r,dz,x,y); return r; } \
__device__ __forceinline__ float4 name(float4 dz, float4 x, float4 y) { float4 r; TERNARY_VEC4(impl,r,dz,x,y); return r; } \
__device__ __forceinline__ float8 name(float8 dz, float8 x, float8 y) { float8 r; TERNARY_VEC4(impl,r.a,dz.a,x.a,y.a); TERNARY_VEC4(impl,r.b,dz.b,x.b,y.b); return r; }

#define MATH_DZ_Z(name, impl) \
__device__ __forceinline__ float  name(float  dz, float  z) { return impl(dz,z); } \
__device__ __forceinline__ float2 name(float2 dz, float2 z) { float2 r; BINARY_VEC2(impl,r,dz,z); return r; } \
__device__ __forceinline__ float4 name(float4 dz, float4 z) { float4 r; BINARY_VEC4(impl,r,dz,z); return r; } \
__device__ __forceinline__ float8 name(float8 dz, float8 z) { float8 r; BINARY_VEC4(impl,r.a,dz.a,z.a); BINARY_VEC4(impl,r.b,dz.b,z.b); return r; }

#define MATH_DZ_X(name, impl) \
__device__ __forceinline__ float  name(float  dz, float  x) { return impl(dz,x); } \
__device__ __forceinline__ float2 name(float2 dz, float2 x) { float2 r; BINARY_VEC2(impl,r,dz,x); return r; } \
__device__ __forceinline__ float4 name(float4 dz, float4 x) { float4 r; BINARY_VEC4(impl,r,dz,x); return r; } \
__device__ __forceinline__ float8 name(float8 dz, float8 x) { float8 r; BINARY_VEC4(impl,r.a,dz.a,x.a); BINARY_VEC4(impl,r.b,dz.b,x.b); return r; }

MATH_Z_XY(ew_add,      _add)
MATH_Z_XY(ew_sub,      _sub)
MATH_Z_XY(ew_mul,      _mul)
MATH_Z_XY(ew_div,      _div)
MATH_Z_XY(ew_maximum, fmaxf)
MATH_Z_XY(ew_minimum, fminf)

MATH_Z_X(ew_neg,     _neg)
MATH_Z_X(ew_rcp,     _rcp)
MATH_Z_X(ew_sqr,     _sqr)
MATH_Z_X(ew_sqrt,   sqrtf)
MATH_Z_X(ew_rsqrt, rsqrtf)
MATH_Z_X(ew_exp,     expf)
MATH_Z_X(ew_log,     logf)
MATH_Z_X(ew_sig,     _sig)
MATH_Z_X(ew_tanh,   tanhf)
MATH_Z_X(ew_relu,   _relu)

MATH_DZ_XY(ew_max_grad, _max_grad)
MATH_DZ_XY(ew_min_grad, _min_grad)
MATH_DZ_XY(ew_div_grad, _div_grad)

MATH_DZ_Z(ew_relu_grad, _relu_grad)
MATH_DZ_Z(ew_sig_grad,   _sig_grad)
MATH_DZ_Z(ew_tanh_grad, _tanh_grad)

MATH_DZ_X(ew_rcp_grad,   _rcp_grad)
MATH_DZ_X(ew_sqr_grad,   _sqr_grad)
MATH_DZ_X(ew_sqrt_grad, _sqrt_grad)
MATH_DZ_X(ew_exp_grad,   _exp_grad)
MATH_DZ_X(ew_log_grad,   _log_grad)