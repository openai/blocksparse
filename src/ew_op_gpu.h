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
    // v = fmaxf(fminf(v, 65504.0f), -65504.0f);
    ehalf r;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r.x) : "f"(v));
    return r;
}

__device__ __forceinline__ ehalf2 to_ehalf(float2 v)
{
    // v.x = fmaxf(fminf(v.x, 65504.0f), -65504.0f);
    // v.y = fmaxf(fminf(v.y, 65504.0f), -65504.0f);
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
    // v.x = fmaxf(fminf(v.x, 65504.0f), -65504.0f);
    // v.y = fmaxf(fminf(v.y, 65504.0f), -65504.0f);
    // v.z = fmaxf(fminf(v.z, 65504.0f), -65504.0f);
    // v.w = fmaxf(fminf(v.w, 65504.0f), -65504.0f);
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
    // v.a.x = fmaxf(fminf(v.a.x, 65504.0f), -65504.0f);
    // v.a.y = fmaxf(fminf(v.a.y, 65504.0f), -65504.0f);
    // v.a.z = fmaxf(fminf(v.a.z, 65504.0f), -65504.0f);
    // v.a.w = fmaxf(fminf(v.a.w, 65504.0f), -65504.0f);
    // v.b.x = fmaxf(fminf(v.b.x, 65504.0f), -65504.0f);
    // v.b.y = fmaxf(fminf(v.b.y, 65504.0f), -65504.0f);
    // v.b.z = fmaxf(fminf(v.b.z, 65504.0f), -65504.0f);
    // v.b.w = fmaxf(fminf(v.b.w, 65504.0f), -65504.0f);
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

__device__ __forceinline__ bhalf  to_bhalf(float  v)
{
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


// 0s-6e-10m, exp 0 => 2**-60, exp 63 => 2**3
__device__ __forceinline__ float  to_float(vhalf  v)
{
    float r;
    if (v.x == 0)
        r = 0.0f;
    else
        asm("{\n\t"
            ".reg .u32 vhalf, float;              \n\t"
            "mov.b32 vhalf, {%1, 0};              \n\t"
            "and.b32 float, vhalf, 0x0000fc00;    \n\t" // strip mantissa
            "shl.b32 float, float, 13;            \n\t" // shift exponent to fp32 position (10 -> 23 == 13)
            "add.u32 float, float, 0x21800000;    \n\t" // add exponent bias ("0x%x" % ((127-60) << 23)
            "bfi.b32 float, vhalf, float, 13, 10; \n\t" // insert 10 bit mantissa at position 13
            "mov.b32 %0, float;                   \n\t"
            "}" : "=f"(r) : "h"(v.x));
    return r;
}
// 1s-6e-9m, exp 0 => 2**-60, exp 63 => 2**3
__device__ __forceinline__ float  to_float(mhalf  v)
{
    float r;
    if (v.x == 0)
        r = 0.0f;
    else
        asm("{\n\t"
            ".reg .u32 mhalf, exp, sign, float;  \n\t"
            "mov.b32 mhalf, {%1, 0};             \n\t"
            "bfe.u32 exp,   mhalf,  9, 6;        \n\t" // extract exponent
            "bfe.u32 sign,  mhalf, 15, 1;        \n\t" // extract sign
            "shl.b32 float, exp,     23;         \n\t" // shift exponent to fp32 position
            "add.u32 float, float, 0x21800000;   \n\t" // add exponent bias ("0x%x" % ((127-60) << 23)
            "bfi.b32 float, mhalf, float, 14, 9; \n\t" // insert 9 bit mantissa at position 14
            "bfi.b32 float, sign,  float, 31, 1; \n\t" // insert 1 bit sign at position 31
            "mov.b32 %0, float;                  \n\t"
            "}" : "=f"(r) : "h"(v.x));
    return r;
}
// 0s-6e-10m, exp 0 => 2**-60, exp 63 => 2**3
__device__ __forceinline__ vhalf  to_vhalf(float  v)
{
    vhalf r;

    // 0x417fe000 = "0x%08x" % (((127+3) << 23) | (1023 << 13))
    // 15.9921875 = unpack("f", pack("I", 0x417fe000))[0]
    v = fminf(v, 15.9921875f);

    // 0x21802000 = "0x%08x" % (((127-60) << 23) | (1 << 13))
    // 8.682087709356578e-19 = unpack("f", pack("I", 0x21802000))[0]
    if (v < 8.682087709356578e-19f)
        r.x = 0;
    else
        asm("{\n\t"
            ".reg .f32 float, round_exp;                         \n\t"
            ".reg .u32 vhalf, exp, ufloat;                       \n\t"
            ".reg .u16 u16;                                      \n\t"
            "mov.b32 float, %1;                                  \n\t"
            "and.b32 round_exp, float, 0xff800000;               \n\t" // mask exponent for rounding
            "fma.rz.ftz.f32 float, round_exp, 0F3a000000, float; \n\t" // round: "0F%08x" % ((127 - 10 - 1) << 23)
            "mov.b32 ufloat, float;                              \n\t"
            "bfe.u32 exp,   ufloat, 23,  8;                      \n\t" // extract exponent
            "bfe.u32 vhalf, ufloat, 13, 10;                      \n\t" // extract mantissa
            "sub.u32 exp, exp, 67;                               \n\t" // subtract exponent bias 127-60 == 67
            "bfi.b32 vhalf, exp, vhalf, 10, 6;                   \n\t" // merge exponent with mantissa
            "mov.b32 { %0, u16 }, vhalf;                         \n\t"
            "}" : "=h"(r.x) : "f"(v));
    return r;
}
// 1s-6e-9m, exp 0 => 2**-60, exp 63 => 2**3
__device__ __forceinline__ mhalf  to_mhalf(float  v)
{
    mhalf r;

    // 0x417fc000 = "0x%08x" % (((127+3) << 23) | (511 << 14))
    // 15.984375 = unpack("f", pack("I", 0x417fc000))[0]
    v = fmaxf(fminf(v, 15.984375f), -15.984375f);

    // 0x21804000 = "0x%08x" % (((127-60) << 23) | (1 << 14))
    // 8.690558038829121e-19 = unpack("f", pack("I", 0x21804000))[0]
    if (abs(v) < 8.690558038829121e-19f)
        r.x = 0;
    else
        asm("{\n\t"
            ".reg .f32 float, round_exp;                        \n\t"
            ".reg .u32 mhalf, exp, sign, ufloat;                \n\t"
            ".reg .u16 u16;                                     \n\t"
            "mov.b32 float, %1;                                 \n\t"
            "and.b32 round_exp, float, 0xff800000;              \n\t" // mask exponent+sign for rounding
            "fma.rz.ftz.f32 float, round_exp, 0F3a800000, float;\n\t" // round: "0F%08x" % ((127 - 9 - 1) << 23)
            "mov.b32 ufloat, float;                             \n\t"
            "bfe.u32 sign,  ufloat, 31, 1;                      \n\t" // extract sign
            "bfe.u32 exp,   ufloat, 23, 8;                      \n\t" // extract exponent
            "bfe.u32 mhalf, ufloat, 14, 9;                      \n\t" // extract mantissa
            "sub.u32 exp, exp, 67;                              \n\t" // subtract exponent bias 127-60 == 67
            "bfi.b32 mhalf,  exp, mhalf,  9, 6;                 \n\t" // merge exponent with mantissa
            "bfi.b32 mhalf, sign, mhalf, 15, 1;                 \n\t" // merge sign with exponent+mantissa
            "mov.b32 { %0, u16 }, mhalf;                        \n\t"
            "}" : "=h"(r.x) : "f"(v));
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
__device__ __forceinline__ vhalf __ldg(const vhalf *ptr)
{
    vhalf ret;
    asm volatile ("ld.global.nc.u16 %0, [%1];" : "=h"(ret.x) : "l"(ptr));
    return ret;
}
__device__ __forceinline__ mhalf __ldg(const mhalf *ptr)
{
    mhalf ret;
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


__device__ __forceinline__ void __stg(ehalf *ptr, ehalf val)
{
    asm volatile ("st.global.wb.u16 [%0], %1;" :: "l"(ptr), "h"(val.x)  );
}
__device__ __forceinline__ void __stg(ehalf2 *ptr, ehalf2 val)
{
    asm volatile ("st.global.wb.u32 [%0], %1;" :: "l"(ptr), "r"(val.x)  );
}
__device__ __forceinline__ void __stg(ehalf4 *ptr, ehalf4 val)
{
    asm volatile ("st.global.wb.v2.u32 [%0], {%1, %2};" :: "l"(ptr), "r"(val.x), "r"(val.y)  );
}
__device__ __forceinline__ void __stg(ehalf8 *ptr, ehalf8 val)
{
    asm volatile ("st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)  );
}
__device__ __forceinline__ void __stg(bhalf *ptr, bhalf val)
{
    asm volatile ("st.global.wb.u16 [%0], %1;" :: "l"(ptr), "h"(val.x)  );
}
__device__ __forceinline__ void __stg(bhalf2 *ptr, bhalf2 val)
{
    asm volatile ("st.global.wb.u32 [%0], %1;" :: "l"(ptr), "r"(val.x)  );
}
__device__ __forceinline__ void __stg(bhalf4 *ptr, bhalf4 val)
{
    asm volatile ("st.global.wb.v2.u32 [%0], {%1, %2};" :: "l"(ptr), "r"(val.x), "r"(val.y)  );
}
__device__ __forceinline__ void __stg(bhalf8 *ptr, bhalf8 val)
{
    asm volatile ("st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)  );
}
__device__ __forceinline__ void __stg(float *ptr, float val)
{
    asm volatile ("st.global.wb.f32 [%0], %1;" :: "l"(ptr), "f"(val)  );
}
__device__ __forceinline__ void __stg(float2 *ptr, float2 val)
{
    asm volatile ("st.global.wb.v2.f32 [%0], {%1, %2};" :: "l"(ptr), "f"(val.x), "f"(val.y)  );
}
__device__ __forceinline__ void __stg(float4 *ptr, float4 val)
{
    asm volatile ("st.global.wb.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(ptr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w)  );
}
__device__ __forceinline__ void __stg(vhalf *ptr, vhalf val)
{
    asm volatile ("st.global.wb.u16 [%0], %1;" :: "l"(ptr), "h"(val.x)  );
}
__device__ __forceinline__ void __stg(mhalf *ptr, mhalf val)
{
    asm volatile ("st.global.wb.u16 [%0], %1;" :: "l"(ptr), "h"(val.x)  );
}

__device__ __forceinline__ void __stg(unsigned char *ptr, uint val)
{
    asm volatile ("st.global.wb.u8 [%0], %1;" :: "l"(ptr), "r"(val) );
}
__device__ __forceinline__ void __stg(ushort *ptr, uint val)
{
    asm volatile ("st.global.wb.u16 [%0], %1;" :: "l"(ptr), "r"(val) );
}
__device__ __forceinline__ void __stg(uint *ptr, uint val)
{
    asm volatile ("st.global.wb.u32 [%0], %1;" :: "l"(ptr), "r"(val) );
}


__device__ __forceinline__ void __stg(char *ptr, int val)
{
    asm volatile ("st.global.wb.s8 [%0], %1;" :: "l"(ptr), "r"(val) );
}
__device__ __forceinline__ void __stg(short *ptr, int val)
{
    asm volatile ("st.global.wb.s16 [%0], %1;" :: "l"(ptr), "r"(val) );
}
__device__ __forceinline__ void __stg(int *ptr, int val)
{
    asm volatile ("st.global.wb.s32 [%0], %1;" :: "l"(ptr), "r"(val) );
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

__device__ __forceinline__  void st_shared_v8(uint a, ehalf8 v)
{
    asm volatile ("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" :: "r"(a), "r"(v.x),"r"(v.y),"r"(v.z),"r"(v.w) );
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

__device__ __forceinline__ void atomicRed(float *ptr, float val, int i=0, bool b=true)
{
    if (b)
        asm volatile ("red.global.add.f32 [%0], %1;" :: "l"(ptr+i), "f"(val)  );
}
__device__ __forceinline__ void atomicRed(float4 *ptr, float4 val, int i=0, bool b=true)
{
    if (b)
        asm volatile ("red.global.add.f32 [%0], %1;" :: "l"(ptr+i), "f"(val.x)  );
}
__device__ __forceinline__ void atomicRedMax(float *ptr, float val)
{
    asm volatile ("red.global.max.u32 [%0], %1;" :: "l"(ptr), "r"(*(uint*)&val) );
}

__device__ __forceinline__ float shfl_xor(float var, int laneMask)
{
    float ret;
# if CUDA_VERSION >= 9020
    asm volatile ("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(ret) : "f"(var), "r"(laneMask));
# else
    asm volatile ("shfl.bfly.b32 %0, %1, %2, 0x1f;" : "=f"(ret) : "f"(var), "r"(laneMask));
# endif
    return ret;
}
__device__ __forceinline__ uint shfl_xor(uint var, int laneMask)
{
    uint ret;
# if CUDA_VERSION >= 9020
    asm volatile ("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(ret) : "r"(var), "r"(laneMask));
# else
    asm volatile ("shfl.bfly.b32 %0, %1, %2, 0x1f;" : "=r"(ret) : "r"(var), "r"(laneMask));
# endif
    return ret;
}
__device__ __forceinline__ int shfl_xor(int var, int laneMask)
{
    int ret;
# if CUDA_VERSION >= 9020
    asm volatile ("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(ret) : "r"(var), "r"(laneMask));
# else
    asm volatile ("shfl.bfly.b32 %0, %1, %2, 0x1f;" : "=r"(ret) : "r"(var), "r"(laneMask));
# endif
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

#define ADD_PTR_S(T) \
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
        "}" : "=l"(dst) : "l"(src), "r"(offset), "r"((int)sizeof(*src)));   \
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
        "}" : "=l"(dst) : "l"(src), "r"(offset), "r"((int)sizeof(*src)));   \
    return dst;                                                             \
}

#define ADD_PTR_U(T) \
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
        "}" : "=l"(dst) : "l"(src), "r"(offset), "r"((int)sizeof(*src)));    \
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
        "}" : "=l"(dst) : "l"(src), "r"(offset), "r"((int)sizeof(*src)));    \
    return dst;                                                              \
}

ADD_PTR_S(ehalf )
ADD_PTR_S(ehalf2)
ADD_PTR_S(ehalf4)
ADD_PTR_S(ehalf8)

ADD_PTR_S(bhalf )
ADD_PTR_S(bhalf2)
ADD_PTR_S(bhalf4)
ADD_PTR_S(bhalf8)

ADD_PTR_S(float )
ADD_PTR_S(float2)
ADD_PTR_S(float4)
ADD_PTR_S(float8)

ADD_PTR_S(mhalf)
ADD_PTR_S(vhalf)

ADD_PTR_S(unsigned char)
ADD_PTR_S(unsigned short)
ADD_PTR_S(unsigned int)
ADD_PTR_S(int)

ADD_PTR_U(ehalf )
ADD_PTR_U(ehalf2)
ADD_PTR_U(ehalf4)
ADD_PTR_U(ehalf8)

ADD_PTR_U(bhalf )
ADD_PTR_U(bhalf2)
ADD_PTR_U(bhalf4)
ADD_PTR_U(bhalf8)

ADD_PTR_U(float )
ADD_PTR_U(float2)
ADD_PTR_U(float4)
ADD_PTR_U(float8)

ADD_PTR_U(mhalf)
ADD_PTR_U(vhalf)

ADD_PTR_U(unsigned char)
ADD_PTR_U(unsigned short)
ADD_PTR_U(unsigned int)
ADD_PTR_U(int)

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

__device__ __forceinline__ void ew_set(float  &a, float val) { a = val; }
__device__ __forceinline__ void ew_set(float2 &a, float val) { a.x = a.y = val; }
__device__ __forceinline__ void ew_set(float4 &a, float val) { a.x = a.y = a.z = a.w = val; }
__device__ __forceinline__ void ew_set(float8 &v, float val) { ew_set(v.a, val); ew_set(v.b, val); }

__device__ __forceinline__ void ew_set(ehalf  &a, uint val) { a.x = (ushort)val; }
__device__ __forceinline__ void ew_set(ehalf2 &a, uint val) { a.x = val; }
__device__ __forceinline__ void ew_set(ehalf4 &a, uint val) { a.x = a.y = val; }
__device__ __forceinline__ void ew_set(ehalf8 &a, uint val) { a.x = a.y = a.z = a.w = val; }

__device__ __forceinline__ void ew_set(bhalf  &a, uint val) { a.x = (ushort)val; }
__device__ __forceinline__ void ew_set(bhalf2 &a, uint val) { a.x = val; }
__device__ __forceinline__ void ew_set(bhalf4 &a, uint val) { a.x = a.y = val; }
__device__ __forceinline__ void ew_set(bhalf8 &a, uint val) { a.x = a.y = a.z = a.w = val; }

__device__ __forceinline__ void ew_zero(vhalf  &a) { a.x = 0; }
__device__ __forceinline__ void ew_zero(mhalf  &a) { a.x = 0; }

// minimize catastrophic cancellation: https://en.wikipedia.org/wiki/Loss_of_significance
// Probably unnecessary, but GPU supports it at no cost (when used sparingly)
__device__ __forceinline__ float precise_sub(float a, float b)
{
    float r;
    asm("{\n\t"
        ".reg .f64 a, b, c;\n\t"
        "cvt.f64.f32 a, %1;\n\t"
        "cvt.f64.f32 b, %2;\n\t"
        "sub.f64 c, a, b;\n\t"
        "cvt.rn.f32.f64 %0, c;\n\t"
        "}" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float _ex2_approx(float x)
{
    asm("ex2.approx.ftz.f32 %0, %0;" : "+f"(x) :);
    return x;
}
__device__ __forceinline__ float _lg2_approx(float x)
{
    asm("lg2.approx.ftz.f32 %0, %0;" : "+f"(x) :);
    return x;
}
__device__ __forceinline__ float _exp_approx(float x)
{
    x *= 1.4426950408889634f;
    asm("ex2.approx.ftz.f32 %0, %0;" : "+f"(x) :);
    return x;
}
__device__ __forceinline__ float _log_approx(float x)
{
    asm("lg2.approx.ftz.f32 %0, %0;" : "+f"(x) :);
    x *= 0.6931471824645996f;
    return x;
}
__device__ __forceinline__ float _sqrt_approx(float x)
{
    asm("sqrt.approx.ftz.f32 %0, %0;" : "+f"(x) :);
    return x;
}
__device__ __forceinline__ float _rcp_approx(float x)
{
    asm("rcp.approx.ftz.f32 %0, %0;" : "+f"(x) :);
    return x;
}
__device__ __forceinline__ float _rsqrt_approx(float x)
{
    asm("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(x) :);
    return x;
}
__device__ __forceinline__ float _zero_inf(float x)
{
    asm("{                         \n\t"
        ".reg .pred p;             \n\t"
        "testp.infinite.f32 p, %0; \n\t"
        "selp.f32 %0, 0.0, %0, p;  \n\t"
        "}" : "+f"(x) :);
    return x;
}
__device__ __forceinline__ float _zero_nan(float x)
{
    asm("{                           \n\t"
        ".reg .pred p;               \n\t"
        "testp.notanumber.f32 p, %0; \n\t"
        "selp.f32 %0, 0.0, %0, p;    \n\t"
        "}" : "+f"(x) :);
    return x;
}
__device__ __forceinline__ float _zero_nan_inf(float x)
{
    asm("{                               \n\t"
        ".reg .pred is_finite;           \n\t"
        "testp.finite.f32 is_finite, %0; \n\t"
        "selp.f32 %0, %0, 0.0, is_finite;\n\t"
        "}" : "+f"(x) :);
    return x;
}

#define SQRT_2_PI 0.7978845608028654f

__device__ __forceinline__ float  _add(float x, float y) { return x + y; }
__device__ __forceinline__ float  _sub(float x, float y) { return x - y; }
__device__ __forceinline__ float  _mul(float x, float y) { return x * y; }
__device__ __forceinline__ float  _div(float x, float y) { return x * _rcp_approx(y); }
__device__ __forceinline__ float  _neg(float x) { return -x; }
__device__ __forceinline__ float  _rcp(float x) { return _rcp_approx(x); }
__device__ __forceinline__ float  _sqr(float x) { return x*x; }
__device__ __forceinline__ float _cube(float x) { return x*x*x; }
__device__ __forceinline__ float  _sig(float x) { return _rcp_approx(1.0f + _exp_approx(-x)); }
//__device__ __forceinline__ float _tanh(float x) { float e2x = _exp_approx(2.0f*x); return (e2x - 1.0f) * _rcp_approx(e2x + 1.0f); }
__device__ __forceinline__ float _relu(float x) { return fmaxf(x, 0.0f); }
__device__ __forceinline__ float  _elu(float x, float a) { return x > 0.0f ? x : a * (_exp_approx(x) - 1.0f); }

__device__ __forceinline__ float _swish(float x, float a) { return x * _sig(a * x); }
__device__ __forceinline__ float  _gelu(float x, float a) { return 0.5f * x * (1.0f + tanhf(SQRT_2_PI * (x + a * _cube(x)))); }

//__device__ __forceinline__ void  _argmax(float &maxval, uint &maxarg, float val, uint idx) { if (val > maxval) { maxval = val; maxarg = idx; } }

__device__ __forceinline__ float  _div_grad(float dz, float x, float y) { return -dz * x * _rcp_approx(y*y); }
__device__ __forceinline__ float  _max_grad(float dz, float x, float y) { return dz * (x >= y); }
__device__ __forceinline__ float  _min_grad(float dz, float x, float y) { return dz * (x <= y); }
__device__ __forceinline__ float _relu_grad(float dz, float z) { return z > 0.0f ? dz : 0.0f; }
__device__ __forceinline__ float  _sig_grad(float dz, float z) { return dz * (z - z*z); }
__device__ __forceinline__ float _tanh_grad(float dz, float z) { return dz * (1.0f - z*z); }
__device__ __forceinline__ float  _rcp_grad(float dz, float x) { return -dz * _rcp_approx(x*x); }
__device__ __forceinline__ float  _sqr_grad(float dz, float x) { return dz * x * 2.0f; }
__device__ __forceinline__ float _cube_grad(float dz, float x) { return dz * x * x * 3.0f; }
__device__ __forceinline__ float _sqrt_grad(float dz, float x) { return 0.5f * dz * _rsqrt_approx(x); }
__device__ __forceinline__ float  _exp_grad(float dz, float x) { return dz * _exp_approx(x); }
__device__ __forceinline__ float  _log_grad(float dz, float x) { return dz / x; }
__device__ __forceinline__ float  _elu_grad(float dz, float x, float a) { return x > 0.0f ? dz : dz * (a * (_exp_approx(x) - 1.0f) + a); }

__device__ __forceinline__ float _swish_grad(float dz, float x, float a)
{
    float sig = _sig(x * a);
    return dz * sig + _sig_grad(dz * x, sig) * a;
}
__device__ __forceinline__ float _gelu_grad(float dz, float x, float a)
{
    float tanh      = tanhf(SQRT_2_PI * (x + a * _cube(x)));
    float tanh_grad = _tanh_grad(0.5f * dz * x, tanh) * SQRT_2_PI;
    return 0.5f * dz * (1.0f + tanh) + tanh_grad + _cube_grad(tanh_grad * a, x);
}

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

__device__ __forceinline__ float  load(const mhalf*  __restrict__ in, int i=0, bool b=true) { mhalf  v; ew_zero(v); if (b) v = __ldg(in + i); return to_float(v); }
__device__ __forceinline__ float  load(const vhalf*  __restrict__ in, int i=0, bool b=true) { vhalf  v; ew_zero(v); if (b) v = __ldg(in + i); return to_float(v); }

__device__ __forceinline__  int   load(const  int*   __restrict__ in, int i=0, bool b=true) {  int v = 0;  if (b) v = __ldg(in + i); return v; }
__device__ __forceinline__ uint   load(const uint*   __restrict__ in, int i=0, bool b=true) { uint v = 0;  if (b) v = __ldg(in + i); return v; }


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

__device__ __forceinline__ void store(ehalf*  out, float  v, int i=0, bool b=true) { ehalf  r = to_ehalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store(ehalf2* out, float2 v, int i=0, bool b=true) { ehalf2 r = to_ehalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store(ehalf4* out, float4 v, int i=0, bool b=true) { ehalf4 r = to_ehalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store(ehalf8* out, float8 v, int i=0, bool b=true) { ehalf8 r = to_ehalf(v); if (b) __stg(out + i, r); }

__device__ __forceinline__ void store(bhalf*  out, float  v, int i=0, bool b=true) { bhalf  r = to_bhalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store(bhalf2* out, float2 v, int i=0, bool b=true) { bhalf2 r = to_bhalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store(bhalf4* out, float4 v, int i=0, bool b=true) { bhalf4 r = to_bhalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store(bhalf8* out, float8 v, int i=0, bool b=true) { bhalf8 r = to_bhalf(v); if (b) __stg(out + i, r); }

__device__ __forceinline__ void store(mhalf*  out, float  v, int i=0, bool b=true) { mhalf  r = to_mhalf(v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store(vhalf*  out, float  v, int i=0, bool b=true) { vhalf  r = to_vhalf(v); if (b) __stg(out + i, r); }

// store from float array
__device__ __forceinline__ void store(float*  out, float* v, int i=0, bool b=true) { if (b) __stg(out + i, *(float *)v); }
__device__ __forceinline__ void store(float2* out, float* v, int i=0, bool b=true) { if (b) __stg(out + i, *(float2*)v); }
__device__ __forceinline__ void store(float4* out, float* v, int i=0, bool b=true) { if (b) __stg(out + i, *(float4*)v); }
__device__ __forceinline__ void store(float8* out, float* v, int i=0, bool b=true) { return; } // not used

__device__ __forceinline__ void store(ehalf*  out, float* v, int i=0, bool b=true) { ehalf  r = to_ehalf(*(float *)v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store(ehalf2* out, float* v, int i=0, bool b=true) { ehalf2 r = to_ehalf(*(float2*)v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store(ehalf4* out, float* v, int i=0, bool b=true) { ehalf4 r = to_ehalf(*(float4*)v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store(ehalf8* out, float* v, int i=0, bool b=true) { ehalf8 r = to_ehalf(*(float8*)v); if (b) __stg(out + i, r); }

__device__ __forceinline__ void store(bhalf*  out, float* v, int i=0, bool b=true) { bhalf  r = to_bhalf(*(float *)v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store(bhalf2* out, float* v, int i=0, bool b=true) { bhalf2 r = to_bhalf(*(float2*)v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store(bhalf4* out, float* v, int i=0, bool b=true) { bhalf4 r = to_bhalf(*(float4*)v); if (b) __stg(out + i, r); }
__device__ __forceinline__ void store(bhalf8* out, float* v, int i=0, bool b=true) { bhalf8 r = to_bhalf(*(float8*)v); if (b) __stg(out + i, r); }


__device__ __forceinline__ void store( int*  out,  int v, int i=0, bool b=true) { if (b) __stg(out + i, v); }
__device__ __forceinline__ void store(uint*  out, uint v, int i=0, bool b=true) { if (b) __stg(out + i, v); }


// For unused code paths but the compiler still needs to process.
__device__ __forceinline__ void store(float* out, float2 v, int i=0, bool b=true) {}
__device__ __forceinline__ void store(ehalf* out, float2 v, int i=0, bool b=true) {}
__device__ __forceinline__ void store(bhalf* out, float2 v, int i=0, bool b=true) {}
__device__ __forceinline__ void store(float2* out, float v, int i=0, bool b=true) {}
__device__ __forceinline__ void store(ehalf2* out, float v, int i=0, bool b=true) {}
__device__ __forceinline__ void store(bhalf2* out, float v, int i=0, bool b=true) {}


__device__ __forceinline__ float ew_sum(float  a) { return a; }
__device__ __forceinline__ float ew_sum(float2 a) { return a.x + a.y; }
__device__ __forceinline__ float ew_sum(float4 a) { return (a.x + a.y) + (a.z + a.w); }
__device__ __forceinline__ float ew_sum(float8 v) { return ew_sum(v.a) + ew_sum(v.b); }

__device__ __forceinline__ float ew_max(float  a) { return a; }
__device__ __forceinline__ float ew_max(float2 a) { return fmaxf(a.x, a.y); }
__device__ __forceinline__ float ew_max(float4 a) { return fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w)); }
__device__ __forceinline__ float ew_max(float8 v) { return fmaxf(ew_max(v.a), ew_max(v.b)); }

__device__ __forceinline__ float  ew_warp_sum(float  a, int i) { a   += shfl_xor(a, i); return a; }
__device__ __forceinline__ float2 ew_warp_sum(float2 a, int i) { a.x += shfl_xor(a.x, i); a.y += shfl_xor(a.y, i); return a; }
__device__ __forceinline__ float4 ew_warp_sum(float4 a, int i) { a.x += shfl_xor(a.x, i); a.y += shfl_xor(a.y, i); a.z += shfl_xor(a.z, i); a.w += shfl_xor(a.w, i); return a; }
__device__ __forceinline__ float8 ew_warp_sum(float8 v, int i) { v.a = ew_warp_sum(v.a, i); v.b = ew_warp_sum(v.b, i); return v; }

__device__ __forceinline__ float  ew_warp_max(float  a, int i) { a   = fmaxf(a,   shfl_xor(a,   i)); return a; }
__device__ __forceinline__ float2 ew_warp_max(float2 a, int i) { a.x = fmaxf(a.x, shfl_xor(a.x, i)); a.y = fmaxf(a.y, shfl_xor(a.y, i)); return a; }
__device__ __forceinline__ float4 ew_warp_max(float4 a, int i) { a.x = fmaxf(a.x, shfl_xor(a.x, i)); a.y = fmaxf(a.y, shfl_xor(a.y, i)); a.z = fmaxf(a.z, shfl_xor(a.z, i)); a.w = fmaxf(a.w, shfl_xor(a.w, i)); return a; }
__device__ __forceinline__ float8 ew_warp_max(float8 v, int i) { v.a = ew_warp_max(v.a, i); v.b = ew_warp_max(v.b, i); return v; }

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
__device__ __forceinline__ float8 name(float8 x) { float8 r; UNARY_VEC4(impl,r.a,x.a); UNARY_VEC4(impl,r.b,x.b); return r; }

#define MATH_Z_XA(name, impl) \
__device__ __forceinline__ float  name(float  x, float a) { return impl(x,a); } \
__device__ __forceinline__ float2 name(float2 x, float a) { float2 r; BINARY_VEC2_S(impl,r,x,a); return r; } \
__device__ __forceinline__ float4 name(float4 x, float a) { float4 r; BINARY_VEC4_S(impl,r,x,a); return r; } \
__device__ __forceinline__ float8 name(float8 x, float a) { float8 r; BINARY_VEC4_S(impl,r.a,x.a,a); BINARY_VEC4_S(impl,r.b,x.b,a); return r; }


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

#define MATH_DZ_XA(name, impl) \
__device__ __forceinline__ float  name(float  dz, float  x, float a) { return impl(dz,x,a); } \
__device__ __forceinline__ float2 name(float2 dz, float2 x, float a) { float2 r; TERNARY_VEC2_S(impl,r,dz,x,a); return r; } \
__device__ __forceinline__ float4 name(float4 dz, float4 x, float a) { float4 r; TERNARY_VEC4_S(impl,r,dz,x,a); return r; } \
__device__ __forceinline__ float8 name(float8 dz, float8 x, float a) { float8 r; TERNARY_VEC4_S(impl,r.a,dz.a,x.a,a); TERNARY_VEC4_S(impl,r.b,dz.b,x.b,a); return r; }


MATH_Z_XY(ew_add,      _add)
MATH_Z_XY(ew_sub,      _sub)
MATH_Z_XY(ew_mul,      _mul)
MATH_Z_XY(ew_div,      _div)
MATH_Z_XY(ew_maximum, fmaxf)
MATH_Z_XY(ew_minimum, fminf)
MATH_Z_XY(ew_precise_sub, precise_sub)


MATH_Z_X(ew_abs,           fabsf)
MATH_Z_X(ew_neg,            _neg)
MATH_Z_X(ew_sqr,            _sqr)
MATH_Z_X(ew_rcp,     _rcp_approx)
MATH_Z_X(ew_sqrt,   _sqrt_approx)
MATH_Z_X(ew_rsqrt, _rsqrt_approx)
MATH_Z_X(ew_ex2,     _ex2_approx)
MATH_Z_X(ew_lg2,     _lg2_approx)
MATH_Z_X(ew_exp,     _exp_approx)
MATH_Z_X(ew_log,     _log_approx)
MATH_Z_X(ew_sig,            _sig)
MATH_Z_X(ew_tanh,          tanhf)
MATH_Z_X(ew_relu,          _relu)
MATH_Z_X(ew_zero_inf,  _zero_inf)
MATH_Z_X(ew_zero_nan,  _zero_nan)
MATH_Z_X(ew_zero_nan_inf, _zero_nan_inf)

MATH_Z_XA(ew_elu,    _elu)
MATH_Z_XA(ew_gelu,  _gelu)
MATH_Z_XA(ew_swish,_swish)

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

MATH_DZ_XA(ew_elu_grad,     _elu_grad)
MATH_DZ_XA(ew_gelu_grad,   _gelu_grad)
MATH_DZ_XA(ew_swish_grad, _swish_grad)
