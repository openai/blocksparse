#include "gpu_types.h"

__device__ __forceinline__ uint div64(uint value, uint magic, uint shift)
{
    // if the divisor is a power of 2 the magic will be 1 and it's just a simple right shift
    // Otherwise multiply by magic and right shift just the high bits
    uint result;
    asm("{                            \n\t"
        ".reg .pred p;                \n\t"
        ".reg .u64 res64;             \n\t"
        ".reg .u32 lo32, hi32;        \n\t"
        "setp.ne.s32 p, %2, 1;        \n\t"
        "mul.wide.u32 res64, %1, %2;  \n\t"
        "mov.b64 {lo32, hi32}, res64; \n\t"
        "selp.u32 hi32, hi32, %1, p;  \n\t"
        "shr.u32 %0, hi32, %3;        \n\t"
        "}" : "=r"(result) : "r"(value), "r"(magic), "r"(shift));
    return result;
}

__device__ __forceinline__ uint to_half2(float* val)
{
    uint ret;
    asm("{                     \n\t"
        ".reg .f16 a, b;       \n\t"
        "cvt.rn.f16.f32 a, %1; \n\t"
        "cvt.rn.f16.f32 b, %2; \n\t"
        "mov.b32 %0, {a, b};   \n\t"
        "}" : "=r"(ret) : "f"(val[0]),"f"(val[1]));
    return ret;
}

__device__ __forceinline__ uint2 to_half4(float4 v)
{
    uint2 r;
    asm("{\n\t"
        ".reg .f16 a, b, c, d; \n\t"
        "cvt.rn.f16.f32 a, %2; \n\t"
        "cvt.rn.f16.f32 b, %3; \n\t"
        "cvt.rn.f16.f32 c, %4; \n\t"
        "cvt.rn.f16.f32 d, %5; \n\t"
        "mov.b32 %0, {a, b};   \n\t"
        "mov.b32 %1, {c, d};   \n\t"
        "}" : "=r"(r.x), "=r"(r.y) : "f"(v.x),"f"(v.y), "f"(v.z),"f"(v.w));
    return r;
}

#define OP_N 0
#define OP_T 1

template <uint OP_A>
struct fragmentA
{
    uint x[8];
    __device__ __forceinline__ static uint get_idx(uint tid, uint stride, uint offset=0)
    {
        uint idx;
        if (OP_A == OP_T)
            idx = (tid & 3)*stride + (tid & 4)*2 + (tid & 16)/4 + offset;
        else
            idx = ((tid & 3) + (tid & 4)*2 + (tid & 16)/4) * stride + offset;
        return idx;
    }
    __device__ __forceinline__ void load(ehalf* hShare, uint idx, uint stride)
    {
        if (OP_A == OP_T)
            for (int i = 0; i < 4; i++)
                *(uint2*)&x[i*2] = *(uint2*)&hShare[idx + i*4*stride];
        else
            for (int i = 0; i < 2; i++)
                *(uint4*)&x[i*4] = *(uint4*)&hShare[idx + i*8];
    }
};

template <uint OP_B>
struct fragmentB
{
    uint x[8];
    __device__ __forceinline__ static uint get_idx(uint tid, uint stride, uint offset=0)
    {
        uint idx;
        if (OP_B == OP_N)
            idx = (tid & 3)*stride + (tid & 8)*1 + (tid & 16)/4 + offset;
        else
            idx = ((tid & 3) + (tid & 8)*1 + (tid & 16)/4) * stride + offset;
        return idx;
    }
    __device__ __forceinline__ void load(ehalf* hShare, uint idx, uint stride)
    {
        if (OP_B == OP_N)
            for (int i = 0; i < 4; i++)
                *(uint2*)&x[i*2] = *(uint2*)&hShare[idx + i*4*stride];
        else
            for (int i = 0; i < 2; i++)
                *(uint4*)&x[i*4] = *(uint4*)&hShare[idx + i*8];
    }
};

template <uint OP_A, uint OP_B>
struct fragmentC
{
    float x[8];
    __device__ __forceinline__ fragmentC()
    {
        for (int i = 0; i < 8; i++)
            asm volatile ("mov.b32 %0, 0;" : "=f"(x[i]) :);
            //x[i] = 0.0f;
    }
    __device__ __forceinline__ static uint get_idx(uint tid, uint stride, uint offset=0)
    {
        return ((tid & 1) + (tid & 4)*2 + (tid & 16)/4)*stride + (tid & 2) + (tid & 8) + offset;
    }
    __device__ __forceinline__ void store(ehalf* hShare, uint idx, uint stride)
    {
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                *(uint*)&hShare[idx + i*4 + j*2*stride] = to_half2(&x[i*4 + j*2]);
    }
    __device__ __forceinline__ void store(float* fShare, uint idx, uint stride)
    {
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                *(float2*)&fShare[idx + i*4 + j*2*stride] = *(float2*)&x[i*4 + j*2];
    }
    __device__ __forceinline__ void mma_sync(fragmentA<OP_A> &fragA, fragmentB<OP_B> &fragB)
    {
        if (OP_A == OP_N && OP_B == OP_N)
        {
            asm("wmma.mma.sync.m16n16k16.row.row.f32.f32            \n\t"
                "        {  %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7 },\n\t"
                "        {  %8,  %9, %10, %11, %12, %13, %14, %15 },\n\t"
                "        { %16, %17, %18, %19, %20, %21, %22, %23 },\n\t"
                "        {  %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7 };\n\t" :
                "+f"(x[0]), "+f"(x[1]), "+f"(x[2]), "+f"(x[3]), "+f"(x[4]), "+f"(x[5]), "+f"(x[6]), "+f"(x[7]) :
                "r"(fragA.x[0]), "r"(fragA.x[1]), "r"(fragA.x[2]), "r"(fragA.x[3]), "r"(fragA.x[4]), "r"(fragA.x[5]), "r"(fragA.x[6]), "r"(fragA.x[7]),
                "r"(fragB.x[0]), "r"(fragB.x[1]), "r"(fragB.x[2]), "r"(fragB.x[3]), "r"(fragB.x[4]), "r"(fragB.x[5]), "r"(fragB.x[6]), "r"(fragB.x[7]));
        }
        if (OP_A == OP_T && OP_B == OP_N)
        {
            asm("wmma.mma.sync.m16n16k16.col.row.f32.f32            \n\t"
                "        {  %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7 },\n\t"
                "        {  %8,  %9, %10, %11, %12, %13, %14, %15 },\n\t"
                "        { %16, %17, %18, %19, %20, %21, %22, %23 },\n\t"
                "        {  %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7 };\n\t" :
                "+f"(x[0]), "+f"(x[1]), "+f"(x[2]), "+f"(x[3]), "+f"(x[4]), "+f"(x[5]), "+f"(x[6]), "+f"(x[7]) :
                "r"(fragA.x[0]), "r"(fragA.x[1]), "r"(fragA.x[2]), "r"(fragA.x[3]), "r"(fragA.x[4]), "r"(fragA.x[5]), "r"(fragA.x[6]), "r"(fragA.x[7]),
                "r"(fragB.x[0]), "r"(fragB.x[1]), "r"(fragB.x[2]), "r"(fragB.x[3]), "r"(fragB.x[4]), "r"(fragB.x[5]), "r"(fragB.x[6]), "r"(fragB.x[7]));
        }
        if (OP_A == OP_N && OP_B == OP_T)
        {
            asm("wmma.mma.sync.m16n16k16.row.col.f32.f32            \n\t"
                "        {  %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7 },\n\t"
                "        {  %8,  %9, %10, %11, %12, %13, %14, %15 },\n\t"
                "        { %16, %17, %18, %19, %20, %21, %22, %23 },\n\t"
                "        {  %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7 };\n\t" :
                "+f"(x[0]), "+f"(x[1]), "+f"(x[2]), "+f"(x[3]), "+f"(x[4]), "+f"(x[5]), "+f"(x[6]), "+f"(x[7]) :
                "r"(fragA.x[0]), "r"(fragA.x[1]), "r"(fragA.x[2]), "r"(fragA.x[3]), "r"(fragA.x[4]), "r"(fragA.x[5]), "r"(fragA.x[6]), "r"(fragA.x[7]),
                "r"(fragB.x[0]), "r"(fragB.x[1]), "r"(fragB.x[2]), "r"(fragB.x[3]), "r"(fragB.x[4]), "r"(fragB.x[5]), "r"(fragB.x[6]), "r"(fragB.x[7]));
        }
    }
};