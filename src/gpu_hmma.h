#ifndef GPU_HMMA_H
#define GPU_HMMA_H

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

__device__ __forceinline__ ushort to_half(float* v)
{
    ushort ret;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(ret) : "f"(v[0]) );
    return ret;
}
__device__ __forceinline__ uint to_half2(float* v)
{
    uint ret;
    asm("{                     \n\t"
        ".reg .f16 a, b;       \n\t"
        "cvt.rn.f16.f32 a, %1; \n\t"
        "cvt.rn.f16.f32 b, %2; \n\t"
        "mov.b32 %0, {a, b};   \n\t"
        "}" : "=r"(ret) : "f"(v[0]),"f"(v[1]));
    return ret;
}
__device__ __forceinline__ uint to_half2(float2 v)
{
    uint ret;
    asm("{                     \n\t"
        ".reg .f16 a, b;       \n\t"
        "cvt.rn.f16.f32 a, %1; \n\t"
        "cvt.rn.f16.f32 b, %2; \n\t"
        "mov.b32 %0, {a, b};   \n\t"
        "}" : "=r"(ret) : "f"(v.x),"f"(v.y));
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

__device__ __forceinline__ ushort load_half(const ehalf* a)
{
    ushort r;
    asm volatile ("ld.global.nc.u16 %0, [%1];" : "=h"(r) : "l"(a));
    return r;
}
__device__ __forceinline__ uint  load_half2(const ehalf* a)
{
    uint r;
    asm volatile ("ld.global.nc.u32 %0, [%1];" : "=r"(r) : "l"(a));
    return r;
}
__device__ __forceinline__ uint2 load_half4(const ehalf* a)
{
    uint2 r;
    asm volatile ("ld.global.nc.v2.u32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(a));
    return r;
}
__device__ __forceinline__ uint4 load_half8(const ehalf* a)
{
    uint4 r;
    asm volatile ("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(a));
    return r;
}

__device__ __forceinline__ void store_half(const ehalf* a, ushort v)
{
    asm volatile ("st.global.wb.u16 [%0], %1;" :: "l"(a), "h"(v));
}
__device__ __forceinline__ void store_half2(const ehalf* a, uint  v)
{
    asm volatile ("st.global.wb.u32 [%0], %1;" :: "l"(a), "r"(v));
}
__device__ __forceinline__ void store_half4(const ehalf* a, uint2 v)
{
    asm volatile ("st.global.wb.v2.u32 [%0], {%1, %2};" :: "l"(a), "r"(v.x), "r"(v.y));
}
__device__ __forceinline__ void store_half8(const ehalf* a, uint4 v)
{
    asm volatile ("st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

__device__ __forceinline__ void zero_half2(const ehalf* a)
{
    asm volatile ("st.global.wb.u32 [%0], 0;" :: "l"(a));
}
__device__ __forceinline__ void zero_half4(const ehalf* a)
{
    asm volatile ("st.global.wb.v2.u32 [%0], {0, 0};" :: "l"(a));
}
__device__ __forceinline__ void zero_half8(const ehalf* a)
{
    asm volatile ("st.global.wb.v4.u32 [%0], {0, 0, 0, 0};" :: "l"(a));
}

__device__ __forceinline__ void reduce_half2(const ehalf* a, uint v)
{
# if CUDA_VERSION >= 9020
    asm volatile ("red.gpu.global.add.noftz.f16x2 [%0], %1;" :: "l"(a), "r"(v) :);
# else
    // Not enabled in older versions of cuda.. just let the lib compile
    // TODO: thwow a warning here.
    asm volatile ("st.global.wb.u32 [%0], %1;" :: "l"(a), "r"(v));
# endif
}

__device__ __forceinline__ float  ld_shared_float1(uint a)
{
    float v;
    asm volatile ("ld.shared.f32 %0, [%1];"  : "=f"(v) : "r"(a*4));
    return v;
}
__device__ __forceinline__ float2 ld_shared_float2(uint a)
{
    float2 v;
    asm volatile ("ld.shared.v2.f32 {%0, %1}, [%2];"  : "=f"(v.x),"=f"(v.y) : "r"(a*4));
    return v;
}
__device__ __forceinline__ float4 ld_shared_float4(uint a)
{
    float4 v;
    asm volatile ("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"  : "=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w) : "r"(a*4));
    return v;
}

#define OP_N 0
#define OP_T 1

#define M16N16K16 0
#define M8N32K16  1
#define M8N8K16   2 // run inside of M16N16K16


template <uint OP_A, uint TILE>
struct fragmentA
{
    uint x[8];
    __device__ __forceinline__ static uint get_idx(uint tid, uint stride, uint offset=0)
    {
        uint idx = 0;
        if (TILE == M16N16K16)
        {
            if (OP_A == OP_T)
                idx = (tid & 3)*stride + (tid & 4)*2 + (tid & 16)/4 + offset;
            else
                idx = ((tid & 3) + (tid & 4)*2 + (tid & 16)/4) * stride + offset;
        }
        else if (TILE == M8N32K16)
        {
            if (OP_A == OP_T)
                idx = (tid & 3)*stride + (tid & 16)/4 + offset;
            else
                idx = ((tid & 3) + (tid & 16)/4) * stride + offset;
        }
        else if (TILE == M8N8K16)
        {
            if (OP_A == OP_N)
                idx = ((tid & 3) + (tid & 16)/4) * stride + offset;
        }
        return idx;
    }
    __device__ __forceinline__ void load(ehalf* hShare, uint idx, uint stride)
    {
        if (OP_A == OP_T)
            for (int i = 0; i < 4; i++)
                *(uint2*)&x[i*2] = *(uint2*)&hShare[idx + i*4*stride];
        else
            for (int i = 0; i < 2; i++)
                // for M8N32K16 we concatonate two 8x8 blocks together to form an 8x16 block
                // but the k dim is not contiguous accross the 2 blocks
                *(uint4*)&x[i*4] = *(uint4*)&hShare[idx + (TILE == M8N32K16 ? i*64 : i*8)];
    }
};

template <uint OP_B, uint TILE>
struct fragmentB
{
    uint x[8];
    __device__ __forceinline__ static uint get_idx(uint tid, uint stride, uint offset=0)
    {
        uint idx = 0;
        if (TILE == M16N16K16)
        {
            if (OP_B == OP_N)
                idx = (tid & 3)*stride + (tid & 8)*1 + (tid & 16)/4 + offset;
            else
                idx = ((tid & 3) + (tid & 8)*1 + (tid & 16)/4) * stride + offset;
        }
        else if (TILE == M8N32K16)
        {
            if (OP_B == OP_N)
                idx = (tid & 3)*stride + (tid & 4)*2 + (tid & 8)*2 + (tid & 16)/4 + offset;
        }
        else if (TILE == M8N8K16)
        {
            if (OP_B == OP_T)
                idx = ((tid & 3) + (tid & 16)/4) * stride + offset;
        }
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

template <uint OP_A, uint OP_B, uint TILE>
struct fragmentC
{
    float x[8];
    __device__ __forceinline__ fragmentC()
    {
        for (int i = 0; i < 8; i++)
            //x[i] = 0.0f;
            asm volatile ("mov.b32 %0, 0;" : "=f"(x[i]) :);
    }
    __device__ __forceinline__ static uint get_idx(uint tid, uint stride, uint offset=0)
    {
        uint idx = 0;
        if (TILE == M16N16K16)
            idx = ((tid & 1) + (tid & 4)*2 + (tid & 16)/4)*stride + (tid & 2) + (tid & 8) + offset;
        else if (TILE == M8N32K16)
            idx = ((tid & 1) + (tid & 16)/4)*stride + (tid & 2) + (tid & 4)*2 + (tid & 8)*2 + offset;
        else if (TILE == M8N8K16)
            idx = ((tid & 1) + (tid & 16)/4)*stride + (tid & 2) + offset;
        return idx;
    }
    __device__ __forceinline__ void store(ehalf* hShare, uint idx, uint stride)
    {
        if (TILE == M8N8K16)
        {
            // only upper left 8x8 quandrant of 16x16 tile is valid (threads 0,1,2,3,16,17,18,19)
            bool valid = (threadIdx.x & 12) == 0;
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                {
                    uint x2 = to_half2(&x[i*4 + j*2]);
                    if (valid)
                        *(uint*)&hShare[idx + i*4 + j*2*stride] = x2;
                }
        }
        else
        {
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    *(uint*)&hShare[idx + i*4 + j*2*stride] = to_half2(&x[i*4 + j*2]);
        }
    }
    __device__ __forceinline__ void store(float* fShare, uint idx, uint stride)
    {
        if (TILE == M8N8K16)
        {
            // only upper left 8x8 quandrant of 16x16 tile is valid (threads 0,1,2,3,16,17,18,19)
            bool valid = (threadIdx.x & 12) == 0;
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    if (valid)
                        *(float2*)&fShare[idx + i*4 + j*2*stride] = *(float2*)&x[i*4 + j*2];
        }
        else
        {
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    *(float2*)&fShare[idx + i*4 + j*2*stride] = *(float2*)&x[i*4 + j*2];
        }
    }


#if CUDA_VERSION > 9020
#define MMA_ALIGNED ".aligned"
#else
#define MMA_ALIGNED ""
#endif
#define MMA_SYNC(tile, opA, opB) \
    asm("wmma.mma.sync" MMA_ALIGNED "." tile "." opA "." opB ".f32.f32     \n\t"   \
        "        {  %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7 },\n\t"   \
        "        {  %8,  %9, %10, %11, %12, %13, %14, %15 },\n\t"   \
        "        { %16, %17, %18, %19, %20, %21, %22, %23 },\n\t"   \
        "        {  %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7 };\n\t" : \
        "+f"(     x[0]), "+f"(     x[1]), "+f"(     x[2]), "+f"(     x[3]), "+f"(     x[4]), "+f"(     x[5]), "+f"(     x[6]), "+f"(     x[7]) : \
        "r"(fragA.x[0]), "r"(fragA.x[1]), "r"(fragA.x[2]), "r"(fragA.x[3]), "r"(fragA.x[4]), "r"(fragA.x[5]), "r"(fragA.x[6]), "r"(fragA.x[7]),  \
        "r"(fragB.x[0]), "r"(fragB.x[1]), "r"(fragB.x[2]), "r"(fragB.x[3]), "r"(fragB.x[4]), "r"(fragB.x[5]), "r"(fragB.x[6]), "r"(fragB.x[7]))

    __device__ __forceinline__ void mma_sync(fragmentA<OP_A,TILE> &fragA, fragmentB<OP_B,TILE> &fragB)
    {
        if (TILE == M16N16K16 || TILE == M8N8K16)
        {
            if (OP_A == OP_N && OP_B == OP_N) MMA_SYNC("m16n16k16", "row", "row");
            if (OP_A == OP_T && OP_B == OP_N) MMA_SYNC("m16n16k16", "col", "row");
            if (OP_A == OP_N && OP_B == OP_T) MMA_SYNC("m16n16k16", "row", "col");
        }
        else if (TILE == M8N32K16)
        {
            // m8n32k16 flag doesn't change mma behavior and isn't backwards compatable with cuda 9.0
            if (OP_A == OP_N && OP_B == OP_N) MMA_SYNC("m16n16k16", "row", "row"); // m8n32k16
            if (OP_A == OP_T && OP_B == OP_N) MMA_SYNC("m16n16k16", "col", "row"); // m8n32k16
        }
    }
};

__device__ __forceinline__ void mma_m8n8k4_nn(float* acc, ehalf4 f4, ehalf4 i4)
{
    asm("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 \n\t"
        "    {  %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7 }, \n\t"
        "    {  %8,  %9 }, { %10, %11 },                 \n\t"
        "    {  %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7 }; \n\t" :
        "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]),
        "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]) :
        "r"(f4.x), "r"(f4.y), "r"(i4.x), "r"(i4.y));
}

__device__ __forceinline__ void mma_m8n8k8_nt(float* acc, ehalf8 e8, ehalf8 i8)
{
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \n\t"
        "    {  %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7 }, \n\t"
        "    {  %8,  %9 }, { %10, %11 },                 \n\t"
        "    {  %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7 }; \n\t"
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \n\t"
        "    {  %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7 }, \n\t"
        "    { %12, %13 }, { %14, %15 },                 \n\t"
        "    {  %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7 }; \n\t" :
        "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]),
        "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]) :
        "r"(e8.x), "r"(e8.y), "r"(i8.x), "r"(i8.y),
        "r"(e8.z), "r"(e8.w), "r"(i8.z), "r"(i8.w));
}

#endif // GPU_HMMA_H