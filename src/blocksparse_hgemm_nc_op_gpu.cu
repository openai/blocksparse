#if GOOGLE_CUDA

#include "ew_op_gpu.h"
#include "gpu_hmma.h"
#include <stdio.h>

#if __CUDA_ARCH__ >= 700

template <bool N64>
__device__  __noinline__  void stg_64x64x64_nx(ehalf* Y, uint offsetY, uint loadY, uint N, uint K, uint n, uint i)
{
    for (uint j = 0; j < 2; j++)
        if (N64 || n + i*16 + j*32 < N)
            store_half4(Y + (offsetY + (i*16 + j*32)*K), to_half4(
                ew_add(
                    ld_shared_float4(loadY + j*64 +   0),
                    ld_shared_float4(loadY + j*64 + 128))
            ));
}

template <bool N64>
__device__  __noinline__  void red_64x64x64_nx(ehalf* Y, uint offsetY, uint loadY, uint N, uint K, uint n, uint i, uint stdC)
{
    for (uint j = 0; j < 2; j++)
        for (uint k = 0; k < 2; k++)
        {
            uint sum2 = to_half2(
                ew_add(
                    ld_shared_float2(loadY + k*8*stdC + j*64 +   0),
                    ld_shared_float2(loadY + k*8*stdC + j*64 + 128)
                )
            );
            uint offset = offsetY + K*(j*32 + k*8 + i*16);
            if (N64 || n + j*32 + k*8 < N)
                reduce_half2(Y + offset, sum2);
        }
}
template <uint OP_B, bool N64, bool GATED>
__global__ void __launch_bounds__(256) hgemm_blocksparse_64x64x64_nx_dsd(
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    const ehalf* __restrict__ X,
    const ehalf* __restrict__ W,
          ehalf*              Y,
    uint* Lock, uint locks, uint N, uint C, uint K, uint blk_a, uint blk_b, uint blk_N)
{
    const uint stdA = 64 + 8;
    const uint stdB = 64 + (OP_B == OP_N ? 16 : 8);
    const uint stdC = 256 + 4;

    __shared__ ehalf hShare[(stdA + stdB)*64];
    float* fShare = (float*)hShare;
    uint2* LutOffsets = (uint2*)&hShare[(stdA + stdB)*64];

    uint tid    = threadIdx.x;
    uint idx_ab = blockIdx.x;
    uint idx_B  = blockIdx.y;
    uint idx_A  = blockIdx.z;

    uint idx_L = idx_A * blk_a + idx_ab / blk_b;
    uint idx_N = idx_B * blk_b + idx_ab % blk_b;

    uint4 lut_head = ((const uint4*)Lut)[idx_L];
    uint lut_offset = lut_head.x;
    uint lut_size   = lut_head.y;
    uint idx_K      = lut_head.z;
    uint idx_Lock   = lut_head.w;

    uint tx = tid % 8;
    uint ty = tid / 8;

    if (lut_size > 0)
    {
        uint* Gates = (uint*)&LutOffsets[lut_size];

        // prefetch the lut and gate data into shared
        Lut += lut_offset;
        #pragma unroll 1
        for (uint i = tid; i < lut_size; i += 256)
        {
            uint2 entry = Lut[i];

            if (GATED)
            {
                float gate = Gate[entry.y];
                uint gate2;
                asm("{                         \n\t"
                    ".reg .f16 gate;           \n\t"
                    "cvt.rn.f16.f32 gate, %1;  \n\t"
                    "mov.b32 %0, {gate, gate}; \n\t"
                    "}" : "=r"(gate2) : "f"(gate));
                Gates[i] = gate2;
            }
            else
                Gates[i] = 1;

            entry.x *= 64;
            entry.y *= 64*64;
            LutOffsets[i] = entry;
        }
        __syncthreads();

        uint storA = ty*stdA + tx*8;
        uint storB = ty*stdB + tx*8 + stdA*64;

        uint loadA = fragmentA<OP_N,M16N16K16>::get_idx(tid, stdA, (tid & 128)*32/128 + (tid & 64)*stdA*32/64);
        uint loadB = fragmentB<OP_B,M16N16K16>::get_idx(tid, stdB, (tid & 128)*(OP_B == OP_N ? stdB : 1)*32/128 + (tid & 32)*(OP_B == OP_N ? 1 : stdB) + stdA*64);

        uint       n = idx_N*64 + ty;
        uint offsetA = n*C + tx*8;

        if (!N64)
        {
            asm(".reg .pred pn00, pn32;\n\t"::);
            asm("setp.lt.u32 pn00, %0, %1;\n\t" :: "r"(n + 0*32), "r"(N));
            asm("setp.lt.u32 pn32, %0, %1;\n\t" :: "r"(n + 1*32), "r"(N));
        }
        asm("mov.b32 %0, %0;" : "+r"(idx_N) : );
        asm("mov.b32 %0, %0;" : "+r"(loadA) : );
        asm("mov.b32 %0, %0;" : "+r"(loadB) : );
        asm("mov.b32 %0, %0;" : "+r"(offsetA) : );

        fragmentC<OP_N,OP_B,M16N16K16> fragC[2][2];

        uint idx_lut = 0;
        #pragma unroll 1
        do
        {
            uint gate = Gates[idx_lut];

            if (gate != 0)
            {
                uint2 entry = LutOffsets[idx_lut];

                const ehalf* pW = W + (entry.y + tid*8);
                uint4 b00 = load_half8(pW + 0*32*64);
                uint4 b32 = load_half8(pW + 1*32*64);
                uint4 a00, a32;
                entry.x += offsetA;

                if (N64)
                {
                    a00 = load_half8(X + (entry.x +  0*C));
                    a32 = load_half8(X + (entry.x + 32*C));
                }
                else
                {
                    asm("mov.b64  {%0, %1}, 0; \n\t"
                        "mov.b64  {%2, %3}, 0; \n\t"
                        "mov.b64  {%4, %5}, 0; \n\t"
                        "mov.b64  {%6, %7}, 0; \n\t"
                        "@pn00 ld.global.nc.v4.u32 { %0,  %1,  %2,  %3}, [%8];\n\t"
                        "@pn32 ld.global.nc.v4.u32 { %4,  %5,  %6,  %7}, [%9];\n\t" :
                        "=r"(a00.x), "=r"(a00.y), "=r"(a00.z), "=r"(a00.w),
                        "=r"(a32.x), "=r"(a32.y), "=r"(a32.z), "=r"(a32.w) :
                        "l"(X + (entry.x +  0*C)),
                        "l"(X + (entry.x + 32*C)));
                }
                if (GATED)
                {
                    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(b00.x) : "r"(gate));
                    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(b00.y) : "r"(gate));
                    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(b00.z) : "r"(gate));
                    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(b00.w) : "r"(gate));
                    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(b32.x) : "r"(gate));
                    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(b32.y) : "r"(gate));
                    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(b32.z) : "r"(gate));
                    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(b32.w) : "r"(gate));
                }

                __syncthreads();
                *(uint4*)&hShare[storB + 0*32*stdB] = b00;
                *(uint4*)&hShare[storB + 1*32*stdB] = b32;
                *(uint4*)&hShare[storA + 0*32*stdA] = a00;
                *(uint4*)&hShare[storA + 1*32*stdA] = a32;
                __syncthreads();

                fragmentA<OP_N,M16N16K16> fragA[2];
                fragmentB<OP_B,M16N16K16> fragB[2];
                for (uint k = 0; k < 2; k++)
                {
                    for (uint i = 0; i < 2; i++)
                    {
                        fragA[i].load(hShare, loadA + k*16 + i*16*stdA, stdA);
                        fragB[i].load(hShare, loadB + (OP_B == OP_N ? stdB : 1)*k*16 + (OP_B == OP_N ? 1 : stdB)*i*16, stdB);
                    }
                    for (uint i = 0; i < 2; i++)
                        for (uint j = 0; j < 2; j++)
                            fragC[i][j].mma_sync(fragA[i], fragB[j]);
                }
            }

        } while (++idx_lut < lut_size);

        uint txc = tid % 16;
        uint tyc = tid / 16;

        n = idx_N*64 + tyc;
        uint loadY   = tyc*stdC + txc*4;
        uint storY   = fragmentC<OP_N,OP_B,M16N16K16>::get_idx(tid, stdC, tid & 224);
        uint offsetY = n*K + idx_K*64 + txc*4;

        if (idx_Lock == 0)
        {
            for (uint i = 0; i < 2; i++)
            {
                __syncthreads();
                for (uint j = 0; j < 2; j++)
                    fragC[i][j].store(fShare, storY + j*16, stdC);
                __syncthreads();

                stg_64x64x64_nx<N64>(Y, offsetY, loadY, N, K, n, i);
            }
        }
        else
        {
            Lock += idx_N*locks + idx_Lock - 1;

            // Critial Section
            if (tid == 0)
                while (atomicCAS(Lock, 0, 1) != 0);
            __syncthreads();

            uint* Count   = Lock + locks * blk_N;
            uint  count   = *Count;
            __syncthreads();

            if (count == 0)
            {
                if (tid == 0)
                    *Count = 1;

                // first block to get here just writes out to init the memory
                for (uint i = 0; i < 2; i++)
                {
                    __syncthreads();
                    for (uint j = 0; j < 2; j++)
                        fragC[i][j].store(fShare, storY + j*16, stdC);
                    __syncthreads();

                    stg_64x64x64_nx<N64>(Y, offsetY, loadY, N, K, n, i);
                }
            }
            else
            {
                txc = tid % 32;
                tyc = tid / 32;

                n       = idx_N*64 + tyc;
                loadY   = tyc*stdC + txc*2;
                offsetY = n*K + idx_K*64 + txc*2;

                // subsequent blocks must accumulate
                for (uint i = 0; i < 2; i++)
                {
                    __syncthreads();
                    for (uint j = 0; j < 2; j++)
                        fragC[i][j].store(fShare, storY + j*16, stdC);
                    __syncthreads();

                    red_64x64x64_nx<N64>(Y, offsetY, loadY, N, K, n, i, stdC);
                }
            }
            __threadfence();
            __syncthreads();

            if (tid == 0)
                atomicExch(Lock, 0);
            // End Critial Section
        }
    }
    else
    {
        uint n = idx_N*64 + ty;
        uint offsetY = n*K + idx_K*64 + tx*8;

        if (N64 || n +  0 < N) zero_half8(Y + (offsetY +  0*K));
        if (N64 || n + 32 < N) zero_half8(Y + (offsetY + 32*K));
    }
}


template <bool N64>
__device__  __noinline__  void stg_64x32x32_nx(ehalf* Y, uint offsetY, uint loadY, uint N, uint K, uint n, uint i)
{
    for (uint j = 0; j < 2; j++)
        if (N64 || n + i*16 + j*32 < N)
            store_half4(Y + (offsetY + (j*32 + i*16)*K), to_half4(ew_add(
                ld_shared_float4(loadY + j*32 +  0),
                ld_shared_float4(loadY + j*32 + 64))));
}
template <bool N64>
__device__  __noinline__  void red_64x32x32_nx(ehalf* Y, uint offsetY, uint loadY, uint N, uint K, uint n, uint i, uint stdC)
{
    for (uint j = 0; j < 2; j++)
        for (uint k = 0; k < 2; k++)
            if (N64 || n + j*32 + k*8 < N)
                reduce_half2(Y + (offsetY + K*(j*32 + k*8 + i*16)), to_half2(ew_add(
                    ld_shared_float2(loadY + k*8*stdC + j*32 +  0),
                    ld_shared_float2(loadY + k*8*stdC + j*32 + 64))));
}
template <uint OP_B, bool N64, bool GATED>
__global__ void __launch_bounds__(128) hgemm_blocksparse_64x32x32_nx_dsd(
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    const ehalf* __restrict__ X,
    const ehalf* __restrict__ W,
          ehalf*              Y,
    uint* Lock, uint locks, uint N, uint C, uint K, uint blk_a, uint blk_b, uint blk_N)
{
    const uint stdA = 32+8;
    const uint stdB = 32 + (OP_B == OP_N ? 16 : 8);
    const uint stdC = 132;

    __shared__ float fShare[stdC*16]; // stdC*16*4 > (stdA*64 + stdB*32)*2
    ehalf* hShare = (ehalf*)fShare;
    uint2* LutOffsets = (uint2*)&hShare[stdA*64 + stdB*32];

    uint tid    = threadIdx.x;
    uint idx_ab = blockIdx.x;
    uint idx_B  = blockIdx.y;
    uint idx_A  = blockIdx.z;

    uint idx_L = idx_A * blk_a + idx_ab / blk_b;
    uint idx_N = idx_B * blk_b + idx_ab % blk_b;

    uint4 lut_head = ((const uint4*)Lut)[idx_L];
    uint lut_offset = lut_head.x;
    uint lut_size   = lut_head.y;
    uint idx_K      = lut_head.z;
    uint idx_Lock   = lut_head.w;

    uint tx = tid % 8;
    uint ty = tid / 8;

    if (lut_size > 0)
    {
        uint* Gates = (uint*)&LutOffsets[lut_size];

        // prefetch the lut and gate data into shared
        Lut += lut_offset;
        #pragma unroll 1
        for (uint i = tid; i < lut_size; i += 128)
        {
            uint2 entry = Lut[i];

            if (GATED)
            {
                float gate = Gate[entry.y];
                uint gate2;
                asm("{                         \n\t"
                    ".reg .f16 gate;           \n\t"
                    "cvt.rn.f16.f32 gate, %1;  \n\t"
                    "mov.b32 %0, {gate, gate}; \n\t"
                    "}" : "=r"(gate2) : "f"(gate));
                Gates[i] = gate2;
            }
            else
                Gates[i] = 1;

            entry.x *= 32;
            entry.y *= 32*32;
            LutOffsets[i] = entry;
        }
        __syncthreads();

        uint storA = ty*stdA + tx*4;
        uint storB = ty*stdB + tx*4 + stdA*64;

        uint loadA = fragmentA<OP_N,M16N16K16>::get_idx(tid, stdA, (tid & 64)*16/64 + (tid & 32)*stdA);
        uint loadB = fragmentB<OP_B,M16N16K16>::get_idx(tid, stdB, (tid & 64)*(OP_B == OP_N ? stdB : 1)*16/64 + stdA*64);

        uint       n = idx_N*64 + ty;
        uint offsetA = n*C + tx*4;

        if (!N64)
        {
            asm(".reg .pred pn<4>;\n\t"::);
            asm("setp.lt.u32 pn0, %0, %1;\n\t" :: "r"(n + 0*16), "r"(N));
            asm("setp.lt.u32 pn1, %0, %1;\n\t" :: "r"(n + 1*16), "r"(N));
            asm("setp.lt.u32 pn2, %0, %1;\n\t" :: "r"(n + 2*16), "r"(N));
            asm("setp.lt.u32 pn3, %0, %1;\n\t" :: "r"(n + 3*16), "r"(N));
        }
        asm("mov.b32 %0, %0;" : "+r"(loadA) : );
        asm("mov.b32 %0, %0;" : "+r"(loadB) : );
        asm("mov.b32 %0, %0;" : "+r"(offsetA) : );

        fragmentC<OP_N,OP_B,M16N16K16> fragC[2][2];

        uint idx_lut = 0;
        #pragma unroll 1
        do
        {
            uint gate = Gates[idx_lut];

            if (gate != 0)
            {
                uint2 entry = LutOffsets[idx_lut];

                const ehalf* pW = W + (entry.y + tid*4);
                uint2 b00 = load_half4(pW + 0*16*32);
                uint2 b16 = load_half4(pW + 1*16*32);
                uint2 a00, a16, a32, a48;

                entry.x += offsetA;
                if (N64)
                {
                    a00 = load_half4(X + (entry.x + 0*16*C));
                    a16 = load_half4(X + (entry.x + 1*16*C));
                    a32 = load_half4(X + (entry.x + 2*16*C));
                    a48 = load_half4(X + (entry.x + 3*16*C));
                }
                else
                {
                    asm("mov.b64  {%0,  %1}, 0; \n\t"
                        "mov.b64  {%2,  %3}, 0; \n\t"
                        "mov.b64  {%4,  %5}, 0; \n\t"
                        "mov.b64  {%6,  %7}, 0; \n\t"
                        "@pn0 ld.global.nc.v2.u32 {%0, %1}, [ %8];\n\t"
                        "@pn1 ld.global.nc.v2.u32 {%2, %3}, [ %9];\n\t"
                        "@pn2 ld.global.nc.v2.u32 {%4, %5}, [%10];\n\t"
                        "@pn3 ld.global.nc.v2.u32 {%6, %7}, [%11];\n\t" :
                        "=r"(a00.x), "=r"(a00.y), "=r"(a16.x), "=r"(a16.y),
                        "=r"(a32.x), "=r"(a32.y), "=r"(a48.x), "=r"(a48.y) :
                        "l"(X + (entry.x + 0*16*C)),
                        "l"(X + (entry.x + 1*16*C)),
                        "l"(X + (entry.x + 2*16*C)),
                        "l"(X + (entry.x + 3*16*C)));
                }

                if (GATED)
                {
                    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(b00.x) : "r"(gate));
                    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(b00.y) : "r"(gate));
                    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(b16.x) : "r"(gate));
                    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(b16.y) : "r"(gate));
                }

                __syncthreads();
                *(uint2*)&hShare[storB + 0*16*stdB] = b00;
                *(uint2*)&hShare[storB + 1*16*stdB] = b16;
                *(uint2*)&hShare[storA + 0*16*stdA] = a00;
                *(uint2*)&hShare[storA + 1*16*stdA] = a16;
                *(uint2*)&hShare[storA + 2*16*stdA] = a32;
                *(uint2*)&hShare[storA + 3*16*stdA] = a48;
                __syncthreads();

                fragmentA<OP_N,M16N16K16> fragA[2];
                fragmentB<OP_B,M16N16K16> fragB[2];
                for (uint i = 0; i < 2; i++)
                {
                    fragA[i].load(hShare, loadA + i*16*stdA, stdA);
                    fragB[i].load(hShare, loadB + (OP_B == OP_N ? 1 : stdB)*i*16, stdB);
                }

                for (uint i = 0; i < 2; i++)
                    for (uint j = 0; j < 2; j++)
                        fragC[i][j].mma_sync(fragA[i], fragB[j]);
            }

        } while (++idx_lut < lut_size);

        uint txc = tid % 8;
        uint tyc = tid / 8;

        n = idx_N*64 + tyc;
        uint loadY   = tyc*stdC + txc*4;
        uint storY   = fragmentC<OP_N,OP_B,M16N16K16>::get_idx(tid, stdC, tid & 96);
        uint offsetY = n*K + idx_K*32 + txc*4;

        if (idx_Lock == 0)
        {
            for (uint i = 0; i < 2; i++)
            {
                __syncthreads();
                for (uint j = 0; j < 2; j++)
                    fragC[i][j].store(fShare, storY + j*16, stdC);
                __syncthreads();

                stg_64x32x32_nx<N64>(Y, offsetY, loadY, N, K, n, i);
            }
        }
        else
        {
            Lock += idx_N*locks + idx_Lock - 1;

            // Critial Section
            if (tid == 0)
                while (atomicCAS(Lock, 0, 1) != 0);
            __syncthreads();

            uint* Count   = Lock + locks * blk_N;
            uint  count   = *Count;
            __syncthreads();

            if (count == 0)
            {
                if (tid == 0)
                    *Count = 1;

                // first block to get here just writes out to init the memory
                for (uint i = 0; i < 2; i++)
                {
                    __syncthreads();
                    for (uint j = 0; j < 2; j++)
                        fragC[i][j].store(fShare, storY + j*16, stdC);
                    __syncthreads();

                    stg_64x32x32_nx<N64>(Y, offsetY, loadY, N, K, n, i);
                }
            }
            else
            {
                txc = tid % 16;
                tyc = tid / 16;

                n       = idx_N*64 + tyc;
                loadY   = tyc*stdC + txc*2;
                offsetY = n*K + idx_K*32 + txc*2;

                // subsequent blocks must accumulate
                for (uint i = 0; i < 2; i++)
                {
                    __syncthreads();
                    for (uint j = 0; j < 2; j++)
                        fragC[i][j].store(fShare, storY + j*16, stdC);
                    __syncthreads();

                    red_64x32x32_nx<N64>(Y, offsetY, loadY, N, K, n, i, stdC);
                }
            }
            __threadfence();
            __syncthreads();

            if (tid == 0)
                atomicExch(Lock, 0);
            // End Critial Section
        }
    }
    else
    {
        uint txc = tid % 4;
        uint tyc = tid / 4;

        uint n = idx_N*64 + tyc;
        uint offsetY = n*K + idx_K*32 + txc*8;

        if (N64 || n +  0 < N) zero_half8(Y + (offsetY +  0*K));
        if (N64 || n + 32 < N) zero_half8(Y + (offsetY + 32*K));
    }
}

template <bool N64, bool GATED>
__global__ void __launch_bounds__(256,3) hgemm_blocksparse_64x64x64_tn_dds(
    struct Plist<ehalf,8> X,
    struct Plist<ehalf,8> DY,
    ehalf*                DW,
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    uint params8, uint N, uint C, uint K, uint loops, uint accumulate)
{
    const uint stdAB =  64 + 16;
    const uint stdC  = 256 +  4;

    __shared__ ehalf hShare[stdAB*2*64];
    float* fShare = (float*)hShare;

    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    float gate = GATED ? Gate[bid] : 1.0f;

    if (gate != 0.0f)
    {
        uint2 lut_head = Lut[bid];

        uint tx = tid % 8;
        uint ty = tid / 8;
        uint n0 = ty;

        uint idx_A = lut_head.x;
        uint idx_B = lut_head.y;
        uint offsetA0 = ty*C + idx_A*64 + tx*8;
        uint offsetB0 = ty*K + idx_B*64 + tx*8;
        uint storAB = ty*stdAB + tx*8;

        uint loadA = fragmentA<OP_T,M16N16K16>::get_idx(tid, stdAB, (tid & 128)*stdAB*32/128 + (tid & 64)/2);
        uint loadB = fragmentB<OP_N,M16N16K16>::get_idx(tid, stdAB, (tid & 128)*stdAB*32/128 + (tid & 32) + stdAB*64);

        fragmentC<OP_T,OP_N,M16N16K16> fragC[2][2];

        uint p8 = 0;
        #pragma unroll 1
        do
        {
            const ehalf* A0;
            const ehalf* B0;
            asm("ld.param.u64 %0, [%2 + 0x160];\n\t"
                "ld.param.u64 %1, [%2 + 0x1a0];"
                : "=l"(A0), "=l"(B0) : "r"(p8));
            p8 += 8;

            uint offsetA = offsetA0;
            uint offsetB = offsetB0;
            uint n       = n0;
            uint loop    = 0;

            #pragma unroll 1
            do
            {
                asm volatile (".pragma \"nounroll\";"::); // ptxas, don't get clever

                uint4 a00 = {0}, a32 = {0};
                uint4 b00 = {0}, b32 = {0};
                if (N64 || n +  0 < N)
                {
                    a00 = load_half8(A0 + (offsetA +  0*C));
                    b00 = load_half8(B0 + (offsetB +  0*K));
                }
                if (N64 || n + 32 < N)
                {
                    a32 = load_half8(A0 + (offsetA + 32*C));
                    b32 = load_half8(B0 + (offsetB + 32*K));
                }
                offsetA += 64*C;
                offsetB += 64*K;
                if (!N64)
                    n += 64;

                __syncthreads();
                *(uint4*)&hShare[storAB +  0*stdAB +  0*stdAB] = a00;
                *(uint4*)&hShare[storAB + 32*stdAB +  0*stdAB] = a32;
                *(uint4*)&hShare[storAB +  0*stdAB + 64*stdAB] = b00;
                *(uint4*)&hShare[storAB + 32*stdAB + 64*stdAB] = b32;
                __syncthreads();

                fragmentA<OP_T,M16N16K16> fragA[2];
                fragmentB<OP_N,M16N16K16> fragB[2];
                for (uint k = 0; k < 2; k++)
                {
                    for (uint i = 0; i < 2; i++)
                    {
                        fragA[i].load(hShare, loadA + k*16*stdAB + i*16, stdAB);
                        fragB[i].load(hShare, loadB + k*16*stdAB + i*16, stdAB);
                    }
                    for (uint i = 0; i < 2; i++)
                        for (uint j = 0; j < 2; j++)
                            fragC[i][j].mma_sync(fragA[i], fragB[j]);
                }

            } while (++loop < loops);

        } while (p8 < params8);

        asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid) :);
        asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(bid) :);

        uint storC = fragmentC<OP_T,OP_N,M16N16K16>::get_idx(tid, stdC, (tid & 224));

        if (accumulate)
        {
            tx = tid % 32;
            ty = tid / 32;
            uint loadC   = ty*stdC   + tx *2;
            uint offsetC = bid*64*64 + tid*2;

            for (uint i = 0; i < 2; i++)
            {
                __syncthreads();
                for (uint j = 0; j < 2; j++)
                    fragC[i][j].store(fShare, storC + j*16, stdC);
                __syncthreads();

                for (uint j = 0; j < 2; j++)
                    for (uint k = 0; k < 2; k++)
                    {
                        uint sum2 = to_half2(
                            ew_add(
                                ld_shared_float2(loadC + k*8*stdC + j*64 +   0),
                                ld_shared_float2(loadC + k*8*stdC + j*64 + 128)
                            )
                        );
                        reduce_half2(DW + offsetC + (i*16 + j*32 + k*8)*64, sum2);
                    }
            }
        }
        else
        {
            tx = tid % 16;
            ty = tid / 16;
            uint loadC   = ty*stdC   + tx *4;
            uint offsetC = bid*64*64 + tid*4;

            for (uint i = 0; i < 2; i++)
            {
                __syncthreads();
                for (uint j = 0; j < 2; j++)
                    fragC[i][j].store(fShare, storC + j*16, stdC);
                __syncthreads();

                for (uint j = 0; j < 2; j++)
                {
                    uint2 sum4 = to_half4(
                        ew_add(
                            ld_shared_float4(loadC + j*64 +   0),
                            ld_shared_float4(loadC + j*64 + 128)
                        )
                    );
                    store_half4(DW + offsetC + (i*16 + j*32)*64, sum4);
                }
            }
        }
    }
    else if (!accumulate) // gate == 0
    {
        DW += bid*64*64 + tid*8;
        zero_half8(DW +  0*64);
        zero_half8(DW + 32*64);
    }
}

template <bool N64, bool GATED>
__global__ void __launch_bounds__(128,6) hgemm_blocksparse_32x32x64_tn_dds(
    struct Plist<ehalf,8> X,
    struct Plist<ehalf,8> DY,
    ehalf*                DW,
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    uint params8, uint N, uint C, uint K, uint loops, uint accumulate)
{
    const uint stdAB = 32+16;
    const uint stdC  = 132;

    __shared__ ehalf hShare[stdAB*2*64];
    float* fShare = (float*)hShare;

    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    float gate = GATED ? Gate[bid] : 1.0f;

    if (gate != 0.0f)
    {
        uint2 lut_head = Lut[bid];

        uint tx = tid % 8;
        uint ty = tid / 8;
        uint n0 = ty;

        uint idx_A = lut_head.x;
        uint idx_B = lut_head.y;
        uint offsetA0 = ty*C + idx_A*32 + tx*4;
        uint offsetB0 = ty*K + idx_B*32 + tx*4;
        uint storAB = ty*stdAB + tx*4;

        uint loadA = fragmentA<OP_T,M16N16K16>::get_idx(tid, stdAB, (tid & 96)*stdAB/2);
        uint loadB = fragmentB<OP_N,M16N16K16>::get_idx(tid, stdAB, (tid & 96)*stdAB/2 + stdAB*64);

        fragmentC<OP_T,OP_N,M16N16K16> fragC[2][2];

        uint p8 = 0;
        #pragma unroll 1
        do
        {
            const ehalf* A0;
            const ehalf* B0;
            asm("ld.param.u64 %0, [%2 + 0x160];\n\t"
                "ld.param.u64 %1, [%2 + 0x1a0];"
                : "=l"(A0), "=l"(B0) : "r"(p8));
            p8 += 8;

            uint offsetA = offsetA0;
            uint offsetB = offsetB0;
            uint n       = n0;
            uint loop    = 0;

            #pragma unroll 1
            do
            {
                asm volatile (".pragma \"nounroll\";"::); // ptxas, don't get clever

                uint2 a00 = {0}, a16 = {0}, a32 = {0}, a48 = {0};
                uint2 b00 = {0}, b16 = {0}, b32 = {0}, b48 = {0};
                if (N64 || n + 0*16 < N)
                {
                    a00 = load_half4(A0 + (offsetA + 0*16*C));
                    b00 = load_half4(B0 + (offsetB + 0*16*K));
                }
                if (N64 || n + 1*16 < N)
                {
                    a16 = load_half4(A0 + (offsetA + 1*16*C));
                    b16 = load_half4(B0 + (offsetB + 1*16*K));
                }
                if (N64 || n + 2*16 < N)
                {
                    a32 = load_half4(A0 + (offsetA + 2*16*C));
                    b32 = load_half4(B0 + (offsetB + 2*16*K));
                }
                if (N64 || n + 3*16 < N)
                {
                    a48 = load_half4(A0 + (offsetA + 3*16*C));
                    b48 = load_half4(B0 + (offsetB + 3*16*K));
                }
                offsetA += 64*C;
                offsetB += 64*K;
                if (!N64)
                    n += 64;

                __syncthreads();
                *(uint2*)&hShare[storAB + 0*16*stdAB +  0*stdAB] = a00;
                *(uint2*)&hShare[storAB + 1*16*stdAB +  0*stdAB] = a16;
                *(uint2*)&hShare[storAB + 2*16*stdAB +  0*stdAB] = a32;
                *(uint2*)&hShare[storAB + 3*16*stdAB +  0*stdAB] = a48;
                *(uint2*)&hShare[storAB + 0*16*stdAB + 64*stdAB] = b00;
                *(uint2*)&hShare[storAB + 1*16*stdAB + 64*stdAB] = b16;
                *(uint2*)&hShare[storAB + 2*16*stdAB + 64*stdAB] = b32;
                *(uint2*)&hShare[storAB + 3*16*stdAB + 64*stdAB] = b48;
                __syncthreads();

                fragmentA<OP_T,M16N16K16> fragA[2];
                fragmentB<OP_N,M16N16K16> fragB[2];
                for (uint i = 0; i < 2; i++)
                {
                    fragA[i].load(hShare, loadA + i*16, stdAB);
                    fragB[i].load(hShare, loadB + i*16, stdAB);
                }
                for (uint i = 0; i < 2; i++)
                    for (uint j = 0; j < 2; j++)
                        fragC[i][j].mma_sync(fragA[i], fragB[j]);

            } while (++loop < loops);

        } while (p8 < params8);

        asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid) :);
        asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(bid) :);

        uint storC = fragmentC<OP_T,OP_N,M16N16K16>::get_idx(tid, stdC, (tid & 96));

        if (accumulate)
        {
            tx = tid % 16;
            ty = tid / 16;
            uint loadC   = ty*stdC + tx*2;
            uint offsetC = bid*32*32 + tid*2;

            for (uint i = 0; i < 2; i++)
            {
                __syncthreads();
                for (uint j = 0; j < 2; j++)
                    fragC[i][j].store(fShare, storC + j*16, stdC);
                __syncthreads();

                for (uint j = 0; j < 2; j++)
                {
                    float2 sum2 = ew_add(
                        ew_add(
                            *(float2*)&fShare[loadC + j*8*stdC + 0*32],
                            *(float2*)&fShare[loadC + j*8*stdC + 1*32]),
                        ew_add(
                            *(float2*)&fShare[loadC + j*8*stdC + 2*32],
                            *(float2*)&fShare[loadC + j*8*stdC + 3*32]));

                    reduce_half2(DW + offsetC + i*16*32 + j*8*32, to_half2(sum2));
                }
            }

        }
        else
        {
            tx = tid % 8;
            ty = tid / 8;
            uint loadC   = ty*stdC + tx*4;
            uint offsetC = bid*32*32 + tid*4;

            for (uint i = 0; i < 2; i++)
            {
                __syncthreads();
                for (uint j = 0; j < 2; j++)
                    fragC[i][j].store(fShare, storC + j*16, stdC);
                __syncthreads();

                float4 sum4 = ew_add(
                    ew_add(
                        *(float4*)&fShare[loadC + 0*32],
                        *(float4*)&fShare[loadC + 1*32]),
                    ew_add(
                        *(float4*)&fShare[loadC + 2*32],
                        *(float4*)&fShare[loadC + 3*32]));

                store_half4(DW + offsetC + i*16*32, to_half4(sum4));
            }
        }
    }
    else if (!accumulate) // gate == 0
        zero_half8(DW + (bid*32*32 + tid*8));
}

#else // __CUDA_ARCH__ >= 700

template <uint OP_B, bool N64, bool GATED>
__global__ void __launch_bounds__(256) hgemm_blocksparse_64x64x64_nx_dsd(
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    const ehalf* __restrict__ X,
    const ehalf* __restrict__ W,
          ehalf*              Y,
    uint* Lock, uint locks, uint N, uint C, uint K, uint blk_a, uint blk_b, uint blk_N)
{
    *Y = 0;
}
template <uint OP_A, bool N64, bool GATED>
__global__ void __launch_bounds__(128) hgemm_blocksparse_64x32x32_nx_dsd(
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    const ehalf* __restrict__ X,
    const ehalf* __restrict__ W,
          ehalf*              Y,
    uint* Lock, uint locks, uint N, uint C, uint K, uint blk_a, uint blk_b, uint blk_N)
{
    *Y = 0;
}
template <bool N64, bool GATED>
__global__ void __launch_bounds__(256) hgemm_blocksparse_64x64x64_tn_dds(
    struct Plist<ehalf,8> X,
    struct Plist<ehalf,8> DY,
    ehalf*                DW,
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    uint params8, uint N, uint C, uint K, uint loops, uint accumulate)
{
    *DW = 0;
}
template <bool N64, bool GATED>
__global__ void __launch_bounds__(128) hgemm_blocksparse_32x32x64_tn_dds(
    struct Plist<ehalf,8> X,
    struct Plist<ehalf,8> DY,
    ehalf*                DW,
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    uint params8, uint N, uint C, uint K, uint loops, uint accumulate)
{
    *DW = 0;
}


#endif // __CUDA_ARCH__ >= 700

cudaError_t hgemm_blocksparse_nx_dsd(const ehalf* X, const ehalf* W, ehalf* Y, bsmm_params* params, uint op)
{
    dim3 grid(params->blk_a*params->blk_b, params->blk_B, params->blk_A);
    uint blk_N = params->blk_b * params->blk_B;

    // cuMemsetD16Async((CUdeviceptr)Y, 0, params->K * params->N, params->stream);
    if (params->locks > 0)
        cuMemsetD32Async((CUdeviceptr)params->Lock, 0, blk_N * params->locks * 2, params->stream);

    const uint2* Lut = (const uint2*)params->Lut;
    uint* Lock       = (uint*)params->Lock;

    bool   N64 = (params->N & 63) == 0;
    int shared = params->shared + params->shared/2;

    if (params->bsize == 32)
    {
        // 132*16*4 - ((32+8)*64 + (32+16)*32)*2
        shared -= op == OP_N ? 256 : 768;
        if (shared < 0) shared = 0;

        if (params->Gate == 0)
        {
            if (op == OP_N)
                if (N64)
                    hgemm_blocksparse_64x32x32_nx_dsd<OP_N, true,false><<<grid,128,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
                else
                    hgemm_blocksparse_64x32x32_nx_dsd<OP_N,false,false><<<grid,128,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
            else
                if (N64)
                    hgemm_blocksparse_64x32x32_nx_dsd<OP_T, true,false><<<grid,128,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
                else
                    hgemm_blocksparse_64x32x32_nx_dsd<OP_T,false,false><<<grid,128,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
        }
        else
        {
            if (op == OP_N)
                if (N64)
                    hgemm_blocksparse_64x32x32_nx_dsd<OP_N, true, true><<<grid,128,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
                else
                    hgemm_blocksparse_64x32x32_nx_dsd<OP_N,false, true><<<grid,128,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
            else
                if (N64)
                    hgemm_blocksparse_64x32x32_nx_dsd<OP_T, true, true><<<grid,128,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
                else
                    hgemm_blocksparse_64x32x32_nx_dsd<OP_T,false, true><<<grid,128,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
        }
    }
    else if (params->bsize == 64)
    {
        if (params->Gate == 0)
        {
            if (op == OP_N)
                if (N64)
                    hgemm_blocksparse_64x64x64_nx_dsd<OP_N, true,false><<<grid,256,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
                else
                    hgemm_blocksparse_64x64x64_nx_dsd<OP_N,false,false><<<grid,256,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
            else
                if (N64)
                    hgemm_blocksparse_64x64x64_nx_dsd<OP_T, true,false><<<grid,256,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
                else
                    hgemm_blocksparse_64x64x64_nx_dsd<OP_T,false,false><<<grid,256,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
        }
        else
        {
            if (op == OP_N)
                if (N64)
                    hgemm_blocksparse_64x64x64_nx_dsd<OP_N, true, true><<<grid,256,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
                else
                    hgemm_blocksparse_64x64x64_nx_dsd<OP_N,false, true><<<grid,256,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
            else
                if (N64)
                    hgemm_blocksparse_64x64x64_nx_dsd<OP_T, true, true><<<grid,256,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
                else
                    hgemm_blocksparse_64x64x64_nx_dsd<OP_T,false, true><<<grid,256,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
        }
    }
    return cudaPeekAtLastError();
}
cudaError_t hgemm_blocksparse_nx_dsd(const bhalf* X, const bhalf* W, bhalf* Y, bsmm_params* params, uint op) { return cudaSuccess; }
cudaError_t hgemm_blocksparse_nx_dsd(const float* X, const float* W, float* Y, bsmm_params* params, uint op) { return cudaSuccess; }


cudaError_t hgemm_blocksparse_tn_dds(const ehalf* X, const ehalf* E, ehalf* U, bsmm_params* params)
{
    struct Plist<ehalf,8>* X8 = (struct Plist<ehalf,8>*)X;
    struct Plist<ehalf,8>* E8 = (struct Plist<ehalf,8>*)E;

    const uint2* Lut = (const uint2*)params->Lut;
    uint accumulate  = params->beta == 1.0f;
    uint pcount8     = params->pcount * 8;
    uint N           = params->N;
    uint C           = params->C;
    uint K           = params->K;
    uint loops       = CEIL_DIV(N, 64);
    bool N64         = (N & 63) == 0;

    dim3 grid(params->blocks, 1, 1);

    if (params->bsize == 32)
    {
        if (params->Gate == 0)
        {
            if (N64)
                hgemm_blocksparse_32x32x64_tn_dds< true,false><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, C, K, loops, accumulate);
            else
                hgemm_blocksparse_32x32x64_tn_dds<false,false><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, C, K, loops, accumulate);
        }
        else
        {
            if (N64)
                hgemm_blocksparse_32x32x64_tn_dds< true, true><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, C, K, loops, accumulate);
            else
                hgemm_blocksparse_32x32x64_tn_dds<false, true><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, C, K, loops, accumulate);
        }
    }
    else if (params->bsize == 64)
    {
        if (params->Gate == 0)
        {
            if (N64)
                hgemm_blocksparse_64x64x64_tn_dds< true,false><<<grid,256,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, C, K, loops, accumulate);
            else
                hgemm_blocksparse_64x64x64_tn_dds<false,false><<<grid,256,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, C, K, loops, accumulate);
        }
        else
        {
            if (N64)
                hgemm_blocksparse_64x64x64_tn_dds< true, true><<<grid,256,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, C, K, loops, accumulate);
            else
                hgemm_blocksparse_64x64x64_tn_dds<false, true><<<grid,256,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, C, K, loops, accumulate);
        }
    }
    return cudaPeekAtLastError();
}
cudaError_t hgemm_blocksparse_tn_dds(const bhalf* X, const bhalf* E, bhalf* U, bsmm_params* params) { return cudaSuccess; }
cudaError_t hgemm_blocksparse_tn_dds(const float* X, const float* E, float* U, bsmm_params* params) { return cudaSuccess; }


#define MAX_NORM 0
#define L2_NORM  1

#if __CUDA_ARCH__ >= 700

template <uint BSIZE, uint NSIZE, uint NORM>
__global__ void __launch_bounds__(256) blocksparse_feature_reduce_nc(
    const struct Plist<ehalf,8> X8, ehalf* Y, uint N, uint C)
{
    const ehalf* X;

    const uint ROW_THDS  = BSIZE/8;
    const uint WARP_ROWS = 32/ROW_THDS;
    const uint C_reduced = C / BSIZE;

    uint tid = threadIdx.x;
    uint bc  = blockIdx.x;
    uint bn  = blockIdx.y;

    // Each warp works on a Plist entry
    uint p  = tid / 32; // index into Plist
    uint tp = tid % 32;
    asm("ld.param.u64 %0, [%1 + 0x160];" : "=l"(X) : "r"(p * 8));

    uint tc = tp % ROW_THDS;
    uint tn = tp / ROW_THDS;
    uint n0 = bn*NSIZE + tn;

    uint offsetX = n0*C + bc*BSIZE + tc*8;
    uint offsetY = (p*N + n0)*C_reduced + bc;

    #pragma unroll 1
    for (uint n = 0; n < NSIZE && bn*NSIZE + n < N; n += WARP_ROWS)
    {
        bool n_valid = n + n0 < N;

        float8 x = load((const ehalf8*)(X + offsetX), 0, n_valid);

        float norm;
        if (NORM == MAX_NORM)
            norm = ew_max(ew_abs(x));
        else
            norm = ew_sum(ew_sqr(x));

        if (NORM == MAX_NORM)
        {
            #pragma unroll
            for (int i = ROW_THDS/2; i > 0; i >>= 1)
                norm = ew_warp_max(norm, i);
        }
        else
        {
            #pragma unroll
            for (int i = ROW_THDS/2; i > 0; i >>= 1)
                norm = ew_warp_sum(norm, i);
            norm = ew_sqrt(norm);
        }

        store(Y + offsetY, norm, 0, n_valid && tc == 0);

        offsetX += WARP_ROWS*C;
        offsetY += WARP_ROWS*C_reduced;
    }
}

template <bool M64, bool ACCUMULATE>
__device__  __noinline__  void store_64x64x32_tn(float* C, uint loadC, uint M, uint N, uint cy, uint cx, uint i, uint stdC, float scale)
{
    for (uint j = 0; j < 2; j++)
        for (uint k = 0; k < 4; k++)
            if (M64 || cy + i*16 + j*32 + k*4 < M)
            {
                float out = ew_zero_nan_inf(ew_mul(
                    ew_add(
                        ld_shared_float1(loadC + j*64 + k*4*stdC +   0),
                        ld_shared_float1(loadC + j*64 + k*4*stdC + 128)), scale));

                uint offsetC = (cy + i*16 + j*32 + k*4)*N + cx;
                if (ACCUMULATE)
                    atomicRed(C + offsetC, out);
                else
                    store(C + offsetC, out);
            }
}
template <bool M64, bool ACCUMULATE>
__global__ void __launch_bounds__(256) hgemm_64x64x32_tn(
    const ehalf* A,
    const ehalf* B,
          float* C,
    uint M, uint N, uint K, uint blk_a, uint blk_b, float scale)
{
    const uint stdAB =  64 + 16;
    const uint stdC  = 256 +  4;

    __shared__ float fShare[stdC*16];
    ehalf* hShare = (ehalf*)fShare;

    uint tid = threadIdx.x;

    uint idx_ab = blockIdx.x;
    uint idx_B  = blockIdx.y;
    uint idx_A  = blockIdx.z;

    idx_A = idx_A * blk_a + idx_ab / blk_b;
    idx_B = idx_B * blk_b + idx_ab % blk_b;

    uint tx = tid % 32;
    uint ty = tid / 32;
    uint ta = idx_A*64 + tx*2;
    uint tb = idx_B*64 + tx*2;

    uint offsetA = ty*M + ta;
    uint offsetB = ty*N + tb;
    uint storAB  = ty*stdAB + tx*2;

    uint loadA = fragmentA<OP_T,M16N16K16>::get_idx(tid, stdAB, (tid & 128)*stdAB*16/128 + (tid & 64)/2);
    uint loadB = fragmentB<OP_N,M16N16K16>::get_idx(tid, stdAB, (tid & 128)*stdAB*16/128 + (tid & 32) + stdAB*32);

    asm(".reg .pred pred_a, pred_b;  \n\t"
        "setp.lt.u32 pred_a, %0, %1; \n\t"
        "setp.lt.u32 pred_b, %2, %3; \n\t" ::
        "r"(ta), "r"(M),
        "r"(tb), "r"(N) );

    fragmentC<OP_T,OP_N,M16N16K16> fragC[2][2];

    #pragma unroll 1
    for (uint k = 0, tk = ty; k < K; k += 32)
    {
        uint a00, a08, a16, a24;
        uint b00, b08, b16, b24;

        asm volatile("{\n\t"
            ".reg .pred ka00, ka08, ka16, ka24;      \n\t"
            ".reg .pred kb00, kb08, kb16, kb24;      \n\t"
            "setp.lt.and.u32 ka00, %16, %20, pred_a; \n\t"
            "setp.lt.and.u32 ka08, %17, %20, pred_a; \n\t"
            "setp.lt.and.u32 ka16, %18, %20, pred_a; \n\t"
            "setp.lt.and.u32 ka24, %19, %20, pred_a; \n\t"
            "setp.lt.and.u32 kb00, %16, %20, pred_b; \n\t"
            "setp.lt.and.u32 kb08, %17, %20, pred_b; \n\t"
            "setp.lt.and.u32 kb16, %18, %20, pred_b; \n\t"
            "setp.lt.and.u32 kb24, %19, %20, pred_b; \n\t"
            "mov.b64 { %0, %1}, 0; \n\t"
            "mov.b64 { %2, %3}, 0; \n\t"
            "mov.b64 { %4, %5}, 0; \n\t"
            "mov.b64 { %6, %7}, 0; \n\t"
            "@ka00 ld.global.nc.u32 %0, [ %8]; \n\t"
            "@ka08 ld.global.nc.u32 %1, [ %9]; \n\t"
            "@ka16 ld.global.nc.u32 %2, [%10]; \n\t"
            "@ka24 ld.global.nc.u32 %3, [%11]; \n\t"
            "@kb00 ld.global.nc.u32 %4, [%12]; \n\t"
            "@kb08 ld.global.nc.u32 %5, [%13]; \n\t"
            "@kb16 ld.global.nc.u32 %6, [%14]; \n\t"
            "@kb24 ld.global.nc.u32 %7, [%15]; \n\t"
            "}" :
            "=r"(a00), "=r"(a08), "=r"(a16), "=r"(a24),
            "=r"(b00), "=r"(b08), "=r"(b16), "=r"(b24) :
            "l"(A + (offsetA +  0*M)),
            "l"(A + (offsetA +  8*M)),
            "l"(A + (offsetA + 16*M)),
            "l"(A + (offsetA + 24*M)),
            "l"(B + (offsetB +  0*N)),
            "l"(B + (offsetB +  8*N)),
            "l"(B + (offsetB + 16*N)),
            "l"(B + (offsetB + 24*N)),
            "r"(tk), "r"(tk+8), "r"(tk+16), "r"(tk+24), "r"(K) );
        offsetA += 32*M;
        offsetB += 32*N;
        tk      += 32;

        __syncthreads();
        *(uint*)&hShare[storAB +  0*stdAB +  0*stdAB] = a00;
        *(uint*)&hShare[storAB +  8*stdAB +  0*stdAB] = a08;
        *(uint*)&hShare[storAB + 16*stdAB +  0*stdAB] = a16;
        *(uint*)&hShare[storAB + 24*stdAB +  0*stdAB] = a24;
        *(uint*)&hShare[storAB +  0*stdAB + 32*stdAB] = b00;
        *(uint*)&hShare[storAB +  8*stdAB + 32*stdAB] = b08;
        *(uint*)&hShare[storAB + 16*stdAB + 32*stdAB] = b16;
        *(uint*)&hShare[storAB + 24*stdAB + 32*stdAB] = b24;
        __syncthreads();

        fragmentA<OP_T,M16N16K16> fragA[2];
        fragmentB<OP_N,M16N16K16> fragB[2];
        for (int i = 0; i < 2; i++)
        {
            fragA[i].load(hShare, loadA + i*16, stdAB);
            fragB[i].load(hShare, loadB + i*16, stdAB);
        }
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                fragC[i][j].mma_sync(fragA[i], fragB[j]);
    }
    asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid)    :);
    asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(idx_ab) :);
    asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(idx_B)  :);
    asm volatile ("mov.u32 %0, %ctaid.z;" : "=r"(idx_A)  :);

    idx_A = idx_A * blk_a + idx_ab / blk_b;
    idx_B = idx_B * blk_b + idx_ab % blk_b;

    tx = tid % 64;
    ty = tid / 64;
    uint cx = idx_B*64 + tx;
    uint cy = idx_A*64 + ty;
    uint loadC = ty*stdC + tx;
    uint storC = fragmentC<OP_T,OP_N,M16N16K16>::get_idx(tid, stdC, (tid & 224));

    bool cx_valid = cx < N;

    for (int i = 0; i < 2; i++)
    {
        __syncthreads();
        for (int j = 0; j < 2; j++)
            fragC[i][j].store(fShare, storC + j*16, stdC);
        __syncthreads();

        if (cx_valid)
            store_64x64x32_tn<M64,ACCUMULATE>(C, loadC, M, N, cy, cx, i, stdC, scale);
    }
}

#else // __CUDA_ARCH__ >= 700

template <uint BSIZE, uint NSIZE, uint NORM>
__global__ void __launch_bounds__(256) blocksparse_feature_reduce_nc(
    const struct Plist<ehalf,8> X8, ehalf* Y, uint N, uint C)
{
    *Y = 0;
}

template <bool M64, bool ACCUMULATE>
__global__ void __launch_bounds__(256) hgemm_64x64x32_tn(
    const ehalf* A,
    const ehalf* B,
          float* C,
    uint M, uint N, uint K, uint blk_a, uint blk_b, float scale)
{
    *C = 0;
}

#endif // __CUDA_ARCH__ >= 700


bool BlocksparseFeatureReduceNC(CUstream stream, ehalf* Y, const struct Plist<ehalf,8>* X8, uint params, uint C, uint N, uint bshift, uint norm_type)
{
    uint gridC   = C >> bshift;
    uint threads = params * 32;
    if (bshift == 5)
    {
        dim3 grid(gridC, CEIL_DIV(N, 32), 1);
        if (norm_type == MAX_NORM)
            blocksparse_feature_reduce_nc<32,32,MAX_NORM><<<grid,threads,0,stream>>>(*X8, Y, N, C);
        else
            blocksparse_feature_reduce_nc<32,32, L2_NORM><<<grid,threads,0,stream>>>(*X8, Y, N, C);
    }
    else if (bshift == 6)
    {
        dim3 grid(gridC, CEIL_DIV(N, 16), 1);
        if (norm_type == MAX_NORM)
            blocksparse_feature_reduce_nc<64,16,MAX_NORM><<<grid,threads,0,stream>>>(*X8, Y, N, C);
        else
            blocksparse_feature_reduce_nc<64,16, L2_NORM><<<grid,threads,0,stream>>>(*X8, Y, N, C);
    }
    return true;
}

bool hGemmTN(CUstream stream, const ehalf* A, const ehalf* B, float* C, uint M, uint N, uint K, uint blk_a, uint blk_b, uint blk_A, uint blk_B, uint accumulate, float scale)
{
    if (scale != 0.0f)
    {
        dim3 grid(blk_a*blk_b, blk_B, blk_A);
        if (M & 63)
            if (accumulate)
                hgemm_64x64x32_tn<false, true><<<grid,256,0,stream>>>(A, B, C, M, N, K, blk_a, blk_b, scale);
            else
                hgemm_64x64x32_tn<false,false><<<grid,256,0,stream>>>(A, B, C, M, N, K, blk_a, blk_b, scale);
        else
            if (accumulate)
                hgemm_64x64x32_tn< true, true><<<grid,256,0,stream>>>(A, B, C, M, N, K, blk_a, blk_b, scale);
            else
                hgemm_64x64x32_tn< true,false><<<grid,256,0,stream>>>(A, B, C, M, N, K, blk_a, blk_b, scale);
    }
    else if (accumulate == 0)
        cuMemsetD32Async((CUdeviceptr)C, 0, M*N, stream);

    return true;
}

#endif // GOOGLE_CUDA



                // if (OP_B == OP_N)
                // printf("%d %d %3d %08x %08x %08x %08x\n",
                //     idx_K, idx_Lock, tid,
                //     b00.x, b00.y, b16.x, b16.y);

                // if (OP_B == OP_N)
                // printf("%d %d %3d %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x\n",
                //     idx_K, idx_Lock, tid,
                //     fragA[0].x[0], fragA[0].x[1], fragA[0].x[2], fragA[0].x[3], fragA[0].x[4], fragA[0].x[5], fragA[0].x[6], fragA[0].x[7],
                //     fragA[1].x[0], fragA[1].x[1], fragA[1].x[2], fragA[1].x[3], fragA[1].x[4], fragA[1].x[5], fragA[1].x[6], fragA[1].x[7]);
                //     // fragB[0].x[0], fragB[0].x[1], fragB[0].x[2], fragB[0].x[3], fragB[0].x[4], fragB[0].x[5], fragB[0].x[6], fragB[0].x[7],
                //     // fragB[1].x[0], fragB[1].x[1], fragB[1].x[2], fragB[1].x[3], fragB[1].x[4], fragB[1].x[5], fragB[1].x[6], fragB[1].x[7]);

        // if (OP_B == OP_N)
        // printf("%d %d %3d %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
        //     idx_K, idx_Lock, tid,
        //     fragC[1][0].x[0], fragC[1][0].x[1], fragC[1][0].x[2], fragC[1][0].x[3], fragC[1][0].x[4], fragC[1][0].x[5], fragC[1][0].x[6], fragC[1][0].x[7],
        //     fragC[1][1].x[0], fragC[1][1].x[1], fragC[1][1].x[2], fragC[1][1].x[3], fragC[1][1].x[4], fragC[1][1].x[5], fragC[1][1].x[6], fragC[1][1].x[7]);

        // printf("%d %d %3d %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
        //     idx_K, idx_Lock, tid,
        //     fragC[0][0].x[0], fragC[0][0].x[1], fragC[0][0].x[2], fragC[0][0].x[3], fragC[0][0].x[4], fragC[0][0].x[5], fragC[0][0].x[6], fragC[0][0].x[7],
        //     fragC[0][1].x[0], fragC[0][1].x[1], fragC[0][1].x[2], fragC[0][1].x[3], fragC[0][1].x[4], fragC[0][1].x[5], fragC[0][1].x[6], fragC[0][1].x[7],
        //     fragC[1][0].x[0], fragC[1][0].x[1], fragC[1][0].x[2], fragC[1][0].x[3], fragC[1][0].x[4], fragC[1][0].x[5], fragC[1][0].x[6], fragC[1][0].x[7],
        //     fragC[1][1].x[0], fragC[1][1].x[1], fragC[1][1].x[2], fragC[1][1].x[3], fragC[1][1].x[4], fragC[1][1].x[5], fragC[1][1].x[6], fragC[1][1].x[7]);


                //if (OP_B == OP_N)
                    // for (uint j = 0; j < 2; j++)
                    // {
                    //     float4 a = *(float4*)&fShare[loadC + j*32 +  0];
                    //     float4 b = *(float4*)&fShare[loadC + j*32 + 64];
                    //     printf("%d %d %d %3d %5d %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
                    //         j, idx_K, idx_Lock, tid, offsetC + K*(j*32 + i*16), a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w
                    //     );
                    //     // uint2 a = to_half4(ew_add(*(float4*)&fShare[loadC + j*32 +  0], *(float4*)&fShare[loadC + j*32 + 64]));
                    //     // printf("%d %d %d %3d %5d %08x %08x\n",
                    //     //     j, idx_K, idx_Lock, tid, offsetC + K*(j*32 + i*16), a.x, a.y
                    //     // );
                    //     //store_half4(C + (offsetC + (j*32 + i*16)*K), a);
                    // }