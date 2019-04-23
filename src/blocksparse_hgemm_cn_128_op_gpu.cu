#if GOOGLE_CUDA

#include "ew_op_gpu.h"
#include "gpu_hmma.h"
#include <stdio.h>

#if __CUDA_ARCH__ >= 700

template <uint OP_A, bool GATED>
__global__ void __launch_bounds__(128) hgemm_blocksparse_32x128x32_xn_sdd(
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint* Lock, uint locks, uint N, uint blk_a, uint blk_b, uint blk_N)
{
    const uint stdA =  32 + 16;
    const uint stdB = 128 + 16;
    const uint stdC = 128 +  4;

    __shared__ ehalf hShare[(stdA + stdB)*32];
    uint2* LutOffsets = (uint2*)&hShare[(stdA + stdB)*32];

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

    uint txa = tid %  8;
    uint tya = tid /  8;
    uint txb = tid % 16;
    uint tyb = tid / 16;

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

            entry.y *= 32*32;
            entry.x *= N*32;
            LutOffsets[i] = entry;
        }
        __syncthreads();

        uint storA = tya*stdA + txa*4;
        uint storB = tyb*stdB + txb*8 + stdA*32;

        uint loadA = fragmentA<OP_A,M16N16K16>::get_idx(tid, stdA);
        uint loadB = fragmentB<OP_N,M16N16K16>::get_idx(tid, stdB, (tid & 96) + stdA*32);

        uint       n = idx_N*128 + txb*8;
        uint offsetB = tyb*N + n;

        asm(".reg .pred pn;\n\tsetp.lt.u32 pn, %0, %1;" :: "r"(n), "r"(N)); // n < N
        asm("mov.b32 %0, %0;" : "+r"(idx_N) : );
        asm("mov.b32 %0, %0;" : "+r"(loadA) : );
        asm("mov.b32 %0, %0;" : "+r"(loadB) : );
        asm("mov.b32 %0, %0;" : "+r"(offsetB) : );

        fragmentC<OP_A,OP_N,M16N16K16> fragC[2][2];

        int idx_lut = 0;
        #pragma unroll 1
        do
        {
            uint gate = Gates[idx_lut];

            if (gate != 0)
            {
                uint2 entry = LutOffsets[idx_lut];

                const ehalf* pA = A + (entry.y + tid*4);
                uint2 a00 = load_half4(pA +  0*32);
                uint2 a16 = load_half4(pA + 16*32);
                uint4 b00, b08, b16, b24;
                entry.x += offsetB;

                asm("mov.b64  { %0,  %1}, 0; \n\t"
                    "mov.b64  { %2,  %3}, 0; \n\t"
                    "mov.b64  { %4,  %5}, 0; \n\t"
                    "mov.b64  { %6,  %7}, 0; \n\t"
                    "mov.b64  { %8,  %9}, 0; \n\t"
                    "mov.b64  {%10, %11}, 0; \n\t"
                    "mov.b64  {%12, %13}, 0; \n\t"
                    "mov.b64  {%14, %15}, 0; \n\t"
                    "@pn ld.global.nc.v4.u32 { %0,  %1,  %2,  %3}, [%16];\n\t"
                    "@pn ld.global.nc.v4.u32 { %4,  %5,  %6,  %7}, [%17];\n\t"
                    "@pn ld.global.nc.v4.u32 { %8,  %9, %10, %11}, [%18];\n\t"
                    "@pn ld.global.nc.v4.u32 {%12, %13, %14, %15}, [%19];\n\t" :
                    "=r"(b00.x), "=r"(b00.y), "=r"(b00.z), "=r"(b00.w),
                    "=r"(b08.x), "=r"(b08.y), "=r"(b08.z), "=r"(b08.w),
                    "=r"(b16.x), "=r"(b16.y), "=r"(b16.z), "=r"(b16.w),
                    "=r"(b24.x), "=r"(b24.y), "=r"(b24.z), "=r"(b24.w) :
                    "l"(B + (entry.x + N* 0)),
                    "l"(B + (entry.x + N* 8)),
                    "l"(B + (entry.x + N*16)),
                    "l"(B + (entry.x + N*24)));

                if (GATED)
                {
                    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(a00.x) : "r"(gate));
                    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(a00.y) : "r"(gate));
                    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(a16.x) : "r"(gate));
                    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(a16.y) : "r"(gate));
                }
                __syncthreads();
                *(uint2*)&hShare[storA +  0*stdA] = a00;
                *(uint2*)&hShare[storA + 16*stdA] = a16;
                *(uint4*)&hShare[storB +  0*stdB] = b00;
                *(uint4*)&hShare[storB +  8*stdB] = b08;
                *(uint4*)&hShare[storB + 16*stdB] = b16;
                *(uint4*)&hShare[storB + 24*stdB] = b24;
                __syncthreads();

                fragmentA<OP_A,M16N16K16> fragA[2];
                fragmentB<OP_N,M16N16K16> fragB[2];
                for (int k = 0; k < 2; k++)
                {
                    for (int i = 0; i < 2; i++)
                    {
                        fragA[i].load(hShare, loadA + (OP_A == OP_N ? stdA : 1)*i*16 + (OP_A == OP_N ? 1 : stdA)*k*16, stdA);
                        fragB[i].load(hShare, loadB + i*16 + k*16*stdB, stdB);
                    }
                    for (int i = 0; i < 2; i++)
                        for (int j = 0; j < 2; j++)
                            fragC[i][j].mma_sync(fragA[i], fragB[j]);
                }
            }

        } while (++idx_lut < lut_size);

        uint txc = tid % 32;
        uint tyc = tid / 32;

        n = idx_N*128 + txc*4;
        uint loadC   = tyc*stdC + txc*4;
        uint storC   = fragmentC<OP_A,OP_N,M16N16K16>::get_idx(tid, stdC, tid & 96);
        uint offsetC = (idx_K*32 + tyc)*N + n;

        __syncthreads();
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                fragC[i][j].store(hShare, storC + i*16*stdC + j*16, stdC);
        __syncthreads();

        if (idx_Lock == 0)
        {
            if (n < N)
                for (int j = 0; j < 8; j++)
                    store_half4(C + (offsetC + j*4*N), *(uint2*)&hShare[loadC + stdC*j*4]);
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
                if (n < N)
                    for (int j = 0; j < 8; j++)
                        store_half4(C + (offsetC + j*4*N), *(uint2*)&hShare[loadC + stdC*j*4]);

                __threadfence();
                __syncthreads();

                if (tid == 0)
                    atomicExch(Lock, 0);
                // End Critial Section
            }
            else
            {
                txc = tid % 64;
                tyc = tid / 64;

                n = idx_N*128 + txc*2;
                loadC   = tyc*stdC + txc*2;
                offsetC = (idx_K*32 + tyc)*N + n;

                // subsequent blocks must accumulate
                if (n < N)
                    for (int j = 0; j < 16; j++)
                        reduce_half2(C + (offsetC + j*2*N), *(uint*)&hShare[loadC + stdC*j*2]);

                __threadfence();
                __syncthreads();

                if (tid == 0)
                    atomicExch(Lock, 0);
                // End Critial Section
            }
        }
    }
    else
    {
        uint n = idx_N*128 + txb*8;
        uint offsetC = (idx_K*32 + tyb)*N + n;

        if (n < N)
            for (int i = 0; i < 4; i++)
                zero_half8(C + (offsetC + i*8*N));
    }
}

template <uint OP_A, bool GATED>
__global__ void __launch_bounds__(128) hgemm_blocksparse_16x128x16_xn_sdd(
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint* Lock, uint locks, uint N, uint blk_a, uint blk_b, uint blk_N)
{
    const uint stdA = 16;
    const uint stdB = 128 + 16;
    const uint stdC = 128 +  4;

    __shared__ ehalf hShare[(stdA + stdB)*16];
    uint2* LutOffsets = (uint2*)&hShare[(stdA + stdB)*16];

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

    uint txb = tid % 16;
    uint tyb = tid / 16;

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

            entry.y *= 16*16;
            entry.x *= N*16;
            LutOffsets[i] = entry;
        }
        __syncthreads();

        uint storB = tyb*stdB + txb*8 + 16*stdA;
        uint loadA = fragmentA<OP_A,M16N16K16>::get_idx(tid, stdA);
        uint loadB = fragmentB<OP_N,M16N16K16>::get_idx(tid, stdB, 16*stdA + (tid & 96));

        uint       n = idx_N*128 + txb*8;
        uint offsetB = tyb*N + n;

        asm(".reg .pred pn;\n\tsetp.lt.u32 pn, %0, %1;" :: "r"(n), "r"(N)); // n < N
        asm("mov.b32 %0, %0;" : "+r"(idx_N)   : );
        asm("mov.b32 %0, %0;" : "+r"(loadA)   : );
        asm("mov.b32 %0, %0;" : "+r"(loadB)   : );
        asm("mov.b32 %0, %0;" : "+r"(offsetB) : );

        fragmentC<OP_A,OP_N,M16N16K16> fragC[2];

        int idx_lut = 0;
        #pragma unroll 1
        do
        {
            uint gate = Gates[idx_lut];

            if (gate != 0)
            {
                uint2 entry = LutOffsets[idx_lut];

                uint  a0 = load_half2(A + (entry.y + tid*2));
                uint4 b0, b8;

                asm("mov.b64  {%0,  %1}, 0; \n\t"
                    "mov.b64  {%2,  %3}, 0; \n\t"
                    "mov.b64  {%4,  %5}, 0; \n\t"
                    "mov.b64  {%6,  %7}, 0; \n\t"
                    "@pn ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%8];\n\t"
                    "@pn ld.global.nc.v4.u32 {%4, %5, %6, %7}, [%9];\n\t" :
                    "=r"(b0.x), "=r"(b0.y), "=r"(b0.z), "=r"(b0.w),
                    "=r"(b8.x), "=r"(b8.y), "=r"(b8.z), "=r"(b8.w) :
                    "l"(B + (entry.x + offsetB + N*0)),
                    "l"(B + (entry.x + offsetB + N*8)));

                // printf("%d %d %3d %08x %08x %08x %08x %08x %08x %08x %08x\n",
                //     idx_K, idx_Lock, tid,
                //     b0.x, b0.y, b0.z, b0.w,
                //     b8.x, b8.y, b8.z, b8.w);

                if (GATED)
                    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(a0) : "r"(gate));

                __syncthreads();
                *(uint *)&hShare[tid*2] = a0;
                *(uint4*)&hShare[storB + 0*stdB] = b0;
                *(uint4*)&hShare[storB + 8*stdB] = b8;
                __syncthreads();

                fragmentA<OP_A,M16N16K16> fragA;
                fragmentB<OP_N,M16N16K16> fragB;

                fragA.load(hShare, loadA, stdA);

                // printf("%d %d %3d %08x %08x %08x %08x %08x %08x %08x %08x\n",
                //     idx_K, idx_Lock, tid,
                //     fragA.x[0], fragA.x[1], fragA.x[2], fragA.x[3], fragA.x[4], fragA.x[5], fragA.x[6], fragA.x[7]);

                #pragma unroll
                for (int j = 0; j < 2; j++)
                {
                    fragB.load(hShare, loadB + j*16, stdB);

                    fragC[j].mma_sync(fragA, fragB);
                }
            }

        } while (++idx_lut < lut_size);

        // printf("%d %d %3d %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
        //     idx_K, idx_Lock, tid,
        //     fragC[0].x[0], fragC[0].x[1], fragC[0].x[2], fragC[0].x[3], fragC[0].x[4], fragC[0].x[5], fragC[0].x[6], fragC[0].x[7],
        //     fragC[1].x[0], fragC[1].x[1], fragC[1].x[2], fragC[1].x[3], fragC[1].x[4], fragC[1].x[5], fragC[1].x[6], fragC[1].x[7]);


        // use thread stride of 4 to allow use of shared stride of 132
        // which minimizes shared bank conflicts on write.
        uint txc = tid % 32;
        uint tyc = tid / 32;

        n = idx_N*128 + txc*4;
        uint loadC   = tyc*stdC + txc*4;
        uint storC   = fragmentC<OP_A,OP_N,M16N16K16>::get_idx(tid, stdC, tid & 96);
        uint offsetC = (idx_K*16 + tyc)*N + n;

        __syncthreads();
        for (int j = 0; j < 2; j++)
            fragC[j].store(hShare, storC + j*16, stdC);
        __syncthreads();

        if (idx_Lock == 0)
        {
            // no lock needed just write out the results
            for (uint i = 0; i < 4; i++)
                if (n < N)
                    store_half4(C + (offsetC + N*i*4), *(uint2*)&hShare[loadC + stdC*i*4]);
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
                for (uint i = 0; i < 4; i++)
                    if (n < N)
                        store_half4(C + (offsetC + N*i*4), *(uint2*)&hShare[loadC + stdC*i*4]);

                __threadfence();
                __syncthreads();

                if (tid == 0)
                    atomicExch(Lock, 0);
                // End Critial Section
            }
            else
            {
                txc = tid % 64;
                tyc = tid / 64;

                n       = idx_N*128 + txc*2;
                loadC   = tyc*stdC  + txc*2;
                offsetC = (idx_K*16 + tyc)*N + n;

                // subsequent blocks must accumulate
                for (uint i = 0; i < 8; i++)
                    if (n < N)
                        reduce_half2(C + (offsetC + N*i*2), *(uint*)&hShare[loadC + stdC*i*2]);

                __threadfence();
                __syncthreads();

                if (tid == 0)
                    atomicExch(Lock, 0);
                // End Critial Section
            }
        }
    }
    else
    {
        uint n = idx_N*128 + txb*8;
        C += (idx_K*16 + tyb)*N + n;

        if (n < N)
        {
            zero_half8(C + N*0);
            zero_half8(C + N*8);
        }
    }
}

template <uint OP_A, bool GATED>
__global__ void __launch_bounds__(128) hgemm_blocksparse_8x128x8_xn_sdd(
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint* Lock, uint locks, uint N, uint blk_a, uint blk_b, uint blk_N)
{
    const uint stdA = 8;
    const uint stdB = 128 + 16;
    const uint stdC = 128 +  4;

    __shared__ ehalf hShare[(stdA + stdB)*16];
    uint2* LutOffsets = (uint2*)&hShare[(stdA + stdB)*16];

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

    if (lut_size > 0)
    {
        ushort* Gates = (ushort*)&LutOffsets[lut_size];

        // prefetch the lut and gate data into shared
        Lut += lut_offset;
        #pragma unroll 1
        for (uint i = tid; i < lut_size; i += 128)
        {
            uint2 entry = Lut[i];

            if (GATED)
            {
                float gate = Gate[entry.y];
                ushort gate16;
                asm("cvt.rn.f16.f32 %0, %1;" : "=h"(gate16) : "f"(gate));
                Gates[i] = gate16;
            }
            else
                Gates[i] = 1;

            entry.y *= 8*8; // 64 entries of A per block
            entry.x *= N*8; // 8 lines of B per block
            LutOffsets[i] = entry;
        }
        if (tid == 0)
            Gates[lut_size] = 0; // needed if lut_size is odd

        __syncthreads();

        uint t64 = tid & 64;
        uint t63 = tid & 63;
        uint txb = tid % 16;
        uint tyb = t63 / 16;

        uint storB = tyb*stdB + txb*8 + t64*stdB*8/64 + 16*stdA;

        uint loadA = fragmentA<OP_A,M8N32K16>::get_idx(tid, stdA);
        uint loadB = fragmentB<OP_N,M8N32K16>::get_idx(tid, stdB, (tid & 96) + 16*stdA);

        uint       n = idx_N*128 + txb*8;
        uint offsetA = t63;
        uint offsetB = tyb*N + n;

        fragmentC<OP_A,OP_N,M8N32K16> fragC;

        uint idx_lut   = t64 / 64;
        uint idx_lut2  = 0;
        uint lut_size2 = (lut_size + 1)/2;

        asm(".reg .pred pn;\n\tsetp.lt.u32 pn, %0, %1;" :: "r"(n), "r"(N)); // n < N
        asm("mov.b32 %0, %0;" : "+r"(idx_N) : );
        asm("mov.b32 %0, %0;" : "+r"(loadA) : );
        asm("mov.b32 %0, %0;" : "+r"(loadB) : );
        asm("mov.b32 %0, %0;" : "+r"(offsetA) : );
        asm("mov.b32 %0, %0;" : "+r"(offsetB) : );

        #pragma unroll 1
        do
        {
            ushort a0 = 0;
            uint4  b0 = {0};
            uint4  b4 = {0};

            ushort gate = Gates[idx_lut];

            // if the gate is zero just skip over memory loads
            // we compute 2 blocks per loop so it's easier to just always do the mma math
            if (gate != 0)
            {
                uint2 entry = LutOffsets[idx_lut];
                a0 = load_half(A + (entry.y + offsetA));

                asm("@pn ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%8];\n\t"
                    "@pn ld.global.nc.v4.u32 {%4, %5, %6, %7}, [%9];\n\t" :
                    "=r"(b0.x), "=r"(b0.y), "=r"(b0.z), "=r"(b0.w),
                    "=r"(b4.x), "=r"(b4.y), "=r"(b4.z), "=r"(b4.w) :
                    "l"(B + (entry.x + offsetB + N*0)),
                    "l"(B + (entry.x + offsetB + N*4)));
            }
            if (GATED)
                asm("mul.rn.f16 %0, %0, %1;" : "+h"(a0) : "h"(gate));

            // if (OP_A == OP_T)
            //     printf("%d %2d A:%08x B: %08x %08x %08x %08x %08x %08x %08x %08x\n", idx_K, tid, a0, b0.x,b0.y,b0.z,b0.w, b4.x,b4.y,b4.z,b4.w);

            __syncthreads();
            *(ushort*)&hShare[tid          ] = a0;
            *(uint4*)&hShare[storB + 0*stdB] = b0;
            *(uint4*)&hShare[storB + 4*stdB] = b4;
            __syncthreads();

            fragmentA<OP_A,M8N32K16> fragA;
            fragmentB<OP_N,M8N32K16> fragB;

            fragA.load(hShare, loadA, stdA);
            fragB.load(hShare, loadB, stdB);

            // if (OP_A == OP_T)
            //     printf("%d %2d A:%08x %08x %08x %08x %08x %08x %08x %08x B:%08x %08x %08x %08x %08x %08x %08x %08x\n", idx_K, tid,
            //         fragA.x[0], fragA.x[1], fragA.x[2], fragA.x[3], fragA.x[4], fragA.x[5], fragA.x[6], fragA.x[7],
            //         fragB.x[0], fragB.x[1], fragB.x[2], fragB.x[3], fragB.x[4], fragB.x[5], fragB.x[6], fragB.x[7]);

            fragC.mma_sync(fragA, fragB);

            idx_lut += 2;

        } while (++idx_lut2 < lut_size2);

        // use thread stride of 4 to allow use of shared stride of 68
        // which minimizes shared bank conflicts on write.
        uint txc = tid % 32;
        uint tyc = tid / 32;

        n = idx_N*128 + txc*4;
        uint loadC   = tyc*stdC + txc*4;
        uint storC   = fragmentC<OP_A,OP_N,M8N32K16>::get_idx(tid, stdC, tid & 96);
        uint offsetC = (idx_K*8 + tyc)*N + n;

        // if (OP_A == OP_T)
        //     printf("%d %d %2d %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n", idx_K, idx_Lock, tid, fragC.x[0], fragC.x[1], fragC.x[2], fragC.x[3], fragC.x[4], fragC.x[5], fragC.x[6], fragC.x[7]);

        __syncthreads();
        fragC.store(hShare, storC, stdC);
        __syncthreads();

        if (idx_Lock == 0)
        {
            // no lock needed just write out the results
            for (uint i = 0; i < 2; i++)
                if (n < N)
                    store_half4(C + (offsetC + N*i*4), *(uint2*)&hShare[loadC + stdC*i*4]);
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
                    if (n < N)
                        store_half4(C + (offsetC + N*i*4), *(uint2*)&hShare[loadC + stdC*i*4]);

                __threadfence();
                __syncthreads();

                if (tid == 0)
                    atomicExch(Lock, 0);
                // End Critial Section
            }
            else
            {
                txc = tid % 64;
                tyc = tid / 64;

                n       = idx_N*128 + txc*2;
                loadC   = tyc*stdC + txc*2;
                offsetC = (idx_K*8 + tyc)*N + n;

                // subsequent blocks must accumulate
                for (uint i = 0; i < 4; i++)
                    if (n < N)
                        reduce_half2(C +(offsetC + N*i*2), *(uint*)&hShare[loadC + stdC*i*2]);

                __threadfence();
                __syncthreads();

                if (tid == 0)
                    atomicExch(Lock, 0);
                // End Critial Section
            }
        }
    }
    else // lut_size == 0
    {
        uint txc = tid % 16;
        uint tyc = tid / 16;

        uint n       = idx_N*128 + txc*8;
        uint offsetC = (idx_K*8 + tyc)*N + n;

        if (n < N)
            zero_half8(C + offsetC);
    }
}

template <bool N128, bool GATED>
__global__ void __launch_bounds__(128,6) hgemm_blocksparse_32x32x128_nt_dds(
    struct Plist<ehalf,8> A,
    struct Plist<ehalf,8> B,
    ehalf*                C,
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    uint params8, uint N, uint loops, uint accumulate)
{
    const uint stdAB = 128 + 8;
    const uint stdC  = 128 + 4;

    __shared__ ehalf hShare[stdAB*2*32];
    float* fShare = (float*)hShare;

    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    float gate = GATED ? Gate[bid] : 1.0f;

    if (gate != 0.0f)
    {
        uint2 lut_head = Lut[bid];

        uint tx = tid % 16;
        uint ty = tid / 16;
        uint n0 = tx  *  8;

        uint idx_A = lut_head.x;
        uint idx_B = lut_head.y;
        uint offsetA0 = (idx_A*32 + ty)*N + n0;
        uint offsetB0 = (idx_B*32 + ty)*N + n0;
        uint storAB = ty*stdAB + n0;

        uint loadA = fragmentA<OP_N,M16N16K16>::get_idx(tid, stdAB, (tid & 96));
        uint loadB = fragmentB<OP_T,M16N16K16>::get_idx(tid, stdAB, (tid & 96) + stdAB*32);

        fragmentC<OP_N,OP_T,M16N16K16> fragC[2][2];

        int p8 = 0;
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
                uint4 a00 = {0}, a08 = {0}, a16 = {0}, a24 = {0};
                uint4 b00 = {0}, b08 = {0}, b16 = {0}, b24 = {0};
                if (N128 || n < N)
                {
                    a00 = load_half8(A0 + (offsetA + 0*8*N));
                    a08 = load_half8(A0 + (offsetA + 1*8*N));
                    a16 = load_half8(A0 + (offsetA + 2*8*N));
                    a24 = load_half8(A0 + (offsetA + 3*8*N));

                    b00 = load_half8(B0 + (offsetB + 0*8*N));
                    b08 = load_half8(B0 + (offsetB + 1*8*N));
                    b16 = load_half8(B0 + (offsetB + 2*8*N));
                    b24 = load_half8(B0 + (offsetB + 3*8*N));
                }
                offsetA += 128;
                offsetB += 128;
                if (!N128)
                    n += 128;

                __syncthreads();
                *(uint4*)&hShare[storAB + 0*8*stdAB +  0*stdAB] = a00;
                *(uint4*)&hShare[storAB + 1*8*stdAB +  0*stdAB] = a08;
                *(uint4*)&hShare[storAB + 2*8*stdAB +  0*stdAB] = a16;
                *(uint4*)&hShare[storAB + 3*8*stdAB +  0*stdAB] = a24;

                *(uint4*)&hShare[storAB + 0*8*stdAB + 32*stdAB] = b00;
                *(uint4*)&hShare[storAB + 1*8*stdAB + 32*stdAB] = b08;
                *(uint4*)&hShare[storAB + 2*8*stdAB + 32*stdAB] = b16;
                *(uint4*)&hShare[storAB + 3*8*stdAB + 32*stdAB] = b24;
                __syncthreads();

                fragmentA<OP_N,M16N16K16> fragA[2];
                fragmentB<OP_T,M16N16K16> fragB[2];
                for (int k = 0; k < 2; k++)
                {
                    for (int i = 0; i < 2; i++)
                    {
                        fragA[i].load(hShare, loadA + k*16 + stdAB*i*16, stdAB);
                        fragB[i].load(hShare, loadB + k*16 + stdAB*i*16, stdAB);
                    }
                    for (int i = 0; i < 2; i++)
                        for (int j = 0; j < 2; j++)
                            fragC[i][j].mma_sync(fragA[i], fragB[j]);

                }

            } while (++loop < loops);

        } while (p8 < params8);

        asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid) :);
        asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(bid) :);

        tx = tid % 16;
        ty = tid / 16;
        uint loadC = ty*stdC + tx*2;
        uint storC = fragmentC<OP_N,OP_T,M16N16K16>::get_idx(tid, stdC, (tid & 96));
        ehalf*  pC = C + (bid*32*32 + tid*2);

        for (int i = 0; i < 2; i++)
        {
            __syncthreads();
            for (int j = 0; j < 2; j++)
                fragC[i][j].store(fShare, storC + j*16, stdC);
            __syncthreads();

            for (uint j = 0; j < 2; j++)
            {
                uint sum2 = to_half2(
                    ew_add(
                        ew_add(
                            *(float2*)&fShare[loadC + j*8*stdC + 0*32],
                            *(float2*)&fShare[loadC + j*8*stdC + 1*32]),
                        ew_add(
                            *(float2*)&fShare[loadC + j*8*stdC + 2*32],
                            *(float2*)&fShare[loadC + j*8*stdC + 3*32])
                    )
                );

                if (accumulate)
                    reduce_half2(pC + i*512 + j*256, sum2);
                else
                    store_half2(pC + i*512 + j*256, sum2);
            }
        }
    }
    else if (!accumulate) // gate == 0
        zero_half8(C + (bid*32*32 + tid*8));
}

template <bool N128, bool GATED>
__global__ void __launch_bounds__(128) hgemm_blocksparse_16x16x128_nt_dds(
    struct Plist<ehalf,8> A,
    struct Plist<ehalf,8> B,
    ehalf*                C,
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    uint params8, uint N, uint loops, uint accumulate)
{
    const uint stdAB = 128 + 8;
    const uint stdC  = 64 + 16;

    __shared__ ehalf hShare[stdAB*2*16];
    float* fShare = (float*)hShare;

    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    float gate = GATED ? Gate[bid] : 1.0f;

    if (gate != 0.0f)
    {
        uint2 lut_head = Lut[bid];

        uint tx = tid % 16;
        uint ty = tid / 16;
        uint n0 = tx  *  8;

        uint idx_A = lut_head.x;
        uint idx_B = lut_head.y;
        uint offsetA0 = (idx_A*16 + ty)*N + n0;
        uint offsetB0 = (idx_B*16 + ty)*N + n0;
        uint storAB = ty*stdAB + n0;
        uint loadA = fragmentA<OP_N,M16N16K16>::get_idx(tid, stdAB, (tid & 96));
        uint loadB = fragmentB<OP_T,M16N16K16>::get_idx(tid, stdAB, (tid & 96) + 16*stdAB);

        fragmentC<OP_N,OP_T,M16N16K16> fragC;

        int p8 = 0;
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
                uint4 a0 = {0}, a8 = {0};
                uint4 b0 = {0}, b8 = {0};
                if (N128 || n < N)
                {
                    a0 = load_half8(A0 + (offsetA + N*0));
                    a8 = load_half8(A0 + (offsetA + N*8));
                    b0 = load_half8(B0 + (offsetB + N*0));
                    b8 = load_half8(B0 + (offsetB + N*8));
                }
                offsetA += 128;
                offsetB += 128;
                if (!N128)
                    n += 128;

                __syncthreads();
                *(uint4*)&hShare[storAB + 0*stdAB +  0*stdAB] = a0;
                *(uint4*)&hShare[storAB + 8*stdAB +  0*stdAB] = a8;
                *(uint4*)&hShare[storAB + 0*stdAB + 16*stdAB] = b0;
                *(uint4*)&hShare[storAB + 8*stdAB + 16*stdAB] = b8;
                __syncthreads();

                fragmentA<OP_N,M16N16K16> fragA;
                fragmentB<OP_T,M16N16K16> fragB;
                #pragma unroll
                for (uint j = 0; j < 2; j++)
                {
                    fragA.load(hShare, loadA + j*16, stdAB);
                    fragB.load(hShare, loadB + j*16, stdAB);

                    fragC.mma_sync(fragA, fragB);
                }

            } while (++loop < loops);

        } while (p8 < params8);

        asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid) :);
        asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(bid) :);

        tx = tid % 8;
        ty = tid / 8;
        uint loadC = ty*stdC + tx*2;
        uint storC = fragmentC<OP_N,OP_T,M16N16K16>::get_idx(tid, stdC, (tid & 96)/2);

        __syncthreads();
        fragC.store(fShare, storC, stdC);
        __syncthreads();

        uint sum2 = to_half2(
            ew_add(
                ew_add(
                    *(float2*)&fShare[loadC + 0*16],
                    *(float2*)&fShare[loadC + 1*16]
                ),
                ew_add(
                    *(float2*)&fShare[loadC + 2*16],
                    *(float2*)&fShare[loadC + 3*16]
                )
            )
        );
        ehalf* pC = C + (bid*16*16 + tid*2);
        if (accumulate)
            reduce_half2(pC, sum2);
        else
            store_half2( pC, sum2);
    }
    else if (!accumulate) // gate == 0
        zero_half2(C + (bid*16*16 + tid*2));
}

template <bool N128, bool GATED>
__global__ void __launch_bounds__(64) hgemm_blocksparse_8x8x128_nt_dds(
    struct Plist<ehalf,8> A,
    struct Plist<ehalf,8> B,
    ehalf*                C,
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    uint params8, uint N, uint loops, uint accumulate)
{
    const uint stdAB = 128 + 8;
    const uint stdC  = 16;

    __shared__ ehalf hShare[stdAB*8*2];
    float* fShare = (float*)hShare;

    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    float gate = GATED ? Gate[bid] : 1.0f;

    if (gate != 0.0f)
    {
        uint2 lut_head = Lut[bid];

        uint tx = tid % 16;
        uint ty = tid / 16;
        uint n0 = tx  *  8;

        uint idx_A = lut_head.x;
        uint idx_B = lut_head.y;
        uint offsetA0 = (idx_A*8 + ty)*N + n0;
        uint offsetB0 = (idx_B*8 + ty)*N + n0;
        uint storAB = ty*stdAB + n0;
        uint loadA = fragmentA<OP_N,M8N8K16>::get_idx(tid, stdAB, 0*stdAB + (tid & 32)*2);
        uint loadB = fragmentB<OP_T,M8N8K16>::get_idx(tid, stdAB, 8*stdAB + (tid & 32)*2);

        fragmentC<OP_N,OP_T,M8N8K16> fragC;

        int p8 = 0;
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

                uint4 a0 = {0}, a4 = {0};
                uint4 b0 = {0}, b4 = {0};

                if (N128 || n < N)
                {
                    a0 = load_half8(A0 + (offsetA + N*0));
                    a4 = load_half8(A0 + (offsetA + N*4));
                    b0 = load_half8(B0 + (offsetB + N*0));
                    b4 = load_half8(B0 + (offsetB + N*4));
                }
                offsetA += 128;
                offsetB += 128;
                if (!N128)
                    n += 128;

                __syncthreads();
                *(uint4*)&hShare[storAB + 0*stdAB + 0*stdAB] = a0;
                *(uint4*)&hShare[storAB + 4*stdAB + 0*stdAB] = a4;
                *(uint4*)&hShare[storAB + 0*stdAB + 8*stdAB] = b0;
                *(uint4*)&hShare[storAB + 4*stdAB + 8*stdAB] = b4;
                __syncthreads();

                fragmentA<OP_N,M8N8K16> fragA;
                fragmentB<OP_T,M8N8K16> fragB;
                #pragma unroll
                for (uint j = 0; j < 4; j++)
                {
                    fragA.load(hShare, loadA + j*16, stdAB);
                    fragB.load(hShare, loadB + j*16, stdAB);

                    fragC.mma_sync(fragA, fragB);
                }

            } while (++loop < loops);

        } while (p8 < params8);

        asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid)   :);
        asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(bid)   :);

        uint storC = fragmentC<OP_N,OP_T,M8N8K16>::get_idx(tid, stdC, (tid & 32)/4);

        __syncthreads();
        fragC.store(fShare, storC, stdC);
        __syncthreads();

        if (tid < 32)
        {
            tx = tid % 4;
            ty = tid / 4;
            uint loadC = ty*stdC + tx*2;
            uint sum2  = to_half2(
                ew_add(
                    *(float2*)&fShare[loadC + 0*8],
                    *(float2*)&fShare[loadC + 1*8]
                )
            );
            C += bid*8*8 + tid*2;
            if (accumulate)
                reduce_half2(C, sum2);
            else
                store_half2(C, sum2);
        }
    }
    else if (!accumulate && tid < 32) // gate == 0
        zero_half2(C + (bid*8*8 + tid*2));
}


#else // __CUDA_ARCH__ >= 700


template <uint OP_A, bool GATED>
__global__ void __launch_bounds__(256,3) hgemm_blocksparse_32x128x32_xn_sdd(
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint* Lock, uint locks, uint N, uint blk_a, uint blk_b, uint blk_N)
{
    *C = 0;
}
template <bool N128, bool GATED>
__global__ void __launch_bounds__(128,6) hgemm_blocksparse_32x32x128_nt_dds(
    struct Plist<ehalf,8> A,
    struct Plist<ehalf,8> B,
    ehalf*                C,
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    uint params8, uint N, uint loops, uint accumulate)
{
    *C = 0;
}
template <uint OP_A, bool GATED>
__global__ void __launch_bounds__(128) hgemm_blocksparse_16x128x16_xn_sdd(
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint* Lock, uint locks, uint N, uint blk_a, uint blk_b, uint blk_N)
{
    *C = 0;
}
template <bool N128, bool GATED>
__global__ void __launch_bounds__(128) hgemm_blocksparse_16x16x128_nt_dds(
    struct Plist<ehalf,8> A,
    struct Plist<ehalf,8> B,
    ehalf*                C,
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    uint params8, uint N, uint loops, uint accumulate)
{
    *C = 0;
}
template <uint OP_A, bool GATED>
__global__ void __launch_bounds__(128) hgemm_blocksparse_8x128x8_xn_sdd(
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint* Lock, uint locks, uint N, uint blk_a, uint blk_b, uint blk_N)
{
    *C = 0;
}
template <bool N128, bool GATED>
__global__ void __launch_bounds__(64) hgemm_blocksparse_8x8x128_nt_dds(
    struct Plist<ehalf,8> A,
    struct Plist<ehalf,8> B,
    ehalf*               C,
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    uint params8, uint N, uint loops, uint accumulate)
{
    *C = 0;
}

#endif // __CUDA_ARCH__ >= 700

cudaError_t hgemm_blocksparse_xn_128_sdd(const ehalf* X, const ehalf* W, ehalf* Y, bsmm_params* params, uint op)
{
    dim3 grid(params->blk_a*params->blk_b, params->blk_B, params->blk_A);
    uint blk_N = params->blk_b * params->blk_B;

    //cuMemsetD16Async((CUdeviceptr)Y, 0, params->K * params->N, params->stream);
    if (params->locks > 0)
        cuMemsetD32Async((CUdeviceptr)params->Lock, 0, blk_N * params->locks * 2, params->stream);

    const uint2* Lut = (const uint2*)params->Lut;
    uint* Lock       = (uint*)params->Lock;

    uint shared = params->shared + params->shared/2;

    if (params->bsize == 8)
    {
        shared += 4;
        if (params->Gate == 0)
        {
            if (op == OP_N)
                hgemm_blocksparse_8x128x8_xn_sdd<OP_N,false><<<grid,128,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
            else
                hgemm_blocksparse_8x128x8_xn_sdd<OP_T,false><<<grid,128,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
        }
        else
        {
            if (op == OP_N)
                hgemm_blocksparse_8x128x8_xn_sdd<OP_N, true><<<grid,128,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
            else
                hgemm_blocksparse_8x128x8_xn_sdd<OP_T, true><<<grid,128,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
        }
    }
    else if (params->bsize == 16)
    {
        if (params->Gate == 0)
        {
            if (op == OP_N)
                hgemm_blocksparse_16x128x16_xn_sdd<OP_N,false><<<grid,128,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
            else
                hgemm_blocksparse_16x128x16_xn_sdd<OP_T,false><<<grid,128,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
        }
        else
        {
            if (op == OP_N)
                hgemm_blocksparse_16x128x16_xn_sdd<OP_N, true><<<grid,128,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
            else
                hgemm_blocksparse_16x128x16_xn_sdd<OP_T, true><<<grid,128,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
        }
    }
    else if (params->bsize == 32)
    {
        if (params->Gate == 0)
        {
            if (op == OP_N)
                hgemm_blocksparse_32x128x32_xn_sdd<OP_N,false><<<grid,128,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
            else
                hgemm_blocksparse_32x128x32_xn_sdd<OP_T,false><<<grid,128,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
        }
        else
        {
            if (op == OP_N)
                hgemm_blocksparse_32x128x32_xn_sdd<OP_N, true><<<grid,128,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
            else
                hgemm_blocksparse_32x128x32_xn_sdd<OP_T, true><<<grid,128,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
        }
    }
    return cudaPeekAtLastError();
}
cudaError_t hgemm_blocksparse_xn_128_sdd(const bhalf* X, const bhalf* W, bhalf* Y, bsmm_params* params, uint op) { return cudaSuccess; }
cudaError_t hgemm_blocksparse_xn_128_sdd(const float* X, const float* W, float* Y, bsmm_params* params, uint op) { return cudaSuccess; }


cudaError_t hgemm_blocksparse_nt_128_dds(const ehalf* X, const ehalf* E, ehalf* U, bsmm_params* params)
{
    struct Plist<ehalf,8>* X8 = (struct Plist<ehalf,8>*)X;
    struct Plist<ehalf,8>* E8 = (struct Plist<ehalf,8>*)E;

    const uint2* Lut = (const uint2*)params->Lut;
    uint accumulate  = params->beta == 1.0f;
    uint pcount8     = params->pcount * 8;
    uint N           = params->N;
    uint loops       = CEIL_DIV(N, 128);
    bool k128        = (N & 127) == 0;

    dim3 grid(params->blocks, 1, 1);

    if (params->bsize == 8)
    {
        if (params->Gate == 0)
        {
            if (k128)
                hgemm_blocksparse_8x8x128_nt_dds< true,false><<<grid,64,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
            else
                hgemm_blocksparse_8x8x128_nt_dds<false,false><<<grid,64,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
        }
        else
        {
            if (k128)
                hgemm_blocksparse_8x8x128_nt_dds< true, true><<<grid,64,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
            else
                hgemm_blocksparse_8x8x128_nt_dds<false, true><<<grid,64,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
        }
    }
    else if (params->bsize == 16)
    {
        if (params->Gate == 0)
        {
            if (k128)
                hgemm_blocksparse_16x16x128_nt_dds< true,false><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
            else
                hgemm_blocksparse_16x16x128_nt_dds<false,false><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
        }
        else
        {
            if (k128)
                hgemm_blocksparse_16x16x128_nt_dds< true, true><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
            else
                hgemm_blocksparse_16x16x128_nt_dds<false, true><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
        }
    }
    else if (params->bsize == 32)
    {
        if (params->Gate == 0)
        {
            if (k128)
                hgemm_blocksparse_32x32x128_nt_dds< true,false><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
            else
                hgemm_blocksparse_32x32x128_nt_dds<false,false><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
        }
        else
        {
            if (k128)
                hgemm_blocksparse_32x32x128_nt_dds< true, true><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
            else
                hgemm_blocksparse_32x32x128_nt_dds<false, true><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
        }
    }
    return cudaPeekAtLastError();
}
cudaError_t hgemm_blocksparse_nt_128_dds(const bhalf* X, const bhalf* E, bhalf* U, bsmm_params* params) { return cudaSuccess; }
cudaError_t hgemm_blocksparse_nt_128_dds(const float* X, const float* E, float* U, bsmm_params* params) { return cudaSuccess; }



#endif // GOOGLE_CUDA
