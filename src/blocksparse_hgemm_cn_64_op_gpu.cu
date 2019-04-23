#if GOOGLE_CUDA

#include "ew_op_gpu.h"
#include "gpu_hmma.h"
#include <stdio.h>

#if __CUDA_ARCH__ >= 700

template <uint OP_A, bool GATED>
__global__ void __launch_bounds__(128) hgemm_blocksparse_32x64x32_xn_sdd(
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint* Lock, uint locks, uint N, uint blk_a, uint blk_b, uint blk_N)
{
    const uint stdA =  32 + 16;
    const uint stdB =  64 + 16;
    const uint stdC = 128 +  4;

    __shared__ float fShare[stdC*16];
    ehalf* hShare = (ehalf*)fShare;
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

            entry.y *= 32*32;
            entry.x *= N*32;
            LutOffsets[i] = entry;
        }
        __syncthreads();

        uint storA = ty*stdA + tx*4;
        uint storB = ty*stdB + tx*8 + stdA*32;

        uint loadA = fragmentA<OP_A,M16N16K16>::get_idx(tid, stdA, (tid & 64)*(OP_A == OP_N ? 1 : stdA)*16/64);
        uint loadB = fragmentB<OP_N,M16N16K16>::get_idx(tid, stdB, (tid & 64)*stdB*16/64 + (tid & 32) + stdA*32);

        uint       n = idx_N*64 + tx*8;
        uint offsetA = tid*4;
        uint offsetB = ty*N + n;

        asm(".reg .pred pn;\n\tsetp.lt.u32 pn, %0, %1;" :: "r"(n), "r"(N)); // n < N
        asm("mov.b32 %0, %0;" : "+r"(loadA) : );
        asm("mov.b32 %0, %0;" : "+r"(loadB) : );
        asm("mov.b32 %0, %0;" : "+r"(offsetA) : );
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

                const ehalf* pA = A + (entry.y + offsetA);
                uint2 a00 = load_half4(pA +  0*32);
                uint2 a16 = load_half4(pA + 16*32);
                uint4 b00, b16;

                asm("mov.b64  {%0,  %1}, 0; \n\t"
                    "mov.b64  {%2,  %3}, 0; \n\t"
                    "mov.b64  {%4,  %5}, 0; \n\t"
                    "mov.b64  {%6,  %7}, 0; \n\t"
                    "@pn ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%8];\n\t"
                    "@pn ld.global.nc.v4.u32 {%4, %5, %6, %7}, [%9];\n\t" :
                    "=r"(b00.x), "=r"(b00.y), "=r"(b00.z), "=r"(b00.w),
                    "=r"(b16.x), "=r"(b16.y), "=r"(b16.z), "=r"(b16.w) :
                    "l"(B + (entry.x + offsetB + N* 0)),
                    "l"(B + (entry.x + offsetB + N*16)));

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
                *(uint4*)&hShare[storB + 16*stdB] = b16;
                __syncthreads();

                fragmentA<OP_A,M16N16K16> fragA[2];
                fragmentB<OP_N,M16N16K16> fragB[2];
                for (int i = 0; i < 2; i++)
                {
                    fragA[i].load(hShare, loadA + (OP_A == OP_N ? stdA : 1)*i*16, stdA);
                    fragB[i].load(hShare, loadB + i*16, stdB);
                }
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 2; j++)
                        fragC[i][j].mma_sync(fragA[i], fragB[j]);
            }

        } while (++idx_lut < lut_size);

        asm volatile ("mov.u32 %0, %tid.x;" : "=r"(tid) :);

        uint txc = tid % 16;
        uint tyc = tid / 16;

        n = idx_N*64 + txc*4;
        uint loadC   = tyc*stdC + txc*4;
        uint storC   = fragmentC<OP_A,OP_N,M16N16K16>::get_idx(tid, stdC, tid & 96);
        uint offsetC = (idx_K*32 + tyc)*N + n;

        if (idx_Lock == 0)
        {
            for (int i = 0; i < 2; i++)
            {
                __syncthreads();
                for (int j = 0; j < 2; j++)
                    fragC[i][j].store(fShare, storC + j*16, stdC);
                __syncthreads();

                if (n < N)
                    for (int j = 0; j < 2; j++)
                        store_half4(C + (offsetC + N*(j*8 + i*16)), to_half4(ew_add(
                            *(float4*)&fShare[loadC + stdC*j*8 +  0],
                            *(float4*)&fShare[loadC + stdC*j*8 + 64])));
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
                for (int i = 0; i < 2; i++)
                {
                    __syncthreads();
                    for (int j = 0; j < 2; j++)
                        fragC[i][j].store(fShare, storC + j*16, stdC);
                    __syncthreads();

                    if (n < N)
                        for (int j = 0; j < 2; j++)
                            store_half4(C + (offsetC + N*(j*8 + i*16)), to_half4(ew_add(
                                *(float4*)&fShare[loadC + stdC*j*8 +  0],
                                *(float4*)&fShare[loadC + stdC*j*8 + 64])));
                }
                __threadfence();
                __syncthreads();

                if (tid == 0)
                    atomicExch(Lock, 0);
                // End Critial Section
            }
            else
            {
                txc = tid % 32;
                tyc = tid / 32;

                n       = idx_N*64 + txc*2;
                loadC   = tyc*stdC + txc*2;
                offsetC = (idx_K*32 + tyc)*N + n;

                // subsequent blocks must accumulate
                for (int i = 0; i < 2; i++)
                {
                    __syncthreads();
                    for (int j = 0; j < 2; j++)
                        fragC[i][j].store(fShare, storC + j*16, stdC);
                    __syncthreads();

                    if (n < N)
                        for (int j = 0; j < 4; j++)
                            reduce_half2(C + (offsetC + N*(j*4 + i*16)), to_half2(ew_add(
                                *(float2*)&fShare[loadC + stdC*j*4 +  0],
                                *(float2*)&fShare[loadC + stdC*j*4 + 64])));
                }
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
        uint n = idx_N*64 + tx*8;
        uint offsetC = (idx_K*32 + ty)*N + n;

        if (n < N)
        {
            zero_half8(C + (offsetC + N *0));
            zero_half8(C + (offsetC + N*16));
        }
    }
}

template <uint OP_A, bool GATED>
__global__ void __launch_bounds__(64) hgemm_blocksparse_16x64x16_xn_sdd(
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint* Lock, uint locks, uint N, uint blk_a, uint blk_b, uint blk_N)
{
    const uint stdA = 16;
    const uint stdB = 80;
    const uint stdC = 68;

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

    uint txb = tid % 8;
    uint tyb = tid / 8;

    if (lut_size > 0)
    {
        uint* Gates = (uint*)&LutOffsets[lut_size];

        // prefetch the lut and gate data into shared
        Lut += lut_offset;
        #pragma unroll 1
        for (uint i = tid; i < lut_size; i += 64)
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

        uint txa = tid % 4;
        uint tya = tid / 4;

        uint storA = tya*stdA + txa*4;
        uint storB = tyb*stdB + txb*8 + 16*stdA;

        uint loadA = fragmentA<OP_A,M16N16K16>::get_idx(tid, stdA);
        uint loadB = fragmentB<OP_N,M16N16K16>::get_idx(tid, stdB, 16*stdA + (tid & 32));

        uint       n = idx_N*64 + txb*8;
        uint offsetA = tid*4;
        uint offsetB = tyb*N + n;

        asm(".reg .pred pn;\n\tsetp.lt.u32 pn, %0, %1;" :: "r"(n), "r"(N)); // n < N
        asm("mov.b32 %0, %0;" : "+r"(loadA) : );
        asm("mov.b32 %0, %0;" : "+r"(loadB) : );
        asm("mov.b32 %0, %0;" : "+r"(offsetA) : );
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

                uint2 a0 = load_half4(A + (entry.y + offsetA));
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

                if (GATED)
                {
                    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(a0.x) : "r"(gate));
                    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(a0.y) : "r"(gate));
                }
                __syncthreads();
                *(uint2*)&hShare[storA] = a0;
                *(uint4*)&hShare[storB + 0*stdB] = b0;
                *(uint4*)&hShare[storB + 8*stdB] = b8;
                __syncthreads();

                fragmentA<OP_A,M16N16K16> fragA;
                fragmentB<OP_N,M16N16K16> fragB;

                fragA.load(hShare, loadA, stdA);
                #pragma unroll
                for (int j = 0; j < 2; j++)
                {
                    fragB.load(hShare, loadB + j*16, stdB);

                    fragC[j].mma_sync(fragA, fragB);
                }
            }

        } while (++idx_lut < lut_size);

        // allow assembler to forget these registers in the main loop
        asm volatile ("mov.u32 %0, %tid.x;" : "=r"(tid) :);

         // use thread stride of 4 to allow use of shared stride of 68
        // which minimizes shared bank conflicts on write.
        uint txc = tid % 16;
        uint tyc = tid / 16;

        n = idx_N*64 + txc*4;
        uint loadC   = tyc*stdC + txc*4;
        uint storC   = fragmentC<OP_A,OP_N,M16N16K16>::get_idx(tid, stdC, tid & 32);
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
                txc = tid % 32;
                tyc = tid / 32;

                n       = idx_N*64 + txc*2;
                loadC   = tyc*stdC + txc*2;
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
        uint n = idx_N*64 + txb*8;
        C += (idx_K*16 + tyb)*N + n;

        if (n < N)
        {
            zero_half8(C + N*0);
            zero_half8(C + N*8);
        }
    }
}

template <uint OP_A, bool GATED>
__global__ void __launch_bounds__(64) hgemm_blocksparse_8x64x8_xn_sdd(
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint* Lock, uint locks, uint N, uint blk_a, uint blk_b, uint blk_N)
{
    const uint stdA = 8;
    const uint stdB = 80;
    const uint stdC = 68;

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
        uint* Gates = (uint*)&LutOffsets[lut_size];

        // prefetch the lut and gate data into shared
        Lut += lut_offset;
        #pragma unroll 1
        for (uint i = tid; i < lut_size; i += 64)
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

            entry.y *= 8*8; // 64 entries of A per block
            entry.x *= N*8; // 8 lines of B per block
            LutOffsets[i] = entry;
        }
        if (tid == 0)
            Gates[lut_size] = 0; // needed if lut_size is odd

        __syncthreads();

        uint t32 = tid & 32;
        uint t31 = tid & 31;
        uint txb = tid % 8;
        uint tyb = t31 / 8;

        uint storA = tid*2;
        uint storB = tyb*stdB + txb*8 + t32*20 + 16*stdA;

        uint loadA = fragmentA<OP_A,M8N32K16>::get_idx(tid, stdA);
        uint loadB = fragmentB<OP_N,M8N32K16>::get_idx(tid, stdB, t32 + 16*stdA);

        uint       n = idx_N*64 + txb*8;
        uint offsetA = t31*2;
        uint offsetB = tyb*N + n;

        fragmentC<OP_A,OP_N,M8N32K16> fragC;

        uint idx_lut   = t32 / 32;
        uint idx_lut2  = 0;
        uint lut_size2 = (lut_size + 1)/2;

        asm(".reg .pred pn;\n\tsetp.lt.u32 pn, %0, %1;" :: "r"(n), "r"(N)); // n < N
        asm("mov.b32 %0, %0;" : "+r"(loadA) : );
        asm("mov.b32 %0, %0;" : "+r"(loadB) : );
        asm("mov.b32 %0, %0;" : "+r"(offsetA) : );
        asm("mov.b32 %0, %0;" : "+r"(offsetB) : );

        #pragma unroll 1
        do
        {
            uint  a0 = 0;
            uint4 b0 = {0};
            uint4 b4 = {0};

            uint gate = Gates[idx_lut];

            // if the gate is zero just skip over memory loads
            // we compute 2 blocks per loop so it's easier to just always do the mma math
            if (gate != 0)
            {
                uint2 entry = LutOffsets[idx_lut];
                a0 = load_half2(A + (entry.y + offsetA));

                asm("@pn ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%8];\n\t"
                    "@pn ld.global.nc.v4.u32 {%4, %5, %6, %7}, [%9];\n\t" :
                    "=r"(b0.x), "=r"(b0.y), "=r"(b0.z), "=r"(b0.w),
                    "=r"(b4.x), "=r"(b4.y), "=r"(b4.z), "=r"(b4.w) :
                    "l"(B + (entry.x + offsetB + N*0)),
                    "l"(B + (entry.x + offsetB + N*4)));

                if (GATED)
                    asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(a0) : "r"(gate));
            }

            // if (OP_A == OP_T)
            //     printf("%d %2d A:%08x B: %08x %08x %08x %08x %08x %08x %08x %08x\n", idx_K, tid, a0, b0.x,b0.y,b0.z,b0.w, b4.x,b4.y,b4.z,b4.w);

            __syncthreads();
            *(uint* )&hShare[storA         ] = a0;
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

        // allow assembler to forget these registers in the main loop
        asm volatile ("mov.u32 %0, %tid.x;" : "=r"(tid) :);

        // use thread stride of 4 to allow use of shared stride of 68
        // which minimizes shared bank conflicts on write.
        uint txc = tid % 16;
        uint tyc = tid / 16;

        n = idx_N*64 + txc*4;
        uint loadC   = tyc*stdC + txc*4;
        uint storC   = fragmentC<OP_A,OP_N,M8N32K16>::get_idx(tid, stdC, tid & 32);
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
                txc = tid % 32;
                tyc = tid / 32;

                n       = idx_N*64 + txc*2;
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
        uint txc = tid % 8;
        uint tyc = tid / 8;

        uint n       = idx_N*64 + txc*8;
        uint offsetC = (idx_K*8 + tyc)*N + n;

        if (n < N)
            zero_half8(C + offsetC);
    }
}

template <bool N64, bool GATED>
__global__ void __launch_bounds__(128) hgemm_blocksparse_32x32x64_nt_dds(
    struct Plist<ehalf,8> A,
    struct Plist<ehalf,8> B,
    ehalf*                C,
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    uint params8, uint N, uint loops, uint accumulate)
{
    const uint stdAB = 72;
    const uint stdC  = 132;

    __shared__ ehalf hShare[stdAB*2*32];
    float* fShare = (float*)hShare;

    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    float gate = GATED ? Gate[bid] : 1.0f;

    if (gate != 0.0f)
    {
        uint2 lut_head = Lut[bid];

        uint tx = tid % 8;
        uint ty = tid / 8;
        uint n0 = tx  * 8;

        uint idx_A = lut_head.x;
        uint idx_B = lut_head.y;
        uint offsetA0 = (idx_A*32 + ty)*N + n0;
        uint offsetB0 = (idx_B*32 + ty)*N + n0;
        uint storAB = ty*stdAB + n0;

        uint loadA = fragmentA<OP_N,M16N16K16>::get_idx(tid, stdAB, (tid & 96)/2);
        uint loadB = fragmentB<OP_T,M16N16K16>::get_idx(tid, stdAB, (tid & 96)/2 + stdAB*32);

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
                asm volatile (".pragma \"nounroll\";"::); // ptxas, don't get clever

                uint4 a00 = {0}, a16 = {0};
                uint4 b00 = {0}, b16 = {0};
                if (N64 || n < N)
                {
                    a00 = load_half8(A0 + (offsetA + N*00));
                    a16 = load_half8(A0 + (offsetA + N*16));
                    b00 = load_half8(B0 + (offsetB + N*00));
                    b16 = load_half8(B0 + (offsetB + N*16));
                }
                offsetA += 64;
                offsetB += 64;
                if (!N64)
                    n += 64;

                __syncthreads();
                *(uint4*)&hShare[storAB +  0*stdAB +  0*stdAB] = a00;
                *(uint4*)&hShare[storAB + 16*stdAB +  0*stdAB] = a16;
                *(uint4*)&hShare[storAB +  0*stdAB + 32*stdAB] = b00;
                *(uint4*)&hShare[storAB + 16*stdAB + 32*stdAB] = b16;
                __syncthreads();

                fragmentA<OP_N,M16N16K16> fragA[2];
                fragmentB<OP_T,M16N16K16> fragB[2];
                for (int i = 0; i < 2; i++)
                {
                    fragA[i].load(hShare, loadA + stdAB*i*16, stdAB);
                    fragB[i].load(hShare, loadB + stdAB*i*16, stdAB);
                }
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 2; j++)
                        fragC[i][j].mma_sync(fragA[i], fragB[j]);

            } while (++loop < loops);

        } while (p8 < params8);

        asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid) :);
        asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(bid) :);

        uint storC = fragmentC<OP_N,OP_T,M16N16K16>::get_idx(tid, stdC, (tid & 96));

        if (accumulate)
        {
            tx = tid % 16;
            ty = tid / 16;
            uint loadC   = ty*stdC + tx*2;
            uint offsetC = bid*32*32 + tid*2;

            for (int i = 0; i < 2; i++)
            {
                __syncthreads();
                for (int j = 0; j < 2; j++)
                    fragC[i][j].store(fShare, storC + j*16, stdC);
                __syncthreads();

                for (uint j = 0; j < 2; j++)
                {
                    float2 sum2 = ew_add(
                        ew_add(
                            *(float2*)&fShare[loadC + j*8*stdC +  0],
                            *(float2*)&fShare[loadC + j*8*stdC + 32]),
                        ew_add(
                            *(float2*)&fShare[loadC + j*8*stdC + 64],
                            *(float2*)&fShare[loadC + j*8*stdC + 96]));

                    reduce_half2(C + offsetC + i*4*128 + j*2*128, to_half2(sum2));
                }
            }
        }
        else
        {
            tx = tid % 8;
            ty = tid / 8;
            uint loadC   = ty*stdC + tx*4;
            uint offsetC = bid*32*32 + tid*4;

            for (int i = 0; i < 2; i++)
            {
                __syncthreads();
                for (int j = 0; j < 2; j++)
                    fragC[i][j].store(fShare, storC + j*16, stdC);
                __syncthreads();

                float4 sum4 = ew_add(
                    ew_add(
                        *(float4*)&fShare[loadC +  0],
                        *(float4*)&fShare[loadC + 32]),
                    ew_add(
                        *(float4*)&fShare[loadC + 64],
                        *(float4*)&fShare[loadC + 96]));

                store_half4(C + offsetC + i*4*128, to_half4(sum4));
            }
        }
    }
    else if (!accumulate) // gate == 0
        zero_half8(C + (bid*32*32 + tid*8));
}

template <bool N64, bool GATED>
__global__ void __launch_bounds__(64) hgemm_blocksparse_16x16x64_nt_dds(
    struct Plist<ehalf,8> A,
    struct Plist<ehalf,8> B,
    ehalf*                C,
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    uint params8, uint N, uint loops, uint accumulate)
{
    const uint stdAB = 72;
    const uint stdC  = 48;

    __shared__ ehalf hShare[stdAB*2*16];
    float* fShare = (float*)hShare;

    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    float gate = GATED ? Gate[bid] : 1.0f;

    if (gate != 0.0f)
    {
        uint2 lut_head = Lut[bid];

        uint tx = tid % 8;
        uint ty = tid / 8;
        uint n0 = tx  * 8;

        uint idx_A = lut_head.x;
        uint idx_B = lut_head.y;
        uint offsetA0 = (idx_A*16 + ty)*N + n0;
        uint offsetB0 = (idx_B*16 + ty)*N + n0;
        uint storAB = ty*stdAB + n0;
        uint loadA = fragmentA<OP_N,M16N16K16>::get_idx(tid, stdAB, (tid & 32));
        uint loadB = fragmentB<OP_T,M16N16K16>::get_idx(tid, stdAB, (tid & 32) + 16*stdAB);

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
                asm volatile (".pragma \"nounroll\";"::); // ptxas, don't get clever

                uint4 a0 = {0}, a8 = {0};
                uint4 b0 = {0}, b8 = {0};
                if (N64 || n < N)
                {
                    a0 = load_half8(A0 + (offsetA + N*0));
                    a8 = load_half8(A0 + (offsetA + N*8));
                    b0 = load_half8(B0 + (offsetB + N*0));
                    b8 = load_half8(B0 + (offsetB + N*8));
                }
                offsetA += 64;
                offsetB += 64;
                if (!N64)
                    n += 64;

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

        uint storC = fragmentC<OP_N,OP_T,M16N16K16>::get_idx(tid, stdC, (tid & 32)/2);

        __syncthreads();
        fragC.store(fShare, storC, stdC);
        __syncthreads();

        if (accumulate)
        {
            tx = tid % 8;
            ty = tid / 8;
            uint loadC   = ty*stdC + tx*2;
            uint offsetC = bid*16*16 + tid*2;

            for (uint i = 0; i < 2; i++)
                reduce_half2(C + offsetC + i*2*64, to_half2(ew_add(
                    *(float2*)&fShare[loadC + i*8*stdC +  0],
                    *(float2*)&fShare[loadC + i*8*stdC + 16])));
        }
        else
        {
            tx = tid % 4;
            ty = tid / 4;
            uint loadC   = ty*stdC + tx*4;
            uint offsetC = bid*16*16 + tid*4;

            store_half4(C + offsetC, to_half4(ew_add(
                *(float4*)&fShare[loadC +  0],
                *(float4*)&fShare[loadC + 16])));
        }
    }
    else if (!accumulate) // gate == 0
        zero_half4(C + (bid*16*16 + tid*4));
}

template <bool N64, bool GATED>
__global__ void __launch_bounds__(32) hgemm_blocksparse_8x8x64_nt_dds(
    struct Plist<ehalf,8> A,
    struct Plist<ehalf,8> B,
    ehalf*                C,
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    uint params8, uint N, uint loops, uint accumulate)
{
    const uint stdAB = 72;
    const uint stdC  = 8;

    __shared__ ehalf hShare[stdAB*8*2];
    float* fShare = (float*)hShare;

    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    float gate = GATED ? Gate[bid] : 1.0f;

    if (gate != 0.0f)
    {
        uint2 lut_head = Lut[bid];

        uint tx = tid % 8;
        uint ty = tid / 8;
        uint n0 = tx  * 8;

        uint idx_A = lut_head.x;
        uint idx_B = lut_head.y;
        uint offsetA0 = (idx_A*8 + ty)*N + n0;
        uint offsetB0 = (idx_B*8 + ty)*N + n0;
        uint storAB = ty*stdAB + n0;
        uint loadA = fragmentA<OP_N,M8N8K16>::get_idx(tid, stdAB, 0*stdAB);
        uint loadB = fragmentB<OP_T,M8N8K16>::get_idx(tid, stdAB, 8*stdAB);

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

                if (N64 || n < N)
                {
                    a0 = load_half8(A0 + (offsetA + N*0));
                    a4 = load_half8(A0 + (offsetA + N*4));
                    b0 = load_half8(B0 + (offsetB + N*0));
                    b4 = load_half8(B0 + (offsetB + N*4));
                }
                offsetA += 64;
                offsetB += 64;
                if (!N64)
                    n += 64;

                *(uint4*)&hShare[storAB + 0*stdAB + 0*stdAB] = a0;
                *(uint4*)&hShare[storAB + 4*stdAB + 0*stdAB] = a4;
                *(uint4*)&hShare[storAB + 0*stdAB + 8*stdAB] = b0;
                *(uint4*)&hShare[storAB + 4*stdAB + 8*stdAB] = b4;

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

        uint storC = fragmentC<OP_N,OP_T,M8N8K16>::get_idx(tid, stdC);

        fragC.store(fShare, storC, stdC);

        C += bid*8*8 + tid*2;

        uint c2 = to_half2(*(float2*)&fShare[tid*2]);

        if (accumulate)
            reduce_half2(C, c2);
        else
            store_half2(C, c2);
    }
    else if (!accumulate) // gate == 0
        zero_half2(C + (bid*8*8 + tid*2));
}


#else // __CUDA_ARCH__ >= 700

template <uint OP_A, bool GATED>
__global__ void __launch_bounds__(128) hgemm_blocksparse_32x64x32_xn_sdd(
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint* Lock, uint locks, uint N, uint blk_a, uint blk_b, uint blk_N)
{
    *C = 0;
}
template <bool N64, bool GATED>
__global__ void __launch_bounds__(128) hgemm_blocksparse_32x32x64_nt_dds(
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
__global__ void __launch_bounds__(64) hgemm_blocksparse_16x64x16_xn_sdd(
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint* Lock, uint locks, uint N, uint blk_a, uint blk_b, uint blk_N)
{
    *C = 0;
}
template <bool N64, bool GATED>
__global__ void __launch_bounds__(64) hgemm_blocksparse_16x16x64_nt_dds(
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
__global__ void __launch_bounds__(64) hgemm_blocksparse_8x64x8_xn_sdd(
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint* Lock, uint locks, uint N, uint blk_a, uint blk_b, uint blk_N)
{
    *C = 0;
}
template <bool N64, bool GATED>
__global__ void __launch_bounds__(32) hgemm_blocksparse_8x8x64_nt_dds(
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

cudaError_t hgemm_blocksparse_xn_64_sdd(const ehalf* X, const ehalf* W, ehalf* Y, bsmm_params* params, uint op)
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
                hgemm_blocksparse_8x64x8_xn_sdd<OP_N,false><<<grid,64,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
            else
                hgemm_blocksparse_8x64x8_xn_sdd<OP_T,false><<<grid,64,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
        }
        else
        {
            if (op == OP_N)
                hgemm_blocksparse_8x64x8_xn_sdd<OP_N, true><<<grid,64,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
            else
                hgemm_blocksparse_8x64x8_xn_sdd<OP_T, true><<<grid,64,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
        }
    }
    else if (params->bsize == 16)
    {
        if (params->Gate == 0)
        {
            if (op == OP_N)
                hgemm_blocksparse_16x64x16_xn_sdd<OP_N,false><<<grid,64,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
            else
                hgemm_blocksparse_16x64x16_xn_sdd<OP_T,false><<<grid,64,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
        }
        else
        {
            if (op == OP_N)
                hgemm_blocksparse_16x64x16_xn_sdd<OP_N, true><<<grid,64,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
            else
                hgemm_blocksparse_16x64x16_xn_sdd<OP_T, true><<<grid,64,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
        }
    }
    else if (params->bsize == 32)
    {
        // 256 = (128+4)*16*4 - (64+16 + 32+16)*32*2
        shared = shared > 256 ? shared - 256 : 0;
        if (params->Gate == 0)
        {
            if (op == OP_N)
                hgemm_blocksparse_32x64x32_xn_sdd<OP_N,false><<<grid,128,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
            else
                hgemm_blocksparse_32x64x32_xn_sdd<OP_T,false><<<grid,128,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
        }
        else
        {
            if (op == OP_N)
                hgemm_blocksparse_32x64x32_xn_sdd<OP_N, true><<<grid,128,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
            else
                hgemm_blocksparse_32x64x32_xn_sdd<OP_T, true><<<grid,128,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
        }
    }
    return cudaPeekAtLastError();
}
cudaError_t hgemm_blocksparse_xn_64_sdd(const bhalf* X, const bhalf* W, bhalf* Y, bsmm_params* params, uint op) { return cudaSuccess; }
cudaError_t hgemm_blocksparse_xn_64_sdd(const float* X, const float* W, float* Y, bsmm_params* params, uint op) { return cudaSuccess; }


cudaError_t hgemm_blocksparse_nt_64_dds(const ehalf* X, const ehalf* E, ehalf* U, bsmm_params* params)
{
    struct Plist<ehalf,8>* X8 = (struct Plist<ehalf,8>*)X;
    struct Plist<ehalf,8>* E8 = (struct Plist<ehalf,8>*)E;

    const uint2* Lut = (const uint2*)params->Lut;
    uint accumulate  = params->beta == 1.0f;
    uint pcount8     = params->pcount * 8;
    uint N           = params->N;
    uint loops       = CEIL_DIV(N, 64);
    bool k64         = (N & 63) == 0;

    dim3 grid(params->blocks, 1, 1);

    if (params->bsize == 8)
    {
        if (params->Gate == 0)
        {
            if (k64)
                hgemm_blocksparse_8x8x64_nt_dds< true,false><<<grid,32,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
            else
                hgemm_blocksparse_8x8x64_nt_dds<false,false><<<grid,32,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
        }
        else
        {
            if (k64)
                hgemm_blocksparse_8x8x64_nt_dds< true, true><<<grid,32,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
            else
                hgemm_blocksparse_8x8x64_nt_dds<false, true><<<grid,32,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
        }
    }
    else if (params->bsize == 16)
    {
        if (params->Gate == 0)
        {
            if (k64)
                hgemm_blocksparse_16x16x64_nt_dds< true,false><<<grid,64,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
            else
                hgemm_blocksparse_16x16x64_nt_dds<false,false><<<grid,64,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
        }
        else
        {
            if (k64)
                hgemm_blocksparse_16x16x64_nt_dds< true, true><<<grid,64,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
            else
                hgemm_blocksparse_16x16x64_nt_dds<false, true><<<grid,64,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
        }
    }
    else if (params->bsize == 32)
    {
        if (params->Gate == 0)
        {
            if (k64)
                hgemm_blocksparse_32x32x64_nt_dds< true,false><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
            else
                hgemm_blocksparse_32x32x64_nt_dds<false,false><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
        }
        else
        {
            if (k64)
                hgemm_blocksparse_32x32x64_nt_dds< true, true><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
            else
                hgemm_blocksparse_32x32x64_nt_dds<false, true><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
        }
    }
    return cudaPeekAtLastError();
}
cudaError_t hgemm_blocksparse_nt_64_dds(const bhalf* X, const bhalf* E, bhalf* U, bsmm_params* params) { return cudaSuccess; }
cudaError_t hgemm_blocksparse_nt_64_dds(const float* X, const float* E, float* U, bsmm_params* params) { return cudaSuccess; }

// dg = sum(dw * w, axis=1,2)
template <typename T, uint BSIZE, uint THREADS>
__global__ void __launch_bounds__(THREADS) blocksparse_gate_grad(T* DW_out, float* DG, const T* __restrict__ DW, const T* __restrict__ W, const float* __restrict__ G)
{
    const uint U = BSIZE*BSIZE/THREADS;

    uint bid = blockIdx.x;
    uint tid = threadIdx.x;
    uint offset = bid*BSIZE*BSIZE + tid;

    float g = G[bid];

    DW += offset;
    W  += offset;
    DW_out += offset;

    float dw[U], w[U];
    for (uint j = 0; j < U; j++)
    {
        dw[j] = load(DW, j*THREADS);
         w[j] = load( W, j*THREADS);

         store(DW_out, dw[j]*g, j*THREADS);
    }

    // Reduce max within this thread
    float dg = 0.0f;
    for (uint j = 0; j < U; j++)
        dg += ew_mul(dw[j], w[j]);

    // reduce within warp
    for (int i = 16; i > 0; i >>= 1)
        dg += shfl_xor(dg, i);

    // if using more than 1 warp, further reduced with shared memory
    if (THREADS > 32)
    {
        __shared__ float Share[32];

        // first thread of each warp store to shared
        if ((tid & 31) == 0)
            Share[tid / 32] = dg;

        __syncthreads();

        if (tid < 32)
        {
            // first warp loads all prior reductions
            dg = Share[tid];

            // reduce within this first warp
            for (int i = THREADS/64; i > 0; i >>= 1)
                dg += shfl_xor(dg, i);
        }
    }
    // first thread has the final reduced max_abs
    if (tid == 0)
        DG[bid] = dg;
}
template <typename T>
bool BlocksparseGateGrad(CUstream stream, T* dw_out, float* dg, const T* dw, const T* w, const float* g, uint blocks, uint bsize)
{
         if (bsize ==  8)
        blocksparse_gate_grad<T, 8,  32><<<blocks,  32,0,stream>>>(dw_out, dg, dw, w, g);
    else if (bsize == 16)
        blocksparse_gate_grad<T,16,  64><<<blocks,  64,0,stream>>>(dw_out, dg, dw, w, g);
    else if (bsize == 32)
        blocksparse_gate_grad<T,32, 256><<<blocks, 256,0,stream>>>(dw_out, dg, dw, w, g);
    else if (bsize == 64)
        blocksparse_gate_grad<T,64,1024><<<blocks,1024,0,stream>>>(dw_out, dg, dw, w, g);
    return true;
}
template bool BlocksparseGateGrad<float>(CUstream stream, float* dw_out, float* dg, const float* dw, const float* w, const float* g, uint blocks, uint bsize);
template bool BlocksparseGateGrad<ehalf>(CUstream stream, ehalf* dw_out, float* dg, const ehalf* dw, const ehalf* w, const float* g, uint blocks, uint bsize);


#define MAX_NORM 0
#define L2_NORM  1

#if __CUDA_ARCH__ >= 700

template <uint BSIZE, uint NORM>
__global__ void __launch_bounds__(256,4) blocksparse_feature_reduce_cn(
    const struct Plist<ehalf,8> X8, ehalf* Y, uint params, uint N)
{
    const ehalf* X;

    uint tid = threadIdx.x;
    uint bn  = blockIdx.x;
    uint bc  = blockIdx.y;

    // Each warp works on a Plist entry
    uint p  = tid / 32; // index into Plist
    uint tp = tid % 32;
    asm("ld.param.u64 %0, [%1 + 0x160];" : "=l"(X) : "r"(p * 8));

    uint tn = tp % 8;
    uint tc = tp / 8;
    uint n  = bn*64 + tn*8;

    X += (bc*BSIZE + tc)*N + n;
    Y += bc*params*N + p*N + n;

    asm("mov.b64 %0, %0;" : "+l"(X) : );
    asm("mov.b64 %0, %0;" : "+l"(Y) : );

    float8 norm;
    ew_zero(norm);
    bool n_valid = n < N;

    #pragma unroll 4
    for (uint c = 0; c < BSIZE; c += 4)
    {
        float8 x = load((const ehalf8*)(X + c*N), 0, n_valid);

        if (NORM == MAX_NORM)
            norm = ew_maximum(ew_abs(x), norm);
        else
            norm = ew_add(ew_sqr(x), norm);
    }

    if (NORM == MAX_NORM)
    {
        #pragma unroll
        for (int i = 16; i > 4; i >>= 1)
            norm = ew_warp_max(norm, i);
    }
    else
    {
        #pragma unroll
        for (int i = 16; i > 4; i >>= 1)
            norm = ew_warp_sum(norm, i);
        norm = ew_sqrt(norm);
    }

    store((ehalf8*)Y, norm, 0, n_valid && tp < 8);
}

template <bool M32, bool ACCUMULATE>
__device__  __noinline__  void store_32x32x64_nt(float* C, uint loadC, uint M, uint N, uint cy, uint cx, uint i, uint stdC, float scale)
{
    for (uint j = 0; j < 4; j++)
        if (M32 || cy + i*16 + j*4 < M)
        {
            float out = ew_zero_nan_inf(ew_mul(ew_add(
                ew_add(
                    ld_shared_float1(loadC + j*4*stdC + 0*32),
                    ld_shared_float1(loadC + j*4*stdC + 1*32)),
                ew_add(
                    ld_shared_float1(loadC + j*4*stdC + 2*32),
                    ld_shared_float1(loadC + j*4*stdC + 3*32))), scale));

            uint offsetC = (cy + i*16 + j*4)*N + cx;
            if (ACCUMULATE)
                atomicRed(C + offsetC, out);
            else
                store(C + offsetC, out);
        }
}
template <bool M32, bool ACCUMULATE>
__global__ void __launch_bounds__(128,8) hgemm_32x32x64_nt(
    const ehalf* A,
    const ehalf* B,
          float* C,
    uint M, uint N, uint K, uint blk_a, uint blk_b, float scale)
{
    const uint stdAB = 72;
    const uint stdC  = 132;

    __shared__ ehalf hShare[stdAB*2*32];
    float* fShare = (float*)hShare;

    uint tid = threadIdx.x;

    uint idx_ab = blockIdx.x;
    uint idx_B  = blockIdx.y;
    uint idx_A  = blockIdx.z;

    idx_A = idx_A * blk_a + idx_ab / blk_b;
    idx_B = idx_B * blk_b + idx_ab % blk_b;

    uint tx = tid % 8;
    uint ty = tid / 8;
    uint tk = tx  * 8;
    uint ta = idx_A*32 + ty;
    uint tb = idx_B*32 + ty;

    uint offsetA = ta*K + tk;
    uint offsetB = tb*K + tk;
    uint storAB  = ty*stdAB + tk;

    uint loadA = fragmentA<OP_N,M16N16K16>::get_idx(tid, stdAB, (tid & 96)/2);
    uint loadB = fragmentB<OP_T,M16N16K16>::get_idx(tid, stdAB, (tid & 96)/2 + stdAB*32);

    asm(".reg .pred a00, a16, b00, b16; \n\t"
        "setp.lt.u32 a00, %0, %2;       \n\t"
        "setp.lt.u32 a16, %1, %2;       \n\t"
        "setp.lt.u32 b00, %3, %5;       \n\t"
        "setp.lt.u32 b16, %4, %5;       \n\t" ::
        "r"(ta), "r"(ta+16), "r"(M),
        "r"(tb), "r"(tb+16), "r"(N) );

    fragmentC<OP_N,OP_T,M16N16K16> fragC[2][2];

    #pragma unroll 1
    for (uint k = 0; k < K; k += 64)
    {
        uint4 a00, a16, b00, b16;

        asm volatile("{\n\t"
            ".reg .pred ka00, ka16, kb00, kb16;   \n\t"
            "setp.lt.and.u32 ka00, %20, %21, a00; \n\t"
            "setp.lt.and.u32 ka16, %20, %21, a16; \n\t"
            "setp.lt.and.u32 kb00, %20, %21, b00; \n\t"
            "setp.lt.and.u32 kb16, %20, %21, b16; \n\t"
            "mov.b64  { %0,  %1}, 0; \n\t"
            "mov.b64  { %2,  %3}, 0; \n\t"
            "mov.b64  { %4,  %5}, 0; \n\t"
            "mov.b64  { %6,  %7}, 0; \n\t"
            "mov.b64  { %8,  %9}, 0; \n\t"
            "mov.b64  {%10, %11}, 0; \n\t"
            "mov.b64  {%12, %13}, 0; \n\t"
            "mov.b64  {%14, %15}, 0; \n\t"
            "@ka00 ld.global.nc.v4.u32 { %0,  %1,  %2,  %3}, [%16]; \n\t"
            "@ka16 ld.global.nc.v4.u32 { %4,  %5,  %6,  %7}, [%17]; \n\t"
            "@kb00 ld.global.nc.v4.u32 { %8,  %9, %10, %11}, [%18]; \n\t"
            "@kb16 ld.global.nc.v4.u32 {%12, %13, %14, %15}, [%19]; \n\t"
            "}" :
            "=r"(a00.x), "=r"(a00.y), "=r"(a00.z), "=r"(a00.w),
            "=r"(a16.x), "=r"(a16.y), "=r"(a16.z), "=r"(a16.w),
            "=r"(b00.x), "=r"(b00.y), "=r"(b00.z), "=r"(b00.w),
            "=r"(b16.x), "=r"(b16.y), "=r"(b16.z), "=r"(b16.w) :
            "l"(A + (offsetA +  0*K)),
            "l"(A + (offsetA + 16*K))
            "l"(B + (offsetB +  0*K)),
            "l"(B + (offsetB + 16*K)),
            "r"(tk), "r"(K) );
        offsetA += 64;
        offsetB += 64;
        tk      += 64;

        __syncthreads();
        *(uint4*)&hShare[storAB +  0*stdAB +  0*stdAB] = a00;
        *(uint4*)&hShare[storAB + 16*stdAB +  0*stdAB] = a16;
        *(uint4*)&hShare[storAB +  0*stdAB + 32*stdAB] = b00;
        *(uint4*)&hShare[storAB + 16*stdAB + 32*stdAB] = b16;
        __syncthreads();

        fragmentA<OP_N,M16N16K16> fragA[2];
        fragmentB<OP_T,M16N16K16> fragB[2];
        for (int i = 0; i < 2; i++)
        {
            fragA[i].load(hShare, loadA + stdAB*i*16, stdAB);
            fragB[i].load(hShare, loadB + stdAB*i*16, stdAB);
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

    tx = tid % 32;
    ty = tid / 32;
    uint cx = idx_B*32 + tx;
    uint cy = idx_A*32 + ty;
    uint loadC = ty*stdC + tx;
    uint storC = fragmentC<OP_N,OP_T,M16N16K16>::get_idx(tid, stdC, (tid & 96));

    bool cx_valid = cx < N;

    for (int i = 0; i < 2; i++)
    {
        __syncthreads();
        for (int j = 0; j < 2; j++)
            fragC[i][j].store(fShare, storC + j*16, stdC);
        __syncthreads();

        if (cx_valid)
            store_32x32x64_nt<M32,ACCUMULATE>(C, loadC, M, N, cy, cx, i, stdC, scale);
    }
}

#else // __CUDA_ARCH__ >= 700

template <uint BSIZE, uint NORM>
__global__ void __launch_bounds__(256,4) blocksparse_feature_reduce_cn(
    const struct Plist<ehalf,8> X8, ehalf* Y, uint params, uint N)
{
    *Y = 0;
}
template <bool M32, bool ACCUMULATE>
__global__ void __launch_bounds__(128,8) hgemm_32x32x64_nt(
    const ehalf* A,
    const ehalf* B,
          float* C,
    uint M, uint N, uint K, uint blk_a, uint blk_b, float scale)
{
    *C = 0;
}

#endif // __CUDA_ARCH__ >= 700


bool BlocksparseFeatureReduceCN(CUstream stream, ehalf* Y, const struct Plist<ehalf,8>* X8, uint params, uint C, uint N, uint bshift, uint norm_type)
{
    dim3 grid(CEIL_DIV(N, 64), C >> bshift, 1);
    uint threads = params * 32;

    if (norm_type == MAX_NORM)
    {
        if (bshift == 3)
            blocksparse_feature_reduce_cn< 8,MAX_NORM><<<grid,threads,0,stream>>>(*X8, Y, params, N);
        else if (bshift == 4)
            blocksparse_feature_reduce_cn<16,MAX_NORM><<<grid,threads,0,stream>>>(*X8, Y, params, N);
        else
            blocksparse_feature_reduce_cn<32,MAX_NORM><<<grid,threads,0,stream>>>(*X8, Y, params, N);
    }
    else
    {
        if (bshift == 3)
            blocksparse_feature_reduce_cn< 8, L2_NORM><<<grid,threads,0,stream>>>(*X8, Y, params, N);
        else if (bshift == 4)
            blocksparse_feature_reduce_cn<16, L2_NORM><<<grid,threads,0,stream>>>(*X8, Y, params, N);
        else
            blocksparse_feature_reduce_cn<32, L2_NORM><<<grid,threads,0,stream>>>(*X8, Y, params, N);
    }
    return true;
}

bool hGemmNT(CUstream stream, const ehalf* A, const ehalf* B, float* C, uint M, uint N, uint K, uint blk_a, uint blk_b, uint blk_A, uint blk_B, uint accumulate, float scale)
{
    if (scale != 0.0f)
    {
        dim3 grid(blk_a*blk_b, blk_B, blk_A);
        if (M & 31)
            if (accumulate)
                hgemm_32x32x64_nt<false, true><<<grid,128,0,stream>>>(A, B, C, M, N, K, blk_a, blk_b, scale);
            else
                hgemm_32x32x64_nt<false,false><<<grid,128,0,stream>>>(A, B, C, M, N, K, blk_a, blk_b, scale);
        else
            if (accumulate)
                hgemm_32x32x64_nt< true, true><<<grid,128,0,stream>>>(A, B, C, M, N, K, blk_a, blk_b, scale);
            else
                hgemm_32x32x64_nt< true,false><<<grid,128,0,stream>>>(A, B, C, M, N, K, blk_a, blk_b, scale);
    }
    else if (accumulate == 0)
        cuMemsetD32Async((CUdeviceptr)C, 0, M*N, stream);

    return true;
}



#endif // GOOGLE_CUDA
