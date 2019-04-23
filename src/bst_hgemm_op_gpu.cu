#if GOOGLE_CUDA

#include "ew_op_gpu.h"
#include "gpu_hmma.h"
#include <stdio.h>

#if __CUDA_ARCH__ >= 700


// C = A * B  or  A.T * B
// Dims: M, N, K
// N64: N even mult of 64
// A is sparse, B is dense, C is dense

// 32x64x16 warp tile
template <uint OP_A, bool N64>
__global__ void __launch_bounds__(256) bst_hgemm_64x64x64_xn(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint szCtxHeadStateB, uint szCtxHeadStateC, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint grid_M, uint grid_N, uint magic_N, uint shift_N)
{
    const uint stdA = 64+16;
    const uint stdB = 64+16;
    const uint stdC = 512+4;

    __shared__ float fShare[stdC*16];
    ehalf* hShare = (ehalf*)fShare;
    uint2* Lut2s = (uint2*)&fShare[stdC*16];

    uint tid    = threadIdx.x;
    uint idx_MN = blockIdx.x; // compound outer product dims
    uint idx_M  = div64(idx_MN, magic_N, shift_N); // idx_M = idx_MN / grid_N;
    uint idx_N  = idx_MN - idx_M*grid_N;           // idx_N = idx_MN % grid_N;
    uint idx_B  = blockIdx.y; // batch dim
    uint idx_H  = blockIdx.z; // head dim

    // assume lower diagonal and schedule large reductions first
    if (OP_A == OP_N)
        idx_M = grid_M - idx_M;

    // each head can optionally have its own lut
    Lut += idx_H*szLut;
    uint2 lut_head   = Lut[idx_M];
    uint  lut_offset = lut_head.x;
    uint  lut_size   = lut_head.y;

    uint tx = tid % 8;
    uint ty = tid / 8;

    if (lut_size > 0)
    {
        // prefetch the lut data into shared
        Lut += lut_offset;
        #pragma unroll 1
        for (uint i = tid; i < lut_size; i += 256)
        {
            uint2 entry = Lut[i];
            entry.x *= 64*64;  // 4096 entries of A per block
            entry.y *= szHeadState*64;   // 64 lines of B per block
            Lut2s[i] = entry;
        }
        __syncthreads();

        uint storAB = ty*stdA + tx*8; // assume stdA == stdB

        uint loadA = fragmentA<OP_A,M16N16K16>::get_idx(tid, stdA, (tid & 192)*(OP_A == OP_N ? 1 : stdA)*16/64 + (tid & 32)*(OP_A == OP_N ? stdA : 1));
        uint loadB = fragmentB<OP_N,M16N16K16>::get_idx(tid, stdB, (tid & 192)*stdB*16/64 + stdA*64);

        uint b = idx_N*64 + tx*8;
        uint offsetA = idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + tid*8;
        uint offsetB = idx_B*szCtxHeadStateB + ty*szHeadState + idx_H*szState + b;

        bool inB = N64 || b < szState;

        fragmentC<OP_A,OP_N,M16N16K16> fragC[2][4];

        int idx_lut = 0;
        #pragma unroll 1
        do
        {
            asm volatile (".pragma \"nounroll\";"::); // ptxas, don't get clever

            uint2 entry = Lut2s[idx_lut];
            uint4 b00 = {0};
            uint4 b32 = {0};

            entry.x += offsetA;
            entry.y += offsetB;

            uint4 a00 = *(uint4*)&A[entry.x +  0*64];
            uint4 a32 = *(uint4*)&A[entry.x + 32*64];
            if (inB)
            {
                b00 = *(uint4*)&B[entry.y +  0*szHeadState];
                b32 = *(uint4*)&B[entry.y + 32*szHeadState];
            }
            __syncthreads();
            *(uint4*)&hShare[storAB +  0*stdA +  0*stdA] = a00;
            *(uint4*)&hShare[storAB + 32*stdA +  0*stdA] = a32;
            *(uint4*)&hShare[storAB +  0*stdB + 64*stdA] = b00;
            *(uint4*)&hShare[storAB + 32*stdB + 64*stdA] = b32;
            __syncthreads();

            fragmentA<OP_A,M16N16K16> fragA[2];
            fragmentB<OP_N,M16N16K16> fragB[4];
            for (int i = 0; i < 2; i++)
                fragA[i].load(hShare, loadA + (OP_A == OP_N ? stdA : 1)*i*16, stdA);

            for (int i = 0; i < 4; i++)
                fragB[i].load(hShare, loadB + i*16, stdB);

            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 4; j++)
                    fragC[i][j].mma_sync(fragA[i], fragB[j]);

        } while (++idx_lut < lut_size);

        asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid  )  :);
        asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(idx_MN) :);
        asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(idx_B)  :);
        asm volatile ("mov.u32 %0, %ctaid.z;" : "=r"(idx_H)  :);
        idx_M  = div64(idx_MN, magic_N, shift_N);
        idx_N  = idx_MN - idx_M*grid_N;

        if (OP_A == OP_N)
            idx_M = grid_M - idx_M;

        uint txc = tid % 16;
        uint tyc = tid / 16;

        uint c       = idx_N*64 + txc*4;
        uint loadC   = tyc*stdC + txc*4;
        uint storC   = fragmentC<OP_A,OP_N,M16N16K16>::get_idx(tid, stdC, (tid & 224)*2);
        uint offsetC = idx_B*szCtxHeadStateC + (idx_M*64 + tyc)*szHeadState + idx_H*szState + c;

        for (int i = 0; i < 2; i++)
        {
            __syncthreads();
            for (int j = 0; j < 4; j++)
                fragC[i][j].store(fShare, storC + j*16, stdC);
            __syncthreads();

            if (N64 || c < szState)
            {
                for (int j = 0; j < 2; j++)
                    *(uint2*)&C[offsetC + szHeadState*(j*32 + i*16)] = to_half4(
                        ew_add(
                            ew_add(
                                *(float4*)&fShare[loadC + j*64 + 0*128],
                                *(float4*)&fShare[loadC + j*64 + 1*128]),
                            ew_add(
                                *(float4*)&fShare[loadC + j*64 + 2*128],
                                *(float4*)&fShare[loadC + j*64 + 3*128])
                        )
                    );
            }
        }
    }
    else
    {
        uint c       = idx_N*64 + tx*8;
        uint offsetC = idx_B*szCtxHeadStateC + (idx_M*64 + ty)*szHeadState + idx_H*szState + c;

        if (N64 || c < szState)
        {
            uint4 zero = {0};
            *(uint4*)&C[offsetC + szHeadState* 0] = zero;
            *(uint4*)&C[offsetC + szHeadState*32] = zero;
        }
    }
}

// C = A * B  or  A.T * B
// Dims: M, N, K
// N64: N even mult of 64
// A is sparse, B is dense, C is dense

template <uint OP_A, bool N64>
__global__ void __launch_bounds__(128) bst_hgemm_32x64x32_xn(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint szCtxHeadStateB, uint szCtxHeadStateC, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint grid_M, uint grid_N, uint magic_N, uint shift_N)
{
    const uint stdA = 48;
    const uint stdB = 80;
    const uint stdC = 132;

    __shared__ float fShare[stdC*16]; // stdC*16*4 > (stdA + stdB)*32*2
    ehalf* hShare = (ehalf*)fShare;
    uint2* Lut2s = (uint2*)&hShare[(stdA + stdB)*32];

    uint tid    = threadIdx.x;
    uint idx_MN = blockIdx.x; // compound outer product dims
    uint idx_M  = div64(idx_MN, magic_N, shift_N); // idx_M = idx_MN / grid_N;
    uint idx_N  = idx_MN - idx_M*grid_N;           // idx_N = idx_MN % grid_N;
    uint idx_B  = blockIdx.y; // batch dim
    uint idx_H  = blockIdx.z; // head dim

    // assume lower diagonal and schedule large reductions first
    if (OP_A == OP_N)
        idx_M = grid_M - idx_M;

    // each head can optionally have its own lut
    Lut += idx_H*szLut;
    uint2 lut_head   = Lut[idx_M];
    uint  lut_offset = lut_head.x;
    uint  lut_size   = lut_head.y;

    uint tx = tid % 8;
    uint ty = tid / 8;

    if (lut_size > 0)
    {
        // prefetch the lut data into shared
        Lut += lut_offset;
        #pragma unroll 1
        for (uint i = tid; i < lut_size; i += 128)
        {
            uint2 entry = Lut[i];
            entry.x *= 32*32;  // 1024 entries of A per block
            entry.y *= szHeadState*32;   // 32 lines of B per block
            Lut2s[i] = entry;
        }
        __syncthreads();

        uint storA = ty*stdA + tx*4;
        uint storB = ty*stdB + tx*8 + stdA*32;

        uint loadA = fragmentA<OP_A,M16N16K16>::get_idx(tid, stdA, (tid & 64)*(OP_A == OP_N ? 1 : stdA)*16/64);
        uint loadB = fragmentB<OP_N,M16N16K16>::get_idx(tid, stdB, (tid & 64)*stdB*16/64 + (tid & 32) + stdA*32);

        uint b = idx_N*64 + tx*8;
        uint offsetA = idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + tid*4;
        uint offsetB = idx_B*szCtxHeadStateB + ty*szHeadState + idx_H*szState + b;

        bool inB = N64 || b < szState;

        fragmentC<OP_A,OP_N,M16N16K16> fragC[2][2];

        int idx_lut = 0;
        #pragma unroll 1
        do
        {
            asm volatile (".pragma \"nounroll\";"::); // ptxas, don't get clever

            uint2 entry = Lut2s[idx_lut];
            uint4 b00 = {0};
            uint4 b16 = {0};

            const ehalf* pA = A + (entry.x + offsetA);
            uint2 a00 = *(uint2*)&pA[ 0*32];
            uint2 a16 = *(uint2*)&pA[16*32];
            if (inB)
            {
                b00 = *(uint4*)&B[entry.y + offsetB +  0*szHeadState];
                b16 = *(uint4*)&B[entry.y + offsetB + 16*szHeadState];
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

        } while (++idx_lut < lut_size);

        asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid  )  :);
        asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(idx_MN) :);
        asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(idx_B)  :);
        asm volatile ("mov.u32 %0, %ctaid.z;" : "=r"(idx_H)  :);
        idx_M  = div64(idx_MN, magic_N, shift_N);
        idx_N  = idx_MN - idx_M*grid_N;

        if (OP_A == OP_N)
            idx_M = grid_M - idx_M;

        uint txc = tid % 16;
        uint tyc = tid / 16;

        uint c       = idx_N*64 + txc*4;
        uint loadC   = tyc*stdC + txc*4;
        uint storC   = fragmentC<OP_A,OP_N,M16N16K16>::get_idx(tid, stdC, tid & 96);
        uint offsetC = idx_B*szCtxHeadStateC + (idx_M*32 + tyc)*szHeadState + idx_H*szState + c;

        for (int i = 0; i < 2; i++)
        {
            __syncthreads();
            for (int j = 0; j < 2; j++)
                fragC[i][j].store(fShare, storC + j*16, stdC);
            __syncthreads();

            if (N64 || c < szState)
            {
                for (int j = 0; j < 2; j++)
                    *(uint2*)&C[offsetC + szHeadState*(j*8 + i*16)] = to_half4(ew_add(
                        *(float4*)&fShare[loadC + stdC*j*8 +  0],
                        *(float4*)&fShare[loadC + stdC*j*8 + 64]));
            }
        }
    }
    else
    {
        uint c       = idx_N*64 + tx*8;
        uint offsetC = idx_B*szCtxHeadStateC + (idx_M*32 + ty)*szHeadState + idx_H*szState + c;

        if (N64 || c < szState)
        {
            uint4 zero = {0};
            *(uint4*)&C[offsetC + szHeadState* 0] = zero;
            *(uint4*)&C[offsetC + szHeadState*16] = zero;
        }
    }
}

// C = A * B  or  A.T * B
// Dims: M, N, K
// N64: N even mult of 64
// A is sparse, B is dense, C is dense

template <uint OP_A, bool N64>
__global__ void __launch_bounds__(64) bst_hgemm_16x64x16_xn(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint szCtxHeadStateB, uint szCtxHeadStateC, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint grid_M, uint grid_N, uint magic_N, uint shift_N)
{
    const uint stdA = 16;
    const uint stdB = 80;
    const uint stdC = 68;

    __shared__ ehalf hShare[(stdA + stdB)*16];
    uint2* Lut2s = (uint2*)&hShare[(stdA + stdB)*16];

    uint tid    = threadIdx.x;
    uint idx_MN = blockIdx.x; // compound outer product dims
    uint idx_M  = div64(idx_MN, magic_N, shift_N); // idx_M = idx_MN / grid_N;
    uint idx_N  = idx_MN - idx_M*grid_N;           // idx_N = idx_MN % grid_N;
    uint idx_B  = blockIdx.y; // batch dim
    uint idx_H  = blockIdx.z; // head dim

    // assume lower diagonal and schedule large reductions first
    if (OP_A == OP_N)
        idx_M = grid_M - idx_M;

    // each head can optionally have its own lut
    Lut += idx_H*szLut;
    uint2 lut_head   = Lut[idx_M];
    uint  lut_offset = lut_head.x;
    uint  lut_size   = lut_head.y;

    uint txb = tid % 8;
    uint tyb = tid / 8;

    if (lut_size > 0)
    {
        // prefetch the lut data into shared
        Lut += lut_offset;
        #pragma unroll 1
        for (uint i = tid; i < lut_size; i += 64)
        {
            uint2 entry = Lut[i];
            entry.x *= 16*16;          // 256 entries of A per block
            entry.y *= szHeadState*16; // 16 lines of B per block
            Lut2s[i] = entry;
        }
        __syncthreads();

        uint txa = tid % 4;
        uint tya = tid / 4;

        uint storA = tya*stdA + txa*4;
        uint storB = tyb*stdB + txb*8 + 16*stdA;

        uint loadA = fragmentA<OP_A,M16N16K16>::get_idx(tid, stdA);
        uint loadB = fragmentB<OP_N,M16N16K16>::get_idx(tid, stdB, 16*stdA + (tid & 32));

        uint       b = idx_N*64 + txb*8;
        bool     inB = N64 || b < szState;
        uint offsetA = idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + tid*4;
        uint offsetB = idx_B*szCtxHeadStateB + tyb*szHeadState + idx_H*szState + b;

        fragmentC<OP_A,OP_N,M16N16K16> fragC[2];

        int idx_lut = 0;
        #pragma unroll 1
        do
        {
            asm volatile (".pragma \"nounroll\";"::); // ptxas, don't get clever

            uint2 entry = Lut2s[idx_lut];
            uint4 b0 = {0};
            uint4 b8 = {0};

            entry.x += offsetA;
            entry.y += offsetB;

            uint2 a0 = *(uint2*)&A[entry.x];
            if (inB)
            {
                b0 = *(uint4*)&B[entry.y + 0*szHeadState];
                b8 = *(uint4*)&B[entry.y + 8*szHeadState];
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

        } while (++idx_lut < lut_size);

        // allow assembler to forget these registers in the main loop
        asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid  )  :);
        asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(idx_MN) :);
        asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(idx_B)  :);
        asm volatile ("mov.u32 %0, %ctaid.z;" : "=r"(idx_H)  :);
        idx_M  = div64(idx_MN, magic_N, shift_N);
        idx_N  = idx_MN - idx_M*grid_N;

        if (OP_A == OP_N)
            idx_M = grid_M - idx_M;

         // use thread stride of 4 to allow use of shared stride of 68
        // which minimizes shared bank conflicts on write.
        uint txc = tid % 16;
        uint tyc = tid / 16;

        uint c       = idx_N*64 + txc*4;
        uint loadC   = tyc*stdC + txc*4;
        uint storC   = fragmentC<OP_A,OP_N,M16N16K16>::get_idx(tid, stdC, tid & 32);
        uint offsetC = idx_B*szCtxHeadStateC + (idx_M*16 + tyc)*szHeadState + idx_H*szState + c;

        __syncthreads();
        for (int j = 0; j < 2; j++)
            fragC[j].store(hShare, storC + j*16, stdC);
        __syncthreads();

        if (N64 || c < szState)
        {
            for (int i = 0; i < 4; i++)
                *(uint2*)&C[offsetC + szHeadState*i*4] = *(uint2*)&hShare[loadC + stdC*i*4];
        }
    }
    else
    {
        uint c       = idx_N*64 + txb*8;
        uint offsetC = idx_B*szCtxHeadStateC + (idx_M*16 + tyb)*szHeadState + idx_H*szState + c;

        if (N64 || c < szState)
        {
            uint4 zero = {0};
            *(uint4*)&C[offsetC + szHeadState*0] = zero;
            *(uint4*)&C[offsetC + szHeadState*8] = zero;
        }
    }
}

template <uint OP_A, bool N64>
__global__ void __launch_bounds__(64) bst_hgemm_8x64x8_xn(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint szCtxHeadStateB, uint szCtxHeadStateC, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint grid_M, uint grid_N, uint magic_N, uint shift_N)
{
    const uint stdA = 8;
    const uint stdB = 80;
    const uint stdC = 68;

    __shared__ ehalf hShare[(stdA + stdB)*16];
    uint2* Lut2s = (uint2*)&hShare[(stdA + stdB)*16];

    uint tid    = threadIdx.x;
    uint idx_MN = blockIdx.x; // compound outer product dims
    uint idx_M  = div64(idx_MN, magic_N, shift_N); // idx_M = idx_MN / grid_N;
    uint idx_N  = idx_MN - idx_M*grid_N;           // idx_N = idx_MN % grid_N;
    uint idx_B  = blockIdx.y; // batch dim
    uint idx_H  = blockIdx.z; // head dim

    // assume lower diagonal and schedule large reductions first
    if (OP_A == OP_N)
        idx_M = grid_M - idx_M;

    // each head can optionally have its own lut
    Lut += idx_H*szLut;
    uint2 lut_head   = Lut[idx_M];
    uint  lut_offset = lut_head.x;
    uint  lut_size   = lut_head.y;

    if (lut_size > 0)
    {
        // prefetch the lut data into shared
        Lut += lut_offset;
        #pragma unroll 1
        for (uint i = tid; i < lut_size; i += 64)
        {
            uint2 entry = Lut[i];
            entry.x *= 8*8;           // 64 entries of A per block
            entry.y *= szHeadState*8; // 8 lines of B per block
            Lut2s[i] = entry;
        }
        __syncthreads();

        uint t32 = tid & 32;
        uint t31 = tid & 31;
        uint txb = tid % 8;
        uint tyb = t31 / 8;

        uint storA = tid*2;
        uint storB = tyb*stdB + txb*8 + t32*20 + 16*stdA;

        uint loadA = fragmentA<OP_A,M8N32K16>::get_idx(tid, stdA);
        uint loadB = fragmentB<OP_N,M8N32K16>::get_idx(tid, stdB, t32 + 16*stdA);

        uint       b = idx_N*64 + txb*8;
        uint offsetA = idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + t31*2;
        uint offsetB = idx_B*szCtxHeadStateB + tyb*szHeadState + idx_H*szState + b;

        fragmentC<OP_A,OP_N,M8N32K16> fragC;

        uint idx_lut   = t32 / 32;
        uint idx_lut2  = 0;
        uint lut_size2 = (lut_size + 1)/2;

        #pragma unroll 1
        do
        {
            uint  a0 = 0;
            uint4 b0 = {0};
            uint4 b4 = {0};
            if (idx_lut < lut_size)
            {
                uint2 entry = Lut2s[idx_lut];
                entry.x += offsetA;
                entry.y += offsetB;
                a0 = *(uint*)&A[entry.x];
                if (b < szState)
                {
                    b0 = *(uint4*)&B[entry.y + 0*szHeadState];
                    b4 = *(uint4*)&B[entry.y + 4*szHeadState];
                }
            }

            __syncthreads();
            *(uint* )&hShare[storA         ] = a0;
            *(uint4*)&hShare[storB + 0*stdB] = b0;
            *(uint4*)&hShare[storB + 4*stdB] = b4;
            __syncthreads();

            fragmentA<OP_A,M8N32K16> fragA;
            fragmentB<OP_N,M8N32K16> fragB;

            fragA.load(hShare, loadA, stdA);
            fragB.load(hShare, loadB, stdB);

            fragC.mma_sync(fragA, fragB);

            idx_lut += 2;

        } while (++idx_lut2 < lut_size2);

        // allow assembler to forget these registers in the main loop
        asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid  )  :);
        asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(idx_MN) :);
        asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(idx_B)  :);
        asm volatile ("mov.u32 %0, %ctaid.z;" : "=r"(idx_H)  :);
        idx_M  = div64(idx_MN, magic_N, shift_N);
        idx_N  = idx_MN - idx_M*grid_N;

        if (OP_A == OP_N)
            idx_M = grid_M - idx_M;

        // use thread stride of 4 to allow use of shared stride of 68
        // which minimizes shared bank conflicts on write.
        uint txc = tid % 16;
        uint tyc = tid / 16;

        uint c       = idx_N*64 + txc*4;
        uint loadC   = tyc*stdC + txc*4;
        uint storC   = fragmentC<OP_A,OP_N,M8N32K16>::get_idx(tid, stdC, tid & 32);
        uint offsetC = idx_B*szCtxHeadStateC + (idx_M*8 + tyc)*szHeadState + idx_H*szState + c;

        __syncthreads();
        fragC.store(hShare, storC, stdC);
        __syncthreads();

        if (N64 || c < szState)
        {
            for (int i = 0; i < 2; i++)
                *(uint2*)&C[offsetC + szHeadState*i*4] = *(uint2*)&hShare[loadC + stdC*i*4];
        }
    }
    else
    {
        uint txc = tid % 8;
        uint tyc = tid / 8;

        uint c       = idx_N*64 + txc*8;
        uint offsetC = idx_B*szCtxHeadStateC + (idx_M*8 + tyc)*szHeadState + idx_H*szState + c;

        if (N64 || c < szState)
        {
            uint4 zero = {0};
            *(uint4*)&C[offsetC + szHeadState*0] = zero;
        }
    }
}


// C = A * B.T
// Dims: M, N, K
// K64: K even mult of 64
// A is dense, B is dense, C is sparse

// 32x32x32 warp tile
template <typename CT, typename CV, bool K64>
__global__ void __launch_bounds__(256) bst_hgemm_64x64x64_nt(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
             CT*              C,
    uint szCtxHeadStateA, uint szCtxHeadStateB, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint loops)
{
    const uint stdA = 64   + 8;
    const uint stdB = 64   + 8;
    const uint stdC = 64*4 + 4;

    __shared__ ehalf hShare[(stdA + stdB)*64];
    float* fShare = (float*)hShare;

    uint tid   = threadIdx.x;
    uint bid   = blockIdx.x; // blockid
    uint idx_B = blockIdx.y; // batch dim
    uint idx_H = blockIdx.z; // head dim

    // each head can optionally have its own lut
    uint2 lut_head = Lut[idx_H*szLut + bid];

    uint tx = tid % 8;
    uint ty = tid / 8;
    uint k  = tx  * 8;

    uint idx_M = lut_head.x;
    uint idx_N = lut_head.y;
    uint offsetA00 = idx_B*szCtxHeadStateA + (idx_M*64 + ty)*szHeadState + idx_H*szState + k;
    uint offsetB00 = idx_B*szCtxHeadStateB + (idx_N*64 + ty)*szHeadState + idx_H*szState + k;
    uint offsetA32 = offsetA00 + szHeadState*32;
    uint offsetB32 = offsetB00 + szHeadState*32;

    uint storA = ty*stdA + k;
    uint storB = ty*stdB + k;
    uint loadA = fragmentA<OP_N,M16N16K16>::get_idx(tid, stdA, (tid & 64)*stdA*32/64 + (tid & 128)*32/128 +  0*stdA);
    uint loadB = fragmentB<OP_T,M16N16K16>::get_idx(tid, stdB, (tid & 32)*stdB*32/32 + (tid & 128)*32/128 + 64*stdA);

    fragmentC<OP_N,OP_T,M16N16K16> fragC[2][2]; // m,n

    uint loop = 0;
    #pragma unroll 1
    do
    {
        asm volatile (".pragma \"nounroll\";"::); // ptxas, don't get clever

        uint4 a00 = {0}, a32 = {0};
        uint4 b00 = {0}, b32 = {0};
        if (K64 || k < szState)
        {
            a00 = *(uint4*)&A[offsetA00];
            a32 = *(uint4*)&A[offsetA32];
            b00 = *(uint4*)&B[offsetB00];
            b32 = *(uint4*)&B[offsetB32];
        }
        offsetA00 += 64;
        offsetA32 += 64;
        offsetB00 += 64;
        offsetB32 += 64;
        if (!K64)
            k += 64;

        __syncthreads();
        *(uint4*)&hShare[storA +  0*stdA +  0*stdA] = a00;
        *(uint4*)&hShare[storA + 32*stdA +  0*stdA] = a32;
        *(uint4*)&hShare[storB +  0*stdB + 64*stdA] = b00;
        *(uint4*)&hShare[storB + 32*stdB + 64*stdA] = b32;
        __syncthreads();

        fragmentA<OP_N,M16N16K16> fragA[2][2]; // m,k
        fragmentB<OP_T,M16N16K16> fragB[2][2]; // n,k
        for (int m = 0; m < 2; m++)
            for (int k = 0; k < 2; k++)
                fragA[m][k].load(hShare, loadA + m*16*stdA + k*16, stdA);

        for (int n = 0; n < 2; n++)
            for (int k = 0; k < 2; k++)
                fragB[n][k].load(hShare, loadB + n*16*stdB + k*16, stdB);

        for (int m = 0; m < 2; m++)
            for (int n = 0; n < 2; n++)
                for (int k = 0; k < 2; k++)
                    fragC[m][n].mma_sync(fragA[m][k], fragB[n][k]);

    } while (++loop < loops);

    asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid)   :);
    asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(bid)   :);
    asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(idx_B) :);
    asm volatile ("mov.u32 %0, %ctaid.z;" : "=r"(idx_H) :);

    uint txc = tid % 16;
    uint tyc = tid / 16;

    uint loadC   = tyc*stdC + txc*4;
    uint storC   = fragmentC<OP_N,OP_T,M16N16K16>::get_idx(tid, stdC, (tid & 224));
    C += idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + bid*64*64 + tid*4;

    for (int m = 0; m < 2; m++)
    {
        __syncthreads();
        for (int n = 0; n < 2; n++)
            fragC[m][n].store(fShare, storC + n*16, stdC);
        __syncthreads();

        for (int i = 0; i < 2; i++)
        {
            float4 sum4 = ew_add(
                    *(float4*)&fShare[loadC + i*64 + 0*128],
                    *(float4*)&fShare[loadC + i*64 + 1*128]
                );
            store((CV*)(C + 64*(i*32 + m*16)), sum4);
        }
    }
}



// C = A * B.T
// Dims: M, N, K
// K64: K even mult of 64
// A is dense, B is dense, C is sparse

template <typename CT, typename CV, bool K64>
__global__ void __launch_bounds__(128) bst_hgemm_32x32x64_nt(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
             CT*              C,
    uint szCtxHeadStateA, uint szCtxHeadStateB, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint loops)
{
    const uint stdA = 72;
    const uint stdB = 72;
    const uint stdC = 132;

    __shared__ ehalf hShare[(stdA + stdB)*32];
    float* fShare = (float*)hShare;

    uint tid   = threadIdx.x;
    uint bid   = blockIdx.x; // blockid
    uint idx_B = blockIdx.y; // batch dim
    uint idx_H = blockIdx.z; // head dim

    // each head can optionally have its own lut
    uint2 lut_head = Lut[idx_H*szLut + bid];

    uint tx = tid % 8;
    uint ty = tid / 8;
    uint k  = tx  * 8;

    uint idx_M = lut_head.x;
    uint idx_N = lut_head.y;
    uint offsetA00 = idx_B*szCtxHeadStateA + (idx_M*32 + ty)*szHeadState + idx_H*szState + k;
    uint offsetB00 = idx_B*szCtxHeadStateB + (idx_N*32 + ty)*szHeadState + idx_H*szState + k;
    uint offsetA16 = offsetA00 + szHeadState*16;
    uint offsetB16 = offsetB00 + szHeadState*16;

    uint storA = ty*stdA + k;
    uint storB = ty*stdB + k;
    uint loadA = fragmentA<OP_N,M16N16K16>::get_idx(tid, stdA, (tid & 96)/2);
    uint loadB = fragmentB<OP_T,M16N16K16>::get_idx(tid, stdB, (tid & 96)/2 + stdA*32);

    fragmentC<OP_N,OP_T,M16N16K16> fragC[2][2];

    uint loop = 0;
    #pragma unroll 1
    do
    {
        asm volatile (".pragma \"nounroll\";"::); // ptxas, don't get clever

        uint4 a00 = {0}, a16 = {0};
        uint4 b00 = {0}, b16 = {0};
        if (K64 || k < szState)
        {
            a00 = *(uint4*)&A[offsetA00];
            a16 = *(uint4*)&A[offsetA16];
            b00 = *(uint4*)&B[offsetB00];
            b16 = *(uint4*)&B[offsetB16];
        }
        offsetA00 += 64;
        offsetA16 += 64;
        offsetB00 += 64;
        offsetB16 += 64;
        if (!K64)
            k += 64;

        __syncthreads();
        *(uint4*)&hShare[storA +  0*stdA +  0*stdA] = a00;
        *(uint4*)&hShare[storA + 16*stdA +  0*stdA] = a16;
        *(uint4*)&hShare[storB +  0*stdB + 32*stdA] = b00;
        *(uint4*)&hShare[storB + 16*stdB + 32*stdA] = b16;
        __syncthreads();

        fragmentA<OP_N,M16N16K16> fragA[2];
        fragmentB<OP_T,M16N16K16> fragB[2];
        for (int i = 0; i < 2; i++)
        {
            fragA[i].load(hShare, loadA + stdA*i*16, stdA);
            fragB[i].load(hShare, loadB + stdB*i*16, stdB);
        }
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                fragC[i][j].mma_sync(fragA[i], fragB[j]);


    } while (++loop < loops);

    asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid)   :);
    asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(bid)   :);
    asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(idx_B) :);
    asm volatile ("mov.u32 %0, %ctaid.z;" : "=r"(idx_H) :);

    tx = tid % 8;
    ty = tid / 8;
    uint loadC   = ty*stdC + tx*4;
    uint storC   = fragmentC<OP_N,OP_T,M16N16K16>::get_idx(tid, stdC, (tid & 96));
    C += idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + bid*32*32 + tid*4;

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

        store((CV*)(C + i*4*128), sum4);
    }
}


// C = A * B.T
// Dims: M, N, K
// K64: K even mult of 64
// dds: A is dense, B is dense, C is sparse

template <typename CT, typename CV, bool K64>
__global__ void __launch_bounds__(64) bst_hgemm_16x16x64_nt(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
             CT*              C,
    uint szCtxHeadStateA, uint szCtxHeadStateB, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint loops)
{
    const uint stdA = 72;
    const uint stdB = 72;
    const uint stdC = 48;

    __shared__ ehalf hShare[(stdA + stdB)*16];
    float* fShare = (float*)hShare;

    uint tid   = threadIdx.x;
    uint bid   = blockIdx.x; // blockid
    uint idx_B = blockIdx.y; // batch dim
    uint idx_H = blockIdx.z; // head dim

    // each head can optionally have its own lut
    uint2 lut_head = Lut[idx_H*szLut + bid];

    uint tx = tid % 8;
    uint ty = tid / 8;
    uint k  = tx  * 8;

    uint idx_M = lut_head.x;
    uint idx_N = lut_head.y;
    uint offsetA0 = idx_B*szCtxHeadStateA + (idx_M*16 + ty)*szHeadState + idx_H*szState + k;
    uint offsetB0 = idx_B*szCtxHeadStateB + (idx_N*16 + ty)*szHeadState + idx_H*szState + k;
    uint offsetA8 = offsetA0 + szHeadState*8;
    uint offsetB8 = offsetB0 + szHeadState*8;

    uint storA = ty*stdA + k;
    uint storB = ty*stdB + k;
    uint loadA = fragmentA<OP_N,M16N16K16>::get_idx(tid, stdA, (tid & 32));
    uint loadB = fragmentB<OP_T,M16N16K16>::get_idx(tid, stdB, (tid & 32) + 16*stdA);

    fragmentC<OP_N,OP_T,M16N16K16> fragC;

    uint loop = 0;
    #pragma unroll 1
    do
    {
        asm volatile (".pragma \"nounroll\";"::); // ptxas, don't get clever

        uint4 a0 = {0}, a8 = {0};
        uint4 b0 = {0}, b8 = {0};
        if (K64 || k < szState)
        {
            a0 = *(uint4*)&A[offsetA0];
            a8 = *(uint4*)&A[offsetA8];
            b0 = *(uint4*)&B[offsetB0];
            b8 = *(uint4*)&B[offsetB8];
        }
        offsetA0 += 64;
        offsetA8 += 64;
        offsetB0 += 64;
        offsetB8 += 64;
        if (!K64)
            k += 64;

        __syncthreads();
        *(uint4*)&hShare[storA + 0*stdA +  0*stdA] = a0;
        *(uint4*)&hShare[storA + 8*stdA +  0*stdA] = a8;
        *(uint4*)&hShare[storB + 0*stdB + 16*stdA] = b0;
        *(uint4*)&hShare[storB + 8*stdB + 16*stdA] = b8;
        __syncthreads();

        fragmentA<OP_N,M16N16K16> fragA;
        fragmentB<OP_T,M16N16K16> fragB;
        #pragma unroll
        for (uint j = 0; j < 2; j++)
        {
            fragA.load(hShare, loadA + j*16, stdA);
            fragB.load(hShare, loadB + j*16, stdB);

            fragC.mma_sync(fragA, fragB);
        }

    } while (++loop < loops);

    asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid)   :);
    asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(bid)   :);
    asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(idx_B) :);
    asm volatile ("mov.u32 %0, %ctaid.z;" : "=r"(idx_H) :);

    tx = tid % 4;
    ty = tid / 4;
    uint loadC   = ty*stdC + tx*4;
    uint storC   = fragmentC<OP_N,OP_T,M16N16K16>::get_idx(tid, stdC, (tid & 32)/2);
    C += idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + bid*16*16 + tid*4;

    __syncthreads();
    fragC.store(fShare, storC, stdC);
    __syncthreads();

    float4 sum4 = ew_add(
        *(float4*)&fShare[loadC +  0],
        *(float4*)&fShare[loadC + 16]);

        store((CV*)C, sum4);
}

template <typename CT, typename CV, bool K64>
__global__ void __launch_bounds__(32) bst_hgemm_8x8x64_nt(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
             CT*              C,
    uint szCtxHeadStateA, uint szCtxHeadStateB, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint loops)
{
    const uint stdAB = 72;
    const uint stdC  = 8;

    __shared__ ehalf hShare[stdAB*8*2];
    float* fShare = (float*)hShare;

    uint tid   = threadIdx.x;
    uint bid   = blockIdx.x; // blockid
    uint idx_B = blockIdx.y; // batch dim
    uint idx_H = blockIdx.z; // head dim

    // each head can optionally have its own lut
    uint2 lut_head = Lut[idx_H*szLut + bid];

    uint tx = tid % 8;
    uint ty = tid / 8;
    uint k  = tx  * 8;

    uint idx_M = lut_head.x;
    uint idx_N = lut_head.y;
    uint offsetA0 = idx_B*szCtxHeadStateA + (idx_M*8 + ty)*szHeadState + idx_H*szState + k;
    uint offsetB0 = idx_B*szCtxHeadStateB + (idx_N*8 + ty)*szHeadState + idx_H*szState + k;
    uint offsetA4 = offsetA0 + szHeadState*4;
    uint offsetB4 = offsetB0 + szHeadState*4;

    uint storAB = ty*stdAB + k;
    uint loadA = fragmentA<OP_N,M8N8K16>::get_idx(tid, stdAB, 0*stdAB);
    uint loadB = fragmentB<OP_T,M8N8K16>::get_idx(tid, stdAB, 8*stdAB);

    fragmentC<OP_N,OP_T,M8N8K16> fragC;

    uint loop = 0;
    #pragma unroll 1
    do
    {
        uint4 a0 = {0}, a4 = {0};
        uint4 b0 = {0}, b4 = {0};
        if (K64 || k < szState)
        {
            a0 = *(uint4*)&A[offsetA0];
            a4 = *(uint4*)&A[offsetA4];
            b0 = *(uint4*)&B[offsetB0];
            b4 = *(uint4*)&B[offsetB4];
        }
        offsetA0 += 64;
        offsetA4 += 64;
        offsetB0 += 64;
        offsetB4 += 64;
        if (!K64)
            k += 64;

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

    asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid)   :);
    asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(bid)   :);
    asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(idx_B) :);
    asm volatile ("mov.u32 %0, %ctaid.z;" : "=r"(idx_H) :);

    uint storC = fragmentC<OP_N,OP_T,M8N8K16>::get_idx(tid, stdC);

    fragC.store(fShare, storC, stdC);

    C += idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + bid*8*8 + tid*2;

    store((CV*)C, *(float2*)&fShare[tid*2]);
}


#else // __CUDA_ARCH__ >= 700

template <uint OP_A, bool N64>
__global__ void __launch_bounds__(256) bst_hgemm_64x64x64_xn(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint szCtxHeadStateB, uint szCtxHeadStateC, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint grid_M, uint grid_N, uint magic_N, uint shift_N)
{
    *C = 0;
}
template <uint OP_A, bool N64>
__global__ void __launch_bounds__(128) bst_hgemm_32x64x32_xn(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint szCtxHeadStateB, uint szCtxHeadStateC, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint grid_M, uint grid_N, uint magic_N, uint shift_N)
{
    *C = 0;
}
template <uint OP_A, bool N64>
__global__ void __launch_bounds__(64) bst_hgemm_16x64x16_xn(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint szCtxHeadStateB, uint szCtxHeadStateC, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint grid_M, uint grid_N, uint magic_N, uint shift_N)
{
    *C = 0;
}
template <uint OP_A, bool N64>
__global__ void __launch_bounds__(64) bst_hgemm_8x64x8_xn(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint szCtxHeadStateB, uint szCtxHeadStateC, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint grid_M, uint grid_N, uint magic_N, uint shift_N)
{
    *C = 0;
}

template <typename CT, typename CV, bool K64>
__global__ void __launch_bounds__(256) bst_hgemm_64x64x64_nt(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
             CT*              C,
    uint szCtxHeadStateA, uint szCtxHeadStateB, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint loops)
{
    *C = 0;
}
template <typename CT, typename CV, bool K64>
__global__ void __launch_bounds__(128) bst_hgemm_32x32x64_nt(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
             CT*              C,
    uint szCtxHeadStateA, uint szCtxHeadStateB, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint loops)
{
    *C = 0;
}
template <typename CT, typename CV, bool K64>
__global__ void __launch_bounds__(64) bst_hgemm_16x16x64_nt(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
             CT*              C,
    uint szCtxHeadStateA, uint szCtxHeadStateB, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint loops)
{
    *C = 0;
}
template <typename CT, typename CV, bool K64>
__global__ void __launch_bounds__(32) bst_hgemm_8x8x64_nt(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
             CT*              C,
    uint szCtxHeadStateA, uint szCtxHeadStateB, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint loops)
{
    *C = 0;
}

#endif // __CUDA_ARCH__ >= 700



bool bst_hgemm_xn(CUstream stream,
    const uint2* lut,
    const ehalf* a,
    const ehalf* b,
          ehalf* c,
    uint block_size, uint blocks,
    uint batch_dim, uint ctx_blks_b, uint ctx_blks_c, uint head_dim, uint state_dim,
    uint lut_heads, uint lut_dim, uint op, uint magic, uint shift, uint max_lut)
{
    uint szState         = state_dim;
    uint szHeadState     = head_dim * szState;
    uint szCtxHeadStateB = ctx_blks_b * block_size * szHeadState;
    uint szCtxHeadStateC = ctx_blks_c * block_size * szHeadState;

    uint szBlocksBlk     = blocks * block_size * block_size;
    uint szHeadBlocksBlk = head_dim * szBlocksBlk;

    // if just one lut head, broadcast block-sparsity to all heads
    uint szLut = lut_heads > 1 ? lut_dim : 0;

    // compound gridDim.x with m and n coords
    uint gridN  = CEIL_DIV(state_dim, 64);
    uint gridM  = ctx_blks_c - 1;
    uint gridX  = ctx_blks_c * gridN;
    uint shared = ((max_lut+1)/2)*2*8; // round up to nearest even, 8 bytes per entry

    dim3 grid(gridX, batch_dim, head_dim);
    if (op == NN_OP) // NN
    {
        if (block_size == 8)
              bst_hgemm_8x64x8_xn<OP_N,false><<<grid, 64,shared,stream>>>(lut, a, b, c, szCtxHeadStateB, szCtxHeadStateC, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic, shift);
        else if (block_size == 16)
            bst_hgemm_16x64x16_xn<OP_N,false><<<grid, 64,shared,stream>>>(lut, a, b, c, szCtxHeadStateB, szCtxHeadStateC, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic, shift);
        else if (block_size == 32)
        {
            shared = shared > 256 ? shared - 256 : 0;
            bst_hgemm_32x64x32_xn<OP_N,false><<<grid,128,shared,stream>>>(lut, a, b, c, szCtxHeadStateB, szCtxHeadStateC, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic, shift);
        }
        else if (block_size == 64)
            bst_hgemm_64x64x64_xn<OP_N,false><<<grid,256,shared,stream>>>(lut, a, b, c, szCtxHeadStateB, szCtxHeadStateC, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic, shift);
    }
    else // TN
    {
        if (block_size == 8)
              bst_hgemm_8x64x8_xn<OP_T,false><<<grid, 64,shared,stream>>>(lut, a, b, c, szCtxHeadStateB, szCtxHeadStateC, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic, shift);
        else if (block_size == 16)
            bst_hgemm_16x64x16_xn<OP_T,false><<<grid, 64,shared,stream>>>(lut, a, b, c, szCtxHeadStateB, szCtxHeadStateC, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic, shift);
        else if (block_size == 32)
        {
            shared = shared > 256 ? shared - 256 : 0;
            bst_hgemm_32x64x32_xn<OP_T,false><<<grid,128,shared,stream>>>(lut, a, b, c, szCtxHeadStateB, szCtxHeadStateC, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic, shift);
        }
        else if (block_size == 64)
            bst_hgemm_64x64x64_xn<OP_T,false><<<grid,256,shared,stream>>>(lut, a, b, c, szCtxHeadStateB, szCtxHeadStateC, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic, shift);
    }
    return true;
}

template <typename CT, typename CV2, typename CV4>
bool bst_hgemm_nt(CUstream stream,
    const uint2* lut,
    const ehalf* a,
    const ehalf* b,
             CT* c,
    uint block_size, uint blocks,
    uint batch_dim, uint ctx_blks_a, uint ctx_blks_b, uint head_dim, uint state_dim,
    uint lut_heads, uint lut_dim)
{
    uint szState         = state_dim;
    uint szHeadState     = head_dim * szState;
    uint szCtxHeadStateA = ctx_blks_a * block_size * szHeadState;
    uint szCtxHeadStateB = ctx_blks_b * block_size * szHeadState;

    uint szBlocksBlk     = blocks * block_size * block_size;
    uint szHeadBlocksBlk = head_dim * szBlocksBlk;

    // if just one lut head, broadcast block-sparsity to all heads
    uint szLut = lut_heads > 1 ? lut_dim : 0;

    uint loops = CEIL_DIV(state_dim, 64);
    bool k64   = (state_dim & 63) == 0;
    dim3 grid(blocks, batch_dim, head_dim);

    if (block_size == 8)
    {
        if (k64)
            bst_hgemm_8x8x64_nt<CT,CV2, true><<<grid, 32,0,stream>>>(lut, a, b, c, szCtxHeadStateA, szCtxHeadStateB, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, loops);
        else
            bst_hgemm_8x8x64_nt<CT,CV2,false><<<grid, 32,0,stream>>>(lut, a, b, c, szCtxHeadStateA, szCtxHeadStateB, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, loops);
    }
    else if (block_size == 16)
    {
        if (k64)
            bst_hgemm_16x16x64_nt<CT,CV4, true><<<grid, 64,0,stream>>>(lut, a, b, c, szCtxHeadStateA, szCtxHeadStateB, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, loops);
        else
            bst_hgemm_16x16x64_nt<CT,CV4,false><<<grid, 64,0,stream>>>(lut, a, b, c, szCtxHeadStateA, szCtxHeadStateB, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, loops);
    }
    else if (block_size == 32)
        bst_hgemm_32x32x64_nt<CT,CV4,false><<<grid,128,0,stream>>>(lut, a, b, c, szCtxHeadStateA, szCtxHeadStateB, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, loops);

    else if (block_size == 64)
        bst_hgemm_64x64x64_nt<CT,CV4,false><<<grid,256,0,stream>>>(lut, a, b, c, szCtxHeadStateA, szCtxHeadStateB, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, loops);

    // cudaError_t error = cudaGetLastError();
    // printf("%s\n%s\n", cudaGetErrorName(error), cudaGetErrorString(error));

    return true;
}
template bool bst_hgemm_nt<ehalf,ehalf2,ehalf4>(CUstream stream, const uint2* lut, const ehalf* a, const ehalf* b, ehalf* c, uint block_size, uint blocks, uint batch_dim, uint ctx_blks_a, uint ctx_blks_b, uint head_dim, uint state_dim, uint lut_heads, uint lut_dim);
template bool bst_hgemm_nt<bhalf,bhalf2,bhalf4>(CUstream stream, const uint2* lut, const ehalf* a, const ehalf* b, bhalf* c, uint block_size, uint blocks, uint batch_dim, uint ctx_blks_a, uint ctx_blks_b, uint head_dim, uint state_dim, uint lut_heads, uint lut_dim);


#endif // GOOGLE_CUDA