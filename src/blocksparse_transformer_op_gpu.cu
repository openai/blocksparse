#if GOOGLE_CUDA

#include "ew_op_gpu.h"
#include "gpu_hmma.h"
#include <stdio.h>

#if __CUDA_ARCH__ >= 700


// C = A * B  or  A.T * B
// Dims: M, N, K
// N64: N even mult of 64
// A is sparse, B is dense, C is dense

template <uint OP_A, bool N64>
__global__ void __launch_bounds__(256) hgemm_blocksparse_64x64x64_xn_sdd(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint szCtxHeadState,  uint szHeadState, uint szState,
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

        uint loadA = fragmentA<OP_A>::get_idx(tid, stdA, (tid & 192)*(OP_A == OP_N ? 1 : stdA)*16/64 + (tid & 32)*(OP_A == OP_N ? stdA : 1));
        uint loadB = fragmentB<OP_N>::get_idx(tid, stdB, (tid & 192)*stdB*16/64 + stdA*64);

        uint b = idx_N*64 + tx*8;
        uint offsetA = idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + tid*8;
        uint offsetB = idx_B*szCtxHeadState + ty*szHeadState + idx_H*szState + b;

        bool inB = N64 || b < szState;

        fragmentC<OP_A,OP_N> fragC[2][4];

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

            fragmentA<OP_A> fragA[2];
            fragmentB<OP_N> fragB[4];
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
        uint storC   = fragmentC<OP_A,OP_N>::get_idx(tid, stdC, (tid & 224)*2);
        uint offsetC = idx_B*szCtxHeadState + (idx_M*64 + tyc)*szHeadState + idx_H*szState + c;

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
        uint offsetC = idx_B*szCtxHeadState + (idx_M*64 + ty)*szHeadState + idx_H*szState + c;

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
__global__ void __launch_bounds__(128) hgemm_blocksparse_32x64x32_xn_sdd(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint szCtxHeadState,  uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint grid_M, uint grid_N, uint magic_N, uint shift_N)
{
    const uint stdA = 48;
    const uint stdB = 80;
    const uint stdC = 132;

    __shared__ ehalf hShare[(stdA + stdB)*32];
    float* fShare = (float*)hShare;
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

    uint txb = tid % 8;
    uint tyb = tid / 8;

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

        uint txa = tid % 4;
        uint tya = tid / 4;

        uint storA = tya*stdA + txa*8;
        uint storB = tyb*stdB + txb*8 + stdA*32;

        uint loadA = fragmentA<OP_A>::get_idx(tid, stdA, (tid & 64)*(OP_A == OP_N ? 1 : stdA)*16/64);
        uint loadB = fragmentB<OP_N>::get_idx(tid, stdB, (tid & 64)*stdB*16/64 + (tid & 32) + stdA*32);

        uint b = idx_N*64 + txb*8;
        uint offsetA = idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + tid*8;
        uint offsetB = idx_B*szCtxHeadState + tyb*szHeadState + idx_H*szState + b;

        bool inB = N64 || b < szState;

        fragmentC<OP_A,OP_N> fragC[2][2];

        int idx_lut = 0;
        #pragma unroll 1
        do
        {
            asm volatile (".pragma \"nounroll\";"::); // ptxas, don't get clever

            uint2 entry = Lut2s[idx_lut];
            uint4 b00 = {0};
            uint4 b16 = {0};

            entry.x += offsetA;
            entry.y += offsetB;

            uint4 a00 = *(uint4*)&A[entry.x];
            if (inB)
            {
                b00 = *(uint4*)&B[entry.y +  0*szHeadState];
                b16 = *(uint4*)&B[entry.y + 16*szHeadState];
            }
            __syncthreads();
            *(uint4*)&hShare[storA] = a00;
            *(uint4*)&hShare[storB +  0*stdB] = b00;
            *(uint4*)&hShare[storB + 16*stdB] = b16;
            __syncthreads();

            fragmentA<OP_A> fragA[2];
            fragmentB<OP_N> fragB[2];
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
        uint storC   = fragmentC<OP_A,OP_N>::get_idx(tid, stdC, tid & 96);
        uint offsetC = idx_B*szCtxHeadState + (idx_M*32 + tyc)*szHeadState + idx_H*szState + c;

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
        uint c       = idx_N*64 + txb*8;
        uint offsetC = idx_B*szCtxHeadState + (idx_M*32 + tyb)*szHeadState + idx_H*szState + c;

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
__global__ void __launch_bounds__(64) hgemm_blocksparse_16x64x16_xn_sdd(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint szCtxHeadState,  uint szHeadState, uint szState,
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

        uint loadA = fragmentA<OP_A>::get_idx(tid, stdA);
        uint loadB = fragmentB<OP_N>::get_idx(tid, stdB, 16*stdA + (tid & 32));

        uint       b = idx_N*64 + txb*8;
        bool     inB = N64 || b < szState;
        uint offsetA = idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + tid*4;
        uint offsetB = idx_B*szCtxHeadState + tyb*szHeadState + idx_H*szState + b;

        fragmentC<OP_A,OP_N> fragC[2];

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

            fragmentA<OP_A> fragA;
            fragmentB<OP_N> fragB;

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
        uint storC   = fragmentC<OP_A,OP_N>::get_idx(tid, stdC, tid & 32);
        uint offsetC = idx_B*szCtxHeadState + (idx_M*16 + tyc)*szHeadState + idx_H*szState + c;

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
        uint offsetC = idx_B*szCtxHeadState + (idx_M*16 + tyb)*szHeadState + idx_H*szState + c;

        if (N64 || c < szState)
        {
            uint4 zero = {0};
            *(uint4*)&C[offsetC + szHeadState*0] = zero;
            *(uint4*)&C[offsetC + szHeadState*8] = zero;
        }
    }
}


// C = A * B.T
// Dims: M, N, K
// K64: K even mult of 64
// A is dense, B is dense, C is sparse

// 32x32x32 warp tile
template <bool K64>
__global__ void __launch_bounds__(256) hgemm_blocksparse_64x64x64_nt_dds(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint szCtxHeadState,  uint szHeadState, uint szState,
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
    uint offsetA00 = idx_B*szCtxHeadState + (idx_M*64 + ty)*szHeadState + idx_H*szState + k;
    uint offsetB00 = idx_B*szCtxHeadState + (idx_N*64 + ty)*szHeadState + idx_H*szState + k;
    uint offsetA32 = offsetA00 + szHeadState*32;
    uint offsetB32 = offsetB00 + szHeadState*32;

    uint storA = ty*stdA + k;
    uint storB = ty*stdB + k;
    uint loadA = fragmentA<OP_N>::get_idx(tid, stdA, (tid & 64)*stdA*32/64 + (tid & 128)*32/128 +  0*stdA);
    uint loadB = fragmentB<OP_T>::get_idx(tid, stdB, (tid & 32)*stdB*32/32 + (tid & 128)*32/128 + 64*stdA);

    fragmentC<OP_N,OP_T> fragC[2][2]; // m,n

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

        fragmentA<OP_N> fragA[2][2]; // m,k
        fragmentB<OP_T> fragB[2][2]; // n,k
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
    uint storC   = fragmentC<OP_N,OP_T>::get_idx(tid, stdC, (tid & 224));
    uint offsetC = idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + bid*64*64 + tid*4;

    for (int m = 0; m < 2; m++)
    {
        __syncthreads();
        for (int n = 0; n < 2; n++)
            fragC[m][n].store(fShare, storC + n*16, stdC);
        __syncthreads();

        for (int i = 0; i < 2; i++)
            *(uint2*)&C[offsetC + 64*(i*32 + m*16)] = to_half4(
                ew_add(
                    *(float4*)&fShare[loadC + i*64 + 0*128],
                    *(float4*)&fShare[loadC + i*64 + 1*128]
                )
            );
    }
}

// 32x64x16 warp tile
// this is slower than 32x32x32 warp tile for small number of loops
// template <bool K64>
// __global__ void __launch_bounds__(256) hgemm_blocksparse_64x64x64_nt_dds2(
//     const uint2* __restrict__ Lut,
//     const ehalf* __restrict__ A,
//     const ehalf* __restrict__ B,
//           ehalf*              C,
//     uint szCtxHeadState,  uint szHeadState, uint szState,
//     uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
//     uint loops)
// {
//     const uint stdA = 64+8;
//     const uint stdB = 64+8;
//     const uint stdC = 512+4;

//     __shared__ float fShare[stdC*16];
//     ehalf* hShare = (ehalf*)fShare;

//     uint tid   = threadIdx.x;
//     uint bid   = blockIdx.x; // blockid
//     uint idx_B = blockIdx.y; // batch dim
//     uint idx_H = blockIdx.z; // head dim

//     // each head can optionally have its own lut
//     uint2 lut_head = Lut[idx_H*szLut + bid];

//     uint tx = tid % 8;
//     uint ty = tid / 8;
//     uint k  = tx  * 8;

//     uint idx_M = lut_head.x;
//     uint idx_N = lut_head.y;
//     uint offsetA00 = idx_B*szCtxHeadState + (idx_M*64 + ty)*szHeadState + idx_H*szState + k;
//     uint offsetB00 = idx_B*szCtxHeadState + (idx_N*64 + ty)*szHeadState + idx_H*szState + k;
//     uint offsetA32 = offsetA00 + szHeadState*32;
//     uint offsetB32 = offsetB00 + szHeadState*32;

//     uint storA = ty*stdA + k;
//     uint storB = ty*stdB + k;
//     uint loadA = fragmentA<OP_N>::get_idx(tid, stdA, (tid & 192)*16/64 + stdA*(tid & 32));
//     uint loadB = fragmentB<OP_T>::get_idx(tid, stdB, (tid & 192)*16/64 + stdA*64);

//     fragmentC<OP_N,OP_T> fragC[2][4];

//     uint loop = 0;
//     #pragma unroll 1
//     do
//     {
//         asm volatile (".pragma \"nounroll\";"::); // ptxas, don't get clever

//         uint4 a00 = {0}, a32 = {0};
//         uint4 b00 = {0}, b32 = {0};
//         if (K64 || k < szState)
//         {
//             a00 = *(uint4*)&A[offsetA00];
//             a32 = *(uint4*)&A[offsetA32];
//             b00 = *(uint4*)&B[offsetB00];
//             b32 = *(uint4*)&B[offsetB32];
//         }
//         offsetA00 += 64;
//         offsetA32 += 64;
//         offsetB00 += 64;
//         offsetB32 += 64;
//         if (!K64)
//             k += 64;

//         __syncthreads();
//         *(uint4*)&hShare[storA +  0*stdA +  0*stdA] = a00;
//         *(uint4*)&hShare[storA + 32*stdA +  0*stdA] = a32;
//         *(uint4*)&hShare[storB +  0*stdB + 64*stdA] = b00;
//         *(uint4*)&hShare[storB + 32*stdB + 64*stdA] = b32;
//         __syncthreads();

//         fragmentA<OP_N> fragA[2];
//         fragmentB<OP_T> fragB[4];
//         for (int i = 0; i < 2; i++)
//             fragA[i].load(hShare, loadA + stdA*i*16, stdA);

//         for (int i = 0; i < 4; i++)
//             fragB[i].load(hShare, loadB + stdB*i*16, stdB);

//         for (int i = 0; i < 2; i++)
//             for (int j = 0; j < 4; j++)
//                 fragC[i][j].mma_sync(fragA[i], fragB[j]);

//     } while (++loop < loops);

//     asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid)   :);
//     asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(bid)   :);
//     asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(idx_B) :);
//     asm volatile ("mov.u32 %0, %ctaid.z;" : "=r"(idx_H) :);

//     uint txc = tid % 16;
//     uint tyc = tid / 16;

//     uint loadC   = tyc*stdC + txc*4;
//     uint storC   = fragmentC<OP_N,OP_T>::get_idx(tid, stdC, (tid & 224)*2);
//     uint offsetC = idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + bid*64*64 + tid*4;

//     for (int i = 0; i < 2; i++)
//     {
//         __syncthreads();
//         for (int j = 0; j < 4; j++)
//             fragC[i][j].store(fShare, storC + j*16, stdC);
//         __syncthreads();

//         for (int j = 0; j < 2; j++)
//             *(uint2*)&C[offsetC + 64*(j*32 + i*16)] = to_half4(
//                 ew_add(
//                     ew_add(
//                         *(float4*)&fShare[loadC + j*64 + 0*128],
//                         *(float4*)&fShare[loadC + j*64 + 1*128]),
//                     ew_add(
//                         *(float4*)&fShare[loadC + j*64 + 2*128],
//                         *(float4*)&fShare[loadC + j*64 + 3*128])
//                 )
//             );
//     }
// }

// C = A * B.T
// Dims: M, N, K
// K64: K even mult of 64
// A is dense, B is dense, C is sparse

template <bool K64>
__global__ void __launch_bounds__(128) hgemm_blocksparse_32x32x64_nt_dds(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint szCtxHeadState,  uint szHeadState, uint szState,
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
    uint offsetA00 = idx_B*szCtxHeadState + (idx_M*32 + ty)*szHeadState + idx_H*szState + k;
    uint offsetB00 = idx_B*szCtxHeadState + (idx_N*32 + ty)*szHeadState + idx_H*szState + k;
    uint offsetA16 = offsetA00 + szHeadState*16;
    uint offsetB16 = offsetB00 + szHeadState*16;

    uint storA = ty*stdA + k;
    uint storB = ty*stdB + k;
    uint loadA = fragmentA<OP_N>::get_idx(tid, stdA, (tid & 96)/2);
    uint loadB = fragmentB<OP_T>::get_idx(tid, stdB, (tid & 96)/2 + stdA*32);

    fragmentC<OP_N,OP_T> fragC[2][2];

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

        fragmentA<OP_N> fragA[2];
        fragmentB<OP_T> fragB[2];
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
    uint storC   = fragmentC<OP_N,OP_T>::get_idx(tid, stdC, (tid & 96));
    uint offsetC = idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + bid*32*32 + tid*4;

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

        *(uint2*)&C[offsetC + i*4*128] = to_half4(sum4);
    }
}


// C = A * B.T
// Dims: M, N, K
// K64: K even mult of 64
// dds: A is dense, B is dense, C is sparse

template <bool K64>
__global__ void __launch_bounds__(64) hgemm_blocksparse_16x16x64_nt_dds(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint szCtxHeadState,  uint szHeadState, uint szState,
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
    uint offsetA0 = idx_B*szCtxHeadState + (idx_M*16 + ty)*szHeadState + idx_H*szState + k;
    uint offsetB0 = idx_B*szCtxHeadState + (idx_N*16 + ty)*szHeadState + idx_H*szState + k;
    uint offsetA8 = offsetA0 + szHeadState*8;
    uint offsetB8 = offsetB0 + szHeadState*8;

    uint storA = ty*stdA + k;
    uint storB = ty*stdB + k;
    uint loadA = fragmentA<OP_N>::get_idx(tid, stdA, (tid & 32));
    uint loadB = fragmentB<OP_T>::get_idx(tid, stdB, (tid & 32) + 16*stdA);

    fragmentC<OP_N,OP_T> fragC;

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

        fragmentA<OP_N> fragA;
        fragmentB<OP_T> fragB;
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
    uint storC   = fragmentC<OP_N,OP_T>::get_idx(tid, stdC, (tid & 32)/2);
    uint offsetC = idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + bid*16*16 + tid*4;

    __syncthreads();
    fragC.store(fShare, storC, stdC);
    __syncthreads();

    float4 sum4 = ew_add(
        *(float4*)&fShare[loadC +  0],
        *(float4*)&fShare[loadC + 16]);

    *(uint2*)&C[offsetC] = to_half4(sum4);
}

#else // __CUDA_ARCH__ >= 700

template <uint OP_A, bool N64>
__global__ void __launch_bounds__(256) hgemm_blocksparse_64x64x64_xn_sdd(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint szCtxHeadState,  uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint grid_M, uint grid_N, uint magic_N, uint shift_N)
{
    *C = 0;
}
template <uint OP_A, bool N64>
__global__ void __launch_bounds__(128) hgemm_blocksparse_32x64x32_xn_sdd(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint szCtxHeadState,  uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint grid_M, uint grid_N, uint magic_N, uint shift_N)
{
    *C = 0;
}
template <uint OP_A, bool N64>
__global__ void __launch_bounds__(64) hgemm_blocksparse_16x64x16_xn_sdd(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint szCtxHeadState,  uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint grid_M, uint grid_N, uint magic_N, uint shift_N)
{
    *C = 0;
}
template <bool K64>
__global__ void __launch_bounds__(256) hgemm_blocksparse_64x64x64_nt_dds(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint szCtxHeadState,  uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint loops)
{
    *C = 0;
}
template <bool K64>
__global__ void __launch_bounds__(128) hgemm_blocksparse_32x32x64_nt_dds(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint szCtxHeadState,  uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint loops)
{
    *C = 0;
}
template <bool K64>
__global__ void __launch_bounds__(64) hgemm_blocksparse_16x16x64_nt_dds(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
          ehalf*              C,
    uint szCtxHeadState,  uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint loops)
{
    *C = 0;
}

#endif // __CUDA_ARCH__ >= 700

bool blocksparse_transformer_xn(CUstream stream,
    const uint2* lut,
    const ehalf* a,
    const ehalf* b,
          ehalf* c,
    uint block_size, uint blocks,
    uint batch_dim, uint ctx_blks, uint head_dim, uint state_dim,
    uint lut_heads, uint lut_dim, uint op, uint magic, uint shift, uint max_lut)
{
    uint szState         = state_dim;
    uint szHeadState     = head_dim * szState;
    uint szCtxHeadState  = ctx_blks * block_size * szHeadState;

    uint szBlocksBlk     = blocks * block_size * block_size;
    uint szHeadBlocksBlk = head_dim * szBlocksBlk;

    // if just one lut head, broadcast block-sparsity to all heads
    uint szLut = lut_heads > 1 ? lut_dim : 0;

    // compound gridDim.x with m and n coords
    uint gridN  = CEIL_DIV(state_dim, 64);
    uint gridM  = ctx_blks - 1;
    uint gridX  = ctx_blks * gridN;
    uint shared = max_lut*8;

    dim3 grid(gridX, batch_dim, head_dim);
    if (op == 1) // NN
    {
        if (block_size == 16)
            hgemm_blocksparse_16x64x16_xn_sdd<OP_N,false><<<grid, 64,shared,stream>>>(lut, a, b, c, szCtxHeadState, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic, shift);
        else if (block_size == 32)
            hgemm_blocksparse_32x64x32_xn_sdd<OP_N,false><<<grid,128,shared,stream>>>(lut, a, b, c, szCtxHeadState, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic, shift);
        else
            hgemm_blocksparse_64x64x64_xn_sdd<OP_N,false><<<grid,256,shared,stream>>>(lut, a, b, c, szCtxHeadState, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic, shift);
    }
    else // TN
    {
        if (block_size == 16)
            hgemm_blocksparse_16x64x16_xn_sdd<OP_T,false><<<grid, 64,shared,stream>>>(lut, a, b, c, szCtxHeadState, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic, shift);
        else if (block_size == 32)
            hgemm_blocksparse_32x64x32_xn_sdd<OP_T,false><<<grid,128,shared,stream>>>(lut, a, b, c, szCtxHeadState, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic, shift);
        else
            hgemm_blocksparse_64x64x64_xn_sdd<OP_T,false><<<grid,256,shared,stream>>>(lut, a, b, c, szCtxHeadState, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic, shift);
    }
    return true;
}

bool blocksparse_transformer_nt(CUstream stream,
    const uint2* lut,
    const ehalf* a,
    const ehalf* b,
          ehalf* c,
    uint block_size, uint blocks,
    uint batch_dim, uint ctx_blks, uint head_dim, uint state_dim,
    uint lut_heads, uint lut_dim)
{
    uint szState         = state_dim;
    uint szHeadState     = head_dim * szState;
    uint szCtxHeadState  = ctx_blks * block_size * szHeadState;

    uint szBlocksBlk     = blocks * block_size * block_size;
    uint szHeadBlocksBlk = head_dim * szBlocksBlk;

    // if just one lut head, broadcast block-sparsity to all heads
    uint szLut = lut_heads > 1 ? lut_dim : 0;

    uint loops = CEIL_DIV(state_dim, 64);

    dim3 grid(blocks, batch_dim, head_dim);
    if (block_size == 16)
        hgemm_blocksparse_16x16x64_nt_dds<false><<<grid, 64,0,stream>>>(lut, a, b, c, szCtxHeadState, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, loops);
    else if (block_size == 32)
        hgemm_blocksparse_32x32x64_nt_dds<false><<<grid,128,0,stream>>>(lut, a, b, c, szCtxHeadState, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, loops);
    else
        hgemm_blocksparse_64x64x64_nt_dds<false><<<grid,256,0,stream>>>(lut, a, b, c, szCtxHeadState, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, loops);

    // cudaError_t error = cudaGetLastError();
    // printf("%s\n%s\n", cudaGetErrorName(error), cudaGetErrorString(error));

    return true;
}


template <uint U, uint BSIZE, typename MASKT>
__global__ void blocksparse_masked_softmax(
    const uint2* __restrict__ Lut,
    const MASKT* __restrict__ Mask,
    const ehalf* __restrict__ X,
          ehalf*              Y,
    uint blocks, uint szLut, uint szMask, uint szHead, uint szBatch, float scale, uint shfl_init, uint use_mask)
{
    __shared__ float Max[32];
    __shared__ float Sum[32];
    uint2* Lut2s = (uint2*)&Sum[32];

    uint tid   = threadIdx.x;
    uint idx_Q = blockIdx.x / BSIZE; // Q dim
    uint idx_q = blockIdx.x % BSIZE; // Q dim
    uint idx_B = blockIdx.y; // batch dim
    uint idx_H = blockIdx.z; // head dim

    // each head can optionally have its own lut
    Lut  += idx_H * szLut;
    Mask += idx_H * szMask + idx_q * blocks;
    uint2 lut_head = Lut[idx_Q];

    if (tid < 32)
    {
        // Allows non-power of 2 threads to work
        Max[tid] = -FLT_MAX;
        Sum[tid] = 0.0f;
    }

    // prefetch the lut data into shared
    uint lut_offset = lut_head.x;
    uint lut_size   = lut_head.y;
    Lut += lut_offset;
    #pragma unroll 1
    for (uint i = tid; i < lut_size; i += blockDim.x)
    {
        uint2 entry = Lut[i];
        entry.y     = use_mask ? (uint)__ldg(Mask + entry.x) : 0xffffffff;
        entry.x    *= BSIZE*BSIZE;
        Lut2s[i]    = entry;

        //printf("%3d %3d %3d %08x\n", idx_Q, idx_q, i, entry.y);
    }
    __syncthreads();

    uint lut_idx = (tid & (1024-BSIZE))*U/BSIZE;
    uint tidx = tid % BSIZE;
    uint mask_bit = 1 << tidx;
    uint offset = idx_B*szBatch + idx_H*szHead + idx_q*BSIZE + tidx;

    float xval[U];
    #pragma unroll
    for (int i = 0; i < U; i++)
    {
        uint2 entry = Lut2s[lut_idx + i];
        uint offsetX = offset + entry.x;
        bool in = lut_idx + i < lut_size;
        float val = load(X + offsetX, 0, in);

        xval[i] = in && (entry.y & mask_bit) != 0 ? val : -FLT_MAX;
    }

    // reduce within thread
    float Xmax[U];
    for (int i = 0; i < U; i++)
        Xmax[i] = xval[i];

    for (int j = U >> 1; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            Xmax[i] = fmaxf(Xmax[i], Xmax[i+j]);
    float xmax = Xmax[0];

    // reduce within warp
    for (int i = 16; i > 0; i >>= 1)
        xmax = fmaxf(xmax, shfl_xor(xmax, i));

    if (blockDim.x > 32)
    {
        // first thread of each warp store to shared
        if ((tid & 31) == 0)
            Max[tid/32] = xmax;
        __syncthreads();
        if (tid < 32)
        {
            // first warp loads all prior reductions
            xmax = Max[tid];
            // reduce within this last warp
            #pragma unroll 1
            for (uint i = shfl_init; i > 0; i >>= 1)
                xmax = fmaxf(xmax, shfl_xor(xmax, i));
            // final reduction to shared
            Max[tid] = xmax;
        }
        __syncthreads();
        xmax = Max[0];
    }
    // compute exponent of softmax
    float Xsum[U];
    for (int i = 0; i < U; i++)
    {
        // use fast approx math: e**x == 2**(x * log2(e))
        float exp = (xval[i] - xmax) * scale;
        asm("ex2.approx.ftz.f32 %0, %0;" : "+f"(exp) :);
        Xsum[i] = xval[i] = exp;
    }

    // reduce within thread
    for (int j = U >> 1; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            Xsum[i] = Xsum[i] + Xsum[i+j];
    float exp_sum = Xsum[0];

    // reduce within warp
    for (int i = 16; i > 0; i >>= 1)
        exp_sum += shfl_xor(exp_sum, i);

    if (blockDim.x > 32)
    {
        // first thread of each warp store to shared
        if ((tid & 31) == 0)
            Sum[tid/32] = exp_sum;
        __syncthreads();

        if (tid < 32)
        {
            // first warp loads all prior reductions
            exp_sum = Sum[tid];
            // reduce within this last warp
            #pragma unroll 1
            for (uint i = shfl_init; i > 0; i >>= 1)
                exp_sum += shfl_xor(exp_sum, i);
            // final reduction to shared
            Sum[tid] = exp_sum;
        }
        __syncthreads();
        exp_sum = Sum[0];
    }
    float rcp_exp_sum = exp_sum;
    asm("rcp.approx.ftz.f32 %0, %0;" : "+f"(rcp_exp_sum) :);

    #pragma unroll
    for (int i = 0; i < U; i++)
    {
        ehalf out;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(out.x) : "f"(xval[i] * rcp_exp_sum));
        uint offsetY = offset + Lut2s[lut_idx + i].x;
        if (lut_idx + i < lut_size)
            __stg(Y + offsetY, out);
    }
}

template <uint U, uint BSIZE, typename MASKT>
__global__ void blocksparse_masked_softmax_grad(
    const uint2* __restrict__ Lut,
    const MASKT* __restrict__ Mask,
    const ehalf* __restrict__ DY,
    const ehalf* __restrict__ Y,
          ehalf*              DX,
    uint blocks, uint szLut, uint szMask, uint szHead, uint szBatch, float scale, uint shfl_init, uint use_mask)
{
    __shared__ float Sum[32];
    uint2* Lut2s = (uint2*)&Sum[32];

    uint tid   = threadIdx.x;
    uint idx_Q = blockIdx.x / BSIZE; // Q dim
    uint idx_q = blockIdx.x % BSIZE; // Q dim
    uint idx_B = blockIdx.y; // batch dim
    uint idx_H = blockIdx.z; // head dim

    // each head can optionally have its own lut and/or mask
    Lut  += idx_H * szLut;
    Mask += idx_H * szMask + idx_q * blocks;
    uint2 lut_head = Lut[idx_Q];

    if (tid < 32)
        Sum[tid] = 0.0f;

    // prefetch the lut data into shared
    uint lut_offset = lut_head.x;
    uint lut_size   = lut_head.y;
    Lut += lut_offset;
    #pragma unroll 1
    for (uint i = tid; i < lut_size; i += blockDim.x)
    {
        uint2 entry = Lut[i];
        entry.y     = use_mask ? (uint)__ldg(Mask + entry.x) : 0xffffffff;
        entry.x    *= BSIZE*BSIZE;
        Lut2s[i]    = entry;
    }
    __syncthreads();

    uint lut_idx = (tid & (1024-BSIZE))*U/BSIZE;
    uint tidx = tid % BSIZE;
    uint mask_bit = 1 << tidx;
    uint offset = idx_B*szBatch + idx_H*szHead + idx_q*BSIZE + tidx;

    float dy[U], y[U];
    #pragma unroll
    for (int i = 0; i < U; i++)
    {
        uint2 entry = Lut2s[lut_idx + i];
        uint offsetY = offset + entry.x;
        bool in = lut_idx + i < lut_size && (entry.y & mask_bit) != 0;
        dy[i] = load(DY + offsetY, 0, in);
        y[i]  = load(Y  + offsetY, 0, in);
    }

    // compute dy * y
    float dyy[U];
    for (int i = 0; i < U; i++)
        dyy[i] = dy[i] * y[i];

    // reduce within thread
    for (int j = U >> 1; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            dyy[i] = dyy[i] + dyy[i+j];
    float sum_dyy = dyy[0];

    // reduce within warp
    for (int i = 16; i > 0; i >>= 1)
        sum_dyy += shfl_xor(sum_dyy, i);

    if (blockDim.x > 32)
    {
        // first thread of each warp store to shared
        if ((tid & 31) == 0)
            Sum[tid/32] = sum_dyy;
        __syncthreads();

        if (tid < 32)
        {
            // first warp loads all prior reductions
            sum_dyy = Sum[tid];
            // reduce within this last warp
            #pragma unroll 1
            for (uint i = shfl_init; i > 0; i >>= 1)
                sum_dyy += shfl_xor(sum_dyy, i);
            // final reduction to shared
            Sum[tid] = sum_dyy;
        }
        __syncthreads();
        sum_dyy = Sum[0];
    }

    // dx = (dy - sum_dyy) * y * scale
    #pragma unroll
    for (int i = 0; i < U; i++)
    {
        float dx = (dy[i] - sum_dyy) * y[i] * scale;
        ehalf out;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(out.x) : "f"(dx));
        uint offsetX = offset + Lut2s[lut_idx + i].x;
        if (lut_idx + i < lut_size)
            __stg(DX + offsetX, out);
    }
}

typedef unsigned long long uint64;

template <uint UNROLL>
__global__ void blocksparse_masked_softmax_64x64(
    const uint2*  __restrict__ Lut,
    const uint64* __restrict__ Mask,
    const ehalf*  __restrict__ X,
          ehalf*               Y,
    uint blocks, uint szLut, uint szMask, uint szHead, uint szBatch, float scale, uint shfl_init, uint max_lut, uint use_mask)
{
    __shared__ float Max[32];
    __shared__ float Sum[32];
    uint64* LutMask64 = (uint64*)&Sum[32];
    uint*   LutMask32 = (uint*)&Sum[32];
    uint*   LutOffset = (uint*)&LutMask64[max_lut];

    uint tid   = threadIdx.x;
    uint idx_Q = blockIdx.x / 64;
    uint idx_q = blockIdx.x % 64;
    uint idx_B = blockIdx.y; // batch dim
    uint idx_H = blockIdx.z; // head dim

    // each head can optionally have its own lut
    Lut  += idx_H * szLut;
    Mask += idx_H * szMask + idx_q * blocks;
    uint2 lut_head = Lut[idx_Q];

    if (tid < 32)
    {
        // Allows non-power of 2 threads to work
        Max[tid] = -FLT_MAX;
        Sum[tid] = 0.0f;
    }

    // prefetch the lut data into shared
    uint lut_offset = lut_head.x;
    uint lut_size   = lut_head.y;

    Lut += lut_offset;
    #pragma unroll 1
    for (uint i = tid; i < max_lut; i += blockDim.x)
    {
        uint64 mask = 0;
        if (i < lut_size)
        {
            uint2 entry  = Lut[i];
            uint blk_id  = entry.x;
            LutOffset[i] = blk_id * 64*64;
            mask = use_mask ? __ldg(Mask + blk_id) : 0xffffffffffffffff;
        }
        LutMask64[i] = mask;
    }
    __syncthreads();

    uint lut_idx = (tid & (1024-32))*UNROLL/32;
    uint tidx    = (tid & 31)*2;
    uint offset  = idx_B*szBatch + idx_H*szHead + idx_q*64 + tidx;

    ehalf2 xval[UNROLL];
    #pragma unroll
    for (int i = 0; i < UNROLL; i++)
    {
        xval[i].x = 0xfc00fc00; // -inf, -inf
        if (lut_idx + i < lut_size)
            xval[i] = __ldg((const ehalf2*)(X + (offset + LutOffset[lut_idx + i])));
    }

    // split the 64 bit mask by half warp
    uint tid16 = (tid & 16)/16;
    uint mask0 = 1 << (tidx - tid16*32);
    uint mask1 = mask0 << 1;

    #pragma unroll
    for (int i = 0; i < UNROLL; i++)
    {
        uint mask32 = LutMask32[(lut_idx + i)*2 + tid16];
        if ((mask32 & mask0) == 0)
            xval[i].x = (xval[i].x & 0xffff0000) | 0x0000fc00;
        if ((mask32 & mask1) == 0)
            xval[i].x = (xval[i].x & 0x0000ffff) | 0xfc000000;
    }


    // reduce within thread
    float Xmax[UNROLL];
    for (int i = 0; i < UNROLL; i++)
        Xmax[i] = ew_max(to_float(xval[i]));

    for (int j = UNROLL/2; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            Xmax[i] = fmaxf(Xmax[i], Xmax[i+j]);
    float xmax = Xmax[0];

    // reduce within warp
    for (int i = 16; i > 0; i >>= 1)
        xmax = fmaxf(xmax, shfl_xor(xmax, i));

    if (blockDim.x > 32)
    {
        // first thread of each warp store to shared
        if ((tid & 31) == 0)
            Max[tid/32] = xmax;
        __syncthreads();
        if (tid < 32)
        {
            // first warp loads all prior reductions
            xmax = Max[tid];
            // reduce within this last warp
            #pragma unroll 1
            for (uint i = shfl_init; i > 0; i >>= 1)
                xmax = fmaxf(xmax, shfl_xor(xmax, i));
            // final reduction to shared
            Max[tid] = xmax;
        }
        __syncthreads();
        xmax = Max[0];
    }

    // subtract xmax and compute exponent
    float2 Xval[UNROLL];
    for (int i = 0; i < UNROLL; i++)
    {
        // use fast approx math: e**x == 2**(x * log2(e))
        Xval[i] = ew_mul(ew_sub(to_float(xval[i]), xmax), scale);
        asm("ex2.approx.ftz.f32 %0, %0;" : "+f"(Xval[i].x) :);
        asm("ex2.approx.ftz.f32 %0, %0;" : "+f"(Xval[i].y) :);
    }

    // reduce within thread
    float Xsum[UNROLL];
    for (int i = 0; i < UNROLL; i++)
        Xsum[i] = ew_sum(Xval[i]);

    for (int j = UNROLL/2; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            Xsum[i] = Xsum[i] + Xsum[i+j];
    float exp_sum = Xsum[0];

    // reduce within warp
    for (int i = 16; i > 0; i >>= 1)
        exp_sum += shfl_xor(exp_sum, i);

    if (blockDim.x > 32)
    {
        // first thread of each warp store to shared
        if ((tid & 31) == 0)
            Sum[tid/32] = exp_sum;
        __syncthreads();

        if (tid < 32)
        {
            // first warp loads all prior reductions
            exp_sum = Sum[tid];
            // reduce within this last warp
            #pragma unroll 1
            for (uint i = shfl_init; i > 0; i >>= 1)
                exp_sum += shfl_xor(exp_sum, i);
            // final reduction to shared
            Sum[tid] = exp_sum;
        }
        __syncthreads();
        exp_sum = Sum[0];
    }
    float rcp_exp_sum = exp_sum;
    asm("rcp.approx.ftz.f32 %0, %0;" : "+f"(rcp_exp_sum) :);

    #pragma unroll
    for (int i = 0; i < UNROLL; i++)
    {
        ehalf2 out = to_ehalf(ew_mul(Xval[i], rcp_exp_sum));

        uint offsetY = offset + LutOffset[lut_idx + i];

        if (lut_idx + i < lut_size)
            __stg((ehalf2*)(Y + offsetY), out);
    }
}


template <uint UNROLL>
__global__ void blocksparse_masked_softmax_64x64_grad(
    const uint2* __restrict__ Lut,
    const ehalf* __restrict__ DY,
    const ehalf* __restrict__ Y,
          ehalf*              DX,
    uint blocks, uint szLut, uint szMask, uint szHead, uint szBatch, float scale, uint shfl_init)
{
    __shared__ float Sum[32];
    uint* LutOffset = (uint*)&Sum[32];

    uint tid   = threadIdx.x;
    uint idx_Q = blockIdx.x / 64;
    uint idx_q = blockIdx.x % 64;
    uint idx_B = blockIdx.y; // batch dim
    uint idx_H = blockIdx.z; // head dim

    // each head can optionally have its own lut
    Lut  += idx_H * szLut;
    uint2 lut_head = Lut[idx_Q];

    if (tid < 32)
        Sum[tid] = 0.0f;

    // prefetch the lut data into shared
    uint lut_offset = lut_head.x;
    uint lut_size   = lut_head.y;

    Lut += lut_offset;
    #pragma unroll 1
    for (uint i = tid; i < lut_size; i += blockDim.x)
        LutOffset[i] = Lut[i].x * 64*64;
    __syncthreads();

    uint lut_idx = (tid & (1024-32))*UNROLL/32;
    uint tidx    = (tid & 31)*2;
    uint offset  = idx_B*szBatch + idx_H*szHead + idx_q*64 + tidx;

    ehalf2 dy[UNROLL], y[UNROLL];
    #pragma unroll
    for (int i = 0; i < UNROLL; i++)
    {
        dy[i].x = y[i].x = 0;
        if (lut_idx + i < lut_size)
        {
            uint offsetY = offset + LutOffset[lut_idx + i];
            dy[i] = __ldg((const ehalf2*)(DY + offsetY));
            y[i]  = __ldg((const ehalf2*)(Y  + offsetY));
        }
    }

    // compute dy * y and start reduction
    float dyy[UNROLL];
    for (int i = 0; i < UNROLL; i++)
        dyy[i] = ew_sum(ew_mul(to_float(dy[i]), to_float(y[i])));

    // reduce within thread
    for (int j = UNROLL/2; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            dyy[i] = dyy[i] + dyy[i+j];
    float sum_dyy = dyy[0];

    // reduce within warp
    for (int i = 16; i > 0; i >>= 1)
        sum_dyy += shfl_xor(sum_dyy, i);

    if (blockDim.x > 32)
    {
        // first thread of each warp store to shared
        if ((tid & 31) == 0)
            Sum[tid/32] = sum_dyy;
        __syncthreads();

        if (tid < 32)
        {
            // first warp loads all prior reductions
            sum_dyy = Sum[tid];
            // reduce within this last warp
            #pragma unroll 1
            for (uint i = shfl_init; i > 0; i >>= 1)
                sum_dyy += shfl_xor(sum_dyy, i);
            // final reduction to shared
            Sum[tid] = sum_dyy;
        }
        __syncthreads();
        sum_dyy = Sum[0];
    }

    // dx = (dy - sum_dyy) * y * scale
    #pragma unroll
    for (int i = 0; i < UNROLL; i++)
    {
        // dx = (dy - sum_dyy) * y * scale
        float2 dx = ew_mul(ew_mul(ew_sub(to_float(dy[i]), sum_dyy), to_float(y[i])), scale);
        if (lut_idx + i < lut_size)
        {
            uint offsetX = offset + LutOffset[lut_idx + i];
            __stg((ehalf2*)(DX + offsetX), to_ehalf(dx));
        }
    }
}

#define LOG2e 1.4426950408889634f

bool BlocksparseMaskedSoftmax(CUstream stream,
    const uint2* lut,
    const  char* mask,
    const ehalf* x,
          ehalf* y,
    uint block_size, uint blocks,
    uint batch_dim,  uint head_dim, uint ctx_blks,
    uint lut_heads,  uint lut_dim,  uint max_lut,
    uint mask_heads, float scale)
{
    uint szLut   = lut_heads  > 1 ? lut_dim  : 0;
    uint szMask  = mask_heads > 1 ? blocks * block_size : 0;
    uint gridQ   = ctx_blks * block_size;
    uint szHead  = blocks * block_size * block_size;
    uint szBatch = head_dim * szHead;
    uint maxK    = max_lut * block_size;
    //cuMemsetD16Async((CUdeviceptr)c, 0, szBatch*batch_dim, stream);

    // combine scaling with fast e**x compute
    scale *= LOG2e;

    dim3 grid(gridQ, batch_dim, head_dim);

    if (block_size == 64)
    {
        // Unroll factor 8 (ctx_size up to 16K)
        if (maxK > 1024*8)
        {
            uint threads   = 1024;
            uint shfl_init = 16;
            uint lut_max   = threads * 8 / 32;
            uint shared    = lut_max * 12;
            blocksparse_masked_softmax_64x64<8><<<grid,threads,shared,stream>>>(lut, (const uint64*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
        }
        // Unroll factor of 4 is preferred (keeps these kernels under 32 registers for max occupancy)
        else if (maxK >= 64*4)
        {
            uint threads   = CEIL_DIV(maxK, 64*4) * 32;
            uint shfl_init = THREAD_POW2(threads) / 64;
            uint lut_max   = threads * 4 / 32;
            uint shared    = lut_max * 12;
            blocksparse_masked_softmax_64x64<4><<<grid,threads,shared,stream>>>(lut, (const uint64*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
        }
        // Unroll factor 1
        else
        {
            uint threads   = CEIL_DIV(maxK, 64) * 32;
            uint shfl_init = THREAD_POW2(threads) / 64;
            uint lut_max   = threads * 1 / 32;
            uint shared    = lut_max * 12;
            blocksparse_masked_softmax_64x64<1><<<grid,threads,shared,stream>>>(lut, (const uint64*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
        }
    }
    else
    {
        uint shared = max_lut*8;

        // Unroll factor 8 (ctx_size up to 8K)
        if (maxK > 1024*4)
        {
            if (block_size == 16)
                blocksparse_masked_softmax<8,16,ushort><<<grid,1024,shared,stream>>>(lut, (const ushort*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, 16, mask != NULL);
            else
                blocksparse_masked_softmax<8,32,  uint><<<grid,1024,shared,stream>>>(lut, (const   uint*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, 16, mask != NULL);
        }
        // Unroll factor of 4 is preferred
        else if (maxK > 32*4)
        {
            uint threads   = CEIL_DIV(maxK, 32*4) * 32;
            uint shfl_init = THREAD_POW2(threads) / 64;
            if (block_size == 16)
                blocksparse_masked_softmax<4,16,ushort><<<grid,threads,shared,stream>>>(lut, (const ushort*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, mask != NULL);
            else
                blocksparse_masked_softmax<4,32,  uint><<<grid,threads,shared,stream>>>(lut, (const   uint*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, mask != NULL);
        }
        // Unroll factor 1
        else
        {
            uint threads   = CEIL_DIV(maxK, 32) * 32;
            uint shfl_init = THREAD_POW2(threads) / 64;
            if (block_size == 16)
                blocksparse_masked_softmax<1,16,ushort><<<grid,threads,shared,stream>>>(lut, (const ushort*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, mask != NULL);
            else
                blocksparse_masked_softmax<1,32,  uint><<<grid,threads,shared,stream>>>(lut, (const   uint*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, mask != NULL);
        }
    }
    return true;
}

bool BlocksparseMaskedSoftmaxGrad(CUstream stream,
    const uint2* lut,
    const  char* mask,
    const ehalf* dy,
    const ehalf* y,
          ehalf* dx,
    uint block_size, uint blocks,
    uint batch_dim,  uint head_dim, uint ctx_blks,
    uint lut_heads,  uint lut_dim,  uint max_lut,
    uint mask_heads, float scale)
{
    uint szLut   = lut_heads  > 1 ? lut_dim  : 0;
    uint szMask  = mask_heads > 1 ? blocks * block_size : 0;
    uint gridQ   = ctx_blks * block_size;
    uint szHead  = blocks * block_size * block_size;
    uint szBatch = head_dim * szHead;
    uint maxK    = max_lut * block_size;
    //cuMemsetD16Async((CUdeviceptr)c, 0, szBatch*batch_dim, stream);

    dim3 grid(gridQ, batch_dim, head_dim);

    if (block_size == 64)
    {
        uint shared = max_lut*4;

        // Unroll factor 8
        if (maxK > 1024*8)
        {
            uint threads   = 1024;
            uint shfl_init = 16;
            blocksparse_masked_softmax_64x64_grad<8><<<grid,threads,shared,stream>>>(lut, dy, y, dx, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init);
        }
        // Unroll factor of 4 is preferred
        else if (maxK >= 64*4)
        {
            uint threads   = CEIL_DIV(maxK, 64*4) * 32;
            uint shfl_init = THREAD_POW2(threads) / 64;
            blocksparse_masked_softmax_64x64_grad<4><<<grid,threads,shared,stream>>>(lut, dy, y, dx, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init);
        }
        // Unroll factor 1
        else
        {
            uint threads   = CEIL_DIV(maxK, 64) * 32;
            uint shfl_init = THREAD_POW2(threads) / 64;
            blocksparse_masked_softmax_64x64_grad<1><<<grid,threads,shared,stream>>>(lut, dy, y, dx, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init);
        }
    }
    else
    {
        uint shared  = max_lut*8;

        // Unroll factor 8
        if (maxK > 1024*4)
        {
            if (block_size == 16)
                blocksparse_masked_softmax_grad<8,16,ushort><<<grid,1024,shared,stream>>>(lut, (const ushort*)mask, dy, y, dx, blocks, szLut, szMask, szHead, szBatch, scale, 16, mask != NULL);
            else
                blocksparse_masked_softmax_grad<8,32,  uint><<<grid,1024,shared,stream>>>(lut, (const   uint*)mask, dy, y, dx, blocks, szLut, szMask, szHead, szBatch, scale, 16, mask != NULL);
        }
        // Unroll factor of 4 is preferred
        else if (maxK > 32*4)
        {
            uint threads   = CEIL_DIV(maxK, 32*4) * 32;
            uint shfl_init = THREAD_POW2(threads) / 64;
            if (block_size == 16)
                blocksparse_masked_softmax_grad<4,16,ushort><<<grid,threads,shared,stream>>>(lut, (const ushort*)mask, dy, y, dx, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, mask != NULL);
            else
                blocksparse_masked_softmax_grad<4,32,  uint><<<grid,threads,shared,stream>>>(lut, (const   uint*)mask, dy, y, dx, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, mask != NULL);
        }
        // Unroll factor 1
        else
        {
            uint threads   = CEIL_DIV(maxK, 32) * 32;
            uint shfl_init = THREAD_POW2(threads) / 64;
            if (block_size == 16)
                blocksparse_masked_softmax_grad<1,16,ushort><<<grid,threads,shared,stream>>>(lut, (const ushort*)mask, dy, y, dx, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, mask != NULL);
            else
                blocksparse_masked_softmax_grad<1,32,  uint><<<grid,threads,shared,stream>>>(lut, (const   uint*)mask, dy, y, dx, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, mask != NULL);
        }
    }

    return true;
}

#endif // GOOGLE_CUDA
