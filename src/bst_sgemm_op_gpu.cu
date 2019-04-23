#if GOOGLE_CUDA

#include "ew_op_gpu.h"
#include "gpu_hmma.h"
#include <stdio.h>

template <uint OP_A, bool N64>
__global__ void __launch_bounds__(128,6) bst_sgemm_32x64x32_xn(
    const uint2* __restrict__ Lut,
    const bhalf* __restrict__ A,
    const float* __restrict__ B,
          float*              C,
    uint szCtxHeadStateB, uint szCtxHeadStateC, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint grid_M, uint grid_N, uint magic_N, uint shift_N)
{
    __shared__ float fShare[(33 + 64)*32];
    uint2* Lut2s = (uint2*)&fShare[(33 + 64)*32];
    char* bShare = (char*)&fShare;

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

    uint txb = tid % 16;
    uint tyb = tid / 16;

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

        uint txa = tid % 8;
        uint tya = tid / 8;

        uint tid16 = tid & 16;
        uint tid96 = tid & 96;

        uint loadB = ((tid / 2) % 8) * 4*4;
        uint loadA =  (tid % 2)      * 4*4;

        // each warp handles a quarter of the weights
        loadA += tid96;

        // second half of warp starts 16 rows down
        loadB += tid16 * 64*4;
        loadA += tid16 * 32*4;

        uint storB = (tyb*64 + txb*4) * 4;
        uint storA;
        if (OP_A == OP_T)
            storA = tid * 4*4;
        else
        {
            // Transpose weights on store to shared
            // Avoid bank conflicts by shifting writes over by 4 every 4 rows (+txa*4)
            storA = (txa*32*4 + tya + txa*4) * 4;
            loadA += tid16 * 4; // shift over 4 floats every 4 rows, second half of warp starts 16 rows down
        }

        uint b = idx_N*64 + txb*4;
        uint offsetA = idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + tid*4;
        uint offsetB = idx_B*szCtxHeadStateB + tyb*szHeadState + idx_H*szState + b;

        bool inB = N64 || b < szState;

        // zero accumulation registers
        float regC[4][8];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 8; j++)
                regC[i][j] = 0.0f;

        // Force compiler to fully compute these prior to loop
        asm("mov.b32 %0, %0;" : "+r"(loadA)   : );
        asm("mov.b32 %0, %0;" : "+r"(loadB)   : );
        asm("mov.b32 %0, %0;" : "+r"(storA)   : );
        asm("mov.b32 %0, %0;" : "+r"(storB)   : );
        asm("mov.b32 %0, %0;" : "+r"(offsetA) : );
        asm("mov.b32 %0, %0;" : "+r"(offsetB) : );

        int idx_lut = 0;
        #pragma unroll 1
        do
        {
            //asm volatile (".pragma \"nounroll\";"::); // ptxas, don't get clever

            uint2 entry = Lut2s[idx_lut];

            const bhalf* pA = add_ptr_u(A, entry.x + offsetA);
            bhalf4 a00 = __ldg((const bhalf4*)(pA +  0*32));
            bhalf4 a16 = __ldg((const bhalf4*)(pA + 16*32));
            float4 b00 = {0.0f}, b08 = {0.0f}, b16 = {0.0f}, b24 = {0.0f};
            entry.y += offsetB;
            if (inB)
            {
                b00 = __ldg((const float4*)(B + (entry.y +  0*szHeadState)));
                b08 = __ldg((const float4*)(B + (entry.y +  8*szHeadState)));
                b16 = __ldg((const float4*)(B + (entry.y + 16*szHeadState)));
                b24 = __ldg((const float4*)(B + (entry.y + 24*szHeadState)));
            }
            __syncthreads();

            float4 fa00 = to_float(a00);
            float4 fa16 = to_float(a16);

            if (OP_A == OP_T)
            {
                *(float4*)&bShare[storA + (0*16*32 + 64*32)*4] = fa00;
                *(float4*)&bShare[storA + (1*16*32 + 64*32)*4] = fa16;
            }
            else
            {
                // transpose the shared store of W
                *(float*)&bShare[storA + (0*32 + 0*16 + 64*32)*4] = fa00.x;
                *(float*)&bShare[storA + (1*32 + 0*16 + 64*32)*4] = fa00.y;
                *(float*)&bShare[storA + (2*32 + 0*16 + 64*32)*4] = fa00.z;
                *(float*)&bShare[storA + (3*32 + 0*16 + 64*32)*4] = fa00.w;

                *(float*)&bShare[storA + (0*32 + 1*16 + 64*32)*4] = fa16.x;
                *(float*)&bShare[storA + (1*32 + 1*16 + 64*32)*4] = fa16.y;
                *(float*)&bShare[storA + (2*32 + 1*16 + 64*32)*4] = fa16.z;
                *(float*)&bShare[storA + (3*32 + 1*16 + 64*32)*4] = fa16.w;
            }

            *(float4*)&bShare[storB +  0*64*4] = b00;
            *(float4*)&bShare[storB +  8*64*4] = b08;
            *(float4*)&bShare[storB + 16*64*4] = b16;
            *(float4*)&bShare[storB + 24*64*4] = b24;
            __syncthreads();

            // computes a 32x64x32 gemm tile with 4x8 register blocking
            float regA[4];
            float regB[8];
            #pragma unroll
            for (int j = 0; j < 16; j++)
            {
                // fetch outer product data
                *(float4*)&regA[0] = *(float4*)&bShare[loadA + (32*j + 64*32 + (OP_A == OP_T ? 0 : (j/4)*4))*4]; // shift over 4 floats every 4 rows
                *(float4*)&regB[0] = *(float4*)&bShare[loadB + (64*j +  0)*4];
                *(float4*)&regB[4] = *(float4*)&bShare[loadB + (64*j + 32)*4];

                // accumulate outer product
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 8; j++)
                        regC[i][j] += regA[i] * regB[j];
            }


        } while (++idx_lut < lut_size);

        asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid  )  :);
        asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(idx_MN) :);
        asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(idx_B)  :);
        asm volatile ("mov.u32 %0, %ctaid.z;" : "=r"(idx_H)  :);
        idx_M  = div64(idx_MN, magic_N, shift_N);
        idx_N  = idx_MN - idx_M*grid_N;

        if (OP_A == OP_N)
            idx_M = grid_M - idx_M;

        // printf("%3d %.0f %.0f %.0f %.0f |  %.0f %.0f %.0f %.0f |  %.0f %.0f %.0f %.0f | %.0f %.0f %.0f %.0f\n", tid,
        //     regC[0][0], regC[0][1], regC[0][2], regC[0][3],
        //     regC[1][0], regC[1][1], regC[1][2], regC[1][3],
        //     regC[2][0], regC[2][1], regC[2][2], regC[2][3],
        //     regC[3][0], regC[3][1], regC[3][2], regC[3][3]);

        tid16 = tid & 16;
        tid96 = tid & 96;

        uint tn =  (tid / 2) % 8;
        uint tm = ((tid % 2) + (tid96 / 16))*4 + (tid16 / 16);

        bool t16 = tid16 != 0;

        float outC[2][8];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 8; j++)
            {
                float swap = t16 ? regC[2*i + 0][j] : regC[2*i + 1][j];
                outC[i][j] = t16 ? regC[2*i + 1][j] : regC[2*i + 0][j];
                outC[i][j] += shfl_xor(swap, 16);
            }

        uint n = idx_N*64 + tn*4;
        bool bn00 = N64 || n +  0 < szState;
        bool bn32 = N64 || n + 32 < szState;

        uint offsetC = idx_B*szCtxHeadStateC + (idx_M*32 + tm)*szHeadState + idx_H*szState + n;

        store((float4*)(C + (offsetC + szHeadState*0 +  0)), *(float4*)&outC[0][0], 0, bn00);
        store((float4*)(C + (offsetC + szHeadState*0 + 32)), *(float4*)&outC[0][4], 0, bn32);
        store((float4*)(C + (offsetC + szHeadState*2 +  0)), *(float4*)&outC[1][0], 0, bn00);
        store((float4*)(C + (offsetC + szHeadState*2 + 32)), *(float4*)&outC[1][4], 0, bn32);
    }
    else
    {
        uint c       = idx_N*64 + txb*4;
        uint offsetC = idx_B*szCtxHeadStateC + (idx_M*32 + tyb)*szHeadState + idx_H*szState + c;

        if (N64 || c < szState)
        {
            float4 zero = {0.0f};
            *(float4*)&C[offsetC + szHeadState* 0] = zero;
            *(float4*)&C[offsetC + szHeadState* 8] = zero;
            *(float4*)&C[offsetC + szHeadState*16] = zero;
            *(float4*)&C[offsetC + szHeadState*24] = zero;
        }
    }
}

template <bool K64>
__global__ void __launch_bounds__(256,3) bst_sgemm_32x32x64_nt(
    const uint2* __restrict__ Lut,
    const float* __restrict__ A,
    const float* __restrict__ B,
          bhalf*              C,
    uint szCtxHeadStateA, uint szCtxHeadStateB, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut, uint loops)
{
    __shared__ float fShare[65*32*2];
    char* bShare = (char*)fShare;

    uint tid   = threadIdx.x;
    uint bid   = blockIdx.x; // blockid
    uint idx_B = blockIdx.y; // batch dim
    uint idx_H = blockIdx.z; // head dim

    // each head can optionally have its own lut
    uint2 lut_head = Lut[idx_H*szLut + bid];

    uint tx = tid % 16;
    uint ty = tid / 16;
    uint k  = tx  * 4;

    uint idx_M = lut_head.x;
    uint idx_N = lut_head.y;
    uint offsetA00 = idx_B*szCtxHeadStateA + (idx_M*32 + ty)*szHeadState + idx_H*szState + k;
    uint offsetB00 = idx_B*szCtxHeadStateB + (idx_N*32 + ty)*szHeadState + idx_H*szState + k;
    uint offsetA16 = offsetA00 + szHeadState*16;
    uint offsetB16 = offsetB00 + szHeadState*16;

    uint tid224 = tid & 224; // 256 - 32

    // avoid bank conflicts when writing transpose (+ tx*2)
    uint storAB = (tx*32*4 + ty + tx*2)*4;

    // 32 threads per tile, each tile reads 8 lines, shifted over by 4
    uint loadA = (((tid & 16) >> 3) | (tid & 1)) << 4;
    uint loadB = ((tid >> 1) & 7) << 4;

    loadA += (tid224 * 32) + (tid224 / 2); // 32*8*4
    loadB += (tid224 * 32) + (tid224 / 2); // 32*8*4

    // This keeps all prior logic outside of the loops.
    asm("mov.b32 %0, %0;" : "+r"(storAB) : );
    asm("mov.b32 %0, %0;" : "+r"(loadA)  : );
    asm("mov.b32 %0, %0;" : "+r"(loadB)  : );

    float regC[8][4];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++)
            regC[i][j] = 0.0f;

    uint loop = 0;
    #pragma unroll 1
    do
    {
        float4 a00 = {0}, a16 = {0};
        float4 b00 = {0}, b16 = {0};
        if (K64 || k < szState)
        {
            a00 = __ldg((const float4*)(add_ptr_u(A, offsetA00)));
            a16 = __ldg((const float4*)(add_ptr_u(A, offsetA16)));
            b00 = __ldg((const float4*)(add_ptr_u(B, offsetB00)));
            b16 = __ldg((const float4*)(add_ptr_u(B, offsetB16)));
        }
        offsetA00 += 64;
        offsetA16 += 64;
        offsetB00 += 64;
        offsetB16 += 64;
        if (!K64)
            k += 64;

        __syncthreads();
        *(float*)&bShare[storAB + (0*32 +  0 + 0*65*32)*4] = a00.x;
        *(float*)&bShare[storAB + (1*32 +  0 + 0*65*32)*4] = a00.y;
        *(float*)&bShare[storAB + (2*32 +  0 + 0*65*32)*4] = a00.z;
        *(float*)&bShare[storAB + (3*32 +  0 + 0*65*32)*4] = a00.w;
        *(float*)&bShare[storAB + (0*32 + 16 + 0*65*32)*4] = a16.x;
        *(float*)&bShare[storAB + (1*32 + 16 + 0*65*32)*4] = a16.y;
        *(float*)&bShare[storAB + (2*32 + 16 + 0*65*32)*4] = a16.z;
        *(float*)&bShare[storAB + (3*32 + 16 + 0*65*32)*4] = a16.w;

        *(float*)&bShare[storAB + (0*32 +  0 + 1*65*32)*4] = b00.x;
        *(float*)&bShare[storAB + (1*32 +  0 + 1*65*32)*4] = b00.y;
        *(float*)&bShare[storAB + (2*32 +  0 + 1*65*32)*4] = b00.z;
        *(float*)&bShare[storAB + (3*32 +  0 + 1*65*32)*4] = b00.w;
        *(float*)&bShare[storAB + (0*32 + 16 + 1*65*32)*4] = b16.x;
        *(float*)&bShare[storAB + (1*32 + 16 + 1*65*32)*4] = b16.y;
        *(float*)&bShare[storAB + (2*32 + 16 + 1*65*32)*4] = b16.z;
        *(float*)&bShare[storAB + (3*32 + 16 + 1*65*32)*4] = b16.w;
        __syncthreads();

        float regA[8], regB[4];
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            // fetch outer product data
            *(float4*)&regA[0] = *(float4*)&bShare[loadA + (32*j +  0)*4];
            *(float4*)&regA[4] = *(float4*)&bShare[loadA + (32*j + 16)*4];
            *(float4*)&regB[0] = *(float4*)&bShare[loadB + (32*j + 65*32)*4];

            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 4; j++)
                    regC[i][j] += regA[i] * regB[j];
        }
        #pragma unroll
        for (int j = 4; j < 8; j++)
        {
            *(float2*)&regA[0] = *(float2*)&bShare[loadA + (32*j +  0 + (j/4)*2)*4];
            *(float2*)&regA[2] = *(float2*)&bShare[loadA + (32*j +  2 + (j/4)*2)*4];
            *(float2*)&regA[4] = *(float2*)&bShare[loadA + (32*j + 16 + (j/4)*2)*4];
            *(float2*)&regA[6] = *(float2*)&bShare[loadA + (32*j + 18 + (j/4)*2)*4];
            *(float2*)&regB[0] = *(float2*)&bShare[loadB + (32*j +  0 + (j/4)*2 + 65*32)*4];
            *(float2*)&regB[2] = *(float2*)&bShare[loadB + (32*j +  2 + (j/4)*2 + 65*32)*4];

            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 4; j++)
                    regC[i][j] += regA[i] * regB[j];
        }


    } while (++loop < loops);

    asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid)   :);
    asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(bid)   :);
    asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(idx_B) :);
    asm volatile ("mov.u32 %0, %ctaid.z;" : "=r"(idx_H) :);

    //printf("%3d %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n", tid, regC[0][0], regC[0][1], regC[0][2], regC[0][3], regC[4][0], regC[4][1], regC[4][2], regC[4][3]);

    // if ((tid & 31) == 0)
    //     printf("%3d %.0f\n", tid, regC[0][0]);

    C += idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + bid*32*32 + tid*2;

    // Arrange 8 tiles horizontally in the X direction: ((tid & 224) >> 1)
    // Add some spacing  to avoid write bank conflicts: (ty << 2)
    ty = ((tid & 16) >> 3) + (tid & 1);
    tx = ((tid >> 1) & 7) + ((tid & 224) >> 2) + (ty << 2);

    uint storC = ty*32*8*4 + tx*4;

    tx = tid % 16;
    ty = tid / 16;

    uint readC = ty*32*8 + tx*2 + ((tid & 192)>>2);

    __syncthreads();
    *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[0];
    *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[1];
    *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[2];
    *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[3];
    __syncthreads();

    float2 c2[8];
    for (int i = 0; i < 8; i++)
        c2[i] = *(float2*)&fShare[readC + i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c2[i] = ew_add(c2[i], c2[i+j]);

    store((bhalf2*)C, c2[0]);

    __syncthreads();
    *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[4];
    *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[5];
    *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[6];
    *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[7];
    __syncthreads();

    for (int i = 0; i < 8; i++)
        c2[i] = *(float2*)&fShare[readC + i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c2[i] = ew_add(c2[i], c2[i+j]);

    store((bhalf2*)(C + 16*32), c2[0]);
}


bool bst_sgemm_xn(CUstream stream,
    const uint2* lut,
    const bhalf* a,
    const float* b,
          float* c,
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
    bool n64    = (state_dim & 63) == 0;

    dim3 grid(gridX, batch_dim, head_dim);
    if (block_size == 32)
    {
        if (op == NN_OP) // NN
        {
            if (n64)
                bst_sgemm_32x64x32_xn<OP_N, true><<<grid,128,shared,stream>>>(lut, a, b, c, szCtxHeadStateB, szCtxHeadStateC, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic, shift);
            else
                bst_sgemm_32x64x32_xn<OP_N,false><<<grid,128,shared,stream>>>(lut, a, b, c, szCtxHeadStateB, szCtxHeadStateC, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic, shift);
        }
        else // TN
        {
            if (n64)
                bst_sgemm_32x64x32_xn<OP_T, true><<<grid,128,shared,stream>>>(lut, a, b, c, szCtxHeadStateB, szCtxHeadStateC, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic, shift);
            else
                bst_sgemm_32x64x32_xn<OP_T,false><<<grid,128,shared,stream>>>(lut, a, b, c, szCtxHeadStateB, szCtxHeadStateC, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic, shift);
        }
    }
    return true;
}

bool bst_sgemm_nt(CUstream stream,
    const uint2* lut,
    const float* a,
    const float* b,
          bhalf* c,
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
    if (block_size == 32)
    {
        if (k64)
            bst_sgemm_32x32x64_nt< true><<<grid,256,0,stream>>>(lut, a, b, c, szCtxHeadStateA, szCtxHeadStateB, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, loops);
        else
            bst_sgemm_32x32x64_nt<false><<<grid,256,0,stream>>>(lut, a, b, c, szCtxHeadStateA, szCtxHeadStateB, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, loops);
    }
    return true;
}



#endif // GOOGLE_CUDA