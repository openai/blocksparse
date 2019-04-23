

#if GOOGLE_CUDA

#include "ew_op_gpu.h"
#include "gpu_hmma.h"
#include <stdio.h>


typedef unsigned long long uint64;

template <uint UNROLL, uint BLOCKS, uint BSIZE, typename T, typename V2, typename MASKT>
__global__ void __launch_bounds__(1024,BLOCKS) bst_masked_softmax(
    const uint2* __restrict__ Lut,
    const MASKT* __restrict__ Mask,
    const bhalf* __restrict__ X,
              T*              Y,
    uint blocks, uint szLut, uint szMask, uint szHead, uint szBatch, float scale, uint shfl_init, uint max_lut, uint use_mask)

{
    __shared__ float Max[32];
    __shared__ float Sum[32];
    uint64* LutMask64 = (uint64*)&Sum[32];
    uint*   LutMask32 = (uint*)&Sum[32];
    uint*   LutOffset = BSIZE == 64 ? (uint*)&LutMask64[max_lut] : &LutMask32[max_lut];

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
    for (uint i = tid; i < max_lut; i += blockDim.x)
    {
        if (BSIZE == 64)
        {
            uint64 mask = 0;
            if (i < lut_size)
            {
                uint2 entry  = Lut[i];
                uint blk_id  = entry.x;
                LutOffset[i] = blk_id * BSIZE*BSIZE;
                mask = use_mask ? __ldg(Mask + blk_id) : 0xffffffffffffffff;
            }
            LutMask64[i] = mask;
        }
        else
        {
            uint mask = 0;
            if (i < lut_size)
            {
                uint2 entry  = Lut[i];
                uint blk_id  = entry.x;
                LutOffset[i] = blk_id * BSIZE*BSIZE;
                mask = use_mask ? (uint)__ldg(Mask + blk_id) : 0xffffffff;
            }
            LutMask32[i] = mask;
        }
    }
    __syncthreads();

    // trim warps that we know are out of lut range
    if ((tid & (1024-32))*2*UNROLL < lut_size*BSIZE)
    {
        uint lut_idx  = (tid & (1024 - BSIZE/2))*2*UNROLL/BSIZE;
        uint tidx     = (tid % (BSIZE/2))*2;
        uint offset   = idx_B*szBatch + idx_H*szHead + idx_q*BSIZE + tidx + LutOffset[lut_idx];
        X += offset;
        asm("mov.b64 %0, %0;" : "+l"(X) : );

        bhalf2 xval[UNROLL];
        #pragma unroll
        for (uint i = 0; i < UNROLL; i++)
        {
            ew_set(xval[i], 0xff80ff80); //-inf, -inf

            if (lut_idx + i < lut_size)
                xval[i] = __ldg((const bhalf2*)(X + i*BSIZE*BSIZE));
        }

        // split the 64 bit mask by half warp
        uint tid16 = BSIZE == 64 ? (tid & 16)/16 : 0;
        uint bit0  = 1 << (tidx - tid16*32);
        uint bit1  = bit0 << 1;
        uint inf   = 0xff80;
        #pragma unroll
        for (int i = 0; i < UNROLL; i++)
        {
            uint mask = LutMask32[(lut_idx + i)*(BSIZE == 64 ? 2 : 1) + tid16];
            asm("{                               \n\t"
                ".reg .pred p0, p1;              \n\t"
                "setp.eq.u32 p0, %2, 0;          \n\t" // if ((mask & bit0) == 0)
                "setp.eq.u32 p1, %3, 0;          \n\t" // if ((mask & bit1) == 0)
                "@p0 prmt.b32 %0, %0, %1, 0x3254;\n\t" // set -inf to lo bits
                "@p1 prmt.b32 %0, %0, %1, 0x5410;\n\t" // set -inf to hi bits
                "}" : "+r"(xval[i].x) : "r"(inf), "r"(mask & bit0), "r"(mask & bit1));
        }

        // reduce within thread
        float Xmax[UNROLL];
        for (int i = 0; i < UNROLL; i++)
            Xmax[i] = ew_max(to_float(xval[i]));

        float xmax = Xmax[0];
        for (int i = 1; i < UNROLL; i++)
            xmax = fmaxf(Xmax[i], xmax);

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
        float exp_sum = 0;
        for (int i = 0; i < UNROLL; i++)
        {
            // use fast approx math: e**x == 2**(x * log2(e))
            // log2(e) is included in scale factor
            float2 Xval = ew_ex2(ew_mul(ew_sub(to_float(xval[i]), xmax), scale));
            exp_sum    += ew_sum(Xval);
            xval[i]     = to_bhalf(Xval);
        }

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
        float rcp_exp_sum = ew_rcp(exp_sum);
        Y += offset;
        asm("mov.b64 %0, %0;" : "+l"(Y) : );

        #pragma unroll
        for (int i = 0; i < UNROLL; i++)
        {
            float2 y2 = ew_mul(to_float(xval[i]), rcp_exp_sum);

            store((V2*)Y, y2, i*BSIZE*BSIZE/2, lut_idx + i < lut_size);
        }
    }
}

template <uint UNROLL, uint BLOCKS, uint BSIZE, typename T, typename V2>
__global__ void __launch_bounds__(1024,BLOCKS) bst_masked_softmax_grad(
    const uint2* __restrict__ Lut,
    const     T* __restrict__ DY,
    const     T* __restrict__ Y,
              T*              DX,
    uint szLut, uint szHead, uint szBatch, float scale, uint shfl_init)
{
    __shared__ float Sum[32];
    uint* LutOffset = (uint*)&Sum[32];

    uint tid   = threadIdx.x;
    uint idx_Q = blockIdx.x / BSIZE;
    uint idx_q = blockIdx.x % BSIZE;
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
        LutOffset[i] = Lut[i].x * BSIZE*BSIZE;
    __syncthreads();

    // trim warps that we know are out of lut range
    if ((tid & (1024-32))*2*UNROLL < lut_size*BSIZE)
    {
        uint lut_idx = (tid & (1024 - BSIZE/2))*2*UNROLL/BSIZE;
        uint tidx    = (tid % (BSIZE/2))*2;
        uint offset  = idx_B*szBatch + idx_H*szHead + idx_q*BSIZE + tidx + LutOffset[lut_idx];
        DY += offset;
        Y  += offset;
        asm("mov.b64 %0, %0;" : "+l"(DY) : );
        asm("mov.b64 %0, %0;" : "+l"(Y)  : );

        V2 dy[UNROLL], y[UNROLL];
        #pragma unroll
        for (uint i = 0; i < UNROLL; i++)
        {
            ew_set(dy[i], 0);
            ew_set( y[i], 0);

            if (lut_idx + i < lut_size)
            {
                dy[i] = __ldg((const V2*)(DY + i*BSIZE*BSIZE));
                 y[i] = __ldg((const V2*)( Y + i*BSIZE*BSIZE));
            }
        }

        // compute dy * y and start reduction
        float sum_dyy = 0.0f;
        for (int i = 0; i < UNROLL; i++)
            sum_dyy += ew_sum(ew_mul(to_float(dy[i]), to_float(y[i])));

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
        DX += offset;
        //asm("mov.b64 %0, %0;" : "+l"(DX) : );

        #pragma unroll
        for (uint i = 0; i < UNROLL; i++)
        {
            // dx = (dy - sum_dyy) * y * scale
            float2 dx2 = ew_mul(ew_mul(ew_sub(to_float(dy[i]), sum_dyy), to_float(y[i])), scale);

            store((V2*)DX, dx2, i*BSIZE*BSIZE/2, lut_idx + i < lut_size);
            // asm (
            //     "{                             \n\t"
            //     ".reg .pred p;                 \n\t"
            //     ".reg .s64 DX, offset;         \n\t"
            //     "setp.lt.u32 p, %3, %4;        \n\t"
            //     "mov.b64 offset, {%1, 0};      \n\t"
            //     "add.s64 DX, %0, offset;       \n\t"
            //     "@p st.global.wb.u32 [DX], %2; \n\t"
            //     "}" :: "l"(DX), "r"(i*BSIZE*BSIZE*2), "r"(dx.x), "r"(lut_idx + i), "r"(lut_size));
        }
    }
}

#define LOG2e 1.4426950408889634f
typedef unsigned char uchar;


template <typename T, typename V>
bool BlocksparseMaskedSoftmax(CUstream stream,
    const uint2* lut,
    const  char* mask,
    const bhalf* x,
              T* y,
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

    // combine scaling with fast exp(x) compute
    scale *= LOG2e;

    dim3 grid(gridQ, batch_dim, head_dim);

    uint unroll, threads;
         if (maxK > 1024*16) { unroll = 16; threads = CEIL_DIV(maxK, 32*16*2) * 32; }
    else if (maxK > 1024* 8) { unroll =  8; threads = CEIL_DIV(maxK, 32* 8*2) * 32; }
    else                     { unroll =  4; threads = CEIL_DIV(maxK, 32* 4*2) * 32; }
    uint bshift    = block_size == 64 ? 5 : block_size == 32 ? 4 : block_size == 16 ? 3 : 2;
    uint shfl_init = THREAD_POW2(threads) / 64;
    uint lut_max   = (threads * unroll) >> bshift;
    uint shared    = lut_max * 8;

    if (block_size == 64)
    {
        shared = lut_max * 12;
            if (unroll == 16)
            bst_masked_softmax<16,1,64,T,V,uint64><<<grid,threads,shared,stream>>>(lut, (const uint64*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
        else if (unroll == 8)
            bst_masked_softmax< 8,2,64,T,V,uint64><<<grid,threads,shared,stream>>>(lut, (const uint64*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
        else // (unroll == 4)
            bst_masked_softmax< 4,2,64,T,V,uint64><<<grid,threads,shared,stream>>>(lut, (const uint64*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
    }
    else if (block_size == 32)
    {
            if (unroll == 16)
            bst_masked_softmax<16,1,32,T,V,  uint><<<grid,threads,shared,stream>>>(lut, (const   uint*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
        else if (unroll == 8)
            bst_masked_softmax< 8,2,32,T,V,  uint><<<grid,threads,shared,stream>>>(lut, (const   uint*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
        else // (unroll == 4)
            bst_masked_softmax< 4,2,32,T,V,  uint><<<grid,threads,shared,stream>>>(lut, (const   uint*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
    }
    else if (block_size == 16)
    {
            if (unroll == 16)
            bst_masked_softmax<16,1,16,T,V,ushort><<<grid,threads,shared,stream>>>(lut, (const ushort*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
        else if (unroll == 8)
            bst_masked_softmax< 8,2,16,T,V,ushort><<<grid,threads,shared,stream>>>(lut, (const ushort*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
        else // (unroll == 4)
            bst_masked_softmax< 4,2,16,T,V,ushort><<<grid,threads,shared,stream>>>(lut, (const ushort*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
    }
    else
    {
            if (unroll == 16)
            bst_masked_softmax<16,1, 8,T,V, uchar><<<grid,threads,shared,stream>>>(lut, (const  uchar*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
        else if (unroll == 8)
            bst_masked_softmax< 8,2, 8,T,V, uchar><<<grid,threads,shared,stream>>>(lut, (const  uchar*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
        else // (unroll == 4)
            bst_masked_softmax< 4,2, 8,T,V, uchar><<<grid,threads,shared,stream>>>(lut, (const  uchar*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
    }
    return true;
}
template bool BlocksparseMaskedSoftmax<ehalf,ehalf2>(CUstream stream, const uint2* lut, const  char* mask, const bhalf* x, ehalf* y, uint block_size, uint blocks, uint batch_dim, uint head_dim, uint ctx_blks, uint lut_heads, uint lut_dim, uint max_lut, uint mask_heads, float scale);
template bool BlocksparseMaskedSoftmax<bhalf,bhalf2>(CUstream stream, const uint2* lut, const  char* mask, const bhalf* x, bhalf* y, uint block_size, uint blocks, uint batch_dim, uint head_dim, uint ctx_blks, uint lut_heads, uint lut_dim, uint max_lut, uint mask_heads, float scale);



template <typename T, typename V>
bool BlocksparseMaskedSoftmaxGrad(CUstream stream,
    const uint2* lut,
    const     T* dy,
    const     T* y,
              T* dx,
    uint block_size, uint blocks,
    uint batch_dim,  uint head_dim, uint ctx_blks,
    uint lut_heads,  uint lut_dim,  uint max_lut,
    float scale)
{
    uint szLut   = lut_heads  > 1 ? lut_dim  : 0;
    uint gridQ   = ctx_blks * block_size;
    uint szHead  = blocks * block_size * block_size;
    uint szBatch = head_dim * szHead;
    uint maxK    = max_lut * block_size;
    //cuMemsetD16Async((CUdeviceptr)c, 0, szBatch*batch_dim, stream);

    dim3 grid(gridQ, batch_dim, head_dim);

    uint unroll, threads;
         if (maxK > 1024*16) { unroll = 16; threads = CEIL_DIV(maxK, 32*16*2) * 32; }
    else if (maxK > 1024* 8) { unroll =  8; threads = CEIL_DIV(maxK, 32* 8*2) * 32; }
    else                     { unroll =  4; threads = CEIL_DIV(maxK, 32* 4*2) * 32; }
    uint bshift    = block_size == 64 ? 5 : block_size == 32 ? 4 : block_size == 16 ? 3 : 2;
    uint shfl_init = THREAD_POW2(threads) / 64;
    uint lut_max   = (threads * unroll) >> bshift;
    uint shared    = lut_max * 4;

         if (unroll == 16)
    {
             if (block_size == 64)
            bst_masked_softmax_grad<16,1,64,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
        else if (block_size == 32)
            bst_masked_softmax_grad<16,1,32,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
        else if (block_size == 16)
            bst_masked_softmax_grad<16,1,16,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
        else
            bst_masked_softmax_grad<16,1, 8,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
    }
    else if (unroll == 8)
    {
             if (block_size == 64)
            bst_masked_softmax_grad< 8,2,64,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
        else if (block_size == 32)
            bst_masked_softmax_grad< 8,2,32,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
        else if (block_size == 16)
            bst_masked_softmax_grad< 8,2,16,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
        else
            bst_masked_softmax_grad< 8,2, 8,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
    }
    else // (unroll == 4)
    {
             if (block_size == 64)
            bst_masked_softmax_grad< 4,2,64,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
        else if (block_size == 32)
            bst_masked_softmax_grad< 4,2,32,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
        else if (block_size == 16)
            bst_masked_softmax_grad< 4,2,16,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
        else
            bst_masked_softmax_grad< 4,2, 8,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
    }
    return true;
}
template bool BlocksparseMaskedSoftmaxGrad<ehalf,ehalf2>(CUstream stream, const uint2* lut, const ehalf* dy, const ehalf* y, ehalf* dx, uint block_size, uint blocks, uint batch_dim, uint head_dim, uint ctx_blks, uint lut_heads, uint lut_dim, uint max_lut, float scale);
template bool BlocksparseMaskedSoftmaxGrad<bhalf,bhalf2>(CUstream stream, const uint2* lut, const bhalf* dy, const bhalf* y, bhalf* dx, uint block_size, uint blocks, uint batch_dim, uint head_dim, uint ctx_blks, uint lut_heads, uint lut_dim, uint max_lut, float scale);


template <int BSIZE, typename MASKT>
__global__ void __launch_bounds__(32) bst_partial_autoregressive_mask(
    const  int2* __restrict__ Lut,
    const MASKT* __restrict__ MaskI, MASKT* MaskO,
    uint blocks, uint szLut, int autoregress_at_k)
{
    uint tid = threadIdx.x;
    uint bid = blockIdx.x; // grid id (each cuda block being assigned to 32 mask blocks)
    uint qid = blockIdx.y; // q dim (row) within block
    uint hid = blockIdx.z; // head dim

    uint block = bid*32 + tid;

    if (block < blocks)
    {
        uint l = hid*szLut + block;
        uint m = hid*blocks*BSIZE + qid*blocks + block;

        int2  entry = Lut[l];
        MASKT  mask = MaskI[m];

        int K = entry.y*BSIZE; // entry.y: block index for keys
        int Q = entry.x*BSIZE; // entry.x: block index for queries
        int q = Q + qid;       // full query index

        // shift amount for the bidirectional to autoregressive transition
        int shift_a = BSIZE - min(max(autoregress_at_k - K, 0), BSIZE);

        // shift amount for the normal autoregressive property (lower triagular)
        int shift_b = min(max(BSIZE-1 + K - q, 0), BSIZE);

        // final shift is min value of these
        int shift_c = min(shift_a, shift_b);

        // apply the unsigned right shift to a pattern of ones to turn the mask off where needed
        // a shift of zero means the mask is unchanged
        // a shift of BSIZE means the mask is turned off for this row/block
        // somewhere in between means it's partially off.
        mask &= (MASKT)-1 >> shift_c;

        MaskO[m] = mask;
    }
}
bool BstPartialAutoregressiveMask(CUstream stream,
    const  int2* lut, const  char* maskI, char* maskO,
    uint block_size, uint blocks, uint lut_heads, uint lut_dim,  int autoregress_at_k)
{
    dim3 grid(CEIL_DIV(blocks,32), block_size, lut_heads);

         if (block_size == 64)
        bst_partial_autoregressive_mask<64,uint64><<<grid,32,0,stream>>>(lut, (const uint64*)maskI, (uint64*)maskO, blocks, lut_dim, autoregress_at_k);
    else if (block_size == 32)
        bst_partial_autoregressive_mask<32,  uint><<<grid,32,0,stream>>>(lut, (const   uint*)maskI, (  uint*)maskO, blocks, lut_dim, autoregress_at_k);
    else if (block_size == 16)
        bst_partial_autoregressive_mask<16,ushort><<<grid,32,0,stream>>>(lut, (const ushort*)maskI, (ushort*)maskO, blocks, lut_dim, autoregress_at_k);
    else
        bst_partial_autoregressive_mask< 8, uchar><<<grid,32,0,stream>>>(lut, (const  uchar*)maskI, ( uchar*)maskO, blocks, lut_dim, autoregress_at_k);

    return true;
}

#endif // GOOGLE_CUDA