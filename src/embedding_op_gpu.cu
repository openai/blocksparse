#if GOOGLE_CUDA

#include "ew_op_gpu.h"
#include <stdio.h>


template <typename TI, typename T>
__global__ void __launch_bounds__(1024) embedding_lookup(T* Y, const TI* __restrict__ I, const T* __restrict__ W, int C, uint K, int nIdx, uint sizeY)
{
    #pragma unroll 1
    for (uint idxY = blockIdx.x*1024 + threadIdx.x; idxY < sizeY; idxY += gridDim.x*1024)
    {
        uint idx = idxY / K;
        uint   k = idxY % K;

        if (idx < nIdx)
        {
            int emb = __ldg(add_ptr_u(I, idx));

            float w = load(add_ptr_u(W, emb*K + k), 0, emb >= 0 && emb < C);

            store(add_ptr_u(Y, idxY), w);
        }
    }
}

template <typename TI, typename TG>
__global__ void __launch_bounds__(1024) embedding_lookup_grad(float* DW, const TI* __restrict__ I, const TG* __restrict__ DY, int C, uint K, int nIdx, uint sizeY)
{
    #pragma unroll 1
    for (uint idxY = blockIdx.x*1024 + threadIdx.x; idxY < sizeY; idxY += gridDim.x*1024)
    {
        uint idx = idxY / K;
        uint   k = idxY % K;

        if (idx < nIdx)
        {
            int emb = __ldg(add_ptr_u(I, idx));

            if (emb >= 0 && emb < C)
            {
                float dy = load(add_ptr_u(DY, idxY));

                atomicRed(add_ptr_u(DW, emb*K + k), dy);
            }
        }
    }
}

__device__ __forceinline__ uint bfe(uint val, int pos)
{
    uint bit;
    asm ("bfe.u32 %0, %1, %2, 1;" : "=r"(bit) : "r"(val), "r"(pos)  );
    return bit;
}

typedef struct __align__(8) EmbMap
{
    int iIdx;
    int iEmb;
} EmbMap;

template <typename TI, typename TG, int UNROLL>
__global__ void sorted_embedding_lookup_grad(float* DW, const TI* __restrict__ I, const TG* __restrict__ DY, int nIdx, int C, int K, int Exp)
{
    extern __shared__ EmbMap emb_map[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    EmbMap init;
    init.iIdx = bid*blockDim.x + tid;
    if (init.iIdx < nIdx)
    {
        init.iEmb = __ldg(add_ptr_u(I, init.iIdx));
        if (init.iEmb < 0 || init.iEmb >= C)
            init.iEmb = -1;
    }
    else
        init.iEmb = -1;
    emb_map[tid] = init;

    __syncthreads();

    // Bittonic sort the embedding indicies to allow reduced atomic add contention.
    for (int i = 1; i <= Exp; ++i)
    {
        int j;
        #pragma unroll 1
        for (j = i - 1; j >= 5; --j)
        {
            // when the comparison stride is 32 or greater,
            // use half of warps and uniform shared memory access to make comparisons
            if (tid < blockDim.x/2)
            {
                // figure out the a and b indexes for the "butterfly" compare operation
                uint m = (tid >> j) << (j + 1);
                uint r =  tid & ((1 << j) - 1);
                uint a = m + r;
                uint b = a + (1 << j);
                bool d = bfe(a, i) != 0;

                EmbMap A = emb_map[a];
                EmbMap B = emb_map[b];

                if((B.iEmb > A.iEmb) ^ d)
                {
                    EmbMap t = A;
                    A = B;
                    B = t;
                }
                emb_map[a] = A;
                emb_map[b] = B;
            }
            __syncthreads();
        }

        // When the comparison stride is less than 32,
        // use all warps and shfl_xor operations to make comparisons in registers

        // Load shared to registers
        EmbMap A = emb_map[tid];

        #pragma unroll 5
        while (j >= 0)
        {
            EmbMap B;
            B.iEmb = shfl_xor(A.iEmb, 1 << j);
            B.iIdx = shfl_xor(A.iIdx, 1 << j);
            bool d = bfe(tid, i) != bfe(tid, j--);

            // in the case of equality we want both shuffle lanes to not swap
            if(((B.iEmb > A.iEmb) ^ d) && B.iEmb != A.iEmb)
                A = B;
        }
        // Load final register values back to shared.
        emb_map[tid] = A;

        __syncthreads();
    }
    int k = blockIdx.y*256 + (tid & 31);

    #pragma unroll 1
    for(int t = 0; t < 256 && k < K; t += UNROLL*32, k += UNROLL*32)
    {
        int iMap  = tid & -32;
        int iPrev = emb_map[iMap].iEmb;
        float dw[UNROLL] = {0};

        #pragma unroll 1
        for (int iTile = 0; iTile < 32 && iPrev != -1; ++iTile, ++iMap)
        {
            EmbMap curr = emb_map[iMap];

            // atomicRed gradient if we hit a new emb index
            if (curr.iEmb != iPrev)
            {
                float* DW_ = add_ptr_u(DW, iPrev*K + k);

                for (int i = 0; i < UNROLL; ++i)
                    atomicRed(DW_, dw[i], i*32, k + i*32 < K);

                for (int i = 0; i < UNROLL; ++i)
                    dw[i] = 0.0f;
            }
            // grab and accumulate this gradient if valid
            if (curr.iEmb != -1)
            {
                const TG* DY_ = add_ptr_u(DY, curr.iIdx*K + k);

                for (int i = 0; i < UNROLL; ++i)
                    dw[i] += load(DY_, i*32, k + i*32 < K);
            }
            iPrev = curr.iEmb;
        }
        // Final atomicRed in case tile size was full 32
        if (iPrev != -1)
        {
            float* DW_ = add_ptr_u(DW, iPrev*K + k);

            for (int i = 0; i < UNROLL; ++i)
                atomicRed(DW_, dw[i], i*32, k + i*32 < K);
        }
    }
}

template <typename TI, typename T>
bool EmbeddingLookup(CUstream stream, int SMs, T* y, const TI* idx, const T* w, int nIdx, int C, int K)
{
    uint sizeY = nIdx*K;
    uint grid  = sizeY > SMs*1024 ? SMs*2 : SMs;
    embedding_lookup<TI,T><<<grid,1024,0,stream>>>(y, idx, w, C, K, nIdx, sizeY);
    return true;
}

template bool EmbeddingLookup<int,float>(CUstream stream, int SMs, float* y, const int* idx, const float* w, int nIdx, int C, int K);
template bool EmbeddingLookup<int,ehalf>(CUstream stream, int SMs, ehalf* y, const int* idx, const ehalf* w, int nIdx, int C, int K);
template bool EmbeddingLookup<int,bhalf>(CUstream stream, int SMs, bhalf* y, const int* idx, const bhalf* w, int nIdx, int C, int K);

template bool EmbeddingLookup<ushort,float>(CUstream stream, int SMs, float* y, const ushort* idx, const float* w, int nIdx, int C, int K);
template bool EmbeddingLookup<ushort,ehalf>(CUstream stream, int SMs, ehalf* y, const ushort* idx, const ehalf* w, int nIdx, int C, int K);
template bool EmbeddingLookup<ushort,bhalf>(CUstream stream, int SMs, bhalf* y, const ushort* idx, const bhalf* w, int nIdx, int C, int K);

template bool EmbeddingLookup<unsigned char,float>(CUstream stream, int SMs, float* y, const unsigned char* idx, const float* w, int nIdx, int C, int K);
template bool EmbeddingLookup<unsigned char,ehalf>(CUstream stream, int SMs, ehalf* y, const unsigned char* idx, const ehalf* w, int nIdx, int C, int K);
template bool EmbeddingLookup<unsigned char,bhalf>(CUstream stream, int SMs, bhalf* y, const unsigned char* idx, const bhalf* w, int nIdx, int C, int K);


template <typename TI, typename TG>
bool EmbeddingLookupGrad(CUstream stream, int SMs, float* dw, const TI* idx, const TG* dy, int nIdx, int C, int K, bool sorted)
{
    cuMemsetD32Async((CUdeviceptr)dw, 0, C*K, stream);

    if (sorted)
    {
        int exp;
             if (nIdx > (SMs << 11)) exp = 10;
        else if (nIdx > (SMs << 10)) exp =  9;
        else if (nIdx > (SMs <<  9)) exp =  8;
        else if (nIdx > (SMs <<  8)) exp =  7;
        else                         exp =  6;
        int threads = 1 << exp;
        int shared  = threads * 8;
        int gridI   = (nIdx >> exp) + ((nIdx & (threads-1)) != 0);
        int gridK   = CEIL_DIV(K, 256);
        dim3 grid(gridI, gridK);
        if (K > 64)
            sorted_embedding_lookup_grad<TI,TG,4><<<grid,threads,shared,stream>>>(dw, idx, dy, nIdx, C, K, exp);
        else if (K > 32)
            sorted_embedding_lookup_grad<TI,TG,2><<<grid,threads,shared,stream>>>(dw, idx, dy, nIdx, C, K, exp);
        else
            sorted_embedding_lookup_grad<TI,TG,1><<<grid,threads,shared,stream>>>(dw, idx, dy, nIdx, C, K, exp);
    }
    else
    {
        uint sizeY = nIdx*K;
        uint grid  = sizeY > SMs*1024 ? SMs*2 : SMs;
        embedding_lookup_grad<TI,TG><<<grid,1024,0,stream>>>(dw, idx, dy, C, K, nIdx, sizeY);
    }
    return true;
}

template bool EmbeddingLookupGrad<int,float>(CUstream stream, int SMs, float* dw, const int* idx, const float* dy, int nIdx, int C, int K, bool sorted);
template bool EmbeddingLookupGrad<int,ehalf>(CUstream stream, int SMs, float* dw, const int* idx, const ehalf* dy, int nIdx, int C, int K, bool sorted);
template bool EmbeddingLookupGrad<int,bhalf>(CUstream stream, int SMs, float* dw, const int* idx, const bhalf* dy, int nIdx, int C, int K, bool sorted);

template bool EmbeddingLookupGrad<ushort,float>(CUstream stream, int SMs, float* dw, const ushort* idx, const float* dy, int nIdx, int C, int K, bool sorted);
template bool EmbeddingLookupGrad<ushort,ehalf>(CUstream stream, int SMs, float* dw, const ushort* idx, const ehalf* dy, int nIdx, int C, int K, bool sorted);
template bool EmbeddingLookupGrad<ushort,bhalf>(CUstream stream, int SMs, float* dw, const ushort* idx, const bhalf* dy, int nIdx, int C, int K, bool sorted);

template bool EmbeddingLookupGrad<unsigned char,float>(CUstream stream, int SMs, float* dw, const unsigned char* idx, const float* dy, int nIdx, int C, int K, bool sorted);
template bool EmbeddingLookupGrad<unsigned char,ehalf>(CUstream stream, int SMs, float* dw, const unsigned char* idx, const ehalf* dy, int nIdx, int C, int K, bool sorted);
template bool EmbeddingLookupGrad<unsigned char,bhalf>(CUstream stream, int SMs, float* dw, const unsigned char* idx, const bhalf* dy, int nIdx, int C, int K, bool sorted);


#endif
