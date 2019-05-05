#if GOOGLE_CUDA

#include "ew_op_gpu.h"
#include <stdio.h>

__device__ __forceinline__ uint bfe(uint val, int pos)
{
    uint bit;
    asm ("bfe.u32 %0, %1, %2, 1;" : "=r"(bit) : "r"(val), "r"(pos)  );
    return bit;
}

typedef struct __align__(8) KeyVal
{
    uint  key;
    float val;
} KeyVal;

template <typename T>
__global__ void top_k(T* Y, uint* A, const T* __restrict__ X, uint Exp, uint topK, uint K, uint rect, uint rebase)
{
    extern __shared__ KeyVal data[];

    uint tid = threadIdx.x;
    uint   n = blockIdx.x;

    uint offset = n*K + tid;

    KeyVal init;
    init.key  = tid;
    init.val  = tid < K ? load(add_ptr_u(X, offset)) : -FLT_MAX;
    data[tid] = init;

    __syncthreads();

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

                KeyVal A = data[a];
                KeyVal B = data[b];

                if((B.val > A.val) ^ d)
                {
                    KeyVal t = A;
                    A = B;
                    B = t;
                }
                data[a] = A;
                data[b] = B;
            }
            __syncthreads();
        }

        // When the comparison stride is less than 32,
        // use all warps and shfl_xor operations to make comparisons in registers

        // Load shared to registers
        KeyVal A = data[tid];

        #pragma unroll 5
        while (j >= 0)
        {
            KeyVal B;
            B.val = shfl_xor(A.val, 1 << j);
            B.key = shfl_xor(A.key, 1 << j);
            bool d = bfe(tid, i) != bfe(tid, j--);

            // in the case of equality we want both shuffle lanes to not swap
            if(((B.val > A.val) ^ d) && B.val != A.val)
                A = B;
        }
        // Load final register values back to shared.
        data[tid] = A;

        __syncthreads();
    }
    if (rect)
    {
        // avoid extra __syncthreads by coalescing to unused shared
        float* coalesce = (float*)&data[blockDim.x];

        // Output same size as input, with zeros for non-topK values.
        // rebase sets the zero line to the min value of the topK

        KeyVal out = data[tid];
        float base = rebase ? fmaxf(data[topK-1].val, 0.0f) : 0.0f;
        float val  = tid < topK ? out.val : 0.0f;

        //if (tid == 0 && n == 0)
        //    printf("base: %f %d\n", base, data[topK-1].key);

        // apply the rectification and coalesce the output
        coalesce[out.key] = fmaxf(val, base) - base;

        __syncthreads();

        if (tid < K)
            store(add_ptr_u(Y, offset), coalesce[tid]);
    }
    else
    {
        // output just top values and their indicies.
        if (tid < topK)
        {
            KeyVal out = data[tid];
            offset = n*topK + tid;
            store(add_ptr_u(Y, offset), out.val);
            __stg(add_ptr_u(A, offset), out.key);
        }
    }
}

template <typename T>
bool TopK(CUstream stream, T* y, uint* a, const T* x, uint topK, uint N, uint K, uint rebase)
{
    uint exp;
         if (K > 512) exp = 10;
    else if (K > 256) exp =  9;
    else if (K > 128) exp =  8;
    else if (K >  64) exp =  7;
    else if (K >  32) exp =  6;
    else              exp =  5;
    uint threads = 1 << exp;
    uint shared  = threads * 16;

    top_k<T><<<N,threads,shared,stream>>>(y, a, x, exp, topK, K, a == NULL, rebase);
    return true;
}

template bool TopK<float>(CUstream stream, float* y, uint* a, const float* x, uint topK, uint N, uint K, uint rebase);
template bool TopK<ehalf>(CUstream stream, ehalf* y, uint* a, const ehalf* x, uint topK, uint N, uint K, uint rebase);
template bool TopK<bhalf>(CUstream stream, bhalf* y, uint* a, const bhalf* x, uint topK, uint N, uint K, uint rebase);


template <typename T>
__global__ void masked_top_k_softmax(T* Y, const float* __restrict__ M, const T* __restrict__ X, uint Exp, uint topK, uint D123, uint D23, uint D3, uint M1, uint M2, uint use_mask, float scale)
{
    extern __shared__ KeyVal block[];
    extern __shared__ float  stage[];

    // x: D0, D1, D2, D3
    // m:  1, D1, D2, D3
    // m:  1,  1, D2, D3
    // m:  1,  1,  1, D3
    uint tid = threadIdx.x;
    uint  d0 = blockIdx.x;
    uint  d1 = blockIdx.y;
    uint  d2 = blockIdx.z;

    uint offsetX = d0*D123 + d1*D23 + d2*D3 + tid;
    uint offsetM =           d1*M1  + d2*M2 + tid;

    M = add_ptr_u(M, offsetM);
    X = add_ptr_u(X, offsetX);

    float mask = tid < D3 ? (use_mask ? __ldg(M) : 1.0f) : 0.0f;
    float xval = mask != 0.0 ? load(X) * mask * scale : -FLT_MAX;

    KeyVal init;
    init.key   = tid;
    init.val   = xval;
    block[tid] = init;

    __syncthreads();

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

                KeyVal A = block[a];
                KeyVal B = block[b];

                if((B.val > A.val) ^ d)
                {
                    KeyVal t = A;
                    A = B;
                    B = t;
                }
                block[a] = A;
                block[b] = B;
            }
            __syncthreads();
        }

        // When the comparison stride is less than 32,
        // use all warps and shfl_xor operations to make comparisons in registers

        // Load shared to registers
        KeyVal A = block[tid];

        #pragma unroll 5
        while (j >= 0)
        {
            KeyVal B;
            B.val = shfl_xor(A.val, 1 << j);
            B.key = shfl_xor(A.key, 1 << j);
            bool d = bfe(tid, i) != bfe(tid, j--);

            // in the case of equality we want both shuffle lanes to not swap
            if(((B.val > A.val) ^ d) && B.val != A.val)
                A = B;
        }
        // Load final register values back to shared.
        block[tid] = A;

        __syncthreads();
    }

    float* vals = &stage[blockDim.x*2];
    float* reds =  &vals[blockDim.x];

    KeyVal out = block[tid];
    float  val = 0.0f;
    if (tid < topK)
        val = expf(out.val - block[0].val);

    vals[out.key] = val;

    // reduce within warp
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1)
        val += shfl_xor(val, i);

    // first thread of each warp store to shared
    if ((tid & 31) == 0)
        reds[tid/32] = val;

    __syncthreads();

    if (tid < blockDim.x/32)
    {
        // first warp loads all prior reductions
        val = reds[tid];
        // reduce within this last warp
        #pragma unroll 1
        for (int i = blockDim.x/64; i > 0; i >>= 1)
            val += shfl_xor(val, i);

        // rcp final reduction to shared
        reds[tid] = 1.0f / val;
    }
    __syncthreads();

    if (tid < D3)
        store(add_ptr_u(Y, offsetX), vals[tid] * reds[0]);
}

template <typename T>
bool MaskedTopKSoftmax(CUstream stream, T* y, const float* m, const T* x, uint topK, uint D0, uint D1, uint D2, uint D3, uint M1, uint M2, float scale)
{
    uint exp;
         if (D3 > 512) exp = 10;
    else if (D3 > 256) exp =  9;
    else if (D3 > 128) exp =  8;
    else if (D3 >  64) exp =  7;
    else if (D3 >  32) exp =  6;
    else               exp =  5;
    uint threads = 1 << exp;
    uint shared  = threads * 16;

    masked_top_k_softmax<T><<<dim3(D0,D1,D2),threads,shared,stream>>>(y, m, x, exp, topK, D1*D2*D3, D2*D3, D3, M1, M2, m != NULL, scale);
    return true;
}

template bool MaskedTopKSoftmax<float>(CUstream stream, float* y, const float* m, const float* x, uint topK, uint D0, uint D1, uint D2, uint D3, uint M1, uint M2, float scale);
template bool MaskedTopKSoftmax<ehalf>(CUstream stream, ehalf* y, const float* m, const ehalf* x, uint topK, uint D0, uint D1, uint D2, uint D3, uint M1, uint M2, float scale);
template bool MaskedTopKSoftmax<bhalf>(CUstream stream, bhalf* y, const float* m, const bhalf* x, uint topK, uint D0, uint D1, uint D2, uint D3, uint M1, uint M2, float scale);

// x *= mask * scale
// y  = exp(x - max(x)) / sum( exp(x - max(x)) )
template <typename T, int U>
__global__ void masked_softmax(
              T*              Y,
    const     T* __restrict__ X,
    const float* __restrict__ M,
    uint D123, uint D23, uint D3, uint M1, uint M2, uint use_mask, float scale, int threads_pow2)
{
    __shared__ float Max[32];
    __shared__ float Sum[32];
    // x: D0, D1, D2, D3
    // m:  1, D1, D2, D3
    // m:  1,  1, D2, D3
    // m:  1,  1,  1, D3
    uint tid = threadIdx.x;
    uint  d0 = blockIdx.x;
    uint  d1 = blockIdx.y;
    uint  d2 = blockIdx.z;

    if (blockDim.x > 32)
    {
        if (tid < 32)
        {
            // Allows non-power of 2 threads to work
            Max[tid] = -FLT_MAX;
            Sum[tid] = 0.0f;
        }
        __syncthreads();
    }

    uint ti = (tid & 0x3fe0)*U + (tid & 31); // 0x3fe0 = -32
    uint offsetX = d0*D123 + d1*D23 + d2*D3 + ti;
    uint offsetM =           d1*M1  + d2*M2 + ti;

    M = add_ptr_u(M, offsetM);
    X = add_ptr_u(X, offsetX);

    // Load mask
    float mask[U]; for (int i = 0; i < U; i++) mask[i]= 1.0f;
    if (use_mask)
    {
        for (int i = 0; i < U; i++)
        {
            mask[i] = 0.0f;
            if (ti + i*32 < D3)
                mask[i] = __ldg(M + i*32);
        }
    }
    // Load X
    float xval[U]; for (int i = 0; i < U; i++) xval[i] = -FLT_MAX;
    for (int i = 0; i < U; i++)
        if (mask[i] != 0.0 && ti + i*32 < D3)
            xval[i] = load(X, i*32) * mask[i] * scale;

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
            for (int i = threads_pow2/64; i > 0; i >>= 1)
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
        Xsum[i] = xval[i] = expf(xval[i] - xmax);

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
            for (int i = threads_pow2/64; i > 0; i >>= 1)
                exp_sum += shfl_xor(exp_sum, i);
            // final reduction to shared
            Sum[tid] = exp_sum;
        }
        __syncthreads();
        exp_sum = Sum[0];
    }
    float rcp_exp_sum = 1.0f / exp_sum;

    Y = add_ptr_u(Y, offsetX);

    for (int i = 0; i < U; i++)
        store(Y, xval[i] * rcp_exp_sum, i*32, ti + i*32 < D3);
}

// x *= mask * scale
// y  = exp(x - max(x)) / sum( exp(x - max(x)) )
template <typename T>
__global__ void __launch_bounds__(32) masked_softmax2(
              T*              Y,
    const     T* __restrict__ X,
    const float* __restrict__ M,
    uint D123, uint D23, uint D3, uint M1, uint M2, uint use_mask, float scale)
{
    // x: D0, D1, D2, D3
    // m:  1, D1, D2, D3
    // m:  1,  1, D2, D3
    // m:  1,  1,  1, D3
    uint tid = threadIdx.x;
    uint  d0 = blockIdx.x;
    uint  d1 = blockIdx.y;
    uint  d2 = blockIdx.z;

    uint offsetX = d0*D123 + d1*D23 + d2*D3 + tid;
    uint offsetM =           d1*M1  + d2*M2 + tid;

    // max(x, axis-1)
    float max_x = -FLT_MAX;
    #pragma unroll 2
    for (uint d3 = tid, xi = offsetX, mi = offsetM; d3 < D3; d3 += 32, xi += 32, mi += 32)
    {
        float m = use_mask ? __ldg(add_ptr_u(M, mi)) : 1.0f;
        float x = m != 0.0 ? load(add_ptr_u(X, xi)) * m * scale : -FLT_MAX;

        max_x = fmaxf(max_x, x);
    }
    for (int i = 16; i > 0; i >>= 1)
        max_x = fmaxf(max_x, shfl_xor(max_x, i));

    float exp_sum = 0.0f;
    #pragma unroll 2
    for (uint d3 = tid, xi = offsetX, mi = offsetM; d3 < D3; d3 += 32, xi += 32, mi += 32)
    {
        float m = use_mask ? __ldg(add_ptr_u(M, mi)) : 1.0f;
        float x = m != 0.0 ? load(add_ptr_u(X, xi)) * m * scale : -FLT_MAX;

        exp_sum += expf(x - max_x);
    }
    for (int i = 16; i > 0; i >>= 1)
        exp_sum += shfl_xor(exp_sum, i);

    float rcp_exp_sum = 1.0f / exp_sum;

    #pragma unroll 2
    for (uint d3 = tid, xi = offsetX, mi = offsetM; d3 < D3; d3 += 32, xi += 32, mi += 32)
    {
        float m = use_mask ? __ldg(add_ptr_u(M, mi)) : 1.0f;
        float x = m != 0.0 ? load(add_ptr_u(X, xi)) * m * scale : -FLT_MAX;

        float y = expf(x - max_x)  * rcp_exp_sum;

        store(add_ptr_u(Y, xi), y);
    }
}

// dx = (dy - sum(dy * y, axis=-1)) * y * m * scale
template <typename T, int U>
__global__ void masked_softmax_grad(
              T*              DX,
    const     T* __restrict__ DY,
    const     T* __restrict__ Y,
    const float* __restrict__ M,
    uint D123, uint D23, uint D3, uint M1, uint M2, uint use_mask, float scale, int threads_pow2)
{
    __shared__ float Sum[32];
    // x: D0, D1, D2, D3
    // m:  1, D1, D2, D3
    // m:  1,  1, D2, D3
    // m:  1,  1,  1, D3
    uint tid = threadIdx.x;
    uint  d0 = blockIdx.x;
    uint  d1 = blockIdx.y;
    uint  d2 = blockIdx.z;

    if (blockDim.x > 32)
    {
        // Allows non-power of 2 threads to work
        if (tid < 32)
            Sum[tid] = 0.0f;
        __syncthreads();
    }

    uint ti = (tid & 0x3fe0)*U + (tid & 31);
    uint offsetY = d0*D123 + d1*D23 + d2*D3 + ti;
    uint offsetM =           d1*M1  + d2*M2 + ti;

    DY = add_ptr_u(DY, offsetY);
    Y  = add_ptr_u( Y, offsetY);
    M  = add_ptr_u( M, offsetM);

    // Load mask
    float mask[U]; for (int i = 0; i < U; i++) mask[i]= 1.0f;
    if (use_mask)
    {
        for (int i = 0; i < U; i++)
        {
            mask[i] = 0.0f;
            if (ti + i*32 < D3)
                mask[i] = __ldg(M + i*32);
        }
    }
    // Load DY
    float dy[U]; for (int i = 0; i < U; i++) dy[i]= 0.0f;
    for (int i = 0; i < U; i++)
        if (mask[i] != 0.0 && ti + i*32 < D3)
            dy[i] = load(DY, i*32);

    // Load Y
    float y[U]; for (int i = 0; i < U; i++) y[i]= 0.0f;
    for (int i = 0; i < U; i++)
        if (mask[i] != 0.0 && ti + i*32 < D3)
            y[i] = load(Y, i*32);

    // compute dy * y and y * mask * scale
    float dyy[U];
    for (int i = 0; i < U; i++)
    {
        dyy[i] = dy[i] * y[i];
        y[i]  *= mask[i] * scale;
    }

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
            for (int i = threads_pow2/64; i > 0; i >>= 1)
                sum_dyy += shfl_xor(sum_dyy, i);
            // final reduction to shared
            Sum[tid] = sum_dyy;
        }
        __syncthreads();
        sum_dyy = Sum[0];
    }
    // dx = (dy - sum_dyy) * y * mask* scale
    DX = add_ptr_u(DX, offsetY);
    for (int i = 0; i < U; i++)
        store(DX, (dy[i] - sum_dyy) * y[i], i*32, ti + i*32 < D3);
}

// dx = (dy - sum(dy * y, axis=-1)) * y * m * scale
template <typename T>
__global__ void __launch_bounds__(32) masked_softmax_grad2(
              T*              DX,
    const     T* __restrict__ DY,
    const     T* __restrict__ Y,
    const float* __restrict__ M,
    uint D123, uint D23, uint D3, uint M1, uint M2, uint use_mask, float scale)
{
    // x: D0, D1, D2, D3
    // m:  1, D1, D2, D3
    // m:  1,  1, D2, D3
    // m:  1,  1,  1, D3
    uint tid = threadIdx.x;
    uint  d0 = blockIdx.x;
    uint  d1 = blockIdx.y;
    uint  d2 = blockIdx.z;

    uint offsetY = d0*D123 + d1*D23 + d2*D3 + tid;
    uint offsetM =           d1*M1  + d2*M2 + tid;

    // sum(dy * y, axis=-1))
    float sum_dy_y = 0.0f;
    #pragma unroll 2
    for (uint d3 = tid, offset = offsetY; d3 < D3; d3 += 32, offset += 32)
    {
        float dy = load(add_ptr_u(DY, offset));
        float  y = load(add_ptr_u(Y,  offset));
        sum_dy_y += dy * y;
    }
    for (int i = 16; i > 0; i >>= 1)
        sum_dy_y += shfl_xor(sum_dy_y, i);

    #pragma unroll 2
    for (uint d3 = tid; d3 < D3; d3 += 32, offsetY += 32, offsetM += 32)
    {
        float dy = load(add_ptr_u(DY, offsetY));
        float  y = load(add_ptr_u(Y,  offsetY));
        float  m = use_mask ? __ldg(add_ptr_u(M,  offsetM)) : 1.0f;

        float dx = (dy - sum_dy_y) * y * m * scale;

        store(add_ptr_u(DX, offsetY), dx);
    }
}

template <typename T>
bool MaskedSoftmax(CUstream stream, T* y, const T* x, const float* m, uint D0, uint D1, uint D2, uint D3, uint M1, uint M2, float scale)
{
    if (D3 > 1024*8)
        masked_softmax2<T><<<dim3(D0,D1,D2),32,0,stream>>>(y, x, m, D1*D2*D3, D2*D3, D3, M1, M2, m != NULL, scale);
    else
    {
        if (D3 > 32*4)
        {
            uint threads = CEIL_DIV(D3, 32*8) * 32;
            int thread2  = THREAD_POW2(threads);
            masked_softmax<T,8><<<dim3(D0,D1,D2),threads,0,stream>>>(y, x, m, D1*D2*D3, D2*D3, D3, M1, M2, m != NULL, scale, thread2);
        }
        else if (D3 > 32*2)
            masked_softmax<T,4><<<dim3(D0,D1,D2),32,0,stream>>>(y, x, m, D1*D2*D3, D2*D3, D3, M1, M2, m != NULL, scale,32);
        else if (D3 > 32*1)
            masked_softmax<T,2><<<dim3(D0,D1,D2),32,0,stream>>>(y, x, m, D1*D2*D3, D2*D3, D3, M1, M2, m != NULL, scale,32);
        else
            masked_softmax<T,1><<<dim3(D0,D1,D2),32,0,stream>>>(y, x, m, D1*D2*D3, D2*D3, D3, M1, M2, m != NULL, scale,32);
    }
    return true;
}

template <typename T>
bool MaskedSoftmaxGrad(CUstream stream, T* dx, const T* dy, const T* y, const float* m, uint D0, uint D1, uint D2, uint D3, uint M1, uint M2, float scale)
{
    if (D3 > 1024*4)
        masked_softmax_grad2<T><<<dim3(D0,D1,D2),32,0,stream>>>(dx, dy, y, m, D1*D2*D3, D2*D3, D3, M1, M2, m != NULL, scale);
    else
    {
        if (D3 > 32*2)
        {
            uint threads = CEIL_DIV(D3, 32*4) * 32;
            int thread2  = THREAD_POW2(threads);
            masked_softmax_grad<T,4><<<dim3(D0,D1,D2),threads,0,stream>>>(dx, dy, y, m, D1*D2*D3, D2*D3, D3, M1, M2, m != NULL, scale, thread2);
        }
        else if (D3 > 32*1)
            masked_softmax_grad<T,2><<<dim3(D0,D1,D2),32,0,stream>>>(dx, dy, y, m, D1*D2*D3, D2*D3, D3, M1, M2, m != NULL, scale,32);
        else
            masked_softmax_grad<T,1><<<dim3(D0,D1,D2),32,0,stream>>>(dx, dy, y, m, D1*D2*D3, D2*D3, D3, M1, M2, m != NULL, scale,32);
    }
    return true;
}

template bool MaskedSoftmax<float>(CUstream stream, float* y, const float* x, const float* m, uint D0, uint D1, uint D2, uint D3, uint M1, uint M2, float scale);
template bool MaskedSoftmax<ehalf>(CUstream stream, ehalf* y, const ehalf* x, const float* m, uint D0, uint D1, uint D2, uint D3, uint M1, uint M2, float scale);
template bool MaskedSoftmax<bhalf>(CUstream stream, bhalf* y, const bhalf* x, const float* m, uint D0, uint D1, uint D2, uint D3, uint M1, uint M2, float scale);

template bool MaskedSoftmaxGrad<float>(CUstream stream, float* dx, const float* dy, const float* y, const float* m, uint D0, uint D1, uint D2, uint D3, uint M1, uint M2, float scale);
template bool MaskedSoftmaxGrad<ehalf>(CUstream stream, ehalf* dx, const ehalf* dy, const ehalf* y, const float* m, uint D0, uint D1, uint D2, uint D3, uint M1, uint M2, float scale);
template bool MaskedSoftmaxGrad<bhalf>(CUstream stream, bhalf* dx, const bhalf* dy, const bhalf* y, const float* m, uint D0, uint D1, uint D2, uint D3, uint M1, uint M2, float scale);

template <typename T, typename V4>
__global__ void __launch_bounds__(256) transpose_64x64_2D(T* B, const T* A, uint dimAY, uint dimAX)
{
    __shared__ T Share[16][64*4 + 4];

    uint tx = threadIdx.x;
    uint ty = threadIdx.y;
    uint bx = blockIdx.x;
    uint by = blockIdx.y;
    uint x  = bx*64 + tx*4;
    uint y  = by*64 + ty*4;

    uint offsetA = y*dimAX + x;

    // Load 4x4 block from global and store to shared
    // Dims need to be even multiples of 4
    if (x < dimAX && y < dimAY)
        for (uint i = 0; i < 4; ++i)
            *(V4*)&Share[ty][i*64 + tx*4] = __ldg((const V4*)(A + (offsetA + i*dimAX)));

    __syncthreads();

    // Swap Dims
    uint dimBX = dimAY;
    uint dimBY = dimAX;
    // block ids hold their dimension affinity but threads get remapped though shared memory
    // We want global writes to be thread contiguous.
    x = by*64 + tx*4;
    y = bx*64 + ty*4;

    if (x < dimBX && y < dimBY)
    {
        // Load 4x4 block from shared in transposed order (tx <=> ty)
        V4 vals[4];
        for (uint i = 0; i < 4; ++i)
            vals[i] = *(V4*)&Share[tx][i*64 + ty*4];

        // Transpose within 4x4 block in registers
        V4 trans[4];
        for (uint i = 0; i < 4; ++i)
            for (uint j = 0; j < 4; ++j)
                ((T*)&trans[j])[i] = ((T*)&vals[i])[j];

        uint offsetB = y*dimBX + x;

        // Store transposed 4x4 block to global
        for (uint i = 0; i < 4; ++i)
            __stg((V4*)(B + (offsetB + i*dimBX)), trans[i]);
    }
}
template <typename T, typename V4>
bool Transpose_2D(CUstream stream, T* y, const T* x, uint D0, uint D1)
{
    uint gridX = CEIL_DIV(D1, 64);
    uint gridY = CEIL_DIV(D0, 64);

    transpose_64x64_2D<T,V4><<<dim3(gridX,gridY,1),dim3(16,16,1),0,stream>>>(y, x, D0, D1);
    return true;
}
template bool Transpose_2D<float,float4>(CUstream stream, float* y, const float* x, uint D0, uint D1);
template bool Transpose_2D<ehalf,ehalf4>(CUstream stream, ehalf* y, const ehalf* x, uint D0, uint D1);
template bool Transpose_2D<bhalf,bhalf4>(CUstream stream, bhalf* y, const bhalf* x, uint D0, uint D1);


// split_heads: (batch, pixel, head, state) -> (batch, head, pixel, state)
// merge_heads: (batch, head, pixel, state) -> (batch, pixel, head, state)
template <typename T, uint U>
__global__ void __launch_bounds__(32) transpose_0213(T* Y, const T* X, uint D123, uint D23, uint D13, uint D2, uint D3)
{
    uint  tid = threadIdx.x;
    uint  d2  = blockIdx.x;
    uint  d1  = blockIdx.y;
    uint  d0  = blockIdx.z;

    uint offset  = d0*D123 + tid;
    uint offsetX = d1*D23 + d2*D3 + offset;
    uint offsetY = d2*D13 + d1*D3 + offset;

    #pragma unroll 1
    while (d2 < D2)
    {
        #pragma unroll 1
        for (uint d3 = tid, xi = offsetX, yi = offsetY; d3 < D3; d3 += U*32, xi += U*32, yi += U*32)
        {
            const T* Xi = add_ptr_u(X, xi);
                  T* Yi = add_ptr_u(Y, yi);

            float x[U];
            for (uint i = 0; i < U; i++)
                x[i] = load(Xi, i*32, d3 + i*32 < D3);

            for (uint i = 0; i < U; i++)
                store(Yi, x[i], i*32, d3 + i*32 < D3);
        }
        offsetX += gridDim.x*D3;
        offsetY += gridDim.x*D13;
        d2      += gridDim.x;
    }
}
template <typename T>
bool Transpose_0213(CUstream stream, T* y, const T* x, uint D0, uint D1, uint D2, uint D3)
{
    // make sure each block has enough work to cover launch overhead
    uint gridX = CEIL_DIV(D2, 4);

    if (D3 <= 64)
        transpose_0213<T,2><<<dim3(gridX,D1,D0),32,0,stream>>>(y, x, D1*D2*D3, D2*D3, D1*D3, D2, D3);
    else
        transpose_0213<T,4><<<dim3(gridX,D1,D0),32,0,stream>>>(y, x, D1*D2*D3, D2*D3, D1*D3, D2, D3);
    return true;
}
template bool Transpose_0213<float>(CUstream stream, float* y, const float* x, uint D0, uint D1, uint D2, uint D3);
template bool Transpose_0213<ehalf>(CUstream stream, ehalf* y, const ehalf* x, uint D0, uint D1, uint D2, uint D3);
template bool Transpose_0213<bhalf>(CUstream stream, bhalf* y, const bhalf* x, uint D0, uint D1, uint D2, uint D3);


__device__ __forceinline__ void xor1(ehalf8 &v)
{
    asm ("xor.b32 %0, %0, 0x00000001;" : "+r"(v.x) : );
    asm ("xor.b32 %0, %0, 0x00000001;" : "+r"(v.y) : );
    asm ("xor.b32 %0, %0, 0x00000001;" : "+r"(v.z) : );
    asm ("xor.b32 %0, %0, 0x00000001;" : "+r"(v.w) : );
}
__device__ __forceinline__ void xor1(ehalf2 &v)
{
    asm ("xor.b32 %0, %0, 0x00000001;" : "+r"(v.x) : );
}

template <uint UNROLL, typename T, typename V, typename TL, uint BLOCKS>
__global__ void __launch_bounds__(1024,BLOCKS) softmax_cross_entropy(
    ehalf* Grad,
    float* Loss,
    const ehalf*  __restrict__ Logits,
    const    TL*  __restrict__ Labels,
    uint K)
{
    __shared__ float Max[32];
    __shared__ float Sum[32];

    uint tid   = threadIdx.x; // feature dim
    uint idx_N = blockIdx.x;  // batch dim

    const uint VSIZE = sizeof(V)/4;

    uint k = ((tid & (1024-32))*UNROLL + (tid & 31))*VSIZE;
    uint offset = idx_N*K + k;
    Logits += offset;
    asm("mov.b64 %0, %0;" : "+l"(Logits) : );

    T logits[UNROLL];
    #pragma unroll
    for (uint i = 0; i < UNROLL; i++)
    {
        ew_set(logits[i], 0xfc00fc00); //-inf, -inf

        if (k + i*VSIZE*32 < K)
            logits[i] = __ldg((const T*)(Logits + i*VSIZE*32));
    }

    // reduce within thread
    float Xmax[UNROLL];
    for (int i = 0; i < UNROLL; i++)
        Xmax[i] = ew_max(to_float(logits[i]));

    float xmax = Xmax[0];
    for (int i = 1; i < UNROLL; i++)
        xmax = fmaxf(Xmax[i], xmax);

    // reduce within warp
    for (int i = 16; i > 0; i >>= 1)
        xmax = fmaxf(xmax, shfl_xor(xmax, i));

    if (blockDim.x > 32)
    {
        if (blockDim.x != 1024)
        {
            if (tid < 32)
            {
                Max[tid] = -FLT_MAX;
                Sum[tid] = 0.0f;
            }
            __syncthreads();
        }
        if ((tid & 31) == 0)
            Max[tid / 32] = xmax;
        __syncthreads();

        xmax = Max[tid & 31];
        for (uint i = 16; i > 0; i >>= 1)
            xmax = fmaxf(xmax, shfl_xor(xmax, i));
    }

    // subtract xmax and compute exponent
    float exp_sum = 0;
    T exp_max[UNROLL];
    for (uint i = 0; i < UNROLL; i++)
    {
        V exp      = ew_exp(ew_sub(to_float(logits[i]), xmax));
        exp_sum   += ew_sum(exp);
        // scale to maximize fp16 bit utilization
        // Warning: this will flush the scaled values less than 2^-24 to zero
        exp_max[i] = to_ehalf(ew_mul(exp, 32768.0f));
        // stupid no-op to force ptxas to pack 2 fp16 values into 32 bit registers, sigh.
        xor1(exp_max[i]);
    }

    // reduce within warp
    for (int i = 16; i > 0; i >>= 1)
        exp_sum += shfl_xor(exp_sum, i);

    if (blockDim.x > 32)
    {
        if ((tid & 31) == 0)
            Sum[tid/32] = exp_sum;
        __syncthreads();

        exp_sum = Sum[tid & 31];
        for (uint i = 16; i > 0; i >>= 1)
            exp_sum += shfl_xor(exp_sum, i);
    }
    float rcp_exp_sum = ew_rcp(exp_sum) * (1.0f/32768.0f); // undo fp16 scaling

    uint label = (uint)__ldg(Labels + idx_N);

    Grad += offset;
    asm("mov.b64 %0, %0;" : "+l"(Grad) : );
    #pragma unroll
    for (uint i = 0; i < UNROLL; i++)
    {
        if (k + i*VSIZE*32 < K)
        {
            // stupid no-op to force ptxas to pack 2 fp16 values into 32 bit registers, sigh.
            xor1(exp_max[i]);
            // Warning: scaled values less than 2^-24 will have been flushed to zero
            V expmax2 = to_float(exp_max[i]);
            // allow softmax for grad computation to have the flushed to zero values
            V softmax = ew_mul(expmax2, rcp_exp_sum);
            // to avoid a nan for log(0), we clip the flushed to zero values to 2^-24
            V loss    = ew_neg(ew_log(ew_mul(ew_maximum(expmax2, 5.960464477539063e-08f), rcp_exp_sum)));

            for (uint j = 0; j < VSIZE; j++)
            {
                if (k + i*VSIZE*32 + j == label)
                {
                    Max[0] = ((float*)&loss)[j];
                    ((float*)&softmax)[j] -= 1.0f;
                }
                ((float*)&softmax)[j] *= 32768.0f; // scale to maximize fp16 bit utilization
            }
            __stg((T*)(Grad + i*VSIZE*32), to_ehalf(softmax));
        }
    }
    __syncthreads();

    if (tid == 0)
        Loss[idx_N] = Max[0];
}

template <typename TL>
bool SoftmaxCrossEntropy(CUstream stream, ehalf* grad, float* loss, const ehalf* logits, const TL* labels, uint N, uint K)
{
         if (K > 1024*32) // <= 64k
        softmax_cross_entropy<8,ehalf8,float8,TL,1><<<N,1024,0,stream>>>(grad, loss, logits, labels, K);
    else if (K > 1024*16) // <= 32k
        softmax_cross_entropy<4,ehalf8,float8,TL,1><<<N,1024,0,stream>>>(grad, loss, logits, labels, K);
    else if (K > 1024* 8) // <= 16k
        softmax_cross_entropy<2,ehalf8,float8,TL,2><<<N,1024,0,stream>>>(grad, loss, logits, labels, K);
    else if (K >=  32* 8) // 256 <= K <= 8k
    {
        uint threads = CEIL_DIV(K, 32*8) * 32;
        softmax_cross_entropy<1,ehalf8,float8,TL,2><<<N,threads,0,stream>>>(grad, loss, logits, labels, K);
    }
    else // K < 256
    {
        uint threads = CEIL_DIV(K, 32*2) * 32;
        softmax_cross_entropy<1,ehalf2,float2,TL,2><<<N,threads,0,stream>>>(grad, loss, logits, labels, K);
    }
    return true;
}

template bool SoftmaxCrossEntropy<         int  >(CUstream stream, ehalf* grad, float* loss, const ehalf* logits, const          int*   labels, uint N, uint K);
template bool SoftmaxCrossEntropy<unsigned short>(CUstream stream, ehalf* grad, float* loss, const ehalf* logits, const unsigned short* labels, uint N, uint K);
template bool SoftmaxCrossEntropy<unsigned char >(CUstream stream, ehalf* grad, float* loss, const ehalf* logits, const unsigned char*  labels, uint N, uint K);

template <typename T, typename V, uint THREADS>
__global__ void softmax_cross_entropy_grad(T* DX, const float* __restrict__ DY, const T* Y, uint NK, uint K)
{
    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    #pragma unroll 1
    for (uint nk = bid*THREADS + tid; nk < NK; nk += gridDim.x*THREADS)
    {
        uint n = nk / K;

        V      y = load( Y + nk);
        float dy = load(DY + n) * (1.0f/32768.0f); // undo fp16 scaling

        store(DX + nk, ew_mul(y, dy));
    }
}

bool SoftmaxCrossEntropyGrad(CUstream stream, uint SMs, ehalf* dx, const float* dy, const ehalf* y, uint NK, uint K)
{
    if (K >= 256)
    {
        NK >>= 3; // use vector loads
        K  >>= 3; // use vector loads
        uint grid = NK > SMs*256*2 ? SMs*4 : NK > SMs*256 ? SMs*2 : SMs;
        softmax_cross_entropy_grad<ehalf8,float8,256><<<grid,256,0,stream>>>((ehalf8*)dx, dy, (const ehalf8*)y, NK, K);
    }
    else
    {
        NK >>= 1; // use vector loads
        K  >>= 1; // use vector loads
        uint grid = NK > SMs*256*4 ? SMs*8 : NK > SMs*256*2 ? SMs*4 : NK > SMs*256 ? SMs*2 : SMs;
        softmax_cross_entropy_grad<ehalf2,float2,256><<<grid,256,0,stream>>>((ehalf2*)dx, dy, (const ehalf2*)y, NK, K);
    }
    return true;
}


// loss[j] = (log(sum_exp_logits) - logits[j]) * 1{ j == label }

// exp_logits[j] / sum_exp_logits - 1{ j == label }

// _CUDA_FP16_DECL__ __half hlog(const __half a) {
//     __half val;
//     asm("{.reg.b32         f, C;           \n"
//         " .reg.b16         r,h;            \n"
//         "  mov.b16         h,%1;           \n"
//         "  cvt.f32.f16     f,h;            \n"
//         "  lg2.approx.f32      f,f;        \n"
//         "  mov.b32         C, 0x3f317218;  \n"
//         "  mul.f32         f,f,C;          \n"
//         "  cvt.rn.f16.f32      r,f;        \n"
//         __SPEC_CASE(h, r, 0X160D, 0x9C00)
//         __SPEC_CASE(h, r, 0X3BFE, 0x8010)
//         __SPEC_CASE(h, r, 0X3C0B, 0x8080)
//         __SPEC_CASE(h, r, 0X6051, 0x1C00)
//         "  mov.b16         %0,r;           \n"
//         "}": "=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)));
//     return val;


#endif
