
#if GOOGLE_CUDA

#include "ew_op_gpu.h"
#include <stdio.h>

// xmean = x - mean(x, axis=1)
// xvar  = var(x, axis=1)
// xstdr = rcp(sqrt(xvar + epsilon))
// xhat  = xmean * xstdr
// y     = xhat*g + b
template <typename T, typename V, int THREADS>
__global__ void __launch_bounds__(THREADS) layer_norm_NC(
              T*              Y,
          float*              Mean,
          float*              Rstd,
    const     T* __restrict__ X,
    const     V* __restrict__ G,
    const     V* __restrict__ B,
    float epsilon, uint K, float rcpK, int relu)
{
    uint tid = threadIdx.x;
    uint n   = blockIdx.x;

    uint offset = n*K + tid;

    // Mean
    V v_mean1, v_mean2;
    ew_zero(v_mean1);
    ew_zero(v_mean2);
    #pragma unroll 4
    for (uint k = tid, offsetX = offset; k < K; k += THREADS, offsetX += THREADS)
    {
        // Single pass over X to compute mean and variance
        // var(x) == mean(x**2) - mean(x)**2
        V x = load(add_ptr_u(X, offsetX));
        v_mean1 = ew_add(v_mean1, x);
        v_mean2 = ew_add(v_mean2, ew_sqr(x));
    }
    float2 stats;
    stats.x = ew_sum(v_mean1) * rcpK;
    stats.y = ew_sum(v_mean2) * rcpK;

    // reduce within warp
    for (int i = 16; i > 0; i >>= 1)
        stats = ew_warp_sum(stats, i);

    // if using more than 1 warp, further reduced with shared memory
    if (THREADS > 32)
    {
        __shared__ float2 Share[32];

        // first thread of each warp store to shared
        if ((tid & 31) == 0)
            Share[tid/32] = stats;

        __syncthreads();

        if (tid < 32)
        {
            // first warp loads all prior reductions
            stats = Share[tid];

            // reduce within this first warp
            for (int i = THREADS/64; i > 0; i >>= 1)
                stats = ew_warp_sum(stats, i);

            // final reduction to shared
            Share[tid] = stats;
        }
        __syncthreads();

        // broadcast result to all threads
        stats = Share[0];
    }
    // var  = avg(x**2) - avg(x)**2
    // rstd = 1/sqrt(var)
    float mean = stats.x;
    float rstd = rsqrtf(precise_sub(stats.y, ew_sqr(mean)) + epsilon);

    if (tid == 0)
    {
        Mean[n] = mean;
        Rstd[n] = rstd;
    }

    // Norm/Gain/Bias
    #pragma unroll 4
    for (uint k = tid; k < K; k += THREADS, offset += THREADS)
    {
        V x = load(add_ptr_u(X, offset));
        V g = load(G, k);
        V b = load(B, k);

        V xhat = ew_mul(ew_sub(x, mean), rstd);
        V    y = ew_add(ew_mul(xhat, g), b);

        if (relu)
            y = ew_relu(y);

        store(add_ptr_u(Y, offset), y);
    }
}

template <typename T, typename V>
bool LayerNormForward_NC(CUstream stream, int SMs,
              T* y,
          float* mean,
          float* rstd,
    const     T* x,
    const float* g,
    const float* b,
    float epsilon, int K, int N, float rcpK, int relu)
{
    dim3 grid(N, 1, 1);

    if ((K & 3) == 0)
    {
        K >>= 2; // use vector loads
                   V* Y = (V*)y;
        const      V* X = (const V*)x;
        const float4* G = (const float4*)g;
        const float4* B = (const float4*)b;
        if (K >= 256)
            layer_norm_NC<V,float4,256><<<grid, 256,0,stream>>>(Y, mean, rstd, X, G, B, epsilon, K, rcpK, relu);
        else
            layer_norm_NC<V,float4, 32><<<grid,  32,0,stream>>>(Y, mean, rstd, X, G, B, epsilon, K, rcpK, relu);
    }
    else
    {
        if (K >= 256)
            layer_norm_NC<T,float ,256><<<grid, 256,0,stream>>>(y, mean, rstd, x, g, b, epsilon, K, rcpK, relu);
        else
            layer_norm_NC<T,float , 32><<<grid,  32,0,stream>>>(y, mean, rstd, x, g, b, epsilon, K, rcpK, relu);
    }
    return true; // TODO
}
template bool LayerNormForward_NC<float,float4>(CUstream stream, int SMs, float* y, float* mean, float* rstd, const float* x, const float* g, const float* b, float epsilon, int K, int N, float rcpK, int relu);
template bool LayerNormForward_NC<ehalf,ehalf4>(CUstream stream, int SMs, ehalf* y, float* mean, float* rstd, const ehalf* x, const float* g, const float* b, float epsilon, int K, int N, float rcpK, int relu);
template bool LayerNormForward_NC<bhalf,bhalf4>(CUstream stream, int SMs, bhalf* y, float* mean, float* rstd, const bhalf* x, const float* g, const float* b, float epsilon, int K, int N, float rcpK, int relu);


// Sum across N axis requries separtate kernel.
// dg = sum(dy * xhat(x), axis=0)
// db = sum(dy, axis=0)
// Don't use vector loads here as we want to maximize the number of blocks
template <typename T, int U>
__global__ void __launch_bounds__(32) layer_norm_dg_db_NC(
          float*              DG,
          float*              DB,
    const     T* __restrict__ DY,
    const     T* __restrict__ X,
    const float* __restrict__ Gain,
    const float* __restrict__ Bias,
    const float* __restrict__ Mean,
    const float* __restrict__ Rstd,
    float epsilon, int K, int N, int relu)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int shift = 5 - U; // 4
    int mask  = (1 << shift) - 1; // 15

    int k  = (bid << shift) + (tid & mask); // b*16 + 0-15
    int n0 = (tid >> shift) << 2; // 0,4
    int nk = n0*K + k;
    bool b = k < K;
    int strideK = K << (2 + U);

    float gain = 1.0f, bias = 0.0f, dg = 0.0f, db = 0.0f;
    if (b && relu)
    {
        gain = Gain[k];
        bias = Bias[k];
    }
    for (int n = n0; n < N; n += (4 << U))
    {
        int n1  = n  + 1;
        int n2  = n  + 2;
        int n3  = n  + 3;
        int nk1 = nk  + K;
        int nk2 = nk1 + K;
        int nk3 = nk2 + K;
        float  x0 = load( X, nk,  b);
        float  x1 = load( X, nk1, b && (n1 < N));
        float  x2 = load( X, nk2, b && (n2 < N));
        float  x3 = load( X, nk3, b && (n3 < N));
        float dy0 = load(DY, nk,  b);
        float dy1 = load(DY, nk1, b && (n1 < N));
        float dy2 = load(DY, nk2, b && (n2 < N));
        float dy3 = load(DY, nk3, b && (n3 < N));

        float mean0 = Mean[n];
        float rstd0 = Rstd[n];
        float mean1 = 0.0f, rstd1 = 0.0f;
        float mean2 = 0.0f, rstd2 = 0.0f;
        float mean3 = 0.0f, rstd3 = 0.0f;
        if (n1 < N)
        {
            mean1 = Mean[n1];
            rstd1 = Rstd[n1];
        }
        if (n2 < N)
        {
            mean2 = Mean[n2];
            rstd2 = Rstd[n2];
        }
        if (n3 < N)
        {
            mean3 = Mean[n3];
            rstd3 = Rstd[n3];
        }
        float xhat0 = (x0 - mean0) * rstd0;
        float xhat1 = (x1 - mean1) * rstd1;
        float xhat2 = (x2 - mean2) * rstd2;
        float xhat3 = (x3 - mean3) * rstd3;

        if (relu)
        {
            dy0 = ew_relu_grad(dy0, xhat0 * gain + bias);
            dy1 = ew_relu_grad(dy1, xhat1 * gain + bias);
            dy2 = ew_relu_grad(dy2, xhat2 * gain + bias);
            dy3 = ew_relu_grad(dy3, xhat3 * gain + bias);
        }
        dg += dy0 * xhat0;
        dg += dy1 * xhat1;
        dg += dy2 * xhat2;
        dg += dy3 * xhat3;
        db += dy0;
        db += dy1;
        db += dy2;
        db += dy3;
        nk += strideK;
    }
    #pragma unroll
    for (int i = 16; i > (1 << (4-U)); i >>= 1)
    {
        dg += shfl_xor(dg, i);
        db += shfl_xor(db, i);
    }
    store(DG, dg, k, b);
    store(DB, db, k, b);
}

// xmean = x - mean(x, axis=1)
// xvar  = var(x, axis=1)
// xstdr = rcp(sqrt(xvar + epsilon))
// xhat  = xmean * xstdr
// dy    = dy * g
// sum1  = sum(xhat * dy, axis=1)
// sum2  = sum(dy, axis=1)
// dx    = (dy - ((xhat * sum1 + sum2) * rcpK)) * xstdr
template <typename T, typename V, int THREADS>
__global__ void __launch_bounds__(THREADS) layer_norm_dx_NC(
              T*              DX,
    const     T* __restrict__ DY,
    const     T* __restrict__ X,
    const     V* __restrict__ Gain,
    const     V* __restrict__ Bias,
    const float* __restrict__ Mean,
    const float* __restrict__ Rstd,
    float epsilon, int K, float rcpK, int relu)
{
    __shared__ float Share1[32];
    __shared__ float Share2[32];

    int tid = threadIdx.x;
    int n   = blockIdx.x;

    int offset = n*K + tid;

    float mean = Mean[n];
    float rstd = Rstd[n];

    const T* X1 = X  + offset;
    const T* Y1 = DY + offset;
    V v_sum1, v_sum2;
    ew_zero(v_sum1);
    ew_zero(v_sum2);
    for (int k = tid; k < K; k += THREADS)
    {
        V  x = load(X1);
        V dy = load(Y1);
        V  g = load(Gain, k);
        V  b = load(Bias, k, relu != 0);

        V xhat = ew_mul(ew_sub(x, mean), rstd);

        if (relu)
            dy = ew_relu_grad(dy, ew_add(ew_mul(xhat, g), b));
        dy = ew_mul(dy, g);

        v_sum1 = ew_add(v_sum1, ew_mul(dy, xhat));
        v_sum2 = ew_add(v_sum2, dy);

        X1 += THREADS;
        Y1 += THREADS;
    }
    float sum1 = ew_sum(v_sum1);
    float sum2 = ew_sum(v_sum2);
    // reduce within warp
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1)
    {
        sum1 += shfl_xor(sum1, i);
        sum2 += shfl_xor(sum2, i);
    }
    // first thread of each warp store to shared
    if ((tid & 31) == 0)
    {
        Share1[tid >> 5] = sum1;
        Share2[tid >> 5] = sum2;
    }
    __syncthreads();
    if (tid < 32)
    {
        // first warp loads all prior reductions
        sum1 = Share1[tid];
        sum2 = Share2[tid];
        // reduce within this last warp
        #pragma unroll
        for (int i = THREADS/64; i > 0; i >>= 1)
        {
            sum1 += shfl_xor(sum1, i);
            sum2 += shfl_xor(sum2, i);
        }
        // outputs final reduction to shared
        Share1[tid] = sum1;
        Share2[tid] = sum2;
    }
    __syncthreads();
    // broadcast result to all threads
    sum1 = Share1[0];
    sum2 = Share2[0];

    X  += offset;
    DY += offset;
    DX += offset;
    for (int k = tid; k < K; k += THREADS)
    {
        V  x = load(X);
        V dy = load(DY);
        V  g = load(Gain, k);
        V  b = load(Bias, k, relu != 0);

        V xhat = ew_mul(ew_sub(x, mean), rstd);

        if (relu)
            dy = ew_relu_grad(dy, ew_add(ew_mul(xhat, g), b));
        dy = ew_mul(dy, g);

        // dx = (dy - ((xhat * sum1 + sum2) * rcpK)) * rstd;
        V dx = ew_mul(ew_sub(dy, ew_mul(ew_add(ew_mul(xhat, sum1), sum2), rcpK)), rstd);

        store(DX, dx);

        X  += THREADS;
        DY += THREADS;
        DX += THREADS;
    }
}

template <typename T, typename V>
bool LayerNormBackward_NC(CUstream stream, int SMs,
              T* dx,
          float* dg,
          float* db,
    const     T* dy,
    const     T* x,
    const float* g,
    const float* b,
    const float* mean,
    const float* rstd,
    float epsilon, int K, int N, float rcpK, int relu)
{

    int K32 = K >> 5;

    // optimize layer_norm_backward1 for highest occupancy
    if (K32 >= 28*16)
    {
        int gridK = K32 + ((K & 31) != 0);
        layer_norm_dg_db_NC<T,0><<<gridK, 32, 0, stream>>>(dg, db, dy, x, g, b, mean, rstd, epsilon, K, N, relu);
    }
    else if (K32 >= 28*8)
    {
        int gridK = (K >> 4) + ((K & 15) != 0);
        layer_norm_dg_db_NC<T,1><<<gridK, 32, 0, stream>>>(dg, db, dy, x, g, b, mean, rstd, epsilon, K, N, relu);
    }
    else if (K32 >= 28*4)
    {
        int gridK = (K >> 3) + ((K & 7) != 0);
        layer_norm_dg_db_NC<T,2><<<gridK, 32, 0, stream>>>(dg, db, dy, x, g, b, mean, rstd, epsilon, K, N, relu);
    }
    else
    {
        int gridK = (K >> 2) + ((K & 3) != 0);
        layer_norm_dg_db_NC<T,3><<<gridK, 32, 0, stream>>>(dg, db, dy, x, g, b, mean, rstd, epsilon, K, N, relu);
    }
    if ((K & 3) == 0)
    {
               V* DX = (      V*)dx;
        const  V* DY = (const V*)dy; // in place op
        const  V*  X = (const V*)x;
        const float4* Gain = (const float4*)g;
        const float4* Bias = (const float4*)b;

        K >>= 2;
        //if      (K >= 1024)
        //    layer_norm_dx_NC<VB,VF,float4,1024><<<N,1024,0,stream>>>(DX, DY, X, mean, rstd, epsilon, K, rcpK);
        if (K >=  256)
            layer_norm_dx_NC<V,float4, 256><<<N, 256,0,stream>>>(DX, DY, X, Gain, Bias, mean, rstd, epsilon, K, rcpK, relu);
        else
            layer_norm_dx_NC<V,float4,  64><<<N,  64,0,stream>>>(DX, DY, X, Gain, Bias, mean, rstd, epsilon, K, rcpK, relu);
    }
    else
    {
        //if      (K >= 1024)
        //    layer_norm_dx_NC<B,F,float,1024><<<N,1024,0,stream>>>(dx, (const B*)dx, x, mean, rstd, epsilon, K, rcpK);
        if (K >=  256)
            layer_norm_dx_NC<T,float, 256><<<N, 256,0,stream>>>(dx, dy, x, g, b, mean, rstd, epsilon, K, rcpK, relu);
        else
            layer_norm_dx_NC<T,float,  64><<<N,  64,0,stream>>>(dx, dy, x, g, b, mean, rstd, epsilon, K, rcpK, relu);
    }
    return true; // TODO
}

template bool LayerNormBackward_NC<float,float4>(CUstream stream, int SMs, float* dx, float* dg, float* db, const float* dy, const float* x, const float* g, const float* b, const float* mean, const float* rstd, float epsilon, int K, int N, float rcpK, int relu);
template bool LayerNormBackward_NC<ehalf,ehalf4>(CUstream stream, int SMs, ehalf* dx, float* dg, float* db, const ehalf* dy, const ehalf* x, const float* g, const float* b, const float* mean, const float* rstd, float epsilon, int K, int N, float rcpK, int relu);
template bool LayerNormBackward_NC<bhalf,bhalf4>(CUstream stream, int SMs, bhalf* dx, float* dg, float* db, const bhalf* dy, const bhalf* x, const float* g, const float* b, const float* mean, const float* rstd, float epsilon, int K, int N, float rcpK, int relu);


// xmean = x - mean(x, axis=1)
// xvar  = var(x, axis=1)
// xstdr = rcp(sqrt(xvar + epsilon))
// xhat  = xmean * xstdr
// y     = xhat*g + b
template <typename T, typename V, int U>
__global__ void layer_norm_segmented_nc(
              T*              Y,
          float*              Mean,
          float*              Rstd,
    const     T* __restrict__ X,
    const     V* __restrict__ G,
    const     V* __restrict__ B,
    float epsilon, uint N, uint SK, uint K, float rcpK, int relu, int thread2)
{
    __shared__ float2 Share[32];

    uint tid = threadIdx.x;

    if (blockDim.x > 32)
    {
        // Allows non-power of 2 threads to work
        float2 zero = {0.0f, 0.0f};
        if (tid < 32)
            Share[tid] = zero;
        __syncthreads();
    }
    uint n = blockIdx.x;
    uint s = blockIdx.y;
    uint t = (tid & 0x3e0)*U + (tid & 31); // 0x3e0 = -32 & 1023
    uint k = s*K + t;
    uint m = s*N + n;

    uint offset = n*SK + k;

    // Load X
    V xval[U];
    X = add_ptr_u(X, offset);
    for (int i = 0; i < U; i++)
        xval[i] = load(X, i*32, t + i*32 < K);

    // Begin mean/variance reductions
    V mean1[U], mean2[U];
    for (int i = 0; i < U; i++)
    {
        mean1[i] = xval[i];
        mean2[i] = ew_sqr(xval[i]);
    }

    // reduce within thread
    for (int j = U >> 1; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
        {
            mean1[i] = ew_add(mean1[i], mean1[i+j]);
            mean2[i] = ew_add(mean2[i], mean2[i+j]);
        }
    float2 stats;
    stats.x = ew_sum(mean1[0]) * rcpK;
    stats.y = ew_sum(mean2[0]) * rcpK;

    // reduce within warp
    for (int i = 16; i > 0; i >>= 1)
        stats = ew_warp_sum(stats, i);

    // reduce across warps
    if (blockDim.x > 32)
    {
        // first thread of each warp store to shared
        if ((tid & 31) == 0)
            Share[tid/32] = stats;
        __syncthreads();

        if (tid < 32)
        {
            // first warp loads all prior reductions
            stats = Share[tid];
            // reduce within this last warp
            #pragma unroll 1
            for (int i = thread2/64; i > 0; i >>= 1)
                stats = ew_warp_sum(stats, i);

            // final reduction to shared
            Share[tid] = stats;
        }
        __syncthreads();
        stats = Share[0];
    }
    // var  = avg(x**2) - avg(x)**2
    // rstd = 1/sqrt(var)
    float mean = stats.x;
    float rstd = rsqrtf(precise_sub(stats.y, ew_sqr(mean)) + epsilon);
    if (tid == 0)
    {
        __stg(add_ptr_u(Mean, m), mean);
        __stg(add_ptr_u(Rstd, m), rstd);
    }

    // Load Gain/Bias
    G = add_ptr_u(G, k);
    B = add_ptr_u(B, k);

    V gain[U], bias[U];
    for (int i = 0; i < U; i++)
    {
        bool  b = t + i*32 < K;
        gain[i] = load(G, i*32, b);
        bias[i] = load(B, i*32, b);
    }

    // Compute and output norm
    Y = add_ptr_u(Y, offset);
    for (int i = 0; i < U; i++)
    {
        V xhat = ew_mul(ew_sub(xval[i], mean), rstd);
        V    y = ew_add(ew_mul(xhat, gain[i]), bias[i]);
        if (relu)
            y = ew_relu(y);

        store(Y, y, i*32, t + i*32 < K);
    }
}
template <typename T, typename V>
bool LayerNormSegmentedForward_NC(CUstream stream, int SMs,
              T* y,
          float* mean,
          float* rstd,
    const     T* x,
    const float* g,
    const float* b,
    float epsilon, uint N, uint S, uint K, float rcpK, int relu)
{
    dim3 grid(N, S, 1);

    if ((K & 3) == 0)
    {
                   V* Y = (V*)y;
        const      V* X = (const V*)x;
        const float4* G = (const float4*)g;
        const float4* B = (const float4*)b;

        if (K >= 256)
        {
            uint threads = CEIL_DIV(K, 32*8) * 32;
            int  thread2 = THREAD_POW2(threads);
            K >>= 2;
            layer_norm_segmented_nc<V,float4,2><<<grid,threads,0,stream>>>(Y, mean, rstd, X, G, B, epsilon, N, S*K, K, rcpK, relu, thread2);
        }
        else
        {
            uint threads = CEIL_DIV(K, 32*4) * 32;
            int  thread2 = THREAD_POW2(threads);
            K >>= 2;
            layer_norm_segmented_nc<V,float4,1><<<grid,threads,0,stream>>>(Y, mean, rstd, X, G, B, epsilon, N, S*K, K, rcpK, relu, thread2);
        }
    }
    else
    {
        if (K >= 256)
        {
            uint threads = CEIL_DIV(K, 32*8) * 32;
            int  thread2 = THREAD_POW2(threads);
            layer_norm_segmented_nc<T,float,8><<<grid,threads,0,stream>>>(y, mean, rstd, x, g, b, epsilon, N, S*K, K, rcpK, relu, thread2);
        }
        else
        {
            uint threads = CEIL_DIV(K, 32*4) * 32;
            int  thread2 = THREAD_POW2(threads);
            layer_norm_segmented_nc<T,float,4><<<grid,threads,0,stream>>>(y, mean, rstd, x, g, b, epsilon, N, S*K, K, rcpK, relu, thread2);
        }
    }
    return true; // TODO
}
template bool LayerNormSegmentedForward_NC<float,float4>(CUstream stream, int SMs, float* y, float* mean, float* rstd, const float* x, const float* g, const float* b, float epsilon, uint N, uint S, uint K, float rcpK, int relu);
template bool LayerNormSegmentedForward_NC<ehalf,ehalf4>(CUstream stream, int SMs, ehalf* y, float* mean, float* rstd, const ehalf* x, const float* g, const float* b, float epsilon, uint N, uint S, uint K, float rcpK, int relu);
template bool LayerNormSegmentedForward_NC<bhalf,bhalf4>(CUstream stream, int SMs, bhalf* y, float* mean, float* rstd, const bhalf* x, const float* g, const float* b, float epsilon, uint N, uint S, uint K, float rcpK, int relu);


// Sum across N axis requries separtate kernel.
// dg = sum(dy * xhat(x), axis=0)
// db = sum(dy, axis=0)
// Don't use vector loads here as we want to maximize the number of blocks
template <typename T>
__global__ void __launch_bounds__(32) layer_norm_segmented_dg_db_nc(
          float*              DG,
          float*              DB,
    const     T* __restrict__ DY,
    const     T* __restrict__ X,
    const float* __restrict__ Gain,
    const float* __restrict__ Bias,
    const float* __restrict__ Mean,
    const float* __restrict__ Rstd,
    uint N, uint SK, uint SKz, uint K, int relu)
{
    uint tid = threadIdx.x;
    uint bn  = blockIdx.x;
    uint bk  = blockIdx.y;
    uint bs  = blockIdx.z;

    uint t = bk*32 + tid;
    uint k = bs*K + t;
    bool b = t < K;

    float gain = 1.0f, bias = 0.0f, dg = 0.0f, db = 0.0f;
    if (b && relu)
    {
        gain = __ldg(add_ptr_u(Gain, k));
        bias = __ldg(add_ptr_u(Bias, k));
    }
    #pragma unroll 1
    for (uint n = bn, m = bs*N + bn, nk = bn*SK + k; n < N; n += gridDim.x, m += gridDim.x, nk += SKz)
    {
        float    x = load(add_ptr_u(X,  nk), 0, b);
        float   dy = load(add_ptr_u(DY, nk), 0, b);
        float mean = load(add_ptr_u(Mean, m));
        float rstd = load(add_ptr_u(Rstd, m));
        float xhat = (x - mean) * rstd;
        if (relu)
            dy = ew_relu_grad(dy, xhat * gain + bias);

        dg += dy * xhat;
        db += dy;
    }
    if (b)
    {
        DG = add_ptr_u(DG, k);
        DB = add_ptr_u(DB, k);
        if (gridDim.x == 1)
        {
            __stg(DG, dg);
            __stg(DB, db);
        }
        else
        {
            atomicRed(DG, dg);
            atomicRed(DB, db);
        }
    }
}

// xmean = x - mean(x, axis=1)
// xvar  = var(x, axis=1)
// xstdr = rcp(sqrt(xvar + epsilon))
// xhat  = xmean * xstdr
// dy    = dy * g
// sum1  = sum(xhat * dy, axis=1)
// sum2  = sum(dy, axis=1)
// dx    = (dy - ((xhat * sum1 + sum2) * rcpK)) * xstdr
template <typename T, typename V, int U>
__global__ void layer_norm_segmented_dx_nc(
              T*              DX,
    const     T* __restrict__ DY,
    const     T* __restrict__ X,
    const     V* __restrict__ Gain,
    const     V* __restrict__ Bias,
    const float* __restrict__ Mean,
    const float* __restrict__ Rstd,
    uint N, uint SK, uint K, float rcpK, int relu, int thread2)
{
    __shared__ float2 Share[32];

    uint tid = threadIdx.x;

    if (blockDim.x > 32)
    {
        // Allows non-power of 2 threads to work
        float2 zero = {0.0f, 0.0f};
        if (tid < 32)
            Share[tid] = zero;
        __syncthreads();
    }
    uint n = blockIdx.x;
    uint s = blockIdx.y;
    uint t = (tid & 0x3e0)*U + (tid & 31); // 0x3e0 = -32 & 1023
    uint k = s*K + t;
    uint m = s*N + n;

    uint offset = n*SK + k;

    float mean = __ldg(add_ptr_u(Mean, m));
    float rstd = __ldg(add_ptr_u(Rstd, m));

    X    = add_ptr_u(X,  offset);
    DY   = add_ptr_u(DY, offset);
    Gain = add_ptr_u(Gain,    k);
    V x[U], dy[U], gain[U];
    for (int i = 0; i < U; i++)
    {
        bool  b = t + i*32 < K;
        x[i]    = load(X,    i*32, b);
        dy[i]   = load(DY,   i*32, b);
        gain[i] = load(Gain, i*32, b);
    }
    V xhat[U];
    if (relu)
    {
        Bias = add_ptr_u(Bias, k);
        for (int i = 0; i < U; i++)
        {
            V bias  = load(Bias, i*32, t + i*32 < K);
            xhat[i] = ew_mul(ew_sub(x[i], mean), rstd);
            dy[i]   = ew_relu_grad(dy[i], ew_add(ew_mul(xhat[i], gain[i]), bias));
        }
    }
    else
    {
        for (int i = 0; i < U; i++)
            xhat[i] = ew_mul(ew_sub(x[i], mean), rstd);
    }
    V sum1[U], sum2[U];
    for (int i = 0; i < U; i++)
    {
        dy[i]   = ew_mul(dy[i], gain[i]);
        sum1[i] = ew_mul(dy[i], xhat[i]);
        sum2[i] = dy[i];
    }

    // reduce within thread
    for (int j = U >> 1; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
        {
            sum1[i] = ew_add(sum1[i], sum1[i+j]);
            sum2[i] = ew_add(sum2[i], sum2[i+j]);
        }
    float2 sums;
    sums.x = ew_sum(sum1[0]);
    sums.y = ew_sum(sum2[0]);

    // reduce within warp
    for (int i = 16; i > 0; i >>= 1)
        sums = ew_warp_sum(sums, i);

    // reduce across warps
    if (blockDim.x > 32)
    {
        // first thread of each warp store to shared
        if ((tid & 31) == 0)
            Share[tid/32] = sums;
        __syncthreads();

        if (tid < 32)
        {
            // first warp loads all prior reductions
            sums = Share[tid];
            // reduce within this last warp
            #pragma unroll 1
            for (int i = thread2/64; i > 0; i >>= 1)
                sums = ew_warp_sum(sums, i);
            // final reduction to shared
            Share[tid] = sums;
        }
        __syncthreads();
        sums = Share[0];
    }
    // Compute and store dx
    DX = add_ptr_u(DX, offset);
    for (int i = 0; i < U; i++)
    {
        // dx = (dy - ((xhat * sum1 + sum2) * rcpK)) * rstd;
        V dx = ew_mul(ew_sub(dy[i], ew_mul(ew_add(ew_mul(xhat[i], sums.x), sums.y), rcpK)), rstd);
        store(DX, dx, i*32, t + i*32 < K);
    }
}


template <typename T, typename V>
bool LayerNormSegmentedBackward_NC(CUstream stream, int SMs,
              T* dx,
          float* dg,
          float* db,
    const     T* dy,
    const     T* x,
    const float* g,
    const float* b,
    const float* mean,
    const float* rstd,
    float epsilon, uint N, uint S, uint K, float rcpK, int relu, int atomics)
{
    uint gridK = CEIL_DIV(K, 32);
    uint gridN = 1;
    if (atomics)
    {
        uint blocksK = gridK * S;
        while (gridN < (N>>3) && gridN * blocksK < 32*SMs) gridN += 1;
        if (gridN * blocksK > 32*SMs && gridN > 1) gridN -= 1;
        if (gridN > 1)
        {
            cuMemsetD32Async((CUdeviceptr)dg, 0, S*K, stream);
            cuMemsetD32Async((CUdeviceptr)db, 0, S*K, stream);
        }
    }
    layer_norm_segmented_dg_db_nc<T><<<dim3(gridN,gridK,S),32,0,stream>>>(dg, db, dy, x, g, b, mean, rstd, N, S*K, S*K*gridN, K, relu);

    dim3 grid(N, S, 1);
    if ((K & 3) == 0 && K >= 512)
    {
                   V* DX = (      V*)dx;
        const      V* DY = (const V*)dy; // in place op
        const      V*  X = (const V*)x;
        const float4*  G = (const float4*)g;
        const float4*  B = (const float4*)b;

        if (K > 4096)
        {
            uint threads = CEIL_DIV(K, 32*8) * 32;
            int  thread2 = THREAD_POW2(threads);
            K >>= 2;
            layer_norm_segmented_dx_nc<V,float4,2><<<grid,threads,0,stream>>>(DX, DY, X, G, B, mean, rstd, N, S*K, K, rcpK, relu, thread2);
        }
        else
        {
            uint threads = CEIL_DIV(K, 32*4) * 32;
            int  thread2 = THREAD_POW2(threads);
            K >>= 2;
            layer_norm_segmented_dx_nc<V,float4,1><<<grid,threads,0,stream>>>(DX, DY, X, G, B, mean, rstd, N, S*K, K, rcpK, relu, thread2);
        }
    }
    else
    {
        if (K > 4096)
        {
            uint threads = CEIL_DIV(K, 32*8) * 32;
            int  thread2 = THREAD_POW2(threads);
            layer_norm_segmented_dx_nc<T,float ,8><<<grid,threads,0,stream>>>(dx, dy, x, g, b, mean, rstd, N, S*K, K, rcpK, relu, thread2);
        }
        else if (K >= 512)
        {
            uint threads = CEIL_DIV(K, 32*4) * 32;
            int  thread2 = THREAD_POW2(threads);
            layer_norm_segmented_dx_nc<T,float ,4><<<grid,threads,0,stream>>>(dx, dy, x, g, b, mean, rstd, N, S*K, K, rcpK, relu, thread2);
        }
        else
        {
            uint threads = CEIL_DIV(K, 32*1) * 32;
            int  thread2 = THREAD_POW2(threads);
            layer_norm_segmented_dx_nc<T,float ,1><<<grid,threads,0,stream>>>(dx, dy, x, g, b, mean, rstd, N, S*K, K, rcpK, relu, thread2);
        }
    }
    return true; // TODO
}
template bool LayerNormSegmentedBackward_NC<float,float4>(CUstream stream, int SMs, float* dx, float* dg, float* db, const float* dy, const float* x, const float* g, const float* b, const float* mean, const float* rstd, float epsilon, uint N, uint S, uint K, float rcpK, int relu, int atomics);
template bool LayerNormSegmentedBackward_NC<ehalf,ehalf4>(CUstream stream, int SMs, ehalf* dx, float* dg, float* db, const ehalf* dy, const ehalf* x, const float* g, const float* b, const float* mean, const float* rstd, float epsilon, uint N, uint S, uint K, float rcpK, int relu, int atomics);
template bool LayerNormSegmentedBackward_NC<bhalf,bhalf4>(CUstream stream, int SMs, bhalf* dx, float* dg, float* db, const bhalf* dy, const bhalf* x, const float* g, const float* b, const float* mean, const float* rstd, float epsilon, uint N, uint S, uint K, float rcpK, int relu, int atomics);


#endif // GOOGLE_CUDA
