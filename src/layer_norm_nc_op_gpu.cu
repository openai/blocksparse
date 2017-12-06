
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "ew_op_gpu.h"


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
    float epsilon, int K, float rcpK, int relu)
{
    __shared__ float Share[THREADS>>5];

    int tid = threadIdx.x;
    int n   = blockIdx.x;

    int offset = n*K + tid;

    // Mean
    const T* X1 = X + offset;
    V v_mean;
    ew_zero(v_mean);
    for (int k = tid; k < K; k += THREADS)
    {
        V x = load(X1);
        v_mean = ew_add(v_mean, x);
        X1 += THREADS;
    }
    float mean = ew_sum(v_mean);
    // reduce within warp
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1)
        mean += __shfl_xor(mean, i);
    // first thread of each warp store to shared
    if ((tid & 31) == 0)
        Share[tid >> 5] = mean;
    __syncthreads();
    if (tid < (THREADS>>5))
    {
        // first warp loads all prior reductions
        mean = Share[tid];
        // reduce within this last warp
        #pragma unroll
        for (int i = (THREADS>>6); i > 0; i >>= 1)
            mean += __shfl_xor(mean, i);
        // outputs final reduction to shared
        Share[tid] = mean * rcpK;
    }
    __syncthreads();
    // broadcast result to all threads
    mean = Share[0];

    // Reciprocal Standard Deviation (rstd)
    const T* X2 = X + offset;
    V v_rstd;
    ew_zero(v_rstd);
    for (int k = tid; k < K; k += THREADS)
    {
        V x = load(X2);
        v_rstd = ew_add(v_rstd, ew_sqr(ew_sub(x, mean)));
        X2   += THREADS;
    }
    float rstd = ew_sum(v_rstd);
    // reduce within warp
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1)
        rstd += __shfl_xor(rstd, i);
    // first thread of each warp store to shared
    if ((tid & 31) == 0)
        Share[tid >> 5] = rstd;
    __syncthreads();
    if (tid < (THREADS>>5))
    {
        // first warp loads all prior reductions
        rstd = Share[tid];
        // reduce within this last warp
        #pragma unroll
        for (int i = (THREADS>>6); i > 0; i >>= 1)
            rstd += __shfl_xor(rstd, i);

        rstd = rsqrtf(rstd*rcpK + epsilon);

        // Outputs final reduction to shared
        // Also cache reductions for backward pass
        if (tid == 0)
        {
            Mean[n]  = mean;
            Rstd[n]  = rstd;
            Share[0] = rstd;
        }
    }
    __syncthreads();
    // broadcast result to all threads
    rstd = Share[0];

    // Norm/Gain/Bias
    X += offset;
    Y += offset;
    for (int k = tid; k < K; k += THREADS)
    {
        V x = load(X);
        V g = load(G, k);
        V b = load(B, k);

        V xhat = ew_mul(ew_sub(x, mean), rstd);
        V    y = ew_add(ew_mul(xhat, g), b);

        if (relu)
            y = ew_relu(y);

        store(Y, y);
        X += THREADS;
        Y += THREADS;
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
        //if (K >= 1024)
        //    layer_norm_NC<V,float4,1024><<<grid,1024,0,stream>>>(Y, mean, rstd, X, G, B, epsilon, K, rcpK);
        if (K >= 256)
            layer_norm_NC<V,float4, 256><<<grid, 256,0,stream>>>(Y, mean, rstd, X, G, B, epsilon, K, rcpK, relu);
        else
            layer_norm_NC<V,float4,  64><<<grid,  64,0,stream>>>(Y, mean, rstd, X, G, B, epsilon, K, rcpK, relu);
    }
    else
    {
        //if (K >= 1024)
        //    layer_norm_forward<T,float ,1024><<<grid,1024,0,stream>>>(y, mean, rstd, x, g, b, epsilon, K, rcpK);
        if (K >= 256)
            layer_norm_NC<T,float , 256><<<grid, 256,0,stream>>>(y, mean, rstd, x, g, b, epsilon, K, rcpK, relu);
        else
            layer_norm_NC<T,float ,  64><<<grid,  64,0,stream>>>(y, mean, rstd, x, g, b, epsilon, K, rcpK, relu);
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
template <typename B, typename F, int U>
__global__ void __launch_bounds__(32) layer_norm_dg_db_NC(
          float*              DG,
          float*              DB,
    const     B* __restrict__ DY,
    const     F* __restrict__ X,
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
        dg += __shfl_xor(dg, i);
        db += __shfl_xor(db, i);
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
template <typename B, typename F, typename V, int THREADS>
__global__ void __launch_bounds__(THREADS) layer_norm_dx_NC(
              B*              DX,
    const     B* __restrict__ DY,
    const     F* __restrict__ X,
    const     V* __restrict__ Gain,
    const     V* __restrict__ Bias,
    const float* __restrict__ Mean,
    const float* __restrict__ Rstd,
    float epsilon, int K, float rcpK, int relu)
{
    __shared__ float Share1[THREADS>>5];
    __shared__ float Share2[THREADS>>5];

    int tid = threadIdx.x;
    int n   = blockIdx.x;

    int offset = n*K + tid;

    float mean = Mean[n];
    float rstd = Rstd[n];

    const F* X1 = X  + offset;
    const B* Y1 = DY + offset;
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
        sum1 += __shfl_xor(sum1, i);
        sum2 += __shfl_xor(sum2, i);
    }
    // first thread of each warp store to shared
    if ((tid & 31) == 0)
    {
        Share1[tid >> 5] = sum1;
        Share2[tid >> 5] = sum2;
    }
    __syncthreads();
    if (tid < (THREADS>>5))
    {
        // first warp loads all prior reductions
        sum1 = Share1[tid];
        sum2 = Share2[tid];
        // reduce within this last warp
        #pragma unroll
        for (int i = (THREADS>>6); i > 0; i >>= 1)
        {
            sum1 += __shfl_xor(sum1, i);
            sum2 += __shfl_xor(sum2, i);
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

template <typename B, typename F, typename VB, typename VF>
bool LayerNormBackward_NC(CUstream stream, int SMs,
              B* dx,
          float* dg,
          float* db,
    const     B* dy,
    const     F* x,
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
        layer_norm_dg_db_NC<B,F,0><<<gridK, 32, 0, stream>>>(dg, db, dy, x, g, b, mean, rstd, epsilon, K, N, relu);
    }
    else if (K32 >= 28*8)
    {
        int gridK = (K >> 4) + ((K & 15) != 0);
        layer_norm_dg_db_NC<B,F,1><<<gridK, 32, 0, stream>>>(dg, db, dy, x, g, b, mean, rstd, epsilon, K, N, relu);
    }
    else if (K32 >= 28*4)
    {
        int gridK = (K >> 3) + ((K & 7) != 0);
        layer_norm_dg_db_NC<B,F,2><<<gridK, 32, 0, stream>>>(dg, db, dy, x, g, b, mean, rstd, epsilon, K, N, relu);
    }
    else
    {
        int gridK = (K >> 2) + ((K & 3) != 0);
        layer_norm_dg_db_NC<B,F,3><<<gridK, 32, 0, stream>>>(dg, db, dy, x, g, b, mean, rstd, epsilon, K, N, relu);
    }
    if ((K & 3) == 0)
    {
              VB* DX = (      VB*)dx;
        const VB* DY = (const VB*)dy; // in place op
        const VF*  X = (const VF*)x;
        const float4* Gain = (const float4*)g;
        const float4* Bias = (const float4*)b;

        K >>= 2;
        //if      (K >= 1024)
        //    layer_norm_dx_NC<VB,VF,float4,1024><<<N,1024,0,stream>>>(DX, DY, X, mean, rstd, epsilon, K, rcpK);
        if (K >=  256)
            layer_norm_dx_NC<VB,VF,float4, 256><<<N, 256,0,stream>>>(DX, DY, X, Gain, Bias, mean, rstd, epsilon, K, rcpK, relu);
        else
            layer_norm_dx_NC<VB,VF,float4,  64><<<N,  64,0,stream>>>(DX, DY, X, Gain, Bias, mean, rstd, epsilon, K, rcpK, relu);
    }
    else
    {
        //if      (K >= 1024)
        //    layer_norm_dx_NC<B,F,float,1024><<<N,1024,0,stream>>>(dx, (const B*)dx, x, mean, rstd, epsilon, K, rcpK);
        if (K >=  256)
            layer_norm_dx_NC<B,F,float, 256><<<N, 256,0,stream>>>(dx, (const B*)dy, x, g, b, mean, rstd, epsilon, K, rcpK, relu);
        else
            layer_norm_dx_NC<B,F,float,  64><<<N,  64,0,stream>>>(dx, (const B*)dy, x, g, b, mean, rstd, epsilon, K, rcpK, relu);
    }
    return true; // TODO
}

template bool LayerNormBackward_NC<float,float,float4,float4>(CUstream stream, int SMs, float* dx, float* dg, float* db, const float* dy, const float* x, const float* g, const float* b, const float* mean, const float* rstd, float epsilon, int K, int N, float rcpK, int relu);
template bool LayerNormBackward_NC<ehalf,ehalf,ehalf4,ehalf4>(CUstream stream, int SMs, ehalf* dx, float* dg, float* db, const ehalf* dy, const ehalf* x, const float* g, const float* b, const float* mean, const float* rstd, float epsilon, int K, int N, float rcpK, int relu);
template bool LayerNormBackward_NC<bhalf,bhalf,bhalf4,bhalf4>(CUstream stream, int SMs, bhalf* dx, float* dg, float* db, const bhalf* dy, const bhalf* x, const float* g, const float* b, const float* mean, const float* rstd, float epsilon, int K, int N, float rcpK, int relu);

template bool LayerNormBackward_NC<float,ehalf,float4,ehalf4>(CUstream stream, int SMs, float* dx, float* dg, float* db, const float* dy, const ehalf* x, const float* g, const float* b, const float* mean, const float* rstd, float epsilon, int K, int N, float rcpK, int relu);
template bool LayerNormBackward_NC<float,bhalf,float4,bhalf4>(CUstream stream, int SMs, float* dx, float* dg, float* db, const float* dy, const bhalf* x, const float* g, const float* b, const float* mean, const float* rstd, float epsilon, int K, int N, float rcpK, int relu);


#endif // GOOGLE_CUDA
