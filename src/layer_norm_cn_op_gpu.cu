
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "ew_op_gpu.h"
//#include <stdio.h>


// mean = mean(x, axis=0)
template <typename T, int UNROLL, int THREADS>
__global__ void __launch_bounds__(THREADS) layer_norm_mean_CN(
          float*              Mean,
    const     T* __restrict__ X,
    int K, int N, float rcpK)
{
    __shared__ float4 Share4[THREADS];
    float* Share = (float*)Share4;

    int tid   = threadIdx.x;
    int idx_K = blockIdx.x;
    int idx_N = blockIdx.y;

    int tid16 = tid >> 4;
    int tid15 = tid & 15;

    int k = idx_K*UNROLL*(THREADS/16) + tid16;
    int n = idx_N*16 + tid15;

    int  N4  = N >> 2;
    bool bn  = n < N4;

    int xi  = k*N4 + n;
    int inc = N4 * (THREADS/16);

    float4 mean4;
    ew_zero(mean4);
    #pragma unroll 4
    for (int j = 0; j < UNROLL; j++)
    {
        float4 x = load(X, xi, bn && k < K);

        mean4 = ew_add(mean4, x);

        k  += (THREADS/16);
        xi += inc;
    }
    Share4[(tid16 << 4) + tid15] = mean4;

    __syncthreads();
    int tid32 = tid >> 5;
    int tid31 = tid & 31;

    if (tid32 == 0)
    {
        n = idx_N*64 + tid31;
        Mean += n;

        float mean = 0.0f;
        #pragma unroll
        for (int i = 0; i < (THREADS/16); i++)
            mean += Share[tid31 + i*64];
        mean *= rcpK;

        if (n < N)
            //*Mean = mean;
            atomicAdd(Mean, mean);
    }
    else if (tid32 == 3)
    {
        n = idx_N*64 + tid31+32;
        Mean += n;

        float mean = 0.0f;
        #pragma unroll
        for (int i = 0; i < (THREADS/16); i++)
            mean += Share[tid31 + i*64 + 32];
        mean *= rcpK;

        if (n < N)
            //*Mean = mean;
            atomicAdd(Mean, mean);
    }
}

// var = var(x, axis=0)
template <typename T, int UNROLL, int THREADS>
__global__ void __launch_bounds__(THREADS) layer_norm_var_CN(
           float*              Var,
    const      T* __restrict__ X,
    const float4* __restrict__ Mean,
    int K, int N, float rcpK)
{
    __shared__ float4 Share4[THREADS];
    float* Share = (float*)Share4;

    int tid   = threadIdx.x;
    int idx_K = blockIdx.x;
    int idx_N = blockIdx.y;

    int tid16 = tid >> 4;
    int tid15 = tid & 15;

    int k = idx_K*UNROLL*(THREADS/16) + tid16;
    int n = idx_N*16 + tid15;

    int  N4  = N >> 2;
    bool bn  = n < N4;

    int xi  = k*N4 + n;
    int inc = N4 * (THREADS/16);

    float4 mean = load(Mean, n, bn);

    float4 var4;
    ew_zero(var4);
    #pragma unroll 4
    for (int j = 0; j < UNROLL; j++)
    {
        float4 x = load(X, xi, bn && k < K);

        // var4 += (x - mean)**2
        if (k < K)
            var4 = ew_add(var4, ew_sqr(ew_sub(x, mean)));

        k  += (THREADS/16);
        xi += inc;
    }
    Share4[(tid16 << 4) + tid15] = var4;

    __syncthreads();
    int tid32 = tid >> 5;
    int tid31 = tid & 31;

    if (tid32 == 0)
    {
        n = idx_N*64 + tid31;
        Var += n;

        float var = 0.0f;
        #pragma unroll
        for (int i = 0; i < (THREADS/16); i++)
            var += Share[tid31 + i*64];
        var *= rcpK;

        if (n < N)
            //*Var = var;
            atomicAdd(Var, var);
    }
    else if (tid32 == 3)
    {
        n = idx_N*64 + tid31+32;
        Var += n;

        float var = 0.0f;
        #pragma unroll
        for (int i = 0; i < (THREADS/16); i++)
            var += Share[tid31 + i*64 + 32];
        var *= rcpK;

        if (n < N)
            //*Var = var;
            atomicAdd(Var, var);
    }
}

// xstdr = rcp(sqrt(xvar + epsilon))
// xhat  = xmean * xstdr
// y     = xhat*g + b
template <typename T, int UNROLL>
__global__ void __launch_bounds__(32) layer_norm_CN(
               T*              Y,
    const      T* __restrict__ X,
    const float4* __restrict__ Mean,
    const float4* __restrict__ Var,
    const  float* __restrict__ G,
    const  float* __restrict__ B,
    int K, int N, float epsilon, int relu)
{
    __shared__ float Gain[UNROLL*2];
    __shared__ float Bias[UNROLL*2];

    int tid   = threadIdx.x;
    int idx_K = blockIdx.x * UNROLL*2;
    int idx_N = blockIdx.y * 16;

    // load gain/bias for this K-block
    int ki = idx_K + tid;
    if (tid < UNROLL*2 && ki < K)
    {
        Gain[tid] = G[ki];
        Bias[tid] = B[ki];
    }

    int tid16 = tid >> 4;
    int tid15 = tid & 15;

    int k = idx_K + tid16;
    int n = idx_N + tid15;

    int  N4 = N  >> 2;
    bool bn = n < N4;

    int xi  = k*N4 + n;
    int inc = N4 * 2;

    float4 var  = load(Var,  n, bn);
    float4 mean = load(Mean, n, bn);

    // rstd = 1 / sqrt(var + epsilon)
    // asm("and.b32 %0, %0, 0xffffc000;" : "+f"(var.x) : );
    // asm("and.b32 %0, %0, 0xffffc000;" : "+f"(var.y) : );
    // asm("and.b32 %0, %0, 0xffffc000;" : "+f"(var.z) : );
    // asm("and.b32 %0, %0, 0xffffc000;" : "+f"(var.w) : );
    float4 rstd = ew_rsqrt(ew_add(var, epsilon));

    #pragma unroll 4
    for (int j = 0; j < UNROLL; j++)
    {
        bool bnk = bn && k < K;
        float4 x = load(X, xi, bnk);

        float g = Gain[tid16];
        float b = Bias[tid16];

        // xhat = (x - mean) / sqrt(var + epsilon)
        //   y  = g * xhat + b
        float4 xhat = ew_mul(ew_sub(x, mean), rstd);
        float4    y = ew_add(ew_mul(xhat, g), b);

        if (relu)
            y = ew_relu(y);

        store_f(Y, y, xi, bnk);

        k     += 2;
        tid16 += 2;
        xi    += inc;
    }
}
template <typename T, typename V>
bool LayerNormForward_CN(CUstream stream, int SMs,
              T* y,
          float* mean,
          float* var,
    const     T* x,
    const float* g,
    const float* b,
    float epsilon, int K, int N, float rcpK, int relu)
{
    const V* X = (const V*)x;
    const float4* Mean = (const float4*)mean;
    const float4* Var  = (const float4*)var;

    cuMemsetD32Async((CUdeviceptr)mean, 0, N, stream);
    cuMemsetD32Async((CUdeviceptr)var,  0, N, stream);

    int gridN = (N >> 6) + ((N &  63) != 0);
    int gridK = (K >> 3) + ((K &   7) != 0);

    if ((K >> 8) < (SMs >> 1))
    {
        dim3 grid((K >> 7) + ((K & 127) != 0), gridN);
        layer_norm_mean_CN<V,16,128><<<grid,128,0,stream>>>(mean, X, K, N, rcpK);
        layer_norm_var_CN <V,16,128><<<grid,128,0,stream>>>(var, X, Mean, K, N, rcpK);
    }
    else
    {
        dim3 grid((K >> 8) + ((K & 255) != 0), gridN);
        layer_norm_mean_CN<V,16,256><<<grid,256,0,stream>>>(mean, X, K, N, rcpK);
        layer_norm_var_CN <V,16,256><<<grid,256,0,stream>>>(var, X, Mean, K, N, rcpK);
    }
    dim3 grid(gridK, gridN);
    layer_norm_CN<V,4><<<grid,32, 0,stream>>>((V*)y, X, Mean, Var, g, b, K, N, epsilon, relu);
    return true; // TODO
}
template bool LayerNormForward_CN<float,float4>(CUstream stream, int SMs, float* y, float* mean, float* rstd, const float* x, const float* g, const float* b, float epsilon, int K, int N, float rcpK, int relu);
template bool LayerNormForward_CN<ehalf,ehalf4>(CUstream stream, int SMs, ehalf* y, float* mean, float* rstd, const ehalf* x, const float* g, const float* b, float epsilon, int K, int N, float rcpK, int relu);
template bool LayerNormForward_CN<bhalf,bhalf4>(CUstream stream, int SMs, bhalf* y, float* mean, float* rstd, const bhalf* x, const float* g, const float* b, float epsilon, int K, int N, float rcpK, int relu);


// dg = sum(dy * xhat(x), axis=1)
// db = sum(dy, axis=1)
template <typename B, typename F>
__global__ void __launch_bounds__(128) layer_norm_dg_db_CN(
           float*              DG,
           float*              DB,
    const      B* __restrict__ DY,
    const      F* __restrict__ X,
    const  float* __restrict__ Gain,
    const  float* __restrict__ Bias,
    const float4* __restrict__ Mean,
    const float4* __restrict__ Var,
    float epsilon, int K, int N, int relu)
{
    __shared__ float gain[8];
    __shared__ float bias[8];

    int tid   = threadIdx.x;
    int idx_K = blockIdx.x * 8;

    // load gain/bias for this K-block
    int ki = idx_K + tid;
    if (relu && tid < 8 && ki < K)
    {
        gain[tid] = Gain[ki];
        bias[tid] = Bias[ki];
    }
    int tid16 = tid >> 4;
    int tid15 = tid & 15;
    int k     = idx_K + tid16;

    __syncthreads();

    if (k < K)
    {
        int N4 = N >> 2;
        int xi = k*N4;
        X  += xi;
        DY += xi;

        float4 dg4, db4;
        ew_zero(dg4);
        ew_zero(db4);
        for (int n = tid15; n < N4; n += 16)
        {
            float4 x    = load(X,    n);
            float4 dy   = load(DY,   n);
            float4 var  = load(Var,  n);
            float4 mean = load(Mean, n);

            // rstd = 1 / sqrt(var + epsilon)
            // xhat = (x - mean) * rstd
            // asm("and.b32 %0, %0, 0xffffc000;" : "+f"(var.x) : );
            // asm("and.b32 %0, %0, 0xffffc000;" : "+f"(var.y) : );
            // asm("and.b32 %0, %0, 0xffffc000;" : "+f"(var.z) : );
            // asm("and.b32 %0, %0, 0xffffc000;" : "+f"(var.w) : );
            float4 rstd = ew_rsqrt(ew_add(var, epsilon));
            float4 xhat = ew_mul(ew_sub(x, mean), rstd);

            if (relu)
            {
                float g = gain[tid16];
                float b = bias[tid16];
                dy = ew_relu_grad(dy, ew_add(ew_mul(xhat, g), b));
            }

            dg4 = ew_add(ew_mul(dy, xhat), dg4);
            db4 = ew_add(dy, db4);
        }
        float dg = ew_sum(dg4);
        float db = ew_sum(db4);

        // reduce each half warp
        for (int i = 8; i > 0; i >>= 1)
        {
            dg += __shfl_xor(dg, i);
            db += __shfl_xor(db, i);
        }
        if (tid15 == 0)
        {
            DG[k] = dg;
            DB[k] = db;
        }
    }
}

// dy    = dy * g
// sum1  = sum(xhat * dy, axis=0)
// sum2  = sum(dy, axis=0)
template <typename B, typename F, int UNROLL, int THREADS>
__global__ void __launch_bounds__(THREADS) layer_norm_dx_sum_CN(
           float*              Sum1,
           float*              Sum2,
    const      B* __restrict__ DY,
    const      F* __restrict__ X,
    const  float* __restrict__ Gain,
    const  float* __restrict__ Bias,
    const float4* __restrict__ Mean,
    const float4* __restrict__ Var,
    float epsilon, int K, int N, int relu)
{
    __shared__ float4 Sum1f4[THREADS];
    __shared__ float4 Sum2f4[THREADS];
    __shared__ float gain[UNROLL*(THREADS/16)];
    __shared__ float bias[UNROLL*(THREADS/16)];
    float* Sum1f1 = (float*)Sum1f4;
    float* Sum2f1 = (float*)Sum2f4;

    int tid   = threadIdx.x;
    int idx_K = blockIdx.x * UNROLL*(THREADS/16);
    int idx_N = blockIdx.y * 16;

    // load gain/bias for this K-block
    int ki = idx_K + tid;
    if (tid < UNROLL*(THREADS/16) && ki < K)
    {
        gain[tid] = Gain[ki];
        bias[tid] = Bias[ki];
    }
    __syncthreads();

    int tid16 = tid >> 4;
    int tid15 = tid & 15;
    int gbi   = tid16;

    int k = idx_K + tid16;
    int n = idx_N + tid15;

    int  N4 = N  >> 2;
    bool bn = n < N4;

    int xi  = k*N4 + n;
    int inc = N4 * (THREADS/16);

    float4 var  = load(Var,  n, bn);
    float4 mean = load(Mean, n, bn);

    // rstd = 1 / sqrt(var + epsilon)
    // asm("and.b32 %0, %0, 0xffffc000;" : "+f"(var.x) : );
    // asm("and.b32 %0, %0, 0xffffc000;" : "+f"(var.y) : );
    // asm("and.b32 %0, %0, 0xffffc000;" : "+f"(var.z) : );
    // asm("and.b32 %0, %0, 0xffffc000;" : "+f"(var.w) : );
    float4 rstd = ew_rsqrt(ew_add(var, epsilon));

    float4 sum1, sum2;
    ew_zero(sum1);
    ew_zero(sum2);
    #pragma unroll 2
    for (int j = 0; j < UNROLL; j++)
    {
        bool bnk = bn & k < K;
        float4  x = load( X, xi, bnk);
        float4 dy = load(DY, xi, bnk);
        float g = gain[gbi];
        float b = bias[gbi];

        float4 xhat = ew_mul(ew_sub(x, mean), rstd);
        if (relu)
            dy = ew_relu_grad(dy, ew_add(ew_mul(xhat, g), b));
        dy = ew_mul(dy, g);

        if (bnk)
        {
            sum1 = ew_add(sum1, ew_mul(dy, xhat));
            sum2 = ew_add(sum2, dy);
        }
        k   += (THREADS/16);
        gbi += (THREADS/16);
        xi  += inc;
    }
    int si = (tid16 << 4) + tid15;
    Sum1f4[si] = sum1;
    Sum2f4[si] = sum2;
    __syncthreads();

    int tid32 = tid >> 5;
    int tid31 = tid & 31;
    n = idx_N*4 + tid31;

    if (tid32 == 0)
    {
        Sum1 += n;

        float sum1 = 0.0f;
        #pragma unroll
        for (int i = 0; i < (THREADS/16); i++)
            sum1 += Sum1f1[tid31 + i*64];

        if (n < N)
            atomicAdd(Sum1, sum1);
    }
    else if (tid32 == 1)
    {
        n += 32;
        Sum1 += n;

        float sum1 = 0.0f;
        #pragma unroll
        for (int i = 0; i < (THREADS/16); i++)
            sum1 += Sum1f1[tid31 + i*64 + 32];

        if (n < N)
            atomicAdd(Sum1, sum1);
    }
    else if (tid32 == 2)
    {
        Sum2 += n;

        float sum2 = 0.0f;
        #pragma unroll
        for (int i = 0; i < (THREADS/16); i++)
            sum2 += Sum2f1[tid31 + i*64];

        if (n < N)
            atomicAdd(Sum2, sum2);
    }
    else if (tid32 == 3)
    {
        n += 32;
        Sum2 += n;

        float sum2 = 0.0f;
        #pragma unroll
        for (int i = 0; i < (THREADS/16); i++)
            sum2 += Sum2f1[tid31 + i*64 + 32];

        if (n < N)
            atomicAdd(Sum2, sum2);
    }
}

// dy = dy * g
// dx = (dy - ((xhat * sum1 + sum2) * rcpK)) * xstdr
template <typename B, typename F, int UNROLL>
__global__ void __launch_bounds__(32) layer_norm_dx_CN(
               B*              DX,
    const      B* __restrict__ DY,
    const      F* __restrict__ X,
    const  float* __restrict__ Gain,
    const  float* __restrict__ Bias,
    const float4* __restrict__ Mean,
    const float4* __restrict__ Var,
    const float4* __restrict__ Sum1,
    const float4* __restrict__ Sum2,
    float epsilon, int K, int N, float rcpK, int relu)
{
    __shared__ float gain[UNROLL*2];
    __shared__ float bias[UNROLL*2];

    int tid   = threadIdx.x;
    int idx_K = blockIdx.x * UNROLL*2;
    int idx_N = blockIdx.y * 16;

    // load gain/bias for this K-block
    int ki = idx_K + tid;
    if (tid < UNROLL*2 && ki < K)
    {
        gain[tid] = Gain[ki];
        bias[tid] = Bias[ki];
    }

    int tid16 = tid >> 4;
    int tid15 = tid & 15;

    int k = idx_K + tid16;
    int n = idx_N + tid15;

    int  N4 = N  >> 2;
    bool bn = n < N4;

    int xi  = k*N4 + n;
    int inc = N4 * 2;

    float4 var  = load(Var,  n, bn);
    float4 mean = load(Mean, n, bn);
    float4 sum1 = load(Sum1, n, bn);
    float4 sum2 = load(Sum2, n, bn);

    // rstd = 1 / sqrt(var + epsilon)
    // asm("and.b32 %0, %0, 0xffffc000;" : "+f"(var.x) : );
    // asm("and.b32 %0, %0, 0xffffc000;" : "+f"(var.y) : );
    // asm("and.b32 %0, %0, 0xffffc000;" : "+f"(var.z) : );
    // asm("and.b32 %0, %0, 0xffffc000;" : "+f"(var.w) : );
    float4 rstd = ew_rsqrt(ew_add(var, epsilon));
    #pragma unroll 4
    for (int j = 0; j < UNROLL; j++)
    {
        bool bnk = bn && k < K;
        float4  x = load( X, xi, bnk);
        float4 dy = load(DY, xi, bnk);
        float   g = gain[tid16];
        float   b = bias[tid16];

        float4 xhat = ew_mul(ew_sub(x, mean), rstd);
        if (relu)
            dy = ew_relu_grad(dy, ew_add(ew_mul(xhat, g), b));
        dy = ew_mul(dy, g);

        // dx = (dy - ((xhat * sum1 + sum2) * rcpK)) * rstd;
        float4 dx = ew_mul(ew_sub(dy, ew_mul(ew_add(ew_mul(xhat, sum1), sum2), rcpK)), rstd);

        store_g(DX, dx, xi, bnk);
        k     += 2;
        tid16 += 2;
        xi    += inc;
    }
}

template <typename B, typename F, typename VB, typename VF>
bool LayerNormBackward_CN(CUstream stream, int SMs,
              B* dx,
          float* dg,
          float* db,
          float* sum1,
          float* sum2,
    const     B* dy,
    const     F* x,
    const float* g,
    const float* b,
    const float* mean,
    const float* var,
    float epsilon, int K, int N, float rcpK, int relu)
{
    int gridK8   = (K >> 3) + ((K &   7) != 0);
    int gridK256 = (K >> 8) + ((K & 255) != 0);
    int gridN64  = (N >> 6) + ((N &  63) != 0);
    dim3 grid8(  gridK8,   gridN64, 1);
    dim3 grid256(gridK256, gridN64, 1);

          VB* DX = (      VB*)dx;
    const VB* DY = (const VB*)dy;
    const VF*  X = (const VF*)x;

    const float4* Mean = (const float4*)mean;
    const float4* Var  = (const float4*)var;
    const float4* Sum1 = (const float4*)sum1;
    const float4* Sum2 = (const float4*)sum2;

    cuMemsetD32Async((CUdeviceptr)sum1, 0, N, stream);
    cuMemsetD32Async((CUdeviceptr)sum2, 0, N, stream);

    layer_norm_dg_db_CN <VB,VF       ><<<gridK8 ,128,0,stream>>>(dg, db, DY, X, g, b, Mean, Var, epsilon, K, N, relu);
    layer_norm_dx_sum_CN<VB,VF,16,256><<<grid256,256,0,stream>>>(sum1, sum2, DY, X, g, b, Mean, Var, epsilon, K, N, relu);
    layer_norm_dx_CN    <VB,VF, 4    ><<<grid8,   32,0,stream>>>(DX, DY, X, g, b, Mean, Var, Sum1, Sum2, epsilon, K, N, rcpK, relu);

    return true; // TODO
}

template bool LayerNormBackward_CN<float,float,float4,float4>(CUstream stream, int SMs, float* dx, float* dg, float* db, float* sum1, float* sum2, const float* dy, const float* x, const float* g, const float* b, const float* mean, const float* rstd, float epsilon, int K, int N, float rcpK, int relu);
template bool LayerNormBackward_CN<ehalf,ehalf,ehalf4,ehalf4>(CUstream stream, int SMs, ehalf* dx, float* dg, float* db, float* sum1, float* sum2, const ehalf* dy, const ehalf* x, const float* g, const float* b, const float* mean, const float* rstd, float epsilon, int K, int N, float rcpK, int relu);
template bool LayerNormBackward_CN<bhalf,bhalf,bhalf4,bhalf4>(CUstream stream, int SMs, bhalf* dx, float* dg, float* db, float* sum1, float* sum2, const bhalf* dy, const bhalf* x, const float* g, const float* b, const float* mean, const float* rstd, float epsilon, int K, int N, float rcpK, int relu);

template bool LayerNormBackward_CN<float,ehalf,float4,ehalf4>(CUstream stream, int SMs, float* dx, float* dg, float* db, float* sum1, float* sum2, const float* dy, const ehalf* x, const float* g, const float* b, const float* mean, const float* rstd, float epsilon, int K, int N, float rcpK, int relu);
template bool LayerNormBackward_CN<float,bhalf,float4,bhalf4>(CUstream stream, int SMs, float* dx, float* dg, float* db, float* sum1, float* sum2, const float* dy, const bhalf* x, const float* g, const float* b, const float* mean, const float* rstd, float epsilon, int K, int N, float rcpK, int relu);


// Sparse Projection Code

template <typename T, typename V, int SHFT>
__global__ void __launch_bounds__(128) gather_scatter(
            T*              Z,
    const   T* __restrict__ X,
    const int* __restrict__ Lut,
    int K, int N)
{
    int tid   = threadIdx.x;
    int idx_K = blockIdx.x;
    int idx_N = blockIdx.y;

    int tidK = tid >> SHFT;
    int tidN = tid & ((1<<SHFT)-1);

    int zk = (idx_K << (7-SHFT)) + tidK;
    int  n = (idx_N <<    SHFT)  + tidN;

    if (zk < K && n < N)
    {
        int xk = load(Lut, zk);

        int zi = zk*N + n;
        int xi = xk*N + n;

        V x = load(X, xi, xk >= 0);

        store(Z, x, zi);
    }
}
template <typename T, typename V, int SHFT>
__global__ void __launch_bounds__(128) scatter_add(
            T*              Z, // large tensor
    const   T* __restrict__ X, // large tensor
    const   T* __restrict__ Y, // small tensor
    const int* __restrict__ Lut,
    int K, int N)
{
    int tid   = threadIdx.x;
    int idx_K = blockIdx.x;
    int idx_N = blockIdx.y;

    int tidK = tid >> SHFT;
    int tidN = tid & ((1<<SHFT)-1);

    int yk = (idx_K << (7-SHFT)) + tidK;
    int  n = (idx_N <<    SHFT) +  tidN;

    if (yk < K && n < N)
    {
        int xk = load(Lut, yk);

        int yi = yk*N + n;
        int xi = xk*N + n;

        V y = load(Y, yi);
        V x = load(X, xi);

        store(Z, ew_add(x, y), xi);
    }
}
template <typename T, typename V, int SHFT>
__global__ void __launch_bounds__(128) scatter_mul(
            T*              Z, // large tensor
    const   T* __restrict__ X, // large tensor
    const   T* __restrict__ Y, // small tensor
    const int* __restrict__ Lut,
    int K, int N)
{
    int tid   = threadIdx.x;
    int idx_K = blockIdx.x;
    int idx_N = blockIdx.y;

    int tidK = tid >> SHFT;
    int tidN = tid & ((1<<SHFT)-1);

    int xk = (idx_K << (7-SHFT)) + tidK;
    int  n = (idx_N <<    SHFT) +  tidN;

    if (xk < K && n < N)
    {
        int yk = load(Lut, xk);

        int xi = xk*N + n;
        int yi = yk*N + n;

        V x = load(X, xi);
        V y = load(Y, yi, yk >= 0);
        V z = yk >= 0 ? ew_mul(x, y) : x; // pass through if unmapped

        store(Z, z, xi);
    }
}
template <typename T, typename V, int SHFT>
__global__ void __launch_bounds__(128) sparse_mul_grad(
            T*              DX, // large tensor
            T*              DY, // small tensor
    const   T* __restrict__ DZ, // large tensor (same pointer as DX)
    const   T* __restrict__ X,  // large tensor
    const   T* __restrict__ Y,  // small tensor
    const int* __restrict__ Lut,
    int K, int N)
{
    int tid   = threadIdx.x;
    int idx_K = blockIdx.x;
    int idx_N = blockIdx.y;

    int tidK = tid >> SHFT;
    int tidN = tid & ((1<<SHFT)-1);

    int yk = (idx_K << (7-SHFT)) + tidK;
    int  n = (idx_N <<    SHFT) +  tidN;

    if (yk < K && n < N)
    {
        int xk = load(Lut, yk);

        int yi = yk*N + n;
        int xi = xk*N + n;

        V y  = load(Y,  yi);
        V x  = load(X,  xi);
        V dz = load(DZ, xi);

        store(DX, ew_mul(dz, y), xi);
        store(DY, ew_mul(dz, x), yi);
    }
}

#define OP_GAT 0
#define OP_SCT 1
#define OP_ADD 2
#define OP_MUL 3

template <typename T, typename V4, typename V8>
bool SparseOp(CUstream stream,
            T* z,
    const   T* x,
    const   T* y,
    const int* lut,
    int op, int K, int N)
{
    int gridN = (N >> 6) + ((N & 63) != 0);

    if (sizeof(T) == 2 && (N & 7) == 0)
    {
              V8* Z = (      V8*)z;
        const V8* X = (const V8*)x;
        const V8* Y = (const V8*)y;

        // blockK = 128 / 8 = 16
        int gridK = (K >> 4) + ((K & 15) != 0);
        dim3 grid(gridK, gridN, 1);
        switch(op)
        {
            case OP_GAT: gather_scatter<V8,float8,3><<<grid,128,0,stream>>>(Z, X,    lut, K, N>>3); break;
            case OP_SCT: gather_scatter<V8,float8,3><<<grid,128,0,stream>>>(Z, X,    lut, K, N>>3); break;
            case OP_ADD:    scatter_add<V8,float8,3><<<grid,128,0,stream>>>(Z, X, Y, lut, K, N>>3); break;
            case OP_MUL:    scatter_mul<V8,float8,3><<<grid,128,0,stream>>>(Z, X, Y, lut, K, N>>3); break;
        }
    }
    else if ((N & 3) == 0)
    {
              V4* Z = (      V4*)z;
        const V4* X = (const V4*)x;
        const V4* Y = (const V4*)y;

        // blockK = 128 / 16 = 8
        int gridK = (K >> 3) + ((K & 7) != 0);
        dim3 grid(gridK, gridN, 1);
        switch(op)
        {
            case OP_GAT: gather_scatter<V4,float4,4><<<grid,128,0,stream>>>(Z, X,    lut, K, N>>2); break;
            case OP_SCT: gather_scatter<V4,float4,4><<<grid,128,0,stream>>>(Z, X,    lut, K, N>>2); break;
            case OP_ADD:    scatter_add<V4,float4,4><<<grid,128,0,stream>>>(Z, X, Y, lut, K, N>>2); break;
            case OP_MUL:    scatter_mul<V4,float4,4><<<grid,128,0,stream>>>(Z, X, Y, lut, K, N>>2); break;
        }
    }
    return true; // TODO
}

template <typename T, typename V4, typename V8>
bool SparseMulGrad(CUstream stream,
            T* dx,
            T* dy,
    const   T* dz,
    const   T* x,
    const   T* y,
    const int* lut,
    int K, int N)
{
    int gridN = (N >> 6) + ((N & 63) != 0);

    if (sizeof(T) == 2 && (N & 7) == 0)
    {
              V8* DX = (      V8*)dx;
              V8* DY = (      V8*)dy;
        const V8* DZ = (const V8*)dz;
        const V8*  X = (const V8*)x;
        const V8*  Y = (const V8*)y;

        // blockK = 128 / 8 = 16
        int gridK = (K >> 4) + ((K & 15) != 0);
        dim3 grid(gridK, gridN, 1);

        sparse_mul_grad<V8,float8,3><<<grid,128,0,stream>>>(DX, DY, DZ, X, Y, lut, K, N>>3);
    }
    else if ((N & 3) == 0)
    {
              V4* DX = (      V4*)dx;
              V4* DY = (      V4*)dy;
        const V4* DZ = (const V4*)dz;
        const V4*  X = (const V4*)x;
        const V4*  Y = (const V4*)y;

        // blockK = 128 / 16 = 8
        int gridK = (K >> 3) + ((K & 7) != 0);
        dim3 grid(gridK, gridN, 1);
        sparse_mul_grad<V4,float4,4><<<grid,128,0,stream>>>(DX, DY, DZ, X, Y, lut, K, N>>2);
    }
    return true; // TODO
}

template bool SparseOp<float,float4,float8>(CUstream stream, float* z, const float* x, const float* y, const int* lut, int op, int K, int N);
template bool SparseOp<ehalf,ehalf4,ehalf8>(CUstream stream, ehalf* z, const ehalf* x, const ehalf* y, const int* lut, int op, int K, int N);
template bool SparseOp<bhalf,bhalf4,bhalf8>(CUstream stream, bhalf* z, const bhalf* x, const bhalf* y, const int* lut, int op, int K, int N);

template bool SparseMulGrad<float,float4,float8>(CUstream stream, float* dx, float* dy, const float* dz, const float* x, const float* y, const int* lut, int K, int N);
template bool SparseMulGrad<ehalf,ehalf4,ehalf8>(CUstream stream, ehalf* dx, ehalf* dy, const ehalf* dz, const ehalf* x, const ehalf* y, const int* lut, int K, int N);
template bool SparseMulGrad<bhalf,bhalf4,bhalf8>(CUstream stream, bhalf* dx, bhalf* dy, const bhalf* dz, const bhalf* x, const bhalf* y, const int* lut, int K, int N);

#endif // GOOGLE_CUDA

// cuobjdump -xelf blocksparse_ops.5.sm_60.cubin blocksparse_ops.so
// cuobjdump -xelf blocksparse_ops.6.sm_61.cubin blocksparse_ops.so

// nvdisasm -c -raw blocksparse_ops.5.sm_60.cubin  > blocksparse_ops.5.sm_60.sass
// nvdisasm -c -raw blocksparse_ops.6.sm_61.cubin  > blocksparse_ops.6.sm_61.sass


