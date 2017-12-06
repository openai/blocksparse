
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "ew_op_gpu.h"

__device__ __forceinline__ int div64(int value, int magic, int shift)
{
    // if the divisor is a power of 2 the magic will be 1 and it's just a simple right shift
    // Otherwise multiply by magic and right shift just the high bits
    int result;
    asm("{\n\t"
        ".reg .pred p;\n\t"
        ".reg .u64 res64;\n\t"
        ".reg .u32 lo32, hi32;\n\t"
        "setp.ne.s32 p, %2, 1;\n\t"
        "mul.wide.u32 res64, %1, %2;\n\t"
        "mov.b64 {lo32, hi32}, res64;\n\t"
        "selp.u32 hi32, hi32, %1, p;\n\t"
        "shr.u32 %0, hi32, %3;\n\t"
        "}" : "=r"(result) : "r"(value), "r"(magic), "r"(shift));
    return result;
}

// y = g * (x - mean) / sqrt(var + eps) + b
template <typename T, int THREADS>
__global__ void __launch_bounds__(THREADS) batchnorm_inference_ncdhw(
              T*              Y,
    const float* __restrict__ M,
    const float* __restrict__ V,
    const     T* __restrict__ X,
    const float* __restrict__ G,
    const float* __restrict__ B,
    int CDHW, int DHW, float epsilon)
{
    const int tid = threadIdx.x;
    const int c   = blockIdx.x;
    const int n   = blockIdx.y;

    int offset = n * CDHW + c * DHW;

    float g = G[c];
    float b = B[c];

    float mean = M[c];
    float var  = V[c];

    float rstdg = rsqrtf(var + epsilon) * g;

    X += offset;
    Y += offset;
    for (int i = tid; i < DHW; i += THREADS)
    {
        float x = load(X, i);
        float y = (x - mean) * rstdg + b;
        store(Y, y, i);
    }
}
template <typename T>
bool BatchNormNCDHW_Inference(CUstream stream,
              T* y,
    const float* m,
    const float* v,
    const     T* x,
    const float* g,
    const float* b,
    int N, int C, int DHW, float epsilon)
{
    int CDHW = C*DHW;
    dim3 grid(C, N, 1);
    if      (DHW < 128*8)
        batchnorm_inference_ncdhw<T, 32><<<grid,  32, 0, stream>>>(y, m, v, x, g, b, CDHW, DHW, epsilon);
    else if (DHW < 512*8)
        batchnorm_inference_ncdhw<T,128><<<grid, 128, 0, stream>>>(y, m, v, x, g, b, CDHW, DHW, epsilon);
    else
        batchnorm_inference_ncdhw<T,512><<<grid, 512, 0, stream>>>(y, m, v, x, g, b, CDHW, DHW, epsilon);
    return true; // TODO
}
template bool BatchNormNCDHW_Inference<float>(CUstream stream,
          float* y,
    const float* m,
    const float* v,
    const float* x,
    const float* g,
    const float* b,
    int N, int C, int DHW, float epsilon);

template bool BatchNormNCDHW_Inference<ehalf>(CUstream stream,
          ehalf* y,
    const float* m,
    const float* v,
    const ehalf* x,
    const float* g,
    const float* b,
    int N, int C, int DHW, float epsilon);

template bool BatchNormNCDHW_Inference<bhalf>(CUstream stream,
          bhalf* y,
    const float* m,
    const float* v,
    const bhalf* x,
    const float* g,
    const float* b,
    int N, int C, int DHW, float epsilon);


// mean = sum(x, axis=(0,2,...)) / NDHW
// var  = sum((x - mean)**2, axis=(0,2,...)) / NDHW
// y    = g * (x - mean) / sqrt(var + eps) + b
template <typename T, int THREADS>
__global__ void __launch_bounds__(THREADS) batchnorm_forward_ncdhw(
              T*              Y,
          float*              M,
          float*              V,
    const     T* __restrict__ X,
    const float* __restrict__ G,
    const float* __restrict__ B,
    int CDHW, int NDHW, int DHW, int magic_DHW, int shift_DHW, float rcpNDHW, float epsilon)
{
    __shared__ float Share[THREADS>>5];

    const int tid = threadIdx.x;
    const int c   = blockIdx.x;

    int offset = c * DHW;

    float g = G[c];
    float b = B[c];

    const T* X1 = X + offset;
    float mean = 0.0f;
    for (int ndhw = tid; ndhw < NDHW; ndhw += THREADS)
    {
        int n   = div64(ndhw, magic_DHW, shift_DHW);
        int dhw = ndhw - n*DHW;
        int i   = n * CDHW + dhw;

        float x = load(X1, i);
        mean += x;
    }
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
        Share[tid] = mean * rcpNDHW;
    }
    __syncthreads();
    // broadcast result to all threads
    mean = Share[0];

    const T* X2 = X + offset;
    float var = 0.0f;
    for (int ndhw = tid; ndhw < NDHW; ndhw += THREADS)
    {
        int n   = div64(ndhw, magic_DHW, shift_DHW);
        int dhw = ndhw - n*DHW;
        int i   = n * CDHW + dhw;

        float x = load(X2, i);
        x -= mean;
        var += x*x;
    }
    // reduce within warp
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1)
        var += __shfl_xor(var, i);
    // first thread of each warp store to shared
    if ((tid & 31) == 0)
        Share[tid >> 5] = var;
    __syncthreads();
    if (tid < (THREADS>>5))
    {
        // first warp loads all prior reductions
        var = Share[tid];
        // reduce within this last warp
        #pragma unroll
        for (int i = (THREADS>>6); i > 0; i >>= 1)
            var += __shfl_xor(var, i);

        // outputs final reduction to shared
        if (tid == 0)
        {
            var *= rcpNDHW;
            M[c] = mean;
            V[c] = var;
            Share[0] = rsqrtf(var + epsilon) * g;
        }
    }
    __syncthreads();
    // broadcast result to all threads
    float rstdg = Share[0];

    X += offset;
    Y += offset;
    for (int ndhw = tid; ndhw < NDHW; ndhw += THREADS)
    {
        int n   = div64(ndhw, magic_DHW, shift_DHW);
        int dhw = ndhw - n*DHW;
        int i   = n * CDHW + dhw;
        float x = load(X, i);
        float y = (x - mean) * rstdg + b;

        store(Y, y, i);
    }
}
template <typename T>
bool BatchNormNCDHW_Forward(CUstream stream,
              T* y,
          float* m,
          float* v,
    const     T* x,
    const float* g,
    const float* b,
    int N, int C, int DHW, int magic_DHW, int shift_DHW, float epsilon)
{
    int NDHW = N*DHW;
    int CDHW = C*DHW;
    float rcpNDHW = 1.0f / (float)NDHW;
    if      (NDHW <  256*8)
        batchnorm_forward_ncdhw<T,  64><<<C,  64,0,stream>>>(y, m, v, x, g, b, CDHW, NDHW, DHW, magic_DHW, shift_DHW, rcpNDHW, epsilon);
    else if (NDHW < 1024*8)
        batchnorm_forward_ncdhw<T, 256><<<C, 256,0,stream>>>(y, m, v, x, g, b, CDHW, NDHW, DHW, magic_DHW, shift_DHW, rcpNDHW, epsilon);
    else
        batchnorm_forward_ncdhw<T,1024><<<C,1024,0,stream>>>(y, m, v, x, g, b, CDHW, NDHW, DHW, magic_DHW, shift_DHW, rcpNDHW, epsilon);
    return true; // TODO
}
template bool BatchNormNCDHW_Forward<float>(CUstream stream,
          float* y,
          float* m,
          float* v,
    const float* x,
    const float* g,
    const float* b,
    int N, int C, int DHW, int magic_DHW, int shift_DHW, float epsilon);

template bool BatchNormNCDHW_Forward<ehalf>(CUstream stream,
          ehalf* y,
          float* m,
          float* v,
    const ehalf* x,
    const float* g,
    const float* b,
    int N, int C, int DHW, int magic_DHW, int shift_DHW, float epsilon);

template bool BatchNormNCDHW_Forward<bhalf>(CUstream stream,
          bhalf* y,
          float* m,
          float* v,
    const bhalf* x,
    const float* g,
    const float* b,
    int N, int C, int DHW, int magic_DHW, int shift_DHW, float epsilon);



template <typename TX, typename TY, int THREADS>
__global__ void __launch_bounds__(THREADS) batchnorm_backward_ncdhw(
             TY*              DX,
          float*              DG,
          float*              DB,
    const    TY* __restrict__ DY,
    const    TX* __restrict__ X,
    const float* __restrict__ G,
    const float* __restrict__ M,
    const float* __restrict__ V,
    int CDHW, int NDHW, int DHW, int magic_DHW, int shift_DHW, float rcpNDHW, float epsilon)
{
    __shared__ float Share1[THREADS>>5];
    __shared__ float Share2[THREADS>>5];

    const int tid = threadIdx.x;
    const int c   = blockIdx.x;

    int offset = c * DHW;

    float    g = G[c];
    float mean = M[c];
    float  var = V[c];
    float rstd = rsqrtf(var + epsilon);

    const TX* X1 = X  + offset;
    const TY* Y1 = DY + offset;
    float dg = 0.0f;
    float db = 0.0f;
    for (int ndhw = tid; ndhw < NDHW; ndhw += THREADS)
    {
        int n   = div64(ndhw, magic_DHW, shift_DHW);
        int dhw = ndhw - n*DHW;
        int i   = n * CDHW + dhw;

        float  x = load(X1, i);
        float dy = load(Y1, i);

        dg += dy * (x - mean) * rstd;
        db += dy;
    }
    // reduce within warp
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1)
    {
        dg += __shfl_xor(dg, i);
        db += __shfl_xor(db, i);
    }
    // first thread of each warp store to shared
    if ((tid & 31) == 0)
    {
        Share1[tid >> 5] = dg;
        Share2[tid >> 5] = db;
    }
    __syncthreads();
    if (tid < (THREADS>>5))
    {
        // first warp loads all prior reductions
        dg = Share1[tid];
        db = Share2[tid];
        // reduce within this last warp
        #pragma unroll
        for (int i = (THREADS>>6); i > 0; i >>= 1)
        {
            dg += __shfl_xor(dg, i);
            db += __shfl_xor(db, i);
        }
        // outputs final reduction to shared
        if (tid == 0)
        {
            DG[c] = dg;
            DB[c] = db;
            Share1[0] = dg;
            Share2[0] = db;
        }
    }
    __syncthreads();
    // broadcast result to all threads
    dg = Share1[0];
    db = Share2[0];

    X  += offset;
    DY += offset;
    DX += offset;
    for (int ndhw = tid; ndhw < NDHW; ndhw += THREADS)
    {
        int n   = div64(ndhw, magic_DHW, shift_DHW);
        int dhw = ndhw - n*DHW;
        int i   = n * CDHW + dhw;

        float  x = load( X, i);
        float dy = load(DY, i);

        float xhat = (x - mean) * rstd;
        float xtmp = (xhat * dg + db) * rcpNDHW;
        float dx   = (dy - xtmp) * rstd * g;

        store(DX, dx, i);
    }
}
template <typename TX, typename TY>
bool BatchNormNCDHW_Backward(CUstream stream,
             TY* dx,
          float* dg,
          float* db,
    const    TY* dy,
    const    TX* x,
    const float* g,
    const float* m,
    const float* v,
    int N, int C, int DHW, int magic_DHW, int shift_DHW, float epsilon)
{
    int NDHW = N*DHW;
    int CDHW = C*DHW;
    float rcpNDHW = 1.0f / (float)NDHW;
    if      (NDHW <  256*8)
        batchnorm_backward_ncdhw<TX,TY,  64><<<C,  64,0,stream>>>(dx, dg, db, dy, x, g, m, v, CDHW, NDHW, DHW, magic_DHW, shift_DHW, rcpNDHW, epsilon);
    else if (NDHW < 1024*8)
        batchnorm_backward_ncdhw<TX,TY, 256><<<C, 256,0,stream>>>(dx, dg, db, dy, x, g, m, v, CDHW, NDHW, DHW, magic_DHW, shift_DHW, rcpNDHW, epsilon);
    else
        batchnorm_backward_ncdhw<TX,TY,1024><<<C,1024,0,stream>>>(dx, dg, db, dy, x, g, m, v, CDHW, NDHW, DHW, magic_DHW, shift_DHW, rcpNDHW, epsilon);
    return true; // TODO
}
template bool BatchNormNCDHW_Backward<float,float>(CUstream stream,
          float* dx,
          float* dg,
          float* db,
    const float* dy,
    const float* x,
    const float* g,
    const float* m,
    const float* v,
    int N, int C, int DHW, int magic_DHW, int shift_DHW, float epsilon);

template bool BatchNormNCDHW_Backward<ehalf,ehalf>(CUstream stream,
          ehalf* dx,
          float* dg,
          float* db,
    const ehalf* dy,
    const ehalf* x,
    const float* g,
    const float* m,
    const float* v,
    int N, int C, int DHW, int magic_DHW, int shift_DHW, float epsilon);

template bool BatchNormNCDHW_Backward<ehalf,float>(CUstream stream,
          float* dx,
          float* dg,
          float* db,
    const float* dy,
    const ehalf* x,
    const float* g,
    const float* m,
    const float* v,
    int N, int C, int DHW, int magic_DHW, int shift_DHW, float epsilon);

template bool BatchNormNCDHW_Backward<bhalf,bhalf>(CUstream stream,
          bhalf* dx,
          float* dg,
          float* db,
    const bhalf* dy,
    const bhalf* x,
    const float* g,
    const float* m,
    const float* v,
    int N, int C, int DHW, int magic_DHW, int shift_DHW, float epsilon);

#endif



