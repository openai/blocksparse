
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

//  y = a*x + b
template <typename T, int THREADS>
__global__ void __launch_bounds__(THREADS) cwise_linear_axpb_forward(
              T*              Y,
    const     T* __restrict__ X,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int CDHW, int DHW, bool bA, bool bB)
{
    const int tid = threadIdx.x;
    const int c   = blockIdx.x;
    const int n   = blockIdx.y;

    int  image_offset = n * CDHW + c * DHW;
    X += image_offset;
    Y += image_offset;

    float a = bA ? A[c] : 1.0f;
    float b = bB ? B[c] : 0.0f;

    for (int i = tid; i < DHW; i += THREADS)
    {
        float x = load(X, i);
        store(Y, a*x + b, i);
    }
}


// dx = a * dy
// da = sum(dy * x)
// db = sum(dy)
template <typename TX, typename TY, int THREADS>
__global__ void __launch_bounds__(THREADS) cwise_linear_axpb_backward(
             TY*              DX,
          float*              DA,
          float*              DB,
    const    TY* __restrict__ DY,
    const    TX* __restrict__ X,
    const float* __restrict__ A,
    int CDHW, int NDHW, int DHW, int magic_DHW, int shift_DHW, bool bDB)
{
    __shared__ float shareDA[THREADS>>5];
    __shared__ float shareDB[THREADS>>5];

    const int tid = threadIdx.x;
    const int c   = blockIdx.x;

    int   image_offset = c * DHW;
    DX += image_offset;
    DY += image_offset;
    X  += image_offset;

    float a = A[c];

    float da = 0.0f;
    float db = 0.0f;
    for (int ndhw = tid; ndhw < NDHW; ndhw += THREADS)
    {
        int n   = div64(ndhw, magic_DHW, shift_DHW);
        int dhw = ndhw - n*DHW;
        int i   = n * CDHW + dhw;

        float dy = load(DY, i);
        float  x = load( X, i);

        da += dy * x;
        db += dy;

        store(DX, a*dy, i);
    }
    // reduce within warp
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1)
    {
        da += shfl_xor(da, i);
        db += shfl_xor(db, i);
    }
    // first thread of each warp store to shared
    if ((tid & 31) == 0)
    {
        shareDA[tid >> 5] = da;
        shareDB[tid >> 5] = db;
    }
    __syncthreads();

    if (tid < (THREADS>>5))
    {
        // first warp loads all prior reductions
        da = shareDA[tid];
        db = shareDB[tid];

        // reduce within this last warp
        #pragma unroll
        for (int i = (THREADS>>6); i > 0; i >>= 1)
        {
            da += shfl_xor(da, i);
            db += shfl_xor(db, i);
        }
        // single thread outputs final reductions
        if (tid == 0)
        {
            DA[c] = da;
            if (bDB)
                DB[c] = db;
        }
    }
}

// db = sum(dy)
template <typename T, int THREADS>
__global__ void __launch_bounds__(THREADS) cwise_linear_xpb_backward(
          float*              DB,
    const     T* __restrict__ DY,
    int CDHW, int NDHW, int DHW, int magic_DHW, int shift_DHW)
{
    __shared__ float shareDB[THREADS>>5];

    const int tid = threadIdx.x;
    const int c   = blockIdx.x;

    DY += c * DHW;

    float db = 0.0f;
    for (int ndhw = tid; ndhw < NDHW; ndhw += THREADS)
    {
        int n   = div64(ndhw, magic_DHW, shift_DHW);
        int dhw = ndhw - n*DHW;
        int i   = n * CDHW + dhw;

        float dy = load(DY, i);

        db += dy;
    }
    // reduce within warp
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1)
        db += shfl_xor(db, i);

    // first thread of each warp store to shared
    if ((tid & 31) == 0)
        shareDB[tid >> 5] = db;

    __syncthreads();

    if (tid < (THREADS>>5))
    {
        // first warp loads all prior reductions
        db = shareDB[tid];

        // reduce within this last warp
        #pragma unroll
        for (int i = (THREADS>>6); i > 0; i >>= 1)
            db += shfl_xor(db, i);

        // single thread outputs final reductions
        if (tid == 0)
            DB[c] = db;
    }
}

template <typename T>
bool CWiseLinearAXPB_Forward(CUstream stream,
              T* y,
    const     T* x,
    const float* a,
    const float* b,
    int N, int C, int DHW)
{
    dim3 grid(C, N, 1);
    if      (DHW < 128*8)
        cwise_linear_axpb_forward<T, 32><<<grid,  32, 0, stream>>>(y, x, a, b, C*DHW, DHW, a!=0, b!=0);
    else if (DHW < 512*8)
        cwise_linear_axpb_forward<T,128><<<grid, 128, 0, stream>>>(y, x, a, b, C*DHW, DHW, a!=0, b!=0);
    else
        cwise_linear_axpb_forward<T,512><<<grid, 512, 0, stream>>>(y, x, a, b, C*DHW, DHW, a!=0, b!=0);
    return true; // TODO
}


template <typename TX, typename TY>
bool CWiseLinearAXPB_Backward(CUstream stream,
             TY* dx,
          float* da,
          float* db,
    const    TY* dy,
    const    TX* x,
    const float* a,
    int C, int NDHW, int DHW, int magic_DHW, int shift_DHW)
{
    dim3 grid(C, 1, 1);
    if      (NDHW <  256*8)
        cwise_linear_axpb_backward<TX,TY,  64><<<grid,   64, 0, stream>>>(dx, da, db, dy, x, a, C*DHW, NDHW, DHW, magic_DHW, shift_DHW, db!=0);
    else if (NDHW < 1024*8)
        cwise_linear_axpb_backward<TX,TY, 256><<<grid,  256, 0, stream>>>(dx, da, db, dy, x, a, C*DHW, NDHW, DHW, magic_DHW, shift_DHW, db!=0);
    else
        cwise_linear_axpb_backward<TX,TY,1024><<<grid, 1024, 0, stream>>>(dx, da, db, dy, x, a, C*DHW, NDHW, DHW, magic_DHW, shift_DHW, db!=0);
    return true; // TODO
}

template <typename T>
bool CWiseLinearXPB_Backward(CUstream stream,
          float* db,
    const     T* dy,
    int C, int NDHW, int DHW, int magic_DHW, int shift_DHW)
{
    dim3 grid(C, 1, 1);
    if      (NDHW <  256*8)
        cwise_linear_xpb_backward<T,  64><<<grid,   64, 0, stream>>>(db, dy, C*DHW, NDHW, DHW, magic_DHW, shift_DHW);
    else if (NDHW < 1024*8)
        cwise_linear_xpb_backward<T, 256><<<grid,  256, 0, stream>>>(db, dy, C*DHW, NDHW, DHW, magic_DHW, shift_DHW);
    else
        cwise_linear_xpb_backward<T,1024><<<grid, 1024, 0, stream>>>(db, dy, C*DHW, NDHW, DHW, magic_DHW, shift_DHW);
    return true; // TODO
}

template bool CWiseLinearAXPB_Forward<float>(CUstream stream,
          float* y,
    const float* x,
    const float* a,
    const float* b,
    int N, int C, int DHW);

template bool CWiseLinearAXPB_Forward<ehalf>(CUstream stream,
          ehalf* y,
    const ehalf* x,
    const float* a,
    const float* b,
    int N, int C, int DHW);

template bool CWiseLinearAXPB_Forward<bhalf>(CUstream stream,
          bhalf* y,
    const bhalf* x,
    const float* a,
    const float* b,
    int N, int C, int DHW);

template bool CWiseLinearAXPB_Backward<float,float>(CUstream stream,
          float* dx,
          float* da,
          float* db,
    const float* dy,
    const float* x,
    const float* a,
    int C, int NDHW, int DHW, int magic_DHW, int shift_DHW);

template bool CWiseLinearAXPB_Backward<ehalf,ehalf>(CUstream stream,
          ehalf* dx,
          float* da,
          float* db,
    const ehalf* dy,
    const ehalf* x,
    const float* a,
   int C, int NDHW, int DHW, int magic_DHW, int shift_DHW);

template bool CWiseLinearAXPB_Backward<bhalf,bhalf>(CUstream stream,
          bhalf* dx,
          float* da,
          float* db,
    const bhalf* dy,
    const bhalf* x,
    const float* a,
   int C, int NDHW, int DHW, int magic_DHW, int shift_DHW);

template bool CWiseLinearAXPB_Backward<ehalf,float>(CUstream stream,
          float* dx,
          float* da,
          float* db,
    const float* dy,
    const ehalf* x,
    const float* a,
   int C, int NDHW, int DHW, int magic_DHW, int shift_DHW);


template bool CWiseLinearXPB_Backward<float>(CUstream stream,
          float* db,
    const float* dy,
    int C, int NDHW, int DHW, int magic_DHW, int shift_DHW);

template bool CWiseLinearXPB_Backward<ehalf>(CUstream stream,
          float* db,
    const ehalf* dy,
   int C, int NDHW, int DHW, int magic_DHW, int shift_DHW);

template bool CWiseLinearXPB_Backward<bhalf>(CUstream stream,
          float* db,
    const bhalf* dy,
   int C, int NDHW, int DHW, int magic_DHW, int shift_DHW);

#endif




