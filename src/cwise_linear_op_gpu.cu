
#if GOOGLE_CUDA

#include "ew_op_gpu.h"

//  y = a*x + b   or    a*(x + b)
template <typename T>
__global__ void cwise_linear_axpb_forward(
              T*              Y,
    const     T* __restrict__ X,
    const float* __restrict__ A,
    const float* __restrict__ B,
    uint CDHW, uint DHW, int bA, int bB, int relu, int swap)
{
    uint tid = threadIdx.x;
    uint c   = blockIdx.x;
    uint n   = blockIdx.y;

    uint offset = n * CDHW + c * DHW + tid;

    float a = bA ? A[c] : 1.0f;
    float b = bB ? B[c] : 0.0f;

    #pragma unroll 1
    for (uint i = tid; i < DHW; i += blockDim.x, offset += blockDim.x)
    {
        float x = load(add_ptr_u(X, offset));
        float y = swap ? a*(x + b) : a*x + b;
        if (relu)
            y = ew_relu(y);

        store(add_ptr_u(Y, offset), y);
    }
}

// y  = a*x + b
// dx = dy * a
// da = sum(dy * x)
// db = sum(dy)

// y  = a*(x + b)
// dx = dy * a
// da = sum(dy*(x + b))
// db = sum(dy * a)
template <typename T>
__global__ void cwise_linear_axpb_backward(
              T*              DX,
          float*              DA,
          float*              DB,
    const     T* __restrict__ DY,
    const     T* __restrict__ X,
    const float* __restrict__ A,
    const float* __restrict__ B,
    uint CDHW, uint NDHW, uint DHW, int bDB, int relu, int swap)
{
    __shared__ float2 Share[32];
    uint tid = threadIdx.x;
    if (blockDim.x > 32)
    {
        float2 zero = {0.0f, 0.0f};
        if (tid < 32)
            Share[tid] = zero;
        __syncthreads();
    }
    uint c = blockIdx.x;
    uint offsetC = c * DHW;

    float a = A[c];
    float b = (relu || swap) && bDB ? B[c] : 0.0f;

    float da = 0.0f, db = 0.0f;
    #pragma unroll 1
    for (int ndhw = tid; ndhw < NDHW; ndhw += blockDim.x)
    {
        uint   n = ndhw / DHW;
        uint dhw = ndhw % DHW;
        uint offset = offsetC + n * CDHW + dhw;

        float dy = load(add_ptr_u(DY, offset));
        float  x = load(add_ptr_u( X, offset));

        if (relu)
            dy = ew_relu_grad(dy, swap ? a*(x + b) : a*x + b);

        float dx = dy * a;
        da += swap ? dy * (x + b) : dy * x;
        db += swap ? dx : dy;

        store(add_ptr_u(DX, offset), dx);
    }
    float2 stats = {da, db};

    // reduce within warp
    for (int i = 16; i > 0; i >>= 1)
        stats = ew_warp_sum(stats, i);

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
            for (int i = blockDim.x/64; i > 0; i >>= 1)
                stats = ew_warp_sum(stats, i);
        }
    }
    // single thread outputs final reductions
    if (tid == 0)
    {
        DA[c] = stats.x;
        if (bDB)
            DB[c] = stats.y;
    }
}

// db = sum(dy)
template <typename T>
__global__ void cwise_linear_xpb_backward(
              T*              DX,
          float*              DB,
    const     T* __restrict__ DY,
    const     T* __restrict__ Y,
    uint CDHW, uint NDHW, uint DHW, int relu)
{
    __shared__ float Share[32];
    uint tid = threadIdx.x;
    if (blockDim.x > 32)
    {
        if (tid < 32)
            Share[tid] = 0.0f;
        __syncthreads();
    }
    uint c = blockIdx.x;
    uint offsetC = c * DHW;

    float db = 0.0f;
    #pragma unroll 1
    for (int ndhw = tid; ndhw < NDHW; ndhw += blockDim.x)
    {
        uint   n = ndhw / DHW;
        uint dhw = ndhw % DHW;
        uint offset = offsetC + n * CDHW + dhw;

        float dy = load(add_ptr_u(DY, offset));
        if (relu)
        {
            float y = load(add_ptr_u(Y, offset));
            dy = ew_relu_grad(dy, y);
            store(add_ptr_u(DX, offset), dy);
        }
        db += dy;
    }
    // reduce within warp
    for (int i = 16; i > 0; i >>= 1)
        db += shfl_xor(db, i);

    if (blockDim.x > 32)
    {
        // first thread of each warp store to shared
        if ((tid & 31) == 0)
            Share[tid/32] = db;
        __syncthreads();

        if (tid < 32)
        {
            // first warp loads all prior reductions
            db = Share[tid];
            // reduce within this last warp
            #pragma unroll 1
            for (int i = blockDim.x/64; i > 0; i >>= 1)
                db += shfl_xor(db, i);

        }
    }
    // single thread outputs final reductions
    if (tid == 0)
        DB[c] = db;
}

template <typename T>
bool CWiseLinear_Forward(CUstream stream,
              T* y,
    const     T* x,
    const float* a,
    const float* b,
    uint N, uint C, uint DHW, bool relu, bool swap)
{
    // target 4 loops per block
    uint threads =
        DHW <=  4*32 ?  32 :
        DHW <=  8*32 ?  64 :
        DHW <= 16*32 ? 128 :
        DHW <= 32*32 ? 256 :
        DHW <= 64*32 ? 512 : 1024;

    cwise_linear_axpb_forward<T><<<dim3(C, N, 1),threads,0,stream>>>(y, x, a, b, C*DHW, DHW, a!=0, b!=0, relu, swap);
    return true; // TODO
}


template <typename T>
bool CWiseLinear_Backward(CUstream stream,
              T* dx,
          float* da,
          float* db,
    const     T* dy,
    const     T* xy,
    const float* a,
    const float* b,
    uint N, uint C, uint DHW, bool relu, bool swap)
{
    dim3 grid(C, 1, 1);
    uint NDHW = N*DHW;
    uint CDHW = C*DHW;

    // target 4 loops per block
    uint threads =
        NDHW <=  4*32 ?  32 :
        NDHW <=  8*32 ?  64 :
        NDHW <= 16*32 ? 128 :
        NDHW <= 32*32 ? 256 :
        NDHW <= 64*32 ? 512 : 1024;

    if (da != NULL)
        cwise_linear_axpb_backward<T><<<grid,threads,0,stream>>>(dx, da, db, dy, xy, a, b, CDHW, NDHW, DHW, db!=0, relu, swap);
    else
        cwise_linear_xpb_backward<T><<<grid,threads,0,stream>>>(dx, db, dy, xy, CDHW, NDHW, DHW, relu);
    return true; // TODO
}

template bool CWiseLinear_Forward<float>(CUstream stream, float* y, const float* x, const float* a, const float* b, uint N, uint C, uint DHW, bool relu, bool swap);
template bool CWiseLinear_Forward<ehalf>(CUstream stream, ehalf* y, const ehalf* x, const float* a, const float* b, uint N, uint C, uint DHW, bool relu, bool swap);
template bool CWiseLinear_Forward<bhalf>(CUstream stream, bhalf* y, const bhalf* x, const float* a, const float* b, uint N, uint C, uint DHW, bool relu, bool swap);

template bool CWiseLinear_Backward<float>(CUstream stream, float* dx, float* da, float* db, const float* dy, const float* xy, const float* a, const float* b, uint N, uint C, uint DHW, bool relu, bool swap);
template bool CWiseLinear_Backward<ehalf>(CUstream stream, ehalf* dx, float* da, float* db, const ehalf* dy, const ehalf* xy, const float* a, const float* b, uint N, uint C, uint DHW, bool relu, bool swap);
template bool CWiseLinear_Backward<bhalf>(CUstream stream, bhalf* dx, float* da, float* db, const bhalf* dy, const bhalf* xy, const float* a, const float* b, uint N, uint C, uint DHW, bool relu, bool swap);

#endif

