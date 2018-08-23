
#if GOOGLE_CUDA

#include "ew_op_gpu.h"


template <typename T>
__global__ void __launch_bounds__(32) edge_bias_forward_nchw(
              T*              Y,
    const float* __restrict__ G,
    const float* __restrict__ B,
    const  uint* __restrict__ Lut,
    uint edges, uint MPQ, uint KMPQ)
{
    uint tid = threadIdx.x;
    uint e   = blockIdx.x;
    uint k   = blockIdx.y;
    uint n   = blockIdx.z;

    uint2 entry = ((const uint2*)Lut)[e];

    uint  image_offset = n * KMPQ + k * MPQ;
    Y += image_offset;
    Lut += entry.x;

    float gain = G[k * edges + e];
    float bias = B[k * edges + e];

    // iterate over Lut mpqOffset entries
    for (uint i = tid; i < entry.y; i += 32)
    {
        uint mpqOffset = Lut[i];
        T* Y_ = Y + mpqOffset;
        float x = load((const T*)Y_);
        store(Y_, x*gain + bias);
    }
}

// dx = g * dy
// dg = sum(dy * x)
// db = sum(dy)
template <typename T>
__global__ void __launch_bounds__(32) edge_bias_backward_nchw(
              T*              DY,
          float*              DG,
          float*              DB,
    const     T* __restrict__ X,
    const float* __restrict__ G,
    const  uint* __restrict__ Lut,
    uint edges, uint MPQ, uint KMPQ, uint N)
{
    uint tid = threadIdx.x;
    uint e   = blockIdx.x;
    uint k   = blockIdx.y;

    uint offsetG = k * edges + e;

    uint2 entry = ((const uint2*)Lut)[e];
    Lut += entry.x;

    float gain = G[offsetG];

    // iterate over Lut mpqOffset entries
    float dg = 0.0f, db = 0.0f;
    for (uint i = tid; i < entry.y; i += 32)
    {
        uint mpqOffset = Lut[i];
        // Sum over N
        for (uint n = 0; n < N; n++)
        {
            uint offset = n * KMPQ + k * MPQ + mpqOffset;

            float dy = load((const T*)DY, offset);
            float x  = load(X, offset);
            dg += dy * x;
            db += dy;
            store(DY, dy * gain, offset);
        }
    }
    for (int i = 16; i > 0; i >>= 1)
    {
        dg += shfl_xor(dg, i);
        db += shfl_xor(db, i);
    }
    if (tid == 0)
    {
        DG[offsetG] = dg;
        DB[offsetG] = db;
    }
}


template <typename T>
__global__ void edge_bias_forward_nhwc(
              T*              Y,
    const float* __restrict__ G,
    const float* __restrict__ B,
    const  uint* __restrict__ Lut,
    uint K, uint MPQK)
{
    uint k = blockIdx.x*blockDim.x + threadIdx.x;
    uint e = blockIdx.y;
    uint n = blockIdx.z;

    uint2 entry = ((const uint2*)Lut)[e];

    bool bk = k < K;
    uint offsetI = n * MPQK + k;
    uint offsetB = e *    K + k;
    uint offsetL = entry.x;
    uint lutsize = entry.y;

    float gain = load(add_ptr_u(G, offsetB), 0, bk);
    float bias = load(add_ptr_u(B, offsetB), 0, bk);

    // iterate over Lut mpqOffset entries
    #pragma unroll 1
    for (uint i = 0; i < lutsize; i += 4)
    {
        uint offset0 = load(add_ptr_u(Lut, offsetL+i+0)) * K + offsetI;
        uint offset1 = load(add_ptr_u(Lut, offsetL+i+1)) * K + offsetI;
        uint offset2 = load(add_ptr_u(Lut, offsetL+i+2)) * K + offsetI;
        uint offset3 = load(add_ptr_u(Lut, offsetL+i+3)) * K + offsetI;

        T* Y0 = add_ptr_u(Y, offset0);
        T* Y1 = add_ptr_u(Y, offset1);
        T* Y2 = add_ptr_u(Y, offset2);
        T* Y3 = add_ptr_u(Y, offset3);

        float y0 = load((const T*)Y0, 0, bk);
        float y1 = load((const T*)Y1, 0, bk & i+1 < lutsize);
        float y2 = load((const T*)Y2, 0, bk & i+2 < lutsize);
        float y3 = load((const T*)Y3, 0, bk & i+3 < lutsize);

        store(Y0, y0*gain + bias, 0, bk);
        store(Y1, y1*gain + bias, 0, bk & i+1 < lutsize);
        store(Y2, y2*gain + bias, 0, bk & i+2 < lutsize);
        store(Y3, y3*gain + bias, 0, bk & i+3 < lutsize);
    }
}

// dx = g * dy
// dg = sum(dy * x)
// db = sum(dy)
template <typename T>
__global__ void edge_bias_backward_nhwc(
              T*              DY,
          float*              DG,
          float*              DB,
    const     T* __restrict__ X,
    const float* __restrict__ G,
    const  uint* __restrict__ Lut,
    uint N, uint K, uint MPQK)
{
    uint k = blockIdx.x*blockDim.x + threadIdx.x;
    uint e = blockIdx.y;

    uint2 entry = ((const uint2*)Lut)[e];

    bool bk = k < K;

    uint offsetB = e*K + k;
    uint offsetL = entry.x;
    uint lutsize = entry.y;

    float gain = load(add_ptr_u(G, offsetB), 0, bk);

    // iterate over Lut mpqOffset entries
    float dg = 0.0f, db = 0.0f;
    #pragma unroll 1
    for (uint i = 0; i < lutsize; i++)
    {
        uint offset = load(add_ptr_u(Lut, offsetL+i)) * K + k;
        // Sum over N
        #pragma unroll 4
        for (uint n = 0; n < N; n++, offset += MPQK)
        {
            T* DY_ = add_ptr_u(DY, offset);

            float dy = load((const T*)DY_, 0, bk);
            float x  = load(add_ptr_u(X, offset), 0, bk);

            dg += dy * x;
            db += dy;
            store(DY_, dy * gain, 0, bk);
        }
    }
    store(add_ptr_u(DG, offsetB), dg, 0, bk);
    store(add_ptr_u(DB, offsetB), db, 0, bk);
}

template <typename T>
bool EdgeBiasForward(CUstream stream,
              T* y,
    const     T* x,
    const float* g,
    const float* b,
    const   int* lut,
    uint edges, uint MPQ, uint K, uint N, int layout, bool inference)
{

    if (!inference)
        cuMemcpyAsync((CUdeviceptr)y, (CUdeviceptr)x, N*K*MPQ*sizeof(T), stream);

    if (layout == 0)
    {
        edge_bias_forward_nchw<T><<<dim3(edges, K, N), 32, 0, stream>>>(y, g, b, (const uint*)lut, edges, MPQ, K*MPQ);
    }
    else
    {
        uint gridK, threads;
             if (K <=  32) { threads =  32; gridK = CEIL_DIV(K,  32); }
        else if (K <=  64) { threads =  64; gridK = CEIL_DIV(K,  64); }
        else               { threads = 128; gridK = CEIL_DIV(K, 128); }

        edge_bias_forward_nhwc<T><<<dim3(gridK, edges, N), threads, 0, stream>>>(y, g, b, (const uint*)lut, K, MPQ*K);
    }
    return true; // TODO
}

template <typename T>
bool EdgeBiasBackward(CUstream stream,
              T* dy,
          float* dg,
          float* db,
    const     T* x,
    const float* g,
    const   int* lut,
    uint edges, uint MPQ, uint K, uint N, int layout)
{
    if (layout == 0)
    {
        edge_bias_backward_nchw<T><<<dim3(edges, K, 1), 32, 0, stream>>>(dy, dg, db, x, g, (const uint*)lut, edges, MPQ, K*MPQ, N);
    }
    else
    {
        uint gridK, threads;
        if (K <=  32) { threads = 32; gridK = CEIL_DIV(K, 32); }
        else          { threads = 64; gridK = CEIL_DIV(K, 64); }

        edge_bias_backward_nhwc<T><<<dim3(gridK, edges, 1), threads, 0, stream>>>(dy, dg, db, x, g, (const uint*)lut, N, K, MPQ*K);
    }
    return true; // TODO
}

template bool EdgeBiasForward <float>(CUstream stream, float* y, const float* x, const float* g, const float* b, const int* lut, uint edges, uint MPQ, uint K, uint N, int layout, bool inference);
template bool EdgeBiasForward <ehalf>(CUstream stream, ehalf* y, const ehalf* x, const float* g, const float* b, const int* lut, uint edges, uint MPQ, uint K, uint N, int layout, bool inference);
template bool EdgeBiasForward <bhalf>(CUstream stream, bhalf* y, const bhalf* x, const float* g, const float* b, const int* lut, uint edges, uint MPQ, uint K, uint N, int layout, bool inference);

template bool EdgeBiasBackward<float>(CUstream stream, float* dy, float* dg, float* db, const float* x, const float* g, const int* lut, uint edges, uint MPQ, uint K, uint N, int layout);
template bool EdgeBiasBackward<ehalf>(CUstream stream, ehalf* dy, float* dg, float* db, const ehalf* x, const float* g, const int* lut, uint edges, uint MPQ, uint K, uint N, int layout);
template bool EdgeBiasBackward<bhalf>(CUstream stream, bhalf* dy, float* dg, float* db, const bhalf* x, const float* g, const int* lut, uint edges, uint MPQ, uint K, uint N, int layout);


#endif


