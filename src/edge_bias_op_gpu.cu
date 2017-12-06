
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "ew_op_gpu.h"


template <typename T>
__global__ void __launch_bounds__(32) edge_bias_forward(
              T*              Y,
    const     T* __restrict__ X,
    const float* __restrict__ B,
    const   int* __restrict__ Lut,
    int edges, int MPQ, int KMPQ)
{
    const int tid = threadIdx.x;
    const int e   = blockIdx.x;
    const int k   = blockIdx.y;
    const int n   = blockIdx.z;

    int2 entry = ((const int2*)Lut)[e];

    int  image_offset = n * KMPQ + k * MPQ;
    X += image_offset;
    Y += image_offset;
    Lut += entry.x;

    float bias = B[k * edges + e];

    // iterate over Lut mpqOffset entries
    for (int i = tid; i < entry.y; i += 32)
    {
        int mpqOffset = Lut[i];
        float x = load(X, mpqOffset);
        store(Y, x + bias, mpqOffset);
    }
}

template <typename T>
__global__ void __launch_bounds__(32) edge_bias_backward(
          float*              DB,
    const     T* __restrict__ DY,
    const   int* __restrict__ Lut,
    int edges, int MPQ, int KMPQ, int N)
{
    const int tid = threadIdx.x;
    const int e   = blockIdx.x;
    const int k   = blockIdx.y;

    int2 entry = ((const int2*)Lut)[e];
    DY += k * MPQ;
    DB += k * edges + e;
    Lut += entry.x;

    // iterate over Lut mpqOffset entries
    float grad_bias = 0.0f;
    for (int i = tid; i < entry.y; i += 32)
    {
        int mpqOffset = Lut[i];
        // Sum over N
        for (int n = 0; n < N; n++){
            float dy = load(DY, n * KMPQ + mpqOffset);
            grad_bias += dy;
        }
    }

    for (int i = 16; i > 0; i >>= 1)
        grad_bias += __shfl_xor(grad_bias, i);

    if (tid == 0)
        *DB = grad_bias;
}

template <typename T>
bool EdgeBiasForward(CUstream stream,
    const     T* x,
    const float* b,
    const   int* lut,
    int edges, int MPQ, int KMPQ, int K, int N)
{
    dim3 grid(edges, K, N);
    dim3 block(32, 1, 1);
    edge_bias_forward<T><<<grid, block, 0, stream>>>((T*)x, x, b, lut, edges, MPQ, KMPQ);
    return true; // TODO
}

template <typename T>
bool EdgeBiasBackward(CUstream stream,
          float* grad_b,
    const     T* grad_y,
    const   int* lut,
    int edges, int MPQ, int KMPQ, int K, int N)
{
    cuMemsetD32Async((CUdeviceptr)grad_b, 0, K * edges, stream);

    dim3 grid(edges, K, 1);
    dim3 block(32, 1, 1);
    edge_bias_backward<T><<<grid, block, 0, stream>>>(grad_b, grad_y, lut, edges, MPQ, KMPQ, N);
    return true; // TODO
}

template bool EdgeBiasForward<float>(CUstream stream,
    const float* x,
    const float* b,
    const   int* lut,
    int edges, int MPQ, int KMPQ, int K, int N);

template bool EdgeBiasForward<ehalf>(CUstream stream,
    const ehalf* x,
    const float* b,
    const   int* lut,
    int edges, int MPQ, int KMPQ, int K, int N);

template bool EdgeBiasForward<bhalf>(CUstream stream,
    const bhalf* x,
    const float* b,
    const   int* lut,
    int edges, int MPQ, int KMPQ, int K, int N);

template bool EdgeBiasBackward<float>(CUstream stream,
          float* grad_b,
    const float* grad_y,
    const   int* lut,
    int edges, int MPQ, int KMPQ, int K, int N);

template bool EdgeBiasBackward<ehalf>(CUstream stream,
          float* grad_b,
    const ehalf* grad_y,
    const   int* lut,
    int edges, int MPQ, int KMPQ, int K, int N);

template bool EdgeBiasBackward<bhalf>(CUstream stream,
          float* grad_b,
    const bhalf* grad_y,
    const   int* lut,
    int edges, int MPQ, int KMPQ, int K, int N);


#endif