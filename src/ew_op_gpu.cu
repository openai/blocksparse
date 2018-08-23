#if GOOGLE_CUDA

#include "ew_op_gpu.h"
#include <stdio.h>

// Forward Kernels
#define OP_Z_XY(func, op)                   \
template <typename T, typename V, int U>    \
__global__ void __launch_bounds__(32) func( \
          T*              Z,                \
    const T* __restrict__ X,                \
    const T* __restrict__ Y,                \
    int size)                               \
{                                           \
    int tid = threadIdx.x;                  \
    int bid = blockIdx.x;                   \
    int i   = bid * U*32 + tid;             \
    for (int j = 0; j < U; j++)             \
    {                                       \
        bool b = i < size;                  \
        V x = load(X, i, b);                \
        V y = load(Y, i, b);                \
        V z = op(x, y);                     \
        store(Z, z, i, b);                  \
        i += 32;                            \
    }                                       \
}
#define OP_Z_X(func, op)                    \
template <typename T, typename V, int U>    \
__global__ void __launch_bounds__(32) func( \
          T*              Z,                \
    const T* __restrict__ X,                \
    int size)                               \
{                                           \
    int tid = threadIdx.x;                  \
    int bid = blockIdx.x;                   \
    int i   = bid * U*32 + tid;             \
    for (int j = 0; j < U; j++)             \
    {                                       \
        bool b = i < size;                  \
        V x = load(X, i, b);                \
        V z = op(x);                        \
        store(Z, z, i, b);                  \
        i += 32;                            \
    }                                       \
}
#define OP_Z_XA(func, op)                   \
template <typename T, typename V, int U>    \
__global__ void __launch_bounds__(32) func( \
          T*              Z,                \
    const T* __restrict__ X,                \
    int size, float alpha)                  \
{                                           \
    int tid = threadIdx.x;                  \
    int bid = blockIdx.x;                   \
    int i   = bid * U*32 + tid;             \
    for (int j = 0; j < U; j++)             \
    {                                       \
        bool b = i < size;                  \
        V x = load(X, i, b);                \
        V z = op(x, alpha);                 \
        store(Z, z, i, b);                  \
        i += 32;                            \
    }                                       \
}
#define OP_Z_XB(func, op)                   \
template <typename T, typename V, int U>    \
__global__ void __launch_bounds__(32) func( \
          T*              Z,                \
    const T* __restrict__ X,                \
    const V* __restrict__ B,                \
    int K)                                  \
{                                           \
    int tid = threadIdx.x;                  \
    int k   = blockIdx.x;                   \
    int n   = blockIdx.y;                   \
    k = k*U*32 + tid;                       \
    int i = n*K + k;                        \
    for (int j = 0; j < U; j++)             \
    {                                       \
        bool b = k < K;                     \
        V x = load(X, i, b);                \
        V g = load(B, k, b);                \
        V z = op(x, g);                     \
        store(Z, z, i, b);                  \
        i += 32;                            \
        k += 32;                            \
    }                                       \
}
//  Backwards Kernels
#define OP_DXDY_DZXY(func, dx_op, dy_op)             \
template <typename B, typename F, typename V, int U> \
__global__ void __launch_bounds__(32) func(          \
          B*              DX,                        \
          B*              DY,                        \
    const B* __restrict__ DZ,                        \
    const F* __restrict__ X,                         \
    const F* __restrict__ Y,                         \
    int size)                                        \
{                                                    \
    int tid = threadIdx.x;                           \
    int bid = blockIdx.x;                            \
    int i   = bid * U*32 + tid;                      \
    for (int j = 0; j < U; j++)                      \
    {                                                \
        bool b = i < size;                           \
        V dz = load(DZ, i, b);                       \
        V  x = load( X, i, b);                       \
        V  y = load( Y, i, b);                       \
        V dx = dx_op;                                \
        V dy = dy_op;                                \
        store(DX, dx, i, b);                         \
        store(DY, dy, i, b);                         \
        i += 32;                                     \
    }                                                \
}
#define OP_DX_DZZ(func, op)                          \
template <typename B, typename F, typename V, int U> \
__global__ void __launch_bounds__(32) func(          \
          B*              DX,                        \
    const B* __restrict__ DZ,                        \
    const F* __restrict__ Z,                         \
    int size)                                        \
{                                                    \
    int tid = threadIdx.x;                           \
    int bid = blockIdx.x;                            \
    int i   = bid * U*32 + tid;                      \
    for (int j = 0; j < U; j++)                      \
    {                                                \
        bool b = i < size;                           \
        V dz = load(DZ, i, b);                       \
        V  z = load( Z, i, b);                       \
        V dx = op(dz, z);                            \
        store(DX, dx, i, b);                         \
        i += 32;                                     \
    }                                                \
}
#define OP_DX_DZX(func, op)                          \
template <typename B, typename F, typename V, int U> \
__global__ void __launch_bounds__(32) func(          \
          B*              DX,                        \
    const B* __restrict__ DZ,                        \
    const F* __restrict__ X,                         \
    int size)                                        \
{                                                    \
    int tid = threadIdx.x;                           \
    int bid = blockIdx.x;                            \
    int i   = bid * U*32 + tid;                      \
    for (int j = 0; j < U; j++)                      \
    {                                                \
        bool b = i < size;                           \
        V dz = load(DZ, i, b);                       \
        V  x = load( X, i, b);                       \
        V dx = op(dz, x);                            \
        store(DX, dx, i, b);                         \
        i += 32;                                     \
    }                                                \
}
#define OP_DX_DZXA(func, op)                         \
template <typename B, typename F, typename V, int U> \
__global__ void __launch_bounds__(32) func(          \
          B*              DX,                        \
    const B* __restrict__ DZ,                        \
    const F* __restrict__ X,                         \
    int size, float alpha)                           \
{                                                    \
    int tid = threadIdx.x;                           \
    int bid = blockIdx.x;                            \
    int i   = bid * U*32 + tid;                      \
    for (int j = 0; j < U; j++)                      \
    {                                                \
        bool b = i < size;                           \
        V dz = load(DZ, i, b);                       \
        V  x = load( X, i, b);                       \
        V dx = op(dz, x, alpha);                     \
        store(DX, dx, i, b);                         \
        i += 32;                                     \
    }                                                \
}
// db = sum(dz, axis=0)
template <typename B, typename V>
__global__ void __launch_bounds__(32) BiasAddGrad(
          V*              DB,
    const B* __restrict__ DZ,
    int N, int K)
{
    int tid = threadIdx.x;
    int k   = blockIdx.x;

    k = k*32 + tid;
    int nk = k;
    bool b = k < K;
    int K2 = K  + K;
    int K3 = K2 + K;
    int K4 = K3 + K;

    V db;
    ew_zero(db);
    for (int n = 0; n < N; n += 4)
    {
        V dz0 = load(DZ, nk  ,  b);
        V dz1 = load(DZ, nk+K,  b && (n+1 < N));
        V dz2 = load(DZ, nk+K2, b && (n+2 < N));
        V dz3 = load(DZ, nk+K3, b && (n+3 < N));
        db = ew_add(db, dz0);
        db = ew_add(db, dz1);
        db = ew_add(db, dz2);
        db = ew_add(db, dz3);
        nk += K4;
    }
    store(DB, db, k, b);
}
// dx = dz * g
// dg = sum(dz * x, axis=0)
template <typename B, typename F, typename V>
__global__ void __launch_bounds__(32) GainMulGrad(
          B*              DX,
          V*              DG,
    const B* __restrict__ DZ,
    const F* __restrict__ X,
    const V* __restrict__ G,
    int N, int K)
{
    int tid = threadIdx.x;
    int k   = blockIdx.x;

    k = k*32 + tid;
    int nk = k;
    bool b = k < K;

    V dg;
    ew_zero(dg);
    V g = load(G, k, b);
    for (int n = 0; n < N; n += 2)
    {
        V dz0 = load(DZ, nk,   b);
        V dz1 = load(DZ, nk+K, b && (n+1 < N));
        V  x0 = load( X, nk,   b);
        V  x1 = load( X, nk+K, b && (n+1 < N));

        V dx0 = ew_mul(dz0, g);
        V dx1 = ew_mul(dz1, g);
        dg = ew_add(dg, ew_mul(dz0, x0));
        dg = ew_add(dg, ew_mul(dz1, x1));

        store(DX, dx0, nk,   b);
        store(DX, dx1, nk+K, b && (n+1 < N));

        nk += K*2;
    }
    store(DG, dg, k, b);
}


// Forward Ops
OP_Z_XY(Add,     ew_add)
OP_Z_XY(Sub,     ew_sub)
OP_Z_XY(Mul,     ew_mul)
OP_Z_XY(Div,     ew_div)
OP_Z_XY(Maximum, ew_maximum)
OP_Z_XY(Minimum, ew_minimum)

OP_Z_X(Neg,      ew_neg)
OP_Z_X(Rcp,      ew_rcp)
OP_Z_X(Sqr,      ew_sqr)
OP_Z_X(Sqrt,    ew_sqrt)
OP_Z_X(Exp,      ew_exp)
OP_Z_X(Log,      ew_log)
OP_Z_X(Sig,      ew_sig)
OP_Z_X(Tanh,    ew_tanh)
OP_Z_X(Relu,    ew_relu)

OP_Z_XA(Elu,     ew_elu)
OP_Z_XB(BiasAdd, ew_add)
OP_Z_XB(GainMul, ew_mul)


// Backward Ops
OP_DXDY_DZXY(    MulGrad,      ew_mul(dz, y),         ew_mul(dz, x)   )
OP_DXDY_DZXY(    DivGrad,      ew_div(dz, y),    ew_div_grad(dz, x, y))
OP_DXDY_DZXY(MaximumGrad, ew_max_grad(dz, x, y), ew_max_grad(dz, y, x))
OP_DXDY_DZXY(MinimumGrad, ew_min_grad(dz, x, y), ew_min_grad(dz, y, x))

OP_DX_DZZ(ReluGrad, ew_relu_grad)
OP_DX_DZZ( SigGrad,  ew_sig_grad)
OP_DX_DZZ(TanhGrad, ew_tanh_grad)

OP_DX_DZX( RcpGrad,  ew_rcp_grad)
OP_DX_DZX( SqrGrad,  ew_sqr_grad)
OP_DX_DZX( ExpGrad,  ew_exp_grad)
OP_DX_DZX( LogGrad,  ew_log_grad)
OP_DX_DZX(SqrtGrad, ew_sqrt_grad)

OP_DX_DZXA(EluGrad,  ew_elu_grad)

template <typename T, typename V>
bool EW_Forward(CUstream stream,
              T* z,
    const     T* x,
    const     T* y,
    const float* b,
    float alpha, int size, int N, int op)
{
    if ((size & 3) == 0 && size >= 256)
    {
        size >>= 2; // use vector loads
        int grid = (size >> 6) + ((size & 63) != 0); // 1 warp with 2 unrolls
                   V* Z = (           V*)z;
        const      V* X = (const      V*)x;
        const      V* Y = (const      V*)y;
        const float4* B = (const float4*)b;
        switch(op)
        {
            case  0 :     Add<V,float4,2><<<grid,32,0,stream>>>(Z, X, Y, size); break;
            case  1 :     Sub<V,float4,2><<<grid,32,0,stream>>>(Z, X, Y, size); break;
            case  2 :     Mul<V,float4,2><<<grid,32,0,stream>>>(Z, X, Y, size); break;
            case  3 :     Div<V,float4,2><<<grid,32,0,stream>>>(Z, X, Y, size); break;
            case  4 : Maximum<V,float4,2><<<grid,32,0,stream>>>(Z, X, Y, size); break;
            case  5 : Minimum<V,float4,2><<<grid,32,0,stream>>>(Z, X, Y, size); break;
            case  6 :     Neg<V,float4,2><<<grid,32,0,stream>>>(Z, X,    size); break;
            case  7 :     Rcp<V,float4,2><<<grid,32,0,stream>>>(Z, X,    size); break;
            case  8 :     Sqr<V,float4,2><<<grid,32,0,stream>>>(Z, X,    size); break;
            case  9 :    Sqrt<V,float4,2><<<grid,32,0,stream>>>(Z, X,    size); break;
            case 10 :     Exp<V,float4,2><<<grid,32,0,stream>>>(Z, X,    size); break;
            case 11 :     Log<V,float4,2><<<grid,32,0,stream>>>(Z, X,    size); break;
            case 12 :     Sig<V,float4,2><<<grid,32,0,stream>>>(Z, X,    size); break;
            case 13 :    Tanh<V,float4,2><<<grid,32,0,stream>>>(Z, X,    size); break;
            case 14 :    Relu<V,float4,2><<<grid,32,0,stream>>>(Z, X,    size); break;
            case 15 :     Elu<V,float4,2><<<grid,32,0,stream>>>(Z, X,    size, alpha);  break;
            case 16 : BiasAdd<V,float4,2><<<dim3(grid,N),32,0,stream>>>(Z, X, B, size); break;
            case 17 : GainMul<V,float4,2><<<dim3(grid,N),32,0,stream>>>(Z, X, B, size); break;
        }
    }
    else
    {
        int grid = (size >> 7) + ((size & 127) != 0); // 1 warp with 4 unrolls
        switch(op)
        {
            case  0 :     Add<T,float,4><<<grid,32,0,stream>>>(z, x, y, size); break;
            case  1 :     Sub<T,float,4><<<grid,32,0,stream>>>(z, x, y, size); break;
            case  2 :     Mul<T,float,4><<<grid,32,0,stream>>>(z, x, y, size); break;
            case  3 :     Div<T,float,4><<<grid,32,0,stream>>>(z, x, y, size); break;
            case  4 : Maximum<T,float,4><<<grid,32,0,stream>>>(z, x, y, size); break;
            case  5 : Minimum<T,float,4><<<grid,32,0,stream>>>(z, x, y, size); break;
            case  6 :     Neg<T,float,4><<<grid,32,0,stream>>>(z, x,    size); break;
            case  7 :     Rcp<T,float,4><<<grid,32,0,stream>>>(z, x,    size); break;
            case  8 :     Sqr<T,float,4><<<grid,32,0,stream>>>(z, x,    size); break;
            case  9 :    Sqrt<T,float,4><<<grid,32,0,stream>>>(z, x,    size); break;
            case 10 :     Exp<T,float,4><<<grid,32,0,stream>>>(z, x,    size); break;
            case 11 :     Log<T,float,4><<<grid,32,0,stream>>>(z, x,    size); break;
            case 12 :     Sig<T,float,4><<<grid,32,0,stream>>>(z, x,    size); break;
            case 13 :    Tanh<T,float,4><<<grid,32,0,stream>>>(z, x,    size); break;
            case 14 :    Relu<T,float,4><<<grid,32,0,stream>>>(z, x,    size); break;
            case 15 :     Elu<T,float,4><<<grid,32,0,stream>>>(z, x,    size, alpha); break;
            case 16 : BiasAdd<T,float,4><<<dim3(grid,N),32,0,stream>>>(z, x, b, size); break;
            case 17 : GainMul<T,float,4><<<dim3(grid,N),32,0,stream>>>(z, x, b, size); break;
        }
    }
    return true;
}

template
bool EW_Forward<float,float4>(CUstream stream,
          float* z,
    const float* x,
    const float* y,
    const float* b,
    float alpha, int size, int N, int op);

template
bool EW_Forward<ehalf,ehalf4>(CUstream stream,
          ehalf* z,
    const ehalf* x,
    const ehalf* y,
    const float* b,
    float alpha, int size, int N, int op);

template
bool EW_Forward<bhalf,bhalf4>(CUstream stream,
          bhalf* z,
    const bhalf* x,
    const bhalf* y,
    const float* b,
    float alpha, int size, int N, int op);

template <typename B, typename F, typename VB, typename VF>
bool EW_Backward(CUstream stream,
              B* dx,
              B* dy,
          float* db,
    const     B* dz,
    const     F* x,
    const     F* y,
    const     F* z,
    const float* g,
    float alpha, int size, int N, int op)
{
    if ((size & 3) == 0 && size >= 16384)
    {
        size >>= 2; // use vector loads
        int grid64 = (size >> 6) + ((size & 63) != 0); // 1 warp with 2 unrolls
        int grid32 = (size >> 5) + ((size & 31) != 0); // 1 warp with 1 unroll
                  VB* DX = (          VB*)dx;
                  VB* DY = (          VB*)dy;
              float4* DB = (      float4*)db;
        const     VB* DZ = (const     VB*)dz;
        const     VF*  X = (const     VF*)x;
        const     VF*  Y = (const     VF*)y;
        const     VF*  Z = (const     VF*)z;
        const float4*  G = (const float4*)g;
        switch(op)
        {
            // Add: no grads, Sub/Neg: use Forward Neg
            case  2 :     MulGrad<VB,VF,float4,2><<<grid64,32,0,stream>>>(DX, DY, DZ, X, Y, size); break;
            case  3 :     DivGrad<VB,VF,float4,2><<<grid64,32,0,stream>>>(DX, DY, DZ, X, Y, size); break;
            case  4 : MaximumGrad<VB,VF,float4,2><<<grid64,32,0,stream>>>(DX, DY, DZ, X, Y, size); break;
            case  5 : MinimumGrad<VB,VF,float4,2><<<grid64,32,0,stream>>>(DX, DY, DZ, X, Y, size); break;
            case  7 :     RcpGrad<VB,VF,float4,2><<<grid64,32,0,stream>>>(DX, DZ, X, size); break;
            case  8 :     SqrGrad<VB,VF,float4,2><<<grid64,32,0,stream>>>(DX, DZ, X, size); break;
            case  9 :    SqrtGrad<VB,VF,float4,2><<<grid64,32,0,stream>>>(DX, DZ, X, size); break;
            case 10 :     ExpGrad<VB,VF,float4,2><<<grid64,32,0,stream>>>(DX, DZ, X, size); break;
            case 11 :     LogGrad<VB,VF,float4,2><<<grid64,32,0,stream>>>(DX, DZ, X, size); break;
            case 12 :     SigGrad<VB,VF,float4,2><<<grid64,32,0,stream>>>(DX, DZ, Z, size); break;
            case 13 :    TanhGrad<VB,VF,float4,2><<<grid64,32,0,stream>>>(DX, DZ, Z, size); break;
            case 14 :    ReluGrad<VB,VF,float4,2><<<grid64,32,0,stream>>>(DX, DZ, Z, size); break;
            case 15 :     EluGrad<VB,VF,float4,2><<<grid64,32,0,stream>>>(DX, DZ, X, size, alpha);    break;
            case 16 : BiasAddGrad<VB,   float4  ><<<grid32,32,0,stream>>>(DB, DZ,           N, size); break;
            case 17 : GainMulGrad<VB,VF,float4  ><<<grid32,32,0,stream>>>(DX, DB, DZ, X, G, N, size); break;
        }
    }
    else
    {
        int grid128 = (size >> 7) + ((size & 127) != 0); // 1 warp with 4 unrolls
        int grid32  = (size >> 5) + ((size &  31) != 0); // 1 warp with 1 unroll
        switch(op)
        {
            // Add: no grads, Sub/Neg: use Forward Neg
            case  2 :     MulGrad<B,F,float,4><<<grid128,32,0,stream>>>(dx, dy, dz, x, y, size); break;
            case  3 :     DivGrad<B,F,float,4><<<grid128,32,0,stream>>>(dx, dy, dz, x, y, size); break;
            case  4 : MaximumGrad<B,F,float,4><<<grid128,32,0,stream>>>(dx, dy, dz, x, y, size); break;
            case  5 : MinimumGrad<B,F,float,4><<<grid128,32,0,stream>>>(dx, dy, dz, x, y, size); break;
            case  7 :     RcpGrad<B,F,float,4><<<grid128,32,0,stream>>>(dx, dz, x, size); break;
            case  8 :     SqrGrad<B,F,float,4><<<grid128,32,0,stream>>>(dx, dz, x, size); break;
            case  9 :    SqrtGrad<B,F,float,4><<<grid128,32,0,stream>>>(dx, dz, x, size); break;
            case 10 :     ExpGrad<B,F,float,4><<<grid128,32,0,stream>>>(dx, dz, x, size); break;
            case 11 :     LogGrad<B,F,float,4><<<grid128,32,0,stream>>>(dx, dz, x, size); break;
            case 12 :     SigGrad<B,F,float,4><<<grid128,32,0,stream>>>(dx, dz, z, size); break;
            case 13 :    TanhGrad<B,F,float,4><<<grid128,32,0,stream>>>(dx, dz, z, size); break;
            case 14 :    ReluGrad<B,F,float,4><<<grid128,32,0,stream>>>(dx, dz, z, size); break;
            case 15 :     EluGrad<B,F,float,4><<<grid128,32,0,stream>>>(dx, dz, x, size, alpha);    break;
            case 16 : BiasAddGrad<B,  float  ><<<grid32 ,32,0,stream>>>(db, dz,           N, size); break;
            case 17 : GainMulGrad<B,F,float  ><<<grid32 ,32,0,stream>>>(dx, db, dz, x, g, N, size); break;
        }
    }
    return true;
}

template
bool EW_Backward<float,float,float4,float4>(CUstream stream,
          float* dx,
          float* dy,
          float* db,
    const float* dz,
    const float* x,
    const float* y,
    const float* z,
    const float* g,
    float alpha, int size, int N, int op);

// template
// bool EW_Backward<float,ehalf,float4,ehalf4>(CUstream stream,
//           float* dx,
//           float* dy,
//           float* db,
//     const float* dz,
//     const ehalf* x,
//     const ehalf* y,
//     const ehalf* z,
//     const float* g,
//     float alpha, int size, int N, int op);

template
bool EW_Backward<ehalf,ehalf,ehalf4,ehalf4>(CUstream stream,
          ehalf* dx,
          ehalf* dy,
          float* db,
    const ehalf* dz,
    const ehalf* x,
    const ehalf* y,
    const ehalf* z,
    const float* g,
    float alpha, int size, int N, int op);

template
bool EW_Backward<bhalf,bhalf,bhalf4,bhalf4>(CUstream stream,
          bhalf* dx,
          bhalf* dy,
          float* db,
    const bhalf* dz,
    const bhalf* x,
    const bhalf* y,
    const bhalf* z,
    const float* g,
    float alpha, int size, int N, int op);

// template
// bool EW_Backward<float,bhalf,float4,bhalf4>(CUstream stream,
//           float* dx,
//           float* dy,
//           float* db,
//     const float* dz,
//     const bhalf* x,
//     const bhalf* y,
//     const bhalf* z,
//     const float* g,
//     float alpha, int size, int N, int op);


template <typename TY, typename TX, typename V, int U>
__global__ void __launch_bounds__(32) float_cast(
          TY*              Y,
    const TX* __restrict__ X,
    int size)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i   = bid * U*32 + tid;
    #pragma unroll
    for (int j = 0; j < U; j++)
    {
        bool b = i < size;
        V x = load(X, i, b);
        store(Y, x, i, b);
        i += 32;
    }
}

template <typename TY, typename TX, typename VY, typename VX>
bool FloatCast(CUstream stream, TY* y, const TX* x, int size)
{
    if ((size & 3) == 0)
    {
        size >>= 2; // use vector loads
        int grid = (size >> 7) + ((size & 127) != 0); // 1 warp with 4 unrolls
        float_cast<VY,VX,float4,4><<<grid,32,0,stream>>>((VY*)y, (const VX*)x, size);
    }
    else
    {
        int grid = (size >> 7) + ((size & 127) != 0); // 1 warp with 4 unrolls
        float_cast<TY,TX,float,4><<<grid,32,0,stream>>>(y, x, size);
    }
    return true; // TODO
}

template bool FloatCast<float,ehalf,float4,ehalf4>(CUstream stream, float* y, const ehalf* x, int size);
template bool FloatCast<ehalf,float,ehalf4,float4>(CUstream stream, ehalf* y, const float* x, int size);
template bool FloatCast<float,bhalf,float4,bhalf4>(CUstream stream, float* y, const bhalf* x, int size);
template bool FloatCast<bhalf,float,bhalf4,float4>(CUstream stream, bhalf* y, const float* x, int size);


__device__ __forceinline__ uint float4_to_uint(float4 v)
{
    uint ret;
    asm("{\n\t"
        ".reg .u32 u0, u1, u2, u3;\n\t"
        "cvt.rni.u32.f32 u0, %1;\n\t"
        "cvt.rni.u32.f32 u1, %2;\n\t"
        "cvt.rni.u32.f32 u2, %3;\n\t"
        "cvt.rni.u32.f32 u3, %4;\n\t"
        "bfi.b32 u0, u1, u0, 8, 8;\n\t"
        "bfi.b32 u2, u3, u2, 8, 8;\n\t"
        "shl.b32 u2, u2, 16;\n\t"
        "add.u32 %0, u0, u2;\n\t" // encourage faster XMAD.PSL over BFI for 3rd bit merge
        "}" : "=r"(ret) : "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w));
    return ret;
}
__device__ __forceinline__ float4 uint_to_float4(uint val)
{
    float4 v;
    asm("{\n\t"
        ".reg .u8 u0, u1, u2, u3;\n\t"
        "mov.b32 {u0, u1, u2, u3}, %4;\n\t"
        "cvt.rn.f32.u8 %0, u0;\n\t"
        "cvt.rn.f32.u8 %1, u1;\n\t"
        "cvt.rn.f32.u8 %2, u2;\n\t"
        "cvt.rn.f32.u8 %3, u3;\n\t"
        "}" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(val) );
    return v;
}
__device__ __forceinline__ float rand_mask(float keep_prob, uint& lfsr0, uint& lfsr1, uint& lfsr2)
{
    lfsr0 = ((lfsr0 & 0xfffffffe) << 12) ^ (((lfsr0 << 13) ^ lfsr0) >> 19);
    lfsr1 = ((lfsr1 & 0xfffffff8) <<  4) ^ (((lfsr1 << 2)  ^ lfsr1) >> 25);
    lfsr2 = ((lfsr2 & 0xfffffff0) << 11) ^ (((lfsr2 << 3)  ^ lfsr2) >> 11);
    uint urand = lfsr0 ^ lfsr1 ^ lfsr2;
    // (float)urand * 2**-32 > keep_prob ? 0.0f : 1.0f;
    float val;
    asm("cvt.rn.f32.u32 %0, %1;\n\t"
        "mul.f32 %0, %0, 0F2f800000;"
        : "=f"(val) : "r"(urand));
    return val > keep_prob ? 0.0f : 1.0f;
}


template <typename T, uint THREADS>
__global__ void __launch_bounds__(THREADS) dropout_forward(
    T* Y, uint* Mask, const T* __restrict__ X, uint size, float keep_prob, float scale)
{
    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    uint lfsr0, lfsr1, lfsr2;
    uint idx = bid * THREADS + tid;
    asm("mov.b32 %0, %%clock_hi;"       : "=r"(lfsr0) :);
    asm("mov.b32 %0, %%clock;"          : "=r"(lfsr1) :);
    asm("mov.b32 %0, %%globaltimer_lo;" : "=r"(lfsr2) :);
    asm("shf.r.clamp.b32 %0,%0,%0,%1;"  : "=r"(lfsr0) : "r"((lfsr0 ^ tid) & 31));
    asm("shf.r.clamp.b32 %0,%0,%0,%1;"  : "=r"(lfsr1) : "r"((lfsr1 ^ tid) & 31));
    asm("shf.r.clamp.b32 %0,%0,%0,%1;"  : "=r"(lfsr2) : "r"((lfsr2 ^ tid) & 31));
    lfsr0 ^= idx ^ (idx << 5)  ^ (idx << 11) ^ (idx << 17) ^ (idx << 23);

    for (uint offset = idx; offset < size; offset += gridDim.x*THREADS)
    {
        float4 x = load(add_ptr_u(X, offset));

        float4 mask;
        mask.x = rand_mask(keep_prob, lfsr0, lfsr1, lfsr2);
        mask.y = rand_mask(keep_prob, lfsr0, lfsr1, lfsr2);
        mask.z = rand_mask(keep_prob, lfsr0, lfsr1, lfsr2);
        mask.w = rand_mask(keep_prob, lfsr0, lfsr1, lfsr2);
        float4 y = ew_mul(ew_mul(x, mask), scale);
        store(add_ptr_u(Y, offset), y);

        Mask[offset] = float4_to_uint(mask);
    }
}

// Forward pass with existing mask (when forward pass needs to be recomputed)
template <typename T, uint THREADS>
__global__ void __launch_bounds__(THREADS) dropout_mask_forward(
    T* Y, const uint* __restrict__ Mask, const T* __restrict__ X, uint size, float scale)
{
    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    for (uint offset = bid*THREADS + tid; offset < size; offset += gridDim.x*THREADS)
    {
        float4 x = load(add_ptr_u(X, offset));
        float4 y = ew_mul(ew_mul(x, uint_to_float4(Mask[offset])), scale);
        store(add_ptr_u(Y, offset), y);
    }
}

template <typename T, uint THREADS>
__global__ void __launch_bounds__(THREADS) dropout_backward(
    T* DX, const uint* __restrict__ Mask, const T* __restrict__ DY, uint size, float scale)
{
    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    for (uint offset = bid*THREADS + tid; offset < size; offset += gridDim.x*THREADS)
    {
        float4 dy = load(add_ptr_u(DY, offset));
        float4 dx = ew_mul(ew_mul(dy, uint_to_float4(Mask[offset])), scale);
        store(add_ptr_u(DX, offset), dx);
    }
}

template <typename T, typename V>
bool DropoutForward(CUstream stream, uint SMs,
          T* y,
       char* m,
    const T* x,
    uint size, float keep_prob, float scale)
{
    size >>= 2; // use vector loads
    dropout_forward<V,512><<<SMs,512,0,stream>>>((V*)y, (uint*)m, (const V*)x, size, keep_prob, scale);
    return true;
}
template bool DropoutForward<float,float4>(CUstream stream, uint SMs, float* y, char* m, const float* x, uint size, float keep_prob, float scale);
template bool DropoutForward<ehalf,ehalf4>(CUstream stream, uint SMs, ehalf* y, char* m, const ehalf* x, uint size, float keep_prob, float scale);
template bool DropoutForward<bhalf,bhalf4>(CUstream stream, uint SMs, bhalf* y, char* m, const bhalf* x, uint size, float keep_prob, float scale);

template <typename T, typename V>
bool DropoutMaskForward(CUstream stream, uint SMs,
             T* y,
    const char* m,
    const    T* x,
    uint size, float scale)
{
    size >>= 2; // use vector loads
    uint grid = size > SMs*1024 ? SMs*2 : SMs;
    dropout_mask_forward<V,1024><<<grid,1024,0,stream>>>((V*)y, (const uint*)m, (const V*)x, size, scale);
    return true;
}
template bool DropoutMaskForward<float,float4>(CUstream stream, uint SMs, float* y, const char* m, const float* x, uint size, float scale);
template bool DropoutMaskForward<ehalf,ehalf4>(CUstream stream, uint SMs, ehalf* y, const char* m, const ehalf* x, uint size, float scale);
template bool DropoutMaskForward<bhalf,bhalf4>(CUstream stream, uint SMs, bhalf* y, const char* m, const bhalf* x, uint size, float scale);


template <typename T, typename V>
bool DropoutBackward(CUstream stream, uint SMs,
             T* dx,
    const char* m,
    const    T* dy,
    uint size, float scale)
{
    size >>= 2; // use vector loads
    uint grid = size > SMs*1024 ? SMs*2 : SMs;
    dropout_backward<V,1024><<<grid,1024,0,stream>>>((V*)dx, (const uint*)m, (const V*)dy, size, scale);
    return true;
}
template bool DropoutBackward<float,float4>(CUstream stream, uint SMs, float* dx, const char* m, const float* dy, uint size, float scale);
template bool DropoutBackward<ehalf,ehalf4>(CUstream stream, uint SMs, ehalf* dx, const char* m, const ehalf* dy, uint size, float scale);
template bool DropoutBackward<bhalf,bhalf4>(CUstream stream, uint SMs, bhalf* dx, const char* m, const bhalf* dy, uint size, float scale);



template <typename T, typename V>
__global__ void __launch_bounds__(32) add_n(
    struct plist8<T> X, T* Z,
    int size, int params)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i   = bid*32 + tid;
    if (i < size)
    {
        V z; ew_zero(z);
        #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            V x = load(X.a[j], i, j < params);
            z = ew_add(x, z);
        }
        store(Z, z, i);
    }
}

template <typename T, typename V>
bool AddN(CUstream stream, struct plist8<T>* x, T* z, int size, int params)
{
    if ((size & 3) == 0 && size >= 256)
    {
        size >>= 2; // use vector loads
        int grid = (size >> 5) + ((size & 31) != 0);

        struct plist8<V>* X = (struct plist8<V>*)x;

        add_n<V,float4><<<grid,32,0,stream>>>(*X, (V*)z, size, params);
    }
    else
    {
        int grid = (size >> 5) + ((size & 31) != 0);
        add_n<T,float><<<grid,32,0,stream>>>(*x, z, size, params);
    }
    return true;
}

template bool AddN<float,float4>(CUstream stream, struct plist8<float>* x, float* z, int size, int params);
template bool AddN<ehalf,ehalf4>(CUstream stream, struct plist8<ehalf>* x, ehalf* z, int size, int params);
template bool AddN<bhalf,bhalf4>(CUstream stream, struct plist8<bhalf>* x, bhalf* z, int size, int params);


template <typename T, typename V>
__global__ void __launch_bounds__(256) bias_relu(
          T*              Y,
    const T* __restrict__ X,
    const V* __restrict__ B,
    uint N, uint K, uint relu)
{
    uint nk = blockIdx.x*256 + threadIdx.x;

    uint n = nk / K;
    uint k = nk % K;

    if (n < N)
    {
        V b = load(add_ptr_u(B,  k));
        V x = load(add_ptr_u(X, nk));
        V y = ew_add(x, b);
        if (relu)
            y = ew_relu(y);

        store(add_ptr_u(Y, nk), y);
    }
}

template <typename T, typename V>
bool EW_Bias_Relu(CUstream stream,
              T* y,
    const     T* x,
    const float* b,
    uint N, uint K, bool relu)
{
    if ((K & 3) == 0)
    {
                   V* Y = (           V*)y;
        const      V* X = (const      V*)x;
        const float4* B = (const float4*)b;

        K >>= 2;
        uint grid = (N*K >> 8) + ((N*K & 255) != 0);
        bias_relu<V,float4><<<grid,256,0,stream>>>(Y, X, B, N, K, relu);
    }
    else
    {
        uint grid = (N*K >> 8) + ((N*K & 255) != 0);
        bias_relu<T,float ><<<grid,256,0,stream>>>(y, x, b, N, K, relu);
    }
    return true;
}

template bool EW_Bias_Relu<float,float4>(CUstream stream, float* y, const float* x, const float* b, uint N, uint K, bool relu);
template bool EW_Bias_Relu<ehalf,ehalf4>(CUstream stream, ehalf* y, const ehalf* x, const float* b, uint N, uint K, bool relu);
template bool EW_Bias_Relu<bhalf,bhalf4>(CUstream stream, bhalf* y, const bhalf* x, const float* b, uint N, uint K, bool relu);


// db = sum(dy, axis=0)
// dx = dy * (y > 0)
template <typename T, typename V, uint THREADS, uint WIDTH>
__global__ void __launch_bounds__(THREADS) bias_relu_grad(
          V*              DB,
          T*              DX,
    const T* __restrict__ DY,
    const T* __restrict__ Y,
    uint N, uint K, uint relu, uint partials)
{
    // Stripe the reduction lines with tid and block_n
    uint tid      = threadIdx.x;
    uint block_k  = blockIdx.x;
    uint block_n  = blockIdx.y;
    uint blocks_n = gridDim.y;

    uint warps = THREADS / 32;
    uint lines = THREADS / WIDTH;
    uint line  = tid     / WIDTH;

    uint k = block_k*WIDTH + (tid % WIDTH);
    uint n = block_n * lines + line;

    uint nk = n*K + k;
    bool bk = k < K;

    uint inc_n  = blocks_n * lines;
    uint inc_nk = inc_n*K;

    V db;
    ew_zero(db);
    while (n < N)
    {
        V dy = load(add_ptr_u(DY, nk), 0, bk);

        if (relu)
        {
            V  y = load(add_ptr_u(Y, nk), 0, bk);
            dy = ew_relu_grad(dy, y);
            store(add_ptr_u(DX, nk), dy, 0, bk);
        }
        db = ew_add(db, dy);

        nk += inc_nk;
        n  += inc_n;
    }
    // if the line width is less than a warp, reduce the lines within a warp
    for (int i = 16; i >= WIDTH; i >>= 1)
        db = ew_warp_sum(db, i);

    if (THREADS > 32)
    {
        __shared__ V Share[THREADS];
        if (tid >= 32)
            Share[tid] = db;

        __syncthreads();

        if (tid < WIDTH)
            for (uint i = 1; i < warps; i++)
                db = ew_add(db, Share[tid + i*32]);
    }
    // if blocks_n==0 then this is the final result
    // otherwise output a partial sum to be reduced with bias_grad2
    if (tid < WIDTH && bk)
    {
        if (gridDim.y == 1 || partials)
            store(add_ptr_u(DB, block_n*K + k), db);
        else
            atomicRed(add_ptr_u(DB, k), db);
    }
}

// Reduce partial sums for bias gradient
__global__ void __launch_bounds__(256) bias_grad2(
          float*              DB,
    const float* __restrict__ DB_Partial,
    uint N, uint K)
{
    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    // load in 8 units of k wide to allow efficient transpose in L1 cache
    uint k = bid*8 + tid/32;
    uint n = tid & 31;

    uint nk = n*K + k;
    bool bk = k < K;

    uint K1 = K*32;
    uint K2 = K*32*2;
    uint K3 = K2 + K1;
    uint K4 = K*32*4;

    float db = 0.0f;
    // We should generally have 128 or fewer partials per bias unit.
    // So pull them all in at once for lowest latency of this tiny kernel.
    #pragma unroll 1
    while (n < N)
    {
        float db0 = load(add_ptr_u(DB_Partial, nk +  0), 0, bk);
        float db1 = load(add_ptr_u(DB_Partial, nk + K1), 0, bk && (n+32*1 < N));
        float db2 = load(add_ptr_u(DB_Partial, nk + K2), 0, bk && (n+32*2 < N));
        float db3 = load(add_ptr_u(DB_Partial, nk + K3), 0, bk && (n+32*3 < N));

        db += (db0 + db1) + (db2 + db3);

        nk += K4;
        n  += 32*4;
    }
    for (uint i = 16; i > 0; i >>= 1)
        db += shfl_xor(db, i);

    store(add_ptr_u(DB, k), db, 0, bk & (tid & 31) == 0);
}

// This is just a rough tuning of the parameter space.  Ideally this would just be autotuned.
void EW_Bias_Relu_Grad_Partial(bool partials, uint N, uint K, uint *gridN, uint *gridK, uint *vec, uint *width)
{
    uint vshift = (K & 3) == 0 && K >= 8 && (partials || N <= 768) ? 2 : 0;
    uint wshift = 0;
    uint K_vec  = K >> vshift;
    uint SMs    = GetCountSMs();

    // narrow and deep reductions
    if (K_vec < 128)
    {
        uint K_vec_32 = K_vec & 31;
        uint K_vec_16 = K_vec & 15;
        uint K_vec_08 = K_vec &  7;

             if (K_vec_32 == 0 || K_vec_32 > 16) wshift = 5;
        else if (K_vec_16 == 0 || K_vec_16 >  8) wshift = 4;
        else if (K_vec_08 == 0 || K_vec_08 >  4) wshift = 3;
        else if (K_vec < 16)                     wshift = 2;
    }
    // wide and shallow reductions (with vector loads)
    else if (vshift && N <= 768)
    {
        uint K_vec_32 = K_vec >> 5;
        uint K_vec_16 = K_vec >> 4;
        uint K_vec_08 = K_vec >> 3;
             if (K_vec_32 >= SMs && K_vec_32 <= SMs*2) wshift = 5;
        else if (K_vec_16 >= SMs && K_vec_16 <= SMs*2) wshift = 4;
        else if (K_vec_08 >= SMs && K_vec_08 <= SMs*2) wshift = 3;
    }
    // anything else
    if (wshift == 0)
        wshift = K_vec < 128 ? 4 : 5;

    uint gk = (K_vec >> wshift) + ((K_vec & ((1 << wshift) - 1)) != 0);
    uint gn = 1;
    // Break up the reduction into blocks if needed
    if (N > 768 && gk < SMs)
    {
        // use 256 or 1024 thread blocks depending on vector loads
        uint tshift = vshift == 2 ? 8 : 10;
        // target as close to full occpancy as possible
        while (gn * gk <= SMs * (2048 >> tshift)) gn += 1;
        gn -= 1;
        if (gn == 0) gn = 1;
    }
    *gridN = gn;
    *gridK = gk;
    *vec   = 1 << vshift;
    *width = 1 << wshift;
}

template <typename T, typename V>
bool EW_Bias_Relu_Grad(CUstream stream,
          float* db,
          float* db_partial,
              T* dx,
    const     T* dy,
    const     T* y,
    uint gridN, uint gridK, uint vec, uint width, uint N, uint K, bool relu, bool partials)
{
    if (gridN > 1 && !partials)
        cuMemsetD32Async((CUdeviceptr)db, 0, K, stream);

    //printf("%d %d %d %d %d %d %d\n", gridN, gridK, vec, width, N, K, relu);
    if (vec == 4)
    {
        const V* DY = (const V*)dy;
        const V* Y  = (const V*)y;
              V* DX = (      V*)dx;
        float4* DB = gridN > 1 && partials ? (float4*)db_partial : (float4*)db;

        if      (width == 32)
            bias_relu_grad<V,float4,256,32><<<dim3(gridK,gridN),256,0,stream>>>(DB, DX, DY, Y, N, K >> 2, relu, partials);
        else if (width == 16)
            bias_relu_grad<V,float4,256,16><<<dim3(gridK,gridN),256,0,stream>>>(DB, DX, DY, Y, N, K >> 2, relu, partials);
        else if (width == 8)
            bias_relu_grad<V,float4,256, 8><<<dim3(gridK,gridN),256,0,stream>>>(DB, DX, DY, Y, N, K >> 2, relu, partials);
        else if (width == 4)
            bias_relu_grad<V,float4,256, 4><<<dim3(gridK,gridN),256,0,stream>>>(DB, DX, DY, Y, N, K >> 2, relu, partials);
    }
    else
    {
        float* DB = gridN > 1 && partials ? db_partial : db;

        if      (width == 32)
            bias_relu_grad<T,float,1024,32><<<dim3(gridK,gridN),1024,0,stream>>>(DB, dx, dy, y, N, K, relu, partials);
        else if (width == 16)
            bias_relu_grad<T,float,1024,16><<<dim3(gridK,gridN),1024,0,stream>>>(DB, dx, dy, y, N, K, relu, partials);
        else if (width == 8)
            bias_relu_grad<T,float,1024, 8><<<dim3(gridK,gridN),1024,0,stream>>>(DB, dx, dy, y, N, K, relu, partials);
        else if (width == 4)
            bias_relu_grad<T,float,1024, 4><<<dim3(gridK,gridN),1024,0,stream>>>(DB, dx, dy, y, N, K, relu, partials);
    }
    if (gridN > 1 && partials)
    {
        gridK = (K >> 3) + ((K & 7) != 0);

        bias_grad2<<<gridK,256,0,stream>>>(db, (const float*)db_partial, gridN, K);
    }
    return true;
}

template bool EW_Bias_Relu_Grad<float,float4>(CUstream stream, float* db, float* db_partial, float* dx, const float* dy, const float* y, uint gridN, uint gridK, uint vec, uint width, uint N, uint K, bool relu, bool partials);
template bool EW_Bias_Relu_Grad<ehalf,ehalf4>(CUstream stream, float* db, float* db_partial, ehalf* dx, const ehalf* dy, const ehalf* y, uint gridN, uint gridK, uint vec, uint width, uint N, uint K, bool relu, bool partials);
template bool EW_Bias_Relu_Grad<bhalf,bhalf4>(CUstream stream, float* db, float* db_partial, bhalf* dx, const bhalf* dy, const bhalf* y, uint gridN, uint gridK, uint vec, uint width, uint N, uint K, bool relu, bool partials);


// x = [128*16*18]
// a = range(0, 128*16) * 18 + idx[128*16], 1
template <typename T, typename TA>
__global__ void __launch_bounds__(64) fancy_gather1(
    T* Y, const TA* __restrict__ A, const T* __restrict__ X,
    uint dim0, uint dim1)
{
    uint idx0 = blockIdx.x*64 + threadIdx.x;
    if (idx0 < dim0)
    {
        uint idx1 = max(load(add_ptr_u(A, idx0)), 0);
        uint idxX = idx0*dim1 + idx1;

        store(add_ptr_u(Y, idx0), load(add_ptr_u(X, idxX), 0, idx1 < dim1));
    }
}
// x = [128*16*18]
// a = range(0, 128*16) * 18 + idx[128*16], 1
template <typename T, typename TA>
__global__ void fancy_gather1_grad(
    T* DX, const TA* __restrict__ A,  const T* __restrict__ DY,
    uint dim0, uint dim1)
{
    uint tid  = threadIdx.x;
    uint idx0 = blockIdx.x;

    uint idx1 = max(load(add_ptr_u(A, idx0)), 0);

    store(add_ptr_u(DX, idx0*dim1 + tid),
        load(add_ptr_u(DY, idx0), 0, idx1 == tid && idx1 < dim1),
        0, tid < dim1);
}
// x = [128*16*51, 64]
// a = range(0, 128*16) * 51 + idx[128*16], 1
template <typename T, typename TA>
__global__ void fancy_gather2(
    T* Y, const TA* __restrict__ A, const T* __restrict__ X,
    uint dim0, uint dim1, uint dim2)
{
    uint tid  = threadIdx.x;
    uint idx0 = blockIdx.x;

    uint idx1 = max(load(add_ptr_u(A, idx0)), 0);

    uint idxX = idx0*dim1*dim2 + idx1*dim2 + tid;
    uint idxY = idx0*dim2 + tid;

    store(add_ptr_u(Y, idxY),
        load(add_ptr_u(X, idxX), 0, tid < dim2 && idx1 < dim1),
         0, tid < dim2);
}
// x = [128*16*51, 64]
// a = range(0, 128*16) * 51 + idx[128*16], 1
template <typename T, typename TA>
__global__ void fancy_gather2_grad(
    T* DX, const TA* __restrict__ A,  const T* __restrict__ DY,
    uint dim0, uint dim1, uint dim2)
{
    uint tid  = threadIdx.x;
    uint idx0 = blockIdx.x;
    uint idx1 = blockIdx.y;

    if (tid < dim2)
    {
        uint idxX = idx0*dim1*dim2 + idx1*dim2 + tid;
        uint idxY = idx0*dim2 + tid;

        uint idx1a = max(load(add_ptr_u(A, idx0)), 0);

        store(add_ptr_u(DX, idxX),
            load(add_ptr_u(DY, idxY), 0, idx1a == idx1));
    }
}
template <typename T, typename TA>
bool EW_Fancy_Gather(CUstream stream, T* y, const TA* a, const T* x, uint dim0, uint dim1, uint dim2)
{
    if (dim2 == 1)
    {
        uint grid = CEIL_DIV(dim0, 64);
        fancy_gather1<T,TA><<<grid,64,0,stream>>>(y, a, x, dim0, dim1);
    }
    else
    {
        uint threads = CEIL_DIV(dim2, 32) * 32;
        fancy_gather2<T,TA><<<dim0,threads,0,stream>>>(y, a, x, dim0, dim1, dim2);
    }
    return true;
}
template <typename T, typename TA>
bool EW_Fancy_Gather_Grad(CUstream stream, T* dx, const TA* a, const T* dy, uint dim0, uint dim1, uint dim2)
{
    if (dim2 == 1)
    {
        uint threads = CEIL_DIV(dim1, 32) * 32;
        fancy_gather1_grad<T,TA><<<dim0,threads,0,stream>>>(dx, a, dy, dim0, dim1);
    }
    else
    {
        uint threads = CEIL_DIV(dim2, 32) * 32;
        fancy_gather2_grad<T,TA><<<dim3(dim0,dim1),threads,0,stream>>>(dx, a, dy, dim0, dim1, dim2);
    }
    return true;
}
template bool EW_Fancy_Gather<  int,int>(CUstream stream,   int* y, const int* a, const   int* x, uint dim0, uint dim1, uint dim2);
template bool EW_Fancy_Gather<float,int>(CUstream stream, float* y, const int* a, const float* x, uint dim0, uint dim1, uint dim2);
template bool EW_Fancy_Gather<ehalf,int>(CUstream stream, ehalf* y, const int* a, const ehalf* x, uint dim0, uint dim1, uint dim2);
template bool EW_Fancy_Gather<bhalf,int>(CUstream stream, bhalf* y, const int* a, const bhalf* x, uint dim0, uint dim1, uint dim2);

template bool EW_Fancy_Gather_Grad<float,int>(CUstream stream, float* dx, const int* a, const float* dy, uint dim0, uint dim1, uint dim2);
template bool EW_Fancy_Gather_Grad<ehalf,int>(CUstream stream, ehalf* dx, const int* a, const ehalf* dy, uint dim0, uint dim1, uint dim2);
template bool EW_Fancy_Gather_Grad<bhalf,int>(CUstream stream, bhalf* dx, const int* a, const bhalf* dy, uint dim0, uint dim1, uint dim2);

template <typename T, typename TA>
__global__ void reduce_column_max(
    T* Y, TA* A, const T* __restrict__ X,
    uint dim0, uint dim1, uint dim2)
{
    uint idx  = blockIdx.x*128 + threadIdx.x;
    uint idx0 = idx / dim2;
    uint idx2 = idx % dim2;

    if (idx0 < dim0)
    {
        uint offset = idx0*dim1*dim2 + idx2;

        float max_val = -FLT_MAX;
        uint  max_idx = 0;
        for (uint idx1 = 0; idx1 < dim1; idx1++)
        {
            float x = load(add_ptr_u(X, offset));

            if (max_val < x)
            {
                max_val = x;
                max_idx = idx1;
            }
            offset += dim2;
        }
        offset = idx0*dim2 + idx2;

        store(add_ptr_u(Y, offset), max_val);
        __stg(add_ptr_u(A, offset), max_idx);
    }
}

template <typename T, typename TA>
__global__ void reduce_column_max_grad(
    T* DX, const TA* __restrict__ A,  const T* __restrict__ DY,
    uint dim0, uint dim1, uint dim2)
{
    uint idx  = blockIdx.x*128 + threadIdx.x;
    uint idx0 = idx / dim2;
    uint idx2 = idx % dim2;

    if (idx0 < dim0)
    {
        uint offset_dy = idx0*dim2      + idx2;
        uint offset_dx = idx0*dim2*dim1 + idx2;

        uint  idx_max = __ldg(add_ptr_u(A, offset_dy));
        float dy = load(add_ptr_u(DY, offset_dy));

        for (uint idx1 = 0; idx1 < dim1; idx1++)
        {
            float dx = idx1 == idx_max ? dy : 0.0f;

            store(add_ptr_u(DX, offset_dx), dx);
            offset_dx += dim2;
        }
    }
}

template <typename T, typename TA>
bool EW_Reduce_Max(CUstream stream, T* y, TA* a, const T* x, uint dim0, uint dim1, uint dim2)
{
    uint dim02 = dim0*dim2;
    uint grid  = CEIL_DIV(dim02, 128);
    reduce_column_max<T,TA><<<grid,128,0,stream>>>(y, a, x, dim0, dim1, dim2);
    return true;
}

template <typename T, typename TA>
bool EW_Reduce_Max_Grad(CUstream stream, T* dx, const TA* a, const T* dy, uint dim0, uint dim1, uint dim2)
{
    uint dim02 = dim0*dim2;
    uint grid  = CEIL_DIV(dim02, 128);
    reduce_column_max_grad<T,TA><<<grid,128,0,stream>>>(dx, a, dy, dim0, dim1, dim2);
    return true;
}

template bool EW_Reduce_Max<float,unsigned char>(CUstream stream, float* y, unsigned char* a, const float* x, uint dim0, uint dim1, uint dim2);
template bool EW_Reduce_Max<ehalf,unsigned char>(CUstream stream, ehalf* y, unsigned char* a, const ehalf* x, uint dim0, uint dim1, uint dim2);
template bool EW_Reduce_Max<bhalf,unsigned char>(CUstream stream, bhalf* y, unsigned char* a, const bhalf* x, uint dim0, uint dim1, uint dim2);

template bool EW_Reduce_Max<float,       ushort>(CUstream stream, float* y,        ushort* a, const float* x, uint dim0, uint dim1, uint dim2);
template bool EW_Reduce_Max<ehalf,       ushort>(CUstream stream, ehalf* y,        ushort* a, const ehalf* x, uint dim0, uint dim1, uint dim2);
template bool EW_Reduce_Max<bhalf,       ushort>(CUstream stream, bhalf* y,        ushort* a, const bhalf* x, uint dim0, uint dim1, uint dim2);

template bool EW_Reduce_Max_Grad<float,unsigned char>(CUstream stream, float* dx, const unsigned char* a, const float* dy, uint dim0, uint dim1, uint dim2);
template bool EW_Reduce_Max_Grad<ehalf,unsigned char>(CUstream stream, ehalf* dx, const unsigned char* a, const ehalf* dy, uint dim0, uint dim1, uint dim2);
template bool EW_Reduce_Max_Grad<bhalf,unsigned char>(CUstream stream, bhalf* dx, const unsigned char* a, const bhalf* dy, uint dim0, uint dim1, uint dim2);

template bool EW_Reduce_Max_Grad<float,       ushort>(CUstream stream, float* dx, const        ushort* a, const float* dy, uint dim0, uint dim1, uint dim2);
template bool EW_Reduce_Max_Grad<ehalf,       ushort>(CUstream stream, ehalf* dx, const        ushort* a, const ehalf* dy, uint dim0, uint dim1, uint dim2);
template bool EW_Reduce_Max_Grad<bhalf,       ushort>(CUstream stream, bhalf* dx, const        ushort* a, const bhalf* dy, uint dim0, uint dim1, uint dim2);



#endif
