#if GOOGLE_CUDA

#include "ew_op_gpu.h"
#include <stdio.h>
#include <type_traits>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

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

OP_Z_XA(Elu,   ew_elu  )
OP_Z_XA(Gelu,  ew_gelu )
OP_Z_XA(Swish, ew_swish)


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

OP_DX_DZXA(EluGrad,     ew_elu_grad)
OP_DX_DZXA(GeluGrad,   ew_gelu_grad)
OP_DX_DZXA(SwishGrad, ew_swish_grad)

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
            case 16 :    Gelu<V,float4,2><<<grid,32,0,stream>>>(Z, X,    size, alpha);  break;
            case 17 :   Swish<V,float4,2><<<grid,32,0,stream>>>(Z, X,    size, alpha);  break;
            case 18 : BiasAdd<V,float4,2><<<dim3(grid,N),32,0,stream>>>(Z, X, B, size); break;
            case 19 : GainMul<V,float4,2><<<dim3(grid,N),32,0,stream>>>(Z, X, B, size); break;
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
            case 16 :    Gelu<T,float,4><<<grid,32,0,stream>>>(z, x,    size, alpha); break;
            case 17 :   Swish<T,float,4><<<grid,32,0,stream>>>(z, x,    size, alpha); break;
            case 18 : BiasAdd<T,float,4><<<dim3(grid,N),32,0,stream>>>(z, x, b, size); break;
            case 19 : GainMul<T,float,4><<<dim3(grid,N),32,0,stream>>>(z, x, b, size); break;
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
            case 16 :    GeluGrad<VB,VF,float4,2><<<grid64,32,0,stream>>>(DX, DZ, X, size, alpha);    break;
            case 17 :   SwishGrad<VB,VF,float4,2><<<grid64,32,0,stream>>>(DX, DZ, X, size, alpha);    break;
            case 18 : BiasAddGrad<VB,   float4  ><<<grid32,32,0,stream>>>(DB, DZ,           N, size); break;
            case 19 : GainMulGrad<VB,VF,float4  ><<<grid32,32,0,stream>>>(DX, DB, DZ, X, G, N, size); break;
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
            case 16 :    GeluGrad<B,F,float,4><<<grid128,32,0,stream>>>(dx, dz, x, size, alpha);    break;
            case 17 :   SwishGrad<B,F,float,4><<<grid128,32,0,stream>>>(dx, dz, x, size, alpha);    break;
            case 18 : BiasAddGrad<B,  float  ><<<grid32 ,32,0,stream>>>(db, dz,           N, size); break;
            case 19 : GainMulGrad<B,F,float  ><<<grid32 ,32,0,stream>>>(dx, db, dz, x, g, N, size); break;
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

// 2**-32
#define URAND_SCALE 2.3283064365386962891e-10f

__global__ void concrete_gate(uint* Entropy, float* Gate, float* Concrete, const float* LogA, float limit_a, float limit_b, float rcp_temp, float epsilon, uint size)
{
    uint tid = threadIdx.x;
    uint bid = blockIdx.x;
    uint idx = bid*blockDim.x + tid;

    if (idx < size)
    {
        uint lfsr0 = __ldg((const uint*)Entropy + (gridDim.x*blockDim.x*0 + idx));
        uint lfsr1 = __ldg((const uint*)Entropy + (gridDim.x*blockDim.x*1 + idx));
        uint lfsr2 = __ldg((const uint*)Entropy + (gridDim.x*blockDim.x*2 + idx));

        #pragma unroll 1
        for (uint offset = idx; offset < size; offset += gridDim.x * blockDim.x)
        {
            float loga = __ldg(LogA + offset);

            lfsr0 = ((lfsr0 & 0xfffffffe) << 12) ^ (((lfsr0 << 13) ^ lfsr0) >> 19);
            lfsr1 = ((lfsr1 & 0xfffffff8) <<  4) ^ (((lfsr1 << 2)  ^ lfsr1) >> 25);
            lfsr2 = ((lfsr2 & 0xfffffff0) << 11) ^ (((lfsr2 << 3)  ^ lfsr2) >> 11);
            uint  urand    = lfsr0 ^ lfsr1 ^ lfsr2;
            float frand    = (float)urand * URAND_SCALE * (1.0f - epsilon*2) + epsilon;
            float concrete = ew_sig((ew_log(frand) - ew_log(1.0f - frand) + loga) * rcp_temp);
            float stretch  = concrete * (limit_b - limit_a) + limit_a;
            float gate     = fminf(fmaxf(stretch, 0.0f), 1.0f);

            __stg(Gate     + offset,     gate);
            __stg(Concrete + offset, concrete);

        }
        __stg(Entropy + (gridDim.x*blockDim.x*0 + idx), lfsr0);
        __stg(Entropy + (gridDim.x*blockDim.x*1 + idx), lfsr1);
        __stg(Entropy + (gridDim.x*blockDim.x*2 + idx), lfsr2);
    }
}
__global__ void concrete_gate_grad(float* DLogA, const float* DGate, const float* Concrete, float limit_a, float limit_b, float rcp_temp, uint size)
{
    uint tid = threadIdx.x;
    uint bid = blockIdx.x;
    uint idx = bid*blockDim.x + tid;

    #pragma unroll 1
    for (uint offset = idx; offset < size; offset += gridDim.x * blockDim.x)
    {
        float dgate    = __ldg(DGate    + offset);
        float concrete = __ldg(Concrete + offset);

        float stretch   = concrete * (limit_b - limit_a) + limit_a;
        float d_tanh    = stretch >= 0.0 && stretch <= 1.0 ? dgate : 0.0f;
        float d_stretch = d_tanh * (limit_b - limit_a);
        float d_loga    = ew_sig_grad(d_stretch, concrete) * rcp_temp;

        __stg(DLogA + offset, d_loga);
    }
}
__global__ void concrete_gate_infer(float* Gate, const float* LogA, float limit_a, float limit_b, uint size)
{
    uint tid = threadIdx.x;
    uint bid = blockIdx.x;
    uint idx = bid*blockDim.x + tid;

    #pragma unroll 1
    for (uint offset = idx; offset < size; offset += gridDim.x * blockDim.x)
    {
        float loga = __ldg(LogA + offset);

        float stretch  = ew_sig(loga) * (limit_b - limit_a) + limit_a;
        float gate     = fminf(fmaxf(stretch, 0.0f), 1.0f);

        __stg(Gate + offset, gate);
    }
}
bool ConcreteGate(CUstream stream, uint SMs,
    uint* Entropy, float* Gate, float* Concrete, const float* LogA, float limit_a, float limit_b, float rcp_temp, float epsilon, uint size)
{
   uint threads =
        size >= SMs*1024*4 ? 1024 :
        size >= SMs* 512*4 ?  512 :
        size >= SMs* 256*4 ?  256 :
                              128 ;
    concrete_gate<<<SMs,threads,0,stream>>>(Entropy, Gate, Concrete, LogA, limit_a, limit_b, rcp_temp, epsilon, size);
    return true;
}
bool ConcreteGateGrad(CUstream stream, uint SMs,
    float* DLogA, const float* DGate, const float* Concrete, float limit_a, float limit_b, float rcp_temp, uint size)
{
   uint threads =
        size >= SMs*1024*2 ? 1024 :
        size >= SMs* 512*2 ?  512 :
        size >= SMs* 256*2 ?  256 :
                              128 ;
    concrete_gate_grad<<<SMs,threads,0,stream>>>(DLogA, DGate, Concrete, limit_a, limit_b, rcp_temp, size);
    return true;
}
bool ConcreteGateInfer(CUstream stream, uint SMs,
    float* Gate, const float* LogA, float limit_a, float limit_b, uint size)
{
   uint threads =
        size >= SMs*1024*2 ? 1024 :
        size >= SMs* 512*2 ?  512 :
        size >= SMs* 256*2 ?  256 :
                              128 ;
    concrete_gate_infer<<<SMs,threads,0,stream>>>(Gate, LogA, limit_a, limit_b, size);
    return true;
}

__global__ void __launch_bounds__(1024) gen_dropout_mask(uint* Entropy, uint* Mask, float keep_prob, uint size32)
{
    uint tid = threadIdx.x;
    uint bid = blockIdx.x;
    uint idx = bid*blockDim.x + tid;

    uint offsetM = idx/32;

    if (offsetM < size32)
    {
        uint lfsr0 = __ldg((const uint*)Entropy + (gridDim.x*blockDim.x*0 + idx));
        uint lfsr1 = __ldg((const uint*)Entropy + (gridDim.x*blockDim.x*1 + idx));
        uint lfsr2 = __ldg((const uint*)Entropy + (gridDim.x*blockDim.x*2 + idx));

        #pragma unroll 1
        for (uint offset = offsetM; offset < size32; offset += gridDim.x * blockDim.x/32)
        {
            lfsr0 = ((lfsr0 & 0xfffffffe) << 12) ^ (((lfsr0 << 13) ^ lfsr0) >> 19);
            lfsr1 = ((lfsr1 & 0xfffffff8) <<  4) ^ (((lfsr1 << 2)  ^ lfsr1) >> 25);
            lfsr2 = ((lfsr2 & 0xfffffff0) << 11) ^ (((lfsr2 << 3)  ^ lfsr2) >> 11);
            uint urand = lfsr0 ^ lfsr1 ^ lfsr2;
            bool keep  = (float)urand * URAND_SCALE < keep_prob;

# if CUDA_VERSION >= 9020
            uint mask  = __ballot_sync(0xffffffff, keep);
# else
            uint mask  = __ballot(keep);
# endif
            if ((tid & 31) == 0)
                __stg(Mask + offset, mask);
        }
        __stg(Entropy + (gridDim.x*blockDim.x*0 + idx), lfsr0);
        __stg(Entropy + (gridDim.x*blockDim.x*1 + idx), lfsr1);
        __stg(Entropy + (gridDim.x*blockDim.x*2 + idx), lfsr2);
    }
}
bool GenDropoutMask(CUstream stream, uint SMs, uint* Entropy, uint* Mask, float keep_prob, uint size)
{
    // try and get good entropy reuse (8x) and make this kernel mostly compute bound
    uint threads = 256;
         if (size >= SMs*1024*8) { threads = 1024; }
    else if (size >= SMs* 512*8) { threads =  512; }

    gen_dropout_mask<<<SMs,threads,0,stream>>>(Entropy, Mask, keep_prob, CEIL_DIV(size, 32));
    return true;
}

template <typename T, typename V, uint THREADS, uint DIMS>
__global__ void __launch_bounds__(THREADS) apply_dropout_mask(
    T* Y, const T* __restrict__ X, const uint* __restrict__ M,
    float scale,  uint x_size, Strides<5> x, Strides<5> m)
{
    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    const uint VSIZE = sizeof(V)/4;

    #pragma unroll 1
    for (uint offsetX = bid*THREADS + tid; offsetX < x_size; offsetX += gridDim.x*THREADS)
    {
        uint x_idx[DIMS];
        x_idx[0] = offsetX  * VSIZE;
        for (int i = 1; i < DIMS; i++)
        {
            x_idx[i  ] = x_idx[i-1] % x.stride[i-1];
            x_idx[i-1] = x_idx[i-1] / x.stride[i-1];
        }

        uint offsetM = 0;
        for (int i = 0; i < DIMS; i++)
            offsetM += x_idx[i] * m.stride[i];

        V    xval =  load(X + offsetX);
        uint mask = __ldg(M + offsetM/32);

        uint shift = offsetM & 31;
        for (uint i = 0; i < VSIZE; i++)
            if (((1 << (shift + i)) & mask) == 0) ((float*)&xval)[i] = 0.0f;

        store(Y + offsetX, ew_mul(xval, scale));
    }
}

template <typename V1, typename V4, typename V8>
bool ApplyDropoutMask(CUstream stream, uint SMs,
            V1* y,
    const   V1* x,
    const uint* m,
    float scale, uint size, int rank, Strides<5> &xs, Strides<5> &ms)
{
    // printf("xs: %d %d\n", xs.stride[0], xs.stride[1]);
    // printf("ms: %d %d\n", ms.stride[0], ms.stride[1]);

    // use vec8 loads
    if (rank == 1 && (size & 7) == 0 && sizeof(V1) == 2)
    {
        size >>= 3;
        uint grid = size > SMs*1024 ? SMs*2 : SMs;
        apply_dropout_mask<V8,float8,1024,1><<<grid,1024,0,stream>>>((V8*)y, (const V8*)x, m, scale, size, xs, ms);
    }
    // use vec4 loads
    else if (rank == 1 && (size & 3) == 0)
    {
        size >>= 2;
        uint grid = size > SMs*1024 ? SMs*2 : SMs;
        apply_dropout_mask<V4,float4,1024,1><<<grid,1024,0,stream>>>((V4*)y, (const V4*)x, m, scale, size, xs, ms);
    }
    // In broadcast mode we can't use vec loads
    else
    {
        uint grid = size > SMs*1024 ? SMs*2 : SMs;
             if (rank == 1)
            apply_dropout_mask<V1, float,1024,1><<<grid,1024,0,stream>>>(y, x, m, scale, size, xs, ms);
        else if (rank == 2)
            apply_dropout_mask<V1, float,1024,2><<<grid,1024,0,stream>>>(y, x, m, scale, size, xs, ms);
        else if (rank == 3)
            apply_dropout_mask<V1, float,1024,3><<<grid,1024,0,stream>>>(y, x, m, scale, size, xs, ms);
        else if (rank == 4)
            apply_dropout_mask<V1, float,1024,4><<<grid,1024,0,stream>>>(y, x, m, scale, size, xs, ms);
        else if (rank == 5)
            apply_dropout_mask<V1, float,1024,5><<<grid,1024,0,stream>>>(y, x, m, scale, size, xs, ms);

    }
    return true;
}
template bool ApplyDropoutMask<float,float4,float8>(CUstream stream, uint SMs, float* y, const float* x, const uint* m, float scale, uint size, int rank, Strides<5> &xs, Strides<5> &ms);
template bool ApplyDropoutMask<ehalf,ehalf4,ehalf8>(CUstream stream, uint SMs, ehalf* y, const ehalf* x, const uint* m, float scale, uint size, int rank, Strides<5> &xs, Strides<5> &ms);
template bool ApplyDropoutMask<bhalf,bhalf4,bhalf8>(CUstream stream, uint SMs, bhalf* y, const bhalf* x, const uint* m, float scale, uint size, int rank, Strides<5> &xs, Strides<5> &ms);




template <typename T, typename V>
__global__ void filter_tensor(T* Y, const T* __restrict__ X, uint size, float scale, float saturate, uint zero_infs, uint zero_nans)
{
    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    for (uint offset = bid*1024 + tid; offset < size; offset += gridDim.x*1024)
    {
        V x = load(X + offset);

        if (zero_infs)
            x = ew_zero_inf(x);
        if (zero_nans)
            x = ew_zero_nan(x);

        x = ew_mul(x, scale);

        if (saturate != 0.0f)
            x = ew_maximum(ew_minimum(x, saturate), -saturate);

        store(Y + offset, x);
    }
}
template <typename T, typename V>
bool FilterTensor(CUstream stream, uint SMs, T* y, const T* x, uint size, float scale, float saturate, bool zero_infs, bool zero_nans)
{
    if (size & 3)
    {
        uint grid = size > SMs*1024 ? SMs*2 : SMs;
        filter_tensor<T,float><<<grid,1024,0,stream>>>(y, x, size, scale, saturate, zero_infs, zero_nans);
    }
    else
    {
        size >>= 2; // use vector loads
        uint grid = size > SMs*1024 ? SMs*2 : SMs;
        filter_tensor<V,float4><<<grid,1024,0,stream>>>((V*)y, (const V*)x, size, scale, saturate, zero_infs, zero_nans);
    }
    return true;
}
template bool FilterTensor<float,float4>(CUstream stream, uint SMs, float* y, const float* x, uint size, float scale, float saturate, bool zero_infs, bool zero_nans);
template bool FilterTensor<ehalf,ehalf4>(CUstream stream, uint SMs, ehalf* y, const ehalf* x, uint size, float scale, float saturate, bool zero_infs, bool zero_nans);
template bool FilterTensor<bhalf,bhalf4>(CUstream stream, uint SMs, bhalf* y, const bhalf* x, uint size, float scale, float saturate, bool zero_infs, bool zero_nans);


template <typename T, typename V, uint THREADS, uint UNROLL>
__global__ void __launch_bounds__(THREADS,2) add_n(struct Plist<T,9> X, T* Y, uint size, uint params)
{
    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    for (uint offset = bid*THREADS + tid; offset < size; offset += gridDim.x*THREADS)
    {
        V y; ew_zero(y);
        #pragma unroll
        for (uint j = 0; j < UNROLL; j++)
        {
            V x = load(X.a[j] + offset, 0, j < params);
            y   = ew_add(x, y);
        }
        store(Y + offset, y);
    }
}

template <typename T, typename V>
bool AddN(CUstream stream, uint SMs, struct Plist<T,9>* x, T* y, uint size, uint params)
{
    if (size & 3)
    {
        uint grid = size > SMs*1024 ? SMs*2 : SMs;
        if (params > 5)
            add_n<T,float,1024,9><<<grid,1024,0,stream>>>(*x, y, size, params);
        else if (params > 3)
            add_n<T,float,1024,5><<<grid,1024,0,stream>>>(*x, y, size, params);
        else
            add_n<T,float,1024,3><<<grid,1024,0,stream>>>(*x, y, size, params);
    }
    // use vector loads
    else
    {
        size >>= 2;
        uint grid = size > SMs*1024 ? SMs*2 : SMs;

        struct Plist<V,9>* X = (struct Plist<V,9>*)x;

        if (params > 5)
            add_n<V,float4, 512,9><<<grid, 512,0,stream>>>(*X, (V*)y, size, params);
        else if (params > 3)
            add_n<V,float4,1024,5><<<grid,1024,0,stream>>>(*X, (V*)y, size, params);
        else
            add_n<V,float4,1024,3><<<grid,1024,0,stream>>>(*X, (V*)y, size, params);
    }
    return true;
}

template bool AddN<float,float4>(CUstream stream, uint SMs, struct Plist<float,9>* x, float* y, uint size, uint params);
template bool AddN<ehalf,ehalf4>(CUstream stream, uint SMs, struct Plist<ehalf,9>* x, ehalf* y, uint size, uint params);
template bool AddN<bhalf,bhalf4>(CUstream stream, uint SMs, struct Plist<bhalf,9>* x, bhalf* y, uint size, uint params);


template <typename T, typename V, uint RELU>
__global__ void __launch_bounds__(256) bias_relu_axis_0(
              T*              Y,
    const     T* __restrict__ X,
    const float* __restrict__ B,
    uint N, uint K)
{
    uint kn = blockIdx.x*256 + threadIdx.x;
    uint k  = kn / N;

    if (k < K)
    {
        float b = load(add_ptr_u(B, k));
        V x = load(add_ptr_u(X, kn));
        V y = ew_add(x, b);
        if (RELU == 2) // fast_gelu
            y = ew_swish(y, 1.702f);
        else if (RELU == 1) // relu
            y = ew_relu(y);

        store(add_ptr_u(Y, kn), y);
    }
}
template <typename T, typename V, uint RELU>
__global__ void __launch_bounds__(256) bias_relu_axis_1(
          T*              Y,
    const T* __restrict__ X,
    const V* __restrict__ B,
    uint N, uint K)
{
    uint nk = blockIdx.x*256 + threadIdx.x;

    uint n = nk / K;
    uint k = nk % K;

    if (n < N)
    {
        V b = load(add_ptr_u(B,  k));
        V x = load(add_ptr_u(X, nk));
        V y = ew_add(x, b);
        if (RELU == 2) // fast_gelu
            y = ew_swish(y, 1.702f);
        else if (RELU == 1) // relu
            y = ew_relu(y);

        store(add_ptr_u(Y, nk), y);
    }
}

template <typename T, typename V>
bool EW_Bias_Relu(CUstream stream,
              T* y,
    const     T* x,
    const float* b,
    uint axis, uint N, uint K, uint relu)
{
    if (axis == 0)
    {
        if ((N & 3) == 0)
        {
                  V* Y = (      V*)y;
            const V* X = (const V*)x;

            N >>= 2;
            uint grid = CEIL_DIV(N*K, 256);
            if (relu == 2)
                bias_relu_axis_0<V,float4,2><<<grid,256,0,stream>>>(Y, X, b, N, K);
            else if (relu == 1)
                bias_relu_axis_0<V,float4,1><<<grid,256,0,stream>>>(Y, X, b, N, K);
            else
                bias_relu_axis_0<V,float4,0><<<grid,256,0,stream>>>(Y, X, b, N, K);
        }
        else
        {
            uint grid = CEIL_DIV(N*K, 256);
            if (relu == 2)
                bias_relu_axis_0<T,float ,2><<<grid,256,0,stream>>>(y, x, b, N, K);
            else if (relu == 1)
                bias_relu_axis_0<T,float ,1><<<grid,256,0,stream>>>(y, x, b, N, K);
            else
                bias_relu_axis_0<T,float ,0><<<grid,256,0,stream>>>(y, x, b, N, K);
        }
    }
    else
    {
        if ((K & 3) == 0)
        {
                       V* Y = (           V*)y;
            const      V* X = (const      V*)x;
            const float4* B = (const float4*)b;

            K >>= 2;
            uint grid = CEIL_DIV(N*K, 256);
            if (relu == 2)
                bias_relu_axis_1<V,float4,2><<<grid,256,0,stream>>>(Y, X, B, N, K);
            else if (relu == 1)
                bias_relu_axis_1<V,float4,1><<<grid,256,0,stream>>>(Y, X, B, N, K);
            else
                bias_relu_axis_1<V,float4,0><<<grid,256,0,stream>>>(Y, X, B, N, K);
        }
        else
        {
            uint grid = CEIL_DIV(N*K, 256);
            if (relu == 2)
                bias_relu_axis_1<T,float ,2><<<grid,256,0,stream>>>(y, x, b, N, K);
            else if (relu == 1)
                bias_relu_axis_1<T,float ,1><<<grid,256,0,stream>>>(y, x, b, N, K);
            else
                bias_relu_axis_1<T,float ,0><<<grid,256,0,stream>>>(y, x, b, N, K);
        }
    }
    return true;
}

template bool EW_Bias_Relu<float,float4>(CUstream stream, float* y, const float* x, const float* b, uint axis, uint N, uint K, uint relu);
template bool EW_Bias_Relu<ehalf,ehalf4>(CUstream stream, ehalf* y, const ehalf* x, const float* b, uint axis, uint N, uint K, uint relu);
template bool EW_Bias_Relu<bhalf,bhalf4>(CUstream stream, bhalf* y, const bhalf* x, const float* b, uint axis, uint N, uint K, uint relu);

// db = sum(dy, axis=0)
// dx = dy * (y > 0)
template <typename T, typename V, uint RELU>
__global__ void bias_relu_axis_0_grad(
          float*              DB,
              T*              DX,
    const     T* __restrict__ DY,
    const     T* __restrict__ Y, // acutally X for fast_gelu
    const float* __restrict__ B,
    uint N, uint K)
{
    __shared__ float Share[32];

    uint tid = threadIdx.x;
    uint   k = blockIdx.x;
    float  b = RELU ? __ldg(B + k) : 0.0f;

    if (tid < 32)
        Share[tid] = 0.0f;

    V db_v;
    ew_zero(db_v);
    for (uint n = tid, kn = k*N + tid; n < N; n += blockDim.x, kn += blockDim.x)
    {
        V dy = load(DY + kn);

        if (RELU == 2) // fast_gelu
        {
            V x = load(Y + kn);

            dy = ew_swish_grad(dy, ew_add(x, b), 1.702f);

            store(DX + kn, dy);
        }
        else if (RELU == 1) // relu
        {
            V  y = load(Y + kn);
            dy = ew_relu_grad(dy, y);
            store(DX + kn, dy);
        }
        db_v = ew_add(db_v, dy);
    }
    // reduce within thread
    float db = ew_sum(db_v);

    // reduce within warp
    for (int i = 16; i > 0; i >>= 1)
        db += shfl_xor(db, i);

    // reduce across warps
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
            for (int i = 16; i > 0; i >>= 1)
                db += shfl_xor(db, i);
        }
    }
    if (tid == 0)
        __stg(DB + k, db);
}

// db = sum(dy, axis=0)
// dx = dy * (y > 0)
template <typename T, typename V, uint THREADS, uint WIDTH, uint RELU>
__global__ void __launch_bounds__(THREADS) bias_relu_axis_1_grad(
          V*              DB,
          T*              DX,
    const T* __restrict__ DY,
    const T* __restrict__ Y, // acutally X for fast_gelu
    const V* __restrict__ B,
    uint N, uint K, uint partials)
{
    // Stripe the reduction lines with tid and block_n
    uint tid      = threadIdx.x;
    uint block_k  = blockIdx.x;
    uint block_n  = blockIdx.y;
    uint blocks_n = gridDim.y;

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

        if (RELU == 2) // fast_gelu
        {
            V x = load(add_ptr_u(Y, nk), 0, bk);
            V b = load(add_ptr_u(B,  k), 0, bk);

            dy = ew_swish_grad(dy, ew_add(x, b), 1.702f);

            store(add_ptr_u(DX, nk), dy, 0, bk);
        }
        else if (RELU == 1)
        {
            V  y = load(add_ptr_u(Y, nk), 0, bk);
            dy = ew_relu_grad(dy, y);
            store(add_ptr_u(DX, nk), dy, 0, bk);
        }
        db = ew_add(db, dy);

        nk += inc_nk;
        n  += inc_n;
    }

    if (THREADS > 32)
    {
        __shared__ V Share[THREADS];
        if (tid >= 64)
            Share[tid] = db;

        __syncthreads();

        if (tid < 64)
        {
            for (uint i = 1; i < THREADS/64; i++)
                db = ew_add(db, Share[tid + i*64]);

            Share[tid] = db;
        }
        __syncthreads();

        if (tid < 32)
            db = ew_add(db, Share[tid + 32]);
    }
    // if the line width is less than a warp, reduce the lines within a warp
    for (int i = 16; i >= WIDTH; i >>= 1)
        db = ew_warp_sum(db, i);

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
    const     T* y, // x for gelu
    const float* b,
    uint axis, uint gridN, uint gridK, uint vec, uint width, uint N, uint K, uint relu, bool partials)
{
    if (axis == 0)
    {
        if ((N & 3) == 0)
        {
            N >>= 2;
            const V* DY = (const V*)dy;
            const V*  Y = (const V*)y;
                  V* DX = (      V*)dx;

            uint threads = MIN(CEIL_DIV(N, 128), 32) * 32;
            if (relu == 2)
                bias_relu_axis_0_grad<V,float4,2><<<K,threads,0,stream>>>(db, DX, DY, Y, b, N, K);
            else if (relu == 1)
                bias_relu_axis_0_grad<V,float4,1><<<K,threads,0,stream>>>(db, DX, DY, Y, b, N, K);
            else
                bias_relu_axis_0_grad<V,float4,0><<<K,threads,0,stream>>>(db, DX, DY, Y, b, N, K);
        }
        else
        {
            uint threads = MIN(CEIL_DIV(N, 128), 32) * 32;
            if (relu == 2)
                bias_relu_axis_0_grad<T, float,2><<<K,threads,0,stream>>>(db, dx, dy, y, b, N, K);
            else if (relu == 1)
                bias_relu_axis_0_grad<T, float,1><<<K,threads,0,stream>>>(db, dx, dy, y, b, N, K);
            else
                bias_relu_axis_0_grad<T, float,0><<<K,threads,0,stream>>>(db, dx, dy, y, b, N, K);
        }
    }
    else
    {
        if (gridN > 1 && !partials)
            cuMemsetD32Async((CUdeviceptr)db, 0, K, stream);

        //printf("%d %d %d %d %d %d %d\n", gridN, gridK, vec, width, N, K, relu);
        if (vec == 4)
        {
            const      V* DY = (const V*)dy;
            const      V* Y  = (const V*)y;
            const float4* B  = (const float4*)b;

                  V* DX = (      V*)dx;
            float4* DB = gridN > 1 && partials ? (float4*)db_partial : (float4*)db;

            if (relu == 2) // fast_gelu
            {
                if      (width == 32)
                    bias_relu_axis_1_grad<V,float4,256,32,2><<<dim3(gridK,gridN),256,0,stream>>>(DB, DX, DY, Y, B, N, K >> 2, partials);
                else if (width == 16)
                    bias_relu_axis_1_grad<V,float4,256,16,2><<<dim3(gridK,gridN),256,0,stream>>>(DB, DX, DY, Y, B, N, K >> 2, partials);
                else if (width == 8)
                    bias_relu_axis_1_grad<V,float4,256, 8,2><<<dim3(gridK,gridN),256,0,stream>>>(DB, DX, DY, Y, B, N, K >> 2, partials);
                else if (width == 4)
                    bias_relu_axis_1_grad<V,float4,256, 4,2><<<dim3(gridK,gridN),256,0,stream>>>(DB, DX, DY, Y, B, N, K >> 2, partials);
            }
            else if (relu == 1) // relu
            {
                if      (width == 32)
                    bias_relu_axis_1_grad<V,float4,256,32,1><<<dim3(gridK,gridN),256,0,stream>>>(DB, DX, DY, Y, B, N, K >> 2, partials);
                else if (width == 16)
                    bias_relu_axis_1_grad<V,float4,256,16,1><<<dim3(gridK,gridN),256,0,stream>>>(DB, DX, DY, Y, B, N, K >> 2, partials);
                else if (width == 8)
                    bias_relu_axis_1_grad<V,float4,256, 8,1><<<dim3(gridK,gridN),256,0,stream>>>(DB, DX, DY, Y, B, N, K >> 2, partials);
                else if (width == 4)
                    bias_relu_axis_1_grad<V,float4,256, 4,1><<<dim3(gridK,gridN),256,0,stream>>>(DB, DX, DY, Y, B, N, K >> 2, partials);
            }
            else // no act
            {
                if      (width == 32)
                    bias_relu_axis_1_grad<V,float4,256,32,0><<<dim3(gridK,gridN),256,0,stream>>>(DB, DX, DY, Y, B, N, K >> 2, partials);
                else if (width == 16)
                    bias_relu_axis_1_grad<V,float4,256,16,0><<<dim3(gridK,gridN),256,0,stream>>>(DB, DX, DY, Y, B, N, K >> 2, partials);
                else if (width == 8)
                    bias_relu_axis_1_grad<V,float4,256, 8,0><<<dim3(gridK,gridN),256,0,stream>>>(DB, DX, DY, Y, B, N, K >> 2, partials);
                else if (width == 4)
                    bias_relu_axis_1_grad<V,float4,256, 4,0><<<dim3(gridK,gridN),256,0,stream>>>(DB, DX, DY, Y, B, N, K >> 2, partials);
            }
        }
        else
        {
            float* DB = gridN > 1 && partials ? db_partial : db;

            if (relu == 2) // fast_gelu
            {
                if      (width == 32)
                    bias_relu_axis_1_grad<T,float,1024,32,2><<<dim3(gridK,gridN),1024,0,stream>>>(DB, dx, dy, y, b, N, K, partials);
                else if (width == 16)
                    bias_relu_axis_1_grad<T,float,1024,16,2><<<dim3(gridK,gridN),1024,0,stream>>>(DB, dx, dy, y, b, N, K, partials);
                else if (width == 8)
                    bias_relu_axis_1_grad<T,float,1024, 8,2><<<dim3(gridK,gridN),1024,0,stream>>>(DB, dx, dy, y, b, N, K, partials);
                else if (width == 4)
                    bias_relu_axis_1_grad<T,float,1024, 4,2><<<dim3(gridK,gridN),1024,0,stream>>>(DB, dx, dy, y, b, N, K, partials);
            }
            else if (relu == 1) // relu
            {
                if      (width == 32)
                    bias_relu_axis_1_grad<T,float,1024,32,1><<<dim3(gridK,gridN),1024,0,stream>>>(DB, dx, dy, y, b, N, K, partials);
                else if (width == 16)
                    bias_relu_axis_1_grad<T,float,1024,16,1><<<dim3(gridK,gridN),1024,0,stream>>>(DB, dx, dy, y, b, N, K, partials);
                else if (width == 8)
                    bias_relu_axis_1_grad<T,float,1024, 8,1><<<dim3(gridK,gridN),1024,0,stream>>>(DB, dx, dy, y, b, N, K, partials);
                else if (width == 4)
                    bias_relu_axis_1_grad<T,float,1024, 4,1><<<dim3(gridK,gridN),1024,0,stream>>>(DB, dx, dy, y, b, N, K, partials);
            }
            else // no act
            {
                if      (width == 32)
                    bias_relu_axis_1_grad<T,float,1024,32,0><<<dim3(gridK,gridN),1024,0,stream>>>(DB, dx, dy, y, b, N, K, partials);
                else if (width == 16)
                    bias_relu_axis_1_grad<T,float,1024,16,0><<<dim3(gridK,gridN),1024,0,stream>>>(DB, dx, dy, y, b, N, K, partials);
                else if (width == 8)
                    bias_relu_axis_1_grad<T,float,1024, 8,0><<<dim3(gridK,gridN),1024,0,stream>>>(DB, dx, dy, y, b, N, K, partials);
                else if (width == 4)
                    bias_relu_axis_1_grad<T,float,1024, 4,0><<<dim3(gridK,gridN),1024,0,stream>>>(DB, dx, dy, y, b, N, K, partials);
            }
        }
        if (gridN > 1 && partials)
        {
            gridK = (K >> 3) + ((K & 7) != 0);

            bias_grad2<<<gridK,256,0,stream>>>(db, (const float*)db_partial, gridN, K);
        }
    }
    return true;
}

template bool EW_Bias_Relu_Grad<float,float4>(CUstream stream, float* db, float* db_partial, float* dx, const float* dy, const float* y, const float* b, uint axis, uint gridN, uint gridK, uint vec, uint width, uint N, uint K, uint relu, bool partials);
template bool EW_Bias_Relu_Grad<ehalf,ehalf4>(CUstream stream, float* db, float* db_partial, ehalf* dx, const ehalf* dy, const ehalf* y, const float* b, uint axis, uint gridN, uint gridK, uint vec, uint width, uint N, uint K, uint relu, bool partials);
template bool EW_Bias_Relu_Grad<bhalf,bhalf4>(CUstream stream, float* db, float* db_partial, bhalf* dx, const bhalf* dy, const bhalf* y, const float* b, uint axis, uint gridN, uint gridK, uint vec, uint width, uint N, uint K, uint relu, bool partials);


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



template <typename T, typename V>
__global__ void assign_add(T* Y, const T* __restrict__ X, uint size)
{
    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    for (uint offset = bid*1024 + tid; offset < size; offset += gridDim.x*1024)
    {
        V x = load(X + offset);
        V y = load((const T*)(Y + offset));

        store(Y + offset, ew_add(x, y));
    }
}
template <typename T, typename V>
bool AssignAdd(CUstream stream, uint SMs, T* y, const T* x, uint size)
{
    if (size & 3)
    {
        uint grid = size > SMs*1024 ? SMs*2 : SMs;
        assign_add<T,float><<<grid,1024,0,stream>>>(y, x, size);
    }
    else
    {
        size >>= 2; // use vector loads
        uint grid = size > SMs*1024 ? SMs*2 : SMs;
        assign_add<V,float4><<<grid,1024,0,stream>>>((V*)y, (const V*)x, size);
    }
    return true;
}
template bool AssignAdd<float,float4>(CUstream stream, uint SMs, float* y, const float* x, uint size);
template bool AssignAdd<ehalf,ehalf4>(CUstream stream, uint SMs, ehalf* y, const ehalf* x, uint size);
template bool AssignAdd<bhalf,bhalf4>(CUstream stream, uint SMs, bhalf* y, const bhalf* x, uint size);



#endif
