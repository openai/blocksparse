#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "ew_op_gpu.h"

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
    if ((size & 3) == 0 && size >= 256)
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

template
bool EW_Backward<float,ehalf,float4,ehalf4>(CUstream stream,
          float* dx,
          float* dy,
          float* db,
    const float* dz,
    const ehalf* x,
    const ehalf* y,
    const ehalf* z,
    const float* g,
    float alpha, int size, int N, int op);

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

template
bool EW_Backward<float,bhalf,float4,bhalf4>(CUstream stream,
          float* dx,
          float* dy,
          float* db,
    const float* dz,
    const bhalf* x,
    const bhalf* y,
    const bhalf* z,
    const float* g,
    float alpha, int size, int N, int op);

template <typename T, typename V>
__global__ void __launch_bounds__(32) LSTM_Forward(
          T*              C_next,
          T*              H_next,
    const T* __restrict__ C_prev,
    const T* __restrict__ H_prev,
    int K, int K4)
{
    int tid = threadIdx.x;
    int k   = blockIdx.x;
    int n   = blockIdx.y;

    int k4 = k*32 + tid;
    int x0 = n*K + k4;
    int x1 = x0 + K4;
    int x2 = x1 + K4;
    int x3 = x2 + K4;
    int  z = n*K4 + k4;
    bool b = k4 < K4;

    V c = load(C_prev,  z, b);
    V i = load(H_prev, x0, b);
    V f = load(H_prev, x1, b);
    V o = load(H_prev, x2, b);
    V u = load(H_prev, x3, b);

    V sig_i = ew_sig(i);
    V sig_f = ew_sig(f);
    V sig_o = ew_sig(o);
    V tan_u = ew_tanh(u);

    V c_nxt = ew_add(ew_mul(sig_f, c), ew_mul(sig_i, tan_u));
    V c_act = ew_tanh(c_nxt);
    V h_nxt = ew_mul(sig_o, c_act);

    store_f(C_next, c_nxt, z, b);
    store_f(H_next, h_nxt, z, b);
}
template <typename T, typename V>
__global__ void __launch_bounds__(32) LSTM4_Forward(
          T*              C_next,
          T*              H_next,
    const T* __restrict__ C,
    const T* __restrict__ I,
    const T* __restrict__ F,
    const T* __restrict__ O,
    const T* __restrict__ U,
    int size)
{
    int tid = threadIdx.x;
    int   x = blockIdx.x*32 + tid;
    bool  b = x < size;

    V c = load(C, x, b);
    V i = load(I, x, b);
    V f = load(F, x, b);
    V o = load(O, x, b);
    V u = load(U, x, b);

    V sig_i = ew_sig(i);
    V sig_f = ew_sig(f);
    V sig_o = ew_sig(o);
    V tan_u = ew_tanh(u);

    V c_nxt = ew_add(ew_mul(sig_f, c), ew_mul(sig_i, tan_u));
    V c_act = ew_tanh(c_nxt);
    V h_nxt = ew_mul(sig_o, c_act);

    store_f(C_next, c_nxt, x, b);
    store_f(H_next, h_nxt, x, b);
}
template <typename B, typename F, typename V>
__global__ void __launch_bounds__(32) LSTM_Backward(
          B*              DC,
          B*              DH,
    const B* __restrict__ EC,
    const B* __restrict__ EH,
    const F* __restrict__ C_prev,
    const F* __restrict__ H_prev,
    int K, int K4, int ec_valid)
{
    int tid = threadIdx.x;
    int k   = blockIdx.x;
    int n   = blockIdx.y;

    int k4 = k*32 + tid;
    int x0 = n*K + k4;
    int x1 = x0 + K4;
    int x2 = x1 + K4;
    int x3 = x2 + K4;
    int  z = n*K4 + k4;
    bool b = k4 < K4;

    V  i = load(H_prev, x0, b);
    V  f = load(H_prev, x1, b);
    V  o = load(H_prev, x2, b);
    V  u = load(H_prev, x3, b);
    V  c = load(C_prev,  z, b);
    V eh = load(EH, z, b);
    V ec = load(EC, z, b && ec_valid);

    V sig_i = ew_sig(i);
    V sig_f = ew_sig(f);
    V sig_o = ew_sig(o);
    V tan_u = ew_tanh(u);

    V c_nxt = ew_add(ew_mul(sig_f, c), ew_mul(sig_i, tan_u));
    V c_act = ew_tanh(c_nxt);

    V dC = ew_add(ew_tanh_grad(ew_mul(eh, sig_o), c_act), ec);
    V dI = ew_sig_grad(ew_mul(dC, tan_u), sig_i);
    V dF = ew_sig_grad(ew_mul(dC,     c), sig_f);
    V dO = ew_sig_grad(ew_mul( eh, c_act), sig_o);
    V dU = ew_tanh_grad(ew_mul(dC, sig_i), tan_u);
    dC = ew_mul(dC, sig_f);

    store_g(DC, dC,  z, b);
    store_g(DH, dI, x0, b);
    store_g(DH, dF, x1, b);
    store_g(DH, dO, x2, b);
    store_g(DH, dU, x3, b);
}
template <typename B, typename A, typename V>
__global__ void __launch_bounds__(32) LSTM4_Backward(
          B*              DC,
          B*              DI,
          B*              DF,
          B*              DO,
          B*              DU,
    const B* __restrict__ EC,
    const B* __restrict__ EH,
    const A* __restrict__ C,
    const A* __restrict__ I,
    const A* __restrict__ F,
    const A* __restrict__ O,
    const A* __restrict__ U,
    int size, int ec_valid)
{
    int tid = threadIdx.x;
    int   x = blockIdx.x*32 + tid;
    bool  b = x < size;

    V  c = load(C, x, b);
    V  i = load(I, x, b);
    V  f = load(F, x, b);
    V  o = load(O, x, b);
    V  u = load(U, x, b);
    V eh = load(EH, x, b);
    V ec = load(EC, x, b && ec_valid);

    V sig_i = ew_sig(i);
    V sig_f = ew_sig(f);
    V sig_o = ew_sig(o);
    V tan_u = ew_tanh(u);

    V c_nxt = ew_add(ew_mul(sig_f, c), ew_mul(sig_i, tan_u));
    V c_act = ew_tanh(c_nxt);

    V dC = ew_add(ew_tanh_grad(ew_mul(eh, sig_o), c_act), ec);
    V dI = ew_sig_grad(ew_mul(dC, tan_u), sig_i);
    V dF = ew_sig_grad(ew_mul(dC,     c), sig_f);
    V dO = ew_sig_grad(ew_mul( eh, c_act), sig_o);
    V dU = ew_tanh_grad(ew_mul(dC, sig_i), tan_u);
    dC = ew_mul(dC, sig_f);

    store_g(DC, dC, x, b);
    store_g(DI, dI, x, b);
    store_g(DF, dF, x, b);
    store_g(DO, dO, x, b);
    store_g(DU, dU, x, b);
}

template <typename T, typename V>
bool LSTM_Gates_Forward(CUstream stream, T* c_next, T* h_next, const T* c_prev, const T* h_prev, int N, int K)
{
    int K4 = K >> 2;
    if ((K4 & 3) == 0)
    {
        K  >>= 2; // use vector loads
        K4 >>= 2;
        dim3 grid((K4 >> 5) + ((K4 & 31) != 0), N);

        V* C_next = (V*)c_next;
        V* H_next = (V*)h_next;
        const V* C_prev = (const V*)c_prev;
        const V* H_prev = (const V*)h_prev;
        LSTM_Forward<V,float4><<<grid,32,0,stream>>>(C_next, H_next, C_prev, H_prev, K, K4);
    }
    else
    {
        dim3 grid((K4 >> 5) + ((K4 & 31) != 0), N);
        LSTM_Forward<T,float ><<<grid,32,0,stream>>>(c_next, h_next, c_prev, h_prev, K, K4);
    }
    return true;
}
template <typename T, typename V>
bool LSTM4_Gates_Forward(CUstream stream, T* c_next, T* h_next, const T* c, const T* i, const T* f, const T* o, const T* u, int N, int K)
{
    int size = N * K;
    if ((size & 3) == 0)
    {
        size >>= 2; // use vector loads
        int grid = (size >> 5) + ((size & 31) != 0);

        V* C_next = (V*)c_next;
        V* H_next = (V*)h_next;
        const V* C = (const V*)c;
        const V* I = (const V*)i;
        const V* F = (const V*)f;
        const V* O = (const V*)o;
        const V* U = (const V*)u;
        LSTM4_Forward<V,float4><<<grid,32,0,stream>>>(C_next, H_next, C, I, F, O, U, size);
    }
    else
    {
        int grid = (size >> 5) + ((size & 31) != 0);
        LSTM4_Forward<T,float ><<<grid,32,0,stream>>>(c_next, h_next, c, i, f, o, u, size);
    }
    return true;
}
template <typename B, typename F, typename VB, typename VF>
bool LSTM_Gates_Backward(CUstream stream, B* dc, B* dh, const B* ec, const B* eh, const F* c_prev, const F* h_prev, int N, int K)
{
    int K4 = K >> 2;
    if ((K4 & 3) == 0)
    {
        K  >>= 2; // use vector loads
        K4 >>= 2;
        dim3 grid((K4 >> 5) + ((K4 & 31) != 0), N);

              VB* DC     = (      VB*)dc;
              VB* DH     = (      VB*)dh;
        const VB* EC     = (const VB*)ec;
        const VB* EH     = (const VB*)eh;
        const VF* C_prev = (const VF*)c_prev;
        const VF* H_prev = (const VF*)h_prev;

        LSTM_Backward<VB,VF,float4><<<grid,32,0,stream>>>(DC, DH, EC, EH, C_prev, H_prev, K, K4, ec != 0);
    }
    else
    {
        dim3 grid((K4 >> 5) + ((K4 & 31) != 0), N);
        LSTM_Backward< B, F,float ><<<grid,32,0,stream>>>(dc, dh, ec, eh, c_prev, h_prev, K, K4, ec != 0);
    }
    return true;
}
template <typename B, typename A, typename VB, typename VA>
bool LSTM4_Gates_Backward(CUstream stream, B* dc, B* di, B* df, B* doo, B* du, const B* ec, const B* eh, const A* c, const A* i, const A* f, const A* o, const A* u, int N, int K)
{
    int size = N * K;
    if ((size & 3) == 0)
    {
        size >>= 2; // use vector loads
        int grid = (size >> 5) + ((size & 31) != 0);

              VB* DC = (      VB*)dc;
              VB* DI = (      VB*)di;
              VB* DF = (      VB*)df;
              VB* DO = (      VB*)doo;
              VB* DU = (      VB*)du;
        const VB* EC = (const VB*)ec;
        const VB* EH = (const VB*)eh;
        const VA* C  = (const VA*)c;
        const VA* I  = (const VA*)i;
        const VA* F  = (const VA*)f;
        const VA* O  = (const VA*)o;
        const VA* U  = (const VA*)u;

        LSTM4_Backward<VB,VA,float4><<<grid,32,0,stream>>>(DC, DI, DF, DO, DU, EC, EH, C, I, F, O, U, size, ec != 0);
    }
    else
    {
        int grid = (size >> 5) + ((size & 31) != 0);
        LSTM4_Backward< B, A,float ><<<grid,32,0,stream>>>(dc, di, df, doo, du, ec, eh, c, i, f, o, u, size, ec != 0);
    }
    return true;
}

template bool LSTM_Gates_Forward <float,float4>(CUstream stream, float* c_next, float* h_next, const float* c_prev, const float* h_prev, int N, int K);
template bool LSTM_Gates_Forward <ehalf,ehalf4>(CUstream stream, ehalf* c_next, ehalf* h_next, const ehalf* c_prev, const ehalf* h_prev, int N, int K);
template bool LSTM_Gates_Forward <bhalf,bhalf4>(CUstream stream, bhalf* c_next, bhalf* h_next, const bhalf* c_prev, const bhalf* h_prev, int N, int K);

template bool LSTM_Gates_Backward<float,float,float4,float4>(CUstream stream, float* dc, float* dh, const float* ec, const float* eh, const float* c_prev, const float* h_prev, int N, int K);
template bool LSTM_Gates_Backward<ehalf,ehalf,ehalf4,ehalf4>(CUstream stream, ehalf* dc, ehalf* dh, const ehalf* ec, const ehalf* eh, const ehalf* c_prev, const ehalf* h_prev, int N, int K);
template bool LSTM_Gates_Backward<float,ehalf,float4,ehalf4>(CUstream stream, float* dc, float* dh, const float* ec, const float* eh, const ehalf* c_prev, const ehalf* h_prev, int N, int K);
template bool LSTM_Gates_Backward<bhalf,bhalf,bhalf4,bhalf4>(CUstream stream, bhalf* dc, bhalf* dh, const bhalf* ec, const bhalf* eh, const bhalf* c_prev, const bhalf* h_prev, int N, int K);
template bool LSTM_Gates_Backward<float,bhalf,float4,bhalf4>(CUstream stream, float* dc, float* dh, const float* ec, const float* eh, const bhalf* c_prev, const bhalf* h_prev, int N, int K);

template bool LSTM4_Gates_Forward <float,float4>(CUstream stream, float* c_next, float* h_next, const float* c, const float* i, const float* f, const float* o, const float* u, int N, int K);
template bool LSTM4_Gates_Forward <ehalf,ehalf4>(CUstream stream, ehalf* c_next, ehalf* h_next, const ehalf* c, const ehalf* i, const ehalf* f, const ehalf* o, const ehalf* u, int N, int K);
template bool LSTM4_Gates_Forward <bhalf,bhalf4>(CUstream stream, bhalf* c_next, bhalf* h_next, const bhalf* c, const bhalf* i, const bhalf* f, const bhalf* o, const bhalf* u, int N, int K);

template bool LSTM4_Gates_Backward<float,float,float4,float4>(CUstream stream, float* dc, float* di, float* df, float* doo, float* du, const float* ec, const float* eh, const float* c, const float* i, const float* f, const float* o, const float* u, int N, int K);
template bool LSTM4_Gates_Backward<ehalf,ehalf,ehalf4,ehalf4>(CUstream stream, ehalf* dc, ehalf* di, ehalf* df, ehalf* doo, ehalf* du, const ehalf* ec, const ehalf* eh, const ehalf* c, const ehalf* i, const ehalf* f, const ehalf* o, const ehalf* u, int N, int K);
template bool LSTM4_Gates_Backward<float,ehalf,float4,ehalf4>(CUstream stream, float* dc, float* di, float* df, float* doo, float* du, const float* ec, const float* eh, const ehalf* c, const ehalf* i, const ehalf* f, const ehalf* o, const ehalf* u, int N, int K);
template bool LSTM4_Gates_Backward<bhalf,bhalf,bhalf4,bhalf4>(CUstream stream, bhalf* dc, bhalf* di, bhalf* df, bhalf* doo, bhalf* du, const bhalf* ec, const bhalf* eh, const bhalf* c, const bhalf* i, const bhalf* f, const bhalf* o, const bhalf* u, int N, int K);
template bool LSTM4_Gates_Backward<float,bhalf,float4,bhalf4>(CUstream stream, float* dc, float* di, float* df, float* doo, float* du, const float* ec, const float* eh, const bhalf* c, const bhalf* i, const bhalf* f, const bhalf* o, const bhalf* u, int N, int K);


template <typename T, typename V>
__global__ void __launch_bounds__(32) Split4(
          T*              Z0,
          T*              Z1,
          T*              Z2,
          T*              Z3,
    const T* __restrict__ X,
    int K, int K4)
{
    int tid = threadIdx.x;
    int k   = blockIdx.x;
    int n   = blockIdx.y;

    int k4 = k*32 + tid;
    int i0 = n*K + k4;
    int i1 = i0 + K4;
    int i2 = i1 + K4;
    int i3 = i2 + K4;
    int  z = n*K4 + k4;
    bool b = k4 < K4;

    V x0 = load(X, i0, b);
    V x1 = load(X, i1, b);
    V x2 = load(X, i2, b);
    V x3 = load(X, i3, b);

    store(Z0, x0, z, b);
    store(Z1, x1, z, b);
    store(Z2, x2, z, b);
    store(Z3, x3, z, b);
}
template <typename T, typename V>
__global__ void __launch_bounds__(32) Concat4(
          T*              DX,
    const T* __restrict__ DZ0,
    const T* __restrict__ DZ1,
    const T* __restrict__ DZ2,
    const T* __restrict__ DZ3,
    int K, int K4)
{
    int tid = threadIdx.x;
    int k   = blockIdx.x;
    int n   = blockIdx.y;

    int k4 = k*32 + tid;
    int i0 = n*K + k4;
    int i1 = i0 + K4;
    int i2 = i1 + K4;
    int i3 = i2 + K4;
    int  z = n*K4 + k4;
    bool b = k4 < K4;

    V dx0 = load(DZ0, z, b);
    V dx1 = load(DZ1, z, b);
    V dx2 = load(DZ2, z, b);
    V dx3 = load(DZ3, z, b);

    store(DX, dx0, i0, b);
    store(DX, dx1, i1, b);
    store(DX, dx2, i2, b);
    store(DX, dx3, i3, b);
}


template <typename T, typename V>
bool Split4_Forward(CUstream stream, T* z0, T* z1, T* z2, T* z3, const T* x, int N, int K)
{
    int K4 = K >> 2;
    if ((K4 & 3) == 0)
    {
        K  >>= 2; // use vector loads
        K4 >>= 2;
        dim3 grid((K4 >> 5) + ((K4 & 31) != 0), N);

        V* Z0 = (V*)z0;
        V* Z1 = (V*)z1;
        V* Z2 = (V*)z2;
        V* Z3 = (V*)z3;
        const V* X = (const V*)x;
        Split4<V,float4><<<grid,32,0,stream>>>(Z0, Z1, Z2, Z3, X, K, K4);
    }
    else
    {
        dim3 grid((K4 >> 5) + ((K4 & 31) != 0), N);
        Split4<T,float ><<<grid,32,0,stream>>>(z0, z1, z2, z3, x, K, K4);
    }
    return true;
}

template <typename T, typename V>
bool Concat4_Forward(CUstream stream, T* dx, const T* z0, const T* z1, const T* z2, const T* z3, int N, int K)
{
    int K4 = K >> 2;
    if ((K4 & 3) == 0)
    {
        K  >>= 2; // use vector loads
        K4 >>= 2;
        dim3 grid((K4 >> 5) + ((K4 & 31) != 0), N);

        V* DX = (V*)dx;
        const V* Z0 = (const V*)z0;
        const V* Z1 = (const V*)z1;
        const V* Z2 = (const V*)z2;
        const V* Z3 = (const V*)z3;
        Concat4<V,float4><<<grid,32,0,stream>>>(DX, Z0, Z1, Z2, Z3, K, K4);
    }
    else
    {
        dim3 grid((K4 >> 5) + ((K4 & 31) != 0), N);
        Concat4<T,float ><<<grid,32,0,stream>>>(dx, z0, z1, z2, z3, K, K4);
    }
    return true;
}

template bool Split4_Forward <float,float4>(CUstream stream, float* z0, float* z1, float* z2, float* z3, const float* x, int N, int K);
template bool Split4_Forward <ehalf,ehalf4>(CUstream stream, ehalf* z0, ehalf* z1, ehalf* z2, ehalf* z3, const ehalf* x, int N, int K);
template bool Split4_Forward <bhalf,bhalf4>(CUstream stream, bhalf* z0, bhalf* z1, bhalf* z2, bhalf* z3, const bhalf* x, int N, int K);

template bool Concat4_Forward<float,float4>(CUstream stream, float* dx, const float* z0, const float* z1, const float* z2, const float* z3, int N, int K);
template bool Concat4_Forward<ehalf,ehalf4>(CUstream stream, ehalf* dx, const ehalf* z0, const ehalf* z1, const ehalf* z2, const ehalf* z3, int N, int K);
template bool Concat4_Forward<bhalf,bhalf4>(CUstream stream, bhalf* dx, const bhalf* z0, const bhalf* z1, const bhalf* z2, const bhalf* z3, int N, int K);


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



// mean   = mean(x, axis=1)
// std    = std(x, axis=1)
// cutoff = mean + alpha*std
// y      = fmaxf(x, cutoff) - cutoff;
template <typename T, typename V, int THREADS>
__global__ void __launch_bounds__(THREADS) sparse_relu_forward(
              T*              Y,
    const     T* __restrict__ X,
    float alpha, int K, float rcpK)
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

    // Standard Deviation (std)
    const T* X2 = X + offset;
    V v_std;
    ew_zero(v_std);
    for (int k = tid; k < K; k += THREADS)
    {
        V x = load(X2);
        v_std = ew_add(v_std, ew_sqr(ew_sub(x, mean)));
        X2   += THREADS;
    }
    float std = ew_sum(v_std);
    // reduce within warp
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1)
        std += __shfl_xor(std, i);
    // first thread of each warp store to shared
    if ((tid & 31) == 0)
        Share[tid >> 5] = std;
    __syncthreads();
    if (tid < (THREADS>>5))
    {
        // first warp loads all prior reductions
        std = Share[tid];
        // reduce within this last warp
        #pragma unroll
        for (int i = (THREADS>>6); i > 0; i >>= 1)
            std += __shfl_xor(std, i);

        std = sqrtf(std*rcpK);

        // Outputs final reduction to shared
        // Also cache reductions for backward pass
        if (tid == 0)
            Share[0] = std;
    }
    __syncthreads();
    // broadcast result to all threads
    std = Share[0];

    // Norm/Gain/Bias
    X += offset;
    Y += offset;
    for (int k = tid; k < K; k += THREADS)
    {
        float cutoff = mean + alpha*std;
        V x = load(X);
        V y = ew_sub(ew_maximum(x, cutoff), cutoff);
        store(Y, y, 0, true);
        X += THREADS;
        Y += THREADS;
    }
}

template <typename T, typename V>
bool SparseReluForward(CUstream stream, T* y, const T* x, float alpha, int K, int N)
{
    dim3 grid(N, 1, 1);
    float rcpK = 1.0f / (float)K;

    if ((K & 3) == 0)
    {
        K >>= 2; // use vector loads
                   V* Y = (V*)y;
        const      V* X = (const V*)x;
        // if (K >= 1024)
        //     sparse_relu_forward<V,float4,1024><<<grid,1024,0,stream>>>(Y, X, alpha, K, rcpK);
        if (K >= 256)
            sparse_relu_forward<V,float4, 256><<<grid, 256,0,stream>>>(Y, X, alpha, K, rcpK);
        else
            sparse_relu_forward<V,float4,  64><<<grid,  64,0,stream>>>(Y, X, alpha, K, rcpK);
    }
    else
    {
        // if (K >= 1024)
        //     sparse_relu_forward<T,float ,1024><<<grid,1024,0,stream>>>(y, x, alpha, K, rcpK);
        if (K >= 256)
            sparse_relu_forward<T,float , 256><<<grid, 256,0,stream>>>(y, x, alpha, K, rcpK);
        else
            sparse_relu_forward<T,float ,  64><<<grid,  64,0,stream>>>(y, x, alpha, K, rcpK);
    }
    return true; // TODO
}
template bool SparseReluForward<float,float4>(CUstream stream, float* y, const float* x,float alpha, int K, int N);
template bool SparseReluForward<ehalf,ehalf4>(CUstream stream, ehalf* y, const ehalf* x,float alpha, int K, int N);
template bool SparseReluForward<bhalf,bhalf4>(CUstream stream, bhalf* y, const bhalf* x,float alpha, int K, int N);



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
__device__ __forceinline__ float rand_mask(float keep_prob, unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2)
{
    lfsr0 = ((lfsr0 & 0xfffffffe) << 12) ^ (((lfsr0 << 13) ^ lfsr0) >> 19);
    lfsr1 = ((lfsr1 & 0xfffffff8) <<  4) ^ (((lfsr1 << 2)  ^ lfsr1) >> 25);
    lfsr2 = ((lfsr2 & 0xfffffff0) << 11) ^ (((lfsr2 << 3)  ^ lfsr2) >> 11);
    unsigned urand = lfsr0 ^ lfsr1 ^ lfsr2;
    // (float)urand * 2**-32 > keep_prob ? 0.0f : 1.0f;
    float val;
    asm("cvt.rn.f32.u32 %0, %1;\n\t"
        "mul.f32 %0, %0, 0F2f800000;"
        : "=f"(val) : "r"(urand));
    return val > keep_prob ? 0.0f : 1.0f;
}


template <typename T, int U>
__global__ void __launch_bounds__(32) dropout_forward(
    T* Y, uint* Mask, const T* __restrict__ X, int size, float keep_prob)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    unsigned lfsr0, lfsr1, lfsr2;
    unsigned idx = bid * 32 + tid;
    asm("mov.b32 %0, %%clock_hi;"       : "=r"(lfsr0) :);
    asm("mov.b32 %0, %%clock;"          : "=r"(lfsr1) :);
    asm("mov.b32 %0, %%globaltimer_lo;" : "=r"(lfsr2) :);
    asm("shf.r.clamp.b32 %0,%0,%0,%1;"  : "=r"(lfsr0) : "r"((lfsr0 & 31)^tid));
    asm("shf.r.clamp.b32 %0,%0,%0,%1;"  : "=r"(lfsr1) : "r"((lfsr1 & 31)^tid));
    asm("shf.r.clamp.b32 %0,%0,%0,%1;"  : "=r"(lfsr2) : "r"((lfsr2 & 31)^tid));
    lfsr0 ^= idx ^ (idx << 5)  ^ (idx << 11) ^ (idx << 17) ^ (idx << 23);

    int i = bid * U*32 + tid;
    for (int j = 0; j < U; j++)
    {
        bool b = i < size;
        float4 x = load(X, i, b);

        float4 mask;
        mask.x = rand_mask(keep_prob, lfsr0, lfsr1, lfsr2);
        mask.y = rand_mask(keep_prob, lfsr0, lfsr1, lfsr2);
        mask.z = rand_mask(keep_prob, lfsr0, lfsr1, lfsr2);
        mask.w = rand_mask(keep_prob, lfsr0, lfsr1, lfsr2);
        float4 y = ew_mul(x, mask);
        store(Y, y, i, b);

        uint mask_out = float4_to_uint(mask);
        if (b) Mask[i] = mask_out;

        i += 32;
    }
}

// Forward pass with existing mask (when forward pass needs to be recomputed)
template <typename T, int U>
__global__ void __launch_bounds__(32) dropout_mask_forward(
    T* Y, const uint* __restrict__ Mask, const T* __restrict__ X, int size)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int i = bid * U*32 + tid;
    for (int j = 0; j < U; j++)
    {
        bool b = i < size;
        float4 x = load(X, i, b);

        uint mask;
        if (b) mask = Mask[i];

        float4 y = ew_mul(x, uint_to_float4(mask));
        store(Y, y, i, b);

        i += 32;
    }
}

template <typename T, int U>
__global__ void __launch_bounds__(32) dropout_backward(
    T* DX, const uint* __restrict__ Mask, const T* __restrict__ DY, int size)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int i = bid * U*32 + tid;
    for (int j = 0; j < U; j++)
    {
        bool b = i < size;
        float4 dy = load(DY, i, b);

        uint mask;
        if (b) mask = Mask[i];

        float4 dx = ew_mul(dy, uint_to_float4(mask));
        store(DX, dx, i, b);

        i += 32;
    }
}

template <typename T, typename V>
bool DropoutForward(CUstream stream,
          T* y,
       char* m,
    const T* x,
    int size, float keep_prob)
{
    size >>= 2; // use vector loads
    int grid = (size >> 6) + ((size & 63) != 0); // 1 warp with 2 unrolls

    dropout_forward<V,2><<<grid,32,0,stream>>>((V*)y, (uint*)m, (const V*)x, size, keep_prob);

    return true;
}
template bool DropoutForward<float,float4>(CUstream stream, float* y, char* m, const float* x, int size, float keep_prob);
template bool DropoutForward<ehalf,ehalf4>(CUstream stream, ehalf* y, char* m, const ehalf* x, int size, float keep_prob);
template bool DropoutForward<bhalf,bhalf4>(CUstream stream, bhalf* y, char* m, const bhalf* x, int size, float keep_prob);

template <typename T, typename V>
bool DropoutMaskForward(CUstream stream,
             T* y,
    const char* m,
    const    T* x,
    int size)
{
    size >>= 2; // use vector loads
    int grid = (size >> 6) + ((size & 63) != 0); // 1 warp with 2 unrolls

    dropout_mask_forward<V,2><<<grid,32,0,stream>>>((V*)y, (const uint*)m, (const V*)x, size);

    return true;
}
template bool DropoutMaskForward<float,float4>(CUstream stream, float* y, const char* m, const float* x, int size);
template bool DropoutMaskForward<ehalf,ehalf4>(CUstream stream, ehalf* y, const char* m, const ehalf* x, int size);
template bool DropoutMaskForward<bhalf,bhalf4>(CUstream stream, bhalf* y, const char* m, const bhalf* x, int size);


template <typename T, typename V>
bool DropoutBackward(CUstream stream,
             T* dx,
    const char* m,
    const    T* dy,
    int size)
{
    size >>= 2; // use vector loads
    int grid = (size >> 6) + ((size & 63) != 0); // 1 warp with 2 unrolls

    dropout_backward<V,2><<<grid,32,0,stream>>>((V*)dx, (const uint*)m, (const V*)dy, size);

    return true;
}
template bool DropoutBackward<float,float4>(CUstream stream, float* dx, const char* m, const float* dy, int size);
template bool DropoutBackward<ehalf,ehalf4>(CUstream stream, ehalf* dx, const char* m, const ehalf* dy, int size);
template bool DropoutBackward<bhalf,bhalf4>(CUstream stream, bhalf* dx, const char* m, const bhalf* dy, int size);



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

#endif