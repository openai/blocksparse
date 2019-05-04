#if GOOGLE_CUDA

#include "ew_op_gpu.h"
#include <stdio.h>

template <typename T, typename V>
__global__ void __launch_bounds__(32) LSTM_Forward(
          T*              C_next,
          T*              H_next,
    const T* __restrict__ C_prev,
    const T* __restrict__ H_prev,
    float forget_bias, int K, int K4)
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

    // Match tf.nn.rnn_cell.BasicLSTMCell layout
    V c = load(C_prev,  z, b);
    V i = load(H_prev, x0, b);
    V u = load(H_prev, x1, b);
    V f = load(H_prev, x2, b);
    V o = load(H_prev, x3, b);

    V sig_i = ew_sig(i);
    V tan_u = ew_tanh(u);
    V sig_f = ew_sig(ew_add(f, forget_bias));
    V sig_o = ew_sig(o);

    V c_nxt = ew_add(ew_mul(sig_f, c), ew_mul(sig_i, tan_u));
    V c_act = ew_tanh(c_nxt);
    V h_nxt = ew_mul(sig_o, c_act);

    store(C_next, c_nxt, z, b);
    store(H_next, h_nxt, z, b);
}
template <typename T, typename V>
__global__ void __launch_bounds__(32) LSTM_Bias_Forward(
          T*              C_next,
          T*              H_next,
    const T* __restrict__ C_prev,
    const T* __restrict__ H_prev,
    const V* __restrict__ Bias,
    float forget_bias, int K, int K4)
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

    // Match tf.nn.rnn_cell.BasicLSTMCell layout
    V ib = load(Bias, k4,             b);
    V ub = load(Bias, k4 + K4,        b);
    V fb = load(Bias, k4 + K4*2,      b);
    V ob = load(Bias, k4 + K4*2 + K4, b);

    V c = load(C_prev,  z, b);
    V i = load(H_prev, x0, b);
    V u = load(H_prev, x1, b);
    V f = load(H_prev, x2, b);
    V o = load(H_prev, x3, b);

    V sig_i = ew_sig(ew_add(i, ib));
    V sig_f = ew_sig(ew_add(ew_add(f, fb), forget_bias));
    V sig_o = ew_sig(ew_add(o, ob));
    V tan_u = ew_tanh(ew_add(u, ub));

    V c_nxt = ew_add(ew_mul(sig_f, c), ew_mul(sig_i, tan_u));
    V c_act = ew_tanh(c_nxt);
    V h_nxt = ew_mul(sig_o, c_act);

    store(C_next, c_nxt, z, b);
    store(H_next, h_nxt, z, b);
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
    float forget_bias, int size)
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
    V sig_f = ew_sig(ew_add(f, forget_bias));
    V sig_o = ew_sig(o);
    V tan_u = ew_tanh(u);

    V c_nxt = ew_add(ew_mul(sig_f, c), ew_mul(sig_i, tan_u));
    V c_act = ew_tanh(c_nxt);
    V h_nxt = ew_mul(sig_o, c_act);

    store(C_next, c_nxt, x, b);
    store(H_next, h_nxt, x, b);
}
template <typename B, typename F, typename V>
__global__ void __launch_bounds__(32) LSTM_Backward(
          B*              DC,
          B*              DH,
    const B* __restrict__ EC,
    const B* __restrict__ EH,
    const F* __restrict__ C_prev,
    const F* __restrict__ H_prev,
    int K, int K4, int ec_valid, float forget_bias)
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
    V  u = load(H_prev, x1, b);
    V  f = load(H_prev, x2, b);
    V  o = load(H_prev, x3, b);
    V  c = load(C_prev,  z, b);
    V eh = load(EH, z, b);
    V ec = load(EC, z, b && ec_valid);

    V sig_i = ew_sig(i);
    V sig_f = ew_sig(ew_add(f, forget_bias));
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

    store(DC, dC,  z, b);
    store(DH, dI, x0, b);
    store(DH, dU, x1, b);
    store(DH, dF, x2, b);
    store(DH, dO, x3, b);
}
template <typename B, typename F, typename V>
__global__ void __launch_bounds__(32) LSTM_Bias_Backward(
          B*              DC,
          B*              DH,
    const B* __restrict__ EC,
    const B* __restrict__ EH,
    const F* __restrict__ C_prev,
    const F* __restrict__ H_prev,
    const V* __restrict__ Bias,
    int K, int K4, int ec_valid, float forget_bias)
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

    // Match tf.nn.rnn_cell.BasicLSTMCell layout
    V ib = load(Bias, k4,             b);
    V ub = load(Bias, k4 + K4,        b);
    V fb = load(Bias, k4 + K4*2,      b);
    V ob = load(Bias, k4 + K4*2 + K4, b);

    V  i = load(H_prev, x0, b);
    V  u = load(H_prev, x1, b);
    V  f = load(H_prev, x2, b);
    V  o = load(H_prev, x3, b);
    V  c = load(C_prev,  z, b);
    V eh = load(EH, z, b);
    V ec = load(EC, z, b && ec_valid);

    V sig_i = ew_sig(ew_add(i, ib));
    V sig_f = ew_sig(ew_add(ew_add(f, fb), forget_bias));
    V sig_o = ew_sig(ew_add(o, ob));
    V tan_u = ew_tanh(ew_add(u, ub));

    V c_nxt = ew_add(ew_mul(sig_f, c), ew_mul(sig_i, tan_u));
    V c_act = ew_tanh(c_nxt);

    V dC = ew_add(ew_tanh_grad(ew_mul(eh, sig_o), c_act), ec);
    V dI = ew_sig_grad(ew_mul(dC, tan_u), sig_i);
    V dF = ew_sig_grad(ew_mul(dC,     c), sig_f);
    V dO = ew_sig_grad(ew_mul( eh, c_act), sig_o);
    V dU = ew_tanh_grad(ew_mul(dC, sig_i), tan_u);
    dC = ew_mul(dC, sig_f);

    store(DC, dC,  z, b);
    store(DH, dI, x0, b);
    store(DH, dU, x1, b);
    store(DH, dF, x2, b);
    store(DH, dO, x3, b);
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
    int size, int ec_valid, float forget_bias)
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
    V sig_f = ew_sig(ew_add(f, forget_bias));
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

    store(DC, dC, x, b);
    store(DI, dI, x, b);
    store(DF, dF, x, b);
    store(DO, dO, x, b);
    store(DU, dU, x, b);
}

template <typename T, typename V>
bool LSTM_Gates_Forward(CUstream stream, T* c_next, T* h_next, const T* c_prev, const T* h_prev, const float* bias, float forget_bias, int N, int K)
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
        const float4* Bias = (const float4*)bias;
        if (bias == NULL)
            LSTM_Forward<V,float4><<<grid,32,0,stream>>>(C_next, H_next, C_prev, H_prev, forget_bias, K, K4);
        else
            LSTM_Bias_Forward<V,float4><<<grid,32,0,stream>>>(C_next, H_next, C_prev, H_prev, Bias, forget_bias, K, K4);

    }
    else
    {
        dim3 grid((K4 >> 5) + ((K4 & 31) != 0), N);
        if (bias == NULL)
            LSTM_Forward<T,float><<<grid,32,0,stream>>>(c_next, h_next, c_prev, h_prev, forget_bias, K, K4);
        else
            LSTM_Bias_Forward<T,float><<<grid,32,0,stream>>>(c_next, h_next, c_prev, h_prev, bias, forget_bias, K, K4);

    }
    return true;
}
template <typename T, typename V>
bool LSTM4_Gates_Forward(CUstream stream, T* c_next, T* h_next, const T* c, const T* i, const T* f, const T* o, const T* u, float forget_bias, int N, int K)
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
        LSTM4_Forward<V,float4><<<grid,32,0,stream>>>(C_next, H_next, C, I, F, O, U, forget_bias, size);
    }
    else
    {
        int grid = (size >> 5) + ((size & 31) != 0);
        LSTM4_Forward<T,float ><<<grid,32,0,stream>>>(c_next, h_next, c, i, f, o, u, forget_bias, size);
    }
    return true;
}
template <typename B, typename F, typename VB, typename VF>
bool LSTM_Gates_Backward(CUstream stream, B* dc, B* dh, const B* ec, const B* eh, const F* c_prev, const F* h_prev, const float* bias, int N, int K, float forget_bias)
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
        const float4* Bias = (const float4*)bias;

        if (bias == NULL)
            LSTM_Backward<VB,VF,float4><<<grid,32,0,stream>>>(DC, DH, EC, EH, C_prev, H_prev, K, K4, ec != 0, forget_bias);
        else
            LSTM_Bias_Backward<VB,VF,float4><<<grid,32,0,stream>>>(DC, DH, EC, EH, C_prev, H_prev, Bias, K, K4, ec != 0, forget_bias);
    }
    else
    {
        dim3 grid((K4 >> 5) + ((K4 & 31) != 0), N);
        if (bias == NULL)
            LSTM_Backward< B, F,float ><<<grid,32,0,stream>>>(dc, dh, ec, eh, c_prev, h_prev, K, K4, ec != 0, forget_bias);
        else
            LSTM_Bias_Backward< B, F,float ><<<grid,32,0,stream>>>(dc, dh, ec, eh, c_prev, h_prev, bias, K, K4, ec != 0, forget_bias);

    }
    return true;
}
template <typename B, typename A, typename VB, typename VA>
bool LSTM4_Gates_Backward(CUstream stream, B* dc, B* di, B* df, B* doo, B* du, const B* ec, const B* eh, const A* c, const A* i, const A* f, const A* o, const A* u, int N, int K, float forget_bias)
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

        LSTM4_Backward<VB,VA,float4><<<grid,32,0,stream>>>(DC, DI, DF, DO, DU, EC, EH, C, I, F, O, U, size, ec != 0, forget_bias);
    }
    else
    {
        int grid = (size >> 5) + ((size & 31) != 0);
        LSTM4_Backward< B, A,float ><<<grid,32,0,stream>>>(dc, di, df, doo, du, ec, eh, c, i, f, o, u, size, ec != 0, forget_bias);
    }
    return true;
}

template bool LSTM_Gates_Forward <float,float4>(CUstream stream, float* c_next, float* h_next, const float* c_prev, const float* h_prev, const float* bias, float forget_bias, int N, int K);
template bool LSTM_Gates_Forward <ehalf,ehalf4>(CUstream stream, ehalf* c_next, ehalf* h_next, const ehalf* c_prev, const ehalf* h_prev, const float* bias, float forget_bias, int N, int K);
template bool LSTM_Gates_Forward <bhalf,bhalf4>(CUstream stream, bhalf* c_next, bhalf* h_next, const bhalf* c_prev, const bhalf* h_prev, const float* bias, float forget_bias, int N, int K);

template bool LSTM_Gates_Backward<float,float,float4,float4>(CUstream stream, float* dc, float* dh, const float* ec, const float* eh, const float* c_prev, const float* h_prev, const float* bias, int N, int K, float forget_bias);
template bool LSTM_Gates_Backward<ehalf,ehalf,ehalf4,ehalf4>(CUstream stream, ehalf* dc, ehalf* dh, const ehalf* ec, const ehalf* eh, const ehalf* c_prev, const ehalf* h_prev, const float* bias, int N, int K, float forget_bias);
//template bool LSTM_Gates_Backward<float,ehalf,float4,ehalf4>(CUstream stream, float* dc, float* dh, const float* ec, const float* eh, const ehalf* c_prev, const ehalf* h_prev, const float* bias, int N, int K, float forget_bias);
template bool LSTM_Gates_Backward<bhalf,bhalf,bhalf4,bhalf4>(CUstream stream, bhalf* dc, bhalf* dh, const bhalf* ec, const bhalf* eh, const bhalf* c_prev, const bhalf* h_prev, const float* bias, int N, int K, float forget_bias);
//template bool LSTM_Gates_Backward<float,bhalf,float4,bhalf4>(CUstream stream, float* dc, float* dh, const float* ec, const float* eh, const bhalf* c_prev, const bhalf* h_prev, const float* bias, int N, int K, float forget_bias);

template bool LSTM4_Gates_Forward <float,float4>(CUstream stream, float* c_next, float* h_next, const float* c, const float* i, const float* f, const float* o, const float* u, float forget_bias, int N, int K);
template bool LSTM4_Gates_Forward <ehalf,ehalf4>(CUstream stream, ehalf* c_next, ehalf* h_next, const ehalf* c, const ehalf* i, const ehalf* f, const ehalf* o, const ehalf* u, float forget_bias, int N, int K);
template bool LSTM4_Gates_Forward <bhalf,bhalf4>(CUstream stream, bhalf* c_next, bhalf* h_next, const bhalf* c, const bhalf* i, const bhalf* f, const bhalf* o, const bhalf* u, float forget_bias, int N, int K);

template bool LSTM4_Gates_Backward<float,float,float4,float4>(CUstream stream, float* dc, float* di, float* df, float* doo, float* du, const float* ec, const float* eh, const float* c, const float* i, const float* f, const float* o, const float* u, int N, int K, float forget_bias);
template bool LSTM4_Gates_Backward<ehalf,ehalf,ehalf4,ehalf4>(CUstream stream, ehalf* dc, ehalf* di, ehalf* df, ehalf* doo, ehalf* du, const ehalf* ec, const ehalf* eh, const ehalf* c, const ehalf* i, const ehalf* f, const ehalf* o, const ehalf* u, int N, int K, float forget_bias);
//template bool LSTM4_Gates_Backward<float,ehalf,float4,ehalf4>(CUstream stream, float* dc, float* di, float* df, float* doo, float* du, const float* ec, const float* eh, const ehalf* c, const ehalf* i, const ehalf* f, const ehalf* o, const ehalf* u, int N, int K, float forget_bias);
template bool LSTM4_Gates_Backward<bhalf,bhalf,bhalf4,bhalf4>(CUstream stream, bhalf* dc, bhalf* di, bhalf* df, bhalf* doo, bhalf* du, const bhalf* ec, const bhalf* eh, const bhalf* c, const bhalf* i, const bhalf* f, const bhalf* o, const bhalf* u, int N, int K, float forget_bias);
//template bool LSTM4_Gates_Backward<float,bhalf,float4,bhalf4>(CUstream stream, float* dc, float* di, float* df, float* doo, float* du, const float* ec, const float* eh, const bhalf* c, const bhalf* i, const bhalf* f, const bhalf* o, const bhalf* u, int N, int K, float forget_bias);


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


// mean   = mean(x, axis=1)
// std    = std(x, axis=1)
// cutoff = mean + alpha*std
// y      = fmaxf(x, cutoff) - cutoff;
template <typename T, typename V, int THREADS>
__global__ void __launch_bounds__(THREADS) sparse_relu_forward(
              T*              Y,
    const     T* __restrict__ X,
    float alpha, uint K, float rcpK)
{
    int tid = threadIdx.x;
    int n   = blockIdx.x;

    int offset = n*K + tid;

    // Mean
    const T* X1 = X + offset;
    V v_mean1, v_mean2;
    ew_zero(v_mean1);
    ew_zero(v_mean2);
    #pragma unroll 4
    for (int k = tid; k < K; k += THREADS)
    {
        V x = load(X1);
        v_mean1 = ew_add(v_mean1, x);
        v_mean2 = ew_add(v_mean2, ew_sqr(x));
        X1 += THREADS;
    }
    float2 mean;
    mean.x = ew_sum(v_mean1) * rcpK;
    mean.y = ew_sum(v_mean2) * rcpK;

    // reduce within warp
    for (int i = 16; i > 0; i >>= 1)
    {
        mean.x += shfl_xor(mean.x, i);
        mean.y += shfl_xor(mean.y, i);
    }
    // if using more than 1 warp, further reduced with shared memory
    if (THREADS > 32)
    {
        __shared__ float2 Share[32];

        // first thread of each warp store to shared
        if ((tid & 31) == 0)
            Share[tid/32] = mean;

        __syncthreads();

        if (tid < 32)
        {
            // first warp loads all prior reductions
            mean = Share[tid];

            // reduce within this first warp
            for (int i = THREADS/64; i > 0; i >>= 1)
            {
                mean.x += shfl_xor(mean.x, i);
                mean.y += shfl_xor(mean.y, i);
            }
            // outputs final reduction to shared
            Share[tid] = mean;
        }
        __syncthreads();

        // broadcast result to all threads
        mean = Share[0];
    }
    // var = avg(x**2) - avg(x)**2
    // std = sqrt(var)
    float std = sqrtf(precise_sub(mean.y, mean.x*mean.x));

    // Norm/Gain/Bias
    X += offset;
    Y += offset;
    for (int k = tid; k < K; k += THREADS)
    {
        float cutoff = mean.x + alpha*std;
        V x = load(X);
        V y = ew_sub(ew_maximum(x, cutoff), cutoff);
        store(Y, ew_relu(y), 0, true);
        X += THREADS;
        Y += THREADS;
    }
}

template <typename T, typename V>
bool SparseReluForward(CUstream stream, T* y, const T* x, float alpha, uint K, uint N)
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
template bool SparseReluForward<float,float4>(CUstream stream, float* y, const float* x,float alpha, uint K, uint N);
template bool SparseReluForward<ehalf,ehalf4>(CUstream stream, ehalf* y, const ehalf* x,float alpha, uint K, uint N);
template bool SparseReluForward<bhalf,bhalf4>(CUstream stream, bhalf* y, const bhalf* x,float alpha, uint K, uint N);


#endif
