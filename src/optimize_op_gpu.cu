
#if GOOGLE_CUDA

#include "ew_op_gpu.h"
#include <stdio.h>
#include <type_traits>

// # kernel 1
// new_vr = decay * vr + (1 - decay) * np.mean(grad**2 + eps1, axis=1, keepdims=True)
// tf.assign(vr, new_vr)
// ltm = np.mean(new_vr, keepdims=True)

template <typename T, typename V>
__global__ void adafactor_row_variance(
    float* RV, float* RV_MEAN, const T* __restrict__ Grad, float grad_scale, float decay, float epsilon, uint K, float rcpC, float rcpK, uint sat_infs, uint zero_nans)
{
    uint tid = threadIdx.x;
    uint c   = blockIdx.x;

    V var_sum;
    ew_zero(var_sum);

    #pragma unroll 1
    for (uint k = tid, offset = c*K + tid; k < K; k += blockDim.x, offset += blockDim.x)
    {
        V grad  = load(Grad + offset);

        grad = ew_mul(grad, grad_scale);

        // Nans => zero
        if (zero_nans)
            grad = ew_zero_nan(grad);

        // Saturate fp16 infinity values
        if (std::is_same<T, ehalf4>::value || std::is_same<T, ehalf>::value || sat_infs)
            grad = ew_maximum(ew_minimum(grad, 65504.0f), -65504.0f);

        var_sum = ew_add(var_sum, ew_add(ew_sqr(grad), epsilon));
    }
    float row_var = ew_sum(var_sum);

    // reduce within warp
    for (int i = 16; i > 0; i >>= 1)
        row_var += shfl_xor(row_var, i);

    // if using more than 1 warp, further reduced with shared memory
    if (blockDim.x > 32)
    {
        __shared__ float Share[32];

        // first thread of each warp store to shared
        if ((tid & 31) == 0)
            Share[tid/32] = row_var;

        __syncthreads();

        if (tid < blockDim.x/32)
        {
            // first warp loads all prior reductions
            row_var = Share[tid];

            // reduce within this first warp
            #pragma unroll 1
            for (int i = blockDim.x/64; i > 0; i >>= 1)
                row_var += shfl_xor(row_var, i);
        }
    }
    if (tid == 0)
    {
        row_var *= rcpK;

        RV += c;
        float old_rv = __ldg((const float*)RV);

        row_var = decay * old_rv + (1.0f - decay) * row_var;
        __stg(RV, row_var);
        atomicRed(RV_MEAN, row_var * rcpC);
    }
}

// # kernel 2
// new_vc = decay * vc + (1 - decay) * np.mean(grad**2 + eps1, axis=0, keepdims=True)
// tf.assign(vc, new_vc)

template <typename T, typename V, uint THREADS>
__global__ void __launch_bounds__(THREADS) adafactor_col_variance(
    V* CV, const T* __restrict__ Grad, float grad_scale, float decay, float epsilon, uint C, uint K, float rcpC, uint sat_infs, uint zero_nans)
{
    uint tid = threadIdx.x;
    uint k   = blockIdx.x*32 + (tid & 31);
    uint c   = tid / 32;

    uint ck = c*K + k;

    V var_sum;
    ew_zero(var_sum);
    if (k < K)
    {
        #pragma unroll 1
        while (c < C)
        {
            V grad = load(Grad + ck);

            grad = ew_mul(grad, grad_scale);

            // Nans => zero
            if (zero_nans)
                grad = ew_zero_nan(grad);

            // Saturate fp16 infinity values
            if (std::is_same<T, ehalf>::value || sat_infs)
                grad = ew_maximum(ew_minimum(grad, 65504.0f), -65504.0f);

            var_sum = ew_add(var_sum, ew_add(ew_sqr(grad), epsilon));

            ck += K*THREADS/32;
            c  +=   THREADS/32;
        }
    }
    if (THREADS > 32)
    {
        __shared__ V Share[THREADS];
        if (tid >= 64)
            Share[tid] = var_sum;

        __syncthreads();

        if (tid < 64)
        {
            for (uint i = 1; i < THREADS/64; i++)
                var_sum = ew_add(var_sum, Share[tid + i*64]);

            Share[tid] = var_sum;
        }
        __syncthreads();

        if (tid < 32)
            var_sum = ew_add(var_sum, Share[tid + 32]);
    }
    if (tid < 32 && k < K)
    {
        CV += k;
        V col_var = ew_mul(var_sum, rcpC);
        V old_cv  = __ldg((const V*)CV);

        col_var = ew_add(ew_mul(old_cv, decay), ew_mul(col_var, 1.0f - decay));

        __stg(CV, col_var);
    }
}

// # kernel 3
// x = grad * np.rsqrt(new_vr / ltm) * np.rsqrt(new_vc)
// rms_x = np.mean(x**2, keepdims=True)

template <typename T, typename V>
__global__ void adafactor_normalize_2d(
    V* X, float* RMS_X, const T* __restrict__ Grad, const float* __restrict__ RV, const V* __restrict__ CV, const float* __restrict__ RV_MEAN, float grad_scale, uint K, float rcpCK, uint sat_infs, uint zero_nans)
{
    uint tid = threadIdx.x;
    uint c   = blockIdx.x;

    float rv = rsqrtf(RV[c] / *RV_MEAN);

    V rms_sum;
    ew_zero(rms_sum);

    #pragma unroll 1
    for (uint k = tid, offset = c*K + tid; k < K; k += blockDim.x, offset += blockDim.x)
    {
        V grad = load(Grad + offset);
        V cv   = ew_rsqrt(CV[k]);

        grad = ew_mul(grad, grad_scale);

        // Nans => zero
        if (zero_nans)
            grad = ew_zero_nan(grad);

        // Saturate fp16 infinity values
        if (std::is_same<T, ehalf4>::value || std::is_same<T, ehalf>::value || sat_infs)
            grad = ew_maximum(ew_minimum(grad, 65504.0f), -65504.0f);

        V x = ew_mul(grad, ew_mul(cv, rv));

        rms_sum = ew_add(rms_sum, ew_sqr(x));

        store(X + offset, x);
    }
    float rms_x = ew_sum(rms_sum);

    // reduce within warp
    for (int i = 16; i > 0; i >>= 1)
        rms_x += shfl_xor(rms_x, i);

    // if using more than 1 warp, further reduced with shared memory
    if (blockDim.x > 32)
    {
        __shared__ float Share[32];

        // first thread of each warp store to shared
        if ((tid & 31) == 0)
            Share[tid/32] = rms_x;

        __syncthreads();

        if (tid < blockDim.x/32)
        {
            // first warp loads all prior reductions
            rms_x = Share[tid];

            // reduce within this first warp
            #pragma unroll 1
            for (int i = blockDim.x/64; i > 0; i >>= 1)
                rms_x += shfl_xor(rms_x, i);
        }
    }
    if (tid == 0)
        atomicRed(RMS_X, rms_x * rcpCK);
}

// new_v = decay * v + (1 - decay) * (grad**2 + eps1)
// tf.assign(v, new_v)
// x = grad * tf.rsqrt(new_v)
// rms_x = np.mean(x**2, keepdims=True)
template <typename T>
__global__ void __launch_bounds__(32) adafactor_normalize_1d(
    float* CV, float* X, float* RMS_X, const T* __restrict__ Grad, float grad_scale, float decay, float epsilon, uint K, float rcpK, uint sat_infs, uint zero_nans)
{
    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    float rms_x = 0.0f;

    #pragma unroll 1
    for (uint k = bid*32 + tid; k < K; k += gridDim.x*32)
    {
        float grad = load(Grad + k);
        float cv   = CV[k];

        grad = ew_mul(grad, grad_scale);

        // Nans => zero
        if (zero_nans)
            grad = ew_zero_nan(grad);

        // Saturate fp16 infinity values
        if (std::is_same<T, ehalf>::value || sat_infs)
            grad = ew_maximum(ew_minimum(grad, 65504.0f), -65504.0f);

        float new_cv = decay * cv + (1.0f - decay) * (grad*grad + epsilon);
        float x      = grad * rsqrtf(new_cv);

        CV[k] = new_cv;
        X[k]  = x;

        rms_x += x*x;
    }
    // reduce within warp
    for (int i = 16; i > 0; i >>= 1)
        rms_x += shfl_xor(rms_x, i);

    if (tid == 0)
        atomicRed(RMS_X, rms_x * rcpK);
}

// # kernel 4
// tf.assign_sub(param, learning_rate * x / np.maximum(1.0, np.sqrt(rms_x) / clipping_threshold) )
template <typename V>
__global__ void adafactor_apply(
    V* P, const V* __restrict__ X, const float* __restrict__ RMS_X, float learning_rate, float rcp_clip, uint size)
{
    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    float update_rate = learning_rate / fmaxf(sqrtf(*RMS_X) * rcp_clip, 1.0f);

    #pragma unroll 1
    for (uint i = bid*blockDim.x + tid; i < size; i += gridDim.x*blockDim.x)
        P[i] = ew_sub(P[i], ew_mul(X[i], update_rate));
}

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

template <typename T, typename V>
bool Adafactor(CUstream stream, uint SMs, float* cv, float* rv, float* x, float* means, float* param, const T* grad, float scale, float learning_rate, float decay, float epsilon, float clip, uint C, uint K, bool sat_infs, bool zero_nans)
{
    cuMemsetD32Async((CUdeviceptr)means, 0, 2, stream); // used for row variance mean and RMS_X
    float rcpK    = 1.0f / (float)K;
    float rcpClip = 1.0f / clip;

    float* rv_mean = means;
    float* rms_x   = means + 1;

    // 1D case
    if (C == 1)
    {
        uint gridK = MIN(MAX(SMs*2, CEIL_DIV(K, 32*4)), SMs*32*2);

        adafactor_normalize_1d<T><<<gridK,32,0,stream>>>(cv, x, rms_x, grad, scale, decay, epsilon, K, rcpK, sat_infs, zero_nans);
        adafactor_apply<float><<<gridK,32,0,stream>>>(param, x, (const float*)rms_x, learning_rate, rcpClip, K);
    }
    else
    {
        float rcpC = 1.0f / (float)C;
        uint gridK = CEIL_DIV(K, 32);

        adafactor_col_variance<T,float,1024><<<gridK,1024,0,stream>>>(cv, grad, scale, decay, epsilon, C, K, rcpC, sat_infs, zero_nans);

        if (K & 3)
        {
            uint CK = C*K;
            uint gridCK = CK > SMs*1024 ? SMs*2 : SMs;

            adafactor_row_variance<T,float><<<C,1024,0,stream>>>(rv, rv_mean, grad, scale, decay, epsilon, K, rcpC, rcpK, sat_infs, zero_nans);
            adafactor_normalize_2d<T,float><<<C,1024,0,stream>>>(x, rms_x, grad, (const float*)rv, (const float*)cv, (const float*)rv_mean, scale, K, rcpC*rcpK, sat_infs, zero_nans);
            adafactor_apply<float><<<gridCK,1024,0,stream>>>(param, (const float*)x, (const float*)rms_x, learning_rate, rcpClip, CK);
        }
        else
        {
            K >>= 2;
            uint CK = C*K;
            uint gridCK =
                CK <= SMs*256*1 ? SMs*1 :
                CK <= SMs*256*2 ? SMs*2 :
                CK <= SMs*256*4 ? SMs*4 :
                                  SMs*8 ;

            adafactor_row_variance<V,float4><<<C,256,0,stream>>>(rv, rv_mean, (const V*)grad, scale, decay, epsilon, K, rcpC, rcpK, sat_infs, zero_nans);
            adafactor_normalize_2d<V,float4><<<C,256,0,stream>>>((float4*)x, rms_x, (const V*)grad, (const float*)rv, (const float4*)cv, (const float*)rv_mean, scale, K, rcpC*rcpK, sat_infs, zero_nans);
            adafactor_apply<float4><<<gridCK,256,0,stream>>>((float4*)param, (const float4*)x, (const float*)rms_x, learning_rate, rcpClip, CK);
        }
    }
    return true;
}
template bool Adafactor<float,float4>(CUstream stream, uint SMs, float* cv, float* rv, float* x, float* means, float* param, const float* grad, float scale, float learning_rate, float decay, float epsilon, float clip, uint C, uint K, bool sat_infs, bool zero_nans);
template bool Adafactor<ehalf,ehalf4>(CUstream stream, uint SMs, float* cv, float* rv, float* x, float* means, float* param, const ehalf* grad, float scale, float learning_rate, float decay, float epsilon, float clip, uint C, uint K, bool sat_infs, bool zero_nans);
template bool Adafactor<bhalf,bhalf4>(CUstream stream, uint SMs, float* cv, float* rv, float* x, float* means, float* param, const bhalf* grad, float scale, float learning_rate, float decay, float epsilon, float clip, uint C, uint K, bool sat_infs, bool zero_nans);


template <typename TG, typename TR>
__global__ void apply_lazy_adam(
          float*              Param,
             TR*              Mean,
             TR*              Var,
    const    TG* __restrict__ Grad,
    float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint K, uint sat_infs, uint zero_nans)
{
    uint tid = threadIdx.x;
    uint c = blockIdx.x;
    uint k = blockIdx.y*blockDim.x + tid;
    uint offset = c*K + k;

    float g = load(add_ptr_u(Grad, offset), 0, k < K);

    // Nans => zero
    if (zero_nans)
        asm("{                               \n\t"
            ".reg .pred is_number;           \n\t"
            "testp.number.f32 is_number, %0; \n\t"
            "selp.f32 %0, %0, 0.0, is_number;\n\t"
            "}" : "+f"(g) :);

    // Saturate fp16 infinity values
    if (std::is_same<TG, ehalf>::value || sat_infs)
        g = fmaxf(fminf(g, 65504.0f), -65504.0f);

    // max reduce gradient within this block.
    // If the whole block is zero that means that this embedding vector was not selected.
    // If the emb vector is bigger than the block then at least the probability is high of non-selection.
    // Make Adam a no-op in this case.
    float gmax = fabsf(g);
    for (int i = 16; i > 0; i >>= 1)
        gmax = fmaxf(gmax, shfl_xor(gmax, i));

    if (blockDim.x > 32)
    {
        __shared__ float Share[32];

        // first thread of each warp store to shared
        if ((tid & 31) == 0)
            Share[tid/32] = gmax;
        __syncthreads();

        if (tid < 32)
        {
            // first warp loads all prior reductions
            gmax = Share[tid];
            // reduce within this last warp
            #pragma unroll 1
            for (int i = blockDim.x/64; i > 0; i >>= 1)
                gmax = fmaxf(gmax, shfl_xor(gmax, i));
            // final reduction to shared
            Share[tid] = gmax;
        }
        __syncthreads();
        gmax = Share[0];
    }

    if (k < K && gmax > 0.0f)
    {
        float v = load(add_ptr_u((const    TR*)Var,   offset));
        float m = load(add_ptr_u((const    TR*)Mean,  offset));
        float p = load(add_ptr_u((const float*)Param, offset));

        g *= grad_scale;
        v  = decay_var * v + (1.0f - decay_var) * g*g;

        float sigma = sqrtf(v);
        if (clip_sigma != 0.0f)
        {
            float clip = clip_sigma * sigma;
            g = fmaxf(g, -clip);
            g = fminf(g,  clip);
        }
        m  = decay_mean * m + (1.0f - decay_mean) * g;
        p -= lr * m / (sigma + epsilon);

        store(add_ptr_u(Mean,  offset), m);
        store(add_ptr_u(Var,   offset), v);
        store(add_ptr_u(Param, offset), p);
    }
}

template <typename TG, typename TR>
__global__ void apply_adam(
          float*              Param,
             TR*              Mean,
             TR*              Var,
    const    TG* __restrict__ Grad,
    float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint size, uint sat_infs, uint zero_nans)
{
    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    for (uint offset = bid*blockDim.x + tid; offset < size; offset += gridDim.x*blockDim.x)
    {
        float g = load(add_ptr_u(              Grad,  offset));
        float v = load(add_ptr_u((const    TR*)Var,   offset));
        float m = load(add_ptr_u((const    TR*)Mean,  offset));
        float p = load(add_ptr_u((const float*)Param, offset));

        // Nans => zero
        if (zero_nans)
            asm("{                               \n\t"
                ".reg .pred is_number;           \n\t"
                "testp.number.f32 is_number, %0; \n\t"
                "selp.f32 %0, %0, 0.0, is_number;\n\t"
                "}" : "+f"(g) :);

        // Saturate fp16 infinity values
        if (std::is_same<TG, ehalf>::value || sat_infs)
            g = fmaxf(fminf(g, 65504.0f), -65504.0f);

        g *= grad_scale;
        v  = decay_var * v + (1.0f - decay_var) * g*g;

        float sigma = sqrtf(v);
        if (clip_sigma != 0.0f)
        {
            float clip = clip_sigma * sigma;
            g = fmaxf(g, -clip);
            g = fminf(g,  clip);
        }
        m  = decay_mean * m + (1.0f - decay_mean) * g;
        p -= lr * m / (sigma + epsilon);

        store(add_ptr_u(Mean,  offset), m);
        store(add_ptr_u(Var,   offset), v);
        store(add_ptr_u(Param, offset), p);
    }
}
template <typename TG, typename TR>
bool ApplyAdam(
  CUstream stream, uint SMs,
  const TG* grad,
  float* param,
  TR* mean,
  TR* var,
  float lr,
  float decay_mean,
  float decay_var,
  float epsilon,
  float grad_scale,
  float clip_sigma,
  uint  size,
  uint  lazy_update,
  bool  sat_infs,
  bool  zero_nans)
{
    if (lazy_update)
    {
        uint K = lazy_update;
        uint C = size;
        uint threads, gridK;
        if (K <= 1024) {
            threads = THREAD_POW2(K);
            gridK   = 1;
        }
        else {
            threads = 256;
            gridK   = CEIL_DIV(K, 256);
        }
        apply_lazy_adam<TG,TR><<<dim3(C,gridK,1),threads,0,stream>>>(param, mean, var, grad, lr, decay_mean, decay_var, epsilon, grad_scale, clip_sigma, K, sat_infs, zero_nans);
    }
    else
    {
        uint grid = SMs, threads = 64;
             if (size > SMs*1024) { threads = 1024; grid *= 2; }
        else if (size > SMs* 512) { threads = 1024; }
        else if (size > SMs* 256) { threads =  512; }
        else if (size > SMs* 128) { threads =  256; }
        else if (size > SMs*  64) { threads =  128; }

        apply_adam<TG,TR><<<grid,threads,0,stream>>>(param, mean, var, grad, lr, decay_mean, decay_var, epsilon, grad_scale, clip_sigma, size, sat_infs, zero_nans);
    }
    return true;
}
template bool ApplyAdam<float,float>(CUstream stream, uint SMs, const float* grad, float* param, float* mean, float* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint size, uint lazy_update, bool sat_infs, bool zero_nans);
template bool ApplyAdam<ehalf,float>(CUstream stream, uint SMs, const ehalf* grad, float* param, float* mean, float* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint size, uint lazy_update, bool sat_infs, bool zero_nans);
template bool ApplyAdam<bhalf,float>(CUstream stream, uint SMs, const bhalf* grad, float* param, float* mean, float* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint size, uint lazy_update, bool sat_infs, bool zero_nans);


template <typename TG, typename TR, uint BSIZE, uint THREADS>
__global__ void __launch_bounds__(THREADS) apply_adam_gated(
          float*              Param,
             TR*              Mean,
             TR*              Var,
    const    TG* __restrict__ Grad,
    const float* __restrict__ Gate,
    float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma)
{
    const uint U = BSIZE*BSIZE/THREADS;

    uint bid = blockIdx.x;
    uint tid = threadIdx.x;

    if (Gate[bid] != 0.0f)
    {
        uint offset = bid*BSIZE*BSIZE + tid;

        Grad  += offset;
        Mean  += offset;
        Var   += offset;
        Param += offset;

        float g[U], m[U], v[U], p[U];
        for (uint j = 0; j < U; j++) g[j] = load((const    TG*)Grad,  j*THREADS);
        for (uint j = 0; j < U; j++) m[j] = load((const    TR*)Mean,  j*THREADS);
        for (uint j = 0; j < U; j++) v[j] = load((const    TR*)Var,   j*THREADS);
        for (uint j = 0; j < U; j++) p[j] = load((const float*)Param, j*THREADS);

        for (uint j = 0; j < U; j++)
        {
            g[j] *= grad_scale;
            v[j]  = decay_var  * v[j] + (1.0f - decay_var ) * g[j] * g[j];

            float sig = sqrtf(v[j]);
            if (clip_sigma != 0.0f)
            {
                float clip = clip_sigma * sig;
                g[j] = fmaxf(g[j], -clip);
                g[j] = fminf(g[j],  clip);
            }
            m[j]  = decay_mean * m[j] + (1.0f - decay_mean) * g[j];
            p[j] -= lr * m[j] / (sqrtf(v[j]) + epsilon);
        }

        for (uint j = 0; j < U; j++) store(Mean,  m[j], j*THREADS);
        for (uint j = 0; j < U; j++) store(Var,   v[j], j*THREADS);
        for (uint j = 0; j < U; j++) store(Param, p[j], j*THREADS);
    }
}

template <typename TG, typename TR>
bool ApplyAdamGated(
  CUstream stream,
  const float* gate,
  const TG* grad,
  float* param,
  TR* mean,
  TR* var,
  float lr,
  float decay_mean,
  float decay_var,
  float epsilon,
  float grad_scale,
  float clip_sigma,
  uint  blocks,
  uint  bsize)
{
    if (bsize == 8)
        apply_adam_gated<TG,TR, 8, 32><<<blocks, 32,0,stream>>>(param, mean, var, grad, gate, lr, decay_mean, decay_var, epsilon, grad_scale, clip_sigma);
    else if (bsize == 16)
        apply_adam_gated<TG,TR,16, 64><<<blocks, 64,0,stream>>>(param, mean, var, grad, gate, lr, decay_mean, decay_var, epsilon, grad_scale, clip_sigma);
    else
        apply_adam_gated<TG,TR,32,256><<<blocks,256,0,stream>>>(param, mean, var, grad, gate, lr, decay_mean, decay_var, epsilon, grad_scale, clip_sigma);
    return true;
}

template bool ApplyAdamGated<float,float>(CUstream stream, const float* gate, const float* grad, float* param, float* mean, float* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint blocks, uint bsize);
template bool ApplyAdamGated<ehalf,float>(CUstream stream, const float* gate, const ehalf* grad, float* param, float* mean, float* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint blocks, uint bsize);
template bool ApplyAdamGated<bhalf,float>(CUstream stream, const float* gate, const bhalf* grad, float* param, float* mean, float* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint blocks, uint bsize);



template <typename T, uint U>
__global__ void __launch_bounds__(32) apply_ema(
          T*              Ema,
    const T* __restrict__ Param,
    float decay, uint size)
{
    uint tid = threadIdx.x;
    uint bid = blockIdx.x;
    uint offset = bid * U*32 + tid;

    bool pred[U];
    for (uint j = 0; j < U; j++)
        pred[j] = offset + j*32 < size;

    Ema   += offset;
    Param += offset;

    float e[U], p[U];
    for (uint j = 0; j < U; j++) e[j]  = load((const T*)Ema, j*32, pred[j]);
    for (uint j = 0; j < U; j++) p[j]  = load(        Param, j*32, pred[j]);
    for (uint j = 0; j < U; j++) e[j] -= (1.0f - decay) * (e[j] - p[j]);
    for (uint j = 0; j < U; j++) store(Ema, e[j], j*32, pred[j]);
}

template <typename T>
bool ApplyEma(CUstream stream, T* ema, const T* param, float decay, uint size)
{
    uint grid = (size >> 7) + ((size & 127) != 0); // 1 warp with 4 unrolls
    if (grid > 200)
    {
        apply_ema<T,4><<<grid,32,0,stream>>>(ema, param, decay, size);
    }
    else
    {
        grid  = (size >> 5) + ((size &  31) != 0); // 1 warp with 1 unroll
        apply_ema<T,1><<<grid,32,0,stream>>>(ema, param, decay, size);
    }
    return true;
}

template bool ApplyEma<float>(CUstream stream, float* ema, const float* param, float decay, uint size);



template <typename T, uint BSIZE, uint THREADS>
__global__ void __launch_bounds__(THREADS) apply_ema_gated(
              T*              Ema,
    const     T* __restrict__ Param,
    const float* __restrict__ Gate,
    float decay)
{
    const uint U = BSIZE*BSIZE/THREADS;

    uint bid = blockIdx.x;
    uint tid = threadIdx.x;

    if (Gate[bid] != 0.0f)
    {
        uint offset = bid*BSIZE*BSIZE + tid;

        Ema   += offset;
        Param += offset;

        float e[U], p[U];
        for (uint j = 0; j < U; j++) e[j]  = load((const T*)Ema, j*THREADS);
        for (uint j = 0; j < U; j++) p[j]  = load(        Param, j*THREADS);
        for (uint j = 0; j < U; j++) e[j] -= (1.0f - decay) * (e[j] - p[j]);
        for (uint j = 0; j < U; j++) store(Ema, e[j],  j*THREADS);
    }
}

template <typename T>
bool ApplyEmaGated(CUstream stream, T* ema, const T* param, const float* gate, float decay, uint blocks, uint bsize)
{
    if (bsize == 8)
        apply_ema_gated<T, 8, 32><<<blocks, 32,0,stream>>>(ema, param, gate, decay);
    else if (bsize == 16)
        apply_ema_gated<T,16, 64><<<blocks, 64,0,stream>>>(ema, param, gate, decay);
    else
        apply_ema_gated<T,32,256><<<blocks,256,0,stream>>>(ema, param, gate, decay);
    return true;
}

template bool ApplyEmaGated<float>(CUstream stream, float* ema, const float* param, const float* gate, float decay, uint blocks, uint bsize);

template <typename T, uint BSIZE, uint THREADS, uint GATED>
__global__ void __launch_bounds__(THREADS) blocksparse_l2_decay(T* Param, const float* __restrict__ Gate, float rate, float epsilon)
{
    const uint U = BSIZE*BSIZE/THREADS;

    uint bid = blockIdx.x;
    uint tid = threadIdx.x;

    if (GATED == 0 || Gate[bid] != 0.0f)
    {
        uint offset = bid*BSIZE*BSIZE + tid;

        Param += offset;

        float p[U];
        for (uint j = 0; j < U; j++)
            p[j] = load((const T*)Param, j*THREADS);

        // Reduce sum squared within this thread
        float sum_sqared = 0.0f;
        for (uint j = 0; j < U; j++)
            sum_sqared += p[j] * p[j];

        // reduce within warp
        for (int i = 16; i > 0; i >>= 1)
            sum_sqared += shfl_xor(sum_sqared, i);

        // if using more than 1 warp, further reduced with shared memory
        if (THREADS > 32)
        {
            __shared__ float Share[THREADS/32];

            // first thread of each warp store to shared
            if ((tid & 31) == 0)
                Share[tid >> 5] = sum_sqared;

            __syncthreads();

            if (tid < THREADS/32)
            {
                // first warp loads all prior reductions
                sum_sqared = Share[tid];

                // reduce within this first warp
                for (int i = THREADS/64; i > 0; i >>= 1)
                    sum_sqared += shfl_xor(sum_sqared, i);

                // outputs final reduction to shared
                Share[tid] = sum_sqared;
            }
            __syncthreads();

            // broadcast result to all threads
            sum_sqared = Share[0];
        }

        // apply weight decay and store updated paramm
        float decay = fminf(rsqrtf(sum_sqared + epsilon) * rate, 1.0f);
        for (uint j = 0; j < U; j++)
            store(Param, p[j] - p[j] * decay, j*THREADS);
    }
}

template <typename T>
bool BlocksparseL2Decay(CUstream stream, T* param, const float* gate, float rate, float epsilon, uint blocks, uint bsize)
{

    if (gate != NULL)
    {
        if (bsize == 8)
            blocksparse_l2_decay<T, 8, 32,1><<<blocks, 32,0,stream>>>(param, gate, rate, epsilon);
        else if (bsize == 16)
            blocksparse_l2_decay<T,16, 64,1><<<blocks, 64,0,stream>>>(param, gate, rate, epsilon);
        else
            blocksparse_l2_decay<T,32,256,1><<<blocks,256,0,stream>>>(param, gate, rate, epsilon);
    }
    else
    {
        if (bsize == 8)
            blocksparse_l2_decay<T, 8, 32,0><<<blocks, 32,0,stream>>>(param, gate, rate, epsilon);
        else if (bsize == 16)
            blocksparse_l2_decay<T,16, 64,0><<<blocks, 64,0,stream>>>(param, gate, rate, epsilon);
        else
            blocksparse_l2_decay<T,32,256,0><<<blocks,256,0,stream>>>(param, gate, rate, epsilon);
    }

    return true;
}
template bool BlocksparseL2Decay<float>(CUstream stream, float* param, const float* gate, float rate, float epsilon, uint blocks, uint bsize);


template <typename T, uint BSIZE, uint THREADS>
__global__ void __launch_bounds__(THREADS) blocksparse_maxnorm_prune(const T* __restrict__ Param, float* Gate, float threshold)
{
    const uint U = BSIZE*BSIZE/THREADS;

    uint bid = blockIdx.x;
    uint tid = threadIdx.x;
    uint offset = bid*BSIZE*BSIZE + tid;

    Param += offset;

    float p[U];
    for (uint j = 0; j < U; j++)
        p[j] = load(Param, j*THREADS);

    // Reduce max within this thread
    float max_abs = 0.0f;
    for (uint j = 0; j < U; j++)
        max_abs = fmaxf(fabsf(p[j]), max_abs);

    // reduce within warp
    for (int i = 16; i > 0; i >>= 1)
        max_abs = fmaxf(max_abs, shfl_xor(max_abs, i));

    // if using more than 1 warp, further reduced with shared memory
    if (THREADS > 32)
    {
        __shared__ float Share[THREADS/32];

        // first thread of each warp store to shared
        if ((tid & 31) == 0)
            Share[tid >> 5] = max_abs;

        __syncthreads();

        if (tid < THREADS/32)
        {
            // first warp loads all prior reductions
            max_abs = Share[tid];

            // reduce within this first warp
            for (int i = THREADS/64; i > 0; i >>= 1)
                max_abs = fmaxf(max_abs, shfl_xor(max_abs, i));
        }
    }
    // first thread has the final reduced max_abs
    // compare against threshhold and update gate if needed.
    // if (bid < 2 && tid == 0)
    //     printf("%d %d %.5f %.5f\n", bid, gridDim.x, max_abs, threshold);
    if (tid == 0)
        Gate[bid] = max_abs < threshold ? 0.0f : 1.0f;
}

template <typename T>
bool BlocksparseMaxnormPrune(CUstream stream, const T* param, float* gate, float threshold, uint blocks, uint bsize)
{
    if (bsize == 8)
        blocksparse_maxnorm_prune<T, 8, 32><<<blocks, 32,0,stream>>>(param, gate, threshold);
    else if (bsize == 16)
        blocksparse_maxnorm_prune<T,16, 64><<<blocks, 64,0,stream>>>(param, gate, threshold);
    else
        blocksparse_maxnorm_prune<T,32,256><<<blocks,256,0,stream>>>(param, gate, threshold);
    return true;
}
template bool BlocksparseMaxnormPrune<float>(CUstream stream, const float* param, float* gate, float threshold, uint blocks, uint bsize);

#endif
