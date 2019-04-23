
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
    float* RV, float* RV_MEAN, const T* __restrict__ Grad, const float* __restrict__ Norm, float grad_scale, float decay, float epsilon, uint K, float rcpC, float rcpK, float saturate, uint zero_infs, uint zero_nans, uint use_norm)
{
    float norm_scale = use_norm ? __ldg(Norm) : 1.0f;

    // skip all optimization if the global norm is bad.
    if (norm_scale != 0.0f)
    {
        uint tid = threadIdx.x;
        uint c   = blockIdx.x;

        V var_sum;
        ew_zero(var_sum);

        #pragma unroll 1
        for (uint k = tid, offset = c*K + tid; k < K; k += blockDim.x, offset += blockDim.x)
        {
            V grad  = load(Grad + offset);

            if (zero_infs)
                grad = ew_zero_inf(grad);
            if (zero_nans)
                grad = ew_zero_nan(grad);
            if (saturate != 0.0f)
                grad = ew_maximum(ew_minimum(grad, saturate), -saturate);

            grad = ew_mul(grad, grad_scale * norm_scale);

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

            if (tid < 32)
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
}

// # kernel 2
// new_vc = decay * vc + (1 - decay) * np.mean(grad**2 + eps1, axis=0, keepdims=True)
// tf.assign(vc, new_vc)

template <typename T, typename V, uint THREADS>
__global__ void __launch_bounds__(THREADS) adafactor_col_variance(
    V* CV, const T* __restrict__ Grad, const float* __restrict__ Norm, float grad_scale, float decay, float epsilon, uint C, uint K, float rcpC, float saturate, uint zero_infs, uint zero_nans, uint use_norm)
{
    float norm_scale = use_norm ? __ldg(Norm) : 1.0f;

    // skip all optimization if the global norm is bad.
    if (norm_scale != 0.0f)
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

                if (zero_infs)
                    grad = ew_zero_inf(grad);
                if (zero_nans)
                    grad = ew_zero_nan(grad);
                if (saturate != 0.0f)
                    grad = ew_maximum(ew_minimum(grad, saturate), -saturate);

                grad = ew_mul(grad, grad_scale * norm_scale);

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
}

// # kernel 3
// x = grad * np.rsqrt(new_vr / ltm) * np.rsqrt(new_vc)
// rms_x = np.mean(x**2, keepdims=True)

template <typename T, typename V>
__global__ void adafactor_normalize_2d(
    V* X, float* RMS_X, const T* __restrict__ Grad, const float* __restrict__ Norm, const float* __restrict__ RV, const V* __restrict__ CV, const float* __restrict__ RV_MEAN, float grad_scale, uint K, float rcpCK, float saturate, uint zero_infs, uint zero_nans, uint use_norm)
{
    float norm_scale = use_norm ? __ldg(Norm) : 1.0f;

    // skip all optimization if the global norm is bad.
    if (norm_scale != 0.0f)
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

            if (zero_infs)
                grad = ew_zero_inf(grad);
            if (zero_nans)
                grad = ew_zero_nan(grad);
            if (saturate != 0.0f)
                grad = ew_maximum(ew_minimum(grad, saturate), -saturate);

            grad = ew_mul(grad, grad_scale * norm_scale);

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

            if (tid < 32)
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
}

// new_v = decay * v + (1 - decay) * (grad**2 + eps1)
// tf.assign(v, new_v)
// x = grad * tf.rsqrt(new_v)
// rms_x = np.mean(x**2, keepdims=True)
template <typename T>
__global__ void __launch_bounds__(32) adafactor_normalize_1d(
    float* CV, float* X, float* RMS_X, const T* __restrict__ Grad, const float* __restrict__ Norm, float grad_scale, float decay, float epsilon, uint K, float rcpK, float saturate, uint zero_infs, uint zero_nans, uint use_norm)
{
    float norm_scale = use_norm ? __ldg(Norm) : 1.0f;

    // skip all optimization if the global norm is bad.
    if (norm_scale != 0.0f)
    {
        uint tid = threadIdx.x;
        uint bid = blockIdx.x;

        float rms_x = 0.0f;

        #pragma unroll 1
        for (uint k = bid*32 + tid; k < K; k += gridDim.x*32)
        {
            float grad = load(Grad + k);
            float cv   = CV[k];

            if (zero_infs)
                grad = ew_zero_inf(grad);
            if (zero_nans)
                grad = ew_zero_nan(grad);
            if (saturate != 0.0f)
                grad = ew_maximum(ew_minimum(grad, saturate), -saturate);

            grad = ew_mul(grad, grad_scale * norm_scale);

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
}

// # kernel 4
// tf.assign_sub(param, learning_rate * x / np.maximum(1.0, np.sqrt(rms_x) / clipping_threshold) )
template <typename V>
__global__ void adafactor_apply(
    V* P, const V* __restrict__ X, const float* __restrict__ RMS_X, const float* __restrict__ Norm, float learning_rate, float rcp_clip, uint size, uint use_norm)
{
    float norm_scale = use_norm ? __ldg(Norm) : 1.0f;

    // skip all optimization if the global norm is bad.
    if (norm_scale != 0.0f)
    {
        uint tid = threadIdx.x;
        uint bid = blockIdx.x;

        float update_rate = learning_rate / fmaxf(sqrtf(__ldg(RMS_X)) * rcp_clip, 1.0f);

        #pragma unroll 1
        for (uint i = bid*blockDim.x + tid; i < size; i += gridDim.x*blockDim.x)
            P[i] = ew_sub(P[i], ew_mul(X[i], update_rate));
    }
}

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

template <typename T, typename V>
bool Adafactor(CUstream stream, uint SMs, float* cv, float* rv, float* x, float* means, float* param, const T* grad, const float* norm_scale, float grad_scale, float learning_rate, float decay, float epsilon, float clip_thresh, uint C, uint K, float saturate, bool zero_infs, bool zero_nans)
{
    cuMemsetD32Async((CUdeviceptr)means, 0, 2, stream); // used for row variance mean and RMS_X
    float rcpK    = 1.0f / (float)K;
    float rcpClip = 1.0f / clip_thresh;

    float* rv_mean = means;
    float* rms_x   = means + 1;

    // 1D case
    if (C == 1)
    {
        uint gridK = MIN(MAX(SMs*2, CEIL_DIV(K, 32*4)), SMs*32*2);

        adafactor_normalize_1d<T><<<gridK,32,0,stream>>>(cv, x, rms_x, grad, norm_scale, grad_scale, decay, epsilon, K, rcpK, saturate, zero_infs, zero_nans, norm_scale != 0);
        adafactor_apply<float><<<gridK,32,0,stream>>>(param, x, (const float*)rms_x, norm_scale, learning_rate, rcpClip, K, norm_scale != 0);
    }
    else
    {
        float rcpC = 1.0f / (float)C;
        uint gridK = CEIL_DIV(K, 32);

        adafactor_col_variance<T,float,1024><<<gridK,1024,0,stream>>>(cv, grad, norm_scale, grad_scale, decay, epsilon, C, K, rcpC, saturate, zero_infs, zero_nans, norm_scale != 0);

        if (K & 3)
        {
            uint CK = C*K;
            uint gridCK = CK > SMs*1024 ? SMs*2 : SMs;

            adafactor_row_variance<T,float><<<C,1024,0,stream>>>(rv, rv_mean, grad, norm_scale, grad_scale, decay, epsilon, K, rcpC, rcpK, saturate, zero_infs, zero_nans, norm_scale != 0);
            adafactor_normalize_2d<T,float><<<C,1024,0,stream>>>(x, rms_x, grad, norm_scale, (const float*)rv, (const float*)cv, (const float*)rv_mean, grad_scale, K, rcpC*rcpK, saturate, zero_infs, zero_nans, norm_scale != 0);
            adafactor_apply<float><<<gridCK,1024,0,stream>>>(param, (const float*)x, (const float*)rms_x, norm_scale, learning_rate, rcpClip, CK, norm_scale != 0);
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

            adafactor_row_variance<V,float4><<<C,256,0,stream>>>(rv, rv_mean, (const V*)grad, norm_scale, grad_scale, decay, epsilon, K, rcpC, rcpK, saturate, zero_infs, zero_nans, norm_scale != 0);
            adafactor_normalize_2d<V,float4><<<C,256,0,stream>>>((float4*)x, rms_x, (const V*)grad, norm_scale, (const float*)rv, (const float4*)cv, (const float*)rv_mean, grad_scale, K, rcpC*rcpK, saturate, zero_infs, zero_nans, norm_scale != 0);
            adafactor_apply<float4><<<gridCK,256,0,stream>>>((float4*)param, (const float4*)x, (const float*)rms_x, norm_scale, learning_rate, rcpClip, CK, norm_scale != 0);
        }
    }
    return true;
}
template bool Adafactor<float,float4>(CUstream stream, uint SMs, float* cv, float* rv, float* x, float* means, float* param, const float* grad, const float* norm_scale, float grad_scale, float learning_rate, float decay, float epsilon, float clip, uint C, uint K, float saturate, bool zero_infs, bool zero_nans);
template bool Adafactor<ehalf,ehalf4>(CUstream stream, uint SMs, float* cv, float* rv, float* x, float* means, float* param, const ehalf* grad, const float* norm_scale, float grad_scale, float learning_rate, float decay, float epsilon, float clip, uint C, uint K, float saturate, bool zero_infs, bool zero_nans);
template bool Adafactor<bhalf,bhalf4>(CUstream stream, uint SMs, float* cv, float* rv, float* x, float* means, float* param, const bhalf* grad, const float* norm_scale, float grad_scale, float learning_rate, float decay, float epsilon, float clip, uint C, uint K, float saturate, bool zero_infs, bool zero_nans);


template <typename TG, typename RM, typename RV>
__global__ void apply_lazy_emb_adam(
          float*              Param,
             RM*              Mean,
             RV*              Var,
    const    TG* __restrict__ Grad,
    const float* __restrict__ Norm,
    float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint K, float saturate, uint zero_infs, uint zero_nans, uint use_norm)
{
    float norm_scale = use_norm ? __ldg(Norm) : 1.0f;

    if (norm_scale != 0.0f)
    {
        uint tid = threadIdx.x;
        uint c   = blockIdx.x;
        uint k   = blockIdx.y*blockDim.x + tid;
        uint offset = c*K + k;

        float g = load(add_ptr_u(Grad, offset), 0, k < K);

        if (zero_infs)
            g = ew_zero_inf(g);
        if (zero_nans)
            g = ew_zero_nan(g);
        if (saturate != 0.0f)
            g = ew_maximum(ew_minimum(g, saturate), -saturate);

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
            float m = load(add_ptr_u((const    RM*)Mean,  offset));
            float v = load(add_ptr_u((const    RV*)Var,   offset));
            float p = load(add_ptr_u((const float*)Param, offset));

            g *= grad_scale * norm_scale;
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
}

template <typename TG, typename RM, typename RV>
__global__ void apply_adam(
          float*              Param,
             RM*              Mean,
             RV*              Var,
    const    TG* __restrict__ Grad,
    const float* __restrict__ Norm,
    float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint size, float saturate, uint zero_infs, uint zero_nans, uint use_norm)
{
    float norm_scale = use_norm ? __ldg(Norm) : 1.0f;

    // skip all optimization if the global norm is bad.
    if (norm_scale != 0.0f)
    {
        uint tid = threadIdx.x;
        uint bid = blockIdx.x;

        for (uint offset = bid*blockDim.x + tid; offset < size; offset += gridDim.x*blockDim.x)
        {
            float g = load(add_ptr_u(              Grad,  offset));
            float m = load(add_ptr_u((const    RM*)Mean,  offset));
            float v = load(add_ptr_u((const    RV*)Var,   offset));
            float p = load(add_ptr_u((const float*)Param, offset));

            if (zero_infs)
                g = ew_zero_inf(g);
            if (zero_nans)
                g = ew_zero_nan(g);
            if (saturate != 0.0f)
                g = ew_maximum(ew_minimum(g, saturate), -saturate);

            g *= grad_scale * norm_scale;
            v  = decay_var * v + (1.0f - decay_var) * g*g;

            float sigma = ew_sqrt(v);
            if (clip_sigma != 0.0f)
            {
                float clip = clip_sigma * sigma;
                g = fmaxf(g, -clip);
                g = fminf(g,  clip);
            }
            m  = decay_mean * m + (1.0f - decay_mean) * g;
            p -= lr * m * ew_rcp(sigma + epsilon);

            store(add_ptr_u(Mean,  offset), m);
            store(add_ptr_u(Var,   offset), v);
            store(add_ptr_u(Param, offset), p);
        }
    }
}

template <typename TG, typename TRM, typename TRV>
bool ApplyAdam(CUstream stream, uint SMs, const TG* grad, const float* norm_scale, float* param, TRM* mean, TRV* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint  size, uint  lazy_emb, float saturate, bool zero_infs, bool zero_nans)
{
    if (lazy_emb)
    {
        uint K = lazy_emb;
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
        apply_lazy_emb_adam<TG,TRM,TRV><<<dim3(C,gridK,1),threads,0,stream>>>(param, mean, var, grad, norm_scale, lr, decay_mean, decay_var, epsilon, grad_scale, clip_sigma, K, saturate, zero_infs, zero_nans, norm_scale != 0);
    }
    else
    {
        uint grid = SMs, threads = 64;
             if (size > SMs*1024) { threads = 1024; grid *= 2; }
        else if (size > SMs* 512) { threads = 1024; }
        else if (size > SMs* 256) { threads =  512; }
        else if (size > SMs* 128) { threads =  256; }
        else if (size > SMs*  64) { threads =  128; }

        apply_adam<TG,TRM,TRV><<<grid,threads,0,stream>>>(param, mean, var, grad, norm_scale, lr, decay_mean, decay_var, epsilon, grad_scale, clip_sigma, size, saturate, zero_infs, zero_nans, norm_scale != 0);
    }
    return true;
}
template bool ApplyAdam<float,float,float>(CUstream stream, uint SMs, const float* grad, const float* norm_scale, float* param, float* mean, float* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint size, uint lazy_update, float saturate, bool zero_infs, bool zero_nans);
template bool ApplyAdam<ehalf,float,float>(CUstream stream, uint SMs, const ehalf* grad, const float* norm_scale, float* param, float* mean, float* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint size, uint lazy_update, float saturate, bool zero_infs, bool zero_nans);
template bool ApplyAdam<bhalf,float,float>(CUstream stream, uint SMs, const bhalf* grad, const float* norm_scale, float* param, float* mean, float* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint size, uint lazy_update, float saturate, bool zero_infs, bool zero_nans);

template bool ApplyAdam<float,mhalf,vhalf>(CUstream stream, uint SMs, const float* grad, const float* norm_scale, float* param, mhalf* mean, vhalf* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint size, uint lazy_update, float saturate, bool zero_infs, bool zero_nans);
template bool ApplyAdam<ehalf,mhalf,vhalf>(CUstream stream, uint SMs, const ehalf* grad, const float* norm_scale, float* param, mhalf* mean, vhalf* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint size, uint lazy_update, float saturate, bool zero_infs, bool zero_nans);
template bool ApplyAdam<bhalf,mhalf,vhalf>(CUstream stream, uint SMs, const bhalf* grad, const float* norm_scale, float* param, mhalf* mean, vhalf* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint size, uint lazy_update, float saturate, bool zero_infs, bool zero_nans);


template <typename TG, typename RM, typename RV, uint BSIZE, uint THREADS>
__global__ void __launch_bounds__(THREADS) apply_adam_gated(
          float*              Param,
             RM*              Mean,
             RV*              Var,
    const    TG* __restrict__ Grad,
    const float* __restrict__ Norm,
    const float* __restrict__ Gate,
    float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, float saturate, uint zero_infs, uint zero_nans, uint use_norm)
{
    const uint U = BSIZE*BSIZE/THREADS;

    float norm_scale = use_norm ? __ldg(Norm) : 1.0f;
    if (norm_scale != 0.0f)
    {
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
            for (uint j = 0; j < U; j++) m[j] = load((const    RM*)Mean,  j*THREADS);
            for (uint j = 0; j < U; j++) v[j] = load((const    RV*)Var,   j*THREADS);
            for (uint j = 0; j < U; j++) p[j] = load((const float*)Param, j*THREADS);

            for (uint j = 0; j < U; j++)
            {
                if (zero_infs)
                    g[j] = ew_zero_inf(g[j]);
                if (zero_nans)
                    g[j] = ew_zero_nan(g[j]);
                if (saturate != 0.0f)
                    g[j] = ew_maximum(ew_minimum(g[j], saturate), -saturate);

                g[j] *= grad_scale * norm_scale;
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
}

template <typename TG, typename TRM, typename TRV>
bool ApplyAdamGated(CUstream stream, const float* gate, const TG* grad, const float* norm_scale, float* param, TRM* mean, TRV* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint  blocks, uint  bsize, float saturate, bool zero_infs, bool zero_nans)
{
    if (bsize == 8)
        apply_adam_gated<TG,TRM,TRV, 8,  32><<<blocks,  32,0,stream>>>(param, mean, var, grad, norm_scale, gate, lr, decay_mean, decay_var, epsilon, grad_scale, clip_sigma, saturate, zero_infs, zero_nans, norm_scale != 0);
    else if (bsize == 16)
        apply_adam_gated<TG,TRM,TRV,16,  64><<<blocks,  64,0,stream>>>(param, mean, var, grad, norm_scale, gate, lr, decay_mean, decay_var, epsilon, grad_scale, clip_sigma, saturate, zero_infs, zero_nans, norm_scale != 0);
    else if (bsize == 32)
        apply_adam_gated<TG,TRM,TRV,32, 256><<<blocks, 256,0,stream>>>(param, mean, var, grad, norm_scale, gate, lr, decay_mean, decay_var, epsilon, grad_scale, clip_sigma, saturate, zero_infs, zero_nans, norm_scale != 0);
    else if (bsize == 64)
        apply_adam_gated<TG,TRM,TRV,64,1024><<<blocks,1024,0,stream>>>(param, mean, var, grad, norm_scale, gate, lr, decay_mean, decay_var, epsilon, grad_scale, clip_sigma, saturate, zero_infs, zero_nans, norm_scale != 0);
    return true;
}

template bool ApplyAdamGated<float,float,float>(CUstream stream, const float* gate, const float* grad, const float* norm_scale, float* param, float* mean, float* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint blocks, uint bsize, float saturate, bool zero_infs, bool zero_nans);
template bool ApplyAdamGated<ehalf,float,float>(CUstream stream, const float* gate, const ehalf* grad, const float* norm_scale, float* param, float* mean, float* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint blocks, uint bsize, float saturate, bool zero_infs, bool zero_nans);
template bool ApplyAdamGated<bhalf,float,float>(CUstream stream, const float* gate, const bhalf* grad, const float* norm_scale, float* param, float* mean, float* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint blocks, uint bsize, float saturate, bool zero_infs, bool zero_nans);

template bool ApplyAdamGated<float,mhalf,vhalf>(CUstream stream, const float* gate, const float* grad, const float* norm_scale, float* param, mhalf* mean, vhalf* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint blocks, uint bsize, float saturate, bool zero_infs, bool zero_nans);
template bool ApplyAdamGated<ehalf,mhalf,vhalf>(CUstream stream, const float* gate, const ehalf* grad, const float* norm_scale, float* param, mhalf* mean, vhalf* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint blocks, uint bsize, float saturate, bool zero_infs, bool zero_nans);
template bool ApplyAdamGated<bhalf,mhalf,vhalf>(CUstream stream, const float* gate, const bhalf* grad, const float* norm_scale, float* param, mhalf* mean, vhalf* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint blocks, uint bsize, float saturate, bool zero_infs, bool zero_nans);

template <typename TG, typename RM, typename RV, uint BSIZE, uint THREADS>
__global__ void __launch_bounds__(THREADS) apply_blocksparse_adam(
          float*              Param,
             RM*              Mean,
             RV*              Var,
    const    TG* __restrict__ Grad,
    const float* __restrict__ Select,
    const float* __restrict__ Norm,
    float lr_old, float lr_new, float decay_mean, float decay_var, float epsilon, float grad_scale, float saturate, uint zero_infs, uint zero_nans, uint use_select, uint use_norm)
{
    const uint U = BSIZE*BSIZE/THREADS;

    float norm_scale = use_norm ? __ldg(Norm) : 1.0f;
    if (norm_scale != 0.0f)
    {
        grad_scale *= norm_scale;

        uint bid = blockIdx.x;
        uint tid = threadIdx.x;

        float select = use_select ? __ldg(Select + bid) : 0.0f;
        float lr = select == 0.0f ? lr_old : lr_new;

        uint offset = bid*BSIZE*BSIZE + tid;

        Grad  += offset;
        Mean  += offset;
        Var   += offset;
        Param += offset;

        float g[U], m[U], v[U], p[U];
        for (uint j = 0; j < U; j++) g[j] = load((const    TG*)Grad,  j*THREADS);
        for (uint j = 0; j < U; j++) m[j] = load((const    RM*)Mean,  j*THREADS);
        for (uint j = 0; j < U; j++) v[j] = load((const    RV*)Var,   j*THREADS);
        for (uint j = 0; j < U; j++) p[j] = load((const float*)Param, j*THREADS);

        for (uint j = 0; j < U; j++)
        {
            if (zero_infs)
                g[j] = ew_zero_inf(g[j]);
            if (zero_nans)
                g[j] = ew_zero_nan(g[j]);
            if (saturate != 0.0f)
                g[j] = ew_maximum(ew_minimum(g[j], saturate), -saturate);

            g[j] *= grad_scale;
            v[j]  = decay_var  * v[j] + (1.0f - decay_var ) * g[j] * g[j];
            m[j]  = decay_mean * m[j] + (1.0f - decay_mean) * g[j];
            p[j] -= lr * m[j]  * ew_rcp((ew_sqrt(v[j]) + epsilon));
        }
        for (uint j = 0; j < U; j++) store(Mean,  m[j], j*THREADS);
        for (uint j = 0; j < U; j++) store(Var,   v[j], j*THREADS);
        for (uint j = 0; j < U; j++) store(Param, p[j], j*THREADS);
    }
}

template <typename TG, typename TRM, typename TRV> bool BlocksparseAdam(CUstream stream,
  float* param, TRM* mean, TRV* var,
  const TG* grad,
  const float* lr_select,
  const float* norm_scale,
  float lr_old, float lr_new,
  float decay_mean, float decay_var, float epsilon,
  float grad_scale, float saturate, bool zero_infs, bool zero_nans,
  uint blocks, uint bsize)
{
    if (bsize == 8)
        apply_blocksparse_adam<TG,TRM,TRV, 8,  32><<<blocks,  32,0,stream>>>(param, mean, var, grad, lr_select, norm_scale, lr_old, lr_new, decay_mean, decay_var, epsilon, grad_scale, saturate, zero_infs, zero_nans, lr_select != 0, norm_scale != 0);
    else if (bsize == 16)
        apply_blocksparse_adam<TG,TRM,TRV,16,  64><<<blocks,  64,0,stream>>>(param, mean, var, grad, lr_select, norm_scale, lr_old, lr_new, decay_mean, decay_var, epsilon, grad_scale, saturate, zero_infs, zero_nans, lr_select != 0, norm_scale != 0);
    else if (bsize == 32)
        apply_blocksparse_adam<TG,TRM,TRV,32, 256><<<blocks, 256,0,stream>>>(param, mean, var, grad, lr_select, norm_scale, lr_old, lr_new, decay_mean, decay_var, epsilon, grad_scale, saturate, zero_infs, zero_nans, lr_select != 0, norm_scale != 0);
    else if (bsize == 64)
        apply_blocksparse_adam<TG,TRM,TRV,64,1024><<<blocks,1024,0,stream>>>(param, mean, var, grad, lr_select, norm_scale, lr_old, lr_new, decay_mean, decay_var, epsilon, grad_scale, saturate, zero_infs, zero_nans, lr_select != 0, norm_scale != 0);
    return true;
}
template bool BlocksparseAdam<float,float,float>(CUstream stream, float* param, float* mean, float* var, const float* grad, const float* lr_select, const float* norm_scale, float lr_old, float lr_new, float decay_mean, float decay_var, float epsilon, float grad_scale, float saturate, bool zero_infs, bool zero_nans, uint blocks, uint bsize);
template bool BlocksparseAdam<ehalf,float,float>(CUstream stream, float* param, float* mean, float* var, const ehalf* grad, const float* lr_select, const float* norm_scale, float lr_old, float lr_new, float decay_mean, float decay_var, float epsilon, float grad_scale, float saturate, bool zero_infs, bool zero_nans, uint blocks, uint bsize);
template bool BlocksparseAdam<float,mhalf,vhalf>(CUstream stream, float* param, mhalf* mean, vhalf* var, const float* grad, const float* lr_select, const float* norm_scale, float lr_old, float lr_new, float decay_mean, float decay_var, float epsilon, float grad_scale, float saturate, bool zero_infs, bool zero_nans, uint blocks, uint bsize);
template bool BlocksparseAdam<ehalf,mhalf,vhalf>(CUstream stream, float* param, mhalf* mean, vhalf* var, const ehalf* grad, const float* lr_select, const float* norm_scale, float lr_old, float lr_new, float decay_mean, float decay_var, float epsilon, float grad_scale, float saturate, bool zero_infs, bool zero_nans, uint blocks, uint bsize);


template <typename T, uint U>
__global__ void __launch_bounds__(32) apply_ema(T* Ema, const float* __restrict__ Param, float decay, uint size)
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
bool ApplyEma(CUstream stream, T* ema, const float* param, float decay, uint size)
{
    uint grid = CEIL_DIV(size, 128); // 1 warp with 4 unrolls
    if (grid > 200)
    {
        apply_ema<T,4><<<grid,32,0,stream>>>(ema, param, decay, size);
    }
    else
    {
        grid = CEIL_DIV(size, 32); // 1 warp with 1 unroll
        apply_ema<T,1><<<grid,32,0,stream>>>(ema, param, decay, size);
    }
    return true;
}
template bool ApplyEma<float>(CUstream stream, float* ema, const float* param, float decay, uint size);
template bool ApplyEma<ehalf>(CUstream stream, ehalf* ema, const float* param, float decay, uint size);



template <typename T, uint BSIZE, uint THREADS>
__global__ void __launch_bounds__(THREADS) apply_ema_gated(
              T*              Ema,
    const float* __restrict__ Param,
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
bool ApplyEmaGated(CUstream stream, T* ema, const float* param, const float* gate, float decay, uint blocks, uint bsize)
{
    if (bsize == 8)
        apply_ema_gated<T, 8,  32><<<blocks,  32,0,stream>>>(ema, param, gate, decay);
    else if (bsize == 16)
        apply_ema_gated<T,16,  64><<<blocks,  64,0,stream>>>(ema, param, gate, decay);
    else if (bsize == 32)
        apply_ema_gated<T,32, 256><<<blocks, 256,0,stream>>>(ema, param, gate, decay);
    else if (bsize == 64)
        apply_ema_gated<T,64,1024><<<blocks,1024,0,stream>>>(ema, param, gate, decay);
    return true;
}
template bool ApplyEmaGated<float>(CUstream stream, float* ema, const float* param, const float* gate, float decay, uint blocks, uint bsize);
template bool ApplyEmaGated<ehalf>(CUstream stream, ehalf* ema, const float* param, const float* gate, float decay, uint blocks, uint bsize);


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
            __shared__ float Share[32];

            // first thread of each warp store to shared
            if ((tid & 31) == 0)
                Share[tid / 32] = sum_sqared;

            __syncthreads();

            if (tid < 32)
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
            blocksparse_l2_decay<T, 8,  32,1><<<blocks,  32,0,stream>>>(param, gate, rate, epsilon);
        else if (bsize == 16)
            blocksparse_l2_decay<T,16,  64,1><<<blocks,  64,0,stream>>>(param, gate, rate, epsilon);
        else if (bsize == 32)
            blocksparse_l2_decay<T,32, 256,1><<<blocks, 256,0,stream>>>(param, gate, rate, epsilon);
        else if (bsize == 64)
            blocksparse_l2_decay<T,64,1024,1><<<blocks,1024,0,stream>>>(param, gate, rate, epsilon);
    }
    else
    {
        if (bsize == 8)
            blocksparse_l2_decay<T, 8,  32,0><<<blocks,  32,0,stream>>>(param, gate, rate, epsilon);
        else if (bsize == 16)
            blocksparse_l2_decay<T,16,  64,0><<<blocks,  64,0,stream>>>(param, gate, rate, epsilon);
        else if (bsize == 32)
            blocksparse_l2_decay<T,32, 256,0><<<blocks, 256,0,stream>>>(param, gate, rate, epsilon);
        else if (bsize == 64)
            blocksparse_l2_decay<T,64,1024,0><<<blocks,1024,0,stream>>>(param, gate, rate, epsilon);
    }
    return true;
}
template bool BlocksparseL2Decay<float>(CUstream stream, float* param, const float* gate, float rate, float epsilon, uint blocks, uint bsize);


#define MAX_NORM 0
#define L2_NORM  1

template <typename T, uint BSIZE, uint THREADS, uint NORM>
__global__ void __launch_bounds__(THREADS) blocksparse_norm(float* Norm, const T* __restrict__ Param)
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
    float norm = 0.0f;
    for (uint j = 0; j < U; j++)
        if (NORM == MAX_NORM)
            norm  = fmaxf(fabsf(p[j]), norm);
        else
            norm += ew_sqr(p[j]);

    // reduce within warp
    for (int i = 16; i > 0; i >>= 1)
        if (NORM == MAX_NORM)
            norm  = fmaxf(norm, shfl_xor(norm, i));
        else
            norm += shfl_xor(norm, i);

    // if using more than 1 warp, further reduced with shared memory
    if (THREADS > 32)
    {
        __shared__ float Share[32];

        // first thread of each warp store to shared
        if ((tid & 31) == 0)
            Share[tid / 32] = norm;

        __syncthreads();

        if (tid < 32)
        {
            // first warp loads all prior reductions
            norm = Share[tid];

            // reduce within this first warp
            for (int i = THREADS/64; i > 0; i >>= 1)
                if (NORM == MAX_NORM)
                    norm  = fmaxf(norm, shfl_xor(norm, i));
                else
                    norm += shfl_xor(norm, i);
        }
    }
    // first thread has the final reduced max_abs
    if (tid == 0)
    {
        if (NORM == L2_NORM)
            norm = ew_sqrt(norm);
        Norm[bid] = norm;
    }
}

template <typename T>
bool BlocksparseNorm(CUstream stream, float* norm, const T* param, uint blocks, uint bsize, uint norm_type)
{
    if (norm_type == MAX_NORM)
    {
        if (bsize == 8)
            blocksparse_norm<T, 8,  32,MAX_NORM><<<blocks,  32,0,stream>>>(norm, param);
        else if (bsize == 16)
            blocksparse_norm<T,16,  64,MAX_NORM><<<blocks,  64,0,stream>>>(norm, param);
        else if (bsize == 32)
            blocksparse_norm<T,32, 256,MAX_NORM><<<blocks, 256,0,stream>>>(norm, param);
        else if (bsize == 64)
            blocksparse_norm<T,64,1024,MAX_NORM><<<blocks,1024,0,stream>>>(norm, param);
    }
    else
    {
        if (bsize == 8)
            blocksparse_norm<T, 8,  32, L2_NORM><<<blocks,  32,0,stream>>>(norm, param);
        else if (bsize == 16)
            blocksparse_norm<T,16,  64, L2_NORM><<<blocks,  64,0,stream>>>(norm, param);
        else if (bsize == 32)
            blocksparse_norm<T,32, 256, L2_NORM><<<blocks, 256,0,stream>>>(norm, param);
        else if (bsize == 64)
            blocksparse_norm<T,64,1024, L2_NORM><<<blocks,1024,0,stream>>>(norm, param);
    }
    return true;
}
template bool BlocksparseNorm<float>(CUstream stream, float* norm, const float* param, uint blocks, uint bsize, uint norm_type);
template bool BlocksparseNorm<ehalf>(CUstream stream, float* norm, const ehalf* param, uint blocks, uint bsize, uint norm_type);
template bool BlocksparseNorm<bhalf>(CUstream stream, float* norm, const bhalf* param, uint blocks, uint bsize, uint norm_type);


__global__ void __launch_bounds__(256) blocksparse_prune(float* Gate, const uint* __restrict__ Idx, uint blocks, uint keep)
{
    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    #pragma unroll 1
    for (uint i = bid*256 + tid; i < blocks; i += gridDim.x*256)
    {
        Gate[Idx[i]] = i < keep ? 1.0f : 0.0f;
    }
}
bool BlocksparsePrune(CUstream stream, uint SMs, float* gate, const uint* idx, uint blocks, uint keep)
{
    uint grid = blocks > SMs*512 ? SMs*4 : blocks > SMs*256 ? SMs*2 : SMs;
    blocksparse_prune<<<grid,256,0,stream>>>(gate, idx, blocks, keep);
    return true;
}



template <typename T, uint BSIZE, uint THREADS, uint NORM>
__global__ void __launch_bounds__(THREADS) blocksparse_threshold_prune(const T* __restrict__ Param, float* Gate, float threshold)
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
    float norm = 0.0f;
    for (uint j = 0; j < U; j++)
        if (NORM == MAX_NORM)
            norm  = fmaxf(fabsf(p[j]), norm);
        else
            norm += ew_sqr(p[j]);

    // reduce within warp
    for (int i = 16; i > 0; i >>= 1)
        if (NORM == MAX_NORM)
            norm  = fmaxf(norm, shfl_xor(norm, i));
        else
            norm += shfl_xor(norm, i);

    // if using more than 1 warp, further reduced with shared memory
    if (THREADS > 32)
    {
        __shared__ float Share[32];

        // first thread of each warp store to shared
        if ((tid & 31) == 0)
            Share[tid/32] = norm;

        __syncthreads();

        if (tid < 32)
        {
            // first warp loads all prior reductions
            norm = Share[tid];

            // reduce within this first warp
            for (int i = THREADS/64; i > 0; i >>= 1)
                if (NORM == MAX_NORM)
                    norm  = fmaxf(norm, shfl_xor(norm, i));
                else
                    norm += shfl_xor(norm, i);
        }
    }
    // first thread has the final reduced max_abs
    // compare against threshhold and update gate if needed.
    // if (bid < 2 && tid == 0)
    //     printf("%d %d %.5f %.5f\n", bid, gridDim.x, max_abs, threshold);
    if (tid == 0)
    {
        if (NORM == L2_NORM)
            norm = ew_sqrt(norm);
        Gate[bid] = norm < threshold ? 0.0f : 1.0f;
    }
}

template <typename T>
bool BlocksparseThresholdPrune(CUstream stream, const T* param, float* gate, float threshold, uint blocks, uint bsize, uint norm_type)
{
    if (norm_type == MAX_NORM)
    {
        if (bsize == 8)
            blocksparse_threshold_prune<T, 8,  32,MAX_NORM><<<blocks,  32,0,stream>>>(param, gate, threshold);
        else if (bsize == 16)
            blocksparse_threshold_prune<T,16,  64,MAX_NORM><<<blocks,  64,0,stream>>>(param, gate, threshold);
        else if (bsize == 32)
            blocksparse_threshold_prune<T,32, 256,MAX_NORM><<<blocks, 256,0,stream>>>(param, gate, threshold);
        else if (bsize == 64)
            blocksparse_threshold_prune<T,64,1024,MAX_NORM><<<blocks,1024,0,stream>>>(param, gate, threshold);
    }
    else
    {
        if (bsize == 8)
            blocksparse_threshold_prune<T, 8,  32, L2_NORM><<<blocks,  32,0,stream>>>(param, gate, threshold);
        else if (bsize == 16)
            blocksparse_threshold_prune<T,16,  64, L2_NORM><<<blocks,  64,0,stream>>>(param, gate, threshold);
        else if (bsize == 32)
            blocksparse_threshold_prune<T,32, 256, L2_NORM><<<blocks, 256,0,stream>>>(param, gate, threshold);
        else if (bsize == 64)
            blocksparse_threshold_prune<T,64,1024, L2_NORM><<<blocks,1024,0,stream>>>(param, gate, threshold);
    }
    return true;
}
template bool BlocksparseThresholdPrune<float>(CUstream stream, const float* param, float* gate, float threshold, uint blocks, uint bsize, uint norm_type);



template <typename T, typename V>
__global__ void __launch_bounds__(1024) reduce_sum_squared(float* SumSquared, const T* X, uint size, float grad_scale, float saturate, uint zero_infs, uint zero_nans)
{
    __shared__ float Share[32];

    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    // check if this block has any work to do
    if (bid * 1024 < size)
    {
        float sum_squared = 0.0f;
        #pragma unroll 1
        for (uint offset = bid * 1024 + tid; offset < size; offset += gridDim.x*1024)
        {
            V x = load(X + offset);

            if (zero_infs)
                x = ew_zero_inf(x);
            if (zero_nans)
                x = ew_zero_nan(x);
            if (saturate != 0.0f)
                x = ew_maximum(ew_minimum(x, saturate), -saturate);

            x = ew_mul(x, grad_scale);
            sum_squared += ew_sum(ew_sqr(x));
        }

        // reduce within warp
        #pragma unroll
        for (int i = 16; i > 0; i >>= 1)
            sum_squared += shfl_xor(sum_squared, i);

        // first thread of each warp store to shared
        if ((tid & 31) == 0)
            Share[tid/32] = sum_squared;

        __syncthreads();

        if (tid < 32)
        {
            // first warp loads all prior reductions
            sum_squared = Share[tid];

            // reduce within this last warp
            for (int i = 16; i > 0; i >>= 1)
                sum_squared += shfl_xor(sum_squared, i);

            if (tid == 0)
                atomicRed(SumSquared, sum_squared);
        }
    }
}


template <typename T, typename V>
bool ReduceSumSquared(CUstream stream, uint SMs, float* sum_squared, const T* x, uint size, float grad_scale, float saturate, bool zero_infs, bool zero_nans, uint tensor_idx, uint tensor_cnt)
{
    if (tensor_idx == 0)
       cuMemsetD32Async((CUdeviceptr)sum_squared, 0, tensor_cnt, stream);

    sum_squared += tensor_idx;

    if ((size & 3) == 0 && size > SMs*1024)
    {
        size >>= 2;
        uint grid = size > SMs*1024 ? SMs*2 : SMs;
        reduce_sum_squared<V,float4><<<grid,1024,0,stream>>>(sum_squared, (const V*)x, size, grad_scale, saturate, zero_infs, zero_nans);
    }
    else
    {
        uint grid = size > SMs*1024 ? SMs*2 : SMs;
        reduce_sum_squared<T,float><<<grid,1024,0,stream>>>(sum_squared, x, size, grad_scale, saturate, zero_infs, zero_nans);
    }
    return true;
}
template bool ReduceSumSquared<float,float4>(CUstream stream, uint SMs, float* sum_squared, const float* x, uint size, float grad_scale, float saturate, bool zero_infs, bool zero_nans, uint tensor_idx, uint tensor_cnt);
template bool ReduceSumSquared<bhalf,bhalf4>(CUstream stream, uint SMs, float* sum_squared, const bhalf* x, uint size, float grad_scale, float saturate, bool zero_infs, bool zero_nans, uint tensor_idx, uint tensor_cnt);
template bool ReduceSumSquared<ehalf,ehalf4>(CUstream stream, uint SMs, float* sum_squared, const ehalf* x, uint size, float grad_scale, float saturate, bool zero_infs, bool zero_nans, uint tensor_idx, uint tensor_cnt);

__global__ void compute_clip_norm(float* Norm, float* Scale, const float* SumSquared, float clip_norm, uint tensor_cnt)
{
    __shared__ float Share[32];

    uint tid = threadIdx.x;

    float sum_squared = 0.0f;
    #pragma unroll 1
    for (uint offset = tid; offset < tensor_cnt; offset += 1024)
        sum_squared += __ldg(SumSquared + offset);

    // reduce within warp
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1)
        sum_squared += shfl_xor(sum_squared, i);

    // first thread of each warp store to shared
    if ((tid & 31) == 0)
        Share[tid/32] = sum_squared;

    __syncthreads();

    if (tid < 32)
    {
        // first warp loads all prior reductions
        sum_squared = Share[tid];

        // reduce within this last warp
        for (int i = 16; i > 0; i >>= 1)
            sum_squared += shfl_xor(sum_squared, i);

        if (tid == 0)
        {
            float global_norm = sqrtf(sum_squared);

            uint is_finite;
            asm("{                               \n\t"
                ".reg .pred is_finite;           \n\t"
                "testp.finite.f32 is_finite, %1; \n\t"
                "selp.u32 %0, 1, 0, is_finite;   \n\t"
                "}" : "=r"(is_finite) : "f"(global_norm));

            if (is_finite == 1)
                *Scale = clip_norm / fmaxf(global_norm, clip_norm);
            else
                *Scale = 0.0f; // use zero for sentinal value to skip updates

            *Norm = global_norm;
        }
    }
}

bool ComputeClipNorm(CUstream stream, float* l2norm, float* scale, float* sum_squared, float clip_norm, uint tensor_cnt)
{
    compute_clip_norm<<<1,1024,0,stream>>>(l2norm, scale, (const float*)sum_squared, clip_norm, tensor_cnt);
    return true;
}


#endif
