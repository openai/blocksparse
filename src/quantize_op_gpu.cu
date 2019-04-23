#if GOOGLE_CUDA

#include "ew_op_gpu.h"
#include <stdio.h>
#include <type_traits>

template <typename T, uint STOCHASTIC, uint THREADS, uint UNROLL>
__global__ void __launch_bounds__(THREADS) quantize(uint* E, T* Y, const T* X, float round_scale, uint trunc_mask, float max_float, float min_float, uint exp_norm, uint size)
{
    const uint tid = threadIdx.x;
    const uint bid = blockIdx.x;

    const uint offsetX = bid       * THREADS * UNROLL + tid;
    const uint strideX = gridDim.x * THREADS * UNROLL;

    if (offsetX < size)
    {
        uint lfsr0, lfsr1, lfsr2;
        if (STOCHASTIC == 1)
        {
            // Grab some entropy wherever we can and evenly distribute it
            uint idx = bid * THREADS + tid;
            asm("mov.b32 %0, %%clock_hi;"       : "=r"(lfsr0) :);
            asm("mov.b32 %0, %%clock;"          : "=r"(lfsr1) :);
            asm("mov.b32 %0, %%globaltimer_lo;" : "=r"(lfsr2) :);
            asm("shf.r.clamp.b32 %0,%0,%0,%1;"  : "=r"(lfsr0) : "r"((lfsr0^tid) & 31)); // rotate bits
            asm("shf.r.clamp.b32 %0,%0,%0,%1;"  : "=r"(lfsr1) : "r"((lfsr1^tid) & 31)); // rotate bits
            asm("shf.r.clamp.b32 %0,%0,%0,%1;"  : "=r"(lfsr2) : "r"((lfsr2^tid) & 31)); // rotate bits
            lfsr0 ^= idx ^ (idx << 5)  ^ (idx << 11) ^ (idx << 17) ^ (idx << 23);
        }
        else if (STOCHASTIC == 2)
        {
            lfsr0 = __ldg(add_ptr_u((const uint*)E, gridDim.x*THREADS*0 + bid*THREADS + tid));
            lfsr1 = __ldg(add_ptr_u((const uint*)E, gridDim.x*THREADS*1 + bid*THREADS + tid));
            lfsr2 = __ldg(add_ptr_u((const uint*)E, gridDim.x*THREADS*2 + bid*THREADS + tid));
        }

        #pragma unroll 1
        for (uint offset = offsetX; offset < size; offset += strideX)
        {
            const T* Xp = add_ptr_u(X, offset);
                  T* Yp = add_ptr_u(Y, offset);

            #pragma unroll
            for (uint j = 0; j < UNROLL; j++)
            {
                bool in_bounds = offset + j*THREADS < size;

                float x = load(Xp, j*THREADS, in_bounds);

                float rscale = round_scale;
                if (STOCHASTIC)
                {
                    // tausworthe generator (low quality rng is just fine)
                    lfsr0 = ((lfsr0 & 0xfffffffe) << 12) ^ (((lfsr0 << 13) ^ lfsr0) >> 19);
                    lfsr1 = ((lfsr1 & 0xfffffff8) <<  4) ^ (((lfsr1 << 2)  ^ lfsr1) >> 25);
                    lfsr2 = ((lfsr2 & 0xfffffff0) << 11) ^ (((lfsr2 << 3)  ^ lfsr2) >> 11);
                    rscale *= (float)(lfsr0 ^ lfsr1 ^ lfsr2);
                }

                asm("{                                     \n\t"
                    ".reg .f32 sign_exp, val;              \n\t"
                    "and.b32 sign_exp, %0, 0xff800000;     \n\t" // extract sign/exponent
                    "fma.rz.ftz.f32 val, sign_exp, %1, %0; \n\t" // add the round amount just below the final ulp position
                    "and.b32 %0, val, %2;                  \n\t" // truncate off unused mantissa
                    "}" : "+f"(x) : "f"(rscale), "r"(trunc_mask));

                x = fmaxf(x, -max_float);
                x = fminf(x,  max_float);
                if (fabs(x) < min_float)
                    x = 0.0f;
                else
                {
                    // Denorm Quantization:
                    // First subtract value off of exponent that will bring min_float to an unbiased exponent of 1.
                    // Then mul by 2**-23 to force truncation of any unused sub normal bits.
                    // Then scale back to origal exponent by reversing this process.
                    asm("{                            \n\t"
                        ".reg .f32 f;                 \n\t"
                        ".reg .u32 u;                 \n\t"
                        "mov.b32 u, %0;               \n\t"
                        "sub.u32 u, u, %1;            \n\t"
                        "mov.b32 f, u;                \n\t"
                        "mul.rn.f32 f, f, 0F34000000; \n\t" // 2 **-23, round to nearest denorm
                        "mul.rz.f32 f, f, 0F4b000000; \n\t" // 2 ** 23
                        "mov.b32 u, f;                \n\t"
                        "add.u32 u, u, %1;            \n\t"
                        "mov.b32 %0, u;               \n\t"
                        "}" : "+f"(x) : "r"(exp_norm));
                }
                store(Yp, x, j*THREADS, in_bounds);
            }
        }
        if (STOCHASTIC == 2)
        {
            __stg(add_ptr_u(E, gridDim.x*THREADS*0 + bid*THREADS + tid), lfsr0);
            __stg(add_ptr_u(E, gridDim.x*THREADS*1 + bid*THREADS + tid), lfsr1);
            __stg(add_ptr_u(E, gridDim.x*THREADS*2 + bid*THREADS + tid), lfsr2);
        }
    }
}


template <typename T>
__global__ void __launch_bounds__(1024) quantization_stats(float* S, const T* X, float max_float, float ftz_float, float rcp_size, uint size)
{
    __shared__ float4 Share4[32];
    __shared__ float  Share1[32];

    uint tid = threadIdx.x;
    uint bid = blockIdx.x;

    // check if this block has any work to do
    if (bid * 1024 < size)
    {
        float4 stats = {0,0,0,0}; // mean, mean**2, n_sat, n_ftz
        float  x_max = 0;
        for (uint offset = bid * 1024 + tid; offset < size; offset += gridDim.x*1024)
        {
            float x = load(add_ptr_u(X, offset));

            // Nans => Inf
            asm("{                               \n\t"
                ".reg .pred is_number;           \n\t"
                "testp.number.f32 is_number, %0; \n\t"
                "selp.f32 %0, %0, 0F7f800000, is_number;\n\t"
                "}" : "+f"(x) :);

            // Saturate fp16 infinity values
            if (std::is_same<T, ehalf>::value)
                x = fmaxf(fminf(x, 65504.0f), -65504.0f);

            stats.x += abs(x);
            stats.y += x*x;
            stats.z += abs(x) >= max_float ? 1.0f : 0.0f;
            stats.w += x != 0.0f && abs(x) < ftz_float ? 1.0f : 0.0f;
            x_max    = fmaxf(x_max, abs(x));
        }

        // reduce within warp
        #pragma unroll
        for (int i = 16; i > 0; i >>= 1)
        {
            stats.x += shfl_xor(stats.x, i);
            stats.y += shfl_xor(stats.y, i);
            stats.z += shfl_xor(stats.z, i);
            stats.w += shfl_xor(stats.w, i);
            x_max    = fmaxf(x_max, shfl_xor(x_max, i));
        }
        // first thread of each warp store to shared
        if ((tid & 31) == 0)
        {
            Share4[tid >> 5] = stats;
            Share1[tid >> 5] = x_max;
        }
        __syncthreads();

        if (tid < 32)
        {
            // first warp loads all prior reductions
            stats = Share4[tid];
            x_max = Share1[tid];

            // reduce within this last warp
            #pragma unroll
            for (int i = 16; i > 0; i >>= 1)
            {
                stats.x += shfl_xor(stats.x, i);
                stats.y += shfl_xor(stats.y, i);
                stats.z += shfl_xor(stats.z, i);
                stats.w += shfl_xor(stats.w, i);
                x_max    = fmaxf(x_max, shfl_xor(x_max, i));
            }
            // All threads in this warp now have the same final reduction values
            // First 4 stats are sums, so add them in one shot
            float sum =
                tid == 0 ? stats.x * rcp_size :
                tid == 1 ? stats.y * rcp_size :
                tid == 2 ? stats.z * rcp_size :
                           stats.w * rcp_size ;
            if (tid < 4)
                atomicRed(add_ptr_u(S, tid), sum);

            // Last stat needs to be maxed seperately
            if (tid == 0)
                atomicRedMax(S + 4, x_max);
        }
    }
}


template <typename T>
bool Quantize(CUstream stream, uint SMs, uint* entropy, T* y, const T* x, float round_scale, uint trunc_mask, float max_float, float min_float, uint exp_norm, uint size, int stochastic)
{
    if (stochastic)
    {
        // stochastic mode does more compute per load and needs fewer total threads.
        uint grid =
            size >= SMs*8*128*4 ? SMs*8 :
            size >= SMs*4*128*4 ? SMs*4 :
            size >= SMs*2*128*4 ? SMs*2 :
                                  SMs   ;
        if (entropy != NULL)
            quantize<T,2,128,4><<<grid,128,0,stream>>>(entropy, y, x, round_scale, trunc_mask, max_float, min_float, exp_norm, size);
        else
            quantize<T,1,128,4><<<grid,128,0,stream>>>(entropy, y, x, round_scale, trunc_mask, max_float, min_float, exp_norm, size);
    }
    else
    {
        uint grid =
            size >= SMs*16*128*4 ? SMs*16 :
            size >= SMs* 8*128*4 ? SMs* 8 :
            size >= SMs* 4*128*4 ? SMs* 4 :
            size >= SMs* 2*128*4 ? SMs* 2 :
                                   SMs    ;

        quantize<T,0,128,4><<<grid,128,0,stream>>>(entropy, y, x, round_scale, trunc_mask, max_float, min_float, exp_norm, size);
    }
    return true; // TODO
}

template <typename T>
QuantStats QuantizationStats(CUstream stream, uint SMs, float* s, const T* x, float max_float, float ftz_float, uint size)
{
    QuantStats stats;
    uint grid = size > SMs*1024 ? SMs*2 : SMs;

    cuMemsetD8Async((CUdeviceptr)s, 0, sizeof(stats), stream);

    quantization_stats<T><<<grid,1024,0,stream>>>(s, x, max_float, ftz_float, 1.0f/(float)size, size);

    cuMemcpyDtoHAsync((void*)&stats, (CUdeviceptr)s, sizeof(stats), stream);

    // var(x) == mean(x**2) - mean(x)**2
    stats.stdv     = sqrtf(stats.stdv - stats.mean*stats.mean);
    stats.sat_pct *= 100.0f;
    stats.ftz_pct *= 100.0f;
    return stats;
}


template bool Quantize<float>(CUstream stream, uint SMs, uint* entropy, float* y, const float* x, float round_scale, uint trunc_mask, float max_float, float min_float, uint exp_norm, uint size, int stochastic);
template bool Quantize<bhalf>(CUstream stream, uint SMs, uint* entropy, bhalf* y, const bhalf* x, float round_scale, uint trunc_mask, float max_float, float min_float, uint exp_norm, uint size, int stochastic);

template QuantStats QuantizationStats<float>(CUstream stream, uint SMs, float* s, const float* x, float max_float, float ftz_float, uint size);
template QuantStats QuantizationStats<bhalf>(CUstream stream, uint SMs, float* s, const bhalf* x, float max_float, float ftz_float, uint size);
template QuantStats QuantizationStats<ehalf>(CUstream stream, uint SMs, float* s, const ehalf* x, float max_float, float ftz_float, uint size);


#endif
