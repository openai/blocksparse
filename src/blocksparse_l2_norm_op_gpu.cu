
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "ew_op_gpu.h"

__device__ __forceinline__ int div16(int numerator, int magic, int shift)
{
    int res;
    asm("vmad.s32.u32.u32 %0, %1.h0, %2.h0, 0;" : "=r"(res) : "r"(numerator), "r"(magic));
    return res >> shift;
}
__device__ __forceinline__ int mod16(int numerator, int div, int maxdiv)
{
    int res;
    asm("vmad.s32.u32.u32 %0, -%1.h0, %2.h0, %3;" : "=r"(res) : "r"(div), "r"(maxdiv), "r"(numerator));
    return res;
}
__device__ __forceinline__ int mad16(int a, int b, int c)
{
    int res;
    asm("vmad.s32.u32.u32 %0, %1.h0, %2.h0, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
    return res;
}


// y = gain * x / sqrt(max(sum(x**2), epsilon))
template <typename TY, typename TX>
__global__ void __launch_bounds__(32) l2_normalize_KCTRS(
             TY*              Y,
          float*              S,
    const    TX* __restrict__ X,
    const float* __restrict__ G,
    const  int2* __restrict__ Lut,
    float epsilon, int apply_gain)
{
    int tid = threadIdx.x;
    int k   = blockIdx.x;

    int2 block_data = Lut[k];

    float gain = 1.0f;
    if (apply_gain) gain = G[k];

    int offset = block_data.x + tid; // block_F + idx_k * CTRS + tid
    int CTRS   = block_data.y;       // block_C * TRS

    const TX* X1 = X + offset;
    const TX* X2 = X + offset;
    Y += offset;

    // sum_sqr_x = sum(x**2)
    float sum_sqr_x = 0.0f;
    for (int i = tid; i < CTRS; i += 32)
    {
        float x = load(X1);
        X1 += 32;
        sum_sqr_x += x * x;
    }
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1)
        sum_sqr_x += __shfl_xor(sum_sqr_x, i);

    // store reduction for gradient pass
    if (tid == 0)
        store(S, sum_sqr_x, k);

    // rnorm = 1.0f / sqrt(max(sum_sqr_x, epsilon)) * gain
    float rnorm = rsqrtf(fmaxf(sum_sqr_x, epsilon)) * gain;

    // y = x * rnorm
    for (int i = tid; i < CTRS; i += 32)
    {
        float x = load(X2);
        store(Y, x * rnorm);
        X2 += 32;
        Y  += 32;
    }
}

// y = gain * x / sqrt(max(sum(x**2), epsilon))
template <typename TY, typename TX>
__global__ void __launch_bounds__(32) l2_normalize_CKTRS(
             TY*              Y,
          float*              S,
    const    TX* __restrict__ X,
    const float* __restrict__ G,
    const  int4* __restrict__ Lut,
    float epsilon, int apply_gain, int TRS, int magic_TRS, int shift_TRS)
{
    int tid = threadIdx.x;
    int k   = blockIdx.x;

    int4 block_data = Lut[k];

    float gain = 1.0f;
    if (apply_gain) gain = G[k];


    int idx_k   = block_data.x;
    int CTRS    = block_data.y;
    int KTRS    = block_data.z;
    int block_F = block_data.w;

    int offset_F = block_F + idx_k * TRS;

    const TX* X1 = X + offset_F;
    const TX* X2 = X + offset_F;
    Y += offset_F;

    // y_val = sum(x**2)
    float sum_sqr_x = 0.0f;
    for (int ctrs = tid; ctrs < CTRS; ctrs += 32)
    {
        //       c = i / TRS;
        //     trs = i % TRS;
        //  offset = c * KTRS + trs
        int      c = div16(ctrs, magic_TRS, shift_TRS);
        int    trs = mod16(ctrs, c, TRS);
        int offset = mad16(c, KTRS, trs);

        float x = load(X1, offset);

        sum_sqr_x += x * x;
    }
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1)
        sum_sqr_x += __shfl_xor(sum_sqr_x, i);

    // store reduction for gradient pass
    if (tid == 0)
        store(S, sum_sqr_x, k);

    // rnorm = 1.0f / sqrt(max(sum_sqr_x, epsilon)) * gain
    float rnorm = rsqrtf(fmaxf(sum_sqr_x, epsilon)) * gain;

    // y = x * rnorm
    for (int ctrs = tid; ctrs < CTRS; ctrs += 32)
    {
        int      c = div16(ctrs, magic_TRS, shift_TRS);
        int    trs = mod16(ctrs, c, TRS);
        int offset = mad16(c, KTRS, trs);

        float x = load(X2, offset);

        store(Y, x * rnorm, offset);
    }
}


// y = gain * x / sqrt(max(sum(x**2), epsilon))
template <typename TY, typename TX>
__global__ void __launch_bounds__(128) l2_normalize_CK_32(
             TY*              Y,
          float*              S,
    const    TX* __restrict__ X,
    const float* __restrict__ G,
    const   int* __restrict__ Lut,
    float epsilon, int apply_gain)
{
    extern __shared__   int iShare[]; // 96 + max(lut_size)
    extern __shared__ float fShare[]; // 96 + max(lut_size)

    int tid   = threadIdx.x;
    int idx_L = blockIdx.x;

    int4 lut_head = ((const int4*)Lut)[idx_L];
    // unpack lut header
    int lut_offset = lut_head.x;
    int lut_size   = lut_head.y;
    int idx_K      = lut_head.z;

    int k = idx_K*32 + (tid & 31);

    float gain = 1.0f;
    if (apply_gain) gain = G[k];

    Lut += lut_offset;
    #pragma unroll 1
    for (int i = tid; i < lut_size; i += 128)
        iShare[i + 96] = Lut[i] * 32 * 32;

    __syncthreads();

    // sum_sqr_x = sum(x**2)
    float sum_sqr_x = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < lut_size; i++)
    {
        const TX* X1 = X + iShare[i + 96] + tid;

        #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            float x = load(X1, j*128);
            sum_sqr_x += x * x;
        }
    }
    // reduce sum_sqr_x across the 4 warps
    if (tid >= 32)
        fShare[tid-32] = sum_sqr_x;
    __syncthreads();

    if (tid < 32)
    {
        sum_sqr_x += fShare[tid] + fShare[tid + 32] + fShare[tid + 64];
        fShare[tid] = sum_sqr_x;
        // store reduction for gradient pass
        store(S, sum_sqr_x, k);
    }
    __syncthreads();

    // get the final reduced value for all warps:
    sum_sqr_x = fShare[tid & 31];

    // rnorm = 1.0f / sqrt(max(sum_sqr_x, epsilon)) * gain
    float rnorm = rsqrtf(fmaxf(sum_sqr_x, epsilon)) * gain;

    // y = x * rnorm
    #pragma unroll 1
    for (int i = 0; i < lut_size; i++)
    {
        int block_offset = iShare[i + 96];

        const TX* X2 = X + block_offset + tid;
              TY* Y2 = Y + block_offset + tid;

        #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            float x = load(X2, j*128);
            store(Y2, x * rnorm, j*128);
        }
    }
}

// y = gain * x / sqrt(max(sum(x**2), epsilon))
template <typename TY, typename TX>
__global__ void __launch_bounds__(32) l2_normalize_CK_16(
             TY*              Y,
          float*              S,
    const    TX* __restrict__ X,
    const float* __restrict__ G,
    const   int* __restrict__ Lut,
    float epsilon, int apply_gain)
{
    extern __shared__ int lut[]; // max(lut_size)

    int tid   = threadIdx.x;
    int idx_L = blockIdx.x;

    int4 lut_head = ((const int4*)Lut)[idx_L];
    // unpack lut header
    int lut_offset = lut_head.x;
    int lut_size   = lut_head.y;
    int idx_K      = lut_head.z;

    int k = idx_K*16 + (tid & 15);

    float gain = 1.0f;
    if (apply_gain) gain = G[k];

    Lut += lut_offset;
    #pragma unroll 1
    for (int i = tid; i < lut_size; i += 32)
        lut[i] = Lut[i] * 16 * 16;

    // sum_sqr_x = sum(x**2)
    float sum_sqr_x = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < lut_size; i++)
    {
        const TX* X0 = X + lut[i] + tid;

        #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            float x = load(X0, j*32);
            sum_sqr_x += x * x;
        }
    }
    // reduce sum_sqr_x across the 4 rows of the warp
    sum_sqr_x += __shfl_xor(sum_sqr_x, 16);

    store(S, sum_sqr_x, k, tid < 16);

    // rnorm = 1.0f / sqrt(max(sum_sqr_x, epsilon)) * gain
    float rnorm = rsqrtf(fmaxf(sum_sqr_x, epsilon)) * gain;

    // y = x * rnorm
    #pragma unroll 1
    for (int i = 0; i < lut_size; i++)
    {
        int block_offset = lut[i];

        const TX* X0 = X + block_offset + tid;
              TY* Y0 = Y + block_offset + tid;

        #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            float x = load(X0, j*32);
            store(Y0, x * rnorm, j*32);
        }
    }
}

// y = gain * x / sqrt(max(sum(x**2), epsilon))
template <typename TY, typename TX>
__global__ void __launch_bounds__(32) l2_normalize_CK_8(
             TY*              Y,
          float*              S,
    const    TX* __restrict__ X,
    const float* __restrict__ G,
    const   int* __restrict__ Lut,
    float epsilon, int apply_gain)
{
    extern __shared__ int lut[]; // max(lut_size)


    int tid   = threadIdx.x;
    int idx_L = blockIdx.x;

    int4 lut_head = ((const int4*)Lut)[idx_L];
    // unpack lut header
    int lut_offset = lut_head.x;
    int lut_size   = lut_head.y;
    int idx_K      = lut_head.z;

    int k = idx_K*8 + (tid & 7);

    float gain = 1.0f;
    if (apply_gain) gain = G[k];

    Lut += lut_offset;
    #pragma unroll 1
    for (int i = tid; i < lut_size; i += 32)
        lut[i] = Lut[i] * 8 * 8;

    // sum_sqr_x = sum(x**2)
    float sum_sqr_x = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < lut_size; i++)
    {
        const TX* X0 = X + lut[i] + tid;

        float x0 = load(X0, 0*32);
        float x1 = load(X0, 1*32);

        sum_sqr_x += x0 * x0 + x1 * x1;
    }
    // reduce sum_sqr_x across the 4 rows of the warp
    sum_sqr_x += __shfl_xor(sum_sqr_x, 16);
    sum_sqr_x += __shfl_xor(sum_sqr_x, 8);

    store(S, sum_sqr_x, k, tid < 8);

    // rnorm = 1.0f / sqrt(max(sum_sqr_x, epsilon)) * gain
    float rnorm = rsqrtf(fmaxf(sum_sqr_x, epsilon)) * gain;

    // y = x * rnorm
    #pragma unroll 1
    for (int i = 0; i < lut_size; i++)
    {
        int block_offset = lut[i];

        const TX* X0 = X + block_offset + tid;
              TY* Y0 = Y + block_offset + tid;

        float x0 = load(X0, 0*32);
        float x1 = load(X0, 1*32);

        store(Y0, x0 * rnorm, 0*32);
        store(Y0, x1 * rnorm, 1*32);
    }
}


template <typename TY, typename TX>
bool L2NormalizeKCTRS(CUstream stream, TY* y, float* sum_sqr_x, const TX* x, const float* g, const int* lut, float epsilon, int K)
{
    dim3 grid(K, 1, 1);
    dim3 block(32, 1, 1);
    l2_normalize_KCTRS<TY,TX><<<grid, block, 0, stream>>>(y, sum_sqr_x, x, g, (const int2*)lut, epsilon, g != 0);
    return true; // TODO
}

template <typename TY, typename TX>
bool L2NormalizeCKTRS(CUstream stream, TY* y, float* sum_sqr_x, const TX* x, const float* g, const int* lut, float epsilon, int K, int TRS, int magic_TRS, int shift_TRS)
{
    dim3 grid(K, 1, 1);
    dim3 block(32, 1, 1);
    l2_normalize_CKTRS<TY,TX><<<grid, block, 0, stream>>>(y, sum_sqr_x, x, g, (const int4*)lut, epsilon, g != 0, TRS, magic_TRS, shift_TRS);
    return true; // TODO
}

template <typename TY, typename TX>
bool L2NormalizeCK(CUstream stream, TY* y, float* sum_sqr_x, const TX* x, const float* g, const int* lut, float epsilon, int K, int shared, int bsize)
{
    if (bsize == 32)
    {
        dim3 grid(K>>5, 1, 1);
        dim3 block(128, 1, 1);
        l2_normalize_CK_32<TY,TX><<<grid, block, shared+96*4, stream>>>(y, sum_sqr_x, x, g, lut, epsilon, g != 0);
    }
    else if (bsize == 16)
    {
        dim3 grid(K>>4, 1, 1);
        dim3 block(32, 1, 1);
        l2_normalize_CK_16<TY,TX><<<grid, block, shared, stream>>>(y, sum_sqr_x, x, g, lut, epsilon, g != 0);
    }
    else // if (bsize == 8)
    {
        dim3 grid(K>>3, 1, 1);
        dim3 block(32, 1, 1);
        l2_normalize_CK_8<TY,TX><<<grid, block, shared, stream>>>(y, sum_sqr_x, x, g, lut, epsilon, g != 0);
    }
    return true; // TODO
}


/////////////////////////////////////// Gradients ///////////////////////////////////////////


// sum_sqr_x = sum(x**2)
// norm_x    = sqrt(maximum(sum_sqr_x, epsilon))
// grad_x    = ( grad_y*g  +  x * (sum_sqr_x >= epsilon) * sum(-grad_y*g * x / norm_x**2) ) / norm_x
// grad_g    = sum(grad_y * l2_norm(x))
template <typename TY, typename TX>
__global__ void __launch_bounds__(32) l2_normalize_grad_KCTRS(
             TX*              DX,
          float*              DG,
    const    TY* __restrict__ DY,
    const    TX* __restrict__ X,
    const float* __restrict__ G,
    const float* __restrict__ S,
    const  int2* __restrict__ Lut,
    float epsilon, int apply_gain)
{
    int tid = threadIdx.x;
    int k   = blockIdx.x;

    int2 block_data = Lut[k];

    float gain = 1.0f;
    if (apply_gain) gain = G[k];

    int offset = block_data.x + tid; // block_F + idx_k * CTRS + tid
    int CTRS   = block_data.y;       // block_C * TRS

    const TX* X1 =  X + offset;
    const TX* X2 =  X + offset;
    const TY* DY1 = DY + offset;
    const TY* DY2 = DY + offset;

    DX += offset;

    float sum_sqr_x     = S[k];
    float max_sum_sqr_x = fmaxf(sum_sqr_x, epsilon);
    float norm_xi       = rsqrtf(max_sum_sqr_x);
    float norm_x2i      = 1.0f / max_sum_sqr_x;

    // sum(-d * x / norm_x**2)
    float red_val = 0.0f;
    float dg = 0.0f;
    for (int i = tid; i < CTRS; i += 32)
    {
        float dy = load(DY1);
        float  x = load(X1);
        DY1 += 32;
        X1  += 32;

        dg += dy * x * norm_xi;
        red_val += (-dy * x * gain) * norm_x2i;
    }
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1)
    {
        red_val += __shfl_xor(red_val, i);
        dg      += __shfl_xor(dg, i);
    }
    if (apply_gain && tid == 0)
        DG[k] = dg;

    red_val *= sum_sqr_x >= epsilon;

    for (int i = tid; i < CTRS; i += 32)
    {
        float dy = load(DY2);
        float  x = load(X2);

        float dx = dy * gain + x * red_val;

        store(DX, dx * norm_xi, 0);

        DY2 += 32;
        X2  += 32;
        DX  += 32;
    }
}

// sum_sqr_x = sum(x**2)
// norm_x    = sqrt(maximum(sum_sqr_x, epsilon))
// grad_x    = ( grad_y*g  +  x * (sum_sqr_x >= epsilon) * sum(-grad_y*g * x / norm_x**2) ) / norm_x
// grad_g    = sum(grad_y * l2_norm(x))
template <typename TY, typename TX>
__global__ void __launch_bounds__(32) l2_normalize_grad_CKTRS(
             TX*              DX,
          float*              DG,
    const    TY* __restrict__ DY,
    const    TX* __restrict__ X,
    const float* __restrict__ G,
    const float* __restrict__ S,
    const  int4* __restrict__ Lut,
    float epsilon, int apply_gain, int TRS, int magic_TRS, int shift_TRS)
{
    int tid = threadIdx.x;
    int k   = blockIdx.x;

    int4 block_data = Lut[k];

    float gain = 1.0f;
    if (apply_gain) gain = G[k];

    int idx_k   = block_data.x;
    int CTRS    = block_data.y;
    int KTRS    = block_data.z;
    int block_F = block_data.w;

    int offset_F = block_F + idx_k * TRS;

    const TX* X1 = X + offset_F;
    const TX* X2 = X + offset_F;
    const TY* DY1 = DY + offset_F;
    const TY* DY2 = DY + offset_F;

    DX += offset_F;

    float sum_sqr_x     = S[k];
    float max_sum_sqr_x = fmaxf(sum_sqr_x, epsilon);
    float norm_xi       = rsqrtf(max_sum_sqr_x);
    float norm_x2i      = 1.0f / max_sum_sqr_x;

    // sum(-d * x / norm_x**2)
    float red_val = 0.0f;
    float dg = 0.0f;
    for (int ctrs = tid; ctrs < CTRS; ctrs += 32)
    {
        //       c = i / TRS;
        //     trs = i % TRS;
        //  offset = c * KTRS + trs
        int      c = div16(ctrs, magic_TRS, shift_TRS);
        int    trs = mod16(ctrs, c, TRS);
        int offset = mad16(c, KTRS, trs);

        float  x = load( X1, offset);
        float dy = load(DY1, offset);

        dg += dy * x * norm_xi;

        red_val += (-dy * x * gain) * norm_x2i;
    }
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1)
    {
        red_val += __shfl_xor(red_val, i);
        dg      += __shfl_xor(dg, i);
    }
    if (apply_gain && tid == 0)
        DG[k] = dg;

    red_val *= sum_sqr_x >= epsilon;

    for (int ctrs = tid; ctrs < CTRS; ctrs += 32)
    {
        int      c = div16(ctrs, magic_TRS, shift_TRS);
        int    trs = mod16(ctrs, c, TRS);
        int offset = mad16(c, KTRS, trs);

        float  x = load( X2, offset);
        float dy = load(DY2, offset);

        float dx = dy * gain + x * red_val;

        store(DX, dx * norm_xi, offset);
    }
}

// sum_sqr_x = sum(x**2)
// norm_x    = sqrt(maximum(sum_sqr_x, epsilon))
// grad_x    = ( grad_y*g  +  x * (sum_sqr_x >= epsilon) * sum(-grad_y*g * x / norm_x**2) ) / norm_x
// grad_g    = sum(grad_y * l2_norm(x))
template <typename TY, typename TX>
__global__ void __launch_bounds__(128) l2_normalize_grad_CK_32(
             TX*              DX,
          float*              DG,
    const    TY* __restrict__ DY,
    const    TX* __restrict__ X,
    const float* __restrict__ G,
    const float* __restrict__ S,
    const   int* __restrict__ Lut,
    float epsilon, int apply_gain)
{
    extern __shared__ float fShare[]; // 96*2 + max(lut_size)
    extern __shared__   int iShare[]; // 96*2 + max(lut_size)

    float* redShare1 = &fShare[96*0];
    float* redShare2 = &fShare[96*1];
    int*   lutShare  = &iShare[96*2];

    int tid   = threadIdx.x;
    int idx_L = blockIdx.x;

    int4 lut_head = ((const int4*)Lut)[idx_L];
    // unpack lut header
    int lut_offset = lut_head.x;
    int lut_size   = lut_head.y;
    int idx_K      = lut_head.z;

    int k = idx_K*32 + (tid & 31);

    float gain = 1.0f;
    if (apply_gain) gain = G[k];

    float sum_sqr_x = S[k];

    Lut += lut_offset;
    #pragma unroll 1
    for (int i = tid; i < lut_size; i += 128)
        lutShare[i] = Lut[i] * 32 * 32;

    __syncthreads();

    float max_sum_sqr_x = fmaxf(sum_sqr_x, epsilon);
    float norm_xi       = rsqrtf(max_sum_sqr_x);
    float norm_x2i      = 1.0f / max_sum_sqr_x;

    float red_val = 0.0f;
    float dg = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < lut_size; i++)
    {
        int offset = lutShare[i] + tid;

        const TY* DY1 = DY + offset;
        const TX*  X1 =  X + offset;

        #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            float  x = load( X1, j*128);
            float dy = load(DY1, j*128);

            red_val += (-dy * gain * x) * norm_x2i;

            dg += dy * x * norm_xi;
        }
    }

    // reduce red_val across the 4 warps
    if (tid >= 32)
    {
        redShare1[tid-32] = red_val;
        redShare2[tid-32] = dg;
    }
    __syncthreads();

    if (tid < 32)
    {
        red_val += redShare1[tid] + redShare1[tid + 32] + redShare1[tid + 64];
        dg      += redShare2[tid] + redShare2[tid + 32] + redShare2[tid + 64];
        redShare1[tid] = red_val;

        if (apply_gain)
            DG[k] = dg;
    }
    __syncthreads();

    // get the final reduced value for all warps:
    red_val = redShare1[tid & 31];

    red_val *= sum_sqr_x >= epsilon;

    #pragma unroll 1
    for (int i = 0; i < lut_size; i++)
    {
        int offset = lutShare[i] + tid;

              TX* DX2 = DX + offset;
        const TY* DY2 = DY + offset;
        const TX* X2  = X  + offset;

        #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            float  x = load( X2, j*128);
            float dy = load(DY2, j*128);

            float dx = dy * gain + x * red_val;

            store(DX2, dx * norm_xi, j*128);
        }
    }
}

// sum_sqr_x = sum(x**2)
// norm_x    = sqrt(maximum(sum_sqr_x, epsilon))
// grad_x    = ( grad_y*g  +  x * (sum_sqr_x >= epsilon) * sum(-grad_y*g * x / norm_x**2) ) / norm_x
// grad_g    = sum(grad_y * l2_norm(x))
template <typename TY, typename TX>
__global__ void __launch_bounds__(32) l2_normalize_grad_CK_16(
             TX*              DX,
          float*              DG,
    const    TY* __restrict__ DY,
    const    TX* __restrict__ X,
    const float* __restrict__ G,
    const float* __restrict__ S,
    const   int* __restrict__ Lut,
    float epsilon, int apply_gain)
{
    extern __shared__ int lut[]; // max(lut_size)

    int tid   = threadIdx.x;
    int idx_L = blockIdx.x;

    int4 lut_head = ((const int4*)Lut)[idx_L];
    // unpack lut header
    int lut_offset = lut_head.x;
    int lut_size   = lut_head.y;
    int idx_K      = lut_head.z;

    int k = idx_K*16 + (tid & 15);

    float gain = 1.0f;
    if (apply_gain) gain = G[k];

    float sum_sqr_x = S[k];

    Lut += lut_offset;
    #pragma unroll 1
    for (int i = tid; i < lut_size; i += 32)
        lut[i] = Lut[i] * 16 * 16;

    float max_sum_sqr_x = fmaxf(sum_sqr_x, epsilon);
    float norm_xi       = rsqrtf(max_sum_sqr_x);
    float norm_x2i      = 1.0f / max_sum_sqr_x;

    float red_val = 0.0f;
    float dg = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < lut_size; i++)
    {
        int offset = lut[i] + tid;

        const TY* DY1 = DY + offset;
        const TX*  X1 =  X + offset;

        #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            float  x = load( X1, j*32);
            float dy = load(DY1, j*32);

            red_val += (-dy * gain * x) * norm_x2i;

            dg += dy * x * norm_xi;
        }
    }
    // reduce red_val,dg across the 4 rows of the warp
    red_val += __shfl_xor(red_val, 16);
    dg      += __shfl_xor(dg,      16);

    store(DG, dg, k, apply_gain && tid < 16);

    red_val *= sum_sqr_x >= epsilon;

    #pragma unroll 1
    for (int i = 0; i < lut_size; i++)
    {
        int offset = lut[i] + tid;

              TX* DX2 = DX + offset;
        const TY* DY2 = DY + offset;
        const TX* X2  = X  + offset;

        #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            float  x = load( X2, j*32);
            float dy = load(DY2, j*32);

            float dx = dy * gain + x * red_val;

            store(DX2, dx * norm_xi, j*32);
        }
    }
}

// sum_sqr_x = sum(x**2)
// norm_x    = sqrt(maximum(sum_sqr_x, epsilon))
// grad_x    = ( grad_y*g  +  x * (sum_sqr_x >= epsilon) * sum(-grad_y*g * x / norm_x**2) ) / norm_x
// grad_g    = sum(grad_y * l2_norm(x))
template <typename TY, typename TX>
__global__ void __launch_bounds__(32) l2_normalize_grad_CK_8(
             TX*              DX,
          float*              DG,
    const    TY* __restrict__ DY,
    const    TX* __restrict__ X,
    const float* __restrict__ G,
    const float* __restrict__ S,
    const   int* __restrict__ Lut,
    float epsilon, int apply_gain)
{
    extern __shared__ int lut[]; // max(lut_size)

    int tid   = threadIdx.x;
    int idx_L = blockIdx.x;

    int4 lut_head = ((const int4*)Lut)[idx_L];
    // unpack lut header
    int lut_offset = lut_head.x;
    int lut_size   = lut_head.y;
    int idx_K      = lut_head.z;

    int k = idx_K*8 + (tid & 7);

    float gain = 1.0f;
    if (apply_gain) gain = G[k];

    float sum_sqr_x = S[k];

    Lut += lut_offset;
    #pragma unroll 1
    for (int i = tid; i < lut_size; i += 32)
        lut[i] = Lut[i] * 8 * 8;

    float max_sum_sqr_x = fmaxf(sum_sqr_x, epsilon);
    float norm_xi       = rsqrtf(max_sum_sqr_x);
    float norm_x2i      = 1.0f / max_sum_sqr_x;

    float red_val = 0.0f;
    float dg = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < lut_size; i++)
    {
        int offset = lut[i] + tid;

        const TY* DY1 = DY + offset;
        const TX*  X1 =  X + offset;

        #pragma unroll
        for (int j = 0; j < 2; j++)
        {
            float  x = load( X1, j*32);
            float dy = load(DY1, j*32);

            red_val += (-dy * gain * x) * norm_x2i;

            dg += dy * x * norm_xi;
        }
    }
    // reduce red_val,dg across the 4 rows of the warp
    red_val += __shfl_xor(red_val, 16);
    dg      += __shfl_xor(dg,      16);
    red_val += __shfl_xor(red_val, 8);
    dg      += __shfl_xor(dg,      8);

    store(DG, dg, k, apply_gain && tid < 8);

    red_val *= sum_sqr_x >= epsilon;

    #pragma unroll 1
    for (int i = 0; i < lut_size; i++)
    {
        int offset = lut[i] + tid;

              TX* DX2 = DX + offset;
        const TY* DY2 = DY + offset;
        const TX* X2  = X  + offset;

        #pragma unroll
        for (int j = 0; j < 2; j++)
        {
            float  x = load( X2, j*32);
            float dy = load(DY2, j*32);

            float dx = dy * gain + x * red_val;

            store(DX2, dx * norm_xi, j*32);
        }
    }
}

template <typename TY, typename TX>
bool L2NormalizeGradKCTRS(CUstream stream, TX* grad_x, float* grad_g, const TY* grad_y, const TX* x, const float* g, const float* sum_sqr_x_p, const int* lut, float epsilon, int K)
{
    dim3 grid(K, 1, 1);
    dim3 block(32, 1, 1);
    l2_normalize_grad_KCTRS<TY,TX><<<grid, block, 0, stream>>>(grad_x, grad_g, grad_y, x, g, sum_sqr_x_p, (const int2*)lut, epsilon, g != 0);
    return true; // TODO
}

template <typename TY, typename TX>
bool L2NormalizeGradCKTRS(CUstream stream, TX* grad_x, float* grad_g, const TY* grad_y, const TX* x, const float* g, const float* sum_sqr_x_p, const int* lut, float epsilon, int K, int TRS, int magic_TRS, int shift_TRS)
{
    dim3 grid(K, 1, 1);
    dim3 block(32, 1, 1);
    l2_normalize_grad_CKTRS<TY,TX><<<grid, block, 0, stream>>>(grad_x, grad_g, grad_y, x, g, sum_sqr_x_p, (const int4*)lut, epsilon, g != 0, TRS, magic_TRS, shift_TRS);
    return true; // TODO
}

template <typename TY, typename TX>
bool L2NormalizeGradCK   (CUstream stream, TX* grad_x, float* grad_g, const TY* grad_y, const TX* x, const float* g, const float* sum_sqr_x_p, const int* lut, float epsilon, int K, int shared, int bsize)
{
    if (bsize == 32)
    {
        dim3 grid(K>>5, 1, 1);
        dim3 block(128, 1, 1);
        l2_normalize_grad_CK_32<TY,TX><<<grid, block, shared+96*2*4, stream>>>(grad_x, grad_g, grad_y, x, g, sum_sqr_x_p, lut, epsilon, g != 0);
    }
    else if (bsize == 16)
    {
        dim3 grid(K>>4, 1, 1);
        dim3 block(32, 1, 1);
        l2_normalize_grad_CK_16<TY,TX><<<grid, block, shared, stream>>>(grad_x, grad_g, grad_y, x, g, sum_sqr_x_p, lut, epsilon, g != 0);
    }
    else // if (bsize == 8)
    {
        dim3 grid(K>>3, 1, 1);
        dim3 block(32, 1, 1);
        l2_normalize_grad_CK_8<TY,TX><<<grid, block, shared, stream>>>(grad_x, grad_g, grad_y, x, g, sum_sqr_x_p, lut, epsilon, g != 0);
    }
    return true; // TODO
}


template bool L2NormalizeKCTRS<float, float>(CUstream stream, float* y, float* sum_sqr_x, const float* x, const float* g, const int* lut, float epsilon, int K);
template bool L2NormalizeCKTRS<float, float>(CUstream stream, float* y, float* sum_sqr_x, const float* x, const float* g, const int* lut, float epsilon, int K, int TRS, int magic_TRS, int shift_TRS);
template bool L2NormalizeCK   <float, float>(CUstream stream, float* y, float* sum_sqr_x, const float* x, const float* g, const int* lut, float epsilon, int K, int shared, int bsize);

template bool L2NormalizeGradKCTRS<float, float>(CUstream stream, float* grad_x, float* grad_g, const float* grad_y, const float* x, const float* g, const float* sum_sqr_x_p, const int* lut, float epsilon, int K);
template bool L2NormalizeGradCKTRS<float, float>(CUstream stream, float* grad_x, float* grad_g, const float* grad_y, const float* x, const float* g, const float* sum_sqr_x_p, const int* lut, float epsilon, int K, int TRS, int magic_TRS, int shift_TRS);
template bool L2NormalizeGradCK   <float, float>(CUstream stream, float* grad_x, float* grad_g, const float* grad_y, const float* x, const float* g, const float* sum_sqr_x_p, const int* lut, float epsilon, int K, int shared, int bsize);


template bool L2NormalizeKCTRS<ehalf, ehalf>(CUstream stream, ehalf* y, float* sum_sqr_x, const ehalf* x, const float* g, const int* lut, float epsilon, int K);
template bool L2NormalizeCKTRS<ehalf, ehalf>(CUstream stream, ehalf* y, float* sum_sqr_x, const ehalf* x, const float* g, const int* lut, float epsilon, int K, int TRS, int magic_TRS, int shift_TRS);
template bool L2NormalizeCK   <ehalf, ehalf>(CUstream stream, ehalf* y, float* sum_sqr_x, const ehalf* x, const float* g, const int* lut, float epsilon, int K, int shared, int bsize);

template bool L2NormalizeGradKCTRS<ehalf, ehalf>(CUstream stream, ehalf* grad_x, float* grad_g, const ehalf* grad_y, const ehalf* x, const float* g, const float* sum_sqr_x_p, const int* lut, float epsilon, int K);
template bool L2NormalizeGradCKTRS<ehalf, ehalf>(CUstream stream, ehalf* grad_x, float* grad_g, const ehalf* grad_y, const ehalf* x, const float* g, const float* sum_sqr_x_p, const int* lut, float epsilon, int K, int TRS, int magic_TRS, int shift_TRS);
template bool L2NormalizeGradCK   <ehalf, ehalf>(CUstream stream, ehalf* grad_x, float* grad_g, const ehalf* grad_y, const ehalf* x, const float* g, const float* sum_sqr_x_p, const int* lut, float epsilon, int K, int shared, int bsize);


template bool L2NormalizeKCTRS<ehalf, float>(CUstream stream, ehalf* y, float* sum_sqr_x, const float* x, const float* g, const int* lut, float epsilon, int K);
template bool L2NormalizeCKTRS<ehalf, float>(CUstream stream, ehalf* y, float* sum_sqr_x, const float* x, const float* g, const int* lut, float epsilon, int K, int TRS, int magic_TRS, int shift_TRS);
template bool L2NormalizeCK   <ehalf, float>(CUstream stream, ehalf* y, float* sum_sqr_x, const float* x, const float* g, const int* lut, float epsilon, int K, int shared, int bsize);

template bool L2NormalizeGradKCTRS<ehalf, float>(CUstream stream, float* grad_x, float* grad_g, const ehalf* grad_y, const float* x, const float* g, const float* sum_sqr_x_p, const int* lut, float epsilon, int K);
template bool L2NormalizeGradCKTRS<ehalf, float>(CUstream stream, float* grad_x, float* grad_g, const ehalf* grad_y, const float* x, const float* g, const float* sum_sqr_x_p, const int* lut, float epsilon, int K, int TRS, int magic_TRS, int shift_TRS);
template bool L2NormalizeGradCK   <ehalf, float>(CUstream stream, float* grad_x, float* grad_g, const ehalf* grad_y, const float* x, const float* g, const float* sum_sqr_x_p, const int* lut, float epsilon, int K, int shared, int bsize);


template bool L2NormalizeKCTRS<bhalf, bhalf>(CUstream stream, bhalf* y, float* sum_sqr_x, const bhalf* x, const float* g, const int* lut, float epsilon, int K);
template bool L2NormalizeCKTRS<bhalf, bhalf>(CUstream stream, bhalf* y, float* sum_sqr_x, const bhalf* x, const float* g, const int* lut, float epsilon, int K, int TRS, int magic_TRS, int shift_TRS);
template bool L2NormalizeCK   <bhalf, bhalf>(CUstream stream, bhalf* y, float* sum_sqr_x, const bhalf* x, const float* g, const int* lut, float epsilon, int K, int shared, int bsize);

template bool L2NormalizeGradKCTRS<bhalf, bhalf>(CUstream stream, bhalf* grad_x, float* grad_g, const bhalf* grad_y, const bhalf* x, const float* g, const float* sum_sqr_x_p, const int* lut, float epsilon, int K);
template bool L2NormalizeGradCKTRS<bhalf, bhalf>(CUstream stream, bhalf* grad_x, float* grad_g, const bhalf* grad_y, const bhalf* x, const float* g, const float* sum_sqr_x_p, const int* lut, float epsilon, int K, int TRS, int magic_TRS, int shift_TRS);
template bool L2NormalizeGradCK   <bhalf, bhalf>(CUstream stream, bhalf* grad_x, float* grad_g, const bhalf* grad_y, const bhalf* x, const float* g, const float* sum_sqr_x_p, const int* lut, float epsilon, int K, int shared, int bsize);


template bool L2NormalizeKCTRS<bhalf, float>(CUstream stream, bhalf* y, float* sum_sqr_x, const float* x, const float* g, const int* lut, float epsilon, int K);
template bool L2NormalizeCKTRS<bhalf, float>(CUstream stream, bhalf* y, float* sum_sqr_x, const float* x, const float* g, const int* lut, float epsilon, int K, int TRS, int magic_TRS, int shift_TRS);
template bool L2NormalizeCK   <bhalf, float>(CUstream stream, bhalf* y, float* sum_sqr_x, const float* x, const float* g, const int* lut, float epsilon, int K, int shared, int bsize);

template bool L2NormalizeGradKCTRS<bhalf, float>(CUstream stream, float* grad_x, float* grad_g, const bhalf* grad_y, const float* x, const float* g, const float* sum_sqr_x_p, const int* lut, float epsilon, int K);
template bool L2NormalizeGradCKTRS<bhalf, float>(CUstream stream, float* grad_x, float* grad_g, const bhalf* grad_y, const float* x, const float* g, const float* sum_sqr_x_p, const int* lut, float epsilon, int K, int TRS, int magic_TRS, int shift_TRS);
template bool L2NormalizeGradCK   <bhalf, float>(CUstream stream, float* grad_x, float* grad_g, const bhalf* grad_y, const float* x, const float* g, const float* sum_sqr_x_p, const int* lut, float epsilon, int K, int shared, int bsize);


#endif // GOOGLE_CUDA
