
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

// #include <stdio.h>
#include "ew_op_gpu.h"
#include <stdio.h>

template <bool Fprop, typename TW, typename TX, typename TY>
__global__ void __launch_bounds__(32) gemm_blocksparse_08x64x08x8_xprop(
    const  int2* __restrict__ Lut,
    const    TW* __restrict__ W,
    const    TX* __restrict__ X,
    TY* Y, int* Lock, int locks, int N /* N is in units of groups of 8 elements each (N/8) */)
{
    if (Fprop)
        asm(".shared .align 16 .b32 share[576];" ::); // 576 =  8*8 + 64*8
    else
        asm(".shared .align 16 .b32 share[608];" ::); // 608 = 12*8 + 64*8


    extern __shared__ int2 Lut2_s[];
    int2* Lut2s = &Lut2_s[Fprop ? 576/2 : 608/2];

    int tid   = threadIdx.x;
    int idx_N = blockIdx.x;
    int idx_L = blockIdx.y;

    int4 lut_head = ((const int4*)Lut)[idx_L];

    int tid7  = tid  & 7;
    int tid8  = tid >> 3;
    int tid16 = tid & 16;

    int readXs = ((tid >> 1) & 7) << 4;
    int readWs =  (tid  & 1)      << 4;

    // second half of warp starts 4 rows down
    readXs += tid16 << 6; // 64*4*4
    readWs += tid16 << 3; //  8*4*4

    int storXs = (tid8*64 + tid7*4) << 2;
    int storWs;
    if (Fprop)
        storWs = tid << 3;
    else
    {
        // Transpose weights on store to shared
        // Avoid bank conflicts by shifting writes over by 8 every 2 rows (+tid3*8)
        int tid3 = tid &  3;
        int tid4 = tid >> 2;
        storWs = (tid3*8*2 + tid4 + tid3*8) << 2;
        readWs += tid16 << 2; // shift over 8 floats every 2 rows, second half of warp starts 4 rows down
    }

    int  n = idx_N*8 + tid7;
    bool bn = n < N;

    int offsetX = (tid8*N + n)*8*2;

    // unpack lut header
    int lut_offset = lut_head.x;
    int lut_size   = lut_head.y;
    int idx_K      = lut_head.z;
    int idx_Lock   = lut_head.w;

    int N8 = N*8*8*2; // 8 lines, 8 elements per index, two bytes per element

    // prefetch the lut data into shared
    Lut += lut_offset;
    #pragma unroll 1
    for (int i = tid; i < lut_size; i += 32)
    {
        int2 entry = Lut[i];
        entry.x *= N8;    // 8 lines of N per block
        entry.y *= 64*2;  // 64 entries of W per block, 2 bytes each
        Lut2s[i] = entry;
    }

    // zero accumulation registers
    float regY[4][8];
    for (int w = 0; w < 4; w++)
        for (int x = 0; x < 8; x++)
            regY[w][x] = 0;

    // loop over each lut entry to compute a gemm block
    int i = 0;
    #pragma unroll 1
    do
    {
        int2 entry = Lut2s[i++];

        entry.x += offsetX;
        entry.y += tid*4;

        const TW* W0;
        const TX* X0;
        const TX* X4;
        // Simplify pointer arithmatic by letting compiler assume all offsets fit in 32 bits.
        asm("{\n\t"
            ".reg .u64 x0, x4, w0;\n\t"
            "mov.b64 w0, {%5, 0};\n\t"
            "mov.b64 x0, {%6, 0};\n\t"
            "mov.b64 x4, {%7, 0};\n\t"
            "add.u64 %0, w0, %3;\n\t"
            "add.u64 %1, x0, %4;\n\t"
            "add.u64 %2, x4, %4;\n\t"
            "}" : "=l"(W0),"=l"(X0),"=l"(X4) : "l"(W), "l"(X), "r"(entry.y), "r"(entry.x), "r"(entry.x + N*4*8*2) );

        // Fetch 8 rows at a time from W and X
        float2 w0 = load(W0, 0);
        float8 x0 = load(X0, 0, bn);
        float8 x4 = load(X4, 0, bn);

        // store to shared.
        if (Fprop)
            st_shared_v2(storWs + 64*8*4, w0);
        else
        {
            // transpose the shared store of W
            st_shared_v1(storWs + 0*4 + 64*8*4, w0.x);
            st_shared_v1(storWs + 8*4 + 64*8*4, w0.y);
        }
        st_shared_v4(storXs + (0*64 + 0*32)*4, x0.a);
        st_shared_v4(storXs + (0*64 + 1*32)*4, x0.b);
        st_shared_v4(storXs + (4*64 + 0*32)*4, x4.a);
        st_shared_v4(storXs + (4*64 + 1*32)*4, x4.b);

        // computes an 8x64x8 gemm tile with 4x8 register blocking
        float regW[4];
        float regX[8];
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            // fetch outer product data
            ld_shared_v4(readWs + ( 8*j + 64*8 + (Fprop ? 0 : (j>>1)*8))*4, regW ); // shift over 8 floats every 2 rows
            ld_shared_v4(readXs + (64*j +  0)*4, &regX[0] );
            ld_shared_v4(readXs + (64*j + 32)*4, &regX[4] );

            // accumulate outer product
            for (int w = 0; w < 4; w++)
                for (int x = 0; x < 8; x++)
                    regY[w][x] += regW[w] * regX[x];
        }

    } while (i < lut_size);

    int tidN = (tid >> 1) & 7;
    int tidK = tid  & 1;
    int tidk = tid >> 4;

    bool t16 = tid16 != 0;

    float outY[2][8];
    for (int w = 0; w < 2; w++)
    {
        for (int x = 0; x < 8; x++)
        {
            float swap = t16 ? regY[2*w + 0][x] : regY[2*w + 1][x];
            outY[w][x] = t16 ? regY[2*w + 1][x] : regY[2*w + 0][x];
            outY[w][x] += __shfl_xor(swap, 16);
        }
    }

    n  = idx_N*64/8 + tidN;
    bn = n < N;

    Y += (idx_K*8 + tidK*4 + tidk)*N + n;

    if (idx_Lock == 0)
    {
        // no lock needed just write out the results
        store(Y, *(float8*)outY[0], N*0, bn);
        store(Y, *(float8*)outY[1], N*2, bn);
    }
    else
    {
        int offsetL = idx_N*locks + idx_Lock - 1;
        Lock += offsetL;

        // Critial Section
        if (tid == 0)
            while (atomicCAS(Lock, 0, 1) != 0);

        int offsetC = locks*gridDim.x;
        int* Count = Lock + offsetC;
        int  count = *Count;

        if (count == 0)
        {
            if (tid == 0)
                *Count = 1;

            // first block to get here just writes out to init the memory
            store(Y, *(float8*)outY[0], N*0, bn);
            store(Y, *(float8*)outY[1], N*2, bn);

            __threadfence();

            if (tid == 0)
                atomicExch(Lock, 0);
            // End Critial Section
        }
        else
        {
            // subsequent blocks must accumulate
            float8 y0 = load_c(Y, N*0, bn);
            float8 y2 = load_c(Y, N*2, bn);

            y0 = ew_add(y0, *(float8*)outY[0]);
            y2 = ew_add(y2, *(float8*)outY[1]);

            store(Y, y0, N*0, bn);
            store(Y, y2, N*2, bn);

            __threadfence();

            if (tid == 0)
                atomicExch(Lock, 0);
            // End Critial Section
        }
    }
}


template <bool Fprop, typename TW, typename TX, typename TY>
__global__ void __launch_bounds__(32) gemm_blocksparse_08x64x08x4_xprop(
    const  int2* __restrict__ Lut,
    const    TW* __restrict__ W,
    const    TX* __restrict__ X,
    TY* Y, int* Lock, int locks, int N)
{
    __shared__ float shrX1[64*8];
    __shared__ float shrW1[Fprop ? 8*8 : 12*8];
   float2* shrW2 = (float2*)shrW1;
   float2* shrX2 = (float2*)shrX1;
   float4* shrX4 = (float4*)shrX1;

    extern __shared__ int2 Lut2_s[];

    int tid   = threadIdx.x;
    int idx_N = blockIdx.x;
    int idx_L = blockIdx.y;

    int4 lut_head = ((const int4*)Lut)[idx_L];

    // used to zero accumulation registers
    shrX1[tid] = 0.0f;

    int tid15 = tid & 15;
    int tid16 = tid >> 4;

    float2* storW2;
    float*  storW1;
    if (Fprop)
        storW2 = &shrW2[tid];
    else
    {
        int tid3 = tid &  3;
        int tid4 = tid >> 2;
        // Transpose weights on store to shared
        // Avoid bank conflicts by shifting writes over by 8 every 2 rows (+tid3*8)
        storW1 = &shrW1[tid3*16 + tid4 + tid3*8];
    }
    float4* storX4 = &shrX4[tid];
    // float2* readX2 = &shrX2[tid15];
    // float2* readW2 = &shrW2[tid16];

    float2* readX2 = &shrX2[tid >> 1];
    float2* readW2 = &shrW2[tid & 1];

    int N4 = N  >> 2;
    int N2 = N4 << 1;
    int N8 = N4 << 3;

    int  n4 = idx_N*16 + tid15;
    bool bn = n4 < N4;

    const TX* X0 = X + tid16*N4 + n4;
    const TX* X2 = X0 + N2;
    const TX* X4 = X2 + N2;
    const TX* X6 = X4 + N2;
    const TW* W0 = W + tid;

    // unpack lut header
    int lut_offset = lut_head.x;
    int lut_size   = lut_head.y;
    int idx_K      = lut_head.z;
    int idx_Lock   = lut_head.w;

    // prefetch the lut data into shared
    Lut += lut_offset;
    #pragma unroll 1
    for (int i = tid; i < lut_size; i += 32)
    {
        int2 entry = Lut[i];
        entry.x *= N8;
        entry.y *= 32;
        Lut2_s[i] = entry;
    }
    float regX[4];
    float regW[4];
    float regY[4][4];

    // zero accumulation registers
    for (int w = 0; w < 4; w++)
        *(float4*)regY[w] = shrX4[w];

    // loop over each lut entry to compute a gemm block
    #pragma unroll 1
    for (int i = 0; i < lut_size; i++)
    {
        int2 entry = Lut2_s[i];

        // Fetch 8 rows at a time from W and X
        TW w0;
        TX x0, x2, x4, x6;
        w0 = W0[entry.y];
        if (bn)
        {
            x0 = X0[entry.x];
            x2 = X2[entry.x];
            x4 = X4[entry.x];
            x6 = X6[entry.x];
        }
        // Convert to float if needed and store to shared.
        if (Fprop)
            storW2[0] = to_float(w0);
        else
        {
            // transpose the shared store of W
            float2 w2 = to_float(w0);
            storW1[0] = w2.x;
            storW1[8] = w2.y;
        }
        storX4[0*16] = to_float(x0);
        storX4[2*16] = to_float(x2);
        storX4[4*16] = to_float(x4);
        storX4[6*16] = to_float(x6);

        // computes an 8x64x8 gemm block
        #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            // fetch outer product data
            *(float2*)&regX[0] = readX2[32*j +  0];
            *(float2*)&regW[0] = readW2[ 4*j +  0 + (Fprop ? 0 : (j>>1)*4)]; // shift over 8 floats every 2 rows
            *(float2*)&regX[2] = readX2[32*j + 16];
            *(float2*)&regW[2] = readW2[ 4*j +  2 + (Fprop ? 0 : (j>>1)*4)];
            // accumulate outer product
            for (int w = 0; w < 4; w++)
                for (int x = 0; x < 4; x++)
                    regY[w][x] += regW[w] * regX[x];
        }
    }
    tid   = threadIdx.x;
    idx_N = blockIdx.x;
    tid15 = tid >> 1;
    tid16 = tid  & 1;
    N2    = N >> 1;

    int n = idx_N*32 + tid15;
    int yi[4];
    yi[0] = (idx_K*8 + tid16*2)*N2 + n;
    yi[1] = yi[0] + N2;
    yi[2] = yi[0] + N2*4;
    yi[3] = yi[2] + N2;

    bool bn0  = n+0  < N2;
    bool bn16 = n+16 < N2;

    if (idx_Lock == 0)
    {
        // no lock needed just write out the results
        for (int i = 0; i < 4; i++)
        {
            store(Y, *(float2*)&regY[i][0], yi[i]+0,  bn0 );
            store(Y, *(float2*)&regY[i][2], yi[i]+16, bn16);
        }
    }
    else
    {
        int offsetL = idx_N*locks + idx_Lock - 1;
        Lock += offsetL;

        // Critial Section
        if (tid == 0)
            while (atomicCAS(Lock, 0, 1) != 0);

        int offsetC = locks*gridDim.x;
        int* Count = Lock + offsetC;
        int  count = *Count;

        if (count == 0)
        {
            if (tid == 0)
                *Count = 1;

            // first block to get here just writes out to init the memory
            for (int i = 0; i < 4; i++)
            {
                store(Y, *(float2*)&regY[i][0], yi[i]+0,  bn0 );
                store(Y, *(float2*)&regY[i][2], yi[i]+16, bn16);
            }

            __threadfence();

            if (tid == 0)
                atomicExch(Lock, 0);
            // End Critial Section
        }
        else
        {
            // subsequent blocks must accumulate
            float2 y[8];
            for (int i = 0; i < 4; i++)
            {
                y[i + 0] = load_c(Y, yi[i]+0,  bn0 );
                y[i + 4] = load_c(Y, yi[i]+16, bn16);

                y[i + 0].x += regY[i][0];
                y[i + 0].y += regY[i][1];
                y[i + 4].x += regY[i][2];
                y[i + 4].y += regY[i][3];
            }

            for (int i = 0; i < 4; i++)
            {
                store(Y, y[i + 0], yi[i]+0,  bn0 );
                store(Y, y[i + 4], yi[i]+16, bn16);
            }

            __threadfence();

            if (tid == 0)
                atomicExch(Lock, 0);
            // End Critial Section
        }
    }
}

template <bool Fprop, typename TW, typename TX, typename TY>
__global__ void __launch_bounds__(64) gemm_blocksparse_16x64x16x8_xprop(
    const  int2* __restrict__ Lut,
    const    TW* __restrict__ W,
    const    TX* __restrict__ X,
    TY* Y, int* Lock, int locks, int N /* N is in units of groups of 8 elements each (N/8) */)
{
    if (Fprop)
        asm(".shared .align 16 .b32 share[1280];" ::); // 1280 = 16*16 + 64*16
    else
        asm(".shared .align 16 .b32 share[1312];" ::); // 1312 = 16*16 + 64*16 + 4*8

    extern __shared__ int2 Lut2_s[];
    int2* Lut2s = &Lut2_s[Fprop ? 1280/2 : 1312/2];

    int tid   = threadIdx.x;
    int idx_N = blockIdx.x;
    int idx_L = blockIdx.y;

    int4 lut_head = ((const int4*)Lut)[idx_L];

    int tid7  = tid  & 7;
    int tid8  = tid >> 3;
    int tid16 = tid & 16;
    int tid32 = tid & 32;

    int readXs = ((tid >> 1) & 7) << 4;
    int readWs =  (tid  & 1)      << 4;

    // split weights in half over two warps
    readWs += tid32;

    // second half of warp starts 8 rows down
    readXs += tid16 << 7; // 64*8*4
    readWs += tid16 << 5; // 16*8*4

    int storXs = (tid8*64 + tid7*4) << 2;
    int storWs;
    if (Fprop)
        storWs = tid << 4;
    else
    {
        // Transpose weights on store to shared
        // Avoid bank conflicts by shifting writes over by 8 every 4 rows (+tid3*8)
        int tid3 = tid &  3;
        int tid4 = tid >> 2;
        storWs = (tid3*16*4 + tid4 + tid3*8) << 2;
        readWs += tid16 << 2; // shift over 8 floats every 4 rows, second half of warp starts 8 rows down
    }

    int  n = idx_N*8 + tid7;
    bool bn = n < N;

    int offsetX = (tid8*N + n)*8*2;

    // unpack lut header
    int lut_offset = lut_head.x;
    int lut_size   = lut_head.y;
    int idx_K      = lut_head.z;
    int idx_Lock   = lut_head.w;

    int N16 = N*16*8*2; // 16 lines, 8 elements per index, two bytes per element

    // prefetch the lut data into shared
    Lut += lut_offset;
    #pragma unroll 1
    for (int i = tid; i < lut_size; i += 64)
    {
        int2 entry = Lut[i];
        entry.x *= N16;    // 16 lines of N per block
        entry.y *= 256*2;  // 256 entries of W per block, 2 bytes each
        Lut2s[i] = entry;
    }
    __syncthreads();

    // Force compiler to fully compute these prior to loop
    asm("mov.b32 %0, %0;" : "+r"(storXs)  : );
    asm("mov.b32 %0, %0;" : "+r"(storWs)  : );
    asm("mov.b32 %0, %0;" : "+r"(readXs)  : );
    asm("mov.b32 %0, %0;" : "+r"(readWs)  : );
    asm("mov.b32 %0, %0;" : "+r"(offsetX) : );

    // zero accumulation registers
    float regY[4][8];
    for (int w = 0; w < 4; w++)
        for (int x = 0; x < 8; x++)
            regY[w][x] = 0;

    // loop over each lut entry to compute a gemm block
    int i = 0;
    #pragma unroll 1
    do
    {
        int2 entry = Lut2s[i++];

        entry.x += offsetX;
        entry.y += tid*4*2;

        const TW* W0;
        const TX* X0;
        const TX* X8;
        // Simplify pointer arithmetic by letting compiler assume all offsets fit in 32 bits.
        asm("{\n\t"
            ".reg .u64 x0, x4, w0;\n\t"
            "mov.b64 w0, {%5, 0};\n\t"
            "mov.b64 x0, {%6, 0};\n\t"
            "mov.b64 x4, {%7, 0};\n\t"
            "add.u64 %0, w0, %3;\n\t"
            "add.u64 %1, x0, %4;\n\t"
            "add.u64 %2, x4, %4;\n\t"
            "}" : "=l"(W0),"=l"(X0),"=l"(X8) : "l"(W), "l"(X), "r"(entry.y), "r"(entry.x), "r"(entry.x + N*8*8*2) );

        // Fetch 8 rows at a time from W and X
        float4 w0 = load(W0, 0);
        float8 x0 = load(X0, 0, bn);
        float8 x8 = load(X8, 0, bn);

        __syncthreads();

        // store to shared.
        if (Fprop)
            st_shared_v4(storWs + 64*16*4, w0);
        else
        {
            // transpose the shared store of W
            st_shared_v1(storWs + 0*16*4 + 64*16*4, w0.x);
            st_shared_v1(storWs + 1*16*4 + 64*16*4, w0.y);
            st_shared_v1(storWs + 2*16*4 + 64*16*4, w0.z);
            st_shared_v1(storWs + 3*16*4 + 64*16*4, w0.w);
        }
        // avoid bank conflicts by splitting 8 floats in two groups of 4
        // We can rejoin them on the output
        st_shared_v4(storXs + (0*64 + 0*32)*4, x0.a);
        st_shared_v4(storXs + (0*64 + 1*32)*4, x0.b);
        st_shared_v4(storXs + (8*64 + 0*32)*4, x8.a);
        st_shared_v4(storXs + (8*64 + 1*32)*4, x8.b);

        __syncthreads();

        // computes a 16x64x16 gemm tile with 4x8 register blocking
        float regW[4];
        float regX[8];
        #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            // fetch outer product data
            ld_shared_v4(readWs + (16*j + 64*16 + (Fprop ? 0 : (j>>2)*8))*4, regW ); // shift over 8 floats every 4 rows
            ld_shared_v4(readXs + (64*j +  0)*4, &regX[0] );
            ld_shared_v4(readXs + (64*j + 32)*4, &regX[4] );

            // accumulate outer product
            for (int w = 0; w < 4; w++)
                for (int x = 0; x < 8; x++)
                    regY[w][x] += regW[w] * regX[x];

        }

    } while (i < lut_size);

    tid16 = tid & 16;
    tid32 = tid & 32;

    int tidN = (tid >> 1) & 7;
    int tidK = (tid & 1) + (tid32 >> 4);
    int tidk = tid16 >> 4;

    bool t16 = tid16 != 0;

    float outY[2][8];
    for (int w = 0; w < 2; w++)
    {
        for (int x = 0; x < 8; x++)
        {
            float swap = t16 ? regY[2*w + 0][x] : regY[2*w + 1][x];
            outY[w][x] = t16 ? regY[2*w + 1][x] : regY[2*w + 0][x];
            outY[w][x] += __shfl_xor(swap, 16);
        }
    }

    n  = idx_N*64/8 + tidN;
    bn = n < N;

    Y += (idx_K*16 + tidK*4 + tidk)*N + n;

    if (idx_Lock == 0)
    {
        // no lock needed just write out the results
        store(Y, *(float8*)outY[0], N*0, bn);
        store(Y, *(float8*)outY[1], N*2, bn);
    }
    else
    {
        int offsetL = idx_N*locks + idx_Lock - 1;
        Lock += offsetL;

        // Critial Section
        if (tid == 0)
            while (atomicCAS(Lock, 0, 1) != 0);
        __syncthreads();

        int offsetC = locks*gridDim.x;
        int* Count = Lock + offsetC;
        int  count = *Count;

        __syncthreads(); // all threads have the count now and safe to update

        if (count == 0)
        {
            if (tid == 0)
                *Count = 1;

            // first block to get here just writes out to init the memory
            store(Y, *(float8*)outY[0], N*0, bn);
            store(Y, *(float8*)outY[1], N*2, bn);

            __threadfence();
            __syncthreads();

            if (tid == 0)
                atomicExch(Lock, 0);
            // End Critial Section
        }
        else
        {
            // subsequent blocks must accumulate
            float8 y0 = load_c(Y, N*0, bn);
            float8 y2 = load_c(Y, N*2, bn);

            y0 = ew_add(y0, *(float8*)outY[0]);
            y2 = ew_add(y2, *(float8*)outY[1]);

            store(Y, y0, N*0, bn);
            store(Y, y2, N*2, bn);

            __threadfence();
            __syncthreads();

            if (tid == 0)
                atomicExch(Lock, 0);
            // End Critial Section
        }
    }
}

template <bool fprop, typename TW, typename TX, typename TY>
__global__ void __launch_bounds__(64) gemm_blocksparse_16x64x16x4_xprop(
    const  int2* __restrict__ Lut,
    const    TW* __restrict__ W,
    const    TX* __restrict__ X,
    TY* Y, int* Lock, int locks, int N)
{
    __shared__ float shrX1[64*16];
    __shared__ float shrW1[fprop ? 16*16 : 16*16 + 4*8];
   float2* shrW2 = (float2*)shrW1;
   float2* shrX2 = (float2*)shrX1;
   float4* shrW4 = (float4*)shrW1;
   float4* shrX4 = (float4*)shrX1;

    extern __shared__ int2 Lut2_s[];

    int tid   = threadIdx.x;
    int idx_N = blockIdx.x;
    int idx_L = blockIdx.y;

    int4 lut_head = ((const int4*)Lut)[idx_L];

    // used to zero accumulation registers
    shrX1[tid] = 0.0f;

    int tid15 = tid & 15;
    int tid16 = tid >> 4;

    float4* storW4;
    float*  storW1;
    if (fprop)
        storW4 = &shrW4[tid];
    else
    {
        int tid3 = tid &  3;
        int tid4 = tid >> 2;
        // Transpose weights on store to shared
        // Avoid bank conflicts by shifting writes over by 8 every 4 rows (+tid3*8)
        storW1 = &shrW1[tid3*16*4 + tid4 + tid3*8];
    }
    float4* storX4 = &shrX4[tid];
    float2* readX2 = &shrX2[(tid >> 1) & 15];
    float2* readW2 = &shrW2[((tid & 32) >> 4) | (tid & 1)];

    int N4  = N  >> 2;
    int N16 = N4 << 4;

    int  n4 = idx_N*16 + tid15;
    bool bn = n4 < N4;

    const TX* X4[4];
    X4[0] = X + tid16*N4 + n4;
    for (int i = 1; i < 8; i++)
        X4[i] = X4[i-1] + N;

    const TW* W0 = W + tid;

    // unpack lut header
    int lut_offset = lut_head.x;
    int lut_size   = lut_head.y;
    int idx_K      = lut_head.z;
    int idx_Lock   = lut_head.w;

    // prefetch the lut data into shared
    Lut += lut_offset;
    #pragma unroll 1
    for (int i = tid; i < lut_size; i += 64)
    {
        int2 entry = Lut[i];
        entry.x *= N16;
        entry.y *= 64;
        Lut2_s[i] = entry;
    }
    __syncthreads();

    float regX[4];
    float regW[4];
    float regY[4][4];

    // zero accumulation registers
    for (int w = 0; w < 4; w++)
        *(float4*)regY[w] = shrX4[w];

    // loop over each lut entry to compute a gemm block
    #pragma unroll 1
    for (int i = 0; i < lut_size; i++)
    {
        int2 entry = Lut2_s[i];

        // Fetch 16 rows at a time from W and X
        TW w0 = W0[entry.y];
        TX x0[4];
        if (bn)
        {
            for (int i = 0; i < 4; i++)
                x0[i] = X4[i][entry.x];
        }
        float4 w4 = to_float(w0);
        float4 x4[4];
        for (int i = 0; i < 4; i++)
            x4[i] = to_float(x0[i]);

        // Convert to float if needed and store to shared.
        __syncthreads();

        if (fprop)
            storW4[0] = w4;
        else
        {
            // transpose the shared store of W
            storW1[0*16] = w4.x;
            storW1[1*16] = w4.y;
            storW1[2*16] = w4.z;
            storW1[3*16] = w4.w;
        }
        for (int i = 0; i < 4; i++)
            storX4[i*4*16] = x4[i];

        __syncthreads();

        // computes a 16x64x16 gemm block
        #pragma unroll
        for (int j = 0; j < 16; j++)
        {
            // fetch outer product data
            *(float2*)&regX[0] = readX2[32*j +  0];
            *(float2*)&regW[0] = readW2[ 8*j +  0 + (fprop ? 0 : (j>>2)*4)]; // shift over 8 floats every 4 rows
            *(float2*)&regX[2] = readX2[32*j + 16];
            *(float2*)&regW[2] = readW2[ 8*j +  4 + (fprop ? 0 : (j>>2)*4)];
            // accumulate outer product
            for (int w = 0; w < 4; w++)
                for (int x = 0; x < 4; x++)
                    regY[w][x] += regW[w] * regX[x];
        }
    }
    tid   = threadIdx.x;
    idx_N = blockIdx.x;
    tid15 = (tid >> 1) & 15;
    tid16 = ((tid & 32) >> 4) | (tid & 1);
    int N2 = N >> 1;

    //printf("%2d %2d %2d\n", tid, tid15, tid16);

    int n = idx_N*32 + tid15;
    int yi[4];
    yi[0] = (idx_K*16 + tid16*2)*N2 + n;
    yi[1] = yi[0] + N2;
    yi[2] = yi[0] + N2*8;
    yi[3] = yi[2] + N2;

    bool bn0  = n+0  < N2;
    bool bn16 = n+16 < N2;

    if (idx_Lock == 0)
    {
        // no lock needed just write out the results
        for (int i = 0; i < 4; i++)
        {
            store(Y, *(float2*)&regY[i][0], yi[i]+0,  bn0 );
            store(Y, *(float2*)&regY[i][2], yi[i]+16, bn16);
        }
    }
    else
    {
        int offsetL = idx_N*locks + idx_Lock - 1;
        Lock += offsetL;

        // Critial Section
        if (tid == 0)
            while (atomicCAS(Lock, 0, 1) != 0);
        __syncthreads();

        int offsetC = locks*gridDim.x;
        int* Count = Lock + offsetC;
        int  count = *Count;

        __syncthreads(); // all threads have the count now and safe to update

        if (count == 0)
        {
            if (tid == 0)
                *Count = 1;

            // first block to get here just writes out to init the memory
            for (int i = 0; i < 4; i++)
            {
                store(Y, *(float2*)&regY[i][0], yi[i]+0,  bn0 );
                store(Y, *(float2*)&regY[i][2], yi[i]+16, bn16);
            }
            __threadfence();
            __syncthreads();
            if (tid == 0)
                atomicExch(Lock, 0);
            // End Critial Section
        }
        else
        {
            // subsequent blocks must accumulate
            float2 y[8];
            for (int i = 0; i < 4; i++)
            {
                y[i + 0] = load_c(Y, yi[i]+0,  bn0 );
                y[i + 4] = load_c(Y, yi[i]+16, bn16);

                y[i + 0].x += regY[i][0];
                y[i + 0].y += regY[i][1];
                y[i + 4].x += regY[i][2];
                y[i + 4].y += regY[i][3];
            }
            for (int i = 0; i < 4; i++)
            {
                store(Y, y[i + 0], yi[i]+0,  bn0 );
                store(Y, y[i + 4], yi[i]+16, bn16);
            }
            __threadfence();
            __syncthreads();
            if (tid == 0)
                atomicExch(Lock, 0);
            // End Critial Section
        }
    }
}

template <typename TX, typename TE, typename TU>
__global__ void __launch_bounds__(32) gemm_blocksparse_08x64x08x8_updat(
    struct plist8<TX> X, struct plist8<TE> E,
    const  int2* __restrict__ Lut,
    TU* U,
    int params8, int N, int loops, float alpha, float beta)
{
    // add padding for bank-conflict-free stores
    //__shared__ float2 shrU2[64*8 + 4*8]; // add padding for bank-conflict-free stores

    asm(".shared .align 16 .b32 share[1088];" ::); // 1088 = (64*8 + 4*8)*2

    extern __shared__ float2 shrU2[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int2 lut_head = Lut[bid];

    int tid7 = tid  & 7;
    int tid8 = tid >> 3;

    int tid1 = tid &  1;
    int tid2 = (tid >> 1) & 1;
    int tid4 = tid >> 2;

    // avoid bank conflicts when writing transpose (+tid7*4)
    int storeS = tid7*8*8 + tid8 + tid7*4;

    // 4 threads read blocks of 8x8 each shifted over by 4
    int iread  = tid4*8*8 + tid4*4;
    int readXs = iread + tid2*2;
    int readEs = iread + tid1*2 + 64*8 + 4*8;

    int idx_X = lut_head.x;
    int idx_E = lut_head.y;

    int offsetX = (idx_X*8 + tid8)*N + tid7*8;
    int offsetE = (idx_E*8 + tid8)*N + tid7*8;

    int N4 = N;

    // This keeps all prior logic outside of the loops.
    asm("shl.b32 %0, %0, 3;" : "+r"(N4)      : );
    asm("shl.b32 %0, %0, 2;" : "+r"(storeS)  : );
    asm("shl.b32 %0, %0, 2;" : "+r"(readXs)  : );
    asm("shl.b32 %0, %0, 2;" : "+r"(readEs)  : );
    asm("shl.b32 %0, %0, 1;" : "+r"(offsetX) : );
    asm("shl.b32 %0, %0, 1;" : "+r"(offsetE) : );

    float regU[4][4]; //[x][e]
    for (int x = 0; x < 4; x++)
        for (int e = 0; e < 4; e++)
            regU[x][e] = 0;

    int p = 0;
    #pragma unroll 1
    do
    {
        const TX* X0;
        const TE* E0;
        asm("{\n\t"
            ".reg .u64 X, E, offsetX, offsetE;\n\t"
# if __CUDA_ARCH__ >= 700
            "ld.param.u64 X, [%2 + 0x160];\n\t" // WARNING: hard coded param offset.
            "ld.param.u64 E, [%2 + 0x1a0];\n\t" // WARNING: hard coded param offset.
# else
            "ld.param.u64 X, [%2 + 0x140];\n\t" // WARNING: hard coded param offset.
            "ld.param.u64 E, [%2 + 0x180];\n\t" // WARNING: hard coded param offset.
# endif
            "cvta.to.global.u64 X, X;\n\t"
            "cvta.to.global.u64 E, E;\n\t"
            "mov.b64 offsetX, {%3, 0};\n\t"
            "mov.b64 offsetE, {%4, 0};\n\t"
            "add.u64 %0, X, offsetX;\n\t"
            "add.u64 %1, E, offsetE;\n\t"
            "}" : "=l"(X0), "=l"(E0) : "r"(p), "r"(offsetX), "r"(offsetE));
        p += 8;

        int n = (tid & 7) << 3;
        int loop = 0;
        #pragma unroll 1
        do
        {
            const TX* X4;
            const TE* E4;
            asm("{\n\t"
                ".reg .u64 N4;\n\t"
                "mov.b64 N4, {%4, 0};\n\t"
                "add.u64 %0, N4, %2;\n\t"
                "add.u64 %1, N4, %3;\n\t"
                "}" : "=l"(X4),"=l"(E4) : "l"(X0), "l"(E0), "r"(N4) );

            TX x0, x4;
            TE e0, e4;
            ew_zero(x0); ew_zero(x4);
            ew_zero(e0); ew_zero(e4);
            if (n < N)
            {
                x0 = __ldg(X0);
                x4 = __ldg(X4);
                e0 = __ldg(E0);
                e4 = __ldg(E4);
            }

            // Convert to float if needed and store to shared as transpose.
            float8 fx0 = to_float(x0);
            float8 fx4 = to_float(x4);

            // advance pointer by 64*2
            asm ("add.u64 %0, %0, 128;" : "+l"(X0));
            n += 64;

            st_shared_v1(storeS + (0*8 + 0 + (64*8 + 4*8)*0)*4, fx0.a.x);
            st_shared_v1(storeS + (1*8 + 0 + (64*8 + 4*8)*0)*4, fx0.a.y);
            st_shared_v1(storeS + (2*8 + 0 + (64*8 + 4*8)*0)*4, fx0.a.z);
            st_shared_v1(storeS + (3*8 + 0 + (64*8 + 4*8)*0)*4, fx0.a.w);
            st_shared_v1(storeS + (4*8 + 0 + (64*8 + 4*8)*0)*4, fx0.b.x);
            st_shared_v1(storeS + (5*8 + 0 + (64*8 + 4*8)*0)*4, fx0.b.y);
            st_shared_v1(storeS + (6*8 + 0 + (64*8 + 4*8)*0)*4, fx0.b.z);
            st_shared_v1(storeS + (7*8 + 0 + (64*8 + 4*8)*0)*4, fx0.b.w);

            st_shared_v1(storeS + (0*8 + 4 + (64*8 + 4*8)*0)*4, fx4.a.x);
            st_shared_v1(storeS + (1*8 + 4 + (64*8 + 4*8)*0)*4, fx4.a.y);
            st_shared_v1(storeS + (2*8 + 4 + (64*8 + 4*8)*0)*4, fx4.a.z);
            st_shared_v1(storeS + (3*8 + 4 + (64*8 + 4*8)*0)*4, fx4.a.w);
            st_shared_v1(storeS + (4*8 + 4 + (64*8 + 4*8)*0)*4, fx4.b.x);
            st_shared_v1(storeS + (5*8 + 4 + (64*8 + 4*8)*0)*4, fx4.b.y);
            st_shared_v1(storeS + (6*8 + 4 + (64*8 + 4*8)*0)*4, fx4.b.z);
            st_shared_v1(storeS + (7*8 + 4 + (64*8 + 4*8)*0)*4, fx4.b.w);

            float8 fe0 = to_float(e0);
            float8 fe4 = to_float(e4);

            // advance pointer by 64*2
            asm ("add.u64 %0, %0, 128;" : "+l"(E0));

            st_shared_v1(storeS + (0*8 + 0 + (64*8 + 4*8)*1)*4, fe0.a.x);
            st_shared_v1(storeS + (1*8 + 0 + (64*8 + 4*8)*1)*4, fe0.a.y);
            st_shared_v1(storeS + (2*8 + 0 + (64*8 + 4*8)*1)*4, fe0.a.z);
            st_shared_v1(storeS + (3*8 + 0 + (64*8 + 4*8)*1)*4, fe0.a.w);
            st_shared_v1(storeS + (4*8 + 0 + (64*8 + 4*8)*1)*4, fe0.b.x);
            st_shared_v1(storeS + (5*8 + 0 + (64*8 + 4*8)*1)*4, fe0.b.y);
            st_shared_v1(storeS + (6*8 + 0 + (64*8 + 4*8)*1)*4, fe0.b.z);
            st_shared_v1(storeS + (7*8 + 0 + (64*8 + 4*8)*1)*4, fe0.b.w);

            st_shared_v1(storeS + (0*8 + 4 + (64*8 + 4*8)*1)*4, fe4.a.x);
            st_shared_v1(storeS + (1*8 + 4 + (64*8 + 4*8)*1)*4, fe4.a.y);
            st_shared_v1(storeS + (2*8 + 4 + (64*8 + 4*8)*1)*4, fe4.a.z);
            st_shared_v1(storeS + (3*8 + 4 + (64*8 + 4*8)*1)*4, fe4.a.w);
            st_shared_v1(storeS + (4*8 + 4 + (64*8 + 4*8)*1)*4, fe4.b.x);
            st_shared_v1(storeS + (5*8 + 4 + (64*8 + 4*8)*1)*4, fe4.b.y);
            st_shared_v1(storeS + (6*8 + 4 + (64*8 + 4*8)*1)*4, fe4.b.z);
            st_shared_v1(storeS + (7*8 + 4 + (64*8 + 4*8)*1)*4, fe4.b.w);

            float regX[4];
            float regE[4];

            #pragma unroll
            for (int j = 0; j < 8; j++)
            {
                // fetch outer product data
                ld_shared_v2(readXs + (8*j + 0)*4, &regX[0] );
                ld_shared_v2(readEs + (8*j + 0)*4, &regE[0] );
                ld_shared_v2(readXs + (8*j + 4)*4, &regX[2] );
                ld_shared_v2(readEs + (8*j + 4)*4, &regE[2] );

                for (int x = 0; x < 4; x++)
                    for (int e = 0; e < 4; e++)
                        regU[x][e] += regX[x] * regE[e];
            }
            loop++;

        } while (loop < loops);

    } while (p < params8);

    tid = threadIdx.x;
    bid = blockIdx.x;

    int offset = bid*32 + tid;
    TU t2; ew_zero(t2);
    if (beta != 0.0f)
        t2 = U[offset];

    tid1 = tid & 1;
    tid2 = (tid >> 1)  & 1;
    tid4 = (tid & -4) << 3;

    float2* storU2 = &shrU2[tid4 + tid2*4*2 + tid1];

    storU2[0*4 + 0] = *(float2*)&regU[0][0];
    storU2[0*4 + 2] = *(float2*)&regU[0][2];
    storU2[1*4 + 0] = *(float2*)&regU[1][0];
    storU2[1*4 + 2] = *(float2*)&regU[1][2];
    storU2[4*4 + 0] = *(float2*)&regU[2][0];
    storU2[4*4 + 2] = *(float2*)&regU[2][2];
    storU2[5*4 + 0] = *(float2*)&regU[3][0];
    storU2[5*4 + 2] = *(float2*)&regU[3][2];

    float2* readU2 = &shrU2[tid];

    float2 u[8];
    for (int i = 0; i < 8; i++)
        u[i] = readU2[i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
        {
            u[i].x = u[i].x + u[i+j].x;
            u[i].y = u[i].y + u[i+j].y;
        }

    float2 u2 = *(float2*)u;
    float2 b2 = to_float(t2);
    u2.x = alpha*u2.x + beta*b2.x;
    u2.y = alpha*u2.y + beta*b2.y;
    store(U, u2, offset);
}

template <typename TX, typename TE, typename TU>
__global__ void __launch_bounds__(32) gemm_blocksparse_08x64x08x4_updat(
    struct plist8<TX> X, struct plist8<TE> E,
    const  int2* __restrict__ Lut,
    TU* U,
    int params8, int N, int loops, float alpha, float beta)
{
    __shared__ float shrX1[64*8 + 2*16]; // add padding for bank-conflict-free stores
    __shared__ float shrE1[64*8 + 2*16];
   float2* shrU2 = (float2*)shrX1;
   float2* shrX2 = (float2*)shrX1;
   float2* shrE2 = (float2*)shrE1;

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int2 lut_head = Lut[bid];

    int tid15 = tid & 15;
    int tid16 = tid >> 4;

    int tid1 = tid &  1;
    int tid2 = (tid >> 1) & 1;
    int tid4 = tid >> 2;

    // avoid bank conflicts when writing transpose (+tid15*2)
    int istore = tid15*8*4 + tid16 + tid15*2;

    float* storX = &shrX1[istore];
    float* storE = &shrE1[istore];

    // 4 threads read blocks of 8x8 each shifted over by 4 (2 shifts of 2 from store)
    int iread = tid4*4*8 + tid4*2;

    float2* readX2 = &shrX2[iread + tid2];
    float2* readE2 = &shrE2[iread + tid1];

    int N4 = N  >> 2;
    int N2 = N4 << 1;

    int idx_X = lut_head.x;
    int idx_E = lut_head.y;

    int offsetX = (idx_X*8 + tid16)*N4;
    int offsetE = (idx_E*8 + tid16)*N4;

    float regX[4];
    float regE[4];
    float regU[4][4]; //[x][e]

    for (int x = 0; x < 4; x++)
        for (int e = 0; e < 4; e++)
            regU[x][e] = 0;

    int p = 0;
    #pragma unroll 1
    do
    {
        int n = tid15;

        const TX* X0;
        const TE* E0;

        asm("{\n\t"
            ".reg .u64 X, E;\n\t"
# if __CUDA_ARCH__ >= 700
            "ld.param.u64 X, [%2 + 0x160];\n\t" // WARNING: hard coded param offset.
            "ld.param.u64 E, [%2 + 0x1a0];\n\t" // WARNING: hard coded param offset.
# else
            "ld.param.u64 X, [%2 + 0x140];\n\t" // WARNING: hard coded param offset.
            "ld.param.u64 E, [%2 + 0x180];\n\t" // WARNING: hard coded param offset.
# endif
            "cvta.to.global.u64 %0, X;\n\t"
            "cvta.to.global.u64 %1, E;\n\t"
            "}" : "=l"(X0), "=l"(E0) : "r"(p));
        p += 8;

        X0 += offsetX;
        E0 += offsetE;

        const TX* X2 = X0 + N2;
        const TX* X4 = X2 + N2;
        const TX* X6 = X4 + N2;

        const TE* E2 = E0 + N2;
        const TE* E4 = E2 + N2;
        const TE* E6 = E4 + N2;

        #pragma unroll 1
        for (int i = 0; i < loops; i++)
        {
            bool bn = n < N4;

            TX x0, x2, x4, x6;
            TE e0, e2, e4, e6;
            ew_zero(x0); ew_zero(x2); ew_zero(x4); ew_zero(x6);
            ew_zero(e0); ew_zero(e2); ew_zero(e4); ew_zero(e6);
            if (bn)
            {
                x0 = __ldg(X0+n); x2 = __ldg(X2+n); x4 = __ldg(X4+n); x6 = __ldg(X6+n);
                e0 = __ldg(E0+n); e2 = __ldg(E2+n); e4 = __ldg(E4+n); e6 = __ldg(E6+n);
            }
            n += 16;

            // Convert to float if needed and store to shared as transpose.
            float4 fx0 = to_float(x0);
            float4 fx2 = to_float(x2);
            float4 fx4 = to_float(x4);
            float4 fx6 = to_float(x6);
            storX[0*8 + 0] = fx0.x;
            storX[1*8 + 0] = fx0.y;
            storX[2*8 + 0] = fx0.z;
            storX[3*8 + 0] = fx0.w;

            storX[0*8 + 2] = fx2.x;
            storX[1*8 + 2] = fx2.y;
            storX[2*8 + 2] = fx2.z;
            storX[3*8 + 2] = fx2.w;

            storX[0*8 + 4] = fx4.x;
            storX[1*8 + 4] = fx4.y;
            storX[2*8 + 4] = fx4.z;
            storX[3*8 + 4] = fx4.w;

            storX[0*8 + 6] = fx6.x;
            storX[1*8 + 6] = fx6.y;
            storX[2*8 + 6] = fx6.z;
            storX[3*8 + 6] = fx6.w;

            float4 fe0 = to_float(e0);
            float4 fe2 = to_float(e2);
            float4 fe4 = to_float(e4);
            float4 fe6 = to_float(e6);
            storE[0*8 + 0] = fe0.x;
            storE[1*8 + 0] = fe0.y;
            storE[2*8 + 0] = fe0.z;
            storE[3*8 + 0] = fe0.w;

            storE[0*8 + 2] = fe2.x;
            storE[1*8 + 2] = fe2.y;
            storE[2*8 + 2] = fe2.z;
            storE[3*8 + 2] = fe2.w;

            storE[0*8 + 4] = fe4.x;
            storE[1*8 + 4] = fe4.y;
            storE[2*8 + 4] = fe4.z;
            storE[3*8 + 4] = fe4.w;

            storE[0*8 + 6] = fe6.x;
            storE[1*8 + 6] = fe6.y;
            storE[2*8 + 6] = fe6.z;
            storE[3*8 + 6] = fe6.w;

            #pragma unroll
            for (int j = 0; j < 8; j++)
            {
                // shift over 2 floats every 4 rows
                *(float2*)&regX[0] = readX2[4*j + 0 + (j>>2)];
                *(float2*)&regE[0] = readE2[4*j + 0 + (j>>2)];
                *(float2*)&regX[2] = readX2[4*j + 2 + (j>>2)];
                *(float2*)&regE[2] = readE2[4*j + 2 + (j>>2)];

                for (int x = 0; x < 4; x++)
                    for (int e = 0; e < 4; e++)
                        regU[x][e] += regX[x] * regE[e];
            }
        }
    } while (p < params8);

    tid = threadIdx.x;
    bid = blockIdx.x;

    int offset = bid*32 + tid;
    TU t2; ew_zero(t2);
    if (beta != 0.0f)
        t2 = U[offset];

    tid1 = tid & 1;
    tid2 = (tid >> 1)  & 1;
    tid4 = (tid & -4) << 3;

    float2* storU2 = &shrU2[tid4 + tid2*4*2 + tid1];

    storU2[0*4 + 0] = *(float2*)&regU[0][0];
    storU2[0*4 + 2] = *(float2*)&regU[0][2];
    storU2[1*4 + 0] = *(float2*)&regU[1][0];
    storU2[1*4 + 2] = *(float2*)&regU[1][2];
    storU2[4*4 + 0] = *(float2*)&regU[2][0];
    storU2[4*4 + 2] = *(float2*)&regU[2][2];
    storU2[5*4 + 0] = *(float2*)&regU[3][0];
    storU2[5*4 + 2] = *(float2*)&regU[3][2];

    float2* readU2 = &shrU2[tid];

    float2 u[8];
    for (int i = 0; i < 8; i++)
        u[i] = readU2[i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
        {
            u[i].x = u[i].x + u[i+j].x;
            u[i].y = u[i].y + u[i+j].y;
        }

    float2 u2 = *(float2*)u;
    float2 b2 = to_float(t2);
    u2.x = alpha*u2.x + beta*b2.x;
    u2.y = alpha*u2.y + beta*b2.y;
    store(U, u2, offset);
}

template <typename TX, typename TE, typename TU>
__global__ void __launch_bounds__(64) gemm_blocksparse_16x64x16x4_updat(
    struct plist8<TX> X, struct plist8<TE> E,
    const  int2* __restrict__ Lut,
    TU* U,
    int params8, int N, int loops, float alpha, float beta)
{
    __shared__ float shrX1[64*16 + 2*16]; // add padding for bank-conflict-free stores
    __shared__ float shrE1[64*16 + 2*16];
    float2* shrU2 = (float2*)shrX1;
    float2* shrX2 = (float2*)shrX1;
    float2* shrE2 = (float2*)shrE1;
    float4* shrX4 = (float4*)shrX1;
    float4* shrU4 = (float4*)shrX1;

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int2 lut_head = Lut[bid];

    // used to zero accumulation registers
    shrX1[tid] = 0.0f;

    int tid15 = tid & 15;
    int tid16 = tid >> 4;

    int tid3 = (tid >> 1) & 3;
    int tid4 = ((tid & 8) >> 2) | (tid & 1);

    // avoid bank conflicts when writing transpose (+tid15*2)
    int istore = tid15*16*4 + tid16 + tid15*2;

    float* storX = &shrX1[istore];
    float* storE = &shrE1[istore];

    // 16 threads read blocks of 16x16 each shifted over by 8 (4 shifts of 2 from store)
    int iread = tid16*8*16 + tid16*4;

    float2* readX2 = &shrX2[iread + tid4];
    float2* readE2 = &shrE2[iread + tid3];

    int N4 = N  >> 2;

    int idx_X = lut_head.x;
    int idx_E = lut_head.y;

    int offsetX = (idx_X*16 + tid16)*N4;
    int offsetE = (idx_E*16 + tid16)*N4;

    float regX[4];
    float regE[4];
    float regU[4][4]; //[x][e]

    __syncthreads();

    for (int x = 0; x < 4; x++)
        *(float4*)regU[x] = shrX4[x];

    #pragma unroll 1
    for (int p = 0; p < params8; p += 8)
    {
        int n = tid15;

        const TX* X0;
        const TE* E0;

        asm("{\n\t"
            ".reg .u64 X, E;\n\t"
# if __CUDA_ARCH__ >= 700
            "ld.param.u64 X, [%2 + 0x160];\n\t" // WARNING: hard coded param offset.
            "ld.param.u64 E, [%2 + 0x1a0];\n\t" // WARNING: hard coded param offset.
# else
            "ld.param.u64 X, [%2 + 0x140];\n\t" // WARNING: hard coded param offset.
            "ld.param.u64 E, [%2 + 0x180];\n\t" // WARNING: hard coded param offset.
# endif
            "cvta.to.global.u64 %0, X;\n\t"
            "cvta.to.global.u64 %1, E;\n\t"
            "}" : "=l"(X0), "=l"(E0) : "r"(p));

        X0 += offsetX;
        E0 += offsetE;

        const TX* X1 = X0 + N;
        const TX* X2 = X1 + N;
        const TX* X3 = X2 + N;

        const TE* E1 = E0 + N;
        const TE* E2 = E1 + N;
        const TE* E3 = E2 + N;

        #pragma unroll 1
        for (int i = 0; i < loops; i++)
        {
            TX x0, x1, x2, x3;
            TE e0, e1, e2, e3;
            ew_zero(x0); ew_zero(x1); ew_zero(x2); ew_zero(x3);
            ew_zero(e0); ew_zero(e1); ew_zero(e2); ew_zero(e3);
            if (n < N4)
            {
                x0 = __ldg(X0+n); x1 = __ldg(X1+n); x2 = __ldg(X2+n); x3 = __ldg(X3+n);
                e0 = __ldg(E0+n); e1 = __ldg(E1+n); e2 = __ldg(E2+n); e3 = __ldg(E3+n);
            }
            n += 16;

            __syncthreads();

            // Convert to float if needed and store to shared as transpose.
            float4 fx0 = to_float(x0);
            float4 fx1 = to_float(x1);
            float4 fx2 = to_float(x2);
            float4 fx3 = to_float(x3);
            float4 fe0 = to_float(e0);
            float4 fe1 = to_float(e1);
            float4 fe2 = to_float(e2);
            float4 fe3 = to_float(e3);

            storX[0*16 + 0*4] = fx0.x;
            storX[1*16 + 0*4] = fx0.y;
            storX[2*16 + 0*4] = fx0.z;
            storX[3*16 + 0*4] = fx0.w;

            storX[0*16 + 1*4] = fx1.x;
            storX[1*16 + 1*4] = fx1.y;
            storX[2*16 + 1*4] = fx1.z;
            storX[3*16 + 1*4] = fx1.w;

            storX[0*16 + 2*4] = fx2.x;
            storX[1*16 + 2*4] = fx2.y;
            storX[2*16 + 2*4] = fx2.z;
            storX[3*16 + 2*4] = fx2.w;

            storX[0*16 + 3*4] = fx3.x;
            storX[1*16 + 3*4] = fx3.y;
            storX[2*16 + 3*4] = fx3.z;
            storX[3*16 + 3*4] = fx3.w;

            storE[0*16 + 0*4] = fe0.x;
            storE[1*16 + 0*4] = fe0.y;
            storE[2*16 + 0*4] = fe0.z;
            storE[3*16 + 0*4] = fe0.w;

            storE[0*16 + 1*4] = fe1.x;
            storE[1*16 + 1*4] = fe1.y;
            storE[2*16 + 1*4] = fe1.z;
            storE[3*16 + 1*4] = fe1.w;

            storE[0*16 + 2*4] = fe2.x;
            storE[1*16 + 2*4] = fe2.y;
            storE[2*16 + 2*4] = fe2.z;
            storE[3*16 + 2*4] = fe2.w;

            storE[0*16 + 3*4] = fe3.x;
            storE[1*16 + 3*4] = fe3.y;
            storE[2*16 + 3*4] = fe3.z;
            storE[3*16 + 3*4] = fe3.w;

            __syncthreads();

            #pragma unroll
            for (int j = 0; j < 16; j++)
            {
                // shift over 2 floats every 4 rows
                *(float2*)&regX[0] = readX2[8*j + 0 + (j>>2)];
                *(float2*)&regE[0] = readE2[8*j + 0 + (j>>2)];
                *(float2*)&regX[2] = readX2[8*j + 4 + (j>>2)];
                *(float2*)&regE[2] = readE2[8*j + 4 + (j>>2)];
                for (int x = 0; x < 4; x++)
                    for (int e = 0; e < 4; e++)
                        regU[x][e] += regX[x] * regE[e];
            }
        }
    }
    tid = threadIdx.x;
    bid = blockIdx.x;

    int offset4 = bid*64 + tid;
    TU t4; ew_zero(t4);
    if (beta != 0.0f)
        t4 = U[offset4];

    tid3  = (tid >> 1) & 3;
    tid4  = ((tid & 8) >> 2) | (tid & 1);
    tid16 = (tid & -16) << 3;

    float2* storU2 = &shrU2[tid16 + tid4*8*2 + tid3];

    __syncthreads();

    storU2[0*8 + 0] = *(float2*)&regU[0][0];
    storU2[0*8 + 4] = *(float2*)&regU[0][2];
    storU2[1*8 + 0] = *(float2*)&regU[1][0];
    storU2[1*8 + 4] = *(float2*)&regU[1][2];
    storU2[8*8 + 0] = *(float2*)&regU[2][0];
    storU2[8*8 + 4] = *(float2*)&regU[2][2];
    storU2[9*8 + 0] = *(float2*)&regU[3][0];
    storU2[9*8 + 4] = *(float2*)&regU[3][2];

    __syncthreads();

    float4* readU4 = &shrU4[tid];

    float4 u[4];
    for (int i = 0; i < 4; i++)
        u[i] = readU4[i*64];

    // Tree reduce
    for (int j = 2; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
        {
            u[i].x = u[i].x + u[i+j].x;
            u[i].y = u[i].y + u[i+j].y;
            u[i].z = u[i].z + u[i+j].z;
            u[i].w = u[i].w + u[i+j].w;
        }

    float4 u4 = *(float4*)u;
    float4 b4 = to_float(t4);
    u4.x = alpha*u4.x + beta*b4.x;
    u4.y = alpha*u4.y + beta*b4.y;
    u4.z = alpha*u4.z + beta*b4.z;
    u4.w = alpha*u4.w + beta*b4.w;
    store(U, u4, offset4);
}

template <typename TX, typename TE, typename TU>
__global__ void __launch_bounds__(64) gemm_blocksparse_16x64x16x8_updat(
    struct plist8<TX> X, struct plist8<TE> E,
    const  int2* __restrict__ Lut,
    TU* U,
    int params8, int N, int loops, float alpha, float beta)
{
    //asm(".shared .align 16 .b32 share[2560];" ::); // 2560 = (64*16 + 4*64)*2
    __shared__ float shrU[(64*16 + 4*64)*2];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int2 lut_head = Lut[bid];

    int tid7 = tid  & 7;
    int tid8 = tid >> 3;

    // avoid bank conflicts when writing transpose (+tid7*4)
    int storeS = tid7*16*8 + tid8 + tid7*4;

    // 8 threads read blocks of 16x16 each shifted over by 4
    int iread  = tid8*8*16 + tid8*4;
    int readXs = iread + (tid & 1)*4;
    int readEs = iread + ((tid >> 1) & 3)*4;
    readXs <<= 2;
    readEs <<= 2;

    int idx_X = lut_head.x;
    int idx_E = lut_head.y;

    int offsetX = (idx_X*16 + tid8)*N + tid7*8;
    int offsetE = (idx_E*16 + tid8)*N + tid7*8;
    int N8 = N << 4;

    // This keeps all prior logic outside of the loops.
    asm("shl.b32 %0, %0, 2;" : "+r"(storeS)  : );
    asm("shl.b32 %0, %0, 1;" : "+r"(offsetX) : );
    asm("shl.b32 %0, %0, 1;" : "+r"(offsetE) : );

    float regU[8][4]; //[x][e]
    for (int x = 0; x < 8; x++)
        for (int e = 0; e < 4; e++)
            regU[x][e] = 0;

    int p = 0;
    #pragma unroll 1
    do
    {
        const TX* X0;
        const TE* E0;
        asm("{\n\t"
            ".reg .u64 X, E, offsetX, offsetE;\n\t"
# if __CUDA_ARCH__ >= 700
            "ld.param.u64 X, [%2 + 0x160];\n\t" // WARNING: hard coded param offset.
            "ld.param.u64 E, [%2 + 0x1a0];\n\t" // WARNING: hard coded param offset.
# else
            "ld.param.u64 X, [%2 + 0x140];\n\t" // WARNING: hard coded param offset.
            "ld.param.u64 E, [%2 + 0x180];\n\t" // WARNING: hard coded param offset.
# endif
            "cvta.to.global.u64 X, X;\n\t"
            "cvta.to.global.u64 E, E;\n\t"
            "mov.b64 offsetX, {%3, 0};\n\t"
            "mov.b64 offsetE, {%4, 0};\n\t"
            "add.u64 %0, X, offsetX;\n\t"
            "add.u64 %1, E, offsetE;\n\t"
            "}" : "=l"(X0), "=l"(E0) : "r"(p), "r"(offsetX), "r"(offsetE));
        p += 8;

        int n = (tid & 7) << 3;
        int loop = 0;
        #pragma unroll 1
        do
        {
            const TX* X8;
            const TE* E8;
            asm("{\n\t"
                ".reg .u64 N8;\n\t"
                "mov.b64 N8, {%4, 0};\n\t"
                "add.u64 %0, N8, %2;\n\t"
                "add.u64 %1, N8, %3;\n\t"
                "}" : "=l"(X8),"=l"(E8) : "l"(X0), "l"(E0), "r"(N8) );

            TX x0, x8;
            TE e0, e8;
            ew_zero(x0); ew_zero(x8);
            ew_zero(e0); ew_zero(e8);
            if (n < N)
            {
                x0 = __ldg(X0);
                x8 = __ldg(X8);
                e0 = __ldg(E0);
                e8 = __ldg(E8);
            }
            __syncthreads();

            // Convert to float if needed and store to shared as transpose.
            float8 fx0 = to_float(x0);
            float8 fx8 = to_float(x8);

            // advance pointer by 64*2
            asm ("add.u64 %0, %0, 128;" : "+l"(X0));
            n += 64;

            st_shared_v1(storeS + (0*16 + 0 + (64*16 + 4*8)*0)*4, fx0.a.x);
            st_shared_v1(storeS + (1*16 + 0 + (64*16 + 4*8)*0)*4, fx0.a.y);
            st_shared_v1(storeS + (2*16 + 0 + (64*16 + 4*8)*0)*4, fx0.a.z);
            st_shared_v1(storeS + (3*16 + 0 + (64*16 + 4*8)*0)*4, fx0.a.w);
            st_shared_v1(storeS + (4*16 + 0 + (64*16 + 4*8)*0)*4, fx0.b.x);
            st_shared_v1(storeS + (5*16 + 0 + (64*16 + 4*8)*0)*4, fx0.b.y);
            st_shared_v1(storeS + (6*16 + 0 + (64*16 + 4*8)*0)*4, fx0.b.z);
            st_shared_v1(storeS + (7*16 + 0 + (64*16 + 4*8)*0)*4, fx0.b.w);

            st_shared_v1(storeS + (0*16 + 8 + (64*16 + 4*8)*0)*4, fx8.a.x);
            st_shared_v1(storeS + (1*16 + 8 + (64*16 + 4*8)*0)*4, fx8.a.y);
            st_shared_v1(storeS + (2*16 + 8 + (64*16 + 4*8)*0)*4, fx8.a.z);
            st_shared_v1(storeS + (3*16 + 8 + (64*16 + 4*8)*0)*4, fx8.a.w);
            st_shared_v1(storeS + (4*16 + 8 + (64*16 + 4*8)*0)*4, fx8.b.x);
            st_shared_v1(storeS + (5*16 + 8 + (64*16 + 4*8)*0)*4, fx8.b.y);
            st_shared_v1(storeS + (6*16 + 8 + (64*16 + 4*8)*0)*4, fx8.b.z);
            st_shared_v1(storeS + (7*16 + 8 + (64*16 + 4*8)*0)*4, fx8.b.w);

            float8 fe0 = to_float(e0);
            float8 fe8 = to_float(e8);

            // advance pointer by 64*2
            asm ("add.u64 %0, %0, 128;" : "+l"(E0));

            st_shared_v1(storeS + (0*16 + 0 + (64*16 + 4*8)*1)*4, fe0.a.x);
            st_shared_v1(storeS + (1*16 + 0 + (64*16 + 4*8)*1)*4, fe0.a.y);
            st_shared_v1(storeS + (2*16 + 0 + (64*16 + 4*8)*1)*4, fe0.a.z);
            st_shared_v1(storeS + (3*16 + 0 + (64*16 + 4*8)*1)*4, fe0.a.w);
            st_shared_v1(storeS + (4*16 + 0 + (64*16 + 4*8)*1)*4, fe0.b.x);
            st_shared_v1(storeS + (5*16 + 0 + (64*16 + 4*8)*1)*4, fe0.b.y);
            st_shared_v1(storeS + (6*16 + 0 + (64*16 + 4*8)*1)*4, fe0.b.z);
            st_shared_v1(storeS + (7*16 + 0 + (64*16 + 4*8)*1)*4, fe0.b.w);

            st_shared_v1(storeS + (0*16 + 8 + (64*16 + 4*8)*1)*4, fe8.a.x);
            st_shared_v1(storeS + (1*16 + 8 + (64*16 + 4*8)*1)*4, fe8.a.y);
            st_shared_v1(storeS + (2*16 + 8 + (64*16 + 4*8)*1)*4, fe8.a.z);
            st_shared_v1(storeS + (3*16 + 8 + (64*16 + 4*8)*1)*4, fe8.a.w);
            st_shared_v1(storeS + (4*16 + 8 + (64*16 + 4*8)*1)*4, fe8.b.x);
            st_shared_v1(storeS + (5*16 + 8 + (64*16 + 4*8)*1)*4, fe8.b.y);
            st_shared_v1(storeS + (6*16 + 8 + (64*16 + 4*8)*1)*4, fe8.b.z);
            st_shared_v1(storeS + (7*16 + 8 + (64*16 + 4*8)*1)*4, fe8.b.w);

            __syncthreads();

            float regX[8];
            float regE[4];

            #pragma unroll
            for (int j = 0; j < 8; j++)
            {
                // fetch outer product data
                ld_shared_v4(readEs + (16*j + 64*16 + 4*8)*4, regE);
                ld_shared_v4(readXs + (16*j + 0)*4, &regX[0] );
                ld_shared_v4(readXs + (16*j + 8)*4, &regX[4] );

                for (int e = 0; e < 4; e++)
                    for (int x = 0; x < 8; x++)
                        regU[x][e] += regX[x] * regE[e];
            }

            loop++;

        } while (loop < loops);

    } while (p < params8);

    tid = threadIdx.x;
    bid = blockIdx.x;

    int offset = bid*64 + tid;
    TU t4; ew_zero(t4);
    if (beta != 0.0f)
        t4 = U[offset];

    tid8 = tid >> 3;

    int tid1  = tid & 1;
    int tid4  = (tid >> 1) & 3;

    float4* storU4 = (float4*)&shrU[tid1*16*4 + tid1*16 + tid4*4 + tid8*16*16 + tid8*16*4];

    __syncthreads();

    storU4[ 0*4 + 0] = *(float4*)regU[0];
    storU4[ 1*4 + 0] = *(float4*)regU[1];
    storU4[ 2*4 + 0] = *(float4*)regU[2];
    storU4[ 3*4 + 0] = *(float4*)regU[3];
    storU4[ 8*4 + 8] = *(float4*)regU[4];
    storU4[ 9*4 + 8] = *(float4*)regU[5];
    storU4[10*4 + 8] = *(float4*)regU[6];
    storU4[11*4 + 8] = *(float4*)regU[7];

    __syncthreads();

    int tid16 = tid & -16;

    float4* readU4 = (float4*)&shrU[tid*4 + tid16];

    float4 u[8];
    for (int i = 0; i < 8; i++)
        u[i] = readU4[i*64 + i*16];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            u[i] = ew_add(u[i], u[i+j]);

    float4 u4 = *(float4*)u;
    float4 b4 = to_float(t4);

    u4 = ew_add(ew_mul(u4, alpha), ew_mul(b4, beta));

    store(U, u4, offset);
}

template <bool Fprop, typename TW, typename TX, typename TY>
__global__ void __launch_bounds__(128) gemm_blocksparse_32x64x32x8_xprop(
    const  int2* __restrict__ Lut,
    const    TW* __restrict__ W,
    const    TX* __restrict__ X,
    TY* Y, int* Lock, int locks, int N /* N is in units of groups of 8 elements each (N/8) */)
{
    if (Fprop)
        asm(".shared .align 16 .b32 share[3072];" ::); // 3072 = 32*32 + 64*32
    else
        asm(".shared .align 16 .b32 share[3104];" ::); // 3104 = 32*32 + 64*32 + 4*8

    extern __shared__ int2 Lut2_s[];
    int2* Lut2s = &Lut2_s[Fprop ? 3072/2 : 3104/2];

    int tid   = threadIdx.x;
    int idx_N = blockIdx.x;
    int idx_L = blockIdx.y;

    int4 lut_head = ((const int4*)Lut)[idx_L];

    int tid7  = tid  & 7;
    int tid8  = tid >> 3;
    int tid16 = tid & 16;
    int tid32 = tid & -32;

    int readXs = ((tid >> 1) & 7) << 4;
    int readWs =  (tid  & 1)      << 4;

    // each warp handles a quarter of the weights
    readWs += tid32;

    // second half of warp starts 16 rows down
    readXs += tid16 << 8; // 64*16*4
    readWs += tid16 << 7; // 32*16*4

    int storXs = (tid8*64 + tid7*4) << 2;
    int storWs;
    if (Fprop)
        storWs = tid << 4;
    else
    {
        // Transpose weights on store to shared
        // Avoid bank conflicts by shifting writes over by 4 every 4 rows (+tid7*4)
        storWs = (tid7*32*4 + tid8 + tid7*4) << 2;
        readWs += tid16 << 2; // shift over 4 floats every 4 rows, second half of warp starts 16 rows down
    }

    int  n = idx_N*8 + tid7;
    bool bn = n < N;

    int offsetX = (tid8*N + n)*8*2;

    // unpack lut header
    int lut_offset = lut_head.x;
    int lut_size   = lut_head.y;
    int idx_K      = lut_head.z;
    int idx_Lock   = lut_head.w;

    int N32 = N*32*8*2; // 32 lines, 8 elements per index, two bytes per element

    // prefetch the lut data into shared
    Lut += lut_offset;
    #pragma unroll 1
    for (int i = tid; i < lut_size; i += 128)
    {
        int2 entry = Lut[i];
        entry.x *= N32;      // 32 lines of N per block
        entry.y *= 32*32*2;  // 1024 entries of W per block, 2 bytes each
        Lut2s[i] = entry;
    }
    __syncthreads();

    // Force compiler to fully compute these prior to loop
    asm("mov.b32 %0, %0;" : "+r"(storXs)  : );
    asm("mov.b32 %0, %0;" : "+r"(storWs)  : );
    asm("mov.b32 %0, %0;" : "+r"(readXs)  : );
    asm("mov.b32 %0, %0;" : "+r"(readWs)  : );
    asm("mov.b32 %0, %0;" : "+r"(offsetX) : );

    // zero accumulation registers
    float regY[4][8];
    for (int w = 0; w < 4; w++)
        for (int x = 0; x < 8; x++)
            regY[w][x] = 0;

    // loop over each lut entry to compute a gemm block
    int i = 0;
    #pragma unroll 1
    do
    {
        int2 entry = Lut2s[i++];

        entry.x += offsetX;
        entry.y += tid*4*2;

        const TW* W0;
        const TX* X00;
        const TX* X16;
        // Simplify pointer arithmetic by letting compiler assume all offsets fit in 32 bits.
        asm("{\n\t"
            ".reg .u64 x0, x4, w0;\n\t"
            "mov.b64 w0, {%5, 0};\n\t"
            "mov.b64 x0, {%6, 0};\n\t"
            "mov.b64 x4, {%7, 0};\n\t"
            "add.u64 %0, w0, %3;\n\t"
            "add.u64 %1, x0, %4;\n\t"
            "add.u64 %2, x4, %4;\n\t"
            "}" : "=l"(W0),"=l"(X00),"=l"(X16) : "l"(W), "l"(X), "r"(entry.y), "r"(entry.x), "r"(entry.x + N*16*8*2) );

        // Fetch 8 rows at a time from W and X
        TW w00 = __ldg(W0);
        TW w16 = __ldg(W0 + 128);
        TX x00, x16;
        ew_zero(x00); ew_zero(x16);
        if (bn)
        {
            x00 = __ldg(X00);
            x16 = __ldg(X16);
        }

        __syncthreads();

        float4 fw00 = to_float(w00);
        float4 fw16 = to_float(w16);
        float8 fx00 = to_float(x00);
        float8 fx16 = to_float(x16);

        // store to shared.
        if (Fprop)
        {
            st_shared_v4(storWs + (0*16*32 + 64*32)*4, fw00);
            st_shared_v4(storWs + (1*16*32 + 64*32)*4, fw16);
        }
        else
        {
            // transpose the shared store of W
            st_shared_v1(storWs + (0*32 + 0*16 + 64*32)*4, fw00.x);
            st_shared_v1(storWs + (1*32 + 0*16 + 64*32)*4, fw00.y);
            st_shared_v1(storWs + (2*32 + 0*16 + 64*32)*4, fw00.z);
            st_shared_v1(storWs + (3*32 + 0*16 + 64*32)*4, fw00.w);

            st_shared_v1(storWs + (0*32 + 1*16 + 64*32)*4, fw16.x);
            st_shared_v1(storWs + (1*32 + 1*16 + 64*32)*4, fw16.y);
            st_shared_v1(storWs + (2*32 + 1*16 + 64*32)*4, fw16.z);
            st_shared_v1(storWs + (3*32 + 1*16 + 64*32)*4, fw16.w);
        }
        // avoid bank conflicts by splitting 8 floats in two groups of 4
        // We can rejoin them on the output
        st_shared_v4(storXs + ( 0*64 + 0*32)*4, fx00.a);
        st_shared_v4(storXs + ( 0*64 + 1*32)*4, fx00.b);
        st_shared_v4(storXs + (16*64 + 0*32)*4, fx16.a);
        st_shared_v4(storXs + (16*64 + 1*32)*4, fx16.b);

        __syncthreads();

        // computes a 16x64x16 gemm tile with 4x8 register blocking
        float regW[4];
        float regX[8];
        #pragma unroll
        for (int j = 0; j < 16; j++)
        {
            // fetch outer product data
            ld_shared_v4(readWs + (32*j + 64*32 + (Fprop ? 0 : (j>>2)*4))*4, regW ); // shift over 4 floats every 4 rows
            ld_shared_v4(readXs + (64*j +  0)*4, &regX[0] );
            ld_shared_v4(readXs + (64*j + 32)*4, &regX[4] );

            // accumulate outer product
            for (int w = 0; w < 4; w++)
                for (int x = 0; x < 8; x++)
                    regY[w][x] += regW[w] * regX[x];
        }

    } while (i < lut_size);

    tid16 = tid &  16;
    tid32 = tid & -32;

    int tidN = (tid >> 1) & 7;
    int tidK = (tid & 1) + (tid32 >> 4);
    int tidk = tid16 >> 4;

    bool t16 = tid16 != 0;

    float outY[2][8];
    for (int w = 0; w < 2; w++)
    {
        for (int x = 0; x < 8; x++)
        {
            float swap = t16 ? regY[2*w + 0][x] : regY[2*w + 1][x];
            outY[w][x] = t16 ? regY[2*w + 1][x] : regY[2*w + 0][x];
            outY[w][x] += __shfl_xor(swap, 16);
        }
    }

    n  = idx_N*8 + tidN;
    bn = n < N;

    Y += (idx_K*32 + tidK*4 + tidk)*N + n;

    if (idx_Lock == 0)
    {
        // no lock needed just write out the results
        store(Y, *(float8*)outY[0], N*0, bn);
        store(Y, *(float8*)outY[1], N*2, bn);
    }
    else
    {
        int offsetL = idx_N*locks + idx_Lock - 1;
        Lock += offsetL;

        // Critial Section
        if (tid == 0)
            while (atomicCAS(Lock, 0, 1) != 0);
        __syncthreads();

        int offsetC = locks*gridDim.x;
        int* Count = Lock + offsetC;
        int  count = *Count;

        __syncthreads(); // all threads have the count now and safe to update

        if (count == 0)
        {
            if (tid == 0)
                *Count = 1;

            // first block to get here just writes out to init the memory
            store(Y, *(float8*)outY[0], N*0, bn);
            store(Y, *(float8*)outY[1], N*2, bn);

            __threadfence();
            __syncthreads();

            if (tid == 0)
                atomicExch(Lock, 0);
            // End Critial Section
        }
        else
        {
            // subsequent blocks must accumulate
            float8 y0 = load_c(Y, N*0, bn);
            float8 y2 = load_c(Y, N*2, bn);

            y0 = ew_add(y0, *(float8*)outY[0]);
            y2 = ew_add(y2, *(float8*)outY[1]);

            store(Y, y0, N*0, bn);
            store(Y, y2, N*2, bn);

            __threadfence();
            __syncthreads();

            if (tid == 0)
                atomicExch(Lock, 0);
            // End Critial Section
        }
    }
}

template <bool Fprop, typename TW, typename TX, typename TY>
__global__ void __launch_bounds__(128) gemm_blocksparse_32x64x32x4_xprop(
    const  int2* __restrict__ Lut,
    const    TW* __restrict__ W,
    const    TX* __restrict__ X,
    TY* Y, int* Lock, int locks, int N /* N is in units of groups of 4 elements each (N/4) */)
{
    if (Fprop)
        asm(".shared .align 16 .b32 share[3072];" ::); // 3072 = 32*32 + 64*32
    else
        asm(".shared .align 16 .b32 share[3104];" ::); // 3104 = 32*32 + 64*32 + 4*8

    extern __shared__ int2 Lut2_s[];
    int2* Lut2s = &Lut2_s[Fprop ? 3072/2 : 3104/2];

    int tid   = threadIdx.x;
    int idx_N = blockIdx.x;
    int idx_L = blockIdx.y;

    int4 lut_head = ((const int4*)Lut)[idx_L];

    int tid7  = tid  & 7;
    int tid8  = tid >> 3;
    int tid15 = tid & 15;
    int tid4 =  tid >> 4;
    int tid16 = tid & 16;
    int tid32 = tid & -32;

    int readXs = ((tid >> 1) & 7) << 4;
    int readWs =  (tid  & 1)      << 4;

    // each warp handles a quarter of the weights
    readWs += tid32;

    // second half of warp starts 16 rows down
    readXs += tid16 << 8; // 64*16*4
    readWs += tid16 << 7; // 32*16*4

    int storXs = (tid4*64 + tid15*4) << 2;
    int storWs;
    if (Fprop)
        storWs = tid << 4;
    else
    {
        // Transpose weights on store to shared
        // Avoid bank conflicts by shifting writes over by 4 every 4 rows (+tid7*4)
        storWs = (tid7*32*4 + tid8 + tid7*4) << 2;
        readWs += tid16 << 2; // shift over 4 floats every 4 rows, second half of warp starts 16 rows down
    }

    int  n = idx_N*64/4 + tid15;
    bool bn = n < N;

    int offsetX = (tid4*N + n)*sizeof(TX);

    // unpack lut header
    int lut_offset = lut_head.x;
    int lut_size   = lut_head.y;
    int idx_K      = lut_head.z;
    int idx_Lock   = lut_head.w;

    int N32 = N*32*sizeof(TX); // 32 lines, 4 elements per index, sizeof bytes per element
    int N8 =  N*8*sizeof(TX);

    // prefetch the lut data into shared
    Lut += lut_offset;
    #pragma unroll 1
    for (int i = tid; i < lut_size; i += 128)
    {
        int2 entry = Lut[i];
        entry.x *= N32;      // 32 lines of N per block
        entry.y *= 32*8*sizeof(TW);  // 1024 entries of W per block, sizeof bytes each
        Lut2s[i] = entry;
    }
    __syncthreads();

    // Force compiler to fully compute these prior to loop
    asm("mov.b32 %0, %0;" : "+r"(storXs)  : );
    asm("mov.b32 %0, %0;" : "+r"(storWs)  : );
    asm("mov.b32 %0, %0;" : "+r"(readXs)  : );
    asm("mov.b32 %0, %0;" : "+r"(readWs)  : );
    asm("mov.b32 %0, %0;" : "+r"(offsetX) : );
    asm("mov.b32 %0, %0;" : "+r"(N8) : );

    // zero accumulation registers
    float regY[4][8];
    for (int w = 0; w < 4; w++)
        for (int x = 0; x < 8; x++)
            regY[w][x] = 0;


    // loop over each lut entry to compute a gemm block
    int i = 0;
    #pragma unroll 1
    do
    {
        int2 entry = Lut2s[i++];

        entry.x += offsetX;
        entry.y += tid*sizeof(TW);

        const TW* W0;
        const TX* X00;
        const TX* X08;
        const TX* X16;
        const TX* X24;
        // Simplify pointer arithmetic by letting compiler assume all offsets fit in 32 bits.
        asm("{\n\t"
            ".reg .u64 w0, x0, N8, X00, X08, X16, X24;\n\t"
            "mov.b64 x0, {%7, 0};\n\t"
            "mov.b64 w0, {%8, 0};\n\t"
            "mov.b64 N8, {%9, 0};\n\t"
            "add.u64 X00, %5, x0;\n\t"
            "add.u64 X08, X00, N8;\n\t"
            "add.u64 X16, X08, N8;\n\t"
            "add.u64 X24, X16, N8;\n\t"
            "mov.b64 %0, X00;\n\t"
            "mov.b64 %1, X08;\n\t"
            "mov.b64 %2, X16;\n\t"
            "mov.b64 %3, X24;\n\t"
            "add.u64 %4, %6, w0;\n\t"
            "}" : "=l"(X00),"=l"(X08),"=l"(X16),"=l"(X24), "=l"(W0) : "l"(X), "l"(W), "r"(entry.x), "r"(entry.y), "r"(N8) );

        // Fetch 8 rows at a time from W and X
        TW w00 = __ldg(W0);
        TW w16 = __ldg(W0 + 128);
        TX x00, x08, x16, x24;
        ew_zero(x00);
        ew_zero(x08);
        ew_zero(x16);
        ew_zero(x24);
        if (bn)
        {
            x00 = __ldg(X00);
            x08 = __ldg(X08);
            x16 = __ldg(X16);
            x24 = __ldg(X24);
        }

        __syncthreads();

        float4 fw00 = to_float(w00);
        float4 fw16 = to_float(w16);

        float4 fx00 = to_float(x00);
        float4 fx08 = to_float(x08);
        float4 fx16 = to_float(x16);
        float4 fx24 = to_float(x24);

        // fw00 = round3w(fw00);
        // fw16 = round3w(fw16);
        // fx00 = round4(fx00);
        // fx08 = round4(fx08);
        // fx16 = round4(fx16);
        // fx24 = round4(fx24);

        // store to shared.
        if (Fprop)
        {
            st_shared_v4(storWs + (0*16*32 + 64*32)*4, fw00);
            st_shared_v4(storWs + (1*16*32 + 64*32)*4, fw16);
        }
        else
        {
            // transpose the shared store of W
            st_shared_v1(storWs + (0*32 + 0*16 + 64*32)*4, fw00.x);
            st_shared_v1(storWs + (1*32 + 0*16 + 64*32)*4, fw00.y);
            st_shared_v1(storWs + (2*32 + 0*16 + 64*32)*4, fw00.z);
            st_shared_v1(storWs + (3*32 + 0*16 + 64*32)*4, fw00.w);

            st_shared_v1(storWs + (0*32 + 1*16 + 64*32)*4, fw16.x);
            st_shared_v1(storWs + (1*32 + 1*16 + 64*32)*4, fw16.y);
            st_shared_v1(storWs + (2*32 + 1*16 + 64*32)*4, fw16.z);
            st_shared_v1(storWs + (3*32 + 1*16 + 64*32)*4, fw16.w);
        }

        st_shared_v4(storXs +  0*64*4, fx00);
        st_shared_v4(storXs +  8*64*4, fx08);
        st_shared_v4(storXs + 16*64*4, fx16);
        st_shared_v4(storXs + 24*64*4, fx24);

        __syncthreads();

        // computes a 16x64x16 gemm tile with 4x8 register blocking
        float regW[4];
        float regX[8];
        #pragma unroll
        for (int j = 0; j < 16; j++)
        {
            // fetch outer product data
            ld_shared_v4(readWs + (32*j + 64*32 + (Fprop ? 0 : (j>>2)*4))*4, regW ); // shift over 4 floats every 4 rows
            ld_shared_v4(readXs + (64*j +  0)*4, &regX[0] );
            ld_shared_v4(readXs + (64*j + 32)*4, &regX[4] );

            // accumulate outer product
            for (int w = 0; w < 4; w++)
                for (int x = 0; x < 8; x++)
                {
                    regY[w][x] += regW[w] * regX[x];
                    //asm("fma.rz.ftz.f32 %0, %1, %2, %0;" : "+f"(regY[w][x]) : "f"(regW[w]), "f"(regX[x]));
                    // if (j == 7)
                    //     asm("and.b32 %0, %0, 0xfffffff0;" : "+f"(regY[w][x]) : );
                    // else
                    //asm("and.b32 %0, %0, 0xffffff00;" : "+f"(regY[w][x]) : );
                }
        }

    } while (i < lut_size);

    tid16 = tid &  16;
    tid32 = tid & -32;

    int tidN = (tid >> 1) & 7;
    int tidK = (tid & 1) + (tid32 >> 4);
    int tidk = tid16 >> 4;

    bool t16 = tid16 != 0;

    float outY[2][8];
    for (int w = 0; w < 2; w++)
    {
        for (int x = 0; x < 8; x++)
        {
            float swap = t16 ? regY[2*w + 0][x] : regY[2*w + 1][x];
            outY[w][x] = t16 ? regY[2*w + 1][x] : regY[2*w + 0][x];
            outY[w][x] += __shfl_xor(swap, 16);
        }
    }

    n  = idx_N*64/4 + tidN;
    bool bn0 = n+0 < N;
    bool bn8 = n+8 < N;

    Y += (idx_K*32 + tidK*4 + tidk)*N + n;

    if (idx_Lock == 0)
    {
        // no lock needed just write out the results
        store(Y, *(float4*)&outY[0][0], N*0 + 0, bn0);
        store(Y, *(float4*)&outY[0][4], N*0 + 8, bn8);
        store(Y, *(float4*)&outY[1][0], N*2 + 0, bn0);
        store(Y, *(float4*)&outY[1][4], N*2 + 8, bn8);
    }
    else
    {
        int offsetL = idx_N*locks + idx_Lock - 1;
        Lock += offsetL;

        // Critial Section
        if (tid == 0)
            while (atomicCAS(Lock, 0, 1) != 0);
        __syncthreads();

        int offsetC = locks*gridDim.x;
        int* Count = Lock + offsetC;
        int  count = *Count;

        __syncthreads(); // all threads have the count now and safe to update

        if (count == 0)
        {
            if (tid == 0)
                *Count = 1;

            // first block to get here just writes out to init the memory
            store(Y, *(float4*)&outY[0][0], N*0 + 0, bn0);
            store(Y, *(float4*)&outY[0][4], N*0 + 8, bn8);
            store(Y, *(float4*)&outY[1][0], N*2 + 0, bn0);
            store(Y, *(float4*)&outY[1][4], N*2 + 8, bn8);

            __threadfence();
            __syncthreads();

            if (tid == 0)
                atomicExch(Lock, 0);
            // End Critial Section
        }
        else
        {
            // subsequent blocks must accumulate
            float4 y00 = load_c(Y, N*0 + 0, bn0);
            float4 y08 = load_c(Y, N*0 + 8, bn8);
            float4 y20 = load_c(Y, N*2 + 0, bn0);
            float4 y28 = load_c(Y, N*2 + 8, bn8);

            y00 = ew_add(y00, *(float4*)&outY[0][0]);
            y08 = ew_add(y08, *(float4*)&outY[0][4]);
            y20 = ew_add(y20, *(float4*)&outY[1][0]);
            y28 = ew_add(y28, *(float4*)&outY[1][4]);

            store(Y, y00, N*0 + 0, bn0);
            store(Y, y08, N*0 + 8, bn8);
            store(Y, y20, N*2 + 0, bn0);
            store(Y, y28, N*2 + 8, bn8);

            __threadfence();
            __syncthreads();

            if (tid == 0)
                atomicExch(Lock, 0);
            // End Critial Section
        }
    }
}

template <typename TX, typename TE, typename TU>
__global__ void __launch_bounds__(256) gemm_blocksparse_32x64x32x8_updat(
    struct plist8<TX> X, struct plist8<TE> E,
    const  int2* __restrict__ Lut,
    TU* U,
    int params8, int N, int loops, float alpha, float beta)
{
    //asm(".shared .align 16 .b32 share[2560];" ::); // 2560 = (64*16 + 4*64)*2
    __shared__ float shrU[(64*32 + 4*8)*2];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int2 lut_head = Lut[bid];

    int tid7 = tid  & 7;
    int tid8 = tid >> 3;
    int tid32 = tid & -32;

    // avoid bank conflicts when writing transpose (+tid7*4)
    int storeS = tid7*32*8 + tid8 + tid7*4;

    // 32 threads per tile, each tile reads 8 lines, shifted over by 4
    int readXs = (((tid & 16) >> 3) | (tid & 1)) << 4;
    int readEs = ((tid >> 1) & 7) << 4;

    readXs += (tid32 << 5) + (tid32 >> 1); // 32*8*4
    readEs += (tid32 << 5) + (tid32 >> 1); // 32*8*4

    int idx_X = lut_head.x;
    int idx_E = lut_head.y;

    int offsetX = (idx_X*32 + tid8)*N + tid7*8;
    int offsetE = (idx_E*32 + tid8)*N + tid7*8;

    // This keeps all prior logic outside of the loops.
    asm("shl.b32 %0, %0, 2;" : "+r"(storeS)  : );
    asm("shl.b32 %0, %0, 1;" : "+r"(offsetX) : );
    asm("shl.b32 %0, %0, 1;" : "+r"(offsetE) : );
    asm("mov.b32 %0, %0;" : "+r"(readXs) : );
    asm("mov.b32 %0, %0;" : "+r"(readEs) : );

    float regU[8][4]; //[x][e]
    for (int x = 0; x < 8; x++)
        for (int e = 0; e < 4; e++)
            regU[x][e] = 0;

    int p = 0;
    #pragma unroll 1
    do
    {
        const TX* X0;
        const TE* E0;
        asm("{\n\t"
            ".reg .u64 X, E, offsetX, offsetE;\n\t"
# if __CUDA_ARCH__ >= 700
            "ld.param.u64 X, [%2 + 0x160];\n\t" // WARNING: hard coded param offset.
            "ld.param.u64 E, [%2 + 0x1a0];\n\t" // WARNING: hard coded param offset.
# else
            "ld.param.u64 X, [%2 + 0x140];\n\t" // WARNING: hard coded param offset.
            "ld.param.u64 E, [%2 + 0x180];\n\t" // WARNING: hard coded param offset.
# endif
            "cvta.to.global.u64 X, X;\n\t"
            "cvta.to.global.u64 E, E;\n\t"
            "mov.b64 offsetX, {%3, 0};\n\t"
            "mov.b64 offsetE, {%4, 0};\n\t"
            "add.u64 %0, X, offsetX;\n\t"
            "add.u64 %1, E, offsetE;\n\t"
            "}" : "=l"(X0), "=l"(E0) : "r"(p), "r"(offsetX), "r"(offsetE));
        p += 8;

        int n = (tid & 7) << 3;
        int loop = 0;
        #pragma unroll 1
        do
        {
            TX x0;
            TE e0;
            ew_zero(x0);
            ew_zero(e0);
            if (n < N)
            {
                x0 = __ldg(X0);
                e0 = __ldg(E0);
            }
            __syncthreads();

            // Convert to float if needed and store to shared as transpose.
            float8 fx0 = to_float(x0);
            st_shared_v1(storeS + (0*32 + (64*32 + 4*8)*0)*4, fx0.a.x);
            st_shared_v1(storeS + (1*32 + (64*32 + 4*8)*0)*4, fx0.a.y);
            st_shared_v1(storeS + (2*32 + (64*32 + 4*8)*0)*4, fx0.a.z);
            st_shared_v1(storeS + (3*32 + (64*32 + 4*8)*0)*4, fx0.a.w);
            st_shared_v1(storeS + (4*32 + (64*32 + 4*8)*0)*4, fx0.b.x);
            st_shared_v1(storeS + (5*32 + (64*32 + 4*8)*0)*4, fx0.b.y);
            st_shared_v1(storeS + (6*32 + (64*32 + 4*8)*0)*4, fx0.b.z);
            st_shared_v1(storeS + (7*32 + (64*32 + 4*8)*0)*4, fx0.b.w);

            float8 fe0 = to_float(e0);
            st_shared_v1(storeS + (0*32 + (64*32 + 4*8)*1)*4, fe0.a.x);
            st_shared_v1(storeS + (1*32 + (64*32 + 4*8)*1)*4, fe0.a.y);
            st_shared_v1(storeS + (2*32 + (64*32 + 4*8)*1)*4, fe0.a.z);
            st_shared_v1(storeS + (3*32 + (64*32 + 4*8)*1)*4, fe0.a.w);
            st_shared_v1(storeS + (4*32 + (64*32 + 4*8)*1)*4, fe0.b.x);
            st_shared_v1(storeS + (5*32 + (64*32 + 4*8)*1)*4, fe0.b.y);
            st_shared_v1(storeS + (6*32 + (64*32 + 4*8)*1)*4, fe0.b.z);
            st_shared_v1(storeS + (7*32 + (64*32 + 4*8)*1)*4, fe0.b.w);

            // advance pointers by 64*2
            asm ("add.u64 %0, %0, 128;" : "+l"(X0));
            asm ("add.u64 %0, %0, 128;" : "+l"(E0));
            n += 64;

            __syncthreads();

            float regX[8];
            float regE[4];

            #pragma unroll
            for (int j = 0; j < 8; j++)
            {
                // fetch outer product data
                ld_shared_v4(readEs + (32*j + 64*32 + 4*8)*4, regE);
                ld_shared_v4(readXs + (32*j +  0)*4, &regX[0] );
                ld_shared_v4(readXs + (32*j + 16)*4, &regX[4] );

                for (int e = 0; e < 4; e++)
                    for (int x = 0; x < 8; x++)
                        regU[x][e] += regX[x] * regE[e];
            }
            loop++;

        } while (loop < loops);

    } while (p < params8);

    tid = threadIdx.x;
    bid = blockIdx.x;

    int offset = bid*32*32/2 + tid;
    TU t00, t16;
    ew_zero(t00);
    ew_zero(t16);
    if (beta != 0.0f)
    {
        t00 = U[offset];
        t16 = U[offset + 256];
    }

    // Arrange 8 tiles horizontally in the X direction: ((tid & -32) >> 1)
    // Add some spacing  to avoid write bank conflicts: (tidY << 2)
    int tidY = ((tid & 16) >> 3) + (tid & 1);
    int tidX = ((tid >> 1) & 7) + ((tid & -32) >> 2) + (tidY << 2);

    int offsetS = tidY*32*8*4 + tidX*4;

    float4* storU4 = (float4*)&shrU[offsetS];

    __syncthreads();

    storU4[ 0*32*8/4 ] = *(float4*)regU[0];
    storU4[ 1*32*8/4 ] = *(float4*)regU[1];
    storU4[ 2*32*8/4 ] = *(float4*)regU[2];
    storU4[ 3*32*8/4 ] = *(float4*)regU[3];

    __syncthreads();

    int tid15 = tid & 15;
    int tid16 = tid >> 4;
    int tid64 = tid & -64;

    float2* readU2 = (float2*)&shrU[tid16*32*8 + tid15*2 + (tid64>>2)];

    float2 u[8];
    for (int i = 0; i < 8; i++)
        u[i] = readU2[i*32/2];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            u[i] = ew_add(u[i], u[i+j]);

    float2 u2 = *(float2*)u;
    float2 b2 = to_float(t00);

    u2 = ew_add(ew_mul(u2, alpha), ew_mul(b2, beta));

    store(U, u2, offset);

    __syncthreads();

    storU4[ 0*32*8/4 ] = *(float4*)regU[4];
    storU4[ 1*32*8/4 ] = *(float4*)regU[5];
    storU4[ 2*32*8/4 ] = *(float4*)regU[6];
    storU4[ 3*32*8/4 ] = *(float4*)regU[7];

    __syncthreads();

    for (int i = 0; i < 8; i++)
        u[i] = readU2[i*32/2];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            u[i] = ew_add(u[i], u[i+j]);

    u2 = *(float2*)u;
    b2 = to_float(t16);

    u2 = ew_add(ew_mul(u2, alpha), ew_mul(b2, beta));

    store(U, u2, offset + 256);
}

template <typename TX, typename TE, typename TU>
__global__ void __launch_bounds__(256) gemm_blocksparse_32x64x32x4_updat(
    struct plist8<TX> X, struct plist8<TE> E,
    const  int2* __restrict__ Lut,
    TU* U,
    int params8, int N, int loops, float alpha, float beta)
{
    __shared__ float shrU[(64*32 + 2*16)*2];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int2 lut_head = Lut[bid];

    int tid15 = tid & 15;
    int tid16 = tid >> 4;
    int tid32 = tid & -32;

    // avoid bank conflicts when writing transpose (+tid15*2)
    int storeS = (tid15*32*4 + tid16 + tid15*2)*4;

    // 32 threads per tile, each tile reads 8 lines, shifted over by 4
    int readXs = (((tid & 16) >> 3) | (tid & 1)) << 4;
    int readEs = ((tid >> 1) & 7) << 4;

    readXs += (tid32 << 5) + (tid32 >> 1); // 32*8*4
    readEs += (tid32 << 5) + (tid32 >> 1); // 32*8*4

    int idx_X = lut_head.x;
    int idx_E = lut_head.y;

    int offsetX = ((idx_X*32 + tid16)*N + tid15*4) * (sizeof(TX)/4);
    int offsetE = ((idx_E*32 + tid16)*N + tid15*4) * (sizeof(TE)/4);

    // This keeps all prior logic outside of the loops.
    asm("mov.b32 %0, %0;" : "+r"(storeS)  : );
    asm("mov.b32 %0, %0;" : "+r"(offsetX) : );
    asm("mov.b32 %0, %0;" : "+r"(offsetE) : );
    asm("mov.b32 %0, %0;" : "+r"(readXs)  : );
    asm("mov.b32 %0, %0;" : "+r"(readEs)  : );

    float regU[8][4]; //[x][e]
    for (int x = 0; x < 8; x++)
        for (int e = 0; e < 4; e++)
            regU[x][e] = 0;

    int p = 0;
    #pragma unroll 1
    do
    {
        const TX *X00, *X16;
        const TE *E00, *E16;
        int xn16 = N*16*(sizeof(TX)/4);
        int en16 = N*16*(sizeof(TE)/4);
        asm("{\n\t"
            ".reg .u64 X, E;\n\t"
# if __CUDA_ARCH__ >= 700
            "ld.param.u64 X, [%4 + 0x160];\n\t" // WARNING: hard coded param offset.
            "ld.param.u64 E, [%4 + 0x1a0];\n\t" // WARNING: hard coded param offset.
# else
            "ld.param.u64 X, [%4 + 0x140];\n\t" // WARNING: hard coded param offset.
            "ld.param.u64 E, [%4 + 0x180];\n\t" // WARNING: hard coded param offset.
# endif
            "cvta.to.global.u64 X, X;\n\t"
            "cvta.to.global.u64 E, E;\n\t"
            "add.u64 X, X, %5;\n\t"
            "add.u64 E, E, %6;\n\t"
            "mov.b64 %0, X;\n\t"
            "mov.b64 %1, E;\n\t"
            "add.u64 %2, X, %7;\n\t"
            "add.u64 %3, E, %8;\n\t"
            "}" : "=l"(X00), "=l"(E00), "=l"(X16), "=l"(E16) : "r"(p), "l"((int64_t)offsetX), "l"((int64_t)offsetE), "l"((int64_t)xn16), "l"((int64_t)en16) );
        p += 8;

        int n = (tid & 15) << 2;
        int loop = 0;
        #pragma unroll 1
        do
        {
            TX x00, x16;
            TE e00, e16;
            ew_zero(x00);
            ew_zero(x16);
            ew_zero(e00);
            ew_zero(e16);
            if (n < N)
            {
                x00 = __ldg(X00);
                x16 = __ldg(X16);
                e00 = __ldg(E00);
                e16 = __ldg(E16);
            }
            __syncthreads();

            // Convert to float if needed and store to shared as transpose.
            float4 fx00 = to_float(x00);
            float4 fx16 = to_float(x16);
            // fx00 = round4(fx00);
            // fx16 = round4(fx16);
            st_shared_v1(storeS + (0*32 +  0 + (64*32 + 2*16)*0)*4, fx00.x);
            st_shared_v1(storeS + (1*32 +  0 + (64*32 + 2*16)*0)*4, fx00.y);
            st_shared_v1(storeS + (2*32 +  0 + (64*32 + 2*16)*0)*4, fx00.z);
            st_shared_v1(storeS + (3*32 +  0 + (64*32 + 2*16)*0)*4, fx00.w);

            st_shared_v1(storeS + (0*32 + 16 + (64*32 + 2*16)*0)*4, fx16.x);
            st_shared_v1(storeS + (1*32 + 16 + (64*32 + 2*16)*0)*4, fx16.y);
            st_shared_v1(storeS + (2*32 + 16 + (64*32 + 2*16)*0)*4, fx16.z);
            st_shared_v1(storeS + (3*32 + 16 + (64*32 + 2*16)*0)*4, fx16.w);

            float4 fe00 = to_float(e00);
            float4 fe16 = to_float(e16);
            // fe00 = round4(fe00);
            // fe16 = round4(fe16);
            st_shared_v1(storeS + (0*32 +  0 + (64*32 + 2*16)*1)*4, fe00.x);
            st_shared_v1(storeS + (1*32 +  0 + (64*32 + 2*16)*1)*4, fe00.y);
            st_shared_v1(storeS + (2*32 +  0 + (64*32 + 2*16)*1)*4, fe00.z);
            st_shared_v1(storeS + (3*32 +  0 + (64*32 + 2*16)*1)*4, fe00.w);

            st_shared_v1(storeS + (0*32 + 16 + (64*32 + 2*16)*1)*4, fe16.x);
            st_shared_v1(storeS + (1*32 + 16 + (64*32 + 2*16)*1)*4, fe16.y);
            st_shared_v1(storeS + (2*32 + 16 + (64*32 + 2*16)*1)*4, fe16.z);
            st_shared_v1(storeS + (3*32 + 16 + (64*32 + 2*16)*1)*4, fe16.w);

            // advance pointers by 64
            int xn64 = 64*(sizeof(TX)/4);
            int en64 = 64*(sizeof(TE)/4);
            asm ("add.u64 %0, %0, %1;" : "+l"(X00) : "l"((int64_t)xn64));
            asm ("add.u64 %0, %0, %1;" : "+l"(E00) : "l"((int64_t)en64));
            asm ("add.u64 %0, %0, %1;" : "+l"(X16) : "l"((int64_t)xn64));
            asm ("add.u64 %0, %0, %1;" : "+l"(E16) : "l"((int64_t)en64));
            n += 64;

            __syncthreads();

            float regX[8];
            float regE[4];

            #pragma unroll
            for (int j = 0; j < 4; j++)
            {
                // fetch outer product data
                ld_shared_v4(readEs + (32*j + 64*32 + 2*16)*4, regE);
                ld_shared_v4(readXs + (32*j +  0)*4, &regX[0] );
                ld_shared_v4(readXs + (32*j + 16)*4, &regX[4] );

                for (int e = 0; e < 4; e++)
                    for (int x = 0; x < 8; x++)
                        regU[x][e] += regX[x] * regE[e];
            }

            #pragma unroll
            for (int j = 4; j < 8; j++)
            {
                // fetch outer product data
                ld_shared_v2(readEs + (32*j +  0 + (j>>2)*2 + 64*32 + 2*16)*4, &regE[0]);
                ld_shared_v2(readEs + (32*j +  2 + (j>>2)*2 + 64*32 + 2*16)*4, &regE[2]);
                ld_shared_v2(readXs + (32*j +  0 + (j>>2)*2)*4, &regX[0]);
                ld_shared_v2(readXs + (32*j +  2 + (j>>2)*2)*4, &regX[2]);
                ld_shared_v2(readXs + (32*j + 16 + (j>>2)*2)*4, &regX[4]);
                ld_shared_v2(readXs + (32*j + 18 + (j>>2)*2)*4, &regX[6]);

                for (int e = 0; e < 4; e++)
                    for (int x = 0; x < 8; x++)
                    {
                        regU[x][e] += regX[x] * regE[e];
                        //asm("fma.rz.ftz.f32 %0, %1, %2, %0;" : "+f"(regU[x][e]) : "f"(regX[x]), "f"(regE[e]));
                        // if (j == 7)
                        //     asm("and.b32 %0, %0, 0xfffffff0;" : "+f"(regU[x][e]) : );
                        // else
                        //asm("and.b32 %0, %0, 0xffffff00;" : "+f"(regU[x][e]) : );
                    }
            }
            loop++;

        } while (loop < loops);

    } while (p < params8);

    tid = threadIdx.x;
    bid = blockIdx.x;

    int offset = bid*32*32/2 + tid;
    TU t00, t16;
    ew_zero(t00);
    ew_zero(t16);
    if (beta != 0.0f)
    {
        t00 = U[offset];
        t16 = U[offset + 256];
    }

    // Arrange 8 tiles horizontally in the X direction: ((tid & -32) >> 1)
    // Add some spacing  to avoid write bank conflicts: (tidY << 2)
    int tidY = ((tid & 16) >> 3) + (tid & 1);
    int tidX = ((tid >> 1) & 7) + ((tid & -32) >> 2) + (tidY << 2);

    int offsetS = tidY*32*8*4 + tidX*4;

    float4* storU4 = (float4*)&shrU[offsetS];

    __syncthreads();

    storU4[ 0*32*8/4 ] = *(float4*)regU[0];
    storU4[ 1*32*8/4 ] = *(float4*)regU[1];
    storU4[ 2*32*8/4 ] = *(float4*)regU[2];
    storU4[ 3*32*8/4 ] = *(float4*)regU[3];

    __syncthreads();

    tid15 = tid & 15;
    tid16 = tid >> 4;
    int tid64 = tid & -64;

    float2* readU2 = (float2*)&shrU[tid16*32*8 + tid15*2 + (tid64>>2)];

    float2 u[8];
    for (int i = 0; i < 8; i++)
        u[i] = readU2[i*32/2];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            u[i] = ew_add(u[i], u[i+j]);

    float2 u2 = *(float2*)u;
    float2 b2 = to_float(t00);

    u2 = ew_add(ew_mul(u2, alpha), ew_mul(b2, beta));

    store_g(U, u2, offset);

    __syncthreads();

    storU4[ 0*32*8/4 ] = *(float4*)regU[4];
    storU4[ 1*32*8/4 ] = *(float4*)regU[5];
    storU4[ 2*32*8/4 ] = *(float4*)regU[6];
    storU4[ 3*32*8/4 ] = *(float4*)regU[7];

    __syncthreads();

    for (int i = 0; i < 8; i++)
        u[i] = readU2[i*32/2];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            u[i] = ew_add(u[i], u[i+j]);

    u2 = *(float2*)u;
    b2 = to_float(t16);

    u2 = ew_add(ew_mul(u2, alpha), ew_mul(b2, beta));

    store_g(U, u2, offset + 256);
}

#define GridDim(size, shift) ((size >> shift) + ((size & ((1<<shift)-1)) != 0))

template <bool Fprop, CTYPE3(TX, TW, TY)>
cudaError_t BsmmXprop_CN(const TX* X, const TW* W, TY* Y, bsmm_params* params)
{
    dim3 grid(GridDim(params->N, 6), params->segments, 1);

    const int2* L2 = (const int2*)params->Lut;
    const  TW2* W2 = (const  TW2*)W;
    const  TW4* W4 = (const  TW4*)W;
    const  TX4* X4 = (const  TX4*)X;
    const  TX8* X8 = (const  TX8*)X;
           TY2* Y2 = (       TY2*)Y;
           TY4* Y4 = (       TY4*)Y;
           TY8* Y8 = (       TY8*)Y;

    if (params->locks > 0)
        cuMemsetD32Async((CUdeviceptr)params->Lock, 0, grid.x * params->locks * 2, params->stream);

    if (params->bshift == 3)
    {
        if (sizeof(TW) == 2 && sizeof(TX) == 2 && (params->N & 7) == 0)
            gemm_blocksparse_08x64x08x8_xprop<Fprop,TW2,TX8,TY8><<<grid,32,params->shared,params->stream>>>(L2, W2, X8, Y8, params->Lock, params->locks, params->N>>3);
        else
            gemm_blocksparse_08x64x08x4_xprop<Fprop,TW2,TX4,TY2><<<grid,32,params->shared,params->stream>>>(L2, W2, X4, Y2, params->Lock, params->locks, params->N);
    }
    else if (params->bshift == 4)
    {
        if (sizeof(TW) == 2 && sizeof(TX) == 2 && (params->N & 7) == 0)
            gemm_blocksparse_16x64x16x8_xprop<Fprop,TW4,TX8,TY8><<<grid,64,params->shared,params->stream>>>(L2, W4, X8, Y8, params->Lock, params->locks, params->N>>3);
        else
            gemm_blocksparse_16x64x16x4_xprop<Fprop,TW4,TX4,TY2><<<grid,64,params->shared,params->stream>>>(L2, W4, X4, Y2, params->Lock, params->locks, params->N);
    }
    else
    {
        if (sizeof(TW) == 2 && sizeof(TX) == 2 && (params->N & 7) == 0)
            gemm_blocksparse_32x64x32x8_xprop<Fprop,TW4,TX8,TY8><<<grid,128,params->shared,params->stream>>>(L2, W4, X8, Y8, params->Lock, params->locks, params->N>>3);
        else
            gemm_blocksparse_32x64x32x4_xprop<Fprop,TW4,TX4,TY4><<<grid,128,params->shared,params->stream>>>(L2, W4, X4, Y4, params->Lock, params->locks, params->N>>2);
    }
    return cudaPeekAtLastError();
}

template cudaError_t BsmmXprop_CN<true,  VTYPE3(float,float,float)>(const float* X, const float* W, float* Y, bsmm_params* params);
template cudaError_t BsmmXprop_CN<true,  VTYPE3(ehalf,ehalf,ehalf)>(const ehalf* X, const ehalf* W, ehalf* Y, bsmm_params* params);
template cudaError_t BsmmXprop_CN<true,  VTYPE3(bhalf,bhalf,bhalf)>(const bhalf* X, const bhalf* W, bhalf* Y, bsmm_params* params);

template cudaError_t BsmmXprop_CN<false, VTYPE3(float,float,float)>(const float* X, const float* W, float* Y, bsmm_params* params);
template cudaError_t BsmmXprop_CN<false, VTYPE3(ehalf,ehalf,ehalf)>(const ehalf* X, const ehalf* W, ehalf* Y, bsmm_params* params);
template cudaError_t BsmmXprop_CN<false, VTYPE3(bhalf,bhalf,bhalf)>(const bhalf* X, const bhalf* W, bhalf* Y, bsmm_params* params);

// template cudaError_t BsmmXprop_CN<false, VTYPE3(float,ehalf,float)>(const float* X, const ehalf* W, float* Y, bsmm_params* params);
// template cudaError_t BsmmXprop_CN<false, VTYPE3(float,bhalf,float)>(const float* X, const bhalf* W, float* Y, bsmm_params* params);

template <CTYPE3(TX, TE, TU)>
cudaError_t BsmmUpdat_CN(const TX* X, const TE* E, TU* U, bsmm_params* params)
{
    dim3 grid(params->blocks, 1, 1);
    int loops = GridDim(params->N, 6);

    struct plist8<TX4>* X4 = (struct plist8<TX4>*)X;
    struct plist8<TE4>* E4 = (struct plist8<TE4>*)E;
    struct plist8<TX8>* X8 = (struct plist8<TX8>*)X;
    struct plist8<TE8>* E8 = (struct plist8<TE8>*)E;

    const int2* L2 = (const int2*)params->Lut;
           TU2* U2 = (       TU2*)U;
           TU4* U4 = (       TU4*)U;

    if (params->bshift == 3)
    {
        if (sizeof(TX) == 2 && sizeof(TE) == 2 && (params->N & 7) == 0)
            gemm_blocksparse_08x64x08x8_updat<TX8,TE8,TU2><<<grid,32,0,params->stream>>>(*X8, *E8, L2, U2, params->pcount*8, params->N, loops, params->alpha, params->beta);
        else
            gemm_blocksparse_08x64x08x4_updat<TX4,TE4,TU2><<<grid,32,0,params->stream>>>(*X4, *E4, L2, U2, params->pcount*8, params->N, loops, params->alpha, params->beta);
    }
    else if (params->bshift == 4)
    {
        if (sizeof(TX) == 2 && sizeof(TE) == 2 && (params->N & 7) == 0)
            gemm_blocksparse_16x64x16x8_updat<TX8,TE8,TU4><<<grid,64,0,params->stream>>>(*X8, *E8, L2, U4, params->pcount*8, params->N, loops, params->alpha, params->beta);
        else
            gemm_blocksparse_16x64x16x4_updat<TX4,TE4,TU4><<<grid,64,0,params->stream>>>(*X4, *E4, L2, U4, params->pcount*8, params->N, loops, params->alpha, params->beta);
    }
    else
    {
        if (sizeof(TX) == 2 && sizeof(TE) == 2 && (params->N & 7) == 0)
            gemm_blocksparse_32x64x32x8_updat<TX8,TE8,TU2><<<grid,256,0,params->stream>>>(*X8, *E8, L2, U2, params->pcount*8, params->N, loops, params->alpha, params->beta);
        else
            gemm_blocksparse_32x64x32x4_updat<TX4,TE4,TU2><<<grid,256,0,params->stream>>>(*X4, *E4, L2, U2, params->pcount*8, params->N, loops, params->alpha, params->beta);
    }
    return cudaPeekAtLastError();
}



template cudaError_t BsmmUpdat_CN<VTYPE3(float,float,float)>(const float* X, const float* E, float* U, bsmm_params* params);
template cudaError_t BsmmUpdat_CN<VTYPE3(ehalf,ehalf,ehalf)>(const ehalf* X, const ehalf* E, ehalf* U, bsmm_params* params);
template cudaError_t BsmmUpdat_CN<VTYPE3(bhalf,bhalf,bhalf)>(const bhalf* X, const bhalf* E, bhalf* U, bsmm_params* params);

// template cudaError_t BsmmUpdat_CN<VTYPE3(ehalf,ehalf,float)>(const ehalf* X, const ehalf* E, float* U, bsmm_params* params);
// template cudaError_t BsmmUpdat_CN<VTYPE3(ehalf,float,float)>(const ehalf* X, const float* E, float* U, bsmm_params* params);
// template cudaError_t BsmmUpdat_CN<VTYPE3(ehalf,float,ehalf)>(const ehalf* X, const float* E, ehalf* U, bsmm_params* params);

// template cudaError_t BsmmUpdat_CN<VTYPE3(bhalf,bhalf,float)>(const bhalf* X, const bhalf* E, float* U, bsmm_params* params);
// template cudaError_t BsmmUpdat_CN<VTYPE3(bhalf,float,float)>(const bhalf* X, const float* E, float* U, bsmm_params* params);
// template cudaError_t BsmmUpdat_CN<VTYPE3(bhalf,float,bhalf)>(const bhalf* X, const float* E, bhalf* U, bsmm_params* params);



__global__ void __launch_bounds__(1024) identity_init_CK(float* W, const int2* __restrict__ lut, int CB, int KB, int bshift)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int2 entry = lut[bid];

    int cb = entry.x;
    int kb = entry.y;
    int c  = tid >> bshift;
    int k  = tid & ((1 << bshift) - 1);

    float init = 0.0f;
    if (c == k && (cb % KB) == (kb % CB))
        init = 1.0f;

    W[(bid<<(bshift + bshift)) + tid] = init;
}

bool IdentityInitCK(CUstream stream, float* W, const int* lut, int CB, int KB, int blocks, int bshift)
{
    int threads = 1 << (bshift + bshift);
    identity_init_CK<<<blocks, threads, 0, stream>>>(W, (const int2*)lut, CB, KB, bshift);
    return true; // TODO
}



#endif // GOOGLE_CUDA

// nvcc -arch sm_60 -cubin blocksparse_matmul_op_gpu.cu
// nvdisasm -c -raw blocksparse_matmul_op_gpu.cubin > blocksparse_matmul_op_gpu.sass
